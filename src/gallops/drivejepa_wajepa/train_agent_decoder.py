"""Stage agent-decoder (agent decoder spec, 22 Oct). Model and loss:
drive_jepa_perception_based/agent_decoder.py; data, split and metrics: agent_decoder_data.py.

  overfit <out dir>           check 1: 64 training scenes (seed 0), the same batch every step, constant lr
                              2e-4, eval
                              every 100 steps on those 64 (final layer, Hungarian match): pass at centre
                              error < 0.5 m
                              (x-y, matched) and class accuracy > 95 % (matched slots); gives up after 4000
                              steps.
  train <out dir> <ckpt dir>  AdamW 2e-4, wd 0.01, batch 64, fp16 autocast + GradScaler, cosine over 12
                              epochs (per step),
                              grad-norm clip 35 (DETR3D). Check 3 = the first 50 steps (peak memory, step
                              time, loader
                              wait, every loss finite; stops on a non-finite loss). Held-out metrics after
                              each epoch;
                              stops when held-out recall (all classes, 2 m) has not improved for 2 epochs.
                              One checkpoint
                              per epoch, never overwritten.
  eval <ckpt> <out.json>      held-out metrics of one checkpoint.
env: Drive-JEPA's interpreter, PYTHONPATH = the agent decoder tree's navsim_v2 (code-agent-decoder), and
WAJEPA_TOKEN_CACHE, NAVTRAIN_AGENTS, HELDOUT_LOGS. QUERIES / LAYERS (default 64 / 3, the 22 Oct baseline) set
the
decoder for train; checkpoints record them and eval reads them back. Run 3A's decoder is D: 300 / 3, epoch 11.

Ported from paddock-ops agent-decoder/train_agent_decoder.py; changed: this docstring and the import of its
data module (common.py there).

    python -m gallops.drivejepa_wajepa.train_agent_decoder {overfit <out>|train <out> <ckpt dir>|eval <ckpt>
        <json>}
"""

import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from navsim.agents.drive_jepa_perception_based.agent_decoder import AgentDecoder, AgentLoss

from gallops.drivejepa_wajepa.agent_decoder_data import AGENTS, CacheSet, metrics, predict, split

dev = "cuda"
LR, WD, BS, EPOCHS, PATIENCE, CLIP = 2e-4, 0.01, 64, 12, 2, 35.0
NWORK = int(os.environ.get("NWORK", 10))
QUERIES, LAYERS = int(os.environ.get("QUERIES", 64)), int(os.environ.get("LAYERS", 3))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def to_dev(b):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def new_file(p):
    if os.path.exists(p):
        raise FileExistsError(f"refusing to overwrite {p}")
    return p


def loader(keys, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        CacheSet(keys),
        batch_size=BS,
        shuffle=shuffle,
        num_workers=NWORK,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False,
        generator=g,
        drop_last=shuffle,
    )


@torch.no_grad()
def matched_stats(model, crit, b):
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        outs = model(b["tokens"], b["lidar2img"])
    _, parts = crit(outs, b)
    bi, pi, ti = parts["_match"]
    out = outs[-1]
    ctr = (out["box"][bi, pi, :2] - b["agent_box"][bi, ti, :2]).norm(dim=-1)
    acc = (out["logits"][bi, pi].argmax(-1) == b["agent_class"][bi, ti]).float()
    unm = torch.ones(out["logits"].shape[:2], dtype=torch.bool, device=dev)
    unm[bi, pi] = False
    none_acc = (out["logits"][unm].argmax(-1) == 3).float().mean()
    model.train()
    return float(ctr.mean()), float(acc.mean()), float(none_acc)


def overfit(out_dir):
    agents = pickle.load(open(AGENTS, "rb"))
    train_keys, _ = split(agents)
    keys = random.Random(0).sample(train_keys, 64)
    ds = CacheSet(keys)
    b = to_dev(torch.utils.data.default_collate([ds[i] for i in range(64)]))
    log("overfit: 64 scenes,", int(b["agent_mask"].sum()), "agents")
    torch.manual_seed(0)
    model, crit = AgentDecoder().to(dev), AgentLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler()
    hist = []
    for step in range(1, 4001):
        with torch.autocast("cuda", dtype=torch.float16):
            outs = model(b["tokens"], b["lidar2img"])
        loss, parts = crit(outs, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0:
            ctr, acc, none_acc = matched_stats(model, crit, b)
            hist.append(
                dict(
                    step=step,
                    loss=round(float(loss), 4),
                    centre_err_m=round(ctr, 3),
                    class_acc=round(acc, 4),
                    none_acc=round(none_acc, 4),
                )
            )
            log(json.dumps(hist[-1]))
            if ctr < 0.5 and acc > 0.95:
                break
    ok = hist[-1]["centre_err_m"] < 0.5 and hist[-1]["class_acc"] > 0.95
    json.dump(
        {"pass": ok, "keys": keys, "history": hist},
        open(new_file(f"{out_dir}/check1_overfit.json"), "w"),
        indent=1,
    )
    torch.save({"model": model.state_dict(), "keys": keys}, new_file(f"{out_dir}/check1_overfit_model.pt"))
    log("CHECK1", "PASS" if ok else "FAIL", hist[-1])


def train(out_dir, ckpt_dir):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    agents = pickle.load(open(AGENTS, "rb"))
    train_keys, held_keys = split(agents)
    log(
        "train",
        len(train_keys),
        "held-out",
        len(held_keys),
        "workers",
        NWORK,
        "queries",
        QUERIES,
        "layers",
        LAYERS,
    )
    torch.manual_seed(0)
    model, crit = AgentDecoder(layers=LAYERS, queries=QUERIES).to(dev), AgentLoss()
    log("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_ep = len(train_keys) // BS
    total = EPOCHS * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    scaler = torch.amp.GradScaler()
    held = loader(held_keys, False)
    best, best_ep, check3, gstep = -1.0, -1, [], 0
    mlog = open(f"{out_dir}/heldout_metrics.jsonl", "a")
    for ep in range(EPOCHS):
        torch.cuda.reset_peak_memory_stats()
        t_end, run = time.time(), {}
        for step, b in enumerate(loader(train_keys, True, seed=ep)):
            t0 = time.time()
            wait = t0 - t_end
            b = to_dev(b)
            with torch.autocast("cuda", dtype=torch.float16):
                outs = model(b["tokens"], b["lidar2img"])
            loss, parts = crit(outs, b)
            finite = bool(torch.isfinite(loss)) and all(
                math.isfinite(v) for k, v in parts.items() if k[0] != "_"
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            scaler.step(opt)
            scaler.update()
            sched.step()
            torch.cuda.synchronize()
            t_end = time.time()
            gstep += 1
            for k, v in parts.items():
                if k[0] != "_":
                    run[k] = run.get(k, 0.0) + v
            run["loss"] = run.get("loss", 0.0) + float(loss)
            if gstep <= 50:
                check3.append(
                    dict(
                        step=gstep,
                        loss=float(loss),
                        finite=finite,
                        step_s=round(t_end - t0, 3),
                        loader_wait_s=round(wait, 3),
                        peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        grad_norm=round(gnorm, 2),
                        scale=scaler.get_scale(),
                    )
                )
                if not finite:
                    json.dump(
                        {"pass": False, "steps": check3},
                        open(new_file(f"{out_dir}/check3_first50.json"), "w"),
                        indent=1,
                    )
                    log("CHECK3 FAIL: non-finite loss at step", gstep, parts)
                    return
                if gstep == 50:
                    s = check3[5:]  # after warm-up
                    summ = dict(
                        pass_=True,
                        peak_mem_gib=max(c["peak_mem_gib"] for c in check3),
                        step_s_median=float(np.median([c["step_s"] for c in s])),
                        loader_wait_s_median=float(np.median([c["loader_wait_s"] for c in s])),
                        loader_wait_s_mean=float(
                            np.mean([c["loader_wait_s"] for c in s])
                        ),  # waits come in bursts
                        epoch_eta_h=round(
                            steps_ep * float(np.mean([c["step_s"] + c["loader_wait_s"] for c in s])) / 3600, 2
                        ),
                    )
                    json.dump(
                        {"summary": summ, "steps": check3},
                        open(new_file(f"{out_dir}/check3_first50.json"), "w"),
                        indent=1,
                    )
                    log("CHECK3 PASS", json.dumps(summ))
            if (step + 1) % 200 == 0:
                log(
                    f"ep {ep} step {step + 1}/{steps_ep}",
                    {k: round(v / 200, 4) for k, v in run.items()},
                    f"step {t_end - t0:.2f}s wait {wait:.2f}s lr {sched.get_last_lr()[0]:.2e}",
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                )
                run = {}
        ck = new_file(f"{ckpt_dir}/epoch_{ep:02d}.pt")
        torch.save(
            {"model": model.state_dict(), "epoch": ep, "step": gstep, "queries": QUERIES, "layers": LAYERS},
            ck,
        )
        t0 = time.time()
        m = metrics(predict(model, held), held_keys, agents)
        rec = m["detection"]["all"]["recall"]
        mlog.write(json.dumps({"epoch": ep, "ckpt": ck, **m}) + "\n")
        mlog.flush()
        log(
            f"epoch {ep} held-out ({time.time() - t0:.0f}s): recall {rec} "
            f"precision {m['detection']['all']['precision']}",
            "moving veh 20m",
            m["moving_vehicles_20m"],
            "e1 b0",
            m["e1"]["b0"],
        )
        if rec > best:
            best, best_ep = rec, ep
            with open(f"{out_dir}/best_epoch.txt", "a") as fh:
                fh.write(f"{ep} {rec} {ck}\n")
        elif ep - best_ep >= PATIENCE:
            log("held-out recall has not improved for", PATIENCE, "epochs; stopping at epoch", ep)
            break
    log("TRAIN_DONE best epoch", best_ep, "recall", best)


def evaluate(ckpt, out_json):
    agents = pickle.load(open(AGENTS, "rb"))
    _, held_keys = split(agents)
    sd = torch.load(ckpt, map_location=dev)
    model = AgentDecoder(layers=sd.get("layers", 3), queries=sd.get("queries", 64)).to(dev)
    model.load_state_dict(sd["model"])
    m = metrics(predict(model, loader(held_keys, False)), held_keys, agents)
    json.dump({"ckpt": ckpt, **m}, open(new_file(out_json), "w"), indent=1)
    log("EVAL_DONE", json.dumps(m["detection"]["all"]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "overfit":
        overfit(sys.argv[2])
    elif mode == "train":
        train(sys.argv[2], sys.argv[3])
    elif mode == "eval":
        evaluate(sys.argv[2], sys.argv[3])
