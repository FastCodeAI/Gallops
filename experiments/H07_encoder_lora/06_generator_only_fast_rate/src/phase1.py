"""Phase 1: train only the generator on KIT, encoder (pass-3 LoRA merged) frozen.

    setsid nohup ./run.sh src/phase1.py all > logs/phase1.log 2>&1 &     # preflight, G1, G2, training
    ./run.sh src/phase1.py all --resume                                 # continue after the last saved pass

Model: release checkpoint (sha256-checked) + pass-3 LoRA merged (s0 g0_gpu.build = s3 close.py
loading + s2 f_final.merge).  Frozen: encoder, projector, scorer.  Trainable: generator.
Encoder runs live from the s2 frame cache in eval mode; only _backbone.grid_mask is switched to
training mode during training batches (grid_mask.py:89 checks self.training), so GridMask
(p = 0.7) is applied there and nowhere else.  A forward hook on grid_mask records every call.
Loss: upstream agent.trajectory_loss_anchors (drive_jepa_agent.py:221-269), unchanged, with empty
vocabulary lists.  Adam lr 1e-4 constant, wd 0, no clipping, bf16 autocast, seed 0, batch 32,
drop_last.  Evaluation: batch 1 through s2 evalsets.forward_dump (agent.eval(), fp32), picking
scenes only.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cache as K           # s2
import common as C          # s1
import evalsets as E        # s2
import g0_gpu as G0         # 05_generator_survey
import lora as L            # s1
import kitdata as DT        # this stage
from s2common import wait_for_gpu

HERE = Path(__file__).resolve().parent.parent
OUT, LOGS = HERE / "out", HERE / "logs"
DATA = DT.DATA
CK = DATA / "ckpt"
DUMPS = DATA / "dumps"
S0OUT = HERE.parent / "05_generator_survey" / "out"

GEN_PREFIXES = ("hist_encoding.", "init_feature.", "_trajectory_head.")
GEN_TOTAL = 2_389_827
# BEVFormerEncoder's fixed reference grids: nn.Parameter(..., requires_grad=False) upstream
# (bevformer/encoder.py:60-64).  Counted in the generator total, never trained upstream.
FIXED_SUFFIXES = ("reference_points", "ref_3d", "hybird_ref_2d", "ref_2d")
BATCH, LR, SEED = 32, 1e-4, 0
MAX_PASSES, PATIENCE = 10, 3


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def wj(name: str, obj) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(OUT / name)


# --------------------------------------------------------------------------- #
# Room checks (inside the run, per the standing rules)
# --------------------------------------------------------------------------- #


def avail_ram_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return 0.0


def preflight() -> dict:
    t0 = time.time()
    while True:
        load5 = os.getloadavg()[1]
        ram = avail_ram_gb()
        disk = shutil.disk_usage("/").free / 2**30
        if load5 < 32 and ram > 40 and disk > 5:
            break
        log(f"WAIT room: load5 {load5:.1f} ram {ram:.1f} GB disk {disk:.0f} GB; retry in 300 s")
        if time.time() - t0 > 6 * 3600:
            raise SystemExit("preflight: no room for 6 h, standing down")
        time.sleep(300)
    gw = wait_for_gpu(14000, "phase1", log=log)
    return {"load5": load5, "ram_gb": round(ram, 1), "disk_gb": round(disk, 1), "gpu_wait_s": round(gw, 1)}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Model:
    def __init__(self):
        self.agent = G0.build(pass3=True, merged=True)     # all requires_grad False, eval
        self.m = self.agent._pad_model
        assert not L.lora_modules(self.m), "LoRA must be merged"
        gen = [(n, p) for n, p in self.m.named_parameters() if n.startswith(GEN_PREFIXES)]
        self.gen_total = sum(p.numel() for _, p in gen)
        assert self.gen_total == GEN_TOTAL, self.gen_total
        self.fixed = [(n, p) for n, p in gen if n.endswith(FIXED_SUFFIXES)]
        self.train_params = [(n, p) for n, p in gen if not n.endswith(FIXED_SUFFIXES)]
        for _, p in self.train_params:
            p.requires_grad_(True)
        self.n_train = sum(p.numel() for _, p in self.train_params)
        assert sum(p.requires_grad for p in self.m.parameters()) == len(self.train_params)
        self.gm = {"calls": 0, "applied": 0, "train_flag_calls": 0, "encoder_eval_at_call": 0}
        self.m._backbone.grid_mask.register_forward_hook(self._gm_hook)

    def _gm_hook(self, mod, inp, out):
        self.gm["calls"] += 1
        self.gm["train_flag_calls"] += int(mod.training)
        self.gm["applied"] += int(not torch.equal(inp[0], out))
        bk = self.m._backbone
        self.gm["encoder_eval_at_call"] += int(not any(x.training for x in bk.img_backbone.modules())
                                               and not bk.projector.training and not bk.training)

    def reset_gm(self):
        for k in self.gm:
            self.gm[k] = 0

    def train_mode(self):
        self.m.eval()
        self.m._backbone.grid_mask.train()
        for mod in (self.m.hist_encoding, self.m.init_feature, self.m._trajectory_head):
            mod.train()

    def gen_state(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.train_params}

    def load_gen_state(self, sd: dict) -> None:
        cur = dict(self.train_params)
        assert set(sd) == set(cur)
        with torch.no_grad():
            for n, t in sd.items():
                cur[n].copy_(t.to(cur[n].device))

    def loss(self, feats: dict, target: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.m(feats)
        pl = [q.float() for q in out["proposal_list"]]
        empty = [np.zeros(0, dtype=np.int64)] * target.shape[0]
        return self.agent.trajectory_loss_anchors(pl, target, self.agent._config, empty)[0]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, moments, meta):
        self.mo, self.meta, self.fc = moments, meta, None

    def __len__(self):
        return len(self.mo)

    def __getitem__(self, i):
        if self.fc is None:
            self.fc = K.FrameCache()
        m = self.mo[i]
        f = DT.assemble(m, self.meta[m["scene_id"]], self.fc)
        f["target"] = torch.as_tensor(m["future_poses"], dtype=torch.float32)
        return f


def to_gpu(batch: dict):
    target = batch.pop("target").cuda(non_blocking=True)
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}, target


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def key(m) -> tuple:
    return (m["scene_id"], m["t"])


def evaluate(model: Model, moments, meta) -> dict:
    """Batch 1, fp32, agent.eval() inside forward_dump: the s0 / s2 evaluation path."""
    fc = K.FrameCache()
    before = dict(model.gm)
    dump = {}
    for m in moments:
        d = E.forward_dump(model.agent, DT.assemble(m, meta[m["scene_id"]], fc))
        dump[key(m)] = {k: d[k] for k in ("proposals", "pdm_score", "chosen")}
    assert model.gm["applied"] == before["applied"] and model.gm["train_flag_calls"] == before["train_flag_calls"], \
        "GridMask active during evaluation"
    return dump


def per_moment(dump: dict, human: dict, start: dict | None) -> dict:
    """05_generator_survey REPORT section G definitions, waypoints 1-6 (0.5-3 s); plus drift (metric 5)."""
    iu = np.triu_indices(32, 1)
    per = {}
    for k in dump:
        p = np.asarray(dump[k]["proposals"], dtype=np.float64)[:, :6, :2]
        h = np.asarray(human[k], dtype=np.float64)[:6, :2]
        err = np.linalg.norm(p - h[None], axis=-1)
        ade = err.mean(1)
        i = int(ade.argmin())
        pair = np.linalg.norm(p[:, None] - p[None], axis=-1).mean(-1)
        r = {"m1a": float(ade[i]), "m1b": float(err[i, 5]), "min_fde3": float(err[:, 5].min()),
             "m2_lt1": int((ade < 1).sum()), "m2_lt2": int((ade < 2).sum()), "m3_variety": float(pair[iu].mean()),
             "m6_pick_ade": float(ade[int(dump[k]["chosen"])])}
        if start is not None:
            s = np.asarray(start[k]["proposals"], dtype=np.float64)[:, :6, :2]
            r["m5_drift"] = float(np.linalg.norm(p - s, axis=-1).mean())
        per[k] = r
    return per


def aggregate(per: dict, worst: list | None = None) -> dict:
    keys = sorted(per)
    names = per[keys[0]].keys()
    out = {"n": len(keys)}
    for nm in names:
        v = [per[k][nm] for k in keys]
        out[nm] = {"mean": float(np.mean(v)), "median": float(np.median(v))}
    if worst:
        for nm in ("m1a", "m1b"):
            v = [per[tuple(k)][nm] for k in worst]
            out["m4_worst20_" + nm] = {"mean": float(np.mean(v)), "median": float(np.median(v))}
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_g1(model: Model, main, meta, human, cont_keys) -> dict:
    """Starting model on the 103 = s0 g_menu.json exactly; baselines on MAIN; worst-20 % list."""
    s0 = json.loads((S0OUT / "g_menu.json").read_text())["menu"]
    start = evaluate(model, main, meta)
    model_orig = G0.build(pass3=False)
    fc = K.FrameCache()
    orig = {}
    for m in main:
        d = E.forward_dump(model_orig, DT.assemble(m, meta[m["scene_id"]], fc))
        orig[key(m)] = {k: d[k] for k in ("proposals", "pdm_score", "chosen")}
    del model_orig
    torch.cuda.empty_cache()

    # s0's own metric function on the 103, against s0's saved numbers (exact)
    h103 = {k: human[k] for k in cont_keys}
    o103 = {k: orig[k] for k in cont_keys}
    s103 = {k: start[k] for k in cont_keys}
    o_agg, o_per = G0.menu_metrics(o103, h103)
    base_best = {k: o_per[k]["best_ade"] for k in o_per}
    o_agg, _ = G0.menu_metrics(o103, h103, base_best)
    s_agg, _ = G0.menu_metrics(s103, h103, base_best)
    rt = lambda x: json.loads(json.dumps(x, default=str))
    exact = {"original": rt(o_agg) == s0["original"], "start (pass3 merged)": rt(s_agg) == s0["pass3_merged"]}
    # this stage's metric function equals s0's on the same moments
    mine = aggregate(per_moment(s103, h103, None))
    same_fn = all(abs(mine[a]["mean"] - s_agg[b]["mean"]) == 0 for a, b in
                  (("m1a", "best_ade"), ("m1b", "best_ade_fde3"), ("min_fde3", "min_fde3"), ("m2_lt1", "n_below_1m"),
                   ("m2_lt2", "n_below_2m"), ("m3_variety", "variety_m"), ("m6_pick_ade", "chosen_ade")))
    # worst 20 % of MAIN by the ORIGINAL model's closest-of-32 average distance error
    o_main = per_moment(orig, human, start)
    n_w = math.ceil(0.2 * len(main))
    worst = [list(k) for k in sorted(o_main, key=lambda k: (-o_main[k]["m1a"], k))[:n_w]]
    wj("worst20_main.json", {"rule": "top ceil(0.2 n) MAIN moments by the original model's closest-of-32 ADE",
                             "n": n_w, "moments": worst})
    s_main = per_moment(start, human, start)
    with open(DUMPS / "original.pkl", "wb") as fh:
        pickle.dump(orig, fh)
    with open(DUMPS / "start.pkl", "wb") as fh:
        pickle.dump(start, fh)
    res = {"exact_vs_s0_on_103": exact, "metric_fn_equals_s0": same_fn,
           "pass": all(exact.values()) and same_fn,
           "original_main": aggregate(o_main, worst), "start_main": aggregate(s_main, worst),
           "original_cont": aggregate({k: o_main[k] for k in cont_keys}),
           "start_cont": aggregate({k: s_main[k] for k in cont_keys})}
    return res, start, worst


def gate_g2(model: Model, train, meta) -> dict:
    """20 steps on a throwaway optimiser; generator restored afterwards."""
    saved = model.gen_state()
    opt = torch.optim.Adam([p for _, p in model.train_params], lr=LR)
    rng = torch.Generator().manual_seed(12345)        # not the training stream
    idx = torch.randperm(len(train), generator=rng)[:20 * BATCH].tolist()
    ds = TrainSet(train, meta)
    batches = [torch.utils.data.default_collate([ds[i] for i in idx[b * BATCH:(b + 1) * BATCH]]) for b in range(20)]
    fixed_feats, fixed_t = to_gpu(dict(batches[0]))

    def fixed_loss():
        model.m.eval()
        with torch.no_grad():
            return float(model.loss({k: v.clone() for k, v in fixed_feats.items()}, fixed_t))

    l_before = fixed_loss()
    torch.cuda.reset_peak_memory_stats()
    model.reset_gm()
    losses, times, grad_ok, other_grads = [], [], [], []
    others = [(n, p) for n, p in model.m.named_parameters() if not n.startswith(GEN_PREFIXES)] + model.fixed
    for b in batches:
        feats, t = to_gpu(dict(b))
        model.train_mode()
        torch.cuda.synchronize(); t0 = time.time()
        loss = model.loss(feats, t)
        loss.backward()
        grad_ok.append(sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for _, p in model.train_params))
        other_grads.append(sum(int(p.grad is not None) for _, p in others))
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        losses.append(float(loss))
    gm = dict(model.gm)              # the 20 training calls only (the check below adds one eval call)
    l_after = fixed_loss()
    model.load_gen_state(saved)
    restored = all(torch.equal(saved[n], p.detach().cpu()) for n, p in model.train_params)
    res = {"steps": 20, "losses": [round(x, 4) for x in losses],
           "fixed_batch_loss_before_after (no GridMask)": [l_before, l_after],
           "train_tensors": len(model.train_params), "nonzero_grad_tensors_each_step_min": min(grad_ok),
           "grads_on_encoder_projector_scorer_fixed_grids_max": max(other_grads),
           "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "s_per_step_mean_after_first": round(float(np.mean(times[1:])), 3), "s_per_step": [round(x, 3) for x in times],
           "gridmask_calls": gm, "generator_restored_bit_exact": restored}
    res["pass"] = (min(grad_ok) == len(model.train_params) and max(other_grads) == 0 and l_after < l_before
                   and restored and gm["train_flag_calls"] == gm["calls"] == 20
                   and gm["encoder_eval_at_call"] == gm["calls"] and 0 < gm["applied"] < 20)
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(resume: bool) -> None:
    CK.mkdir(parents=True, exist_ok=True)
    DUMPS.mkdir(parents=True, exist_ok=True)
    state = json.loads((OUT / "passes.json").read_text()) if resume else {}
    if not resume:
        state["preflight"] = preflight()
        log(f"preflight {state['preflight']}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    meta = torch.load(DATA / "scene_meta.pt", weights_only=False)
    mom = torch.load(DATA / "moments.pt", weights_only=False)
    train, main_set = mom["train"], mom["selection"]
    assert len(train) == 13937 and len(main_set) == 1763, (len(train), len(main_set))
    human = {key(m): m["future_poses"] for m in main_set}
    cont_keys = sorted((e["scene_id"], e["keyframe"]) for e in E.load_selection_features())
    assert len(cont_keys) == 103 and set(cont_keys) <= set(human)

    model = Model()
    state["model"] = {"generator_total": model.gen_total, "trainable": model.n_train,
                      "trainable_tensors": len(model.train_params),
                      "fixed_grids_frozen": {n: p.numel() for n, p in model.fixed}}
    log(f"model {state['model']}")

    if not resume:
        g1, start, worst = gate_g1(model, main_set, meta, human, cont_keys)
        state["G1"] = g1
        wj("passes.json", state)
        log(f"G1 pass={g1['pass']} {g1['exact_vs_s0_on_103']} metric_fn={g1['metric_fn_equals_s0']}")
        if not g1["pass"]:
            raise SystemExit("G1 failed: stopping")
        g2 = gate_g2(model, train, meta)
        state["G2"] = g2
        wj("passes.json", state)
        log(f"G2 pass={g2['pass']} {json.dumps({k: v for k, v in g2.items() if k != 's_per_step'})}")
        if not g2["pass"]:
            raise SystemExit("G2 failed: stopping")
        state["passes"] = []
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)   # training starts from seed 0
    else:
        with open(DUMPS / "start.pkl", "rb") as fh:
            start = pickle.load(fh)
        worst = json.loads((OUT / "worst20_main.json").read_text())["moments"]

    start_agg = state["G1"]["start_main"]
    opt = torch.optim.Adam([p for _, p in model.train_params], lr=LR, weight_decay=0)
    shuf = torch.Generator().manual_seed(SEED)
    done = len(state["passes"])
    perms = [torch.randperm(len(train), generator=shuf).tolist() for _ in range(MAX_PASSES)]
    if done:
        ck = torch.load(CK / f"pass_{done:02d}.pt", weights_only=False)
        model.load_gen_state(ck["generator"]); opt.load_state_dict(ck["optimizer"])
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"]); random.setstate(ck["py_rng"])
        log(f"resumed after pass {done}")
    ds = TrainSet(train, meta)
    best, since_best = float("inf"), 0
    for p in state["passes"]:
        if p["main"]["m1b"]["mean"] < best:
            best, since_best = p["main"]["m1b"]["mean"], 0
        else:
            since_best += 1

    for pi in range(done + 1, MAX_PASSES + 1):
        perm = perms[pi - 1]
        batches = [perm[i:i + BATCH] for i in range(0, len(perm) - BATCH + 1, BATCH)]   # drop_last
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=6, pin_memory=True,
                                             prefetch_factor=4, persistent_workers=False)
        model.reset_gm()
        t0, wait_t, losses = time.time(), 0.0, []
        tw = time.time()
        for step, b in enumerate(loader, 1):
            wait_t += time.time() - tw
            feats, t = to_gpu(b)
            model.train_mode()
            loss = model.loss(feats, t)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at pass {pi} step {step}")
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            if step % 50 == 0:
                ram = avail_ram_gb()
                log(f"pass {pi} step {step}/{len(batches)} loss {np.mean(losses[-50:]):.4f} ram {ram:.0f} GB "
                    f"gm {model.gm['applied']}/{model.gm['calls']}")
                if ram < 20:
                    raise SystemExit(f"available RAM {ram:.1f} GB < 20 GB: stopping (pass {pi} not saved)")
            tw = time.time()
        train_s = time.time() - t0
        gm_train = dict(model.gm)
        te = time.time()
        dump = evaluate(model, main_set, meta)
        eval_s = time.time() - te
        per = per_moment(dump, human, start)
        agg = aggregate(per, worst)
        cont = aggregate({k: per[k] for k in cont_keys})
        torch.save({"generator": model.gen_state(), "optimizer": opt.state_dict(), "pass": pi,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "py_rng": random.getstate(),
                    "settings": {"lr": LR, "batch": BATCH, "seed": SEED, "loss": "upstream trajectory_loss_anchors, empty vocabulary lists"}},
                   CK / f"pass_{pi:02d}.pt")
        with open(DUMPS / f"pass_{pi:02d}.pkl", "wb") as fh:
            pickle.dump(dump, fh)
        rec = {"pass": pi, "steps": len(batches), "train_loss_mean": float(np.mean(losses)),
               "train_loss_last50": float(np.mean(losses[-50:])), "gridmask_train": gm_train,
               "train_s": round(train_s, 1), "data_wait_s": round(wait_t, 1), "eval_s": round(eval_s, 1),
               "main": agg, "cont": {k: cont[k] for k in ("m1a", "m1b")}}
        guards = {"variety_below_90pct": agg["m3_variety"]["mean"] < 0.9 * start_agg["m3_variety"]["mean"],
                  "within2m_below_start": agg["m2_lt2"]["mean"] < start_agg["m2_lt2"]["mean"]}
        rec["guards"] = guards
        if agg["m1b"]["mean"] < best:
            best, since_best = agg["m1b"]["mean"], 0
        else:
            since_best += 1
        state["passes"].append(rec)
        state["best_pass"] = min(state["passes"], key=lambda q: (q["main"]["m1b"]["mean"], q["pass"]))["pass"]
        wj("passes.json", state)
        log(f"pass {pi}: 1b {agg['m1b']['mean']:.4f} (start {start_agg['m1b']['mean']:.4f}) 1a {agg['m1a']['mean']:.4f} "
            f"<2m {agg['m2_lt2']['mean']:.2f} var {agg['m3_variety']['mean']:.3f} drift {agg['m5_drift']['mean']:.3f} "
            f"train {train_s:.0f}s eval {eval_s:.0f}s guards {guards}")
        if any(guards.values()):
            state["stopped"] = f"guard fired after pass {pi}: {[g for g, v in guards.items() if v]}"
            wj("passes.json", state)
            log(state["stopped"])
            return
        if since_best >= PATIENCE:
            state["stopped"] = f"no 1b improvement for {PATIENCE} passes after pass {pi}"
            wj("passes.json", state)
            log(state["stopped"])
            return
    state["stopped"] = f"reached {MAX_PASSES} passes"
    wj("passes.json", state)
    log(state["stopped"])


if __name__ == "__main__":
    assert sys.argv[1] == "all"
    main(resume="--resume" in sys.argv)
