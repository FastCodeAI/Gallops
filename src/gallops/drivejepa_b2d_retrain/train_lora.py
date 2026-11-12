"""Phase 1 trainer: our copy of Sid's E1t/src/train.py (specs/phase1_lora_v3.md in H16's folder). Kept: Setup,
step(), guesser_groups, set_lr, the warm-up then main order, the same lora / players / masks2. Changed: the
data (Bench2Drive `native` cache), the splits, no stop rule (1,000 warm-up steps, then 3 passes), new-path
saves with the slow copy, optimiser, sampler and all RNG state (train_018b._save_resume), resume anywhere,
memory STOP file, reports. Reports: fixed-set loss on training and held-out clips at the end of warm-up and
every pass (spec v3: no planner report).
    python -m gallops.drivejepa_b2d_retrain.train_lora --name <run> [--resume <save dir>]
        [--warmup-steps N --passes N --stop-at-main N --save-every N --no-reports]"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np

from gallops.drivejepa_b2d_retrain import wiring as W

L, P, M = W.setup()
import torch  # noqa: E402

from gallops.drivejepa_b2d_retrain import ade as A  # noqa: E402
from gallops.drivejepa_b2d_retrain import frames as D  # noqa: E402

S = W.SETTINGS
PHASE_CODE = {"warmup": 1, "main": 2}


class StopRun(RuntimeError):
    pass


def log(m):
    print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def guesser_groups(g):
    decay = [p for n, p in g.named_parameters() if "bias" not in n and p.ndim != 1]
    nodecay = [p for n, p in g.named_parameters() if "bias" in n or p.ndim == 1]
    return [
        {
            "params": decay,
            "lr": S["lr_guesser"],
            "weight_decay": S["wd_guesser"],
            "base_lr": S["lr_guesser"],
            "name": "guesser",
        },
        {
            "params": nodecay,
            "lr": S["lr_guesser"],
            "weight_decay": 0.0,
            "base_lr": S["lr_guesser"],
            "name": "guesser_nowd",
        },
    ]


def set_lr(opt, phase_step):
    f = min(1.0, (phase_step + 1) / S["ramp_steps"])
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f
    return f


def grad_norm(params):
    gs = [p.grad for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs)))) if gs else 0.0


def lora_state(enc):
    return {n: p.detach().cpu().clone() for n, p in enc.named_parameters() if ".lora_" in n}


def load_lora_state(enc, sd):
    params = dict(enc.named_parameters())
    assert set(sd) == {n for n in params if ".lora_" in n}
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].device))


class Setup:
    def __init__(self, hash_files=True):
        self.hashes = {}
        if hash_files:
            from common import sha256_file

            self.hashes = {"agent": sha256_file(W.AGENT_CKPT), "e50": sha256_file(W.E50)}
            assert self.hashes["agent"] == W.AGENT_SHA and self.hashes["e50"] == W.E50_SHA, self.hashes
        self.splits = json.load(open(W.SPLITS))
        self.agent = W.build_agent()
        self.learner = self.agent.get_submodule(W.ENC_PATH)
        assert len(L.wrap(self.learner, S["lora_rank"], S["lora_alpha"])) == 96
        torch.manual_seed(W.SEED)
        for m in L.lora_modules(self.learner).values():
            torch.nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
        self.key = copy.deepcopy(self.learner)
        for p in self.key.parameters():
            p.requires_grad_(False)
        self.key.eval()
        self.guesser = P.build_guesser(torch.load(W.E50, map_location="cpu", mmap=True)).cuda()
        self.gens = M.make()
        self.train_ds = D.CachedPairs(self.splits["train"], S["pair_gap_files"], S["stride_files"])

    def loader(self, pass_idx, start_batch=0, workers=4):
        g = torch.Generator()
        g.manual_seed(W.SEED * 1000 + pass_idx)
        perm = torch.randperm(len(self.train_ds), generator=g).tolist()
        batches = [perm[i : i + S["batch"]] for i in range(0, len(perm), S["batch"])]
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_sampler=batches[start_batch:],
            num_workers=workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )

    def n_batches(self):
        return math.ceil(len(self.train_ds) / S["batch"])

    def step(self, opt, x, phase, phase_step, train_lora):
        masks = M.sample(self.gens, x.shape[0], M.step_rng(W.SEED, PHASE_CODE[phase], phase_step), "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, per = P.game_loss(self.learner, self.key, self.guesser, x, masks)
        if not torch.isfinite(loss):
            raise StopRun(f"non-finite loss {float(loss)} at {phase} step {phase_step}")
        loss.backward()
        gl = grad_norm(L.lora_parameters(self.learner)) if train_lora else 0.0
        gg = grad_norm(self.guesser.parameters())
        if not (math.isfinite(gl) and math.isfinite(gg)):
            raise StopRun(f"non-finite grad norm at {phase} step {phase_step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        if train_lora:
            L.momentum_update(self.key, self.learner, S["momentum"])
        return {
            "loss": float(loss),
            "per_mask": per,
            "gn_lora": gl,
            "gn_guesser": gg,
            "hidden": [int(p.shape[1]) for _, p in masks],
        }

    def opt_warmup(self):
        return torch.optim.AdamW(guesser_groups(self.guesser), betas=S["betas"], eps=S["eps"])

    def opt_main(self):
        return torch.optim.AdamW(
            [
                {
                    "params": L.lora_parameters(self.learner),
                    "lr": S["lr_lora"],
                    "weight_decay": S["wd_lora"],
                    "base_lr": S["lr_lora"],
                    "name": "lora",
                },
                *guesser_groups(self.guesser),
            ],
            betas=S["betas"],
            eps=S["eps"],
        )


def save(path, st, opt, state):
    """Everything to continue from here, at a new path (train_018b._save_resume, written atomically)."""
    assert not path.exists(), f"never overwrite: {path}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.mkdir(parents=True)
    torch.save(
        dict(
            lora=lora_state(st.learner),
            key_lora=lora_state(st.key),
            guesser=st.guesser.state_dict(),
            optimizer=opt.state_dict(),
            state=state,
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
            ),
        ),
        tmp / "state.pt",
    )
    json.dump(
        {**S, **state, "saved": time.strftime("%F %T")},
        open(tmp / "settings.json", "w"),
        indent=1,
        default=str,
    )
    tmp.rename(path)
    (path.parent / "LATEST").with_suffix(f".{state['seq']:04d}").write_text(str(path))
    log(f"saved {path}")


def report(st, sets, tag, run, moments, do_ade):
    t0 = time.time()
    r = dict(
        tag=tag,
        train_loss_fixed=sets["train"].loss(st.learner, st.key, st.guesser),
        heldout_loss_fixed=sets["held"].loss(st.learner, st.key, st.guesser),
    )
    if do_ade:
        r["ade"] = A.evaluate(st.agent, moments, tag, run)
        st.learner.train()
        st.guesser.train()
    r["seconds"] = round(time.time() - t0)
    p, k = run / f"report_{tag}.json", 1
    while p.exists():  # a resumed run repeats a report: new path, nothing overwritten
        k += 1
        p = run / f"report_{tag}_r{k}.json"
    json.dump(r, open(p, "w"), indent=1)
    log(
        f"REPORT {tag}: train {r['train_loss_fixed']:.5f} held-out {r['heldout_loss_fixed']:.5f} "
        + (
            f"closest {r['ade']['closest_ade_3s']:.3f} pick {r['ade']['pick_ade_3s']:.3f} variety {r['ade']['variety_m']:.3f} "  # noqa: E501
            if do_ade
            else ""
        )
        + f"({r['seconds']} s)"
    )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--warmup-steps", type=int, default=S["warmup_steps"])
    ap.add_argument("--passes", type=int, default=S["main_passes"])
    ap.add_argument(
        "--stop-at-main", type=int, default=0, help="checks: stop after this many main-phase steps"
    )
    ap.add_argument(
        "--stop-at-warmup", type=int, default=0, help="checks: stop after this many warm-up steps"
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--no-reports", action="store_true")
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()
    run = W.RUNS / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)  # run_train.sh makes it for ACTIVE / DONE
    assert a.resume or not list(run.glob("save_*")), f"never overwrite: {run} has saves"
    random.seed(W.SEED)
    np.random.seed(W.SEED)
    st = Setup(hash_files=not a.no_hash)
    state = dict(phase="warmup", phase_step=0, pass_idx=0, batch_in_pass=0, seq=0)
    opt = st.opt_warmup()
    L.set_lora_trainable(st.learner, False)
    if a.resume:
        ck = torch.load(os.path.join(a.resume, "state.pt"), map_location="cpu", weights_only=False)
        load_lora_state(st.learner, ck["lora"])
        load_lora_state(st.key, ck["key_lora"])
        st.guesser.load_state_dict(ck["guesser"])
        state = ck["state"]
        if state["phase"] == "main":
            L.set_lora_trainable(st.learner, True)
            opt = st.opt_main()
        opt.load_state_dict(ck["optimizer"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log(f"resumed from {a.resume}: {state}")
    for p in st.guesser.parameters():
        p.requires_grad_(True)
    nb = st.n_batches()
    info = dict(
        args=vars(a),
        settings=S,
        hashes=st.hashes,
        train_clips=len(st.splits["train"]),
        heldout_clips=len(st.splits["heldout"]),
        train_pairs=len(st.train_ds),
        batches_per_pass=nb,
        sid_files="sid_e75aaac",
        devkit=str(W.DEVKIT),
        cache=str(W.CACHE),
        started=time.strftime("%F %T"),
    )
    json.dump(info, open(run / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1, default=str)
    log(
        f"train pairs {len(st.train_ds)}, {nb} batches per pass, warm-up {a.warmup_steps} steps, {a.passes} passes, total {a.warmup_steps + a.passes * nb} steps"  # noqa: E501
    )
    sets = moments = None
    if not a.no_reports:
        sets = dict(train=D.FixedSet(st.splits["train"], P, M), held=D.FixedSet(st.splits["heldout"], P, M))
        moments = torch.load(W.RUNS / "ade_set.pt", weights_only=False)["moments"]
        log(
            f"fixed sets: train {len(sets['train'].items)} pairs, held-out {len(sets['held'].items)} pairs; ADE moments {len(moments)}"  # noqa: E501
        )
    steplog = open(run / "steps.jsonl", "a")
    st.learner.train()
    st.guesser.train()
    stop_file = run / "STOP"

    def one(x, train_lora):
        f = set_lr(opt, state["phase_step"])
        t0 = time.time()
        r = st.step(opt, x.cuda(non_blocking=True), state["phase"], state["phase_step"], train_lora)
        torch.cuda.synchronize()
        state["phase_step"] += 1
        state["batch_in_pass"] += 1
        return f, time.time() - t0, r

    def write(f, dt, wait, r):
        steplog.write(
            json.dumps(
                dict(
                    phase=state["phase"],
                    step=state["phase_step"],
                    pass_idx=state["pass_idx"],
                    lr_factor=f,
                    step_s=round(dt, 3),
                    data_wait_s=round(wait, 3),
                    peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    **r,
                )
            )
            + "\n"
        )
        if state["phase_step"] % 100 == 0:
            steplog.flush()
            log(
                f"{state['phase']} pass {state['pass_idx']} step {state['phase_step']} loss {r['loss']:.4f} gn_lora {r['gn_lora']:.3g} gn_guesser {r['gn_guesser']:.3g} {dt:.2f} s wait {wait:.2f} s"  # noqa: E501
            )

    def periodic(tag):
        if state["phase_step"] % a.save_every == 0:
            state["seq"] += 1
            save(run / f"save_{state['seq']:04d}_{tag}_{state['phase_step']:06d}", st, opt, state)

    def stopped():
        if stop_file.exists():
            state["seq"] += 1
            save(
                run / f"save_{state['seq']:04d}_stop_{state['phase']}_{state['phase_step']:06d}",
                st,
                opt,
                state,
            )
            log(f"STOPPED: {stop_file.read_text().strip()}")
            return True
        return False

    # warm-up: guesser only, slow copy not updated
    while state["phase"] == "warmup" and state["phase_step"] < a.warmup_steps:
        pidx, start = divmod(state["phase_step"], nb)
        state["pass_idx"], state["batch_in_pass"] = pidx, start
        t = time.time()
        for x, _ in st.loader(1000 + pidx, start):
            wait = time.time() - t
            f, dt, r = one(x, False)
            write(f, dt, wait, r)
            periodic("warmup")
            if stopped():
                return
            if a.stop_at_warmup and state["phase_step"] >= a.stop_at_warmup:
                steplog.flush()
                log("CHECK_DONE warmup")
                return
            if state["phase_step"] >= a.warmup_steps:
                break
            t = time.time()
    if state["phase"] == "warmup":
        steplog.flush()
        if not a.no_reports:
            report(st, sets, "warmup_end", run, moments, False)
        L.set_lora_trainable(st.learner, True)
        opt = st.opt_main()  # a fresh optimiser for the main phase, as Sid's
        state.update(phase="main", phase_step=0, pass_idx=1, batch_in_pass=0)
        state["seq"] += 1
        save(run / f"save_{state['seq']:04d}_warmup_end", st, opt, state)
        log("main phase")
    elif (
        state["pass_idx"] > 1
        and state["batch_in_pass"] == 0
        and not a.no_reports
        and not (run / f"report_pass_{state['pass_idx'] - 1:02d}.json").exists()
    ):
        report(
            st, sets, f"pass_{state['pass_idx'] - 1:02d}", run, moments, False
        )  # resumed from a pass-end save whose report was cut off
        st.learner.train()
        st.guesser.train()
    while state["pass_idx"] <= a.passes:
        t = time.time()
        for x, _ in st.loader(state["pass_idx"], state["batch_in_pass"]):
            wait = time.time() - t
            f, dt, r = one(x, True)
            write(f, dt, wait, r)
            periodic("main")
            if stopped():
                return
            if a.stop_at_main and state["phase_step"] >= a.stop_at_main:
                steplog.flush()
                log("CHECK_DONE main")
                return
            t = time.time()
        steplog.flush()
        p = state["pass_idx"]
        state["seq"] += 1
        state["batch_in_pass"] = 0
        state["pass_idx"] = p + 1
        save(run / f"save_{state['seq']:04d}_pass_{p:02d}", st, opt, state)
        if not a.no_reports:
            report(st, sets, f"pass_{p:02d}", run, moments, False)
            st.learner.train()
            st.guesser.train()
    (run / "TRAIN_DONE").write_text(time.strftime("%F %T"))
    log("TRAIN_DONE")


if __name__ == "__main__":
    try:
        main()
    except StopRun as e:
        log(f"STOP: {e}")
        sys.exit(2)
