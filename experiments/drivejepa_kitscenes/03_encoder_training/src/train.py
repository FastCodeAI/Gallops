"""Parts D and E: warm-up (guesser only) then main training (LoRA + guesser).

    ./run.sh src/train.py warmup      # resumable every 250 steps
    ./run.sh src/train.py main        # resumable at pass boundaries

Run detached (tmux).  The learner is the agent's own encoder, wrapped with LoRA
in place, so after each pass agent.forward runs with the current (unmerged)
LoRA exactly as s1 T2 did -- no weight copying between "training" and
"driving" models.  The answer key is a frozen deepcopy of that wrapped encoder.

Stops (and writes STOP reason) instead of working around: warm-up fixed-set loss
not lower at step 500 than at step 0; any non-finite loss; selection drift / ADE
identical to the zero-LoRA run after a full main pass.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import shutil
import sys
import time

import numpy as np
import torch

import cache as K
import common as C
import drift as D
import evalsets as E
import lora as L
import masks2 as M
import players as P
import splits as S
from s2common import CKPT, LOGS, OUT, SEED, SETTINGS, wait_for_gpu, write_json

RUN = C.kit_runner_module()
BATCH = SETTINGS["batch"]
RAMP = SETTINGS["optimizer"]["ramp_steps"]
LR_LORA = SETTINGS["optimizer"]["lora"]["lr"]
LR_GUESS = SETTINGS["optimizer"]["guesser"]["lr"]
WD_GUESS = SETTINGS["optimizer"]["guesser"]["weight_decay"]
MOMENTUM = SETTINGS["momentum"]
WARMUP_EVERY, WARMUP_WINDOW, WARMUP_MAX, WARMUP_MIN_GAIN = 250, 500, 3000, 0.01
MAX_PASSES, PATIENCE, KEEP_BEST = 10, 3, 3
PHASE_CODE = {"warmup": 1, "main": 2}


class StopRun(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def guesser_groups(guesser) -> list[dict]:
    """Upstream init_opt split (vjepa2/app/vjepa/utils.py:224-236): no weight
    decay on biases and 1-D tensors."""
    decay = [p for n, p in guesser.named_parameters() if "bias" not in n and p.ndim != 1]
    no_decay = [p for n, p in guesser.named_parameters() if "bias" in n or p.ndim == 1]
    return [{"params": decay, "lr": LR_GUESS, "weight_decay": WD_GUESS, "base_lr": LR_GUESS, "name": "guesser"},
            {"params": no_decay, "lr": LR_GUESS, "weight_decay": 0.0, "base_lr": LR_GUESS, "name": "guesser_nowd"}]


def set_lr(opt, phase_step: int) -> float:
    f = min(1.0, (phase_step + 1) / RAMP)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f
    return f


def grad_norm(params) -> float:
    gs = [p.grad for p in params if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs))))


def lora_state(enc) -> dict:
    return {n: p.detach().cpu().clone() for n, p in enc.named_parameters() if ".lora_" in n}


def load_lora_state(enc, sd: dict) -> None:
    params = dict(enc.named_parameters())
    assert set(sd) == {n for n in params if ".lora_" in n}
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].device))


class Setup:
    """Agent with LoRA-wrapped encoder (learner), answer key, guesser, data."""

    def __init__(self):
        self.data = S.load_splits()
        assert C.sha256_file(C.AGENT_CKPT) == C.AGENT_CKPT_SHA256
        assert C.sha256_file(C.E50) == C.E50_SHA256
        self.agent = RUN.build_agent(C.AGENT_CKPT, "drive_jepa_perception_based_agent")
        for p in self.agent.parameters():
            p.requires_grad_(False)
        self.learner = self.agent.get_submodule(E.ENC_PATH)
        wrapped = L.wrap(self.learner, L.RANK, L.ALPHA)
        assert len(wrapped) == 96
        torch.manual_seed(SEED)  # LoRA A init
        for m in L.lora_modules(self.learner).values():
            torch.nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
        self.key = copy.deepcopy(self.learner)
        for p in self.key.parameters():
            p.requires_grad_(False)
        self.key.eval()
        self.guesser = P.build_guesser(torch.load(C.E50, map_location="cpu", mmap=True)).cuda()
        self.gens = M.make()
        self.train_ds = K.CachedPairs(self.data["train"])
        self.fixed = E.FixedSet(self.data["selection"])
        self.sel = E.load_selection_features()
        with open(OUT / "c_selection_zero_dump.pkl", "rb") as fh:
            self.zero_dump = pickle.load(fh)
        self.zero_ade = json.loads((OUT / "c_reference.json").read_text())["C1"]["selection_before"]

    def loader(self, pass_idx: int, start_batch: int = 0):
        g = torch.Generator()
        g.manual_seed(SEED * 1000 + pass_idx)
        perm = torch.randperm(len(self.train_ds), generator=g).tolist()
        batches = [perm[i:i + BATCH] for i in range(0, len(perm), BATCH)]
        return len(batches), torch.utils.data.DataLoader(
            self.train_ds, batch_sampler=batches[start_batch:], num_workers=8, pin_memory=True,
            prefetch_factor=4, persistent_workers=False)

    def step(self, opt, x, phase: str, gstep: int, train_lora: bool) -> dict:
        masks = M.sample(self.gens, x.shape[0], M.step_rng(SEED, PHASE_CODE[phase], gstep), "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, per = P.game_loss(self.learner, self.key, self.guesser, x, masks)
        if not torch.isfinite(loss):
            raise StopRun(f"non-finite loss {float(loss)} at {phase} step {gstep}")
        loss.backward()
        gn_l = grad_norm(L.lora_parameters(self.learner)) if train_lora else 0.0
        gn_g = grad_norm(self.guesser.parameters())
        if not (math.isfinite(gn_l) and math.isfinite(gn_g)):
            raise StopRun(f"non-finite grad norm at {phase} step {gstep}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        if train_lora:
            L.momentum_update(self.key, self.learner, MOMENTUM)
        return {"loss": float(loss), "per_mask": per, "gn_lora": gn_l, "gn_guesser": gn_g,
                "hidden": [int(p.shape[1]) for _, p in masks]}


def save(path, payload: dict, extra_settings: dict) -> None:
    """Weights and settings.json written together (Gallops CLAUDE.md: the label
    is written in the same function that saves the weights)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp / "state.pt")
    write_json(tmp / "settings.json", {**SETTINGS, **extra_settings,
                                       "saved": time.strftime("%F %T"), "masks_cfgs": M.wide_cfgs()})
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


# --------------------------------------------------------------------------- #
# Part D
# --------------------------------------------------------------------------- #


def warmup() -> None:
    wait = wait_for_gpu(8192, "warm-up", log)
    st = Setup()
    L.set_lora_trainable(st.learner, False)
    for p in st.guesser.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(guesser_groups(st.guesser), betas=(0.9, 0.999), eps=1e-8)
    wdir = CKPT / "warmup_resume"
    state = {"step": 0, "curve": [], "gpu_wait_s": wait, "wall_s": 0.0}
    if (wdir / "state.pt").exists():
        ck = torch.load(wdir / "state.pt", map_location="cpu", weights_only=False)
        st.guesser.load_state_dict(ck["guesser"])
        opt.load_state_dict(ck["optimizer"])
        state = ck["state"]
        state["gpu_wait_s"] += wait
        log(f"resumed warm-up at step {state['step']}")
    if not state["curve"]:
        l0 = st.fixed.loss(st.learner, st.key, st.guesser)
        state["curve"].append({"step": 0, "fixed_loss": l0})
        log(f"warm-up step 0 fixed-set loss {l0:.5f}")
    n_batches = math.ceil(len(st.train_ds) / BATCH)
    steplog = open(LOGS / "warmup_steps.jsonl", "a")
    st.learner.train(); st.guesser.train()
    reason = None
    t_start = time.time() - state["wall_s"]
    while reason is None:
        pass_idx, start = divmod(state["step"], n_batches)
        _, loader = st.loader(1000 + pass_idx, start)
        t_data = time.time()
        for x, _ in loader:
            data_s = time.time() - t_data
            gstep = state["step"]
            f = set_lr(opt, gstep)
            t0 = time.time()
            r = st.step(opt, x.cuda(non_blocking=True), "warmup", gstep, train_lora=False)
            state["step"] = gstep + 1
            steplog.write(json.dumps({"step": gstep, "lr_factor": f, "step_s": round(time.time() - t0, 3),
                                      "data_wait_s": round(data_s, 3), **r}) + "\n")
            if state["step"] % WARMUP_EVERY == 0:
                lf = st.fixed.loss(st.learner, st.key, st.guesser)
                state["curve"].append({"step": state["step"], "fixed_loss": lf})
                log(f"warm-up step {state['step']} fixed-set loss {lf:.5f} (train loss {r['loss']:.4f})")
                steplog.flush()
                curve = {c["step"]: c["fixed_loss"] for c in state["curve"]}
                s = state["step"]
                if s == WARMUP_WINDOW and not curve[s] < curve[0]:
                    reason = f"STOP: fixed-set loss did not fall in the first 500 steps ({curve[0]:.5f} -> {curve[s]:.5f})"
                elif s >= WARMUP_WINDOW and (curve[s - WARMUP_WINDOW] - curve[s]) / curve[s - WARMUP_WINDOW] < WARMUP_MIN_GAIN:
                    reason = (f"rule: fixed-set loss improved {100 * (curve[s - WARMUP_WINDOW] - curve[s]) / curve[s - WARMUP_WINDOW]:.2f} % "
                              f"over steps {s - WARMUP_WINDOW}-{s} (< 1 %)")
                elif s >= WARMUP_MAX:
                    reason = f"cap: {WARMUP_MAX} steps"
                state["wall_s"] = time.time() - t_start
                save(wdir, {"guesser": st.guesser.state_dict(), "optimizer": opt.state_dict(), "state": state},
                     {"phase": "warmup", "step": state["step"]})
                if reason:
                    break
            t_data = time.time()
    steplog.close()
    state["wall_s"] = time.time() - t_start
    state["stop_reason"] = reason
    save(CKPT / "warmup_final", {"guesser": st.guesser.state_dict(), "state": state},
         {"phase": "warmup", "step": state["step"], "stop_reason": reason})
    write_json(OUT / "d_warmup.json", state)
    log(reason)
    if reason.startswith("STOP"):
        sys.exit(2)


# --------------------------------------------------------------------------- #
# Part E
# --------------------------------------------------------------------------- #


def evaluate(st: Setup) -> dict:
    st.agent.eval()
    dump, rows = E.selection_eval(st.agent, st.sel, RUN)
    st.learner.train()
    agg = E.aggregate(rows)
    dr = D.drift(st.zero_dump, dump)["summary"]
    return {"selection": agg, "drift_vs_zero": dr, "ade_list": [r["ade"] for r in rows], "dump": dump}


def main_phase() -> None:
    wait = wait_for_gpu(8192, "main", log)
    st = Setup()
    wu = torch.load(CKPT / "warmup_final" / "state.pt", map_location="cpu", weights_only=False)
    st.guesser.load_state_dict(wu["guesser"])
    L.set_lora_trainable(st.learner, True)
    for p in st.guesser.parameters():
        p.requires_grad_(True)
    lora_params = L.lora_parameters(st.learner)
    opt = torch.optim.AdamW([{"params": lora_params, "lr": LR_LORA, "weight_decay": 0.0, "base_lr": LR_LORA, "name": "lora"},
                             *guesser_groups(st.guesser)], betas=(0.9, 0.999), eps=1e-8)
    hist_path = OUT / "e_passes.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {"passes": [], "gpu_wait_s_restarts": []}
    hist["gpu_wait_s_restarts"].append(wait)
    gstep = 0
    if hist["passes"]:
        last = hist["passes"][-1]
        ck = torch.load(CKPT / f"pass_{last['pass']:02d}" / "state.pt", map_location="cpu", weights_only=False)
        load_lora_state(st.learner, ck["lora"])
        load_lora_state(st.key, ck["key_lora"])
        st.guesser.load_state_dict(ck["guesser"])
        opt.load_state_dict(ck["optimizer"])
        gstep = ck["global_step"]
        log(f"resumed after pass {last['pass']} (step {gstep})")
    n_batches = math.ceil(len(st.train_ds) / BATCH)
    steplog = open(LOGS / "main_steps.jsonl", "a")
    st.learner.train(); st.guesser.train()
    reason = None
    while reason is None:
        pidx = len(hist["passes"]) + 1
        if pidx > MAX_PASSES:
            reason = f"cap: {MAX_PASSES} passes"
            break
        t_pass = time.time()
        data_wait = 0.0
        _, loader = st.loader(pidx)
        t_data = time.time()
        losses = []
        for x, _ in loader:
            data_s = time.time() - t_data
            data_wait += data_s
            f = set_lr(opt, gstep)
            t0 = time.time()
            r = st.step(opt, x.cuda(non_blocking=True), "main", gstep, train_lora=True)
            steplog.write(json.dumps({"pass": pidx, "step": gstep, "lr_factor": f, "step_s": round(time.time() - t0, 3),
                                      "data_wait_s": round(data_s, 3), **r}) + "\n")
            losses.append(r["loss"])
            gstep += 1
            if gstep % 100 == 0:
                log(f"pass {pidx} step {gstep} loss {r['loss']:.4f} gn_lora {r['gn_lora']:.3g} gn_guesser {r['gn_guesser']:.3g}")
                steplog.flush()
            t_data = time.time()
        steplog.flush()
        train_s = time.time() - t_pass
        t_eval = time.time()
        fixed_loss = st.fixed.loss(st.learner, st.key, st.guesser)
        ev = evaluate(st)
        row = {"pass": pidx, "global_step": gstep, "steps": n_batches, "train_loss_mean": float(np.mean(losses)),
               "fixed_loss": fixed_loss, "selection": ev["selection"], "drift_vs_zero": ev["drift_vs_zero"],
               "train_s": round(train_s, 1), "data_wait_s": round(data_wait, 1), "eval_s": round(time.time() - t_eval, 1),
               "gpu_wait_s": round(wait, 1),
               "lora_B_norm": float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(
                   [p for n, p in st.learner.named_parameters() if n.endswith("lora_B")]))))}
        wait = 0.0
        log(f"pass {pidx}: fixed {fixed_loss:.5f}  sel ADE {row['selection']['ade']['mean']:.4f}  "
            f"drift {row['drift_vs_zero']}")
        dr = ev["drift_vs_zero"]
        if (dr["mean_proposal_shift_m"] == 0.0 and dr["mean_grade_change"] == 0.0) \
                or row["selection"]["ade"]["mean"] == st.zero_ade["ade"]["mean"]:
            reason = f"STOP: pass {pidx} drift/ADE identical to the zero-LoRA run (LoRA not reaching the model?)"
        with open(OUT / f"e_dump_pass_{pidx:02d}.pkl", "wb") as fh:
            pickle.dump(ev["dump"], fh)
        save(CKPT / f"pass_{pidx:02d}",
             {"lora": lora_state(st.learner), "key_lora": lora_state(st.key), "guesser": st.guesser.state_dict(),
              "optimizer": opt.state_dict(), "global_step": gstep, "pass": pidx, "row": row},
             {"phase": "main", "pass": pidx, "global_step": gstep, "selection_ade": row["selection"]["ade"]["mean"]})
        hist["passes"].append(row)
        # keep best 3 by selection ADE plus the latest
        ranked = sorted(hist["passes"], key=lambda r: r["selection"]["ade"]["mean"])
        keep = {r["pass"] for r in ranked[:KEEP_BEST]} | {pidx}
        for r in hist["passes"]:
            d = CKPT / f"pass_{r['pass']:02d}"
            if r["pass"] not in keep and d.exists():
                shutil.rmtree(d)
                r["checkpoint_deleted"] = True
        hist["kept"] = sorted(keep)
        best = ranked[0]
        hist["best_pass"] = best["pass"]
        write_json(hist_path, hist)
        if reason:
            break
        worse_streak = 0
        for r in reversed(hist["passes"]):
            if r["pass"] == best["pass"]:
                break
            worse_streak += 1
        if worse_streak >= PATIENCE:
            reason = f"rule: selection ADE worse than best (pass {best['pass']}) for {PATIENCE} passes in a row"
    steplog.close()
    hist["stop_reason"] = reason
    write_json(hist_path, hist)
    log(reason)
    if reason.startswith("STOP"):
        sys.exit(2)


if __name__ == "__main__":
    try:
        {"warmup": warmup, "main": main_phase}[sys.argv[1]]()
    except StopRun as exc:
        log(f"STOP: {exc}")
        write_json(OUT / f"STOP_{sys.argv[1]}.json", {"reason": str(exc), "time": time.strftime("%F %T")})
        sys.exit(2)
