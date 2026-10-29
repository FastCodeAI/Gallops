"""Phase 1 rerun: generator only, lr 3e-5, with and without a stay-near-original term.

    setsid nohup ./run.sh src/rerun.py all > logs/rerun.log 2>&1 &     # preflight, G1, G2, runs w = 0, 0.3, 1.0

Everything not listed here is 06_generator_only_fast_rate unchanged (imported): model build, data, GridMask
handling, evaluation path, per-moment metrics.
- lr 3e-5 (was 1e-4).
- Stay-near-original term.  A frozen copy of the starting generator (hist_encoding,
  init_feature, the shared refiner) runs, without gradient, on the SAME image features the
  trained generator gets in that step: a forward hook on _backbone catches its output, so the
  GridMask draw is shared.  On each refinement pass the winner is recomputed exactly as in
  upstream's human term (drive_jepa_agent.py:228-233: L1 over x, y, heading, mean over the 8
  waypoints, min over the 32); for the other 31 the same distance is taken to the same-index
  candidate of the starting menu, averaged over the 31 and the batch.  Passes are combined
  like upstream (tl = prev_weight * tl + term: 0.001 / 0.01 / 0.1 / 1).
  Loss = upstream trajectory_loss_anchors (unchanged, empty vocabulary lists) + w * stay.
- A check (full evaluation on the 1,763 picking moments) at step 0, mid-pass and end of pass.
  Each run restarts from the starting generator with seed 0, so the three runs see the same
  batches and the same GridMask draws; only w differs.
"""

from __future__ import annotations

import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cache as K           # s2 (experiments/drivejepa_kitscenes)
import evalsets as E        # s2 (experiments/drivejepa_kitscenes)
import g0_gpu as G0         # 05_generator_survey
import kitdata as DT        # 06_generator_only_fast_rate
import phase1 as P1         # 06_generator_only_fast_rate

HERE = Path(__file__).resolve().parent.parent
OUT, LOGS = HERE / "out", HERE / "logs"
S1 = HERE.parent / "06_generator_only_fast_rate"
S1DATA = DT.DATA                                    # moments.pt, scene_meta.pt, s1 dumps (read only)
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s2")
LR, BATCH, SEED = 3e-5, 32, 0
WEIGHTS = (0.0, 0.3, 1.0)
MAX_PASSES, TURN_FAILS, NO_IMPROVE = 10, 2, 6
SLOW_V, TURN_RAD = 15.0, 0.3

log = P1.log


def wj(name: str, obj) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(OUT / name)


# --------------------------------------------------------------------------- #
# Model: s1 Model + frozen starting generator + stay term
# --------------------------------------------------------------------------- #


class Model(P1.Model):
    def __init__(self):
        super().__init__()
        m = self.m
        gen_mods = (m.hist_encoding, m.init_feature, m._trajectory_head)
        # train() vs eval() must not change the generator, so the frozen copy (eval) and the
        # trained one (train mode) compute the same function at step 0.
        bad = [n for g in gen_mods for n, x in g.named_modules()
               if (isinstance(x, torch.nn.Dropout) and x.p > 0) or isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
        assert not bad, bad
        self.f_hist = copy.deepcopy(m.hist_encoding).eval()
        self.f_init = copy.deepcopy(m.init_feature).eval()
        self.f_head = copy.deepcopy(m._trajectory_head).eval()     # deepcopy keeps the 4 entries one shared refiner
        assert len({id(r) for r in self.f_head}) == 1 and len({id(r) for r in m._trajectory_head}) == 1
        for mod in (self.f_hist, self.f_init, self.f_head):
            for p in mod.parameters():
                p.requires_grad_(False)
        self._img = None
        m._backbone.register_forward_hook(self._bk_hook)
        self.w = 0.0

    def _bk_hook(self, mod, inp, out):
        self._img = out

    def frozen_menu(self, image_feature, ego_status) -> list:
        """drive_jepa_model.py:59-68 with the frozen copy."""
        with torch.no_grad():
            bev = self.f_hist(ego_status)[:, None] + self.f_init.weight[None]
            pl = []
            for refine in self.f_head:
                bev, prop = refine(bev, image_feature)
                pl.append(prop)
        return pl

    def losses(self, feats: dict, target: torch.Tensor) -> dict:
        ego = feats["ego_status"][:, -1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.m(feats)
            img, self._img = self._img, None
            fl = self.frozen_menu(img, ego)
        pl = [q.float() for q in out["proposal_list"]]
        fl = [q.float() for q in fl]
        B = target.shape[0]
        empty = [np.zeros(0, dtype=np.int64)] * B
        up = self.agent.trajectory_loss_anchors(pl, target, self.agent._config, empty)
        cfg = self.agent._config
        stay, human, stay_list, human_list = 0, 0, [], []
        for p, f, _ in zip(pl, fl, empty):          # same pass count as upstream's zip (min(4, B))
            d_h = torch.linalg.norm(p - target[:, None], dim=-1, ord=1).mean(-1)      # (B, 32)
            win = d_h.argmin(1)
            h = d_h.gather(1, win[:, None]).squeeze(1).mean()
            d_s = torch.linalg.norm(p - f, dim=-1, ord=1).mean(-1)                    # (B, 32)
            keep = torch.ones_like(d_s, dtype=torch.bool)
            keep[torch.arange(B, device=d_s.device), win] = False
            s = d_s[keep].reshape(B, -1).mean(1).mean()
            human = cfg.prev_weight * human + h
            stay = cfg.prev_weight * stay + s
            human_list.append(h); stay_list.append(s)
        return {"upstream": up[0], "upstream_min_list": up[3], "human": human, "human_list": human_list,
                "stay": stay, "stay_list": stay_list, "total": up[0] + self.w * stay}


# --------------------------------------------------------------------------- #
# Splits and per-check summary
# --------------------------------------------------------------------------- #


def splits(main_set) -> dict:
    speed, turn3 = {}, {}
    for m in main_set:
        k = P1.key(m)
        speed[k] = float(np.linalg.norm(np.asarray(m["ego_status"])[-1, 3:5]))     # ego velocity (x, y)
        turn3[k] = abs(float(np.asarray(m["future_poses"])[5, 2]))                    # heading at 3 s
    slow = sorted(k for k in speed if speed[k] < SLOW_V)
    return {"ALL": sorted(speed), "SLOW": slow, "FAST": sorted(k for k in speed if speed[k] >= SLOW_V),
            "SLOW_TURNS": [k for k in slow if turn3[k] > TURN_RAD]}


def summarise(per: dict, start_per: dict, sp: dict, cont_keys) -> dict:
    def mm(keys, nm):
        v = [per[k][nm] for k in keys]
        return {"mean": float(np.mean(v)), "median": float(np.median(v))}
    r = {g: {"m1b": mm(ks, "m1b")} for g, ks in sp.items()}
    for nm in ("m1a", "min_fde3", "m2_lt1", "m2_lt2", "m3_variety", "m5_drift", "m6_pick_ade"):
        r["ALL"][nm] = mm(sp["ALL"], nm)
    for nm in ("m1a", "m2_lt2", "m3_variety", "m5_drift", "m6_pick_ade"):
        r["SLOW"][nm] = mm(sp["SLOW"], nm)
    r["SLOW_TURNS"]["m3_variety"] = mm(sp["SLOW_TURNS"], "m3_variety")
    for g in ("SLOW", "SLOW_TURNS", "FAST"):
        d = np.array([per[k]["m1b"] - start_per[k]["m1b"] for k in sp[g]])
        r[g]["better_gt_10cm"] = int((d < -0.1).sum()); r[g]["worse_gt_10cm"] = int((d > 0.1).sum())
    r["CONT_m1b_mean"] = float(np.mean([per[k]["m1b"] for k in cont_keys]))
    return r


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_g1(model, main_set, meta, human, cont_keys) -> tuple:
    s0 = json.loads((S0OUT_FILE).read_text())["menu"]
    start = P1.evaluate(model, main_set, meta)
    s103 = {k: start[k] for k in cont_keys}
    h103 = {k: human[k] for k in cont_keys}
    with open(S1DATA / "dumps" / "original.pkl", "rb") as fh:
        orig = pickle.load(fh)
    _, o_per = G0.menu_metrics({k: orig[k] for k in cont_keys}, h103)
    base_best = {k: o_per[k]["best_ade"] for k in o_per}
    s_agg, _ = G0.menu_metrics(s103, h103, base_best)
    rt = lambda x: json.loads(json.dumps(x, default=str))
    exact_103 = rt(s_agg) == s0["pass3_merged"]
    s1g1 = json.loads((S1 / "out" / "passes.json").read_text())["G1"]
    worst = json.loads((S1 / "out" / "worst20_main.json").read_text())["moments"]
    start_main = P1.aggregate(P1.per_moment(start, human, start), worst)
    exact_1763 = rt(start_main) == s1g1["start_main"]
    with open(S1DATA / "dumps" / "start.pkl", "rb") as fh:
        s1start = pickle.load(fh)
    bit = set(s1start) == set(start) and all(
        np.array_equal(s1start[k]["proposals"], start[k]["proposals"]) and s1start[k]["chosen"] == start[k]["chosen"]
        for k in start)
    with open(DATA / "dumps" / "start.pkl", "wb") as fh:
        pickle.dump(start, fh)
    res = {"start_on_103_equals_s0_g_menu": exact_103, "start_on_1763_equals_s1_start_row": exact_1763,
           "start_menus_bit_identical_to_s1_start_dump": bit, "pass": exact_103 and exact_1763 and bit}
    return res, start, orig, worst


def gate_g2(model, train, meta) -> dict:
    saved = model.gen_state()
    ds = P1.TrainSet(train, meta)
    rng = torch.Generator().manual_seed(12345)                     # not the training stream
    idx = torch.randperm(len(train), generator=rng)[:6 * BATCH].tolist()
    batches = [torch.utils.data.default_collate([ds[i] for i in idx[b * BATCH:(b + 1) * BATCH]]) for b in range(6)]
    fx, ft = P1.to_gpu(dict(batches[0]))
    clone = lambda: {k: v.clone() for k, v in fx.items()}
    res = {}
    # (a) + (b) at step 0, in training mode (GridMask may act) and in eval mode
    for mode in ("train", "eval"):
        model.reset_gm()
        model.train_mode() if mode == "train" else model.m.eval()
        with torch.no_grad():
            L = model.losses(clone(), ft)
        a_list = [bool(torch.equal(h, u)) for h, u in zip(L["human_list"], L["upstream_min_list"])]
        res[f"step0_{mode}"] = {
            "gridmask_applied": model.gm["applied"],
            "a_winner_human_term_equals_upstream_per_pass": a_list,
            "a_combined_equals_upstream_total": bool(torch.equal(L["human"], L["upstream"])),
            "upstream_total": float(L["upstream"]), "recomputed_total": float(L["human"]),
            "b_stay_term_per_pass": [float(s) for s in L["stay_list"]], "b_stay_term": float(L["stay"])}
    # (c) gradients: total loss at w = 1 at step 0, then stay term alone after 5 steps
    model.w = 1.0
    others = [(n, p) for n, p in model.m.named_parameters() if not n.startswith(P1.GEN_PREFIXES)] + model.fixed
    frozen = [p for mod in (model.f_hist, model.f_init, model.f_head) for p in mod.parameters()]
    f_before = [p.detach().clone() for p in frozen]

    def grad_report(loss):
        loss.backward()
        r = {"nonzero_trainable": sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for _, p in model.train_params),
             "any_grad_encoder_projector_scorer_grids": sum(int(p.grad is not None) for _, p in others),
             "any_grad_frozen_copy": sum(int(p.grad is not None) for p in frozen)}
        for _, p in model.train_params:
            p.grad = None
        return r
    model.train_mode()
    res["c_step0_total_w1"] = grad_report(model.losses(clone(), ft)["total"])
    opt = torch.optim.Adam([p for _, p in model.train_params], lr=LR)
    torch.cuda.reset_peak_memory_stats()
    times, stays = [], []
    for b in batches[1:]:
        feats, t = P1.to_gpu(dict(b))
        model.train_mode()
        torch.cuda.synchronize(); t0 = time.time()
        L = model.losses(feats, t)
        L["total"].backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        stays.append(float(L["stay"]))
    model.train_mode()
    L = model.losses(clone(), ft)
    res["c_after5_stay_only"] = grad_report(L["stay"])
    res["c_after5_stay_value"] = float(L["stay"])
    res["smoke_stay_values"] = stays
    res["frozen_copy_unchanged"] = all(torch.equal(a, p) for a, p in zip(f_before, frozen))
    res["peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["s_per_step"] = round(float(np.mean(times[1:])), 3)
    model.load_gen_state(saved)
    model.w = 0.0
    res["generator_restored_bit_exact"] = all(torch.equal(saved[n], p.detach().cpu()) for n, p in model.train_params)
    n = len(model.train_params)
    ok_c = all(res[c]["nonzero_trainable"] == n and res[c]["any_grad_encoder_projector_scorer_grids"] == 0
               and res[c]["any_grad_frozen_copy"] == 0 for c in ("c_step0_total_w1", "c_after5_stay_only"))
    ok_ab = all(all(res[s]["a_winner_human_term_equals_upstream_per_pass"]) and res[s]["a_combined_equals_upstream_total"]
                and res[s]["b_stay_term"] == 0.0 for s in ("step0_train", "step0_eval"))
    res["pass"] = ok_ab and ok_c and res["c_after5_stay_value"] > 0 and res["frozen_copy_unchanged"] \
        and res["generator_restored_bit_exact"]
    return res


S0OUT_FILE = HERE.parent / "05_generator_survey" / "out" / "g_menu.json"


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def run(model, w, train, main_set, meta, human, start, start_per, sp, cont_keys, start_sum) -> dict:
    tag = f"w{w:g}"
    ck = DATA / tag / "ckpt"; dd = DATA / tag / "dumps"
    ck.mkdir(parents=True, exist_ok=True); dd.mkdir(parents=True, exist_ok=True)
    model.load_gen_state(START_GEN)
    model.w = w
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    opt = torch.optim.Adam([p for _, p in model.train_params], lr=LR, weight_decay=0)
    shuf = torch.Generator().manual_seed(SEED)
    perms = [torch.randperm(len(train), generator=shuf).tolist() for _ in range(MAX_PASSES)]
    ds = P1.TrainSet(train, meta)
    turn_ref = start_sum["SLOW_TURNS"]["m1b"]["mean"]
    rec = {"w": w, "checks": [], "start_slow_turn_1b": turn_ref}
    s0c = {"pass": 0.0, "step": 0, "summary": start_sum, "turning_ok": True}
    rec["checks"].append(s0c)
    best_slow, since, turn_fail_run = start_sum["SLOW"]["m1b"]["mean"], 0, 0
    gstep = 0

    def check(progress):
        nonlocal best_slow, since, turn_fail_run
        te = time.time()
        dump = P1.evaluate(model, main_set, meta)
        per = P1.per_moment(dump, human, start)
        s = summarise(per, start_per, sp, cont_keys)
        name = f"p{progress:04.1f}"
        torch.save({"generator": model.gen_state(), "w": w, "progress_passes": progress, "step": gstep,
                    "settings": {"lr": LR, "batch": BATCH, "seed": SEED}}, ck / f"{name}.pt")
        with open(dd / f"{name}.pkl", "wb") as fh:
            pickle.dump(dump, fh)
        ok = s["SLOW_TURNS"]["m1b"]["mean"] <= turn_ref
        turn_fail_run = 0 if ok else turn_fail_run + 1
        if s["SLOW"]["m1b"]["mean"] < best_slow:
            best_slow, since = s["SLOW"]["m1b"]["mean"], 0
        else:
            since += 1
        c = {"pass": progress, "step": gstep, "summary": s, "turning_ok": ok, "eval_s": round(time.time() - te, 1),
             "train_loss_recent": float(np.mean(losses[-50:])) if losses else None,
             "stay_recent": float(np.mean(stays[-50:])) if stays else None}
        rec["checks"].append(c)
        log(f"[{tag}] pass {progress:.1f}: SLOW 1b {s['SLOW']['m1b']['mean']:.4f} TURN 1b {s['SLOW_TURNS']['m1b']['mean']:.4f} "
            f"(start {turn_ref:.4f}) FAST 1b {s['FAST']['m1b']['mean']:.3f} ALL 1b {s['ALL']['m1b']['mean']:.4f} "
            f"SLOW var {s['SLOW']['m3_variety']['mean']:.3f} drift {s['ALL']['m5_drift']['mean']:.3f} "
            f"loss {c['train_loss_recent']} stay {c['stay_recent']}")
        if turn_fail_run >= TURN_FAILS:
            return f"turning check failed at {TURN_FAILS} checks in a row (pass {progress:.1f})"
        if since >= NO_IMPROVE:
            return f"no SLOW 1b improvement for {NO_IMPROVE} checks (pass {progress:.1f})"
        return None

    losses, stays = [], []
    stop = None
    for pi in range(1, MAX_PASSES + 1):
        perm = perms[pi - 1]
        batches = [perm[i:i + BATCH] for i in range(0, len(perm) - BATCH + 1, BATCH)]   # drop_last
        mid = (len(batches) + 1) // 2
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=6, pin_memory=True,
                                             prefetch_factor=4, persistent_workers=False)
        model.reset_gm()
        for step, b in enumerate(loader, 1):
            feats, t = P1.to_gpu(b)
            model.train_mode()
            L = model.losses(feats, t)
            if not torch.isfinite(L["total"]):
                raise SystemExit(f"[{tag}] non-finite loss at pass {pi} step {step}")
            L["total"].backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            gstep += 1
            losses.append(float(L["total"])); stays.append(float(L["stay"]))
            if step % 100 == 0:
                ram = P1.avail_ram_gb()
                if ram < 20:
                    raise SystemExit(f"available RAM {ram:.1f} GB < 20 GB: stopping")
            if step == mid:
                stop = check(pi - 0.5)
                if stop:
                    break
        if stop:
            break
        rec.setdefault("gridmask_per_pass", []).append(dict(model.gm))
        stop = check(float(pi))
        wj(f"run_{tag}.json", rec)
        if stop:
            break
    rec["stopped"] = stop or f"reached {MAX_PASSES} passes"
    ok = [c for c in rec["checks"] if c["turning_ok"] and c["pass"] > 0]
    rec["best"] = min(ok, key=lambda c: (c["summary"]["SLOW"]["m1b"]["mean"], c["pass"]))["pass"] if ok else None
    wj(f"run_{tag}.json", rec)
    log(f"[{tag}] stopped: {rec['stopped']}; best check pass {rec['best']}")
    return rec


START_GEN: dict = {}


def main() -> None:
    (DATA / "dumps").mkdir(parents=True, exist_ok=True)
    state = {"preflight": P1.preflight()}
    log(f"preflight {state['preflight']}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    meta = torch.load(S1DATA / "scene_meta.pt", weights_only=False)
    mom = torch.load(S1DATA / "moments.pt", weights_only=False)
    train, main_set = mom["train"], mom["selection"]
    assert len(train) == 13937 and len(main_set) == 1763
    human = {P1.key(m): m["future_poses"] for m in main_set}
    cont_keys = sorted((e["scene_id"], e["keyframe"]) for e in E.load_selection_features())
    assert len(cont_keys) == 103 and set(cont_keys) <= set(human)
    sp = splits(main_set)
    state["splits"] = {g: len(v) for g, v in sp.items()}
    state["slow_turns_below_50"] = len(sp["SLOW_TURNS"]) < 50
    log(f"splits {state['splits']}")

    model = Model()
    START_GEN.update(model.gen_state())
    state["model"] = {"trainable": model.n_train, "trainable_tensors": len(model.train_params)}
    assert model.n_train == 2_381_123

    g1, start, orig, worst = gate_g1(model, main_set, meta, human, cont_keys)
    state["G1"] = g1
    wj("state.json", state)
    log(f"G1 {g1}")
    if not g1["pass"]:
        raise SystemExit("G1 failed: stopping")
    g2 = gate_g2(model, train, meta)
    state["G2"] = g2
    wj("state.json", state)
    log(f"G2 pass={g2['pass']} {json.dumps(g2)}")
    if not g2["pass"]:
        raise SystemExit("G2 failed: stopping")

    start_per = P1.per_moment(start, human, start)
    state["start_summary"] = summarise(start_per, start_per, sp, cont_keys)
    state["original_summary"] = summarise(P1.per_moment(orig, human, start), start_per, sp, cont_keys)
    wj("state.json", state)
    for w in WEIGHTS:
        rec = run(model, w, train, main_set, meta, human, start, start_per, sp, cont_keys, state["start_summary"])
        state.setdefault("runs", {})[f"w{w:g}"] = {"stopped": rec["stopped"], "best": rec["best"]}
        wj("state.json", state)
    log("all runs done")


if __name__ == "__main__":
    assert sys.argv[1] == "all"
    main()
