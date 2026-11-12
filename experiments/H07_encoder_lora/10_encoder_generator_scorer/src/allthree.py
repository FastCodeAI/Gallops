"""Encoder LoRA, generator and scorer of Drive-JEPA trained together on KIT, in one run.

    setsid nohup ./run.sh src/allthree.py all > logs/allthree.log 2>&1 &
    ./run.sh src/allthree.py smoke                                        # wiring only, writes nothing

Start: release checkpoint (sha256-checked) + pass-3 encoder LoRA (unmerged, trainable), with the
guesser and answer key from the same pass-3 state.pt.  Generator and scorer are the RELEASED
weights, so step 0 is "Encoder only, tuned on KIT".

Trained (one backward of the summed loss; every optimiser steps):
  encoder LoRA  AdamW 1e-4, wd 0      <- generator loss + scorer loss + hide-and-guess
  guesser       AdamW 1e-4, wd 0.04 on weights, 0 on biases / 1-D (s2 split)
  answer key    momentum 0.99925 after every step
  generator     Adam 3e-5, 41 tensors (fixed grids frozen) <- upstream trajectory_loss_anchors, empty vocab lists
  pred_score    Adam 1e-4, whole head (its 6 logits share both hidden layers, G0b) <- scorer loss
Frozen: projector, pred_col_agent, pred_area, the rest of the scorer, encoder base weights.

Scorer loss (new: KIT has no NAVSIM grades).  d_k = mean over waypoints 1-6 (0.5-3 s) of the xy
distance of final candidate k to the human path; target = softmax(-d / 0.5 m) over the 32;
prediction = softmax over the 32 of the last pred_score logit (before sigmoid); cross-entropy,
averaged over the batch.

Why a replay.  The scorer reads only the generator's final query state (scorer.py:41), never image
features directly (G0b).  It gets that state from a replay of the 4 refinement passes on the same
image features, with detached copies of the generator weights, the ego input and learned queries
computed without gradient, and the trained forward's candidates detached.  Image features stay
attached, so the scorer loss reaches the LoRA but never the generator weights.  G2 checks that the
replayed state equals the trained forward's bit for bit.

Pick rule: upstream's, unchanged (argmax of sigmoid of the last logit, drive_jepa_model.py:85-91; G0a).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cache as K           # experiments/drivejepa_kitscenes s2
import common as C          # experiments/drivejepa_kitscenes s1
import evalsets as E        # experiments/drivejepa_kitscenes s2
import f_final as FF        # experiments/drivejepa_kitscenes s2 (merge)
import g0_gpu as G0         # 05_generator_survey
import kitdata as DT        # 06_generator_only_fast_rate
import lora as L            # experiments/drivejepa_kitscenes s1
import masks2 as M          # experiments/drivejepa_kitscenes s2
import phase1 as P1         # 06_generator_only_fast_rate
import phase2 as Q          # 08_generator_and_encoder_together (streams, drift, tails, evaluation context)
import players as P         # experiments/drivejepa_kitscenes s1
import rerun as R           # 07_generator_only_gentle_rate
from s2common import CKPT as ENC_CKPT

HERE = Path(__file__).resolve().parent.parent
OUT, LOGS = HERE / "out", HERE / "logs"
GJK = HERE.parent
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s5")
SEED, BATCH = 0, 32
LR_GEN, LR_SCORE, LR_LORA, LR_GUESS, WD_GUESS = 3e-5, 1e-4, 1e-4, 1e-4, 0.04
MOMENTUM, GAME_W, TAU, N_WP = 0.99925, 1.0, 0.5, 6
MAX_PASSES, HARD_FAILS, NO_IMPROVE = 10, 2, 6
N_LORA, N_SCORE = 192, 10
GROUPS = ("ALL", "SLOW", "FAST", "SLOW_TURNS")
METRICS = ("m1a", "m1b", "m6_pick_ade", "pick_fde1", "pick_fde2", "pick_fde3", "pick_gap", "m2_lt2", "m3_variety")
JOINT = "Generator and encoder trained together"

log = P1.log


def wj(name: str, obj) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(OUT / name)


def nonzero(ps) -> int:
    return sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in ps)


def anygrad(ps) -> int:
    return sum(int(p.grad is not None) for p in ps)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Model:
    def __init__(self, pass3_state: dict):
        self.agent = G0.build(pass3=True, merged=False)        # release (sha256 asserted) + pass-3 LoRA
        self.m = m = self.agent._pad_model
        self.enc = m._backbone.img_backbone
        gen = [(n, p) for n, p in m.named_parameters() if n.startswith(P1.GEN_PREFIXES)]
        assert sum(p.numel() for _, p in gen) == P1.GEN_TOTAL
        self.fixed = [(n, p) for n, p in gen if n.endswith(P1.FIXED_SUFFIXES)]
        self.train_params = [(n, p) for n, p in gen if not n.endswith(P1.FIXED_SUFFIXES)]
        assert sum(p.numel() for _, p in self.train_params) == 2_381_123 and len(self.train_params) == 41
        for _, p in self.train_params:
            p.requires_grad_(True)
        self.score = [(n, p) for n, p in m.named_parameters() if n.startswith("scorer.pred_score.")]
        assert len(self.score) == N_SCORE
        for _, p in self.score:
            p.requires_grad_(True)
        heads = m._trajectory_head
        assert len(heads) == 4 and len({id(h) for h in heads}) == 1     # one shared refiner (drive_jepa_model.py:35-36)
        self.bev_refiner = heads[0].Bev_refiner
        # LoRA, answer key, guesser exactly as 08_generator_and_encoder_together Model(joint=True)
        params = dict(self.enc.named_parameters())
        assert all(torch.equal(params[n].cpu(), t) for n, t in pass3_state["lora"].items())
        self.lora = L.lora_parameters(self.enc)
        assert len(self.lora) == N_LORA
        L.set_lora_trainable(self.enc, True)
        self.key = copy.deepcopy(self.enc).eval()
        for p in self.key.parameters():
            p.requires_grad_(False)
        Q.Model.load_lora(self.key, pass3_state["key_lora"])
        self.guesser = P.build_guesser(torch.load(C.E50, map_location="cpu", mmap=True)).cuda()
        self.guesser.load_state_dict(pass3_state["guesser"])
        for p in self.guesser.parameters():
            p.requires_grad_(True)
        self.n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

        # hooks are closures over dicts, not bound methods: FF.merge deep-copies the agent with them
        self.gm = gm = {"calls": 0, "applied": 0, "train_flag_calls": 0, "last_applied": 0}
        self.cap = cap = {"on": False}
        bk = m._backbone

        def gm_hook(mod, inp, out):
            gm["calls"] += 1
            gm["train_flag_calls"] += int(mod.training)
            gm["last_applied"] = int(not torch.equal(inp[0], out))
            gm["applied"] += gm["last_applied"]

        def img_hook(mod, inp, out):
            if cap["on"]:
                cap["img"] = out

        def scorer_pre(mod, args):
            if cap["on"]:
                cap["bev"] = args[1]
        bk.grid_mask.register_forward_hook(gm_hook)
        bk.register_forward_hook(img_hook)
        m.scorer.register_forward_pre_hook(scorer_pre)

    # --- state -------------------------------------------------------------- #
    def full_state(self) -> dict:
        g = lambda ps: {n: p.detach().cpu().clone() for n, p in ps}
        return {"generator": g(self.train_params), "pred_score": g(self.score),
                "lora": Q.Model.lora_state(self.enc), "key_lora": Q.Model.lora_state(self.key),
                "guesser": {k: v.detach().cpu().clone() for k, v in self.guesser.state_dict().items()}}

    def load_full(self, s: dict) -> None:
        with torch.no_grad():
            for key, ps in (("generator", self.train_params), ("pred_score", self.score)):
                cur = dict(ps)
                assert set(s[key]) == set(cur)
                for n, t in s[key].items():
                    cur[n].copy_(t.to(cur[n].device))
        Q.Model.load_lora(self.enc, s["lora"]); Q.Model.load_lora(self.key, s["key_lora"])
        self.guesser.load_state_dict(s["guesser"])

    # --- training ----------------------------------------------------------- #
    def train_mode(self) -> None:
        """08_generator_and_encoder_together: whole model eval; GridMask, generator and guesser train.  The scorer stays in
        eval, so pred_col_agent / pred_area / map heads are not even run (scorer.py:46);
        pred_score has no dropout or batch norm, so its mode does not matter."""
        self.m.eval()
        self.m._backbone.grid_mask.train()
        for mod in (self.m.hist_encoding, self.m.init_feature, self.m._trajectory_head):
            mod.train()
        self.guesser.train()

    def replay(self, ego, img, proposal_list):
        """The scorer's input recomputed: drive_jepa_model.py:61-67 with every generator input but
        the image features detached.  Bev_refiner gets the trained forward's candidates (traj_refiner.py:24-27;
        it detaches them itself, bev_refiner.py:114)."""
        det = {n: p.detach() for n, p in self.bev_refiner.named_parameters()}
        with torch.no_grad():
            bev = self.m.hist_encoding(ego)[:, None] + self.m.init_feature.weight[None]
        for prop in proposal_list:
            bev = torch.func.functional_call(self.bev_refiner, det, (prop.detach(), bev, img))
        return bev

    def forward(self, feats, target, stream: Q.GridMaskStream | None) -> dict:
        ego = feats["ego_status"][:, -1]
        self.cap.clear(); self.cap["on"] = True
        ctx = stream if stream is not None else _Null()
        with ctx:
            before = np.random.get_state()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.m(feats)
                self.cap["on"] = False
                bev = self.replay(ego, self.cap["img"], out["proposal_list"])
                logit = self.m.scorer(out["proposals"].detach(), bev)[0]
            after = np.random.get_state()
        pl = [q.float() for q in out["proposal_list"]]
        empty = [np.zeros(0, dtype=np.int64)] * target.shape[0]
        gl = self.agent.trajectory_loss_anchors(pl, target, self.agent._config, empty)[0]
        sl, d, tgt, n_wp = scorer_loss(logit, out["proposals"], target)
        return {"gen": gl, "score": sl, "d": d, "target": tgt, "n_wp": n_wp, "proposals": out["proposals"],
                "logit": logit, "bev_replay": bev,
                "bev_trained": self.cap.pop("bev"), "logit_trained": out["pred_logit"],
                "np_state": (Q.np_state_hash(before), Q.np_state_hash(after)), "gm_applied": self.gm["last_applied"]}

    def game_loss(self, x, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return P.game_loss(self.enc, self.key, self.guesser, x, masks)

    def optimisers(self) -> list:
        gd = [p for n, p in self.guesser.named_parameters() if "bias" not in n and p.ndim != 1]
        gn = [p for n, p in self.guesser.named_parameters() if "bias" in n or p.ndim == 1]
        return [torch.optim.Adam([p for _, p in self.train_params], lr=LR_GEN, weight_decay=0),
                torch.optim.Adam([p for _, p in self.score], lr=LR_SCORE, weight_decay=0),
                torch.optim.AdamW([{"params": self.lora, "lr": LR_LORA, "weight_decay": 0.0},
                                   {"params": gd, "lr": LR_GUESS, "weight_decay": WD_GUESS},
                                   {"params": gn, "lr": LR_GUESS, "weight_decay": 0.0}],
                                  betas=(0.9, 0.999), eps=1e-8)]

    def after_step(self) -> None:
        L.momentum_update(self.key, self.enc, MOMENTUM)

    def zero_grad(self) -> None:
        for p in list(self.m.parameters()) + list(self.guesser.parameters()) + list(self.key.parameters()):
            p.grad = None

    def eval_agent(self):
        return FF.merge(self.agent)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def scorer_loss(logit, proposals, target):
    p = proposals.detach().float()[:, :, :N_WP, :2]                       # B x 32 x 6 x xy (0.5-3 s)
    h = target[:, None, :N_WP, :2]
    d = torch.linalg.norm(p - h, dim=-1).mean(-1)                         # B x 32
    tgt = torch.softmax(-d / TAU, dim=1)
    logp = torch.log_softmax(logit[..., -1].float(), dim=1)               # last pred_score logit only
    return -(tgt * logp).sum(1).mean(), d, tgt, p.shape[2]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(model: Model, agent, moments, meta) -> dict:
    """08_generator_and_encoder_together evaluate, also keeping pred_logit (for the other-logit drift)."""
    fc = K.FrameCache()
    before = dict(model.gm)
    dump = {}
    for mo in moments:
        d = E.forward_dump(agent, DT.assemble(mo, meta[mo["scene_id"]], fc))
        dump[P1.key(mo)] = {k: d[k] for k in ("proposals", "pdm_score", "chosen", "pred_logit")}
    assert model.gm["applied"] == before["applied"] and model.gm["train_flag_calls"] == before["train_flag_calls"], \
        "GridMask active during evaluation"
    return dump


def per_moment(dump: dict, human: dict, start: dict | None) -> dict:
    """s1 phase1.per_moment unchanged, plus the pick's FDE at 1 / 2 / 3 s (waypoints 2, 4, 6) and the pick gap."""
    per = P1.per_moment(dump, human, start)
    for k, r in per.items():
        p = np.asarray(dump[k]["proposals"], dtype=np.float64)[int(dump[k]["chosen"]), :6, :2]
        e = np.linalg.norm(p - np.asarray(human[k], dtype=np.float64)[:6, :2], axis=-1)
        r.update(pick_fde1=float(e[1]), pick_fde2=float(e[3]), pick_fde3=float(e[5]), pick_gap=r["m6_pick_ade"] - r["m1a"])
    return per


def summary(per: dict, sp: dict, cont: list) -> dict:
    out = {g: {x: float(np.mean([per[k][x] for k in sp[g]])) for x in METRICS} for g in GROUPS}
    out["CONT103"] = {x: float(np.mean([per[k][x] for k in cont])) for x in METRICS}
    return out


def ties(dump: dict) -> int:
    return sum(int((np.asarray(d["pdm_score"]) == np.asarray(d["pdm_score"]).max()).sum() > 1) for d in dump.values())


def logit_drift(dump: dict, ref: dict, keys: list) -> dict:
    a = np.stack([np.asarray(dump[k]["pred_logit"], dtype=np.float64) for k in keys])
    b = np.stack([np.asarray(ref[k]["pred_logit"], dtype=np.float64) for k in keys])
    return {"other5_mean_abs_change": float(np.abs(a[..., :5] - b[..., :5]).mean()),
            "other5_mean_abs_change_per_logit": [float(v) for v in np.abs(a[..., :5] - b[..., :5]).mean((0, 1))],
            "last_mean_abs_change": float(np.abs(a[..., 5] - b[..., 5]).mean()),
            "last_mean_level": float(a[..., 5].mean()), "last_max": float(a[..., 5].max())}


# --------------------------------------------------------------------------- #
# Fixed batches (gates and the per-check gradient shares)
# --------------------------------------------------------------------------- #


def fixed_batches(ctx, n: int, seed: int = 12345):
    rng = torch.Generator().manual_seed(seed)                # not the training streams
    idx = torch.randperm(len(ctx.train), generator=rng)[:n * BATCH].tolist()
    ds = P1.TrainSet(ctx.train, ctx.meta)
    pairs = K.CachedPairs(ctx.data["train"])
    pidx = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed + 1))[:n * BATCH].tolist()
    gens = M.make()
    out = []
    for s in range(n):
        b = torch.utils.data.default_collate([ds[i] for i in idx[s * BATCH:(s + 1) * BATCH]])
        x = torch.utils.data.default_collate([pairs[i] for i in pidx[s * BATCH:(s + 1) * BATCH]])[0]
        out.append((b, x, M.sample(gens, BATCH, M.step_rng(seed, 2, s), "cuda")))
    return out


def lora_grad_shares(model: Model, fb) -> dict:
    """LoRA gradient norm from each loss alone on one fixed batch (no GridMask), and its share of the sum."""
    b, x, masks = fb
    feats, t = P1.to_gpu(dict(b))
    model.train_mode()
    model.m._backbone.grid_mask.eval()
    f = model.forward(feats, t, None)
    jl, _ = model.game_loss(x.cuda(), masks)
    norms = {}
    for name, loss in (("generator", f["gen"]), ("scorer", f["score"]), ("hide_and_guess", GAME_W * jl)):
        g = torch.autograd.grad(loss, model.lora, retain_graph=True, allow_unused=True)
        norms[name] = float(torch.sqrt(sum((gi.float() ** 2).sum() for gi in g if gi is not None)))
    tot = sum(norms.values())
    model.zero_grad()
    return {"norm": norms, "share": {k: v / tot for k, v in norms.items()},
            "losses": {"generator": float(f["gen"]), "scorer": float(f["score"]), "hide_and_guess": float(jl)}}


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_g1(ctx, dump0: dict) -> dict:
    """Step 0 = "Encoder only, tuned on KIT" (upstream pick rule)."""
    rt = lambda v: json.loads(json.dumps(v, default=str))
    bit = set(dump0) == set(ctx.start) and all(
        np.asarray(dump0[k][f]).tobytes() == np.asarray(ctx.start[k][f]).tobytes()
        for k in dump0 for f in ("proposals", "pdm_score", "chosen"))
    s0 = json.loads((GJK / "05_generator_survey" / "out" / "g_menu.json").read_text())["menu"]
    with open(DT.DATA / "dumps" / "original.pkl", "rb") as fh:
        orig = pickle.load(fh)
    h103 = {k: ctx.human[k] for k in ctx.cont}
    _, o_per = G0.menu_metrics({k: orig[k] for k in ctx.cont}, h103)
    base_best = {k: o_per[k]["best_ade"] for k in o_per}
    agg103, _ = G0.menu_metrics({k: dump0[k] for k in ctx.cont}, h103, base_best)
    s1g1 = json.loads((GJK / "06_generator_only_fast_rate" / "out" / "passes.json").read_text())["G1"]
    worst = json.loads((GJK / "06_generator_only_fast_rate" / "out" / "worst20_main.json").read_text())["moments"]
    start_main = P1.aggregate(P1.per_moment(dump0, ctx.human, dump0), worst)
    s2state = json.loads((GJK / "07_generator_only_gentle_rate" / "out" / "state.json").read_text())
    p0 = P1.per_moment(dump0, ctx.human, dump0)
    s2sum = R.summarise(p0, p0, ctx.sp, ctx.cont)
    res = {"menus_1763_bit_identical_to_s2_start_dump": bit,
           "103_equals_s0_g_menu_pass3_merged": rt(agg103) == s0["pass3_merged"],
           "103_equals_s0_g_menu_pass3_unmerged (record only)": rt(agg103) == s0["pass3_unmerged"],
           "1763_equals_s1_phase1_start_row": rt(start_main) == s1g1["start_main"],
           "1763_equals_s2_phase1_rerun_start_row": rt(s2sum) == s2state["start_summary"]}
    res["pass"] = all(v for k, v in res.items() if "record only" not in k)
    res["this_run_pick_rule"] = "same as upstream (G0a): step 0 under this run's rule is the same row"
    return res


def gate_g2(model: Model, fb) -> dict:
    """Scorer loss on one fixed batch, training mode; no optimiser step."""
    b, x, masks = fb
    feats, t = P1.to_gpu(dict(b))
    model.zero_grad()
    model.train_mode()
    f = model.forward(feats, t, Q.GridMaskStream(12345))
    res = {"gridmask_applied_on_this_batch": f["gm_applied"],
           "replay_bev_max_abs_diff": float((f["bev_replay"].float() - f["bev_trained"].float()).abs().max()),
           "replay_bev_bit_identical": bool(torch.equal(f["bev_replay"], f["bev_trained"])),
           "replay_logit_bit_identical": bool(torch.equal(f["logit"], f["logit_trained"])),
           "d_k_waypoints_used": int(f["n_wp"]),
           "target_row_sum_max_abs_err": float((f["target"].sum(1) - 1).abs().max()),
           "argmin_d_is_argmax_target_rows": f"{int((f['d'].argmin(1) == f['target'].argmax(1)).sum())} / {t.shape[0]}",
           "scorer_loss": float(f["score"]), "loss_finite": bool(torch.isfinite(f["score"]))}
    # d_k from waypoints 1-6 only: equal to an independent recomputation over 0.5-3 s, not over all 8
    pr = f["proposals"].detach().float()
    d6 = torch.linalg.norm(pr[:, :, :6, :2] - t[:, None, :6, :2], dim=-1).mean(-1)
    d8 = torch.linalg.norm(pr[:, :, :8, :2] - t[:, None, :8, :2], dim=-1).mean(-1)
    res["d_k_equals_6_waypoint_recompute"] = bool(torch.equal(d6, f["d"]))
    res["d_k_differs_from_8_waypoints"] = bool(not torch.equal(d8, f["d"]))
    named = dict(model.m.named_parameters())
    groups = {
        "generator": [p for _, p in model.train_params],
        "pred_score": [p for _, p in model.score],
        "lora": model.lora,
        "other_scorer_heads": [p for n, p in named.items() if n.startswith("scorer.") and not n.startswith("scorer.pred_score.")],
        "projector": [p for n, p in named.items() if n.startswith("_backbone.projector.")],
        "answer_key": list(model.key.parameters()),
        "guesser": list(model.guesser.parameters()),
        "encoder_base": [p for n, p in model.enc.named_parameters() if ".lora_" not in n],
        "fixed_grids": [p for _, p in model.fixed]}
    f["logit"].retain_grad()
    f["score"].backward(retain_graph=True)
    lg = f["logit"].grad.float()
    res["logit_grad_nonzero_columns"] = [int(bool(lg[..., c].abs().sum() > 0)) for c in range(lg.shape[-1])]
    res["scorer_loss_alone"] = {g: {"nonzero": nonzero(ps), "any": anygrad(ps), "total": len(ps)} for g, ps in groups.items()}
    model.zero_grad()
    f["gen"].backward()
    res["generator_loss_alone"] = {g: {"nonzero": nonzero(ps), "any": anygrad(ps), "total": len(ps)} for g, ps in groups.items()}
    model.zero_grad()
    jl, _ = model.game_loss(x.cuda(), masks)
    jl.backward()
    res["hide_and_guess_alone"] = {g: {"nonzero": nonzero(ps), "any": anygrad(ps), "total": len(ps)} for g, ps in groups.items()}
    model.zero_grad()
    del f
    # the same bit-identity with GridMask acting (it masks the images before the encoder, so it must not matter)
    for seed in range(1, 50):
        feats, t = P1.to_gpu(dict(b))            # fresh dict: the model forward slices lidar2img in place
        model.train_mode()
        g = model.forward(feats, t, Q.GridMaskStream(seed))
        if g["gm_applied"]:
            break
    res["with_gridmask_applied"] = {"stream_seed": seed, "applied": g["gm_applied"],
                                    "bev_bit_identical": bool(torch.equal(g["bev_replay"], g["bev_trained"])),
                                    "logit_bit_identical": bool(torch.equal(g["logit"], g["logit_trained"]))}
    del g
    model.zero_grad()
    s = res["scorer_loss_alone"]
    res["pass"] = (res["replay_bev_max_abs_diff"] == 0.0 and res["replay_bev_bit_identical"] and res["replay_logit_bit_identical"]
                   and res["target_row_sum_max_abs_err"] < 1e-6
                   and res["argmin_d_is_argmax_target_rows"] == f"{t.shape[0]} / {t.shape[0]}"
                   and res["loss_finite"] and res["logit_grad_nonzero_columns"] == [0, 0, 0, 0, 0, 1]
                   and res["d_k_waypoints_used"] == 6 and res["d_k_equals_6_waypoint_recompute"]
                   and res["d_k_differs_from_8_waypoints"]
                   and res["with_gridmask_applied"]["applied"] == 1 and res["with_gridmask_applied"]["bev_bit_identical"]
                   and res["with_gridmask_applied"]["logit_bit_identical"]
                   and s["pred_score"]["nonzero"] == N_SCORE and s["lora"]["nonzero"] == N_LORA
                   and all(s[g]["any"] == 0 for g in ("generator", "other_scorer_heads", "projector", "answer_key",
                                                       "guesser", "encoder_base", "fixed_grids")))
    return res


def gate_g3(model: Model, fbs: list) -> dict:
    """5 steps of the full loss on a throwaway optimiser; weights restored, seeds reset by the caller."""
    saved = model.full_state()
    opts = model.optimisers()
    stream = Q.GridMaskStream(12345)
    allowed = {f"mask_tokens.{i}" for i in range(1, 10)}       # the game uses mask token 0 only (08_generator_and_encoder_together G2)
    res = {"steps": []}
    torch.cuda.reset_peak_memory_stats()
    times = []
    # which loss gives the generator its gradient (first batch, before any step)
    b, x, masks = fbs[0]
    feats, t = P1.to_gpu(dict(b))
    model.train_mode()
    f = model.forward(feats, t, Q.GridMaskStream(777))
    jl, _ = model.game_loss(x.cuda(), masks)
    gen = [p for _, p in model.train_params]
    res["generator_grad_from"] = {}
    for name, loss in (("generator", f["gen"]), ("scorer", f["score"]), ("hide_and_guess", jl)):
        g = torch.autograd.grad(loss, gen, retain_graph=True, allow_unused=True)
        res["generator_grad_from"][name] = sum(int(gi is not None and bool(gi.abs().sum() > 0)) for gi in g)
    del f, jl
    model.zero_grad()
    for s, (b, x, masks) in enumerate(fbs):
        feats, t = P1.to_gpu(dict(b))
        model.train_mode()
        torch.cuda.synchronize(); t0 = time.time()
        f = model.forward(feats, t, stream)
        jl, _ = model.game_loss(x.cuda(non_blocking=True), masks)
        loss = f["gen"] + f["score"] + GAME_W * jl
        loss.backward()
        gz = [n for n, p in model.guesser.named_parameters() if not (p.grad is not None and bool(p.grad.abs().sum() > 0))]
        res["steps"].append({"gen_loss": float(f["gen"]), "scorer_loss": float(f["score"]), "game_loss": float(jl),
                             "generator_nonzero": nonzero(gen), "pred_score_nonzero": nonzero([p for _, p in model.score]),
                             "lora_nonzero": nonzero(model.lora), "guesser_zero_grad": gz,
                             "guesser_total": len(list(model.guesser.parameters()))})
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        model.after_step()
        model.zero_grad()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    res["peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["s_per_step_after_first"] = round(float(np.mean(times[1:])), 3)
    model.load_full(saved)
    now = model.full_state()
    res["restored_bit_exact"] = all(torch.equal(saved[g][k], now[g][k]) for g in saved for k in saved[g])
    gf = res["generator_grad_from"]
    res["pass"] = (gf["generator"] == 41 and gf["scorer"] == 0 and gf["hide_and_guess"] == 0
                   and all(r["generator_nonzero"] == 41 and r["lora_nonzero"] == N_LORA and r["pred_score_nonzero"] == N_SCORE
                           and set(r["guesser_zero_grad"]) <= allowed for r in res["steps"])
                   and res["restored_bit_exact"])
    return res


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def reference_row() -> dict:
    """G0d: "Generator and encoder trained together" (08_generator_and_encoder_together run_phase2.json, chosen check pass 1.0),
    read as recorded.  SLOW TURNS closest / pick ADE were not recorded there (rerun.summarise)."""
    r = json.loads((GJK / "08_generator_and_encoder_together" / "out" / "run_phase2.json").read_text())
    assert r["best"] == 1.0
    s = next(c for c in r["checks"] if c["pass"] == 1.0)["summary"]
    return {"SLOW": {"m1a": s["SLOW"]["m1a"]["mean"], "m6_pick_ade": s["SLOW"]["m6_pick_ade"]["mean"]},
            "SLOW_TURNS": {"m1a": None, "m6_pick_ade": None}}


def flags(summ: dict, ref: dict) -> dict:
    out = {}
    for g in ("SLOW", "SLOW_TURNS"):
        for x in ("m1a", "m6_pick_ade"):
            out[f"{g}.{x}"] = "not recorded" if ref[g][x] is None else ("FLAG" if summ[g][x] > ref[g][x] else "ok")
    return out


def main(which: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    smoke = which == "smoke"
    state = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not smoke:
        state["preflight"] = P1.preflight()
        log(f"preflight {state['preflight']}")
    ctx = Q.Ctx()
    p3 = torch.load(ENC_CKPT / "pass_03" / "state.pt", map_location="cpu", weights_only=False)
    model = Model(p3)
    state["trainable_values"] = model.n_trainable
    state["pick_rule"] = "upstream (argmax sigmoid(last pred_score logit)); G0a: upstream uses the last logit alone"
    log(f"model built, trainable {model.n_trainable}")
    fbs = fixed_batches(ctx, 6)
    if smoke:
        log(f"G2 {json.dumps(gate_g2(model, fbs[0]))[:3000]}")
        g3 = gate_g3(model, fbs[1:3])
        log(f"G3 {json.dumps(g3)[:3000]}")
        a = model.eval_agent()
        d = evaluate(model, a, ctx.main[:5], ctx.meta)
        log(f"eval ok; same as s2 start on 5: {all(np.array_equal(d[k]['proposals'], ctx.start[k]['proposals']) for k in d)}")
        log(f"shares {lora_grad_shares(model, fbs[0])}")
        log("smoke done")
        return

    # step 0 and G1
    agent = model.eval_agent()
    dump0 = evaluate(model, agent, ctx.main, ctx.meta)
    ref_proj = Q.projector_out(agent, ctx.sel)
    del agent
    torch.cuda.empty_cache()
    state["G1"] = gate_g1(ctx, dump0)
    wj("state.json", state)
    log(f"G1 {state['G1']}")
    if not state["G1"]["pass"]:
        raise SystemExit("G1 failed: stopping")
    state["G2"] = gate_g2(model, fbs[0])
    wj("state.json", state)
    log(f"G2 pass={state['G2']['pass']} {json.dumps(state['G2'])[:2500]}")
    if not state["G2"]["pass"]:
        raise SystemExit("G2 failed: stopping")
    state["G3"] = gate_g3(model, fbs[1:6])
    wj("state.json", state)
    log(f"G3 pass={state['G3']['pass']} peak {state['G3']['peak_allocated_GiB']} GiB {state['G3']['s_per_step_after_first']} s/step")
    if not state["G3"]["pass"]:
        raise SystemExit("G3 failed: stopping")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ck_dir, dd = DATA / "ckpt", DATA / "dumps"
    ck_dir.mkdir(parents=True, exist_ok=True); dd.mkdir(parents=True, exist_ok=True)
    with open(dd / "p00.0.pkl", "wb") as fh:
        pickle.dump(dump0, fh)
    opts = model.optimisers()
    stream = Q.GridMaskStream(SEED)
    shuf = torch.Generator().manual_seed(SEED)
    perms = [torch.randperm(len(ctx.train), generator=shuf).tolist() for _ in range(MAX_PASSES)]
    ds = P1.TrainSet(ctx.train, ctx.meta)
    pairs = Q.pair_stream(K.CachedPairs(ctx.data["train"]), SEED)
    gens = M.make()
    fixed = E.FixedSet(ctx.data["selection"])
    ref = reference_row()
    per0 = per_moment(dump0, ctx.human, dump0)
    s0 = summary(per0, ctx.sp, ctx.cont)
    rec = {"run": "all three", "checks": [], "reference_joint_row": ref,
           "losses": {"generator": [], "scorer": [], "hide_and_guess": []}}
    c0 = {"pass": 0.0, "step": 0, "summary": s0, "turning_ok": True, "menu_ok": True,
          "tails": Q.tails(per0, {"step0": per0}, ctx.sp["SLOW_TURNS"]), "encoder_drift_cos": 0.0,
          "game_loss_picking_pairs": fixed.loss(model.enc, model.key, model.guesser),
          "other_logits": logit_drift(dump0, dump0, ctx.cont), "pdm_score_ties_1763": ties(dump0),
          "lora_grad": lora_grad_shares(model, fbs[0]), "flags": flags(s0, ref)}
    rec["checks"].append(c0)
    wj("run.json", rec)
    turn_ref, menu_ref = s0["SLOW_TURNS"]["m6_pick_ade"], s0["SLOW"]["m1a"]
    best, since, turn_fail, menu_fail = s0["SLOW"]["m6_pick_ade"], 0, 0, 0
    H = rec["losses"]
    gstep = 0
    log(f"step 0: SLOW pick ADE {best:.4f} closest ADE {menu_ref:.4f}; SLOW TURNS pick ADE {turn_ref:.4f}; "
        f"shares {c0['lora_grad']['share']}")

    def check(progress: float):
        nonlocal best, since, turn_fail, menu_fail
        te = time.time()
        agent = model.eval_agent()
        dump = evaluate(model, agent, ctx.main, ctx.meta)
        drift = Q.cos_drift(Q.projector_out(agent, ctx.sel), ref_proj)
        del agent
        torch.cuda.empty_cache()
        per = per_moment(dump, ctx.human, dump0)
        s = summary(per, ctx.sp, ctx.cont)
        c = {"pass": progress, "step": gstep, "summary": s, "tails": Q.tails(per, {"step0": per0}, ctx.sp["SLOW_TURNS"]),
             "encoder_drift_cos": drift, "game_loss_picking_pairs": fixed.loss(model.enc, model.key, model.guesser),
             "other_logits": logit_drift(dump, dump0, ctx.cont), "pdm_score_ties_1763": ties(dump),
             "lora_grad": lora_grad_shares(model, fbs[0]), "flags": flags(s, ref),
             "train_recent": {k: float(np.mean(v[-50:])) for k, v in H.items()}}
        c["turning_ok"] = s["SLOW_TURNS"]["m6_pick_ade"] <= turn_ref
        c["menu_ok"] = s["SLOW"]["m1a"] <= menu_ref
        turn_fail = 0 if c["turning_ok"] else turn_fail + 1
        menu_fail = 0 if c["menu_ok"] else menu_fail + 1
        if s["SLOW"]["m6_pick_ade"] < best:
            best, since = s["SLOW"]["m6_pick_ade"], 0
        else:
            since += 1
        c["eval_s"] = round(time.time() - te, 1)
        tag = f"p{progress:04.1f}"
        torch.save({**model.full_state(), "progress_passes": progress, "step": gstep,
                    "start": "release + pass-3 LoRA (unmerged) + released generator and scorer, fresh optimiser state"},
                   ck_dir / f"{tag}.pt")
        with open(dd / f"{tag}.pkl", "wb") as fh:
            pickle.dump(dump, fh)
        rec["checks"].append(c)
        wj("run.json", rec)
        log(f"pass {progress:.1f}: SLOW pick {s['SLOW']['m6_pick_ade']:.4f} closest {s['SLOW']['m1a']:.4f} gap {s['SLOW']['pick_gap']:.3f} | "
            f"TURNS pick {s['SLOW_TURNS']['m6_pick_ade']:.4f} (ref {turn_ref:.4f}) | var {s['SLOW']['m3_variety']:.3f} "
            f"drift {drift:.5f} game {c['game_loss_picking_pairs']:.5f} ties {c['pdm_score_ties_1763']} "
            f"other-logit {c['other_logits']['other5_mean_abs_change']:.3f} shares "
            f"{ {k: round(v, 3) for k, v in c['lora_grad']['share'].items()} } flags {c['flags']} eval {c['eval_s']}s")
        if turn_fail >= HARD_FAILS:
            return f"turning check failed at {HARD_FAILS} checks in a row (pass {progress:.1f})"
        if menu_fail >= HARD_FAILS:
            return f"menu guard failed at {HARD_FAILS} checks in a row (pass {progress:.1f})"
        if since >= NO_IMPROVE:
            return f"no improvement in SLOW scorer pick ADE (3 s) for {NO_IMPROVE} checks (pass {progress:.1f})"
        return None

    stop = None
    t_run = time.time()
    per_pass = [Q.planner_batches(len(ctx.train), pi, perms) for pi in range(1, MAX_PASSES + 1)]
    planner_it = iter(Q.spawn_loader(ds, [b for bs in per_pass for b in bs], 4, SEED))
    hashes = []
    for pi in range(1, MAX_PASSES + 1):
        batches = per_pass[pi - 1]
        mid = (len(batches) + 1) // 2
        for step in range(1, len(batches) + 1):
            b = next(planner_it)
            feats, t = P1.to_gpu(b)
            model.train_mode()
            f = model.forward(feats, t, stream)
            hashes.append(hashlib.sha1(json.dumps([batches[step - 1], *f["np_state"], f["gm_applied"]]).encode()).hexdigest()[:16])
            x, _ = next(pairs)
            masks = M.sample(gens, BATCH, M.step_rng(SEED, 2, gstep), "cuda")
            jl, _ = model.game_loss(x.cuda(non_blocking=True), masks)
            loss = f["gen"] + f["score"] + GAME_W * jl
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at pass {pi} step {step}")
            loss.backward()
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)
            model.after_step()
            model.zero_grad()
            H["generator"].append(float(f["gen"])); H["scorer"].append(float(f["score"])); H["hide_and_guess"].append(float(jl))
            del f, loss, jl
            gstep += 1
            if step % 25 == 0:
                ram = P1.avail_ram_gb()
                log(f"pass {pi} step {step} gen {np.mean(H['generator'][-25:]):.4f} scorer {np.mean(H['scorer'][-25:]):.4f} "
                    f"game {np.mean(H['hide_and_guess'][-25:]):.4f} ram {ram:.0f} GB")
                if ram < 20:
                    raise SystemExit(f"available RAM {ram:.1f} GB < 20 GB: stopping")
            if step == mid:
                stop = check(pi - 0.5)
                if stop:
                    break
        if stop:
            break
        stop = check(float(pi))
        wj("hashes.json", hashes)
        if stop:
            break
    rec["stopped"] = stop or f"reached {MAX_PASSES} passes"
    ok = [c for c in rec["checks"] if c["turning_ok"] and c["menu_ok"] and c["pass"] > 0]
    rec["chosen"] = min(ok, key=lambda c: (c["summary"]["SLOW"]["m6_pick_ade"], c["pass"]))["pass"] if ok else None
    rec["gridmask_total"] = dict(model.gm)
    rec["wall_s"] = round(time.time() - t_run, 1)
    wj("run.json", rec)
    wj("hashes.json", hashes)
    state.update(stopped=rec["stopped"], chosen=rec["chosen"])
    wj("state.json", state)
    log(f"stopped: {rec['stopped']}; chosen check {rec['chosen']}")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
    sys.stdout.flush()
    os._exit(0)          # 08_generator_and_encoder_together hung at exit after "done" (spawned loader workers)
