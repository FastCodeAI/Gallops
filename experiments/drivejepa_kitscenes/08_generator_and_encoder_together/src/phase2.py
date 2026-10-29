"""Phase 2: generator + encoder LoRA trained together on KIT, and a generator-only control.

    setsid nohup ./run.sh src/phase2.py all > logs/phase2.log 2>&1 &      # control first, then phase 2

Both runs start from the release checkpoint (sha256-checked) + pass-3 LoRA + the generator
weights of the phase-1 rerun's best check (w = 0, pass 9.5), with fresh optimiser state.

control  pass-3 LoRA merged and frozen; generator trained exactly as the phase-1 rerun at w = 0
         (upstream trajectory_loss_anchors, empty vocabulary lists; Adam 3e-5).
phase 2  pass-3 LoRA unmerged and trainable (AdamW 1e-4, no weight decay), plus the encoder
         hide-and-guess game from experiments/drivejepa_kitscenes s2 (guesser and answer key from pass-3
         state.pt, guesser AdamW 1e-4 with s2's weight-decay split, key moved by momentum
         0.99925 after every step, players.game_loss unchanged).  Loss = generator loss +
         1.0 x game loss; one backward; both optimisers step.

Keeping the runs comparable: the planner's moment order comes from its own torch.Generator
(seed 0, as s2); GridMask draws from numpy's global stream, so training forwards run inside
GridMaskStream, which swaps in a numpy state seeded 0 and keeps it between steps, so nothing
else (evaluation, the game) can move it.  The game's pairs and masks have their own
generators.  Every step logs a hash of the moment ids, the GridMask stream state before and
after the forward, and whether GridMask acted; phase 2 compares each step with the control.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cache as K           # experiments/drivejepa_kitscenes s2
import common as C          # experiments/drivejepa_kitscenes s1
import evalsets as E        # experiments/drivejepa_kitscenes s2
import f_final as FF        # experiments/drivejepa_kitscenes s2 (merge)
import g0_gpu as G0         # 05_generator_survey
import kitdata as DT        # 06_generator_only_fast_rate
import lora as L            # experiments/drivejepa_kitscenes s1
import masks2 as M          # experiments/drivejepa_kitscenes s2
import phase1 as P1         # 06_generator_only_fast_rate
import players as P         # experiments/drivejepa_kitscenes s1
import rerun as R           # 07_generator_only_gentle_rate
import splits as S          # experiments/drivejepa_kitscenes s1
from s2common import CKPT as ENC_CKPT

HERE = Path(__file__).resolve().parent.parent
OUT, LOGS = HERE / "out", HERE / "logs"
DATA = Path("{runs}/drivejepa_kitscenes/generator_joint_kit/s3")
S2DATA = R.DATA
P95 = S2DATA / "w0" / "ckpt" / "p09.5.pt"
P95_DUMP = S2DATA / "w0" / "dumps" / "p09.5.pkl"
SEED, BATCH = 0, 32
LR_GEN, LR_LORA, LR_GUESS, WD_GUESS, MOMENTUM, GAME_W = 3e-5, 1e-4, 1e-4, 0.04, 0.99925, 1.0
MAX_PASSES, TURN_FAILS, NO_IMPROVE = 10, 2, 6
N_LORA_TENSORS = 192

log = P1.log


def wj(name: str, obj) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(OUT / name)


class GridMaskStream:
    """numpy global state used by GridMask (grid_mask.py:89, np.random) during training forwards only."""

    def __init__(self, seed: int):
        self.state = np.random.RandomState(seed).get_state()

    def __enter__(self):
        self.outer = np.random.get_state()
        np.random.set_state(self.state)
        return self

    def __exit__(self, *a):
        self.state = np.random.get_state()
        np.random.set_state(self.outer)


def np_state_hash(st) -> str:
    return hashlib.sha1(st[1].tobytes() + str(st[2:]).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Model:
    def __init__(self, joint: bool, pass3_state: dict):
        self.joint = joint
        self.agent = G0.build(pass3=True, merged=not joint)
        self.m = m = self.agent._pad_model
        self.enc = m._backbone.img_backbone
        gen = [(n, p) for n, p in m.named_parameters() if n.startswith(P1.GEN_PREFIXES)]
        assert sum(p.numel() for _, p in gen) == P1.GEN_TOTAL
        self.fixed = [(n, p) for n, p in gen if n.endswith(P1.FIXED_SUFFIXES)]
        self.train_params = [(n, p) for n, p in gen if not n.endswith(P1.FIXED_SUFFIXES)]
        self.n_gen = sum(p.numel() for _, p in self.train_params)
        assert self.n_gen == 2_381_123 and len(self.train_params) == 41
        ck = torch.load(P95, map_location="cpu", weights_only=False)
        assert ck["w"] == 0.0 and ck["progress_passes"] == 9.5
        self.load_gen(ck["generator"])
        for _, p in self.train_params:
            p.requires_grad_(True)
        if joint:
            params = dict(self.enc.named_parameters())
            assert all(torch.equal(params[n].cpu(), t) for n, t in pass3_state["lora"].items())
            self.lora = L.lora_parameters(self.enc)
            assert len(self.lora) == N_LORA_TENSORS
            L.set_lora_trainable(self.enc, True)
            self.key = copy.deepcopy(self.enc).eval()
            for p in self.key.parameters():
                p.requires_grad_(False)
            self.load_lora(self.key, pass3_state["key_lora"])
            self.guesser = P.build_guesser(torch.load(C.E50, map_location="cpu", mmap=True)).cuda()
            self.guesser.load_state_dict(pass3_state["guesser"])
            for p in self.guesser.parameters():
                p.requires_grad_(True)
        else:
            assert not L.lora_modules(m)
        self.n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.gm = {"calls": 0, "applied": 0, "train_flag_calls": 0, "encoder_eval_at_call": 0, "last_applied": 0}
        gm, bk = self.gm, m._backbone

        def hook(mod, inp, out):      # a closure, not a bound method: FF.merge deep-copies the agent
            gm["calls"] += 1
            gm["train_flag_calls"] += int(mod.training)
            gm["last_applied"] = int(not torch.equal(inp[0], out))
            gm["applied"] += gm["last_applied"]
            gm["encoder_eval_at_call"] += int(not any(x.training for x in bk.img_backbone.modules())
                                              and not bk.projector.training)
        m._backbone.grid_mask.register_forward_hook(hook)

    # --- state -------------------------------------------------------------- #
    def gen_state(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.train_params}

    def load_gen(self, sd: dict) -> None:
        cur = dict(self.train_params)
        assert set(sd) == set(cur)
        with torch.no_grad():
            for n, t in sd.items():
                cur[n].copy_(t.to(cur[n].device))

    @staticmethod
    def lora_state(enc) -> dict:
        return {n: p.detach().cpu().clone() for n, p in enc.named_parameters() if ".lora_" in n}

    @staticmethod
    def load_lora(enc, sd: dict) -> None:
        params = dict(enc.named_parameters())
        assert set(sd) == {n for n in params if ".lora_" in n}
        with torch.no_grad():
            for n, t in sd.items():
                params[n].copy_(t.to(params[n].device, params[n].dtype))

    def full_state(self) -> dict:
        s = {"generator": self.gen_state()}
        if self.joint:
            s.update(lora=self.lora_state(self.enc), key_lora=self.lora_state(self.key),
                     guesser={k: v.detach().cpu().clone() for k, v in self.guesser.state_dict().items()})
        return s

    def load_full(self, s: dict) -> None:
        self.load_gen(s["generator"])
        if self.joint:
            self.load_lora(self.enc, s["lora"]); self.load_lora(self.key, s["key_lora"])
            self.guesser.load_state_dict(s["guesser"])

    # --- training ----------------------------------------------------------- #
    def train_mode(self) -> None:
        """As s1/s2: whole model eval, GridMask and generator train.  The encoder stays in
        eval in both runs: it has no dropout / drop-path (s0 E), and its activation
        checkpointing does not depend on the mode (vision_transformer.py:198)."""
        self.m.eval()
        self.m._backbone.grid_mask.train()
        for mod in (self.m.hist_encoding, self.m.init_feature, self.m._trajectory_head):
            mod.train()
        if self.joint:
            self.guesser.train()      # s2 trained the guesser in train mode (its checkpointing)

    def planner_loss(self, feats, target, stream: GridMaskStream):
        with stream:
            before = np.random.get_state()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.m(feats)
            after = np.random.get_state()
        pl = [q.float() for q in out["proposal_list"]]
        empty = [np.zeros(0, dtype=np.int64)] * target.shape[0]
        loss = self.agent.trajectory_loss_anchors(pl, target, self.agent._config, empty)[0]
        return loss, (np_state_hash(before), np_state_hash(after), self.gm["last_applied"])

    def game_loss(self, x, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return P.game_loss(self.enc, self.key, self.guesser, x, masks)

    def optimisers(self):
        opts = [torch.optim.Adam([p for _, p in self.train_params], lr=LR_GEN, weight_decay=0)]
        if self.joint:
            # s2 train.py guesser_groups (upstream init_opt split: no decay on biases / 1-D tensors)
            gd = [p for n, p in self.guesser.named_parameters() if "bias" not in n and p.ndim != 1]
            gn = [p for n, p in self.guesser.named_parameters() if "bias" in n or p.ndim == 1]
            opts.append(torch.optim.AdamW([{"params": self.lora, "lr": LR_LORA, "weight_decay": 0.0},
                                           {"params": gd, "lr": LR_GUESS, "weight_decay": WD_GUESS},
                                           {"params": gn, "lr": LR_GUESS, "weight_decay": 0.0}],
                                          betas=(0.9, 0.999), eps=1e-8))
        return opts

    def after_step(self) -> None:
        if self.joint:
            L.momentum_update(self.key, self.enc, MOMENTUM)

    def eval_agent(self):
        return FF.merge(self.agent) if self.joint else self.agent


# --------------------------------------------------------------------------- #
# Data streams
# --------------------------------------------------------------------------- #


def planner_batches(n: int, pass_idx: int, perms) -> list:
    perm = perms[pass_idx - 1]
    return [perm[i:i + BATCH] for i in range(0, len(perm) - BATCH + 1, BATCH)]    # drop_last


def spawn_loader(ds, batches: list, workers: int, seed: int):
    """One long-lived loader per stream.  Workers are spawned, not forked: attempt 1 forked
    10 workers per pass from a ~16 GB parent and the run stalled in swap on a shared host."""
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=workers, pin_memory=True,
                                       prefetch_factor=4, persistent_workers=False, multiprocessing_context="spawn",
                                       generator=torch.Generator().manual_seed(seed))


def pair_stream(ds, seed: int, epochs: int = 12):
    """Batches of 32 train-scene pairs, reshuffled each epoch by its own stream (drop_last)."""
    batches = []
    for e in range(epochs):
        g = torch.Generator().manual_seed(seed * 1000 + e)
        perm = torch.randperm(len(ds), generator=g).tolist()
        batches += [perm[i:i + BATCH] for i in range(0, len(perm) - BATCH + 1, BATCH)]
    for x, idx in spawn_loader(ds, batches, 3, seed):
        yield x, idx


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(model: Model, agent, moments, meta) -> dict:
    fc = K.FrameCache()
    before = dict(model.gm)
    dump = {}
    for m in moments:
        d = E.forward_dump(agent, DT.assemble(m, meta[m["scene_id"]], fc))
        dump[P1.key(m)] = {k: d[k] for k in ("proposals", "pdm_score", "chosen")}
    assert model.gm["applied"] == before["applied"] and model.gm["train_flag_calls"] == before["train_flag_calls"], \
        "GridMask active during evaluation"
    return dump


@torch.no_grad()
def projector_out(agent, sel) -> list:
    agent.eval()
    m = agent._pad_model
    out = []
    for e in sel:
        f = e["features"]
        x = torch.cat([m.transform(f["camera_feature_2"][None].cuda())[:, None],
                       m.transform(f["camera_feature_1"][None].cuda())[:, None]], dim=1)
        feat, *_ = m._backbone(x, img_metas={})
        out.append(feat.permute(2, 0, 1, 3).reshape(-1, feat.shape[-1]).float().cpu())
    return out


def cos_drift(a: list, ref: list) -> float:
    return float(np.mean([float((1 - F.cosine_similarity(x, y, dim=-1)).mean()) for x, y in zip(a, ref)]))


def tails(per: dict, refs: dict, turns: list) -> dict:
    """T1 count worse by > 0.5 m, T2 worst single increase, per reference; T3 mean 1b of the worst 10 %."""
    n3 = int(0.1 * len(turns))
    out = {"T3_worst10pct_mean_1b": float(np.mean(sorted((per[k]["m1b"] for k in turns), reverse=True)[:n3])),
           "T3_n": n3}
    for name, rp in refs.items():
        d = {k: per[k]["m1b"] - rp[k]["m1b"] for k in turns}
        kmax = max(d, key=lambda k: (d[k], k))
        out[f"T1_worse_gt_0.5m_vs_{name}"] = int(sum(v > 0.5 for v in d.values()))
        out[f"T2_worst_increase_vs_{name}"] = {"m": float(d[kmax]), "moment": list(kmax)}
    return out


class Ctx:
    """Everything evaluation needs, built once."""

    def __init__(self):
        self.meta = torch.load(DT.DATA / "scene_meta.pt", weights_only=False)
        mom = torch.load(DT.DATA / "moments.pt", weights_only=False)
        self.train, self.main = mom["train"], mom["selection"]
        assert len(self.train) == 13937 and len(self.main) == 1763
        self.human = {P1.key(m): m["future_poses"] for m in self.main}
        sel = E.load_selection_features()
        # keep only what is used later (cameras for the encoder-drift check), ~0.3 GB instead of ~2 GB
        self.sel = [{"scene_id": e["scene_id"], "keyframe": e["keyframe"],
                     "features": {k: e["features"][k].clone() for k in ("camera_feature_1", "camera_feature_2")}}
                    for e in sel]
        del sel
        self.cont = sorted((e["scene_id"], e["keyframe"]) for e in self.sel)
        assert len(self.cont) == 103 and set(self.cont) <= set(self.human)
        self.sp = R.splits(self.main)
        assert (len(self.sp["SLOW"]), len(self.sp["FAST"]), len(self.sp["SLOW_TURNS"])) == (1681, 82, 342)
        self.start = pickle.load(open(S2DATA / "dumps" / "start.pkl", "rb"))
        self.start_per = P1.per_moment(self.start, self.human, self.start)
        self.p95 = pickle.load(open(P95_DUMP, "rb"))
        self.p95_per = P1.per_moment(self.p95, self.human, self.start)
        self.refs = {"pass3_start": self.start_per, "p9.5": self.p95_per}
        self.data = S.load_splits()
        assert len(self.data["train"]) == 482, len(self.data["train"])

    def score(self, dump) -> dict:
        per = P1.per_moment(dump, self.human, self.start)
        return {"summary": R.summarise(per, self.start_per, self.sp, self.cont),
                "tails": tails(per, self.refs, self.sp["SLOW_TURNS"])}


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_g1(model: Model, ctx: Ctx, dump: dict) -> dict:
    bit = set(dump) == set(ctx.p95) and all(
        np.asarray(dump[k][f]).tobytes() == np.asarray(ctx.p95[k][f]).tobytes()
        for k in dump for f in ("proposals", "pdm_score", "chosen"))
    h103 = {k: ctx.human[k] for k in ctx.cont}
    now, _ = G0.menu_metrics({k: dump[k] for k in ctx.cont}, h103)
    then, _ = G0.menu_metrics({k: ctx.p95[k] for k in ctx.cont}, h103)
    rec = json.loads((R.OUT / "run_w0.json").read_text())
    p95_cont = next(c for c in rec["checks"] if c["pass"] == 9.5)["summary"]["CONT_m1b_mean"]
    cont_now = float(np.mean([P1.per_moment({k: dump[k]}, ctx.human, None)[k]["m1b"] for k in ctx.cont]))
    res = {"menus_1763_bit_identical_to_p9.5_dump": bit, "metrics_103_equal": now == then,
           "cont_1b_equals_s2_record": cont_now == p95_cont}
    res["pass"] = all(res.values())
    return res


def gate_g2(model: Model, ctx: Ctx, n_steps: int = 5) -> dict:
    saved = model.full_state()
    opts = model.optimisers()
    rng = torch.Generator().manual_seed(12345)           # not the training streams
    idx = torch.randperm(len(ctx.train), generator=rng)[:n_steps * BATCH].tolist()
    ds = P1.TrainSet(ctx.train, ctx.meta)
    stream = GridMaskStream(12345)
    pairs = pair_stream(K.CachedPairs(ctx.data["train"]), 99) if model.joint else None
    gens = M.make()
    m = model.m
    named = dict(m.named_parameters())
    proj = [p for n, p in named.items() if n.startswith("_backbone.projector.")]
    scorer = [p for n, p in named.items() if n.startswith("scorer.")]
    enc_base = [p for n, p in model.enc.named_parameters() if ".lora_" not in n]
    others = {"projector": proj, "scorer": scorer, "fixed_grids": [p for _, p in model.fixed], "encoder_base": enc_base,
              "rest": [p for n, p in named.items() if not n.startswith(P1.GEN_PREFIXES + ("_backbone.", "scorer."))]}
    if model.joint:
        others["answer_key"] = list(model.key.parameters())
    nz = lambda ps: sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in ps)
    anyg = lambda ps: sum(int(p.grad is not None) for p in ps)
    res = {"steps": [], "mode": "phase2" if model.joint else "control"}
    torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(n_steps):
        b = torch.utils.data.default_collate([ds[i] for i in idx[s * BATCH:(s + 1) * BATCH]])
        feats, t = P1.to_gpu(dict(b))
        model.train_mode()
        torch.cuda.synchronize(); t0 = time.time()
        gl, _ = model.planner_loss(feats, t, stream)
        r = {}
        if model.joint:
            x, _ = next(pairs)
            masks = M.sample(gens, BATCH, M.step_rng(12345, 2, s), "cuda")
            if s == 0:             # generator loss alone must reach the LoRA through the encoder
                gl.backward(retain_graph=True)
                r["generator_loss_only_lora_nonzero"] = nz(model.lora)
                r["generator_loss_only_guesser_any"] = anyg(model.guesser.parameters())
                for p in m.parameters():
                    p.grad = None
            jl, _ = model.game_loss(x.cuda(non_blocking=True), masks)
            loss = gl + GAME_W * jl
            r["game_loss"] = float(jl)
        else:
            loss = gl
        loss.backward()
        r.update(gen_loss=float(gl), generator_nonzero=nz([p for _, p in model.train_params]),
                 others_any_grad={k: anyg(v) for k, v in others.items()})
        if model.joint:
            gz = [n for n, p in model.guesser.named_parameters() if not (p.grad is not None and bool(p.grad.abs().sum() > 0))]
            r.update(lora_nonzero=nz(model.lora), guesser_nonzero=nz(model.guesser.parameters()),
                     guesser_total=len(list(model.guesser.parameters())), guesser_zero_grad=gz)
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        model.after_step()
        torch.cuda.synchronize(); times.append(time.time() - t0)
        res["steps"].append(r)
    res["peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["s_per_step_after_first"] = round(float(np.mean(times[1:])), 3)
    model.load_full(saved)
    now = model.full_state()
    res["restored_bit_exact"] = all(torch.equal(saved[g][k], now[g][k]) for g in saved for k in saved[g])
    ok = all(r["generator_nonzero"] == 41 and not any(r["others_any_grad"].values()) for r in res["steps"])
    if model.joint:
        # The guesser has 10 mask tokens and the game uses index 0 only (players.MASK_TOKEN_INDEX).
        allowed = {f"mask_tokens.{i}" for i in range(1, 10)}
        ok = ok and res["steps"][0]["generator_loss_only_lora_nonzero"] == N_LORA_TENSORS \
            and res["steps"][0]["generator_loss_only_guesser_any"] == 0 \
            and all(r["lora_nonzero"] == N_LORA_TENSORS and set(r["guesser_zero_grad"]) <= allowed for r in res["steps"])
    res["pass"] = ok and res["restored_bit_exact"]
    return res


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def run(name: str, ctx: Ctx, pass3_state: dict, state: dict, ref_hashes: list | None) -> None:
    joint = name == "phase2"
    model = Model(joint, pass3_state)
    st = state[name] = {"trainable_values": model.n_trainable}
    log(f"[{name}] model built, trainable {model.n_trainable}")
    agent = model.eval_agent()
    dump0 = evaluate(model, agent, ctx.main, ctx.meta)
    if joint:
        ref_proj = projector_out(agent, ctx.sel)
        fixed = E.FixedSet(ctx.data["selection"])
    del agent
    torch.cuda.empty_cache()
    st["G1"] = gate_g1(model, ctx, dump0)
    wj("state.json", state)
    log(f"[{name}] G1 {st['G1']}")
    if not st["G1"]["pass"]:
        raise SystemExit(f"G1 failed for {name}: stopping")
    st["G2"] = gate_g2(model, ctx)
    wj("state.json", state)
    log(f"[{name}] G2 pass={st['G2']['pass']} peak {st['G2']['peak_allocated_GiB']} GiB "
        f"{st['G2']['s_per_step_after_first']} s/step; last step {json.dumps(st['G2']['steps'][-1])[:600]}")
    if not st["G2"]["pass"]:
        raise SystemExit(f"G2 failed for {name}: stopping")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ck_dir, dd = DATA / name / "ckpt", DATA / name / "dumps"
    ck_dir.mkdir(parents=True, exist_ok=True); dd.mkdir(parents=True, exist_ok=True)
    opts = model.optimisers()
    stream = GridMaskStream(SEED)
    shuf = torch.Generator().manual_seed(SEED)
    perms = [torch.randperm(len(ctx.train), generator=shuf).tolist() for _ in range(MAX_PASSES)]
    ds = P1.TrainSet(ctx.train, ctx.meta)
    pairs = pair_stream(K.CachedPairs(ctx.data["train"]), SEED) if joint else None
    gens = M.make()
    hashes: list = []
    rec = {"run": name, "checks": [], "fresh_optimiser_state": True}
    s0 = ctx.score(dump0)
    c0 = {"pass": 0.0, "step": 0, **s0, "turning_ok": True}
    if joint:
        c0["game_loss_picking_pairs"] = fixed.loss(model.enc, model.key, model.guesser)
        c0["encoder_drift_cos"] = 0.0
    rec["checks"].append(c0)
    turn_ref = s0["summary"]["SLOW_TURNS"]["m1b"]["mean"]
    best_slow, since, turn_fail = s0["summary"]["SLOW"]["m1b"]["mean"], 0, 0
    gl_hist, jl_hist = [], []
    gstep = 0
    log(f"[{name}] step 0: SLOW 1b {best_slow:.4f} TURN 1b {turn_ref:.4f} " +
        (f"game {c0['game_loss_picking_pairs']:.5f}" if joint else ""))

    def check(progress: float):
        nonlocal best_slow, since, turn_fail
        te = time.time()
        agent = model.eval_agent()
        dump = evaluate(model, agent, ctx.main, ctx.meta)
        c = {"pass": progress, "step": gstep, **ctx.score(dump)}
        if joint:
            c["encoder_drift_cos"] = cos_drift(projector_out(agent, ctx.sel), ref_proj)
            c["game_loss_picking_pairs"] = fixed.loss(model.enc, model.key, model.guesser)
            c["game_loss_train_recent"] = float(np.mean(jl_hist[-50:]))
        del agent
        torch.cuda.empty_cache()
        c["gen_loss_train_recent"] = float(np.mean(gl_hist[-50:]))
        c["eval_s"] = round(time.time() - te, 1)
        tag = f"p{progress:04.1f}"
        torch.save({**model.full_state(), "run": name, "progress_passes": progress, "step": gstep,
                    "start": "release + pass-3 LoRA + phase-1 rerun w0 p9.5 generator, fresh optimiser state"},
                   ck_dir / f"{tag}.pt")
        with open(dd / f"{tag}.pkl", "wb") as fh:
            pickle.dump(dump, fh)
        s = c["summary"]
        c["turning_ok"] = s["SLOW_TURNS"]["m1b"]["mean"] <= turn_ref
        turn_fail = 0 if c["turning_ok"] else turn_fail + 1
        if s["SLOW"]["m1b"]["mean"] < best_slow:
            best_slow, since = s["SLOW"]["m1b"]["mean"], 0
        else:
            since += 1
        rec["checks"].append(c)
        wj(f"run_{name}.json", rec)
        log(f"[{name}] pass {progress:.1f}: SLOW 1b {s['SLOW']['m1b']['mean']:.4f} TURN 1b {s['SLOW_TURNS']['m1b']['mean']:.4f} "
            f"(ref {turn_ref:.4f}) FAST {s['FAST']['m1b']['mean']:.3f} var {s['SLOW']['m3_variety']['mean']:.3f} "
            f"T1 {c['tails']['T1_worse_gt_0.5m_vs_p9.5']} T3 {c['tails']['T3_worst10pct_mean_1b']:.3f} "
            + (f"drift {c['encoder_drift_cos']:.5f} game {c['game_loss_picking_pairs']:.5f} " if joint else "")
            + f"eval {c['eval_s']}s")
        if turn_fail >= TURN_FAILS:
            return f"turning check failed at {TURN_FAILS} checks in a row (pass {progress:.1f})"
        if since >= NO_IMPROVE:
            return f"no SLOW 1b improvement for {NO_IMPROVE} checks (pass {progress:.1f})"
        return None

    stop = None
    t_run = time.time()
    per_pass = [planner_batches(len(ctx.train), pi, perms) for pi in range(1, MAX_PASSES + 1)]
    planner_it = iter(spawn_loader(ds, [b for bs in per_pass for b in bs], 4, SEED))
    for pi in range(1, MAX_PASSES + 1):
        batches = per_pass[pi - 1]
        mid = (len(batches) + 1) // 2
        for step in range(1, len(batches) + 1):
            b = next(planner_it)
            ids = batches[step - 1]
            feats, t = P1.to_gpu(b)
            model.train_mode()
            gl, (h0, h1, applied) = model.planner_loss(feats, t, stream)
            hsh = hashlib.sha1(json.dumps([ids, h0, h1, applied]).encode()).hexdigest()[:16]
            hashes.append(hsh)
            if ref_hashes is not None and gstep < len(ref_hashes) and hsh != ref_hashes[gstep]:
                state["G3"] = {"pass": False, "first_mismatch_step": gstep}
                wj("state.json", state)
                raise SystemExit(f"planner batch hash differs from the control at step {gstep}: stopping")
            if ref_hashes is not None and gstep == 19:
                state["G3"] = {"pass": True, "first_20_match": True}
                wj("state.json", state)
                log("G3 pass: first 20 planner-batch hashes match the control")
            loss = gl
            if joint:
                x, _ = next(pairs)
                masks = M.sample(gens, BATCH, M.step_rng(SEED, 2, gstep), "cuda")
                jl, _ = model.game_loss(x.cuda(non_blocking=True), masks)
                loss = gl + GAME_W * jl
                jl_hist.append(float(jl))
            if not torch.isfinite(loss):
                raise SystemExit(f"[{name}] non-finite loss at pass {pi} step {step}")
            loss.backward()
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)
            model.after_step()
            gl_hist.append(float(gl))
            gstep += 1
            if step % 100 == 0:
                ram = P1.avail_ram_gb()
                log(f"[{name}] pass {pi} step {step} gen {np.mean(gl_hist[-100:]):.4f}"
                    + (f" game {np.mean(jl_hist[-100:]):.4f}" if joint else "") + f" ram {ram:.0f} GB")
                if ram < 20:
                    raise SystemExit(f"available RAM {ram:.1f} GB < 20 GB: stopping")
            if step == mid:
                stop = check(pi - 0.5)
                if stop:
                    break
        if stop:
            break
        stop = check(float(pi))
        wj(f"hashes_{name}.json", hashes)
        if stop:
            break
    rec["stopped"] = stop or f"reached {MAX_PASSES} passes"
    ok = [c for c in rec["checks"] if c["turning_ok"] and c["pass"] > 0]
    rec["best"] = min(ok, key=lambda c: (c["summary"]["SLOW"]["m1b"]["mean"], c["pass"]))["pass"] if ok else None
    rec["beats_step0"] = bool(ok) and min(c["summary"]["SLOW"]["m1b"]["mean"] for c in ok) < s0["summary"]["SLOW"]["m1b"]["mean"]
    rec["gridmask_total"] = dict(model.gm)
    rec["wall_s"] = round(time.time() - t_run, 1)
    wj(f"run_{name}.json", rec)
    wj(f"hashes_{name}.json", hashes)
    st.update(stopped=rec["stopped"], best=rec["best"], beats_step0=rec["beats_step0"])
    wj("state.json", state)
    log(f"[{name}] stopped: {rec['stopped']}; best {rec['best']}")
    del model
    torch.cuda.empty_cache()


def main(which: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    state = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() and which == "phase2" else {}
    state["preflight"] = P1.preflight()
    log(f"preflight {state['preflight']}")
    ctx = Ctx()
    p3 = torch.load(ENC_CKPT / "pass_03" / "state.pt", map_location="cpu", weights_only=False)
    log(f"pass-3 state keys {sorted(p3)}")
    if which in ("all", "control"):
        run("control", ctx, p3, state, None)
    if which in ("all", "phase2"):
        ref = json.loads((OUT / "hashes_control.json").read_text())
        run("phase2", ctx, p3, state, ref)
        n = min(len(ref), len(json.loads((OUT / "hashes_phase2.json").read_text())))
        same = sum(a == b for a, b in zip(ref, json.loads((OUT / "hashes_phase2.json").read_text())))
        state["G3"]["all_common_steps"] = {"n": n, "matching": same}
        wj("state.json", state)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
