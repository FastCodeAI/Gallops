"""The two things every pass is judged on, both built from selection scenes only.

1. The 103 selection moments (baseline rule, valid_keyframes(10, 90, 40),
   RUN:236-238).  Their agent features are built once through the baseline path
   (ADP.build_sample -> agent.get_feature_builders(), abstract_agent.py:79-80)
   and saved; the features do not depend on the encoder, so every later pass
   only re-runs agent.forward.  forward_dump() reproduces compute_trajectory
   (abstract_agent.py:76-94) on those features; c_reference.py checks that it
   is bit-identical to compute_trajectory on all 103.
2. The fixed guesser-loss set: 2,000 selection pairs from the frame cache with
   masks fixed by seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch

import cache as K
import masks2 as M
import players as P
from s2common import CACHE, SEED, SEL_FEATS

ENC_PATH = "_pad_model._backbone.img_backbone"
FIXED_N = 2000
FIXED_BATCH = 32


# --------------------------------------------------------------------------- #
# Selection moments
# --------------------------------------------------------------------------- #


def load_selection_features() -> list[dict]:
    return torch.load(SEL_FEATS, map_location="cpu", weights_only=False)


def forward_dump(agent, feats_cpu: dict) -> dict:
    """compute_trajectory minus the feature build: batch dim + .cuda()
    (abstract_agent.py:83), the three extra keys (:84-86), forward under
    no_grad, trajectory squeezed to numpy (:89-91).  fp32: no autocast."""
    agent.eval()
    feats = {k: v.unsqueeze(0).cuda() for k, v in feats_cpu.items()}
    feats["past_ego_simulated_states"] = None
    feats["metric_cache"] = None
    feats["observation_interval"] = None
    with torch.no_grad():
        o = agent.forward(feats)
    return {
        "proposals": o["proposals"][0].cpu().numpy(),
        "pred_logit": o["pred_logit"][0].cpu().numpy(),
        "pdm_score": o["pdm_score"][0].cpu().numpy(),
        "chosen": int(torch.argmax(o["pdm_score"], dim=1)[0]),
        "trajectory": o["trajectory"].squeeze(0).cpu().numpy(),
    }


def aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("ade", "fde@1s", "fde@2s", "fde@3s"):
        v = [r[k] for r in rows]
        out[k] = {"mean": float(np.mean(v)), "median": float(np.median(v))}
    return out


def selection_eval(agent, entries: list[dict], run_mod) -> tuple[dict, list[dict]]:
    """Dump + RUN.displacement_errors (RUN:259-261, horizon 3 s) per moment."""
    dump, rows = {}, []
    for e in entries:
        d = forward_dump(agent, e["features"])
        key = (e["scene_id"], e["keyframe"])
        dump[key] = d
        m = run_mod.displacement_errors(d["trajectory"], e["future_poses"], e["future_interval_s"], 3.0)
        rows.append({"scene_id": e["scene_id"], "keyframe_index": e["keyframe"], **m})
    return dump, rows


# --------------------------------------------------------------------------- #
# Fixed guesser-loss set
# --------------------------------------------------------------------------- #


def fixed_items(selection: list[str], scene_frames: dict) -> list[tuple[str, int]]:
    items = [(s, t) for s in sorted(selection) for t in K.pair_ends(scene_frames[s])]
    return sorted(random.Random(SEED).sample(items, FIXED_N))


def fixed_masks(n_batches: int, gens=None) -> list:
    """Batch b uses step_rng(SEED, 7, b): the same masks every time."""
    gens = gens or M.make()
    out = []
    for b in range(n_batches):
        size = min(FIXED_BATCH, FIXED_N - b * FIXED_BATCH)
        out.append(M.sample(gens, size, M.step_rng(SEED, 7, b)))
    return out


class FixedSet:
    def __init__(self, selection: list[str]):
        import json

        nf = json.loads((CACHE / "index.json").read_text())["scene_frames"]
        self.items = fixed_items(selection, nf)
        n_batches = (FIXED_N + FIXED_BATCH - 1) // FIXED_BATCH
        self.masks = fixed_masks(n_batches)
        fc = K.FrameCache()
        # 2,000 pairs x 3 MB = 6.3 GB as float32; keep the uint8 frames instead
        # (1.6 GB) and normalise per batch with the same functions.
        self.frames = [(fc.frame(s, t - K.FRAME_STEP), fc.frame(s, t)) for s, t in self.items]

    def batches(self, device="cuda"):
        for b, masks in enumerate(self.masks):
            chunk = self.frames[b * FIXED_BATCH:(b + 1) * FIXED_BATCH]
            x = torch.stack([K.pair_from_arrays(o, n) for o, n in chunk]).to(device, non_blocking=True)
            yield x, [(e.to(device), p.to(device)) for e, p in masks]

    @torch.no_grad()
    def loss(self, learner, key, guesser) -> float:
        was = learner.training, guesser.training
        learner.eval(); guesser.eval()
        tot, n = 0.0, 0
        for x, masks in self.batches():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = P.game_loss(learner, key, guesser, x, masks)
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
        learner.train(was[0]); guesser.train(was[1])
        return tot / n
