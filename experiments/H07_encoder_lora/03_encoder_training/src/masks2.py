"""Part B: the covering pattern for the wide 16x32 token grid.

Two changes to what s1 used (players.mask_generators -> upstream _MaskGenerator,
vjepa2/src/masks/multiseq_multiblock3d.py:80-240), nothing else:

1. Allowed shapes.  Upstream sizes a block as (:136-152)
       n = int(H*W*scale);  h = round(sqrt(n*ar));  w = round(sqrt(n/ar)),  h<=H, w<=W
   with ar in (0.75, 1.5) -- shapes chosen for a square grid.  The block area n
   is already a share of the grid, so on 16x32 n doubles, and with the same ar
   h = sqrt(358*ar) >= 16 for every ar in range: the large blocks are clipped to
   the full height every time (s1 C.3).  We pass ar * H/W instead, i.e. the
   aspect ratio is measured in grid units rather than tokens.  The block is then
   the square-grid block stretched 2x along the width: same share of the grid,
   same share of the height.  The upstream class is used unchanged; only its
   aspect_ratio argument differs.

2. No batch-minimum trimming.  Upstream draws each sample's blocks, then cuts
   every sample's hidden list to the batch minimum (:212-213) so they stack; on
   our grid that took 62 % -> 49 % and 72 % -> 59 % at batch 16 (s1 C.3).
   Here the block size is drawn once per batch, as upstream does (:186-192),
   sample 0's positions are drawn as upstream does (uniform top/left, :156-158),
   and every other sample is redrawn until it hides exactly as many tokens as
   sample 0 (rejection sampling, vectorised).  Because sample 0's count K is a
   draw from the per-sample distribution and the others are conditioned on that
   same K, every sample's hidden count still has the per-sample distribution,
   and no token is added or cut.
"""

from __future__ import annotations

import numpy as np
import torch

import players as P  # s1: MASK_CFGS, _MaskGenerator import, GRID_H/W

GRID_H, GRID_W = P.GRID_H, P.GRID_W
N_TOKENS = GRID_H * GRID_W


def wide_cfgs(cfgs=P.MASK_CFGS, h=GRID_H, w=GRID_W) -> list[dict]:
    """Same block counts and area shares; aspect ratio scaled by H/W."""
    return [{**c, "aspect_ratio": (c["aspect_ratio"][0] * h / w, c["aspect_ratio"][1] * h / w)} for c in cfgs]


def upstream_generator(cfg: dict, crop=(256, 512)):
    """Upstream _MaskGenerator, exactly as s1 players.mask_generators builds it."""
    return P._MaskGenerator(
        crop_size=crop, num_frames=2, spatial_patch_size=(16, 16), temporal_patch_size=2,
        spatial_pred_mask_scale=cfg["spatial_scale"], temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=cfg["aspect_ratio"], npred=cfg["num_blocks"], max_context_frames_ratio=1.0,
        max_keep=None, full_complement=False, pred_full_complement=False, inv_block=False,
    )


class EqualCountMasks:
    """One mask config on the 16x32 grid, equal hidden count across the batch."""

    CANDIDATES = 4096
    MAX_ROUNDS = 500

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.up = upstream_generator(cfg)  # used for _sample_block_size only
        self.npred = cfg["num_blocks"]

    def block_size(self, rng: np.random.Generator) -> tuple[int, int]:
        g = torch.Generator()
        g.manual_seed(int(rng.integers(0, 2**62)))
        t, h, w = self.up._sample_block_size(g, self.up.temporal_pred_mask_scale,
                                             self.up.spatial_pred_mask_scale, self.up.aspect_ratio)
        assert t == 1
        return h, w

    def _candidates(self, rng, h, w, n) -> np.ndarray:
        """(n, H*W) bool: True = hidden.  Uniform top/left per block, as upstream :156-158."""
        top = rng.integers(0, GRID_H - h + 1, size=(n, self.npred))
        left = rng.integers(0, GRID_W - w + 1, size=(n, self.npred))
        r = np.arange(GRID_H)
        c = np.arange(GRID_W)
        rows = (r >= top[..., None]) & (r < top[..., None] + h)  # (n, npred, H)
        cols = (c >= left[..., None]) & (c < left[..., None] + w)  # (n, npred, W)
        hidden = (rows[..., :, None] & cols[..., None, :]).any(axis=1)  # (n, H, W)
        return hidden.reshape(n, N_TOKENS)

    def __call__(self, batch_size: int, rng: np.random.Generator, equalise: bool = True):
        h, w = self.block_size(rng)
        redraws = 0
        while True:
            first = self._candidates(rng, h, w, 1)
            while first.all():  # upstream redraws an empty context (:197-208)
                first = self._candidates(rng, h, w, 1)
            k = int(first.sum())
            chosen = [first[0]]
            rounds = 0
            while len(chosen) < batch_size and rounds <= self.MAX_ROUNDS:
                cand = self._candidates(rng, h, w, self.CANDIDATES)
                if equalise:
                    cand = cand[cand.sum(1) == k]
                else:
                    cand = cand[~cand.all(1)]
                chosen.extend(cand[: batch_size - len(chosen)])
                rounds += 1
            if len(chosen) == batch_size:
                break
            # Added 2026-10-16 after main step 1848 found no match for k=431
            # (8 blocks of 5x14, almost no overlap) in 500 rounds x 4096.  Only
            # such extreme-tail k reach here; sample 0 is redrawn with the same
            # block size.  Every batch that used to succeed is unchanged.
            redraws += 1
            if redraws > 20:
                raise RuntimeError(f"could not equalise hidden count for block {h}x{w}")
        m = np.stack(chosen)
        if equalise:
            pred = torch.from_numpy(np.stack([np.flatnonzero(x) for x in m]))
            enc = torch.from_numpy(np.stack([np.flatnonzero(~x) for x in m]))
            return enc, pred, {"h": h, "w": w, "k": k, "rounds": rounds, "redraws": redraws}
        return m, {"h": h, "w": w}


def make() -> list[EqualCountMasks]:
    return [EqualCountMasks(c) for c in wide_cfgs()]


def sample(gens, batch_size: int, rng: np.random.Generator, device="cpu"):
    """Same structure as s1 players.sample_masks: [(enc (B,Kc), pred (B,Kh)) per config]."""
    out = []
    for g in gens:
        enc, pred, _ = g(batch_size, rng)
        out.append((enc.to(device), pred.to(device)))
    return out


def step_rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])
