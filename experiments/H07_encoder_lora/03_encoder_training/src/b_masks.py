"""Part B measurements (CPU only).

    ./run.sh src/b_masks.py      # writes out/b_masks.json, out/masks/*.png
"""

from __future__ import annotations

import time

import numpy as np
import torch

import masks2 as M
import players as P
from s2common import OUT, SEED, write_json

DRAWS = 5000
BATCHES = 300


def stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "mean": round(float(v.mean()) * 100, 2), "std": round(float(v.std()) * 100, 2),
            "p5": round(float(np.percentile(v, 5)) * 100, 1), "p50": round(float(np.median(v)) * 100, 1),
            "p95": round(float(np.percentile(v, 95)) * 100, 1)}


def upstream_draws(cfg, crop, batch, n, grid_h=16):
    """Unchanged upstream generator; hidden fraction per sample after its own
    batch trimming, and share of calls whose block is full height."""
    g = M.upstream_generator(cfg, crop)
    tokens = (crop[0] // 16) * (crop[1] // 16)
    fr, full = [], []
    for _ in range(n):
        seed_next = g._itr_counter.value + 1
        gen = torch.Generator(); gen.manual_seed(seed_next)
        _, h, _ = g._sample_block_size(gen, g.temporal_pred_mask_scale, g.spatial_pred_mask_scale, g.aspect_ratio)
        enc, pred = g(batch)
        if pred.ndim == 1:
            pred = pred[None]
        fr += [pred.shape[1] / tokens] * pred.shape[0]
        full.append(h == grid_h)
    return fr, float(np.mean(full))


def main() -> None:
    torch.manual_seed(SEED)
    out = {"configs_s1": [dict(c) for c in P.MASK_CFGS], "configs_wide": M.wide_cfgs(), "rows": {}}
    names = ["cfg0 (8 blocks x 0.15)", "cfg1 (2 blocks x 0.70)"]
    for i, (c_sq, c_wide) in enumerate(zip(P.MASK_CFGS, M.wide_cfgs())):
        row = {}
        fr, full = upstream_draws(c_sq, (256, 256), 1, DRAWS)
        row["target_square_16x16_upstream_batch1"] = {**stats(fr), "full_height_share": round(full * 100, 1)}
        fr, full = upstream_draws(c_sq, (256, 512), 1, DRAWS)
        row["s1_wide_upstream_batch1"] = {**stats(fr), "full_height_share": round(full * 100, 1)}
        fr, _ = upstream_draws(c_sq, (256, 512), 32, BATCHES)
        row["s1_wide_upstream_batch32_trimmed"] = stats(fr)

        gen = M.EqualCountMasks(c_wide)
        rng = M.step_rng(SEED, 1, i)
        fr, full = [], []
        for _ in range(DRAWS):
            m, info = gen(1, rng, equalise=False)
            fr.append(m[0].mean())
            full.append(info["h"] == 16)
        row["new_wide_batch1"] = {**stats(fr), "full_height_share": round(float(np.mean(full)) * 100, 1)}
        fr, full, rounds, secs = [], [], [], []
        for _ in range(BATCHES):
            t0 = time.perf_counter()
            enc, pred, info = gen(32, rng)
            secs.append(time.perf_counter() - t0)
            assert enc.shape[1] + pred.shape[1] == M.N_TOKENS
            assert all(len(set(e.tolist()) | set(p.tolist())) == M.N_TOKENS for e, p in zip(enc, pred))
            fr += [pred.shape[1] / M.N_TOKENS] * 32
            full.append(info["h"] == 16)
            rounds.append(info["rounds"])
        row["new_wide_batch32_equalised"] = {**stats(fr), "full_height_share": round(float(np.mean(full)) * 100, 1),
                                             "rounds_mean": round(float(np.mean(rounds)), 2),
                                             "rounds_max": int(np.max(rounds)),
                                             "ms_per_batch_mean": round(1000 * float(np.mean(secs)), 1),
                                             "ms_per_batch_max": round(1000 * float(np.max(secs)), 1)}
        t, n, b = (row["target_square_16x16_upstream_batch1"]["mean"], row["new_wide_batch1"]["mean"],
                   row["new_wide_batch32_equalised"]["mean"])
        row["gates"] = {
            "B2_i_mean_within_5pts_of_target": abs(n - t) <= 5, "B2_i_diff_pts": round(n - t, 2),
            "B2_ii_full_height_under_30pct": (row["new_wide_batch1"]["full_height_share"] < 30) if i == 1 else None,
            "B3_batch32_within_3pts_of_per_sample": abs(b - n) <= 3, "B3_diff_pts": round(b - n, 2),
        }
        out["rows"][names[i]] = row
        print(names[i], row, flush=True)
    out["all_gates_pass"] = all(v for r in out["rows"].values() for v in r["gates"].values() if isinstance(v, bool))
    write_json(OUT / "b_masks.json", out)
    images()


def images() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = OUT / "masks"
    folder.mkdir(exist_ok=True)
    gens = M.make()
    for k in range(8):
        rng = M.step_rng(SEED, 2, k)
        fig, axes = plt.subplots(2, 3, figsize=(10, 3.6))
        for i, g in enumerate(gens):
            enc, pred, info = g(32, rng)
            for j in range(3):
                grid = np.zeros(M.N_TOKENS)
                grid[pred[j].numpy()] = 1
                ax = axes[i, j]
                ax.imshow(grid.reshape(16, 32), cmap="Greys", vmin=0, vmax=1)
                ax.set_title(f"cfg{i} sample {j}: block {info['h']}x{info['w']}, hidden {grid.mean():.0%}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"New masks, batch draw {k}: 3 of 32 samples per config (black = hidden; equal count across the batch)",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"mask_example_{k}.png", dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
