"""How often a path's memory starts empty on the val logs, and where (as run for MEMORY_H15_STEP2.md, D2).

A path starts empty when its moment starts a chain (no moment 0.5 s before) or when no old path is
within align.LIMIT of it. Breakdowns by the car's turn over the 0.5 s step, its speed, and chain position.

    PYTHONPATH=src python experiments/H15_prediction_memory/scripts/match_stats_val.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.memory import align, ticks

cfg = SiteConfig.load()
cache = Path(cfg.resolve("{data.navtrain.cache}"))
logs_dir = Path(cfg.datasets["navtrain"].extra["logs"])
table = ticks.load(cache / "ticks")
proposals = np.load(cache / "release" / "proposals.npy", mmap_mode="r")
logits = np.load(cache / "release" / "logits.npy", mmap_mode="r")
prev = align.previous_rows(table)
val = table["split"] == "val"
rows_val = np.flatnonzero(val)
graded = np.asarray(table["graded"]) & np.asarray(table["inputs_ok"])

poses = np.zeros((len(table["log"]), 4, 4))
for log in sorted(set(table["log"][rows_val].tolist())):
    frames = align._log_frames(logs_dir, log)
    for r in np.flatnonzero(table["log"] == log):
        poses[r] = align.pose_of(frames[int(table["frame_idx"][r])])

pairs = rows_val[prev[rows_val] >= 0]
unmatched = np.zeros((len(pairs), 32), bool)
closest = np.zeros((len(pairs), 32))
same = np.zeros((len(pairs), 32), bool)
pick_unmatched = np.zeros(len(pairs), bool)
yaw = np.zeros(len(pairs))
speed = np.zeros(len(pairs))
for k, r in enumerate(pairs):
    p = prev[r]
    aligned = align.align_previous(proposals[p], poses[p], poses[r])
    m, m_d = align.match(align.distances(aligned, proposals[r]))
    unmatched[k], closest[k], same[k] = m == -1, m_d, m == np.arange(32)
    pick_unmatched[k] = m[int(np.asarray(logits[r])[:, -1].argmax())] == -1
    rel = np.linalg.inv(poses[p]) @ poses[r]
    yaw[k] = abs(np.arctan2(rel[1, 0], rel[0, 0]))
    speed[k] = np.hypot(rel[0, 3], rel[1, 3]) / 0.5

q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]  # noqa: E731
frac = lambda m: round(float(np.mean(m)), 4)  # noqa: E731


def by(edges, values, label):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):  # noqa: B905
        m = (values >= lo) & (values < hi)
        if m.any():
            out.append(
                {
                    label: [lo, hi],
                    "pairs": int(m.sum()),
                    "paths_unmatched": frac(unmatched[m]),
                    "pick_unmatched": frac(pick_unmatched[m]),
                    "closest_m_p50": round(float(np.median(closest[m])), 3),
                }
            )
    return out


chain_pos = table["chain_pos"][pairs]
start_val, start_graded = prev[rows_val] < 0, (prev[rows_val] < 0) & graded[rows_val]
result = {
    "val_logs": int(len(set(table["log"][rows_val].tolist()))),
    "val_moments": int(len(rows_val)),
    "chain_starts": int(start_val.sum()),
    "chain_start_fraction_of_moments": frac(start_val),
    "graded_val_moments": int(graded[rows_val].sum()),
    "graded_chain_starts": int(start_graded.sum()),
    "back_to_back_pairs": int(len(pairs)),
    "new_path_closest_old_m_p10_50_90": q(closest),
    "new_path_closest_old_is_same_slot": frac(same),
    "paths_unmatched_beyond_limit": frac(unmatched),
    "moments_with_any_path_unmatched": frac(unmatched.any(1)),
    "moments_with_8_or_more_unmatched": frac(unmatched.sum(1) >= 8),
    "moments_with_all_32_unmatched": frac(unmatched.all(1)),
    "pick_unmatched": frac(pick_unmatched),
    "paths_starting_empty_incl_chain_starts": round(
        float((unmatched.sum() + 32 * start_val.sum()) / (32 * len(rows_val))), 4
    ),
    "by_turn_rad_per_step": by([0, 0.02, 0.05, 0.1, 0.2, 4], yaw, "turn_rad"),
    "by_speed_mps": by([0, 1, 5, 10, 15, 40], speed, "speed_mps"),
    "by_chain_pos": by([1, 2, 3, 5, 9, 17, 100], chain_pos, "chain_pos"),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result, indent=1))
