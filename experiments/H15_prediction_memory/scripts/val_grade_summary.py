"""True grades on the val logs, for the released pick (momentum off), as run for MEMORY_H15_STEP2.md (B3).

Pick = argmax of the released final logit (momentum off, H15 arm (c) offline). Strict final grade unless
stated. A braking path: its simulated longitudinal acceleration falls below -1.5 m/s^2 within its first
2 s (the threshold of the probe's braking onset). Top-two gap: the two highest strict grades at a moment.

    PYTHONPATH=src python experiments/H15_prediction_memory/scripts/val_grade_summary.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.memory import grade, ticks

ACCEL_X, BRAKE, FIRST_2S = 5, -1.5, 21  # StateIndex.ACCELERATION_X; states at 0.1 s, 0..2.0 s

cfg = SiteConfig.load()
cache = Path(cfg.resolve("{data.navtrain.cache}"))
out = cache / "grades"
table = ticks.load(cache / "ticks")
row = np.load(out / "row.npy")
done = np.load(out / "done.npy")
val = (table["split"][row] == "val") & done
k = np.flatnonzero(val)
rows = row[k]
scores = np.asarray(np.load(out / "scores.npy", mmap_mode="r")[k], np.float64)
forgiving = np.asarray(np.load(out / "forgiving.npy", mmap_mode="r")[k], np.float64)
accel = np.asarray(np.load(out / "states.npy", mmap_mode="r")[k][:, :, :FIRST_2S, ACCEL_X], np.float64)
logits = np.asarray(np.load(cache / "release" / "logits.npy", mmap_mode="r")[rows][:, :, -1])

final = scores[..., -1]
n = len(k)
pick = logits.argmax(1)
at = np.arange(n)
g_pick, g_best = final[at, pick], final.max(1)
top2 = np.sort(final, 1)[:, -2:]
gap = top2[:, 1] - top2[:, 0]
brakes = accel.min(-1) < BRAKE
is_best = final == g_best[:, None]
q = lambda v: [round(float(x), 4) for x in np.percentile(v, [10, 25, 50, 75, 90])]  # noqa: E731
f = lambda m: round(float(np.mean(m)), 4)  # noqa: E731
names = list(grade.SCORE_NAMES)
sub_pick = {name: f(scores[at, pick, j] == 1) for j, name in enumerate(names[:-1])}
result = {
    "val_graded_moments": int(n),
    "val_logs": int(len(set(table["log"][rows].tolist()))),
    "all_val_done": bool(done[table["split"][row] == "val"].all()),
    "pick_grade_mean": round(float(g_pick.mean()), 4),
    "pick_grade_p10_25_50_75_90": q(g_pick),
    "pick_grade_is_1": f(g_pick == 1),
    "pick_grade_is_0": f(g_pick == 0),
    "pick_passes_each_subscore": sub_pick,
    "best_grade_mean": round(float(g_best.mean()), 4),
    "headroom_best_minus_pick_mean": round(float((g_best - g_pick).mean()), 4),
    "headroom_p50_p90": [round(float(x), 4) for x in np.percentile(g_best - g_pick, [50, 90])],
    "mean_grade_all_32": round(float(final.mean()), 4),
    "pick_is_a_best_path_ties_count": f(is_best[at, pick]),
    "best_is_unique": f(is_best.sum(1) == 1),
    "pick_is_the_unique_best": f(is_best[at, pick] & (is_best.sum(1) == 1)),
    "paths_tied_at_best_p50_p90": [int(x) for x in np.percentile(is_best.sum(1), [50, 90])],
    "true_top_two_gap_p10_50_90": [round(float(x), 4) for x in np.percentile(gap, [10, 50, 90])],
    "true_top_two_gap_is_0": f(gap == 0),
    "true_top_two_gap_mean": round(float(gap.mean()), 4),
    "braking_paths_per_moment_mean": round(float(brakes.sum(1).mean()), 2),
    "pick_brakes": f(brakes[at, pick]),
    "a_best_path_brakes_and_pick_does_not": f(
        (is_best & brakes).any(1) & ~brakes[at, pick] & ~is_best[at, pick]
    ),
    "every_best_path_brakes_and_pick_does_not": f(
        (~is_best | brakes).all(1) & ~brakes[at, pick] & ~is_best[at, pick]
    ),
    "forgiving_lane_keeping_of_pick_passes": f(forgiving[at, pick, 0] == 1),
    "strict_lane_keeping_of_pick_passes": f(scores[at, pick, names.index("lane_keeping")] == 1),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result, indent=1))
