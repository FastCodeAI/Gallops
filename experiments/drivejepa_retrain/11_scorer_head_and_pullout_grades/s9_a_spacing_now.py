#!/usr/bin/env python
"""S9 part A support (CPU, read-only on S6's UNSEEN dump): the size of the proposed spacing term on today's pass-9 menus, so the weight
sweep in A_spec.md is anchored. Term per plan = mean over k of |d(k+1) - 2 d(k) + d(k-1)|, d(k) = distance between consecutive waypoints
(d(1) from the car's position), 6 values per plan, metres. Also the generator's own distance term for scale: mean over the 8 poses of
the L1 distance (x, y, heading) between the human path and the closest of the 32 plans (drive_jepa_agent_s35.py:313-317)."""
import pathlib, numpy as np, pandas as pd
from s7_common import load, RD
HERE = pathlib.Path(__file__).resolve().parent; D = load("unseen", ["menu_9", "human", "grade_9"]); VOC = np.load(RD / "04_training_set/vocab_10240.npy")[:, 4::5]
def term(P):
    xy = P[..., :2].astype(np.float64); prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], -2); d = np.linalg.norm(xy - prev, axis=-1); return np.abs(d[..., 2:] - 2 * d[..., 1:-1] + d[..., :-2]).mean(-1)
tm = term(D["menu_9"]); th = term(D["human"]); tc = term(VOC); l1 = np.abs(D["menu_9"].astype(np.float64) - D["human"][:, None].astype(np.float64)).sum(-1).mean(-1).min(1); hc = D["grade_9"][..., 7]; R = []
for sl in ("all", "highway", "close-pass", "ordinary"):
    k = np.ones(len(tm), bool) if sl == "all" else D["slice"] == sl
    R.append(dict(slice=sl, instants=int(k.sum()), spacing_term_all_32_plans_mean_m=float(tm[k].mean()), median_m=float(np.median(tm[k])), p90_m=float(np.percentile(tm[k], 90)), human_path_mean_m=float(th[k].mean()), human_p90_m=float(np.percentile(th[k], 90)),
                  plans_failing_comfort_mean_m=float(tm[k][hc[k] < 1].mean()), plans_passing_comfort_mean_m=float(tm[k][hc[k] >= 1].mean()), generator_distance_term_mean=float(l1[k].mean()), spacing_over_distance_term=float(tm[k].mean() / l1[k].mean())))
for lab, k in (("highway, car >= 20 m/s", (D["slice"] == "highway") & (D["speed"] >= 20)), ("highway, car < 20 m/s", (D["slice"] == "highway") & (D["speed"] < 20))):
    R.append(dict(slice=lab, instants=int(k.sum()), spacing_term_all_32_plans_mean_m=float(tm[k].mean()), median_m=float(np.median(tm[k])), p90_m=float(np.percentile(tm[k], 90)), human_path_mean_m=float(th[k].mean()), human_p90_m=float(np.percentile(th[k], 90)),
                  plans_failing_comfort_mean_m=float(tm[k][hc[k] < 1].mean()), plans_passing_comfort_mean_m=float(tm[k][hc[k] >= 1].mean()), generator_distance_term_mean=float(l1[k].mean()), spacing_over_distance_term=float(tm[k].mean() / l1[k].mean())))
R.append(dict(slice="catalogue curves (10,240)", instants=len(VOC), spacing_term_all_32_plans_mean_m=float(tc.mean()), median_m=float(np.median(tc)), p90_m=float(np.percentile(tc, 90))))
o = pd.DataFrame(R); o.to_csv(HERE / "a_spacing_now.csv", index=False); pd.set_option("display.width", 250); print(o.round(3).T.to_string())
