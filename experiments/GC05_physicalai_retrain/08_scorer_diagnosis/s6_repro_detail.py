#!/usr/bin/env python
"""S6 Part 0.4 detail: how far our 32-bit CPU run is from S4's 16-bit GPU evaluation, example by example (menu-level and pick-level)."""
import glob, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent / "06_first_training_run/run/readouts"; D = pathlib.Path("{runs}/drivejepa_retrain/08_scorer_diagnosis/dump/unseen")
Z = [np.load(f) for f in sorted(glob.glob(str(D / "shard_*.npz"))) if ".tmp." not in f]; cat = lambda k: np.concatenate([z[k] for z in Z]); tok = cat("token"); hum = cat("human"); n = len(tok); ar = np.arange(n); rows = []
for p in (0, 1, 9):
    r = pd.read_parquet(RD / f"val_full_pass{p:02d}.parquet").set_index("token").loc[tok]; g = cat(f"grade_{p}")[:, :, -1]; s = 1 / (1 + np.exp(-cat(f"logit_{p}")[:, :, -1].astype(np.float64))); menu = cat(f"menu_{p}")
    ch = s.argmax(1); d4 = np.linalg.norm(menu[:, :, -1, :2] - hum[:, None, -1, :2], axis=2); dd = np.abs(d4[ar, ch] - r.dist4.to_numpy()); db = np.abs(d4.min(1) - r.best_dist4.to_numpy()); dg = np.abs(g[ar, ch] - r.ref_chosen.to_numpy()); dbg = np.abs(g.max(1) - r.ref_best.to_numpy())
    st = np.diff(np.concatenate([np.zeros((n, 32, 1, 2)), menu[:, :, :, :2]], 2), axis=2); fe = np.linalg.norm(st[:, :, -1], axis=2).max(1) / 0.5; s2 = np.sort(s, 1)
    rows.append(dict(passno=p, n=n, menu_closest_plan_dist_median_abs_diff_m=np.median(db), menu_closest_plan_dist_p90_abs_diff_m=np.quantile(db, .9), menu_fastest_speed_median_abs_diff_mps=np.median(np.abs(fe - r.fastest_end.to_numpy())),
                     menu_best_grade_same_within_0p02=(dbg < .02).mean(), pick_dist_same_within_0p5m=(dd < .5).mean(), pick_dist_median_abs_diff_m=np.median(dd), pick_grade_same_within_0p02=(dg < .02).mean(),
                     top2_score_gap_median=np.median(s2[:, -1] - s2[:, -2]), top2_score_gap_below_0p001=((s2[:, -1] - s2[:, -2]) < 1e-3).mean(), top2_score_gap_below_0p01=((s2[:, -1] - s2[:, -2]) < 1e-2).mean(), top5_score_gap_median=np.median(s2[:, -1] - s2[:, -5])))
t = pd.DataFrame(rows); t.to_csv(HERE / f"repro_detail_{n}.csv", index=False); print(t.round(4).T.to_string())
