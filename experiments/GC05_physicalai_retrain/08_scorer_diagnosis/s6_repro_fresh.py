#!/usr/bin/env python
"""S6 Part 0.4, fresh half: our CPU dump vs S4's stored per-example readouts on val_full, example by example and in aggregate.
Usage: s6_repro_fresh.py [dump dir name, default 'dump']"""
import sys, glob, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent / "06_first_training_run/run/readouts"; D = pathlib.Path("{runs}/drivejepa_retrain/08_scorer_diagnosis") / (sys.argv[1] if len(sys.argv) > 1 else "dump") / "unseen"
Z = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(str(D / "shard_*.npz"))) if ".tmp." not in f]; cat = lambda k: np.concatenate([z[k] for z in Z]); tok = cat("token"); hum = cat("human"); rows = []; CL = dict(pd.read_parquet(HERE / 'set_unseen.parquet')[['sample_id', 'clip_uuid']].values); print("examples in dump:", len(tok), "shards", len(Z))
for p, name in [(0, "shipped (pass 0)"), (1, "pass 1"), (9, "pass 9")]:
    r = pd.read_parquet(RD / f"val_full_pass{p:02d}.parquet").set_index("token").loc[tok]; g = cat(f"grade_{p}")[:, :, -1]; s = 1 / (1 + np.exp(-cat(f"logit_{p}")[:, :, -1])); menu = cat(f"menu_{p}"); n = len(tok); ar = np.arange(n)
    ch = s.argmax(1); gc = g[ar, ch]; top1 = gc >= g.max(1) - 1e-9; d4 = np.linalg.norm(menu[:, :, -1, :2] - hum[:, None, -1, :2], axis=2); cp = (r.type == "B_pass").to_numpy()
    rows.append(dict(scorer=name, n=n, fresh_pick_is_best=top1.mean(), s4_pick_is_best_same_examples=r.top1.mean(), same_verdict_per_example=(top1 == r.top1.to_numpy()).mean(), max_abs_diff_grade_of_pick=np.abs(gc - r.ref_chosen.to_numpy()).max(),
                     share_grade_of_pick_within_1e3=(np.abs(gc - r.ref_chosen.to_numpy()) < 1e-3).mean(), max_abs_diff_best_grade=np.abs(g.max(1) - r.ref_best.to_numpy()).max(), max_abs_diff_dist4_m=np.abs(d4[ar, ch] - r.dist4.to_numpy()).max(), share_same_pick_by_distance=(np.abs(d4[ar, ch] - r.dist4.to_numpy()) < 1e-2).mean(), median_gap_top2_scores=float(np.median(np.sort(s, 1)[:, -1] - np.sort(s, 1)[:, -2])), median_spread_scores=float(np.median(s.max(1) - s.min(1))),
                     fresh_rank_agreement=float(np.nanmean([pd.Series(s[i]).corr(pd.Series(g[i]), method='spearman') for i in range(n)])), s4_rank_agreement=float(r.rank_corr.mean()), clips=int(r.index.to_series().map(CL).nunique()),
                     closepass_n=int(cp.sum()), fresh_pick_within_1p5m=(d4[ar, ch][cp] <= 1.5).mean() if cp.any() else np.nan, s4_pick_within_1p5m=(r.dist4[cp] <= 1.5).mean() if cp.any() else np.nan,
                     fresh_menu_has_1p5m=(d4.min(1)[cp] <= 1.5).mean() if cp.any() else np.nan, s4_menu_has_1p5m=(r.best_dist4[cp] <= 1.5).mean() if cp.any() else np.nan))
t = pd.DataFrame(rows); t.to_csv(HERE / f"repro_fresh_{len(tok)}.csv", index=False); print(t.round(5).T.to_string())
