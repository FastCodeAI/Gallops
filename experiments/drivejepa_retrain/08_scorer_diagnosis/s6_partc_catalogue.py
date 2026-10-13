#!/usr/bin/env python
"""S6 part c, the two catalogue items (CPU, no model run; reads the UNSEEN dump and S3's stored catalogue grades).
(1) Spot check: on 20 UNSEEN examples (one per clip, seeded), re-grade 20 stored-graded catalogue curves each with the same calculator
    call S6 uses for menus (s3_scorer.score) and compare with the stored total grade (anchors_scores_final.npy, 16-bit).
(2) P-c2, first half: for every UNSEEN example and every pass-9 menu plan, the nearest catalogue curve (mean xy distance over the 8
    poses; catalogue cut to the same 8 poses by [:, 4::5]) and its stored total grade, against the plan's true grade from the dump.
    Curves the S3 pre-filter skipped have no stored grade; those plans are counted and left out of the difference.
Intervals: clip-level bootstrap, 95 %, 1000 resamples, seed 20261007."""
import sys, glob, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent; C = pathlib.Path("{runs}/drivejepa_retrain/04_training_set/s1/cache"); SEED = 20261007
man = pd.read_parquet(HERE / "set_unseen.parquet").set_index("sample_id"); VOC40 = np.load(RD / "04_training_set/vocab_10240.npy"); VOC = VOC40[:, 4::5]; rng = np.random.default_rng(SEED)
# ---- (1) spot check
from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS
import s3_scorer as SS
one = man.reset_index().groupby("clip_uuid").sample(1, random_state=SEED).sample(20, random_state=SEED); rows = []
for _, q in one.iterrows():
    d = C / q.clip_uuid / q.sample_id; fin = np.load(d / "anchors_scores_final.npy").astype(np.float64); ok = np.where(np.isfinite(fin))[0]; ii = np.sort(rng.choice(ok, size=min(20, len(ok)), replace=False))
    sc, _ = SS.score(CNS.load_metric_cache(str(d / "metric_cache.pkl")), VOC[ii].astype(np.float32))
    for j, a, b in zip(ii, fin[ii], sc[:, -1]): rows.append(dict(sample_id=q.sample_id, slice=q["slice"], curve=int(j), stored=a, regraded=float(b), diff=float(b - a)))
s = pd.DataFrame(rows); s.to_csv(HERE / "partc_spotcheck_20x20.csv", index=False)
print("spot check:", len(s), "curves on", s.sample_id.nunique(), "examples | max |diff| %.5f | share within 0.001 (16-bit storage) %.3f | share within 0.01 %.3f | mean stored %.3f" % (s["diff"].abs().max(), (s["diff"].abs() < 1e-3).mean(), (s["diff"].abs() < 1e-2).mean(), s.stored.mean()), flush=True)
# ---- (2) nearest-catalogue grade vs true grade, pass-9 menu
V2 = VOC[:, :, :2].reshape(len(VOC), -1).astype(np.float32); vn = (V2 ** 2).sum(1); R = []
for f in sorted(glob.glob("{runs}/drivejepa_retrain/08_scorer_diagnosis/dump/unseen/shard_*.npz")):
    z = np.load(f); tok = z["token"]; menu = z["menu_9"]; G = z["grade_9"][:, :, -1]
    for i, t in enumerate(tok):
        g = G[i]; q = man.loc[t]; fin = np.load(C / q.clip_uuid / t / "anchors_scores_final.npy").astype(np.float32); P = menu[i][:, :, :2]; P2 = P.reshape(32, -1)
        cand = np.argsort((P2 ** 2).sum(1)[:, None] - 2 * P2 @ V2.T + vn[None], axis=1)[:, :16]                      # 16 nearest by summed squared distance, then exact mean distance
        dm = np.linalg.norm(P[:, None] - VOC[cand][:, :, :, :2], axis=-1).mean(-1); k = dm.argmin(1); nn = cand[np.arange(32), k]; st = fin[nn]; has = np.isfinite(st)
        R.append(dict(token=t, clip=q.clip_uuid, slice=q["slice"], plans_with_stored=int(has.sum()), nearest_dist_m=float(dm[np.arange(32), k].mean()), true_mean=float(g[has].mean()) if has.any() else np.nan, stored_mean=float(st[has].mean()) if has.any() else np.nan,
                      diff_mean=float((g[has] - st[has]).mean()) if has.any() else np.nan, share_true_lower_by_0p1=float((g[has] < st[has] - 0.1).mean()) if has.any() else np.nan, share_true_higher_by_0p1=float((g[has] > st[has] + 0.1).mean()) if has.any() else np.nan,
                      true_best=float(g.max()), stored_best_of_nearest=float(st[has].max()) if has.any() else np.nan, catalogue_best=float(np.nanmax(fin)) if np.isfinite(fin).any() else np.nan))
    print(pathlib.Path(f).name, len(R), flush=True)
d = pd.DataFrame(R); d.to_parquet(HERE / "partc_nearest_catalogue_unseen.parquet")
out = []
def boot(x, fn):
    u, inv = np.unique(x["clip"].to_numpy(), return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(u))]; r2 = np.random.default_rng(SEED); v = fn(x); o = []
    for _ in range(1000): o.append(fn(x.iloc[np.concatenate([idx[k] for k in r2.integers(0, len(u), len(u))])]))
    lo, hi = np.nanpercentile(o, [2.5, 97.5]); return v, lo, hi
for sl in ("all", "highway", "close-pass", "ordinary"):
    x = d if sl == "all" else d[d["slice"] == sl]; r = dict(slice=sl, examples=len(x), clips=x["clip"].nunique(), plans_with_a_stored_nearest_grade_share=x.plans_with_stored.sum() / (32 * len(x)), nearest_dist_m=x.nearest_dist_m.mean())
    for k in ("true_mean", "stored_mean", "diff_mean", "share_true_lower_by_0p1", "share_true_higher_by_0p1", "true_best", "catalogue_best"): r[k], r[k + "_lo"], r[k + "_hi"] = boot(x, lambda q: np.nanmean(q[k]))
    r["menu_has_ge095"] = float((x.true_best >= 0.95).mean()); r["catalogue_has_ge095"] = float((x.catalogue_best >= 0.95).mean()); out.append(r)
o = pd.DataFrame(out); o.to_csv(HERE / "table_c_nearest_catalogue_unseen.csv", index=False); pd.set_option("display.width", 250); print(o.round(3).T.to_string())
