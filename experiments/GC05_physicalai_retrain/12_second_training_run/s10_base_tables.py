#!/usr/bin/env python
"""S10: baseline tables from a finished UNSEEN dump folder (S6's batch-8 dump, the batch-1 dump of s10_dump_b1.py, or a new generator's dump
made by the same script). Definitions and the clip-level bootstrap (1,000 resamples, seed 20261007) are those of the task-1 audit
(loop/.../01_audit-baselines-and-means/report/audit_baselines.py); slices and car speed come from 08_scorer_diagnosis/set_unseen.parquet by token.
Reads only arrays menu_9 (or --menu KEY), grade_9, logit_9, human. CPU, no model.
Usage: s10_base_tables.py DUMP_DIR OUT_DIR [--tag NAME] [--menu menu_9 --grade grade_9 --logit logit_9]"""
import sys, json, glob, pathlib, argparse, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent; sys.path.insert(0, str(RD / "09_three_hypotheses"))
from s7_common import step_speeds, spearman_rows, SEED_BOOT, NB
ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("--tag", default=""); ap.add_argument("--menu", default="menu_9"); ap.add_argument("--grade", default="grade_9"); ap.add_argument("--logit", default="logit_9")
a = ap.parse_args(); DUMP = pathlib.Path(a.dump); OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True); TAG = a.tag or DUMP.name
fs = sorted(f for f in glob.glob(str(DUMP / "shard_*.npz")) if ".tmp." not in f); D = {k: [] for k in ("token", a.menu, a.grade, a.logit, "human")}
for f in fs:
    z = np.load(f)
    for k in D: D[k].append(z[k])
D = {k: np.concatenate(v) for k, v in D.items()}; man = pd.read_parquet(RD / "08_scorer_diagnosis/set_unseen.parquet").set_index("sample_id")
assert len(D["token"]) == len(man) == 11842 and set(D["token"]) == set(man.index), ("dump is not the full UNSEEN graded set", len(fs), len(D["token"]))
mm = man.loc[D["token"]]; SLICE = mm["slice"].to_numpy(); CLIP = mm.clip_uuid.to_numpy(); SPEED = mm.speed.to_numpy(float); N = len(SLICE)
G = D[a.grade][..., -1].astype(np.float64); HC = D[a.grade][..., 7] >= 1; M = D[a.menu].astype(np.float64); H = D["human"].astype(np.float64)
sp = step_speeds(M); swing = sp.max(-1) - sp.min(-1); ar = np.arange(N); gmax = G.max(1); good = gmax >= 0.95
d4 = np.linalg.norm(M[:, :, -1, :2] - H[:, None, -1, :2], axis=-1); best = np.where(G >= gmax[:, None] - 1e-9, d4, np.inf).argmin(1)      # S7 rule: highest grade, ties -> end closest to the human
score = D[a.logit][..., -1].astype(np.float64); pick = score.argmax(1); gp = G[ar, pick]; l1 = np.abs(M - H[:, None]).sum(-1).mean(-1).min(1); rho = spearman_rows(score, G)
def boot(mask, stat):
    ii = np.where(mask)[0]; u, inv = np.unique(CLIP[ii], return_inverse=True); idx = [ii[inv == k] for k in range(len(u))]; rng = np.random.default_rng(SEED_BOOT); v = np.asarray(stat(ii), float); out = []
    for _ in range(NB): out.append(stat(np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])))
    lo, hi = np.nanpercentile(np.asarray(out, float), [2.5, 97.5], axis=0); return v, lo, hi, len(ii), len(u)
rows = []
def add(name, sl, mask, stat):
    if not mask.any(): return
    v, lo, hi, n, nc = boot(mask, stat); rows.append(dict(dump=TAG, measure=name, slice=sl, instants=n, clips=nc, value=float(v), lo95=float(lo), hi95=float(hi))); print(f"{name:62s} {sl:24s} n={n:6d} clips={nc:4d}  {v:.4f} ({lo:.4f}-{hi:.4f})", flush=True)
HW = SLICE == "highway"; HW20 = HW & (SPEED >= 20)
SL = {"highway": HW, "close-pass": SLICE == "close-pass", "ordinary": SLICE == "ordinary", "all": np.ones(N, bool), "highway, car >= 20 m/s": HW20}
for sl, k in SL.items():
    add("share of instants with a menu plan graded >= 0.95", sl, k, lambda i: good[i].mean())
    add("swing, all 32 plans, median m/s", sl, k, lambda i: np.median(swing[i]))
    add("swing, best plan (S7 tie-break), median m/s", sl, k, lambda i: np.median(swing[i, best[i]]))
    add("swing, best plan on no-good-plan instants, median m/s", sl, k & ~good, lambda i: np.median(swing[i, best[i]]))
    add("share of all 32 plans passing comfort", sl, k, lambda i: HC[i].mean())
    add("human-path term (L1 x,y,heading; mean 8 poses; best of 32), mean", sl, k, lambda i: l1[i].mean())
    add("best-of-32 distance to the human at 4 s, mean m", sl, k, lambda i: d4[i].min(1).mean())
    add("best-of-32 distance to the human at 4 s, median m", sl, k, lambda i: np.median(d4[i].min(1)))
    add("scorer picks graded exactly 0, share", sl, k, lambda i: (gp[i] == 0).mean())
    add("scorer regret (best grade - pick grade), mean", sl, k, lambda i: (gmax[i] - gp[i]).mean())
    add("rank agreement (Spearman), median over instants", sl, k, lambda i: np.nanmedian(rho[i]))
    add("pick = true best (ties count), share", sl, k, lambda i: (gp[i] >= gmax[i] - 1e-9).mean())
T = pd.DataFrame(rows); T.to_csv(OUT / f"base_tables_{TAG}.csv", index=False)
v, lo, hi, n, nc = boot(HW20, lambda i: HC[i].mean(0)); slot = pd.DataFrame(dict(dump=TAG, slot=np.arange(32), comfort_pass_rate=v, lo95=lo, hi95=hi, swing_median_mps=np.median(swing[HW20], 0), mean_grade=G[HW20].mean(0), instants=n, clips=nc))
slot.to_csv(OUT / f"slot_comfort_highway_ge20_{TAG}.csv", index=False)
g = lambda meas, sl: float(T[(T.measure == meas) & (T["slice"] == sl)].value.iloc[0])
key = dict(dump=str(DUMP), shards=len(fs), instants=N, clips=int(len(np.unique(CLIP))),
           highway_share_good_plan_pct=100 * g("share of instants with a menu plan graded >= 0.95", "highway"), highway_ge20_swing_all32_median_mps=g("swing, all 32 plans, median m/s", "highway, car >= 20 m/s"),
           highway_ge20_slots_comfort_above_25pct=int((v > 0.25).sum()), highway_ge20_max_slot_comfort_rate=float(v.max()), highway_human_path_term=g("human-path term (L1 x,y,heading; mean 8 poses; best of 32), mean", "highway"),
           all_picks_graded_0_pct=100 * g("scorer picks graded exactly 0, share", "all"), all_regret_mean=g("scorer regret (best grade - pick grade), mean", "all"), all_rank_agreement_median=g("rank agreement (Spearman), median over instants", "all"),
           highway_ge20_swing_best_plan_median_mps=g("swing, best plan (S7 tie-break), median m/s", "highway, car >= 20 m/s"), highway_swing_best_plan_no_good_plan_median_mps=g("swing, best plan on no-good-plan instants, median m/s", "highway"),
           highway_ge20_share_good_plan_pct=100 * g("share of instants with a menu plan graded >= 0.95", "highway, car >= 20 m/s"), all_pick_equals_best_pct=100 * g("pick = true best (ties count), share", "all"))
json.dump(key, open(OUT / f"base_tables_{TAG}.json", "w"), indent=1); print(json.dumps(key, indent=1))
