#!/usr/bin/env python
"""S6 parts c.2/c.3 and a.2-a.4 from the dumps (s6_dump.py). No model run. One row per (set, table, scorer, selection, slice).
table "own"  (PRIMARY)   : checkpoint p whole, on its own menu, graded by the EPDMS calculator.
table "swap" (OFF-DESIGN): score head of checkpoint q on the pass-9 trunk's features, pass-9 menu and grades.
Rulers on the pick: strict match, graded >= 0.9, within 0.05 of the best grade, regret, within 1.5 m of the human at 4 s (S4's ruler).
Intervals: clip-level bootstrap, 95 %, 1000 resamples, seed 20261007.
Usage: s6_analyze.py [--partial]   (--partial: use whatever shards exist and say so in the output names)"""
import sys, glob, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = pathlib.Path(__file__).resolve().parent; NAS = pathlib.Path("{runs}/drivejepa_retrain/08_scorer_diagnosis/dump"); PARTIAL = "--partial" in sys.argv; SEED = 20261007; NB = 1000
TERMS = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]; SL = ["all", "highway", "close-pass", "ordinary"]

def load(name):
    fs = [f for f in sorted(glob.glob(str(NAS / name / "shard_*.npz"))) if ".tmp." not in f]
    if not fs: return None
    Z = [np.load(f) for f in fs]; D = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files if k != "states_main"}; D["_files"] = fs
    man = pd.read_parquet(HERE / f"set_{name}.parquet").set_index("sample_id")
    if not PARTIAL: assert len(D["token"]) == len(man) and set(D["token"]) == set(man.index), f"{name}: dump incomplete ({len(D['token'])} of {len(man)})"
    m = man.loc[D["token"]]; D["slice"] = m["slice"].to_numpy(); D["clip"] = m.clip_uuid.to_numpy(); D["anchor_t"] = m.anchor_t.to_numpy(); return D

def spearman_rows(s, g, s4=False):
    """Per-example Spearman over the 32 plans. Standard: average ranks. s4=True: S4's version (s4_train.py:86-88), ties by slot order."""
    rk = (lambda x: np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1, kind="stable").astype(float)) if s4 else (lambda x: rankdata(x, axis=1))
    a = rk(s); b = rk(g); a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True); den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)

def boot(clip, cols, stat):
    """clip-level bootstrap of stat(dict of per-example arrays) -> (value, lo, hi)"""
    v = stat(cols); u, inv = np.unique(clip, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(u))]; rng = np.random.default_rng(SEED); out = []
    for _ in range(NB):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))]); out.append(stat({k: x[ii] for k, x in cols.items()}))
    lo, hi = np.nanpercentile(out, [2.5, 97.5]); return float(v), float(lo), float(hi)

def momentum_pick(D, s):
    """Momentum-aware selection as adapter 0.1.4b (adapters/drivejepa014b/momentum.py:216-221 -> drive_jepa_model.py:98-151): the score
    becomes (14 * score + 2 * comfort) / 16, comfort = two-frame extended comfort between the previous tick's chosen plan and each
    current plan. Offline stand-in: the previous graded example of the same clip, 0.5 s earlier, and ITS momentum pick. Pass 9 only."""
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
    fs = D["_files"]; st = np.concatenate([np.load(f)["states_main"] for f in fs]); n = len(s); pick = s.argmax(1).copy(); used = np.zeros(n, bool); o = 5; tp = np.arange(41 - o) * 0.1
    order = np.lexsort((D["anchor_t"], D["clip"])); prev = None
    for i in order:
        if prev is not None and D["clip"][prev] == D["clip"][i] and abs(D["anchor_t"][i] - D["anchor_t"][prev] - 0.5) < 1e-3:
            cur = np.repeat(st[prev, pick[prev]][None, o:], 32, 0); nxt = st[i][:, :-o]; c = ego_is_two_frame_extended_comfort(cur, nxt, tp).astype(np.float64)
            pick[i] = int(((14.0 * s[i] + 2.0 * c) / 16.0).argmax()); used[i] = True
        prev = i
    return pick, used

def rulers(D, g9, d4, pick_onehot_or_w):
    """pick given as weights over the 32 plans (one-hot for a pick; uniform for the random baseline, giving expected values)"""
    w = pick_onehot_or_w; gt = g9[:, :, -1]; best = gt.max(1, keepdims=True)
    return dict(strict=(w * (gt >= best - 1e-9)).sum(1), ge09=(w * (gt >= 0.9)).sum(1), within005=(w * (best - gt <= 0.05)).sum(1), regret=(w * (best - gt)).sum(1), within1p5m=(w * (d4 <= 1.5)).sum(1), grade=(w * gt).sum(1))

def onehot(ix): w = np.zeros((len(ix), 32)); w[np.arange(len(ix)), ix] = 1.0; return w

def summarise(setname, table, scorer, sel, D, R, extra=None):
    rows = []
    for sl in SL:
        m = np.ones(len(D["token"]), bool) if sl == "all" else D["slice"] == sl
        if m.sum() == 0: continue
        r = dict(set=setname, table=table, scorer=scorer, selection=sel, slice=sl, examples=int(m.sum()), clips=int(len(set(D["clip"][m]))))
        for k in ("strict", "ge09", "within005", "within1p5m", "regret", "grade"):
            r[k], r[k + "_lo"], r[k + "_hi"] = boot(D["clip"][m], {"x": R[k][m]}, lambda c: np.mean(c["x"]))
        r["regret_p90"], r["regret_p90_lo"], r["regret_p90_hi"] = boot(D["clip"][m], {"x": R["regret"][m]}, lambda c: np.quantile(c["x"], 0.9))
        r["gap_ge09_minus_1p5m"], r["gap_lo"], r["gap_hi"] = boot(D["clip"][m], {"a": R["ge09"][m], "b": R["within1p5m"][m]}, lambda c: np.mean(c["a"]) - np.mean(c["b"]))
        for k, x in (extra or {}).items():
            r[k + "_median"], r[k + "_median_lo"], r[k + "_median_hi"] = boot(D["clip"][m], {"x": x[m]}, lambda c: np.nanmedian(c["x"])); r[k + "_mean"], r[k + "_mean_lo"], r[k + "_mean_hi"] = boot(D["clip"][m], {"x": x[m]}, lambda c: np.nanmean(c["x"]))
        rows.append(r)
    return rows

def menu_distance(A, B):
    """mean over plans of A of the distance to the nearest plan of B (mean over the 8 poses of the xy distance), metres; per example"""
    d = np.linalg.norm(A[:, :, None, :, :2] - B[:, None, :, :, :2], axis=-1).mean(-1); return d.min(2).mean(1)

OUT = []; MENU = []; MD = []; MARGIN = []
ONLY = [x.split("=")[1] for x in sys.argv if x.startswith("--set=")]
for name in (ONLY or ("unseen", "seen")):
    D = load(name)
    if D is None: print(name, "no dump yet"); continue
    n = len(D["token"]); ar = np.arange(n); hum = D["human"]; S = name.upper(); print(S, n, "examples", len(set(D["clip"])), "clips", "(PARTIAL)" if PARTIAL else "")
    sig = lambda x: 1 / (1 + np.exp(-x.astype(np.float64)))
    for p, pname in [(0, "shipped"), (1, "pass 1"), (9, "pass 9")]:
        g = D[f"grade_{p}"]; s = sig(D[f"logit_{p}"][:, :, -1]); menu = D[f"menu_{p}"]; d4 = np.linalg.norm(menu[:, :, -1, :2] - hum[:, None, -1, :2], axis=2); gt = g[:, :, -1]; s2 = np.sort(s, 1)
        ex = dict(rank_agreement=spearman_rows(s, gt), rank_agreement_s4def=spearman_rows(s, gt, s4=True), top2_gap=s2[:, -1] - s2[:, -2])
        OUT += summarise(S, "own", pname, "raw score", D, rulers(D, g, d4, onehot(s.argmax(1))), ex)
        if p == 9:
            try:
                mp, used = momentum_pick(D, s); OUT += [dict(r, momentum_used_share=float(used.mean()), pick_changed_share=float((mp != s.argmax(1)).mean())) for r in summarise(S, "own", pname, "momentum-aware (0.1.4b)", D, rulers(D, g, d4, onehot(mp)))]
            except Exception as e: print("momentum-aware selection failed:", repr(e)[:200])
        if p == 9:   # baselines on the pass-9 menu
            OUT += summarise(S, "own", "baseline: random pick (expected)", "-", D, rulers(D, g, d4, np.full((n, 32), 1 / 32)))
            OUT += summarise(S, "own", "baseline: slot 0 (the menu has no order; first slot)", "-", D, rulers(D, g, d4, onehot(np.zeros(n, int))))
            OUT += summarise(S, "own", "baseline: plan closest to the human", "-", D, rulers(D, g, d4, onehot(d4.argmin(1))))
            OUT += summarise(S, "own", "ceiling: best-graded plan (closest to human among ties)", "-", D, rulers(D, g, d4, onehot(np.where(gt >= gt.max(1, keepdims=True) - 1e-9, d4, np.inf).argmin(1))))
            for q, qname in [(0, "shipped head on pass-9 trunk"), (1, "pass-1 head on pass-9 trunk")]:
                sq = sig(D[f"swap_{q}"][:, :, -1]); q2 = np.sort(sq, 1); OUT += summarise(S, "swap", qname, "raw score", D, rulers(D, g, d4, onehot(sq.argmax(1))), dict(rank_agreement=spearman_rows(sq, gt), rank_agreement_s4def=spearman_rows(sq, gt, s4=True), top2_gap=q2[:, -1] - q2[:, -2]))
            OUT += [dict(r, table="swap") for r in summarise(S, "own", "pass 9 (same rows as primary, for reference)", "raw score", D, rulers(D, g, d4, onehot(s.argmax(1))), ex)]
        # top-2 margin (PREDICTIONS_v4.md, P-a5): score gap between the two top-scored plans and the true-grade gap between those same two
        o2 = np.argsort(-s, axis=1, kind="stable")[:, :2]; sg = s[ar, o2[:, 0]] - s[ar, o2[:, 1]]; gg = np.abs(gt[ar, o2[:, 0]] - gt[ar, o2[:, 1]]); worse = gt[ar, o2[:, 0]] < gt[ar, o2[:, 1]] - 0.1
        for sl in SL:
            m = np.ones(n, bool) if sl == "all" else D["slice"] == sl
            if m.sum() == 0: continue
            r = dict(set=S, scorer=pname, slice=sl, examples=int(m.sum()), clips=int(len(set(D["clip"][m])))); c = {"sg": sg[m], "gg": gg[m], "w": worse[m].astype(float)}; near = lambda q: q["sg"] < 0.01
            for k, f in dict(score_gap_median=lambda q: np.median(q["sg"]), share_score_gap_lt_0p01=lambda q: np.mean(near(q)), grade_gap_median=lambda q: np.median(q["gg"]), share_grade_gap_gt_0p1=lambda q: np.mean(q["gg"] > 0.1),
                             of_near_ties_share_grade_gap_gt_0p1=lambda q: np.mean(q["gg"][near(q)] > 0.1) if near(q).any() else np.nan, of_near_ties_share_top_is_the_worse_by_gt_0p1=lambda q: np.mean(q["w"][near(q)]) if near(q).any() else np.nan).items():
                r[k], r[k + "_lo"], r[k + "_hi"] = boot(D["clip"][m], c, f)
            MARGIN.append(r)
        # menu facts: good plan on the menu, can the two rulers agree, per-check pass rates
        best = gt.max(1, keepdims=True); isb = gt >= best - 1e-9
        for sl in SL:
            m = np.ones(n, bool) if sl == "all" else D["slice"] == sl
            if m.sum() == 0: continue
            r = dict(set=S, menu=pname, slice=sl, examples=int(m.sum()), clips=int(len(set(D["clip"][m]))))
            for k, x in dict(menu_has_grade_ge095=(gt.max(1) >= 0.95), menu_has_grade_ge09=(gt.max(1) >= 0.9), menu_has_plan_within_1p5m=(d4.min(1) <= 1.5), best_graded_plan_within_1p5m_any_tie=((isb & (d4 <= 1.5)).any(1)),
                             best_graded_plan_within_1p5m_all_ties=((isb & (d4 <= 1.5)).sum(1) == isb.sum(1)), plans_tied_for_best=isb.sum(1), plans_graded_ge09=(gt >= 0.9).sum(1), best_grade=gt.max(1), mean_grade=gt.mean(1), d3_exempt_any=D[f"exempt_{p}"].any(1)).items():
                r[k], r[k + "_lo"], r[k + "_hi"] = boot(D["clip"][m], {"x": x[m].astype(float)}, lambda c: np.mean(c["x"]))
            for j, t in enumerate(TERMS[:-1]): r[f"plans_failing_{t}"] = float((g[m][:, :, j] < 1.0).mean())
            MENU.append(r)
    for q, qname in [(1, "pass 1"), (0, "shipped")]:
        a = menu_distance(D["menu_9"], D[f"menu_{q}"]); b = menu_distance(D[f"menu_{q}"], D["menu_9"])
        for sl in SL:
            m = np.ones(n, bool) if sl == "all" else D["slice"] == sl
            if m.sum() == 0: continue
            r = dict(set=S, pair=f"pass 9 vs {qname}", slice=sl, examples=int(m.sum()))
            r["nearest_plan_dist_9_to_other_m"], r["lo1"], r["hi1"] = boot(D["clip"][m], {"x": a[m]}, lambda c: np.mean(c["x"])); r["nearest_plan_dist_other_to_9_m"], r["lo2"], r["hi2"] = boot(D["clip"][m], {"x": b[m]}, lambda c: np.mean(c["x"])); MD.append(r)
tag = ("_PARTIAL" if PARTIAL else "") + ("_" + "_".join(ONLY) if ONLY else "")
pd.DataFrame(OUT).to_csv(HERE / f"table_a_c{tag}.csv", index=False); pd.DataFrame(MENU).to_csv(HERE / f"table_menu{tag}.csv", index=False); pd.DataFrame(MD).to_csv(HERE / f"table_menu_distance{tag}.csv", index=False); pd.DataFrame(MARGIN).to_csv(HERE / f"table_top2_margin{tag}.csv", index=False)
o = pd.DataFrame(OUT); pd.set_option("display.width", 250)
print(o[o.slice == "all"][["set", "table", "scorer", "selection", "examples", "strict", "ge09", "within005", "within1p5m", "regret", "regret_p90", "gap_ge09_minus_1p5m", "rank_agreement_median", "rank_agreement_mean", "rank_agreement_s4def_mean", "top2_gap_median"]].round(3).to_string(index=False))
print(pd.DataFrame(MENU).query("slice == 'all'")[["set", "menu", "menu_has_grade_ge095", "menu_has_plan_within_1p5m", "best_graded_plan_within_1p5m_any_tie", "plans_tied_for_best", "plans_graded_ge09"]].round(3).to_string(index=False)); print(pd.DataFrame(MD).round(3).to_string(index=False))
print(pd.DataFrame(MARGIN)[["set", "scorer", "slice", "examples", "score_gap_median", "share_score_gap_lt_0p01", "of_near_ties_share_grade_gap_gt_0p1", "of_near_ties_share_top_is_the_worse_by_gt_0p1", "share_grade_gap_gt_0p1"]].round(4).to_string(index=False))
