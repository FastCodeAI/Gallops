#!/usr/bin/env python
"""S8 L3: EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION. Pull-out scores in absolute terms on d2715023 and b44a4a44.
Source 1 (the brief's): S7's dump of the trial's replans, 09_three_hypotheses/h2_exam_inspection_scores32.csv (adapter 0.1.6, pass-9 weights;
raw score = S_plain, the scorer's output before the momentum step). Leave / stay rule as S7 H2.
Source 2: the SHIPPED weights. The trial folder (07_closed_loop_test/) holds no shipped-model scores for these replans (menus/ has only d016_*).
The 0.1.4b baseline run (shipped weights) of the same two scenes is stored outside it, analysis_3.0/Re-ranker/baseline_014b/menus/
b014b_<scene>/replan_*.npz; it is a DIFFERENT RUN, so not the same replans: same scenes, same replan times after handover, the car
nearly stationary in both. Read-only. Nothing here feeds anything else. Bands (fixed in PREDICTIONS.md): best leave score < 0.2 = scored
as failing, 0.2-0.6 in between, >= 0.6 merely worse."""
import glob, pathlib, numpy as np, pandas as pd
from s7_common import RD
HERE = pathlib.Path(__file__).resolve().parent; B = RD.parent / "Re-ranker/baseline_014b/menus"; R = []
a = pd.read_csv(RD / "09_three_hypotheses/h2_exam_inspection_scores32.csv")
for (sc, t), x in a.groupby(["scene", "t_after_handover_s"], sort=False):
    l = x[x.tag == "leave"]; s = x[x.tag == "stay"]; bl = l.loc[l.raw_score.idxmax()]; bs = s.loc[s.raw_score.idxmax()]
    R.append(dict(scene=sc, model="0.1.6 (pass 9), trial run", t_after_handover_s=t, leave_plans=len(l), best_stay_raw=bs.raw_score, best_leave_raw=bl.raw_score, gap=round(bs.raw_score - bl.raw_score, 4), best_leave_end_x_m=bl.end_x_m, best_leave_end_y_m=bl.end_y_m, leave_median_raw=l.raw_score.median(), leave_min_raw=l.raw_score.min(),
                  leave_plans_below_0p2=int((l.raw_score < 0.2).sum()), leave_plans_ge_0p6=int((l.raw_score >= 0.6).sum()), stay_median_raw=s.raw_score.median(), best_far_leave_raw=(l[(l.end_x_m ** 2 + l.end_y_m ** 2) ** 0.5 >= 5].raw_score.max() if ((l.end_x_m ** 2 + l.end_y_m ** 2) ** 0.5 >= 5).any() else np.nan), car_speed_mps=np.nan))
have = {}
for sc in ("d2715023", "b44a4a44"):
    fs = sorted(glob.glob(str(B / f"b014b_{sc}" / "replan_*.npz"))); have[sc] = len(fs)
    for f in fs:
        z = np.load(f, allow_pickle=True); t = float(z["t_after_ho"])
        if t < -0.01: continue
        P = z["proposals"]; S = z["S_plain"].astype(float) if "S_plain" in z.files else z["pdm_score"].astype(float); x, y = P[:, -1, 0], P[:, -1, 1]; lv = (np.abs(y) >= 2) | (x >= 5); far = lv & (np.hypot(x, y) >= 5)
        if not lv.any() or lv.all(): R.append(dict(scene=sc, model="shipped weights (0.1.4b), different run", t_after_handover_s=round(t, 2), leave_plans=int(lv.sum()), car_speed_mps=float(z["ego_v"]))); continue
        bl = np.where(lv)[0][S[lv].argmax()]; bs = np.where(~lv)[0][S[~lv].argmax()]
        R.append(dict(scene=sc, model="shipped weights (0.1.4b), different run", t_after_handover_s=round(t, 2), leave_plans=int(lv.sum()), best_stay_raw=round(S[bs], 4), best_leave_raw=round(S[bl], 4), gap=round(S[bs] - S[bl], 4), best_leave_end_x_m=round(float(x[bl]), 2), best_leave_end_y_m=round(float(y[bl]), 2),
                      leave_median_raw=float(np.median(S[lv])), leave_min_raw=float(S[lv].min()), leave_plans_below_0p2=int((S[lv] < 0.2).sum()), leave_plans_ge_0p6=int((S[lv] >= 0.6).sum()), stay_median_raw=float(np.median(S[~lv])), best_far_leave_raw=(float(S[far].max()) if far.any() else np.nan), car_speed_mps=float(z["ego_v"]),
                      raw_field=("S_plain" if "S_plain" in z.files else "pdm_score (nudged; no raw stored)")))
d = pd.DataFrame(R); d.to_csv(HERE / "l3_exam_inspection_replans.csv", index=False); S = []
for (sc, m), x in d.groupby(["scene", "model"], sort=False):
    v = x.dropna(subset=["best_leave_raw"]); b = v.best_leave_raw
    S.append(dict(scene=sc, model=m, replans=len(x), replans_with_both_kinds=len(v), leave_plans_median=x.leave_plans.median(), best_stay_median=v.best_stay_raw.median(), best_stay_min=v.best_stay_raw.min(), best_stay_max=v.best_stay_raw.max(), best_leave_median=b.median(), best_leave_min=b.min(), best_leave_max=b.max(),
                  replans_best_leave_below_0p2=int((b < 0.2).sum()), replans_best_leave_0p2_to_0p6=int(((b >= 0.2) & (b < 0.6)).sum()), replans_best_leave_ge_0p6=int((b >= 0.6).sum()), gap_median=v.gap.median(), median_leave_plan_score=v.leave_median_raw.median(), leave_plans_below_0p2_median=v.leave_plans_below_0p2.median(),
                  leave_plans_ge_0p6_median=v.leave_plans_ge_0p6.median(), best_leave_ending_5m_or_more_away_median=v.best_far_leave_raw.median(), best_leave_end_x_median=v.best_leave_end_x_m.median(), best_leave_end_abs_y_median=v.best_leave_end_y_m.abs().median(), best_leave_first_5s_median=v[v.t_after_handover_s <= 5].best_leave_raw.median(),
                  best_leave_after_5s_median=v[v.t_after_handover_s > 5].best_leave_raw.median(), raw_field=(x.raw_field.dropna().iloc[0] if "raw_field" in x and x.raw_field.notna().any() else "S_plain")))
s = pd.DataFrame(S); s.to_csv(HERE / "l3_exam_inspection_summary.csv", index=False); pd.set_option("display.width", 300, "display.max_rows", 300, "display.max_colwidth", 50)
print("EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION | 0.1.4b replan files found:", have); print(s.T.to_string())
print(d[["scene", "model", "t_after_handover_s", "leave_plans", "best_stay_raw", "best_leave_raw", "gap", "leave_plans_below_0p2", "leave_plans_ge_0p6", "best_far_leave_raw", "best_leave_end_x_m", "best_leave_end_y_m"]].round(3).to_string(index=False))
