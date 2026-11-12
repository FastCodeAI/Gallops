#!/usr/bin/env python
"""S7 H2: EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION. Read-only on the S5 trial's stored per-replan menus of the two pull-out scenes
d2715023 and b44a4a44 (adapter 0.1.6, 07_closed_loop_test/menus/d016_<scene>/replan_*.npz). No model is run, nothing is written under 07_closed_loop_test/,
and nothing here feeds H1, H3, a threshold or any choice. Definitions: PREDICTIONS.md, P-H2.
raw = S_plain (scorer output before the momentum step), nudged = pdm_score ((14 raw + 2 comfort) / 16; 07_closed_loop_test/s5_menus.py:49-50).
leave = end position >= 2 m to the side (|y| >= 2) or >= 5 m forward (x >= 5) of the car at that replan; else stay."""
import glob, numpy as np, pandas as pd
from s7_common import HERE, RD
T = RD / "07_closed_loop_test/menus"; R = []; A = []
for sc in ("d2715023", "b44a4a44"):
    for f in sorted(glob.glob(str(T / f"d016_{sc}" / "replan_*.npz"))):
        z = np.load(f, allow_pickle=True); t = float(z["t_after_ho"])
        if t < -0.01: continue
        P = z["proposals"]; raw = z["S_plain"].astype(float); nud = z["pdm_score"].astype(float); Sc = z["S_c"].astype(float); x, y = P[:, -1, 0], P[:, -1, 1]; leave = (np.abs(y) >= 2.0) | (x >= 5.0); stay = ~leave
        r = dict(scene=sc, t_after_handover_s=round(t, 2), car_speed_mps=round(float(z["ego_v"]), 2), momentum_code=str(z["code"]), leave_plans=int(leave.sum()), stay_plans=int(stay.sum()), leave_by_side_only=int(((np.abs(y) >= 2) & (x < 5)).sum()), leave_by_forward=int((x >= 5).sum()),
                 car_pick_slot=int(z["chosen"]), car_pick_is_leave=bool(leave[int(z["chosen"])]), raw_top_slot=int(raw.argmax()), raw_top_is_leave=bool(leave[raw.argmax()]), nudged_top_slot=int(nud.argmax()), nudged_top_is_leave=bool(leave[nud.argmax()]),
                 comfort_term_available=bool(np.isfinite(Sc).all()), raw_equals_nudged=bool(np.allclose(raw, nud, atol=1e-6)))
        for nm, S in (("raw", raw), ("nudged", nud)):
            if leave.any() and stay.any():
                bl = int(np.where(leave)[0][S[leave].argmax()]); bs = int(np.where(stay)[0][S[stay].argmax()]); rank = int((S > S[bl]).sum()) + 1
                r.update({f"{nm}_best_stay": round(S[bs], 4), f"{nm}_best_leave": round(S[bl], 4), f"{nm}_gap_stay_minus_leave": round(S[bs] - S[bl], 4), f"{nm}_rank_best_leave": rank, f"{nm}_best_leave_slot": bl, f"{nm}_best_leave_end_x_m": round(float(x[bl]), 2), f"{nm}_best_leave_end_y_m": round(float(y[bl]), 2)})
        if np.isfinite(Sc).all() and leave.any() and stay.any(): r["comfort_term_mean_leave"] = round(float(Sc[leave].mean()), 3); r["comfort_term_mean_stay"] = round(float(Sc[stay].mean()), 3)
        r["flip_leave_to_stay"] = bool(r["raw_top_is_leave"] and not r["nudged_top_is_leave"]); r["flip_stay_to_leave"] = bool((not r["raw_top_is_leave"]) and r["nudged_top_is_leave"]); R.append(r)
        for j in range(32): A.append(dict(scene=sc, t_after_handover_s=round(t, 2), slot=j, tag="leave" if leave[j] else "stay", end_x_m=round(float(x[j]), 2), end_y_m=round(float(y[j]), 2), raw_score=round(raw[j], 4), nudged_score=round(nud[j], 4), comfort_term=(round(Sc[j], 3) if np.isfinite(Sc[j]) else np.nan), car_pick=bool(j == int(z["chosen"]))))
d = pd.DataFrame(R); d.to_csv(HERE / "h2_exam_inspection_replans.csv", index=False); pd.DataFrame(A).to_csv(HERE / "h2_exam_inspection_scores32.csv", index=False)
S = []
for sc, x in d.groupby("scene", sort=False):
    v = x.dropna(subset=["raw_gap_stay_minus_leave"]); g = v.raw_gap_stay_minus_leave; gn = v.nudged_gap_stay_minus_leave
    S.append(dict(scene=sc, replans=len(x), replans_with_both_kinds=len(v), leave_plans_median=x.leave_plans.median(), leave_plans_min=x.leave_plans.min(), leave_plans_max=x.leave_plans.max(), raw_gap_median=g.median(), raw_gap_min=g.min(), raw_gap_max=g.max(), nudged_gap_median=gn.median(), nudged_gap_min=gn.min(), nudged_gap_max=gn.max(),
                  share_raw_gap_below_0p02=float((g < 0.02).mean()), share_raw_gap_abs_below_0p02=float((g.abs() < 0.02).mean()), share_raw_gap_negative=float((g < 0).mean()), raw_rank_best_leave_median=v.raw_rank_best_leave.median(), raw_rank_best_leave_min=v.raw_rank_best_leave.min(), raw_rank_best_leave_max=v.raw_rank_best_leave.max(),
                  nudged_rank_best_leave_median=v.nudged_rank_best_leave.median(), raw_top_is_leave=int(x.raw_top_is_leave.sum()), nudged_top_is_leave=int(x.nudged_top_is_leave.sum()), car_pick_is_leave=int(x.car_pick_is_leave.sum()), flips_leave_to_stay=int(x.flip_leave_to_stay.sum()), share_flipped=float(x.flip_leave_to_stay.mean()),
                  flips_stay_to_leave=int(x.flip_stay_to_leave.sum()), raw_equals_nudged_replans=int(x.raw_equals_nudged.sum()), comfort_term_available_replans=int(x.comfort_term_available.sum()), raw_best_stay_median=v.raw_best_stay.median(), raw_best_leave_median=v.raw_best_leave.median(),
                  gap_first_5s_median=v[v.t_after_handover_s <= 5].raw_gap_stay_minus_leave.median(), gap_after_5s_median=v[v.t_after_handover_s > 5].raw_gap_stay_minus_leave.median()))
s = pd.DataFrame(S); s.to_csv(HERE / "h2_exam_inspection_summary.csv", index=False); pd.set_option("display.width", 250, "display.max_rows", 200)
print("EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION"); print(s.set_index("scene").T.to_string())
print(d[["scene", "t_after_handover_s", "car_speed_mps", "momentum_code", "leave_plans", "raw_best_stay", "raw_best_leave", "raw_gap_stay_minus_leave", "raw_rank_best_leave", "nudged_gap_stay_minus_leave", "nudged_rank_best_leave", "raw_best_leave_end_x_m", "raw_best_leave_end_y_m", "raw_top_is_leave", "car_pick_is_leave"]].to_string(index=False))
