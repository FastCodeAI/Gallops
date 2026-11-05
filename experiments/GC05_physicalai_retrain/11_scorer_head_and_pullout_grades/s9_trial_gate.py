#!/usr/bin/env python
"""S9 C.4, EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION. On the two stuck trial scenes (d2715023, b44a4a44; adapter 0.1.6): per replan,
the gate P of every plan from part B's gate head and from the C.4 gate head (retrained with the pull-out collision labels), and the pick
of each two-stage head. Reads c4_trial_features.npz (features recomputed read-only from the trial's logged frames; plans checked equal
to the trial's), b_models.pt, c4_models.pt. Leave / stay rule and "best leave plan" (highest raw scorer score among leave plans) as S7 H2.
Nothing here feeds a threshold, a choice or any other table."""
import pathlib, numpy as np, pandas as pd, torch
from s9_lib import *
F = np.load(HERE / "c4_trial_features.npz"); B = torch.load(HERE / "b_models.pt", weights_only=False); C4 = torch.load(HERE / "c4_models.pt", weights_only=False); t = float(B["selection"]["t"]); assert float(C4["selection"]["t"]) == t
Z = ((F["slot_9"].astype(np.float32) - B["mu"]) / B["sd"]).astype(np.float32); H = {}
for nm, M in (("B", B), ("C4", C4)):
    g = gate_net(); g.load_state_dict(M["gate"]); r = rank_net(); r.load_state_dict(M["rankers"][str(t)]); pg, P = gate_probs(g, Z); o = fwd(r, Z).squeeze(-1); pk, v = select(o, P, t); H[nm] = dict(pg=pg, P=P, out=o, pick=pk, veto=v)
P_ = F["menu"]; x, y = P_[:, :, -1, 0], P_[:, :, -1, 1]; leave = (np.abs(y) >= 2.0) | (x >= 5.0); raw = F["S_plain"].astype(float); R = []; A = []
for i in range(len(raw)):
    if F["t_after_ho"][i] < -0.01: continue
    l = leave[i]; bl = int(np.where(l)[0][raw[i][l].argmax()]); bs = int(np.where(~l)[0][raw[i][~l].argmax()]); ch = int(F["chosen"][i]); r = dict(scene=str(F["scene"][i]), t_after_handover_s=round(float(F["t_after_ho"][i]), 2), car_speed_mps=round(float(F["ego_v"][i]), 2), leave_plans=int(l.sum()),
        best_leave_slot=bl, best_leave_raw_score=round(raw[i, bl], 4), best_leave_end_x_m=round(float(x[i, bl]), 2), best_leave_end_y_m=round(float(y[i, bl]), 2), best_stay_slot=bs, best_stay_raw_score=round(raw[i, bs], 4), car_pick_slot=ch, car_pick_is_leave=bool(l[ch]))
    for nm, h in H.items():
        r.update({f"{nm}_gateP_best_leave": round(h["P"][i, bl], 4), f"{nm}_p_kerb_best_leave": round(h["pg"][i, bl, 0], 4), f"{nm}_p_oncoming_best_leave": round(h["pg"][i, bl, 1], 4), f"{nm}_p_collision_best_leave": round(h["pg"][i, bl, 2], 4), f"{nm}_gateP_best_stay": round(h["P"][i, bs], 4),
                  f"{nm}_gateP_car_pick": round(h["P"][i, ch], 4), f"{nm}_gateP_leave_plans_median": round(float(np.median(h["P"][i][l])), 4), f"{nm}_gateP_leave_plans_min": round(float(h["P"][i][l].min()), 4), f"{nm}_gateP_stay_plans_median": round(float(np.median(h["P"][i][~l])), 4),
                  f"{nm}_leave_plans_vetoed": int(h["veto"][i][l].sum()), f"{nm}_stay_plans_vetoed": int(h["veto"][i][~l].sum()), f"{nm}_pick_slot": int(h["pick"][i]), f"{nm}_pick_is_leave": bool(l[h["pick"][i]]), f"{nm}_pick_end_x_m": round(float(x[i, h["pick"][i]]), 2),
                  f"{nm}_pick_end_y_m": round(float(y[i, h["pick"][i]]), 2), f"{nm}_pick_gateP": round(h["P"][i, h["pick"][i]], 4), f"{nm}_pick_same_as_car": bool(h["pick"][i] == ch), f"{nm}_ranker_top_unvetoed_leave_rank": int((np.where(h["veto"][i], -np.inf, h["out"][i]) > np.where(h["veto"][i] | ~l, -np.inf, h["out"][i]).max()).sum() + 1)})
    R.append(r)
    for j in range(32): A.append(dict(scene=r["scene"], t_after_handover_s=r["t_after_handover_s"], slot=j, tag="leave" if l[j] else "stay", end_x_m=round(float(x[i, j]), 2), end_y_m=round(float(y[i, j]), 2), raw_score=round(raw[i, j], 4), B_gateP=round(H["B"]["P"][i, j], 4), C4_gateP=round(H["C4"]["P"][i, j], 4),
                                      C4_p_kerb=round(H["C4"]["pg"][i, j, 0], 4), C4_p_oncoming=round(H["C4"]["pg"][i, j, 1], 4), C4_p_collision=round(H["C4"]["pg"][i, j, 2], 4), B_p_collision=round(H["B"]["pg"][i, j, 2], 4), B_ranker=round(H["B"]["out"][i, j], 4), C4_ranker=round(H["C4"]["out"][i, j], 4)))
d = pd.DataFrame(R); d.to_csv(HERE / "c4_exam_inspection_replans.csv", index=False); pd.DataFrame(A).to_csv(HERE / "c4_exam_inspection_plans32.csv", index=False); S = []
for sc, q in d.groupby("scene", sort=False):
    r = dict(scene=sc, replans=len(q), threshold_t=t)
    for nm in ("B", "C4"):
        p = q[f"{nm}_gateP_best_leave"]; r.update({f"{nm}_gateP_best_leave_median": p.median(), f"{nm}_gateP_best_leave_min": p.min(), f"{nm}_gateP_best_leave_max": p.max(), f"{nm}_replans_gateP_best_leave_below_0p5": int((p < 0.5).sum()), f"{nm}_replans_gateP_best_leave_at_or_below_t": int((p <= t).sum()),
                   f"{nm}_p_kerb_best_leave_median": q[f"{nm}_p_kerb_best_leave"].median(), f"{nm}_p_oncoming_best_leave_median": q[f"{nm}_p_oncoming_best_leave"].median(), f"{nm}_p_collision_best_leave_median": q[f"{nm}_p_collision_best_leave"].median(), f"{nm}_gateP_best_stay_median": q[f"{nm}_gateP_best_stay"].median(),
                   f"{nm}_gateP_leave_plans_median": q[f"{nm}_gateP_leave_plans_median"].median(), f"{nm}_gateP_lowest_leave_plan_median": q[f"{nm}_gateP_leave_plans_min"].median(), f"{nm}_replans_some_leave_plan_below_0p5": int((q[f"{nm}_gateP_leave_plans_min"] < 0.5).sum()), f"{nm}_leave_plans_vetoed_median": q[f"{nm}_leave_plans_vetoed"].median(),
                   f"{nm}_stay_plans_vetoed_median": q[f"{nm}_stay_plans_vetoed"].median(), f"{nm}_pick_is_leave": int(q[f"{nm}_pick_is_leave"].sum()), f"{nm}_pick_same_as_car": int(q[f"{nm}_pick_same_as_car"].sum()), f"{nm}_pick_end_x_median": q[f"{nm}_pick_end_x_m"].median(), f"{nm}_pick_gateP_median": q[f"{nm}_pick_gateP"].median()})
    r["car_pick_is_leave"] = int(q.car_pick_is_leave.sum()); r["best_leave_raw_score_median"] = q.best_leave_raw_score.median(); r["leave_plans_median"] = q.leave_plans.median(); S.append(r)
s = pd.DataFrame(S); s.to_csv(HERE / "c4_exam_inspection_summary.csv", index=False); pd.set_option("display.width", 250, "display.max_rows", 300); print("EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION"); print(s.set_index("scene").T.round(4).to_string())
print(d[["scene", "t_after_handover_s", "leave_plans", "best_leave_raw_score", "B_gateP_best_leave", "C4_gateP_best_leave", "C4_p_kerb_best_leave", "C4_p_oncoming_best_leave", "C4_p_collision_best_leave", "B_p_collision_best_leave", "C4_gateP_best_stay", "C4_leave_plans_vetoed", "B_pick_is_leave", "C4_pick_is_leave", "C4_pick_end_x_m"]].to_string(index=False))
q = s[s.scene == "d2715023"].iloc[0]; print("P-C second half: d2715023, C.4 gate P of the best leave plan < 0.5 at %d of %d replans (B head: %d)" % (q.C4_replans_gateP_best_leave_below_0p5, q.replans, q.B_replans_gateP_best_leave_below_0p5))
