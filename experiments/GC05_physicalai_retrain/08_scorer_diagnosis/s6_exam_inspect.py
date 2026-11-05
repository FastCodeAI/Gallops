#!/usr/bin/env python
"""S6 flag table: EXAM CLIPS, INSPECTION ONLY, NOT AN EVALUATION. Read-only on the S5 trial's stored per-replan menus of the two pull-out
scenes d2715023 and b44a4a44 (adapter 0.1.6, 07_closed_loop_test/menus/d016_<scene>/replan_*.npz). No model is run; nothing here feeds a, b or c,
a threshold, or any choice. Per replan after handover: the 32 scorer scores, the pick, and the rank and score of the best pull-out plan.
Pull-out plan = the trial's "departing plan": ends >= 2 m from the car (07_closed_loop_test/s5_reads.py:36). "Best" = highest scorer score among
them. Scores are the scorer's own (pdm_score); the pick is the deployed one (after the momentum-aware step), stored as 'chosen'."""
import glob, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; T = HERE.parent / "07_closed_loop_test/menus"; rows = []; allS = []
for sc in ("d2715023", "b44a4a44"):
    for f in sorted(glob.glob(str(T / f"d016_{sc}" / "replan_*.npz"))):
        z = np.load(f, allow_pickle=True); t = float(z["t_after_ho"])
        if t < -0.01: continue
        P = z["proposals"]; S = z["pdm_score"].astype(float); ch = int(z["chosen"]); disp = np.linalg.norm(P[:, -1, :2], axis=1); dep = np.where(disp >= 2.0)[0]; order = np.argsort(-S); rank = np.empty(32, int); rank[order] = np.arange(1, 33)
        b = int(dep[S[dep].argmax()]) if len(dep) else -1; s2 = np.sort(S)
        rows.append(dict(scene=sc, t_after_handover_s=round(t, 2), car_speed_mps=round(float(z["ego_v"]), 2), pick_slot=ch, pick_score=round(S[ch], 4), pick_rank=int(rank[ch]), pick_end_distance_m=round(float(disp[ch]), 2), pick_changed_by_momentum=bool(z["changed"]),
                         pullout_plans_of_32=int(len(dep)), best_pullout_slot=b, best_pullout_score=(round(S[b], 4) if b >= 0 else np.nan), best_pullout_rank=(int(rank[b]) if b >= 0 else np.nan), best_pullout_end_distance_m=(round(float(disp[b]), 2) if b >= 0 else np.nan),
                         best_pullout_end_lateral_m=(round(float(P[b, -1, 1]), 2) if b >= 0 else np.nan), score_gap_pick_minus_best_pullout=(round(S[ch] - S[b], 4) if b >= 0 else np.nan), top2_score_gap=round(s2[-1] - s2[-2], 4), plans_scored_above_best_pullout=(int(rank[b]) - 1 if b >= 0 else np.nan)))
        allS.append(dict(scene=sc, t_after_handover_s=round(t, 2), **{f"score_slot_{j:02d}": round(S[j], 4) for j in range(32)}, **{f"end_distance_m_slot_{j:02d}": round(float(disp[j]), 2) for j in range(32)}))
d = pd.DataFrame(rows); d.to_csv(HERE / "exam_inspection_pullout_replans.csv", index=False); pd.DataFrame(allS).to_csv(HERE / "exam_inspection_pullout_scores32.csv", index=False)
pd.set_option("display.width", 260); print(d.drop(columns=["best_pullout_end_lateral_m", "plans_scored_above_best_pullout", "pick_changed_by_momentum"]).to_string(index=False))
print(d.groupby("scene").agg(replans=("pick_slot", "size"), pullout_picked=("pick_end_distance_m", lambda x: int((x >= 2).sum())), rank_median=("best_pullout_rank", "median"), rank_best=("best_pullout_rank", "min"), rank_worst=("best_pullout_rank", "max"), score_pick_median=("pick_score", "median"),
      score_best_pullout_median=("best_pullout_score", "median"), gap_median=("score_gap_pick_minus_best_pullout", "median"), gap_min=("score_gap_pick_minus_best_pullout", "min"), gap_max=("score_gap_pick_minus_best_pullout", "max"), top2_gap_median=("top2_score_gap", "median"), pullout_plans_median=("pullout_plans_of_32", "median"), momentum_changed=("pick_changed_by_momentum", "sum")).T.to_string())
