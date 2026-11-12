#!/usr/bin/env python
"""T04 Part A (timeline, what is in front, fed inputs) and Part B (menus at every replan). CPU only.
  CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python t04_menu.py
"""
import json, math, multiprocessing as mp, pickle
import numpy as np, pandas as pd
from shapely.geometry import LineString, Point, Polygon
import t04_common as T
C = T.C


def poly(c, yaw, dims):
    l, w = dims[0] / 2, dims[1] / 2; R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return Polygon((np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) @ R.T) + c)


def one(job):
    ver, s, keep_props = job; o = T.load(ver, s); M = T.menus(ver, s); ho = o["ho"]; hs = T.human_start(o); props, reps, val = [], [], []
    g = o["gt"]; gm = g[g[:, 0] >= ho - 1]; hline = LineString(gm[:, 1:3])       # the human's path from handover on (see README: human-like)
    for tn, z in M.items():
        rel = (tn - ho) / 1e6
        if rel < -1e-6 or tn not in o["egot"]: continue
        e = o["egot"][tn]; ego = e[1:4]; v_ego = float(np.hypot(e[4], e[5])); P = z["proposals"].astype(float); ch = int(z["chosen"]); Sf = z["pdm_score"].astype(float); Sp = z["S_plain"].astype(float)
        rf, rp = T.ranks(Sf), T.ranks(Sp); hv = float(np.interp(rel, o["gt_t"], o["gt_v"])); rows = []
        # the human's own future path from this tick through the same check
        tq = tn + T.TS * 1e6; okt = tq <= g[-1, 0]; hw = T.gt_at(o, tq[okt])
        for mg in (0.0, 0.3):
            fr, ft, fa, hz = T.box_check(o, tn, hw, T.TS[okt], mg); val.append(dict(scene=s, version=ver, t_after_handover_s=rel, margin_m=mg, human_path_box_free=fr, first_overlap_s=ft, actor=fa, checked_horizon_s=hz, human_speed_mps=hv))
        for i in range(32):
            p9 = np.r_[np.zeros((1, 3)), P[i]]; seg = np.hypot(*np.diff(p9[:, :2], axis=0).T); arc = np.r_[0, np.cumsum(seg)]; d = T.dense(p9); w = C.to_world(d, ego)
            leave = bool(np.hypot(*P[i, -1, :2]) >= 2.0); prompt = bool(arc[2] >= 0.5 and arc[4] >= 1.5)
            f0, t0, a0, hz = T.box_check(o, tn, w, T.TS, 0.0); f3, t3, a3, _ = T.box_check(o, tn, w, T.TS, 0.3)
            w2, w4 = w[20, :2], w[40, :2]; n2, n4 = hline.distance(Point(w2)), hline.distance(Point(w4))
            h2, h4 = T.gt_at(o, [tn + 2e6])[0, :2], T.gt_at(o, [tn + 4e6])[0, :2]
            s2 = float(np.hypot(*(w2 - h2))) if tn + 2e6 <= g[-1, 0] else np.nan; s4 = float(np.hypot(*(w4 - h4))) if tn + 4e6 <= g[-1, 0] else np.nan
            rows.append(dict(scene=s, version=ver, replan_idx=int(z["idx"]), t_after_handover_s=rel, proposal=i, leave=leave, prompt=prompt, late_go=leave and not prompt, box_free=f0, box_free_margin0p3=f3,
                             first_overlap_s=t0, first_overlap_actor=a0, first_overlap_margin0p3_s=t3, first_overlap_margin0p3_actor=a3, box_checked_horizon_s=hz,
                             human_nearest_2s_m=n2, human_nearest_4s_m=n4, human_like=bool(n2 <= 1.0 and n4 <= 1.0), human_same_time_2s_m=s2, human_same_time_4s_m=s4,
                             arc_1s_m=arc[2], arc_2s_m=arc[4], arc_4s_m=arc[8], end_x_fwd_m=P[i, -1, 0], end_y_left_m=P[i, -1, 1], score_plain=Sp[i], rank_plain=int(rp[i]), score_final=Sf[i], rank_final=int(rf[i]), chosen=bool(i == ch),
                             usable=bool(leave and prompt and f0), usable_margin0p3=bool(leave and prompt and f3)))
        D = pd.DataFrame(rows); st = ~D.leave; R = dict(scene=s, version=ver, replan_idx=int(z["idx"]), t_after_handover_s=rel, ego_speed_mps=v_ego, model_fed_speed_mps=float(z["ego_v"]), human_speed_mps=hv, human_moving=bool(hv > 0.5),
                                                    chosen=ch, chosen_is_argmax_final=bool(ch == int(np.argmax(Sf))), plain_pick=int(z["shipped"]), chosen_class="leave" if D.leave[ch] else "stay", chosen_prompt=bool(D.prompt[ch]), chosen_arc_2s_m=D.arc_2s_m[ch], chosen_arc_4s_m=D.arc_4s_m[ch],
                                                    chosen_box_free=bool(D.box_free[ch]), n_leave=int(D.leave.sum()), n_stay=int(st.sum()), n_leave_prompt=int((D.leave & D.prompt).sum()), n_late_go=int(D.late_go.sum()), n_leave_box_free=int((D.leave & D.box_free).sum()),
                                                    n_late_go_box_free=int((D.late_go & D.box_free).sum()), n_usable=int(D.usable.sum()), n_usable_margin0p3=int(D.usable_margin0p3.sum()), n_usable_human_like=int((D.usable & D.human_like).sum()), n_stay_box_free=int((st & D.box_free).sum()),
                                                    usable_exists=bool(D.usable.any()), usable_human_like_exists=bool((D.usable & D.human_like).any()), leave_box_free_exists=bool((D.leave & D.box_free).any()),
                                                    chosen_score_final=Sf[ch], chosen_score_plain=Sp[ch], best_stay_score_final=float(D.score_final[st].max()) if st.any() else np.nan, best_stay_score_plain=float(D.score_plain[st].max()) if st.any() else np.nan)
        for nm, m in (("usable", D.usable), ("usable_margin0p3", D.usable_margin0p3), ("leave", D.leave), ("leave_box_free", D.leave & D.box_free), ("late_go", D.late_go)):
            for sc, rk in (("final", "rank_final"), ("plain", "rank_plain")):
                if m.any():
                    br = int(D[rk][m].min()); bs = float(D["score_" + sc][m].max()); R[f"best_{nm}_rank_{sc}"] = br; R[f"best_{nm}_score_{sc}"] = bs; R[f"best_{nm}_proposal_{sc}"] = int(D.proposal[m][D[rk][m] == br].iloc[0])
                    R[f"gap_chosen_minus_best_{nm}_{sc}"] = R[f"chosen_score_{sc}"] - bs; R[f"gap_best_stay_minus_best_{nm}_{sc}"] = R[f"best_stay_score_{sc}"] - bs; R[f"n_stay_above_best_{nm}_{sc}"] = int((st & (D[rk] < br)).sum())
                else:
                    for c in ("rank", "score", "proposal"): R[f"best_{nm}_{c}_{sc}"] = np.nan
                    R[f"gap_chosen_minus_best_{nm}_{sc}"] = np.nan; R[f"gap_best_stay_minus_best_{nm}_{sc}"] = np.nan; R[f"n_stay_above_best_{nm}_{sc}"] = np.nan
        reps.append(R)
        if keep_props: props += rows
    return props, reps, val, (ver, s, hs)


def summarise(R, sub, label):
    out = []
    for (s, v), g in sub.groupby(["scene", "version"], sort=False):
        u = g[g.usable_exists]; lb = g[g.leave_box_free_exists]; n = len(g)
        d = dict(scene=s, version=v, replans_subset=label, n_replans=n, n_human_moving=int(g.human_moving.sum()), n_chosen_stay=int((g.chosen_class == "stay").sum()), share_usable_escape=len(u) / n if n else np.nan, n_usable_escape=len(u),
                 share_usable_margin0p3=float((g.n_usable_margin0p3 > 0).mean()) if n else np.nan, share_usable_and_human_like=float(g.usable_human_like_exists.mean()) if n else np.nan,
                 share_leave_box_free=len(lb) / n if n else np.nan, share_of_leave_box_free_with_prompt_one=float(lb.usable_exists.mean()) if len(lb) else np.nan,
                 median_n_leave=float(g.n_leave.median()) if n else np.nan, median_n_leave_prompt=float(g.n_leave_prompt.median()) if n else np.nan, median_n_late_go=float(g.n_late_go.median()) if n else np.nan, median_n_usable=float(g.n_usable.median()) if n else np.nan,
                 median_chosen_score_final=float(g.chosen_score_final.median()) if n else np.nan, median_best_leave_rank_final=float(g.best_leave_rank_final.median()) if n else np.nan, median_best_leave_score_final=float(g.best_leave_score_final.median()) if n else np.nan)
        for sc in ("final", "plain"):
            r = u[f"best_usable_rank_{sc}"]
            d.update({f"median_best_usable_rank_{sc}": float(r.median()) if len(u) else np.nan, f"best_usable_rank_{sc}": float(r.min()) if len(u) else np.nan, f"share_usable_replans_rank_worse_than_5_{sc}": float((r > 5).mean()) if len(u) else np.nan,
                      f"share_usable_replans_below_every_stay_{sc}": float((u[f"n_stay_above_best_usable_{sc}"] == u.n_stay).mean()) if len(u) else np.nan, f"median_n_stay_above_best_usable_{sc}": float(u[f"n_stay_above_best_usable_{sc}"].median()) if len(u) else np.nan,
                      f"median_gap_chosen_minus_best_usable_{sc}": float(u[f"gap_chosen_minus_best_usable_{sc}"].median()) if len(u) else np.nan, f"median_gap_best_stay_minus_best_usable_{sc}": float(u[f"gap_best_stay_minus_best_usable_{sc}"].median()) if len(u) else np.nan,
                      f"median_best_usable_score_{sc}": float(u[f"best_usable_score_{sc}"].median()) if len(u) else np.nan})
        out.append(d)
    return out


def main():
    S = T.scenes(); jobs = [("0.1.6", s, s in T.FOUR) for s in S] + [("0.1.4b", s, True) for s in T.FOUR]
    with mp.Pool(16) as pool: res = pool.map(one, jobs, chunksize=1)
    P = pd.DataFrame([r for x in res for r in x[0]]); R = pd.DataFrame([r for x in res for r in x[1]]); V = pd.DataFrame([r for x in res for r in x[2]]); hs = {(v, s): h for x in res for v, s, h in [x[3]]}
    P.to_csv(T.HERE / "proposals.csv", index=False, float_format="%.5g"); R4 = R[R.scene.isin(T.FOUR)]; R4.to_csv(T.HERE / "replans.csv", index=False, float_format="%.5g"); R.to_pickle(T.HERE / "cache" / "replans_all.pkl")
    V.to_csv(T.HERE / "boxcheck_validation.csv", index=False, float_format="%.5g")
    sm = summarise(R, R4, "all (handover to end)") + summarise(R, R4[R4.ego_speed_mps < 0.5], "ego < 0.5 m/s") + summarise(R, R4[R4.human_moving], "human moving (> 0.5 m/s)") + summarise(R, R4[(R4.ego_speed_mps < 0.5) & R4.human_moving], "ego < 0.5 m/s and human moving") + summarise(R, R4[R4.t_after_handover_s <= 12.5], "full 4 s box horizon (t <= 12.5 s)")
    pd.DataFrame(sm).to_csv(T.HERE / "scene_summary.csv", index=False, float_format="%.5g")
    # all-30 scan (0.1.6)
    A = R[(R.version == "0.1.6") & (R.ego_speed_mps < 0.5) & (R.human_speed_mps > 1.0) & (R.chosen_class == "stay")]; rows = []
    for s in S:
        g = A[A.scene == s]; u = g[g.usable_exists]; n_all = int(((R.version == "0.1.6") & (R.scene == s)).sum())
        rows.append(dict(scene=s, n_replans=n_all, n_stationary_stay_human_going=len(g), of_which_usable_escape=len(u), of_which_usable_margin0p3=int((g.n_usable_margin0p3 > 0).sum()), of_which_usable_human_like=int(g.usable_human_like_exists.sum()),
                         first_t_s=float(g.t_after_handover_s.min()) if len(g) else np.nan, last_t_s=float(g.t_after_handover_s.max()) if len(g) else np.nan,
                         median_best_usable_rank_final=float(u.best_usable_rank_final.median()) if len(u) else np.nan, best_usable_rank_final=float(u.best_usable_rank_final.min()) if len(u) else np.nan,
                         median_best_usable_rank_plain=float(u.best_usable_rank_plain.median()) if len(u) else np.nan, median_gap_chosen_minus_best_usable_final=float(u.gap_chosen_minus_best_usable_final.median()) if len(u) else np.nan,
                         median_n_stay_above_best_usable_final=float(u.n_stay_above_best_usable_final.median()) if len(u) else np.nan, median_n_stay=float(u.n_stay.median()) if len(u) else np.nan))
    pd.DataFrame(rows).to_csv(T.HERE / "all30_stationary_stays.csv", index=False, float_format="%.5g")
    json.dump({f"{v}|{s}": h for (v, s), h in hs.items()}, open(T.HERE / "cache" / "human_start.json", "w"))
    print(len(P), len(R), len(V)); print("chosen==argmax(final):", R.chosen_is_argmax_final.mean(), "validation pass:", V.groupby("margin_m").human_path_box_free.mean().to_dict())


if __name__ == "__main__":
    main()
