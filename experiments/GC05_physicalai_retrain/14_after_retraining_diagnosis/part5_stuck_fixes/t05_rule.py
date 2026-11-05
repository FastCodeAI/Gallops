#!/usr/bin/env python
"""T05 Part 1: the anti-stall rule R(T, d, guard) replayed open-loop on the recorded 0.1.6 menus of all 30 scenes; controller check of
every first firing; stationary episodes. -> antistall_grid.csv, antistall_first_firings.csv, stationary_episodes.csv, cache/rule.pkl
  CUDA_VISIBLE_DEVICES="" nurec-venv/bin/python t05_rule.py"""
import itertools, math, multiprocessing as mp, pickle, sys
import numpy as np, pandas as pd
import t05_common as F
T = F.T; C = F.C
TS_ = (1.0, 2.0, 3.0, 5.0); DS = (0.1, 0.2, 0.3, 0.5, 1.0); GUARDS = ("none", "G_box0", "G_box03"); CLOCKS = ("incl_pre_handover", "from_handover")
GCOL = {"none": None, "G_box0": "box_free", "G_box03": "box_free_margin0p3"}; SCOL = {"incl_pre_handover": "standing_s", "from_handover": "standing_s_from_handover"}


def firings(D):
    """Every (clock, T, d, guard, scene, replan) at which the rule fires on the recorded menus."""
    out = []
    for (s, tn), g in D.groupby(["scene", "tn"], sort=True):
        ch = g[g.chosen].iloc[0]
        if ch.leave: continue
        for gd in GUARDS:
            c = g[g.leave & g.prompt & (g[GCOL[gd]] if GCOL[gd] else True)]
            if not len(c): continue
            b = c.loc[c.score_final.idxmax()]
            for ck in CLOCKS:
                for Tm in TS_:
                    if ch[SCOL[ck]] < Tm - 1e-9: continue
                    for d in DS:
                        if b.score_final >= ch.score_final - d: out.append(dict(clock=ck, T_s=Tm, d=d, guard=gd, scene=s, tn=tn, t_after_handover_s=ch.t_after_handover_s, proposal=int(b.proposal), chosen_stay=int(ch.proposal)))
    return pd.DataFrame(out)


def check_scene(job):
    """Standalone controller, warm (teacher-forced from the session start), 2.0 s hold of each switched-to plan. One scene, ticks in order."""
    s, items = job; sys.path.insert(0, str(F.HERE.parent / "part3_swerve_fixes")); import t03_common as T3
    o = T.load("0.1.6", s); d = T3.load(s); M = T.menus("0.1.6", s); sim = C.Sim(d, d["ticks"][0]); res = {}
    for ts in sorted({t for t, _ in items}):
        sim.feed_recorded(ts); ego = o["egot"][ts][1:4]
        for pi in sorted({p for t, p in items if t == ts}):
            p9 = F.plan9(M[ts]["proposals"][pi])
            def fn(sm, p9=p9, ts=ts):
                sm.hold(C.plan_pb(ts, p9), dur_us=2_000_000); return sm.rows()
            rows = T3.branch(sim, fn); m = rows["timestamp_us"] > ts; t = (rows["timestamp_us"][m] - ts) / 1e6; x, y = rows["x"][m], rows["y"][m]; w = np.c_[x, y, 2 * np.arctan2(rows["qz"][m], rows["qw"][m])]
            disp = np.hypot(x - ego[0], y - ego[1]); f0 = T.box_check(o, ts, w, t, 0.0); f3 = T.box_check(o, ts, w, t, 0.3); cl = F.clearance(o, ts, w, t)
            res[(s, ts, pi)] = dict(moved_1s_m=float(disp[int(np.argmin(np.abs(t - 1.0)))]), moved_2s_m=float(disp[-1]), speed_at_2s_mps=float(np.hypot(rows["vx"][m], rows["vy"][m])[-1]), peak_abs_steer_rad=float(np.abs(rows["front_steering_angle"][m]).max()),
                                    lateral_at_2s_m=float(C.to_rig(w, ego)[-1, 1]), driven_overlap_2s=not f0[0], driven_first_overlap_s=f0[1], driven_overlap_actor=f0[2], driven_overlap_label=F.label(o, f0[2]),
                                    driven_overlap_margin0p3_2s=not f3[0], driven_first_overlap_margin0p3_s=f3[1], driven_min_clearance_2s_m=cl[0], driven_min_clearance_label=F.label(o, cl[2]), driven_t=t, driven_w=w)
    return res


def human(o, rel):
    hv = float(np.interp(rel, o["gt_t"], o["gt_v"])); m = (o["gt_t"] >= rel - 1e-9) & (o["gt_v"] > 0.5)
    return hv, (float(max(o["gt_t"][np.argmax(m)] - rel, 0.0)) if m.any() else np.nan)


def ahead(o, t_us, ego):
    """Nearest actor box ahead inside the ego-width corridor (gap from the front bumper) and nearest box anywhere, at one time."""
    i = int(np.argmin(np.abs(o["ap_t"] - t_us))); a = o["A"][i]; ok = np.flatnonzero(np.isfinite(a[:, 0])); r = dict(ahead_gap_m=np.nan, ahead_label="", ahead_actor="", nearest_m=np.nan, nearest_label="")
    if not len(ok): return r
    k = F.corners(a[ok, :2].astype(float), a[ok, 2].astype(float), o["dims"][ok] / 2); hw = o["ego_dims"][1] / 2; front = o["rig_to_aabb_x"] + o["ego_dims"][0] / 2; best = np.inf
    for j, kk in zip(ok, k):
        cr = C.to_rig(np.c_[kk, np.zeros(4)], ego)
        if cr[:, 1].min() < hw and cr[:, 1].max() > -hw and cr[:, 0].max() > front:
            gap = max(cr[:, 0].min() - front, 0.0)
            if gap < best: best = gap; r.update(ahead_gap_m=float(gap), ahead_label=o["labels"][j], ahead_actor=o["ids"][j])
    c = np.array([ego[0] + o["rig_to_aabb_x"] * math.cos(ego[2]), ego[1] + o["rig_to_aabb_x"] * math.sin(ego[2])]); dd = F.rect_dist(c, float(ego[2]), o["ego_dims"] / 2, a[ok, :2].astype(float), a[ok, 2].astype(float), o["dims"][ok] / 2)
    j = int(np.argmin(dd)); r.update(nearest_m=float(dd[j]), nearest_label=o["labels"][ok[j]]); return r


def episodes(D):
    out = []
    for s in T.scenes():
        o = T.load("0.1.6", s); ho = o["ho"]; te, ve, st = F.standing_since(o); below = ve < F.V_STAND; i = 0; g = D[D.scene == s].drop_duplicates("tn"); chs = D[(D.scene == s) & D.chosen]
        while i < len(te):
            if not below[i]: i += 1; continue
            j = i
            while j + 1 < len(te) and below[j + 1]: j += 1
            a, b = (te[i] - ho) / 1e6, (te[j] - ho) / 1e6
            if b - a >= 1.0 and b > 0:
                a0 = max(a, 0.0); hs = T.gt_at(o, ho + np.arange(a0, b + 1e-6, 0.05) * 1e6); hd = float(np.hypot(*np.diff(hs[:, :2], axis=0).T).sum()); hv_a, wait = human(o, a0); tt = np.arange(a0, b + 1e-6, 0.05); hvv = np.interp(tt, o["gt_t"], o["gt_v"])
                em = o["ego"][i:j + 1]; em = em[em[:, 0] >= ho + a0 * 1e6 - 1]; cd = float(np.hypot(*np.diff(em[:, 1:3], axis=0).T).sum()) if len(em) > 1 else 0.0; mid = em[len(em) // 2]; ah = ahead(o, mid[0], mid[1:4]); ah0 = ahead(o, em[0, 0], em[0, 1:4])
                rp = chs[(chs.t_after_handover_s >= a0 - 1e-6) & (chs.t_after_handover_s <= b + 1e-6)]
                out.append(dict(scene=s, start_s=a, end_s=b, duration_s=b - a, post_handover_duration_s=b - a0, runs_to_end_of_clip=bool(j == len(te) - 1), car_path_m=cd, human_path_m=hd, human_speed_at_start_mps=hv_a, human_starts_moving_after_s=wait, human_share_below_0p5=float((hvv < 0.5).mean()), human_max_speed_mps=float(hvv.max()),
                                n_replans=len(rp), n_replans_chosen_stay=int((~rp.leave).sum()), ahead_gap_at_start_m=ah0["ahead_gap_m"], ahead_label_at_start=ah0["ahead_label"], ahead_gap_mid_m=ah["ahead_gap_m"], ahead_label_mid=ah["ahead_label"], nearest_box_mid_m=ah["nearest_m"], nearest_label_mid=ah["nearest_label"],
                                episode_class=("human also standing (human path < 2 m)" if hd < 2.0 else "human drives off (human path >= car path + 2 m)" if hd >= cd + 2.0 else "other")))
            i = j + 1
    return pd.DataFrame(out)


def main():
    D = pd.read_pickle(F.HERE / "cache" / "props_all.pkl"); D = D[D.version == "0.1.6"].reset_index(drop=True); Fr = firings(D); print("firing rows", len(Fr))
    first = Fr.sort_values("tn").groupby(["clock", "T_s", "d", "guard", "scene"], as_index=False).first(); cnt = Fr.groupby(["clock", "T_s", "d", "guard", "scene"]).size().rename("n_replans_rule_fires_recorded_menus").reset_index(); first = first.merge(cnt)
    need = {}
    for r in first[["scene", "tn", "proposal"]].drop_duplicates().itertuples(): need.setdefault(r.scene, []).append((int(r.tn), int(r.proposal)))
    print("controller checks:", sum(len(v) for v in need.values()), "in", len(need), "scenes", flush=True)
    with mp.Pool(8) as pool: res = {k: v for x in pool.map(check_scene, list(need.items()), chunksize=1) for k, v in x.items()}
    key = D.set_index(["scene", "tn", "proposal"]); os_ = {s: T.load("0.1.6", s) for s in need}; rows = []
    for r in first.itertuples():
        p = key.loc[(r.scene, r.tn, r.proposal)]; st = key.loc[(r.scene, r.tn, r.chosen_stay)]; o = os_[r.scene]; hv, wait = human(o, r.t_after_handover_s); cc = res[(r.scene, int(r.tn), int(r.proposal))]
        overlap = bool((not p.box_free) or cc["driven_overlap_2s"]); wanted = bool(hv > 0.5 or (np.isfinite(wait) and wait <= 2.0))
        rows.append(dict(clock=r.clock, T_s=r.T_s, d=r.d, guard=r.guard, scene=r.scene, first_firing_t_s=r.t_after_handover_s, n_replans_rule_fires_recorded_menus=r.n_replans_rule_fires_recorded_menus, proposal=r.proposal, proposal_score_final=p.score_final, proposal_rank_final=int(p.rank_final), proposal_score_plain=p.score_plain,
                         chosen_stay=r.chosen_stay, chosen_stay_score_final=st.score_final, score_gap=st.score_final - p.score_final, standing_s=st.standing_s, standing_s_from_handover=st.standing_s_from_handover, ego_speed_mps=st.ego_speed_mps, human_speed_mps=hv, human_state="moving" if hv > 0.5 else "standing", human_starts_moving_in_s=wait,
                         plan_end_x_fwd_m=p.end_x_fwd_m, plan_end_y_left_m=p.end_y_left_m, plan_arc_2s_m=p.arc_2s_m, plan_arc_4s_m=p.arc_4s_m, plan_box_free=bool(p.box_free), plan_first_overlap_s=p.first_overlap_s, plan_first_overlap_actor=p.first_overlap_actor, plan_first_overlap_label=p.first_overlap_label,
                         plan_box_free_margin0p3=bool(p.box_free_margin0p3), plan_first_overlap_margin0p3_s=p.first_overlap_margin0p3_s, plan_first_overlap_margin0p3_label=p.first_overlap_margin0p3_label, plan_min_clearance_m=p.min_clearance_m, plan_min_clearance_label=p.min_clearance_label, plan_box_checked_horizon_s=p.box_checked_horizon_s,
                         **{k: v for k, v in cc.items() if not k.startswith("driven_t") and k != "driven_w"}, label="unsafe go" if overlap else "wanted go" if wanted else "premature go"))
    FF = pd.DataFrame(rows); FF.to_csv(F.HERE / "antistall_first_firings.csv", index=False, float_format="%.5g")
    # grid
    chs = D[D.chosen]; grid = []; elig0 = chs[~chs.leave]
    Fk = Fr.merge(key[["box_free", "box_free_margin0p3"]].reset_index(), on=["scene", "tn", "proposal"]).merge(chs[["scene", "tn", "human_speed_mps"]], on=["scene", "tn"])
    for ck, Tm, d, gd in itertools.product(CLOCKS, TS_, DS, GUARDS):
        f = FF[(FF.clock == ck) & (FF.T_s == Tm) & (FF.d == d) & (FF.guard == gd)]; e = elig0[elig0[SCOL[ck]] >= Tm - 1e-9]; fk = Fk[(Fk.clock == ck) & (Fk.T_s == Tm) & (Fk.d == d) & (Fk.guard == gd)]
        eo = e[~e.scene.isin(F.STUCK) & (e.human_speed_mps < 0.5)]; fo = fk[~fk.scene.isin(F.STUCK) & (fk.human_speed_mps < 0.5)]; g = dict(clock=ck, T_s=Tm, d=d, guard=gd, n_scenes_first_firing=len(f))
        for lb in ("wanted go", "premature go", "unsafe go"): g["n_" + lb.replace(" ", "_")] = int((f.label == lb).sum()); g["scenes_" + lb.replace(" ", "_")] = " ".join(f.scene[f.label == lb])
        for s in F.STUCK: x = f[f.scene == s]; g[f"{s}_first_firing_t_s"] = float(x.first_firing_t_s.iloc[0]) if len(x) else np.nan; g[f"{s}_label"] = x.label.iloc[0] if len(x) else "does not fire"
        g.update(n_eligible_replans_all30=len(e), n_firing_replans_all30=len(fk), other28_human_standing_eligible_replans=len(eo), other28_human_standing_firing_replans=len(fo), other28_human_standing_firing_share=len(fo) / len(eo) if len(eo) else np.nan,
                 other28_human_standing_firings_plan_overlaps_0m=int((~fo.box_free).sum()), other28_human_standing_firings_plan_overlaps_0p3m=int((~fo.box_free_margin0p3).sum()), other28_human_standing_firing_scenes=" ".join(sorted(fo.scene.unique())),
                 other28_human_standing_overlap0_scenes=" ".join(sorted(fo.scene[~fo.box_free].unique())))
        grid.append(g)
    G = pd.DataFrame(grid); G.to_csv(F.HERE / "antistall_grid.csv", index=False, float_format="%.4g")
    E = episodes(D); E.to_csv(F.HERE / "stationary_episodes.csv", index=False, float_format="%.4g"); pickle.dump(dict(firings=Fk, checks=res), open(F.HERE / "cache" / "rule.pkl", "wb"))
    pd.set_option("display.width", 250); m = G[(G.clock == "incl_pre_handover") & (G.T_s == 3) & (G.d == 0.3)]
    print(m.drop(columns=[c for c in m if c.startswith("other28")]).T.to_string()); print(E.round(2).to_string())


if __name__ == "__main__":
    main()
