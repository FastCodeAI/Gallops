#!/usr/bin/env python
"""T03: tables, plots, README.md and metrics.json from the CSVs written by t03_testbed.py and t03_fixc.py. No controller runs here."""
import json, pickle, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = pathlib.Path(__file__).resolve().parent
VARS = ["baseline", "A1", "A2", "A3", "B1", "B2"]; FIX = VARS[1:]
COL = dict(zip(VARS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))      # dataviz reference categorical palette, fixed order
GROUPS = ["highway", "bd07fb91", "other>=10", "<10"]; IX = ["scene", "tick_after_handover_s"]
INK = "#3d3d3a"; plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b0aea5", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#e8e6dc", "grid.linewidth": 0.6,
                                      "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight", "lines.linewidth": 2})
C5 = "T2_long0p5_steer50"


def md(df, nd=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{nd}f}".rstrip("0").rstrip(".") if abs(x) < 1e6 else str(x))
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in d.values]) + "\n"


def main():
    tb = pd.read_csv(HERE / "testbed.csv"); alt = pd.read_csv(HERE / "fixA_alteration.csv"); pk = pd.read_csv(HERE / "fixB_picks.csv"); lt = pd.read_csv(HERE / "later_braking_cases.csv")
    sw = pd.read_csv(HERE / "fixC_sweep.csv"); tr = pd.read_csv(HERE / "fixC_tracking.csv"); longs = pickle.load(open(HERE / "long_hold.pkl", "rb"))
    tb["state"] = np.where(tb.pre_swerve, "pre-swerve", "disturbed"); post = tb[tb.phase == "post_handover"]; pre = tb[tb.phase != "post_handover"]; M = {}; out = []
    W = post.pivot_table(index=IX, columns="variant", values="peak_abs_steer_rad")[VARS]; B = post[post.variant == "baseline"].set_index(IX)
    n_scene = tb.scene.nunique(); n_all = len(tb[tb.variant == "baseline"]); n_post = len(B)

    # ---- 1 baseline reproduction
    key = [("04749bb9", 0.0), ("acd22430", 0.0), ("04749bb9", 0.5), ("acd22430", 0.5), ("04749bb9", 3.0), ("acd22430", 3.0), ("bd07fb91", 6.0)]
    K = W.loc[key].reset_index(); K.insert(2, "car_speed_mps", [B.loc[k, "car_speed_mps"] for k in key]); K.insert(3, "pre_swerve", [bool(B.loc[k, "pre_swerve"]) for k in key]); K.insert(4, "plan_a1_mps2", [B.loc[k, "plan_a1_mps2"] for k in key])
    M["baseline_reproduction"] = dict(peak_steer_04749bb9_handover_rad=round(float(W.loc[("04749bb9", 0.0), "baseline"]), 4), peak_steer_acd22430_handover_rad=round(float(W.loc[("acd22430", 0.0), "baseline"]), 4), t02_values_rad=[0.525, 0.401],
                                      solves_not_solved_in_all_holds=int(tb.n_solves_not_solved.sum()), replans_run=n_all, replans_post_handover=n_post, scenes=n_scene)
    M["peak_steer_rad_at_key_ticks"] = {f"{s}+{r}s": {v: round(float(W.loc[(s, r), v]), 4) for v in VARS} for s, r in key}

    # ---- 2 counts > 0.3
    def counts(df, by):
        g = df.assign(over=df.peak_abs_steer_rad > 0.3).groupby(by + ["variant"]).over.agg(["sum", "size"]).reset_index()
        t = g.pivot_table(index=by, columns="variant", values="sum")[VARS].astype(int); t.insert(0, "n_replans", g[g.variant == "baseline"].set_index(by)["size"]); return t.reset_index()
    cnt = counts(post, ["group", "state"]); cnt["group"] = pd.Categorical(cnt.group, GROUPS); cnt = cnt.sort_values(["group", "state"], ascending=[True, False])
    cnt_v = counts(post.assign(speed=np.where(post.speed_ge_10, "car >= 10 m/s", "car < 10 m/s")), ["speed", "state"]); cnt_pre = counts(pre.assign(speed=np.where(pre.speed_ge_10, "car >= 10 m/s", "car < 10 m/s")), ["speed"])
    cnt01 = post.assign(over=post.peak_abs_steer_rad > 0.1, speed=np.where(post.speed_ge_10, "car >= 10 m/s", "car < 10 m/s")).pivot_table(index=["speed", "state"], columns="variant", values="over", aggfunc="sum")[VARS].astype(int).reset_index()
    p2 = post.assign(peak_abs_steer_rad=post.peak_abs_steer_2s_rad); cnt2 = counts(p2.assign(speed=np.where(p2.speed_ge_10, "car >= 10 m/s", "car < 10 m/s")), ["speed", "state"]); W2 = post.pivot_table(index=IX, columns="variant", values="peak_abs_steer_2s_rad")[VARS]; K2 = W2.loc[key].reset_index()
    M["replans_peak_steer_gt_0p3_within_2s_by_car_speed_state"] = {f"{r.speed}|{r.state}": dict(n=int(r.n_replans), **{v: int(getattr(r, v)) for v in VARS}) for r in cnt2.itertuples()}
    M["peak_steer_rad_within_2s_at_key_ticks"] = {f"{s}+{r}s": {v: round(float(W2.loc[(s, r), v]), 4) for v in VARS} for s, r in key}
    M["replans_peak_steer_gt_0p3_by_group_state"] = {f"{r.group}|{r.state}": dict(n=int(r.n_replans), **{v: int(getattr(r, v)) for v in VARS}) for r in cnt.itertuples()}
    M["replans_peak_steer_gt_0p3_by_car_speed_state"] = {f"{r.speed}|{r.state}": dict(n=int(r.n_replans), **{v: int(getattr(r, v)) for v in VARS}) for r in cnt_v.itertuples()}
    M["pre_handover_hypothetical_gt_0p3"] = {r.speed: dict(n=int(r.n_replans), **{v: int(getattr(r, v)) for v in VARS}) for r in cnt_pre.itertuples()}

    # ---- 3 change relative to baseline
    rows = []
    for v in FIX:
        d = W[v] - W.baseline; x = post[post.variant == v].set_index(IX); ch = x.plan_changed
        for nm, m in (("car < 10 m/s, pre-swerve", (B.car_speed_mps < 10) & B.pre_swerve), ("car < 10 m/s, disturbed", (B.car_speed_mps < 10) & ~B.pre_swerve), ("car >= 10 m/s, pre-swerve", (B.car_speed_mps >= 10) & B.pre_swerve), ("car >= 10 m/s, disturbed", (B.car_speed_mps >= 10) & ~B.pre_swerve)):
            rows.append(dict(variant=v, subset=nm, n=int(m.sum()), n_plan_or_pick_changed=int(ch[m].sum()), n_abs_change_gt_0p02=int((d[m].abs() > 0.02).sum()), n_worse_by_gt_0p05=int((d[m] > 0.05).sum()), n_better_by_gt_0p05=int((d[m] < -0.05).sum()),
                             max_increase_rad=float(d[m].max()), max_decrease_rad=float(-d[m].min()), max_peak_steer_rad=float(W[v][m].max()), baseline_max_peak_steer_rad=float(W.baseline[m].max()),
                             median_speed_after_1s_minus_baseline_mps=float((x.speed_end_mps - B.speed_end_mps)[m & ch].median()) if (m & ch).any() else np.nan, max_speed_after_1s_minus_baseline_mps=float((x.speed_end_mps - B.speed_end_mps)[m].max())))
    chg = pd.DataFrame(rows); M["change_vs_baseline"] = {f"{r.variant}|{r.subset}": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r._asdict().items() if k not in ("Index", "variant", "subset")} for r in chg.itertuples()}
    share = {v: dict(post_handover=f"{int(post[post.variant == v].plan_changed.sum())}/{n_post}", car_ge_10=f"{int(post[(post.variant == v) & post.speed_ge_10].plan_changed.sum())}/{int(B.speed_ge_10.sum())}") for v in FIX}; M["share_plan_or_pick_changed"] = share
    worst = pd.DataFrame(dict(car_speed_mps=B.car_speed_mps, state=B.state, baseline=W.baseline, A1=W.A1, increase=W.A1 - W.baseline, baseline_lat_from_plan_m=B.peak_abs_lateral_from_plan_path_m,
                              A1_lat_from_plan_m=post[post.variant == "A1"].set_index(IX).peak_abs_lateral_from_plan_path_m, plan_arc_4s_m=B.plan_arc_4s_m)).sort_values("increase", ascending=False).head(12).reset_index()

    # ---- 4 alteration
    ap = alt[alt.phase == "post_handover"]; rows = []
    for v in ("A1", "A2", "A3"):
        for gname, g in [(x, ap[(ap.variant == v) & (ap.group == x)]) for x in GROUPS] + [("all", ap[ap.variant == v]), ("car >= 10 m/s", ap[(ap.variant == v) & ap.speed_ge_10])]:
            r = dict(variant=v, group=gname, n=len(g), n_changed=int(g.plan_changed.sum()))
            for h in ("0p5s", "1s", "2s", "4s"):
                c = g[f"pos_diff_{h}_m"]; r.update({f"{h}_median_m": c.median(), f"{h}_p95_m": c.quantile(0.95), f"{h}_max_m": c.max()})
            rows.append(r)
    alt_t = pd.DataFrame(rows); M["alteration_position_diff_m"] = {f"{r['variant']}|{r['group']}": {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("variant", "group")} for r in rows}
    a1 = ap[ap.variant == "A1"]; a2 = ap[ap.variant == "A2"]
    M["fixA_bookkeeping"] = dict(A1_needs_const_decel_gt_5_n=int((a1.A1_accel_mps2 < -5).sum()), A1_needs_const_decel_gt_4_n=int((a1.A1_accel_mps2 < -4).sum()), A1_needs_const_decel_gt_5_at_car_ge_10_n=int(((a1.A1_accel_mps2 < -5) & a1.speed_ge_10).sum()), of_replans=n_post,
                                 A1_stops_before_4s_n=int(a1.A1_stops_before_4s.sum()), A2_cap_bites_n=int(a2.A2_cap_bites.sum()), A2_end_beyond_model_end_m=[round(x, 2) for x in a2[a2.A2_cap_bites].A2_end_beyond_model_end_m], A3_triggered_n=int(a1.A3_triggered.sum()),
                                 A1_most_negative_accel_mps2=float(a1.A1_accel_mps2.min()))
    hard = a1[a1.A1_accel_mps2 < -4][IX + ["group", "car_speed_mps", "pre_swerve", "a1_mps2", "plan_arc_4s_m", "A1_accel_mps2", "A1_stops_before_4s", "A2_cap_bites", "A2_end_beyond_model_end_m"]].merge(W[["baseline", "A1", "A2", "A3"]].reset_index(), on=IX)

    # ---- 5 B1
    pp = pk[pk.phase == "post_handover"]; hs = pp[pp.speed_ge_10]; chd = pk[pk.pick_changed]
    surv = pp.assign(speed=np.where(pp.speed_ge_10, "car >= 10 m/s", "car < 10 m/s")).groupby("speed").agg(n=("n_survive_of_32", "size"), survive_min=("n_survive_of_32", "min"), survive_median=("n_survive_of_32", "median"), survive_mean=("n_survive_of_32", "mean"),
                                                                                                        n_all_32_survive=("n_survive_of_32", lambda x: int((x == 32).sum())), n_le_1_survive=("n_survive_of_32", lambda x: int((x <= 1).sum())), n_none=("none_left_kept_original", "sum"),
                                                                                                        n_pick_changed=("pick_changed", "sum"), n_dropped_a1_low=("n_dropped_a1_low", "sum"), n_dropped_a1_high=("n_dropped_a1_high", "sum"), n_dropped_too_fast=("n_dropped_too_fast", "sum")).reset_index()
    chd_t = chd[["scene", "tick_after_handover_s", "phase", "car_speed_mps", "pre_swerve", "n_survive_of_32", "original_pick", "new_pick", "new_pick_rank_final_score", "new_pick_rank_plain_score", "original_a1_mps2", "new_a1_mps2", "original_mean_speed_4s_mps", "new_mean_speed_4s_mps",
                 "human_mean_speed_4s_mps", "new_lateral_from_original_path_2s_m", "new_lateral_from_original_path_4s_m"]].merge(tb.pivot_table(index=IX, columns="variant", values="peak_abs_steer_rad")[["baseline", "B1", "B2"]].reset_index(), on=IX)
    cp = chd[chd.phase == "post_handover"]
    M["fixB"] = dict(pick_changed_post_handover=f"{int(pp.pick_changed.sum())}/{len(pp)}", pick_changed_car_ge_10=f"{int(hs.pick_changed.sum())}/{len(hs)}", pick_changed_pre_handover_hypothetical=f"{int(pk[pk.phase != 'post_handover'].pick_changed.sum())}/{int((pk.phase != 'post_handover').sum())}",
                     none_left=int(pp.none_left_kept_original.sum()), survivors_min_car_ge_10=int(hs.n_survive_of_32.min()), survivors_median_car_ge_10=float(hs.n_survive_of_32.median()), new_pick_rank_final_score_median=float(cp.new_pick_rank_final_score.median()), new_pick_rank_final_score_max=int(cp.new_pick_rank_final_score.max()),
                     new_minus_original_mean_speed_4s_mps_median=float((cp.new_mean_speed_4s_mps - cp.original_mean_speed_4s_mps).median()), new_minus_human_mean_speed_4s_mps_median=float((cp.new_mean_speed_4s_mps - cp.human_mean_speed_4s_mps).median()),
                     original_minus_human_mean_speed_4s_mps_median=float((cp.original_mean_speed_4s_mps - cp.human_mean_speed_4s_mps).median()), abs_lateral_from_original_2s_m_median=float(cp.new_lateral_from_original_path_2s_m.abs().median()), abs_lateral_from_original_2s_m_max=float(cp.new_lateral_from_original_path_2s_m.abs().max()),
                     abs_lateral_from_original_4s_m_median=float(cp.new_lateral_from_original_path_4s_m.abs().median()), abs_lateral_from_original_4s_m_max=float(cp.new_lateral_from_original_path_4s_m.abs().max()))

    # ---- 6 later braking
    M["later_braking_cases"] = {f"{f}|{p}": int(n) for (f, p), n in lt.groupby(["fix", "phase"]).size().items()}
    M["later_braking_cases_followed_by_recorded_collision_within_4s"] = {f"{f}|{p}": int(n) for (f, p), n in lt.groupby(["fix", "phase"]).recorded_collision_any_within_4s.sum().items()}
    lt_t = lt[["fix", "scene", "tick_after_handover_s", "phase", "car_speed_mps", "pre_swerve", "original_plan_end_speed_mps", "original_plan_a1_mps2", "original_const_accel_to_end_mps2", "fixed_plan_end_speed_mps", "ends_beyond_model_end_m", "recorded_collision_any_within_4s",
                "recorded_collision_front_or_lateral_within_4s", "recorded_offroad_within_4s", "recorded_min_distance_to_obstacle_within_4s_m"]]

    # ---- 7 long holds
    rows = []
    for (s, rel, v), L in longs.items():
        f2 = L["t"] <= 2.0 + 1e-9; r = dict(scene=s, tick_after_handover_s=rel, variant=v, peak_abs_steer_first_2s_rad=float(np.abs(L["steer"][f2]).max()), peak_abs_lateral_from_plan_path_first_2s_m=float(np.nanmax(np.abs(L["lat_plan"][f2]))), min_u_long_first_2s_mps2=float(L["ulong"][f2].min()), peak_abs_steer_rad=float(np.abs(L["steer"]).max()), peak_abs_lateral_from_plan_path_m=float(np.nanmax(np.abs(L["lat_plan"]))), min_u_long_mps2=float(L["ulong"].min()), plan_arc_4s_m=float(np.hypot(*np.diff(L["plan"][:, :2], axis=0).T).sum()), dist_travelled_4s_m=L["summary"]["dist_travelled_m"])
        for t in (1.0, 2.0, 3.0, 4.0):
            i = int(np.argmin(np.abs(L["t"] - t))); r.update({f"steer_{t:.0f}s_rad": float(L["steer"][i]), f"speed_{t:.0f}s_mps": float(L["speed"][i]), f"lat_{t:.0f}s_m": float(L["lat_plan"][i])})
        rows.append(r)
    lh = pd.DataFrame(rows); lh.to_csv(HERE / "long_hold_summary.csv", index=False)
    M["long_hold_4s"] = {f"{r.scene}+{r.tick_after_handover_s}s|{r.variant}": dict(peak_abs_steer_first_2s_rad=round(r.peak_abs_steer_first_2s_rad, 4), peak_abs_lateral_from_plan_path_first_2s_m=round(r.peak_abs_lateral_from_plan_path_first_2s_m, 3), speed_2s_mps=round(r.speed_2s_mps, 2), peak_abs_steer_whole_4s_rad=round(r.peak_abs_steer_rad, 4), speed_4s_mps=round(r.speed_4s_mps, 2)) for r in lh.itertuples()}

    # ---- 8 Fix C
    S = sw[sw.kind == "sweep"]; order = list(dict.fromkeys(["default"] + [c for c in S.controller.unique() if c != "default"]))
    sweep_t = {s: S[S.seed_scene == s].pivot_table(index="controller", columns="const_decel_mps2", values="peak_abs_steer_rad").loc[order].reset_index() for s in ("04749bb9", "bd07fb91")}
    spd_t = {s: S[S.seed_scene == s].pivot_table(index="controller", columns="const_decel_mps2", values="speed_end_mps").loc[order].reset_index() for s in ("04749bb9", "bd07fb91")}
    def onset(g, thr):
        g = g.sort_values("const_decel_mps2"); o = g[g.peak_abs_steer_rad > thr]; return float(o.const_decel_mps2.iloc[0]) if len(o) else np.nan
    on = pd.DataFrame([dict(controller=c, **{f"onset_gt_{str(th).replace('.', 'p')}_rad_at_{v}": onset(S[(S.controller == c) & (S.seed_scene == s)], th) for s, v in (("04749bb9", "23p6mps"), ("bd07fb91", "15p7mps")) for th in (0.1, 0.3)}) for c in order])
    O = sw[sw.kind == "original_plan"].copy(); O["case"] = O.seed_scene + "+" + O.tick_after_handover_s.astype(str) + "s " + O.seeding
    op = O.pivot_table(index="controller", columns="case", values="peak_abs_steer_rad").loc[order].reset_index(); opu = O.pivot_table(index="controller", columns="case", values="speed_end_mps").loc[order].reset_index()
    def trk(df):
        return df.groupby("controller").agg(n_scenes=("scene", "size"), max_lateral_dev_m=("max_abs_lateral_dev_from_recorded_m", "max"), max_longitudinal_dev_m=("max_abs_longitudinal_dev_from_recorded_m", "max"), peak_steer_rad=("peak_abs_steer_rad", "max"),
                                            recorded_peak_steer_rad=("recorded_peak_abs_steer_rad", "max"), max_steer_diff_from_recorded_rad=("max_abs_steer_diff_from_recorded_rad", "max"), solves_not_solved=("n_solves_not_solved", "sum")).loc[order].reset_index()
    tr_clean = trk(tr[tr.scene_set == "clean5"]); turn_sc = sorted(tr[(tr.controller == "default") & (tr.recorded_peak_abs_steer_rad.between(0.05, 0.35)) & ~tr.scene.isin(["04749bb9", "acd22430"])].scene); tr_turn = trk(tr[tr.scene.isin(turn_sc)])
    M["fixC"] = dict(C5_is=C5, onset_decel_mps2={r.controller: {k: (None if pd.isna(v) else v) for k, v in r._asdict().items() if k.startswith("onset")} for r in on.itertuples()},
                     original_plan_peak_steer_rad={r["controller"]: {k: round(float(v), 4) for k, v in r.items() if k != "controller"} for r in op.to_dict("records")},
                     tracking_clean5={r.controller: dict(max_lateral_dev_m=round(r.max_lateral_dev_m, 4), max_longitudinal_dev_m=round(r.max_longitudinal_dev_m, 4), peak_steer_rad=round(r.peak_steer_rad, 4)) for r in tr_clean.itertuples()},
                     tracking_turning_scenes=dict(scenes=turn_sc, **{r.controller: dict(max_lateral_dev_m=round(r.max_lateral_dev_m, 4), max_longitudinal_dev_m=round(r.max_longitudinal_dev_m, 4), peak_steer_rad=round(r.peak_steer_rad, 4)) for r in tr_turn.itertuples()}))

    # ---------------------------------------------------------------- plots
    ps = B[B.pre_swerve]; sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 25), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#86b6ef", "#3987e5", "#0d366b"]))
    for n, v in ((1, "A1"), (2, "B1")):
        fig, ax = plt.subplots(figsize=(6.2, 5.4)); ax.plot([0, 0.8], [0, 0.8], color="#87867f", lw=1, ls="--", label="no change (diagonal)"); ax.axhline(0.3, color="#b0aea5", lw=0.8); ax.axvline(0.3, color="#b0aea5", lw=0.8)
        o = np.argsort(ps.car_speed_mps.values); ax.scatter(W.baseline[ps.index].values[o], W[v][ps.index].values[o], c=sm.to_rgba(ps.car_speed_mps.values[o]), s=28, edgecolor="white", linewidth=0.6, zorder=3)
        for (s, r) in [k for k in key if k in ps.index]: ax.annotate(f"{s} +{r:g}s", (W.baseline[(s, r)], W[v][(s, r)]), xytext=(-4, 10 + 11 * [k_ for k_ in key if k_ in ps.index].index((s, r))), textcoords="offset points", fontsize=7.5, color=INK, ha="right")
        fig.colorbar(sm, ax=ax, label="car speed at the replan (m/s)"); ax.set_xlabel("baseline (plan actually chosen): peak |front steering angle| in 1.0 s (rad)"); ax.set_ylabel(f"{v}: peak |front steering angle| in 1.0 s (rad)")
        ax.set_title(f"Plot {n}. Peak steering, baseline vs {v}: every pre-swerve replan after handover\n(n = {len(ps)}, 30 scenes; grey lines = 0.3 rad)", fontsize=10); ax.legend(loc="upper left", frameon=False); ax.set_xlim(-0.02, 0.82); ax.set_ylim(-0.02, 0.82)
        fig.savefig(HERE / f"plot{n}_peak_steer_baseline_vs_{v}.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, st in zip(axs, ("pre-swerve", "disturbed")):
        c = cnt[cnt.state == st].set_index("group").reindex(GROUPS); x = np.arange(len(GROUPS)); wd = 0.13
        for i, v in enumerate(VARS):
            bars = ax.bar(x + (i - 2.5) * wd, c[v].fillna(0).values, wd * 0.86, color=COL[v], label=v)
            for b_, val in zip(bars, c[v].fillna(0).values): ax.text(b_.get_x() + b_.get_width() / 2, val, f"{int(val)}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([f"{g}\nn = {int(c.n_replans[g]) if pd.notna(c.n_replans[g]) else 0}" for g in GROUPS]); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); ax.set_ylim(0, max(2.4, ax.get_ylim()[1] * 1.08)); ax.set_ylabel("replans with peak |steer| > 0.3 rad (count)"); ax.set_xlabel("scene group (n = replans in the group)"); ax.grid(axis="x", visible=False)
        ax.set_title(f"{st} replans" + (" (valid)" if st == "pre-swerve" else " (indicative only)"), fontsize=10)
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.13)); fig.suptitle("Plot 3. Replans whose held plan steers > 0.3 rad within 1.0 s, per variant (post-handover replans, 30 scenes)", fontsize=10.5); fig.savefig(HERE / "plot3_counts_gt_0p3.png"); plt.close(fig)
    for n, kq, yl, ttl in ((4, "steer", "front steering angle (rad)", "steering"), (5, "speed", "speed (m/s)", "speed")):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, s in zip(axs, ("04749bb9", "acd22430")):
            for v in VARS:
                L = longs[(s, 0.0, v)]; ax.plot(L["t"], L[kq], color=COL[v], label=v, ls="-" if v in ("baseline", "A1", "B1") else (0, (4, 2)))
            ax.axvspan(2.0, 4.0, color="#f0eee6", zorder=0); ax.set_title(f"{s}, handover tick, plan held 4.0 s", fontsize=10); ax.set_xlabel("time since the handover tick (s)")
        axs[0].set_ylabel(yl); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.07)); fig.suptitle(f"Plot {n}. Highway handover ticks: {ttl} over a 4.0 s hold, baseline and the five fix variants\nshaded 2-4 s = test artefact: the 2 s MPC horizon reaches past the plan's last pose (4 s), where the reference stands still", fontsize=10, y=1.04); fig.savefig(HERE / f"plot{n}_long_hold_{kq}.png"); plt.close(fig)
    show = ["default", "C1_track_from_step0", "C2_long_weight_0p5", "C3_heading_weight_10", "C4_steer_rate_weight_50", C5]; cc = dict(zip(show, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (s, v) in zip(axs, (("04749bb9", "23.6"), ("bd07fb91", "15.7"))):
        for c in show:
            g = S[(S.controller == c) & (S.seed_scene == s)].sort_values("const_decel_mps2"); ax.plot(g.const_decel_mps2, g.peak_abs_steer_rad, marker="o", ms=4, color=cc[c], label=c + (" (= C5)" if c == C5 else ""))
        ax.axhline(0.3, color="#b0aea5", lw=0.8); ax.set_title(f"car at {v} m/s ({s} handover state)", fontsize=10); ax.set_xlabel("constant deceleration of the plan (m/s$^2$)")
    axs[0].set_ylabel("peak |front steering angle| in 1.0 s (rad)"); axs[0].legend(frameon=False, fontsize=8); fig.suptitle("Plot 6. Fix C: peak steering vs deceleration of a straight constant-deceleration plan, default controller and variants\n(the other 5 tries are in fixC_sweep.csv)", fontsize=10.5); fig.savefig(HERE / "plot6_fixC_sweep.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4)); tc = tr[tr.scene_set == "clean5"]; scs = sorted(tc.scene.unique()); y = np.arange(len(order))
    for ax, (col, lab) in zip(axs, (("max_abs_lateral_dev_from_recorded_m", "max |lateral deviation| from recorded (m)"), ("max_abs_longitudinal_dev_from_recorded_m", "max |longitudinal deviation| from recorded (m)"), ("peak_abs_steer_rad", "peak |front steering angle| (rad)"))):
        p = tc.pivot_table(index="controller", columns="scene", values=col).loc[order]
        for j, s in enumerate(scs): ax.scatter(p[s].values, y, s=30, color=list(COL.values())[j], edgecolor="white", linewidth=0.6, label=s, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(order if ax is axs[0] else []); ax.set_xlabel(lab); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, title="clean scene", ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.12)); axs[0].set_ylabel("controller variant")
    fig.suptitle("Plot 7. Fix C tracking check: 5 clean scenes, 5 s after handover, closed loop on recorded plans (all five scenes are near-straight: recorded peak steer <= 0.004 rad)", fontsize=10); fig.savefig(HERE / "plot7_fixC_tracking.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2)); bins = np.r_[0, np.logspace(-2, 1.4, 28)]
    for ax, h in zip(axs, ("1s", "4s")):
        for v, ls in (("A1", "-"), ("A2", (0, (4, 2))), ("A3", "-")):
            x = ap[ap.variant == v][f"pos_diff_{h}_m"].values; ax.hist(np.clip(x, 0.011, None), bins=bins[1:], histtype="step", lw=2, color=COL[v], ls=ls, label=f"{v} (n changed > 1 cm: {(x > 0.01).sum()} of {len(x)})")
        ax.set_xscale("log"); ax.set_xlabel(f"position difference from the chosen plan at {h[:-1]} s (m, log; first bin = <= 1 cm)"); ax.set_ylabel("replans (count)"); ax.legend(frameon=False, fontsize=8); ax.set_title(f"at {h[:-1]}.0 s", fontsize=10)
    fig.suptitle("Plot 8. How much Fix A alters the chosen plan: post-handover replans, 30 scenes", fontsize=10.5); fig.savefig(HERE / "plot8_alteration_hist.png"); plt.close(fig)

    # ---------------------------------------------------------------- README
    w = out.append
    w("# T03 - offline validation of three candidate fixes for the high-speed swerve (Drive-JEPA adapter 0.1.6)\n")
    w("Offline, CPU only. Scripts: `t03_common.py` (imports t02's reader/controller, adds the fork-based branch, Fix A/B constructions, a Sim with other MPC settings), `t03_testbed.py`, `t03_fixc.py`, `t03_report.py` (this file is generated). "
      "Interpreter `{runs}/alpasim_bulk/nurec-venv/bin/python`. Nothing under `alpasim/` or `adapters/` was edited. Measured facts only; no ranking.\n")
    w("## 0. What was run\n")
    w(f"- **Full set, no subsampling**: all {n_all} replans of all {n_scene} scenes x 6 variants = {len(tb)} held-plan runs (66 s wall on 15 processes). {int(tb.n_solves_not_solved.sum())} MPC solves returned a status other than solved.\n"
      f"- {n_post} replans are at or after handover (34 per scene). The other {n_all - n_post} (5 per scene, -2.5 s .. -0.5 s) are **before handover**: the controller request on those ticks carries the 151-pose ground-truth trajectory, not the model's plan, so the model's plan never reached the controller. "
      "They are run as a hypothetical ('what if this plan had been handed to the controller at the recorded state') with the menu's chosen plan as baseline, tagged `pre_handover_hypothetical`, and counted separately everywhere.\n"
      "- Procedure per replan: one simulator per scene is replayed teacher-forced from the session's first request (t02's `warm_tf`); at each replan tick the process is forked once per distinct plan, the child holds that plan for 1.0 s in one request (10 solves, 50 ms reporting) and returns the metrics; the parent continues the recorded replay. "
      "Checked: after branching, the parent still reproduces the recorded steering to 3e-16 rad; two branches of the same plan give identical numbers.\n"
      "- Baseline plan after handover = the 9 poses in the recorded controller request (max difference from the regenerated menu's chosen proposal asserted < 1 mm on every replan).\n"
      "- Tags: `car_speed_mps` = recorded speed at the tick; `pre_swerve` = no recorded |front steering angle| > 0.2 rad in the preceding 1.0 s and no off-road tick so far; groups by handover speed: highway (04749bb9, acd22430), bd07fb91, other >= 10 m/s (9 scenes), < 10 m/s (18 scenes).\n"
      "- `peak_abs_lateral_from_plan_path_m` = perpendicular distance of the car from the held plan's own polyline (continued straight past its end; blank when the plan is shorter than 0.5 m); `peak_abs_lateral_rig_m` = t02's definition (y in the rig frame at the tick).\n")
    w("Variant constructions (path = polyline through the 9 poses, arc-length parametrised, as t02's `Path`):\n"
      "- **A1** s(t) = v0 t + a t^2/2 with a = 2 (L - 4 v0)/16, L = the plan's 4 s arc length; if v0 + 4a < 0 the deceleration that stops exactly at L is used and the plan stays there. Heading = path tangent (3-point smoothed, as t02 B4e); where neighbouring poses are < 0.05 m apart the tangent is undefined and the path's re-sampled recorded heading is used.\n"
      "- **A2** = A1 with a >= -4.0 m/s2 when car speed >= 10 m/s. **A3** = recorded plan unless a1 < -4.0 and speed >= 10; then poses at 0.5/1.0/1.5 s are re-spaced on the path with the constant acceleration that reaches the plan's 2 s arc length (heading re-sampled), poses 2.0-4.0 s untouched.\n"
      "- **B1** drop proposals with a1 < -4.0 (speed >= 10) or a1 > +4.0 or mean 4 s speed > 1.5 v0 + 5; pick = argmax of the adapter's final score (`pdm_score`, the score after the momentum recalibration; argmax of it equals the recorded `chosen` on 1,170/1,170 replans) among the survivors. **B2** = A1 applied to the B1 pick.\n"
      "- a1 = 2 (s(1 s) - v0), t02's ordering variable.\n")
    w("## 1. Baseline reproduction and the key ticks\n")
    w(f"Baseline at the two highway handover ticks: **{W.loc[('04749bb9', 0.0), 'baseline']:.3f} / {W.loc[('acd22430', 0.0), 'baseline']:.3f} rad** (t02: 0.525 / 0.401). Peak |front steering angle| within 1.0 s (rad):\n")
    w(md(K))
    w("## 2. Replans whose held plan steers > 0.3 rad within 1.0 s (post-handover; denominators in `n_replans`)\n"); w(md(cnt)); w("By car speed at the tick:\n"); w(md(cnt_v)); w("Same split, threshold 0.1 rad:\n"); w(md(cnt01))
    w("**Added (not asked): the same holds continued to 2.0 s** (the longest hold whose 2 s MPC horizon stays inside the 4 s plan; first 1.0 s identical to the 1.0 s hold). Replans steering > 0.3 rad within 2.0 s:\n"); w(md(cnt2)); w("Peak |steer| within 2.0 s at the key ticks (rad):\n"); w(md(K2))
    w("Pre-handover hypothetical replans (not part of the counts above):\n"); w(md(cnt_pre))
    w("## 3. Change relative to baseline, by car speed and state\n")
    w("`n_plan_or_pick_changed` = plan differs from the recorded one by > 1e-6 (A1/A2 re-time every plan; B1 only when the pick changes). Speed after 1 s minus baseline: positive = the car is faster after the 1.0 s hold than with the recorded plan (median over changed plans).\n"); w(md(chg))
    w(f"Share of replans whose plan or pick changed: " + "; ".join(f"{v}: {d['post_handover']} post-handover, {d['car_ge_10']} at car speed >= 10 m/s" for v, d in share.items()) + ".\n")
    w("The 12 largest increases of peak steering under A1 (all at car speed < 2 m/s except one disturbed highway tick; lateral distance from the plan path stays in millimetres on the slow ones):\n"); w(md(worst))
    w("## 4. Fix A: how much the plan is altered (position difference from the recorded chosen plan, m; post-handover replans)\n"); w(md(alt_t))
    fb = M["fixA_bookkeeping"]
    w(f"- A1's constant acceleration is below -5 m/s2 on {fb['A1_needs_const_decel_gt_5_n']} of {n_post} replans ({fb['A1_needs_const_decel_gt_5_at_car_ge_10_n']} of them at car speed >= 10 m/s) and below -4 m/s2 on {fb['A1_needs_const_decel_gt_4_n']}; most negative {fb['A1_most_negative_accel_mps2']:.2f} m/s2. "
      f"A1 uses the stop-at-end-point rule on {fb['A1_stops_before_4s_n']} replans.\n- A2's cap bites on {fb['A2_cap_bites_n']} replans; the plan then ends {fb['A2_end_beyond_model_end_m']} m beyond the model's end point. A3 triggers on {fb['A3_triggered_n']} replans.\n")
    w("Replans where A1 needs more than 4 m/s2 of constant deceleration (peak steer per variant, rad):\n"); w(md(hard))
    w("## 5. Fix B1: menu filter\n"); w(md(surv))
    fbm = M["fixB"]
    w(f"Pick changed on {fbm['pick_changed_post_handover']} post-handover replans ({fbm['pick_changed_car_ge_10']} at car speed >= 10 m/s; {fbm['pick_changed_pre_handover_hypothetical']} of the pre-handover hypothetical ones). The filter never emptied the menu. Every replan whose pick changed:\n"); w(md(chd_t))
    w("## 6. Cost side: replans where the fix brakes later or less than the model asked\n")
    w("A2 rows = the cap bites (plan ends beyond the model's end point). B1 rows = the pick changed and the new pick's 4 s arc length is > 0.1 m longer than the original's. The run's per-tick metrics contain collision_*, offroad and min_distance_to_obstacle_m but **no traffic-light metric**, so 'red light followed' cannot be read from the recorded metrics. "
      "The recorded-run columns describe what happened in the recorded run (with the original plan and the swerve), within 4 s after the tick.\n"); w(md(lt_t))
    w("## 7. Stability check: 4.0 s holds\n")
    w("**Only the first 2.0 s of a 4 s hold are a fair test.** The plan has poses up to 4.0 s and the MPC looks 2.0 s ahead; `_tvp_fun` clamps reference times to the plan's last pose, so from 2.0 s into the hold the reference for the far horizon steps stands still at the end point. "
      "Every variant, including the re-timed ones, then gets a full -9 m/s2 brake command and a steering swing to 0.68-0.785 rad between 3 and 4 s (in the run this never happens: a new plan arrives every 0.5 s). The `first_2s` columns are the valid part; the whole-4 s columns are shown as asked.\n")
    w(md(lh))
    w("## 8. Fix C: controller settings\n")
    w("Controller config of every 0.1.6 run (`runs/drivejepa/drivejepa_016_<scene>/controller-config.yaml`; the task's path `controller/*.yaml` does not exist, `controller/` holds only the controller CSV):\n\n```\nmpc_implementation: nonlinear\nn_horizon: 20\ndt_mpc: 0.1\ngains:\n  long_position_weight: 2.0\n  lat_position_weight: 1.0\n  heading_weight: 1.0\n  acceleration_weight: 0.1\n  rel_front_steering_angle_weight: 5.0\n  rel_acceleration_weight: 1.0\n  idx_start_penalty: 10\n```\n")
    w("Cost as written in `nonlinear_mpc.py::_setup_mpc` (re-read; stage term and terminal term are the same expression): `tracking_enabled_k * ( long_w * (x - x_ref)^2 + lat_w * (y - y_ref)^2 + acc_w * acceleration^2 + heading_w * wrap(yaw - heading_ref)^2 )`, "
      "x, y in the rig frame at the solve (no rotation into the reference heading), `tracking_enabled_k` = 1 for horizon step k >= idx_start_penalty else 0 (`_tvp_fun`). **The gate multiplies the whole bracket**, so for the first 10 steps (0-0.9 s) the heading and acceleration-state terms are off as well, not only the position terms. "
      "Plus do_mpc's rterm `rel_front_steering_angle_weight * (d steer_cmd)^2 + rel_acceleration_weight * (d acc_cmd)^2` on the change of each command between consecutive steps (active on every step). With the run's values: [2 x^2 + 1 y^2 + 0.1 acc^2 + 1 yaw^2] on steps 10..20, rterm 5 / 1. Bounds: acceleration_cmd in [-9, +6] m/s2.\n\n"
      "**Exposed without a code change** (fields of `MPCGains` / `ControllerConfig`, i.e. the controller yaml): the six weights, `idx_start_penalty`, `n_horizon`, `dt_mpc`, `mpc_implementation`. **Would need a code change**: the command bounds (acceleration_cmd in [-9, +6] m/s2, steering cmd +-2 rad), the state bounds (steering +-0.785 rad, yaw +-0.78 rad, ...), "
      "any penalty on the steering angle itself or on lateral acceleration (none exists), the vehicle parameters. The only exposed steering penalty is the steering-command rate term `rel_front_steering_angle_weight`.\n")
    w(f"Variants: C1 idx_start_penalty 10 -> 0; C2 long_position_weight 2 -> 0.5; C3 heading_weight 1 -> 10; C4 rel_front_steering_angle_weight 5 -> 50; six further tries T1-T6 (names say what was changed). **C5 = `{C5}`**, the try with the smallest sum of peak steer over the three original-plan cases below (criterion fixed after the fact; T4 is within 0.002 rad of it).\n")
    w("Peak |steer| (rad) vs constant deceleration of a straight plan, car at 23.6 m/s (04749bb9 handover state):\n"); w(md(sweep_t["04749bb9"], 2)); w("Car at 15.7 m/s (bd07fb91 handover state):\n"); w(md(sweep_t["bd07fb91"], 2))
    w("Speed after the 1.0 s hold (m/s), 23.6 m/s case (how much braking was actually delivered):\n"); w(md(spd_t["04749bb9"], 1))
    w("Onset = smallest swept deceleration whose peak steer exceeds the threshold (blank = none up to 12 m/s2):\n"); w(md(on, 1))
    w("ORIGINAL chosen plan held 1.0 s; peak |steer| (rad). Highway ticks are warm (replayed to handover with the variant controller); bd07fb91 +6.0 s is seeded cold from the recorded state for every controller (default warm shown for comparison):\n"); w(md(op)); w("Speed after that 1.0 s hold (m/s):\n"); w(md(opu, 2))
    w("Tracking check, 5 clean scenes (02e075b9, 16b83346, 6725df2d, 01d503d4, 048b974e), 5 s after handover, closed loop on the simulator's own pose with the recorded plans fixed in the world; worst scene per controller. **All five scenes are near-straight in that window (recorded peak steer <= 0.004 rad), so this check exercises longitudinal tracking and says almost nothing about steering response.**\n"); w(md(tr_clean, 4))
    w(f"Supplementary (not asked): the same check on the scenes whose recorded peak steer in those 5 s is 0.05-0.35 rad ({', '.join(turn_sc)}; all handed over below 7 m/s). Per scene x controller rows for all 30 scenes are in `fixC_tracking.csv`.\n"); w(md(tr_turn, 4))
    w("## 9. The task's predictions next to the measurements\n")
    hv = cnt_v[(cnt_v.speed == "car >= 10 m/s") & (cnt_v.state == "pre-swerve")].iloc[0]; sl = chg[(chg.variant == "A1") & chg.subset.str.startswith("car < 10")]
    w(f"- F-A, handover ticks < 0.05 rad: A1 {W.loc[('04749bb9', 0.0), 'A1']:.3f} / {W.loc[('acd22430', 0.0), 'A1']:.3f}; A2 the same (cap does not bite there); A3 {W.loc[('04749bb9', 0.0), 'A3']:.3f} / {W.loc[('acd22430', 0.0), 'A3']:.3f}.\n"
      f"- F-A, pre-swerve replans at car speed >= 10 m/s steering > 0.3 rad: baseline {hv.baseline} of {hv.n_replans}, A1 {hv.A1}, A2 {hv.A2}, A3 {hv.A3} (B1 {hv.B1}, B2 {hv.B2}). The baseline count is {hv.baseline}, so the '>= 90 %' statement rests on {hv.baseline} replans.\n"
      f"- F-A, 4 s end point unchanged: A1 max position difference at 4 s = {alt_t[(alt_t.variant == 'A1') & (alt_t.group == 'all')]['4s_max_m'].iloc[0]:.3f} m (A2: {alt_t[(alt_t.variant == 'A2') & (alt_t.group == 'all')]['4s_max_m'].iloc[0]:.2f} m where the cap bites).\n"
      f"- F-A, slow replans (< 10 m/s) change by <= 0.02 rad: under A1 {int(sl.n_abs_change_gt_0p02.sum())} of {int(sl.n.sum())} slow replans change by more than 0.02 rad (max increase {sl.max_increase_rad.max():.3f} rad), {int(sl[sl.subset.str.contains('pre-swerve')].n_abs_change_gt_0p02.sum())} of them pre-swerve. A3 and B1 change none.\n"
      f"- F-A, plans needing > 5 m/s2 constant deceleration: {fb['A1_needs_const_decel_gt_5_n']} of {n_post} replans ({100 * fb['A1_needs_const_decel_gt_5_n'] / n_post:.1f} %).\n"
      f"- F-B, handover ticks < 0.05 rad: B1 {W.loc[('04749bb9', 0.0), 'B1']:.3f} / {W.loc[('acd22430', 0.0), 'B1']:.3f}; +0.5 s: {W.loc[('04749bb9', 0.5), 'B1']:.3f} / {W.loc[('acd22430', 0.5), 'B1']:.3f}; +3.0 s: {W.loc[('04749bb9', 3.0), 'B1']:.3f} / {W.loc[('acd22430', 3.0), 'B1']:.3f}. Pick changed on {fbm['pick_changed_car_ge_10']} replans at car speed >= 10 m/s.\n"
      f"- F-C: see section 8 (C1 = tracking from step 0; C4 = steering-rate weight x10; C2 = lower longitudinal weight).\n")
    w("## 10. What this test bed cannot show\n")
    w("- Each row is one plan held for 1.0 s from the recorded state. After the first changed plan the car would be elsewhere and the model would see other images, so later replans of the recorded run are not what a fixed system would face. `pre_swerve` rows are the valid ones; `disturbed` rows are indicative only (the recorded car is already steering hard or off-road there, e.g. all highway ticks from +0.5 s).\n"
      "- A held plan is not the 20 Hz stream: in the run the adapter re-anchors the plan every 50 ms and replans every 0.5 s.\n- Nothing here measures collisions, comfort or progress of a fixed system; the 'cost side' columns come from the recorded (unfixed) run.\n"
      "- Fix C variants were not tested on any turning scene above 7 m/s; the clean-scene tracking check is on near-straight driving.\n- At car speed below ~2 m/s, steering angles of 0.2-0.5 rad occur in ordinary manoeuvring; '> 0.3 rad' is not a swerve criterion there (lateral distance from the plan path is in the CSV).\n")
    w("## Files\n\n`testbed.csv` (scene x replan x variant), `fixA_alteration.csv`, `fixB_picks.csv`, `later_braking_cases.csv`, `long_hold.csv` (time series), `long_hold_summary.csv`, `fixC_sweep.csv` (kind = sweep / original_plan), `fixC_tracking.csv`, `metrics.json`, plots 1-8.\n")
    (HERE / "README.md").write_text("\n".join(out)); json.dump(M, open(HERE / "metrics.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("ok", len("\n".join(out)))


if __name__ == "__main__":
    main()
