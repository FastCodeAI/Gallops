#!/usr/bin/env python
"""T05: Part 2 tables (scorer_probe.csv, scorer_geometry.csv, clearance_bins.csv), plots, README.md, cache/metrics.json.
  nurec-venv/bin/python t05_report.py"""
import json, math, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MP
from scipy.stats import spearmanr
import t05_common as F, t05_rule as R
T = F.T; C = F.C; H = F.HERE
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"   # dataviz reference palette, fixed order
GOOD, WARN, CRIT, INK, MUTED, SURF = "#0ca30c", "#fab219", "#d03b3b", "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "savefig.dpi": 130, "lines.linewidth": 2})
MAINCK = "incl_pre_handover"; BINS = [(-1, 0, "0 (overlap)"), (0, 0.3, "0-0.3"), (0.3, 0.6, "0.3-0.6"), (0.6, 1.0, "0.6-1.0"), (1.0, np.inf, "> 1.0")]


def md(df, nd=2):
    df = df.copy()
    for c in df:
        if df[c].dtype.kind == "f": df[c] = df[c].map(lambda x: "" if pd.isna(x) else ("inf" if np.isinf(x) else f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"))
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def cbin(x):
    for lo, hi, nm in BINS:
        if (x <= 0 and hi == 0) or (lo < x <= hi and hi > 0): return nm
    return "> 1.0"


def boxes(ax, o, t_us, ego, lim=None, **kw):
    i = int(np.argmin(np.abs(o["ap_t"] - t_us))); a = o["A"][i]; ok = np.flatnonzero(np.isfinite(a[:, 0]))
    for j, k in zip(ok, F.corners(a[ok, :2].astype(float), a[ok, 2].astype(float), o["dims"][ok] / 2)):
        r = C.to_rig(np.c_[k, np.zeros(4)], ego)
        if np.abs(r[:, 0]).min() > 40: continue
        ax.add_patch(MP(np.c_[-r[:, 1], r[:, 0]], closed=True, **kw))
        if kw.get("label_it"): pass


def footprint(ax, o, pose_rig, **kw):
    x, y, yaw = pose_rig; c = np.array([[x + o["rig_to_aabb_x"] * math.cos(yaw), y + o["rig_to_aabb_x"] * math.sin(yaw)]]); k = F.corners(c, np.array([yaw]), o["ego_dims"][None] / 2)[0]; ax.add_patch(MP(np.c_[-k[:, 1], k[:, 0]], closed=True, **kw))


def topax(ax, title, xl=(-9, 6), yl=(-6, 22)):
    ax.set_aspect("equal"); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xlabel("lateral, left is negative here (m; left of the car drawn on the left)"); ax.set_ylabel("forward (m)"); ax.set_title(title)


def main():
    D = pd.read_pickle(H / "cache" / "props_all.pkl"); G = pd.read_csv(H / "antistall_grid.csv"); FF = pd.read_csv(H / "antistall_first_firings.csv"); E = pd.read_csv(H / "stationary_episodes.csv"); RK = pickle.load(open(H / "cache" / "rule.pkl", "rb"))
    PJ = json.load(open(H / "cache" / "probe_raw.json")); PP = json.load(open(H / "cache" / "probe_paths.json")); MET = {}
    # ---------------- Part 2 tables
    two = D[D.scene.isin(F.STUCK)].copy(); two["set"] = "stuck scene"; two["clearance_bin"] = two.min_clearance_m.map(cbin)
    # comparable situations in the other 28 scenes (0.1.6): car < 0.5 m/s, human moving or starting within 2 s, an actor box within 4 m ahead in the ego-width corridor
    comp = []
    for s in T.scenes():
        if s in F.STUCK: continue
        o = T.load("0.1.6", s); g = D[(D.scene == s) & (D.version == "0.1.6") & D.chosen & (D.ego_speed_mps < 0.5)]
        for r in g.itertuples():
            hv, wait = R.human(o, r.t_after_handover_s); ah = R.ahead(o, r.tn, o["egot"][r.tn][1:4])
            if (hv > 0.5 or (np.isfinite(wait) and wait <= 2.0)) and np.isfinite(ah["ahead_gap_m"]) and ah["ahead_gap_m"] <= 4.0: comp.append(dict(scene=s, tn=r.tn, t_after_handover_s=r.t_after_handover_s, ahead_gap_m=ah["ahead_gap_m"], ahead_label=ah["ahead_label"], human_speed_mps=hv, human_starts_moving_in_s=wait, chosen_leave=bool(r.leave)))
    CP = pd.DataFrame(comp); oth = D[(D.version == "0.1.6")].merge(CP[["scene", "tn"]], on=["scene", "tn"]) if len(CP) else D.iloc[:0]; oth = oth.copy(); oth["set"] = "comparable replan in another scene"; oth["clearance_bin"] = oth.min_clearance_m.map(cbin)
    SG = pd.concat([two, oth]); cols = ["set", "scene", "version", "replan_idx", "t_after_handover_s", "proposal", "leave", "prompt", "box_free", "box_free_margin0p3", "min_clearance_m", "min_clearance_label", "clearance_bin", "end_x_fwd_m", "end_y_left_m", "arc_2s_m", "arc_4s_m", "score_plain", "score_final", "rank_final", "chosen", "ego_speed_mps", "human_speed_mps"]
    SG[cols].to_csv(H / "scorer_geometry.csv", index=False, float_format="%.5g")
    sp = []
    for (st, s, v), g in SG.groupby(["set", "scene", "version"], sort=False):
        for sub, gg in (("all proposals", g), ("leave+prompt only", g[g.leave & g.prompt]), ("leave only", g[g.leave])):
            if len(gg) < 10: continue
            row = dict(set=st, scene=s, version=v, subset=sub, n=len(gg))
            for nm, c in (("end_lateral_left_m", "end_y_left_m"), ("min_clearance_m", "min_clearance_m"), ("arc_4s_m", "arc_4s_m")):
                x = gg[c].replace(np.inf, 1e3); row[f"spearman_final_vs_{nm}"] = float(spearmanr(x, gg.score_final)[0]); row[f"spearman_plain_vs_{nm}"] = float(spearmanr(x, gg.score_plain)[0])
            sp.append(row)
    SP = pd.DataFrame(sp); SP.to_csv(H / "scorer_geometry_spearman.csv", index=False, float_format="%.3f")
    lp = two[two.leave & two.prompt]; cb = []
    for (s, v), g in lp.groupby(["scene", "version"], sort=False):
        for _, _, nm in BINS:
            x = g[g.clearance_bin == nm]; cb.append(dict(scene=s, version=v, clearance_bin_m=nm, n_leave_prompt=len(x), share_of_leave_prompt=len(x) / len(g), median_score_final=x.score_final.median(), q25_score_final=x.score_final.quantile(.25), q75_score_final=x.score_final.quantile(.75), median_score_plain=x.score_plain.median(), median_rank_final=x.rank_final.median(), best_score_final=x.score_final.max()))
    CB = pd.DataFrame(cb); CB.to_csv(H / "clearance_bins.csv", index=False, float_format="%.4g")
    stay = two[~two.leave].groupby(["scene", "version"]).agg(n_stay=("score_final", "size"), median_stay_score_final=("score_final", "median")).reset_index(); chm = two[two.chosen].groupby(["scene", "version"]).score_final.median().rename("median_chosen_score_final").reset_index()
    # probe
    PR = pd.DataFrame(PJ["rows"]); cl = {(p["scene"], p["t_after_handover_s"], k): c for p in PP for k, c in p["classes"].items()}; meta = {(p["scene"], p["t_after_handover_s"]): p for p in PP}
    def cls(r):
        c = cl.get((r.scene, r.t_after_handover_s, r.path))
        if c is None: return pd.Series(dict(path_class="", path_min_clearance_m=np.nan, path_end_x_fwd_m=np.nan, path_end_y_left_m=np.nan, path_arc_4s_m=np.nan, path_first_overlap_label=""))
        return pd.Series(dict(path_class=("leave" if c["leave"] else "stay") + ("/prompt" if c["prompt"] else "/not-prompt") + ("/box-free(0.3)" if c["box_free_margin0p3"] else "/box-free(0)" if c["box_free"] else "/overlap"), path_min_clearance_m=c["min_clearance_m"], path_end_x_fwd_m=c["end_x_fwd_m"], path_end_y_left_m=c["end_y_left_m"], path_arc_4s_m=c["arc_4s_m"], path_first_overlap_label=c["first_overlap_label"]))
    PR = pd.concat([PR, PR.apply(cls, axis=1)], axis=1); PR["injection_mode"] = PR.pop("mode"); PR.to_csv(H / "scorer_probe.csv", index=False, float_format="%.5g")
    I = PR[PR.injection_mode != "none"]; ctl = PR[PR.injection_mode == "none"]
    summ = I[I.path != "self_own_waypoints"].groupby(["scene", "t_after_handover_s", "path", "path_class", "path_min_clearance_m", "injection_mode"], sort=False).agg(plain_min=("score_plain", "min"), plain_median=("score_plain", "median"), plain_max=("score_plain", "max"), final_min=("score_final", "min"), final_median=("score_final", "median"), final_max=("score_final", "max")).reset_index()
    # ---------------- plots
    # 1 grid heat-map
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, gd in zip(axs, R.GUARDS):
        g = G[(G.clock == MAINCK) & (G.guard == gd)]; M = g.pivot(index="T_s", columns="d", values="n_scenes_first_firing"); im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=8, aspect="auto"); ax.grid(False)
        for i, Tm in enumerate(M.index):
            for j, d in enumerate(M.columns):
                r = g[(g.T_s == Tm) & (g.d == d)].iloc[0]; ax.text(j, i, f"{r.n_scenes_first_firing} fired\n{r.n_wanted_go} W / {r.n_premature_go} P / {r.n_unsafe_go} U", ha="center", va="center", fontsize=8, color="white" if r.n_scenes_first_firing >= 6 else INK)
        ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns); ax.set_yticks(range(len(M.index))); ax.set_yticklabels([f"{t:.0f}" for t in M.index]); ax.set_xlabel("score margin d (final-score units)"); ax.set_title(f"guard: {gd}")
    axs[0].set_ylabel("standing time T (s)"); fig.colorbar(im, ax=axs, label="scenes with a first firing (of 30)", shrink=0.9)
    fig.suptitle("Plot 1 - Anti-stall rule R(T, d, guard) on the recorded 0.1.6 menus: scenes fired, split wanted (W) / premature (P) / unsafe (U) go", y=1.03); fig.savefig(H / "plot1_grid_heatmap.png", bbox_inches="tight"); plt.close(fig)
    # 2 b44a4a44 top view at first firing of R(3, 0.3, G_box0)
    r = FF[(FF.clock == MAINCK) & (FF.T_s == 3) & (FF.d == 0.3) & (FF.guard == "G_box0") & (FF.scene == "b44a4a44")].iloc[0]; s = "b44a4a44"; o = T.load("0.1.6", s); ho = o["ho"]; tn = int(ho + r.first_firing_t_s * 1e6); ego = o["egot"][tn][1:4]; Mn = T.menus("0.1.6", s)[tn]
    fig, ax = plt.subplots(figsize=(6.5, 8.5)); boxes(ax, o, tn, ego, fc="#d9d8d3", ec=MUTED, lw=1); boxes(ax, o, tn + 2_000_000, ego, fc="none", ec=MUTED, lw=1, ls=":"); footprint(ax, o, (0, 0, 0), fc="none", ec=INK, lw=1.5)
    p9 = F.plan9(Mn["proposals"][int(r.proposal)]); ax.plot(-p9[:, 1], p9[:, 0], "o-", color=BLUE, ms=5, label=f"switched-to plan (slot {int(r.proposal)}, final score {r.proposal_score_final:.2f}; 4 s)")
    st9 = F.plan9(Mn["proposals"][int(r.chosen_stay)]); ax.plot(-st9[:, 1], st9[:, 0], "s-", color=MAGENTA, ms=4, label=f"chosen stay (slot {int(r.chosen_stay)}, final score {r.chosen_stay_score_final:.2f})")
    hw = C.to_rig(T.gt_at(o, tn + np.arange(0, 8.01, 0.1) * 1e6), ego); ax.plot(-hw[:, 1], hw[:, 0], "-", color=ORANGE, label="human path, next 8 s (starts moving +2.98 s)")
    ck = RK["checks"][(s, tn, int(r.proposal))]; dw = C.to_rig(ck["driven_w"], ego); ax.plot(-dw[:, 1], dw[:, 0], "-", color=AQUA, lw=3, label=f"driven path, controller check 2 s ({r.moved_2s_m:.2f} m)"); footprint(ax, o, dw[-1], fc="none", ec=AQUA, lw=1.2, ls="--")
    topax(ax, f"Plot 2 - b44a4a44, first firing of R(3 s, 0.3, G_box0) at +{r.first_firing_t_s:.1f} s after handover\nactor boxes: filled = at the tick, dotted = 2 s later; dashed aqua = car footprint after 2 s", (-8, 8), (-6, 24)); ax.legend(loc="upper left", fontsize=7.5); fig.savefig(H / "plot2_b44a4a44_first_firing_topview.png", bbox_inches="tight"); plt.close(fig)
    # 3 timelines
    sc = list(E.scene.unique()); Fk = RK["firings"]; Fk = Fk[(Fk.clock == MAINCK) & (Fk.T_s == 3) & (Fk.d == 0.3)]; fig, axs = plt.subplots(3, 3, figsize=(14, 9)); axs = axs.ravel()
    for ax, s in zip(axs, sc):
        o = T.load("0.1.6", s); te, ve, _ = F.standing_since(o); ax.plot((te - o["ho"]) / 1e6, ve, color=BLUE, label="car speed (0.1.6 run)"); ax.plot(o["gt_t"], o["gt_v"], color=ORANGE, label="human speed (recording)"); top = max(ve.max(), o["gt_v"].max(), 1.0)
        for k, (gd, mk, col) in enumerate((("none", "v", VIOLET), ("G_box0", "o", AQUA), ("G_box03", "s", YELLOW))):
            x = Fk[(Fk.scene == s) & (Fk.guard == gd)].t_after_handover_s; ax.plot(x, np.full(len(x), -top * (0.08 + 0.07 * k)), mk, color=col, ms=5, ls="none", label=f"rule fires, guard {gd}")
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.axvline(0, color=MUTED, lw=0.8); ax.set_title(s); ax.set_xlabel("time after handover (s)"); ax.set_ylabel("speed (m/s)")
    for ax in axs[len(sc):]: ax.axis("off")
    axs[0].legend(fontsize=7); fig.suptitle("Plot 3 - Scenes with a stationary episode: car and human speed; markers below zero = replans at which R(3 s, 0.3, guard) fires on the recorded menus (dashed: 0.5 m/s)", y=1.0); fig.tight_layout(); fig.savefig(H / "plot3_timelines.png", bbox_inches="tight"); plt.close(fig)
    # 4, 5 probe
    roles = ["chosen_stay_slot", "base_escape_slot", "best_late_go_slot", "lowest_ranked_slot"]; rc = dict(zip(roles, (BLUE, ORANGE, AQUA, YELLOW)))
    for n, s in ((4, "d2715023"), (5, "b44a4a44")):
        ticks = [p["t_after_handover_s"] for p in PP if p["scene"] == s]; fig, axs = plt.subplots(len(ticks), 1, figsize=(13, 10), sharex=True)
        for ax, t in zip(axs, ticks):
            g = I[(I.scene == s) & (I.t_after_handover_s == t) & (I.injection_mode == "last") & (I.path != "self_own_waypoints")]; names = list(dict.fromkeys(g.path)); c0 = ctl[(ctl.scene == s) & (ctl.t_after_handover_s == t)].set_index("slot"); sl = meta[(s, t)]["slots"]
            for k, ro in enumerate(roles):
                x = g[g.slot_role == ro].set_index("path").loc[names]; ax.bar(np.arange(len(names)) + (k - 1.5) * 0.2, x.score_final, 0.18, color=rc[ro], label=f"injected into {ro.replace('_', ' ')} (slot {sl[ro]}; its own final score {c0.score_final[sl[ro]]:.2f})"); ax.plot(np.arange(len(names)) + (k - 1.5) * 0.2, x.score_plain, "_", color=INK, ms=9, mew=1.5, label="plain scorer output" if k == 0 else None)
            ax.axhline(c0.score_final[sl["chosen_stay_slot"]], color=BLUE, lw=1, ls="--"); ax.axhline(c0.score_final[sl["base_escape_slot"]], color=ORANGE, lw=1, ls="--"); ax.set_ylim(0, 1.3); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("final score (0-1)"); ax.legend(fontsize=6.5, ncol=3, loc="upper center")
            ax.set_title(f"+{t:.1f} s after handover (dashed: logged final score of the chosen stay, blue, and of the base escape, orange; base = {meta[(s, t)]['base_kind']})"); ax.set_xticks(range(len(names)))
            ax.set_xticklabels([f"{nm.replace('_', ' ').replace('base escape ', 'escape, ').replace(' from tick', '').replace(' transplanted', ' at car pose')}\n{g[g.path == nm].path_class.iloc[0].replace('/', ' / ', 1).replace('/box', '\\nbox').replace('/overlap', '\\noverlap')}\nclearance {g[g.path == nm].path_min_clearance_m.iloc[0]:.2f} m" for nm in names], fontsize=6.5)
        axs[-1].set_xlabel("injected path (class and minimum clearance to recorded actor boxes over 4 s)"); fig.suptitle(f"Plot {n} - {s}: score of a path injected into one slot's waypoints in the last refinement pass (bars = final score, black ticks = plain scorer output)", y=1.0); fig.tight_layout(); fig.savefig(H / f"plot{n}_probe_{s}.png", bbox_inches="tight"); plt.close(fig)
    # 6 clearance bins
    fig, ax = plt.subplots(figsize=(10, 4.5)); keys = [("d2715023", "0.1.6"), ("d2715023", "0.1.4b"), ("b44a4a44", "0.1.6"), ("b44a4a44", "0.1.4b")]
    for k, ((s, v), col) in enumerate(zip(keys, (BLUE, ORANGE, AQUA, YELLOW))):
        x = CB[(CB.scene == s) & (CB.version == v)].set_index("clearance_bin_m").loc[[b[2] for b in BINS]]; xs = np.arange(5) + (k - 1.5) * 0.2; ax.bar(xs, x.median_score_final, 0.18, color=col, label=f"{s} {v}")
        for xi, (mv, nn) in zip(xs, zip(x.median_score_final, x.n_leave_prompt)): ax.text(xi, (0 if pd.isna(mv) else mv) + 0.01, f"n={nn}", ha="center", fontsize=6.5, color="#52514e", rotation=90, va="bottom")
    for (s, v), col in zip(keys, (BLUE, ORANGE, AQUA, YELLOW)): ax.axhline(float(chm[(chm.scene == s) & (chm.version == v)].median_chosen_score_final.iloc[0]), color=col, lw=1, ls="--")
    ax.set_xticks(range(5)); ax.set_xticklabels([b[2] for b in BINS]); ax.set_xlabel("minimum clearance of the proposal to any recorded actor box over 4 s (m)"); ax.set_ylabel("median final score of leave+prompt proposals (0-1)"); ax.set_ylim(0, 1.1); ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("Plot 6 - Score of leave+prompt proposals by clearance bin, all post-handover replans (dashed: median final score of the chosen stay, same colour)"); fig.savefig(H / "plot6_clearance_bins.png", bbox_inches="tight"); plt.close(fig)
    # 7 d2715023 top view at the human-start tick with the injected paths
    s = "d2715023"; t = 1.0; p = meta[(s, t)]; o = T.load("0.1.6", s); tn = p["tn"]; ego = o["egot"][tn][1:4]; fig, ax = plt.subplots(figsize=(7.5, 8.5)); boxes(ax, o, tn, ego, fc="#d9d8d3", ec=MUTED, lw=1); footprint(ax, o, (0, 0, 0), fc="none", ec=INK, lw=1.5)
    g = I[(I.scene == s) & (I.t_after_handover_s == t) & (I.injection_mode == "last")]; show = [("i_human_future_from_tick", ORANGE, "o"), ("iii_base_escape_shift_0.0m", BLUE, "s"), ("iii_base_escape_shift_1.0m", AQUA, "^"), ("iii_base_escape_shift_2.0m", VIOLET, "D"), ("v_chosen_stay_path", MAGENTA, "v")]
    for nm, col, mk in show:
        q = F.plan9(np.array(p["paths"][nm])); x = g[g.path == nm].set_index("slot_role").score_final; ax.plot(-q[:, 1], q[:, 0], marker=mk, color=col, ms=5, label=f"{nm.replace('_', ' ')}: final score {x['chosen_stay_slot']:.2f} in the stay slot, {x['base_escape_slot']:.2f} in the base-escape slot, {x['best_late_go_slot']:.2f} late-go slot, {x['lowest_ranked_slot']:.2f} lowest slot")
    topax(ax, f"Plot 7 - d2715023 at the human-start tick (+1.0 s): injected paths and their scores (last-pass injection)\nlogged final score: chosen stay {ctl[(ctl.scene == s) & (ctl.t_after_handover_s == t)].set_index('slot').score_final[p['slots']['chosen_stay_slot']]:.2f}, base escape {ctl[(ctl.scene == s) & (ctl.t_after_handover_s == t)].set_index('slot').score_final[p['slots']['base_escape_slot']]:.2f}", (-9, 7), (-5, 20)); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.33), fontsize=7); fig.savefig(H / "plot7_d2715023_injected_topview.png", bbox_inches="tight"); plt.close(fig)
    # 8 score vs clearance scatter (case b view), two scenes x two versions
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (s, v) in zip(axs.ravel(), keys):
        g = two[(two.scene == s) & (two.version == v)]; x = g.min_clearance_m.clip(upper=3.0)
        for m, col, lb in ((~g.leave, MAGENTA, "stay"), (g.leave & ~g.prompt, YELLOW, "leave, late go"), (g.leave & g.prompt, BLUE, "leave + prompt")): ax.plot(x[m], g.score_final[m], "o", ms=3, alpha=0.5, color=col, ls="none", label=f"{lb} (n={int(m.sum())})")
        rr = SP[(SP.scene == s) & (SP.version == v) & (SP.subset == "leave+prompt only")]; ax.set_title(f"{s} {v} - Spearman(final, clearance) on leave+prompt: {rr.spearman_final_vs_min_clearance_m.iloc[0]:.2f}"); ax.legend(fontsize=7)
    for ax in axs[-1]: ax.set_xlabel("minimum clearance to any recorded actor box over 4 s (m, clipped at 3)")
    for ax in axs[:, 0]: ax.set_ylabel("final score (0-1)")
    fig.suptitle("Plot 8 - Final score against clearance, every proposal of every post-handover replan", y=1.0); fig.tight_layout(); fig.savefig(H / "plot8_score_vs_clearance.png", bbox_inches="tight"); plt.close(fig)
    # ---------------- README
    g3 = G[(G.T_s == 3) & (G.d == 0.3)]; W = []; A = W.append
    A("# T05 - anti-stall rule replayed on the recorded menus, and what the retrained scorer says about a pull-out\n")
    A("Offline, CPU only (`CUDA_VISIBLE_DEVICES=\"\"`). Nothing under `alpasim/` or `adapters/` was edited. Scripts (this directory): `t05_common.py` (imports `part4_why_it_stays_stopped/t04_common.py`; adds clearance and the standing clock), `t05_menu.py` (per-proposal table, 30 scenes), `t05_rule.py` (Part 1), `t05_probe_inputs.py` / `t05_probe_paths.py` / `t05_probe.py` (Part 2), `t05_report.py` (tables, plots, this file - generated). Measured facts only.\n")
    A("## 0. Definitions as applied\n")
    A("- leave / prompt / box-free / usable escape: as in `part4_why_it_stays_stopped/README.md` (leave = end point >= 2.0 m; prompt = arc >= 0.5 m by 1 s and >= 1.5 m by 2 s; box-free = ego footprint at 0.1 s steps over 4 s never overlaps a recorded actor box at the same time stamp; margin 0.3 m = ego box grown 0.3 m on every side). Recomputed here for all 30 scenes; on the 8704 rows shared with `part4_why_it_stays_stopped/proposals.csv` leave, prompt, both box checks and the chosen flag are identical (0 differences).")
    A("- **Actors are a playback** and do not react. **The box guard uses the recorded actor boxes, i.e. information a real system would have to get from perception; it does not come from the scorer.** The actor log ends at +16.8 s: for replans after +12.8 s the box check covers less than 4 s (`plan_box_checked_horizon_s`), and a controller check started after +14.8 s is box-checked for less than 2 s.")
    A("- **Rule R(T, d, guard)**: at a replan, if the car has been below 0.5 m/s for >= T seconds and the chosen plan is a stay (end point < 2.0 m), take the leave+prompt proposal (passing the guard) with the highest final score (`pdm_score`); switch if its final score >= chosen score - d.")
    A(f"- **Standing clock.** Main (`incl_pre_handover`): continuous time below 0.5 m/s in the controller-state rows of the run, which start 3.0 s before handover (there the state is the simulator-reported replay), so at most 3.0 s of pre-handover standing is credited. Variant (`from_handover`): the same, capped at the time since handover. Both are in the CSVs; tables below are the main clock unless stated.")
    A("- **Open loop**: after a first firing the real car would move and later menus would differ. Per scene the FIRST firing is reported; `n_replans_rule_fires_recorded_menus` counts the replans at which the rule fires on the recorded (still-standing) menus and is indicative only.")
    A("- **Controller check**: standalone AlpaSim controller + vehicle model (`t02_common.Sim`), teacher-forced from the session start to the firing tick, then the switched-to plan held 2.0 s (`t03_common.branch`).")
    A("- **Labels** (by rule): `unsafe go` = the plan overlaps a recorded actor box at 0 m margin within the checked horizon, or the driven footprint overlaps one within 2 s; else `wanted go` = human above 0.5 m/s at that time or exceeding it within 2 s; else `premature go` (human standing for more than 2 s more, or never moving again in the clip).")
    A("- **Clearance** = minimum distance between the ego box and any recorded actor box over the 41 samples (0 = overlap).\n")
    A("## Part 1 - anti-stall rule\n### 1.1 Stationary episodes in the 30 runs (0.1.6)\nEvery run of >= 1.0 s below 0.5 m/s that extends past handover (start can be negative = already standing in the replayed part). 21 of 30 scenes have none. `ahead_*` = nearest actor box ahead inside the ego-width corridor, gap from the front bumper, at the episode start / middle; `episode_class` by rule from the human's path length during the episode.\n")
    A(md(E[["scene", "start_s", "end_s", "post_handover_duration_s", "runs_to_end_of_clip", "car_path_m", "human_path_m", "human_speed_at_start_mps", "human_starts_moving_after_s", "human_share_below_0p5", "human_max_speed_mps", "n_replans", "n_replans_chosen_stay", "ahead_gap_at_start_m", "ahead_label_at_start", "ahead_gap_mid_m", "ahead_label_mid", "nearest_box_mid_m", "nearest_label_mid", "episode_class"]]))
    ec = E.episode_class.value_counts().to_dict(); MET["stationary_episodes"] = dict(n=len(E), n_scenes=int(E.scene.nunique()), by_class=ec)
    A(f"\nCounts: {len(E)} episodes in {E.scene.nunique()} scenes; by class: {ec}. Replans with the car standing >= 3 s and a chosen stay (the rule's eligible set at T = 3): {int(g3.n_eligible_replans_all30.iloc[0])} in all 30 scenes, of which 68 on the two stuck scenes.\n")
    A("### 1.2 Grid (main clock): scenes with a first firing, split wanted / premature / unsafe\nCell = `fired: W/P/U`.\n")
    for gd in R.GUARDS:
        g = G[(G.clock == MAINCK) & (G.guard == gd)].copy(); g["cell"] = g.n_scenes_first_firing.astype(str) + ": " + g.n_wanted_go.astype(str) + "/" + g.n_premature_go.astype(str) + "/" + g.n_unsafe_go.astype(str); pv = g.pivot(index="T_s", columns="d", values="cell").reset_index(); pv.columns = ["T_s \\ d"] + [f"d={c}" for c in pv.columns[1:]]; A(f"**guard {gd}**\n\n" + md(pv, 0) + "\n")
    A("First-firing time (s after handover) of the two stuck scenes (blank = never fires):\n")
    for s in F.STUCK:
        for ck in R.CLOCKS:
            g = G[G.clock == ck].copy(); g["row"] = g.guard + ", T=" + g.T_s.map("{:.0f}".format); pv = g.pivot(index="row", columns="d", values=f"{s}_first_firing_t_s").reset_index(); pv.columns = ["guard, T"] + [f"d={c}" for c in pv.columns[1:]]; A(f"**{s}, clock {ck}**\n\n" + md(pv, 1) + "\n")
    A("Scene ids per cell and the `from_handover` clock: `antistall_grid.csv`. Same grid, `from_handover` clock:\n")
    for gd in R.GUARDS:
        g = G[(G.clock == "from_handover") & (G.guard == gd)].copy(); g["cell"] = g.n_scenes_first_firing.astype(str) + ": " + g.n_wanted_go.astype(str) + "/" + g.n_premature_go.astype(str) + "/" + g.n_unsafe_go.astype(str); pv = g.pivot(index="T_s", columns="d", values="cell").reset_index(); pv.columns = ["T_s \\ d"] + [f"d={c}" for c in pv.columns[1:]]; A(f"**guard {gd}** (from_handover)\n\n" + md(pv, 0) + "\n")
    A("### 1.3 Replans where the car and the human both stand (other 28 scenes)\nEligible = car standing >= T, chosen plan a stay, human below 0.5 m/s at that time, scene not one of the two stuck ones. The count does not depend on the clock or the guard in this data.\n")
    g = G[(G.clock == MAINCK) & (G.guard == "G_box03")][["T_s", "d", "other28_human_standing_eligible_replans", "other28_human_standing_firing_replans", "other28_human_standing_firing_share", "other28_human_standing_firings_plan_overlaps_0m", "other28_human_standing_firings_plan_overlaps_0p3m", "other28_human_standing_firing_scenes"]]; A(md(g))
    A("\n### 1.4 Per-scene tables for selected settings\nShown: R(3, 0.3) with each guard under both clocks (requested), and R(5, 0.3, G_box0) and R(3, 0.1, G_box0) because in the grid they are the cells with zero unsafe and zero premature first firings at the two ends of the T and d axes. This is a selection for display, not a recommendation.\n")
    showc = ["scene", "first_firing_t_s", "n_replans_rule_fires_recorded_menus", "proposal", "proposal_score_final", "proposal_rank_final", "chosen_stay_score_final", "standing_s", "human_state", "human_speed_mps", "human_starts_moving_in_s", "plan_box_free", "plan_first_overlap_s", "plan_first_overlap_label", "plan_box_free_margin0p3", "plan_first_overlap_margin0p3_s", "plan_first_overlap_margin0p3_label", "plan_min_clearance_m", "plan_box_checked_horizon_s", "moved_1s_m", "moved_2s_m", "peak_abs_steer_rad", "driven_overlap_2s", "driven_min_clearance_2s_m", "label"]
    for ck, Tm, d, gd in [(MAINCK, 3, 0.3, "none"), (MAINCK, 3, 0.3, "G_box0"), (MAINCK, 3, 0.3, "G_box03"), ("from_handover", 3, 0.3, "none"), ("from_handover", 3, 0.3, "G_box0"), ("from_handover", 3, 0.3, "G_box03"), (MAINCK, 5, 0.3, "G_box0"), (MAINCK, 3, 0.1, "G_box0")]:
        f = FF[(FF.clock == ck) & (FF.T_s == Tm) & (FF.d == d) & (FF.guard == gd)]; A(f"**R(T={Tm} s, d={d}, {gd}), clock {ck}** - {len(f)} scenes fire; the other {30 - len(f)} never do.\n\n" + md(f[showc]) + "\n")
    trunc = FF[(FF.plan_box_checked_horizon_s < 4.0)].drop_duplicates(["scene", "first_firing_t_s"])[["scene", "first_firing_t_s", "plan_box_checked_horizon_s"]]
    A("First firings whose plan box check covers less than 4 s because the actor log ends (their label rests on the shorter check):\n\n" + md(trunc) + "\n")
    # ---------------- Part 2 text
    A("## Part 2 - the retrained scorer and a pull-out\n### 2.1 What the scorer takes as input (code)\n")
    MP_ = "adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based"
    arch = [f"The score head reads no waypoints. `Scorer.forward(proposals, bev_feature)` uses `proposals` only for its shape (`{MP_}/score_module/scorer.py:37-39`); the per-proposal input is `bev_feature.reshape(B, 32, 8, -1).amax(-2)` - the element-wise maximum over the 8 poses of a 256-d feature per (slot, pose) - fed to one MLP with 6 outputs (`scorer.py:41-42`); the score is the sigmoid of the last output (`drive_jepa_model.py:85`).",
            f"Those features are the generator's own query features: `bev_feature = hist_encoding(ego_status) + init_feature.weight` - one learned embedding per (slot, pose), 32 x 8 (`drive_jepa_model.py:33, 61-62`) - passed 4 times (`drive_jepa_config.py:14` `ref_num=4`) through ONE shared `Traj_refiner` (`drive_jepa_model.py:35-36, 65-67`).",
            f"Each pass decodes waypoints from the features with an MLP (`traj_refiner.py:24`) and then refines the features with `Bev_refiner(proposals, bev_feature, image_feature)` (`traj_refiner.py:26-27`), where the just-decoded waypoints enter only as detached reference points (`bevformer/bev_refiner.py:112-114`): position `(xy + 32) / 64` for the self-attention and the four corners of the ego footprint at that pose, projected into the camera, as the sampling locations of the image cross-attention (`bevformer/encoder.py:234-253, 273-274`). The returned menu is the waypoints decoded in the 4th pass (`drive_jepa_model.py:69`); the scorer gets the features after the 4th refinement (`drive_jepa_model.py:71-73`).",
            "So: (1) the scorer consumes a per-slot learned feature from the generator, with image features sampled at the proposal's footprint; (2) it has no interface for a trajectory the generator did not produce; (3) a foreign path can only be put in as the reference points of a refinement pass, while the slot's query feature still comes from the slot's own earlier passes (and that feature is what the waypoint MLP decodes the slot's own path from). Such a score is a hybrid, not the scorer's opinion of the path alone. Neither case (a) nor case (b) of the task holds cleanly; both sets of outputs were produced.",
            "Final score = `(14 * S + 2 * S_c) / 16` with `S_c` in {0, 1} a two-frame comfort term computed from the proposal's simulated states against the previous plan (`drive_jepa_model.py:98-151`, formula at `:149`; adapter side `adapters/drivejepa016/momentum.py:234-296`)."]
    for a in arch: A("- " + a)
    MET["architecture_answer"] = arch
    gate = pd.DataFrame(PJ["gate"]); pin = json.load(open(H / "cache" / "probe_inputs.json"))
    A("\n### 2.2 Forward-pass gate\n" + f"Step 1 (`t05_probe_inputs.py`, method of `s9_trial_inputs.py`: 0.1.6 planner, bridge worker on CPU, replans in time order from the session start): picks equal the logged ones on {sum(p['replans_run'] for p in pin)} of {sum(p['replans_run'] for p in pin)} replans run (through +5.0 s; {sum(p['pick_mismatches'] for p in pin)} mismatches); worst waypoint difference against the stored S5 menus {max(p['worst_plan_dev_vs_trial_m'] for p in pin):.1e} m. Step 2 (`t05_probe.py`, `MomentumDriveJEPA` loaded directly in the worker interpreter, same weights, with an instance-level `Traj_refiner.forward` and no replacement): against the values saved in step 1 -\n")
    A(md(gate, 6)); MET["forward_pass_gate"] = dict(step1_replans=int(sum(p["replans_run"] for p in pin)), step1_pick_mismatches=int(sum(p["pick_mismatches"] for p in pin)), step1_worst_waypoint_diff_vs_s5_menu_m=float(max(p["worst_plan_dev_vs_trial_m"] for p in pin)), step2_max_abs_waypoint_diff_m=float(gate.max_abs_all_diff.max()), step2_max_abs_plain_score_diff=float(gate.max_abs_plain_score_diff.max()), step2_max_abs_final_score_diff=float(gate.max_abs_final_score_diff.max()), step2_picks_equal=bool((gate.pick == gate.logged_pick).all()), n_ticks=len(gate))
    so = I[I.path == "self_own_waypoints"]; e_last = float((so[so.injection_mode == "last"].score_plain - so[so.injection_mode == "last"].slot_orig_plain).abs().max()); e_all = float((so[so.injection_mode == "all"].score_plain - so[so.injection_mode == "all"].slot_orig_plain).abs().max())
    A(f"\nInjecting a slot's own final waypoints: last-pass mode changes its plain score by at most {e_last:.1e}; all-pass mode by up to {e_all:.2f} (the slot's waypoints of passes 1-3 differ from its final ones, so this mode is not an identity). Largest change of any OTHER slot's plain score caused by one injection: last-pass {I[I.injection_mode == 'last'].max_other_slot_plain_change.max():.1e}, all-pass {I[I.injection_mode == 'all'].max_other_slot_plain_change.max():.2f}.\n")
    A("### 2.3 Injection probe (hybrid; see 2.1)\nOne path replaces the waypoints of ONE slot per forward pass. Mode `last`: in the 4th refinement pass only. Mode `all`: in all 4 passes. Four slots per tick: the chosen stay's slot, the base escape's slot, the best-scored late-go slot, the lowest-scored slot. Paths: (i) the human's own future from the tick in the car's frame (8 poses, 0.5 s, recorded heading relative to the car's); (ii) the human's pull-out from min(tick, human-start tick) placed at the car's pose; (iii) the base escape (best usable escape of that menu, or the best leave+prompt where the menu has no usable one) with its end shifted 0 / 0.5 / 1.0 / 1.5 / 2.0 m to the left, scaled by arc-length fraction; (iv) all poses at the origin; (v) the chosen stay's own waypoints. `control_no_replacement` rows in `scorer_probe.csv` are the unmodified menu. At +5 s the human is already "
      + ", ".join(f"{p['human_offset_from_car_m']:.1f} m ({p['scene']})" for p in PP if p["t_after_handover_s"] == 5.0) + " away from the standing car, so path (i) starts that far ahead.\n")
    for s in F.STUCK:
        for p in [p for p in PP if p["scene"] == s]:
            t = p["t_after_handover_s"]; c0 = ctl[(ctl.scene == s) & (ctl.t_after_handover_s == t)].set_index("slot"); g = I[(I.scene == s) & (I.t_after_handover_s == t) & (I.path != "self_own_waypoints")]
            tab = g[g.injection_mode == "last"].pivot_table(index=["path", "path_class", "path_min_clearance_m"], columns="slot_role", values="score_final", sort=False)[roles]; tab.columns = [f"final, {c.replace('_', ' ')} ({p['slots'][c]})" for c in tab.columns]
            tp = g[g.injection_mode == "last"].pivot_table(index=["path", "path_class", "path_min_clearance_m"], columns="slot_role", values="score_plain", sort=False)[roles]; tab["plain, min..max over the 4 slots"] = [f"{a:.2f}..{b:.2f}" for a, b in zip(tp.min(1), tp.max(1))]
            ta = g[g.injection_mode == "all"].pivot_table(index=["path", "path_class", "path_min_clearance_m"], columns="slot_role", values="score_final", sort=False)[roles]; tab["final, all-pass mode, min..max"] = [f"{a:.2f}..{b:.2f}" for a, b in zip(ta.min(1), ta.max(1))]
            A(f"**{s}, +{t:.1f} s** - logged final score of the slots: " + ", ".join(f"{ro.replace('_', ' ')} {p['slots'][ro]}: {c0.score_final[p['slots'][ro]]:.2f} (plain {c0.score_plain[p['slots'][ro]]:.2f})" for ro in roles) + f"; base = {p['base_kind']}. Last-pass mode:\n\n" + md(tab.reset_index()) + "\n")
    L = I[(I.injection_mode == "last") & (I.path != "self_own_waypoints")]; rs = L.groupby(["scene", "t_after_handover_s", "path"]).score_plain.agg(lambda x: x.max() - x.min()).mean(); rp = L.groupby(["scene", "t_after_handover_s", "slot_role"]).score_plain.agg(lambda x: x.max() - x.min()).mean()
    A(f"Slot dependence (last-pass mode, plain score): mean range over the 4 slots for one and the same injected path {rs:.2f}; mean range over the 9 paths inside one slot {rp:.2f}. Spearman between the injected score and the slot's own original score {spearmanr(L.score_plain, L.slot_orig_plain)[0]:.2f} (n = {len(L)}).\n")
    MET["probe_slot_dependence"] = dict(mean_range_over_slots_same_path=float(rs), mean_range_over_paths_same_slot=float(rp), spearman_injected_vs_slot_original=float(spearmanr(L.score_plain, L.slot_orig_plain)[0]))
    key = []
    for p in PP:
        s, t = p["scene"], p["t_after_handover_s"]; c0 = ctl[(ctl.scene == s) & (ctl.t_after_handover_s == t)].set_index("slot"); g = L[(L.scene == s) & (L.t_after_handover_s == t)]; row = dict(scene=s, t_after_handover_s=t, logged_chosen_stay_final=float(c0.score_final[p["slots"]["chosen_stay_slot"]]), logged_base_escape_final=float(c0.score_final[p["slots"]["base_escape_slot"]]))
        for nm, k in (("i_human_future_from_tick", "human_future"), ("ii_human_pullout_transplanted", "human_pullout_transplanted"), ("iv_stay_zeros", "stay_zeros"), ("iii_base_escape_shift_2.0m", "base_shift_2m")):
            x = g[g.path == nm]; row[f"{k}_final_min"] = float(x.score_final.min()); row[f"{k}_final_max"] = float(x.score_final.max()); row[f"{k}_final_in_stay_slot"] = float(x[x.slot_role == "chosen_stay_slot"].score_final.iloc[0]); row[f"{k}_final_in_base_slot"] = float(x[x.slot_role == "base_escape_slot"].score_final.iloc[0]); row[f"{k}_class"] = x.path_class.iloc[0]
        key.append(row)
    KEY = pd.DataFrame(key); KEY.to_csv(H / "scorer_probe_key.csv", index=False, float_format="%.4g"); MET["probe_key_last_pass"] = key
    A("Key numbers (last-pass mode, final score; min and max over the 4 slots): `scorer_probe_key.csv`.\n\n" + md(KEY[["scene", "t_after_handover_s", "logged_chosen_stay_final", "logged_base_escape_final", "human_future_class", "human_future_final_min", "human_future_final_max", "human_pullout_transplanted_class", "human_pullout_transplanted_final_min", "human_pullout_transplanted_final_max", "stay_zeros_final_min", "stay_zeros_final_max", "base_shift_2m_class", "base_shift_2m_final_min", "base_shift_2m_final_max"]]) + "\n")
    A("### 2.4 Score against the proposals' own geometry (no injection)\nAll 32 proposals of all 34 post-handover replans, both scenes, 0.1.6 and 0.1.4b (`scorer_geometry.csv`); Spearman rank correlations (`scorer_geometry_spearman.csv`):\n\n" + md(SP) + "\n")
    if len(CP): A(f"Comparable replans in the other 28 scenes (0.1.6; car below 0.5 m/s, human above 0.5 m/s or exceeding it within 2 s, an actor box within 4 m ahead in the ego-width corridor): {len(CP)} replans in {CP.scene.nunique()} scene(s):\n\n" + md(CP.drop(columns='tn')) + "\n")
    else: A("Comparable replans in the other 28 scenes (car below 0.5 m/s, human moving or starting within 2 s, an actor box within 4 m ahead in the corridor): none.\n")
    MET["comparable_replans_other28"] = dict(n=len(CP), scenes=sorted(CP.scene.unique()) if len(CP) else [])
    A("### 2.5 Score of leave+prompt proposals by clearance bin\n" + md(CB) + "\n\nFor reference, stays on the same replans:\n\n" + md(stay.merge(chm)) + "\n")
    MET["clearance_bin_median_final"] = {f"{r.scene}|{r.version}|{r.clearance_bin_m}": dict(n=int(r.n_leave_prompt), median_final=None if pd.isna(r.median_score_final) else round(float(r.median_score_final), 3)) for r in CB.itertuples()}
    A("## Plots\n" + "\n".join(f"- `{f}`" for f in ("plot1_grid_heatmap.png", "plot2_b44a4a44_first_firing_topview.png", "plot3_timelines.png", "plot4_probe_d2715023.png", "plot5_probe_b44a4a44.png", "plot6_clearance_bins.png", "plot7_d2715023_injected_topview.png", "plot8_score_vs_clearance.png")))
    A("\nColours: the dataviz reference palette in its fixed order; its validator script could not be run here (Node 18 rejects the file as a CommonJS script and prints nothing when it is loaded as a module).\n\n## Files\n`antistall_grid.csv`, `antistall_first_firings.csv`, `stationary_episodes.csv`, `scorer_probe.csv`, `scorer_probe_key.csv`, `scorer_geometry.csv`, `scorer_geometry_spearman.csv`, `clearance_bins.csv`. Scratch (model inputs of the 6 probe ticks, camera tensors, 66 MB, not deleted): `{runs}/drivejepa_retrain/t05_scratch_20261009/`.")
    open(H / "README.md", "w").write("\n".join(W) + "\n")
    # grid metrics
    MET["grid_main_clock"] = {f"T{r.T_s:.0f}_d{r.d}_{r.guard}": dict(fired=int(r.n_scenes_first_firing), wanted=int(r.n_wanted_go), premature=int(r.n_premature_go), unsafe=int(r.n_unsafe_go)) for r in G[G.clock == MAINCK].itertuples()}
    MET["grid_from_handover_clock"] = {f"T{r.T_s:.0f}_d{r.d}_{r.guard}": dict(fired=int(r.n_scenes_first_firing), wanted=int(r.n_wanted_go), premature=int(r.n_premature_go), unsafe=int(r.n_unsafe_go)) for r in G[G.clock == "from_handover"].itertuples()}
    f = FF[(FF.T_s == 3) & (FF.d == 0.3) & FF.scene.isin(F.STUCK)]; MET["stuck_scenes_R3_0p3"] = [dict(clock=r.clock, guard=r.guard, scene=r.scene, first_firing_t_s=r.first_firing_t_s, proposal=int(r.proposal), score_final=round(r.proposal_score_final, 3), rank=int(r.proposal_rank_final), stay_score=round(r.chosen_stay_score_final, 3), human_speed_mps=round(r.human_speed_mps, 2), human_starts_in_s=round(r.human_starts_moving_in_s, 2), plan_box_free_0=bool(r.plan_box_free), plan_box_free_0p3=bool(r.plan_box_free_margin0p3), plan_first_overlap_s=None if pd.isna(r.plan_first_overlap_s) else r.plan_first_overlap_s, plan_first_overlap_label=None if pd.isna(r.plan_first_overlap_label) else r.plan_first_overlap_label, moved_1s_m=round(r.moved_1s_m, 2), moved_2s_m=round(r.moved_2s_m, 2), peak_steer_rad=round(r.peak_abs_steer_rad, 2), driven_overlap_2s=bool(r.driven_overlap_2s), label=r.label) for r in f.itertuples()]
    MET["stuck_scenes_not_firing_R3_0p3"] = [dict(clock=r.clock, guard=r.guard, scene=s) for r in g3.itertuples() for s in F.STUCK if pd.isna(getattr(r, f"_{list(g3.columns).index(s + '_first_firing_t_s') + 1}", np.nan)) and pd.isna(r._asdict().get(s + "_first_firing_t_s", np.nan)) and pd.isna(g3.loc[r.Index, s + "_first_firing_t_s"])]
    u = FF.drop_duplicates(["scene", "first_firing_t_s", "proposal", "label"]); MET["premature_first_firings_distinct"] = [f"{r.scene} +{r.first_firing_t_s}s slot {int(r.proposal)}" for r in u[u.label == "premature go"].itertuples()]; MET["unsafe_first_firings_distinct"] = [f"{r.scene} +{r.first_firing_t_s}s slot {int(r.proposal)} ({'plan overlaps ' + str(r.plan_first_overlap_label) + ' at ' + str(r.plan_first_overlap_s) + ' s' if not r.plan_box_free else 'driven overlap ' + str(r.driven_overlap_label)})" for r in u[u.label == "unsafe go"].itertuples()]
    MET["d2715023_fires_by_d_main_clock"] = {f"T{r.T_s:.0f}_d{r.d}_{r.guard}": (None if pd.isna(r.d2715023_first_firing_t_s) else r.d2715023_first_firing_t_s) for r in G[G.clock == MAINCK].itertuples()}
    o28 = G[(G.clock == MAINCK) & (G.guard == "G_box03")]; MET["other28_human_standing"] = {f"T{r.T_s:.0f}_d{r.d}": dict(eligible=int(r.other28_human_standing_eligible_replans), fires=int(r.other28_human_standing_firing_replans), plan_overlaps_0m=int(r.other28_human_standing_firings_plan_overlaps_0m)) for r in o28.itertuples()}
    MET["spearman"] = SP.round(3).to_dict("records"); MET["truncated_box_horizon_first_firings"] = trunc.to_dict("records")
    json.dump(MET, open(H / "cache" / "metrics.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print("ok", len(CP), CB.round(2).to_string()); print(SP.round(2).to_string()); print(KEY.round(2).T.to_string())


if __name__ == "__main__":
    main()
