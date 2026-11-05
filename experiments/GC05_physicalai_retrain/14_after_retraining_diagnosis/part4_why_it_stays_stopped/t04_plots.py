#!/usr/bin/env python
"""T04 plots 1-8.  CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python t04_plots.py"""
import math, pickle
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MP
from matplotlib.collections import LineCollection
import t04_common as T
from t04_menu import poly
C = T.C
P = pd.read_csv(T.HERE / "proposals.csv"); R = pd.read_csv(T.HERE / "replans.csv"); VC = {"0.1.6": "#1f77b4", "0.1.4b": "#d95f02"}


def topview(s, ver, fn, num):
    o = T.load(ver, s); ho = o["ho"]; hs = T.human_start(o); rel = math.ceil(hs / 0.5 - 1e-9) * 0.5; ts = int(ho + rel * 1e6); z = T.menus(ver, s)[ts]; ego = o["egot"][ts][1:4]
    p = P[(P.scene == s) & (P.version == ver) & (np.abs(P.t_after_handover_s - rel) < 1e-6)].set_index("proposal"); fig, ax = plt.subplots(figsize=(13, 7)); i0 = int(np.argmin(np.abs(o["ap_t"] - ts)))
    for j, a in enumerate(o["ids"]):
        if not np.isfinite(o["A"][i0, j, 0]): continue
        c = C.to_rig(np.c_[np.array(poly(o["A"][i0, j, :2].astype(float), float(o["A"][i0, j, 2]), o["dims"][j]).exterior.coords)[:4], np.zeros(4)], ego)[:, :2]
        if np.abs(c).max() > 40: continue
        tr = o["A"][:, j]; ok = np.isfinite(tr[:, 0]) & (o["ap_t"] >= ts) & (o["ap_t"] <= ts + 4e6); mv = ok.sum() > 1 and np.hypot(*(tr[ok][-1, :2] - tr[ok][0, :2])) > 0.5
        ax.add_patch(MP(c, closed=True, fc="#f4a582" if mv else "#bbbbbb", ec="k", lw=0.8, zorder=2))
        if mv: q = C.to_rig(tr[ok].astype(float), ego); ax.plot(q[:, 0], q[:, 1], color="#b2182b", lw=1, ls=":", zorder=2)
    ax.add_patch(MP(np.array([[o["rig_to_aabb_x"] + sx * o["ego_dims"][0] / 2, sy * o["ego_dims"][1] / 2] for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]), closed=True, fc="none", ec="blue", lw=2, zorder=5, label="ego footprint at this tick"))
    g = o["gt"]; hw = C.to_rig(g[g[:, 0] >= ho - 1][:, 1:4], ego); ax.plot(hw[:, 0], hw[:, 1], color="k", lw=1, ls="--", zorder=3, label="human path (handover to end of clip)")
    h4 = C.to_rig(T.gt_at(o, ts + T.TS * 1e6), ego); ax.plot(h4[:, 0], h4[:, 1], color="k", lw=3, zorder=4, label="human's next 4 s from this tick")
    cm = plt.cm.viridis
    for i in np.argsort(z["pdm_score"]):
        q = np.r_[np.zeros((1, 3)), z["proposals"][i]]; u = bool(p.usable[i]); ax.plot(q[:, 0], q[:, 1], color=cm(float(z["pdm_score"][i])), lw=4 if u else 1.3, marker="o", ms=2.5, zorder=7 if u else 6, solid_capstyle="round")
    ch = int(z["chosen"]); q = np.r_[np.zeros((1, 3)), z["proposals"][ch]]; ax.plot(q[:, 0], q[:, 1], color="red", lw=1.5, ls="--", zorder=8); ax.plot(q[-1, 0], q[-1, 1], marker="*", ms=20, mfc="red", mec="k", zorder=9, ls="none", label=f"chosen plan end (proposal {ch}, final score {z['pdm_score'][ch]:.2f})")
    ax.plot([], [], color=cm(0.5), lw=4, label=f"usable escape (leave + prompt + box-free): {int(p.usable.sum())} of 32"); ax.plot([], [], color=cm(0.5), lw=1.3, label="other proposals")
    ax.add_patch(MP(np.zeros((1, 2)), fc="#f4a582", ec="k", label="actor that moves > 0.5 m in the next 4 s (dotted: its track)")); ax.add_patch(MP(np.zeros((1, 2)), fc="#bbbbbb", ec="k", label="actor box at this tick (static)"))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)); fig.colorbar(sm, ax=ax, label="final score of the proposal (0-1)", shrink=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-6, 26); ax.set_ylim(-8, 10); ax.set_xlabel("forward of the car at this tick (m)"); ax.set_ylabel("left of the car at this tick (m)"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"Plot {num}. {s}, adapter {ver}: the 32-proposal menu at +{rel:.1f} s after handover (the tick the human starts moving)\nleave {int(p.leave.sum())}, prompt leave {int((p.leave & p.prompt).sum())}, late-go {int(p.late_go.sum())}, stay {int((~p.leave).sum())}; ego speed {np.hypot(*o['egot'][ts][4:6]):.2f} m/s")
    fig.tight_layout(); fig.savefig(T.HERE / fn, dpi=110); plt.close(fig)


n = 1
for ver in T.VERS:
    for s in T.STUCK: topview(s, ver, f"plot{n}_topview_{s}_{ver.replace('.', '')}.png", n); n += 1

# 5: scores against time
fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, s in zip(axs, T.STUCK):
    for ver in T.VERS:
        r = R[(R.scene == s) & (R.version == ver)]; c = VC[ver]
        ax.plot(r.t_after_handover_s, r.chosen_score_final, color=c, lw=2, label=f"{ver}: chosen plan"); ax.plot(r.t_after_handover_s, r.best_usable_score_final, color=c, lw=0, marker="o", ms=6, label=f"{ver}: best usable escape (where one exists)")
        ax.plot(r.t_after_handover_s, r.best_leave_score_final, color=c, lw=1.2, ls="--", label=f"{ver}: best leave proposal of any kind")
    hs = T.human_start(T.load("0.1.6", s)); ax.axvline(hs, color="k", ls=":"); ax.text(hs + 0.1, 0.02, f"human starts moving\n(+{hs:.2f} s)", fontsize=9)
    ax.set_title(f"{s}"); ax.set_xlabel("time after handover (s)"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
axs[0].set_ylabel("final score (0-1)"); axs[1].legend(fontsize=8, loc="center right"); fig.suptitle("Plot 5. Stuck scenes: final score of the chosen plan, of the best usable escape and of the best leave proposal, per replan, both adapters")
fig.tight_layout(); fig.savefig(T.HERE / "plot5_scores_vs_time.png", dpi=110); plt.close(fig)

# 6: class counts
fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, s in zip(axs, T.STUCK):
    p = P[(P.scene == s) & (P.version == "0.1.6")]; g = p.groupby("t_after_handover_s")
    cls = [("stay", g.apply(lambda d: (~d.leave).sum()), "#999999"), ("late-go leave (not prompt)", g.apply(lambda d: d.late_go.sum()), "#fdae61"), ("prompt leave, overlaps an actor box", g.apply(lambda d: (d.leave & d.prompt & ~d.box_free).sum()), "#d7191c"), ("usable escape (leave + prompt + box-free)", g.apply(lambda d: d.usable.sum()), "#1a9641")]
    b = np.zeros(len(cls[0][1]))
    for nm, v, c in cls: ax.bar(v.index, v.values, 0.42, bottom=b, color=c, label=nm); b += v.values
    hs = T.human_start(T.load("0.1.6", s)); ax.axvline(hs, color="k", ls=":"); ax.set_title(f"{s} (dotted: human starts moving, +{hs:.2f} s)"); ax.set_xlabel("time after handover (s)"); ax.set_ylim(0, 32)
axs[0].set_ylabel("number of proposals (of 32)"); axs[1].legend(fontsize=9, loc="upper right"); fig.suptitle("Plot 6. Adapter 0.1.6, stuck scenes: number of proposals per class at every replan"); fig.tight_layout(); fig.savefig(T.HERE / "plot6_class_counts_016.png", dpi=110); plt.close(fig)

# 7: controller check
ser = pickle.load(open(T.HERE / "cache" / "controller_series.pkl", "rb")); CC = pd.read_csv(T.HERE / "controller_check.csv"); fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
col = {"i_chosen": "red", "ii_best_usable": "#1a9641", "iii_best_leave": "#fdae61", "iv_best_late_go": "#7b3294", "v_human_path": "k", "vi_best_prompt_leave": "#2b83ba"}
for ax, s in zip(axs, T.STUCK):
    for (ss, tick, nm), (t, d) in ser.items():
        if ss != s or tick != "human_start": continue
        r = CC[(CC.scene == s) & (CC.tick == tick) & (CC.plan == nm)].iloc[0]; ax.plot(t, d, color=col[nm], lw=2, ls="--" if nm in ("iv_best_late_go", "vi_best_prompt_leave") else "-", label=f"{nm} ({r.plan_class}, final rank {r.plan_rank_final:.0f})" if nm != "v_human_path" else "v_human_path")
    ax.axhline(1.0, color="grey", ls=":"); ax.set_title(f"{s}, tick +{CC[(CC.scene == s) & (CC.tick == 'human_start')].t_after_handover_s.iloc[0]:.1f} s"); ax.set_xlabel("time since the plan was handed to the controller (s)"); ax.set_ylabel("distance of the car from its starting point (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Plot 7. Controller check (0.1.6, standalone controller + vehicle model, one plan held 2 s) at the tick the human starts moving"); fig.tight_layout(); fig.savefig(T.HERE / "plot7_controller_check.png", dpi=110); plt.close(fig)

# 8: moving standstill scenes
fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, s in zip(axs, T.MOVE):
    for ver in T.VERS:
        r = R[(R.scene == s) & (R.version == ver)]; c = VC[ver]; ax.plot(r.t_after_handover_s, r.chosen_score_final, color=c, lw=2, label=f"{ver}: chosen plan"); ax.plot(r.t_after_handover_s, r.best_usable_score_final, color=c, lw=0, marker="o", ms=6, label=f"{ver}: best usable escape (where one exists)")
        st = r[r.ego_speed_mps < 0.5]; ax.plot(st.t_after_handover_s, np.full(len(st), 0.03 if ver == "0.1.6" else 0.06), color=c, lw=0, marker="s", ms=5, label=f"{ver}: replans with the car below 0.5 m/s")
    hs = T.human_start(T.load("0.1.6", s)); ax.axvline(hs, color="k", ls=":"); ax.set_title(f"{s} (dotted: human starts moving, +{hs:.2f} s)"); ax.set_xlabel("time after handover (s)"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
axs[0].set_ylabel("final score (0-1)"); axs[1].legend(fontsize=8, loc="lower right"); fig.suptitle("Plot 8. The two standstill scenes that do move: final score of the chosen plan and of the best usable escape, per replan, both adapters"); fig.tight_layout(); fig.savefig(T.HERE / "plot8_moving_scenes_scores.png", dpi=110); plt.close(fig)
print("ok")
