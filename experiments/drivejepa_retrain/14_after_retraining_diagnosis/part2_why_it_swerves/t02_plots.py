#!/usr/bin/env python
"""T02 plots (reads gate_series.pkl, run_series.pkl and the CSVs written by t02_gate.py / t02_run.py).
  {runs}/alpasim_bulk/nurec-venv/bin/python t02_plots.py"""
import pathlib, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parent
G = pickle.load(open(H / "gate_series.pkl", "rb")); S = pickle.load(open(H / "run_series.pkl", "rb"))
cf = pd.read_csv(H / "counterfactuals.csv"); sw = pd.read_csv(H / "braking_sweep.csv"); an = pd.read_csv(H / "plan_anatomy.csv"); mp = pd.read_csv(H / "menu_proposals.csv")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})
HW = ["04749bb9", "acd22430"]

# 1 gate
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5)); ax = ax.ravel()
for a, ((scene, rel), d) in zip(ax, G.items()):
    a.plot(d["t"], d["rec_steer"], "k-", lw=3, alpha=0.35, label="recorded run (controller CSV)")
    a.plot(d["t"], d["warm_tf"]["steer"], "C0-", lw=1.2, label="standalone, replayed from session start")
    a.plot(d["t"], d["cold_tf"]["steer"], "C1--", lw=1.2, label="standalone, fresh controller at start tick")
    a.plot(d["t"], d["cold_free"]["steer"], "C2:", lw=1.4, label="standalone, fresh + own pose (free-running)")
    a.axhline(0.3, color="r", lw=0.6); a.axhline(-0.3, color="r", lw=0.6)
    a.set_title(f"{scene} 0.1.6, start = handover + {rel:.1f} s" + ("  (negative control)" if scene == "02e075b9" else "")); a.set_xlabel("time after start tick (s)"); a.set_ylabel("front steering angle (rad)")
    if np.abs(d["rec_steer"]).max() < 0.05: a.set_ylim(-0.35, 0.35)
ax[0].legend(fontsize=7, loc="lower left"); ax[5].axis("off"); ax[5].text(0.02, 0.5, "red lines: |steer| = 0.3 rad\nplans fed: the recorded plan of every 50 ms tick", fontsize=9)
fig.suptitle("Plot 1 - Gate: standalone controller replay vs recorded steering (3 s, recorded plans fed at recorded times)"); fig.tight_layout(); fig.savefig(H / "plot1_gate.png"); plt.close(fig)

# 2, 3 counterfactual steering at handover
for n, scene in ((2, HW[0]), (3, HW[1])):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5)); names = [v for v in cf[(cf.version == "0.1.6") & (cf.scene == scene) & (cf.tick_after_handover_s == 0)].variant if "coldseed" not in v]
    for i, v in enumerate(names):
        d = S[("0.1.6", scene, 0.0, v)]; pk = np.abs(d["steer"]).max(); st = dict(color=f"C{i}", lw=2.4 if v.startswith("B0") else 1.3, ls="-" if i < 10 else "--")
        ax[0].plot(d["t"], d["steer"], label=f"{v}  (peak {pk:.3f} rad)", **st); ax[1].plot(d["t"], d["ulong"], label=v, **st)
    ax[0].axhline(0.3, color="r", lw=0.6); ax[0].axhline(-0.3, color="r", lw=0.6); ax[0].set_xlabel("time after handover (s), plan held fixed"); ax[0].set_ylabel("front steering angle (rad)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("time after handover (s), plan held fixed"); ax[1].set_ylabel("commanded longitudinal acceleration (m/s^2)")
    ax[0].set_title("steering"); ax[1].set_title("longitudinal command")
    fig.suptitle(f"Plot {n} - {scene} (0.1.6), handover tick: counterfactual plans B0-B7 through the standalone controller, 1.0 s"); fig.tight_layout(); fig.savefig(H / f"plot{n}_counterfactuals_{scene}.png"); plt.close(fig)

# 4 braking sweep
fig, ax = plt.subplots(1, 3, figsize=(21, 6.2)); XC = ("const_decel_mps2", "first_0p5s_accel_mps2", "accel_equiv_to_1s_mps2")
for i, ((s, pk), g) in enumerate(sw.groupby(["seed_scene", "path"], sort=False)):
    g = g.sort_values("const_decel_mps2"); lab = f"sweep at {g.ego_speed_mps.iloc[0]:.1f} m/s ({s}, {pk.replace('_', ' ')})"; st = dict(marker="o", ms=3, ls="--" if pk == "recorded_path" else "-", color=f"C{i}")
    for a, xc in zip(ax, XC): a.plot(g[xc], g.peak_abs_steer_rad, label=lab, **st)
for (ver, mk) in (("0.1.6", "*"), ("0.1.4b", "X")):
    for s, col in zip(HW, ("k", "m")):
        r = cf[(cf.version == ver) & (cf.scene == s) & (cf.tick_after_handover_s == 0) & (cf.variant == "B0_recorded")].iloc[0]
        for a, xc in zip(ax[1:], XC[1:]): a.scatter(r[xc], r.peak_abs_steer_rad, marker=mk, s=130, color=col, zorder=5, label=f"recorded handover plan {ver} {s}")
        ax[0].axvline(-r.first_0p5s_accel_mps2, color=col, lw=1.0, ls=":" if ver == "0.1.6" else "-.", label=f"first-0.5 s demand of recorded plan {ver} {s}: {r.first_0p5s_accel_mps2:.1f} m/s^2")
for s, col in zip(HW, ("k", "m")):
    r = cf[(cf.version == "0.1.6") & (cf.scene == s) & (cf.tick_after_handover_s == 0) & (cf.variant == "B5_first_step_respaced")].iloc[0]
    for a, xc in zip(ax[1:], XC[1:]): a.scatter(r[xc], r.peak_abs_steer_rad, marker="s", s=70, facecolors="none", edgecolors=col, zorder=5, label=f"B5 first step re-spaced, 0.1.6 {s}")
for a in ax: a.axhline(0.1, color="r", lw=0.6); a.set_ylabel("peak |front steering angle| in 1.0 s (rad)")
ax[0].set_xlabel("constant deceleration of the plan (m/s^2)"); ax[0].set_title("vs the plan's constant deceleration")
ax[1].set_xlabel("first-0.5 s demanded acceleration, (step-1 speed - current speed)/0.5 s (m/s^2)"); ax[1].set_title("same runs vs first-0.5 s demand (= -a/2 for constant a)")
ax[2].set_xlabel("constant acceleration that reaches the plan's 1.0 s position, 2 (s(1 s) - v0) (m/s^2)"); ax[2].set_title("same runs vs position shortfall at 1.0 s (= -a for constant a)")
ax[0].legend(fontsize=6.5, loc="center right"); ax[1].legend(fontsize=6.5, loc="center left"); ax[1].set_xlim(-18.5, 0.5); ax[2].set_xlim(-18.5, 0.5)
fig.suptitle("Plot 4 - Braking sweep: peak steering vs demanded deceleration, one line per seed speed (red line: 0.1 rad onset threshold)"); fig.tight_layout(); fig.savefig(H / "plot4_braking_sweep.png"); plt.close(fig)

# 5 anatomy
fig, ax = plt.subplots(2, 2, figsize=(14, 8)); a0 = an[an.source == "handover_replan"]
for i, ver in enumerate(("0.1.6", "0.1.4b")):
    for j, s in enumerate(HW):
        g = a0[(a0.version == ver) & (a0.scene == s)].sort_values("pose"); a = ax[i, j]; x = g.t_s.values - 0.25
        a.step(np.r_[0, g.t_s.values], np.r_[g.step_speed_human_mps.values[0], g.step_speed_human_mps.values], where="pre", color="C2", lw=2, label="human (clip ground truth), same 0.5 s steps")
        a.step(np.r_[0, g.t_s.values], np.r_[g.step_speed_ctrl_plan_mps.values[0], g.step_speed_ctrl_plan_mps.values], where="pre", color="C0", lw=2, label="plan the controller received at the handover tick (9 poses, 0.5 s apart)")
        a.plot(x, g.step_speed_raw_waypoints_mps, "kx", ms=8, label="raw 8 model waypoints (regenerated menu, chosen proposal)")
        c = an[(an.source == "cached_tick") & (an.version == ver) & (an.scene == s) & (an.pose == 1)]
        a.plot(c.tick_after_handover_s, c.step_speed_ctrl_plan_mps, "C3.-", lw=1, label="plans sent at +0.05 ... +0.45 s (20 Hz stream): car-to-first-pose distance / 0.5 s")
        a.axhline(g.ego_speed_mps.iloc[0], color="gray", lw=0.8, ls="--", label=f"car speed at handover {g.ego_speed_mps.iloc[0]:.1f} m/s")
        a.set_title(f"{ver}  {s}"); a.set_xlabel("time after handover (s)"); a.set_ylabel("speed over each 0.5 s step (m/s)"); a.set_ylim(0, 30)
ax[0, 0].legend(fontsize=6.5, loc="lower left")
fig.suptitle("Plot 5 - Handover plan anatomy: step speeds of the raw 8 waypoints, the plan the controller received, and the human"); fig.tight_layout(); fig.savefig(H / "plot5_plan_anatomy.png"); plt.close(fig)

# 6, 7 menu scatter at handover
for n, ver in ((6, "0.1.6"), (7, "0.1.4b")):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    for a, s in zip(ax, HW):
        b = mp[(mp.version == ver) & (mp.scene == s) & (mp.replan_after_handover_s == 0)]; lo, hi = -35.0, 12.0; x = b.first_0p5s_accel_mps2.clip(lo, hi); y = b.mean_speed_4s_mps.clip(0, 40)
        sc = a.scatter(x, y, c=b.scorer_rank, cmap="viridis_r", s=45, edgecolors="k", linewidths=0.4, vmin=1, vmax=32)
        c = b[b.chosen]; a.scatter(c.first_0p5s_accel_mps2, c.mean_speed_4s_mps, marker="*", s=420, facecolors="none", edgecolors="r", linewidths=1.8, label=f"chosen (rank {int(c.scorer_rank.iloc[0])})")
        k = b[b.clean_human_like]; a.scatter(k.first_0p5s_accel_mps2, k.mean_speed_4s_mps, marker="o", s=260, facecolors="none", edgecolors="C1", linewidths=2, label=f"clean human-like (n = {len(k)})")
        a.axhline(b.human_mean_speed_4s_mps.iloc[0], color="C2", lw=1.5, label=f"human mean speed over 4 s {b.human_mean_speed_4s_mps.iloc[0]:.1f} m/s")
        a.axvline(b.safe_threshold_first_0p5s_accel_mps2.iloc[0], color="gray", ls="--", lw=1, label=f"controller-safe threshold {b.safe_threshold_first_0p5s_accel_mps2.iloc[0]:.1f} m/s^2")
        nclip = int(((b.first_0p5s_accel_mps2 > hi) | (b.first_0p5s_accel_mps2 < lo) | (b.mean_speed_4s_mps > 40)).sum())
        a.set_title(f"{s}  ({nclip} of 32 proposals outside the axes, drawn at the edge)"); a.set_xlabel("first-0.5 s demanded acceleration (m/s^2)"); a.set_ylabel("mean speed over the 4 s plan (m/s)"); a.legend(fontsize=7, loc="lower right"); a.set_xlim(lo - 1, hi + 1); a.set_ylim(0, 41)
        fig.colorbar(sc, ax=a, label="scorer rank (1 = best)")
    fig.suptitle(f"Plot {n} - Menu at the handover replan, {ver}: 32 proposals, mean speed vs first-0.5 s demanded acceleration"); fig.tight_layout(); fig.savefig(H / f"plot{n}_menu_handover_{ver.replace('.', '')}.png"); plt.close(fig)

# 8 bd07fb91 top view
m = S[("menu", "0.1.6", "bd07fb91", 6.0)]; fig, a = plt.subplots(figsize=(14, 5.6)); P = m["P"]; order = np.argsort(m["S"])
for i in order:
    q = np.r_[np.zeros((1, 2)), P[i][:, :2]]; a.plot(q[:, 0], q[:, 1], "-", color=plt.cm.viridis((m["S"][i] - m["S"].min()) / (np.ptp(m["S"]) + 1e-9)), lw=1.2, marker=".", ms=3)
q = np.r_[np.zeros((1, 2)), P[m["chosen"]][:, :2]]; a.plot(q[:, 0], q[:, 1], "r-o", lw=2.6, ms=5, label=f"chosen proposal {m['chosen']} (score {m['S'][m['chosen']]:.3f}, rank 1)")
hd = m["human_dense"]; t = np.arange(-3.0, 10.0, 0.05); k = (t >= 0) & (t <= 4.0); a.plot(hd[k, 0], hd[k, 1], "k-", lw=3, label="human path, next 4 s (clip ground truth)"); kk = (t >= -3) & (t < 0); a.plot(hd[kk, 0], hd[kk, 1], "k:", lw=1.5, label="human path, previous 3 s")
a.plot(0, 0, "ks", ms=9, label=f"car at +6.0 s ({m['ego_v']:.1f} m/s; human mean over next 4 s {m['human_mean']:.1f} m/s)")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(m["S"].min(), m["S"].max())); fig.colorbar(sm, ax=a, label="scorer score")
a.set_xlabel("x forward (m)"); a.set_ylabel("y left (m)"); a.set_title("Plot 8 - bd07fb91 (0.1.6), replan at handover + 6.0 s: the 32 proposals, top view in the car frame (dots = 0.5 s waypoints)"); a.legend(fontsize=8, loc="lower left"); a.set_aspect("equal"); a.set_xlim(-45, 75)
fig.tight_layout(); fig.savefig(H / "plot8_bd07fb91_menu_topview.png"); plt.close(fig)
print("plots done")
