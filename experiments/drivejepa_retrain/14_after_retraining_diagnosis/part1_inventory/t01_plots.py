#!/usr/bin/env python
"""T01 plots, from honest_table.csv and series.pkl (both written by t01_build.py).
  cd paddock-sid/Alpasim && alpasim/.venv/bin/python "analysis_3.0/14_after_retraining_diagnosis/part1_inventory/t01_plots.py" """
import pathlib, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
h = pd.read_csv(HERE / "honest_table.csv", dtype={"scene": str}); ser = pickle.load(open(HERE / "series.pkl", "rb"))
B, A = "0.1.4b", "0.1.6"; CB, CA, CH = "#8a8a8a", "#1f6fb4", "#111111"
D = {v: h[h.version == v].set_index("scene") for v in h.version.unique()}; order = D[A].sort_values("handover_speed_mps").index; x = np.arange(len(order))
lab = [f"{s} ({D[A].handover_speed_mps[s]:.1f})" for s in order]

# 1 -- progress before/after with scored-window fraction
fig, ax = plt.subplots(figsize=(17, 6.5))
for off, v, c in ((-0.2, B, CB), (0.2, A, CA)):
    ax.bar(x + off, D[v].progress_rel_to_total[order], 0.38, color=c, label=f"{v}: progress_rel_to_total at end of scored window")
    ax.scatter(x + off, D[v].progress_rel_to_total_full_clip[order], marker="_", s=120, color="k", zorder=3, label=("same metric at end of clip (unscored part included)" if v == A else None))
    for i, s in enumerate(order): ax.text(i + off, D[v].progress_rel_to_total[s] + 0.015, f"{D[v].scored_window_fraction[s]:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90, color=c)
ax.scatter(x, D[A].progress_rel_to_total_at_handover[order], marker="v", s=22, color="#c0392b", zorder=4, label="progress already credited at handover (3 s of replayed human driving)")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7, rotation=60); ax.set_ylim(0, 1.25); ax.set_ylabel("progress_rel_to_total (fraction of the human's whole-clip distance)")
ax.set_xlabel("scene, sorted by speed at handover (m/s, in brackets)"); ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.set_title("Drive-JEPA 0.1.4b (before) vs 0.1.6 (after retrain): progress per scene. Number above each bar = scored-window fraction (scored ticks / ticks after handover)")
fig.tight_layout(); fig.savefig(HERE / "plot1_progress_before_after.png", dpi=110); plt.close(fig)

# 2 -- distance travelled after handover
fig, ax = plt.subplots(figsize=(17, 6))
ax.bar(x - 0.27, D[B].dist_traveled_m[order], 0.27, color=CB, label="0.1.4b (before)"); ax.bar(x, D[A].dist_traveled_m[order], 0.27, color=CA, label="0.1.6 (after)")
ax.bar(x + 0.27, D[A].human_dist_traveled_m[order], 0.27, color=CH, label="human (recorded drive)")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7, rotation=60); ax.set_ylabel("distance travelled from handover to end of clip (m)"); ax.set_xlabel("scene, sorted by speed at handover (m/s, in brackets)")
ax.set_title("Distance travelled after handover, whole clip (about 16.8 s), per scene: before, after, human"); ax.legend(); fig.tight_layout(); fig.savefig(HERE / "plot2_distance_before_after_human.png", dpi=110); plt.close(fig)

# 3 -- highway scenes, first 5 s
fig, axs = plt.subplots(3, 2, figsize=(15, 9.5), sharex=True)
for j, s in enumerate(("04749bb9", "acd22430")):
    z = ser[(A, s)]; t, V, st, men = z["t"], z["V"], z["steer"], [m for m in z["menus"] if -0.01 <= m["t"] <= 5.01]; k = (t >= 0) & (t <= 5); ks = (st[:, 0] >= 0) & (st[:, 0] <= 5)
    off = z["tw"][z["offroad"] & (z["tw"] >= 0)]; t_off = off.min() if len(off) else None
    hk = (z["gt_t"] >= 0) & (z["gt_t"] <= 5.2); hv = np.gradient(z["gt_d"], z["gt_t"])
    axs[0, j].plot(t[k], V[k], color=CA, label="car speed (0.1.6)"); axs[0, j].plot(z["gt_t"][hk], hv[hk], color=CH, ls="--", label="human speed"); axs[0, j].set_ylabel("speed (m/s)"); axs[0, j].set_ylim(0, 27)
    axs[1, j].plot(st[ks, 0], st[ks, 1], color=CA, label="front steering angle"); axs[1, j].axhline(0.3, color="#c0392b", lw=0.8, ls=":"); axs[1, j].axhline(-0.3, color="#c0392b", lw=0.8, ls=":", label="+/- 0.3 rad rule"); axs[1, j].set_ylabel("steering angle (rad)"); axs[1, j].set_ylim(-0.85, 0.85)
    axs[2, j].step([m["t"] for m in men], [m["acc"] for m in men], where="post", color=CA, marker="o", label="driven plan: demanded accel over its first 0.5 s"); axs[2, j].plot(st[ks, 0], st[ks, 3], color="#888", lw=0.9, label="acceleration reported by the controller")
    axs[2, j].axhline(0, color="k", lw=0.5); axs[2, j].set_ylabel("longitudinal acceleration (m/s^2)"); axs[2, j].set_xlabel("time after handover (s)")
    for a in axs[:, j]:
        for m in men: a.axvline(m["t"], color="#bbb", lw=0.6, zorder=0)
        if t_off is not None and t_off <= 5: a.axvline(t_off, color="#c0392b", lw=1.4)
        a.legend(fontsize=7, loc="lower left")
    axs[0, j].set_title(f"{s} under 0.1.6, first 5 s after handover (handover {D[A].handover_speed_mps[s]:.1f} m/s)\ngrey lines = replans (every 0.5 s); red line = first off-road tick" + (f" (+{t_off:.2f} s)" if t_off is not None else ""), fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "plot3_highway_first5s_016.png", dpi=110); plt.close(fig)

# 4 -- pull-outs, whole clip
fig, axs = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True)
for j, s in enumerate(("d2715023", "b44a4a44")):
    hv = None
    for v, c in ((B, CB), (A, CA)):
        z = ser[(v, s)]; k = z["t"] >= 0; kw = z["tw"] >= 0
        axs[0, j].plot(z["t"][k], z["V"][k], color=c, label=f"car speed ({v})"); axs[1, j].plot(z["tw"][kw], z["dist"][kw], color=c, label=f"car distance ({v})")
        col = z["tw"][z["coll"] & kw]
        if len(col) and v == A:
            for a in axs[:, j]: a.axvline(col.min(), color="#c0392b", lw=1.2)
    hk = z["gt_t"] >= 0; axs[0, j].plot(z["gt_t"][hk], np.gradient(z["gt_d"], z["gt_t"])[hk], color=CH, ls="--", label="human speed"); axs[1, j].plot(z["gt_t"][hk], z["gt_d"][hk], color=CH, ls="--", label="human distance")
    axs[0, j].set_ylabel("speed (m/s)"); axs[1, j].set_ylabel("distance travelled since handover (m)"); axs[1, j].set_xlabel("time after handover (s)"); axs[0, j].legend(fontsize=8); axs[1, j].legend(fontsize=8)
    axs[0, j].set_title(f"{s}: whole clip after handover (handover {D[A].handover_speed_mps[s]:.2f} m/s). 0.1.6 travels {D[A].dist_traveled_m[s]:.2f} m, human {D[A].human_dist_traveled_m[s]:.1f} m\nred line = first collision tick under 0.1.6" , fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "plot4_pullouts_whole_clip.png", dpi=110); plt.close(fig)
print("ok")
