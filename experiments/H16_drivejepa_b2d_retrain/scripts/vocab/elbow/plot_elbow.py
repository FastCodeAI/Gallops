"""Six small charts of the elbow runs (runs/*.json), mean of 3 seeds. python3 plot_elbow.py <out.png>"""
import collections, glob, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R = "{ops}/b2d_drivejepa/vocab/elbow/runs/"
runs = collections.defaultdict(list)
for f in glob.glob(R + "*.json"):
    d = json.load(open(f)); runs[d["K"]].append(d)
Ks = [k for k in sorted(runs) if k >= 512]
S0 = 20261105
g = lambda f: np.array([np.mean([f(d) for d in runs[k]]) for k in Ks])
t0 = lambda f: np.array([f([d for d in runs[k] if d["seed"] == S0][0]) for k in Ks])
INK, MUTED, GRID, S1, S2, SURF = "#0b0b0b", "#52514e", "#e7e6e1", "#2a78d6", "#eb6834", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.edgecolor": "#bdbcb5",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
x = np.log2(Ks)
def base(a, title, ylabel):
    a.set_title(title)
    a.set_ylabel(ylabel)
    a.set_xticks(x, [f"{k:,}" if k in (512, 1024, 2048, 4096, 8192) else "" for k in Ks])
    a.set_xlabel("vocabulary size")
    a.axvline(10, color="#9b9a93", lw=1, ls=(0, (4, 3)), zorder=0)
def line(a, y, c, label=None, fmt="{:.2f}", ends=(0, -1)):
    a.plot(x, y, color=c, marker="o", ms=5.5, mec=SURF, mew=1.2, label=label)
    for i in ends:
        a.annotate(fmt.format(y[i]), (x[i], y[i]), xytext=(0, 8), textcoords="offset points", ha="center", color=INK, fontsize=9)
ho, tr = g(lambda d: d["heldout_moving"]["mean"]), t0(lambda d: d["train_moving"]["mean"])
a = ax[0, 0]; base(a, "Coverage: training and held-out split apart after 1,024", "ADE to nearest entry, moving paths (m)")
line(a, ho, S1, "held-out", ends=(-1,)); line(a, tr, S2, "training", ends=(-1,)); a.set_ylim(0, .56); a.legend(loc="lower left")
a = ax[0, 1]; base(a, "No sharp elbow: each doubling gains a little less", "held-out ADE saved per doubling (m)")
kk = np.array(Ks, float); gain = (ho[:-1] - ho[1:]) / np.log2(kk[1:] / kk[:-1]); xm = (x[:-1] + x[1:]) / 2
a.plot(xm, gain, color=S1, marker="o", ms=5.5, mec=SURF, mew=1.2); a.set_ylim(0, .08)
for i in (0, len(gain) - 1):
    a.annotate(f"{gain[i]:.3f}", (xm[i], gain[i]), xytext=(0, 8), textcoords="offset points", ha="center", color=INK, fontsize=9)
a = ax[0, 2]; base(a, "Memorising: entries built from a single clip", "share of entries (%)")
line(a, 100 * g(lambda d: d["entries_one_clip_share"]), S1, fmt="{:.1f}%", ends=(0, 2 * 2, 6, -1)); a.set_ylim(0, 60)
a = ax[1, 0]; base(a, "Speed or shape? Both errors fall together", "held-out error at the nearest entry (m)")
line(a, g(lambda d: d["heldout_err_along_m"]), S1, "along the path (speed)", ends=(0, -1)); line(a, g(lambda d: d["heldout_err_across_m"]), S2, "across the path (shape)", ends=(0, -1)); a.set_ylim(0, .46); a.legend(loc="upper right")
a = ax[1, 1]; base(a, "Teacher pool: near-copies per expert path", "entries within 1 m ADE, median")
line(a, g(lambda d: d["teachers_within_1m_median"]), S1, fmt="{:.0f}", ends=(0, 4, 6, -1)); a.axhline(4, color=MUTED, lw=1); a.text(x[-1], 5, "4 teachers sampled per round", ha="right", color=MUTED, fontsize=9); a.set_ylim(0, 60)
a = ax[1, 2]; base(a, "Speed slicing: gap between straight entries", "median gap in mean speed (m/s)")
line(a, g(lambda d: d["straight_speed_gap_median"]), S1, fmt="{:.3f}", ends=(0, 2 * 2, -1)); a.set_ylim(0, .062)
fig.suptitle("Bench2Drive vocabulary size: 36,897 training paths, 47 held-out clips, mean of 3 seeds. Dashed line: 1,024", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(sys.argv[1], dpi=110)
print("wrote", sys.argv[1], Ks)
