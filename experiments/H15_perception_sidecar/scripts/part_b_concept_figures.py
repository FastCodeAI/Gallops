"""Part B's concept figures that need geometry rather than boxes: B-1's veto, on an illustrative scene.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/part_b_concept_figures.py

Nothing here is data: one car crossing from the left and four hand-made proposals. The clearances are
computed by `geometry.terms` itself, the code B-1 would use, so the picture shows its definition exactly:
the gap between the ego's footprint and the agent's, each a circle of its box's area, at each 0.5 s of the
plan, the agent following its 1 s path and then constant velocity. Writes ../figures/b1-clearance.png.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("GALLOPS_CONFIG", str(REPO / "gallops.toml"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from gallops.drivejepa_perception import geometry, views  # noqa: E402

MARGIN_M, HORIZON_S = 1.0, 2.0
TIMES = geometry.TIMES.numpy()

# Rear-axle poses (x, y, yaw) at 0.5 s steps, and the release's score for each: made up, for the picture.
PROPOSALS = {
    "keep speed": ([(3.75 * k, 0.0, 0.0) for k in range(1, 9)], 0.91),
    "slow down": (list(zip([3.4, 6.2, 7.9, 8.8, 9.6, 10.6, 11.9, 13.4], [0.0] * 8, [0.0] * 8)), 0.85),  # noqa: B905
    "nudge left": (
        [(3.6, 0.2, 0.05), (7.1, 0.7, 0.12), (10.4, 1.5, 0.15), (13.6, 2.3, 0.12), (16.8, 2.8, 0.08), (20.0, 3.1, 0.05),
         (23.2, 3.3, 0.03), (26.4, 3.4, 0.0)],
        0.88,
    ),
    "turn right": (
        [(3.4, -0.2, -0.1), (6.6, -0.8, -0.25), (9.4, -1.9, -0.42), (11.7, -3.4, -0.6), (13.4, -5.2, -0.8),
         (14.6, -7.2, -0.98), (15.4, -9.3, -1.12), (15.9, -11.5, -1.25)],
        0.62,
    ),
}
COLOURS = {"keep speed": "#d62728", "slow down": "#1baf7a", "nudge left": "#9467bd", "turn right": "#8c6d31"}
# One car, 15 m ahead and 7 m to the left, crossing to the right at 4 m/s; its 1 s path is the side-car's
# path head's output, constant velocity after.
CAR = {"xy": (15.0, 7.0), "size": (4.5, 1.9), "vel": (0.0, -4.0), "path": ((0.0, -2.0), (0.0, -4.0))}


def clearances() -> dict:
    """Per proposal, its clearance to the car at each of the 8 plan times (geometry.terms' definition)."""
    names = list(PROPOSALS)
    candidates = torch.tensor([PROPOSALS[n][0] for n in names], dtype=torch.float32)[None]  # 1, K, 8, 3
    xy = torch.tensor([[CAR["xy"]]], dtype=torch.float32)
    size = torch.tensor([[CAR["size"]]], dtype=torch.float32)
    vel = torch.tensor([[CAR["vel"]]], dtype=torch.float32)
    path = torch.tensor([[CAR["path"]]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)
    features, pairwise, _ = geometry.terms(candidates, xy, size, vel, path, mask)
    per_time = (features[0, :, :8] * 10.0).numpy()  # the features keep per-time minima / 10
    return dict(zip(names, per_time))  # noqa: B905


def main() -> None:
    views.use_style()
    per_time = clearances()
    within = TIMES <= HORIZON_S
    vetoed = {n: bool((c[within] < MARGIN_M).any()) for n, c in per_time.items()}
    kept = [n for n in PROPOSALS if not vetoed[n]]
    release_pick = max(PROPOSALS, key=lambda n: PROPOSALS[n][1])
    new_pick = max(kept, key=lambda n: PROPOSALS[n][1]) if kept else release_pick
    car_track = geometry.agent_centres(
        torch.tensor([[CAR["xy"]]]), torch.tensor([[CAR["vel"]]]), torch.tensor([[CAR["path"]]])
    )[0, 0].numpy()
    r_ego = math.sqrt(geometry.EGO_LENGTH * geometry.EGO_WIDTH / math.pi)
    r_car = math.sqrt(CAR["size"][0] * CAR["size"][1] / math.pi)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) from above: x forward up, y left to the left.
    ego = np.array([[geometry.EGO_CENTRE_AHEAD + s * geometry.EGO_LENGTH / 2, t * geometry.EGO_WIDTH / 2]
                    for s, t in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    ax.add_patch(Polygon(np.c_[-ego[:, 1], ego[:, 0]], closed=True, fc="#6b7280", ec="none", alpha=0.8))
    ax.annotate("ego", (0.0, 1.5), ha="center", va="center", fontsize=8.5, color="white")
    for name, (poses, score) in PROPOSALS.items():
        p = np.asarray(poses)
        style = dict(color=COLOURS[name], lw=2.2 if name in (release_pick, new_pick) else 1.3,
                     ls="--" if vetoed[name] else "-")
        tag = f"{name} · the release's score {score:.2f} · " + ("vetoed" if vetoed[name] else "kept")
        ax.plot(-np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], label=tag, **style)
        ax.plot(-p[:, 1], p[:, 0], "o", ms=3, color=COLOURS[name])
    for k, (x, y) in enumerate(car_track):
        alpha = 0.85 if k < 2 else 0.35
        ax.add_patch(Circle((-y, x), r_car, fill=False, ec="#eb6834", lw=1.0, alpha=alpha, ls="-" if k < 2 else ":"))
    ax.plot(-car_track[:, 1], car_track[:, 0], "-", color="#eb6834", lw=1.0, alpha=0.6)
    ax.add_patch(Circle((-CAR["xy"][1], CAR["xy"][0]), r_car, fc="#eb6834", ec="none", alpha=0.35))
    ax.annotate("the car now,\ncrossing at 4 m/s:\nat 0.5 and 1.0 s\nfrom its path head,\nthen constant\nvelocity",
                (-CAR["xy"][1], CAR["xy"][0] + r_car), xytext=(-16.5, 29.5), textcoords="data", ha="left", va="top",
                fontsize=8, color="#eb6834", arrowprops=dict(arrowstyle="-", color="#eb6834", lw=0.8))
    # The closest approach of the release's pick, footprints drawn as the clearance sees them.
    worst = int(np.argmin(per_time[release_pick][within]))
    pose = np.asarray(PROPOSALS[release_pick][0])[worst]
    centre = pose[:2] + geometry.EGO_CENTRE_AHEAD * np.array([math.cos(pose[2]), math.sin(pose[2])])
    ax.add_patch(Circle((-centre[1], centre[0]), r_ego, fill=False, ec=COLOURS[release_pick], lw=1.4))
    ax.add_patch(Circle((-car_track[worst, 1], car_track[worst, 0]), r_car, fill=False, ec="#eb6834", lw=1.4))
    ax.annotate(f"at {TIMES[worst]:.1f} s the footprints overlap:\nclearance {per_time[release_pick][worst]:+.1f} m",
                (-centre[1] - r_ego, centre[0]), xytext=(40, -95), textcoords="offset points", fontsize=8.5,
                color=COLOURS[release_pick], arrowprops=dict(arrowstyle="-", color=COLOURS[release_pick], lw=0.8))
    ax.set_xlim(-17, 14)
    ax.set_ylim(-3, 31)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{-v + 0.0:g}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=7.5, frameon=False)
    ax.set_xlabel("y (m): + left, − right")
    ax.set_ylabel("x (m), forward")
    ax.set_title(f"(a) from above: the release would pick “{release_pick}”; the veto leaves “{new_pick}”")

    # (b) the rule: any clearance under the margin within the horizon vetoes the proposal.
    bx.axvspan(0, HORIZON_S, color="#fef3c7", alpha=0.7, lw=0)
    bx.axhline(MARGIN_M, color="#d97706", lw=1.2, ls="--")
    bx.axhline(0, color=views.AXIS, lw=0.8)
    for name, c in per_time.items():
        bx.plot(TIMES, c, marker="o", ms=3.5, color=COLOURS[name], lw=1.6, ls="--" if vetoed[name] else "-",
                label=f"{name}: {'vetoed' if vetoed[name] else 'kept'} (smallest {c[within].min():+.1f} m)")
    bx.annotate(f"margin {MARGIN_M:g} m", (3.95, MARGIN_M), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color="#d97706")
    bx.annotate(f"horizon {HORIZON_S:g} s", (HORIZON_S, -2.6), xytext=(-4, 0),
                textcoords="offset points", ha="right", fontsize=8.5, color="#d97706")
    bx.set_xlim(0, 4.1)
    bx.set_ylim(-3, 10)
    bx.set_xlabel("plan time (s)")
    bx.set_ylabel("clearance to the car (m): below 0, the footprints overlap")
    bx.set_title("(b) the rule: vetoed if any clearance within the horizon is under the margin")
    bx.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("B-1, the analytic veto, on a made-up scene: one car, four of the release's proposals "
                 "(clearances from geometry.terms)", fontsize=11)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = HERE.parent / "figures" / "b1-clearance.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(REPO)}: vetoed {[n for n in vetoed if vetoed[n]]}; pick {release_pick} -> {new_pick}")


if __name__ == "__main__":
    main()
