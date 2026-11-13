"""H15 Part B, explained: the real scenes behind B-0's and B-1's numbers, drawn.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/part_b_explained_figures.py

For docs/drivejepa-perception-sidecar/part-b/B-0-explained.md and B-1-explained.md. It reads what B-0's and
B-1's passes wrote under {runs}/drivejepa_perception, NAVSIM's own logs and camera frames, and the metric caches
the scorers read. It recomputes nothing a result rests on: the clearances it draws come from
`geometry.clearances` at B-1's frozen settings, the code B-1 ran, and the side-car's hits from `evaluate.match`,
the readout's own rule. Every example is the extreme of a rule written in `pick_examples` and printed, with the
numbers behind it, to ../figures/part-b-explained-examples.json.

Writes ../figures/b0x-* and ../figures/b1x-*: camera pictures as JPEG, charts as PNG.
"""

from __future__ import annotations

import json
import lzma
import os
import pickle
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon  # noqa: E402

from gallops.config import SiteConfig  # noqa: E402
from gallops.drivejepa_perception import evaluate, geometry, navsim_data, train_sidecar, veto, views  # noqa: E402
from gallops.drivejepa_perception import navtest as nt  # noqa: E402
from gallops.drivejepa_perception import targets as tg  # noqa: E402

FIGURES = HERE.parent / "figures"
CFG = SiteConfig.load()
OUT = Path(CFG.resolve("{runs}/drivejepa_perception"))
NAVSIM = Path(CFG.resolve("{data.navsim}"))
NAVTEST_CACHE = Path(CFG.resolve("{data.navsim_navtest_metric_cache}"))
FRESH_CACHE = OUT / "metric-cache-fresh-v2"
B1 = OUT / "part-b/b1"
#: B-1's frozen settings (recipes/drivejepa_perception/part_b1_frozen.json): 0.5 m within 1 s, confidence 0.7.
MARGIN, HORIZON, CONFIDENCE = 0.5, 1.0, 0.7
SEED = 0
EGO_RADIUS = float(np.sqrt(geometry.EGO_LENGTH * geometry.EGO_WIDTH / np.pi))
#: Plan times, seconds: the release's 8 poses.
TIMES = geometry.TIMES.numpy()
FAIL, PASS, HIT, MISS = views.MISS, views.GREY, views.HIT, views.MISS
#: What each figure shows, beyond its picture, for the documents: written beside the examples.
DETAILS: dict = {}
views.use_style()


# ------------------------------------------------------------------------------------- the data


def official(directory: Path) -> dict:
    m = json.loads((directory / "manifest.json").read_text())
    return nt.per_token(str(directory / m["csv"]))


class Split:
    """Everything one split's passes wrote, by row."""

    def __init__(self, split: str):
        tok, lab, pdm, agents = veto.O_DIRS[split]
        self.split, self.data_split = split, navsim_data.DATA_SPLIT[split]
        self.labels = dict(np.load(OUT / lab / "labels.npz"))
        self.manifest = json.loads((OUT / lab / "manifest.json").read_text())
        self.tokens = [str(t) for t in self.labels["token"]]
        self.row = {t: i for i, t in enumerate(self.tokens)}
        self.logs = np.array(self.manifest["logs"])[self.labels["log_id"]]
        self.proposals = np.load(OUT / tok / "proposals.npy", mmap_mode="r")
        self.scores = np.load(OUT / tok / "scores.npy", mmap_mode="r")
        self.pick = np.load(OUT / tok / "pick.npy").astype(np.int64)
        self.lidar2img = np.load(OUT / tok / "lidar2img.npy", mmap_mode="r")
        self.ego_status = np.load(OUT / tok / "ego_status.npy", mmap_mode="r")
        z = np.load(OUT / pdm / "pdm.npz")
        self.pdm, self.human_pdm = z["candidates"], z["human"]
        self.collision_agent, self.collision_step = z["collision_agent"], z["collision_step"]
        self.ttc_agent, self.ttc_step = z["ttc_agent"], z["ttc_step"]
        self.agents = {k: dict(np.load(OUT / agents.format(seed=k) / "agents.npz")) for k in (0, 1, 2)}
        self.cache_root = NAVTEST_CACHE if split == "navtest" else FRESH_CACHE
        self._failing = veto.ceiling_vetoed(self.pdm)

    def projection(self, row: int) -> np.ndarray:
        return tg.front_projection(np.asarray(self.lidar2img[row]))

    def labelled(self, row: int) -> dict:
        return navsim_data.agents_of(self.labels, row)

    def targets(self, row: int) -> dict:
        return tg.build(self.labelled(row), self.projection(row))

    def detections(self, row: int, seed: int = SEED) -> dict:
        return train_sidecar.decoded(self.agents[seed], row)

    def frames(self, row: int) -> list:
        return views.window_frames(NAVSIM, self.logs[row], self.tokens[row], self.data_split)

    def image(self, row: int, frame: int = navsim_data.CURRENT) -> np.ndarray:
        frames = self.frames(row)
        return views.read_image(views.camera_path(NAVSIM, frames[frame], "CAM_F0", self.data_split))

    def cache(self, row: int):
        path = self.cache_root / self.logs[row] / "unknown" / self.tokens[row] / "metric_cache.pkl"
        with lzma.open(path, "rb") as fh:
            return pickle.load(fh)

    def failing(self) -> np.ndarray:
        """(N, 32) the grader's verdict per proposal: fails NC (0 or 0.5) or TTC (B-1-1's labels)."""
        return self._failing


def match_row(dets: dict, target: dict):
    """The readout's own match (evaluate.match, 2 m, class, confidence 0.5): detection index -> hit?,
    and target index -> the detection that hit it (-1 for a miss)."""
    m = target["mask"]
    gxy, gcls = target["box"][m, :2].astype(np.float64), target["cls"][m]
    det, tp, hit_of_gt = evaluate.match(dets["conf"], dets["cls"], dets["xyz"][:, :2], gxy, gcls)
    return det, tp, hit_of_gt


def clearance_of(s: Split, row: int, agents: dict, keep: np.ndarray):
    """(32, A, 8) clearances of the release's 32 proposals to `agents` (geometry.clearances, B-1's code)."""
    with torch.no_grad():
        c = geometry.clearances(
            torch.from_numpy(np.asarray(s.proposals[row : row + 1], np.float32)),
            torch.from_numpy(np.asarray(agents["xyz"][None, :, :2], np.float32)),
            torch.from_numpy(np.asarray(agents["lwh"][None, :, :2], np.float32)),
            torch.from_numpy(np.asarray(agents["vel"][None], np.float32)),
            torch.from_numpy(np.asarray(agents["path"][None], np.float32)),
            torch.from_numpy(np.asarray(keep[None], bool)),
        )
    return c[0].numpy()


def sidecar_raw(s: Split, row: int, seed: int = SEED) -> dict:
    """The side-car's agents on a row in `geometry`'s format: xyz, lwh, vel, path offsets, conf, cls."""
    a = s.agents[seed]
    return {k: np.asarray(a[k][row]) for k in ("xyz", "lwh", "vel", "path", "conf", "cls", "yaw")}


def _one_row(s: Split, row: int) -> dict:
    """A one-row view of the packed labels, for targets.truth_agents."""
    out = {k: s.labels[k] for k in navsim_data.AGENT_KEYS}
    out["count"] = s.labels["count"][row : row + 1]
    out["offset"] = s.labels["offset"][row : row + 1]
    return out


def truth_one(s: Split, row: int) -> dict:
    t = tg.truth_agents(_one_row(s, row), s.projection(row)[None], fill="constant_velocity")
    return {k: v[0] for k, v in t.items()}


# --------------------------------------------------------------------------------- the drawing


def bev(ax, xlim=(-20, 20), ylim=(-8, 45), title=""):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("← left      lateral (m)      right →")
    ax.set_ylabel("ahead (m)")
    if title:
        ax.set_title(title, loc="left")


def xy_to_plot(xy: np.ndarray) -> tuple:
    """Ego (x forward, y left) -> the page (left on the left, ahead up)."""
    xy = np.asarray(xy)
    return -xy[..., 1], xy[..., 0]


def draw_road(ax, cache) -> None:
    """The drivable area and the route's centreline the grader reads, in the ego frame."""
    ego = np.array([cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading])
    for polygon in cache.drivable_area_map._geometries:
        xs, ys = polygon.exterior.xy
        u, v = xy_to_plot(navsim_data.to_local(np.c_[xs, ys], ego))
        ax.fill(u, v, color=views.GRID, zorder=0, lw=0)
    centre = navsim_data.to_local(np.array([[p.x, p.y] for p in cache.centerline.discrete_path]), ego)
    u, v = xy_to_plot(centre)
    ax.plot(u, v, color=views.AXIS, lw=1, zorder=1)


def grader_agents(ax, cache, step: int = 0, highlight=None, colour=views.INK_2) -> None:
    """Every agent the grader reads (the metric cache's observation) at t, from above; with `highlight` (a track
    id), that agent outlined at t and filled at the 10 Hz `step` of the collision."""
    ego = np.array([cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading])
    for k, fill in ((0, True), (step, False)) if highlight is not None else ((0, True),):
        now = cache.observation[k]
        for token, polygon in zip(now.tokens, now._geometries):  # noqa: B905 -- Python 3.9
            xs, ys = polygon.exterior.xy
            u, v = xy_to_plot(navsim_data.to_local(np.c_[xs, ys], ego))
            hit = highlight is not None and navsim_data.track_id(token) == highlight
            if fill and not hit:
                ax.fill(u, v, color=colour, alpha=0.25, zorder=3, lw=0)
            elif hit and fill:
                ax.fill(u, v, fc="none", ec=FAIL, lw=1.2, zorder=6)
            elif hit:
                ax.fill(u, v, color=FAIL, alpha=0.85, zorder=6, lw=0)


def ego_box_at(ax, proposal: np.ndarray, seconds: float) -> None:
    """The ego's footprint where a plan puts it at `seconds` (the plan's own poses, interpolated: the grader
    simulates a controller tracking them, so this is close, not exact)."""
    t = np.r_[0.0, TIMES]
    poses = np.vstack([[0.0, 0.0, 0.0], np.asarray(proposal, np.float64)])
    x, y, yaw = (np.interp(seconds, t, poses[:, i]) for i in range(3))
    centre = np.array([x, y]) + geometry.EGO_CENTRE_AHEAD * np.array([np.cos(yaw), np.sin(yaw)])
    box = np.array([centre[0], centre[1], 0.0, geometry.EGO_LENGTH, geometry.EGO_WIDTH, 1.6, yaw])
    u, v = xy_to_plot(views.footprint(box[None])[0])
    ax.add_patch(Polygon(np.c_[u, v], closed=True, fc="none", ec=views.INK, lw=1.4, ls=(0, (3, 2)), zorder=8))


def footprints(ax, boxes: np.ndarray, colours, fill: bool, lw: float = 1.5, alpha: float = 0.35, zorder: int = 4):
    for box, colour in zip(np.asarray(boxes).reshape(-1, 7), colours):  # noqa: B905 -- Python 3.9
        u, v = xy_to_plot(views.footprint(box[None])[0])
        if fill:
            ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=colour, ec="none", alpha=alpha, zorder=zorder))
        else:
            ax.add_patch(Polygon(np.c_[u, v], closed=True, fc="none", ec=colour, lw=lw, zorder=zorder))


def plans(ax, s: Split, row: int, picks: dict, vetoed=None, human: bool = True) -> None:
    """The 32 proposals coloured by the grader's verdict, the vetoed ones crossed, and the named picks."""
    failing = s.failing()[row]
    for k in range(32):
        u, v = xy_to_plot(np.asarray(s.proposals[row, k, :, :2]))
        ax.plot(u, v, color=FAIL if failing[k] else PASS, lw=0.9, alpha=0.9 if failing[k] else 0.55, zorder=5)
        if vetoed is not None and vetoed[k]:
            ax.plot(u[-1], v[-1], marker="x", ms=5, color=views.INK_2, mew=1.2, zorder=6)
    if human:
        u, v = xy_to_plot(s.labels["ego_future"][row][:, :2])
        ax.plot(u, v, color=views.INK_2, lw=1.5, ls=(0, (1, 2)), zorder=7)
    same = len({k for k, _ in picks.values()}) < len(picks)
    for i, (name, (k, marker)) in enumerate(picks.items()):
        u, v = xy_to_plot(np.asarray(s.proposals[row, k, :, :2]))
        ax.plot(u, v, color=views.INK, lw=2.4, zorder=8, marker=marker, ms=5, mfc=views.SURFACE, mec=views.INK, mew=1.2)
        if same and i:
            continue
        label = " = ".join(picks) if same else name
        ax.annotate(label, (u[-1], v[-1]), xytext=(6, 2), textcoords="offset points", fontsize=8, color=views.INK, zorder=9)
    ax.plot([0], [0], marker="^", color=views.INK, ms=9, zorder=9)


def plan_legend(ax, extra=(), loc="lower left", outside: bool = False) -> None:
    handles = [
        Line2D([], [], color=PASS, lw=1.2, label="proposal the grader passes"),
        Line2D([], [], color=FAIL, lw=1.2, label="proposal failing NC or TTC (grader)"),
        Line2D([], [], color=views.INK_2, lw=1.5, ls=(0, (1, 2)), label="the human's plan"),
        *extra,
    ]
    if outside:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    else:
        ax.legend(handles=handles, loc=loc, fontsize=7)


def image_boxes(ax, image512, projection, boxes, colours, lws, styles=None, labels=None) -> None:
    ax.imshow(image512)
    for b, (box, colour, lw) in enumerate(zip(np.asarray(boxes).reshape(-1, 7), colours, lws)):  # noqa: B905
        ls = "-" if styles is None else styles[b]
        for (u0, v0), (u1, v1) in views.box_edges_2d(projection, box[None])[0]:
            ax.plot([u0, u1], [v0, v1], color=colour, lw=lw, ls=ls)
        if labels is not None and labels[b]:
            uv, depth = views.project(projection, views.box_corners(box[None])[0])
            ok = depth > 0.5
            if ok.any():
                u, v = float(np.clip(uv[ok, 0].min(), 2, 470)), float(np.clip(uv[ok, 1].min() - 3, 8, 250))
                ax.text(u, v, labels[b], color=views.INK, fontsize=6.5,
                        bbox={"fc": views.SURFACE, "ec": "none", "pad": 0.6, "alpha": 0.85})
    ax.set_xlim(0, 512)
    ax.set_ylim(256, 0)
    ax.axis("off")


def det_box(raw: dict, q: int) -> np.ndarray:
    """A detection's box (x, y, z, l, w, h, yaw) from the agents file's own arrays (`sidecar_raw`)."""
    return np.r_[raw["xyz"][q], raw["lwh"][q], raw["yaw"][q]]


def save(fig, name: str) -> None:
    """Camera pictures as JPEG (about a quarter of a PNG's bytes, as views.show_jpeg keeps notebooks), charts
    as PNG (JPEG blurs thin lines and text)."""
    if name.endswith(".jpg"):
        fig.savefig(FIGURES / name, dpi=130, bbox_inches="tight", pil_kwargs={"quality": 85})
    else:
        fig.savefig(FIGURES / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


# ---------------------------------------------------------------------------- choosing examples


def row_f1(s: Split, row: int, seed: int = SEED, reach: float = 30.0) -> tuple:
    """(F1, targets, detections) at 2 m within `reach`, one row, by the readout's match."""
    t = s.targets(row)
    d = s.detections(row, seed)
    m = t["mask"] & (np.hypot(t["box"][:, 0], t["box"][:, 1]) <= reach)
    sub = {k: v.copy() for k, v in t.items()}
    sub["mask"] = m
    det, tp, hit_of_gt = match_row(d, sub)
    near = np.hypot(d["xyz"][det, 0], d["xyz"][det, 1]) <= reach
    n_t, n_d, n_tp = int(m.sum()), int(near.sum()), int(tp[near].sum())
    f1 = 2 * n_tp / (n_t + n_d) if n_t + n_d else 1.0
    return f1, n_t, n_d


def pick_examples(nav: Split, fresh: Split) -> dict:
    """Every example, by the rule in its "rule" field; ties go to the first token in sorted order."""
    out = {}
    n = np.arange(len(nav.pick))
    R = official(OUT / "part-b/b0/navtest-R")
    arm = official(B1 / f"navtest/score-B-1-3-s{SEED}")
    pick13 = np.load(B1 / f"navtest/plans-B-1-3-s{SEED}/plans.npz")["pick"].astype(np.int64)
    delta = np.array([arm[t]["epdms_hf"] - R[t]["epdms_hf"] for t in nav.tokens])
    switched = np.flatnonzero(pick13 != nav.pick)
    failing_release = nav.failing()[n, nav.pick]

    def first(rows, key):
        rows = list(rows)
        best = min(key(r) for r in rows)
        return sorted([r for r in rows if key(r) == best], key=lambda r: nav.tokens[r])[0]

    good = first(switched, lambda r: -delta[r])
    out["b1_good"] = {"row": int(good), "rule": "B-1-3 (seed 0) switched the pick; the largest official EPDMS gain over R",
                      "delta": float(delta[good]), "release_pick_fails": bool(failing_release[good])}
    bad = first(switched, lambda r: delta[r])
    out["b1_bad_wrong"] = {"row": int(bad), "rule": "B-1-3 (seed 0) switched the pick; the largest official EPDMS loss",
                           "delta": float(delta[bad]), "release_pick_fails": bool(failing_release[bad])}

    # The switch caused by a false alarm with the largest loss: the agent behind the veto of the release's pick,
    # matched to no labelled target of its class within 2 m.
    alarms = []
    for r in switched:
        a = sidecar_raw(nav, r)
        keep = a["conf"] >= CONFIDENCE
        c = clearance_of(nav, r, a, keep)[nav.pick[r], :, : veto.horizon_steps(HORIZON)]
        if c.min() >= MARGIN:
            continue
        q = int(np.unravel_index(c.argmin(), c.shape)[0])
        t = nav.targets(r)
        one = {"conf": np.array([1.0]), "cls": a["cls"][q : q + 1], "xyz": a["xyz"][q : q + 1]}
        _, tp, _ = match_row(one, t)
        if not tp.any():
            alarms.append(r)
    alarm = first(alarms, lambda r: delta[r])
    out["b1_bad_false_alarm"] = {"row": int(alarm), "rule": "B-1-3 (seed 0) switched because of a false alarm; the largest loss",
                                 "delta": float(delta[alarm]), "false_alarm_switches": len(alarms)}

    # A failure the veto misses: the release's pick fails NC (official), the at-fault agent is a target the
    # side-car found, and B-1-3 kept the pick. The nearest such agent at t.
    missed = []
    for r in np.flatnonzero((np.array([R[t]["NC"] for t in nav.tokens]) < 1) & (pick13 == nav.pick)):
        track = int(nav.collision_agent[r, nav.pick[r]])
        agents = nav.labelled(r)
        where = np.flatnonzero(agents["track"] == track)
        if not len(where):
            continue
        index = tg.target_index(agents, nav.projection(r))
        slot = np.flatnonzero(index == where[0])
        if not len(slot):
            continue
        _, _, hit_of_gt = match_row(nav.detections(r), nav.targets(r))
        if hit_of_gt[slot[0]] >= 0:
            missed.append((r, float(np.hypot(*agents["box"][where[0], :2]))))
    miss = first([r for r, _ in missed], lambda r: dict(missed)[r])
    out["b1_miss"] = {"row": int(miss), "rule": "the release's pick fails NC (official), the at-fault agent is a target "
                      "seed 0 found, and B-1-3 kept the pick; the nearest such agent", "candidates": len(missed)}

    # EC: a scene B-1-3 did not switch whose official score still moved (its successor's plan did).
    unswitched = np.flatnonzero((pick13 == nav.pick) & (delta != 0))
    ec = first(unswitched, lambda r: delta[r])
    out["b1_ec"] = {"row": int(ec), "rule": "B-1-3 kept the pick, yet the official EPDMS fell (EC reads the next frame's "
                    "plan); the largest such fall", "delta": float(delta[ec]), "candidates": int(len(unswitched))}

    # The side-car, at its best and worst, among navtest rows with at least 6 targets within 30 m (seed 0).
    scored = []
    for r in range(0, len(nav.tokens), 7):
        f1, n_t, n_d = row_f1(nav, r)
        if n_t >= 6:
            scored.append((r, f1, n_t))
    by = {r: (f1, n_t) for r, f1, n_t in scored}
    best = first([r for r, _, _ in scored], lambda r: (-by[r][0], -by[r][1]))
    worst = first([r for r, _, _ in scored], lambda r: (by[r][0], -by[r][1]))
    out["b0_sidecar_good"] = {"row": int(best), "rule": "every 7th navtest row with >= 6 targets within 30 m: the best F1 "
                              "(seed 0), then the most targets", "f1": by[best][0], "targets": by[best][1]}
    out["b0_sidecar_bad"] = {"row": int(worst), "rule": "the same rows: the worst F1, then the most targets",
                             "f1": by[worst][0], "targets": by[worst][1]}

    # Headroom: a failure behind which the grader names an agent out of the camera's view.
    behind = []
    for r in np.flatnonzero((np.array([R[t]["NC"] for t in nav.tokens]) < 1) & ~nav.failing().all(1)):
        track = int(nav.collision_agent[r, nav.pick[r]])
        agents = nav.labelled(r)
        where = np.flatnonzero(agents["track"] == track)
        if len(where) and agents["cls"][where[0]] < tg.NUM_CLASSES:
            if not len(np.flatnonzero(tg.target_index(agents, nav.projection(r)) == where[0])):
                behind.append(r)
    out["b0_out_of_view"] = {"row": int(sorted(behind, key=lambda r: nav.tokens[r])[0]), "rule": "the release's pick fails "
                             "NC (official), a safe proposal exists, and the at-fault agent is out of CAM_F0's view or "
                             "beyond 50 m; the first token", "candidates": len(behind)}

    # R-clean: an unknown-command window where the release's pick fails NC (vendored); the first token.
    unknown = ~veto.known_command(OUT / veto.O_DIRS["fresh"][0])
    m = np.arange(len(fresh.pick))
    ood = np.flatnonzero(unknown & (fresh.pdm[m, fresh.pick, veto.NC] < 1))
    out["b0_rclean_unknown"] = {"row": int(sorted(ood, key=lambda r: fresh.tokens[r])[0]), "rule": "an R-clean window with "
                                "the 'unknown' command where the release's pick fails NC (vendored); the first token",
                                "candidates": int(len(ood))}
    for key, value in out.items():
        split = fresh if key == "b0_rclean_unknown" else nav
        value["token"] = split.tokens[value["row"]]
        value["log"] = str(split.logs[value["row"]])
    return out


# -------------------------------------------------------------------------------- B-0's figures


def fig_preprocess(s: Split, row: int) -> None:
    frames = s.frames(row)
    full = s.image(row)
    prev = views.read_image(views.camera_path(NAVSIM, frames[navsim_data.CURRENT - 1], "CAM_F0", s.data_split))
    fig = plt.figure(figsize=(13, 7.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax = fig.add_subplot(grid[0, :2])
    ax.imshow(full)
    for r in (tg.CROP_ROWS, full.shape[0] - tg.CROP_ROWS):
        ax.axhline(r, color=views.MISS, lw=1.2)
    ax.set_title(f"1 · CAM_F0 as recorded, {full.shape[1]}x{full.shape[0]}; red: the {tg.CROP_ROWS}-row crop, top and bottom",
                 loc="left")
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(views.model_view(prev))
    ax.set_title("2 · the frame 0.5 s before, as the ViT reads it", loc="left")
    ax.axis("off")
    ax = fig.add_subplot(grid[1, :2])
    image = views.model_view(full)
    ax.imshow(image)
    for x in range(0, 513, views.PATCH):
        ax.axvline(x - 0.5, color=views.SURFACE, lw=0.4, alpha=0.6)
    for y in range(0, 257, views.PATCH):
        ax.axhline(y - 0.5, color=views.SURFACE, lw=0.4, alpha=0.6)
    t = s.targets(row)
    for (box, cls) in zip(t["box"][t["mask"]], t["cls"][t["mask"]]):  # noqa: B905
        for (u0, v0), (u1, v1) in views.box_edges_2d(s.projection(row), box[None])[0]:
            ax.plot([u0, u1], [v0, v1], color=views.CLASS_COLOURS[cls], lw=1.1)
    ax.set_xlim(0, 512)
    ax.set_ylim(256, 0)
    ax.set_title("3 · resized to 512x256, 16x32 tokens (grid); boxes: the labelled targets", loc="left")
    ax.axis("off")
    ax = fig.add_subplot(grid[1, 2])
    agents = s.labelled(row)
    index = tg.target_index(agents, s.projection(row))
    colours = [views.CLASS_COLOURS[c] if i in set(index.tolist()) else views.GREY for i, c in enumerate(agents["cls"].clip(max=2))]
    footprints(ax, agents["box"], colours, fill=True, alpha=0.6)
    fov = views.view_edges(s.projection(row), reach=50)
    u, v = xy_to_plot(fov[[1, 0, 2]])
    ax.plot(u, v, color=views.INK_2, lw=1)
    for radius in (10, 20, 30, 50):
        ax.add_patch(Circle((0, 0), radius, fill=False, ec=views.AXIS, lw=0.7))
    ax.plot([0], [0], marker="^", color=views.INK, ms=8)
    bev(ax, (-35, 35), (-15, 55), "4 · from above: which labels are targets")
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(views.CLASS_COLOURS, views.CLASS_NAMES)]  # noqa: B905
              + [Patch(color=views.GREY, label="static, out of view, or beyond 50 m")], loc="lower left", fontsize=7)
    fig.suptitle(f"What every model in Part B reads: navtest token {s.tokens[row]} (log {s.logs[row]})", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "b0x-preprocess.jpg")


def fig_sidecar(s: Split, row: int, name: str, title: str) -> None:
    t, d, raw = s.targets(row), s.detections(row), sidecar_raw(s, row)
    det, tp, hit_of_gt = match_row(d, t)
    proj = s.projection(row)
    boxes, colours, lws, styles, texts = [], [], [], [], []
    for q, ok in zip(det, tp):  # noqa: B905
        boxes.append(det_box(raw, q))
        colours.append(HIT if ok else MISS)
        lws.append(1.6)
        styles.append("-")
        texts.append(f"{d['conf'][q]:.2f}")
    m = np.flatnonzero(t["mask"])
    for j, g in enumerate(m):
        if hit_of_gt[j] < 0:
            boxes.append(t["box"][g])
            colours.append(MISS)
            lws.append(1.0)
            styles.append((0, (2, 2)))
            texts.append("")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.9, 1]})
    image_boxes(axes[0], views.model_view(s.image(row)), proj, boxes, colours, lws, styles, texts)
    f1, n_t, n_d = row_f1(s, row)
    near_det = np.hypot(d["xyz"][det, 0], d["xyz"][det, 1]) <= 30
    near_gt = np.hypot(t["box"][m, 0], t["box"][m, 1]) <= 30
    counts = (int((tp & near_det).sum()), int((~tp & near_det).sum()), int(((hit_of_gt < 0) & near_gt).sum()))
    far = (int((tp & ~near_det).sum()), int((~tp & ~near_det).sum()), int(((hit_of_gt < 0) & ~near_gt).sum()))
    axes[0].set_title(f"seed 0, within 30 m: {counts[0]} hits, {counts[1]} false alarms, {counts[2]} misses "
                      f"(F1 {f1:.2f}); beyond: {far[0]}, {far[1]}, {far[2]}", loc="left")
    DETAILS[name] = {"within_30m": dict(zip(("hits", "false_alarms", "misses"), counts)),  # noqa: B905
                     "beyond_30m": dict(zip(("hits", "false_alarms", "misses"), far)), "f1_30m": f1}  # noqa: B905
    ax = axes[1]
    footprints(ax, t["box"][m], [views.GREY] * len(m), fill=True, alpha=0.6)
    footprints(ax, np.array([det_box(raw, q) for q in det]).reshape(-1, 7), [HIT if ok else MISS for ok in tp], fill=False)
    for q in det:
        u, v = xy_to_plot(d["path"][q])
        u0, v0 = xy_to_plot(d["xyz"][q, :2])
        ax.plot([u0, *u], [v0, *v], color=views.INK, lw=1)
    for radius in (10, 20, 30, 50):
        ax.add_patch(Circle((0, 0), radius, fill=False, ec=views.AXIS, lw=0.7))
    ax.plot([0], [0], marker="^", color=views.INK, ms=8)
    bev(ax, (-30, 30), (-5, 55), "from above, with each detection's 1 s path")
    ax.legend(handles=[Patch(color=views.GREY, alpha=0.6, label="labelled target"),
                       Patch(fc="none", ec=HIT, label="detection that hits (2 m, class)"),
                       Patch(fc="none", ec=MISS, label="false alarm"),
                       Line2D([], [], color=MISS, lw=1, ls=(0, (2, 2)), label="missed target (image)"),
                       Line2D([], [], color=views.INK, lw=1, label="predicted path, 0.5 s and 1 s")],
              loc="lower left", fontsize=7)
    fig.suptitle(title + f" · navtest {s.tokens[row]}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, name)


def fig_planner(s: Split, row: int, name: str, title: str, step=None, highlight=None) -> None:
    cache = s.cache(row)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    image_boxes(axes[0], views.model_view(s.image(row)), s.projection(row), [], [], [])
    axes[0].set_title("the front camera at t (what the release and the side-car read)", loc="left")
    ax = axes[1]
    draw_road(ax, cache)
    grader_agents(ax, cache, step or 0, highlight)
    plans(ax, s, row, {"release": (int(s.pick[row]), "s")})
    extra = [Patch(color=views.INK_2, alpha=0.25, label="agents at t, as the grader reads them")]
    if highlight is not None:
        ego_box_at(ax, np.asarray(s.proposals[row, s.pick[row]]), step / 10)
        extra += [Patch(fc="none", ec=FAIL, label="the agent hit, at t"), Patch(color=FAIL, label=f"it, at {step / 10:.1f} s"),
                  Line2D([], [], color=views.INK, lw=1.4, ls=(0, (3, 2)), label=f"the ego, planned, at {step / 10:.1f} s")]
    plan_legend(ax, extra, loc="upper left", outside=True)
    bev(ax, (-22, 22), (-8, 48), "from above: the grader's road, agents and verdicts")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, name)


def fig_rclean(fresh: Split, row: int, nav: Split, nav_row: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (s, r, label) in enumerate(((fresh, row, "R-clean, 'unknown' command"), (nav, nav_row, "navtest"))):
        image_boxes(axes[i, 0], views.model_view(s.image(r)), s.projection(r), [], [], [])
        command = ("left", "straight", "right", "unknown")[int(np.argmax(s.ego_status[r][-1, 7:11]))]
        speed = float(s.labels["ego_speed"][r][0])
        axes[i, 0].set_title(f"{label}: {s.tokens[r]} · command '{command}' · ego {speed:.1f} m/s", loc="left")
        ax = axes[i, 1]
        cache = s.cache(r)
        draw_road(ax, cache)
        grader_agents(ax, cache)
        plans(ax, s, r, {"release": (int(s.pick[r]), "s")})
        plan_legend(ax, [Patch(color=views.INK_2, alpha=0.25, label="agents (grader)")], outside=True)
        bev(ax, (-22, 22), (-8, 40), "from above")
    fig.suptitle("Why B-1 chose only on R-clean's known-command windows (D25)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "b0x-rclean-example.jpg")


def fig_speed(fresh: Split, nav: Split) -> None:
    known = veto.known_command(OUT / veto.O_DIRS["fresh"][0])
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bins = np.linspace(0, 20, 41)
    for values, colour, label in (
        (nav.labels["ego_speed"][:, 0], views.CLASS_COLOURS[0], f"navtest ({len(nav.tokens):,} scenes)"),
        (fresh.labels["ego_speed"][:, 0], views.CLASS_COLOURS[1], f"R-clean, all ({len(fresh.tokens):,})"),
        (fresh.labels["ego_speed"][known, 0], views.CLASS_COLOURS[2], f"R-clean, known command ({int(known.sum()):,})"),
    ):
        share, _ = np.histogram(values, bins)
        ax.step(bins[:-1], share / len(values), where="post", color=colour, lw=2, label=label)
    ax.set_xlabel("the ego's speed at t (m/s)")
    ax.set_ylabel("share of windows")
    ax.set_title("The ego's speed: R-clean is mostly a stopped car; navtest mostly drives", loc="left")
    ax.legend()
    fig.tight_layout()
    save(fig, "b0x-rclean-speed.png")


# -------------------------------------------------------------------------------- B-1's figures


def veto_state(s: Split, row: int, source: str = "sidecar") -> dict:
    """B-1's rule on one row at the frozen settings: per proposal the clearance over time (min over kept agents),
    which agent, vetoed or not, and the pick."""
    a = sidecar_raw(s, row) if source == "sidecar" else truth_one(s, row)
    keep = a["conf"] >= CONFIDENCE if source == "sidecar" else a["mask"]
    c = clearance_of(s, row, a, keep)  # 32, A, 8
    per_time = c.min(axis=1)
    who = c.argmin(axis=1)
    vetoed = veto.vetoed(per_time[None], MARGIN, HORIZON)[0]
    pick, _ = veto.pick_with_veto(np.asarray(s.scores[row : row + 1]), vetoed[None], s.pick[row : row + 1])
    return {"agents": a, "keep": keep, "per_time": per_time, "who": who, "vetoed": vetoed, "pick": int(pick[0])}


def draw_agents(ax, a: dict, keep: np.ndarray, show_forecast: bool = True) -> None:
    """The agents the rule read, by class, with their forecast: the 1 s path, then constant velocity."""
    centres = geometry.agent_centres(torch.from_numpy(a["xyz"][None, :, :2].astype(np.float32)),
                                     torch.from_numpy(a["vel"][None].astype(np.float32)),
                                     torch.from_numpy(a["path"][None].astype(np.float32)))[0].numpy()
    for q in np.flatnonzero(keep):
        colour = views.CLASS_COLOURS[int(a["cls"][q]) % 3]
        box = np.r_[a["xyz"][q], a["lwh"][q], a["yaw"][q]]
        footprints(ax, box[None], [colour], fill=True, alpha=0.55)
        if show_forecast:
            u, v = xy_to_plot(np.vstack([a["xyz"][q, :2], centres[q]]))
            ax.plot(u, v, color=colour, lw=1, alpha=0.9, zorder=4)
            ax.plot(u[1:], v[1:], ls="none", marker="o", ms=2.5, color=colour, zorder=4)


def fig_rule(s: Split, row: int, name: str, title: str, verdict: dict) -> dict:
    """B-1's rule on one scene, step by step: the agents and their forecasts, the clearances over time, the veto."""
    st = veto_state(s, row)
    a, keep = st["agents"], st["keep"]
    t = s.targets(row)
    kept = np.flatnonzero(keep)
    real = []
    for q in kept:
        one = {"conf": np.array([1.0]), "cls": a["cls"][q : q + 1], "xyz": a["xyz"][q : q + 1]}
        _, tp, _ = match_row(one, t)
        real.append(bool(tp.any()))
    steps = veto.horizon_steps(HORIZON)
    k_rel = int(s.pick[row])
    when = int(np.argmin(st["per_time"][k_rel, :steps]))
    vetoer = int(st["who"][k_rel, when])
    detail = {
        "vetoed": int(st["vetoed"].sum()),
        "release_pick": k_rel,
        "b13_pick": st["pick"],
        "b13_pick_rank": int((np.asarray(s.scores[row]) > s.scores[row][st["pick"]]).sum()) + 1,
        "release_pick_vetoed": bool(st["vetoed"][k_rel]),
        "release_pick_fails": bool(s.failing()[row, k_rel]),
        "b13_pick_fails": bool(s.failing()[row, st["pick"]]),
    }
    if st["vetoed"][k_rel]:
        detail["vetoed_by"] = {
            "class": views.CLASS_NAMES[int(a["cls"][vetoer]) % 3],
            "xy": [round(float(x), 1) for x in a["xyz"][vetoer, :2]],
            "range_m": round(float(np.hypot(*a["xyz"][vetoer, :2])), 1),
            "confidence": round(float(a["conf"][vetoer]), 2),
            "clearance_m": round(float(st["per_time"][k_rel, when]), 2),
            "at_s": float(TIMES[when]),
            "real_agent": bool(dict(zip(kept.tolist(), real)).get(vetoer, False)),  # noqa: B905
        }
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], height_ratios=[1, 1, 1])
    # 1. The camera, with the agents the rule reads.
    ax = fig.add_subplot(grid[0, 0])
    boxes = [np.r_[a["xyz"][q], a["lwh"][q], a["yaw"][q]] for q in kept]
    colours = [views.CLASS_COLOURS[int(a["cls"][q]) % 3] if ok else MISS for q, ok in zip(kept, real)]  # noqa: B905
    texts = [f"{a['conf'][q]:.2f}" + ("" if ok else " false alarm") for q, ok in zip(kept, real)]  # noqa: B905
    image_boxes(ax, views.model_view(s.image(row)), s.projection(row), boxes, colours, [1.6] * len(boxes), None, texts)
    ax.set_title(f"1 · the side-car's agents at confidence >= {CONFIDENCE} (seed 0); red: matched to no label", loc="left")
    # 2. The clearance of each proposal over time.
    ax = fig.add_subplot(grid[1, 0])
    ax.axvspan(0.25, HORIZON + 0.1, color=views.GRID, alpha=0.8, lw=0, zorder=0)
    for k in range(32):
        ax.plot(TIMES, st["per_time"][k], color=FAIL if s.failing()[row, k] else PASS, lw=0.9, alpha=0.8, zorder=2)
    same = k_rel == st["pick"]
    for k, label, marker in ((k_rel, "release = B-1-3" if same else "release", "s"), (st["pick"], "B-1-3", "o")):
        ax.plot(TIMES, st["per_time"][k], color=views.INK, lw=2.2, marker=marker, ms=5, mfc=views.SURFACE, zorder=3)
        if same and label == "B-1-3":
            continue
        ax.annotate(label, (TIMES[-1], min(st["per_time"][k, -1], 11.5)), xytext=(5, 0), textcoords="offset points",
                    fontsize=8)
    ax.axhline(MARGIN, color=views.INK_2, lw=1)
    ax.set_ylim(-3, 12)
    ax.text(0.3, MARGIN + 0.3, f"margin {MARGIN} m", fontsize=8, color=views.INK_2)
    ax.text(0.3, 11.0, f"horizon {HORIZON:g} s", fontsize=8, color=views.INK_2)
    ax.set_xlabel("plan time (s)")
    ax.set_ylabel("clearance to the nearest kept agent (m)")
    ax.set_title(f"2 · each proposal's clearance over time: {detail['vetoed']} of 32 under the margin within the horizon "
                 "(vetoed)", loc="left")
    ax.legend(handles=[Line2D([], [], color=PASS, lw=1.2, label="passes NC and TTC (grader)"),
                       Line2D([], [], color=FAIL, lw=1.2, label="fails NC or TTC (grader)")],
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=7)
    # 3. The release's scores and the pick.
    ax = fig.add_subplot(grid[2, 0])
    scores = np.asarray(s.scores[row])
    order = list(np.argsort(-scores)[:11])
    if st["pick"] not in order:
        order.append(st["pick"])
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, scores[order], color=[FAIL if s.failing()[row, k] else PASS for k in order], height=0.6)
    for y, k in zip(ypos, order):  # noqa: B905
        tags = [f"rank {int((scores > scores[k]).sum()) + 1}"]
        if k == k_rel:
            tags.append("the release's pick")
        if k == st["pick"]:
            tags.append("B-1-3's pick")
        if st["vetoed"][k]:
            tags.append("vetoed")
        ax.text(scores[order].min() - 0.24 * (scores[order].max() - scores[order].min()), y, f"#{k}  " + ", ".join(tags),
                va="center", fontsize=7.5, color=views.INK)
    span = scores[order].max() - scores[order].min()
    ax.set_xlim(scores[order].min() - 0.25 * span, scores[order].max() + 0.02 * span)
    ax.set_yticks([])
    ax.set_xlabel("the release's own score (sigmoid of its last logit); bar colour: the grader's verdict")
    ax.set_title("3 · the release's best-scored proposals: B-1-3 picks the best one not vetoed", loc="left")
    # 4. From above.
    ax = fig.add_subplot(grid[:, 1])
    draw_road(ax, s.cache(row))
    draw_agents(ax, a, keep)
    for k, ls in ((k_rel, "-"), (st["pick"], (0, (2, 2)))):
        centres = geometry.ego_centres(torch.from_numpy(np.asarray(s.proposals[row, k], np.float32)[None, None]))[0, 0].numpy()
        for j in range(steps):
            u, v = xy_to_plot(centres[j])
            ax.add_patch(Circle((u, v), EGO_RADIUS, fill=False, ec=views.INK, lw=1, ls=ls, zorder=7))
    plans(ax, s, row, {"release": (k_rel, "s"), "B-1-3": (st["pick"], "o")}, st["vetoed"], human=True)
    if "vetoed_by" in detail:
        vb = detail["vetoed_by"]
        u, v = xy_to_plot(np.array(vb["xy"]))
        ax.annotate(f"vetoes the release's pick:\n{vb['clearance_m']:.1f} m at {vb['at_s']:g} s"
                    + ("" if vb["real_agent"] else "\n(a false alarm)"), (u, v), xytext=(-110, -40),
                    textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "color": views.INK_2, "lw": 0.8},
                    bbox={"fc": views.SURFACE, "ec": views.AXIS, "pad": 2})
    bev(ax, (-14, 14), (-6, 40), "4 · from above: the agents and their forecast (dots, 0.5 s apart); the ego's footprint "
        "circles at 0.5 s and 1 s on the two picks")
    plan_legend(ax, [Line2D([], [], color=views.INK, lw=2.4, marker="s", mfc=views.SURFACE, label="the release's pick"),
                     Line2D([], [], color=views.INK, lw=2.4, marker="o", mfc=views.SURFACE, label="B-1-3's pick"),
                     Line2D([], [], ls="none", marker="x", color=views.INK_2, label="vetoed (at the plan's end)"),
                     Patch(color=views.CLASS_COLOURS[0], alpha=0.55, label="an agent the rule reads (by class)")],
                loc="lower left")
    r = verdict
    changed = ", ".join(f"{k} {r['R_terms'][k]:.2f}->{r['A_terms'][k]:.2f}" for k in nt.SHORT
                        if not _same_value(r["R_terms"][k], r["A_terms"][k]))
    fig.suptitle(f"{title} · navtest {s.tokens[row]}\nofficial EPDMS: the release {r['R']:.3f}, B-1-3 {r['A']:.3f}"
                 + (f" ({changed})" if changed else ""), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, name)
    DETAILS[name] = detail
    return detail


def _same_value(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def verdict_of(R: dict, A: dict, token: str) -> dict:
    def ec(x):
        return "-" if np.isnan(x) else f"{x:.0f}"

    return {
        "R": R[token]["epdms_hf"], "A": A[token]["epdms_hf"],
        "R_terms": {k: R[token][k] for k in nt.SHORT}, "A_terms": {k: A[token][k] for k in nt.SHORT},
        "R_EC": ec(R[token]["EC"]), "A_EC": ec(A[token]["EC"]),
    }


def fig_ec(s: Split, row: int, verdict: dict) -> None:
    """Two frames in one log: B-1-3 switched at t+1 and not at t, so t's two-frame comfort (EC) fails."""
    nxt = None
    for r in np.flatnonzero(s.logs == s.logs[row]):
        dt = s.labels["timestamp"][r] - s.labels["timestamp"][row]
        if 0 < dt < 550_000:
            nxt = int(r)
    pick13 = np.load(B1 / f"navtest/plans-B-1-3-s{SEED}/plans.npz")["pick"].astype(np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.4))
    ego_next = s.labels["ego_future"][row][0]  # the ego's pose at t + 0.5 s, in frame t
    c, sn = np.cos(ego_next[2]), np.sin(ego_next[2])
    for ax, (label, p_t, p_next) in zip(axes, (("the release", s.pick[row], s.pick[nxt]),  # noqa: B905
                                                ("B-1-3", pick13[row], pick13[nxt]))):
        cache = s.cache(row)
        draw_road(ax, cache)
        now = np.asarray(s.proposals[row, p_t, :, :2])
        later = np.asarray(s.proposals[nxt, p_next, :, :2]) @ np.array([[c, sn], [-sn, c]]) + ego_next[:2]
        u, v = xy_to_plot(now)
        ax.plot(u, v, color=views.INK, lw=2.2, marker="s", ms=4, mfc=views.SURFACE, label="plan at t")
        u, v = xy_to_plot(later)
        ax.plot(u, v, color=views.CLASS_COLOURS[0], lw=2.2, marker="o", ms=4, mfc=views.SURFACE,
                label="plan at t + 0.5 s, moved into frame t")
        ax.plot([0], [0], marker="^", color=views.INK, ms=9)
        gap = np.linalg.norm(now[1:] - later[:-1], axis=-1).mean()
        bev(ax, (-10, 10), (-3, 40), f"{label}: #{p_t} at t, #{p_next} at t + 0.5 s\noverlap gap {gap:.2f} m")
        DETAILS.setdefault("b1x-ec.png", {})[label] = {"pick_t": int(p_t), "pick_next": int(p_next), "gap_m": float(gap)}
        ax.legend(loc="lower left", fontsize=7)
    fig.suptitle(f"Why EC falls when a veto flickers · navtest {s.tokens[row]} and its next frame\nEC at t: "
                 f"the release {verdict['R_EC']}, B-1-3 {verdict['A_EC']}; EPDMS {verdict['R']:.3f} -> {verdict['A']:.3f}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "b1x-ec.png")


def fig_forest() -> None:
    r = json.loads((B1 / "navtest/navtest.json").read_text())
    r4 = json.loads((B1 / "navtest-b14/navtest.json").read_text())
    rows = [("B-1-1, the ceiling (reads the labels)", r["pairs"]["B-1-1"]), ("B-1-2, labelled agents", r["pairs"]["B-1-2"])]
    rows += [(f"B-1-3, side-car, seed {k}", r["pairs"][f"B-1-3-s{k}"]) for k in (0, 1, 2)]
    rows += [(f"B-1-4, soft veto, seed {k}", r4["pairs"][f"B-1-4-s{k}"]) for k in (0, 1, 2)]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for y, (label, p) in enumerate(rows[::-1]):
        d, se = p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]
        ceiling = label.startswith("B-1-1")
        ax.plot([d - 2 * se, d + 2 * se], [y, y], color=views.CLASS_COLOURS[0], lw=2)
        ax.plot(d, y, marker="o", ms=8, mfc=views.SURFACE if ceiling else views.CLASS_COLOURS[0],
                mec=views.CLASS_COLOURS[0], mew=2)
        ax.text(d + 2 * se + 0.05, y, f"{d:+.2f}", va="center", fontsize=8, color=views.INK)
    ax.axvline(0, color=views.INK_2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows[::-1]])
    ax.set_xlabel("EPDMS (human filter) minus the release's 87.783, points: per-scene paired mean ± 2 SE (136 logs)")
    ax.set_title("B-1 on navtest, read once: only the ceiling, which reads the answer, gains", loc="left")
    fig.tight_layout()
    save(fig, "b1x-navtest-results.png")


def fig_grid() -> None:
    g = json.loads((B1 / "grid-fresh/grid.json").read_text())
    g4 = json.loads((B1 / "grid-b14-fresh/grid.json").read_text())
    frozen = json.loads((REPO / "recipes/drivejepa_perception/part_b1_frozen.json").read_text())["chosen"]
    frozen4 = json.loads((REPO / "recipes/drivejepa_perception/part_b14_frozen.json").read_text())["chosen"]
    groups = [
        ("B-1-2", {k: v["truth"]["minus_release"] for k, v in g["results"]["B-1-2"].items()}, frozen["B-1-2"]),
        ("B-1-3", {k: v["mean_over_seeds"]["minus_release"] for k, v in g["results"]["B-1-3"].items()}, frozen["B-1-3"]),
        ("B-1-4", {k: v["mean_over_seeds"]["minus_release"] for k, v in g4["results"].items()}, frozen4["B-1-4"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (label, values, chosen) in zip(axes, groups):  # noqa: B905
        names = sorted(values, key=lambda k: -values[k])
        x = np.arange(len(names))
        y = np.array([values[k] * 100 for k in names])
        ax.plot(x, y, ls="none", marker="o", ms=6, color=views.CLASS_COLOURS[0])
        c = names.index(chosen)
        ax.plot(c, y[c], marker="o", ms=11, mfc="none", mec=views.INK, mew=1.6)
        ax.annotate(f"frozen: {chosen}", (c, y[c]), xytext=(8, 8), textcoords="offset points", fontsize=8)
        ax.axhline(0, color=views.INK_2, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=70, fontsize=6.5)
        ax.set_title(f"{label}: {len(names)} settings", loc="left")
    axes[0].set_ylabel("selection value minus the release's (points)")
    fig.suptitle("The settings on R-clean's known-command windows (city-reweighted): every setting is at or below the "
                 "release, so the least intrusive is frozen", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "b1x-rclean-grid.png")


# ----------------------------------------------------------------------------------------- main


def main() -> None:
    from gallops.drivejepa_perception import env

    env.enter(CFG)  # navsim's own classes, for unpickling the metric caches the scorers read
    nav, fresh = Split("navtest"), Split("fresh")
    examples = pick_examples(nav, fresh)
    R = official(OUT / "part-b/b0/navtest-R")
    A = official(B1 / f"navtest/score-B-1-3-s{SEED}")
    for key in ("b1_good", "b1_bad_wrong", "b1_bad_false_alarm", "b1_miss", "b1_ec"):
        examples[key]["official"] = verdict_of(R, A, examples[key]["token"])
    (FIGURES / "part-b-explained-examples.json").write_text(json.dumps(examples, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "official"} for k, v in examples.items()}, indent=1))

    good = examples["b1_good"]["row"]
    fig_preprocess(nav, good)
    fig_sidecar(nav, examples["b0_sidecar_good"]["row"], "b0x-sidecar-good.jpg", "The side-car at its best on navtest")
    fig_sidecar(nav, examples["b0_sidecar_bad"]["row"], "b0x-sidecar-bad.jpg", "The side-car at its worst on navtest")
    miss = examples["b1_miss"]["row"]
    fig_planner(nav, miss, "b0x-headroom-in-view.jpg",
                f"Room for an agent input: the release's pick collides at "
                f"{nav.collision_step[miss, nav.pick[miss]] / 10:.1f} s with an agent the camera sees and the side-car "
                f"found; {int((~nav.failing()[miss]).sum())} of 32 proposals pass",
                step=int(nav.collision_step[miss, nav.pick[miss]]), highlight=int(nav.collision_agent[miss, nav.pick[miss]]))
    out = examples["b0_out_of_view"]["row"]
    fig_planner(nav, out, "b0x-headroom-out-of-view.jpg",
                f"Room only for a better pick: the release's pick collides at "
                f"{nav.collision_step[out, nav.pick[out]] / 10:.1f} s with an agent outside the front camera's view",
                step=int(nav.collision_step[out, nav.pick[out]]), highlight=int(nav.collision_agent[out, nav.pick[out]]))
    fig_rclean(fresh, examples["b0_rclean_unknown"]["row"], nav, good)
    fig_speed(fresh, nav)

    fig_rule(nav, good, "b1x-good.jpg", "A veto that helps (the largest gain)", examples["b1_good"]["official"])
    fig_rule(nav, examples["b1_bad_wrong"]["row"], "b1x-bad-real-agent.jpg", "A veto that hurts (the largest loss)",
             examples["b1_bad_wrong"]["official"])
    fig_rule(nav, examples["b1_bad_false_alarm"]["row"], "b1x-bad-false-alarm.jpg",
             "A veto on a false alarm (the largest loss among them)", examples["b1_bad_false_alarm"]["official"])
    fig_rule(nav, miss, "b1x-miss.jpg", "A failure the veto misses", examples["b1_miss"]["official"])
    fig_ec(nav, examples["b1_ec"]["row"], examples["b1_ec"]["official"])
    fig_forest()
    fig_grid()
    examples["figures"] = DETAILS
    (FIGURES / "part-b-explained-examples.json").write_text(json.dumps(examples, indent=1, default=str))


if __name__ == "__main__":
    main()
