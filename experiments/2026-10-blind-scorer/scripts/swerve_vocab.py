"""The proposed 32-path swerve vocabulary, built on real round-0 ticks, graded, and drawn.

For each chosen episode (released model drove, round0_v5), pick one tick by a fixed rule,
build the 32 vocabulary paths from the released pick, grade the 32 native candidates and the
32 vocabulary paths with the gallops grader (every vehicle, lane reference, drivable raster),
and draw them on the front camera and in a top-down view.

    python swerve_vocab.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "{gallops}/src")
from gallops.expert.drivable import DrivableArea  # noqa: E402
from gallops.expert.grader import actor_at, corners, grade  # noqa: E402
from gallops.expert.lanes import load_references, reference_at  # noqa: E402

R = Path("{runs}")
COLLECT, DUMPS, LABELS, DRIVABLE = R / "collect", R / "collect_dumps", R / "labelled_round0_v5", R / "drivable"
CALIB = Path("{paddock_runs}/ops/ncap_viewer/work")

OFFSETS = (-3.5, -2.5, -1.5, -1.0, 1.0, 1.5, 2.5, 3.5)  # metres, left positive
SPEEDS = (1.0, 0.7)  # 0.7 = brake to 40% of the speed by 4 s, about 70% of the distance
SHAPES = ("hold", "return")
DT = 0.5
T = np.arange(1, 9) * DT
OFF_LANE_M = 1.75
VEHICLE_MIN_LENGTH = 1.2  # pedestrians are ~0.5 m long; neuroNCAP does not count them
TARGET_TTC_S = 3.0

EPISODES = ["frontal-0053", "frontal-0064", "frontal-0070", "frontal-0075", "stationary-0070", "side-0070"]

STATUS = {"clear": "#0ca30c", "off-lane": "#fab219", "off-road": "#ec835a", "hit": "#d03b3b"}
ORDER = ("clear", "off-lane", "off-road", "hit")


def ease(u):
    return (1 - np.cos(np.pi * np.clip(u, 0.0, 1.0))) / 2


def lateral(shape: str, t: np.ndarray) -> np.ndarray:
    """Fraction of the offset reached at time t."""
    if shape == "hold":
        return ease(t / 2.0)
    out = np.where(t <= 1.5, ease(t / 1.5), 1.0)
    return np.where(t > 2.5, ease((4.0 - t) / 1.5), out)


def retime(base: np.ndarray, factor: float) -> np.ndarray:
    """The pick's own path, driven with speed scaled by 1 - k*t (k chosen so 4 s keeps `factor` of the distance)."""
    if factor == 1.0:
        return base[:, :2].copy()
    k = (1 - factor) / 2.0  # integral of (1 - k t) over 0..4 = 4 - 8k = 4*factor
    poly = np.vstack([[0.0, 0.0], base[:, :2]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    mid = T - DT / 2
    new_arc = np.cumsum(seg * (1 - k * mid))
    if arc[-1] < 1e-6:
        return base[:, :2].copy()
    return np.stack([np.interp(new_arc, arc, poly[:, 0]), np.interp(new_arc, arc, poly[:, 1])], axis=1)


def at_speed(pick: np.ndarray, v0: float, factor: float) -> np.ndarray:
    """The pick's path geometry, driven from the current speed v0 (held, or braking to keep `factor` of the distance).

    The path is extended straight along its last heading when the pick itself is shorter.
    """
    k = (1 - factor) / 2.0
    dist = v0 * (T - k * T**2 / 2)
    poly = np.vstack([[0.0, 0.0], pick[:, :2]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] < 0.5:  # a stopped pick has no direction of its own: straight ahead
        poly, arc = np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.0, 1.0])
    d = poly[-1] - poly[-2] if np.linalg.norm(poly[-1] - poly[-2]) > 1e-6 else np.array([1.0, 0.0])
    d = d / np.linalg.norm(d)
    extra = max(0.0, dist.max() - arc[-1]) + 1.0
    poly = np.vstack([poly, poly[-1] + extra * d])
    arc = np.r_[arc, arc[-1] + extra]
    return np.stack([np.interp(dist, arc, poly[:, 0]), np.interp(dist, arc, poly[:, 1])], axis=1)


def vocabulary(pick: np.ndarray, v0: float | None = None) -> tuple[np.ndarray, list[dict]]:
    """32 paths from the pick. v0=None: version A, the pick's own speed. v0 given: version B, the current speed."""
    paths, meta = [], []
    for speed in SPEEDS:
        along = retime(pick, speed) if v0 is None else at_speed(pick, v0, speed)
        for shape in SHAPES:
            frac = lateral(shape, T)
            for off in OFFSETS:
                xy = along.copy()
                xy[:, 1] = xy[:, 1] + off * frac
                step = np.diff(np.vstack([[0.0, 0.0], xy]), axis=0)
                heading = np.arctan2(step[:, 1], step[:, 0])
                paths.append(np.column_stack([xy, heading]))
                meta.append({"offset": off, "speed": speed, "shape": shape})
    return np.stack(paths), meta


def category(v, ) -> str:
    if v.hit_at is not None:
        return "hit"
    if v.off_road_at is not None:
        return "off-road"
    if v.max_deviation > OFF_LANE_M:
        return "off-lane"
    return "clear"


def dump_index() -> dict[int, Path]:
    index = {}
    for path in DUMPS.glob("*.jsonl"):
        with path.open() as f:
            for line in f:
                index[json.loads(line)["timestamp"]] = path
    return index


def dump_rows(episode: Path, index: dict[int, Path]) -> dict[int, dict]:
    plans = json.loads((episode / "trajectories.json").read_text())
    stamps = sorted(int(t) for t in plans)
    files = {index[s] for s in stamps if s in index}
    for path in files:
        rows = {r["timestamp"]: r for r in map(json.loads, path.read_text().splitlines())}
        if set(stamps) <= set(rows):
            drift = max(np.abs(np.asarray(rows[s]["trajectory"])[:6, :2] - np.asarray(plans[str(s)])).max() for s in stamps)
            if drift < 1e-3:
                return rows
    raise LookupError(f"no dump matches {episode}")


def to_ego(points_world: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    rot, origin = ego2world[:2, :2], ego2world[:2, 3]
    return (np.asarray(points_world)[:, :2] - origin) @ rot


def project(plan_xy: np.ndarray, cam2ego, K):
    """The engine's FC_TRAJ projection (as ops/ncap_viewer/build.py:_draw_plan), returning pixel runs."""
    plan = np.vstack([[0.0, 0.0], plan_xy])
    ego2cam = np.linalg.inv(cam2ego)
    cam2img = np.eye(4)
    cam2img[:3, :3] = K
    ego2img = cam2img @ ego2cam
    s = np.linspace(0, 1, len(plan) * 20)
    x = np.interp(s, np.linspace(0, 1, len(plan)), plan[:, 0])
    y = np.interp(s, np.linspace(0, 1, len(plan)), plan[:, 1])
    pts = np.stack([x, y, np.full_like(x, 0.25), np.ones_like(x)], axis=1)
    cam = (ego2cam @ pts.T).T
    img = (ego2img @ pts.T).T
    uv = img[:, :2] / img[:, 2:3]
    keep = (cam[:, 2] > 0.5) & (uv > -400).all(1) & (uv < [2000, 1300]).all(1)
    return uv, keep


def draw_line(draw, uv, keep, rgb, width, dashed=False):
    run = []
    for i, (p, k) in enumerate(zip(uv, keep)):
        if k and (not dashed or (i // 12) % 2 == 0):
            run.append(tuple(p))
        elif len(run) > 1:
            draw.line(run, fill=rgb, width=width, joint="curve")
            run = []
        else:
            run = []
    if len(run) > 1:
        draw.line(run, fill=rgb, width=width, joint="curve")


def hex_rgb(h: str, a: int = 255):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) + (a,)


def render_camera(frame: Path, scene: str, natives, pick, vocab, meta, cats, out: Path):
    from PIL import Image, ImageDraw

    calib = json.loads((CALIB / f"calib_{scene}.json").read_text())
    cam2ego, K = np.asarray(calib["cam2ego"]), np.asarray(calib["K"])
    img = Image.open(frame).convert("RGB")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)
    for p in natives:
        uv, keep = project(p[:, :2], cam2ego, K)
        draw_line(draw, uv, keep, (235, 235, 235, 70), 2)
    for p, m, c in zip(vocab, meta, cats):
        if m["speed"] != 1.0 or m["shape"] != "hold":
            continue
        uv, keep = project(p[:, :2], cam2ego, K)
        draw_line(draw, uv, keep, (0, 0, 0, 150), 8, dashed=m["shape"] == "return")
        draw_line(draw, uv, keep, hex_rgb(STATUS[c]), 5, dashed=m["shape"] == "return")
    uv, keep = project(pick[:, :2], cam2ego, K)
    draw_line(draw, uv, keep, (0, 0, 0, 200), 11)
    draw_line(draw, uv, keep, (255, 255, 255, 255), 7)
    img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    img = img.resize((1200, 675))
    img.save(out, quality=86)


def render_bev(ego2world, ts, vehicles, target_ids, reference, area, natives, pick, vocab, meta, cats, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    surface, ink, muted, grid_ink = "#1a1a19", "#ffffff", "#c3c2b7", "#3a3a37"
    fig, axes = plt.subplots(1, 4, figsize=(16, 7.4), facecolor=surface)
    xs = np.arange(-8, 48, 0.4)
    ys = np.arange(-16, 16, 0.4)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    rot, origin = ego2world[:2, :2], ego2world[:2, 3]
    world = np.stack([gx.ravel(), gy.ravel()], 1) @ rot.T + origin
    drivable = area.contains(world).reshape(gx.shape)
    ref_ego = to_ego(reference, ego2world)
    panels = [(1.0, "hold"), (1.0, "return"), (0.7, "hold"), (0.7, "return")]
    titles = {(1.0, "hold"): "own speed · move over and stay", (1.0, "return"): "own speed · pass and come back",
              (0.7, "hold"): "braking · move over and stay", (0.7, "return"): "braking · pass and come back"}
    for ax, (speed, shape) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.pcolormesh(-gy, gx, np.where(drivable, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#33332f"]),
                      shading="auto", zorder=0)
        ax.plot(-ref_ego[:, 1], ref_ego[:, 0], color=muted, lw=1.0, ls=(0, (4, 4)), zorder=1)
        for actor in vehicles:
            state = actor_at(actor, float(ts))
            if state is None:
                continue
            box = to_ego(corners(state[0], state[1], actor["dims"][1], actor["dims"][0]), ego2world)
            is_target = actor["uuid"] in target_ids
            ax.add_patch(Polygon(np.c_[-box[:, 1], box[:, 0]], closed=True, fill=is_target,
                                 fc="#d9d6cc" if is_target else "none", ec=ink if is_target else "#7d7c76",
                                 lw=1.4 if is_target else 0.9, zorder=3))
            if is_target:
                for k in (1, 2, 3, 4):
                    later = actor_at(actor, float(ts) + k * 1e6)
                    if later is None:
                        continue
                    c = to_ego(later[0][None], ego2world)[0]
                    box = to_ego(corners(later[0], later[1], actor["dims"][1], actor["dims"][0]), ego2world)
                    ax.add_patch(Polygon(np.c_[-box[:, 1], box[:, 0]], closed=True, fill=False, ec=ink, lw=0.8,
                                         ls=(0, (2, 2)), alpha=0.75, zorder=3))
                    ax.text(-c[1] + 1.4, c[0], f"+{k} s", color=muted, fontsize=8, va="center", zorder=6)
        ego = corners(np.array([0.5, 0.0]), 0.0, 4.018, 1.85)
        ax.add_patch(Polygon(np.c_[-ego[:, 1], ego[:, 0]], closed=True, fill=False, ec=ink, lw=1.2, zorder=4))
        for p in natives:
            q = np.vstack([[0, 0], p[:, :2]])
            ax.plot(-q[:, 1], q[:, 0], color="#8f8e88", lw=0.7, alpha=0.5, zorder=2)
        counts = {c: 0 for c in ORDER}
        for p, m, c in zip(vocab, meta, cats):
            if m["speed"] != speed or m["shape"] != shape:
                continue
            counts[c] += 1
            q = np.vstack([[0, 0], p[:, :2]])
            ax.plot(-q[:, 1], q[:, 0], color=surface, lw=3.6, zorder=4, solid_capstyle="round")
            ax.plot(-q[:, 1], q[:, 0], color=STATUS[c], lw=2.0, zorder=5, solid_capstyle="round")
            ax.plot(-q[-1, 1], q[-1, 0], "o", ms=3.5, color=STATUS[c], zorder=5)
        q = np.vstack([[0, 0], pick[:, :2]])
        ax.plot(-q[:, 1], q[:, 0], color=surface, lw=4.6, zorder=6)
        ax.plot(-q[:, 1], q[:, 0], color=ink, lw=2.4, zorder=7)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-6, 44)
        ax.set_aspect("equal")
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid_ink)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_xticklabels(["10 L", "5 L", "0", "5 R", "10 R"])
        ax.grid(color=grid_ink, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(titles[(speed, shape)], color=ink, fontsize=10.5, loc="left", pad=20)
        summary = "   ".join(f"{c} {counts[c]}" for c in ORDER if counts[c])
        ax.text(0, 1.012, summary, transform=ax.transAxes, color=muted, fontsize=8.5, va="bottom")
    axes[0].set_ylabel("metres ahead", color=muted, fontsize=9)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out, dpi=100, facecolor=surface)
    plt.close(fig)


def main() -> None:
    out_dir = Path(sys.argv[1])
    episodes = [a for a in sys.argv[2:] if not a.startswith("--")] or EPISODES
    out_dir.mkdir(parents=True, exist_ok=True)
    index = dump_index()
    report = []
    for name in episodes:
        shape, scene_no = name.split("-")
        scene = f"scene-{scene_no}"
        area = DrivableArea.load(DRIVABLE / f"{scene}.npz")
        chosen = None
        for run in range(10):
            episode = COLLECT / name / f"run_{run}"
            label_file = LABELS / f"{name}__run_{run}.jsonl"
            if not label_file.exists():
                continue
            labels = {r["timestamp"]: r for r in map(json.loads, label_file.read_text().splitlines())}
            rows = dump_rows(episode, index)
            poses = json.loads((episode / "ego_poses.json").read_text())
            actors = json.loads((episode / "actors.json").read_text())
            vehicles = [a for a in actors if a["dims"][1] >= VEHICLE_MIN_LENGTH]
            targets = [a for a in actors if a["is_target_actor"]]
            logged, lanes = load_references(episode)
            best = None
            fallback = None
            per_tick = []
            for ts, lab in sorted(labels.items()):
                if ts not in rows or str(ts) not in poses:
                    continue
                ego2world = np.asarray(poses[str(ts)])
                reference, ref_kind = reference_at(ts, logged, lanes)
                row = rows[ts]
                natives = np.asarray(row["proposals"], dtype=float)
                pick_i = int(np.argmax(row["pdm_score"]))
                pick = natives[pick_i]
                speed = float(row["speed"])
                vocab_a, meta = vocabulary(pick)
                vocab, _ = vocabulary(pick, speed)
                here = ego2world[:2, 3]
                near = [a for a in vehicles if (s := actor_at(a, float(ts))) is not None and np.linalg.norm(s[0] - here) < 70]
                nat_v = [grade(p, ego2world, ts, near, reference, area) for p in natives]
                voc_v = [grade(p, ego2world, ts, near, reference, area) for p in vocab]
                voc_a = [category(grade(p, ego2world, ts, near, reference, area)) for p in vocab_a]
                nat_c = [category(v) for v in nat_v]
                voc_c = [category(v) for v in voc_v]
                cv = np.column_stack([speed * T, np.zeros(8), np.zeros(8)])
                cv_hit = grade(cv, ego2world, ts, targets).hit_at
                rec = {"ts": ts, "speed": speed, "in_frame": lab.get("hazard_in_frame"), "bearing": lab.get("hazard_bearing_deg"),
                       "pick": nat_c[pick_i], "pick_hit": nat_v[pick_i].hit_at, "cv_hit": cv_hit,
                       "native_clear": nat_c.count("clear"), "vocab": {c: voc_c.count(c) for c in ORDER},
                       "vocab_clear_or_offlane": sum(c in ("clear", "off-lane") for c in voc_c),
                       "vocab_a": {c: voc_a.count(c) for c in ORDER},
                       "vocab_a_ok": sum(c in ("clear", "off-lane") for c in voc_a),
                       "pick_travel": float(np.linalg.norm(np.diff(np.vstack([[0, 0], pick[:, :2]]), axis=0), axis=1).sum()),
                       "native_ok": sum(c in ("clear", "off-lane") for c in nat_c), "ref": ref_kind}
                per_tick.append(rec)
                state = actor_at(targets[0], float(ts)) if targets else None
                if state is not None and lab.get("hazard_in_frame") and speed >= 1.5:
                    ahead = to_ego(state[0][None], ego2world)[0]
                    if ahead[0] > 8:
                        gap = abs(np.linalg.norm(ahead) - 20.0)
                        if fallback is None or gap < fallback[0]:
                            fallback = (gap, ts, ego2world, reference, natives, pick, vocab, meta, voc_c, rec, lab, vocab_a, voc_a)
                if lab.get("hazard_in_frame") and speed >= 3 and cv_hit is not None:
                    gap = abs(cv_hit - TARGET_TTC_S)
                    if best is None or gap < best[0]:
                        best = (gap, ts, ego2world, reference, natives, pick, vocab, meta, voc_c, rec, lab, vocab_a, voc_a)
            if best is not None:
                chosen = (run, episode, vehicles, {a["uuid"] for a in targets}, best, per_tick, "rule")
                break
            if fallback is not None and len(sys.argv) > 2 and "--fallback" in sys.argv:
                chosen = (run, episode, vehicles, {a["uuid"] for a in targets}, fallback, per_tick, "fallback")
                break
        if chosen is None:
            print(f"{name}: no qualifying tick in runs 0-9")
            continue
        run, episode, vehicles, target_ids, best, per_tick, rule = chosen
        _, ts, ego2world, reference, natives, pick, vocab, meta, voc_c, rec, lab, vocab_a, voc_a = best
        stem = f"{name}_run{run}_{ts}"
        render_camera(Path(lab["frame"]), scene, natives, pick, vocab, meta, voc_c, out_dir / f"{stem}_cam.jpg")
        render_bev(ego2world, ts, vehicles, target_ids, reference, area, natives, pick, vocab, meta, voc_c,
                   out_dir / f"{stem}_bev.png")
        if name == "frontal-0053":
            render_bev(ego2world, ts, vehicles, target_ids, reference, area, natives, pick, vocab_a, meta, voc_a,
                       out_dir / f"{stem}_bev_A.png")
        hits = [r for r in per_tick if r["pick_hit"] is not None and r["in_frame"]]
        summary = {
            "episode": name, "run": run, "tick": ts, "stem": stem, "tick_detail": rec, "rule": rule,
            "t_in_episode": (ts - per_tick[0]["ts"]) / 1e6,
            "vocab_at_tick": [{**m, "verdict": c} for m, c in zip(meta, voc_c)],
            "episode_ticks": len(per_tick),
            "pick_hit_ticks_in_view": len(hits),
            "of_those_native_ok": sum(r["native_ok"] > 0 for r in hits),
            "of_those_vocab_ok": sum(r["vocab_clear_or_offlane"] > 0 for r in hits),
            "of_those_vocab_clear": sum(r["vocab"]["clear"] > 0 for r in hits),
            "of_those_vocab_a_ok": sum(r["vocab_a_ok"] > 0 for r in hits),
            "of_those_vocab_a_clear": sum(r["vocab_a"]["clear"] > 0 for r in hits),
            "per_tick": per_tick,
        }
        report.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("vocab_at_tick", "per_tick")}), flush=True)
    (out_dir / f"report_{'_'.join(episodes)}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
