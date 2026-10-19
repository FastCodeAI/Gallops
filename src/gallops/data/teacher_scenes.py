"""Phase 1 of the closed-loop expert-data plan: pick nuScenes-train scenes to
reconstruct with NeuRAD for the privileged teacher's training data.

Reads the full nuScenes trainval devkit, scores every train-split scene not
already spoken for (paddock's own neuroNCAP eval set, and the scenes already
reconstructed for the swerve finetune), and picks a diverse set of training
scenes plus a disjoint held-out validation set. Writes one CAM_FRONT video per
selected scene and a markdown rationale next to it, so a human can review the
picks before Phase 2 (NeuRAD reconstruction) starts.

Turn/curve/straight classification reuses the exact 3 s-horizon, 2.0 m lateral
rule paddock's `adapters/benchmarks/nuscenes/loader.py:command_from_future`
uses to derive nav commands -- the same rule paddock's open-loop
failure-mode writeup's `turn_radius_given_correct_direction` finding is built
on -- rather than a from-scratch heading heuristic. A scene only counts as a
genuine turn if that per-sample command stays LEFT or RIGHT for a sustained
run, not a single noisy sample.

A sample with no full HORIZON_SAMPLES-ahead window remaining (the last ~3 s of
every scene) is excluded from every category entirely, the same way paddock's
nuScenes harness drops those samples rather than scoring them against a
shrunk, shorter-duration lookahead -- see `_turn_classification_flags`.

Run inside an env that carries nuscenes-devkit (not this one -- `make setup`
deliberately leaves torch and nuscenes-devkit out, see pyproject.toml):

    /path/to/envs/ncap/bin/python3 -m gallops.data.teacher_scenes \\
        --dataroot /path/to/nuscenes --version v1.0-trainval \\
        --out /path/to/output-dir
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from gallops.data.nuscenes_real import RECONSTRUCTED, neuroncap_scenes

# Scene descriptions containing any of these are skipped outright: dense
# parking-lot maneuvering isn't what this pass is after.
SKIP_KEYWORDS = ("parking lot", "night", "rain", "wet")

# nuScenes visibility tokens: '1'=0-40%, '2'=40-60%, '3'=60-80%, '4'=80-100%.
GOOD_VISIBILITY = {"3", "4"}
VEHICLE_PREFIX = "vehicle."
CLOSE_RANGE_M = 25.0
GOOD_ACTOR_MIN_FRAMES = 3
GOOD_ACTOR_MIN_LIDAR_PTS = 5

# Mirrors paddock's adapters/benchmarks/nuscenes/loader.py:TURN_THRESHOLD_M
# (VAD's gt_ego_fut_cmd rule): lateral offset of the point 3 s (HORIZON_SAMPLES
# at 2 Hz keyframes) ahead, in the ego frame at the current sample.
HORIZON_SAMPLES = 6
TURN_THRESHOLD_M = 2.0
CURVE_THRESHOLD_M = 0.8
# A turn only counts if the per-sample LEFT/RIGHT command holds for this many
# consecutive samples (2 s) -- rules out a single noisy crossing.
SUSTAINED_TURN_MIN_SAMPLES = 4
SUSTAINED_CURVE_MIN_SAMPLES = 6
STRAIGHT_FRACTION_MIN = 0.85
# The turn/curve run must not start in the scene's first 3 s or end in its
# last 3 s -- a scene that's already mid-turn at sample 0 has no lead-in to
# show what's approaching, and one that ends right at the turn has no
# aftermath either.
MIN_LEAD_IN_SAMPLES = 6
MIN_TRAIL_SAMPLES = 6

# Two scenes count as the same location if their ego tracks ever come within
# this many metres of each other -- different logs (different days) can still
# be the same street.
SAME_LOCATION_DIST_M = 25.0

# Scenes a human reviewer already watched and rejected; never pick these
# regardless of what the scoring says.
BLOCKLIST: dict[str, str] = {}

TARGET_TRAIN = 12
TARGET_VAL = 3

# name -> count to pick for each of train/val.
CATEGORY_QUOTAS_TRAIN = {
    "turn_left": 2,
    "turn_right": 2,
    "curved_road": 4,
    "straight": 4,
}
CATEGORY_QUOTAS_VAL = {
    "turn_left": 1,
    "curved_road": 1,
    "straight": 1,
}


@dataclass
class SceneMetrics:
    name: str
    scene_token: str
    log_token: str
    location: str
    description: str
    skipped_reason: str
    n_samples: int
    n_valid_samples: int
    longest_left_run: int
    longest_right_run: int
    longest_curve_run: int
    straight_fraction: float
    total_yaw_deg: float
    has_good_actor: bool
    good_actor_count: int
    category: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


def _heading_deg(rotation: list[float]) -> float:
    from nuscenes.eval.common.utils import quaternion_yaw
    from pyquaternion import Quaternion

    return math.degrees(quaternion_yaw(Quaternion(rotation)))


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _to_ego_frame(dx: float, dy: float, heading_deg: float) -> tuple[float, float]:
    """World-frame delta -> ego-frame (x forward, y left), matching the
    Trajectory convention paddock.policy.protocol defines."""
    h = math.radians(heading_deg)
    c, s = math.cos(h), math.sin(h)
    return c * dx + s * dy, -s * dx + c * dy


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Every maximal (start, end_inclusive) run of True in `flags`."""
    out = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _best_buffered_run_length(flags: list[bool], n: int) -> int:
    """Longest run that leaves >=MIN_LEAD_IN_SAMPLES before it starts and
    >=MIN_TRAIL_SAMPLES after it ends, so the scene has real context on both
    sides of the maneuver rather than starting or ending mid-turn."""
    best = 0
    for start, end in _runs(flags):
        if start < MIN_LEAD_IN_SAMPLES:
            continue
        if (n - 1 - end) < MIN_TRAIL_SAMPLES:
            continue
        best = max(best, end - start + 1)
    return best


def _turn_classification_flags(
    positions: list[tuple[float, float]], headings: list[float]
) -> tuple[list[bool], list[bool], list[bool], list[bool], int]:
    """Per-sample left/right/curve/straight flags, plus how many samples had a
    full HORIZON_SAMPLES-ahead window to score.

    A sample with fewer than HORIZON_SAMPLES ahead of it is excluded from
    every category -- not folded into "straight" -- mirroring paddock's own
    nuScenes harness, which drops any sample whose 3 s future horizon runs
    past the end of the scene rather than scoring it against a shrunk,
    shorter-duration lookahead. A shrunk window produces a systematically
    smaller lateral offset for the same physical turn rate, which both
    inflates straight_fraction with samples that measured a different
    (shorter) quantity than the rest of the scene and can erase a genuine
    turn signal that happens to fall in a scene's last ~3 s, before the
    run-length logic even sees it.
    """
    n = len(positions)
    n_valid = max(n - HORIZON_SAMPLES, 0)
    left_flags, right_flags, curve_flags, straight_flags = [], [], [], []
    for i in range(n):
        if i >= n_valid:
            left_flags.append(False)
            right_flags.append(False)
            curve_flags.append(False)
            straight_flags.append(False)
            continue
        j = i + HORIZON_SAMPLES
        dx = positions[j][0] - positions[i][0]
        dy = positions[j][1] - positions[i][1]
        _, lateral = _to_ego_frame(dx, dy, headings[i])
        left_flags.append(lateral >= TURN_THRESHOLD_M)
        right_flags.append(lateral <= -TURN_THRESHOLD_M)
        curve_flags.append(CURVE_THRESHOLD_M <= abs(lateral) < TURN_THRESHOLD_M)
        straight_flags.append(abs(lateral) < CURVE_THRESHOLD_M)
    return left_flags, right_flags, curve_flags, straight_flags, n_valid


def _score_scene(nusc, scene: dict) -> tuple[SceneMetrics, list[tuple[float, float]]]:
    description = scene["description"]
    skip_reason = next((kw for kw in SKIP_KEYWORDS if kw in description.lower()), "")
    if scene["name"] in BLOCKLIST:
        skip_reason = BLOCKLIST[scene["name"]]

    sample_token = scene["first_sample_token"]
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    samples: list[dict] = []
    while sample_token:
        sample = nusc.get("sample", sample_token)
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar["ego_pose_token"])
        positions.append((ego["translation"][0], ego["translation"][1]))
        headings.append(_heading_deg(ego["rotation"]))
        samples.append(sample)
        sample_token = sample["next"]

    n = len(samples)
    left_flags, right_flags, curve_flags, straight_flags, n_valid = _turn_classification_flags(
        positions, headings
    )
    total_yaw = sum(_angle_diff_deg(headings[i + 1], headings[i]) for i in range(n - 1))

    good_actors: dict[str, int] = {}
    if not skip_reason:
        for i, sample in enumerate(samples):
            ex, ey = positions[i]
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                if not ann["category_name"].startswith(VEHICLE_PREFIX):
                    continue
                dist = math.hypot(ann["translation"][0] - ex, ann["translation"][1] - ey)
                if (
                    dist <= CLOSE_RANGE_M
                    and ann["visibility_token"] in GOOD_VISIBILITY
                    and ann["num_lidar_pts"] >= GOOD_ACTOR_MIN_LIDAR_PTS
                ):
                    good_actors[ann["instance_token"]] = good_actors.get(ann["instance_token"], 0) + 1

    good_actor_count = sum(1 for v in good_actors.values() if v >= GOOD_ACTOR_MIN_FRAMES)
    log = nusc.get("log", scene["log_token"])

    m = SceneMetrics(
        name=scene["name"],
        scene_token=scene["token"],
        log_token=scene["log_token"],
        location=log["location"],
        description=description,
        skipped_reason=skip_reason,
        n_samples=n,
        n_valid_samples=n_valid,
        longest_left_run=_best_buffered_run_length(left_flags, n),
        longest_right_run=_best_buffered_run_length(right_flags, n),
        longest_curve_run=_best_buffered_run_length(curve_flags, n),
        straight_fraction=sum(straight_flags) / max(n_valid, 1),
        total_yaw_deg=total_yaw,
        has_good_actor=good_actor_count >= 1,
        good_actor_count=good_actor_count,
    )
    return m, positions


def _classify(m: SceneMetrics) -> str:
    if m.skipped_reason:
        return "skipped"
    if not m.has_good_actor:
        return "no_good_actor"
    if m.longest_left_run >= SUSTAINED_TURN_MIN_SAMPLES:
        return "turn_left"
    if m.longest_right_run >= SUSTAINED_TURN_MIN_SAMPLES:
        return "turn_right"
    if m.longest_curve_run >= SUSTAINED_CURVE_MIN_SAMPLES:
        return "curved_road"
    if m.straight_fraction >= STRAIGHT_FRACTION_MIN:
        return "straight"
    return "other"


def _overlaps_chosen(
    candidate_pos: list[tuple[float, float]],
    chosen_pos: list[list[tuple[float, float]]],
) -> bool:
    import numpy as np

    c = np.array(candidate_pos)
    for other in chosen_pos:
        o = np.array(other)
        # cheap bbox reject before the O(n*m) pairwise distance check
        if (
            c[:, 0].max() < o[:, 0].min() - SAME_LOCATION_DIST_M
            or c[:, 0].min() > o[:, 0].max() + SAME_LOCATION_DIST_M
            or c[:, 1].max() < o[:, 1].min() - SAME_LOCATION_DIST_M
            or c[:, 1].min() > o[:, 1].max() + SAME_LOCATION_DIST_M
        ):
            continue
        d = np.linalg.norm(c[:, None, :] - o[None, :, :], axis=-1)
        if d.min() < SAME_LOCATION_DIST_M:
            return True
    return False


def select_scenes(
    metrics: list[SceneMetrics], positions: dict[str, list[tuple[float, float]]]
) -> tuple[list[SceneMetrics], list[SceneMetrics]]:
    for m in metrics:
        m.category = _classify(m)

    pool = [m for m in metrics if m.category not in ("skipped", "no_good_actor")]
    used_names: set[str] = set()
    used_logs: set[str] = set()
    chosen_positions: list[list[tuple[float, float]]] = []

    def accept(m: SceneMetrics) -> bool:
        if m.name in used_names or m.log_token in used_logs:
            return False
        if _overlaps_chosen(positions[m.name], chosen_positions):
            return False
        used_names.add(m.name)
        used_logs.add(m.log_token)
        chosen_positions.append(positions[m.name])
        return True

    def take(category: str, n: int, dest: list[SceneMetrics]) -> None:
        candidates = sorted(
            (m for m in pool if m.category == category and m.name not in used_names),
            key=lambda m: m.good_actor_count,
            reverse=True,
        )
        got = 0
        for m in candidates:
            if got >= n:
                break
            if accept(m):
                dest.append(m)
                got += 1

    def fill_rest(n: int, dest: list[SceneMetrics]) -> None:
        rest = sorted(
            (m for m in pool if m.name not in used_names and m.category != "other"),
            key=lambda m: m.good_actor_count,
            reverse=True,
        )
        got = 0
        for m in rest:
            if got >= n:
                break
            if accept(m):
                dest.append(m)
                got += 1

    train: list[SceneMetrics] = []
    for category, n in CATEGORY_QUOTAS_TRAIN.items():
        take(category, n, train)
    fill_rest(TARGET_TRAIN - len(train), train)

    val: list[SceneMetrics] = []
    for category, n in CATEGORY_QUOTAS_VAL.items():
        take(category, n, val)
    fill_rest(TARGET_VAL - len(val), val)

    return train, val


def _render_video(nusc, out_dir: Path, m: SceneMetrics, fps: float = 4.0) -> Path:
    """Every keyframe (2 Hz, guaranteed present for the whole scene) as a
    full-length video, each frame held for two output frames so a 20 s scene
    plays back in ~10 s. Deliberately not `NuScenes.render_scene_channel`:
    that interpolates through the raw camera *sweep* chain, and several
    scenes have gaps in that chain that cut the video short with no error --
    keyframes never have that gap."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{m.name}.mp4"

    sample_token = nusc.get("scene", m.scene_token)["first_sample_token"]
    writer = None
    while sample_token:
        sample = nusc.get("sample", sample_token)
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = Path(nusc.dataroot) / cam["filename"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            sample_token = sample["next"]
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        sample_token = sample["next"]
    if writer is not None:
        writer.release()
    return out_path


def _write_rationale(
    out_dir: Path, train: list[SceneMetrics], val: list[SceneMetrics], excluded: int
) -> None:
    lines = [
        "# Phase 1 scene selection",
        "",
        f"{len(train)} training scenes, {len(val)} held-out validation scenes, "
        f"drawn from the nuScenes train split, excluding {excluded} scenes paddock's own "
        "neuroNCAP eval set or the swerve-finetune's already-reconstructed scenes already "
        "use, and any scene whose description mentions a parking lot, night, rain, or wet "
        "roads.",
        "",
        "Turn/curve/straight labels reuse the exact rule paddock's "
        "`adapters/benchmarks/nuscenes/loader.py:command_from_future` already uses for "
        "nav-command derivation: at every sample with a full 3 s (6-keyframe) lookahead, "
        "take the ego-frame lateral offset of the point 3 s ahead; >=2.0 m is a turn, "
        "0.8-2.0 m is a gentle curve, <0.8 m is straight. A sample in the last 3 s of a "
        "scene has no full lookahead and is excluded from every category, the same way "
        "the harness drops it, rather than scored against a shrunk window. A scene is "
        "only labeled `turn_left`/`turn_right` if that per-sample signal holds for >=2 s "
        "straight (not one noisy sample), and `curved_road` needs >=3 s of sustained "
        "curve-band offset. `straight` needs >=85% of *valid* samples reading straight. "
        "Scenes with no sustained signal in any of those are `other` and not picked. A "
        "turn/curve run only qualifies if it leaves >=3 s of lead-in before it starts and "
        ">=3 s of runout after it ends, so the scene never starts or ends mid-maneuver.",
        "",
        "Two scenes sharing a log (literally the same drive) or whose ego tracks ever "
        "come within 25 m of each other (same street, different day) are never both "
        "picked -- across train and validation together.",
        "",
        "**good actor** gate: at least one vehicle instance seen within 25 m, "
        "visibility >=60%, >=5 lidar points, in >=3 different frames -- just existence, "
        "not a count to maximize.",
        "",
        "## Training scenes",
        "",
        "| scene | city | category | longest turn/curve run (samples) | straight frac | "
        "good actors | description |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for m in train:
        run = max(m.longest_left_run, m.longest_right_run, m.longest_curve_run)
        lines.append(
            f"| {m.name} | {m.location} | {m.category} | {run} | "
            f"{m.straight_fraction:.2f} | {m.good_actor_count} | {m.description} |"
        )

    boston = sum(1 for m in train if m.location == "boston-seaport")
    singapore = len(train) - boston
    lines += [
        "",
        f"City mix: {boston} Boston, {singapore} Singapore.",
        "",
        "## Held-out validation scenes (checkpoint selection only, never neuroNCAP)",
        "",
        "| scene | city | category | good actors | description |",
        "|---|---|---|---:|---|",
    ]
    for m in val:
        lines.append(f"| {m.name} | {m.location} | {m.category} | {m.good_actor_count} | {m.description} |")

    lines += [
        "",
        "## What to check in the videos",
        "",
        "- Videos are the full scene at 2 Hz keyframes (not interpolated sweeps), so "
        "every one covers the whole ~20 s scene, turn included, with real context "
        "before and after it -- flag anything that still looks clipped.",
        "- Does a `turn_left`/`turn_right` scene actually show one committed, sustained "
        "turn, not several small wiggles that happened to chain together?",
        "- Is the flagged 'good actor' actually a clean, mostly-unoccluded vehicle worth "
        "reconstructing, not a sliver visible between two other cars?",
        "- Replace anything that looks wrong from `candidates.json` (every scored scene, "
        "including `other`/`skipped`/`no_good_actor`, is in there) before Phase 2 spends "
        "NeuRAD training time on it.",
    ]
    (out_dir / "SELECTION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory (a machine path; not defaulted here)"
    )
    parser.add_argument(
        "--dataroot",
        required=True,
        type=Path,
        help="nuScenes root (matches [datasets.nuscenes] in gallops.toml)",
    )
    parser.add_argument(
        "--version", default="v1.0-trainval", help="matches [datasets.nuscenes].version in gallops.toml"
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip video rendering (metrics + rationale only)"
    )
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)

    kept_out = {f"scene-{n}" for n in (neuroncap_scenes() | RECONSTRUCTED)}
    train_names = set(splits.train)
    candidate_scenes = [s for s in nusc.scene if s["name"] in train_names and s["name"] not in kept_out]
    print(
        f"scoring {len(candidate_scenes)} candidate train-split scenes "
        f"(excluded {len(kept_out)} already-used scenes)"
    )

    metrics = []
    positions: dict[str, list[tuple[float, float]]] = {}
    for i, scene in enumerate(candidate_scenes):
        m, pos = _score_scene(nusc, scene)
        metrics.append(m)
        positions[m.name] = pos
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(candidate_scenes)}")

    train, val = select_scenes(metrics, positions)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps([m.to_json() for m in metrics], indent=2))
    (args.out / "selected_train.json").write_text(json.dumps([m.to_json() for m in train], indent=2))
    (args.out / "selected_val.json").write_text(json.dumps([m.to_json() for m in val], indent=2))
    _write_rationale(args.out, train, val, len(kept_out))

    if not args.no_video:
        video_dir = args.out / "videos"
        for m in train + val:
            print(f"rendering {m.name}...")
            _render_video(nusc, video_dir, m)

    print(f"done: {len(train)} training scenes, {len(val)} validation scenes")
    print(f"review {args.out / 'SELECTION.md'} and {args.out / 'videos'}")


if __name__ == "__main__":
    main()
