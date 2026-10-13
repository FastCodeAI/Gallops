"""Author inserted-actor scenarios for scenes that are not in any benchmark.

neuroNCAP's own twenty scenarios were placed by hand. We need the same three
shapes on five reconstructed train scenes, so they are computed instead: the
ego's logged path says where a hazard has to be to become one.

A scenario does not invent a vehicle. It takes a car the reconstruction already
holds -- `uuid` is the nuScenes instance token
(`neuro_ncap/components/nuscenes_api.py`) -- and sends it through two to four
timed global waypoints that the engine smooths into a trajectory.

The engine's clock, which the emitted `times` must agree with
(`neuro_ncap/components/scenario.py`):

* a *frame* is a nuScenes keyframe, so 2 Hz, and `start-frame` indexes those;
* `times` are seconds from the scene's FIRST frame, the same clock the engine
  builds `query_times` on, so they must bracket
  [start_offset, start_offset + duration] or the ends get extrapolated;
* `positions` are global XYZ at GROUND level -- the engine adds half the actor's
  height itself;
* heading comes from consecutive interpolated positions, so a car that never
  moves has none and needs `initial-yaw`.

Three shapes, each aimed at a habit measured in
paddock's `docs/models/drivejepa-neuroncap.md`:

    stationary   parked on the path            -> the 1.3 m/s creep
    frontal      driving back down it at us    -> no lateral evasion, the real gap
    side         crossing, timed to meet us    -> the blind flank

    python -m gallops.expert.scenarios --scene scene-0075 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from gallops.config import SiteConfig

#: Seconds into the episode when the ego should reach the hazard. Long enough
#: that an evasion is still possible -- the teacher's own failures are confined
#: to the last second -- and short enough to stay inside one scenario.
MEET_S = 5.0
DURATION_S = 8.0
PRIMING_STEPS = 2
TIME_STEP = 0.1  # match the benchmark: scenarios_10hz steps at 10 Hz, and the
#: model is evaluated there, so training data collected at 2 Hz would be a
#: different control problem (and five times fewer samples per episode).
#: A crossing actor starts this far off the path and ends as far past it.
SIDE_OFFSET_M = 18.0
#: An oncoming actor starts this far beyond the meeting point.
FRONTAL_LEAD_M = 45.0
#: Annotated for at least this long to be worth re-routing: a poorly observed
#: instance is a poorly reconstructed one.
MIN_ANNOTATIONS = 10
#: The reconstruction's own threshold for "this actor moves", and so for whether
#: it exists as something the renderer can place elsewhere.
MIN_MOTION_STD_M = 0.50
#: Classes that are BOTH reconstructible (`ALLOWED_RIGID_CLASSES` in nerfstudio's
#: nuScenes dataparser) and scored as a collision by the benchmark
#: (`COLLISION_CLASSES_TO_CHECK` in neuro_ncap/evaluation/collision.py). Bicycles
#: and motorcycles qualify on both counts but are left out here: they are too
#: narrow to make a lane-blocking hazard, so a swerve around one teaches little.
HAZARD_CLASSES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer")


@dataclass
class Scene:
    name: str
    token: str
    first_timestamp: int
    times: np.ndarray  # seconds from the first frame, per keyframe
    positions: np.ndarray  # (N, 3) global ego positions per keyframe
    ground_z: float


def _tables(root: Path) -> dict:
    return {
        name: json.loads((root / "v1.0-trainval" / f"{name}.json").read_text())
        for name in (
            "scene",
            "sample",
            "sample_data",
            "ego_pose",
            "sample_annotation",
            "instance",
            "category",
            "log",
        )
    }


def load_scene(root: Path, name: str, tables: dict) -> Scene:
    scene = next((s for s in tables["scene"] if s["name"] == name), None)
    if scene is None:
        raise SystemExit(f"no scene named {name}")
    samples = {s["token"]: s for s in tables["sample"] if s["scene_token"] == scene["token"]}
    lidar = {
        r["sample_token"]: r
        for r in tables["sample_data"]
        if r["sample_token"] in samples and r["is_key_frame"] and "LIDAR_TOP" in r["filename"]
    }
    poses = {p["token"]: p["translation"] for p in tables["ego_pose"]}

    ordered, token = [], scene["first_sample_token"]
    while token:
        ordered.append(samples[token])
        token = samples[token]["next"]
    stamps = np.array([s["timestamp"] for s in ordered], dtype=np.int64)
    xyz = np.array([poses[lidar[s["token"]]["ego_pose_token"]] for s in ordered], dtype=float)
    return Scene(
        name=name,
        token=scene["token"],
        first_timestamp=int(stamps[0]),
        times=(stamps - stamps[0]) / 1e6,
        positions=xyz,
        ground_z=float(np.median(xyz[:, 2])),
    )


def pick_actor(root: Path, scene: Scene, tables: dict) -> tuple[str, list[float]]:
    """The best-observed MOVING car in the scene.

    A re-routed actor is rendered from what the reconstruction learned about it,
    so more observations is better. But the reconstruction keeps only actors that
    move: `_traj_dict_to_list` in nerfstudio's nuScenes dataparser drops anything
    whose position standard deviation stays under 0.50 m, and a dropped actor is
    not in the renderer's list, so the engine dies with a bare KeyError on its
    uuid. Ranking on observation count alone picks parked cars -- annotated in
    every frame, never moving, and never reconstructed. So the same test is
    applied here, and only then does the count decide.
    """
    categories = {c["token"]: c["name"] for c in tables["category"]}
    instance_category = {i["token"]: categories[i["category_token"]] for i in tables["instance"]}
    scene_samples = {s["token"] for s in tables["sample"] if s["scene_token"] == scene.token}

    per_instance: dict[str, list[dict]] = {}
    for a in tables["sample_annotation"]:
        if a["sample_token"] in scene_samples and instance_category.get(a["instance_token"], "").startswith(
            HAZARD_CLASSES
        ):
            per_instance.setdefault(a["instance_token"], []).append(a)

    ranked = sorted(per_instance.items(), key=lambda kv: -len(kv[1]))
    for uuid, anns in ranked:
        if len(anns) < MIN_ANNOTATIONS:
            break
        track = np.array([a["translation"][:2] for a in anns], dtype=float)
        if not (track.std(axis=0) > MIN_MOTION_STD_M).any():
            continue  # parked: the reconstruction never learned to move it
        size = np.median([a["size"] for a in anns], axis=0)  # w, l, h
        if 1.4 < size[0] < 3.0 and 3.0 < size[1] < 13.0:
            return uuid, [round(float(v), 2) for v in size]
    raise SystemExit(
        f"{scene.name}: no moving {'/'.join(HAZARD_CLASSES)} annotated >= {MIN_ANNOTATIONS} times "
        "with plausible dimensions -- the reconstruction can only re-route actors it saw move"
    )


def choose_actor(scene: Scene, tables: dict, prefix: str) -> tuple[str, list[float]]:
    """A named car instead of pick_actor's choice, held to the same tests.

    pick_actor ranks by annotation count, and in scene-0053 a tie went to a car
    the log only ever saw 43-84 m away from behind, which NeuRAD renders as a
    blob at 5 m. A car seen close from the side the scenario shows has to be
    chosen by hand until the choice is made by view coverage.
    """
    categories = {c["token"]: c["name"] for c in tables["category"]}
    instance_category = {i["token"]: categories[i["category_token"]] for i in tables["instance"]}
    scene_samples = {s["token"] for s in tables["sample"] if s["scene_token"] == scene.token}
    anns = [
        a
        for a in tables["sample_annotation"]
        if a["sample_token"] in scene_samples and a["instance_token"].startswith(prefix)
    ]
    uuids = {a["instance_token"] for a in anns}
    if len(uuids) != 1:
        raise SystemExit(f"{scene.name}: {len(uuids)} instances start with {prefix!r}")
    uuid = uuids.pop()
    if not instance_category.get(uuid, "").startswith(HAZARD_CLASSES):
        raise SystemExit(f"{scene.name}: {uuid[:8]} is a {instance_category.get(uuid)}, not a hazard class")
    track = np.array([a["translation"][:2] for a in anns], dtype=float)
    if len(anns) < MIN_ANNOTATIONS or not (track.std(axis=0) > MIN_MOTION_STD_M).any():
        raise SystemExit(
            f"{scene.name}: {uuid[:8]} never moves or is barely seen; the reconstruction cannot move it"
        )
    size = np.median([a["size"] for a in anns], axis=0)
    return uuid, [round(float(v), 2) for v in size]


def _project(lane: np.ndarray, point: np.ndarray) -> float:
    """Distance along `lane` of the point on it nearest `point`."""
    steps = np.diff(lane, axis=0)
    length_sq = np.maximum((steps**2).sum(1), 1e-9)
    t = np.clip(((point - lane[:-1]) * steps).sum(1) / length_sq, 0.0, 1.0)
    feet = lane[:-1] + t[:, None] * steps
    i = int(np.argmin(np.linalg.norm(feet - point, axis=1)))
    arc = np.r_[0.0, np.cumsum(np.sqrt(length_sq))]
    return float(arc[i] + t[i] * np.sqrt(length_sq[i]))


def _along(lane: np.ndarray, distance: float) -> tuple[np.ndarray, float]:
    """The point `distance` along `lane`, and the lane's heading there."""
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(lane, axis=0), axis=1))]
    point = np.array([np.interp(distance, arc, lane[:, k]) for k in (0, 1)])
    ahead = np.array([np.interp(distance + 1.0, arc, lane[:, k]) for k in (0, 1)])
    step = ahead - point
    return point, float(math.atan2(step[1], step[0]))


def _arc(scene: Scene) -> np.ndarray:
    steps = np.linalg.norm(np.diff(scene.positions[:, :2], axis=0), axis=1)
    return np.r_[0.0, np.cumsum(steps)]


def _at_time(scene: Scene, seconds: float) -> np.ndarray:
    """Where the ego was, in the plane, at a time on the scene clock."""
    return np.array([np.interp(seconds, scene.times, scene.positions[:, k]) for k in (0, 1)])


def _at_distance(scene: Scene, start_s: float, extra_m: float) -> np.ndarray:
    """A point `extra_m` further along the logged path than the ego at `start_s`."""
    arc = _arc(scene)
    here = float(np.interp(start_s, scene.times, arc))
    return np.array([np.interp(here + extra_m, arc, scene.positions[:, k]) for k in (0, 1)])


def _speed(scene: Scene, seconds: float) -> float:
    """The ego's own speed there, so an oncoming car closes at a plausible rate."""
    a, b = _at_time(scene, seconds - 0.5), _at_time(scene, seconds + 0.5)
    return float(np.linalg.norm(b - a))


def _heading(scene: Scene, seconds: float) -> float:
    a, b = _at_time(scene, seconds), _at_time(scene, seconds + 0.5)
    step = b - a
    return float(math.atan2(step[1], step[0])) if np.linalg.norm(step) > 1e-3 else 0.0


def build(
    scene: Scene, uuid: str, dims: list[float], start_frame: int, lane: np.ndarray | None = None
) -> dict[str, dict]:
    """The three shapes for one scene, as scenario bodies ready to dump.

    With `lane` (the centreline of the lane the car starts in, world frame), the
    hazard is placed in THAT lane, as far along it as the human drove by the
    meeting time. Without it, on the human's path -- which in scene-0058 is one
    lane over by then, leaving our lane empty.
    """
    start_s = float(scene.times[start_frame])
    meet_s = start_s + MEET_S
    meet = _at_time(scene, meet_s)
    heading = _heading(scene, meet_s)
    if lane is not None:
        arc = _arc(scene)
        travelled = float(np.interp(meet_s, scene.times, arc) - np.interp(start_s, scene.times, arc))
        meet_along = _project(lane, _at_time(scene, start_s)) + travelled
        meet, heading = _along(lane, meet_along)
    forward = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-forward[1], forward[0]])
    z = round(scene.ground_z, 2)

    def point(xy):
        return [round(float(xy[0]), 2), round(float(xy[1]), 2), z]

    general = {
        "start-pose": "original",
        "start-velo": "original",
        "start-frame": start_frame,
        "time-step": TIME_STEP,
        "priming-steps": PRIMING_STEPS,
        "duration": DURATION_S,
        "smooth-actors": True,
    }
    # Times bracket the whole window the engine queries, so nothing is extrapolated.
    span = [round(start_s - 1.0, 2), round(start_s + DURATION_S + 1.0, 2)]

    # Parked on the path, facing along the road: the creep has something to creep into.
    stationary = {
        "general": {**general, "start-jitter": 1},
        "actors": [
            {
                "mode": "override",
                "uuids": [uuid],
                "dims": dims,
                "is_target_actor": True,
                "initial-yaw": round(heading, 4),
                "times": span,
                "positions": [point(meet), point(meet)],
                "jitter": {"longitudinal": 5.0, "lateral": 1.0, "long-lat-angle": round(heading, 4)},
            }
        ],
    }

    # Oncoming in our lane: starts beyond the meeting point and drives back at us,
    # so a full stop cannot save the score and only a swerve does.
    #
    # It has to FOLLOW the road, not cut across it. Two waypoints draw a straight
    # chord, which on a curve leaves the actor metres off the lane and the ego
    # sails past -- that missed on two of five scenes. So the path is sampled at
    # several points and the actor drives down it.
    speed = max(_speed(scene, meet_s), 3.0)
    offsets = np.linspace(FRONTAL_LEAD_M, -12.0, 6)
    if lane is not None:
        frontal_positions = [point(_along(lane, meet_along + float(d))[0]) for d in offsets]
    else:
        frontal_positions = [point(_at_distance(scene, meet_s, float(d))) for d in offsets]
    # it passes the meeting point at meet_s, so its time at each offset follows
    # from closing at its own speed
    frontal_times = [round(float(meet_s - d / speed), 2) for d in offsets]
    frontal = {
        "general": general,
        "actors": [
            {
                "mode": "override",
                "uuids": [uuid],
                "dims": dims,
                "is_target_actor": True,
                "times": frontal_times,
                "positions": frontal_positions,
                "jitter": {"longitudinal": 8.0, "lateral": 0.75, "long-lat-angle": round(heading, 4)},
            }
        ],
    }

    # Crossing, timed to arrive as we do.
    side = {
        "general": general,
        "actors": [
            {
                "mode": "override",
                "uuids": [uuid],
                "dims": dims,
                "is_target_actor": True,
                "times": [round(meet_s - SIDE_OFFSET_M / 8.0, 2), round(meet_s + SIDE_OFFSET_M / 8.0, 2)],
                "positions": [point(meet + left * SIDE_OFFSET_M), point(meet - left * SIDE_OFFSET_M)],
                "jitter": {
                    "longitudinal": 2.0,
                    "lateral": 1.0,
                    "long-lat-angle": round(heading + math.pi / 2, 4),
                },
            }
        ],
    }
    return {"stationary": stationary, "frontal": frontal, "side": side}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", action="append", required=True, help="scene-0075, repeatable")
    parser.add_argument("--start-frame", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actor", help="instance token (or its prefix) of the car to move; one --scene only")
    parser.add_argument(
        "--lane", action="store_true", help="place the hazard in the car's own lane (needs nuscenes-devkit)"
    )
    args = parser.parse_args()
    if args.actor and len(args.scene) != 1:
        raise SystemExit("--actor names a car in one scene: pass exactly one --scene")

    root = Path(SiteConfig.load().resolve("{data.nuscenes}"))
    tables = _tables(root)
    for name in args.scene:
        scene = load_scene(root, name, tables)
        uuid, dims = (
            choose_actor(scene, tables, args.actor) if args.actor else pick_actor(root, scene, tables)
        )
        lane = None
        if args.lane:
            from nuscenes.map_expansion.map_api import NuScenesMap

            from .lanes import lane_path

            logs = {entry["token"]: entry["location"] for entry in tables["log"]}
            record = next(s for s in tables["scene"] if s["name"] == name)
            start_s = float(scene.times[args.start_frame])
            x, y = _at_time(scene, start_s)
            nusc_map = NuScenesMap(dataroot=str(root), map_name=logs[record["log_token"]])
            lane = lane_path(nusc_map, float(x), float(y), _heading(scene, start_s), scene.positions[:, :2])
            if lane is None:
                raise SystemExit(f"{name}: no lane aligned with the car at the start frame")
        bodies = build(scene, uuid, dims, args.start_frame, lane)
        for shape, body in bodies.items():
            path = args.out / shape / f"{name.removeprefix('scene-')}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False, default_flow_style=None))
        where = "its own lane" if lane is not None else "the logged path"
        print(f"{name}: actor {uuid[:8]} dims {dims}, hazard on {where} -> {args.out}")


if __name__ == "__main__":
    main()
