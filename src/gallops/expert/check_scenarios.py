"""Would the inserted actor actually meet the ego? Geometry only, no renderer.

A scenario whose actor passes harmlessly behind the ego costs a full episode of
GPU and teaches nothing, and nothing in the YAML shows it. This replays the
actor exactly as the engine would -- same interpolation, same clock -- against
the ego's logged path, and reports how close the two boxes come.

It answers "is this scenario a hazard at all", not "does it render". A scenario
that passes here can still look wrong; one that fails here cannot be saved by
any amount of rendering.

The ego is assumed to drive its logged path, which is what it does before a
model reacts. A model that brakes or swerves will diverge -- that is the point of
the benchmark -- so treat the closest approach here as the hazard the scenario
sets up, not as a prediction of the outcome.

    python -m gallops.expert.check_scenarios --dir <scenario dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from gallops.config import SiteConfig

from .grader import EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE, corners, overlap
from .scenarios import _tables, load_scene


def actor_poses(actor: dict, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The engine's own interpolation: linear in position, heading from motion."""
    times = np.asarray(actor["times"], dtype=float)
    positions = np.asarray(actor["positions"], dtype=float)
    xy = np.column_stack([np.interp(query_times, times, positions[:, k]) for k in (0, 1)])
    # extrapolate the ends the way scipy's fill_value="extrapolate" would
    for k in (0, 1):
        if query_times[0] < times[0] or query_times[-1] > times[-1]:
            slope_lo = (positions[1, k] - positions[0, k]) / max(times[1] - times[0], 1e-9)
            slope_hi = (positions[-1, k] - positions[-2, k]) / max(times[-1] - times[-2], 1e-9)
            lo, hi = query_times < times[0], query_times > times[-1]
            xy[lo, k] = positions[0, k] + slope_lo * (query_times[lo] - times[0])
            xy[hi, k] = positions[-1, k] + slope_hi * (query_times[hi] - times[-1])
    steps = np.diff(xy, axis=0)
    steps = np.vstack([steps, steps[-1:]])
    moving = np.linalg.norm(steps, axis=1) > 1e-6
    yaw = np.where(moving, np.arctan2(steps[:, 1], steps[:, 0]), float(actor.get("initial-yaw", 0.0)))
    return xy, yaw


def check(scene, body: dict) -> dict:
    general = body["general"]
    actor = body["actors"][0]
    start_s = float(scene.times[general["start-frame"]])
    duration = float(general["duration"])
    query = np.linspace(start_s, start_s + duration, int(duration * 2) + 1)

    ego_xy = np.column_stack([np.interp(query, scene.times, scene.positions[:, k]) for k in (0, 1)])
    steps = np.vstack([np.diff(ego_xy, axis=0), np.diff(ego_xy, axis=0)[-1:]])
    ego_yaw = np.arctan2(steps[:, 1], steps[:, 0])

    xy, yaw = actor_poses(actor, query)
    dims = actor["dims"]

    gaps, hit_at = [], None
    for k, t in enumerate(query):
        forward = np.array([math.cos(ego_yaw[k]), math.sin(ego_yaw[k])])
        ego = corners(ego_xy[k] + REAR_AXLE_TO_CENTRE * forward, ego_yaw[k], EGO_LENGTH, EGO_WIDTH)
        box = corners(xy[k], yaw[k], dims[1], dims[0])
        gaps.append(float(np.linalg.norm(ego_xy[k] - xy[k])))
        if hit_at is None and overlap(ego, box):
            hit_at = float(t - start_s)
    closest = int(np.argmin(gaps))
    return {
        "closest_m": round(min(gaps), 2),
        "at_s": round(float(query[closest] - start_s), 1),
        "hit_at_s": hit_at,
        "ego_travel_m": round(float(np.linalg.norm(ego_xy[-1] - ego_xy[0])), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(SiteConfig.load().resolve("{data.nuscenes}"))
    tables = _tables(root)
    cache: dict[str, object] = {}

    print(f"{'scenario':26} {'closest':>9} {'when':>7} {'would hit':>10} {'ego moves':>10}")
    bad = []
    for path in sorted(args.dir.glob("*/*.yaml")):
        name = f"{path.parent.name}/{path.stem}"
        scene_name = f"scene-{path.stem}"
        if scene_name not in cache:
            cache[scene_name] = load_scene(root, scene_name, tables)
        got = check(cache[scene_name], yaml.safe_load(path.read_text()))
        flag = "" if got["hit_at_s"] is not None or got["closest_m"] < 4.0 else "   <-- no hazard"
        if flag:
            bad.append(name)
        print(
            f"{name:26} {got['closest_m']:8.2f}m {got['at_s']:6.1f}s "
            f"{('yes @' + str(got['hit_at_s']) + 's') if got['hit_at_s'] is not None else 'no':>10} "
            f"{got['ego_travel_m']:9.1f}m{flag}"
        )
    print(f"\n{len(bad)} scenario(s) set up no hazard: {', '.join(bad) if bad else 'none'}")
    print(json.dumps({"no_hazard": bad}))


if __name__ == "__main__":
    main()
