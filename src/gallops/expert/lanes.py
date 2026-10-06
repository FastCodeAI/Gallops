"""The lane the car is in, as the path the expert follows -- not the human's drive.

The expert used to follow the logged human path: every plan was that path plus a
bump that returned to it. Where the human drove off-centre or changed lanes, the
expert taught the same. Measured against the nuScenes map, the logged path in
scene-0070 runs a median 0.88 m (up to 1.62 m) from its lane centre, and in
scene-0058 it reaches 1.45 m, crossing toward the next lane. Reviewed on the
drives themselves: in 0058 a driver in the original lane would slow and then go
round, not switch lanes at the last moment as the human happened to.

So the reference is the centre of the lane the car is in at THAT tick, chained
forward through the lane graph. At a junction the chain takes the outgoing lane
nearest the human's path, which keeps the route (a turn stays a turn) without
copying a lane change, because a lane change is never an outgoing lane.

This needs nuscenes-devkit's map API, which only the ncap environment has, so it
runs once per collected episode and writes `lane_reference.json` beside the
episode's other files. Labelling and training read that file; neither imports the
map library.

    <ncap python> -m gallops.expert.lanes --collect <collect dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

#: How far a lane may sit from the car and still be "the lane it is in".
SEARCH_RADIUS_M = 5.0
#: A lane pointing more than this away from the car is the opposite carriageway or a crossing.
ALIGN_DEG = 45.0
AHEAD_M = 80.0
BEHIND_M = 10.0
SPACING_M = 1.0
#: How much of an outgoing lane is compared with the human's path at a junction.
JUNCTION_LOOK_M = 15.0


def load_references(episode: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The logged human path, and the per-tick lane centrelines when they were computed.

    One definition for labelling and for the trainer's referee: if they read
    different paths, the scorer is graded against a lane the labels never used,
    and the training preflight fails on exactly that disagreement.
    """
    logged = json.loads((episode / "reference_trajectory.json").read_text())
    path = np.asarray([v for _, v in sorted(logged.items(), key=lambda kv: int(kv[0]))], dtype=float)[:, :2]
    lanes_file = episode / "lane_reference.json"
    lanes = {}
    if lanes_file.exists():
        lanes = {int(k): np.asarray(v, dtype=float) for k, v in json.loads(lanes_file.read_text()).items()}
    return path, lanes


def reference_at(stamp: int, logged: np.ndarray, lanes: dict[int, np.ndarray]) -> tuple[np.ndarray, str]:
    """The path a tick is drawn and graded against, and which kind it is."""
    lane = lanes.get(int(stamp))
    return (lane, "lane") if lane is not None else (logged, "logged")


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(steps)]
    if arc[-1] < spacing:
        return points
    grid = np.arange(0.0, arc[-1], spacing)
    return np.column_stack([np.interp(grid, arc, points[:, 0]), np.interp(grid, arc, points[:, 1])])


def lane_path(nusc_map, x: float, y: float, yaw: float, guide: np.ndarray) -> np.ndarray | None:
    """World (N, 2) centreline of the lane at (x, y) facing `yaw`, BEHIND_M back to AHEAD_M on.

    `guide` is the human's path in world coordinates, used only to pick the
    outgoing lane at a junction. None when no lane is aligned with the car.
    """
    records = nusc_map.get_records_in_radius(x, y, SEARCH_RADIUS_M, ["lane", "lane_connector"])
    tokens = list(records.get("lane", [])) + list(records.get("lane_connector", []))
    if not tokens:
        return None
    discrete = nusc_map.discretize_lanes(tokens, 0.5)
    best = None
    for token in tokens:
        pts = np.asarray(discrete[token])
        if len(pts) < 2:
            continue
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        i = int(np.argmin(d))
        if abs(_wrap(float(pts[i, 2]) - yaw)) > math.radians(ALIGN_DEG):
            continue
        if best is None or d[i] < best[0]:
            best = (float(d[i]), token, i, pts)
    if best is None:
        return None
    _, token, i, pts = best
    forward = [pts[i:, :2]]
    length = float(np.linalg.norm(np.diff(pts[i:, :2], axis=0), axis=1).sum())
    for _ in range(30):
        if length >= AHEAD_M:
            break
        outgoing = nusc_map.get_outgoing_lane_ids(token)
        if not outgoing:
            break
        nxt = nusc_map.discretize_lanes(outgoing, 0.5)

        def off_route(tok, nxt=nxt):
            lane = np.asarray(nxt[tok])[:, :2]
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(lane, axis=0), axis=1))]
            head = lane[arc <= JUNCTION_LOOK_M]
            return float(np.mean([np.min(np.hypot(*(guide - p).T)) for p in head]))

        token = min(outgoing, key=off_route)
        lane = np.asarray(nxt[token])[:, :2]
        forward.append(lane[1:])
        length += float(np.linalg.norm(np.diff(lane, axis=0), axis=1).sum())
    ahead = np.vstack(forward)
    direction = ahead[1] - ahead[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    behind = ahead[0] - direction * np.arange(BEHIND_M, 0.0, -SPACING_M)[:, None]
    return _resample(np.vstack([behind, ahead]), SPACING_M)


def episode_lanes(nusc_map, episode: Path) -> dict[str, list]:
    """{tick timestamp: lane centreline} for every logged pose of one episode."""
    poses = json.loads((episode / "ego_poses.json").read_text())
    logged = json.loads((episode / "reference_trajectory.json").read_text())
    guide = np.asarray([v for _, v in sorted(logged.items(), key=lambda kv: int(kv[0]))], dtype=float)[:, :2]
    out = {}
    for stamp, pose in poses.items():
        pose = np.asarray(pose)
        path = lane_path(
            nusc_map, float(pose[0, 3]), float(pose[1, 3]), math.atan2(pose[1, 0], pose[0, 0]), guide
        )
        if path is not None:
            out[stamp] = np.round(path, 2).tolist()
    return out


def main() -> None:
    from nuscenes.map_expansion.map_api import NuScenesMap

    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collect", type=Path, required=True)
    args = parser.parse_args()

    root = Path(SiteConfig.load().resolve("{data.nuscenes}"))
    tables = root / "v1.0-trainval"
    logs = {entry["token"]: entry["location"] for entry in json.loads((tables / "log.json").read_text())}
    location = {s["name"]: logs[s["log_token"]] for s in json.loads((tables / "scene.json").read_text())}
    maps: dict[str, object] = {}
    for episode in sorted(p.parent for p in args.collect.glob("*-*/run_*/ego_poses.json")):
        scene = "scene-" + episode.parent.name.rsplit("-", 1)[-1]
        city = location[scene]
        if city not in maps:
            maps[city] = NuScenesMap(dataroot=str(root), map_name=city)
        lanes = episode_lanes(maps[city], episode)
        (episode / "lane_reference.json").write_text(json.dumps(lanes))
        print(
            f"  {episode.parent.name}/{episode.name}: lane for {len(lanes)} of "
            f"{len(json.loads((episode / 'ego_poses.json').read_text()))} ticks",
            flush=True,
        )


if __name__ == "__main__":
    main()
