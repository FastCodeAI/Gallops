"""nuScenes for the auxiliary labels: keyframes, boxes, CAN bus and map, read from the tables.

The tables are read directly, as `gallops.data.nuscenes_real` reads them, and that module's
pose helpers, keyframe rule, command rule and scene exclusions are imported, not restated:
a keyframe here is a keyframe there.

    python -m gallops.drivejepa_aux.nusc extract --out <dir>     # the train scenes, once

`extract` parses the 1.3 GB sample_data table once and keeps what the labels and the model's
inputs need, as `<dir>/extract.pkl`, plus each scene's calibration as `<dir>/calib/<scene>.json`
in the layout `swerve_data.scene_calibration` reads (set GALLOPS_SCENE_CALIB to that directory).

**Train scenes only**, from the devkit's official split, and never the scenes behind neuroNCAP
or the five reconstructed ones (`nuscenes_real`'s exclusions): paddock's nuScenes benchmark and
neuroNCAP's scenarios come from `val`.

**The map through the devkit, with one shim.** nuscenes-devkit 1.1.11's `map_api` calls
`plt.style.use("seaborn-whitegrid")` on import; matplotlib 3.6 renamed that style, so the import
fails in every environment on this site (Drive-JEPA's and ncap's). `devkit_map` registers the
new name under the old one first. Geometry is indexed once per location (`MapView`): the
devkit's `get_records_in_patch` extracts every record's polygon on every call.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from gallops.data.nuscenes_real import (
    CAMERAS,
    FUTURE_STEPS,
    HISTORY_DT_US,
    HISTORY_TOLERANCE_US,
    NEURONCAP_MINI,
    RECONSTRUCTED,
    command,
    nearest,
    neuroncap_scenes,
    pose_matrix,
    to_ego,
)
from gallops.drivejepa_aux import bev, targets

#: Drive-JEPA crops this many rows off the top and the bottom of the frame before resizing it
#: (`image[28:-28]` in the vendor's `_get_camera_feature`, `swerve_data.TickDataset.frame`).
MODEL_CROP_ROWS = 28

#: nuScenes categories, by prefix, for each BEV object class and for the box and lead targets.
VEHICLE = ("vehicle.",)
PEDESTRIAN = ("human.pedestrian.",)
STATIC_OBJECT = ("movable_object.barrier", "movable_object.trafficcone", "static_object.bicycle_rack")

#: Lane-graph hops searched for "the ego followed its lane": 4 s at urban speed crosses a lane
#: and a connector or two.
LANE_HOPS = 4
#: Map records are gathered within this radius of the ego; the grid's far corners are 45 m away.
MAP_RADIUS_M = 50.0
#: The devkit's `box_velocity`: a difference over more than this is not a velocity.
MAX_VELOCITY_DT_S = 1.5


def excluded_scene_numbers() -> set:
    """neuroNCAP's scenes and the five reconstructed ones, by number: `nuscenes_real`'s exclusions.

    `nuscenes_real.neuroncap_scenes` imports paddock's neuroNCAP harness, which needs Python 3.11
    (`paddock.config` imports tomllib), and Drive-JEPA's interpreter is 3.9. There the harness is
    read rather than run: its OFFICIAL_SET is a literal, taken from paddock's own source with
    `ast`, so the set is still paddock's and not a copy of it.
    """
    if sys.version_info >= (3, 11):  # noqa: UP036 -- this module also runs under Drive-JEPA's 3.9
        return neuroncap_scenes() | RECONSTRUCTED
    import ast

    import paddock

    path = Path(paddock.__file__).parent / "adapters" / "benchmarks" / "neuroncap" / "harness.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "OFFICIAL_SET":
            official = ast.literal_eval(node.value)
            return {s for scenes in official.values() for s in scenes} | NEURONCAP_MINI | RECONSTRUCTED
    raise SystemExit(f"no OFFICIAL_SET literal in {path}")


def train_scene_names() -> list[str]:
    """The devkit's train split, less neuroNCAP's scenes and the reconstructed ones."""
    from nuscenes.utils import splits

    held = excluded_scene_numbers()
    names = sorted(n for n in splits.train if n.split("-")[1] not in held)
    assert not set(names) & set(splits.val)
    return names


def _velocity(annotation: dict, by_token: dict, sample_time: dict) -> list:
    """The devkit's `NuScenes.box_velocity`: a central difference over the neighbouring annotations."""
    prev = by_token.get(annotation["prev"]) if annotation["prev"] else None
    nxt = by_token.get(annotation["next"]) if annotation["next"] else None
    if prev is None and nxt is None:
        return [math.nan] * 3
    first = prev if prev is not None else annotation
    last = nxt if nxt is not None else annotation
    dt = (sample_time[last["sample_token"]] - sample_time[first["sample_token"]]) * 1e-6
    limit = MAX_VELOCITY_DT_S * (2 if prev is not None and nxt is not None else 1)
    if dt <= 0 or dt > limit:
        return [math.nan] * 3
    return ((np.asarray(last["translation"]) - np.asarray(first["translation"])) / dt).tolist()


def extract(root: Path, names: list[str]) -> dict:
    """Keyframes, calibration and annotations of the named scenes, from the trainval tables."""
    tables = root / "v1.0-trainval"

    def load(name: str):
        return json.loads((tables / f"{name}.json").read_text())

    scenes = {s["name"]: s for s in load("scene")}
    logs = {x["token"]: x for x in load("log")}
    samples = {x["token"]: x for x in load("sample")}
    sensors = {x["token"]: x["channel"] for x in load("sensor")}
    calib = {x["token"]: x for x in load("calibrated_sensor")}
    channel_of = {t: sensors[c["sensor_token"]] for t, c in calib.items()}

    chains = {}
    for name in names:
        chain, token = [], scenes[name]["first_sample_token"]
        while token:
            chain.append(token)
            token = samples[token]["next"]
        chains[name] = chain
    wanted = {t for chain in chains.values() for t in chain}

    keyframes = {}  # (sample_token, channel) -> record
    for record in load("sample_data"):
        if record["is_key_frame"] and record["sample_token"] in wanted:
            channel = channel_of[record["calibrated_sensor_token"]]
            if channel in CAMERAS:
                keyframes[(record["sample_token"], channel)] = {
                    k: record[k]
                    for k in (
                        "filename",
                        "ego_pose_token",
                        "calibrated_sensor_token",
                        "timestamp",
                        "width",
                        "height",
                    )
                }
    need = {r["ego_pose_token"] for (_, ch), r in keyframes.items() if ch == "CAM_FRONT"}
    poses = {
        p["token"]: pose_matrix(p["translation"], p["rotation"])
        for p in load("ego_pose")
        if p["token"] in need
    }

    categories = {c["token"]: c["name"] for c in load("category")}
    instance_category = {i["token"]: categories[i["category_token"]] for i in load("instance")}
    visibility = {v["token"]: v["level"] for v in load("visibility")}
    sample_time = {t: samples[t]["timestamp"] for t in wanted}
    by_token = {a["token"]: a for a in load("sample_annotation") if a["sample_token"] in wanted}
    annotations = defaultdict(list)
    for a in by_token.values():
        annotations[a["sample_token"]].append(
            {
                "category": instance_category[a["instance_token"]],
                "instance": a["instance_token"],
                "translation": a["translation"],
                "size_wlh": a["size"],
                "rotation": a["rotation"],
                "velocity": _velocity(a, by_token, sample_time),
                "visibility": visibility[a["visibility_token"]],
                "lidar_points": a["num_lidar_pts"],
            }
        )

    out = {"version": 1, "scenes": {}, "annotations": dict(annotations)}
    for name, chain in chains.items():
        front = [keyframes[(t, "CAM_FRONT")] for t in chain]
        calibration = {}
        for channel in CAMERAS:
            entry = calib[keyframes[(chain[0], channel)]["calibrated_sensor_token"]]
            calibration[channel] = {
                "pose": pose_matrix(entry["translation"], entry["rotation"]).tolist(),
                "intrinsics": entry["camera_intrinsic"] or None,
            }
        out["scenes"][name] = {
            "location": logs[scenes[name]["log_token"]]["location"],
            "keyframes": [
                {
                    "sample": token,
                    "timestamp": record["timestamp"],
                    "file": record["filename"],
                    "ego": poses[record["ego_pose_token"]].tolist(),
                }
                for token, record in zip(chain, front)  # noqa: B905 -- Python 3.9
            ],
            "calibration": calibration,
            "image_size": [front[0]["width"], front[0]["height"]],
        }
    return out


def ticks(name: str, scene: dict, root: Path) -> list[dict]:
    """The scene's trainable keyframes, as rows `inputs.TablesTickDataset` takes.

    `nuscenes_real`'s rule: the keyframe 0.5 s back exists (within 0.05 s) and so do 4 s of
    future keyframes. Speed and acceleration come from the CAN bus as paddock's nuScenes harness
    reads them, the command from paddock's own rule on the human's next 3 s. Empty when the scene
    has no CAN-bus pose messages.
    """
    can_path = root / "can_bus" / f"{name}_pose.json"
    if not can_path.exists():
        return []
    can = sorted(json.loads(can_path.read_text()), key=lambda m: m["utime"])
    can_t = [m["utime"] for m in can]
    frames = scene["keyframes"]
    stamps = [k["timestamp"] for k in frames]
    ego = [np.asarray(k["ego"]) for k in frames]
    rows = []
    for k, (t, pose) in enumerate(zip(stamps, ego)):  # noqa: B905 -- Python 3.9
        if k == 0 or k + FUTURE_STEPS >= len(stamps):
            continue
        if abs((t - stamps[k - 1]) - HISTORY_DT_US) > HISTORY_TOLERANCE_US:
            continue
        future = to_ego(pose, ego[k + 1 : k + 1 + FUTURE_STEPS])
        message = can[nearest(can_t, t)]
        rows.append(
            {
                "scene": name,
                "sample": frames[k]["sample"],
                "keyframe": k,
                "timestamp": int(t),
                "speed": float(message["vel"][0]),
                "accel": float(message["accel"][0]),
                "command": command(future[:, :2]),
                "expert": future.round(4).tolist(),
                "location": scene["location"],
            }
        )
    return rows


def write_calibration(scenes: dict, directory: Path) -> None:
    """Each scene's calibration where `swerve_data.scene_calibration` looks (GALLOPS_SCENE_CALIB)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, scene in scenes.items():
        (directory / f"{name}.json").write_text(json.dumps(scene["calibration"]))


# --- the map --------------------------------------------------------------------------------------


def devkit_map(root: Path, location: str):
    """The devkit's NuScenesMap, importable with matplotlib >= 3.6 (see the module docstring)."""
    import matplotlib.style

    if (
        "seaborn-whitegrid" not in matplotlib.style.library
        and "seaborn-v0_8-whitegrid" in matplotlib.style.library
    ):
        matplotlib.style.library["seaborn-whitegrid"] = matplotlib.style.library["seaborn-v0_8-whitegrid"]
    from nuscenes.map_expansion.map_api import NuScenesMap

    return NuScenesMap(dataroot=str(root), map_name=location)


def _yaw(pose: np.ndarray) -> float:
    return math.atan2(pose[1, 0], pose[0, 0])


def _to_ego_2d(points: np.ndarray, ego2global: np.ndarray) -> np.ndarray:
    """Global (N, 2) points in the ego's ground frame (x forward, y left)."""
    c, s = math.cos(_yaw(ego2global)), math.sin(_yaw(ego2global))
    rel = np.asarray(points, dtype=float)[:, :2] - ego2global[:2, 3]
    return np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], axis=1)


class MapView:
    """One location's map, indexed once: polygons, lines and lane centrelines with their bounds."""

    def __init__(self, nmap):
        self.nmap = nmap
        self.road = self._polygons("drivable_area")
        self.walkway = self._polygons("walkway")
        self.dividers = self._lines("lane_divider") + self._lines("road_divider")
        self.lanes = {}  # token -> shapely polygon, lanes and lane connectors
        for layer in ("lane", "lane_connector"):
            for record in getattr(nmap, layer):
                self.lanes[record["token"]] = nmap.extract_polygon(record["polygon_token"])
        tokens = list(self.lanes)
        centrelines = nmap.discretize_lanes(tokens, 0.5)
        self.centrelines = {t: np.asarray(centrelines[t])[:, :2] for t in tokens if len(centrelines[t]) > 1}
        self._lane_bounds = np.array([self.lanes[t].bounds for t in tokens])
        self._lane_tokens = tokens

    def _polygons(self, layer: str) -> list:
        out = []
        for record in getattr(self.nmap, layer):
            for token in record.get("polygon_tokens", [record.get("polygon_token")]):
                polygon = self.nmap.extract_polygon(token)
                if not polygon.is_valid:
                    # One in the four maps: a boston-seaport walkway (b7ada811) that crosses itself, on
                    # which clipping raises GEOS's TopologyException. buffer(0) repairs it, area unchanged.
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    out.append((polygon, np.array(polygon.bounds)))
        return out

    def _lines(self, layer: str) -> list:
        out = []
        for record in getattr(self.nmap, layer):
            line = self.nmap.extract_line(record["line_token"])
            if not line.is_empty:
                coords = np.asarray(line.coords)
                out.append((coords, np.r_[coords.min(0), coords.max(0)]))
        return out

    @staticmethod
    def _near(bounds: np.ndarray, x: float, y: float, radius: float) -> bool:
        return (
            bounds[0] <= x + radius
            and bounds[2] >= x - radius
            and bounds[1] <= y + radius
            and bounds[3] >= y - radius
        )

    def layers(self, ego2global: np.ndarray, radius: float = MAP_RADIUS_M) -> dict:
        """Road, walkway, centreline and divider masks on the BEV grid, around this pose."""
        from shapely.geometry import Polygon

        x, y = ego2global[0, 3], ego2global[1, 3]
        # The grid's own rectangle in global coordinates. Map polygons are clipped to it before
        # rasterising: a drivable-area polygon spans city blocks, thousands of vertices, and the
        # rasteriser's cost is vertices x cells.
        c, s = math.cos(_yaw(ego2global)), math.sin(_yaw(ego2global))
        corners = np.array(
            [[0.0, -bev.Y_HALF], [bev.X_MAX, -bev.Y_HALF], [bev.X_MAX, bev.Y_HALF], [0.0, bev.Y_HALF]]
        )
        window = Polygon(corners @ np.array([[c, s], [-s, c]]) + [x, y])

        def polygons(items):
            pieces = []
            for polygon, bounds in items:
                if not self._near(bounds, x, y, radius):
                    continue
                clipped = polygon.intersection(window)
                for part in getattr(clipped, "geoms", [clipped]):
                    if part.geom_type == "Polygon" and not part.is_empty:
                        exterior = _to_ego_2d(np.asarray(part.exterior.coords), ego2global)
                        holes = [_to_ego_2d(np.asarray(h.coords), ego2global) for h in part.interiors]
                        pieces.append((exterior, holes))
            return pieces

        lines = [_to_ego_2d(c, ego2global) for c, b in self.dividers if self._near(b, x, y, radius)]
        centre = [
            _to_ego_2d(c, ego2global)
            for t, c in self.centrelines.items()
            if self._near(np.r_[c.min(0), c.max(0)], x, y, radius)
        ]
        return {
            "road": bev.polygon_mask(polygons(self.road)),
            "walkway": bev.polygon_mask(polygons(self.walkway)),
            "centreline": bev.polyline_mask(centre),
            "lane_divider": bev.polyline_mask(lines),
        }

    def lanes_at(self, x: float, y: float) -> set:
        """Every lane and lane connector whose polygon holds the point."""
        from shapely.geometry import Point

        b = self._lane_bounds
        near = np.flatnonzero((b[:, 0] <= x) & (b[:, 2] >= x) & (b[:, 1] <= y) & (b[:, 3] >= y))
        point = Point(x, y)
        return {self._lane_tokens[i] for i in near if self.lanes[self._lane_tokens[i]].contains(point)}

    def reachable(self, lanes: set, hops: int = LANE_HOPS) -> set:
        """Every lane the lane graph leads to from `lanes` within `hops` steps."""
        seen, frontier = set(), set(lanes)
        for _ in range(hops):
            frontier = {n for lane in frontier for n in self.nmap.get_outgoing_lane_ids(lane)} - seen
            seen |= frontier
        return seen

    def lateral_shift(self, lanes: set, now_xy, later_xy) -> float:
        """How far (left positive) the ego moved across its lane between two points: the change in
        signed offset from one centreline -- that of `lanes` nearest to where it starts."""
        best = None
        for lane in lanes:
            line = self.centrelines.get(lane)
            if line is None:
                continue
            _, d, _ = targets.path_coordinates(np.array([now_xy, later_xy]), line)
            if best is None or abs(d[0]) < best[0]:
                best = (abs(d[0]), float(d[1] - d[0]))
        return 0.0 if best is None else best[1]


# --- one keyframe's labels ------------------------------------------------------------------------


def _boxes(annotations: list, ego2global: np.ndarray, prefixes: tuple) -> tuple:
    """Boxes of the given categories in the ego frame: (N, 5) [x, y, heading, length, width],
    (N, 3) centres with height, (N, 2) ground velocity in ego axes."""
    chosen = [a for a in annotations if a["category"].startswith(prefixes)]
    if not chosen:
        return np.zeros((0, 5)), np.zeros((0, 3)), np.zeros((0, 2))
    inverse = np.linalg.inv(ego2global)
    centres = np.array([inverse @ np.r_[a["translation"], 1.0] for a in chosen])[:, :3]
    yaw = _yaw(ego2global)
    heading = np.array([_yaw(pose_matrix([0, 0, 0], a["rotation"])) - yaw for a in chosen])
    heading = (heading + math.pi) % (2 * math.pi) - math.pi
    wlh = np.array([a["size_wlh"] for a in chosen])
    boxes = np.stack([centres[:, 0], centres[:, 1], heading, wlh[:, 1], wlh[:, 0]], axis=1)
    c, s = math.cos(yaw), math.sin(yaw)
    v = np.array([a["velocity"][:2] for a in chosen], dtype=float)
    velocity = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], axis=1)
    return boxes, centres, velocity


def keyframe_labels(scene: dict, tick: dict, annotations: list, view_map: MapView) -> dict:
    """Every auxiliary target for one keyframe, plus what was used to build them."""
    frames = scene["keyframes"]
    k = tick["keyframe"]
    ego2global = np.asarray(frames[k]["ego"])
    camera = scene["calibration"]["CAM_FRONT"]
    intrinsics, cam2ego = np.asarray(camera["intrinsics"]), np.asarray(camera["pose"])
    width, height = scene["image_size"]
    rows = (MODEL_CROP_ROWS, height - MODEL_CROP_ROWS)
    view = bev.view_mask(intrinsics, cam2ego, width, rows)

    vehicles, vehicle_centres, vehicle_velocity = _boxes(annotations, ego2global, VEHICLE)
    pedestrians = _boxes(annotations, ego2global, PEDESTRIAN)[0]
    static = _boxes(annotations, ego2global, STATIC_OBJECT)[0]
    layers = view_map.layers(ego2global)
    layers.update(
        {
            "vehicle": bev.box_mask(vehicles),
            "pedestrian": bev.box_mask(pedestrians),
            "static_object": bev.box_mask(static),
        }
    )
    semantic = bev.semantic_map(layers, view)
    seen = (
        bev.visible(vehicle_centres, intrinsics, cam2ego, width, rows) if len(vehicles) else np.zeros(0, bool)
    )
    agent_states, agent_valid = targets.agent_targets(vehicles, seen)

    future = np.asarray(tick["expert"])
    lead = targets.lead_vehicle(vehicles, vehicle_velocity, seen, future[:, :2], tick["speed"])

    later = np.asarray(frames[k + FUTURE_STEPS]["ego"])
    now_xy, later_xy = ego2global[:2, 3], later[:2, 3]
    lanes_now = view_map.lanes_at(*now_xy)
    lanes_later = view_map.lanes_at(*later_xy)
    reachable = view_map.reachable(lanes_now) if lanes_now else set()
    shift = view_map.lateral_shift(lanes_now, now_xy, later_xy) if lanes_now else 0.0
    change = targets.lane_change(lanes_now, lanes_later, reachable, shift)

    return {
        "bev": semantic,
        "view": view,
        "agent_states": agent_states,
        "agent_valid": agent_valid,
        "lead_present": lead["present"],
        "lead_gap": lead["gap"],
        "lead_closing": lead["closing"],
        "lane_change": change,
        # what the targets were built from, for checks and plots
        "road": layers["road"],  # the whole grid, not only the camera's view: where a plan leaves the road
        "vehicles": vehicles,
        "vehicles_seen": seen,
        "lead_index": lead["index"],
        "lanes_now": sorted(lanes_now),
        "lanes_later": sorted(lanes_later),
        "lane_shift": shift,
    }


def future_boxes(scene: dict, tick: dict, annotations: dict) -> list:
    """Every annotated box at each of the next FUTURE_STEPS keyframes, in THIS keyframe's ego frame,
    (N, 5) [x, y, heading, length, width] per step: what a plan made here would drive into.

    Every category, as paddock's nuScenes benchmark counts them for its collision rate (its index
    keeps every sample_annotation); `annotations` is the extract's, keyed by sample token.
    """
    frames = scene["keyframes"]
    k = tick["keyframe"]
    ego2global = np.asarray(frames[k]["ego"])
    return [
        _boxes(annotations.get(frames[k + j]["sample"], []), ego2global, ("",))[0]
        for j in range(1, FUTURE_STEPS + 1)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="keyframes, calibration and annotations of the train scenes")
    ex.add_argument("--nuscenes", type=Path, help="default: gallops.toml's {data.nuscenes}")
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--limit", type=int, default=0, help="only the first N scenes (a quick look)")
    args = ap.parse_args()

    root = args.nuscenes
    if root is None:
        from gallops.config import SiteConfig

        root = Path(SiteConfig.load().resolve("{data.nuscenes}"))
    names = train_scene_names()
    if args.limit:
        names = names[: args.limit]
    out = args.out / "extract.pkl"
    if out.exists():
        raise SystemExit(f"{out} exists; an extract is written once")
    print(f"extracting {len(names)} train scenes from the trainval tables...", flush=True)
    data = extract(root, names)
    data["ticks"] = {name: ticks(name, scene, root) for name, scene in data["scenes"].items()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_calibration(data["scenes"], args.out / "calib")
    with open(out, "wb") as f:
        pickle.dump(data, f, protocol=4)
    n_ticks = sum(len(v) for v in data["ticks"].values())
    no_can = sorted(n for n, v in data["ticks"].items() if not v)
    print(f"{len(names)} scenes, {n_ticks} trainable keyframes; {len(no_can)} scenes without CAN-bus poses")


if __name__ == "__main__":
    main()
