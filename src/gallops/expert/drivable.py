"""Where the road is, as a small raster the grader can read without a map library.

The expert had no idea where the road was. It chose which side to swerve toward
purely from which side the actor sat on, so on a two-lane street with the ego in
the right-hand lane it steered toward the parked cars instead of the empty
oncoming lane. Nothing downstream noticed, because the grader only tested
collision and distance from the logged path.

Rather than make the grader depend on nuScenes' map API -- which lives in another
interpreter and costs seconds per query -- the drivable area around each scene is
rasterised once into a boolean grid. `build` needs the devkit and runs as a
preprocessing step; `DrivableArea.load` needs only numpy, so the grader stays a
pure-numpy module that a training loop can call a million times.

    <ncap python> -m gallops.expert.drivable --scene scene-0075 --out <dir>

A note on what this does and does not say. The raster is nuScenes' `drivable_area`
layer: the surface a vehicle may be driven on, which includes the oncoming
carriageway. It is not a lane-keeping test. Leaving the drivable area means a
pavement, a building or a parked-car bay; crossing into oncoming traffic is a
judgement the collision test already covers, because oncoming traffic is an actor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Metres per raster cell. The ego is 1.85 m wide and the margins that matter are
#: half a metre, so 0.25 m resolves the question without a large array.
CELL_M = 0.25
#: How far around the ego's logged path to rasterise. A trajectory is capped at
#: 4 m of lateral deviation and reaches about 40 m ahead, so this is generous.
PAD_M = 80.0


@dataclass(frozen=True)
class DrivableArea:
    """A boolean grid: is this world point somewhere a vehicle may be driven?"""

    mask: np.ndarray  # (rows, cols), True where drivable
    origin: np.ndarray  # world (x, y) of cell [0, 0]
    cell: float
    scene: str
    location: str

    @classmethod
    def load(cls, path: Path) -> DrivableArea:
        raw = np.load(path)
        return cls(
            mask=raw["mask"],
            origin=raw["origin"],
            cell=float(raw["cell"]),
            scene=str(raw["scene"]),
            location=str(raw["location"]),
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """True for each world (x, y) that lies on the drivable surface.

        A point outside the rasterised window counts as NOT drivable: the window
        covers everything a legal trajectory can reach, so leaving it means the
        trajectory went somewhere this was never meant to answer for.
        """
        points = np.atleast_2d(points)
        index = np.floor((points - self.origin) / self.cell).astype(int)
        rows, cols = self.mask.shape
        inside = (index[:, 1] >= 0) & (index[:, 1] < rows) & (index[:, 0] >= 0) & (index[:, 0] < cols)
        out = np.zeros(len(points), dtype=bool)
        out[inside] = self.mask[index[inside, 1], index[inside, 0]]
        return out


def build(scene_name: str, nuscenes_root: Path, out_dir: Path) -> Path:
    """Rasterise the drivable area around one scene's ego path. Needs the devkit."""
    from nuscenes.map_expansion.map_api import NuScenesMap

    tables = nuscenes_root / "v1.0-trainval"
    scenes = {s["name"]: s for s in json.loads((tables / "scene.json").read_text())}
    logs = {log["token"]: log for log in json.loads((tables / "log.json").read_text())}
    scene = scenes[scene_name]
    location = logs[scene["log_token"]]["location"]

    samples = {s["token"]: s for s in json.loads((tables / "sample.json").read_text())}
    sensors = {s["token"]: s["channel"] for s in json.loads((tables / "sensor.json").read_text())}
    calib = {
        c["token"]: sensors[c["sensor_token"]]
        for c in json.loads((tables / "calibrated_sensor.json").read_text())
    }
    poses = {p["token"]: p["translation"] for p in json.loads((tables / "ego_pose.json").read_text())}
    lidar = {
        r["sample_token"]: r
        for r in json.loads((tables / "sample_data.json").read_text())
        if r["is_key_frame"] and calib.get(r["calibrated_sensor_token"]) == "LIDAR_TOP"
    }

    token, path = scene["first_sample_token"], []
    while token:
        path.append(poses[lidar[token]["ego_pose_token"]][:2])
        token = samples[token]["next"]
    path = np.asarray(path)

    low = path.min(0) - PAD_M
    high = path.max(0) + PAD_M
    cols = int(np.ceil((high[0] - low[0]) / CELL_M))
    rows = int(np.ceil((high[1] - low[1]) / CELL_M))

    nusc_map = NuScenesMap(dataroot=str(nuscenes_root), map_name=location)
    centres_x = low[0] + (np.arange(cols) + 0.5) * CELL_M
    centres_y = low[1] + (np.arange(rows) + 0.5) * CELL_M
    mask = np.zeros((rows, cols), dtype=bool)
    # `get_records_in_patch` is far cheaper than asking about every cell, so the
    # polygons are fetched once and each row tested against them.
    records = nusc_map.get_records_in_patch(
        (low[0], low[1], high[0], high[1]), ["drivable_area"], mode="intersect"
    )["drivable_area"]
    polygons = []
    for token in records:
        for polygon_token in nusc_map.get("drivable_area", token)["polygon_tokens"]:
            polygons.append(nusc_map.extract_polygon(polygon_token))

    from shapely.geometry import Point
    from shapely.prepared import prep

    prepared = [prep(polygon) for polygon in polygons]
    for row, y in enumerate(centres_y):
        for col, x in enumerate(centres_x):
            point = Point(x, y)
            if any(p.contains(point) for p in prepared):
                mask[row, col] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scene_name}.npz"
    np.savez_compressed(out, mask=mask, origin=low, cell=CELL_M, scene=scene_name, location=location)
    print(
        f"  {scene_name} ({location}): {rows}x{cols} cells at {CELL_M} m, {mask.mean():.1%} drivable -> {out}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--nuscenes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for scene in args.scene:
        build(scene, args.nuscenes, args.out)


if __name__ == "__main__":
    main()
