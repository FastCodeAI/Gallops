"""Real nuScenes train scenes, written as episodes the scorer cache already reads.

The scorer learned its collision answer from four or five reconstructed scenes and
did not carry it to a held-out one. Real nuScenes train has hundreds of scenes full
of vehicles in adjacent and oncoming lanes; graded against where those vehicles
really went, plans that swerve into them are labelled as hits -- the rule frontal
needs, from many more situations. Nothing downstream changes: each scene becomes an
episode directory in the layout the neuroNCAP collector writes, plus labelled rows,
so `scorer_cache`, `scorer_labels`, the folds and the viewer read it as-is.

Per scene, written under --collect/real-SSSS/run_0/:
  CAM_FRONT/<us>.jpg      symlinks to the real CAM_FRONT keyframes
  ego_poses.json          {us: 4x4 ego2world} at those keyframes (the camera sample's own ego pose)
  actors.json             every annotated vehicle (nuScenes category vehicle.*), 2 Hz boxes;
                          none is a target -- there is no inserted hazard
  reference_trajectory.json  the logged path, {us: [x, y, yaw]}
And --labelled/real-SSSS__run_0.jsonl: per tick, the CAN-bus speed and acceleration
(paddock's nuScenes harness uses the same messages), VAD's 3 s / 2 m command rule
(paddock's `command_from_future`), and the human's next 4 s as `expert`.

A tick is kept only if the CAM_FRONT keyframe 0.5 s back exists within 0.05 s (the
model reads that frame; see `swerve_data.HISTORY_TOLERANCE_S`) and 4 s of future
keyframes exist (the progress label needs the human's 4 s).

Excluded by name: every scene with a neuroNCAP scenario (scene-0061 is in train),
and the five reconstructed scenes, so the real log of a held-out rendered scene
never trains its fold. Needs the nuScenes devkit (the ncap env) for the maps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

NEURONCAP_SCENES = {
    "0061",
    "0099",
    "0101",
    "0103",
    "0106",
    "0108",
    "0110",
    "0278",
    "0331",
    "0346",
    "0783",
    "0796",
    "0921",
    "0923",
    "0966",
}
RECONSTRUCTED = {"0053", "0058", "0064", "0070", "0075"}
TURN_THRESHOLD_M = 2.0
COMMAND_STEPS = 6  # 3 s at 2 Hz
FUTURE_STEPS = 8  # 4 s
HISTORY_DT_US = 500_000
HISTORY_TOLERANCE_US = 50_000
#: neuroNCAP's Command enum: RIGHT 0, LEFT 1, STRAIGHT 2.
RIGHT, LEFT, STRAIGHT = 0, 1, 2
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "LIDAR_TOP")


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (x * w + y * z), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_matrix(translation, rotation) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_matrix(rotation)
    m[:3, 3] = translation
    return m


def to_ego(ego2world: np.ndarray, poses: list[np.ndarray]) -> np.ndarray:
    """(N, 3) [x, y, yaw] of world poses in the ego frame of `ego2world`."""
    inv = np.linalg.inv(ego2world)
    out = []
    for p in poses:
        rel = inv @ p
        out.append([rel[0, 3], rel[1, 3], math.atan2(rel[1, 0], rel[0, 0])])
    return np.asarray(out)


def command(future_xy: np.ndarray) -> int:
    """VAD's rule as paddock's nuScenes harness applies it: lateral offset at the 3 s point."""
    lateral = future_xy[COMMAND_STEPS - 1][1]
    if abs(lateral) < TURN_THRESHOLD_M:
        return STRAIGHT
    return LEFT if lateral > 0 else RIGHT


def nearest(times: list[int], t: int) -> int:
    i = int(np.searchsorted(times, t))
    return min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t))


def rasterise(polygons, low, rows: int, cols: int, cell: float) -> np.ndarray:
    """Drivable mask at cell centres -- point-in-polygon, exterior minus holes, as drivable.build."""
    from matplotlib.path import Path as MPath

    mask = np.zeros((rows, cols), dtype=bool)
    cx = low[0] + (np.arange(cols) + 0.5) * cell
    cy = low[1] + (np.arange(rows) + 0.5) * cell
    for polygon in polygons:
        x0, y0, x1, y1 = polygon.bounds
        c0, c1 = max(0, int((x0 - low[0]) / cell) - 1), min(cols, int((x1 - low[0]) / cell) + 2)
        r0, r1 = max(0, int((y0 - low[1]) / cell) - 1), min(rows, int((y1 - low[1]) / cell) + 2)
        if c0 >= c1 or r0 >= r1:
            continue
        gx, gy = np.meshgrid(cx[c0:c1], cy[r0:r1])
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        inside = MPath(np.asarray(polygon.exterior.coords)).contains_points(pts)
        for hole in polygon.interiors:
            inside &= ~MPath(np.asarray(hole.coords)).contains_points(pts)
        mask[r0:r1, c0:c1] |= inside.reshape(r1 - r0, c1 - c0)
    return mask


def main() -> None:
    from nuscenes.map_expansion.map_api import NuScenesMap

    from gallops.expert.drivable import CELL_M, PAD_M
    from gallops.expert.lanes import episode_lanes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuscenes", type=Path, required=True)
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--labelled", type=Path, required=True, help="training scenes' rows")
    ap.add_argument("--labelled-holdout", type=Path, required=True, help="held-out real scenes' rows")
    ap.add_argument("--drivable", type=Path, required=True)
    ap.add_argument(
        "--calib", type=Path, required=True, help="per-scene calibration JSON, read by swerve_data"
    )
    ap.add_argument(
        "--val-scenes", type=Path, required=True, help="paddock's val_scenes.json: these are not train"
    )
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles scenes i, i+n, ...")
    ap.add_argument(
        "--check",
        help="a reconstructed scene (e.g. scene-0070): reproduce its drivable.build raster "
        "and its swerve_data calibration exactly, then exit",
    )
    ap.add_argument("--check-raster", type=Path, help="that scene's drivable.build npz")
    args = ap.parse_args()

    tables = args.nuscenes / "v1.0-trainval"
    load = lambda name: json.loads((tables / f"{name}.json").read_text())  # noqa: E731
    print("loading tables...", flush=True)
    scenes = sorted(load("scene"), key=lambda s: s["name"])
    logs = {x["token"]: x for x in load("log")}
    samples = {x["token"]: x for x in load("sample")}
    sensors = {x["token"]: x["channel"] for x in load("sensor")}
    calib = {x["token"]: x for x in load("calibrated_sensor")}
    channel_of = {t: sensors[c["sensor_token"]] for t, c in calib.items()}
    keyframes = {}  # (sample_token, channel) -> sample_data record
    for r in load("sample_data"):
        if r["is_key_frame"]:
            ch = channel_of[r["calibrated_sensor_token"]]
            if ch in CAMERAS:
                keyframes[(r["sample_token"], ch)] = r
    need = {r["ego_pose_token"] for r in keyframes.values()}
    poses = {
        p["token"]: pose_matrix(p["translation"], p["rotation"])
        for p in load("ego_pose")
        if p["token"] in need
    }
    categories = {c["token"]: c["name"] for c in load("category")}
    instance_cat = {i["token"]: categories[i["category_token"]] for i in load("instance")}
    annotations = defaultdict(list)
    for a in load("sample_annotation"):
        if instance_cat[a["instance_token"]].startswith("vehicle."):
            annotations[a["sample_token"]].append(a)
    print("tables loaded", flush=True)

    val = set(json.loads(args.val_scenes.read_text())["val"])
    names = [
        s["name"]
        for s in scenes
        if s["name"] not in val and s["name"].split("-")[1] not in NEURONCAP_SCENES | RECONSTRUCTED
    ]
    rng = random.Random(0)
    holdout = set(rng.sample(names, int(round(len(names) * args.holdout_frac))))
    if args.check:
        names = [args.check]
    elif args.limit:
        names = names[: args.limit]
    i_shard, n_shard = (int(v) for v in args.shard.split("/"))
    names = names[i_shard::n_shard]
    maps: dict[str, NuScenesMap] = {}
    by_name = {s["name"]: s for s in scenes}

    for done, name in enumerate(names):
        scene = by_name[name]
        number = name.split("-")[1]
        episode = args.collect / f"real-{number}" / "run_0"
        out_rows = (
            args.labelled_holdout if name in holdout else args.labelled
        ) / f"real-{number}__run_0.jsonl"
        if out_rows.exists() and not args.check:
            continue
        can_path = args.nuscenes / "can_bus" / f"{name}_pose.json"
        if not can_path.exists():
            print(f"  {name}: no CAN-bus pose messages, skipped", flush=True)
            continue
        can = sorted(json.loads(can_path.read_text()), key=lambda m: m["utime"])
        can_t = [m["utime"] for m in can]
        location = logs[scene["log_token"]]["location"]
        if location not in maps:
            maps[location] = NuScenesMap(dataroot=str(args.nuscenes), map_name=location)
        nmap = maps[location]

        chain, token = [], scene["first_sample_token"]
        while token:
            chain.append(token)
            token = samples[token]["next"]
        front = [keyframes[(t, "CAM_FRONT")] for t in chain]
        stamps = [r["timestamp"] for r in front]
        ego = [poses[r["ego_pose_token"]] for r in front]

        # calibration at the first keyframe, as swerve_data.scene_calibration reads it
        cal = {}
        for ch in CAMERAS:
            entry = calib[keyframes[(chain[0], ch)]["calibrated_sensor_token"]]
            cal[ch] = {
                "pose": pose_matrix(entry["translation"], entry["rotation"]).tolist(),
                "intrinsics": entry["camera_intrinsic"] or None,
            }
        if args.check:
            from gallops.train.swerve_data import scene_calibration

            os.environ.pop("GALLOPS_SCENE_CALIB", None)
            ref = scene_calibration(str(args.nuscenes), name)
            worst = max(float(np.abs(np.asarray(cal[ch]["pose"]) - ref[ch][0]).max()) for ch in ref)
            worst_k = max(
                float(np.abs(np.asarray(cal[ch]["intrinsics"]) - ref[ch][1]).max())
                for ch in ref
                if ref[ch][1] is not None
            )
            print(
                f"  CHECK calibration {name}: max pose diff {worst:.2e}, max intrinsics diff {worst_k:.2e}",
                flush=True,
            )
            assert worst < 1e-9 and worst_k < 1e-3, "calibration differs from swerve_data.scene_calibration"
        args.calib.mkdir(parents=True, exist_ok=True)
        (args.calib / f"{name}.json").write_text(json.dumps(cal))

        # drivable raster around the LIDAR_TOP keyframe ego path, as drivable.build
        lidar_path = np.asarray([poses[keyframes[(t, "LIDAR_TOP")]["ego_pose_token"]][:2, 3] for t in chain])
        low = lidar_path.min(0) - PAD_M
        high = lidar_path.max(0) + PAD_M
        cols = int(np.ceil((high[0] - low[0]) / CELL_M))
        rows = int(np.ceil((high[1] - low[1]) / CELL_M))
        recs = nmap.get_records_in_patch(
            (low[0], low[1], high[0], high[1]), ["drivable_area"], mode="intersect"
        )["drivable_area"]
        polys = [
            nmap.extract_polygon(pt) for r in recs for pt in nmap.get("drivable_area", r)["polygon_tokens"]
        ]
        mask = rasterise(polys, low, rows, cols, CELL_M)
        if args.check:
            ref = np.load(args.check_raster)
            same_grid = ref["mask"].shape == mask.shape and np.allclose(ref["origin"], low)
            agree = float((ref["mask"] == mask).mean()) if same_grid else float("nan")
            print(
                f"  CHECK raster {name}: grid identical {same_grid}, cells agreeing {agree:.6f} "
                f"({int((ref['mask'] != mask).sum()) if same_grid else -1} of {mask.size} differ)",
                flush=True,
            )
            assert same_grid and agree > 0.9999, "raster differs from drivable.build"
            return
        args.drivable.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.drivable / f"{name}.npz", mask=mask, origin=low, cell=CELL_M, scene=name, location=location
        )

        # actors: every annotated vehicle, by instance, at 2 Hz
        tracks = defaultdict(list)
        for t in chain:
            for a in annotations.get(t, []):
                tracks[a["instance_token"]].append(
                    (samples[t]["timestamp"], pose_matrix(a["translation"], a["rotation"]), a["size"])
                )
        actors = []
        for inst, items in tracks.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            actors.append(
                {
                    "uuid": inst,
                    "timestamps": [x[0] for x in items],
                    "poses": [x[1].round(4).tolist() for x in items],
                    "dims": items[0][2],
                    "is_target_actor": False,
                }
            )

        episode.mkdir(parents=True, exist_ok=True)
        (episode / "CAM_FRONT").mkdir(exist_ok=True)
        for r in front:
            link = episode / "CAM_FRONT" / f"{r['timestamp']}.jpg"
            if not link.exists():
                os.symlink(args.nuscenes / r["filename"], link)
        (episode / "ego_poses.json").write_text(
            json.dumps({str(t): p.round(6).tolist() for t, p in zip(stamps, ego, strict=True)})
        )
        (episode / "actors.json").write_text(json.dumps(actors))
        (episode / "reference_trajectory.json").write_text(
            json.dumps(
                {
                    str(t): [float(p[0, 3]), float(p[1, 3]), float(math.atan2(p[1, 0], p[0, 0]))]
                    for t, p in zip(stamps, ego, strict=True)
                }
            )
        )
        (episode / "trajectories.json").write_text("{}")
        lanes = episode_lanes(nmap, episode)
        (episode / "lane_reference.json").write_text(json.dumps(lanes))

        rows_out = []
        for k, (t, p) in enumerate(zip(stamps, ego, strict=True)):
            if k == 0 or k + FUTURE_STEPS >= len(stamps):
                continue
            if abs((t - stamps[k - 1]) - HISTORY_DT_US) > HISTORY_TOLERANCE_US:
                continue
            fut = to_ego(p, ego[k + 1 : k + 1 + FUTURE_STEPS])
            m = can[nearest(can_t, t)]
            rows_out.append(
                {
                    "episode": f"real-{number}/run_0",
                    "timestamp": int(t),
                    "frame": str(episode / "CAM_FRONT" / f"{t}.jpg"),
                    "speed": float(m["vel"][0]),
                    "accel": float(m["accel"][0]),
                    "command": command(fut[:, :2]),
                    "expert": fut.round(4).tolist(),
                    "expert_kind": "human",
                    "hazard_in_frame": False,
                    "usable_for_imitation": True,
                    "location": location,
                    "holdout": name in holdout,
                }
            )
        out_rows.parent.mkdir(parents=True, exist_ok=True)
        out_rows.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
        print(
            f"  [{done + 1}/{len(names)}] {name} ({location}{', HOLDOUT' if name in holdout else ''}): "
            f"{len(rows_out)} of {len(stamps)} keyframes, {len(actors)} vehicles, lanes on {len(lanes)}, "
            f"{mask.mean():.1%} drivable",
            flush=True,
        )


if __name__ == "__main__":
    main()
