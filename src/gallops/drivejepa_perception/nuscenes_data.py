"""H15's nuScenes: its own scenes, keyframes, labels and inputs for the out-of-domain step (ood.py).

    <devkit python> -m gallops.drivejepa_perception.nuscenes_data train-scenes --out T.json
    <paddock python> -m gallops.drivejepa_perception.nuscenes_data draw --train-scenes T.json --n 100 --seed 0

**The scenes** are H15's own seeded draw (`draw_scenes`), committed as OOD_SCENES and re-drawn on
every load, which refuses an edited list. Candidates are nuScenes train scenes (the devkit's official
split) that have CAN-bus poses, without any scene behind a neuroNCAP scenario or a reconstructed one.
Val is never a candidate, because paddock's nuScenes benchmark reads val.

**Two interpreters, one draw.** The official split comes from the nuScenes devkit and the neuroNCAP
scenes from paddock's harness, both imported, never restated. At this site no interpreter has both:
paddock's config needs Python >= 3.11 (`tomllib`), and the devkit is installed only in older ones
(Drive-JEPA's 3.9, the ncap env's 3.10). So `train-scenes` runs where the devkit is and writes the
readable train scenes; `draw` runs where paddock imports, drops the neuroNCAP and reconstructed scenes,
and draws. The committed record names both sources: the devkit's version, paddock's commit and the
scenes it kept out.

**One definition each, imported:** poses, the command rule, the exclusions and the keyframe timing
from `gallops.data.nuscenes_real`; the model's inputs from `gallops.train.swerve_data`'s TickDataset;
the ego's footprint from `gallops.expert.grader`.

**A keyframe is kept** when the keyframe 0.5 s back exists (within HISTORY_TOLERANCE_US) and 4 s of
future keyframes exist: the model reads the frame 0.5 s back, and the plan is read against 4 s.

numpy only at import, so the tests run in gallops' own interpreter; the devkit, torch and paddock
are imported where they are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from gallops.data import nuscenes_real as nr

REPO = Path(__file__).resolve().parents[3]
OOD_SCENES = REPO / "recipes" / "drivejepa_perception" / "ood_scenes.json"
#: The devkit's rule for an annotation's velocity: a central difference over its neighbours, given
#: up when they are further apart than this (twice this when both exist).
MAX_VELOCITY_DT_S = 1.5
#: Plans are read over 3 s: the collision rate and the best of 32 by ADE (6 waypoints at 2 Hz).
HORIZON_WAYPOINTS = 6


# --------------------------------------------------------------------------------------- scenes


def readable_train_scenes(train: list, can_bus: Path) -> list:
    """The train scenes the ticks can read: their CAN-bus poses exist (speed and acceleration)."""
    return sorted(name for name in set(train) if (Path(can_bus) / f"{name}_pose.json").exists())


def candidate_scenes(readable: list, kept_out: set) -> list:
    """The readable train scenes without any whose number is in `kept_out` (neuroNCAP's, reconstructed)."""
    return sorted(name for name in set(readable) if name.split("-")[1] not in kept_out)


def draw_scenes(candidates: list, n: int, seed: int) -> list:
    """`n` of the candidates, chosen by `seed`, sorted. The same names give the same draw in any order."""
    names = sorted(set(candidates))
    if n > len(names):
        raise ValueError(f"{n} scenes asked of {len(names)} candidates")
    picked = np.random.default_rng(seed).choice(len(names), size=n, replace=False)
    return sorted(names[i] for i in picked)


def _sha256(names: list) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def write_scenes(path: Path, candidates: list, n: int, seed: int, sources=None) -> dict:
    """Draw and write the record; `sources` says where the candidates came from (kept as given)."""
    record = {
        "rule": "numpy default_rng(seed).choice over the sorted candidates, n of them, without replacement",
        "candidates_are": (
            "nuScenes train (devkit split) with CAN-bus poses, minus neuroNCAP and reconstructed scenes"
        ),
        "n": n,
        "seed": seed,
        **({"sources": sources} if sources else {}),
        "candidates_sha256": _sha256(candidates),
        "candidates": sorted(candidates),
        "scenes": draw_scenes(candidates, n, seed),
    }
    Path(path).write_text(json.dumps(record, indent=1) + "\n")
    return record


def load_scenes(path: Path = OOD_SCENES) -> list:
    """The committed draw, re-drawn from its own candidates; an edited list is refused."""
    record = json.loads(Path(path).read_text())
    if _sha256(record["candidates"]) != record["candidates_sha256"]:
        raise ValueError(f"{path}: the candidate list was edited")
    if draw_scenes(record["candidates"], record["n"], record["seed"]) != record["scenes"]:
        raise ValueError(f"{path}: the scene list is not the draw its rule makes")
    return record["scenes"]


# ------------------------------------------------------------------------------------ the tables


def _velocity(annotation: dict, by_token: dict, sample_time: dict) -> list:
    """The devkit's box velocity: a central difference over the neighbouring annotations, in m/s."""
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


def extract(root: Path, names: list) -> dict:
    """The named scenes' front keyframes, ego poses, calibration and annotations, from the tables."""
    tables = Path(root) / "v1.0-trainval"

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
    keyframes = {}  # (sample token, channel) -> sample_data record
    for record in load("sample_data"):
        if record["is_key_frame"] and record["sample_token"] in wanted:
            channel = channel_of[record["calibrated_sensor_token"]]
            if channel in nr.CAMERAS:
                keyframes[(record["sample_token"], channel)] = record
    front = {s: r for (s, ch), r in keyframes.items() if ch == "CAM_FRONT"}
    need = {r["ego_pose_token"] for r in front.values()}
    poses = {
        p["token"]: nr.pose_matrix(p["translation"], p["rotation"])
        for p in load("ego_pose")
        if p["token"] in need
    }
    categories = {c["token"]: c["name"] for c in load("category")}
    category_of = {i["token"]: categories[i["category_token"]] for i in load("instance")}
    sample_time = {t: samples[t]["timestamp"] for t in wanted}
    by_token = {a["token"]: a for a in load("sample_annotation") if a["sample_token"] in wanted}
    annotations = defaultdict(list)
    for a in by_token.values():
        annotations[a["sample_token"]].append(
            {
                "category": category_of[a["instance_token"]],
                "instance": a["instance_token"],
                "translation": a["translation"],
                "size_wlh": a["size"],
                "rotation": a["rotation"],
                "velocity": _velocity(a, by_token, sample_time),
                "lidar_points": a["num_lidar_pts"],
            }
        )
    out = {"scenes": {}, "annotations": dict(annotations)}
    for name, chain in chains.items():
        calibration = {}
        for channel in nr.CAMERAS:
            entry = calib[keyframes[(chain[0], channel)]["calibrated_sensor_token"]]
            calibration[channel] = {
                "pose": nr.pose_matrix(entry["translation"], entry["rotation"]).tolist(),
                "intrinsics": entry["camera_intrinsic"] or None,
            }
        out["scenes"][name] = {
            "location": logs[scenes[name]["log_token"]]["location"],
            # the scene's own words ("Night, rain, ..."): step 7 reads night and rain apart
            "description": scenes[name]["description"],
            "keyframes": [
                {
                    "sample": token,
                    "timestamp": int(front[token]["timestamp"]),
                    "file": front[token]["filename"],
                    "ego": poses[front[token]["ego_pose_token"]].tolist(),
                }
                for token in chain
            ],
            "calibration": calibration,
            "image_size": [front[chain[0]]["width"], front[chain[0]]["height"]],
        }
    return out


def ticks(name: str, scene: dict, can_bus: Path) -> list:
    """The scene's usable keyframes: 0.5 s of history and 4 s of future keyframes exist.

    Speed and acceleration are the CAN bus's at the keyframe, as paddock's nuScenes harness reads
    them; the command is paddock's rule on the human's next 3 s (`nuscenes_real.command`); `expert`
    is the human's next 4 s in this keyframe's ego frame.
    """
    path = Path(can_bus) / f"{name}_pose.json"
    if not path.exists():
        return []
    can = sorted(json.loads(path.read_text()), key=lambda m: m["utime"])
    can_t = [m["utime"] for m in can]
    frames = scene["keyframes"]
    stamps = [f["timestamp"] for f in frames]
    ego = [np.asarray(f["ego"]) for f in frames]
    rows = []
    for k, t in enumerate(stamps):
        if k == 0 or k + nr.FUTURE_STEPS >= len(stamps):
            continue
        if abs((t - stamps[k - 1]) - nr.HISTORY_DT_US) > nr.HISTORY_TOLERANCE_US:
            continue
        future = nr.to_ego(ego[k], ego[k + 1 : k + 1 + nr.FUTURE_STEPS])
        message = can[nr.nearest(can_t, t)]
        rows.append(
            {
                "scene": name,
                "sample": frames[k]["sample"],
                "keyframe": k,
                "timestamp": int(t),
                "speed": float(message["vel"][0]),
                "accel": float(message["accel"][0]),
                "command": nr.command(future[:, :2]),
                "expert": future.round(4).tolist(),
            }
        )
    return rows


def write_calibration(scenes: dict, directory: Path) -> None:
    """Each scene's calibration where `swerve_data.scene_calibration` reads it (GALLOPS_SCENE_CALIB)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, scene in scenes.items():
        (directory / f"{name}.json").write_text(json.dumps(scene["calibration"]))


# ------------------------------------------------------------------------------------ the boxes


def _yaw(matrix: np.ndarray) -> float:
    return math.atan2(matrix[1, 0], matrix[0, 0])


def boxes(annotations: list, ego2global: np.ndarray) -> dict:
    """Annotated boxes in the frame of `ego2global`: centres (N, 3), yaw, length, width, height,
    and ground velocity (N, 2) in its axes (nan where the devkit's rule gives none)."""
    n = len(annotations)
    if not n:
        empty = np.zeros(0)
        return {
            "xyz": np.zeros((0, 3)),
            "yaw": empty,
            "length": empty,
            "width": empty,
            "height": empty,
            "vel": np.zeros((0, 2)),
        }
    inverse = np.linalg.inv(ego2global)
    xyz = (np.c_[np.array([a["translation"] for a in annotations]), np.ones(n)] @ inverse.T)[:, :3]
    yaw = np.array([_yaw(nr.pose_matrix([0, 0, 0], a["rotation"])) for a in annotations]) - _yaw(ego2global)
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    wlh = np.array([a["size_wlh"] for a in annotations], dtype=float)
    c, s = math.cos(_yaw(ego2global)), math.sin(_yaw(ego2global))
    v = np.array([a["velocity"][:2] for a in annotations], dtype=float)
    vel = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], axis=1)
    return {"xyz": xyz, "yaw": yaw, "length": wlh[:, 1], "width": wlh[:, 0], "height": wlh[:, 2], "vel": vel}


def future_boxes(scene: dict, k: int, annotations: dict) -> list:
    """Every annotated box at each of the next FUTURE_STEPS keyframes, in keyframe k's frame:
    (M, 5) [x, y, yaw, length, width] per step -- what a plan made at k would drive into."""
    frames = scene["keyframes"]
    ego2global = np.asarray(frames[k]["ego"])
    out = []
    for j in range(1, nr.FUTURE_STEPS + 1):
        b = boxes(annotations.get(frames[k + j]["sample"], []), ego2global)
        out.append(np.stack([b["xyz"][:, 0], b["xyz"][:, 1], b["yaw"], b["length"], b["width"]], axis=1))
    return out


# -------------------------------------------------------------------------------- reading plans


def collides(plan: np.ndarray, steps_boxes: list, steps: int = HORIZON_WAYPOINTS) -> bool:
    """Whether the ego's footprint at a waypoint (x, y, yaw of its rear axle) overlaps a box logged
    at that waypoint's time. `steps_boxes[j]` is (M, 5) [x, y, yaw, length, width] in the plan's frame."""
    from gallops.expert.grader import EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE, corners, overlap

    reach = 0.5 * math.hypot(EGO_LENGTH, EGO_WIDTH) + REAR_AXLE_TO_CENTRE
    for j in range(min(steps, len(plan), len(steps_boxes))):
        near = steps_boxes[j]
        if not len(near):
            continue
        heading = float(plan[j, 2])
        centre = plan[j, :2] + REAR_AXLE_TO_CENTRE * np.array([math.cos(heading), math.sin(heading)])
        # a cheap circle test first: most boxes are nowhere near the ego
        span = reach + 0.5 * np.hypot(near[:, 3], near[:, 4])
        near = near[np.hypot(near[:, 0] - centre[0], near[:, 1] - centre[1]) < span]
        if not len(near):
            continue
        ego = corners(centre, heading, EGO_LENGTH, EGO_WIDTH)
        if any(overlap(ego, corners(b[:2], float(b[2]), float(b[3]), float(b[4]))) for b in near):
            return True
    return False


def collided(plans: np.ndarray, rows_boxes: list, steps: int = HORIZON_WAYPOINTS) -> np.ndarray:
    """(N,) `collides` for each row's plan."""
    return np.array([collides(p, b, steps) for p, b in zip(plans, rows_boxes)], dtype=bool)  # noqa: B905 -- 3.9


def min_ade(proposals: np.ndarray, expert: np.ndarray, steps: int = HORIZON_WAYPOINTS) -> np.ndarray:
    """(N,) the best of each row's proposals (N, P, 8, 3) by mean distance to the human over `steps`."""
    gap = np.linalg.norm(proposals[..., :steps, :2] - expert[:, None, :steps, :2], axis=-1)
    return gap.mean(-1).min(-1)


# -------------------------------------------------------------------------------------- inputs


def keyframe_dataset(rows: list, scenes: dict, nuscenes_root: str, calibration_dir: Path):
    """swerve_data's TickDataset fed from an extract: the same input path, frames and poses from the
    scene's keyframes instead of an episode directory."""
    import os

    from gallops.train.swerve_data import TickDataset

    if not Path(calibration_dir).is_dir():
        raise FileNotFoundError(f"{calibration_dir}: no calibration (write_calibration)")
    os.environ["GALLOPS_SCENE_CALIB"] = str(calibration_dir)  # else each scene parses sample_data

    class KeyframeDataset(TickDataset):
        def _episode(self, tick: dict) -> dict:
            name = tick["scene"]
            if name not in self._episodes:
                frames = scenes[name]["keyframes"]
                self._episodes[name] = {
                    "dir": None,
                    "poses": {int(f["timestamp"]): np.asarray(f["ego"]) for f in frames},
                    "frames": {int(f["timestamp"]): Path(nuscenes_root) / f["file"] for f in frames},
                    "scene": name,
                }
            return self._episodes[name]

    return KeyframeDataset(rows, "cpu", nuscenes_root)


def _git_head(repo: Path) -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def write_train_scenes(path: Path, can_bus: Path) -> dict:
    """The devkit's official train split, kept where CAN-bus poses exist (run where the devkit is)."""
    from importlib.metadata import version

    from nuscenes.utils.splits import create_splits_scenes

    train = create_splits_scenes()["train"]
    readable = readable_train_scenes(train, can_bus)
    record = {
        "rule": "the devkit's create_splits_scenes()['train'], kept where can_bus/<scene>_pose.json exists",
        "devkit": version("nuscenes-devkit"),
        "train_scenes": len(train),
        "readable_sha256": _sha256(readable),
        "readable": readable,
    }
    Path(path).write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.nuscenes_data")
    parser.add_argument("command", choices=("train-scenes", "draw"))
    parser.add_argument("--out", help="train-scenes: where to write the readable train scenes")
    parser.add_argument("--train-scenes", help="draw: train-scenes' output")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    if args.command == "train-scenes":
        if not args.out:
            parser.error("train-scenes needs --out")
        record = write_train_scenes(Path(args.out), Path(cfg.resolve("{data.nuscenes}")) / "can_bus")
        print(f"{len(record['readable'])} of {record['train_scenes']} train scenes have CAN-bus poses")
        return
    if not (args.train_scenes and args.n is not None and args.seed is not None):
        parser.error("draw needs --train-scenes, --n and --seed")
    if OOD_SCENES.exists():
        raise SystemExit(f"{OOD_SCENES} exists; the draw is made once and committed")
    train = json.loads(Path(args.train_scenes).read_text())
    if _sha256(train["readable"]) != train["readable_sha256"]:
        raise SystemExit(f"{args.train_scenes}: the scene list was edited")
    kept_out = nr.neuroncap_scenes() | nr.RECONSTRUCTED
    candidates = candidate_scenes(train["readable"], kept_out)
    sources = {
        "train_scenes": {k: train[k] for k in ("rule", "devkit", "train_scenes", "readable_sha256")},
        "readable": len(train["readable"]),
        "kept_out": sorted(kept_out),
        "kept_out_from": "gallops.data.nuscenes_real.neuroncap_scenes() (paddock's harness) | RECONSTRUCTED",
        "paddock_commit": _git_head(Path(cfg.resolve("{code.paddock}"))),
    }
    record = write_scenes(OOD_SCENES, candidates, args.n, args.seed, sources)
    print(f"{len(record['scenes'])} of {len(candidates)} candidate train scenes -> {OOD_SCENES}")


if __name__ == "__main__":
    main()
