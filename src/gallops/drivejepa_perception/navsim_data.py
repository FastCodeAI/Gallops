"""NAVSIM navtrain for H15: the sample index, the three log splits, and the agent labels.

    python -m gallops.drivejepa_perception.navsim_data splits --write recipes/drivejepa_perception/splits.json
    python -m gallops.drivejepa_perception.navsim_data labels --out {runs}/drivejepa_perception/labels-v1
    python -m gallops.drivejepa_perception.navsim_data check --labels {runs}/drivejepa_perception/labels-v1

The passes run in Drive-JEPA's interpreter (`env.enter`). The rules below are numpy-only, so
tests/ runs them in gallops' interpreter too.

**A sample** is a navtrain token: a window of 14 log frames at 2 Hz, 4 of history ending at the
current frame t (index 3) and 10 of future. Windows come from the vendor's own `filter_scenes`,
imported, so the index is exactly the one NAVSIM's `SceneLoader` builds.

**A label** (docs/drivejepa-perception-sidecar/first-pass-plan.md, S0.4) is every annotated agent at t within
RANGE_M of the ego, in frame t's ego coordinates (x forward, y left, metres):
- class, box (x, y, z, length, width, height, yaw) and velocity (vx, vy);
- its position at t + 0.5 s and t + 1.0 s: frames t+1 and t+2, joined on `track_token`, moved from
  frame k's ego coordinates through global into frame t's. A track that ends early leaves that
  step masked, never extrapolated.

Three facts the rules rest on:
- Boxes are in each frame's lidar coordinates, which in every OpenScene log are the ego's: lidar2ego
  is the identity (paddock's drivejepa policy docstring).
- `gt_velocity_3d` is the agent's world velocity expressed in frame k's axes. At t it is already in
  frame t's axes and nothing subtracts the ego's own speed. Only t's velocity is used, so no
  velocity crosses frames.
- NAVSIM moves poses in SE2 (`Scene.get_future_trajectory`), and so do the labels. z is t's own,
  never transformed. The `check` subcommand holds all of this to NAVSIM's code on real windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

#: OpenScene's annotation names, and the side-car's classes. The side-car learns the first three
#: (H04's grouping); static objects are kept in the labels for analysis and later rules.
NAMES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object")
CLASSES = ("vehicle", "pedestrian", "cyclist", "static")
CLASS_OF_NAME = {
    "vehicle": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "traffic_cone": 3,
    "barrier": 3,
    "czone_sign": 3,
    "generic_object": 3,
}
#: Agents kept around the ego at t, metres. The side-car's targets use 50 m inside CAM_F0's view; the
#: margin lets a later rule widen without a new pass.
RANGE_M = 60.0
#: navtrain.yaml: 4 frames of history ending at the current one, then 10 of future, at 2 Hz.
HISTORY = 4
FUTURE = 10
CURRENT = HISTORY - 1
#: The path steps: frames t+1 and t+2, which are t + 0.5 s and t + 1.0 s.
PATH_STEPS = (1, 2)
#: The human's plan in the planner's format: eight poses at 0.5 s.
EGO_FUTURE = 8
#: Ego speed from t to t + 2 s, for the pre-registered "slowing or stopped" subset.
EGO_SPEED_STEPS = 5

#: The select logs are H04's 51 held-out navtrain logs, copied with their source; the report logs are
#: drawn from the rest by this rule. docs/drivejepa-perception-sidecar/first-pass-plan.md, S0.3.
SELECT_SOURCE = {
    "branch": "drivejepa-wajepa/run3a",
    "commit": "e5ca907",
    "path": "recipes/drivejepa_wajepa/heldout_logs.txt",
}
REPORT_SEED = 15
REPORT_N = 70


# ------------------------------------------------------------------------------------------ SE2


def to_global(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(N, 2) points in the ego frame of global pose (x, y, yaw) -> global (N, 2)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    x, y = points[:, 0], points[:, 1]
    return np.stack([pose[0] + c * x - s * y, pose[1] + s * x + c * y], axis=1)


def to_local(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(N, 2) global points -> the ego frame of global pose (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    dx, dy = points[:, 0] - pose[0], points[:, 1] - pose[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def wrap(angle: np.ndarray) -> np.ndarray:
    """Angles into [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def relative_poses(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """(N, 3) global poses -> (x, y, heading) relative to the global pose `origin`."""
    xy = to_local(poses[:, :2], origin)
    return np.concatenate([xy, wrap(poses[:, 2:3] - origin[2])], axis=1)


# --------------------------------------------------------------------------------------- labels


def track_id(token: str) -> int:
    """A stable 63-bit integer for a track token, so labels carry identity without strings."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little") >> 1


def window_labels(frames: list, poses: np.ndarray) -> dict:
    """The labels of one navtrain window.

    `frames` are the window's raw log frames, with the current one at CURRENT. `poses` are their
    global ego poses (x, y, yaw), as NAVSIM's `Scene._build_ego_status` reads them; the pass passes
    the vendor's own values, and tests pass synthetic ones.
    """
    t = CURRENT
    anns = frames[t]["anns"]
    names = list(anns["gt_names"])
    boxes = np.asarray(anns["gt_boxes"], dtype=np.float64).reshape(-1, 7)
    velocity = np.asarray(anns["gt_velocity_3d"], dtype=np.float64).reshape(-1, 3)[:, :2]
    tracks = list(anns["track_tokens"])
    keep = np.array([name in CLASS_OF_NAME for name in names], dtype=bool) & (
        np.hypot(boxes[:, 0], boxes[:, 1]) <= RANGE_M
    )
    index = np.flatnonzero(keep)
    n = len(index)

    path = np.full((n, len(PATH_STEPS), 2), np.nan, dtype=np.float64)
    path_ok = np.zeros((n, len(PATH_STEPS)), dtype=bool)
    size_change = np.full((n, len(PATH_STEPS)), np.nan, dtype=np.float64)
    for j, step in enumerate(PATH_STEPS):
        later = frames[t + step]["anns"]
        where = {token: i for i, token in enumerate(later["track_tokens"])}
        later_boxes = np.asarray(later["gt_boxes"], dtype=np.float64).reshape(-1, 7)
        rows = np.array([where.get(tracks[i], -1) for i in index], dtype=np.int64)
        found = rows >= 0
        if found.any():
            centres = later_boxes[rows[found], :2]
            path[found, j] = to_local(to_global(centres, poses[t + step]), poses[t])
            path_ok[found, j] = True
            size_change[found, j] = np.abs(later_boxes[rows[found], 3:6] - boxes[index[found], 3:6]).max(
                axis=1
            )

    ego_future = relative_poses(poses[t + 1 : t + 1 + EGO_FUTURE], poses[t])
    speed = np.array(
        [
            np.hypot(*np.asarray(frames[t + k]["ego_dynamic_state"][:2], dtype=np.float64))
            for k in range(EGO_SPEED_STEPS)
        ]
    )
    return {
        "cls": np.array([CLASS_OF_NAME[names[i]] for i in index], dtype=np.int8),
        "name": np.array([NAMES.index(names[i]) for i in index], dtype=np.int8),
        "box": boxes[index].astype(np.float32),
        "velocity": velocity[index].astype(np.float32),
        "path": path.astype(np.float32),
        "path_ok": path_ok,
        "size_change": size_change.astype(np.float32),
        "track": np.array([track_id(tracks[i]) for i in index], dtype=np.int64),
        "ego_future": ego_future.astype(np.float32),
        "ego_speed": speed.astype(np.float32),
    }


#: Per-agent arrays, concatenated over samples; the rest are per sample.
AGENT_KEYS = ("cls", "name", "box", "velocity", "path", "path_ok", "size_change", "track")
SAMPLE_KEYS = ("ego_future", "ego_speed")


def pack(samples: list) -> dict:
    """Window labels -> one dict of arrays: agents concatenated, with `offset` / `count` per sample."""
    counts = np.array([len(s["cls"]) for s in samples], dtype=np.int64)
    out = {
        "count": counts.astype(np.int32),
        "offset": np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64),
    }
    for key in AGENT_KEYS:
        out[key] = np.concatenate([s[key] for s in samples]) if samples else np.zeros(0)
    for key in SAMPLE_KEYS:
        out[key] = np.stack([s[key] for s in samples])
    return out


def agents_of(labels: dict, row: int) -> dict:
    """The agent arrays of one sample out of `pack`'s layout."""
    start, count = int(labels["offset"][row]), int(labels["count"][row])
    return {key: labels[key][start : start + count] for key in AGENT_KEYS}


# --------------------------------------------------------------------------------------- splits


def draw_splits(navtrain_logs: list, select_logs: list) -> dict:
    """The three log lists: H04's select logs, a seeded draw of report logs, and train."""
    logs = sorted(set(navtrain_logs))
    select = sorted(set(select_logs))
    outside = sorted(set(select) - set(logs))
    if outside:
        raise ValueError(f"{len(outside)} select logs are not navtrain logs, e.g. {outside[:2]}")
    candidates = [log for log in logs if log not in set(select)]
    picked = np.random.default_rng(REPORT_SEED).choice(len(candidates), REPORT_N, replace=False)
    report = sorted(candidates[i] for i in picked)
    train = [log for log in candidates if log not in set(report)]
    return {
        "rule": "select = H04's held-out navtrain logs; report = default_rng(seed).choice over the "
        "sorted navtrain logs outside select, n of them, without replacement; train = the rest",
        "select_source": SELECT_SOURCE,
        "select_sha256": _sha256_lines(select),
        "candidates_sha256": _sha256_lines(candidates),
        "report_seed": REPORT_SEED,
        "report_n": REPORT_N,
        "select": select,
        "report": report,
        "train": train,
    }


def load_splits(path: Path, navtrain_logs: list) -> dict:
    """The committed splits, re-drawn from their rule and refused if the file was edited."""
    stored = json.loads(Path(path).read_text())
    redrawn = draw_splits(navtrain_logs, stored["select"])
    if stored["select_sha256"] != redrawn["select_sha256"]:
        raise SystemExit(f"{path}: the select list does not match its recorded sha256")
    for key in ("report", "train", "candidates_sha256", "report_seed", "report_n"):
        if stored[key] != redrawn[key]:
            raise SystemExit(f"{path}: `{key}` differs from what its rule draws; the file was edited")
    return stored


def _sha256_lines(lines: list) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ----------------------------------------------------------------------------------- the passes


def navtrain_filter(code: Path) -> dict:
    """navtrain's scene filter, read from the Drive-JEPA tree that runs (not restated here)."""
    import yaml

    path = code / "navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    filt = yaml.safe_load(path.read_text())
    if (filt["num_history_frames"], filt["num_future_frames"], filt["frame_interval"]) != (
        HISTORY,
        FUTURE,
        1,
    ):
        raise SystemExit(f"{path}: the window is not the one these labels assume")
    filt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return filt


def log_windows(log: str, logs_dir: Path, tokens: set) -> dict:
    """token -> its 14 raw frames, for one log, by the vendor's own `filter_scenes`."""
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import filter_scenes

    scene_filter = SceneFilter(
        num_history_frames=HISTORY,
        num_future_frames=FUTURE,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=[log],
        tokens=list(tokens),
    )
    windows, _ = filter_scenes(logs_dir, scene_filter)
    return windows


def global_poses(frames: list) -> np.ndarray:
    """The global ego pose of each frame, by NAVSIM's own `Scene._build_ego_status`."""
    from navsim.common.dataclasses import Scene

    return np.array([Scene._build_ego_status(frame).ego_pose for frame in frames], dtype=np.float64)


_WORKER: dict = {}


def _labels_of_log(log: str):
    windows = log_windows(log, _WORKER["logs_dir"], _WORKER["tokens"])
    rows = []
    for token, frames in windows.items():
        rows.append((token, window_labels(frames, global_poses(frames))))
    return log, rows


def build_labels(cfg, out: Path, workers: int, limit_logs) -> None:
    """The labels pass: every navtrain window, written once under `out`, never overwritten."""
    from multiprocessing import get_context

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    sha = git_sha()  # at the start: the tree may move during a long pass
    entered = env.enter(cfg)
    filt = navtrain_filter(Path(entered["code"]))
    logs = sorted(filt["log_names"])[: limit_logs or None]
    _WORKER.update(logs_dir=Path(entered["navsim_root"]) / "navsim_logs/trainval", tokens=set(filt["tokens"]))

    tokens, log_ids, samples = [], [], []
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        for k, (log, rows) in enumerate(pool.imap(_labels_of_log, logs)):
            for token, labels in rows:
                tokens.append(token)
                log_ids.append(logs.index(log))
                samples.append(labels)
            if k % 100 == 0:
                print(
                    f"{k + 1}/{len(logs)} logs, {len(tokens)} samples, {time.time() - t0:.0f} s", flush=True
                )

    packed = pack(samples)
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(
        staging / "labels.npz", token=np.array(tokens), log_id=np.array(log_ids, dtype=np.int16), **packed
    )
    counts = np.bincount(packed["cls"].astype(np.int64), minlength=len(CLASSES)) if len(packed["cls"]) else []
    manifest = {
        "pass": "labels",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "navtrain_yaml_sha256": filt["sha256"],
        "logs": logs,
        "samples": len(tokens),
        "navtrain_tokens": len(filt["tokens"]),
        "agents": int(len(packed["cls"])),
        "agents_per_class": dict(zip(CLASSES, [int(c) for c in counts])),  # noqa: B905 -- Python 3.9
        "range_m": RANGE_M,
        "path_steps": list(PATH_STEPS),
        "path_ok_share": [float(x) for x in packed["path_ok"].mean(axis=0)] if len(packed["cls"]) else [],
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "logs"}, indent=1))


def check_labels(cfg, labels_dir: Path, n_windows: int) -> list:
    """Hold the labels to NAVSIM's own code on real windows; return the problems found.

    - the ego future equals `Scene.get_future_trajectory(8)` (1e-4 m, 1e-4 rad);
    - each joined path point equals nuPlan's `relative_to_absolute_poses` followed by NAVSIM's
      `convert_absolute_to_relative_se2_array` (1e-4 m);
    - a re-run of `window_labels` equals what the pass stored (exactly, after float32);
    - a joined track keeps its size (at most 0.5 m on 99% of joins: auto-labels drift a little).
    """
    from gallops.drivejepa_perception import env

    entered = env.enter(cfg)  # first: navsim comes from {code.drivejepa}, put on sys.path here
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses

    stored = dict(np.load(Path(labels_dir) / "labels.npz"))
    manifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    logs = manifest["logs"]
    logs_dir = Path(entered["navsim_root"]) / "navsim_logs/trainval"
    rng = np.random.default_rng(0)
    rows = sorted(rng.choice(len(stored["token"]), size=min(n_windows, len(stored["token"])), replace=False))
    problems, worst = [], {"ego": 0.0, "path": 0.0, "joins": 0, "size_over": 0}
    for row in rows:
        token, log = str(stored["token"][row]), logs[int(stored["log_id"][row])]
        frames = log_windows(log, logs_dir, {token})[token]
        poses = global_poses(frames)
        scene = Scene.from_scene_dict_list(
            frames,
            Path(entered["navsim_root"]) / "sensor_blobs/trainval",
            HISTORY,
            FUTURE,
            SensorConfig.build_no_sensors(),
        )
        vendor = scene.get_future_trajectory(EGO_FUTURE).poses
        ego = np.asarray(stored["ego_future"][row], dtype=np.float64)
        worst["ego"] = max(worst["ego"], float(np.abs(wrap(vendor - ego)).max()))
        again = window_labels(frames, poses)
        mine = agents_of(stored, row)
        for key in ("cls", "box", "path_ok", "track"):
            if not np.array_equal(again[key].astype(mine[key].dtype), mine[key]):
                problems.append(f"{token}: re-run `{key}` differs from the stored labels")
        anns = frames[CURRENT]["anns"]
        tracks = list(anns["track_tokens"])
        for j, step in enumerate(PATH_STEPS):
            later = frames[CURRENT + step]["anns"]
            where = {tok: i for i, tok in enumerate(later["track_tokens"])}
            for a in np.flatnonzero(mine["path_ok"][:, j])[:20]:
                tid = int(mine["track"][a])
                source = [tok for tok in tracks if track_id(tok) == tid][0]
                bx, by = np.asarray(later["gt_boxes"][where[source]][:2], dtype=np.float64)
                (absolute,) = relative_to_absolute_poses(
                    StateSE2(*poses[CURRENT + step]), [StateSE2(bx, by, 0.0)]
                )
                ref = convert_absolute_to_relative_se2_array(
                    StateSE2(*poses[CURRENT]), np.array([[absolute.x, absolute.y, absolute.heading]])
                )[0, :2]
                worst["path"] = max(worst["path"], float(np.abs(ref - mine["path"][a, j]).max()))
        joined = np.isfinite(mine["size_change"])
        worst["joins"] += int(joined.sum())
        worst["size_over"] += int((mine["size_change"][joined] > 0.5).sum())
    if worst["ego"] > 1e-4:
        problems.append(f"ego future off NAVSIM's by {worst['ego']:.2e}")
    if worst["path"] > 1e-4:
        problems.append(f"path points off nuPlan/NAVSIM's transform by {worst['path']:.2e} m")
    if worst["joins"] and worst["size_over"] > 0.01 * worst["joins"]:
        problems.append(f"{worst['size_over']} of {worst['joins']} joins change size by > 0.5 m")
    print(json.dumps({"windows": len(rows), **worst, "problems": problems}, indent=1))
    return problems


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.navsim_data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("splits")
    p.add_argument("--write", required=True)
    p.add_argument("--select", required=True, help="H04's heldout_logs.txt, as copied from its branch")
    p = sub.add_parser("labels")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit-logs", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--labels", required=True)
    p.add_argument("--windows", type=int, default=60)
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    if args.cmd == "splits":
        filt = navtrain_filter(Path(cfg.resolve("{code.drivejepa}")))
        select = [line.strip() for line in Path(args.select).read_text().splitlines() if line.strip()]
        splits = draw_splits(filt["log_names"], select)
        Path(args.write).write_text(json.dumps(splits, indent=1) + "\n")
        print({k: len(splits[k]) for k in ("select", "report", "train")})
    elif args.cmd == "labels":
        build_labels(cfg, Path(args.out), args.workers, args.limit_logs)
    else:
        sys.exit(1 if check_labels(cfg, Path(args.labels), args.windows) else 0)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
