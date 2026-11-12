"""NAVSIM for H15: the sample index, the three log splits, Part B's two lists, and the agent labels.

    python -m gallops.drivejepa_perception.navsim_data splits --write recipes/drivejepa_perception/splits.json
    python -m gallops.drivejepa_perception.navsim_data labels --out {runs}/drivejepa_perception/labels-v1
    python -m gallops.drivejepa_perception.navsim_data check --labels {runs}/drivejepa_perception/labels-v1
    python -m gallops.drivejepa_perception.navsim_data navtest-list --write recipes/.../navtest.json
    python -m gallops.drivejepa_perception.navsim_data fresh-list --write recipes/.../fresh.json
    python -m gallops.drivejepa_perception.navsim_data labels --source navtest --out {out}/labels-navtest-v1

The passes run in Drive-JEPA's interpreter (`env.enter`). The rules below are numpy-only, so
tests/ runs them in gallops' interpreter too.

**Sources** (Part B, docs/drivejepa-perception-sidecar/part-b/B-0-foundations.md, B-0-1). A labels
pass reads one of three:
- `navtrain`, the default and all Part A read: the select, report and train logs of `splits.json`;
- `navtest`, the vendor's navtest filter, from the test data split: where Part B reads headlines,
  once per experiment;
- `fresh`, the R-clean split Part B chooses settings on: the trainval logs that no NAVSIM scene
  filter names, so the release never trained on them, and every routed window of theirs.
The two lists Part B adds are committed (`navtest.json`, `fresh.json`), re-derived on every load and
refused if edited, like `splits.json`. A pass records its source in its manifest; `labels-v1`
predates sources and is navtrain.

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

#: This file's checkout, for the committed lists: the cwd moves into navsim_v2/ (env.enter).
REPO = Path(__file__).resolve().parents[3]
#: Where a pass's windows come from, and the OpenScene data split their logs and blobs sit in.
SOURCES = ("navtrain", "navtest", "fresh")
DATA_SPLIT = {"navtrain": "trainval", "navtest": "test", "fresh": "trainval"}
#: Part B's committed lists, relative to the checkout.
LISTS = {
    "navtest": "recipes/drivejepa_perception/navtest.json",
    "fresh": "recipes/drivejepa_perception/fresh.json",
}
#: The vendor's scene filters, relative to {code.drivejepa}.
FILTER_DIR = "navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter"


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


def subsets(labels: dict, proposals: np.ndarray) -> dict:
    """The three pre-registered subsets (PREDICTIONS.md), from the labels, not from any model.

    Moved here from `train_scorer` (which keeps the name) so it is numpy-only and tested anywhere.
    - `lead_30m`: a labelled vehicle ahead within 30 m and 1.75 m of the ego's line;
    - `agent_within_10m`: any labelled agent within 10 m of any proposal's first 2 s of poses;
    - `slowing_or_stopped`: the ego slows by more than 1 m/s over 2 s, or is under 0.5 m/s.
    """
    n = len(labels["token"])
    lead, near = np.zeros(n, bool), np.zeros(n, bool)
    for row in range(n):
        a = agents_of(labels, row)
        xy = a["box"][:, :2]
        lead[row] = bool(
            ((a["cls"] == 0) & (xy[:, 0] > 0) & (xy[:, 0] < 30) & (np.abs(xy[:, 1]) <= 1.75)).any()
        )
        if len(xy):
            poses = proposals[row, :, :4, :2].reshape(-1, 2)
            near[row] = bool((np.linalg.norm(poses[:, None] - xy[None], axis=-1) < 10.0).any())
    speed = labels["ego_speed"]
    slowing = (speed[:, 0] - speed[:, -1] > 1.0) | (speed[:, 0] < 0.5)
    return {"lead_30m": lead, "slowing_or_stopped": slowing, "agent_within_10m": near}


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


# ------------------------------------------------------------------------- Part B's two lists


def navtest_record(filt: dict) -> dict:
    """navtest's list: the vendor's own filter, named by its sha256, with its logs and tokens' sha256.

    The tokens stay in the vendor's file; the record pins which file. `filt` is `scene_filter`'s.
    """
    tokens = sorted(filt["tokens"])
    return {
        "rule": "navtest = the vendor's navtest scene filter in {code.drivejepa}, unchanged: its logs and "
        "tokens, read from the test data split. Part B reads it once per experiment (B-0, Q2)",
        "source": {"file": filt["file"], "sha256": filt["sha256"]},
        "data_split": DATA_SPLIT["navtest"],
        "logs": sorted(filt["log_names"]),
        "n_tokens": len(tokens),
        "tokens_sha256": _sha256_lines(tokens),
    }


def load_navtest(path: Path, filt: dict) -> dict:
    """The committed navtest list, re-derived from the vendor's filter and refused on any difference."""
    stored = json.loads(Path(path).read_text())
    again = navtest_record(filt)
    differ = sorted(key for key in set(stored) | set(again) if stored.get(key) != again.get(key))
    if differ:
        raise SystemExit(f"{path}: {differ} differ from what the vendor's navtest filter gives")
    return stored


def named_logs(filters: dict) -> set:
    """Every log any of the vendor's scene filters names."""
    return {log for filt in filters.values() for log in (filt.get("log_names") or [])}


def fresh_record(trainval_logs: list, filters: dict, windows: dict, cities: dict) -> dict:
    """R-clean's list (B-0, Q3): the trainval logs no NAVSIM scene filter names, and their routed windows.

    `filters` are all the vendor's scene filters (`all_filters`); `windows` maps each candidate log to
    its routed window tokens, by navtrain's window (`log_windows` with tokens None); `cities` maps a
    log that has windows to its `map_location`. A log without a routed window drops out: every NAVSIM
    metric needs a route. Refuses a window token that a filter names (a private test token, say).
    """
    candidates = sorted(set(trainval_logs) - named_logs(filters))
    logs = [log for log in candidates if windows.get(log)]
    tokens = sorted(token for log in logs for token in windows[log])
    named = {token for filt in filters.values() for token in (filt.get("tokens") or [])}
    if set(tokens) & named:
        raise SystemExit(f"{len(set(tokens) & named)} R-clean windows are tokens a NAVSIM filter names")
    per_city: dict = {}
    for log in logs:
        per_city[cities[log]] = per_city.get(cities[log], 0) + len(windows[log])
    return {
        "rule": "fresh (R-clean) = the trainval logs on disk that no scene filter in the vendor's "
        "scene_filter/ names, and every window of theirs with a route, by navtrain's window (4 + 10 "
        "frames at 2 Hz). The routed test logs outside navtest are not included (D24, B-0 (a))",
        "sources": {name: filt["sha256"] for name, filt in sorted(filters.items())},
        "trainval_logs": len(trainval_logs),
        "trainval_logs_sha256": _sha256_lines(sorted(trainval_logs)),
        "n_candidates": len(candidates),
        "candidates_sha256": _sha256_lines(candidates),
        "data_split": DATA_SPLIT["fresh"],
        "logs": logs,
        "cities": {log: cities[log] for log in logs},
        "windows": {log: len(windows[log]) for log in logs},
        "windows_per_city": dict(sorted(per_city.items())),
        "n_tokens": len(tokens),
        "tokens_sha256": _sha256_lines(tokens),
    }


def load_fresh(path: Path, trainval_logs: list, filters: dict) -> dict:
    """The committed R-clean list, held to its rule without reading a log: refused if edited.

    Re-derives the candidates from the logs on disk and the vendor's filters; the windows themselves
    are re-derived by `check_fresh` (it reads every candidate log), which the list writer and the
    labels pass run.
    """
    stored = json.loads(Path(path).read_text())
    candidates = sorted(set(trainval_logs) - named_logs(filters))
    problems = []
    if stored["sources"] != {name: filt["sha256"] for name, filt in sorted(filters.items())}:
        problems.append("the vendor's scene filters changed since the list was written")
    if stored["trainval_logs_sha256"] != _sha256_lines(sorted(trainval_logs)):
        problems.append("the trainval logs on disk changed since the list was written")
    if stored["candidates_sha256"] != _sha256_lines(candidates):
        problems.append("the candidate logs differ from what the rule gives")
    if not set(stored["logs"]) <= set(candidates):
        problems.append("a listed log is not a candidate")
    if sorted(stored["windows"]) != stored["logs"] or sum(stored["windows"].values()) != stored["n_tokens"]:
        problems.append("the window counts do not add up to n_tokens")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return stored


# ----------------------------------------------------------------------------------- the passes


def scene_filter(code: Path, name: str, window: bool = True) -> dict:
    """One of the vendor's scene filters, read from the Drive-JEPA tree that runs (not restated here).

    Adds `file` (relative to the tree) and `sha256`. With `window`, refuses a filter whose window is
    not the one these labels assume.
    """
    import yaml

    path = Path(code) / FILTER_DIR / f"{name}.yaml"
    filt = yaml.safe_load(path.read_text())
    if window and (filt["num_history_frames"], filt["num_future_frames"], filt["frame_interval"]) != (
        HISTORY,
        FUTURE,
        1,
    ):
        raise SystemExit(f"{path}: the window is not the one these labels assume")
    filt["file"] = f"{FILTER_DIR}/{name}.yaml"
    filt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return filt


def navtrain_filter(code: Path) -> dict:
    """navtrain's scene filter, read from the Drive-JEPA tree that runs (not restated here)."""
    return scene_filter(code, "navtrain")


def all_filters(code: Path) -> dict:
    """Every scene filter the vendor ships, by name: what "no NAVSIM split uses" is read against."""
    return {
        path.stem: scene_filter(code, path.stem, window=False)
        for path in sorted((Path(code) / FILTER_DIR).glob("*.yaml"))
    }


def trainval_logs(navsim_root: Path) -> list:
    """The trainval logs on disk, by name."""
    return sorted(path.stem for path in (Path(navsim_root) / "navsim_logs/trainval").glob("*.pkl"))


def source_of(manifest: dict) -> str:
    """The source a labels pass read: navtrain unless its manifest says otherwise (labels-v1 predates it)."""
    return manifest.get("source", "navtrain")


def source_spec(code: Path, navsim_root: Path, source: str) -> dict:
    """What a pass reads for `source`: its logs, its window tokens (None: every routed window, checked
    against the list's sha256 afterwards), its data split, and the provenance its manifest records."""
    if source == "navtrain":
        filt = navtrain_filter(code)
        return {
            "source": source,
            "data_split": DATA_SPLIT[source],
            "logs": sorted(filt["log_names"]),
            "tokens": set(filt["tokens"]),
            "provenance": {"navtrain_yaml_sha256": filt["sha256"]},
        }
    path = REPO / LISTS[source]
    list_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if source == "navtest":
        filt = scene_filter(code, "navtest")
        record = load_navtest(path, filt)
        return {
            "source": source,
            "data_split": DATA_SPLIT[source],
            "logs": record["logs"],
            "tokens": set(filt["tokens"]),
            "provenance": {
                "navtest_yaml_sha256": filt["sha256"],
                "list": LISTS[source],
                "list_sha256": list_sha,
            },
        }
    if source == "fresh":
        record = load_fresh(path, trainval_logs(navsim_root), all_filters(code))
        return {
            "source": source,
            "data_split": DATA_SPLIT[source],
            "logs": record["logs"],
            "tokens": None,
            "tokens_sha256": record["tokens_sha256"],
            "cities": record["cities"],
            "provenance": {"list": LISTS[source], "list_sha256": list_sha},
        }
    raise SystemExit(f"unknown source {source!r}; one of {SOURCES}")


def log_windows(log: str, logs_dir: Path, tokens) -> dict:
    """token -> its 14 raw frames, for one log, by the vendor's own `filter_scenes`.

    `tokens` limits the windows to those tokens; None keeps every routed window (the R-clean list).
    """
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import filter_scenes

    scene_filter = SceneFilter(
        num_history_frames=HISTORY,
        num_future_frames=FUTURE,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=[log],
        tokens=None if tokens is None else list(tokens),
    )
    windows, _ = filter_scenes(logs_dir, scene_filter)
    return windows


def city_of(windows: dict):
    """A log's city, from its windows' current frames (`map_location`, as the vendor reads it); None if
    it has no window. Refuses a log whose windows disagree."""
    cities = {frames[CURRENT]["map_location"] for frames in windows.values()}
    if len(cities) > 1:
        raise SystemExit(f"one log's windows name {sorted(cities)}")
    return cities.pop() if cities else None


def _routed_windows_of(log: str):
    windows = log_windows(log, _WORKER["logs_dir"], None)
    return log, sorted(windows), city_of(windows)


def fresh_windows(navsim_root: Path, candidates: list, workers: int) -> tuple:
    """Each candidate log's routed window tokens and its city, read from the logs."""
    from multiprocessing import get_context

    _WORKER.update(logs_dir=Path(navsim_root) / "navsim_logs/trainval")
    windows, cities = {}, {}
    with get_context("fork").Pool(workers) as pool:
        for log, tokens, city in pool.imap(_routed_windows_of, candidates):
            windows[log] = tokens
            if city is not None:
                cities[log] = city
    return windows, cities


def write_list(cfg, which: str, path: Path, workers: int) -> dict:
    """Write navtest's or R-clean's list once; if it exists, derive it again and refuse any difference.

    So a rerun is a check: `fresh` reads every candidate log again (`check_fresh` in one call).
    """
    from gallops.drivejepa_perception import env

    entered = env.enter(cfg)
    code, root = Path(entered["code"]), Path(entered["navsim_root"])
    if which == "navtest":
        record = navtest_record(scene_filter(code, "navtest"))
    else:
        logs, filters = trainval_logs(root), all_filters(code)
        candidates = sorted(set(logs) - named_logs(filters))
        windows, cities = fresh_windows(root, candidates, workers)
        record = fresh_record(logs, filters, windows, cities)
    path = Path(path)
    if path.exists():
        stored = json.loads(path.read_text())
        differ = sorted(key for key in set(stored) | set(record) if stored.get(key) != record.get(key))
        if differ:
            raise SystemExit(f"{path}: {differ} differ from what the rule gives now")
        print(f"{path}: derived again, equal")
    else:
        path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("logs", "cities", "windows")}, indent=1))
    return record


def global_poses(frames: list) -> np.ndarray:
    """The global ego pose of each frame, by NAVSIM's own `Scene._build_ego_status`."""
    from navsim.common.dataclasses import Scene

    return np.array([Scene._build_ego_status(frame).ego_pose for frame in frames], dtype=np.float64)


_WORKER: dict = {}


def _labels_of_log(log: str):
    windows = log_windows(log, _WORKER["logs_dir"], _WORKER["tokens"])
    rows = []
    for token, frames in windows.items():
        rows.append((token, window_labels(frames, global_poses(frames)), int(frames[CURRENT]["timestamp"])))
    return log, rows, city_of(windows)


def build_labels(cfg, out: Path, workers: int, limit_logs, source: str = "navtrain") -> None:
    """The labels pass: every window of `source`, written once under `out`, never overwritten.

    Refuses to finish unless the windows read are exactly the source's: navtrain's and navtest's
    tokens, or the R-clean list's sha256 and cities (which re-derives that list from the logs).
    """
    from multiprocessing import get_context

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    sha = git_sha()  # at the start: the tree may move during a long pass
    entered = env.enter(cfg)
    spec = source_spec(Path(entered["code"]), Path(entered["navsim_root"]), source)
    logs = spec["logs"][: limit_logs or None]
    _WORKER.update(
        logs_dir=Path(entered["navsim_root"]) / "navsim_logs" / spec["data_split"], tokens=spec["tokens"]
    )

    tokens, log_ids, samples, stamps, cities = [], [], [], [], []
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        for k, (log, rows, city) in enumerate(pool.imap(_labels_of_log, logs)):
            cities.append(city)
            for token, labels, stamp in rows:
                tokens.append(token)
                log_ids.append(logs.index(log))
                samples.append(labels)
                stamps.append(stamp)
            if k % 100 == 0:
                print(
                    f"{k + 1}/{len(logs)} logs, {len(tokens)} samples, {time.time() - t0:.0f} s", flush=True
                )

    if not limit_logs:
        if spec["tokens"] is not None and set(tokens) != spec["tokens"]:
            raise SystemExit(
                f"{len(spec['tokens'] - set(tokens))} of the source's tokens have no window here, and "
                f"{len(set(tokens) - spec['tokens'])} windows are not the source's"
            )
        if spec["tokens"] is None and _sha256_lines(sorted(tokens)) != spec["tokens_sha256"]:
            raise SystemExit(f"the {len(tokens)} windows read differ from the {source} list's")
        if "cities" in spec and dict(zip(logs, cities)) != spec["cities"]:  # noqa: B905 -- Python 3.9
            raise SystemExit(f"the logs' cities differ from the {source} list's")

    packed = pack(samples)
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(
        staging / "labels.npz",
        token=np.array(tokens),
        log_id=np.array(log_ids, dtype=np.int16),
        timestamp=np.array(stamps, dtype=np.int64),
        **packed,
    )
    counts = np.bincount(packed["cls"].astype(np.int64), minlength=len(CLASSES)) if len(packed["cls"]) else []
    manifest = {
        "pass": "labels",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "source": source,
        "data_split": spec["data_split"],
        **spec["provenance"],
        "logs": logs,
        "cities": cities,
        "samples": len(tokens),
        "tokens_expected": len(spec["tokens"]) if spec["tokens"] is not None else None,
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
    split = DATA_SPLIT[source_of(manifest)]
    logs_dir = Path(entered["navsim_root"]) / "navsim_logs" / split
    rng = np.random.default_rng(0)
    rows = sorted(rng.choice(len(stored["token"]), size=min(n_windows, len(stored["token"])), replace=False))
    problems, worst = [], {"ego": 0.0, "path": 0.0, "joins": 0, "size_over": 0}
    for row in rows:
        token, log = str(stored["token"][row]), logs[int(stored["log_id"][row])]
        frames = log_windows(log, logs_dir, {token})[token]
        poses = global_poses(frames)
        scene = Scene.from_scene_dict_list(
            frames,
            Path(entered["navsim_root"]) / "sensor_blobs" / split,
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
    p.add_argument("--source", choices=SOURCES, default="navtrain")
    p = sub.add_parser("check")
    p.add_argument("--labels", required=True)
    p.add_argument("--windows", type=int, default=60)
    for name in ("navtest-list", "fresh-list"):
        p = sub.add_parser(name)
        p.add_argument("--write", required=True, help="the committed list; if it exists, derived again")
        p.add_argument("--workers", type=int, default=24)
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    if args.cmd == "splits":
        filt = navtrain_filter(Path(cfg.resolve("{code.drivejepa}")))
        select = [line.strip() for line in Path(args.select).read_text().splitlines() if line.strip()]
        splits = draw_splits(filt["log_names"], select)
        Path(args.write).write_text(json.dumps(splits, indent=1) + "\n")
        print({k: len(splits[k]) for k in ("select", "report", "train")})
    elif args.cmd == "labels":
        build_labels(cfg, Path(args.out), args.workers, args.limit_logs, args.source)
    elif args.cmd in ("navtest-list", "fresh-list"):
        write_list(cfg, args.cmd.split("-")[0], Path(args.write), args.workers)
    else:
        sys.exit(1 if check_labels(cfg, Path(args.labels), args.windows) else 0)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
