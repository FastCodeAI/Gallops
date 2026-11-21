"""Which navtrain frames are moments, how they chain, and which are graded.

    python -m gallops.memory.ticks build [--out DIR] [--limit N]    # one pass over the log pickles
    python -m gallops.memory.ticks report [--out DIR]               # counts, chains, missing-image check

**A moment** is a log frame whose front image (CAM_F0) exists and whose predecessor, at most
0.55 s earlier, has one too: the two frames the encoder reads (`drive_jepa_features.py`
`_get_camera_feature` takes ``cameras[-1]`` and ``cameras[-2]``). The model also reads the current
frame's back, left and right cameras for its image geometry (`bev_feature_build.py`
`LoadMultiViewImageFromFiles` skips a missing camera, which would shift the front camera out of the
slot the model keeps), so a moment records whether all four are on disk: `inputs_ok`.

**Images** come from one root, `blobs`, or two when a site's copy is incomplete: `blobs` first,
then `extra_blobs` (`image_roots`). At the site that built the first cache the main copy lacks the
current frames of 298 logs, and a second unpack of navtrain_current_1 has 134 of them. A frame is
looked up by file name in each root's directory listing.

**A chain** is a run of consecutive frames that all have a front image, each at most 0.55 s after
the last. A moment's `chain_pos` is how many frames precede it in its chain, so the memory has
`chain_pos` earlier moments to warm up on; the chain's first frame is not a moment.

**Graded** moments are navtrain tokens the train metric cache has an entry for: the ones whose 32
proposals can be scored against the vendor's target. **Split** is the vendor's log split
(`default_train_val_test_log_split.yaml`): 978 train and 214 val of the 1,192 graded logs.

**Dynamics**, for the missing-image check (`report`): forward speed and acceleration are the log's
own `ego_dynamic_state` (vx, vy, ax, ay, ego frame); an *event* is a frame whose forward speed
changes by more than 1 m/s within the next 2 s, and a *braking onset* a frame not braking now
(ax >= -1.5 m/s^2) with ax below -1.5 m/s^2 within the next 2 s. Both are judged only where the
log has the full 2 s ahead.

Runs in either interpreter (numpy only); the build reads 14 GB of pickles and takes ~10 min.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

#: The longest gap between two frames of one chain: 0.5 s at NAVSIM's rate, plus the jitter
#: the logs carry (all observed gaps are 0.5 s exactly, 2026-11-03).
MAX_GAP_US = 550_000
#: The cameras the model reads at the current frame, in the vendor's order (b0, f0, l0, r0).
CURRENT_CAMERAS = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0")
FRONT = "CAM_F0"
#: NAVSIM's history depth: the model is handed four frames, 0.5 s apart.
HISTORY_FRAMES = 4

EVENT_DV = 1.0  # m/s
BRAKE_AX = -1.5  # m/s^2
HORIZON_S = 2.0
FRAME_DT_S = 0.5

SPLIT_FILE = "navsim_v2/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
SPLITS = ("train_logs", "val_logs", "test_logs")

#: Per-moment columns of moments.npz, besides the string columns `log` and `token`.
INT_COLUMNS = ("frame_idx", "chain", "chain_pos", "root_now", "root_prev")
BOOL_COLUMNS = ("graded", "inputs_ok")


# --- pure parts, tested by make check ----------------------------------------------------------


def log_split(split_yaml: Path) -> dict[str, str]:
    """log name -> "train" / "val" / "test", from the vendor's split file.

    Parsed by line rather than with yaml: the file is 16k flat entries, and the shape is fixed.
    """
    split, out = None, {}
    for line in Path(split_yaml).read_text().splitlines():
        head = re.match(r"^(\w+):", line)
        if head:
            split = head.group(1) if head.group(1) in SPLITS else None
            continue
        item = line.strip()
        if split and item.startswith("- "):
            out[item[2:].strip()] = split[: -len("_logs")]
    return out


def graded_tokens(metadata_csv: Path) -> dict[str, set]:
    """log -> the tokens the train metric cache has an entry for.

    Each row is `<root>/<log>/unknown/<token>/metric_cache.pkl`.
    """
    out: dict[str, set] = defaultdict(set)
    lines = Path(metadata_csv).read_text().splitlines()
    for line in lines[1:]:
        parts = line.strip().split("/")
        if len(parts) >= 4:
            out[parts[-4]].add(parts[-2])
    return dict(out)


def image_root(frame: dict, camera: str, listings: list) -> int:
    """Index of the first root whose listing holds this frame's image for `camera`; -1 if none.

    `listings` is one dict per root, camera -> set of file names (or None: no such directory).
    """
    name = os.path.basename(frame["cams"][camera]["data_path"])
    for i, listing in enumerate(listings):
        files = listing.get(camera)
        if files is not None and name in files:
            return i
    return -1


def log_moments(frames: list, listings: list, graded: set) -> list[dict]:
    """Every moment of one log, in frame order, with its chain and roots.

    `frames` is the log pickle's list, `listings` as in `image_root`, `graded` the log's graded
    tokens.
    """
    front = [image_root(f, FRONT, listings) for f in frames]
    out = []
    chain, pos = -1, 0
    for i, f in enumerate(frames):
        if front[i] < 0:
            pos = 0
            continue
        joined = i > 0 and front[i - 1] >= 0 and 0 < f["timestamp"] - frames[i - 1]["timestamp"] <= MAX_GAP_US
        if not joined:
            chain += 1
            pos = 0
            continue  # a chain's first frame has no predecessor image: not a moment
        pos += 1
        roots = [image_root(f, c, listings) for c in CURRENT_CAMERAS]
        out.append(
            {
                "log": f["log_name"],
                "token": f["token"],
                "frame_idx": i,
                "timestamp": int(f["timestamp"]),
                "chain": chain,
                "chain_pos": pos,
                "root_now": front[i],
                "root_prev": front[i - 1],
                "graded": f["token"] in graded,
                "inputs_ok": all(r >= 0 for r in roots),
            }
        )
    return out


def history_indices(frame_idx: int) -> list[int]:
    """The four frames handed to the model for a moment, oldest first, clamped at the log start.

    Paddock's node picks, for each of now, -0.5, -1.0 and -1.5 s, the nearest frame it has
    (`drivejepa/policy.py` `_select_history`); on a 2 Hz log that is the frame itself, and at the
    log start the earliest frame, repeated.
    """
    return [max(frame_idx - k, 0) for k in range(HISTORY_FRAMES - 1, -1, -1)]


def dynamics(frames: list) -> dict[str, np.ndarray]:
    """Per frame: forward speed and acceleration, and the event / braking-onset flags.

    The flags are judged only where the log reaches `HORIZON_S` ahead (`judged`).
    """
    state = np.asarray([f["ego_dynamic_state"] for f in frames], dtype=np.float64).reshape(len(frames), -1)
    vx, ax = state[:, 0], state[:, 2]
    n = len(frames)
    steps = int(round(HORIZON_S / FRAME_DT_S))
    judged = np.zeros(n, bool)
    event = np.zeros(n, bool)
    onset = np.zeros(n, bool)
    braking = ax < BRAKE_AX
    for i in range(n - steps):
        judged[i] = True
        ahead = slice(i + 1, i + steps + 1)
        event[i] = bool(np.any(np.abs(vx[ahead] - vx[i]) > EVENT_DV))
        onset[i] = (not braking[i]) and bool(np.any(braking[ahead]))
    return {"speed": vx, "accel": ax, "judged": judged, "event": event, "onset": onset, "braking": braking}


def chain_histogram(moments: list[dict]) -> dict:
    """Chain lengths (frames, first included) and how many moments end at least 8 / 16 frames."""
    lengths = Counter()
    for m in moments:
        lengths[(m["log"], m["chain"])] = max(lengths[(m["log"], m["chain"])], m["chain_pos"] + 1)
    bins = Counter(lengths.values())

    def ends(rows, k):
        return sum(1 for m in rows if m["chain_pos"] + 1 >= k)

    graded = [m for m in moments if m["graded"]]
    return {
        "chains": len(lengths),
        "length_counts": {str(k): bins[k] for k in sorted(bins)},
        "moments_with_8_frames": ends(moments, 8),
        "moments_with_16_frames": ends(moments, 16),
        "graded_with_8_frames": ends(graded, 8),
        "graded_with_16_frames": ends(graded, 16),
    }


# --- the build ---------------------------------------------------------------------------------


def _listing(root: Path, log: str) -> dict:
    out = {}
    for camera in CURRENT_CAMERAS:
        try:
            out[camera] = set(os.listdir(root / log / camera))
        except FileNotFoundError:
            out[camera] = None
    return out


def image_roots(data) -> list[Path]:
    """The navtrain image roots, searched in order: `blobs`, then `extra_blobs` where a site has one."""
    roots = [data.extra["blobs"]]
    if data.extra.get("extra_blobs"):
        roots.append(data.extra["extra_blobs"])
    return roots


def build(cfg, out: Path, limit: int = 0) -> Path:
    """One pass over every graded log: moments.npz, logs.json (per-log dynamics), manifest.json."""
    from gallops.drivejepa_b2d.provenance import git_sha

    data = cfg.datasets["navtrain"]
    logs_dir, roots = data.extra["logs"], image_roots(data)
    metadata = sorted((data.extra["metric_cache"] / "metadata").glob("*.csv"))
    if len(metadata) != 1:
        raise SystemExit(
            f"expected one metadata csv under {data.extra['metric_cache']}/metadata, got {metadata}"
        )
    split = log_split(Path(cfg.resolve("{code.drivejepa}")) / SPLIT_FILE)
    graded = graded_tokens(metadata[0])

    final = Path(out)
    if final.exists():
        raise SystemExit(f"{final} exists; the moments table is written once -- choose a new --out")
    staging = final.with_name(final.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.time()
    provenance = {"git_sha": git_sha(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    names = sorted(graded)
    if limit:
        names = names[:limit]
    moments, per_log = [], {}
    for k, log in enumerate(names):
        with open(logs_dir / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        listings = [_listing(r, log) for r in roots]
        rows = log_moments(frames, listings, graded[log])
        for r in rows:
            r["split"] = split.get(log, "none")
        moments.extend(rows)
        dyn = dynamics(frames)
        tokens = graded[log]
        at_token = np.asarray([f["token"] in tokens for f in frames])
        use = at_token & dyn["judged"]
        first = frames[0]
        per_log[log] = {
            "split": split.get(log, "none"),
            "city": first.get("map_location"),
            "vehicle": first.get("vehicle_name"),
            "frames": len(frames),
            "graded_tokens": int(at_token.sum()),
            # graded tokens whose own front image is on disk: zero for the logs neither root has
            "graded_with_front_now": int(
                sum(image_root(f, FRONT, listings) >= 0 for f in frames if f["token"] in tokens)
            ),
            "judged": int(use.sum()),
            "events": int((dyn["event"] & use).sum()),
            "not_braking": int((~dyn["braking"] & use).sum()),
            "onsets": int((dyn["onset"] & use).sum()),
            "speed_sum": float(dyn["speed"][at_token].sum()),
        }
        if (k + 1) % 100 == 0:
            print(
                f"  {k + 1} of {len(names)} logs, {len(moments)} moments, {time.time() - started:.0f} s",
                flush=True,
            )

    arrays = {
        "log": np.asarray([m["log"] for m in moments]),
        "token": np.asarray([m["token"] for m in moments]),
        "split": np.asarray([m["split"] for m in moments]),
        "timestamp": np.asarray([m["timestamp"] for m in moments], np.int64),
        **{c: np.asarray([m[c] for m in moments], np.int32) for c in INT_COLUMNS},
        **{c: np.asarray([m[c] for m in moments], bool) for c in BOOL_COLUMNS},
    }
    np.savez(staging / "moments.npz", **arrays)
    (staging / "logs.json").write_text(json.dumps(per_log, indent=1) + "\n")
    manifest = {
        **provenance,
        "logs": len(names),
        "moments": len(moments),
        "roots": [str(r) for r in roots],
        "metadata_csv": str(metadata[0]),
        "split_file": SPLIT_FILE,
        "limit": limit or None,
        "seconds": round(time.time() - started, 1),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    staging.rename(final)
    print(f"{len(moments)} moments from {len(names)} logs in {manifest['seconds']:.0f} s -> {final}")
    return final


def load(directory: Path) -> dict:
    """moments.npz as a dict of arrays, logs.json and the manifest."""
    directory = Path(directory)
    with np.load(directory / "moments.npz") as z:
        out = {k: z[k] for k in z.files}
    out["logs"] = json.loads((directory / "logs.json").read_text())
    out["manifest"] = json.loads((directory / "manifest.json").read_text())
    return out


# --- the report: counts, chains and the missing-image check ------------------------------------

#: The missing-image check's flag: a relative difference this large, with a CI excluding zero.
DIFFERENT_RELATIVE = 0.25


def _rates(logs: list[dict]) -> dict[str, float]:
    judged = sum(r["judged"] for r in logs)
    not_braking = sum(r["not_braking"] for r in logs)
    tokens = sum(r["graded_tokens"] for r in logs)
    return {
        "event_rate": sum(r["events"] for r in logs) / judged if judged else float("nan"),
        "onset_rate": sum(r["onsets"] for r in logs) / not_braking if not_braking else float("nan"),
        "mean_speed": sum(r["speed_sum"] for r in logs) / tokens if tokens else float("nan"),
    }


def compare_groups(per_log: dict, missing: set, reps: int = 2000, seed: int = 0) -> dict:
    """Missing logs vs kept logs: pooled rates, their difference and a log-clustered bootstrap CI.

    Each bootstrap draw resamples logs with replacement within each group and pools again, so a
    log's moments move together. `flag` is DIFFERENT when the event or the braking-onset rate
    differs by more than `DIFFERENT_RELATIVE` of the kept value and its 95 % CI excludes zero.
    """
    miss = [per_log[n] for n in sorted(per_log) if n in missing]
    kept = [per_log[n] for n in sorted(per_log) if n not in missing]
    rng = np.random.default_rng(seed)
    a, b = _rates(miss), _rates(kept)
    draws = {k: [] for k in a}
    for _ in range(reps):
        ra = _rates([miss[i] for i in rng.integers(0, len(miss), len(miss))])
        rb = _rates([kept[i] for i in rng.integers(0, len(kept), len(kept))])
        for k in a:
            draws[k].append(ra[k] - rb[k])
    out = {"missing_logs": len(miss), "kept_logs": len(kept), "metrics": {}}
    different = False
    for k in a:
        lo, hi = (float(x) for x in np.percentile(draws[k], [2.5, 97.5]))
        diff = a[k] - b[k]
        relative = diff / b[k] if b[k] else float("nan")
        excludes_zero = lo > 0 or hi < 0
        out["metrics"][k] = {
            "missing": a[k],
            "kept": b[k],
            "difference": diff,
            "ci95": [lo, hi],
            "relative": relative,
            "ci_excludes_zero": excludes_zero,
        }
        if k in ("event_rate", "onset_rate") and abs(relative) > DIFFERENT_RELATIVE and excludes_zero:
            different = True
    out["flag"] = "DIFFERENT" if different else "SIMILAR"

    def mix(rows, key):
        total = sum(r["graded_tokens"] for r in rows) or 1
        counts = Counter()
        for r in rows:
            counts[r[key]] += r["graded_tokens"]
        return {str(k): round(v / total, 4) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}

    out["city_share"] = {"missing": mix(miss, "city"), "kept": mix(kept, "city")}
    out["vehicle_share"] = {"missing": mix(miss, "vehicle"), "kept": mix(kept, "vehicle")}
    return out


def ours_only_positions(table: dict) -> np.ndarray:
    """Each moment's chain position counting only frames in the first root (Naren's copy).

    The plan's chain counts (27 Oct) were made on that copy alone; this lets them be reconciled.
    -1 where the moment or its predecessor is not in the first root.
    """
    order = np.lexsort((table["frame_idx"], table["log"]))
    pos = np.full(len(order), -1, np.int32)
    last_log, last_frame, run = None, None, 0
    for i in order:
        log, frame = table["log"][i], int(table["frame_idx"][i])
        ours = table["root_now"][i] == 0 and table["root_prev"][i] == 0
        if not ours:
            last_log, run = None, 0
            continue
        run = run + 1 if (log == last_log and frame == last_frame + 1) else 1
        pos[i] = run
        last_log, last_frame = log, frame
    return pos


def report(directory: Path) -> dict:
    table = load(directory)
    per_log = table["logs"]
    graded = table["graded"]
    ok = table["inputs_ok"]
    split = table["split"]
    moments = [
        {
            "log": table["log"][i],
            "chain": int(table["chain"][i]),
            "chain_pos": int(table["chain_pos"][i]),
            "graded": bool(graded[i]),
        }
        for i in range(len(graded))
    ]
    hist = chain_histogram(moments)
    ours = ours_only_positions(table)
    missing = {n for n, r in per_log.items() if r["graded_tokens"] and not r["graded_with_front_now"]}
    total_graded_tokens = sum(r["graded_tokens"] for r in per_log.values())
    out = {
        "moments": int(len(graded)),
        "moments_inputs_ok": int(ok.sum()),
        "graded_moments": int(graded.sum()),
        "graded_moments_inputs_ok": int((graded & ok).sum()),
        "graded_tokens_in_metric_cache": int(total_graded_tokens),
        "graded_tokens_with_front_now": int(sum(r["graded_with_front_now"] for r in per_log.values())),
        "per_split": {
            str(s): {
                "moments": int((split == s).sum()),
                "graded": int((graded & (split == s)).sum()),
                "graded_inputs_ok": int((graded & ok & (split == s)).sum()),
                "logs": len({str(x) for x in table["log"][split == s]}),
            }
            for s in sorted(set(split.tolist()))
        },
        "chains": hist,
        "ours_only": {
            "graded_with_8_frames": int((graded & (ours + 1 >= 8)).sum()),
            "graded_with_16_frames": int((graded & (ours + 1 >= 16)).sum()),
        },
        "moments_from_second_root": int(((table["root_now"] == 1) | (table["root_prev"] == 1)).sum()),
        "missing_logs": len(missing),
        "missing_check": compare_groups(per_log, missing),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("build", "report"))
    ap.add_argument("--out", type=Path, help="default: {data.navtrain.cache}/ticks")
    ap.add_argument("--limit", type=int, default=0, help="only the first N logs (a quick look)")
    args = ap.parse_args()

    from gallops.config import SiteConfig

    cfg = SiteConfig.load()
    out = args.out or Path(cfg.resolve("{data.navtrain.cache}")) / "ticks"
    if args.part == "build":
        build(cfg, out, args.limit)
    else:
        result = report(out)
        (out / "report.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
