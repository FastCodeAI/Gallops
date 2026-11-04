"""Which of the previous moment's 32 paths each new path inherits its memory from.

H15 hands each path's memory from one moment to the next (plan, "How one path's memory reaches the
next moment"). Slot numbers are not enough: the slot that is closest is the same slot only 38% of
the time (Naren's slot-stability measurement, 40 val logs). So, for every pair of back-to-back
moments in a chain (same log and chain, chain_pos one apart, 0.5 s):

1. **Align.** The previous moment's proposals are in its own ego frame. Their waypoints 2-8 (1.0 to
   4.0 s after it, so 0.5 to 3.5 s after the current moment) are moved into the current ego frame
   with the two moments' logged poses (`ego2global` from the log pickles).
2. **Compare** them with the current proposals' waypoints 1-7 (0.5 to 3.5 s): the distance between
   two paths is the mean over those 7 waypoints of the x, y distance (`distances`, old x new).
3. **Match.** Each new path takes the memory of the old path closest to it; -1 when that is farther
   than `LIMIT` (2 m, fixed in H15's PREDICTIONS.md) or when the moment starts its chain.

This is `slot_stability.py`'s geometry (paddock-runs `ops/memory_h15/`), which measured from the old
path's side (`D.min(1)`, for each old path its closest new one); the memory needs the new path's side
(`D.min(0)`), and `pair_stats` reports both.

Stored (`{data.navtrain.cache}/align/`), in moments-table order:

    poses.npy       (N, 4, 4) float64  ego2global of each moment's frame
    match.npy       (N, 32) int16      the old slot each new path inherits from, or -1
    match_m.npy     (N, 32) float32    its distance in metres (NaN at a chain start)
    prev.npy        (N,) int64         the previous moment's row, or -1 at a chain start

    python -m gallops.memory.align build                 # every moment
    python -m gallops.memory.align stats --logs val      # match statistics, as slot_stability reports them
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict

LIMIT = 2.0  #: metres; beyond it a new path starts with an empty memory (PREDICTIONS.md)
SLOT_STABILITY_SEED = 20261105  #: slot_stability.py's sample of 40 val logs


# --- geometry (tested) -----------------------------------------------------------------------------


def pose_of(frame: dict) -> np.ndarray:
    """A log frame's ego2global as 4 x 4, from the matrix or from translation + quaternion (w, x, y, z)."""
    if "ego2global" in frame:
        return np.asarray(frame["ego2global"], float)
    w, x, y, z = frame["ego2global_rotation"]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = frame["ego2global_translation"]
    return out


def align_previous(previous: np.ndarray, pose_previous: np.ndarray, pose_now: np.ndarray) -> np.ndarray:
    """(32, 7, 2): the previous proposals' waypoints 2-8, in the current ego frame."""
    relative = np.linalg.inv(pose_previous) @ pose_now  # the current frame, seen from the previous one
    c, s = relative[0, 0], relative[1, 0]
    tx, ty = relative[0, 3], relative[1, 3]
    points = np.asarray(previous, float)[:, 1:, :2]
    dx, dy = points[..., 0] - tx, points[..., 1] - ty
    return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)


def distances(aligned_previous: np.ndarray, now: np.ndarray) -> np.ndarray:
    """(old 32, new 32) mean distance over the 3.5 s the two share (old 2-8 against new 1-7)."""
    current = np.asarray(now, float)[:, :7, :2]
    return np.linalg.norm(aligned_previous[:, None] - current[None], axis=-1).mean(-1)


def match(d: np.ndarray, limit: float = LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """For each new path (column), the closest old path and its distance; -1 beyond `limit`."""
    best = d.argmin(0)
    best_m = d[best, np.arange(d.shape[1])]
    return np.where(best_m <= limit, best, -1).astype(np.int16), best_m.astype(np.float32)


def previous_rows(table: dict) -> np.ndarray:
    """Each row's predecessor (same log and chain, chain_pos one less), or -1."""
    logs, chains, pos = table["log"].tolist(), table["chain"].tolist(), table["chain_pos"].tolist()
    index = {key: i for i, key in enumerate(zip_strict(logs, chains, pos))}
    return np.array([index.get((lg, ch, p - 1), -1) for lg, ch, p in zip_strict(logs, chains, pos)], np.int64)


# --- building and statistics -----------------------------------------------------------------------


def _log_frames(logs_dir: Path, log: str) -> list:
    with open(logs_dir / f"{log}.pkl", "rb") as f:
        return pickle.load(f)


def build(cfg, out: Path) -> dict:
    from gallops.memory import ticks

    cache = Path(cfg.resolve("{data.navtrain.cache}"))
    logs_dir = Path(cfg.datasets["navtrain"].extra["logs"])
    table = ticks.load(cache / "ticks")
    proposals = np.load(cache / "release" / "proposals.npy", mmap_mode="r")
    n = len(table["log"])
    prev = previous_rows(table)
    poses = np.zeros((n, 4, 4))
    matched = np.full((n, 32), -1, np.int16)
    matched_m = np.full((n, 32), np.nan, np.float32)
    started = time.time()
    names = sorted(set(table["log"].tolist()))
    for k, log in enumerate(names):
        rows = np.flatnonzero(table["log"] == log)
        frames = _log_frames(logs_dir, log)
        for r in rows:
            poses[r] = pose_of(frames[int(table["frame_idx"][r])])
        for r in rows:
            p = prev[r]
            if p < 0:
                continue
            d = distances(align_previous(proposals[p], poses[p], poses[r]), proposals[r])
            matched[r], matched_m[r] = match(d)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1} of {len(names)} logs ({time.time() - started:.0f} s)", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, array in (("poses", poses), ("match", matched), ("match_m", matched_m), ("prev", prev)):
        np.save(out / f"{name}.npy", array)
    from gallops.drivejepa_b2d.provenance import git_sha

    manifest = {
        "git_sha": git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": n,
        "pairs": int((prev >= 0).sum()),
        "limit_m": LIMIT,
        "seconds": round(time.time() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def pair_stats(table: dict, prev: np.ndarray, poses: np.ndarray, proposals, logits, rows) -> dict:
    """slot_stability.py's numbers, plus the new-path side the memory uses, over back-to-back `rows`."""
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]  # noqa: E731
    same, old_near, old_is_same, new_near, new_is_same, none = [], [], [], [], [], []
    pick_same, pick_d, pick_best = [], [], []
    idx = np.arange(32)
    for r in rows:
        p = prev[r]
        d = distances(align_previous(proposals[p], poses[p], poses[r]), proposals[r])
        same.append(d[idx, idx])
        old_near.append(d.min(1))
        old_is_same.append(d.argmin(1) == idx)
        m, m_d = match(d)
        new_near.append(m_d)
        new_is_same.append(m == idx)
        none.append(m == -1)
        la, lb = np.asarray(logits[p])[:, -1], np.asarray(logits[r])[:, -1]
        pa, pb = int(la.argmax()), int(lb.argmax())
        pick_same.append(pa == pb)
        pick_d.append(d[pa, pb])
        pick_best.append(d[pa].min())
    cat = np.concatenate
    return {
        "moment_pairs": int(len(rows)),
        "logs": int(len(set(table["log"][rows].tolist()))),
        "same_slot_m_p10_50_90": q(cat(same)),
        "nearest_slot_m_p10_50_90": q(cat(old_near)),
        "nearest_is_same_slot": round(float(cat(old_is_same).mean()), 4),
        "picked_slot_unchanged": round(float(np.mean(pick_same)), 4),
        "pick_to_pick_m_p10_50_90": q(np.array(pick_d)),
        "pick_to_pick_over_1m": round(float((np.array(pick_d) > 1).mean()), 4),
        "best_available_for_old_pick_m_p10_50_90": q(np.array(pick_best)),
        "new_path_closest_old_m_p10_50_90": q(cat(new_near)),
        "new_path_closest_old_is_same_slot": round(float(cat(new_is_same).mean()), 4),
        "new_path_unmatched_beyond_limit": round(float(cat(none).mean()), 4),
    }


def stats(cfg, which: str) -> dict:
    """Match statistics over `which` = 'slot40' (slot_stability.py's sample) or 'val' (all val logs)."""
    from gallops.memory import ticks

    cache = Path(cfg.resolve("{data.navtrain.cache}"))
    logs_dir = Path(cfg.datasets["navtrain"].extra["logs"])
    table = ticks.load(cache / "ticks")
    proposals = np.load(cache / "release" / "proposals.npy", mmap_mode="r")
    logits = np.load(cache / "release" / "logits.npy", mmap_mode="r")
    val = np.unique(table["log"][table["split"] == "val"])
    if which == "slot40":
        names = np.random.default_rng(SLOT_STABILITY_SEED).choice(val, size=40, replace=False)
    else:
        names = val
    prev = previous_rows(table)
    rows = np.flatnonzero(np.isin(table["log"], names) & (prev >= 0))
    poses = np.zeros((len(table["log"]), 4, 4))
    for log in sorted(set(table["log"][rows].tolist())):
        frames = _log_frames(logs_dir, log)
        for r in np.flatnonzero(table["log"] == log):
            poses[r] = pose_of(frames[int(table["frame_idx"][r])])
    result = pair_stats(table, prev, poses, proposals, logits, rows)
    result["which"] = which
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("build", "stats"))
    ap.add_argument("--logs", choices=("slot40", "val"), default="slot40", help="stats: which logs")
    ap.add_argument("--out", type=Path, help="build: default {data.navtrain.cache}/align")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from gallops.config import SiteConfig

    cfg = SiteConfig.load()
    if args.part == "build":
        result = build(cfg, (args.out or Path(cfg.resolve("{data.navtrain.cache}")) / "align").resolve())
    else:
        result = stats(cfg, args.logs)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
