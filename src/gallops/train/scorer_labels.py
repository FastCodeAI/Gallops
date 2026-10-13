"""Grade every plan in a scorer cache: the answers the scorer's logits are trained on.

For each cached episode (``scorer_cache``), every native plan and every vocabulary
path is graded with `fastgrade` against every vehicle in the episode (neuroNCAP
counts a collision with any of them, ego stopped or not), the lane centre the
labeller used, and the scene's drivable raster. The result sits beside the cache
file as ``<episode>.labels.npz``.

Before an episode's labels are written, a sample of its plans is graded again with
`grader.grade` -- the function every earlier label came from -- and any
disagreement stops the run. The vectorised grader exists to be fast, never to be
different.

    python -m gallops.train.scorer_labels --cache <dir> --drivable <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gallops.expert import fastgrade, grader
from gallops.expert.drivable import DrivableArea
from gallops.expert.lanes import load_references, reference_at

#: neuroNCAP excludes pedestrians; annotated walkers are about 0.5 m long.
VEHICLE_MIN_LENGTH_M = 1.2
NEAR_M = 70.0
CHECKS_PER_EPISODE = 12


def human_travel(logged_times: np.ndarray, logged_xy: np.ndarray, t0: int, horizon_s: float = 4.0) -> float:
    """Distance the human covered over the next `horizon_s`, from the logged path."""
    times = np.r_[
        t0, logged_times[(logged_times > t0) & (logged_times < t0 + horizon_s * 1e6)], t0 + horizon_s * 1e6
    ]
    xs = np.interp(times, logged_times, logged_xy[:, 0])
    ys = np.interp(times, logged_times, logged_xy[:, 1])
    return float(np.hypot(np.diff(xs), np.diff(ys)).sum())


def label_episode(cache_file: Path, drivable_dir: Path, rng: np.random.Generator) -> dict:
    data = np.load(cache_file)
    episode = Path(str(data["frame_dir"]))
    scene = "scene-" + episode.parent.name.rsplit("-", 1)[-1]
    area = DrivableArea.load(drivable_dir / f"{scene}.npz")
    poses = json.loads((episode / "ego_poses.json").read_text())
    actors = json.loads((episode / "actors.json").read_text())
    vehicles = [a for a in actors if a["dims"][1] >= VEHICLE_MIN_LENGTH_M and len(a["timestamps"]) >= 2]
    targets = [a for a in vehicles if a.get("is_target_actor")]
    logged, lanes = load_references(episode)
    raw = json.loads((episode / "reference_trajectory.json").read_text())
    log_t = np.array(sorted(int(k) for k in raw), dtype=float)
    log_xy = np.array([raw[str(int(k))][:2] for k in log_t], dtype=float)

    plans_all = np.concatenate([data["natives"], data["vocab"]], axis=1)  # (T, 32+N, 8, 3)
    keys = (
        "hit_at",
        "hit5_at",
        "hit_target_at",
        "off_road_at",
        "off_path_at",
        "max_deviation",
        "lane_run_s",
        "ttc_at",
        "travel",
    )
    out = {k: np.full(plans_all.shape[:2], np.nan, dtype=np.float32) for k in keys}
    out["human_travel"] = np.full(len(plans_all), np.nan, dtype=np.float32)
    out["reference_is_lane"] = np.zeros(len(plans_all), dtype=bool)
    check_ticks = set(rng.choice(len(plans_all), size=min(3, len(plans_all)), replace=False).tolist())
    checked = 0
    for i, ts in enumerate(data["timestamp"]):
        ts = int(ts)
        ego2world = np.asarray(poses[str(ts)])
        here = ego2world[:2, 3]
        near = [
            a
            for a in vehicles
            if (s := grader.actor_at(a, float(ts))) is not None and np.linalg.norm(s[0] - here) < NEAR_M
        ]
        reference, kind = reference_at(ts, logged, lanes)
        graded = fastgrade.grade_many(plans_all[i], ego2world, ts, near, reference, area, targets=targets)
        for k in keys:
            out[k][i] = graded[k]
        out["human_travel"][i] = human_travel(log_t, log_xy, ts)
        out["reference_is_lane"][i] = kind == "lane"
        if i in check_ticks:
            for j in rng.choice(plans_all.shape[1], size=CHECKS_PER_EPISODE // 3, replace=False):
                slow = grader.grade(plans_all[i, j], ego2world, ts, near, reference, area)
                for key, value in (
                    ("hit_at", slow.hit_at),
                    ("off_road_at", slow.off_road_at),
                    ("off_path_at", slow.off_path_at),
                ):
                    got = graded[key][j]
                    same = (value is None and np.isnan(got)) or (
                        value is not None and abs(value - got) < 1e-6
                    )
                    if not same:
                        raise SystemExit(
                            f"fastgrade disagrees with grader on {cache_file.name} tick {ts} plan {j} {key}: "
                            f"{value} vs {got}"
                        )
                if abs(slow.max_deviation - graded["max_deviation"][j]) > 1e-6:
                    raise SystemExit(f"max_deviation disagrees on {cache_file.name} tick {ts} plan {j}")
                checked += 1
    target = cache_file.with_name(cache_file.name.replace(".npz", ".labels.npz"))
    tmp = target.with_name(target.name + ".tmp.npz")
    # the key of the cache file graded, so labels are never read against another build of it
    stamp = {
        "timestamp": data["timestamp"],
        "n_vehicles": np.int32(len(vehicles)),
        "cache_key": data["cache_key"],
    }
    np.savez(tmp, **out, **stamp)
    tmp.rename(target)
    return {"ticks": len(plans_all), "checked": checked}


def labels_current(cache_file: Path) -> bool:
    """Whether `cache_file`'s labels exist and graded this build of it; refuses stale ones."""
    labels = cache_file.with_name(cache_file.name.replace(".npz", ".labels.npz"))
    if not labels.exists():
        return False
    with np.load(cache_file) as data, np.load(labels) as graded:
        built = str(data["cache_key"])
        if "cache_key" not in graded.files or str(graded["cache_key"]) != built:
            raise SystemExit(f"{labels} graded another build of {cache_file.name}; remove it to relabel")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--drivable", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only", nargs="*", default=None, help="source directory names to label (to run shards in parallel)"
    )
    args = parser.parse_args()
    from gallops.train.scorer_cache import require_key

    require_key(args.cache)
    rng = np.random.default_rng(args.seed)
    files = sorted(
        p for p in args.cache.glob("*/*.npz") if not p.name.endswith(".labels.npz") and ".tmp" not in p.name
    )
    if args.only:
        files = [f for f in files if f.parent.name in args.only]
    started, done, checked = time.time(), 0, 0
    for f in files:
        if labels_current(f):
            continue
        result = label_episode(f, args.drivable, rng)
        done += 1
        checked += result["checked"]
        print(
            f"  {f.parent.name}/{f.name}: {result['ticks']} ticks, "
            f"{checked} plans re-checked against grader so far, "
            f"{(time.time() - started) / done:.1f} s/episode",
            flush=True,
        )


if __name__ == "__main__":
    main()
