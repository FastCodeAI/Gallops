"""True grades of the released model's 32 proposals at every graded navtrain moment.

For each graded moment of the moments table (`ticks`), the 32 proposals the release cache stored
(`release/proposals.npy`, ego frame, 8 poses at 0.5 s) go through the vendor's own training-time
grader with that token's train metric cache: `before_score_proposals` (the PDM simulator over 4 s at
0.1 s) and `PDMScorer.score_proposals` / `after_score_proposals`
(`score_module/compute_navsim_score.py`). That is `get_sub_score` without its anchors load: the
anchors index is the catalogue's precomputed grades, which the 32 proposals do not use, and it does
not exist for navtest (H15 PREDICTIONS.md, "Fixed before training").

Stored, in graded-row order (`row.npy` maps each to its row of the moments table):

    scores.npy   (G, 32, 9)      SCORE_NAMES: the eight sub-scores and the final grade
    states.npy   (G, 32, 41, 11) each proposal's simulated ego states (StateIndex layout), which
                                 two-frame comfort and momentum need, with x and y as offsets from
                                 `origin` (see below)
    origin.npy   (G, 2) float64  the moment's ego rear-axle x, y in the map frame
    done.npy     (G,)

The simulator works in map coordinates (UTM, about 4e6 m), where float32 resolves only about 0.25 m;
so x, y are stored relative to the moment's own ego position (float32 offsets of a few hundred metres
keep well under 1 mm) and `origin` adds them back exactly. Every other channel is stored as it comes.

Row 0 of the simulator's output is the PDM reference path the vendor scores the proposals against
(its progress normalises theirs); `after_score_proposals` already drops it from the scores, and
`proposal_states` drops it from the states, so both hold the 32 proposals only.

**Resumable per log**, like `cache.build`: the arrays are created whole, `progress.json` lists each
finished log. **Order:** the val logs, then the selection logs (H15's `selection_logs.txt`), then
the rest, so the report set is graded first. Workers are processes (`--workers`), one log per task;
pin them with taskset outside.

    python -m gallops.memory.grade build   [--workers 4] [--selection LIST]
    python -m gallops.memory.grade time    [--n 200] [--workers 4]       # rate, bytes; writes nothing big
    python -m gallops.memory.grade regrade --run <navsim run dir> [--n 200]   # navtest check vs its CSV
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict

#: after_score_proposals' columns, in order (compute_navsim_score.py)
SCORE_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "traffic_light_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "final",
)
#: the sub-scores NAVSIM's evaluator CSV also reports (its final score is weighted differently)
SHARED_WITH_CSV = SCORE_NAMES[:-1]
N_PROPOSALS, N_STATES, STATE_DIM = 32, 41, 11
ARRAYS = {"scores": (N_PROPOSALS, len(SCORE_NAMES)), "states": (N_PROPOSALS, N_STATES, STATE_DIM)}


# --- pure parts (tested without the vendor) ------------------------------------------------------


def metric_cache_file(root: Path, log: str, token: str) -> Path:
    """The vendor's layout: <root>/<log>/unknown/<token>/metric_cache.pkl."""
    return Path(root) / log / "unknown" / token / "metric_cache.pkl"


def proposal_states(simulated: np.ndarray, origin) -> np.ndarray:
    """Row 0 (the PDM reference path) dropped from `simulate_proposals`' (1 + P, 41, 11) output, and x, y
    made relative to `origin` (the ego rear axle, map frame) before the cast to float32."""
    simulated = np.asarray(simulated, np.float64)
    if simulated.ndim != 3 or simulated.shape[1:] != (N_STATES, STATE_DIM):
        raise ValueError(f"expected (1 + P, {N_STATES}, {STATE_DIM}) simulated states, got {simulated.shape}")
    states = simulated[1:].copy()
    states[..., 0] -= origin[0]
    states[..., 1] -= origin[1]
    return states.astype(np.float32)


def log_order(logs, split, selection) -> list:
    """Each log once: val logs, then selection logs, then the rest; sorted within each group."""
    logs, split = np.asarray(logs), np.asarray(split)
    names = sorted(set(logs.tolist()))
    val = set(logs[split == "val"].tolist())
    chosen = set(selection) - val
    first = [n for n in names if n in val]
    second = [n for n in names if n in chosen]
    rest = [n for n in names if n not in val and n not in chosen]
    return first + second + rest


def graded_rows(table: dict) -> np.ndarray:
    """Rows of the moments table that get a grade: graded tokens whose cache row was built."""
    return np.flatnonzero(np.asarray(table["graded"]) & np.asarray(table["inputs_ok"]))


def open_arrays(out: Path, n: int, mode: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, shape in ARRAYS.items():
        path = out / f"{name}.npy"
        if mode == "w+" and not path.exists():
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, *shape))
        else:
            arrays[name] = np.load(path, mmap_mode="r+" if mode != "r" else "r")
    origin = out / "origin.npy"
    if mode == "w+" and not origin.exists():
        arrays["origin"] = np.lib.format.open_memmap(origin, mode="w+", dtype=np.float64, shape=(n, 2))
    else:
        arrays["origin"] = np.load(origin, mmap_mode="r+" if mode != "r" else "r")
    done = out / "done.npy"
    if not done.exists():
        np.save(done, np.zeros(n, bool))
    arrays["done"] = np.load(done, mmap_mode="r+" if mode != "r" else "r")
    return arrays


# --- the vendor's grader ---------------------------------------------------------------------------

_VENDOR: dict = {}


def init_worker(cfg) -> None:
    """Enter Drive-JEPA's environment and import its grader once per process."""
    from gallops.memory import cache

    cache.enter(cfg)
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as cns

    _VENDOR["cns"] = cns


def grade_poses(metric_cache_path: Path, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, 9) scores, (P, 41, 11) simulated states (x, y relative to the origin) and the (2,) float64
    origin, for P ego-frame proposals (P, 8, 3).

    `get_sub_score` step for step (compute_navsim_score.py), without the anchors index, with
    `test=True` (the key-agent extras it skips are training targets, not grades).
    """
    cns = _VENDOR["cns"]
    metric_cache = cns.load_metric_cache(metric_cache_path)
    simulated = cns.before_score_proposals(metric_cache, poses, metric_cache.ego_state)
    final = cns.scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        None,  # log-replay agents, as get_sub_score
        metric_cache.past_human_trajectory,
    )
    scores = cns.after_score_proposals(cns.scorer, final, metric_cache.ego_state, True)[0]
    axle = metric_cache.ego_state.rear_axle
    origin = np.array([axle.x, axle.y], np.float64)
    return np.asarray(scores, np.float32), proposal_states(simulated, origin), origin


def _grade_rows(task):
    """Worker: grade every (row, token) of one log; returns (log, rows, scores, states, seconds, bytes)."""
    log, rows, tokens, metric_root, proposals_path = task
    proposals = np.load(proposals_path, mmap_mode="r")
    t0, nbytes = time.time(), 0
    scores = np.zeros((len(rows), *ARRAYS["scores"]), np.float32)
    states = np.zeros((len(rows), *ARRAYS["states"]), np.float32)
    origins = np.zeros((len(rows), 2), np.float64)
    for j, (row, token) in enumerate(zip_strict(rows, tokens)):
        path = metric_cache_file(metric_root, log, token)
        nbytes += path.stat().st_size
        scores[j], states[j], origins[j] = grade_poses(path, np.asarray(proposals[row], np.float64))
    return log, rows, scores, states, origins, time.time() - t0, nbytes


def _pool(cfg, workers: int):
    import multiprocessing as mp

    return mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(cfg,))


def _paths(cfg) -> tuple[Path, Path, Path]:
    cache_root = Path(cfg.resolve("{data.navtrain.cache}"))
    metric_root = Path(cfg.datasets["navtrain"].extra["metric_cache"])
    return cache_root, metric_root, cache_root / "release" / "proposals.npy"


# --- commands --------------------------------------------------------------------------------------


def build(cfg, out: Path, workers: int, selection: list, limit: int = 0) -> dict:
    """Grade every graded moment into `out`; resumable per log."""
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.memory import ticks

    cache_root, metric_root, proposals_path = _paths(cfg)
    table = ticks.load(cache_root / "ticks")
    rows = graded_rows(table)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "row.npy").exists():
        np.save(out / "row.npy", rows)
    if not np.array_equal(np.load(out / "row.npy"), rows):
        raise SystemExit(f"{out}/row.npy is not this moments table's graded rows: refusing to mix")
    manifest_path, progress_path = out / "manifest.json", out / "progress.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                {
                    "git_sha": git_sha(),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "rows": int(len(rows)),
                    "metric_cache": str(metric_root),
                    "proposals": str(proposals_path),
                    "grader": "compute_navsim_score: before_score_proposals + score_proposals + "
                    "after_score_proposals(test=True); no anchors load; "
                    "states row 0 (PDM reference) dropped; states x, y relative to origin.npy",
                    "score_names": SCORE_NAMES,
                    "workers": workers,
                },
                indent=1,
            )
            + "\n"
        )
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"logs_done": []}
    finished = set(progress["logs_done"])
    arrays = open_arrays(out, len(rows), "w+")
    position = {int(r): k for k, r in enumerate(rows)}
    logs = table["log"][rows]
    names = [n for n in log_order(logs, table["split"][rows], selection) if n not in finished]
    if limit:
        names = names[:limit]
    tasks = []
    for name in names:
        mine = rows[logs == name]
        tokens = [str(t) for t in table["token"][mine]]
        tasks.append((name, [int(r) for r in mine], tokens, metric_root, proposals_path))
    started, graded = time.time(), 0
    with _pool(cfg, workers) as pool:
        # imap keeps the order, so logs finish (and are recorded) val first
        for k, (log, done_rows, scores, states, origins, _s, _b) in enumerate(pool.imap(_grade_rows, tasks)):
            at = [position[r] for r in done_rows]
            arrays["scores"][at], arrays["states"][at], arrays["origin"][at] = scores, states, origins
            arrays["done"][at] = True
            for a in arrays.values():
                a.flush()
            finished.add(log)
            graded += len(at)
            progress.update(
                logs_done=sorted(finished),
                graded_this_run=graded,
                logs_left=len(tasks) - k - 1,
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                seconds_this_run=round(time.time() - started, 1),
            )
            progress_path.write_text(json.dumps(progress, indent=1) + "\n")
            if (k + 1) % 10 == 0:
                rate = graded / (time.time() - started)
                print(f"  {k + 1} of {len(tasks)} logs, {graded} moments, {rate:.2f}/s", flush=True)
    return {"graded": graded, "logs": len(tasks), "seconds": round(time.time() - started, 1)}


def timing(cfg, n: int, workers: int, seed: int = 0) -> dict:
    """Grade n random graded *train* moments end to end; nothing is stored but the result."""
    from gallops.memory import ticks

    cache_root, metric_root, proposals_path = _paths(cfg)
    table = ticks.load(cache_root / "ticks")
    rows = graded_rows(table)
    rows = rows[table["split"][rows] == "train"]
    pick = np.sort(np.random.default_rng(seed).choice(rows, size=n, replace=False))
    # one task per moment, so 4 workers share the load evenly however the logs fall
    tasks = [
        (str(table["log"][r]), [int(r)], [str(table["token"][r])], metric_root, proposals_path) for r in pick
    ]
    t0 = time.time()
    with _pool(cfg, workers) as pool:
        warm = time.time() - t0
        t1 = time.time()
        results = pool.map(_grade_rows, tasks, chunksize=1)
        wall = time.time() - t1
    per = np.array([r[5] for r in results])
    nbytes = int(sum(r[6] for r in results))
    finite = bool(all(np.isfinite(r[2]).all() and np.isfinite(r[3]).all() for r in results))
    scores = np.concatenate([r[2] for r in results])
    return {
        "moments": n,
        "logs": int(len(set(t[0] for t in tasks))),
        "workers": workers,
        "pool_start_s": round(warm, 1),
        "wall_s": round(wall, 1),
        "wall_s_per_moment": wall / n,
        "worker_s_per_moment_p10_50_90": [float(x) for x in np.percentile(per, [10, 50, 90])],
        "metric_cache_bytes_per_moment": nbytes / n,
        "finite": finite,
        "final_grade_mean": float(scores[..., -1].mean()),
        "rows": pick.tolist(),
    }


def regrade(cfg, run: Path, n: int, workers: int) -> dict:
    """Navtest check: the picks of a paddock NAVSIM run re-graded here vs that run's per-token CSV."""
    import csv

    navtest_cache = Path(cfg.resolve("{data.navsim.exp}")) / "Drive-JEPA-cache" / "metric_cache_v2"
    run = Path(run)
    lines = (run / "trajectories.jsonl").read_text().splitlines()
    picks = [json.loads(line) for line in lines if line.strip()]
    csv_path = sorted(run.glob("*.csv"))[0]
    with open(csv_path) as f:
        reported = {r["token"]: r for r in csv.DictReader(f)}
    picks = [p for p in picks if p["token"] in reported][:n]
    tasks = [(p["log_name"], p["token"], np.asarray(p["model_waypoints"], np.float64)) for p in picks]
    with _pool(cfg, workers) as pool:
        graded = pool.starmap(_regrade_one, [(t, navtest_cache) for t in tasks], chunksize=1)
    diffs = {name: [] for name in SHARED_WITH_CSV}
    per_token = []
    for (log, token, _), ours in zip_strict(tasks, graded):
        row = reported[token]
        d = {name: float(ours[k]) - float(row[name]) for k, name in enumerate(SHARED_WITH_CSV)}
        for name, v in d.items():
            diffs[name].append(v)
        per_token.append(
            {"token": token, "log": log, "ours": ours.tolist(), "csv_score": float(row["score"]), "diff": d}
        )
    summary = {}
    for name, v in diffs.items():
        a = np.abs(np.asarray(v))
        summary[name] = {
            "n": int(len(a)),
            "exact": int((a == 0).sum()),
            "within_1e-6": int((a <= 1e-6).sum()),
            "within_1e-3": int((a <= 1e-3).sum()),
            "max_abs": float(a.max()),
            "p50_p90_p99_abs": [float(x) for x in np.percentile(a, [50, 90, 99])],
        }
    return {
        "run": str(run),
        "csv": csv_path.name,
        "n": len(tasks),
        "summary": summary,
        "per_token": per_token,
    }


def _regrade_one(task, navtest_cache: Path) -> np.ndarray:
    log, token, pose = task
    scores, _, _ = grade_poses(metric_cache_file(navtest_cache, log, token), pose[None])
    return scores[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("build", "time", "regrade"))
    ap.add_argument("--out", type=Path, help="build: default {data.navtrain.cache}/grades")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selection", type=Path, help="build: the selection logs, one per line (graded second)")
    ap.add_argument("--limit", type=int, default=0, help="build: only the first N logs of the order")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--run", type=Path, help="regrade: a paddock NAVSIM run directory")
    ap.add_argument("--json", type=Path, help="time / regrade: where to write the result")
    args = ap.parse_args()

    from gallops.config import SiteConfig

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one thread per worker process; the cores are the workers
    cfg = SiteConfig.load()
    started = time.time()
    if args.part == "build":
        selection = args.selection.read_text().split() if args.selection else []
        out = (args.out or Path(cfg.resolve("{data.navtrain.cache}")) / "grades").resolve()
        result = build(cfg, out, args.workers, selection, args.limit)
    elif args.part == "time":
        result = timing(cfg, args.n, args.workers)
    else:
        if args.run is None:
            raise SystemExit("regrade needs --run <paddock NAVSIM run directory>")
        result = regrade(cfg, args.run, args.n, args.workers)
    result["total_s"] = round(time.time() - started, 1)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_token", "rows")}, indent=1))


if __name__ == "__main__":
    main()
