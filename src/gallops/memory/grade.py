"""True grades of the released model's 32 proposals at every graded navtrain moment.

For each graded moment of the moments table (`ticks`), the 32 proposals the release cache stored
(`release/proposals.npy`, ego frame, 8 poses at 0.5 s) go through the vendor's own training-time
grader with that token's train metric cache: `before_score_proposals` (the PDM simulator over 4 s at
0.1 s) and `PDMScorer.score_proposals` / `after_score_proposals`
(`score_module/compute_navsim_score.py`). That is `get_sub_score` without its anchors load: the
anchors index is the catalogue's precomputed grades, which the 32 proposals do not use, and it does
not exist for navtest (H19 PREDICTIONS.md, "Fixed before training").

**Two gradings, because the training grader and NAVSIM's benchmark differ in two known ways**
(measured on 200 navtest picks; MEMORY_H15_STEP2.md, part C):

- *strict* (`scores`) is the vendor's training grader as it is, the grade the released scorer was
  trained against and the one H19 trains on (Sid D22). Its lane keeping is a coarse check (every
  0.5 s, fail after 4 samples in a row off the centreline), and it never forgives a path for what the
  human driver also did.
- *forgiving* (`forgiving`) re-states lane keeping and traffic light the benchmark's way: lane
  keeping checked every 0.1 s (fail after 20 states in a row, `pdm_scorer.py` `_calculate_lane_keeping`),
  and either metric set to 1 where the human driver's own path fails it (`evaluate/pdm_score.py`, the
  `human_penalty_filter` branch, on in `pdm_scorer.yaml`). The human's path is graded with the same
  grader against the same reference path, and its row is kept (`human`), so any other metric can be
  forgiven later the same way.

Stored, in graded-row order (`row.npy` maps each to its row of the moments table):

    scores.npy     (G, 32, 9)      SCORE_NAMES, strict: the eight sub-scores and the final grade
    forgiving.npy  (G, 32, 2)      FORGIVING_NAMES: lane keeping and traffic light, the benchmark's way
    human.npy      (G, 10)         HUMAN_NAMES: the human path's strict scores and its fine lane keeping
    states.npy     (G, 32, 41, 11) each proposal's simulated ego states (StateIndex layout), which
                                   two-frame comfort and momentum need, with x and y as offsets from
                                   `origin` (see below)
    origin.npy     (G, 2) float64  the moment's ego rear-axle x, y in the map frame
    done.npy       (G,)

The simulator works in map coordinates (UTM, about 4e6 m), where float32 resolves only about 0.25 m;
so x, y are stored relative to the moment's own ego position (float32 offsets of a few hundred metres
keep well under 1 mm) and `origin` adds them back exactly. Every other channel is stored as it comes.

Row 0 of the simulator's output is the PDM reference path the vendor scores the proposals against
(its progress normalises theirs); `after_score_proposals` already drops it from the scores, and
`proposal_states` drops it from the states, so both hold the 32 proposals only.

**Resumable per log**, like `cache.build`: the arrays are created whole, `progress.json` lists each
finished log. **Order:** the val logs, then the selection logs (H19's `selection_logs.txt`), then
the rest, so the report set is graded first; `--batch` restricts a build to one of those groups.
Workers are processes (`--workers`), one log per task; pin them with taskset outside.

    python -m gallops.memory.grade build   [--batch val|selection|rest] [--workers 4] [--selection LIST]
    python -m gallops.memory.grade time    [--n 200] [--workers 4]       # rate, bytes; writes nothing big
    python -m gallops.memory.grade regrade --run <navsim run dir> [--n 200]   # navtest check vs its CSV
    python -m gallops.memory.grade repeat  --ref <npz>                   # strict grades equal a reference
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
FORGIVING_NAMES = ("lane_keeping", "traffic_light_compliance")
HUMAN_NAMES = (*SCORE_NAMES, "lane_keeping_fine")
LANE, LIGHT = SCORE_NAMES.index("lane_keeping"), SCORE_NAMES.index("traffic_light_compliance")
N_PROPOSALS, N_STATES, STATE_DIM = 32, 41, 11
ARRAYS = {
    "scores": (N_PROPOSALS, len(SCORE_NAMES)),
    "forgiving": (N_PROPOSALS, len(FORGIVING_NAMES)),
    "human": (len(HUMAN_NAMES),),
    "states": (N_PROPOSALS, N_STATES, STATE_DIM),
}
BATCHES = ("val", "selection", "rest")


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


def forgive(lane_fine, light, human_lane_fine: float, human_light: float) -> np.ndarray:
    """(P, 2) lane keeping and traffic light the benchmark's way: 1 where the human's path fails too."""
    lane = np.where(human_lane_fine == 0, 1.0, np.asarray(lane_fine, np.float64))
    light = np.where(human_light == 0, 1.0, np.asarray(light, np.float64))
    return np.stack([lane, light], -1).astype(np.float32)


def log_groups(logs, split, selection) -> dict:
    """The three batches, each sorted: val logs, selection logs (not val), the rest."""
    logs, split = np.asarray(logs), np.asarray(split)
    names = sorted(set(logs.tolist()))
    val = set(logs[split == "val"].tolist())
    chosen = set(selection) - val
    return {
        "val": [n for n in names if n in val],
        "selection": [n for n in names if n in chosen],
        "rest": [n for n in names if n not in val and n not in chosen],
    }


def log_order(logs, split, selection) -> list:
    """Each log once: val logs, then selection logs, then the rest; sorted within each group."""
    groups = log_groups(logs, split, selection)
    return groups["val"] + groups["selection"] + groups["rest"]


def graded_rows(table: dict) -> np.ndarray:
    """Rows of the moments table that get a grade: graded tokens whose cache row was built."""
    return np.flatnonzero(np.asarray(table["graded"]) & np.asarray(table["inputs_ok"]))


def open_arrays(out: Path, n: int, mode: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    shapes = {**{k: (np.float32, s) for k, s in ARRAYS.items()}, "origin": (np.float64, (2,))}
    arrays = {}
    for name, (dtype, shape) in shapes.items():
        path = out / f"{name}.npy"
        if mode == "w+" and not path.exists():
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n, *shape))
        else:
            arrays[name] = np.load(path, mmap_mode="r+" if mode != "r" else "r")
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
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    _VENDOR.update(
        cns=cns,
        as_array=get_trajectory_as_array,
        transform=transform_trajectory,
        center=BBCoordsIndex.CENTER,
        intersection=SemanticMapLayer.INTERSECTION,
        point=Point,
    )


def lane_keeping_fine(scorer) -> np.ndarray:
    """Lane keeping per row of the scorer's last call, the benchmark's way.

    The loop of NAVSIM's evaluator scorer (`pdm_scorer.py` `_calculate_lane_keeping`): every state at
    0.1 s, failing after `horizon / 0.1` = 20 states in a row more than 0.5 m off the centreline,
    intersections skipped. The training scorer runs the same loop on every 5th state and fails after
    4 (`train_pdm_scorer.py`, "coarse estimation"), which spans 1.5 s, not 2 s.
    """
    v = _VENDOR
    interval = scorer.proposal_sampling.interval_length
    need = int(np.ceil(scorer._config.lane_keeping_horizon_window / interval))
    limit = scorer._config.lane_keeping_deviation_limit
    line = scorer._centerline.linestring
    out = np.ones(scorer._num_proposals)
    for p in range(scorer._num_proposals):
        run = 0
        for t in range(scorer.proposal_sampling.num_poses + 1):
            point = v["point"](*scorer._ego_coords[p, t, v["center"]])
            if scorer._drivable_area_map.is_in_layer(point, layer=v["intersection"]):
                continue
            run = run + 1 if point.distance(line) > limit else 0
            if run >= need:
                out[p] = 0.0
                break
    return out


def _score(metric_cache, simulated):
    """Strict (rows, 9) scores without the reference row, and the fine lane keeping of the same rows."""
    cns = _VENDOR["cns"]
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
    fine = lane_keeping_fine(cns.scorer)[1:]  # read before after_score_proposals; the scorer is shared
    scores = cns.after_score_proposals(cns.scorer, final, metric_cache.ego_state, True)[0]
    return np.asarray(scores, np.float64), fine


def human_simulated(metric_cache) -> np.ndarray:
    """The PDM reference and the human driver's logged path, simulated: (2, 41, 11)."""
    v, cns = _VENDOR, _VENDOR["cns"]
    ego, sampling = metric_cache.ego_state, cns.simulator.proposal_sampling
    reference = v["as_array"](metric_cache.trajectory, sampling, ego.time_point)[:, :3]
    human = v["as_array"](v["transform"](metric_cache.human_trajectory, ego), sampling, ego.time_point)[:, :3]
    extended = np.zeros((2, *reference.shape[:-1], STATE_DIM))
    extended[0, :, :3], extended[1, :, :3] = reference, human
    return cns.simulator.simulate_proposals(extended, ego)


def grade_poses(metric_cache_path: Path, poses: np.ndarray) -> dict:
    """Every grade of P ego-frame proposals (P, 8, 3) at one moment.

    `scores` (P, 9) strict, `forgiving` (P, 2), `human` (10,), `states` (P, 41, 11) relative to
    `origin` (2,) float64. The strict part is `get_sub_score` step for step (compute_navsim_score.py),
    without the anchors index, with `test=True` (the key-agent extras it skips are training targets).
    """
    cns = _VENDOR["cns"]
    metric_cache = cns.load_metric_cache(metric_cache_path)
    simulated = cns.before_score_proposals(metric_cache, poses, metric_cache.ego_state)
    scores, fine = _score(metric_cache, simulated)
    human_scores, human_fine = _score(metric_cache, human_simulated(metric_cache))
    human = np.concatenate([human_scores[0], human_fine[:1]])
    axle = metric_cache.ego_state.rear_axle
    origin = np.array([axle.x, axle.y], np.float64)
    return {
        "scores": scores.astype(np.float32),
        "forgiving": forgive(fine, scores[:, LIGHT], human_fine[0], human_scores[0, LIGHT]),
        "human": human.astype(np.float32),
        "states": proposal_states(simulated, origin),
        "origin": origin,
    }


def _grade_rows(task) -> dict:
    """Worker: grade every (row, token) of one log, in the order given."""
    log, rows, tokens, metric_root, proposals_path = task
    proposals = np.load(proposals_path, mmap_mode="r")
    t0, nbytes = time.time(), 0
    out = {k: np.zeros((len(rows), *shape), np.float32) for k, shape in ARRAYS.items()}
    out["origin"] = np.zeros((len(rows), 2), np.float64)
    for j, (row, token) in enumerate(zip_strict(rows, tokens)):
        path = metric_cache_file(metric_root, log, token)
        nbytes += path.stat().st_size
        for k, v in grade_poses(path, np.asarray(proposals[row], np.float64)).items():
            out[k][j] = v
    out.update(log=log, rows=rows, seconds=time.time() - t0, bytes=nbytes)
    return out


def _pool(cfg, workers: int):
    import multiprocessing as mp

    return mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(cfg,))


def _paths(cfg) -> tuple[Path, Path, Path]:
    cache_root = Path(cfg.resolve("{data.navtrain.cache}"))
    metric_root = Path(cfg.datasets["navtrain"].extra["metric_cache"])
    return cache_root, metric_root, cache_root / "release" / "proposals.npy"


# --- commands --------------------------------------------------------------------------------------


def build(cfg, out: Path, workers: int, selection: list, limit: int = 0, batch: str = "") -> dict:
    """Grade every graded moment (or one batch of logs) into `out`; resumable per log."""
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
                    "states row 0 (PDM reference) dropped; states x, y relative to origin.npy; "
                    "forgiving = lane keeping at 0.1 s and traffic light, 1 where the human fails",
                    "score_names": SCORE_NAMES,
                    "forgiving_names": FORGIVING_NAMES,
                    "human_names": HUMAN_NAMES,
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
    if batch:
        order = log_groups(logs, table["split"][rows], selection)[batch]
    else:
        order = log_order(logs, table["split"][rows], selection)
    names = [n for n in order if n not in finished]
    if limit:
        names = names[:limit]
    tasks = []
    for name in names:
        mine = rows[logs == name]
        tokens = [str(t) for t in table["token"][mine]]
        tasks.append((name, [int(r) for r in mine], tokens, metric_root, proposals_path))
    started, graded, nbytes = time.time(), 0, 0
    print(f"batch {batch or 'all'}: {len(tasks)} logs to grade, {len(finished)} already done", flush=True)
    with _pool(cfg, workers) as pool:
        # imap keeps the order, so logs finish (and are recorded) in batch order
        for k, result in enumerate(pool.imap(_grade_rows, tasks)):
            at = [position[r] for r in result["rows"]]
            for name in (*ARRAYS, "origin"):
                arrays[name][at] = result[name]
            arrays["done"][at] = True
            for a in arrays.values():
                a.flush()
            finished.add(result["log"])
            graded += len(at)
            nbytes += result["bytes"]
            progress.update(
                logs_done=sorted(finished),
                batch=batch or "all",
                graded_this_run=graded,
                metric_cache_bytes_this_run=nbytes,
                logs_left=len(tasks) - k - 1,
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                seconds_this_run=round(time.time() - started, 1),
            )
            progress_path.write_text(json.dumps(progress, indent=1) + "\n")
            if (k + 1) % 10 == 0 or k + 1 == len(tasks):
                rate = graded / (time.time() - started)
                mbps = nbytes / 1e6 / (time.time() - started)
                print(
                    f"  {k + 1} of {len(tasks)} logs, {graded} moments, {rate:.2f}/s, NAS {mbps:.1f} MB/s",
                    flush=True,
                )
    return {
        "batch": batch or "all",
        "graded": graded,
        "logs": len(tasks),
        "seconds": round(time.time() - started, 1),
        "metric_cache_bytes": nbytes,
    }


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
        results = pool.map(_grade_rows, tasks, chunksize=1)
    wall = time.time() - t0
    per = np.array([r["seconds"] for r in results])
    finite = bool(all(np.isfinite(r[k]).all() for r in results for k in ARRAYS))
    return {
        "moments": n,
        "logs": int(len(set(t[0] for t in tasks))),
        "workers": workers,
        "wall_s": round(wall, 1),
        "wall_s_per_moment": wall / n,
        "worker_s_per_moment_p10_50_90": [float(x) for x in np.percentile(per, [10, 50, 90])],
        "metric_cache_bytes_per_moment": sum(r["bytes"] for r in results) / n,
        "finite": finite,
        "final_grade_mean": float(np.concatenate([r["scores"] for r in results])[..., -1].mean()),
        "rows": pick.tolist(),
    }


def repeat(cfg, ref: Path) -> dict:
    """Grade a reference file's rows again; strict scores, states and origin must equal it exactly."""
    from gallops.memory import ticks

    cache_root, metric_root, proposals_path = _paths(cfg)
    table = ticks.load(cache_root / "ticks")
    with np.load(ref) as z:
        reference = {k: z[k] for k in z.files}
    init_worker(cfg)
    proposals = np.load(proposals_path, mmap_mode="r")
    same = {"scores": [], "states": [], "origin": []}
    for j, r in enumerate(reference["rows"].tolist()):
        path = metric_cache_file(metric_root, str(table["log"][r]), str(table["token"][r]))
        got = grade_poses(path, np.asarray(proposals[r], np.float64))
        for k in same:
            same[k].append(bool(np.array_equal(got[k], reference[k][j])))
    return {
        "reference": str(ref),
        "reference_git_sha": str(reference.get("git_sha", "")),
        "rows": int(len(reference["rows"])),
        "identical": {k: int(sum(v)) for k, v in same.items()},
        "all_identical": bool(all(all(v) for v in same.values())),
    }


def regrade(cfg, run: Path, n: int, workers: int) -> dict:
    """Navtest check: the picks of a paddock NAVSIM run re-graded here vs that run's per-token CSV.

    Each pick is compared strict (the training grader) and, for lane keeping and traffic light,
    forgiving (the benchmark's way), which is what the CSV reports.
    """
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
    columns = {**{name: ("strict", k) for k, name in enumerate(SHARED_WITH_CSV)}}
    columns.update({f"{name} (forgiving)": ("forgiving", k) for k, name in enumerate(FORGIVING_NAMES)})
    diffs = {name: [] for name in columns}
    per_token = []
    for (log, token, _), ours in zip_strict(tasks, graded):
        row = reported[token]
        d = {}
        for name, (kind, k) in columns.items():
            value = float(ours["scores"][0, k]) if kind == "strict" else float(ours["forgiving"][0, k])
            d[name] = value - float(row[name.replace(" (forgiving)", "")])
            diffs[name].append(d[name])
        per_token.append(
            {
                "token": token,
                "log": log,
                "ours": ours["scores"][0].tolist(),
                "forgiving": ours["forgiving"][0].tolist(),
                "human": ours["human"].tolist(),
                "csv_score": float(row["score"]),
                "diff": d,
            }
        )
    summary = {}
    for name, v in diffs.items():
        a = np.abs(np.asarray(v))
        summary[name] = {
            "n": int(len(a)),
            "exact": int((a == 0).sum()),
            "within_1e-6": int((a <= 1e-6).sum()),
            "within_1e-4": int((a <= 1e-4).sum()),
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


def _regrade_one(task, navtest_cache: Path) -> dict:
    log, token, pose = task
    return grade_poses(metric_cache_file(navtest_cache, log, token), pose[None])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("build", "time", "regrade", "repeat"))
    ap.add_argument("--out", type=Path, help="build: default {data.navtrain.cache}/grades")
    ap.add_argument(
        "--batch", choices=BATCHES, help="build: only this group of logs (default: all, in order)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selection", type=Path, help="build: the selection logs, one per line (graded second)")
    ap.add_argument("--limit", type=int, default=0, help="build: only the first N logs of the order")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--run", type=Path, help="regrade: a paddock NAVSIM run directory")
    ap.add_argument("--ref", type=Path, help="repeat: an npz of rows, scores, states and origin")
    ap.add_argument("--json", type=Path, help="time / regrade / repeat: where to write the result")
    args = ap.parse_args()

    from gallops.config import SiteConfig

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one thread per worker process; the cores are the workers
    cfg = SiteConfig.load()
    started = time.time()
    if args.part == "build":
        if args.batch in ("selection", "rest") and not args.selection:
            raise SystemExit("--batch selection/rest needs --selection, or the groups would be wrong")
        selection = args.selection.read_text().split() if args.selection else []
        out = (args.out or Path(cfg.resolve("{data.navtrain.cache}")) / "grades").resolve()
        result = build(cfg, out, args.workers, selection, args.limit, args.batch or "")
    elif args.part == "time":
        result = timing(cfg, args.n, args.workers)
    elif args.part == "repeat":
        if args.ref is None:
            raise SystemExit("repeat needs --ref <npz>")
        result = repeat(cfg, args.ref)
    else:
        if args.run is None:
            raise SystemExit("regrade needs --run <paddock NAVSIM run directory>")
        result = regrade(cfg, args.run, args.n, args.workers)
    result["total_s"] = round(time.time() - started, 1)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_token", "rows")}, indent=1))
    if args.part == "repeat" and not result["all_identical"]:
        raise SystemExit("strict grades differ from the reference: STOP")


if __name__ == "__main__":
    main()
