"""Part B's official scorer: NAVSIM v2's own `run_pdm_score.py`, on navtest, for plans this repo stored.

    python -m gallops.drivejepa_perception.navtest verify-cache --cache {data.navsim_navtest_metric_cache} \
        --labels {out}/labels-navtest-v1 --out {out}/part-b/b0/navtest-cache-verify
    python -m gallops.drivejepa_perception.navtest build-cache --labels {out}/labels-fresh-v1 \
        --out {out}/metric-cache-fresh-v2

docs/drivejepa-perception-sidecar/part-b/B-0-foundations.md, B-0-2. Every Part B headline is read the
way the release's 87.78 was: the official v2 `run_pdm_score.py` on all 12,146 navtest scenes, no
momentum, with EPDMS under the human filter and navsim's CSV score by H04's arithmetic (`score`,
ported below). A modified model's plans reach it through a replay agent (`replay_agent.py`), which
returns the plan stored here for each token.

**The metric caches.**
- navtest's is Paddock's (D24, B-0 (c)): built on 10 Nov from another Drive-JEPA tree and
  interpreter. `verify-cache` holds it to this site before anything is scored on it: one entry per
  navtest token; its builder's scene filter equal to `navtest.yaml`; and 32 seeded tokens cached
  again here, with this tree, scored bit-equal against both caches by the vendored scorer (eight
  fixed plans) and by the official one (the human's plan). Its manifest pins the cache's metadata CSV
  by sha256, which `score` re-checks before every run, since Paddock owns the directory.
- R-clean's is built here (`build-cache`), by navsim's own `run_metric_caching.py` on a scene filter
  generated from `fresh.json` (logs only: the filter's own rule picks the routed windows), and
  refused unless it holds exactly the list's windows.
- The metadata CSV a cache carries holds absolute paths, and navsim's `MetricCacheLoader` reads
  them; a cache built in `.staging` gets them rewritten before the rename.

H04's arithmetic is ported from branch drivejepa-wajepa/run3a (e5ca907),
`src/gallops/drivejepa_wajepa/navtest_score.py`, with its tests; only `zip(strict=True)`, which
Python 3.9 lacks, became a length check.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: This file's checkout: the cwd moves into navsim_v2/ (env.enter).
REPO = Path(__file__).resolve().parents[3]
#: Part B's Hydra configs (agents; generated scene filters go beside each run's output).
NAVSIM_CONFIG = REPO / "recipes/drivejepa_perception/navsim_config"
SCRIPTS = "navsim_v2/navsim/planning/script"

# ------------------------------------------------------------------- H04's arithmetic (ported)

SUMMARY = {"average_all_frames", "pseudo_closed_loop", "average_expert_frames"}
GATES = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
]
SOFT = [
    ("ego_progress", 5),
    ("time_to_collision_within_bound", 5),
    ("lane_keeping", 2),
    ("history_comfort", 2),
    ("two_frame_extended_comfort", 2),
]
SHORT = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def num(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


def filtered(r: dict) -> float:
    """One scene's EPDMS from the CSV's columns, which navsim has already human-filtered: the four gates
    times the weighted mean of the soft terms, a blank term (EC, where the previous frame was not
    scored) left out of that scene's weights."""
    g = math.prod(num(r[c]) for c in GATES)
    sv = [(num(r[c]), w) for c, w in SOFT if not math.isnan(num(r[c]))]
    return g * sum(v * w for v, w in sv) / sum(w for _, w in sv)


def mean(xs: list) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs)


def resolve(pattern: str) -> str:
    """A CSV path, or a glob over navsim's timestamped run directories: the last match in sorted order."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no CSV matches {pattern}")
    return matches[-1]


def rows_of(path: str) -> list:
    """The valid scene rows of a run's CSV: the three summary rows and invalid rows dropped."""
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["token"] not in SUMMARY and r["valid"] == "True"]


def score(path: str) -> dict:
    """H04's two numbers: EPDMS under the human filter and the CSV score, with every sub-score mean."""
    rows = rows_of(path)
    cols = GATES + [c for c, _ in SOFT]
    if len(SHORT) != len(cols):
        raise SystemExit("SHORT and the columns disagree")
    return {
        "scenes": len(rows),
        "epdms_human_filter": 100 * mean([filtered(r) for r in rows]),
        "csv_score": 100 * mean([num(r["score"]) for r in rows]),
        "sub_scores": {s: mean([num(r[c]) for r in rows]) for s, c in zip(SHORT, cols)},  # noqa: B905 -- Python 3.9
    }


def per_token(path: str) -> dict:
    """token -> its EPDMS (human filter), its CSV score, and each sub-score, from a run's CSV."""
    cols = GATES + [c for c, _ in SOFT]
    out = {}
    for r in rows_of(path):
        entry = {"epdms_hf": filtered(r), "csv": num(r["score"])}
        entry.update({s: num(r[c]) for s, c in zip(SHORT, cols)})  # noqa: B905 -- Python 3.9
        out[r["token"]] = entry
    return out


# ------------------------------------------------------------------------------ the plan store


def write_plans(out: Path, tokens: list, poses: np.ndarray, meta: dict, **arrays) -> Path:
    """A plan store: `plans.npz` (token, poses (N, 8, 3) ego-frame x y yaw at 0.5 s, and any per-row
    arrays given, e.g. pick and row) plus a manifest. Refuses an existing store and a bad plan."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a plan store is never overwritten")
    poses = np.asarray(poses, dtype=np.float32)
    problems = coverage_problems({"token": np.asarray(tokens), "poses": poses}, set(tokens))
    if problems:
        raise SystemExit("; ".join(problems))
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(staging / "plans.npz", token=np.asarray(tokens), poses=poses, **arrays)
    manifest = {
        "rows": len(tokens),
        "tokens_sha256": hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest(),
        "plans_sha256": _sha256(staging / "plans.npz"),
        **meta,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    return out


def load_plans(path: Path) -> dict:
    """token -> its (8, 3) plan, from a plan store's directory or its plans.npz."""
    path = Path(path)
    z = np.load(path / "plans.npz" if path.is_dir() else path)
    return {str(t): np.asarray(p, dtype=np.float64) for t, p in zip(z["token"], z["poses"])}  # noqa: B905


def coverage_problems(plans: dict, tokens: set) -> list:
    """Why a plan store cannot be replayed on `tokens`: a token missing or extra, a repeated token, a
    plan of the wrong shape or not finite. `run_pdm_score.py` crashes on any failed token, so the
    replay agent must never meet one."""
    stored = [str(t) for t in plans["token"]]
    poses = np.asarray(plans["poses"])
    problems = []
    if len(set(stored)) != len(stored):
        problems.append(f"{len(stored) - len(set(stored))} tokens repeat")
    missing, extra = set(tokens) - set(stored), set(stored) - set(tokens)
    if missing or extra:
        problems.append(f"{len(missing)} tokens have no plan, {len(extra)} are extra")
    if poses.shape[1:] != (8, 3):
        problems.append(f"plans have shape {poses.shape[1:]}, not (8, 3)")
    elif not np.isfinite(poses).all():
        problems.append(f"{int((~np.isfinite(poses)).any(axis=(1, 2)).sum())} plans are not finite")
    return problems


# ---------------------------------------------------------------------------- the metric caches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_csv(root: Path) -> Path:
    """The metadata CSV navsim's MetricCacheLoader reads: the first .csv in `metadata/`. Refuses a
    cache with more than one, where which one is read would depend on the directory's order."""
    found = sorted(p for p in (Path(root) / "metadata").iterdir() if ".csv" in p.name)
    if len(found) != 1:
        raise SystemExit(f"{root}/metadata holds {len(found)} CSVs, not one")
    return found[0]


def cache_paths(root: Path) -> dict:
    """token -> the metric cache file the CSV names (as MetricCacheLoader reads it)."""
    lines = metadata_csv(root).read_text().splitlines()[1:]
    return {line.split("/")[-2]: line for line in lines if line}


def cache_identity(root: Path) -> dict:
    """What pins a cache: its CSV's name and sha256, and the counts it lists."""
    csv_path = metadata_csv(root)
    paths = cache_paths(root)
    return {
        "root": str(Path(root)),
        "csv": csv_path.name,
        "csv_sha256": _sha256(csv_path),
        "tokens": len(paths),
        "logs": len({Path(p).parents[2].name for p in paths.values()}),
    }


def coverage_of(root: Path, logs_of_tokens: dict) -> list:
    """Why a cache does not cover exactly `logs_of_tokens` (token -> log): a token missing or extra in
    its CSV, a CSV line outside the root or at the wrong log, or a file the CSV names that is absent."""
    root = Path(root)
    paths = cache_paths(root)
    problems = []
    if len(paths) != len(metadata_csv(root).read_text().splitlines()[1:]):
        problems.append("the CSV names a token twice")
    missing, extra = set(logs_of_tokens) - set(paths), set(paths) - set(logs_of_tokens)
    if missing or extra:
        problems.append(f"{len(missing)} tokens have no entry and {len(extra)} entries are not the split's")
    wrong = [
        t
        for t, p in paths.items()
        if t in logs_of_tokens and Path(p) != expected_path(root, logs_of_tokens[t], t)
    ]
    if wrong:
        problems.append(f"{len(wrong)} entries are not at <root>/<log>/unknown/<token>, e.g. {wrong[:2]}")
    absent = [t for t, p in paths.items() if not Path(p).exists()]
    if absent:
        problems.append(f"{len(absent)} files the CSV names do not exist, e.g. {absent[:2]}")
    return problems


def expected_path(root: Path, log: str, token: str) -> Path:
    return Path(root) / log / "unknown" / token / "metric_cache.pkl"


def rewrite_csv_root(csv_path: Path, old: str, new: str) -> int:
    """Move a cache's CSV from its staging root to its final one; returns the lines rewritten."""
    lines = Path(csv_path).read_text().splitlines()
    head, body = lines[0], lines[1:]
    if any(not line.startswith(old + "/") for line in body):
        raise SystemExit(f"{csv_path}: a line is not under {old}")
    Path(csv_path).write_text("\n".join([head] + [new + line[len(old) :] for line in body]) + "\n")
    return len(body)


def compose(config_dir: Path, name: str, overrides: list):
    """A navsim Hydra config, composed as its script would compose it (searchpath included)."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return hydra_compose(config_name=name, overrides=overrides)


def fixed_plans(ego_future: np.ndarray, speed: float) -> np.ndarray:
    """Eight plans that exercise every term against a cache: the human's; constant velocity; a stop;
    the human's shifted 0.5 m and 1.5 m to either side; and the human's at 0.8 of its length."""
    t = np.arange(1, 9) * 0.5
    human = np.asarray(ego_future, dtype=np.float64)
    cv = np.stack([speed * t, np.zeros(8), np.zeros(8)], axis=1)
    plans = [human, cv, np.zeros((8, 3))]
    for dy in (-1.5, -0.5, 0.5, 1.5):
        shifted = human.copy()
        shifted[:, 1] += dy
        plans.append(shifted)
    slower = human.copy()
    slower[:, :2] *= 0.8
    plans.append(slower)
    return np.stack(plans)


def verify_cache(cfg, cache: Path, labels_dir: Path, out: Path, rebuild: int) -> dict:
    """Hold Paddock's navtest metric cache to this site (B0-G3); write the manifest `score` re-checks."""
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, navsim_data, pdm_labels

    out, cache = Path(out), Path(cache)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    code = Path(entered["code"])
    filt = navsim_data.scene_filter(code, "navtest")
    navsim_data.load_navtest(REPO / navsim_data.LISTS["navtest"], filt)
    labels = np.load(Path(labels_dir) / "labels.npz")
    lmanifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    if navsim_data.source_of(lmanifest) != "navtest":
        raise SystemExit(f"{labels_dir} is not a navtest labels pass")
    logs = lmanifest["logs"]
    tokens = [str(t) for t in labels["token"]]
    log_of = {t: logs[i] for t, i in zip(tokens, labels["log_id"])}  # noqa: B905 -- Python 3.9
    problems = []
    if set(tokens) != set(filt["tokens"]):
        problems.append("the labels' tokens are not navtest.yaml's")
    problems += coverage_of(cache, log_of)

    # The builder's own record of its scene filter: the same windows as this site's navtest.yaml.
    import yaml

    built = yaml.safe_load((cache / "metadata/code/hydra/config.yaml").read_text())
    bf = built["train_test_split"]["scene_filter"]
    builder = {
        "data_split": built["train_test_split"]["data_split"],
        "window": [bf["num_history_frames"], bf["num_future_frames"], bf["frame_interval"], bf["has_route"]],
        "logs_equal": sorted(bf["log_names"]) == sorted(filt["log_names"]),
        "tokens_equal": sorted(bf["tokens"]) == sorted(filt["tokens"]),
        "sampling": [built["proposal_sampling"]["num_poses"], built["proposal_sampling"]["interval_length"]],
    }
    if builder != {
        "data_split": "test",
        "window": [navsim_data.HISTORY, navsim_data.FUTURE, 1, True],
        "logs_equal": True,
        "tokens_equal": True,
        "sampling": [40, 0.1],
    }:
        problems.append(f"the builder's config differs from this site's navtest: {builder}")

    # Rebuild a seeded sample here, with this tree, and score both caches.
    picked = sorted(np.random.default_rng(0).choice(sorted(tokens), rebuild, replace=False).tolist())
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    rebuilt = staging / "rebuilt"
    mc = compose(
        code / SCRIPTS / "config/metric_caching",
        "default_metric_caching",
        ["train_test_split=navtest", f"metric_cache_path={rebuilt}"],
    )
    from navsim.planning.metric_caching.caching import cache_scenarios

    by_log: dict = {}
    for t in picked:
        by_log.setdefault(log_of[t], []).append(t)
    for log, toks in sorted(by_log.items()):
        cache_scenarios([{"cfg": mc, "log_file": log, "tokens": toks}])

    scoring = compose(
        code / SCRIPTS / "config/pdm_scoring",
        "default_run_pdm_score",
        ["train_test_split=navtest", "experiment_name=h15_verify"],
    )
    official = _official_scorer(scoring)
    row_of = {t: i for i, t in enumerate(tokens)}
    speed = labels["ego_speed"]
    compared = []
    for t in picked:
        theirs, ours = expected_path(cache, log_of[t], t), expected_path(rebuilt, log_of[t], t)
        if not ours.exists():
            problems.append(f"{t}: this tree did not cache it")
            continue
        r = row_of[t]
        plans = fixed_plans(labels["ego_future"][r], float(speed[r][0]))
        a, b = pdm_labels.score(theirs, plans), pdm_labels.score(ours, plans)
        vendored_equal = all(np.array_equal(x, y) for x, y in zip(a, b))  # noqa: B905 -- Python 3.9
        o1, o2 = official(theirs, plans[0]), official(ours, plans[0])
        official_equal = o1 == o2
        compared.append({"token": t, "vendored_equal": vendored_equal, "official_equal": official_equal})
        if not vendored_equal:
            problems.append(f"{t}: the vendored scorer differs between the two caches")
        if not official_equal:
            problems.append(f"{t}: the official scorer differs between the two caches: {o1} / {o2}")

    manifest = {
        "pass": "navtest-cache-verify",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "cache": cache_identity(cache),
        "builder": builder,
        "builder_log_head": _builder_head(cache),
        "navtest_yaml_sha256": filt["sha256"],
        "rebuilt_tokens": picked,
        "rebuild_seed": 0,
        "fixed_plans": "human, constant velocity, stop, human shifted -1.5/-0.5/+0.5/+1.5 m, human x0.8",
        "compared": compared,
        "problems": problems,
        "verified": not problems,
    }
    shutil.rmtree(rebuilt, ignore_errors=True)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("entered", "compared", "rebuilt_tokens")},
            indent=1,
        )
    )
    if problems:
        raise SystemExit("the navtest cache is not verified:\n- " + "\n- ".join(problems))
    return manifest


def _builder_head(cache: Path) -> list:
    """The first lines of the builder's own log, which name the tree and interpreter it ran from."""
    path = Path(cache) / "metadata/log.txt"
    return path.read_text(errors="replace").splitlines()[:12] if path.exists() else []


def _official_scorer(scoring_cfg):
    """A function (cache file, plan) -> the official per-token score row, as `run_pdm_score.py` makes it."""
    import lzma
    import pickle

    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    simulator = instantiate(scoring_cfg.simulator)
    scorer = instantiate(scoring_cfg.scorer)
    policy = instantiate(scoring_cfg.traffic_agents_policy, simulator.proposal_sampling)

    def official(path: Path, plan: np.ndarray) -> list:
        with lzma.open(path, "rb") as f:
            metric_cache = pickle.load(f)
        trajectory = Trajectory(
            np.asarray(plan, dtype=np.float32), TrajectorySampling(time_horizon=4, interval_length=0.5)
        )
        row, _ = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=policy,
        )
        return [
            [float(v) for v in np.ravel(np.asarray(row[c].iloc[0], dtype=np.float64))]
            for c in row.columns
            if np.issubdtype(np.asarray(row[c].iloc[0]).dtype, np.number)
        ]

    return official


def fresh_scene_filter(record: dict) -> dict:
    """The Hydra scene filter that caches R-clean: its logs, navtrain's window, every routed window."""
    return {
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "num_history_frames": 4,
        "num_future_frames": 10,
        "frame_interval": 1,
        "has_route": True,
        "max_scenes": None,
        "log_names": list(record["logs"]),
        "tokens": None,
    }


def build_cache(cfg, labels_dir: Path, out: Path, workers: int) -> dict:
    """R-clean's metric cache, by navsim's own run_metric_caching.py (B0-G4)."""
    import yaml

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env, navsim_data

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    code = Path(entered["code"])
    spec = navsim_data.source_spec(code, Path(entered["navsim_root"]), "fresh")
    record = json.loads((REPO / navsim_data.LISTS["fresh"]).read_text())
    labels = np.load(Path(labels_dir) / "labels.npz")
    lmanifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    if navsim_data.source_of(lmanifest) != "fresh":
        raise SystemExit(f"{labels_dir} is not an R-clean labels pass")
    tokens = [str(t) for t in labels["token"]]
    log_of = {t: lmanifest["logs"][i] for t, i in zip(tokens, labels["log_id"])}  # noqa: B905
    if hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest() != spec["tokens_sha256"]:
        raise SystemExit("the labels' windows are not fresh.json's")

    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        raise SystemExit(f"{staging} exists: a failed build; look at it, then remove it")
    config = staging / "hydra_config/train_test_split"
    (config / "scene_filter").mkdir(parents=True)
    (config / "fresh.yaml").write_text(
        yaml.safe_dump({"defaults": [{"scene_filter": "fresh"}], "data_split": "trainval"})
    )
    (config / "scene_filter/fresh.yaml").write_text(
        yaml.safe_dump(fresh_scene_filter(record), sort_keys=False)
    )
    argv = [
        sys.executable,
        str(code / SCRIPTS / "run_metric_caching.py"),
        f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{staging / 'hydra_config'}]",
        "train_test_split=fresh",
        f"metric_cache_path={staging}",
        "worker=single_machine_thread_pool",
        f"worker.max_workers={workers}",
        "worker.use_process_pool=true",
    ]
    environ = {**os.environ, "PYTHONPATH": os.pathsep.join([str(code / "navsim_v2"), str(REPO / "src")])}
    t0 = time.time()
    done = subprocess.run(argv, cwd=code / "navsim_v2", env=environ)
    if done.returncode:
        raise SystemExit(f"run_metric_caching.py exited {done.returncode}; {staging} is left to look at")
    csv_path = metadata_csv(staging)
    rewritten = rewrite_csv_root(csv_path, str(staging), str(out))
    staged = {t: p.replace(str(out), str(staging), 1) for t, p in cache_paths(staging).items()}
    problems = []
    if set(staged) != set(tokens):
        problems.append(
            f"{len(set(tokens) - set(staged))} windows not cached, {len(set(staged) - set(tokens))} extra"
        )
    absent = [t for t, p in staged.items() if not Path(p).exists()]
    if absent:
        problems.append(f"{len(absent)} cached files are missing")
    wrong = [t for t in staged if t in log_of and Path(staged[t]) != expected_path(staging, log_of[t], t)]
    if wrong:
        problems.append(f"{len(wrong)} windows are cached under another log")
    if problems:
        raise SystemExit("the R-clean cache is not complete:\n- " + "\n- ".join(problems))
    manifest = {
        "pass": "metric-cache",
        "source": "fresh",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "list": navsim_data.LISTS["fresh"],
        "list_sha256": spec["provenance"]["list_sha256"],
        "labels": str(labels_dir),
        "tokens": len(staged),
        "logs": len(record["logs"]),
        "csv_lines_rewritten": rewritten,
        "command": argv[1:],
        "workers": workers,
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    manifest["cache"] = cache_identity(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "entered"}, indent=1))
    return manifest


# ------------------------------------------------------------------------------- scoring runs

#: H04's account of the release on navtest (docs/drivejepa_wajepa.md on run3a): B0-G8 holds the
#: release's replayed picks to it.
ANCHOR = {"epdms_human_filter": 87.78, "csv_score": 84.49}
ANCHOR_TOLERANCE = 0.02
#: B0-G9: the live release against its replayed picks.
LIVE_PICKS_EQUAL, LIVE_COLUMNS_EQUAL, LIVE_EPDMS = 0.999, 0.999, 0.02
#: A token's EC is scored against its successor: the next frame of its log, under 0.55 s later.
SUCCESSOR_US = 550_000


def release_plans(tokens_dir: Path, labels_dir: Path, out: Path) -> Path:
    """R's plan store: the release's own pick of its 32 proposals for every navtest row."""
    labels = np.load(Path(labels_dir) / "labels.npz")
    proposals = np.load(Path(tokens_dir) / "proposals.npy", mmap_mode="r")
    pick = np.load(Path(tokens_dir) / "pick.npy").astype(np.int64)
    rows = np.arange(len(pick))
    meta = {
        "arm": "R",
        "what": "the release's pick of its 32 proposals, from the token pass",
        "proposals": str(tokens_dir),
        "proposals_sha256": _sha256(Path(tokens_dir) / "proposals.npy"),
        "pick_sha256": _sha256(Path(tokens_dir) / "pick.npy"),
    }
    return write_plans(
        out, [str(t) for t in labels["token"]], np.asarray(proposals[rows, pick]), meta, pick=pick, row=rows
    )


def verified_cache(verified_dir: Path) -> Path:
    """The navtest metric cache `verify-cache` passed, refused if its CSV changed since (Paddock owns it)."""
    m = json.loads((Path(verified_dir) / "manifest.json").read_text())
    if not m.get("verified"):
        raise SystemExit(f"{verified_dir}: the navtest cache is not verified")
    root = Path(m["cache"]["root"])
    now = cache_identity(root)
    if (now["csv_sha256"], now["tokens"]) != (m["cache"]["csv_sha256"], m["cache"]["tokens"]):
        raise SystemExit(f"{root} changed since it was verified: verify it again before scoring on it")
    return root


def run_official(
    entered: dict, overrides: list, cache: Path, staging: Path, workers: int, gpu: bool, name: str
):
    """navsim's run_pdm_score.py on all of navtest, writing its CSV into `staging`; returns (argv, CSV)."""
    code = Path(entered["code"])
    argv = [
        sys.executable,
        str(code / SCRIPTS / "run_pdm_score.py"),
        f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{NAVSIM_CONFIG}]",
        "train_test_split=navtest",
        *overrides,
        "worker=single_machine_thread_pool",
        f"worker.max_workers={workers}",
        "worker.use_process_pool=true",
        f"metric_cache_path={cache}",
        f"experiment_name=h15_{name}",
        f"output_dir={staging}",
    ]
    environ = {**os.environ, "PYTHONPATH": os.pathsep.join([str(code / "navsim_v2"), str(REPO / "src")])}
    if not gpu:
        environ["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run(argv, cwd=code / "navsim_v2", env=environ)
    if done.returncode:
        raise SystemExit(f"run_pdm_score.py exited {done.returncode}; {staging} is left to look at")
    return argv[1:], resolve(str(Path(staging) / "*.csv"))


def read_dumps(dump_dir: Path) -> tuple:
    """token -> its dumped record, and the tokens dumped more than once."""
    out, repeated = {}, []
    for path in sorted(Path(dump_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["token"] in out:
                repeated.append(record["token"])
            out[record["token"]] = record
    return out, repeated


def _navtest_tokens(entered: dict) -> set:
    from gallops.drivejepa_perception import navsim_data

    filt = navsim_data.scene_filter(Path(entered["code"]), "navtest")
    navsim_data.load_navtest(REPO / navsim_data.LISTS["navtest"], filt)
    return set(filt["tokens"])


def score_plans(cfg, plans_dir: Path, verified_dir: Path, out: Path, workers: int, name: str) -> dict:
    """A plan store, scored on all of navtest by the official scorer through the replay agent.

    Refused unless the cache is the verified one and the store covers navtest exactly; after the run,
    every navtest token must be valid in the CSV and every plan the agent handed over must equal the store's.
    """
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; navtest is scored once per plan store")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    navtest = _navtest_tokens(entered)
    cache = verified_cache(verified_dir)
    z = np.load(Path(plans_dir) / "plans.npz")
    problems = coverage_problems(z, navtest)
    if problems:
        raise SystemExit(f"{plans_dir} cannot be replayed on navtest: " + "; ".join(problems))
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        raise SystemExit(f"{staging} exists: a failed run; look at it, then remove it")
    dumps = staging / "dumps"
    dumps.mkdir(parents=True)
    t0 = time.time()
    overrides = [
        "agent=gallops_replay",
        f"agent.plans={Path(plans_dir).resolve()}",
        f"agent.dump_dir={dumps}",
    ]
    argv, csv_path = run_official(entered, overrides, cache, staging, workers, gpu=False, name=name)
    valid = {r["token"] for r in rows_of(csv_path)}
    if valid != navtest:
        problems.append(f"{len(navtest - valid)} navtest tokens are not valid in the CSV")
    handed, repeated = read_dumps(dumps)
    stored = load_plans(plans_dir)
    differ = [
        t
        for t, r in handed.items()
        if not np.array_equal(np.asarray(r["poses"], np.float32), stored[t].astype(np.float32))
    ]
    if set(handed) != navtest or repeated or differ:
        unlike = f"{len(differ)} unlike the store's"
        problems.append(f"{len(handed)} plans handed over, {len(repeated)} twice, {unlike}")
    if problems:
        raise SystemExit("the replay is not the store's:\n- " + "\n- ".join(problems))
    manifest = {
        "pass": "navtest-score",
        "name": name,
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "plans": str(Path(plans_dir).resolve()),
        "plans_manifest": json.loads((Path(plans_dir) / "manifest.json").read_text()),
        "cache": cache_identity(cache),
        "verified": str(verified_dir),
        "command": argv,
        "csv": Path(csv_path).name,
        "replayed_equal_to_store": len(handed),
        "score": score(csv_path),
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(
        json.dumps(
            {k: manifest[k] for k in ("name", "score", "replayed_equal_to_store", "seconds")}, indent=1
        )
    )
    return manifest


def live_release(
    cfg, verified_dir: Path, tokens_dir: Path, labels_dir: Path, out: Path, workers: int
) -> dict:
    """The release itself through the official scorer, its plans, scores and picks dumped (B0-G9)."""
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; the live release is run once")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    navtest = _navtest_tokens(entered)
    cache = verified_cache(verified_dir)
    release = Path(entered["release"])
    recorded = json.loads((Path(tokens_dir) / "manifest.json").read_text())["release_sha256"]
    if _sha256(release) != recorded:
        raise SystemExit(f"{release} is not the release the token pass ran (sha256 {recorded[:12]})")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        raise SystemExit(f"{staging} exists: a failed run; look at it, then remove it")
    dumps = staging / "dumps"
    dumps.mkdir(parents=True)
    t0 = time.time()
    overrides = ["agent=gallops_live_release", f"agent.checkpoint_path={release}", f"agent.dump_dir={dumps}"]
    argv, csv_path = run_official(entered, overrides, cache, staging, workers, gpu=True, name="live_release")
    valid = {r["token"] for r in rows_of(csv_path)}
    handed, repeated = read_dumps(dumps)
    problems = []
    if valid != navtest:
        problems.append(f"{len(navtest - valid)} navtest tokens are not valid in the CSV")
    if set(handed) != navtest or repeated:
        problems.append(f"{len(navtest - set(handed))} tokens were not dumped, {len(repeated)} twice")
    if problems:
        raise SystemExit("the live run is not complete:\n- " + "\n- ".join(problems))
    labels = np.load(Path(labels_dir) / "labels.npz")
    order = [str(t) for t in labels["token"]]
    write_plans(
        staging / "plans",
        order,
        np.array([handed[t]["poses"] for t in order], np.float32),
        {"arm": "R-live", "what": "the release's own plans, as navsim ran it, dumped by LiveReleaseAgent"},
        pick=np.array([handed[t]["pick"] for t in order], np.int64),
        scores=np.array([handed[t]["scores"] for t in order], np.float32),
    )
    manifest = {
        "pass": "navtest-live",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "release_sha256": recorded,
        "cache": cache_identity(cache),
        "command": argv,
        "csv": Path(csv_path).name,
        "score": score(csv_path),
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: manifest[k] for k in ("score", "seconds")}, indent=1))
    return manifest


def successors(labels_dir: Path) -> dict:
    """token -> the next navtest token of its log under SUCCESSOR_US later, whose plan its EC reads."""
    labels = np.load(Path(labels_dir) / "labels.npz")
    tokens, logs, stamps = [str(t) for t in labels["token"]], labels["log_id"], labels["timestamp"]
    out = {}
    for log in np.unique(logs):
        index = np.flatnonzero(logs == log)
        index = index[np.argsort(stamps[index], kind="stable")]
        for a, b in zip(index[:-1], index[1:]):  # noqa: B905 -- Python 3.9
            if stamps[b] - stamps[a] < SUCCESSOR_US:
                out[tokens[a]] = tokens[b]
    return out


def compare_live(
    replay: dict, live: dict, replay_plans: dict, live_plans: dict, release_pick: dict, after: dict
) -> dict:
    """B0-G9's numbers: pick agreement, per-token equality where the plans (and their successors') agree,
    and the two EPDMS. `replay`, `live`: per_token(csv); plans: token -> (8, 3); `after`: successors."""
    tokens = sorted(replay)
    live_pick = {t: int(np.asarray(live_plans["pick"][i])) for i, t in enumerate(live_plans["token"])}
    picks_equal = float(np.mean([live_pick[t] == release_pick[t] for t in tokens]))

    def same_plan(t):
        return np.abs(replay_plans[t] - live_plans["poses_of"][t]).max() <= 1e-3

    comparable = [t for t in tokens if same_plan(t) and (t not in after or same_plan(after[t]))]
    columns = [k for k in replay[tokens[0]] if k != "epdms_hf"]
    equal = [t for t in comparable if all(_same(replay[t][k], live[t][k]) for k in columns)]
    mean_r = 100 * mean([replay[t]["epdms_hf"] for t in tokens])
    mean_l = 100 * mean([live[t]["epdms_hf"] for t in tokens])
    return {
        "tokens": len(tokens),
        "picks_equal": picks_equal,
        "comparable": len(comparable),
        "columns_equal": len(equal) / max(1, len(comparable)),
        "unequal_sample": [t for t in comparable if t not in set(equal)][:20],
        "epdms_replay": mean_r,
        "epdms_live": mean_l,
        "epdms_gap": abs(mean_r - mean_l),
    }


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def anchor(replay_dir: Path, live_dir: Path, tokens_dir: Path, labels_dir: Path, out: Path) -> dict:
    """B0-G8 and B0-G9: the replayed release against H04's account, and against the live release."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    replay_m = json.loads((Path(replay_dir) / "manifest.json").read_text())
    live_m = json.loads((Path(live_dir) / "manifest.json").read_text())
    got = replay_m["score"]
    a1 = {
        k: {"measured": got[k], "account": v, "within": abs(got[k] - v) <= ANCHOR_TOLERANCE}
        for k, v in ANCHOR.items()
    }
    replay = per_token(str(Path(replay_dir) / replay_m["csv"]))
    live = per_token(str(Path(live_dir) / live_m["csv"]))
    z = np.load(Path(live_dir) / "plans/plans.npz")
    live_plans = {"token": [str(t) for t in z["token"]], "pick": z["pick"]}
    live_plans["poses_of"] = {t: np.asarray(p, np.float64) for t, p in zip(live_plans["token"], z["poses"])}  # noqa: B905
    labels = np.load(Path(labels_dir) / "labels.npz")
    pick = np.load(Path(tokens_dir) / "pick.npy")
    release_pick = {str(t): int(p) for t, p in zip(labels["token"], pick)}  # noqa: B905 -- Python 3.9
    g9 = compare_live(
        replay, live, load_plans(replay_m["plans"]), live_plans, release_pick, successors(labels_dir)
    )
    gates = {
        "B0-G8": all(v["within"] for v in a1.values()),
        "B0-G9": g9["tokens"] == len(live)
        and g9["picks_equal"] >= LIVE_PICKS_EQUAL
        and g9["columns_equal"] >= LIVE_COLUMNS_EQUAL
        and g9["epdms_gap"] <= LIVE_EPDMS,
    }
    record = {"anchor": a1, "live": g9, "gates": gates, "tolerance": ANCHOR_TOLERANCE}
    out.mkdir(parents=True)
    (out / "anchor.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    if not all(gates.values()):
        raise SystemExit(f"the anchor does not hold: {gates}; nothing on navtest is compared until it does")
    return record


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.navtest")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify-cache", help="hold Paddock's navtest metric cache to this site (B0-G3)")
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rebuild", type=int, default=32)
    p = sub.add_parser("build-cache", help="R-clean's metric cache (B0-G4)")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("plans", help="R's plan store: the release's pick of its own 32, every navtest row")
    p.add_argument("--tokens", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score", help="a plan store through the official scorer, via the replay agent")
    p.add_argument("--plans", required=True)
    p.add_argument("--verified", required=True, help="verify-cache's output for the navtest cache")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("live", help="the release itself through the official scorer (B0-G9; one GPU)")
    p.add_argument("--verified", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("anchor", help="B0-G8 and B0-G9")
    p.add_argument("--replay", required=True)
    p.add_argument("--live", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score-csv", help="H04's two numbers from run_pdm_score CSVs")
    p.add_argument("runs", nargs="+", help="NAME=CSV (a glob: the last match)")
    p.add_argument("--json", default="")
    args = parser.parse_args(argv)
    if args.cmd == "score-csv":
        out = {}
        for arg in args.runs:
            name, pattern = arg.split("=", 1)
            path = resolve(pattern)
            out[name] = {"csv": path, **score(path)}
            print(name, json.dumps(out[name]))
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2))
        return
    if args.cmd == "plans":
        release_plans(Path(args.tokens), Path(args.labels), Path(args.out))
        return
    if args.cmd == "anchor":
        anchor(Path(args.replay), Path(args.live), Path(args.tokens), Path(args.labels), Path(args.out))
        return
    cfg = SiteConfig.load()
    if args.cmd == "verify-cache":
        verify_cache(cfg, Path(args.cache), Path(args.labels), Path(args.out), args.rebuild)
    elif args.cmd == "build-cache":
        build_cache(cfg, Path(args.labels), Path(args.out), args.workers)
    elif args.cmd == "score":
        score_plans(cfg, Path(args.plans), Path(args.verified), Path(args.out), args.workers, args.name)
    else:
        live_release(
            cfg, Path(args.verified), Path(args.tokens), Path(args.labels), Path(args.out), args.workers
        )


if __name__ == "__main__":
    main()
