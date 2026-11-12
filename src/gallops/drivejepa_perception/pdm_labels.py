"""Phase B's labels: every candidate's NAVSIM v2 sub-scores, by the scorer the release trained against.

    python -m gallops.drivejepa_perception.pdm_labels build --tokens {out}/tokens-v1 \
        --labels {out}/labels-v1 --out {out}/pdm-v1

The scorer is Drive-JEPA's own (`score_module/compute_navsim_score.py`), imported from
`{code.drivejepa}`, never copied. It is called step by step -- `before_score_proposals`,
`scorer.score_proposals`, `after_score_proposals` -- because `get_sub_score` also opens the
release's anchor index, which these candidates are not in. It reads the pruned v2 train metric cache
(`{data.navsim_metric_cache}`), whose observations carry the replayed agents.

Per sample:
- the release's 32 candidates (tokens-v1/proposals.npy), scored together in one call, as the
  release's training scored its proposals: (32, 9);
- the human's plan (labels ego_future), scored alone, so it cannot move the candidates' progress
  normalisation: (9,);
- per candidate, the at-fault collision agent: `navsim_data.track_id` of its track token, and the
  10 Hz step of the collision, or -1. This is what B1.6's grounding test reads;
- per candidate (passes from 11 Nov), the agent behind a TTC failure and its 10 Hz step, the same
  way: Part B's headroom check (B-0-3) names the agent behind every NC or TTC failure.

Both agents are read past the scorer's base list, `observation.collided_track_ids`: the tracks
already touching the ego at t, which every proposal's list starts as a copy of. That base was empty
on every navtrain row (the old reading would have raised on `int(inf)` otherwise); off navtrain it
need not be.

    python -m gallops.drivejepa_perception.pdm_labels build --root {out}/metric-cache-fresh-v2 \
        --tokens {out}/tokens-fresh-v1 --labels {out}/labels-fresh-v1 --out {out}/pdm-fresh-v1

The columns are `after_score_proposals`'s: NC, DAC, TLC, DDC, EP, TTC, LK, HC, final.

**This scorer is not NAVSIM's official one.** Progress is normalised against PDM-Closed only, lane
keeping is checked every 0.5 s, drivable area uses a fast point test, and EC and the human filter are
absent. Its TTC loop is the official one, line for line (until 11 Nov this said TTC was checked every
0.5 s; it is not). Labels must equal what the trainer's scorer computes, so `check` re-scores rows and
requires exact equality. Agreement with the official scorer is a separate measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

COLUMNS = ("nc", "dac", "tlc", "ddc", "ep", "ttc", "lk", "hc", "final")
NO_AGENT = -1

_STATE: dict = {}


def metric_cache_path(root: Path, log: str, token: str) -> Path:
    return Path(root) / log / "unknown" / token / "metric_cache.pkl"


def score(path: Path, trajectories: np.ndarray):
    """Score (N, 8, 3) ego-frame plans in one call, with the PDM-Closed reference the scorer adds.

    Returns (N, 9) scores, the at-fault collision track id per plan, and its 10 Hz step.
    """
    detail = score_detail(path, trajectories)
    return detail["scores"], detail["nc_agent"], detail["nc_step"]


def score_detail(path: Path, trajectories: np.ndarray) -> dict:
    """`score`, with the agent behind each plan's TTC failure and its step as well.

    Keys: scores (N, 9) float32; nc_agent, ttc_agent (N,) int64 track ids or NO_AGENT; nc_step,
    ttc_step (N,) int16 10 Hz steps or NO_AGENT.
    """
    from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as cns

    from gallops.drivejepa_perception.navsim_data import track_id

    cache = cns.load_metric_cache(path)
    states = cns.before_score_proposals(cache, trajectories, cache.ego_state)
    final = cns.scorer.score_proposals(
        states,
        cache.observation,
        cache.centerline,
        cache.route_lane_ids,
        cache.drivable_area_map,
        cache.map_parameters,
        None,
        cache.past_human_trajectory,
    )
    # One row per plan: `after_score_proposals` drops the PDM-Closed reference itself (its last
    # line returns scores[1:]). The scorer's own arrays keep the reference at index 0, so plan k's
    # collision is at index k + 1 there. Measured: 32 plans in, (33, 41, 11) states, (32, 9) scores.
    scores, _, _, _ = cns.after_score_proposals(cns.scorer, final, cache.ego_state, True)
    if len(scores) != len(trajectories):
        raise SystemExit(
            f"{len(trajectories)} plans scored into {len(scores)} rows: the vendor's indexing moved"
        )
    base = len(cns.scorer._observation.collided_track_ids)
    out = {key: [] for key in ("nc_agent", "nc_step", "ttc_agent", "ttc_step")}
    for k in range(len(trajectories)):
        for kind, ids, steps in (
            ("nc", cns.scorer.proposal_fault_collided_track_ids[k + 1], cns.scorer._collision_time_idcs),
            ("ttc", cns.scorer.ttc_collided_track_ids[k + 1], cns.scorer._ttc_time_idcs),
        ):
            added = ids[base:]
            out[f"{kind}_agent"].append(track_id(added[0]) if added else NO_AGENT)
            out[f"{kind}_step"].append(int(steps[k + 1]) if added else NO_AGENT)
    return {
        "scores": np.asarray(scores, dtype=np.float32),
        "nc_agent": np.array(out["nc_agent"], dtype=np.int64),
        "nc_step": np.array(out["nc_step"], dtype=np.int16),
        "ttc_agent": np.array(out["ttc_agent"], dtype=np.int64),
        "ttc_step": np.array(out["ttc_step"], dtype=np.int16),
    }


def _score_row(row: int):
    s = _STATE
    path = metric_cache_path(s["root"], s["logs"][row], s["tokens"][row])
    if not path.exists():
        return row, None
    d = score_detail(path, np.asarray(s["proposals"][row], dtype=np.float64))
    human, _, _ = score(path, np.asarray(s["human"][row], dtype=np.float64)[None])
    return row, (d["scores"], d["nc_agent"], d["nc_step"], d["ttc_agent"], d["ttc_step"], human[0])


def build(cfg, tokens_dir: Path, labels_dir: Path, out: Path, workers: int, limit: int, root=None) -> None:
    """Every row's labels, from the metric cache at `root` (default: {data.navsim_metric_cache})."""
    from multiprocessing import get_context

    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_perception import env

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pass never overwrites one")
    started, sha = time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha()
    entered = env.enter(cfg)
    root = Path(root) if root else Path(cfg.resolve("{data.navsim_metric_cache}"))
    labels = np.load(Path(labels_dir) / "labels.npz")
    lmanifest = json.loads((Path(labels_dir) / "manifest.json").read_text())
    logs = lmanifest["logs"]
    n = len(labels["token"]) if not limit else min(limit, len(labels["token"]))
    _STATE.update(
        root=root,
        logs=[logs[i] for i in labels["log_id"][:n]],
        tokens=[str(t) for t in labels["token"][:n]],
        proposals=np.load(Path(tokens_dir) / "proposals.npy", mmap_mode="r"),
        human=labels["ego_future"],
    )
    candidates = np.full((n, 32, len(COLUMNS)), np.nan, np.float32)
    human = np.full((n, len(COLUMNS)), np.nan, np.float32)
    agents = np.full((n, 32), NO_AGENT, np.int64)
    steps = np.full((n, 32), NO_AGENT, np.int16)
    ttc_agents = np.full((n, 32), NO_AGENT, np.int64)
    ttc_steps = np.full((n, 32), NO_AGENT, np.int16)
    missing = []
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        for k, (row, result) in enumerate(pool.imap_unordered(_score_row, range(n), chunksize=16)):
            if result is None:
                missing.append(row)
            else:
                candidates[row], agents[row], steps[row], ttc_agents[row], ttc_steps[row], human[row] = result
            if (k + 1) % 5000 == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"{k + 1}/{n} rows, {rate:.1f}/s, eta {(n - k - 1) / rate / 60:.0f} min", flush=True)
    staging = out.with_name(out.name + ".staging")
    staging.mkdir(parents=True)
    np.savez(
        staging / "pdm.npz",
        candidates=candidates,
        human=human,
        collision_agent=agents,
        collision_step=steps,
        ttc_agent=ttc_agents,
        ttc_step=ttc_steps,
    )
    manifest = {
        "pass": "pdm_labels",
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gallops_sha": sha,
        "entered": entered,
        "metric_cache": str(root),
        "source": lmanifest.get("source", "navtrain"),
        "tokens": str(tokens_dir),
        "labels": str(labels_dir),
        "rows": n,
        "missing_metric_cache": len(missing),
        "missing_rows_sample": missing[:20],
        "columns": list(COLUMNS),
        "seconds": round(time.time() - t0),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    staging.rename(out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entered"}, indent=1))


def built_root(cfg, pdm_dir: Path) -> Path:
    """The metric cache a labels pass read, as its manifest names it.

    pdm-v1's manifest names where navtrain's cache was before it moved out of navsim/exp/ on 4 Nov;
    for a navtrain pass whose recorded cache is gone, the site's `{data.navsim_metric_cache}` is it.
    """
    built = json.loads((Path(pdm_dir) / "manifest.json").read_text())
    root = Path(built["metric_cache"])
    if not root.exists() and built.get("source", "navtrain") == "navtrain":
        root = Path(cfg.resolve("{data.navsim_metric_cache}"))
    return root


def check(cfg, pdm_dir: Path, tokens_dir: Path, labels_dir: Path, rows: int) -> list:
    """Re-score `rows` random rows and require exact equality with the stored labels, from the metric
    cache the build read (its manifest names it)."""
    from gallops.drivejepa_perception import env

    env.enter(cfg)
    root = built_root(cfg, pdm_dir)
    stored = np.load(Path(pdm_dir) / "pdm.npz")
    labels = np.load(Path(labels_dir) / "labels.npz")
    logs = json.loads((Path(labels_dir) / "manifest.json").read_text())["logs"]
    proposals = np.load(Path(tokens_dir) / "proposals.npy", mmap_mode="r")
    scored = np.flatnonzero(np.isfinite(stored["candidates"][:, 0, -1]))
    picked = np.random.default_rng(0).choice(scored, min(rows, len(scored)), replace=False)
    problems = []
    for row in picked:
        path = metric_cache_path(root, logs[int(labels["log_id"][row])], str(labels["token"][row]))
        d = score_detail(path, np.asarray(proposals[row], dtype=np.float64))
        same = np.array_equal(d["scores"], stored["candidates"][row]) and np.array_equal(
            d["nc_agent"], stored["collision_agent"][row]
        )
        if "ttc_agent" in stored:
            same = same and np.array_equal(d["ttc_agent"], stored["ttc_agent"][row])
        if not same:
            problems.append(f"row {row}: re-scored candidates differ from the stored labels")
    print(json.dumps({"rows": len(picked), "problems": problems}))
    return problems


def main(argv=None) -> None:
    from gallops.config import SiteConfig

    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.pdm_labels")
    parser.add_argument("cmd", choices=("build", "check"))
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--root", default="", help="build: the metric cache (default: navtrain's)")
    args = parser.parse_args(argv)
    cfg = SiteConfig.load()
    if args.cmd == "build":
        build(cfg, Path(args.tokens), Path(args.labels), Path(args.out), args.workers, args.limit, args.root)
    else:
        raise SystemExit(
            1 if check(cfg, Path(args.out), Path(args.tokens), Path(args.labels), args.rows) else 0
        )


if __name__ == "__main__":
    main()
