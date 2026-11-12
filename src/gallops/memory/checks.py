"""Sample checks on the cache path, before the full run: does it compute what paddock computes?

    python -m gallops.memory.checks fidelity    --run DIR [--n 200]   # navtest picks vs paddock's plans
    python -m gallops.memory.checks determinism [--n 20]              # two fresh processes, same arrays
    python -m gallops.memory.checks sight       [--n 5]               # a blank image changes the encoder map
    python -m gallops.memory.checks contact     --png PATH [--n 20]   # training moments, by eye
    python -m gallops.memory.checks time        [--n 200]             # training moments end to end
    python -m gallops.memory.checks workers     [--n 20] [--workers 4] # 1 vs N decode workers, same arrays
    python -m gallops.memory.checks rate        [--n 200] [--workers 4] # the build's own loop, timed

Each writes a JSON verdict to {data.navtrain.cache}/checks/<name>.json and prints a summary.

**fidelity** runs the cache's own input path and forward (`inputs.agent_input`, `cache.run`) on
navtest tokens a paddock queue run drove (`trajectories.jsonl`, plans rounded to 1e-4 m), and asks:
is the cached pick paddock's plan, within 1e-3 m, on every token? It also checks, on the same
tokens, that the pick is exactly the top-scoring of the cached 32 (self-consistency): argmax of the
final score's sigmoid over the stored logits, whose proposal is the stored trajectory bit for bit.
NAVSIM sends four frames through `/infer_sequence`, which resets the node, so momentum never applies
there and the plain argmax is what paddock drove.

**determinism** runs the same tokens in two fresh processes and compares every stored array.
**workers** builds the same training moments with 1 decode worker and with N, through the build's
own loop (`cache.log_inputs`, read-ahead included), and requires every stored array to be equal.
**rate** times that loop on whole runs of consecutive moments from random training logs, as the
full build meets them (pickle loads timed apart: the full run loads each log once).
**sight** blanks the five images the model reads and requires the encoder map to change.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict

WAYPOINT_TOL_M = 1e-3
#: GPU memory a check waits for before loading the model (MiB).
GPU_FREE_MIB = 12 * 1024


def wait_for_gpu(need_mib: int = GPU_FREE_MIB, poll_s: int = 60) -> int:
    """Block until the GPU has `need_mib` free; returns the free MiB seen."""
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        free = int(out.split()[0])
        if free >= need_mib:
            return free
        print(f"  waiting for {need_mib} MiB free on the GPU (have {free})", flush=True)
        time.sleep(poll_s)


def _cfg():
    from gallops.config import SiteConfig

    return SiteConfig.load()


def _verdict_path(cfg, name: str) -> Path:
    out = Path(cfg.resolve("{data.navtrain.cache}")) / "checks"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{name}.json"


def _write(cfg, name: str, verdict: dict) -> None:
    verdict["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _verdict_path(cfg, name).write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "rows"}, indent=1))


# --- navtest -----------------------------------------------------------------------------------


def navtest_sample(run_dir: Path, n: int, seed: int = 0) -> list[dict]:
    """`n` tokens of a paddock NAVSIM run's plan dump, a fixed random sample."""
    rows = [
        json.loads(line) for line in (Path(run_dir) / "trajectories.jsonl").read_text().splitlines() if line
    ]
    random.Random(seed).shuffle(rows)
    return rows[:n]


class Navtest:
    """navtest log pickles by name, read once each; images from the navtest blob root."""

    def __init__(self, cfg):
        root = cfg.datasets["navsim"].resolve()
        self.logs = root / "navsim_logs" / "test"
        self.roots = [root / "sensor_blobs" / "test"]
        self._frames: dict = {}

    def moment(self, log: str, token: str) -> tuple[list, int]:
        if log not in self._frames:
            with open(self.logs / f"{log}.pkl", "rb") as f:
                self._frames[log] = pickle.load(f)
        frames = self._frames[log]
        for i, frame in enumerate(frames):
            if frame["token"] == token:
                return frames, i
        raise KeyError(f"token {token} not in {log}")


def _load(cfg, device: str):
    from gallops.memory import cache

    checkpoint = cache.enter(cfg)
    free = wait_for_gpu()
    agent, builder = cache.build_model(checkpoint, device)
    return agent, builder, checkpoint, free


def fidelity(args) -> int:
    from gallops.memory import cache, inputs

    cfg = _cfg()
    sample = navtest_sample(args.run, args.n)
    nav = Navtest(cfg)
    agent, builder, checkpoint, free = _load(cfg, args.device)
    rows = []
    for rec in sample:
        frames, i = nav.moment(rec["log_name"], rec["token"])
        result = cache.run(agent, builder, inputs.agent_input(frames, i, nav.roots), args.device)
        paddock = np.asarray(rec["model_waypoints"], dtype=np.float64)
        ours = result["trajectory"].astype(np.float64)
        stored_pick = cache.released_pick(result["logits"])
        rows.append(
            {
                "token": rec["token"],
                "max_abs_m": float(np.max(np.abs(ours[:, :2] - paddock[:, :2]))),
                "max_abs_heading": float(np.max(np.abs(ours[:, 2] - paddock[:, 2]))),
                "pick": result["pick"],
                "stored_pick": stored_pick,
                "pick_is_top": stored_pick == result["pick"],
                "pick_proposal_exact": bool(
                    np.array_equal(result["proposals"][stored_pick], result["trajectory"])
                ),
            }
        )
    gaps = np.asarray([r["max_abs_m"] for r in rows])
    same = [r["max_abs_m"] <= WAYPOINT_TOL_M for r in rows]
    consistent = [r["pick_is_top"] and r["pick_proposal_exact"] for r in rows]
    from gallops.drivejepa_b2d.provenance import git_sha
    from gallops.drivejepa_b2d.sidecar import sha256

    verdict = {
        "check": "fidelity + self-consistency",
        "run": str(args.run),
        "checkpoint": checkpoint,
        # what `cache.build` requires of a verdict before it builds (`cache.start_problems`)
        "checkpoint_sha256": sha256(Path(checkpoint)),
        "code_tree": cache.code_tree(),
        "git_sha": git_sha(),
        "gpu_free_mib_at_start": free,
        "tokens": len(rows),
        "same_pick_within_1e-3_m": int(sum(same)),
        "max_abs_m": float(gaps.max()),
        "median_abs_m": float(np.median(gaps)),
        "max_abs_heading_rad": float(max(r["max_abs_heading"] for r in rows)),
        "fidelity": "PASS" if all(same) else "FAIL",
        "self_consistent": int(sum(consistent)),
        "self_consistency": "PASS" if all(consistent) else "FAIL",
        "worst": sorted(rows, key=lambda r: -r["max_abs_m"])[:5],
        "rows": rows,
    }
    _write(cfg, "fidelity", verdict)
    return 0 if verdict["fidelity"] == "PASS" and verdict["self_consistency"] == "PASS" else 1


# --- determinism -------------------------------------------------------------------------------


def dump(args) -> int:
    """One process's arrays for `args.n` navtest tokens, to `args.npz` (determinism's worker)."""
    from gallops.memory import cache, inputs

    cfg = _cfg()
    sample = navtest_sample(args.run, args.n, seed=1)
    nav = Navtest(cfg)
    agent, builder, _, _ = _load(cfg, args.device)
    out = {name: [] for name in cache.ARRAYS}
    for rec in sample:
        frames, i = nav.moment(rec["log_name"], rec["token"])
        result = cache.run(agent, builder, inputs.agent_input(frames, i, nav.roots), args.device)
        for name in cache.ARRAYS:
            out[name].append(result[name])
    np.savez(
        args.npz, tokens=np.asarray([r["token"] for r in sample]), **{k: np.stack(v) for k, v in out.items()}
    )
    return 0


def determinism(args) -> int:
    from gallops.memory import cache

    cfg = _cfg()
    scratch = Path(cfg.resolve("{data.navtrain.cache}")) / "checks"
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [scratch / f"determinism_{k}.npz" for k in (1, 2)]
    for path in paths:
        command = [sys.executable, "-m", "gallops.memory.checks", "dump", "--run", str(args.run)]
        command += ["--n", str(args.n), "--npz", str(path), "--device", args.device]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3], env=os.environ.copy())
    a, b = np.load(paths[0]), np.load(paths[1])
    diffs = {name: float(np.max(np.abs(a[name] - b[name]))) for name in cache.ARRAYS}
    verdict = {
        "check": "determinism",
        "tokens": int(len(a["tokens"])),
        "same_tokens": bool(np.array_equal(a["tokens"], b["tokens"])),
        "max_abs_diff": diffs,
        "identical": all(np.array_equal(a[name], b[name]) for name in cache.ARRAYS),
    }
    verdict["determinism"] = "PASS" if verdict["identical"] and verdict["same_tokens"] else "FAIL"
    _write(cfg, "determinism", verdict)
    return 0 if verdict["determinism"] == "PASS" else 1


# --- sight -------------------------------------------------------------------------------------


def sight(args) -> int:
    from gallops.memory import cache, inputs

    cfg = _cfg()
    sample = navtest_sample(args.run, args.n, seed=2)
    nav = Navtest(cfg)
    agent, builder, _, _ = _load(cfg, args.device)
    rows = []
    for rec in sample:
        frames, i = nav.moment(rec["log_name"], rec["token"])
        seen = cache.run(agent, builder, inputs.agent_input(frames, i, nav.roots), args.device)
        blind = cache.run(agent, builder, inputs.agent_input(frames, i, nav.roots, blank=True), args.device)
        rows.append(
            {
                "token": rec["token"],
                "encoder_max_abs_diff": float(np.max(np.abs(seen["encoder"] - blind["encoder"]))),
                "encoder_rms": float(np.sqrt(np.mean(seen["encoder"] ** 2))),
                "proposals_max_abs_diff_m": float(np.max(np.abs(seen["proposals"] - blind["proposals"]))),
                "pick_changed": seen["pick"] != blind["pick"],
            }
        )
    changed = [r["encoder_max_abs_diff"] > 0 for r in rows]
    verdict = {"check": "sight", "tokens": len(rows), "encoder_changed": int(sum(changed)), "rows": rows}
    verdict["sight"] = "PASS" if all(changed) else "FAIL"
    _write(cfg, "sight", {**verdict, "summary": rows})
    return 0 if verdict["sight"] == "PASS" else 1


# --- training moments: contact sheet and timing -------------------------------------------------


def _training_sample(cfg, n: int, seed: int, graded_only: bool = False):
    from gallops.memory import ticks

    table = ticks.load(Path(cfg.resolve("{data.navtrain.cache}")) / "ticks")
    ok = table["inputs_ok"] & (table["graded"] if graded_only else True)
    index = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    return table, sorted(rng.choice(index, size=n, replace=False).tolist())


def _run_rows(agent, builder, cfg, table, rows, workers: int, device: str):
    """The build's loop (`cache.log_inputs` + `cache.run`) over `rows`, grouped by log in row order.

    Returns ({array: stacked in `rows` order}, pickle seconds, loop seconds, failures).
    """
    import torch

    from gallops.memory import cache, ticks

    data = cfg.datasets["navtrain"]
    roots = ticks.image_roots(data)
    by_log: dict = {}
    for i in rows:
        by_log.setdefault(str(table["log"][i]), []).append(int(i))
    frames_by_log, t0 = {}, time.time()
    for log in by_log:
        with open(data.extra["logs"] / f"{log}.pkl", "rb") as f:
            frames_by_log[log] = pickle.load(f)
    pickle_s = time.time() - t0
    pool, ahead = cache.decoders(workers)
    out, failed, t0 = {}, 0, time.time()
    try:
        for log, log_rows in by_log.items():
            idx = [int(table["frame_idx"][i]) for i in log_rows]
            for i, ai in zip_strict(log_rows, cache.log_inputs(frames_by_log[log], idx, roots, pool, ahead)):
                if isinstance(ai, FileNotFoundError):
                    failed += 1
                    continue
                out[i] = cache.run(agent, builder, ai, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        loop_s = time.time() - t0
    finally:
        for executor in (ahead, pool):
            if executor is not None:
                executor.shutdown()
    arrays = {name: np.stack([out[i][name] for i in rows if i in out]) for name in cache.ARRAYS}
    return arrays, pickle_s, loop_s, failed


def workers(args) -> int:
    """The same moments with 1 decode worker and with `args.workers`: every array equal."""
    import torch

    from gallops.memory import cache

    cfg = _cfg()
    table, rows = _training_sample(cfg, args.n, seed=5)
    agent, builder, _, _ = _load(cfg, args.device)
    torch.set_num_threads(2)  # as cache.build
    one, _, one_s, f1 = _run_rows(agent, builder, cfg, table, rows, 1, args.device)
    many, _, many_s, fn = _run_rows(agent, builder, cfg, table, rows, args.workers, args.device)
    diffs = {name: float(np.max(np.abs(one[name] - many[name]))) for name in cache.ARRAYS}
    equal = {name: bool(np.array_equal(one[name], many[name])) for name in cache.ARRAYS}
    verdict = {
        "check": "decode workers",
        "verdict": "PASS" if all(equal.values()) and f1 == fn == 0 else "FAIL",
        "moments": len(rows),
        "workers": [1, args.workers],
        "max_abs_diff": diffs,
        "bit_identical": equal,
        "loop_s": [round(one_s, 1), round(many_s, 1)],
        "failed": [f1, fn],
        "rows": rows,
    }
    _write(cfg, "workers", verdict)
    return 0 if verdict["verdict"] == "PASS" else 1


def _consecutive_sample(table, n: int, seed: int, split: str = "train") -> list[int]:
    """`n` rows: every usable moment of random logs of `split`, in row order, until there are n."""
    ok = table["inputs_ok"] & (table["split"] == split)
    logs = sorted(set(table["log"][ok].tolist()))
    rng = random.Random(seed)
    rng.shuffle(logs)
    rows: list[int] = []
    for log in logs:
        rows += np.flatnonzero(ok & (table["log"] == log)).tolist()
        if len(rows) >= n:
            break
    return sorted(rows[:n]) if len(rows) > n else rows


def rate(args) -> int:
    """The build's loop timed on consecutive moments, with 1 decode worker and with `args.workers`."""
    import torch

    cfg = _cfg()
    from gallops.memory import ticks

    table = ticks.load(Path(cfg.resolve("{data.navtrain.cache}")) / "ticks")
    rows = _consecutive_sample(table, args.n, seed=6)
    agent, builder, _, free = _load(cfg, args.device)
    torch.set_num_threads(2)  # as cache.build
    torch.cuda.reset_peak_memory_stats()
    _run_rows(agent, builder, cfg, table, rows[:5], args.workers, args.device)  # warm-up, untimed
    timed = {}
    for w in sorted({1, args.workers}):
        _, pickle_s, loop_s, failed = _run_rows(agent, builder, cfg, table, rows, w, args.device)
        timed[str(w)] = {
            "loop_s": round(loop_s, 1),
            "per_moment_s": round(loop_s / len(rows), 3),
            "pickle_s": round(pickle_s, 1),
            "failed": failed,
        }
    verdict = {
        "check": "rate",
        "moments": len(rows),
        "logs": len({str(table["log"][i]) for i in rows}),
        "by_workers": timed,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "gpu_free_mib_at_start": free,
    }
    _write(cfg, "rate", verdict)
    return 0


def contact(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gallops.memory import cache, inputs, ticks

    cfg = _cfg()
    data = cfg.datasets["navtrain"]
    roots = ticks.image_roots(data)
    table, rows = _training_sample(cfg, args.n, seed=3)
    cache.enter(cfg)  # navsim on the path; no model
    fig, axes = plt.subplots(args.n, 2, figsize=(9, 2.6 * args.n))
    for r, i in enumerate(rows):
        log, idx = str(table["log"][i]), int(table["frame_idx"][i])
        with open(data.extra["logs"] / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        ai = inputs.agent_input(frames, idx, roots)
        prev, now = ai.cameras[-2].cam_f0.image, ai.cameras[-1].cam_f0.image
        ego = ai.ego_statuses[-1]
        for c, (image, label) in enumerate(((prev, "front, 0.5 s earlier"), (now, "front, now"))):
            axes[r, c].imshow(image)
            axes[r, c].set_axis_off()
            axes[r, c].set_title(label, fontsize=7)
        p = ai.ego_statuses[0].ego_pose
        axes[r, 0].text(
            0,
            -0.35,
            f"{log} frame {idx}  split {table['split'][i]}  graded {bool(table['graded'][i])}  "
            f"chain pos {int(table['chain_pos'][i])}  roots prev/now {int(table['root_prev'][i])}/"
            f"{int(table['root_now'][i])}\n"
            f"v {float(ego.ego_velocity[0]):.2f} m/s  a {float(ego.ego_acceleration[0]):.2f} m/s2  "
            f"cmd {list(ego.driving_command)}  pose -1.5 s ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.2f})",
            transform=axes[r, 0].transAxes,
            fontsize=6,
            va="top",
        )
    fig.tight_layout()
    fig.savefig(args.png, dpi=80)
    verdict = {
        "check": "contact sheet",
        "png": str(args.png),
        "rows": [
            {
                "log": str(table["log"][i]),
                "frame_idx": int(table["frame_idx"][i]),
                "graded": bool(table["graded"][i]),
                "split": str(table["split"][i]),
            }
            for i in rows
        ],
    }
    _write(cfg, "contact", verdict)
    return 0


def timing(args) -> int:
    import torch

    from gallops.memory import cache, inputs, ticks

    cfg = _cfg()
    data = cfg.datasets["navtrain"]
    roots = ticks.image_roots(data)
    table, rows = _training_sample(cfg, args.n, seed=4)
    agent, builder, _, free = _load(cfg, args.device)
    torch.cuda.reset_peak_memory_stats()
    parts = {"pickle": 0.0, "inputs": 0.0, "forward": 0.0}
    started = time.time()
    frames_by_log: dict = {}
    for i in rows:
        log = str(table["log"][i])
        t0 = time.time()
        if log not in frames_by_log:
            with open(data.extra["logs"] / f"{log}.pkl", "rb") as f:
                frames_by_log[log] = pickle.load(f)
        t1 = time.time()
        ai = inputs.agent_input(frames_by_log[log], int(table["frame_idx"][i]), roots)
        t2 = time.time()
        cache.run(agent, builder, ai, args.device)
        torch.cuda.synchronize()
        t3 = time.time()
        parts["pickle"] += t1 - t0
        parts["inputs"] += t2 - t1
        parts["forward"] += t3 - t2
    wall = time.time() - started
    verdict = {
        "check": "timing",
        "moments": len(rows),
        "logs": len(frames_by_log),
        "wall_s": round(wall, 1),
        "per_moment_s": round(wall / len(rows), 3),
        "parts_per_moment_s": {k: round(v / len(rows), 3) for k, v in parts.items()},
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "gpu_free_mib_at_start": free,
        "torch_threads": torch.get_num_threads(),
    }
    _write(cfg, "timing", verdict)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "check",
        choices=("fidelity", "determinism", "dump", "sight", "contact", "time", "workers", "rate"),
    )
    ap.add_argument("--run", type=Path, help="a paddock NAVSIM run dir with trajectories.jsonl")
    ap.add_argument("--n", type=int)
    ap.add_argument("--npz", type=Path, help="dump's output")
    ap.add_argument("--png", type=Path, help="contact's output")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4, help="workers/rate: decode threads to compare with 1")
    args = ap.parse_args()
    defaults = {
        "fidelity": 200,
        "determinism": 20,
        "dump": 20,
        "sight": 5,
        "contact": 20,
        "time": 200,
        "workers": 20,
        "rate": 200,
    }
    args.n = args.n or defaults[args.check]
    # paths before `cache.enter` changes directory
    for name in ("run", "npz", "png"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    run = {
        "fidelity": fidelity,
        "determinism": determinism,
        "dump": dump,
        "sight": sight,
        "contact": contact,
        "time": timing,
        "workers": workers,
        "rate": rate,
    }[args.check]
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
