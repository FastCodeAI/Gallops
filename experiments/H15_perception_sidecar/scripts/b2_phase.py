"""H15 B-2: run one phase's training and dumps (PREDICTIONS.md, Part B · B-2), in parallel on one GPU.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/b2_phase.py --phase 1
        [--parallel 6] [--workers 40] [--gpu 1] [--only b2-1-C-s0,...] [--seeds 3,4 --arms D,HD]

Each run is `python -m gallops.drivejepa_perception.train_b2`, one process, its log in
{runs}/drivejepa_perception/part-b/logs/<run>.log; its directory {runs}/drivejepa_perception/part-b/b2/runs/<run>/.
- B-2-1 (joint): `train --recipe joint`.
- B-2-2, B-2-3 (two stages): `train --stage 1` into <run>-stage1, `dump --split train` of it into
  <run>-stage1/dump-train, then `train --stage 3 --stage1 <run>-stage1` into <run>.
Every finished run is then dumped on navtest, R-clean and select, each with its vendored labels. A step whose
output exists is skipped, so the driver resumes; a failed step stops that run's chain and is reported.

The phases' runs are b2_read.PHASES (phase 2's H3 is a seed-0 pilot); `--seeds/--arms` adds the noise-floor
rule's extra seeds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("GALLOPS_CONFIG", str(REPO / "gallops.toml"))

from gallops.config import SiteConfig  # noqa: E402
from gallops.drivejepa_perception import b2_read  # noqa: E402

RECIPE = {"1": "joint", "2": "lora2", "3": "full2"}
DUMPS = ("navtest", "fresh", "select")


def call(args: list, log: Path, gpu: str) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(REPO / "src"))
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        return subprocess.run(args, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def chain(run: str, version: str, arm: str, seed: int, root: Path, logs: Path, a) -> str:
    """One run's steps, in order; returns 'ok' or what failed."""
    py = sys.executable
    mod = "gallops.drivejepa_perception.train_b2"
    log = logs / f"{run}.log"
    out = root / run
    steps = []
    if version == "1":
        steps.append(
            (
                out / "weights.pt",
                [
                    py,
                    "-m",
                    mod,
                    "train",
                    "--recipe",
                    "joint",
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--workers",
                    str(a.workers),
                    "--out",
                    str(out),
                ],
            )
        )
    else:
        s1 = root / f"{run}-stage1"
        recipe = RECIPE[version]
        steps += [
            (
                s1 / "weights.pt",
                [
                    py,
                    "-m",
                    mod,
                    "train",
                    "--recipe",
                    recipe,
                    "--stage",
                    "1",
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--out",
                    str(s1),
                ],
            ),
            (
                s1 / "dump-train" / "labels.npz",
                [
                    py,
                    "-m",
                    mod,
                    "dump",
                    "--run",
                    str(s1),
                    "--split",
                    "train",
                    "--workers",
                    str(a.workers),
                    "--out",
                    str(s1 / "dump-train"),
                ],
            ),
            (
                out / "weights.pt",
                [
                    py,
                    "-m",
                    mod,
                    "train",
                    "--recipe",
                    recipe,
                    "--stage",
                    "3",
                    "--stage1",
                    str(s1),
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--out",
                    str(out),
                ],
            ),
        ]
    for split in DUMPS:
        steps.append(
            (
                out / f"dump-{split}" / "labels.npz",
                [
                    py,
                    "-m",
                    mod,
                    "dump",
                    "--run",
                    str(out),
                    "--split",
                    split,
                    "--workers",
                    str(a.workers),
                    "--out",
                    str(out / f"dump-{split}"),
                ],
            )
        )
    for done, args in steps:
        if done.exists():
            continue
        t0 = time.time()
        code = call(args, log, a.gpu)
        print(f"{run}: {args[3]} {' '.join(args[4:6])} -> {code} ({time.time() - t0:.0f} s)", flush=True)
        if code != 0:
            return f"failed: {' '.join(args[3:6])} (exit {code}); see {log}"
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=tuple(b2_read.PHASES))
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--gpu", default="1")
    p.add_argument("--only", default="", help="comma-separated run names")
    p.add_argument("--seeds", default="", help="extra seeds for --arms under B-2-1 (the noise-floor rule)")
    p.add_argument("--arms", default="")
    a = p.parse_args()
    cfg = SiteConfig.load()
    out_root = Path(cfg.resolve("{runs}/drivejepa_perception"))
    root, logs = out_root / "part-b/b2/runs", out_root / "part-b/logs"
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(b2_read.name(v, arm, s), v, arm, s) for v, arm, seeds in b2_read.PHASES[a.phase] for s in seeds]
    if a.seeds:
        jobs = [
            (b2_read.name("1", arm, int(s)), "1", arm, int(s))
            for arm in a.arms.split(",")
            for s in a.seeds.split(",")
        ]
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    print(f"phase {a.phase}: {[j[0] for j in jobs]}", flush=True)
    with ThreadPoolExecutor(a.parallel) as pool:
        results = dict(zip([j[0] for j in jobs], pool.map(lambda j: chain(*j, root, logs, a), jobs)))  # noqa: B905
    for run, result in results.items():
        print(f"{run}: {result}")
    if any(r != "ok" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
