"""Stages run2-train and run3a-train: navsim's run_training.py inside a run's tree, as the run was launched.

The Hydra overrides come from the recipe, one per argument after `--`, exactly as the run's own
code/hydra/overrides.yaml recorded them (only the cache path is a template). Checkpoints go to --ckpt-dir,
every epoch plus last.ckpt (the agent's per-epoch ModelCheckpoint reads DRIVEJEPA_CKPT_DIR); run_training.py
also keeps its own best_epoch=N.ckpt by val/score_epoch under navsim's experiment directory. An existing
--ckpt-dir is refused, so a checkpoint is never overwritten.

Environment set here, as both runs had it: WANDB_MODE=offline, DRIVEJEPA_SCORE_WORKERS (the label workers;
12), DRIVEJEPA_CKPT_DIR, OMP_NUM_THREADS=2, and NAVSIM_DEVKIT_ROOT / PYTHONPATH = the tree's navsim_v2. The
recipe sets the rest (NAVSIM_EXP_ROOT, OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, and Run
3A's label inputs).

    python -m gallops.drivejepa_wajepa.train --tree <code-run dir> --ckpt-dir <dir> [--config-name
        run3a_training]
        [--score-workers 12] -- agent=... experiment_name=... cache_path=... ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def command(tree: Path, config_name: str | None, overrides: list[str]) -> list[str]:
    script = tree / "navsim_v2/navsim/planning/script/run_training.py"
    return [sys.executable, str(script), *(["--config-name", config_name] if config_name else []), *overrides]


def environment(tree: Path, ckpt_dir: Path, score_workers: int) -> dict[str, str]:
    devkit = str(tree / "navsim_v2")
    return {
        **os.environ,
        "NAVSIM_DEVKIT_ROOT": devkit,
        "PYTHONPATH": devkit,
        "WANDB_MODE": "offline",
        "DRIVEJEPA_SCORE_WORKERS": str(score_workers),
        "DRIVEJEPA_CKPT_DIR": str(ckpt_dir),
        "OMP_NUM_THREADS": "2",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True, help="the run's tree, from code-run2 / code-run3a")
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument(
        "--config-name", default=None, help="run_training's --config-name (Run 3A: run3a_training)"
    )
    ap.add_argument("--score-workers", type=int, default=12)
    ap.add_argument(
        "--require",
        type=Path,
        action="append",
        default=[],
        help="an input that must exist before training starts; repeatable",
    )
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, after --")
    a = ap.parse_args(argv)
    if not (a.tree / "OVERLAY.json").is_file():
        raise SystemExit(f"{a.tree} has no OVERLAY.json: build it with the code stage first")
    if a.ckpt_dir.exists():
        raise SystemExit(f"refusing: {a.ckpt_dir} exists; checkpoints are never overwritten")
    missing = [str(p) for p in a.require if not p.exists()]
    if missing:
        raise SystemExit(f"refusing: required inputs missing: {missing}")
    cmd = command(a.tree, a.config_name, a.overrides)
    print(" ".join(cmd), flush=True)
    return subprocess.run(
        cmd, cwd=a.tree / "navsim_v2", env=environment(a.tree, a.ckpt_dir, a.score_workers)
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
