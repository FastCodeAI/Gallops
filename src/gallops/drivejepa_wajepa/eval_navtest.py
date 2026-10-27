"""Stages run2-navtest and run3a-navtest: navsim v2's run_pdm_score.py on navtest, inside a run's tree.

The Hydra overrides come from the recipe, as each evaluation's code/hydra/overrides.yaml recorded them. The
agent reads its tokens live (a frozen WA-JEPA encoder per worker), so WAJEPA_CODE and WAJEPA_WEIGHTS must be
set. No momentum-aware selection. navsim writes a CSV per run under
{NAVSIM_EXP_ROOT}/<experiment_name>/<timestamp>/; navtest-score turns it into the numbers
docs/drivejepa_wajepa.md reports.

    python -m gallops.drivejepa_wajepa.eval_navtest --tree <code-run dir> -- train_test_split=navtest
        agent=... ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, after --")
    a = ap.parse_args(argv)
    if not (a.tree / "OVERLAY.json").is_file():
        raise SystemExit(f"{a.tree} has no OVERLAY.json: build it with the code stage first")
    devkit = a.tree / "navsim_v2"
    cmd = [sys.executable, str(devkit / "navsim/planning/script/run_pdm_score.py"), *a.overrides]
    print(" ".join(cmd), flush=True)
    env = {**os.environ, "NAVSIM_DEVKIT_ROOT": str(devkit), "PYTHONPATH": str(devkit), "OMP_NUM_THREADS": "2"}
    return subprocess.run(cmd, cwd=devkit, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
