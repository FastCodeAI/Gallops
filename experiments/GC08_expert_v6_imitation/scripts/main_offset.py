"""neuro-ncap's main.py with the seeds moved: runs FIRST .. FIRST+N-1, not 0 .. N-1.

The engine seeds run i's actor jitter with i (`engine.run(i)` ->
`np.random.default_rng(seed)` in `_setup_scenario`), and the released model is
deterministic, so re-collecting seeds 0-9 under it reproduces round 0 tick for
tick. New data needs new seeds. The pinned neuro-ncap checkout is not edited;
this wraps its own Config and only replaces the loop.

Run with cwd = the neuro-ncap repo, exactly like main.py:

    python main_offset.py --first-seed 10 <every main.py argument>
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

import tyro  # noqa: E402

from main import Config  # noqa: E402


def run(config: Config, first_seed: int) -> None:
    # main() also infers a scenario path and wires spoofing; collection always
    # passes --engine.scenario.path and never spoofs, so neither applies.
    if config.engine.scenario.path is None or config.spoof_renderer or config.spoof_model:
        raise SystemExit("main_offset.py: pass --engine.scenario.path; spoofing is not supported")
    engine = config.engine.setup()
    og_log_dir = config.engine.logger.log_dir
    for seed in range(first_seed, first_seed + config.runs):
        config.engine.logger.log_dir = og_log_dir / f"run_{seed}"
        engine.logger = config.engine.logger.setup()
        asyncio.run(engine.run(seed))


if __name__ == "__main__":
    first = 0
    argv = sys.argv[1:]
    if "--first-seed" in argv:
        at = argv.index("--first-seed")
        first = int(argv[at + 1])
        del argv[at : at + 2]
    run(tyro.cli(Config, args=argv), first)
