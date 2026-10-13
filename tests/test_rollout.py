"""The offline closed loop's bookkeeping: which frame a run starts on, which plans it replays.

The engine itself (`Engine.step`, the LQR vehicle model, the evaluator) needs torch and nuplan,
which `make check` does not install; its faithfulness is measured by `rollout replay` against
recorded episodes, not here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from gallops.expert.rollout import Episode, _finish, match_rows, recorded_plans, start_frame_for


def test_a_run_without_jitter_starts_on_the_scenario_frame():
    general = {"start-frame": 6, "start-jitter": 0}
    assert {start_frame_for(general, seed) for seed in range(10)} == {6}


def test_jittered_runs_start_where_the_engine_starts_them():
    general = {"start-frame": 6, "start-jitter": 1}
    starts = [start_frame_for(general, seed) for seed in range(10)]
    # Engine._setup_scenario: one draw from a fresh default_rng(seed), clamped at frame 0
    assert starts == [6 + int(np.random.default_rng(seed).integers(-1, 2)) for seed in range(10)]
    assert set(starts) <= {5, 6, 7} and len(set(starts)) > 1
    assert start_frame_for({"start-frame": 0, "start-jitter": 3}, 4) >= 0


def test_an_episode_is_named_by_its_directories():
    episode = Episode.from_dir(Path("collect/frontal-0053/run_7"))
    assert (episode.shape, episode.scene, episode.seed) == ("frontal", "0053", 7)


def _plan(offset: float) -> list:
    return (np.arange(12, dtype=np.float64).reshape(6, 2) + offset).tolist()


def test_the_dump_with_the_same_plans_is_the_match():
    plans = {10: _plan(0.0), 20: _plan(1.0)}
    other_seed = {10: np.asarray(_plan(0.5)), 20: np.asarray(_plan(1.5)), 5: np.asarray(_plan(9.0))}
    this_run = {10: np.asarray(_plan(0.0)), 20: np.asarray(_plan(1.00004)), 5: np.asarray(_plan(9.0))}
    assert match_rows(plans, [other_seed, this_run]) is this_run
    assert match_rows(plans, [other_seed]) is None
    assert match_rows({10: _plan(0.0), 99: _plan(0.0)}, [this_run]) is None  # a stamp it never saw


def test_the_engines_own_copy_wins_and_the_dump_fills_the_priming_plan(tmp_path):
    (tmp_path / "trajectories.json").write_text(json.dumps({"20": _plan(1.0)}))
    dump_rows = {5: np.asarray(_plan(9.0)), 20: np.asarray(_plan(1.00004))}
    plans = recorded_plans(tmp_path, dump_rows)
    assert plans[20].dtype == np.float32
    np.testing.assert_array_equal(plans[20], np.asarray(_plan(1.0), dtype=np.float32))
    np.testing.assert_array_equal(plans[5], np.asarray(_plan(9.0), dtype=np.float32))


def test_a_coroutine_that_never_awaits_runs_without_a_loop():
    async def no_wait():
        return 3

    assert _finish(no_wait()) == 3

    async def waits():
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="awaited"):
        _finish(waits())
