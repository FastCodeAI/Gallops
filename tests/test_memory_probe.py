"""The go / no-go probe's parts that decide what a row may see and how it is scored (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.compat import zip_strict
from gallops.memory import probe


def chain_table(chains):
    """A ticks-like table: `chains` is a list of (log, chain, n_moments); moments at chain_pos 1..n."""
    rows = []
    for log, chain, n in chains:
        for pos in range(1, n + 1):
            rows.append((log, chain, pos))
    rows.sort()
    t0 = {}
    table = {
        "log": np.asarray([r[0] for r in rows]),
        "chain": np.asarray([r[1] for r in rows]),
        "chain_pos": np.asarray([r[2] for r in rows]),
    }
    times = []
    for log, chain, pos in rows:
        base = t0.setdefault((log, chain), len(t0) * 100_000_000)
        times.append(base + pos * 500_000)
    table["timestamp"] = np.asarray(times)
    return table


def test_window_is_the_row_and_its_predecessors_newest_first():
    table = chain_table([("a", 0, 10)])
    row = int(np.flatnonzero(table["chain_pos"] == 9)[0])
    idx = probe.window_index(table, np.asarray([row]), 4)
    assert table["chain_pos"][idx[0]].tolist() == [9, 8, 7, 6]


def test_a_planted_future_value_never_reaches_the_features():
    table = chain_table([("a", 0, 12), ("a", 1, 12), ("b", 0, 12)])
    feats = np.zeros((len(table["log"]), 3))
    feats[:, 0] = table["chain_pos"]
    row = int(np.flatnonzero((table["log"] == "a") & (table["chain"] == 0) & (table["chain_pos"] == 8))[0])
    # plant: every later moment of the row's chain, and moments of other chains, carry 999
    later = (table["log"] == "a") & (table["chain"] == 0) & (table["chain_pos"] > 8)
    other = ~((table["log"] == "a") & (table["chain"] == 0))
    feats[later | other, 1:] = 999.0
    for k in probe.WINDOWS:
        x = probe.window_features(feats, probe.window_index(table, np.asarray([row]), k))
        assert not np.any(x == 999.0)
        assert x.reshape(k, 3)[:, 0].tolist() == [8 - j for j in range(k)]


def test_too_little_history_is_refused_not_padded():
    table = chain_table([("a", 0, 5)])
    row = int(np.flatnonzero(table["chain_pos"] == 5)[0])
    probe.window_index(table, np.asarray([row]), 4)
    with pytest.raises(KeyError):
        probe.window_index(table, np.asarray([row]), 8)


def test_check_no_future_catches_a_later_moment():
    table = chain_table([("a", 0, 6)])
    rows = np.asarray([2])
    with pytest.raises(AssertionError):
        probe.check_no_future(table, rows, np.asarray([[2, 3]]))  # 3 is after the row
    with pytest.raises(AssertionError):
        other = chain_table([("a", 0, 6), ("a", 1, 6)])
        probe.check_no_future(other, np.asarray([7]), np.asarray([[7, 1]]))  # another chain
    probe.check_no_future(table, rows, np.asarray([[2, 1, 0]]))


def test_pool_map_global_mean_and_blocks():
    enc = np.zeros((probe.GRID_H, probe.GRID_W, probe.DIM), np.float32)
    enc[:4, :4, 0] = 1.0  # the top-left 4 x 4 block only
    pooled = probe.pool_map(enc.reshape(1, -1, probe.DIM))[0]
    assert pooled.shape == ((1 + probe.POOL_H * probe.POOL_W) * probe.DIM,)
    assert pooled[0] == pytest.approx(16 / 512)
    blocks = pooled[probe.DIM :].reshape(probe.POOL_H * probe.POOL_W, probe.DIM)
    assert blocks[0, 0] == 1.0 and blocks[1:, 0].sum() == 0.0


def frames_with(vx, ax):
    return [
        {
            "ego_dynamic_state": [v, 0.0, a, 0.0],
            "ego2global_translation": [v_sum, 0.0, 0.0],
            "timestamp": i * 500_000,
        }
        for i, (v, a, v_sum) in enumerate(zip_strict(vx, ax, np.cumsum(np.r_[0, np.asarray(vx[:-1]) * 0.5])))
    ]


def test_targets_are_the_log_s_speed_ahead_and_step1_s_onset():
    vx = [10.0] * 6 + [8.0] * 6
    ax = [0.0] * 5 + [-3.0] + [0.0] * 6
    frames = frames_with(vx, ax)
    t = probe.targets_for_log(frames, [0, 2, 5, 6])
    assert t["has_future"].tolist() == [True, True, True, False]
    assert t["speed_1s"][:3].tolist() == [10.0, 10.0, 8.0]
    assert t["speed_3s"][:3].tolist() == [8.0, 8.0, 8.0]
    assert t["onset"].tolist() == [False, True, False, False]  # frame 5 is braking already
    assert t["event"][:3].tolist() == [False, True, True]
    assert t["gap_s"][0] == pytest.approx(3.0)


def test_average_precision_matches_the_step_definition():
    y = np.asarray([1, 0, 1, 0])
    s = np.asarray([0.9, 0.8, 0.7, 0.1])
    # ranks: P@1 = 1 (r .5), P@3 = 2/3 (r 1.0)
    assert probe.average_precision(y, s) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert probe.average_precision(y, s, np.ones(4) * 3) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_ridge_path_is_the_closed_form():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=(50, 2))
    coef = probe.ridge_path(x, y, np.asarray([0.5]))[0]
    assert np.allclose(coef, np.linalg.solve(x.T @ x + 0.5 * np.eye(4), x.T @ y))


def test_log_folds_keep_each_log_whole():
    groups = np.asarray(["a", "a", "b", "c", "c", "c", "d"])
    fold = probe.log_folds(groups, 2, 0)
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1


def test_log_weights_draw_as_many_logs_as_there_are():
    w = probe.log_weights(7, 20, 0)
    assert w.shape == (20, 7) and (w.sum(1) == 7).all()


def test_a_saved_pooled_map_made_from_another_cache_is_refused(tmp_path):
    cache_dir, out = tmp_path / "release", tmp_path / "probe"
    cache_dir.mkdir()
    out.mkdir()
    np.save(cache_dir / "encoder.npy", np.ones((4, 512, 256), np.float32))
    first = np.load(probe.stage_pool(cache_dir, out, 4)).copy()
    assert probe.stage_pool(cache_dir, out, 4) == out / "pooled.npy"  # same inputs: reused
    np.save(cache_dir / "encoder.npy", np.full((4, 512, 256), 7.0, np.float32))  # a rebuilt cache
    with pytest.raises(SystemExit, match="other inputs"):
        probe.stage_pool(cache_dir, out, 4)
    assert np.array_equal(np.load(out / "pooled.npy"), first)


def test_a_saved_pca_fitted_on_other_moments_is_refused(tmp_path):
    rng = np.random.default_rng(0)
    pooled = rng.normal(size=(300, 20)).astype(np.float32)
    key = {"rows": 300}
    a = probe.stage_pca(pooled, np.arange(200), tmp_path, key)
    b = probe.stage_pca(pooled, np.arange(200), tmp_path, key)  # same inputs: reused
    assert all(np.array_equal(a[k], b[k]) for k in a)
    with pytest.raises(SystemExit, match="other inputs"):
        probe.stage_pca(pooled, np.arange(250), tmp_path, key)
    with pytest.raises(SystemExit, match="other inputs"):
        probe.stage_pca(pooled, np.arange(200), tmp_path, {"rows": 301})


def test_a_stage_saved_before_it_had_a_key_is_refused(tmp_path):
    cache_dir, out = tmp_path / "release", tmp_path / "probe"
    cache_dir.mkdir()
    out.mkdir()
    np.save(cache_dir / "encoder.npy", np.ones((2, 512, 256), np.float32))
    np.save(out / "pooled.npy", np.zeros((2, 8448), np.float32))
    with pytest.raises(SystemExit, match="other inputs"):
        probe.stage_pool(cache_dir, out, 2)
