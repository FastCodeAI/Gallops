"""H15's phase B verdict rule (b_report), on synthetic picks: numpy only."""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception import b_report


def test_clustered_se_is_zero_for_a_constant_difference_and_grows_with_cluster_noise():
    logs = np.repeat(np.arange(10), 20)
    assert b_report.clustered_se(np.full(200, 0.3), logs) < 1e-12
    rng = np.random.default_rng(0)
    within = rng.normal(0, 1, 200)
    between = np.repeat(rng.normal(0, 1, 10), 20)
    assert b_report.clustered_se(between, logs) > b_report.clustered_se(within, logs)


def _picks(pick_value_index, seeds=3, n=200):
    out = {"rows": np.arange(n)}
    for s in range(seeds):
        for which in ("kept", "last"):
            out[f"seed{s}_{which}"] = np.full(n, pick_value_index)
    return out


def test_the_rule_needs_a_gain_beyond_seed_range_and_two_se_on_every_seed():
    n = 200
    logs = np.repeat(np.arange(20), 10)
    pdm = np.zeros((n, 32, 9))
    pdm[:, 0, -1] = 0.80  # what C picks
    pdm[:, 1, -1] = 0.90  # what F picks: +0.10 on every row
    masks = {name: np.ones(n, bool) for name in b_report.SUBSETS}
    result = b_report.compare(_picks(1), _picks(0), pdm, np.arange(n), logs, masks, "kept")
    assert result["verdict"] == "F beats C"
    assert np.isclose(result["per_seed"][0]["f_minus_c"], 0.10)
    same = b_report.compare(_picks(0), _picks(0), pdm, np.arange(n), logs, masks, "kept")
    assert same["verdict"] == "NEUTRAL"


def test_b2s_rule_reads_each_arms_own_values_by_the_same_rule():
    logs = np.repeat(np.arange(20), 10)
    c = {f"seed{s}": np.full(200, 0.95) + 0.001 * s for s in range(3)}
    f = {f"seed{s}": np.full(200, 0.96) + 0.001 * s for s in range(3)}
    result = b_report.compare_values(f, c, logs)
    assert result["verdict"] == "F beats C" and np.isclose(result["c_seed_range"], 0.002)
    assert np.isclose(result["per_seed"][0]["f_minus_c"], 0.01)
    assert b_report.compare_values(c, c, logs)["verdict"] == "NEUTRAL"
