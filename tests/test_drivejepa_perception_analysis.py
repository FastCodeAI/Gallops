"""B-0-4's proposal analysis (numpy; the Hungarian match needs scipy and skips without it)."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.drivejepa_perception import b_analysis as ba


def _straight(speed, lateral=0.0, n=8):
    t = np.arange(1, n + 1) * 0.5
    return np.stack([speed * t, np.full(n, lateral), np.zeros(n)], axis=1)


def test_best_of_32_and_the_safe_share():
    pdm = np.ones((2, 4, 9))
    pdm[0, :, ba.FINAL] = [0.2, 0.9, 0.5, 0.1]
    pdm[1, :, ba.FINAL] = [0.4, 0.4, 0.4, 0.4]
    pdm[0, 1, ba.NC] = 0.0
    assert ba.best_of_k(pdm) == pytest.approx(0.65)
    assert ba.safe_share(pdm) == pytest.approx(7 / 8)


def test_spread_and_min_ade():
    proposals = np.stack([np.stack([_straight(4.0, lateral=y) for y in (-1.0, 1.0)])])  # 1 row, 2 proposals
    assert ba.spread(proposals) == pytest.approx({"longitudinal_m": 0.0, "lateral_m": 1.0})
    human = _straight(4.0, lateral=0.5)[None]
    got = ba.min_ade_miss(proposals, human)
    assert got["min_ade_m"] == pytest.approx(0.5) and got["miss_rate"] == 0.0


def test_a_steady_plan_does_not_move_between_frames():
    # A plan at constant speed, re-planned 0.5 s later from where the ego got to: the overlap agrees.
    plans = np.stack([_straight(6.0), _straight(6.0)])
    ego_next = np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    got = ba.steadiness(plans, np.array(["l", "l"]), np.array([0, 500_000]), ego_next)
    assert got == {"pairs": 1, "mean_m": pytest.approx(0.0, abs=1e-12)}
    plans[1] = _straight(6.0, lateral=0.4)  # the next frame's plan sits 0.4 m to the left
    assert ba.steadiness(plans, np.array(["l", "l"]), np.array([0, 500_000]), ego_next)[
        "mean_m"
    ] == pytest.approx(0.4)
    assert ba.steadiness(plans, np.array(["l", "m"]), np.array([0, 500_000]), ego_next)["pairs"] == 0


def test_the_hungarian_match_finds_proposals_that_only_reordered():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 32, 8, 3))
    b = a[:, ::-1].copy()
    got = ba.hungarian_moves(a, b, rng.normal(size=(2, 32)))
    assert got["matched_ade_m"] == pytest.approx(0.0)
    assert (got["match"] == np.arange(31, -1, -1)).all()
