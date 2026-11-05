"""gallops.memory.evaluate: the final-grade formula, the paired bootstrap, and the rules as written."""

from __future__ import annotations

import numpy as np

from gallops.memory import evaluate, grade


def test_final_from_is_gates_times_weighted_mean():
    names = grade.SCORE_NAMES[:8]
    sub = np.ones((2, 8))
    sub[0, names.index("ego_progress")] = 0.5  # (5 * 0.5 + 5 + 2 + 2) / 14
    sub[1, names.index("drivable_area_compliance")] = 0.0  # a failed gate zeroes it
    assert np.allclose(evaluate.final_from(sub, names), [11.5 / 14, 0.0])


def test_boot_point_is_the_mean_and_draws_resample_whole_logs():
    values = np.array([1.0, 1.0, 0.0, 0.0])
    groups = np.array([0, 0, 1, 1])
    weights = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    point, draws = evaluate.boot(values, groups, weights)
    assert point == 0.5 and draws.tolist() == [1.0, 0.0, 0.5]


def _results(ab, bc, ac, flip=-0.01, comfort=0.01):
    def d(v, lo, hi):
        return {"value": v, "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}

    return {
        "differences": {
            "a-b": {"P1_strict": d(*ab), "P2a_flip": d(flip, -1, 1), "P2b_comfort": d(comfort, -1, 1)},
            "b-c": {"P1_strict": d(*bc)},
            "a-c": {"P1_strict": d(*ac)},
        }
    }


def test_verdict_supported_and_memory_helps():
    v = evaluate.verdict(_results((0.01, 0.004, 0.016), (0.0, -0.002, 0.002), (0.01, 0.004, 0.016)))
    assert v["P1"] == "supported"
    assert "(a) beats (b), and (b) ~ (c): memory helps; go on to the gate" in v["rules"]


def test_verdict_below_the_useful_bar():
    v = evaluate.verdict(_results((0.003, 0.001, 0.005), (0.0, -0.002, 0.002), (0.003, 0.001, 0.005)))
    assert v["P1"] == "supported but not useful (below +0.005)"


def test_verdict_refuted_when_the_ci_includes_zero_and_capacity_rule():
    v = evaluate.verdict(_results((0.001, -0.002, 0.004), (0.02, 0.01, 0.03), (0.021, 0.01, 0.03)))
    assert v["P1"] == "refuted (the CI includes 0)"
    assert "(a) ~ (b), and both beat (c): capacity or training; rethink" in v["rules"]
    assert "P1 fails: not supported, whatever P2 shows" in v["rules"]


def test_verdict_flags_an_arm_worse_than_released_and_p2_failing():
    v = evaluate.verdict(
        _results((0.01, 0.004, 0.016), (-0.02, -0.03, -0.01), (-0.01, -0.02, -0.004), flip=0.01)
    )
    assert "an arm is worse than (c): stop, or fix" in v["rules"]
    assert "P1 passes and P2 fails: supported on grade; comfort not shown" in v["rules"]
    assert v["P2a"] == "(a) not better than (b)"
