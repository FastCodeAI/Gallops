"""fastgrade must give grader.grade's verdicts exactly; it only exists to be faster."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gallops.expert import fastgrade, grader


class Disc:
    """A drivable area that is a disc: enough to exercise the off-road test."""

    def __init__(self, centre, radius):
        self.centre, self.radius = np.asarray(centre, float), radius

    def contains(self, points):
        return np.linalg.norm(np.atleast_2d(points) - self.centre, axis=1) <= self.radius


def _pose(x, y, yaw):
    m = np.eye(4)
    m[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    m[:2, 3] = (x, y)
    return m


def _scene(rng):
    ego2world = _pose(*rng.uniform(-50, 50, 2), rng.uniform(-math.pi, math.pi))
    t0 = 1_000_000_000
    actors = []
    for _ in range(6):
        start = ego2world[:2, 3] + rng.uniform(-30, 30, 2)
        velocity = rng.uniform(-8, 8, 2)
        stamps = t0 + np.arange(-1, 7) * 0.5e6
        poses = [
            _pose(*(start + velocity * (s - t0) / 1e6), math.atan2(velocity[1], velocity[0])) for s in stamps
        ]
        actors.append({"timestamps": stamps.tolist(), "poses": poses, "dims": [1.9, 4.5, 1.6]})
    speed = rng.uniform(0, 12)
    t = np.arange(1, 9) * 0.5
    plans = []
    for _ in range(24):
        lateral = rng.uniform(-4, 4) * (1 - np.cos(np.pi * np.clip(t / 2, 0, 1))) / 2
        x = speed * rng.uniform(0.3, 1.2) * t
        heading = np.arctan2(np.diff(np.r_[0, lateral]), np.diff(np.r_[0, x]) + 1e-9)
        plans.append(np.column_stack([x, lateral, heading]))
    reference = np.array(
        [ego2world[:2, 3] + ego2world[:2, :2] @ [s, 0.5 * math.sin(s / 10)] for s in range(-10, 80, 5)]
    )
    area = Disc(ego2world[:2, 3] + ego2world[:2, :2] @ [20, 0], 26)
    return np.stack(plans), ego2world, t0, actors, reference, area


@pytest.mark.parametrize("seed", range(12))
def test_matches_grader(seed):
    plans, ego2world, t0, actors, reference, area = _scene(np.random.default_rng(seed))
    fast = fastgrade.grade_many(plans, ego2world, t0, actors, reference, area)
    for i, plan in enumerate(plans):
        slow = grader.grade(plan, ego2world, t0, actors, reference, area)
        for key, value in (
            ("hit_at", slow.hit_at),
            ("off_road_at", slow.off_road_at),
            ("off_path_at", slow.off_path_at),
        ):
            got = fast[key][i]
            assert (value is None and np.isnan(got)) or value == pytest.approx(got, abs=1e-9), (
                seed,
                i,
                key,
                value,
                got,
            )
        assert fast["max_deviation"][i] == pytest.approx(slow.max_deviation, abs=1e-9)


def test_scenes_actually_exercise_every_verdict():
    """A test that never sees a hit or an off-road plan proves nothing about them."""
    seen = {"hit_at": 0, "off_road_at": 0, "off_path_at": 0, "clear": 0}
    for seed in range(12):
        plans, ego2world, t0, actors, reference, area = _scene(np.random.default_rng(seed))
        fast = fastgrade.grade_many(plans, ego2world, t0, actors, reference, area)
        for key in ("hit_at", "off_road_at", "off_path_at"):
            seen[key] += int((~np.isnan(fast[key])).sum())
        seen["clear"] += int((np.isnan(fast["hit_at"]) & np.isnan(fast["off_road_at"])).sum())
    assert all(v > 5 for v in seen.values()), seen


def test_extended_hit_is_never_later_than_a_4s_hit():
    plans, ego2world, t0, actors, reference, area = _scene(np.random.default_rng(3))
    fast = fastgrade.grade_many(plans, ego2world, t0, actors, reference, area)
    both = ~np.isnan(fast["hit_at"])
    assert (fast["hit5_at"][both] == fast["hit_at"][both]).all()
    assert np.isnan(fast["hit_at"][np.isnan(fast["hit5_at"])]).all()


def test_stopped_plan_has_no_ttc_violation_and_no_travel():
    _, ego2world, t0, actors, reference, area = _scene(np.random.default_rng(0))
    stopped = np.zeros((1, 8, 3))
    fast = fastgrade.grade_many(stopped, ego2world, t0, actors, reference, area)
    assert np.isnan(fast["ttc_at"][0])
    assert fast["travel"][0] == 0
