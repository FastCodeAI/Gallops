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


def test_labels_count_a_car_still_far_away_at_the_tick(tmp_path):
    """#6 review, finding 4: a car 80 m out closing at 21 m/s, and the ego stopped.

    It reaches the ego at 3.8 s. Filtered to vehicles within 70 m at the tick, the
    labels said no collision while `grader.grade` over every vehicle said 3.8 s.
    """
    import json

    from gallops.expert import grader
    from gallops.train import scorer_labels

    ts = 1_000_000
    episode = tmp_path / "collect" / "frontal-0099" / "run_0"
    episode.mkdir(parents=True)
    (episode / "ego_poses.json").write_text(json.dumps({str(ts): np.eye(4).tolist()}))

    def car(x):
        pose = np.eye(4)
        pose[:2, :2] = [[-1.0, 0.0], [0.0, -1.0]]  # heading back down -x, towards us
        pose[0, 3] = x
        return pose.tolist()

    oncoming = {"timestamps": [0, 10_000_000], "poses": [car(101.0), car(-109.0)], "dims": [1.9, 4.5, 1.6]}
    (episode / "actors.json").write_text(json.dumps([{**oncoming, "is_target_actor": False}]))
    (episode / "reference_trajectory.json").write_text(
        json.dumps({str(ts + i * 500_000): [i, 0, 0] for i in range(20)})
    )
    drivable = tmp_path / "drivable"
    drivable.mkdir()
    np.savez(
        drivable / "scene-0099.npz",
        mask=np.ones((400, 400), dtype=bool),
        origin=np.array([-200.0, -200.0]),
        cell=1.0,
        scene="scene-0099",
        location="test",
    )
    cache = tmp_path / "cache" / "gt" / "frontal-0099__run_0.npz"
    cache.parent.mkdir(parents=True)
    np.savez(
        cache,
        natives=np.zeros((1, 32, 8, 3)),
        vocab=np.zeros((1, 0, 8, 3)),
        timestamp=np.array([ts]),
        frame_dir=np.array(str(episode)),
        cache_key=np.array("{}"),
    )

    truth = grader.grade(np.zeros((8, 3)), np.eye(4), ts, [oncoming], None, None).hit_at
    assert truth is not None and 3.0 < truth < 4.0
    scorer_labels.label_episode(cache, drivable, np.random.default_rng(0))
    labels = np.load(cache.with_name("frontal-0099__run_0.labels.npz"))
    assert abs(float(labels["hit_at"][0, 0]) - truth) < 1e-6
    assert np.isfinite(labels["hit5_at"][0, 0])
