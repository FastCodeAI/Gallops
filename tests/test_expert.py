"""The expert's two halves: the grader that judges, and the teacher that draws.

The grader's real test is not here -- it is that it reproduces, verdict for
verdict, the 38,080 candidate judgements paddock recorded in `check.json` for the
neuroNCAP candidate analysis. That needs recorded runs which no clean checkout
has, so these keep the invariants a fixture can hold: the vendor's footprint, the
geometry of an overlap, and the physics of what gets drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

from gallops.expert import draw, grader


def straight(speed: float, poses: int = 8, dt: float = 0.5) -> np.ndarray:
    t = np.arange(1, poses + 1) * dt
    return np.column_stack([speed * t, np.zeros_like(t), np.zeros_like(t)])


def actor_ahead(distance: float, t0: int = 0, width: float = 1.9, length: float = 4.5) -> dict:
    """A box parked `distance` straight ahead, in a world frame the ego sits at."""
    pose = np.eye(4)
    pose[0, 3] = distance
    return {
        "timestamps": [t0 - 10e6, t0 + 10e6],
        "poses": [pose.tolist(), pose.tolist()],
        "dims": [width, length, 1.6],
        "is_target_actor": True,
    }


class TestGrader:
    def test_footprint_still_matches_the_vendor(self, pytestconfig):
        """The ego box is neuroNCAP's, so a crash here means a crash there.

        Skipped when the vendor checkout is not configured; when it is, a drift
        in those constants has to fail loudly rather than quietly rescore
        everything.
        """
        import os

        repo = os.environ.get("NEURO_NCAP_REPO")
        if not repo:
            pytest.skip("set NEURO_NCAP_REPO to check the constants against the vendor")
        source = (__import__("pathlib").Path(repo) / "neuro_ncap" / "evaluation" / "collision.py").read_text()
        assert "4.018" in source and "1.85" in source

    def test_a_box_dead_ahead_is_a_collision(self):
        verdict = grader.grade(straight(10.0), np.eye(4), 0, [actor_ahead(15.0)])
        assert not verdict.clear
        assert verdict.hit_at == pytest.approx(1.2, abs=0.3)

    def test_the_same_box_beside_us_is_not(self):
        beside = actor_ahead(15.0)
        beside["poses"] = [np.array(p) for p in beside["poses"]]
        for p in beside["poses"]:
            p[1, 3] = 6.0
        beside["poses"] = [p.tolist() for p in beside["poses"]]
        assert grader.grade(straight(10.0), np.eye(4), 0, [beside]).clear

    def test_straying_past_the_trust_band_is_graded_even_when_nothing_is_hit(self):
        wide = straight(10.0)
        wide[:, 1] = 6.0  # 6 m off a reference that runs dead straight
        reference = np.column_stack([np.arange(0, 120, 5.0), np.zeros(24)])
        verdict = grader.grade(wide, np.eye(4), 0, [], reference)
        assert not verdict.clear
        assert verdict.max_deviation > grader.TRUST_BAND_M


class TestDrawnTrajectories:
    def test_the_ramp_respects_the_lateral_acceleration_limit(self):
        times = np.linspace(0, 4, 400)
        for offset in (1.0, 2.5, 3.5):
            ramp = draw.min_ramp_for(offset)
            lateral = draw.lateral_profile(offset, ramp, times)
            peak = np.abs(np.gradient(np.gradient(lateral, times), times)).max()
            assert peak <= draw.LATERAL_ACCEL_MAX * 1.05

    def test_a_wider_evasion_needs_a_longer_ramp(self):
        assert draw.min_ramp_for(3.5) > draw.min_ramp_for(1.5)

    def test_distances_are_measured_from_the_ego_not_the_path_start(self):
        """Regression: the nominal plan must start where the car is now.

        Measuring arc length from the beginning of the logged path put the whole
        plan wherever the recording started, which read as a ten-metre excursion
        and cleared obstacles by being nowhere near them.
        """
        reference = np.column_stack([np.arange(-50.0, 50.0, 1.0), np.zeros(100)])
        points = draw.follow(reference, np.array([0.0, 10.0]))
        assert points[0] == pytest.approx([0.0, 0.0], abs=1e-6)
        assert points[1] == pytest.approx([10.0, 0.0], abs=1e-6)

    def test_it_continues_when_the_road_is_empty(self):
        reference = np.column_stack([np.arange(0.0, 200.0, 1.0), np.zeros(200)])
        plan = draw.teach(np.eye(4), 0, 10.0, reference, [])
        assert plan.kind == "continue"
        assert plan.clear

    def test_it_draws_a_swerve_around_a_parked_car_and_the_swerve_is_clear(self):
        reference = np.column_stack([np.arange(0.0, 200.0, 1.0), np.zeros(200)])
        plan = draw.teach(np.eye(4), 0, 10.0, reference, [actor_ahead(25.0)])
        assert plan.kind == "swerve"
        assert plan.clear
        assert abs(plan.offset_m) <= grader.TRUST_BAND_M
        assert abs(plan.offset_m) >= 1.0

    def test_it_never_teaches_a_trajectory_that_would_crash_when_one_exists(self):
        reference = np.column_stack([np.arange(0.0, 200.0, 1.0), np.zeros(200)])
        for distance in (15.0, 20.0, 30.0, 45.0):
            plan = draw.teach(np.eye(4), 0, 10.0, reference, [actor_ahead(distance)])
            assert plan.clear, f"no clear plan found with the car {distance} m ahead"

    def test_the_prediction_uses_what_was_visible_not_the_logged_future(self):
        """A target that swerves away later must not shrink the evasion now."""
        pose = np.eye(4)
        pose[0, 3] = 30.0
        late = np.eye(4)
        late[0, 3], late[1, 3] = 30.0, 40.0  # teleports aside after the horizon
        actor = {
            "timestamps": [-10e6, 0, 10e6],
            "poses": [pose.tolist(), pose.tolist(), late.tolist()],
            "dims": [1.9, 4.5, 1.6],
            "is_target_actor": True,
        }
        predicted = draw.predict_actor(actor, 0, grader.HORIZON)
        assert np.abs(predicted[:, 1]).max() < 1.0
