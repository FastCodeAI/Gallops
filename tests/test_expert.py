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

    def test_deviation_measures_to_the_path_not_to_its_sample_points(self):
        """Regression: nearest-VERTEX inflated a trajectory sitting on the path.

        With nuScenes keyframes about 5 m apart it reported 2.00 m instead of 0,
        which shrank the trust band and, worse, would have marked a candidate that
        genuinely swerved as off-path -- penalising the behaviour being taught.
        """
        reference = np.column_stack([np.arange(0.0, 200.0, 5.0), np.zeros(40)])
        on_path = np.column_stack([np.arange(1, 9) * 5.0, np.zeros(8), np.zeros(8)])
        worst, off = grader.deviation(on_path, np.eye(4), reference)
        assert worst < 0.05
        assert off is None

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
            ramp = draw.min_ramp_for(offset, speed=12.0)
            lateral = draw.lateral_profile(offset, ramp, times)
            peak = np.abs(np.gradient(np.gradient(lateral, times), times)).max()
            assert peak <= draw.LATERAL_ACCEL_MAX * 1.05

    def test_a_wider_evasion_needs_a_longer_ramp(self):
        assert draw.min_ramp_for(3.5, speed=10.0) > draw.min_ramp_for(1.5, speed=10.0)

    def test_the_ramp_lengthens_as_speed_falls(self):
        """Regression: the acceleration budget alone permits an undrivable turn.

        |d2y/dt2| stays inside the budget at any speed, so a 2.45 m offset at
        1.8 m/s was drawn at a 1.7 m turn radius and taught as a target.
        """
        slow = draw.min_ramp_for(2.45, speed=1.8)
        fast = draw.min_ramp_for(2.45, speed=10.0)
        assert slow > fast
        assert slow > draw.HORIZON_S, "a manoeuvre this tight must be refused, not drawn"

    def test_a_drawn_swerve_stays_within_the_minimum_turn_radius(self):
        times = np.arange(1, draw.POSES + 1) * draw.WAYPOINT_DT
        for speed in (3.0, 6.0, 12.0):
            offset = 2.45
            ramp = draw.min_ramp_for(offset, speed)
            if ramp > draw.HORIZON_S:
                continue
            points = np.column_stack(
                [draw.speed_profile(speed, 1.0, times), draw.lateral_profile(offset, ramp, times)]
            )
            radii = []
            for i in range(1, len(points) - 1):
                a, b, c = points[i - 1], points[i], points[i + 1]
                # scalar cross product; np.cross on 2-vectors is deprecated
                area = abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]) / 2
                if area > 1e-9:
                    sides = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
                    radii.append(sides / (4 * area))
            assert min(radii) >= draw.MIN_TURN_RADIUS_M * 0.9, f"{speed} m/s: {min(radii):.1f} m"

    def test_a_commanded_stop_brakes_at_the_stated_rate(self):
        """Regression: the drop was spread over the horizon, so BRAKE_MAX was dead.

        A stop from 1.8 m/s came out as 0.45 m/s^2 over 3.60 m, and every tick
        labelled "brake" would have taught a lift-off.
        """
        times = np.arange(1, draw.POSES + 1) * draw.WAYPOINT_DT
        for speed in (1.8, 3.96, 10.0):
            travelled = draw.speed_profile(speed, 0.0, times)[-1]
            assert travelled == pytest.approx(speed**2 / (2 * draw.BRAKE_MAX), rel=0.02)

    def test_least_impact_prefers_the_slower_contact_over_the_later_one(self):
        """Regression: it maximised hit_at, a time at 0.1 s resolution.

        When every option hit in the same tenth of a second it returned whichever
        was first in the list, once choosing an undrivable swerve over a stop.
        """
        fast = np.column_stack([np.arange(1, 9) * 5.0, np.zeros(8), np.zeros(8)])
        slow = np.column_stack([np.arange(1, 9) * 0.5, np.zeros(8), np.zeros(8)])
        assert draw.speed_at(fast, 1.0) > draw.speed_at(slow, 1.0)
        assert draw.speed_at(slow, 1.0) == pytest.approx(1.0, abs=0.05)

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
