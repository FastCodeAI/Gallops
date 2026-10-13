"""The expert's two halves: the grader that judges, and the teacher that draws.

The grader's real test is not here -- it is that it reproduces, verdict for
verdict, the 38,080 candidate judgements paddock recorded in `check.json` for the
neuroNCAP candidate analysis. That needs recorded runs which no clean checkout
has, so these keep the invariants a fixture can hold: the vendor's footprint, the
geometry of an overlap, and the physics of what gets drawn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gallops.expert import draw, drivable, grader, rounds


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
        assert "0.5" in source, "the box centre sits 0.5 m ahead of the rear axle"

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

    def test_a_three_column_reference_is_accepted(self):
        """The logged route is stored as (x, y, z); `grade` may be called with it."""
        reference = np.column_stack([np.arange(0.0, 200.0, 5.0), np.zeros(40), np.full(40, 1.2)])
        on_path = np.column_stack([np.arange(1, 9) * 5.0, np.zeros(8), np.zeros(8)])
        assert grader.grade(on_path, np.eye(4), 0, [], reference).clear

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

    def test_it_brakes_for_a_parked_car_rather_than_swerving_round_it(self):
        """Braking removes the ego's own closing speed, and a parked car supplies
        none of its own -- so stopping resolves it and is far more robust than a
        lateral move. On neuroNCAP 0103 the released model scores 5.0 ten times
        out of ten by simply stopping, while a model taught to swerve here went
        the wrong way into the obstacle."""
        reference = np.column_stack([np.arange(0.0, 200.0, 1.0), np.zeros(200)])
        plan = draw.teach(np.eye(4), 0, 10.0, reference, [actor_ahead(25.0)])
        assert plan.kind == "brake"
        assert plan.clear
        assert plan.offset_m == 0.0, "braking should not also leave the lane"

    def test_it_still_swerves_for_a_car_coming_at_us(self):
        """The opposite case: an oncoming car supplies the closing speed itself,
        so braking only changes how fast you are going when it arrives."""
        reference = np.column_stack([np.arange(0.0, 200.0, 1.0), np.zeros(200)])
        oncoming = actor_ahead(45.0)
        # driving toward us at 9 m/s, which is what the frontal scenarios do
        poses = []
        for seconds in (-10.0, 0.0, 10.0):
            pose = np.eye(4)
            pose[0, 3] = 45.0 - 9.0 * seconds
            poses.append(pose.tolist())
        oncoming["timestamps"], oncoming["poses"] = [-10e6, 0, 10e6], poses

        plan = draw.teach(np.eye(4), 0, 10.0, reference, [oncoming])
        assert plan.kind == "swerve", f"braked at a closing car ({plan.kind})"
        assert plan.clear
        assert abs(plan.offset_m) >= 1.0
        assert abs(plan.offset_m) <= grader.TRUST_BAND_M

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


def a_road(width_m: float = 7.0, length_m: float = 200.0) -> drivable.DrivableArea:
    """A straight drivable strip along +x, centred on y = 0."""
    cell = drivable.CELL_M
    origin = np.array([-20.0, -30.0])
    cols = int(length_m / cell)
    rows = int(60.0 / cell)
    mask = np.zeros((rows, cols), dtype=bool)
    ys = origin[1] + (np.arange(rows) + 0.5) * cell
    mask[np.abs(ys) <= width_m / 2, :] = True
    return drivable.DrivableArea(mask=mask, origin=origin, cell=cell, scene="test", location="test")


class TestDrivableArea:
    def test_the_road_contains_its_own_centreline_and_not_the_pavement(self):
        area = a_road()
        assert area.contains(np.array([[10.0, 0.0]]))[0]
        assert not area.contains(np.array([[10.0, 9.0]]))[0]

    def test_a_point_outside_the_raster_is_not_drivable(self):
        """Leaving the window means leaving what this was built to answer for."""
        assert not a_road().contains(np.array([[5000.0, 0.0]]))[0]

    def test_a_swerve_off_the_road_is_graded_as_such(self):
        area = a_road()
        straight = np.column_stack([np.arange(1, 9) * 5.0, np.zeros(8), np.zeros(8)])
        assert grader.grade(straight, np.eye(4), 0, [], None, area).clear
        wide = straight.copy()
        wide[:, 1] = 6.0  # onto the pavement of a 7 m road
        verdict = grader.grade(wide, np.eye(4), 0, [], None, area)
        assert not verdict.clear
        assert verdict.off_road_at is not None

    def test_the_teacher_swerves_the_other_way_when_the_near_side_is_blocked(self):
        """Regression: it only ever tried the side away from the actor.

        On a two-lane street with the ego in the right-hand lane, "away from the
        oncoming car" is the parked cars, and the empty carriageway is on the
        side the actor came from. Trying one side steered it off the road.
        """
        # a road drivable only to the LEFT of the ego
        area = a_road(width_m=7.0)
        shifted = drivable.DrivableArea(
            mask=area.mask,
            origin=area.origin + np.array([0.0, 2.5]),
            cell=area.cell,
            scene="test",
            location="test",
        )
        reference = np.column_stack([np.arange(0.0, 200.0, 1.0), np.zeros(200), np.zeros(200)])
        # Oncoming at 8 m/s and slightly LEFT, so "away from it" is right, off the
        # road. Parked, the expert brakes instead, and this test never asserted.
        poses = []
        for t in (-10.0, 10.0):
            pose = np.eye(4)
            pose[:2, :2] = [[-1.0, 0.0], [0.0, -1.0]]
            pose[0, 3], pose[1, 3] = 40.0 - 8.0 * t, 0.4
            poses.append(pose.tolist())
        blocker = {"timestamps": [-10e6, 10e6], "poses": poses, "dims": [1.9, 4.5, 1.6]}
        blocker["is_target_actor"] = True
        plan = draw.teach(np.eye(4), 0, 9.0, reference, [blocker], area=shifted)
        assert plan.kind == "swerve", plan.kind
        assert plan.offset_m > 0, "the only drivable side is to the left"


class TestRounds:
    def _round(self, tmp_path, index, ticks):
        labelled = tmp_path / f"labelled{index}"
        labelled.mkdir(exist_ok=True)
        (labelled / "ep.jsonl").write_text("\n".join("{}" for _ in range(ticks)) + "\n")
        return rounds.Round(
            index=index,
            checkpoint=f"ckpt{index}",
            collected=str(tmp_path),
            labelled=str(labelled),
            episodes=1,
            ticks=ticks,
        )

    def test_the_training_set_aggregates_every_round(self, tmp_path):
        """The whole point: training on the newest round alone is where DAgger
        implementations oscillate."""
        for index in (0, 1, 2):
            rounds.record(tmp_path, self._round(tmp_path, index, 10 * (index + 1)))
        assert len(rounds.training_set(tmp_path)) == 3
        assert len(rounds.training_set(tmp_path, through=1)) == 2

    def test_rounds_stay_ordered_and_a_rerun_replaces_rather_than_duplicates(self, tmp_path):
        rounds.record(tmp_path, self._round(tmp_path, 1, 5))
        rounds.record(tmp_path, self._round(tmp_path, 0, 5))
        rounds.record(tmp_path, self._round(tmp_path, 1, 7))
        got = rounds.load(tmp_path)
        assert [r.index for r in got] == [0, 1]
        assert got[1].ticks == 7

    def test_next_index_follows_the_last(self, tmp_path):
        assert rounds.next_index(tmp_path) == 0
        rounds.record(tmp_path, self._round(tmp_path, 0, 3))
        assert rounds.next_index(tmp_path) == 1

    def test_a_stale_manifest_fails_loudly(self, tmp_path):
        entry = self._round(tmp_path, 0, 3)
        rounds.record(tmp_path, entry)
        for item in Path(entry.labelled).iterdir():
            item.unlink()
        Path(entry.labelled).rmdir()
        with pytest.raises(SystemExit):
            rounds.training_set(tmp_path)


def test_hazard_in_frame_follows_the_one_camera_it_reads():
    """A crossing car at 45 deg is outside CAM_FRONT; one dead ahead is not.

    This is the whole reason side-scenario swerve targets are dropped from the
    imitation loss: the model reads CAM_FRONT alone, and half that frame is
    32.6 deg wide, so it never sees the car it would be taught to avoid.
    """
    from gallops.expert.label import hazard_bearing

    ego2world = np.eye(4)

    bearing, in_frame = hazard_bearing(ego2world, 0, [actor_ahead(20.0)])
    assert bearing < 1.0
    assert in_frame

    # the same box swung out to the left, where a crossing actor actually appears
    crossing = actor_ahead(14.0)
    for pose in crossing["poses"]:
        pose[1][3] = 14.0
    bearing, in_frame = hazard_bearing(ego2world, 0, [crossing])
    assert bearing > 40.0
    assert not in_frame

    assert hazard_bearing(ego2world, 0, []) is None


class TestScorerTargets:
    """The soft targets the scorer is trained on, which are our referee, not PDM's.

    These decide what gets DRIVEN. An earlier version measured progress against
    free-running speed, so a clear swerve scored 1.00 and a clear brake 0.75 --
    the scorer was trained to prefer swerving on every tick while the trajectory
    loss was being taught to brake, and the scorer won. On neuroNCAP 0103 the
    released model slows from 5.6 to 1.4 m/s; the finetuned one accelerated to
    8.0 and swerved, whichever way the expert was changed.
    """

    def test_the_score_peaks_at_the_expert_distance_and_falls_off_both_sides(self):
        """Capping instead of penalising was the bug: a brake and a full-speed
        plan both reached the cap, tied, and the lane term decided. The model then
        accelerated 26% on ticks where the expert braked 38%."""
        from gallops.train.losses import score_target

        matched = score_target(None, None, travelled_m=20.0, expert_travelled_m=20.0)
        further = score_target(None, None, travelled_m=40.0, expert_travelled_m=20.0)
        shorter = score_target(None, None, travelled_m=5.0, expert_travelled_m=20.0)
        assert matched == pytest.approx(1.0, abs=0.01)
        assert further < matched, "overshooting the expert is not penalised"
        assert shorter < matched, "undershooting the expert is not penalised"

    def test_a_brake_outscores_a_swerve_when_the_expert_braked(self):
        """The ordering the whole change exists for."""
        from gallops.train.losses import score_target

        # the expert braked here, so its plan covers 20 m
        brake = score_target(None, None, 20.0, 20.0, deviation_m=0.2)
        swerve = score_target(None, None, 40.0, 20.0, deviation_m=2.5)
        assert brake > swerve, f"brake {brake:.3f} does not beat swerve {swerve:.3f}"

    def test_a_swerve_outscores_a_brake_when_the_expert_swerved(self):
        """And it must not flip the other way: on an oncoming car the expert's
        plan covers the full distance, so the brake falls short on progress."""
        from gallops.train.losses import score_target

        swerve = score_target(None, None, 40.0, 40.0, deviation_m=2.7)
        brake = score_target(None, None, 20.0, 40.0, deviation_m=0.2)
        assert swerve > brake, f"swerve {swerve:.3f} does not beat brake {brake:.3f}"

    def test_a_dead_stop_still_loses_to_driving_on(self):
        """The emergency-stop trap, still shut. An all-zero trajectory hits
        nothing, and if safety alone set the target it would score 1.0 everywhere
        and teach the model to brake for nothing."""
        from gallops.train.losses import score_target

        assert score_target(None, None, 0.0, 30.0) == pytest.approx(0.5, abs=0.01)
        assert score_target(None, None, 30.0, 30.0) > score_target(None, None, 0.0, 30.0)

    def test_a_later_crash_scores_above_an_earlier_one(self):
        """Every proposal crashes on a late frontal tick, so the ranking among
        them is the whole signal -- a hard zero would throw it away."""
        from gallops.train.losses import score_target

        early = score_target(0.5, None, 30.0, 30.0)
        late = score_target(3.5, None, 30.0, 30.0)
        assert late > early > 0.0

    def test_leaving_the_road_costs_but_beats_a_collision(self):
        from gallops.train.losses import score_target

        on_road = score_target(None, None, 30.0, 30.0)
        off_road = score_target(None, 2.0, 30.0, 30.0)
        collision = score_target(2.0, None, 30.0, 30.0)
        assert on_road > off_road > collision


def test_a_displaced_car_is_given_a_drivable_way_back():
    """`follow` returns points ON the logged path, so a plan for a car that is off
    it used to start with a jump straight back -- 3 m sideways in half a second,
    24.5 m/s^2, six times this module's own comfort limit. Round 0 never showed it
    because the released model never leaves the lane; the finetuned one lives
    there."""
    from gallops.expert.draw import LATERAL_ACCEL_MAX, rejoin

    times = np.arange(1, 9) * 0.5
    # a straight path running 3 m to the ego's right, i.e. the car is 3 m left of it
    path = np.column_stack([np.linspace(-20, 80, 200), np.full(200, -3.0)])

    correction = rejoin(path, speed=10.0, distances=10.0 * times)
    assert correction[0] == pytest.approx(3.0, abs=0.6), "does not start where the car is"
    assert abs(correction[-1]) < 0.05, "never actually rejoins the path"

    lateral = np.gradient(np.gradient(correction, times), times)
    assert np.abs(lateral).max() < LATERAL_ACCEL_MAX * 1.5, "the rejoin is not drivable"


def test_a_car_already_on_the_path_gets_no_correction():
    """The common case must be untouched, or every round-0 label moves for nothing."""
    from gallops.expert.draw import rejoin

    times = np.arange(1, 9) * 0.5
    path = np.column_stack([np.linspace(-20, 80, 200), np.zeros(200)])
    assert np.abs(rejoin(path, speed=10.0, distances=10.0 * times)).max() < 1e-3


def test_a_car_braking_to_a_stop_does_not_slide_back_onto_the_path():
    """The rejoin was timed by the clock, so a plan that stopped within a metre still
    covered the whole sideways gap: 934 of 9,495 round-0 targets slid up to 2 m
    sideways, one step pointing 90 deg off the road. Laid out over distance, a car
    that stops early stops part of the way back, still pointing down the road."""
    from gallops.expert.draw import _plan_from, undrivable

    path = np.column_stack([np.linspace(-20, 80, 200), np.full(200, 1.0)])  # car 1 m right of it
    plan = _plan_from(path, speed=2.0, scale=0.0, offset=0.0, ramp_s=1.0)
    assert undrivable(plan) is None
    assert plan[-1, 0] < 1.0, "the fixture should stop within a metre"
    assert abs(plan[-1, 1]) < 0.2, "a stopped car moved sideways"
    assert np.abs(plan[:, 2]).max() < np.radians(20), "a stopped car turned to face sideways"


def test_a_braked_swerve_keeps_its_turning_radius():
    """Sized for the starting speed, a swerve drawn while braking used to tighten as
    the car slowed. Laid out over distance it cannot."""
    from gallops.expert.draw import _plan_from, min_ramp_for, undrivable

    path = np.column_stack([np.linspace(-20, 80, 200), np.zeros(200)])
    for scale in (1.0, 0.6, 0.3, 0.0):
        plan = _plan_from(path, speed=8.0, scale=scale, offset=2.0, ramp_s=min_ramp_for(2.0, 8.0))
        assert undrivable(plan) is None, scale


def test_the_drivability_check_flags_a_sideways_slide():
    from gallops.expert.draw import undrivable

    slide = np.array([[0.3 * k, 0.4 * k, 0.0] for k in range(1, 9)])  # 53 deg off the car's heading at once
    assert undrivable(slide) is not None
    straight = np.array([[2.0 * k, 0.0, 0.0] for k in range(1, 9)])
    assert undrivable(straight) is None


def test_the_trainer_still_fits_the_objects_it_drives():
    """Every method the training loop calls on the dataset and the referee exists.

    `make check` runs where torch is absent, so nothing here can construct a
    TickDataset, and nothing constructs a Referee either. An edit can delete a
    method the loop depends on and the gate stays green. It has happened three
    times: `TickDataset.batch` was dropped by a bad replace and the run died eight
    minutes in; `Referee.reference` was never written while `Referee.targets`
    already called it; and this very test was swallowed by a span-replace of the
    block above it. Parsing the source costs nothing and closes all of them.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "gallops" / "train"
    trainer = ast.parse((source / "swerve.py").read_text())

    def methods_of(tree, name):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
        return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}

    def called_on(tree, names):
        return {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
        }

    dataset = methods_of(ast.parse((source / "swerve_data.py").read_text()), "TickDataset")
    assert {"agent_input", "features", "batch", "build"} <= dataset, dataset
    missing = called_on(trainer, {"dataset", "validation"}) - dataset
    assert not missing, f"the trainer calls TickDataset.{missing}, which do not exist"

    referee = methods_of(trainer, "Referee")
    assert {"targets", "good_anchors", "context", "reference"} <= referee, referee
    missing = called_on(trainer, {"referee"}) - referee
    assert not missing, f"the trainer calls Referee.{missing}, which do not exist"


def test_each_tick_is_graded_where_the_ego_is_at_that_tick(tmp_path):
    """The referee grades a tick from that tick's pose, whichever tick came first.

    It used to cache the whole context per episode, pose included, so every tick
    was graded from wherever the first one looked up had the ego -- 14-16 m off
    on round 0, flipping a fifth of the stationary verdicts the scorer learns
    from. Two ticks 15 m apart with a car parked ahead: the same short plan is
    clear from the first and a crash from the second, and asking about the first
    must not change the answer for the second.
    """
    import json

    from gallops.train.swerve import Referee

    episode = tmp_path / "stationary-0099" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    early, late = 1_000_000, 2_000_000
    poses = {}
    for stamp, x in ((early, 0.0), (late, 15.0)):
        pose = np.eye(4)
        pose[0, 3] = x
        poses[str(stamp)] = pose.tolist()
    (episode / "ego_poses.json").write_text(json.dumps(poses))
    (episode / "actors.json").write_text(json.dumps([actor_ahead(22.0, t0=late)]))

    def tick(stamp):
        return {"frame": str(episode / "CAM_FRONT" / f"{stamp}.jpg"), "timestamp": stamp}

    referee = Referee(anchors=np.zeros((1, 8, 3)))
    plan = straight(speed=2.0)
    first = referee.context(tick(early))
    second = referee.context(tick(late))

    assert second[0][0, 3] == pytest.approx(15.0)
    assert grader.grade(plan, first[0], early, first[1]).hit_at is None
    assert grader.grade(plan, second[0], late, second[1]).hit_at is not None


def test_a_brake_holds_its_speed_before_it_slows():
    """The swerve was given a when-to-begin and the brake was not, so a brake
    started at t=0 however far away the hazard was. With four seconds in hand a
    driver coasts first, which is what makes a reaction to a car 60 m away stop
    looking like a flinch."""
    from gallops.expert.draw import speed_profile

    times = np.arange(1, 9) * 0.5
    at_once = speed_profile(10.0, 0.5, times)
    delayed = speed_profile(10.0, 0.5, times, delay_s=1.5)

    assert delayed[0] == pytest.approx(5.0, abs=0.01), "not holding speed in the first half second"
    assert delayed[2] == pytest.approx(15.0, abs=0.01), "not holding speed through the delay"
    assert delayed[-1] > at_once[-1], "the delayed brake should cover more ground"
    # and it still ends up slowed, not coasting
    assert (delayed[-1] - delayed[-2]) / 0.5 == pytest.approx(5.0, abs=0.2)


def test_a_delayed_brake_still_decelerates_at_the_commanded_rate():
    """The delay must not quietly soften the deceleration -- that was the earlier
    bug where a commanded stop came out as a lift-off."""
    from gallops.expert.draw import BRAKE_MAX, speed_profile

    times = np.linspace(0.05, 4.0, 80)
    distance = speed_profile(12.0, 0.0, times, delay_s=1.0)
    speed = np.gradient(distance, times)
    slowing = np.abs(np.gradient(speed, times))
    assert slowing.max() >= BRAKE_MAX * 0.7, f"peak deceleration only {slowing.max():.1f} m/s^2"


def test_the_expert_follows_the_lane_the_car_is_in_when_one_was_computed(tmp_path):
    """Scene 0058: the human moved one lane over. With the lane centre on file, a
    plan for an empty road stays in the car's own lane instead of following them,
    and the label says which path it was drawn against."""
    import json

    from gallops.expert.label import label_episode

    episode = tmp_path / "stationary-0099" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    stamps = [1_000_000, 1_100_000]
    poses = {str(s): np.eye(4).tolist() for s in stamps}
    (episode / "ego_poses.json").write_text(json.dumps(poses))
    far = actor_ahead(200.0)
    (episode / "actors.json").write_text(json.dumps([{**far, "uuid": "x"}]))
    # the human path runs 3 m to the left, one lane over
    (episode / "reference_trajectory.json").write_text(
        json.dumps({str(i): [float(i) - 20.0, 3.0, 0.0] for i in range(120)})
    )
    lane = [[float(x), 0.0] for x in range(-10, 90)]
    (episode / "lane_reference.json").write_text(json.dumps({str(s): lane for s in stamps}))
    plan = straight(8.0)
    rows = [
        {
            "timestamp": s,
            "speed": 8.0,
            "accel": 0.0,
            "command": 2,
            "trajectory": plan.tolist(),
            "proposals": [plan.tolist()] * 32,
        }
        for s in stamps
    ]

    labelled = label_episode(episode, rows)
    assert all(t["reference"] == "lane" for t in labelled)
    assert all(t["expert_kind"] == "continue" for t in labelled)
    assert max(abs(y) for t in labelled for _, y, _ in t["expert"]) < 0.2, "followed the human, not the lane"

    (episode / "lane_reference.json").unlink()
    drifted = label_episode(episode, rows)
    assert all(t["reference"] == "logged" for t in drifted)
    assert max(abs(y) for t in drifted for _, y, _ in t["expert"]) > 1.0, (
        "fixture should show the old behaviour"
    )


def test_labelling_refuses_an_existing_label_set(tmp_path):
    """labelled_round0_v2 was rewritten in place between two runs that both name it."""
    from gallops.expert.label import fresh_out

    out = tmp_path / "labelled"
    fresh_out(out)
    assert out.is_dir()
    fresh_out(out)  # empty: nothing to overwrite
    (out / "frontal-0053__run_0.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="already holds labels"):
        fresh_out(out)


def test_a_gt_row_without_the_models_pick_is_refused():
    from gallops.expert.label import model_pick

    assert model_pick({"timestamp": 1, "trajectory": [[1, 0, 0]]}) is None  # model drove
    assert model_pick({"timestamp": 1, "drive": "gt", "model_trajectory": [[2, 0, 0]]}) == [[2, 0, 0]]
    with pytest.raises(SystemExit, match="no model_trajectory"):
        model_pick({"timestamp": 1, "drive": "gt", "trajectory": [[1, 0, 0]]})


def _peak_lateral_accel(waypoints) -> float:
    """v^2 kappa, chord by chord, from the car's own pose: what a passenger would feel."""
    points = np.vstack([[0.0, 0.0], np.asarray(waypoints)[:, :2]])
    heading = step = worst = 0.0
    for i in range(1, len(points)):
        delta = points[i] - points[i - 1]
        length = float(np.hypot(*delta))
        if length <= draw.STILL_M:
            continue
        new = float(np.arctan2(delta[1], delta[0]))
        turn = abs((new - heading + np.pi) % (2 * np.pi) - np.pi)
        worst = max(worst, (length / draw.WAYPOINT_DT) ** 2 * turn / ((step + length) / 2))
        heading, step = new, length
    return worst


def test_a_recovery_stays_within_the_lateral_acceleration_limit():
    """#2 review 7b: a 3 m return over RETURN_S alone peaked at 4.7 m/s^2 (limit 4)."""
    from gallops.expert.draw import LATERAL_ACCEL_MAX, _plan_from, min_ramp_for, undrivable

    reference = np.column_stack([np.arange(-20.0, 300.0, 1.0), np.zeros(320)])
    for speed in (9.0, 12.0, 15.0):
        plan = _plan_from(reference, speed, 1.0, 3.0, min_ramp_for(3.0, speed), 0.0, 1.5)
        assert _peak_lateral_accel(plan) <= LATERAL_ACCEL_MAX * 1.01, speed
        assert undrivable(plan) is None


def test_chasing_a_lane_off_the_cars_heading_stays_within_the_limit():
    """#4 review 3: pursuit clamped curvature only; 15 deg off at 12 m/s took 8.7 m/s^2."""
    from gallops.expert.draw import LATERAL_ACCEL_MAX, _plan_from, undrivable

    angle = np.radians(15.0)
    along = np.arange(-10.0, 200.0)
    reference = np.column_stack([along * np.cos(angle), along * np.sin(angle)])
    plan = _plan_from(reference, 12.0, 1.0, 0.0, 1.0)
    assert _peak_lateral_accel(plan) <= LATERAL_ACCEL_MAX * 1.01
    assert undrivable(plan) is None


def test_undrivable_refuses_a_turn_wide_enough_but_too_fast():
    from gallops.expert.draw import undrivable

    # a 20 m radius arc at 15 m/s: 11 m/s^2 sideways, though wider than MIN_TURN_RADIUS_M
    s = 15.0 * 0.5 * np.arange(1, 9)
    arc = np.column_stack([20 * np.sin(s / 20), 20 * (1 - np.cos(s / 20)), s / 20])
    reason = undrivable(arc)
    assert reason is not None and "m/s^2" in reason
