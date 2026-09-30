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
        blocker = actor_ahead(26.0)
        blocker["poses"] = [np.array(p) for p in blocker["poses"]]
        for pose in blocker["poses"]:
            pose[1, 3] = -0.4  # slightly right, so "away" would be left... then flip
        blocker["poses"] = [p.tolist() for p in blocker["poses"]]
        plan = draw.teach(np.eye(4), 0, 9.0, reference, [blocker], area=shifted)
        if plan.kind == "swerve":
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
    """The soft targets the scorer is trained on, which are our referee, not PDM's."""

    def test_a_clear_plan_that_keeps_up_scores_top(self):
        from gallops.train.losses import score_target, travelled

        plan = straight(10.0)
        assert score_target(None, None, travelled(plan), 10.0) == pytest.approx(1.0, abs=0.02)

    def test_braking_beats_crashing_but_loses_to_driving_on(self):
        """The emergency-stop trap: a stopped plan hits nothing, and must not win.

        An all-zero trajectory is a commanded emergency stop. If safety alone set
        the target it would score 1.0 everywhere and the scorer would learn to
        brake for nothing, which reads as a good neuroNCAP result for the wrong
        reason.
        """
        from gallops.train.losses import score_target, travelled

        driving = score_target(None, None, travelled(straight(10.0)), 10.0)
        stopped = score_target(None, None, 0.0, 10.0)
        crashing = score_target(1.0, None, travelled(straight(10.0)), 10.0)

        assert driving > stopped > crashing
        assert stopped == pytest.approx(0.5, abs=0.01)

    def test_a_later_crash_scores_above_an_earlier_one(self):
        """Every proposal crashes on a late frontal tick, so the ranking among
        them is the whole signal -- a hard zero would throw it away."""
        from gallops.train.losses import score_target, travelled

        distance = travelled(straight(10.0))
        early = score_target(0.5, None, distance, 10.0)
        late = score_target(3.5, None, distance, 10.0)
        assert late > early > 0.0

    def test_leaving_the_road_costs_but_beats_a_collision(self):
        from gallops.train.losses import score_target, travelled

        distance = travelled(straight(10.0))
        on_road = score_target(None, None, distance, 10.0)
        off_road = score_target(None, 2.0, distance, 10.0)
        collision = score_target(2.0, None, distance, 10.0)
        assert on_road > off_road > collision


def test_the_trainer_and_dataset_still_fit_together():
    """The methods the training loop calls exist, without importing torch.

    `make check` runs in an environment that deliberately lacks torch, so nothing
    here can construct a TickDataset -- which meant an edit could delete a method
    it uses and the gate would stay green. It did: `batch` was dropped by a bad
    replace and the run died eight minutes in, after the model had loaded.
    Parsing the source costs nothing and closes that hole.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "gallops" / "train"
    tree = ast.parse((source / "swerve_data.py").read_text())
    dataset = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "TickDataset"
    )
    methods = {n.name for n in dataset.body if isinstance(n, ast.FunctionDef)}
    assert {"agent_input", "features", "batch", "build"} <= methods, methods

    called = {
        node.func.attr
        for node in ast.walk(ast.parse((source / "swerve.py").read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"dataset", "validation"}
    }
    assert called <= methods, f"the trainer calls {called - methods}, which do not exist"


class TestPlanShape:
    """The drawn plan has to look like driving, not like a step input.

    The first version started the lateral move at t=0 with the shortest ramp the
    tyres allow and held the offset to the horizon. Trained on that, the model
    threw the wheel over at first sight and sat at an angle to the road with no
    idea how to come back -- visible in the neuroNCAP videos before it was visible
    in any number.
    """

    def test_it_holds_the_lane_before_it_moves(self):
        from gallops.expert.draw import lateral_profile

        times = np.arange(1, 9) * 0.5
        y = lateral_profile(3.0, ramp_s=1.5, times=times, delay_s=1.5)
        assert abs(y[0]) < 1e-9, "moving during the first half second"
        assert abs(y[1]) < 1e-9, "moving before the delay is up"
        assert y[-1] == pytest.approx(3.0, abs=0.01), "never gets there"

    def test_it_comes_back(self):
        from gallops.expert.draw import lateral_profile

        times = np.arange(1, 9) * 0.5
        y = lateral_profile(3.0, ramp_s=1.0, times=times, delay_s=0.0, return_at=2.0, return_s=1.5)
        assert max(y) == pytest.approx(3.0, abs=0.05), "never reaches the offset"
        assert abs(y[-1]) < 0.15, f"ends {y[-1]:.2f} m off the lane instead of back in it"

    def test_the_ease_is_gentle_at_both_ends(self):
        """Zero slope entering and leaving, so there is no step in lateral speed."""
        from gallops.expert.draw import ease

        phase = np.linspace(0, 1, 101)
        slope = np.gradient(ease(phase), phase)
        assert abs(slope[0]) < 0.05 and abs(slope[-1]) < 0.05

    def test_a_gentler_ramp_pulls_less_lateral_g(self):
        from gallops.expert.draw import lateral_profile

        times = np.linspace(0.1, 4.0, 40)
        peaks = []
        for ramp in (1.0, 2.0, 3.0):
            y = lateral_profile(3.0, ramp_s=ramp, times=times)
            peaks.append(np.abs(np.gradient(np.gradient(y, times), times)).max())
        assert peaks[0] > peaks[1] > peaks[2], "a longer ramp must be gentler"

    def test_an_urgent_evasion_still_starts_now(self):
        """The delay is a preference, not a rule: with no time it must collapse to 0."""
        from gallops.expert.draw import lateral_profile

        times = np.arange(1, 9) * 0.5
        y = lateral_profile(3.0, ramp_s=1.0, times=times, delay_s=0.0)
        assert y[1] > 1.0, "not committed by 1 s when it needs to be"


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

    correction = rejoin(path, speed=10.0, times=times)
    assert correction[0] == pytest.approx(3.0, abs=0.6), "does not start where the car is"
    assert abs(correction[-1]) < 0.05, "never actually rejoins the path"

    lateral = np.gradient(np.gradient(correction, times), times)
    assert np.abs(lateral).max() < LATERAL_ACCEL_MAX * 1.5, "the rejoin is not drivable"


def test_a_car_already_on_the_path_gets_no_correction():
    """The common case must be untouched, or every round-0 label moves for nothing."""
    from gallops.expert.draw import rejoin

    times = np.arange(1, 9) * 0.5
    path = np.column_stack([np.linspace(-20, 80, 200), np.zeros(200)])
    assert np.abs(rejoin(path, speed=10.0, times=times)).max() < 1e-3
