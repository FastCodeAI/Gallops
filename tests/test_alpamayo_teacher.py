"""alpamayo_teacher: the thin reshape of draw.teach() to Alpamayo's contract.

Does not re-test the solver itself (collision geometry, swerve drawing,
drivability) -- that is `test_expert.py`'s job, against `draw.py`/`grader.py`
directly. This only checks the reshape, the flip-flag, and the label record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gallops.expert import alpamayo_teacher as at
from gallops.expert import draw


def straight(speed: float, poses: int = 8, dt: float = 0.5) -> np.ndarray:
    t = np.arange(1, poses + 1) * dt
    return np.column_stack([speed * t, np.zeros_like(t), np.zeros_like(t)])


def actor_ahead(distance: float, t0: int = 0, width: float = 1.9, length: float = 4.5) -> dict:
    pose = np.eye(4)
    pose[0, 3] = distance
    return {
        "timestamps": [t0 - 10e6, t0 + 10e6],
        "poses": [pose.tolist(), pose.tolist()],
        "dims": [width, length, 1.6],
        "is_target_actor": True,
    }


def reference(length_m: float = 200.0, spacing_m: float = 5.0) -> np.ndarray:
    xs = np.arange(0.0, length_m, spacing_m)
    return np.column_stack([xs, np.zeros_like(xs)])


class TestTruncation:
    def test_truncates_eight_poses_to_six(self):
        plan, _ = at.teach_alpamayo(np.eye(4), 0, 10.0, reference(), [])
        assert plan.waypoints.shape == (at.ALPAMAYO_POSES, 3)

    def test_truncated_waypoints_match_the_first_six_of_the_untruncated_plan(self):
        full = draw.teach(np.eye(4), 0, 10.0, reference(), [])
        plan, _ = at.teach_alpamayo(np.eye(4), 0, 10.0, reference(), [])
        np.testing.assert_allclose(plan.waypoints, np.asarray(full.waypoints)[:6])

    def test_rejects_a_draw_poses_shrink_rather_than_silently_truncate_further(self, monkeypatch):
        """Regression guard: if draw.POSES ever drops below ALPAMAYO_POSES, this
        must raise, not hand back a short/garbled trajectory."""
        bad = draw.Plan(
            waypoints=np.zeros((4, 3)),
            verdict=draw.grade(straight(1.0, poses=4), np.eye(4), 0, []),
            kind="continue",
            offset_m=0.0,
            progress_m=0.0,
        )
        with pytest.raises(ValueError, match="fewer than"):
            at._truncate(bad)


class TestTeachAlpamayo:
    def test_clear_road_continues(self):
        plan, flipped = at.teach_alpamayo(np.eye(4), 0, 10.0, reference(), [])
        assert plan.kind == "continue"
        assert plan.clear
        assert not flipped

    def test_a_stationary_obstacle_ahead_is_handled(self):
        plan, _ = at.teach_alpamayo(np.eye(4), 0, 10.0, reference(), [actor_ahead(15.0)])
        assert plan.kind in ("brake", "swerve", "swerve+brake", "least-impact")
        assert plan.clear or plan.kind == "least-impact"


class TestTeacherState:
    def test_flags_a_genuine_direction_flip(self):
        state = at.TeacherState()
        state.last_kind, state.last_offset_sign = "swerve", 1.0
        plan = at.AlpamayoPlan(
            waypoints=np.zeros((6, 3)),
            verdict=draw.grade(straight(1.0), np.eye(4), 0, []),
            kind="swerve",
            offset_m=-2.0,
            progress_m=1.0,
            drivable=True,
        )
        assert state.update(plan) is True

    def test_does_not_flag_the_first_swerve_with_nothing_to_compare(self):
        state = at.TeacherState()
        plan = at.AlpamayoPlan(
            waypoints=np.zeros((6, 3)),
            verdict=draw.grade(straight(1.0), np.eye(4), 0, []),
            kind="swerve",
            offset_m=-2.0,
            progress_m=1.0,
            drivable=True,
        )
        assert state.update(plan) is False

    def test_does_not_flag_a_brake_between_two_same_side_swerves(self):
        """A brake tick (offset_m == 0) must not reset the remembered side."""
        state = at.TeacherState()
        state.last_kind, state.last_offset_sign = "swerve", 1.0
        brake = at.AlpamayoPlan(
            waypoints=np.zeros((6, 3)),
            verdict=draw.grade(straight(1.0), np.eye(4), 0, []),
            kind="brake",
            offset_m=0.0,
            progress_m=1.0,
            drivable=True,
        )
        assert state.update(brake) is False
        same_side = at.AlpamayoPlan(
            waypoints=np.zeros((6, 3)),
            verdict=draw.grade(straight(1.0), np.eye(4), 0, []),
            kind="swerve",
            offset_m=1.5,
            progress_m=1.0,
            drivable=True,
        )
        assert state.update(same_side) is False


class TestLabelling:
    def test_label_alpamayo_tick_records_frames_by_camera(self):
        record = at.label_alpamayo_tick(
            episode="side-0103/run_0",
            timestamp=0,
            ego2world=np.eye(4),
            speed=10.0,
            reference_world=reference(),
            targets=[actor_ahead(15.0)],
            frames={
                "CAM_FRONT": "/runs/.../CAM_FRONT/0.jpg",
                "CAM_FRONT_ZOOM": "/runs/.../CAM_FRONT_ZOOM/0.jpg",
            },
        )
        assert record["episode"] == "side-0103/run_0"
        assert record["frames"]["CAM_FRONT_ZOOM"].endswith("0.jpg")
        assert len(record["expert"]) == at.ALPAMAYO_POSES
        assert record["direction_flipped"] is False

    def test_write_episode_labels_is_one_jsonl_record_per_tick(self, tmp_path: Path):
        records = [
            at.label_alpamayo_tick(
                episode="side-0103/run_0",
                timestamp=t,
                ego2world=np.eye(4),
                speed=10.0,
                reference_world=reference(),
                targets=[],
                frames=None,
            )
            for t in range(3)
        ]
        out = tmp_path / "side-0103__run_0.jsonl"
        at.write_episode_labels(records, out)
        lines = out.read_text().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["timestamp"] for line in lines] == [0, 1, 2]
