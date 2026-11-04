"""H15's own nuScenes reader (nuscenes_data): frames, timing, the scene draw, collisions. numpy only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gallops.data import nuscenes_real as nr
from gallops.drivejepa_perception import nuscenes_data as nd


def _ego(x, y, yaw):
    m = np.eye(4)
    m[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    m[:2, 3] = [x, y]
    return m


def _quat_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def test_boxes_are_in_the_egos_frame_with_velocity_in_its_axes():
    ego = _ego(100.0, 50.0, math.pi / 2)  # the ego faces +y in the world
    agent = {
        "translation": [100.0, 60.0, 1.0],  # 10 m along the ego's heading
        "size_wlh": [2.0, 4.5, 1.6],
        "rotation": _quat_yaw(math.pi / 2),  # parallel to the ego
        "velocity": [0.0, 3.0, 0.0],  # moving the ego's way
    }
    b = nd.boxes([agent], ego)
    assert np.allclose(b["xyz"][0], [10.0, 0.0, 1.0])
    assert (
        abs(b["yaw"][0]) < 1e-9 and b["length"][0] == 4.5 and b["width"][0] == 2.0 and b["height"][0] == 1.6
    )
    assert np.allclose(b["vel"][0], [3.0, 0.0])
    assert nd.boxes([], ego)["xyz"].shape == (0, 3)


def test_velocity_is_a_central_difference_given_up_over_long_gaps():
    a = {"token": "a", "prev": "p", "next": "n", "sample_token": "s1", "translation": [1.0, 0.0, 0.0]}
    p = {"token": "p", "prev": "", "next": "a", "sample_token": "s0", "translation": [0.0, 0.0, 0.0]}
    n = {"token": "n", "prev": "a", "next": "", "sample_token": "s2", "translation": [2.0, 0.0, 0.0]}
    times = {"s0": 0, "s1": 500_000, "s2": 1_000_000}
    assert np.allclose(nd._velocity(a, {"p": p, "n": n}, times), [2.0, 0.0, 0.0])
    slow = {"s0": 0, "s1": 2_000_000, "s2": 4_000_000}  # 4 s apart: more than twice 1.5 s
    assert all(math.isnan(v) for v in nd._velocity(a, {"p": p, "n": n}, slow))


def test_future_boxes_are_read_in_the_current_keyframes_frame():
    frames = [
        {"sample": f"s{k}", "ego": _ego(5.0 * k, 0.0, 0.0).tolist()} for k in range(nr.FUTURE_STEPS + 1)
    ]
    car = {
        "translation": [20.0, 3.0, 0.5],
        "size_wlh": [2.0, 4.0, 1.5],
        "rotation": _quat_yaw(0.0),
        "velocity": [0, 0, 0],
    }
    annotations = {f"s{k}": [car] for k in range(nr.FUTURE_STEPS + 1)}
    out = nd.future_boxes({"keyframes": frames}, 0, annotations)
    assert len(out) == nr.FUTURE_STEPS and np.allclose(out[0][0], [20.0, 3.0, 0.0, 4.0, 2.0])


def test_ticks_keep_keyframes_with_history_and_four_seconds_of_future(tmp_path, monkeypatch):
    monkeypatch.setattr(nr, "command", lambda future_xy: 2)  # paddock's rule; not under test here
    n = nr.FUTURE_STEPS + 6  # keyframes 1-5 have 4 s of future
    stamps = [k * nr.HISTORY_DT_US for k in range(n)]
    stamps[2] += 3 * nr.HISTORY_TOLERANCE_US  # breaks two gaps: keyframe 2's past, and keyframe 3's
    frames = [
        {"sample": f"s{k}", "timestamp": t, "ego": _ego(2.0 * k, 0.0, 0.0).tolist()}
        for k, t in enumerate(stamps)
    ]
    can = [{"utime": t, "vel": [4.0, 0.0, 0.0], "accel": [0.1, 0.0, 0.0]} for t in stamps]
    (tmp_path / "scene-0001_pose.json").write_text(json.dumps(can))
    rows = nd.ticks("scene-0001", {"keyframes": frames}, tmp_path)
    kept = [r["keyframe"] for r in rows]
    # 0 has no past, 2 and 3 have a past 0.15 s off, 6 and later lack 4 s of future
    assert kept == [1, 4, 5]
    first = rows[0]
    assert first["speed"] == 4.0 and first["command"] == 2 and len(first["expert"]) == nr.FUTURE_STEPS
    assert np.allclose(first["expert"][0][:2], [2.0, 0.0])
    assert nd.ticks("scene-0002", {"keyframes": frames}, tmp_path) == []  # no CAN bus, no ticks


def test_the_scene_draw_is_reproducible_and_an_edited_list_is_refused(tmp_path):
    candidates = [f"scene-{i:04d}" for i in range(50)]
    path = tmp_path / "scenes.json"
    record = nd.write_scenes(path, candidates, n=10, seed=3)
    assert nd.load_scenes(path) == record["scenes"] == nd.draw_scenes(list(reversed(candidates)), 10, 3)
    edited = json.loads(path.read_text())
    edited["scenes"][0] = "scene-9999"
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError):
        nd.load_scenes(path)


def test_candidates_need_can_bus_poses_and_leave_out_scenes_by_number(tmp_path):
    for name in ("scene-0001", "scene-0061", "scene-0103"):
        (tmp_path / f"{name}_pose.json").write_text("[]")
    train = ["scene-0001", "scene-0002", "scene-0061", "scene-0103"]  # 0002 has no CAN-bus poses
    readable = nd.readable_train_scenes(train, tmp_path)
    assert readable == ["scene-0001", "scene-0061", "scene-0103"]
    # neuroNCAP's and the reconstructed scenes are kept out by their number, as nuscenes_real names them
    assert nd.candidate_scenes(readable, {"0061", "0103"}) == ["scene-0001"]


def test_the_draw_keeps_its_sources_and_still_redraws_exactly(tmp_path):
    candidates = [f"scene-{i:04d}" for i in range(20)]
    sources = {"devkit": "1.1.11", "kept_out": ["0061"], "paddock_commit": "abc"}
    path = tmp_path / "scenes.json"
    record = nd.write_scenes(path, candidates, n=5, seed=0, sources=sources)
    assert json.loads(path.read_text())["sources"] == sources
    assert nd.load_scenes(path) == record["scenes"] == nd.draw_scenes(candidates, 5, 0)


def test_a_box_in_the_egos_path_is_a_collision_and_one_beside_it_is_not():
    plan = np.zeros((8, 3))
    plan[:, 0] = np.arange(1, 9) * 2.0  # straight ahead, 2 m a step
    ahead = [np.array([[6.0, 0.0, 0.0, 4.5, 2.0]])] * 8
    beside = [np.array([[6.0, 6.0, 0.0, 4.5, 2.0]])] * 8
    assert nd.collides(plan, ahead) and not nd.collides(plan, beside)
    assert nd.collided(np.stack([plan, plan]), [ahead, beside]).tolist() == [True, False]
