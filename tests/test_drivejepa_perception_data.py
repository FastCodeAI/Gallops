"""H15's NAVSIM label rules and splits, on synthetic windows (numpy only; no navsim needed).

The expected values are computed with an explicit rotation matrix, independently of the module's
SE2 helpers, so a label that copied frame k's coordinates instead of moving them through global,
or a rotation with the wrong sign, fails here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gallops.drivejepa_perception import navsim_data as nd


def _local(global_xy, pose):
    """Global (x, y) into the ego frame of global pose (x, y, yaw), by an explicit matrix."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])  # ego -> global
    return rotation.T @ (np.asarray(global_xy, dtype=float) - np.asarray(pose[:2], dtype=float))


#: The ego drives 5 m per 0.5 s and turns 0.1 rad per frame: rotation is exercised everywhere.
POSES = np.array([[5.0 * k, 0.3 * k, 0.1 * k] for k in range(nd.HISTORY + nd.FUTURE)])


def _parked(k):
    return np.array([25.0, 2.0])


def _moving(k):
    return np.array([20.0 + 3.0 * k, -1.0 + 0.5 * k])


def _frame(k, agents):
    """One raw log frame: each agent as (name, track, global_xy_fn, size), seen from frame k."""
    boxes, names, tracks, velocity = [], [], [], []
    for name, track, where, size in agents:
        x, y = _local(where(k), POSES[k])
        boxes.append([x, y, 0.5, *size, 0.0])
        names.append(name)
        tracks.append(track)
        velocity.append([1.0, 0.0, 0.0])
    return {
        "anns": {
            "gt_boxes": np.array(boxes).reshape(-1, 7),
            "gt_names": names,
            "track_tokens": tracks,
            "gt_velocity_3d": np.array(velocity).reshape(-1, 3),
        },
        "ego_dynamic_state": [10.0 - k, 0.0, 0.0, 0.0],
    }


def _window(drop_moving_after=None):
    frames = []
    for k in range(nd.HISTORY + nd.FUTURE):
        agents = [
            ("vehicle", "parked", _parked, (4.0, 2.0, 1.5)),
            ("generic_object", "far", lambda k: np.array([500.0, 0.0]), (1.0, 1.0, 1.0)),
            ("animal", "unknown-name", _parked, (1.0, 1.0, 1.0)),
        ]
        if drop_moving_after is None or k <= drop_moving_after:
            agents.append(("pedestrian", "moving", _moving, (0.6, 0.6, 1.8)))
        frames.append(_frame(k, agents))
    return frames


def test_se2_round_trip():
    pose = np.array([3.0, -2.0, 0.7])
    points = np.array([[1.0, 2.0], [-4.0, 0.5]])
    assert np.allclose(nd.to_local(nd.to_global(points, pose), pose), points)
    assert np.allclose(nd.to_local(np.array([[10.0, 7.0]]), np.array([10.0, 5.0, np.pi / 2])), [[2.0, 0.0]])


def test_labels_move_later_frames_through_global_into_frame_t():
    labels = nd.window_labels(_window(), POSES)
    t = nd.CURRENT
    # far (500 m) and unknown names are dropped; parked and moving are kept, in frame order.
    assert list(labels["cls"]) == [0, 1]
    assert np.allclose(labels["box"][0, :2], _local(_parked(t), POSES[t]), atol=1e-5)
    for j, step in enumerate(nd.PATH_STEPS):
        assert labels["path_ok"][:, j].all()
        assert np.allclose(labels["path"][0, j], _local(_parked(t + step), POSES[t]), atol=1e-5)
        assert np.allclose(labels["path"][1, j], _local(_moving(t + step), POSES[t]), atol=1e-5)
    # A parked car's path stays where it stands; frame t+1's own coordinates would not.
    assert np.allclose(labels["path"][0, 1], labels["box"][0, :2], atol=1e-5)
    assert np.allclose(labels["size_change"], 0.0)


def test_a_track_that_ends_early_is_masked_not_extrapolated():
    labels = nd.window_labels(_window(drop_moving_after=nd.CURRENT + 1), POSES)
    assert list(labels["path_ok"][1]) == [True, False]
    assert np.isnan(labels["path"][1, 1]).all()


def test_ego_future_and_speed():
    labels = nd.window_labels(_window(), POSES)
    t = nd.CURRENT
    for i in range(nd.EGO_FUTURE):
        k = t + 1 + i
        assert np.allclose(labels["ego_future"][i, :2], _local(POSES[k, :2], POSES[t]), atol=1e-5)
        assert np.isclose(labels["ego_future"][i, 2], POSES[k, 2] - POSES[t, 2], atol=1e-6)
    assert np.allclose(labels["ego_speed"], [10.0 - (t + k) for k in range(nd.EGO_SPEED_STEPS)])


def test_pack_round_trip():
    a = nd.window_labels(_window(), POSES)
    b = nd.window_labels(_window(drop_moving_after=nd.CURRENT), POSES)
    packed = nd.pack([a, b])
    assert list(packed["count"]) == [2, 2]
    for row, labels in enumerate((a, b)):
        got = nd.agents_of(packed, row)
        for key in nd.AGENT_KEYS:
            assert np.array_equal(got[key], labels[key], equal_nan=key in ("path", "size_change"))


def test_track_ids_are_stable_and_distinct():
    assert nd.track_id("abc") == nd.track_id("abc")
    assert nd.track_id("abc") != nd.track_id("abd")
    assert 0 <= nd.track_id("abc") < 2**63


LOGS = [f"log-{i:04d}" for i in range(200)]


def test_splits_are_disjoint_and_drawn_by_rule(tmp_path):
    splits = nd.draw_splits(LOGS, LOGS[:51])
    sets = [set(splits[k]) for k in ("select", "report", "train")]
    assert [len(s) for s in sets] == [51, nd.REPORT_N, 200 - 51 - nd.REPORT_N]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert nd.draw_splits(LOGS, LOGS[:51]) == splits
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    assert nd.load_splits(path, LOGS)["report"] == splits["report"]
    edited = dict(splits, report=splits["report"][:-1] + [splits["train"][0]])
    path.write_text(json.dumps(edited))
    with pytest.raises(SystemExit):
        nd.load_splits(path, LOGS)


def test_select_logs_must_be_navtrain_logs():
    with pytest.raises(ValueError):
        nd.draw_splits(LOGS, ["not-a-navtrain-log"])
