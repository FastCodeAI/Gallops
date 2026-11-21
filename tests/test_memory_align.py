"""gallops.memory.align: moving the previous moment's paths into the current frame, and matching them."""

from __future__ import annotations

import numpy as np

from gallops.memory import align


def _pose(x, y, yaw):
    out = np.eye(4)
    out[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    out[:2, 3] = [x, y]
    return out


def _paths(rng):
    """32 paths of 8 waypoints (x, y, heading), 0.5 s apart, fanning out ahead of the car."""
    t = np.arange(1, 9) * 0.5
    speed = rng.uniform(2, 15, 32)[:, None]
    bend = rng.uniform(-0.3, 0.3, 32)[:, None]
    x, y = speed * t, bend * (speed * t) ** 2 / 10
    return np.stack([x, y, np.zeros_like(x)], -1)


def test_identity_when_the_car_does_not_move():
    rng = np.random.default_rng(0)
    old = _paths(rng)
    new = np.concatenate([old[:, 1:], old[:, -1:]], 1)  # the same plans, one step on
    pose = _pose(664885.0, 3999644.0, 0.7)
    d = align.distances(align.align_previous(old, pose, pose), new)
    assert np.allclose(np.diag(d), 0)
    m, m_d = align.match(d)
    assert m.tolist() == list(range(32)) and np.allclose(m_d, 0) and m.dtype == np.int16


def test_alignment_follows_the_car_when_it_moves_and_turns():
    rng = np.random.default_rng(1)
    old = _paths(rng)
    # the car drives 3 m forward and turns 0.2 rad; express the old plans in the new frame by hand
    before, after = _pose(100.0, 50.0, 0.3), _pose(100.0 + 3 * np.cos(0.3), 50.0 + 3 * np.sin(0.3), 0.5)
    world = (before[:2, :2] @ old[:, 1:, :2].reshape(-1, 2).T).T + before[:2, 3]
    local = (after[:2, :2].T @ (world - after[:2, 3]).T).T.reshape(32, 7, 2)
    assert np.allclose(align.align_previous(old, before, after), local, atol=1e-9)
    new = np.zeros((32, 8, 3))
    new[:, :7, :2] = local
    m, m_d = align.match(align.distances(align.align_previous(old, before, after), new))
    assert m.tolist() == list(range(32)) and np.allclose(m_d, 0, atol=1e-9)


def test_a_far_path_gets_minus_one_and_the_rest_still_match():
    rng = np.random.default_rng(2)
    old = _paths(rng)
    new = np.concatenate([old[:, 1:], old[:, -1:]], 1)
    new[5, :, 1] += 30.0  # a new path far from every old one
    pose = _pose(0.0, 0.0, 0.0)
    m, m_d = align.match(align.distances(align.align_previous(old, pose, pose), new))
    assert m[5] == -1 and m_d[5] > align.LIMIT
    assert [int(v) for k, v in enumerate(m) if k != 5] == [k for k in range(32) if k != 5]


def test_match_takes_the_closest_old_path_for_each_new_path():
    d = np.array([[0.5, 3.0, 2.5], [0.2, 0.9, 2.1], [4.0, 0.1, 2.2]])  # old x new
    m, m_d = align.match(d, limit=2.0)
    assert m.tolist() == [1, 2, -1]  # new 0 <- old 1, new 1 <- old 2, new 2: nothing within 2 m
    assert np.allclose(m_d, [0.2, 0.1, 2.1])


def test_previous_rows_follow_chains_not_table_order():
    table = {
        "log": np.array(["a", "a", "a", "b", "a"]),
        "chain": np.array([0, 0, 1, 0, 0]),
        "chain_pos": np.array([1, 2, 1, 2, 3]),
    }
    assert align.previous_rows(table).tolist() == [-1, 0, -1, -1, 1]


def test_pose_from_quaternion_matches_the_matrix():
    yaw = 0.4
    frame = {
        "ego2global_rotation": [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
        "ego2global_translation": [1.0, 2.0, 0.5],
    }
    pose = align.pose_of(frame)
    expected = _pose(1.0, 2.0, yaw)
    expected[2, 3] = 0.5
    assert np.allclose(pose, expected)
    assert np.allclose(align.pose_of({"ego2global": expected}), expected)
