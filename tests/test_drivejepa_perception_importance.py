"""H15's derived labels (importance.py): line of sight, the ego's corridor, causal, pedestrian groups."""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception import importance as im
from gallops.drivejepa_perception import targets as tg


def _projection():
    """A front camera 1.5 m up looking along x (f = 400 and principal point (384, 216) on the vendor's
    padded frame), composed with TO_512 as the trainer composes it."""
    to_cam = np.array([[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    k = np.array([[400, 0, 384, 0], [0, 400, 216, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    l2i = np.zeros((4, 4, 4))
    l2i[tg.FRONT_SLOT] = k @ to_cam
    return tg.front_projection(l2i)


def _agents(boxes, cls, path=None, path_ok=None):
    boxes = np.asarray(boxes, dtype=np.float32)
    n = len(boxes)
    return {
        "box": boxes,
        "cls": np.asarray(cls, dtype=np.int8),
        "velocity": np.zeros((n, 2), np.float32),
        "path": np.zeros((n, 2, 2), np.float32) if path is None else np.asarray(path, np.float32),
        "path_ok": np.zeros((n, 2), bool) if path_ok is None else np.asarray(path_ok),
    }


def test_a_car_behind_a_nearer_one_is_hidden_and_one_beside_it_is_not():
    boxes = np.array(
        [
            [20.0, 0.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # far, straight ahead
            [10.0, 0.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # near, in front of it
            [20.0, 8.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # far, off to the left
        ]
    )
    seen = im.visibility(_projection(), boxes)
    assert seen[1] == 1.0 and seen[2] == 1.0
    assert seen[0] < 0.2  # the nearer car's rectangle covers most of it


def test_distance_to_the_egos_path_is_to_the_nearest_segment():
    future = np.array([[5.0 * k, 0.0, 0.0] for k in range(1, 5)])  # straight ahead to 20 m
    d = im.path_distance(future, np.array([[10.0, 1.0], [10.0, -3.0], [30.0, 0.0], [-2.0, 0.0]]))
    assert np.allclose(d, [1.0, 3.0, 10.0, 2.0])


def test_an_agent_is_in_path_now_or_when_its_known_path_enters_the_corridor():
    future = np.array([[5.0 * k, 0.0, 0.0] for k in range(1, 9)])
    agents = _agents(
        [
            [15.0, 0.5, 0.9, 4.5, 2.0, 1.8, 0.0],  # in the corridor now
            [15.0, 6.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # beside it, and stays there
            [15.0, 6.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # beside it, cutting in by +1.0 s
            [15.0, 6.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # would cut in, but that step is unknown
        ],
        [0, 0, 0, 0],
        path=[[[15, 0.5], [15, 0.5]], [[15, 6], [15, 6]], [[15, 4], [15, 1]], [[15, 4], [15, 1]]],
        path_ok=[[True, True], [True, True], [True, True], [True, False]],
    )
    assert im.in_path(agents, future, ego_half_width=1.15).tolist() == [True, False, True, False]


def test_pedestrians_chain_into_groups_named_after_their_nearest_member():
    xy = np.array([[10.8, 0.0], [10.0, 0.0], [10.4, 0.0], [20.0, 0.0], [10.2, 0.0]])
    cls = np.array([1, 1, 1, 1, 0])  # the last is a vehicle
    g = im.groups(xy, cls, radius=0.5)
    assert g.tolist() == [1, 1, 1, 3, -1]  # 10.0-10.4-10.8 chain; the one at 20 m is alone


def test_target_labels_follow_the_targets_order():
    agents = _agents(
        [
            [30.0, 0.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # a vehicle 30 m ahead, in the ego's path
            [8.0, -3.0, 0.9, 0.6, 0.6, 1.8, 0.0],  # a pedestrian near the ego
            [25.0, 9.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # a vehicle far off the path
        ],
        [0, 1, 0],
    )
    future = np.array([[5.0 * k, 0.0, 0.0] for k in range(1, 9)])
    labels = im.target_importance(agents, _projection(), future, ego_half_width=1.15)
    order = tg.target_index(agents, _projection())
    assert order.tolist() == [1, 2, 0]  # nearest first, as targets.build lays them out
    assert labels["causal"].tolist() == [True, False, True]  # near; far and off the path; in the path
    assert labels["rank"].tolist() == [0, 1, 2]


def test_a_pedestrian_walking_towards_the_path_is_in_it_within_the_horizon():
    future = np.array([[5.0 * k, 0.0, 0.0] for k in range(1, 9)])
    agents = _agents([[12.0, 4.0, 0.9, 0.6, 0.6, 1.8, 0.0]], [1])
    agents["velocity"] = np.array([[0.0, -1.5]], np.float32)  # towards the ego's path, 1.5 m/s
    agents["path"] = np.array([[[12.0, 3.25], [12.0, 2.5]]], np.float32)
    agents["path_ok"] = np.ones((1, 2), bool)
    # The labelled second ends 2.5 m from the path, outside 1.15 + 0.3 + 0.5; at 1.5 m/s it is in by 2 s.
    assert im.in_path(agents, future, ego_half_width=1.15).tolist() == [False]
    assert im.in_path(agents, future, ego_half_width=1.15, horizon_s=4.0).tolist() == [True]


def test_the_chosen_rule_widens_the_corridor_to_the_adjacent_lane():
    future = np.array([[5.0 * k, 0.0, 0.0] for k in range(1, 9)])
    agents = _agents([[15.0, 3.6, 0.9, 4.5, 2.0, 1.8, 0.0]], [0])  # a car in the next lane, 3.6 m left
    # 1.15 + 1.0 + 0.5 = 2.65 m reach: outside. With the chosen 2 m margin, 4.15 m: inside.
    assert im.in_path(agents, future, ego_half_width=1.15).tolist() == [False]
    assert im.in_path(agents, future, ego_half_width=1.15, margin_m=2.0).tolist() == [True]


def test_target_importance_gives_causal_under_each_named_rule():
    future = np.array([[5.0 * k, 0.0, 0.0] for k in range(1, 9)])
    agents = _agents(
        [
            [25.0, 3.6, 0.9, 4.5, 2.0, 1.8, 0.0],  # the next lane, 25 m ahead: causal only when chosen
            [30.0, 0.0, 0.9, 4.5, 2.0, 1.8, 0.0],  # in the path: causal under both
        ],
        [0, 0],
    )
    labels = im.target_importance(agents, _projection(), future, ego_half_width=1.15)
    assert labels["causal"].tolist() == labels["causal_strict"].tolist() == [False, True]
    assert labels["causal_chosen"].tolist() == [True, True]
    assert set(im.CAUSAL_RULES) == {"strict", "chosen"}


def test_a_chain_wider_than_the_cap_is_split_nearest_first():
    # Five pedestrians 0.9 m apart in a line: one chain at 1.0 m, 3.6 m wide.
    xy = np.array([[10.0 + 0.9 * k, 0.0] for k in range(5)])
    cls = np.ones(5, int)
    assert im.groups(xy, cls, radius=1.0).tolist() == [0, 0, 0, 0, 0]
    # Capped at 3 m: the first four (2.7 m) stay together, the fifth starts its own group.
    assert im.groups(xy, cls, radius=1.0, max_width=3.0).tolist() == [0, 0, 0, 0, 4]
    # A chain inside the cap is left whole.
    assert im.groups(xy[:3], cls[:3], radius=1.0, max_width=3.0).tolist() == [0, 0, 0]
