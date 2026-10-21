"""The aux runs' split, cache layout and held-out measurements, on hand-built cases (numpy only)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gallops.drivejepa_aux import bev, cache, metrics, split


def straight(speed: float = 5.0) -> np.ndarray:
    """A plan straight ahead at `speed`: waypoint k at (k + 1) x 0.5 s."""
    t = 0.5 * np.arange(1, 9)
    return np.stack([speed * t, np.zeros(8), np.zeros(8)], axis=1)


# --- the split ------------------------------------------------------------------------------------


def test_draw_is_the_names_not_their_order() -> None:
    names = [f"scene-{i:04d}" for i in range(200)]
    assert split.draw(names) == split.draw(list(reversed(names)))
    assert len(split.draw(names)) == 30
    assert split.draw(names, seed=1) != split.draw(names)


def test_the_committed_split_is_its_own_rule() -> None:
    held = split.load()
    assert set(held["held_out"]) <= set(held["candidates"])
    assert len(held["held_out"]) == round(held["fraction"] * len(held["candidates"]))


def test_a_hand_edited_split_is_refused(tmp_path) -> None:
    held = json.loads(split.HELDOUT_FILE.read_text())
    held["held_out"] = held["held_out"][1:] + [n for n in held["candidates"] if n not in held["held_out"]][:1]
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(held))
    with pytest.raises(SystemExit, match="not draw"):
        split.load(path)


def test_partition_keeps_a_scene_on_one_side() -> None:
    scenes = np.array(["a", "a", "b", "c", "c", "c"])
    train, held = split.partition(scenes, ["c"])
    assert train.tolist() == [0, 1, 2] and held.tolist() == [3, 4, 5]


# --- the cache's layout ---------------------------------------------------------------------------


def record(boxes_per_step: list, road_cell=(4, 128)) -> dict:
    road = np.zeros((bev.ROWS, bev.COLS), bool)
    road[road_cell] = True
    return {
        "bev": np.full((bev.ROWS, bev.COLS), bev.IGNORE, np.uint8),
        "road": road,
        "agent_states": np.zeros((30, 5), np.float32),
        "agent_valid": np.zeros(30, bool),
        "lead_present": 0,
        "lead_gap": math.nan,
        "lead_closing": math.nan,
        "lane_change": -1,
        "expert_judged": 8,
        "expert_on_road": 8,
        "future_boxes": [np.asarray(b, np.float32).reshape(-1, 5) for b in boxes_per_step],
    }


def test_future_boxes_and_road_come_back_out_per_row() -> None:
    box = [10.0, 0.0, 0.0, 4.0, 2.0]
    first = record([[box]] + [[]] * 7)
    second = record([[]] * 7 + [[box, box]], road_cell=(100, 3))
    ticks = [{"sample": s, "scene": "scene-0001", "keyframe": k} for k, s in enumerate(["s0", "s1"])]
    labels = cache.stack_records([first, second], ticks)
    steps = cache.future_boxes(labels, 0)
    assert [len(s) for s in steps] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert [len(s) for s in cache.future_boxes(labels, 1)] == [0] * 7 + [2]
    np.testing.assert_array_equal(steps[0][0], box)
    roads = cache.road(labels, [0, 1])
    assert roads.shape == (2, bev.ROWS, bev.COLS)
    assert roads[0, 4, 128] and roads[1, 100, 3] and roads.sum() == 2


# --- the plan -------------------------------------------------------------------------------------


def test_l2_and_min_ade_read_the_right_waypoints() -> None:
    expert = straight()[None]
    plan = expert.copy()
    plan[0, :, 1] = 1.0  # one metre to the left the whole way
    assert metrics.l2(plan, expert)[0, list(metrics.L2_WAYPOINTS.values())].tolist() == [1.0, 1.0, 1.0]
    proposals = np.stack([plan[0], expert[0]])[None]
    assert metrics.min_ade(proposals, expert)[0] == 0.0


def test_a_box_in_the_plan_at_that_moment_is_a_collision() -> None:
    plan = straight(5.0)
    # waypoint 3 is 10 m ahead at 2 s; a car sitting there at 2 s is hit, the same car at 1 s is not
    car = np.array([[10.0, 0.0, 0.0, 4.5, 1.9]])
    at = lambda k: [car if j == k else np.zeros((0, 5)) for j in range(8)]  # noqa: E731
    assert metrics.collides(plan, at(3))
    assert not metrics.collides(plan, at(1))
    # beside the path, its edge clear of the ego's by 10 cm
    beside = car.copy()
    beside[0, 1] = metrics.EGO_WIDTH / 2 + 1.9 / 2 + 0.1
    assert not metrics.collides(plan, [beside if j == 3 else np.zeros((0, 5)) for j in range(8)])
    # past the 3 s horizon it is not counted
    assert not metrics.collides(plan, at(7))


def test_the_footprint_sits_ahead_of_the_rear_axle() -> None:
    footprint = metrics.ego_footprint(np.array([0.0, 0.0, 0.0]))
    assert footprint[:, 0].max() == pytest.approx(metrics.REAR_AXLE_TO_CENTRE + metrics.EGO_LENGTH / 2)
    assert footprint[:, 0].min() == pytest.approx(metrics.REAR_AXLE_TO_CENTRE - metrics.EGO_LENGTH / 2)


def test_off_road_judges_only_waypoints_on_the_grid() -> None:
    road = np.zeros((1, bev.ROWS, bev.COLS), bool)
    plan = straight(2.0)[None]  # 1 m to 6 m ahead over 3 s
    rows, cols = bev.xy_to_cell(plan[0, :6, 0], plan[0, :6, 1])
    road[0, rows, cols] = True
    assert metrics.off_road(plan, road)[0] == 0.0
    road[0, rows[4], cols[4]] = False
    assert metrics.off_road(plan, road)[0] == 1.0
    behind = plan.copy()
    behind[..., 0] *= -1  # every waypoint behind the grid
    assert np.isnan(metrics.off_road(behind, road)[0])


def test_on_road_counts_only_what_the_camera_sees() -> None:
    path = straight(2.0)
    road = np.zeros((bev.ROWS, bev.COLS), bool)
    view = np.zeros((bev.ROWS, bev.COLS), bool)
    rows, cols = bev.xy_to_cell(path[:, 0], path[:, 1])
    view[rows[:4], cols[:4]] = True
    road[rows[:3], cols[:3]] = True
    assert metrics.on_road(path, road, view) == (4, 3)


# --- uncertainty ----------------------------------------------------------------------------------


def test_one_row_per_cluster_is_the_ordinary_standard_error() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    mean, se, g = metrics.clustered(values, np.arange(4))
    assert mean == pytest.approx(3.75) and g == 4
    assert se == pytest.approx(values.std(ddof=1) / 2)


def test_copies_within_a_cluster_add_no_information() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    once = metrics.clustered(values, np.arange(4))
    thrice = metrics.clustered(np.repeat(values, 3), np.repeat(np.arange(4), 3))
    assert thrice[0] == pytest.approx(once[0]) and thrice[1] == pytest.approx(once[1])


def test_clustered_leaves_out_nan_and_paired_differences() -> None:
    mean, _, g = metrics.clustered([1.0, np.nan, 3.0], ["a", "b", "c"])
    assert mean == 2.0 and g == 2
    assert metrics.paired([3.0, 5.0], [1.0, 2.0], ["a", "b"])[0] == pytest.approx(2.5)


# --- the heads ------------------------------------------------------------------------------------


def test_bev_iou_skips_ignored_cells() -> None:
    label = np.full((1, 2, 2), bev.IGNORE, np.uint8)
    label[0, 0] = [1, 2]
    predicted = np.array([[[1, 1], [1, 1]]], np.uint8)  # right on the one labelled road cell, wrong elsewhere
    counts = metrics.bev_counts(predicted, label, classes=3)
    iou = metrics.iou(counts)
    assert iou[1] == pytest.approx(0.5) and iou[2] == 0.0 and np.isnan(iou[0])


def test_prior_bev_is_each_cells_majority() -> None:
    labels = np.array([[[1, bev.IGNORE]], [[1, 2]], [[3, 2]]], np.uint8)
    assert metrics.prior_bev(labels, classes=4).tolist() == [[1, 2]]


def test_box_matches_take_the_nearest_label_once() -> None:
    truth = np.array([[10.0, 0.0, 0, 4, 2], [20.0, 0.0, 0, 4, 2], [0, 0, 0, 0, 0]])
    valid = np.array([True, True, False])
    states = np.array([[10.5, 0, 0, 4, 2], [10.2, 0, 0, 4, 2], [40.0, 0, 0, 4, 2], [20.0, 0, 0, 4, 2]], float)
    logits = np.array([3.0, 2.0, 1.0, -1.0])  # the last is not confident
    tp, fp, fn, errors = metrics.box_matches(states, logits, truth, valid)
    assert (tp, fp, fn) == (1, 2, 1)
    assert errors == [pytest.approx(0.5)]  # the more confident box took the label
    assert metrics.f1(tp, fp, fn) == pytest.approx(2 / 5)


def test_lane_f1_scores_changes_and_skips_unknowns() -> None:
    label = np.array([0, 1, 1, 2, -1])
    predicted = np.array([0, 1, 0, 1, 1])
    scores = metrics.lane_f1(predicted, label)
    assert scores["left"] == pytest.approx(2 / 4) and scores["right"] == 0.0
    assert np.isnan(metrics.lane_f1(np.zeros(3, int), np.zeros(3, int))["left"])
