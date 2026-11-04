"""The auxiliary labels' grid and rules, on hand-built scenes (numpy only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gallops.drivejepa_aux import bev, targets


def test_grid_matches_upstreams_orientation() -> None:
    # upstream: OpenCV pixel (col = x / 0.25, row = y / 0.25 + 128) on a (256, 128) image, then
    # rot90 and a flip -- the transpose. A point 10 m ahead, 5 m left: row 40, column 148.
    upstream = np.zeros((256, 128), dtype=np.uint8)
    upstream[int(5 / 0.25 + 128), int(10 / 0.25)] = 1
    upstream = np.rot90(upstream)[::-1]
    assert upstream.shape == (bev.ROWS, bev.COLS)
    row, col = bev.xy_to_cell(10.0, 5.0)
    assert (int(row), int(col)) == (40, 148)
    assert upstream[row, col] == 1


def test_cell_centres_span_the_front_half() -> None:
    xs, ys = bev.cell_centres()
    assert xs[0] == pytest.approx(0.125) and xs[-1] == pytest.approx(31.875)
    assert ys[0] == pytest.approx(-31.875) and ys[-1] == pytest.approx(31.875)


def test_polygon_with_a_hole() -> None:
    square = np.array([[4.0, -4.0], [12.0, -4.0], [12.0, 4.0], [4.0, 4.0]])
    hole = np.array([[6.0, -2.0], [10.0, -2.0], [10.0, 2.0], [6.0, 2.0]])
    mask = bev.polygon_mask([(square, [hole])])
    for x, y, expected in [(5, 0, True), (8, 0, False), (11, 3, True), (13, 0, False), (8, -5, False)]:
        row, col = bev.xy_to_cell(x, y)
        assert mask[row, col] == expected, (x, y)
    assert mask.sum() == (8 * 8 - 4 * 4) / bev.CELL_M**2


def test_polyline_is_half_a_metre_wide() -> None:
    mask = bev.polyline_mask([np.array([[0.0, 3.0], [30.0, 3.0]])])
    row = bev.xy_to_cell(15.0, 0.0)[0]
    columns = np.flatnonzero(mask[row])
    assert len(columns) == 2  # centres at 2.875 and 3.125
    assert bev.cell_centres()[1][columns].tolist() == [2.875, 3.125]


def test_box_corners_rotate_with_heading() -> None:
    corners = bev.box_corners(10.0, 0.0, math.pi / 2, 4.0, 2.0)
    assert np.allclose(sorted(corners[:, 0]), [9, 9, 11, 11])
    assert np.allclose(sorted(corners[:, 1]), [-2, -2, 2, 2])


def forward_camera() -> tuple[np.ndarray, np.ndarray]:
    """A 1600 x 900 pinhole 1.5 m up, looking along x, roughly nuScenes' CAM_FRONT."""
    intrinsics = np.array([[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1]])
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # camera z forward, x right, y down
    cam2ego[:3, 3] = [1.7, 0.0, 1.5]
    return intrinsics, cam2ego


def test_view_mask_is_a_forward_wedge() -> None:
    intrinsics, cam2ego = forward_camera()
    view = bev.view_mask(intrinsics, cam2ego, 1600, (28, 872))
    half_fov = math.atan(800 / 1266.0)
    for x, y, expected in [
        (20.0, 0.0, True),
        (20.0, 0.9 * (20 - 1.7) * math.tan(half_fov), True),
        (20.0, 1.1 * (20 - 1.7) * math.tan(half_fov), False),
        (20.0, -1.1 * (20 - 1.7) * math.tan(half_fov), False),
        (1.0, 0.0, False),  # under the bonnet, below the image
    ]:
        row, col = bev.xy_to_cell(x, y)
        assert view[row, col] == expected, (x, y)


def test_semantic_map_paints_in_order_and_ignores_the_unseen() -> None:
    road = np.ones((bev.ROWS, bev.COLS), dtype=bool)
    car = np.zeros_like(road)
    car[40, 128] = True
    view = np.zeros_like(road)
    view[30:50, 100:150] = True
    target = bev.semantic_map({"vehicle": car, "road": road}, view)
    assert target[40, 128] == bev.CLASSES.index("vehicle")
    assert target[41, 128] == bev.CLASSES.index("road")
    assert target[0, 0] == bev.IGNORE
    with pytest.raises(ValueError):
        bev.semantic_map({"tree": road}, view)


def test_agent_targets_keep_the_nearest_seen_boxes_on_the_grid() -> None:
    boxes = np.array(
        [
            [20.0, 0.0, 0, 4, 2],  # seen
            [5.0, 1.0, 0, 4, 2],  # seen, nearest
            [-5.0, 0.0, 0, 4, 2],  # behind: off the grid
            [10.0, 0.0, 0, 4, 2],  # not seen
            [40.0, 0.0, 0, 4, 2],  # beyond 32 m
        ]
    )
    seen = np.array([True, True, True, False, True])
    states, valid = targets.agent_targets(boxes, seen)
    assert states.shape == (targets.MAX_AGENTS, 5) and valid.sum() == 2
    assert states[0, 0] == 5.0 and states[1, 0] == 20.0


def test_ego_path_extends_straight_and_stands_in_for_a_stopped_ego() -> None:
    path = targets.ego_path(np.array([[2.0 * k, 0.0] for k in range(1, 9)]))
    assert path[-1].tolist() == pytest.approx([targets.LEAD_RANGE_M, 0.0])
    stopped = targets.ego_path(np.zeros((8, 2)))
    assert stopped.tolist() == [[0.0, 0.0], [targets.LEAD_RANGE_M, 0.0]]


def test_path_coordinates_on_a_left_turn() -> None:
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    s, d, _ = targets.path_coordinates(np.array([[5.0, 1.0], [11.0, 5.0]]), path)
    assert s.tolist() == pytest.approx([5.0, 15.0])
    assert d.tolist() == pytest.approx([1.0, -1.0])  # left of the first leg, right of the second


def test_lead_vehicle_is_the_nearest_in_the_way() -> None:
    future = np.array([[2.0 * k, 0.0] for k in range(1, 9)])
    boxes = np.array(
        [
            [30.0, 0.5, 0.0, 4.5, 1.8],  # in the way, farther
            [15.0, -0.3, 0.0, 4.5, 1.8],  # in the way, nearer: the lead
            [8.0, 3.5, 0.0, 4.5, 1.8],  # next lane
            [12.0, 0.0, math.pi, 4.5, 1.8],  # oncoming
            [10.0, 0.0, 0.0, 4.5, 1.8],  # not seen
        ]
    )
    velocities = np.array([[8.0, 0.0], [6.0, 0.0], [9.0, 0.0], [-9.0, 0.0], [0.0, 0.0]])
    seen = np.array([True, True, True, True, False])
    lead = targets.lead_vehicle(boxes, velocities, seen, future, ego_speed=10.0)
    assert lead["present"] == 1 and lead["index"] == 1
    assert lead["gap"] == pytest.approx(15.0 - 4.5 / 2)
    assert lead["closing"] == pytest.approx(4.0)


def test_no_lead_and_an_unknown_speed() -> None:
    future = np.array([[2.0 * k, 0.0] for k in range(1, 9)])
    assert targets.lead_vehicle(np.zeros((0, 5)), np.zeros((0, 2)), np.zeros(0), future, 5.0)["present"] == 0
    lead = targets.lead_vehicle(
        np.array([[12.0, 0.0, 0.0, 4.0, 2.0]]), np.array([[np.nan, np.nan]]), np.array([True]), future, 5.0
    )
    assert lead["present"] == 1 and math.isnan(lead["closing"])


def test_lane_change_rule() -> None:
    assert targets.lane_change({"a"}, {"a"}, set(), 0.0) == 0
    assert targets.lane_change({"a"}, {"c"}, {"b", "c"}, 3.0) == 0  # followed the graph
    assert targets.lane_change({"a"}, {"x"}, {"b"}, 3.2) == 1
    assert targets.lane_change({"a"}, {"x"}, {"b"}, -3.2) == 2
    assert targets.lane_change(set(), {"x"}, set(), 3.0) == -1


def test_riding_the_line_is_not_a_lane_change() -> None:
    # another lane's polygon holds the ego 4 s later, but it moved only 0.4 m across its lane
    assert targets.lane_change({"a"}, {"x"}, set(), 0.4) == 0
    assert targets.lane_change({"a"}, {"x"}, set(), -(targets.LANE_CHANGE_MIN_SHIFT_M - 0.01)) == 0


def test_a_lane_the_graph_names_but_lacks_is_reachable_and_leads_nowhere() -> None:
    from gallops.drivejepa_aux import nusc

    class Graph:
        connectivity = {"a": {"outgoing": ["b"]}, "b": {"outgoing": ["ghost"]}}

        def get_outgoing_lane_ids(self, lane):
            if lane not in self.connectivity:
                raise ValueError(f"{lane} is not a valid lane.")  # the devkit's behaviour
            return self.connectivity[lane]["outgoing"]

    view = object.__new__(nusc.MapView)
    view.nmap = Graph()
    assert view.reachable({"a"}) == {"b", "ghost"}
