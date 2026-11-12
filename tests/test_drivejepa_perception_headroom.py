"""B-0-3's headroom check: which object is behind a failure, and whether the side-car found it (numpy)."""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception import headroom
from gallops.drivejepa_perception import targets as tg


def _agents():
    """A vehicle 10 m ahead (in view), a cone 6 m ahead (static), a vehicle 10 m behind (out of view)."""
    box = np.zeros((3, 7), np.float32)
    box[:, :3] = [[10.0, 0.0, 0.5], [6.0, 1.0, 0.3], [-10.0, 0.0, 0.5]]
    box[:, 3:6] = [4.0, 2.0, 1.5]
    return {
        "cls": np.array([0, 3, 0], np.int8),
        "box": box,
        "velocity": np.zeros((3, 2), np.float32),
        "path": np.zeros((3, 2, 2), np.float32),
        "path_ok": np.ones((3, 2), bool),
        "track": np.array([11, 22, 33], np.int64),
    }


def _projection():
    """A camera at the ego origin looking along x."""
    proj = np.zeros((4, 4))
    proj[0] = [256.0, -256.0, 0, 0]
    proj[1] = [128.0, 0, -128.0, 0]
    proj[2] = [1.0, 0, 0, 0]
    proj[3] = [0, 0, 0, 1.0]
    return proj


def test_the_object_behind_a_failure_is_classified():
    agents, proj = _agents(), _projection()
    assert headroom.failure_kind(agents, proj, 11) == ("in_view_50m", 0)
    assert headroom.failure_kind(agents, proj, 22) == ("static", -1)
    assert headroom.failure_kind(agents, proj, 33) == ("outside_view_or_range", -1)
    assert headroom.failure_kind(agents, proj, 99) == ("not_in_labels", -1)
    assert headroom.failure_kind(agents, proj, headroom.NO_AGENT) == ("unnamed", -1)


def test_a_target_is_found_by_a_detection_of_its_class_within_two_metres():
    target = tg.build(_agents(), _projection())
    q = 4
    detections = {
        "conf": np.zeros((1, q), np.float32),
        "cls": np.zeros((1, q), np.int64),
        "xyz": np.zeros((1, q, 3), np.float32),
    }
    assert not headroom.found_by(detections, 0, target, 0)
    detections["conf"][0, 0] = 0.9
    detections["xyz"][0, 0] = [11.5, 0.5, 0.5]  # 1.6 m from the vehicle's centre
    assert headroom.found_by(detections, 0, target, 0)
    detections["cls"][0, 0] = 1  # a pedestrian there does not find a vehicle
    assert not headroom.found_by(detections, 0, target, 0)
