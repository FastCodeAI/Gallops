"""The parts of `gallops.memory.inputs` that repeat paddock's node, without navsim or a model."""

from __future__ import annotations

import numpy as np

from gallops.memory import inputs


def test_crop_box_is_the_whole_frame_for_openscene_s_own_camera():
    k = inputs.target_intrinsics().astype(np.float64)
    assert inputs.crop_box(k) == (0, 0, inputs.TARGET_WIDTH, inputs.TARGET_HEIGHT)


def test_crop_box_scales_about_the_principal_point():
    k = np.array([[1545.0 * 2, 0, 1000.0], [0, 1545.0 * 2, 600.0], [0, 0, 1]])
    left, top, right, bottom = inputs.crop_box(k)
    assert (right - left, bottom - top) == (inputs.TARGET_WIDTH * 2, inputs.TARGET_HEIGHT * 2)
    assert (left, top) == (1000 - 960 * 2, 600 - 560 * 2)


def test_command_round_trip_keeps_one_hot_and_maps_the_rest_to_the_argmax():
    for hot in range(4):
        one_hot = [0, 0, 0, 0]
        one_hot[hot] = 1
        assert inputs.command_round_trip(one_hot) == one_hot
    assert inputs.command_round_trip([0, 0, 0, 0]) == [1, 0, 0, 0]  # argmax of zeros is 0: left
    assert inputs.command_round_trip([]) == [0, 0, 0, 1]


def test_relative_pose_in_the_current_frame():
    current = inputs.pose_matrix([10.0, 5.0, np.pi / 2])
    behind = inputs.pose_matrix([10.0, 3.0, np.pi / 2])  # 2 m behind, facing the same way
    np.testing.assert_allclose(inputs.relative_pose(behind, current), [-2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(inputs.relative_pose(current, current), [0.0, 0.0, 0.0], atol=1e-12)
