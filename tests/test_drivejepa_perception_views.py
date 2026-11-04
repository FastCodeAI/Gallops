"""H15's review pictures: a sample's window from its log, box corners, projection, the view from above."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from gallops.drivejepa_perception import targets as tg
from gallops.drivejepa_perception import views


def _projection():
    """A front camera 1.5 m up, looking along x: f = 400 and principal point (384, 216) on the vendor's
    padded frame (0.4 of 1000 and of (960, 540)), composed with TO_512 as the trainer composes it."""
    to_cam = np.array([[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    k = np.array([[400, 0, 384, 0], [0, 400, 216, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    l2i = np.zeros((4, 4, 4))
    l2i[tg.FRONT_SLOT] = k @ to_cam
    return tg.front_projection(l2i)


def test_box_corners_span_the_box_and_turn_with_its_yaw():
    box = np.array([[10.0, 2.0, 1.0, 4.0, 2.0, 2.0, 0.0]])
    c = views.box_corners(box)[0]
    assert np.allclose(c.min(0), [8.0, 1.0, 0.0]) and np.allclose(c.max(0), [12.0, 3.0, 2.0])
    assert np.allclose(c[[0, 1], 0], 12.0)  # corners 0 and 1 are the front edge
    turned = views.box_corners(box * [1, 1, 1, 1, 1, 1, 0] + [0, 0, 0, 0, 0, 0, np.pi / 2])[0]
    assert np.allclose(turned[:, 1].max() - turned[:, 1].min(), 4.0)  # length now runs along y
    assert np.allclose(views.footprint(box)[0], c[:4, :2])


def test_project_and_camera_ray_are_inverses():
    p = _projection()
    uv, depth = views.project(p, np.array([[20.0, 0.0, 1.5]]))  # on the optical axis
    assert np.allclose(uv[0], [384 * 2 / 3, 216 * 0.625 - 7]) and np.isclose(depth[0], 20.0)
    centre, ray = views.camera_ray(p, 100.0, 50.0)
    back, _ = views.project(p, centre + 13.0 * ray)
    assert np.allclose(back, [100.0, 50.0])
    assert views.project(p, np.array([[-5.0, 0.0, 1.0]]))[1][0] < 0  # behind the camera


def test_edges_of_a_box_behind_the_camera_are_dropped():
    p = _projection()
    ahead, behind = views.box_edges_2d(p, np.array([[15, 0, 1, 4, 2, 1.5, 0], [-15, 0, 1, 4, 2, 1.5, 0]]))
    assert len(ahead) == 12 and behind == []


def test_view_edges_open_as_wide_as_the_image():
    centre, left, right = views.view_edges(_projection(), reach=10.0)
    assert np.allclose(centre, [0.0, 0.0])
    # The left image edge is u = 0, which is 256 px left of the axis at f = 400 * 2/3 on the token
    # image: a half-angle of atan(256 / 266.7), about 43.8 degrees.
    half = np.degrees(np.arctan2(left[1], left[0]))
    assert np.isclose(half, np.degrees(np.arctan(256 / (400 * 2 / 3))), atol=0.01)
    assert right[1] < 0 < left[1]


def test_token_of_is_row_major_over_16_by_32():
    uv = np.array([[8.0, 8.0], [511.0, 255.0], [16.0, 16.0], [-1.0, 0.0], [512.0, 0.0]])
    assert views.token_of(uv).tolist() == [0, 511, 33, -1, -1]


def test_window_frames_slices_the_log_as_navsim_does(tmp_path):
    log = tmp_path / "navsim_logs" / "trainval"
    log.mkdir(parents=True)
    with open(log / "L.pkl", "wb") as fh:
        pickle.dump([{"token": f"t{i}"} for i in range(20)], fh)
    window = views.window_frames(tmp_path, "L", "t5")
    assert [f["token"] for f in window] == [f"t{i}" for i in range(2, 16)]
    assert window[3]["token"] == "t5"  # navsim_data.CURRENT
    for token in ("t1", "t15"):  # no full history, no full future
        with pytest.raises(ValueError):
            views.window_frames(tmp_path, "L", token)
