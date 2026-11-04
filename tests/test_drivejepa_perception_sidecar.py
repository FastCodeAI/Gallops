"""H15's side-car: the target rule, the camera geometry, the metrics (numpy) and the model (torch).

The torch tests skip in gallops' own interpreter, which has no torch; they run in Drive-JEPA's,
where the preflight runs the whole suite before any training step.
"""

from __future__ import annotations

import numpy as np
import pytest

from gallops.drivejepa_perception import evaluate
from gallops.drivejepa_perception import targets as tg


def _pinhole_lidar2img():
    """A front camera on the padded 768x448 frame, as the vendor builds it: 0.4 of 1920x1080.

    Ego (x forward, y left, z up) -> camera (right, down, forward); focal 1000 px and principal point
    (960, 540) at full size, so 0.4x: f = 400, c = (384, 216).
    """
    to_cam = np.array([[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    k = np.array([[400, 0, 384, 0], [0, 400, 216, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    l2i = np.zeros((4, 4, 4))
    l2i[tg.FRONT_SLOT] = k @ to_cam
    return l2i


def test_to_512_maps_the_padded_frame_onto_the_cropped_resized_image():
    # A full-size pixel (u, v) is (0.4 u, 0.4 v) on the padded frame and (u * 512 / 1920,
    # (v - 28) * 256 / 1024) on the token image: the crop drops 28 rows, then the resize.
    for u, v in ((0.0, 28.0), (1920.0, 1052.0), (960.0, 540.0)):
        pad = tg.TO_512 @ np.array([0.4 * u, 0.4 * v, 1.0, 1.0])
        assert np.allclose(pad[:2], [u * 512 / 1920, (v - 28) * 256 / 1024])


def test_targets_keep_seen_agents_of_the_three_classes_nearest_first():
    projection = tg.front_projection(_pinhole_lidar2img())
    agents = {
        "cls": np.array([0, 1, 3, 0, 0, 2], dtype=np.int8),
        "box": np.array(
            [
                [30.0, 1.0, 0.5, 4, 2, 1.5, 0.1],  # vehicle ahead
                [10.0, -1.0, 0.0, 0.6, 0.6, 1.8, 0.0],  # pedestrian ahead, nearer
                [12.0, 0.0, 0.0, 1, 1, 1, 0.0],  # static: not a side-car class
                [-10.0, 0.0, 0.5, 4, 2, 1.5, 0.0],  # behind the camera
                [60.0, 0.0, 0.5, 4, 2, 1.5, 0.0],  # beyond RANGE_M
                [20.0, 2.0, 0.0, 2, 0.8, 1.7, 0.0],  # cyclist ahead
            ],
            dtype=np.float32,
        ),
        "velocity": np.zeros((6, 2), np.float32),
        "path": np.array([[[31.0, 1.0], [32.0, 1.0]]] * 6, dtype=np.float32),
        "path_ok": np.array([[True, False]] * 6),
    }
    t = tg.build(agents, projection)
    assert t["mask"].sum() == 3
    assert list(t["cls"][:3]) == [1, 2, 0]  # pedestrian 10 m, cyclist 20 m, vehicle 30 m
    assert np.allclose(t["path"][2, 0], [1.0, 0.0])  # the vehicle's offset at t + 0.5 s
    assert not t["path_ok"][2, 1] and np.allclose(t["path"][2, 1], 0.0)


def test_match_is_class_aware_and_within_two_metres():
    conf = np.array([0.9, 0.8, 0.3])
    cls = np.array([0, 1, 0])
    xy = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    gxy = np.array([[10.5, 0.0], [20.0, 0.0], [30.0, 0.0]])
    gcls = np.array([0, 0, 0])
    det, tp, hit = evaluate.match(conf, cls, xy, gxy, gcls)
    assert list(det) == [0, 1]  # the 0.3 slot is below THR
    assert list(tp) == [True, False]  # a pedestrian cannot hit a vehicle
    assert list(hit) == [0, -1, -1]


def test_readout_scores_a_perfect_prediction_perfectly():
    projection = tg.front_projection(_pinhole_lidar2img())
    agents = {
        "cls": np.array([0, 0], dtype=np.int8),
        "box": np.array([[15.0, 0.0, 0.5, 4, 2, 1.5, 0], [25.0, 3.0, 0.5, 4, 2, 1.5, 0]], dtype=np.float32),
        "velocity": np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "path": np.array([[[16.0, 0.0], [17.0, 0.0]], [[25.0, 3.0], [25.0, 3.0]]], dtype=np.float32),
        "path_ok": np.ones((2, 2), bool),
    }
    gt = tg.build(agents, projection)
    m = gt["mask"]
    pred = {
        "conf": np.where(m, 0.9, 0.1),
        "cls": gt["cls"],
        "xyz": gt["box"][:, :3].astype(np.float64),
        "vel": gt["vel"].astype(np.float64),
        "path": gt["box"][:, None, :2] + gt["path"],
    }
    out = evaluate.readout([pred], [gt], prior_gap=30.0)
    assert out["recall_2m"] == 1.0 and out["precision_2m"] == 1.0
    assert out["centre_error_m"] == 0.0 and out["path_fde_1s_model_moving"] == 0.0
    assert out["lead_gap_mae_m"] == 0.0 and out["lead_gap_prior_mae_m"] == 15.0


def test_the_side_car_is_about_the_planned_size_and_learns():
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception import sidecar

    model = sidecar.SideCar()
    params = sum(p.numel() for p in model.parameters())
    assert 3.5e6 < params < 4.5e6, params  # the plan's ~4.2 M
    projection = torch.from_numpy(tg.front_projection(_pinhole_lidar2img()).astype(np.float32))[None]
    tokens = torch.randn(1, 512, 256)
    outs = model(tokens, projection)
    assert len(outs) == 3 and outs[-1]["path"].shape == (1, tg.QUERIES, tg.PATH_STEPS, 2)
    agents = {
        "cls": np.array([0], dtype=np.int8),
        "box": np.array([[15.0, 0.0, 0.5, 4, 2, 1.5, 0]], dtype=np.float32),
        "velocity": np.zeros((1, 2), np.float32),
        "path": np.array([[[16.0, 0.0], [17.0, 0.0]]], dtype=np.float32),
        "path_ok": np.ones((1, 2), bool),
    }
    tgt = {k: torch.from_numpy(v)[None] for k, v in tg.build(agents, projection[0].numpy()).items()}
    loss, parts = sidecar.SideCarLoss()(outs, tgt)
    assert torch.isfinite(loss) and "path2" in parts
