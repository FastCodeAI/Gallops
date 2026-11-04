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


def test_within_30m_means_30_metres():
    projection = tg.front_projection(_pinhole_lidar2img())
    agents = {
        "cls": np.array([0, 0], dtype=np.int8),
        "box": np.array([[28.0, 0.0, 0.5, 4, 2, 1.5, 0], [32.0, 0.0, 0.5, 4, 2, 1.5, 0]], dtype=np.float32),
        "velocity": np.zeros((2, 2), np.float32),
        "path": np.array([[[28.0, 0.0]] * 2, [[32.0, 0.0]] * 2], dtype=np.float32),
        "path_ok": np.ones((2, 2), bool),
    }
    gt = tg.build(agents, projection)
    assert gt["mask"].sum() == 2
    found = gt["box"][:, 0] > 30  # detect only the agent beyond 30 m
    pred = {
        "conf": np.where(gt["mask"] & found, 0.9, 0.1),
        "cls": gt["cls"],
        "xyz": gt["box"][:, :3].astype(np.float64),
        "vel": gt["vel"].astype(np.float64),
        "path": gt["box"][:, None, :2] + gt["path"],
    }
    out = evaluate.readout([pred], [gt], prior_gap=30.0)
    assert out["recall_2m_within_30m"] == 0.0  # the 28 m agent is within 30 m, and missed
    assert out["recall_by_range"][1] == 0.0 and out["recall_by_range"][2] == 1.0


def test_to_512_is_navsims_matrix_at_navsims_size_and_maps_any_camera_the_same_way():
    assert np.allclose(tg.to_512(1920, 1080), tg.TO_512, atol=1e-12)
    # nuScenes' front camera: (u, v) full-size is (0.4 u, 0.4 v) padded, and on the token image
    # (u * 512 / 1600, (v - 28) * 256 / 844).
    for u, v in ((0.0, 28.0), (1600.0, 872.0), (800.0, 450.0)):
        pad = tg.to_512(1600, 900) @ np.array([0.4 * u, 0.4 * v, 1.0, 1.0])
        assert np.allclose(pad[:2], [u * 512 / 1600, (v - 28) * 256 / 844])


def test_agent_arrays_mask_the_slots_under_threshold_and_leave_other_rows_empty():
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception import evaluate, sidecar, train_sidecar

    torch.manual_seed(0)
    model = sidecar.SideCar(d_in=16, layers=1, queries=8)
    tokens = np.random.default_rng(0).standard_normal((3, 512, 16)).astype(np.float16)
    projections = np.repeat(tg.front_projection(_pinhole_lidar2img())[None], 3, axis=0).astype(np.float32)
    arrays = train_sidecar.agent_arrays(model, tokens, int, [0, 2], projections, torch.device("cpu"))
    assert arrays["xyz"].shape == (3, 8, 3) and arrays["path"].shape == (3, 8, tg.PATH_STEPS, 2)
    assert np.array_equal(arrays["mask"], arrays["conf"] >= evaluate.THR)
    assert not arrays["mask"][1].any() and not arrays["conf"][1].any()  # row 1 was not asked for


def test_readout_detail_gives_each_hit_agents_row_and_the_means_it_averages():
    projection = tg.front_projection(_pinhole_lidar2img())
    agents = {
        "cls": np.array([0], dtype=np.int8),
        "box": np.array([[15.0, 0.0, 0.5, 4, 2, 1.5, 0]], dtype=np.float32),
        "velocity": np.array([[2.0, 0.0]], dtype=np.float32),
        "path": np.array([[[16.0, 0.0], [17.0, 0.0]]], dtype=np.float32),
        "path_ok": np.ones((1, 2), bool),
    }
    gt = tg.build(agents, projection)
    empty = {k: np.zeros_like(v) for k, v in gt.items()}
    pred = {
        "conf": np.where(gt["mask"], 0.9, 0.1),
        "cls": gt["cls"],
        "xyz": gt["box"][:, :3].astype(np.float64),
        "vel": np.zeros_like(gt["vel"], dtype=np.float64),  # a stationary guess: cv_pred misses by 2 m
        "path": gt["box"][:, None, :2] + gt["path"],
    }
    none = {**pred, "conf": np.zeros_like(pred["conf"])}
    out = evaluate.readout([none, pred], [empty, gt], prior_gap=30.0, detail=True)
    d = out["detail"]
    assert d["path_row"].tolist() == [1] and d["path_moving"].tolist() == [True]
    assert d["fde_model"].tolist() == [0.0] and np.isclose(d["fde_cv_pred"][0], 2.0)
    assert np.isclose(out["path_fde_1s_cv_pred_moving"], d["fde_cv_pred"][d["path_moving"]].mean())
