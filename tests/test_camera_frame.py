"""Drive-JEPA must see the camera: the extrinsics put it there, and preflight refuses a model that does not.

Until paddock #62 the cameras were placed relative to nuScenes' LIDAR_TOP, yawed
-90 degrees. The model's image sample points are in ego coordinates (NAVSIM's lidar
frame is the ego frame), so none landed in the front image and it planned from ego
status alone. Every check passed, because each compared one copy of that mistake
with another. These run where torch, navsim and Pillow are absent.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from gallops.train import preflight
from gallops.train.swerve_data import NUSCENES_CAMERAS, camera_extrinsics

#: nuScenes' LIDAR_TOP on the ego: yawed -90 degrees (its x axis to the car's
#: right), 0.94 m forward and 1.84 m up.
LIDAR2EGO = np.array(
    [[0.0, 1.0, 0.0, 0.94], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0.0, 0.0, 0.0, 1.0]]
)
#: A front camera: optical axis along ego x, 1.7 m forward, 1.5 m up.
FRONT2EGO = np.array(
    [[0.0, 0.0, 1.0, 1.7], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
)
K = np.array([[1266.0, 0.0, 800.0], [0.0, 1266.0, 450.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def calibration() -> dict:
    out = {channel: (FRONT2EGO, K) for channel in NUSCENES_CAMERAS.values()}
    out["LIDAR_TOP"] = (LIDAR2EGO, None)
    return out


def test_a_point_ahead_of_the_car_lands_in_the_front_image():
    """The vendor's lidar2img applied to a point 10 m ahead, as the model applies it.

    With LIDAR_TOP composed in, the same point lands behind the camera.
    """
    sensor2lidar = camera_extrinsics(calibration())["cam_f0"]
    np.testing.assert_allclose(sensor2lidar, FRONT2EGO)
    ahead = np.array([10.0, 0.0, 0.5])

    def project(extrinsics):
        in_camera = extrinsics[:3, :3].T @ (ahead - extrinsics[:3, 3])
        return in_camera, (K @ in_camera)[:2] / in_camera[2]

    in_camera, (u, v) = project(sensor2lidar)
    assert in_camera[2] > 0 and 0 < u < 1600 and 0 < v < 900, f"10 m ahead projects to ({u:.0f}, {v:.0f})"
    old, _ = project(np.linalg.inv(LIDAR2EGO) @ FRONT2EGO)
    assert old[2] < 0, "the old composition should put the point behind the camera"


class _Tensor:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float64)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Torch:
    no_grad = staticmethod(nullcontext)

    @staticmethod
    def zeros_like(a):
        return np.zeros_like(a)


class _Dataset:
    def __init__(self, ticks, device):
        self.ticks = ticks

    def batch(self, indices):
        image = np.full((3, 4, 4), 0.5)
        return (
            {"camera_feature_1": image.copy(), "camera_feature_2": image.copy()},
            None,
            [self.ticks[i] for i in indices],
        )


class _Model:
    def __init__(self, looks: bool):
        self.looks, self.training = looks, True

    def eval(self):
        self.training = False

    def train(self, mode):
        self.training = mode

    def __call__(self, features):
        plan = np.zeros((1, 8, 3))
        plan[..., 0] = np.arange(1, 9) * (
            2.0 + (10.0 * features["camera_feature_1"].mean() if self.looks else 0.0)
        )
        return {"trajectory": _Tensor(plan)}


def test_sight_refuses_a_model_that_a_black_frame_does_not_move():
    report, problems = preflight.sight_check(_Model(looks=False), _Dataset, [{}] * 3, _Torch, "cpu")
    assert report["max_move_m"] == 0.0
    assert problems and "not reaching the model" in problems[0]


def test_sight_passes_a_model_that_looks():
    model = _Model(looks=True)
    report, problems = preflight.sight_check(model, _Dataset, [{}] * 3, _Torch, "cpu")
    assert problems == [] and report["compared"] == 3 and report["max_move_m"] > preflight.SIGHT_MIN_MOVE_M
    assert model.training is True, "the model's mode must be restored"
