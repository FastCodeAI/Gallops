"""H15's nuScenes step: classes, the projection's two routes, and labels in H15's layout (numpy only)."""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception import ood
from gallops.drivejepa_perception import targets as tg


def test_classes_follow_navsims_grouping():
    assert ood.class_of("vehicle.car") == 0 and ood.class_of("vehicle.bus.rigid") == 0
    assert ood.class_of("vehicle.bicycle") == 2 and ood.class_of("vehicle.motorcycle") == 2
    assert ood.class_of("human.pedestrian.adult") == 1
    assert ood.class_of("movable_object.trafficcone") == 3 and ood.class_of("animal") == -1


def _camera():
    # A front camera 1.5 m up and 1 m ahead, looking along x: camera z = ego x, x = -ego y, y = -ego z.
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    cam2ego[:3, 3] = [1.0, 0.0, 1.5]
    intrinsics = np.array([[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1]])
    return cam2ego, intrinsics


def test_the_vendors_projection_and_the_calibrations_agree_on_nuscenes_geometry():
    cam2ego, intrinsics = _camera()
    # The vendor's lidar2img (tokens.fast_bev_feature): 0.4 x the pixels of K [R|t]_ego->cam.
    k4 = np.eye(4)
    k4[:3, :3] = intrinsics
    vendor = np.diag([tg.VENDOR_SCALE, tg.VENDOR_SCALE, 1.0, 1.0]) @ k4 @ np.linalg.inv(cam2ego)
    calibration = {"CAM_FRONT": {"pose": cam2ego.tolist(), "intrinsics": intrinsics.tolist()}}
    direct = ood.to_512_direct(calibration, *ood.NUSC_SIZE)
    assert ood.projection_gap(tg.to_512(*ood.NUSC_SIZE) @ vendor, direct) < 1e-6
    # NAVSIM's matrix on nuScenes' camera is off by many pixels: the reason to_512 exists.
    assert ood.projection_gap(tg.TO_512 @ vendor, direct) > 20


def test_keyframe_agents_are_in_the_keyframes_frame_with_their_paths_and_seen_by_lidar():
    shift = np.eye(4)
    shift[:3, 3] = [100.0, 50.0, 0.0]  # the ego at (100, 50), facing +x
    later = [shift.copy(), shift.copy(), shift.copy()]
    for i, m in enumerate(later):
        m[0, 3] += 5.0 * i
    scene = {"keyframes": [{"sample": f"s{i}", "ego": m.tolist()} for i, m in enumerate(later)]}

    def car(instance, x, points=5):
        return {
            "category": "vehicle.car",
            "instance": instance,
            "translation": [100.0 + x, 52.0, 0.8],
            "size_wlh": [1.9, 4.5, 1.6],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [2.0, 0.0, 0.0],
            "lidar_points": points,
        }

    annotations = {
        "s0": [car("a", 10.0), car("hidden", 20.0, points=0)],
        "s1": [car("a", 11.0)],
        "s2": [car("a", 12.0)],
    }
    out = ood.keyframe_agents(scene, {"keyframe": 0}, annotations)
    assert len(out["cls"]) == 1 and out["cls"][0] == 0  # the unseen box is no target
    assert np.allclose(out["box"][0, :2], [10.0, 2.0]) and np.isclose(out["box"][0, 3], 4.5)
    assert np.allclose(out["path"][0], [[11.0, 2.0], [12.0, 2.0]]) and out["path_ok"][0].all()
    assert np.allclose(out["velocity"][0], [2.0, 0.0])
