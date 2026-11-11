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


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Ego -> camera for a camera looking along ego x, turned by yaw (left) and pitched (down)."""
    base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)  # (right, down, forward)
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    yaw = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    pitch = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return base @ (yaw @ pitch).T


def test_rq3_and_camera_of_recover_the_camera_a_projection_was_made_from():
    k = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
    rotation = _rotation(1.0, 2.0)
    centre = np.array([1.6, 0.0, 1.5])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation.T, centre
    projection = ood.to_512_direct({"CAM_FRONT": {"intrinsics": k, "pose": pose}}, *ood.NAVSIM_SIZE)
    cam = ood.camera_of(projection, *ood.NAVSIM_SIZE)
    assert np.allclose(cam["k"], k, atol=1e-6)
    assert np.allclose(cam["rotation"], rotation, atol=1e-9)
    assert np.allclose(cam["centre"], centre, atol=1e-9)
    # On NAVSIM's frame the token-image projection is the vendor's: TO_512 at 0.4 scale.
    k4 = np.eye(4)
    k4[:3, :3] = k
    vendor = np.diag([tg.VENDOR_SCALE, tg.VENDOR_SCALE, 1.0, 1.0]) @ k4 @ np.linalg.inv(pose)
    assert np.allclose(tg.TO_512 @ vendor, projection)


def test_the_homography_carries_every_point_to_where_the_warped_camera_sees_it():
    centre = np.array([1.7, 0.0, 1.5])
    nus = {
        "k": np.array([[1266.0, 0.0, 816.0], [0.0, 1266.0, 491.0], [0.0, 0.0, 1.0]]),
        "rotation": _rotation(0.8, 1.9),
        "centre": centre,
    }
    nav = {
        "k": np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]]),
        "rotation": _rotation(-0.6, 0.4),
        "centre": centre,
    }
    h = ood.homography(nus["k"], nus["rotation"], nav["k"], nav["rotation"])
    assert ood.warp_gap(h, nus, nav) < 1e-6
    # The warped frame's projection is the target camera's, through NAVSIM's crop and resize.
    projection = ood.warped_projection(nav["k"], nav["rotation"], centre)
    point = np.array([25.0, 3.0, 0.5])
    full = nav["k"] @ nav["rotation"] @ (point - centre)
    token = projection @ np.r_[point, 1.0]
    s = ood.crop_resize(*ood.NAVSIM_SIZE)
    assert np.allclose((s @ full)[:2] / full[2], token[:2] / token[2])
    # The same camera on both sides warps by the identity; a wrong rotation does not pass.
    assert np.allclose(ood.homography(nus["k"], nus["rotation"], nus["k"], nus["rotation"]), np.eye(3))
    wrong = ood.homography(nus["k"], nus["rotation"].T, nav["k"], nav["rotation"])
    assert ood.warp_gap(wrong, nus, nav) > 1.0


def test_scene_interval_pools_counts_and_its_interval_brackets_the_point():
    scenes = np.array(["a", "a", "b", "c"])
    counts = {
        "targets": np.array([2, 2, 4, 2]),
        "hits": np.array([1, 2, 2, 2]),
        "detections": np.array([2, 2, 3, 2]),
        "right": np.array([1, 2, 2, 2]),
    }
    out = ood.scene_interval(counts, scenes, draws=200)
    assert np.isclose(out["recall"], 7 / 10) and np.isclose(out["precision"], 7 / 9)
    assert out["recall_95"][0] <= out["recall"] <= out["recall_95"][1] and out["targets"] == 10


def test_scene_groups_name_traffic_side_and_read_night_and_rain_from_the_description():
    scenes = {
        "s1": {"location": "boston-seaport", "description": "Rain, parked cars"},
        "s2": {"location": "singapore-onenorth", "description": "Night, turn right"},
    }
    groups = ood.scene_groups(scenes, np.array(["s1", "s2", "s2"]))
    assert groups["traffic=left"].tolist() == [False, True, True]
    assert groups["rain"].tolist() == [True, False, False] and groups["night"].tolist() == [False, True, True]
    assert groups["location=boston-seaport"].tolist() == [True, False, False]
