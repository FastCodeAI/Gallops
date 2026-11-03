"""Drive-JEPA's input for one navtrain moment, built the way paddock's node builds it on NAVSIM.

On NAVSIM, paddock's proxy agent turns each of a scenario's four history frames into an
`Observation` (`adapters/benchmarks/navsim/agent.py` `_observation`), and the Drive-JEPA node turns
those back into a NAVSIM `AgentInput` (`adapters/models/drivejepa/policy.py` `_agent_input`). The
round trip is not the identity, and the cache must see what the node sees, so this module repeats
it from the log pickle:

* **Ego status.** The pose is re-derived through 4x4 matrices relative to the current frame (the
  node's `relative_pose`); velocity and acceleration keep only their forward component, with the
  lateral one set to 0 (policy.py:301-304); the driving command goes through paddock's `Command`
  and back, so a one-hot vector survives and anything else becomes the argmax's command.
* **Cameras.** Each image is read with PIL and converted to RGB (paddock's `load_frame`), then put
  through the node's "openscene" geometry (`_to_openscene`: crop about the principal point and
  resize to 1920x1080, intrinsics replaced by OpenScene's), with `sensor2lidar` as the camera's
  pose on the car: OpenScene's lidar frame is the ego frame.
* **Which pixels.** The model reads the current frame's four cameras (b0, f0, l0, r0) and the
  previous frame's front camera, nothing else (`drive_jepa_features.py`, `bev_feature_build.py`).
  The node decodes all four cameras of all four frames; the frames the model never reads are left
  empty here, which changes no tensor the model reads (checked against paddock's own plans by
  `checks fidelity`).
* **Decoding in parallel.** The five reads are independent, so `agent_input` can hand them to a
  thread pool (`decode_pool`); PIL releases the GIL while it decodes and resizes. Each image goes
  back to the camera slot it came from, by position, so the input does not depend on the number of
  workers or the order the reads finish in (`checks workers` compares 1 and 4 workers bit for bit).

The geometry constants and the crop rule are paddock's (`drivejepa/kitscenes_adapter.py`
`_crop_box`, `target_intrinsics`), repeated here because gallops does not import paddock's
adapters; `checks fidelity` is what holds the two to the same answer.

Runs in Drive-JEPA's interpreter (Python 3.9), with navsim on the path (`model.enter`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict
from gallops.memory.ticks import CURRENT_CAMERAS, FRONT, history_indices

#: OpenScene's camera, which the node's "openscene" geometry warps every image onto.
TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080
TARGET_FOCAL = 1545.0
TARGET_CU, TARGET_CV = 960.0, 560.0

#: NAVSIM one-hot [left, straight, right, unknown] position -> paddock Command value, and back.
#: paddock's Command: RIGHT 0, LEFT 1, STRAIGHT 2, FOLLOW_REFERENCE 3.
_COMMAND_FROM_ONE_HOT = (1, 2, 0, 3)
_ONE_HOT_FROM_COMMAND = {0: [0, 0, 1, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 0, 1]}

_ALL_SLOTS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
#: navsim slot name of each log camera the model reads.
_SLOT = {"CAM_B0": "cam_b0", "CAM_F0": "cam_f0", "CAM_L0": "cam_l0", "CAM_R0": "cam_r0"}


def yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def pose_matrix(pose) -> np.ndarray:
    """(x, y, heading) -> a 4x4 rigid transform (paddock navsim/convert.py `pose_matrix`)."""
    x, y, heading = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(heading), np.sin(heading)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    m[0, 3], m[1, 3] = x, y
    return m


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(x, y, heading) of `pose` in `reference`'s frame (paddock drivejepa/policy.py)."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], yaw_of(delta)], dtype=np.float64)


def command_round_trip(one_hot) -> list[int]:
    """The driving command as the node hands it to the model, after paddock's `Command`."""
    command = np.asarray(one_hot).ravel()
    hot = int(np.argmax(command)) if command.size else 3
    return _ONE_HOT_FROM_COMMAND[_COMMAND_FROM_ONE_HOT[hot] if hot < 4 else 3]


def crop_box(k: np.ndarray) -> tuple[int, int, int, int]:
    """The crop window mapping a pinhole onto OpenScene's (paddock kitscenes_adapter `_crop_box`)."""
    focal, cu, cv = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
    scale = TARGET_FOCAL / focal
    left = cu - TARGET_CU / scale
    top = cv - TARGET_CV / scale
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + TARGET_WIDTH / scale)),
        int(round(top + TARGET_HEIGHT / scale)),
    )


def target_intrinsics() -> np.ndarray:
    return np.array(
        [[TARGET_FOCAL, 0.0, TARGET_CU], [0.0, TARGET_FOCAL, TARGET_CV], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def load_image(path: Path, k: np.ndarray, blank: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One camera's pixels and intrinsics as the node gives them to the model ("openscene").

    `blank` replaces the pixels with zeros after the warp (the sight check).
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    cropped = image.crop(crop_box(k)).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
    pixels = np.asarray(cropped)
    if blank:
        pixels = np.zeros_like(pixels)
    return pixels, target_intrinsics().astype(np.float64)


def find_image(roots: list, data_path: str) -> Path:
    """The first root holding `data_path`; raises if none does."""
    for root in roots:
        candidate = Path(root) / data_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{data_path} is in none of {[str(r) for r in roots]}")


def decode_pool(workers: int) -> Executor | None:
    """A pool for the image reads, or None for reading them in the calling thread (1 worker)."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="decode") if workers > 1 else None


def load_all(
    jobs: Sequence[tuple],
    pool: Executor | None = None,
    load: Callable[..., tuple[np.ndarray, np.ndarray]] = load_image,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`load(*job)` for every job, results in the jobs' order whatever order the reads finish in."""
    if pool is None:
        return [load(*job) for job in jobs]
    futures = [pool.submit(load, *job) for job in jobs]
    return [future.result() for future in futures]


def agent_input(frames: list, frame_idx: int, roots: list, blank: bool = False, pool: Executor | None = None):
    """The NAVSIM AgentInput paddock's node would build for the moment at `frame_idx`.

    `frames` is the log pickle's list; `roots` the image roots, searched in order; `pool` an
    optional `decode_pool` for the image reads (the result is the same either way).
    """
    from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    from nuplan.common.actor_state.state_representation import StateSE2
    from pyquaternion import Quaternion

    history = [frames[i] for i in history_indices(frame_idx)]
    # NAVSIM's own conversion to the current frame (`AgentInput.from_scene_dict_list`), then the node's.
    global_poses = []
    for f in history:
        translation = f["ego2global_translation"]
        yaw = Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0]
        global_poses.append([translation[0], translation[1], yaw])
    global_poses = np.array(global_poses, dtype=np.float64)
    local = convert_absolute_to_relative_se2_array(StateSE2(*global_poses[-1]), global_poses)
    # navsim stores these float32; the proxy reads them into float64 matrices
    local = np.asarray(local, dtype=np.float32)
    current = pose_matrix(local[-1])

    statuses, jobs, wanted_by_frame = [], [], []
    for j, f in enumerate(history):
        dyn = np.asarray(f["ego_dynamic_state"], dtype=np.float32)
        statuses.append(
            EgoStatus(
                ego_pose=relative_pose(pose_matrix(local[j]), current),
                ego_velocity=np.array([float(dyn[0]), 0.0], dtype=np.float32),
                ego_acceleration=np.array([float(dyn[2]), 0.0], dtype=np.float32),
                driving_command=np.array(command_round_trip(f["driving_command"]), dtype=np.int64),
            )
        )
        wanted = CURRENT_CAMERAS if j == len(history) - 1 else (FRONT,) if j == len(history) - 2 else ()
        wanted_by_frame.append(wanted)
        for name in wanted:
            cam = f["cams"][name]
            k = np.asarray(cam["cam_intrinsic"], dtype=np.float64)[:3, :3]
            # find_image runs here, so a missing image raises before any read is queued
            jobs.append((find_image(roots, cam["data_path"]), k, blank))
    loaded = iter(load_all(jobs, pool))

    cameras = []
    for f, wanted in zip_strict(history, wanted_by_frame):
        slots = {slot: Camera() for slot in _ALL_SLOTS}
        for name in wanted:
            cam = f["cams"][name]
            pixels, intrinsics = next(loaded)
            slots[_SLOT[name]] = Camera(
                image=pixels,
                sensor2lidar_rotation=np.asarray(cam["sensor2lidar_rotation"], dtype=np.float32),
                sensor2lidar_translation=np.asarray(cam["sensor2lidar_translation"], dtype=np.float32),
                intrinsics=intrinsics.astype(np.float32),
            )
        cameras.append(Cameras(**slots))
    return AgentInput(ego_statuses=statuses, cameras=cameras, lidars=[])
