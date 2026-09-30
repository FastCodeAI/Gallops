"""Rebuild the model's own inputs from a collected tick, so training sees what it saw.

A labelled tick names one frame. Drive-JEPA needs more: two front frames half a
second apart, a four-frame ego-status history, and calibration for four camera
slots. All of it is recoverable from the episode directory the engine wrote --
frames are keyed by timestamp, and so are the ego poses.

Two things here are easy to get wrong, and both are checked by `verify`, which
rebuilds a recorded tick and compares the model's 32 proposals against the ones
the dump recorded at the time. If the reconstruction is faithful they agree to
floating-point noise; if it is not, the number moves and training would be
learning from inputs the model never actually saw.

**Only the front camera's pixels exist, and only the front camera's pixels are
read.** `_get_camera_feature` uses `cam_f0` alone (the "stitch l0, f0, r0" comment
in the vendor file is stale), and the four-view tensor the feature builder also
assembles is read nowhere in the model. But the projection matrices ARE built from
all four slots in the order [b0, f0, l0, r0], skipping any slot with no image, and
the model then keeps index 1 -- so dropping the three unread cameras silently
shifts the front camera's projection to another slot. They are supplied as blank
images of the right shape for that reason alone: to hold the ordering. Nothing
reads their contents.

**The ego status is a history, not a state.** The model reads four frames at 0.5 s
spacing; giving it four copies of the current one makes a parked car of a moving
ego, which is the failure `min_input_hz` exists to prevent elsewhere.

Transcribed from paddock's `adapters/models/drivejepa/policy.py`, which built the
same AgentInput when these episodes were driven. The repos do not import each
other, so `verify` is what keeps the copy honest.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CAMERA_SLOTS = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")
NUSCENES_CAMERAS = {
    "cam_f0": "CAM_FRONT",
    "cam_l0": "CAM_FRONT_LEFT",
    "cam_r0": "CAM_FRONT_RIGHT",
    "cam_b0": "CAM_BACK",
}
HISTORY_FRAMES = 4
HISTORY_DT = 0.5
#: NeuroNCAP's Command enum -> NAVSIM's one-hot [left, straight, right, unknown].
COMMAND_ONE_HOT = {0: [0, 0, 1, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 0, 1]}


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`pose` expressed in `reference`'s frame, as (x, y, heading)."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], float(np.arctan2(delta[1, 0], delta[0, 0]))], dtype=np.float32)


@lru_cache(maxsize=32)
def scene_calibration(nuscenes_root: str, scene_name: str) -> dict:
    """Per-camera (camera2ego, intrinsics) for one scene, from nuScenes."""
    tables = Path(nuscenes_root) / "v1.0-trainval"
    sensors = {s["token"]: s["channel"] for s in json.loads((tables / "sensor.json").read_text())}
    calib = {c["token"]: c for c in json.loads((tables / "calibrated_sensor.json").read_text())}
    scene = next(s for s in json.loads((tables / "scene.json").read_text()) if s["name"] == scene_name)
    first = scene["first_sample_token"]
    out = {}
    for record in json.loads((tables / "sample_data.json").read_text()):
        if record["sample_token"] != first or not record["is_key_frame"]:
            continue
        channel = sensors[calib[record["calibrated_sensor_token"]]["sensor_token"]]
        if channel in NUSCENES_CAMERAS.values() or channel == "LIDAR_TOP":
            entry = calib[record["calibrated_sensor_token"]]
            pose = np.eye(4)
            pose[:3, :3] = _quat(entry["rotation"])
            pose[:3, 3] = entry["translation"]
            intrinsics = entry["camera_intrinsic"]
            out[channel] = (
                pose,
                np.asarray(intrinsics, dtype=np.float32) if intrinsics else None,
            )
    return out


def _quat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (x * w + y * z), 1 - 2 * (x * x + y * y)],
        ]
    )


def episode_context(tick: dict, drivable: Path | None = None) -> tuple:
    """The actors, pose and map a tick's trajectories are graded against.

    One definition, used by the trainer and by the reach measurement, so a change
    to how a proposal is judged cannot land in one and not the other.
    """
    from gallops.expert.drivable import DrivableArea

    directory = Path(tick["frame"]).parent.parent
    poses = json.loads((directory / "ego_poses.json").read_text())
    actors = json.loads((directory / "actors.json").read_text())
    area = None
    if drivable:
        path = Path(drivable) / f"scene-{directory.parent.name.rsplit('-', 1)[-1]}.npz"
        if path.exists():
            area = DrivableArea.load(path)
    return (
        np.asarray(poses[str(int(tick["timestamp"]))]),
        [a for a in actors if a["is_target_actor"]],
        area,
    )


class TickDataset:
    """Labelled ticks, rebuilt into the AgentInput the model consumes."""

    def __init__(self, ticks: list[dict], device: str, nuscenes_root: str | None = None):
        import torch

        self.torch = torch
        self.ticks = ticks
        self.device = device
        if nuscenes_root is None:
            from gallops.config import SiteConfig

            nuscenes_root = SiteConfig.load().resolve("{data.nuscenes}")
        self.nuscenes_root = str(nuscenes_root)
        self._episodes: dict[str, dict] = {}
        self._bev_cache: dict[str, dict] = {}
        self._feature_builder = None

    def __len__(self) -> int:
        return len(self.ticks)

    def _episode(self, tick: dict) -> dict:
        directory = Path(tick["frame"]).parent.parent
        key = str(directory)
        if key not in self._episodes:
            poses = json.loads((directory / "ego_poses.json").read_text())
            self._episodes[key] = {
                "dir": directory,
                "poses": {int(k): np.asarray(v) for k, v in poses.items()},
                "frames": {int(p.stem): p for p in (directory / "CAM_FRONT").glob("*.jpg")},
                "scene": "scene-" + directory.parent.name.rsplit("-", 1)[-1],
            }
        return self._episodes[key]

    def agent_input(self, tick: dict):
        from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
        from PIL import Image

        episode = self._episode(tick)
        stamps = sorted(episode["frames"])
        now = int(tick["timestamp"])
        calibration = scene_calibration(self.nuscenes_root, episode["scene"])

        chosen = []
        for back in range(HISTORY_FRAMES - 1, -1, -1):
            target = now - int(back * HISTORY_DT * 1e6)
            chosen.append(min(stamps, key=lambda s: abs(s - target)))

        current_pose = episode["poses"].get(chosen[-1])
        if current_pose is None:
            return None

        statuses, cameras = [], []
        for stamp in chosen:
            pose = episode["poses"].get(stamp, current_pose)
            statuses.append(
                EgoStatus(
                    ego_pose=relative_pose(pose, current_pose),
                    ego_velocity=np.array([tick["speed"], 0.0], dtype=np.float32),
                    ego_acceleration=np.array([tick["accel"], 0.0], dtype=np.float32),
                    driving_command=np.array(COMMAND_ONE_HOT[int(tick["command"])], dtype=np.int64),
                )
            )
            image = np.asarray(Image.open(episode["frames"][stamp]).convert("RGB"))
            blank = np.zeros_like(image)
            # neuroNCAP sends the REAL lidar-to-ego calibration, not identity --
            # the engine reads it from nuScenes (`get_lidar_calibration`). Assuming
            # identity shifts every camera's projection by the lidar's mounting
            # offset, about 1.8 m up and 0.94 m forward, and the proposals move
            # metres. That is what the fidelity check caught.
            ego2lidar = np.linalg.inv(calibration["LIDAR_TOP"][0])
            slots = {}
            for slot in CAMERA_SLOTS:
                channel = NUSCENES_CAMERAS[slot]
                cam2ego, intrinsics = calibration[channel]
                sensor2lidar = ego2lidar @ cam2ego
                slots[slot] = Camera(
                    # only cam_f0's pixels are ever read; the other three are here
                    # to hold the projection ordering, nothing more
                    image=image if slot == "cam_f0" else blank,
                    sensor2lidar_rotation=sensor2lidar[:3, :3].astype(np.float32),
                    sensor2lidar_translation=sensor2lidar[:3, 3].astype(np.float32),
                    intrinsics=intrinsics,
                )
            empty = Camera()
            cameras.append(
                Cameras(
                    cam_f0=slots["cam_f0"],
                    cam_l0=slots["cam_l0"],
                    cam_l1=empty,
                    cam_l2=empty,
                    cam_r0=slots["cam_r0"],
                    cam_r1=empty,
                    cam_r2=empty,
                    cam_b0=slots["cam_b0"],
                )
            )
        return AgentInput(ego_statuses=statuses, cameras=cameras, lidars=[])

    def _builder(self):
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
        from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
            DriveJEPAFeatureBuilder,
        )

        if self._feature_builder is None:
            self._feature_builder = DriveJEPAFeatureBuilder(DriveJEPAConfig())
        return self._feature_builder

    def build(self, agent_input, scene: str, fast: bool = True) -> dict:
        """The model's feature dict, without rebuilding what cannot have changed.

        `_get_bev_feature` casts all four camera images to float32, normalises,
        rescales and stacks them -- about 1.2 s a tick, and the four-view tensor
        it produces is read nowhere: the encoder takes `lidar2img` and `img_shape`
        out of that dict and the model builds its own input from cam_f0 alone.
        Worse, three of the four images are the blanks that exist only to hold the
        projection ordering, so most of that second is spent normalising zeros.

        `lidar2img` and `img_shape` depend on the scene's calibration and the
        frame size, neither of which varies within a scene, so the slow path runs
        once per scene and every later tick reuses it. `verify_fast_path` checks
        the two agree, and the fidelity check against a recorded dump is what says
        the whole reconstruction is faithful in the first place.
        """
        builder = self._builder()
        if not fast:
            return builder.compute_features(agent_input)
        if scene not in self._bev_cache:
            self._bev_cache[scene] = builder.compute_features(agent_input)
        cached = self._bev_cache[scene]
        built = {
            "camera_feature": cached["camera_feature"],
            "img_shape": cached["img_shape"],
            "lidar2img": cached["lidar2img"],
        }
        built.update(builder._get_camera_feature(agent_input))
        # The one piece restated rather than imported, because the vendor builds
        # it inline inside compute_features: a per-frame concat of pose, velocity,
        # acceleration and the one-hot command. verify_fast_path compares it.
        built["ego_status"] = self.torch.stack(
            [
                self.torch.cat(
                    [
                        self.torch.tensor(status.ego_pose, dtype=self.torch.float32),
                        self.torch.tensor(status.ego_velocity, dtype=self.torch.float32),
                        self.torch.tensor(status.ego_acceleration, dtype=self.torch.float32),
                        self.torch.tensor(status.driving_command, dtype=self.torch.float32),
                    ],
                    dim=-1,
                )
                for status in agent_input.ego_statuses
                if status is not None
            ]
        )
        return built

    def features(self, tick: dict, fast: bool = True):
        agent_input = self.agent_input(tick)
        if agent_input is None:
            return None
        built = self.build(agent_input, self._episode(tick)["scene"], fast=fast)
        return {
            key: (value.to(self.device)[None] if hasattr(value, "to") else value)
            for key, value in built.items()
        }


def verify_fast_path(dataset, ticks, torch, model=None) -> None:
    """The fast feature build must not change what the model computes.

    An optimisation that shifts the model's input by a hair is the worst kind of
    bug here: it trains fine and the numbers are quietly about something else.

    Every key the encoder reads is compared exactly. `camera_feature` is the one
    that legitimately differs -- it is the four-view tensor the fast path reuses
    from another tick -- and rather than assert it is unused, passing a `model`
    proves it: the 32 proposals must come out bit-identical either way.
    """
    reused = "camera_feature"
    for tick in ticks:
        slow = dataset.features(tick, fast=False)
        fast = dataset.features(tick, fast=True)
        assert set(slow) == set(fast), f"keys differ: {set(slow) ^ set(fast)}"
        for key in slow:
            if key == reused or not torch.is_tensor(slow[key]):
                continue
            if not torch.equal(slow[key], fast[key]):
                worst = (slow[key].float() - fast[key].float()).abs().max()
                raise SystemExit(f"{key} differs by up to {float(worst):.3e}")
    print(f"  every key the encoder reads matches exactly on {len(ticks)} ticks")

    if model is None:
        print(f"  NOT checked: that {reused} is unused -- pass a model to prove it")
        return
    model.eval()
    with torch.no_grad():
        for tick in ticks:
            slow = model(dataset.features(tick, fast=False))["proposals"]
            fast = model(dataset.features(tick, fast=True))["proposals"]
            worst = float((slow - fast).abs().max())
            if worst > 0:
                raise SystemExit(f"the 32 proposals move by {worst:.3e}: {reused} IS read")
    print(f"  and the 32 proposals are bit-identical, so {reused} is genuinely unread")
