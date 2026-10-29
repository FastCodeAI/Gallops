"""Part B: frame pairs [now-0.5 s, now] built through the baseline's own code.

Nothing here re-implements an image step.  For a pair ending at frame t the
pixels go through, in order:
  1. KITScenesSceneAdapter._load_camera("camera_ring_front", i)   devkit kitscenes_adapter.py:242-257
     (principal-point crop :165-182, bilinear resize to 1920x1080 :247)
  2. DriveJEPAFeatureBuilder._get_camera_feature                   drive_jepa_features.py:38-65
     (rows [28:-28], cv2.resize to 512x256, ToTensor)
  3. DriveJEPAModel.make_transform (ImageNet Normalize)            drive_jepa_model.py:42-47
  4. stacked [older, newer] as in drive_jepa_model.py:53-57
which is exactly what the encoder receives while driving, minus GridMask
(simple_image_encoder.py:44-45), which the baseline never applies in eval and
which our own masking replaces in training.
"""

from __future__ import annotations

import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

import common as C

C.setup()

from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig  # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder  # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel  # noqa: E402
_adp = C.kit_adapter_module()
CAMERA_MAP, HISTORY_STRIDE, KITScenesSceneAdapter = _adp.CAMERA_MAP, _adp.HISTORY_STRIDE, _adp.KITScenesSceneAdapter

FRONT = CAMERA_MAP["cam_f0"]
PAIR_GAP = HISTORY_STRIDE  # 5 frames at 10 Hz = 0.5 s, the gap between history slots 2 and 3

_builder = DriveJEPAFeatureBuilder(DriveJEPAConfig())
# make_transform does not touch self (drive_jepa_model.py:42-47).
_normalise = DriveJEPAModel.make_transform(None)

# The adapter's ground offset only moves camera extrinsics (kitscenes_adapter.py:235-240),
# never pixels; fixing it skips the lidar read in __init__ (:203-205, :209-232).
_PIXEL_ONLY_GROUND_OFFSET = 2.2


@lru_cache(maxsize=64)
def scene_adapter(scene_path: str) -> KITScenesSceneAdapter:
    return KITScenesSceneAdapter(Path(scene_path), ground_offset=_PIXEL_ONLY_GROUND_OFFSET)


def build_pair(scene_path: str, t: int) -> torch.Tensor:
    """(2, 3, 256, 512) float32: [frame t-5, frame t], normalised."""
    adapter = scene_adapter(scene_path)
    older = adapter._load_camera(FRONT, t - PAIR_GAP)
    newer = adapter._load_camera(FRONT, t)
    # The builder reads cameras[-1].cam_f0 (newer) and cameras[-2].cam_f0 (older).
    agent_input = types.SimpleNamespace(
        cameras=[types.SimpleNamespace(cam_f0=older), types.SimpleNamespace(cam_f0=newer)]
    )
    feats = _builder._get_camera_feature(agent_input)
    cam_f_2 = _normalise(feats["camera_feature_2"])
    cam_f_1 = _normalise(feats["camera_feature_1"])
    return torch.stack([cam_f_2, cam_f_1], dim=0)


def scene_num_frames(scene_path: str) -> int:
    """Frames that have both a reference timestamp and a front-camera JPEG."""
    scene = Path(scene_path)
    n_ts = len(np.loadtxt(scene / "timestamp.reference.txt", ndmin=1))
    n_jpg = sum(1 for _ in (scene / FRONT).glob("*.jpg"))
    return min(n_ts, n_jpg)


class PairDataset(torch.utils.data.Dataset):
    """Every frame t >= 5 of every listed scene, as (pair, scene_id, t)."""

    def __init__(self, scene_ids: list[str], scene_paths: dict[str, str]):
        self.index: list[tuple[str, str, int]] = []
        self.frames: dict[str, int] = {}
        for sid in scene_ids:
            path = scene_paths[sid]
            n = scene_num_frames(path)
            self.frames[sid] = n
            self.index.extend((sid, path, t) for t in range(PAIR_GAP, n))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        sid, path, t = self.index[i]
        return build_pair(path, t), sid, t
