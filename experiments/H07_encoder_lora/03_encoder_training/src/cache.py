"""Frame cache: every 5th front-camera frame as the 256x512x3 uint8 array that
feeds ToTensor in the driving feature builder.

The baseline builds a camera feature as (drive_jepa_features.py:48, :51, :58)

    f0 = cameras[-1].cam_f0.image[28:-28]
    resized_image_1 = cv2.resize(f0, (512, 256))      # uint8, (256, 512, 3)
    tensor_image_1 = transforms.ToTensor()(resized_image_1)

and the image itself comes from KITScenesSceneAdapter._load_camera
(kitscenes_adapter.py:242-257: crop, bilinear resize, np.asarray -> uint8).
We store `resized_image_*`, i.e. the last array before ToTensor; ToTensor of a
uint8 HWC array is a pure function (permute, /255), so the float tensor is
reproducible from it exactly.  Gate A checks that against s1 pairs.build_pair.

Layout: shard_NNN.bin = raw frames back to back (393,216 bytes each), and
shard_NNN.json = {"frames": [[scene, t], ...], "sha256": ...} written after the
.bin is complete, so a shard with a .json is done (resume skips it).
index.json maps "scene:t" -> [shard, slot].
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

import pairs as PR  # s1: scene_adapter, FRONT, PAIR_GAP, _normalise, scene_num_frames
from s2common import CACHE, FRAME_STEP

H, W = 256, 512
FRAME_BYTES = H * W * 3
_to_tensor = transforms.ToTensor()


def frame_array(scene_path: str, t: int) -> np.ndarray:
    """drive_jepa_features.py:48+51 applied to kitscenes_adapter._load_camera."""
    image = PR.scene_adapter(scene_path)._load_camera(PR.FRONT, t).image
    out = cv2.resize(image[28:-28], (512, 256))
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)
    return out


def pair_from_arrays(older: np.ndarray, newer: np.ndarray) -> torch.Tensor:
    """The rest of s1 pairs.build_pair: ToTensor (drive_jepa_features.py:58-59),
    ImageNet Normalize (drive_jepa_model.py:42-47), stacked [older, newer]."""
    return torch.stack([PR._normalise(_to_tensor(older)), PR._normalise(_to_tensor(newer))], dim=0)


def frames_for_scene(n_frames: int) -> list[int]:
    return list(range(0, n_frames, FRAME_STEP))


def pair_ends(n_frames: int) -> list[int]:
    """t multiple of 5, t >= 5, t < n_frames (so t-5 exists too)."""
    return list(range(FRAME_STEP, n_frames, FRAME_STEP))


class FrameCache:
    def __init__(self, root: Path = CACHE):
        self.root = root
        self.index = {k: tuple(v) for k, v in json.loads((root / "index.json").read_text())["index"].items()}
        self._maps: dict[int, np.memmap] = {}

    def _shard(self, s: int) -> np.memmap:
        if s not in self._maps:
            path = self.root / f"shard_{s:03d}.bin"
            n = path.stat().st_size // FRAME_BYTES
            self._maps[s] = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, H, W, 3))
        return self._maps[s]

    def frame(self, scene: str, t: int) -> np.ndarray:
        s, slot = self.index[f"{scene}:{t}"]
        return np.array(self._shard(s)[slot])  # copy out of the map

    def pair(self, scene: str, t: int) -> torch.Tensor:
        return pair_from_arrays(self.frame(scene, t - FRAME_STEP), self.frame(scene, t))


class CachedPairs(torch.utils.data.Dataset):
    """(pair, index) for every (scene, t) pair end of the listed scenes."""

    def __init__(self, scenes: list[str], cache_root: Path = CACHE):
        self.root = cache_root
        meta = json.loads((cache_root / "index.json").read_text())
        n_frames = meta["scene_frames"]
        self.items = [(s, t) for s in scenes for t in pair_ends(n_frames[s])]
        self._cache = None  # opened lazily per worker

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        if self._cache is None:
            self._cache = FrameCache(self.root)
        s, t = self.items[i]
        return self._cache.pair(s, t), i
