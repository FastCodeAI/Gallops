"""The model's inputs for a nuScenes keyframe: swerve_data's TickDataset, fed from the tables.

`swerve_data.TickDataset` rebuilds Drive-JEPA's AgentInput from an episode directory (front
frames by timestamp, ego poses by timestamp) -- the layout `nuscenes_real` writes. The aux
pipeline keeps the same keyframes in `nusc.extract` instead, so this subclass answers the one
question TickDataset asks of a directory, "which frames and poses does this tick's episode
have", from the extract, and inherits everything else: the history rule, the camera slots and
extrinsics, the 28-row crop and resize, the ego-status vector and the vendor's own feature
builder. One input path, so the preflight checks one thing.

A tick is a row of `nusc.ticks`: it carries `scene`, `timestamp`, `speed`, `accel` and
`command`, which is all TickDataset reads. Calibration comes from GALLOPS_SCENE_CALIB, the
directory `nusc extract` writes, set by `use_calibration`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from gallops.train.swerve_data import TickDataset


def use_calibration(directory: Path) -> None:
    """Point `swerve_data.scene_calibration` at an extract's calibration, before the first tick.

    Without it, each new scene costs a parse of the 1.3 GB sample_data table.
    """
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"{directory}: no calibration directory (written by `nusc extract`)")
    os.environ["GALLOPS_SCENE_CALIB"] = str(directory)


class TablesTickDataset(TickDataset):
    """TickDataset whose episodes are scenes of a `nusc.extract`, not directories."""

    def __init__(self, ticks: list, device: str, scenes: dict, nuscenes_root: str):
        super().__init__(ticks, device, nuscenes_root)
        self._scenes = scenes

    def _episode(self, tick: dict) -> dict:
        name = tick["scene"]
        if name not in self._episodes:
            frames = self._scenes[name]["keyframes"]
            self._episodes[name] = {
                "dir": None,
                "poses": {int(f["timestamp"]): np.asarray(f["ego"]) for f in frames},
                "frames": {int(f["timestamp"]): Path(self.nuscenes_root) / f["file"] for f in frames},
                "scene": name,
            }
        return self._episodes[name]
