"""Run 3B v2 as a NAVSIM AbstractAgent for navsim_v2's run_pdm_score.py (navtest EPDMS, the procedure run 3A's 87.30 and
run 2's numbers came from: same devkit tree, same metric_cache_v2, 4 process workers, no momentum-aware selection).

Inputs as run 3A's agent with use_wajepa_tokens (drive_jepa_run3a/drive_jepa_agent.py get_sensor_config): cameras f0 l0 r0
b0 at frames [2, 3] (t - 0.5 s, t), four ego statuses; features = DriveJEPA4VFeatureBuilder (lidar2img, ego_status) +
the encoder's clips (run3b_infer.clips_from_agent_input). Model and pick: run3b/v2/model/run3b_infer.py (shared with the
Paddock adapter). Score domain navsim (comfort term on)."""
import os
import sys
from typing import Dict

import torch

sys.path.insert(0, "{runs}/run3b/v2/model")
import run3b_infer as RI  # noqa: E402

from navsim.agents.abstract_agent import AbstractAgent  # noqa: E402
from navsim.agents.drive_jepa_run3a.drive_jepa_config import DriveJEPAConfig  # noqa: E402
from navsim.agents.drive_jepa_run3a.drive_jepa_features import DriveJEPA4VFeatureBuilder  # noqa: E402
from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


class ClipsBuilder(AbstractFeatureBuilder):
    def get_unique_name(self) -> str:
        return "run3b_clips"

    def compute_features(self, agent_input) -> Dict[str, torch.Tensor]:
        return {"clips": RI.clips_from_agent_input(agent_input)}


class Run3BV2Agent(AbstractAgent):
    def __init__(self, checkpoint_path: str, score_domain: str = "navsim", trajectory_sampling=None):
        super().__init__(trajectory_sampling or TrajectorySampling(time_horizon=4, interval_length=0.5))
        self._checkpoint_path = checkpoint_path
        self._domain = score_domain
        self._config = DriveJEPAConfig()
        self._picker = None

    def name(self) -> str:
        return "run3b_v2_agent"

    def initialize(self) -> None:
        net, info = RI.load_run3b(self._checkpoint_path, "cuda")
        self._picker = RI.Picker(net, info["P"], self._domain)
        self._info = info

    def get_sensor_config(self):
        return SensorConfig(cam_f0=[2, 3], cam_l0=[2, 3], cam_l1=[], cam_l2=[], cam_r0=[2, 3], cam_r1=[], cam_r2=[],
                            cam_b0=[2, 3], lidar_pc=[])

    def get_feature_builders(self):
        return [DriveJEPA4VFeatureBuilder(config=self._config), ClipsBuilder()]

    def get_target_builders(self):
        return []

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._picker.forward(features)
