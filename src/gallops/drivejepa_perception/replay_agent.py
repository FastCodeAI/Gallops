"""The agents navsim's run_pdm_score.py instantiates for Part B (recipes/drivejepa_perception/navsim_config).

Imported only inside navsim's runs (`navtest score`, `navtest live`): it needs navsim and torch.

- **`ReplayAgent`** returns, for each scene, the plan a plan store holds for its token
  (`navtest.write_plans`), so the official scorer scores a modified model's plans exactly as it would the
  release's (docs/drivejepa-perception-sidecar/part-b/B-0-foundations.md, Q5). It needs the scene for the
  token (`requires_scene`, as navsim's HumanAgent). With `dump_dir`, each worker writes every plan it handed
  over, so `navtest score` can hold all of them to the store after the run.
- **`LiveReleaseAgent`** is the release itself, run by navsim as it ran for 87.78 (no momentum), with each
  token's plan, 32 scores and pick dumped beside the score: the live half of the fidelity gate (B0-G9).

A failed dump is counted and the run goes on: `run_pdm_score.py` crashes on any failed token, and the
replay's coverage is checked before the run, the dumps' after it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def _dump(dump_dir, record: dict) -> None:
    if dump_dir is None:
        return
    path = Path(dump_dir) / f"agent-{os.getpid()}.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


class ReplayAgent(AbstractAgent):
    """The stored plan for each token; nothing is computed."""

    requires_scene = True

    def __init__(self, plans: str, trajectory_sampling: TrajectorySampling, dump_dir=None):
        super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        from gallops.drivejepa_perception.navtest import load_plans

        self._plans = load_plans(Path(plans))
        self._dump_dir = dump_dir

    def name(self) -> str:
        return "gallops_replay"

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input, scene) -> Trajectory:
        token = scene.scene_metadata.initial_token
        poses = self._plans[token].astype(np.float32)
        try:
            _dump(self._dump_dir, {"token": token, "poses": poses.tolist()})
        except Exception:  # a dump never fails a token; the count is checked after the run
            pass
        return Trajectory(poses, self._trajectory_sampling)


class LiveReleaseAgent(DriveJEPAAgent):
    """The release, as `AbstractAgent.compute_trajectory` runs it, with its outputs dumped."""

    requires_scene = True

    def __init__(self, config, lr: float, checkpoint_path: str, dump_dir: str):
        super().__init__(config=config, lr=lr, checkpoint_path=checkpoint_path)
        self.requires_scene = True
        self._dump_dir = dump_dir

    def compute_trajectory(self, agent_input, scene) -> Trajectory:
        import torch

        # AbstractAgent.compute_trajectory, line for line, keeping the model's outputs.
        self.eval()
        features = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).cuda() for k, v in features.items()}
        features["past_ego_simulated_states"] = None
        features["metric_cache"] = None
        features["observation_interval"] = None
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()
        try:
            scores = predictions["pdm_score"][0].float().cpu().numpy()
            _dump(
                self._dump_dir,
                {
                    "token": scene.scene_metadata.initial_token,
                    "poses": poses.tolist(),
                    "pick": int(scores.argmax()),
                    "scores": scores.tolist(),
                },
            )
        except Exception:  # a dump never fails a token; the count is checked after the run
            pass
        return Trajectory(poses, self._trajectory_sampling)
