"""Run 3B v2 (paddock-ops/run3b/CLAUDE.md) as a paddock Policy, behind the wajepa_dj node's feed.

Feed: exactly run 3A's (`wajepa_dj_run3a`): the four cameras b0 f0 l0 r0, each rescaled to 1920x1080 (non-16:9 padded,
never stretched; `wajepa_dj.policy`), the NAVSIM ``AgentInput`` with four frames 0.5 s apart, run 3A's light features
(`DriveJEPA4VFeatureBuilder`: Drive-JEPA's 768x432 lidar2img, ego_status (4, 11)) and the t - 0.5 s and t frames of each
camera through WA-JEPA's preprocessing (``img[28:-28]`` -> 512x256, ImageNet norm).

Model and pick: run3b/v2/model/run3b_infer.py (`load_run3b`, `Picker`), the code the NAVSIM navtest agent runs too: the
trainer's net with every trainable tensor from the checkpoint, the decoder's own agents, and eval_v2's "pick (expert metres
head)": argmax over the 32 proposals of the T4 fact score with the expert's-metres head as R. RUN3B_SCORE_DOMAIN: navsim
(comfort term on, as for NAVSIM training samples; default) or lead (off, as for LEAD/CARLA samples).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.models.wajepa_dj.policy import WajepaDjPolicy


class _Features:
    """Light features + the encoder's input clips, as one builder (`DriveJepaPolicy.forward` calls compute_features)."""

    def __init__(self, light, clips_fn) -> None:
        self.light, self.clips_fn = light, clips_fn

    def compute_features(self, agent_input) -> dict[str, Any]:
        features = dict(self.light.compute_features(agent_input))
        features["clips"] = self.clips_fn(agent_input)  # (4, 3, 2, 256, 512), b0 f0 l0 r0
        return features


class Run3bV2Policy(WajepaDjPolicy):
    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        super().__init__(checkpoint=checkpoint or os.environ.get("RUN3B_CHECKPOINT"), device=device)
        self.domain = os.environ.get("RUN3B_SCORE_DOMAIN", "navsim")
        self.ckpt_info: dict[str, Any] = {}

    def load(self) -> None:
        if not self.checkpoint:
            raise RuntimeError("no checkpoint: set RUN3B_CHECKPOINT (model.yaml supplies it)")
        if self.momentum is not None:
            raise RuntimeError("DRIVEJEPA_MOMENTUM=on does not apply to run 3B (its pick is the fact score)")
        model_dir = str(Path(os.environ["RUN3B_CODE"]) / "model")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        import run3b_infer as RI  # noqa: E402
        from navsim.agents.drive_jepa_run3a.drive_jepa_config import DriveJEPAConfig
        from navsim.agents.drive_jepa_run3a.drive_jepa_features import DriveJEPA4VFeatureBuilder

        self.net, info = RI.load_run3b(self.checkpoint, self.device)
        self.agent = RI.Picker(self.net, info["P"], self.domain)
        self.feature_builder = _Features(DriveJEPA4VFeatureBuilder(DriveJEPAConfig()), RI.clips_from_agent_input)
        self.ckpt_info = {**info, "score_domain": self.domain, "code": RI.__file__}
        self.lesion = ""
        self.reset()

    def _pick(self, output: dict[str, Any]) -> dict[str, Any]:
        scores = output["pdm_score"][0].float().cpu().numpy()
        index = int(np.argmax(scores))
        return {"chosen_index": index, "chosen_score": float(scores[index]), "momentum_fed": False,
                "expert_metres_4s": float(output["expert_metres"][0, -1]), "ntg": bool(output["ntg"][0])}

    def diagnostics(self) -> dict[str, Any]:
        return {**super().diagnostics(), "agent_package": "run3b_v2 (run3b/v2/model/run3b_infer.py)", **self.ckpt_info}


POLICY = Run3bV2Policy
