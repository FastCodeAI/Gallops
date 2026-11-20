"""Run 3B v2 behind this worktree's wajepa_dj node (rerank-native: the node run 3A's HUGSIM and Bench2Drive runs were made
with: identity lidar2ego feed, WAJEPA_DJ_FOLLOW command fix, per-call records). Same model and pick code as the gate's
adapter (worktrees/gate-oct27 .../run3b_v2): run3b/v2/model/run3b_infer.py `load_run3b` + `Picker` (eval_v2's "pick
(expert metres head)"; RUN3B_SCORE_DOMAIN navsim / lead).

Differences from the gate's copy, all from this worktree's parents: `_cameras(frame, lidar2ego)`; the command fix
(WAJEPA_DJ_FOLLOW=straight: FOLLOW_REFERENCE fed as STRAIGHT, as run 3A on HUGSIM / Bench2Drive); `record_call` writes
run 3B's own per-call record (32 proposals, the fact score, its a / b parts, the pick, the expert's-metres head) when
DRIVEJEPA_RECORD_DIR is set (set DRIVEJEPA_RECORD_FEATS=0: there is no Drive-JEPA scorer module to hook)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.models.wajepa_dj.policy import FOLLOW, WajepaDjPolicy


class _Features:
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
        if os.environ.get("DRIVEJEPA_MOMENTUM", "") == "1":
            raise RuntimeError("DRIVEJEPA_MOMENTUM does not apply to run 3B (its pick is the fact score)")
        if os.environ.get("DRIVEJEPA_RECORD_DIR") and os.environ.get("DRIVEJEPA_RECORD_FEATS", "1") != "0":
            raise RuntimeError("set DRIVEJEPA_RECORD_FEATS=0 with DRIVEJEPA_RECORD_DIR (no Drive-JEPA scorer to hook)")
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
        self.follow = os.environ.get("WAJEPA_DJ_FOLLOW", "unknown")
        if self.follow not in FOLLOW:
            raise ValueError(f"WAJEPA_DJ_FOLLOW={self.follow!r}; expected one of {FOLLOW}")
        self.reset()

    def record_call(self, obs, output: dict[str, Any], **extra: Any) -> None:
        rec_dir = os.environ.get("DRIVEJEPA_RECORD_DIR")
        if not rec_dir:
            return
        if getattr(self, "_record_fh", None) is None:
            os.makedirs(rec_dir, exist_ok=True)
            self._record_fh = open(os.path.join(rec_dir, f"calls_{os.getpid()}_e{getattr(self, '_episode', 0):03d}.jsonl"), "a")
        f = lambda k, d: np.round(output[k][0].float().cpu().numpy(), d).tolist()  # noqa: E731
        rec = {"timestamp_us": int(obs.timestamp_us), "speed": float(obs.speed), "accel": float(obs.accel),
               "command": int(obs.command), "proposals": f("proposals", 3), "pdm_score": f("pdm_score", 6),
               "score_a": f("score_a", 6), "score_b": f("score_b", 6), "expert_metres": f("expert_metres", 3),
               "model_pick": int(output["pdm_score"][0].float().argmax()), "diagnostics": self.diagnostics(), **extra}
        self._record_fh.write(json.dumps(rec) + "\n")
        self._record_fh.flush()

    def _pick(self, output: dict[str, Any]) -> dict[str, Any]:
        scores = output["pdm_score"][0].float().cpu().numpy()
        index = int(np.argmax(scores))
        return {"chosen_index": index, "chosen_score": float(scores[index]), "momentum_fed": False,
                "expert_metres_4s": float(output["expert_metres"][0, -1]), "ntg": bool(output["ntg"][0])}

    def diagnostics(self) -> dict[str, Any]:
        return {**super().diagnostics(), "agent_package": "run3b_v2 (run3b/v2/model/run3b_infer.py)", **self.ckpt_info}


POLICY = Run3bV2Policy
