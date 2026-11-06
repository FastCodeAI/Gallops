"""Phase 2 Drive-JEPA (paddock-ops/b2d_drivejepa/phase2): the stock drivejepa node, with the checkpoint trained on Bench2Drive, and FOLLOW_REFERENCE sent as STRAIGHT.
The model never saw the "unknown" command in training (spec: FOLLOW sent as straight, as wajepa_dj/policy.py does)."""

from __future__ import annotations

from paddock.adapters.models.drivejepa.policy import DriveJepaPolicy


class DriveJepaP2Policy(DriveJepaPolicy):
    def _select_history(self):
        frames = super()._select_history()
        for frame in frames:  # history frames are kept, so the rewrite is done once per frame
            if frame["command"] == 3:  # FOLLOW_REFERENCE
                frame["command"] = 2  # STRAIGHT
        return frames


POLICY = DriveJepaP2Policy
