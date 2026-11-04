"""The auxiliary heads and their losses. Runs in Drive-JEPA's interpreter: torch, Python 3.9.

Two heads are upstream's and are switched on, not written: `MapHead` (a BEV semantic map) and
the box decoder `_agent_head` (30 boxes). `configure` sets the switches before the model is
built, so `Scorer.__init__` constructs them; the release has no weights for either.

Two are new, `SceneHeads`: the lead vehicle and the lane change. Each is one learned query that
attends over the refiner's 256 query features -- the pattern MapHead itself uses -- then an MLP.

Every head reads the same tensor: the query features after the last refinement pass, the input
`Scorer.forward` takes. That is where an auxiliary loss reaches the planner: those features are
made by the refiner, and the refiner is what trains.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gallops.drivejepa_aux import bev

#: Losses on targets in metres are scaled to about one: gap by 10 m, closing speed by 5 m/s.
GAP_SCALE_M = 10.0
CLOSING_SCALE_MPS = 5.0
#: The new heads' loss weight, against upstream's 1.0 / 0.1 / 1.0 for its own. Not tuned.
SCENE_HEAD_WEIGHT = 0.5


def configure(config) -> None:
    """Switch on upstream's two heads, with the lane-divider class added to the map."""
    config.bev_map = True
    config.bev_agent = True
    config.num_bev_classes = len(bev.CLASSES)


class SceneHeads(nn.Module):
    """Lead vehicle (present logit, gap / 10 m, closing speed / 5 m/s) and lane change (3 logits)."""

    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.attend = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.lead = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))
        self.lane = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))

    def forward(self, features: torch.Tensor) -> dict:
        query = self.queries[None].expand(features.shape[0], -1, -1)
        pooled, _ = self.attend(query, features, features)
        return {"lead": self.lead(pooled[:, 0]), "lane_change": self.lane(pooled[:, 1])}


def run_upstream_heads(scorer, features: torch.Tensor) -> dict:
    """Upstream's two heads on the query features, called exactly as `Scorer.forward` calls them
    in training mode (the model itself stays in eval mode; `step.check_heads` compares the two)."""
    out = {}
    if getattr(scorer, "bev_map", False):
        out["bev"] = scorer.map_head(features)
    if getattr(scorer, "bev_agent", False):
        agents = scorer._agent_head(None, features)
        out["agent_states"] = agents[:, :, :-1]
        out["agent_labels"] = agents[:, :, -1]
    return out


def collate(labels: list, device) -> dict:
    """Stack `nusc.keyframe_labels` dicts into the tensors the losses take."""

    def stack(key, dtype):
        return torch.as_tensor(
            np.stack([np.asarray(item[key]) for item in labels]), dtype=dtype, device=device
        )

    return {
        "bev": stack("bev", torch.long),
        "agent_states": stack("agent_states", torch.float32),
        "agent_valid": stack("agent_valid", torch.bool),
        "lead_present": stack("lead_present", torch.float32),
        "lead_gap": stack("lead_gap", torch.float32),
        "lead_closing": stack("lead_closing", torch.float32),
        "lane_change": stack("lane_change", torch.long),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum() if mask.any() else values.sum() * 0.0


def losses(outputs: dict, labels: dict, config) -> dict:
    """Each auxiliary loss, unweighted. A head with no target in the batch costs zero."""
    from navsim.agents.transfuser.transfuser_loss import _agent_loss

    out = {}
    if "bev" in outputs:
        if (labels["bev"] != bev.IGNORE).any():
            out["bev"] = F.cross_entropy(outputs["bev"], labels["bev"], ignore_index=bev.IGNORE)
        else:
            out["bev"] = outputs["bev"].sum() * 0.0
    if "agent_states" in outputs:
        # upstream's Hungarian loss, as the agent's own compute_loss calls it
        out["agent_class"], out["agent_box"] = _agent_loss(
            {"agent_states": labels["agent_states"], "agent_labels": labels["agent_valid"]},
            {"agent_states": outputs["agent_states"], "agent_labels": outputs["agent_labels"]},
            config,
        )
    if "lead" in outputs:
        lead = outputs["lead"]
        present = labels["lead_present"]
        out["lead_present"] = F.binary_cross_entropy_with_logits(lead[:, 0], present)
        has_gap = present > 0.5
        has_closing = has_gap & torch.isfinite(labels["lead_closing"])
        gap = torch.nan_to_num(labels["lead_gap"]) / GAP_SCALE_M
        closing = torch.nan_to_num(labels["lead_closing"]) / CLOSING_SCALE_MPS
        out["lead_gap"] = _masked_mean(F.smooth_l1_loss(lead[:, 1], gap, reduction="none"), has_gap)
        out["lead_closing"] = _masked_mean(
            F.smooth_l1_loss(lead[:, 2], closing, reduction="none"), has_closing
        )
    if "lane_change" in outputs:
        target = labels["lane_change"]
        out["lane_change"] = (
            F.cross_entropy(outputs["lane_change"], target, ignore_index=-1)
            if (target >= 0).any()
            else outputs["lane_change"].sum() * 0.0
        )
    return out


def weights(config) -> dict:
    """The weight of each loss in the total: upstream's for its heads, SCENE_HEAD_WEIGHT for ours."""
    return {
        "bev": config.bev_semantic_weight,
        "agent_class": config.agent_class_weight,
        "agent_box": config.agent_box_weight,
        "lead_present": SCENE_HEAD_WEIGHT,
        "lead_gap": SCENE_HEAD_WEIGHT,
        "lead_closing": SCENE_HEAD_WEIGHT,
        "lane_change": SCENE_HEAD_WEIGHT,
    }


def lead_readout(lead: torch.Tensor) -> dict:
    """A lead prediction in the target's units."""
    return {
        "present_probability": torch.sigmoid(lead[:, 0]),
        "gap_m": lead[:, 1] * GAP_SCALE_M,
        "closing_mps": lead[:, 2] * CLOSING_SCALE_MPS,
    }
