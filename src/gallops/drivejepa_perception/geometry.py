"""Candidate x agent terms with no weights: where each agent will be when the ego gets there (phase B).

For every candidate k (8 poses, x y yaw, at 0.5 s steps) and agent a (decoded side-car output, or
ground truth for arm O), the agent's centre at each plan time t:
- t = 0.5 s and 1.0 s: its own predicted path (centre + offset);
- t > 1.0 s: constant velocity from its 1.0 s position, at its predicted velocity (as PDM's
  simulation of other agents does).

The ego is NAVSIM's (nuPlan's Chrysler Pacifica: 5.176 x 2.297 m, centre 1.461 m ahead of the rear
axle, which is the plan's pose). Clearance is centre distance minus both footprints' radii, each the
radius of a circle of the box's area, sqrt(l * w / pi): rough on purpose, because these are features
for a learned scorer and explanations for a person, not the grader. The grader is PDM's (labels).

Per candidate, the scorer gets `features` (N_FEATURES numbers) and, per agent, the smallest
clearance over the plan (`pairwise`, the attention bias); `nearest` names the agent behind each
candidate's smallest clearance -- the explanation B1.6 tests by deleting it.
"""

from __future__ import annotations

import math

import torch

#: nuPlan's get_pacifica_parameters: front length 4.049, rear length 1.127, width 2.297.
EGO_LENGTH, EGO_WIDTH, EGO_CENTRE_AHEAD = 5.176, 2.297, 1.461
TIMES = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
CLEARANCE_CLIP = (-5.0, 30.0)
#: 8 per-time minima, the overall minimum, its time, the minimum within 1 s and within 2 s, and the
#: number of agents ever closer than 1 m.
N_FEATURES = 13


def agent_centres(xy, vel, path, times=TIMES):
    """(B, A, 2) centres at t, (B, A, 2) velocity, (B, A, 2, 2) path offsets -> (B, A, T, 2) centres."""
    at_1s = xy + path[:, :, 1]
    later = at_1s[:, :, None, :] + vel[:, :, None, :] * (times[2:].to(xy) - 1.0)[None, None, :, None]
    return torch.cat([(xy[:, :, None, :] + path), later], dim=2)


def ego_centres(candidates):
    """(B, K, 8, 3) rear-axle poses -> (B, K, 8, 2) footprint centres."""
    yaw = candidates[..., 2]
    return candidates[..., :2] + EGO_CENTRE_AHEAD * torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def clearances(candidates, xy, size, vel, path, mask):
    """(B, K, A, T): the clearance of each candidate to each agent at each plan time, in metres.

    Masked agents sit at +CLIP, and every value is clamped to CLEARANCE_CLIP. B-1's veto reads this
    per agent and per time (a margin within a horizon, and which agent was behind it); `terms` reduces
    it to the scorer's features.
    """
    ego = ego_centres(candidates)  # B, K, T, 2
    agents = agent_centres(xy, vel, path)  # B, A, T, 2
    distance = torch.linalg.norm(ego[:, :, None] - agents[:, None], dim=-1)  # B, K, A, T
    radius_ego = math.sqrt(EGO_LENGTH * EGO_WIDTH / math.pi)
    radius = torch.sqrt((size[..., 0] * size[..., 1]).clamp(min=0.01) / math.pi)  # B, A
    clearance = distance - radius_ego - radius[:, None, :, None]
    return clearance.masked_fill(~mask[:, None, :, None], CLEARANCE_CLIP[1]).clamp(*CLEARANCE_CLIP)


def terms(candidates, xy, size, vel, path, mask):
    """The analytic terms of K candidates against A agents.

    candidates (B, K, 8, 3); xy (B, A, 2); size (B, A, 2) length, width; vel (B, A, 2);
    path (B, A, 2, 2) offsets; mask (B, A) bool, True for a real agent.
    Returns features (B, K, N_FEATURES), pairwise (B, K, A) smallest clearance per agent (masked
    agents at +CLIP), nearest (B, K) index of the agent with the smallest clearance (-1 if none).
    """
    clearance = clearances(candidates, xy, size, vel, path, mask)  # B, K, A, T
    per_time = clearance.min(dim=2).values  # B, K, T
    overall, when = per_time.min(dim=-1)
    pairwise = clearance.min(dim=-1).values  # B, K, A
    nearest = pairwise.argmin(dim=-1)
    nearest = torch.where(
        mask.any(dim=-1, keepdim=True).expand_as(nearest), nearest, torch.full_like(nearest, -1)
    )
    close = (pairwise < 1.0).float().sum(dim=-1)
    features = torch.cat(
        [
            per_time / 10.0,
            (overall / 10.0)[..., None],
            TIMES.to(candidates)[when][..., None] / 4.0,
            (per_time[..., :2].min(dim=-1).values / 10.0)[..., None],
            (per_time[..., :4].min(dim=-1).values / 10.0)[..., None],
            (close / 4.0)[..., None],
        ],
        dim=-1,
    )
    return features, pairwise, nearest
