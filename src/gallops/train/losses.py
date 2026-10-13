"""The two losses, in the shapes the vendor actually uses.

Stage A's first run taught the refiner to swerve everywhere -- proposals spread
from 1.25 m to 5.22 m sideways, and the plan the scorer picked drifted 1.40 m on
ticks whose target was `continue`. On neuroNCAP scene 0103 that turned two
scenarios the released model solved 10/10 into collisions. The cause was not the
data. It was that the trainer implemented ONE of the three terms in the vendor's
`trajectory_loss_anchors`, and the missing one is the term that holds the other
proposals still.

**Why one term is not enough.** `drive_jepa_model.py` builds the refiner as

    shared_refiner = Traj_refiner(config)
    self._trajectory_head = nn.ModuleList([shared_refiner for _ in range(config.ref_num)])

-- ONE module, listed four times. Every one of the 32 proposals goes through the
same weights, four times over. So a gradient that moves the winning proposal
toward an evasion moves all 32, and a winner-takes-all loss never notices,
because it only ever looks at the closest one.

The vendor's answer is the anchor term: alongside the winner, it pulls proposals
toward up to four trajectories sampled from a fixed vocabulary of good ones, so
the shared weights keep producing a spread of plausible plans. That is the term
this module restores.

**What each loss is, precisely** (drive_jepa_agent.py, released config):

    trajectory   L1 over (x, y, heading), meaned over the 8 poses, MIN over the
                 32 proposals, meaned over the batch. Winner-takes-all, not MSE,
                 and not an average over the 32.
                 + 0.5 x the same min-over-proposals distance to up to 4 sampled
                   good anchors.
                 + inter_weight x a diversity term -- but inter_weight is 0 in the
                   released config, so it is off, and it stays off here.
                 Accumulated over the 4 refinement stages with prev_weight 0.1.

    score        binary cross-entropy with logits against SOFT targets in [0, 1]
                 -- the PDM score, not a one-hot best proposal. Only the last of
                 the 6 logit channels is trained (final_score_weight 1,
                 sub_score_weight 0), which is the channel `pdm_score` reads.

The scorer's soft targets are the one thing that cannot be transcribed. The
vendor computes them by running the PDM simulator against a NAVSIM metric cache
keyed by token, and our neuroNCAP ticks have no such cache. `score_target` below
builds them from the same grader the expert is held to, in the same multiplicative
shape PDM uses. That makes them OUR referee, not NAVSIM's, which is a real
difference and the reason a model trained here has to be re-measured on NAVSIM
rather than assumed unchanged.
"""

from __future__ import annotations

import numpy as np

#: The refinement stages are weighted geometrically, earliest lightest.
PREV_WEIGHT = 0.1
#: The vendor's 0.5 on the anchor term, and its 4 sampled anchors per tick.
ANCHOR_WEIGHT = 0.5
ANCHOR_K = 4
#: Off the road but not in anything: worse than a clear plan, better than a crash.
OFF_ROAD_SCORE = 0.6
#: A collision caps the score here, then scales with how late it happens.
COLLISION_CEILING = 0.3
#: Progress never zeroes a safe plan, so braking stays available, but a plan that
#: stops dead scores half of one that keeps up. An all-zero trajectory is a
#: commanded emergency stop, and must not be the highest-scoring option.
PROGRESS_FLOOR = 0.5
#: How much a plan loses for leaving its lane when it did not have to. Small
#: enough that avoiding a crash always beats staying in lane, large enough to
#: settle the choice between two plans that are both clear.
LANE_PENALTY = 0.3


def score_target(
    hit_at: float | None,
    off_road_at: float | None,
    travelled_m: float,
    expert_travelled_m: float,
    deviation_m: float = 0.0,
    trust_band_m: float = 4.0,
    horizon_s: float = 4.0,
) -> float:
    """One proposal's soft score in [0, 1], in PDM's multiplicative shape.

    PDM multiplies hard safety terms by a weighted average of soft ones. This
    keeps that structure with the terms our grader can measure: whether the plan
    hits anything, whether it leaves the road, how far it gets, and how far from
    its lane it strays.

    Graded rather than binary on collisions, because on a late frontal tick every
    proposal crashes -- a hard zero would leave the scorer no way to rank a 2 m/s
    impact above a 12 m/s one, which is exactly the choice neuroNCAP scores.

    **The expert's own plan sets the progress bar, not free-running speed.** The
    first version measured progress against carrying on at the current speed, so a
    clear swerve at full speed scored 1.00 and a clear brake, covering half the
    ground, scored 0.75. The scorer was therefore trained to rank swerving above
    braking on every tick -- while the trajectory loss was being taught to brake.
    The scorer picks what gets driven, so the scorer won: measured on neuroNCAP
    0103, the released model slows from 5.6 to 1.4 m/s and the finetuned one
    accelerates to 8.0 and swerves, in every scenario, however the expert was
    changed. Scoring against what the expert achieved makes matching a brake worth
    full marks and exceeding it worth nothing extra.

    `deviation_m` settles what progress alone cannot: two plans that are both
    clear and both make the distance, where one holds its lane and the other
    leaves it. A swerve has a direction to get wrong and a brake does not, so the
    lane-keeping one should win unless leaving is what avoids the crash -- and if
    it is, the other plan is not clear and never reaches this comparison.
    """
    if hit_at is None:
        base = OFF_ROAD_SCORE if off_road_at is not None else 1.0
    else:
        # Later is better: a crash at the horizon is one there was time to avoid.
        base = COLLISION_CEILING * min(1.0, max(0.0, hit_at) / max(horizon_s, 1e-6))
    # Peaks AT the expert's distance and falls away on both sides. Capping it
    # instead -- min(1, travelled/expert) -- removed the reward for overshooting
    # without penalising it, so a brake and a full-speed plan both scored 1.0 and
    # tied, leaving the lane term to decide. Measured: the model then accelerated
    # 26% on ticks where the expert braked 38%.
    progress = 1.0 - abs(travelled_m - expert_travelled_m) / max(expert_travelled_m, 1e-6)
    progress = max(0.0, progress)
    lane = 1.0 - LANE_PENALTY * min(1.0, abs(deviation_m) / max(trust_band_m, 1e-6))
    scored = base * (PROGRESS_FLOOR + (1 - PROGRESS_FLOOR) * progress) * lane
    return float(np.clip(scored, 0.0, 1.0))


def travelled(waypoints: np.ndarray) -> float:
    """Path length of a trajectory from the ego, in metres."""
    points = np.vstack([[0.0, 0.0], np.asarray(waypoints)[:, :2]])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def trajectory_loss(proposal_list, target, anchor_sets, torch):
    """The vendor's `trajectory_loss_anchors`, restored in full.

    `anchor_sets` is one array of good anchor trajectories per batch element,
    (K, 8, 3); an empty one contributes nothing, exactly as the vendor skips a
    token whose index array is empty.
    """
    total = 0.0
    winner = anchored = None
    for proposals in proposal_list:
        # Winner-takes-all: only the closest of the 32 is pulled to the target.
        winner = torch.linalg.norm(proposals - target[:, None], dim=-1, ord=1).mean(-1).amin(1).mean()
        # And the rest are held to a vocabulary of plausible plans, which is what
        # keeps a shared refiner from moving all 32 at once.
        pseudo = 0.0
        for index, anchors in enumerate(anchor_sets):
            if anchors is None or len(anchors) == 0:
                continue
            chosen = anchors
            if len(chosen) > ANCHOR_K:
                chosen = chosen[np.random.choice(len(chosen), size=ANCHOR_K, replace=False)]
            targets = torch.from_numpy(np.asarray(chosen)).to(device=proposals.device, dtype=proposals.dtype)
            difference = proposals[index, None] - targets[:, None]
            pseudo = pseudo + (torch.linalg.norm(difference, dim=-1, ord=1).mean(-1).amin(-1).mean())
        anchored = winner + ANCHOR_WEIGHT * pseudo / max(len(anchor_sets), 1)
        total = PREV_WEIGHT * total + anchored
    return total, winner, anchored


def soft_target_floor(targets, torch):
    """The lowest BCE any scorer can reach on these soft targets: their entropy.

    BCE against a soft target y is minimised at p = y, where it equals
    -[y log y + (1 - y) log(1 - y)] -- 0.69 at y = 0.5, 0 only at y = 0 or 1.
    The scorer loss sat at 0.61-0.66 through every run so far, which by itself
    cannot say whether the scorer learned nothing or was already near this
    floor. BCE minus the floor is the part that can go to zero, so that is what
    gets logged beside it.
    """
    y = targets.clamp(1e-6, 1 - 1e-6)
    return -(y * torch.log(y) + (1 - y) * torch.log(1 - y)).mean()


def scorer_loss(pred_logit, targets, torch):
    """BCE with logits on the one channel `pdm_score` reads, against soft targets.

    `pred_logit` is (B, 32, 6) and the model takes channel -1, so that is the only
    one trained -- the vendor's sub_score_weight is 0 and the five other channels
    have no target we can compute.
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_logit[..., -1], targets)
