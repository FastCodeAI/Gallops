"""Judge a trajectory: does it hit anything, leave the road, or stray too far?

This is the half of the expert that has a hard correctness test. It grades any
trajectory on Drive-JEPA's grid, and it does two jobs at once:

* it labels every one of the model's 32 candidates, which is the scorer's
  training signal;
* it checks whatever the teacher draws, so a target that would crash is never
  taught.

Because one function does both, a swerve and a candidate are judged by identical
rules, and "the expert avoided a crash" means exactly what "the model caused one"
means.

**The collision test is the benchmark's own.** Footprint, rear-axle offset and
the separating-axis overlap are transcribed from
`neuro_ncap/evaluation/collision.py` (4.018 x 1.85 m, box centre 0.5 m ahead of
the rear axle). `tests/test_expert_grader.py` reads that vendor file and fails if
the constants drift. Paddock transcribes the same numbers for its own analysis;
the two repos do not import each other, so each cites the source instead.

Open loop, per tick: the trajectory is placed at the ego pose it was planned from
and followed with no replanning, while scripted actors are replayed from the
episode's `actors.json`. "Clear" therefore means *this trajectory, driven from
here, does not hit the target within HORIZON seconds* -- an option existed, not
that taking it would have saved the episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The vendor's ego footprint (`neuro_ncap/evaluation/collision.py`).
EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE = 4.018, 1.85, 0.5

#: Times a trajectory is checked at, in seconds from the tick it was planned on.
HORIZON = np.round(np.arange(0.1, 4.0001, 0.1), 1)

#: How far off the logged path a trajectory may go before the NeuRAD
#: reconstruction stops being something to trust. Reasoned, not measured: one
#: lane width clears a car, and Drive-JEPA's own episodes stay under 4 m while
#: the renders are visibly fine. See docs/models/drivejepa-neuroncap.md.
TRUST_BAND_M = 4.0


@dataclass(frozen=True)
class Verdict:
    """What the grader found. `clear` is the only thing most callers need."""

    hit_at: float | None
    off_path_at: float | None
    max_deviation: float

    @property
    def clear(self) -> bool:
        return self.hit_at is None and self.off_path_at is None


def corners(centre: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    return np.array(
        [
            centre + a * forward * length / 2 + b * left * width / 2
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test between two oriented rectangles."""
    for polygon in (a, b):
        for i in range(4):
            edge = polygon[(i + 1) % 4] - polygon[i]
            normal = np.array([-edge[1], edge[0]])
            pa, pb = a @ normal, b @ normal
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def actor_at(actor: dict, timestamp: float):
    """An actor's (position, yaw) at a time, or None if it is not active then."""
    times = np.asarray(actor["timestamps"], dtype=float)
    if not times[0] <= timestamp <= times[-1]:
        return None
    poses = np.asarray(actor["poses"])
    i = int(np.clip(np.searchsorted(times, timestamp) - 1, 0, len(times) - 2))
    w = (timestamp - times[i]) / (times[i + 1] - times[i])
    position = poses[i][:2, 3] * (1 - w) + poses[i + 1][:2, 3] * w
    y0 = np.arctan2(poses[i][1, 0], poses[i][0, 0])
    y1 = np.arctan2(poses[i + 1][1, 0], poses[i + 1][0, 0])
    return position, y0 + np.angle(np.exp(1j * (y1 - y0))) * w


def densify(waypoints: np.ndarray):
    """(N, 3) waypoints at 0.5 s -> (x, y, heading) at HORIZON, from the origin."""
    stamps = np.r_[0, np.arange(1, len(waypoints) + 1) * 0.5]
    x, y, heading = (np.r_[0, waypoints[:, k]] for k in range(3))
    return (
        np.interp(HORIZON, stamps, x),
        np.interp(HORIZON, stamps, y),
        np.interp(HORIZON, stamps, np.unwrap(heading)),
    )


def first_hit(waypoints: np.ndarray, ego2world: np.ndarray, t0: int, targets: list[dict]) -> float | None:
    """When this trajectory first overlaps a target, in seconds, or None."""
    x, y, heading = densify(waypoints)
    rotation, origin = ego2world[:2, :2], ego2world[:2, 3]
    yaw0 = np.arctan2(rotation[1, 0], rotation[0, 0])
    for k, tau in enumerate(HORIZON):
        position = origin + rotation @ np.array([x[k], y[k]])
        yaw = yaw0 + heading[k]
        ego = corners(
            position + REAR_AXLE_TO_CENTRE * np.array([np.cos(yaw), np.sin(yaw)]),
            yaw,
            EGO_LENGTH,
            EGO_WIDTH,
        )
        for actor in targets:
            state = actor_at(actor, t0 + tau * 1e6)
            if state is None:
                continue
            if overlap(ego, corners(state[0], state[1], actor["dims"][1], actor["dims"][0])):
                return float(tau)
    return None


def deviation(
    waypoints: np.ndarray, ego2world: np.ndarray, reference: np.ndarray
) -> tuple[float, float | None]:
    """Largest distance from the logged path, and when it first exceeds the band.

    The band stands in for "the reconstruction was trained on this view". It is a
    property of the renderer, not of driving, which is why it is graded here
    beside the collision rather than folded into the teacher's preferences.
    """
    x, y, _ = densify(waypoints)
    rotation, origin = ego2world[:2, :2], ego2world[:2, 3]
    worst, first = 0.0, None
    for k, tau in enumerate(HORIZON):
        point = origin + rotation @ np.array([x[k], y[k]])
        gap = float(np.min(np.linalg.norm(reference - point, axis=1)))
        worst = max(worst, gap)
        if first is None and gap > TRUST_BAND_M:
            first = float(tau)
    return worst, first


def grade(
    waypoints: np.ndarray,
    ego2world: np.ndarray,
    t0: int,
    targets: list[dict],
    reference: np.ndarray | None = None,
) -> Verdict:
    """The one judgement, applied to a model candidate and a drawn target alike."""
    worst, off = (0.0, None) if reference is None else deviation(waypoints, ego2world, reference)
    return Verdict(
        hit_at=first_hit(waypoints, ego2world, t0, targets),
        off_path_at=off,
        max_deviation=worst,
    )
