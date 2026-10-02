"""Draw the trajectory the model should have produced, then let the grader judge it.

The obvious way to build a teacher is to enumerate a lattice of manoeuvres and keep
the best survivor. That quantises the answer and, worse, cannot express *when* to
begin: a car closing at 20 m/s leaves about a second and a half, and a fixed ramp
has no way to say "now" rather than "half a second from now". So the evasion is
solved for instead. How far sideways is geometry -- the actor sweeps a box, the ego
has a width, a margin is added -- and how fast is a constraint, not a menu choice.

Two rules keep a drawn target honest.

**Draw from a prediction, verify against the truth.** The offset is sized against a
constant-velocity extrapolation of the actor, roughly what a driver could infer
from two frames, and the result is then graded against the actor's real future. So
the target is always something the model had the information to produce, while a
trajectory that would actually crash is never taught. A lattice hid this problem by
being too coarse to exploit clairvoyance; drawing has to refuse it on purpose.

**Everything drawn goes through the same grader as the model's own 32.** A target
and a candidate are judged by identical rules, so "the expert avoided a crash"
means what "the model caused one" means.

Preference among trajectories that survive: the most forward progress. When
nothing survives, the latest contact -- the lowest impact speed available, which
is what the benchmark scores once a crash is certain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grader import EGO_WIDTH, HORIZON, TRUST_BAND_M, Verdict, actor_at, grade

WAYPOINT_DT = 0.5
POSES = 8

#: Comfort-to-emergency lateral acceleration. 4 m/s^2 is firm; beyond it the tyres
#: and the reconstruction both stop being something to rely on.
LATERAL_ACCEL_MAX = 4.0
#: Emergency braking. nuScenes' own driving almost never goes below -4 m/s^2.
BRAKE_MAX = 5.0
#: Clearance added to the two half-widths before a pass counts as a pass.
MARGIN_M = 0.5
#: Tightest path a passenger car can actually follow. The lateral-acceleration
#: budget does NOT imply this: |d2y/dt2| stays small at low speed however tight
#: the geometry, so a 2.45 m offset drawn at 1.8 m/s came out at a 1.7 m radius --
#: undrivable, and taught as a target. Curvature is the quantity to bound.
MIN_TURN_RADIUS_M = 5.0
#: The planning horizon: 8 poses at 0.5 s. A manoeuvre that cannot be drawn
#: drivably inside it is not available at all.
HORIZON_S = 4.0
#: When to begin, tried latest first. Holding the lane while the situation is
#: still developing is what a driver does, and at 10 Hz it is also what keeps the
#: car from committing to an evasion on a two-frame prediction.
DELAYS_S = (2.0, 1.5, 1.0, 0.5, 0.0)
#: How long to take crossing, tried gentlest first. `min_ramp_for` gives the
#: shortest the car could survive; that is a limit, not a target.
RAMPS_S = (3.0, 2.5, 2.0, 1.5, 1.0)
#: How long the recovery takes once the hazard is behind.
RETURN_S = 1.5
#: A beat after passing before starting back, so the return does not clip the
#: actor the swerve just avoided.
RETURN_GAP_S = 0.3
#: Below this, the actor supplies the closing speed itself and slowing down only
#: means being hit while stationary -- so the lane has to be left. Above it,
#: braking removes the ego's own contribution and that is enough. Measured on the
#: round-0 corpus, the two cases do not overlap anywhere near this line: oncoming
#: ticks close at -8.7 m/s (p90 -7.0), crossing at 0.1 (p10 -1.7), parked at 0.0.
APPROACHING_MPS = -3.0


@dataclass(frozen=True)
class Plan:
    """What the teacher decided, and why it is defensible."""

    waypoints: np.ndarray
    verdict: Verdict
    kind: str  # continue | swerve | brake | swerve+brake | least-impact
    offset_m: float
    progress_m: float

    @property
    def clear(self) -> bool:
        return self.verdict.clear


def to_ego(points_world: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    """World (N, 2) -> ego frame, x forward and y left."""
    rotation, origin = ego2world[:2, :2], ego2world[:2, 3]
    return (np.atleast_2d(points_world) - origin) @ rotation


def predict_actor(actor: dict, t0: int, horizon: np.ndarray) -> np.ndarray | None:
    """Where the actor goes next at constant velocity, from what was visible by t0.

    Deliberately NOT the logged future. Velocity comes from the 0.5 s before the
    tick, which is what two camera frames would show, so the evasion we teach is
    one the model had grounds to plan.
    """
    now = actor_at(actor, t0)
    if now is None:
        return None
    before = actor_at(actor, t0 - 0.5e6)
    velocity = np.zeros(2) if before is None else (now[0] - before[0]) / 0.5
    return now[0][None, :] + velocity[None, :] * horizon[:, None]


def follow(reference_ego: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Points `distances` ahead of the ego along the logged path, in the ego frame.

    The nominal plan is "keep doing what the human did", so a curve in the road
    does not read as a swerve and eat the trust band for nothing.

    Distances are measured from where the ego is *now*, which is the origin of its
    own frame -- so the path is first projected onto to find how far along it the
    car has already travelled. Measuring from the path's start instead puts the
    whole plan wherever the logged run happened to begin, which reads as a
    ten-metre excursion and clears obstacles by being nowhere near them.
    """
    steps = np.linalg.norm(np.diff(reference_ego, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(steps)]
    segment = np.diff(reference_ego, axis=0)
    length_sq = np.maximum((segment**2).sum(1), 1e-9)
    # project the origin onto every segment, clamped to it, and take the nearest
    t = np.clip((-reference_ego[:-1] * segment).sum(1) / length_sq, 0.0, 1.0)
    feet = reference_ego[:-1] + t[:, None] * segment
    i = int(np.argmin(np.linalg.norm(feet, axis=1)))
    here = arc[i] + t[i] * np.sqrt(length_sq[i])
    return np.column_stack(
        [
            np.interp(here + distances, arc, reference_ego[:, 0]),
            np.interp(here + distances, arc, reference_ego[:, 1]),
        ]
    )


def _trajectory(points: np.ndarray) -> np.ndarray:
    steps = np.diff(np.vstack([[0.0, 0.0], points]), axis=0)
    return np.column_stack([points, np.arctan2(steps[:, 1], steps[:, 0])])


def ease(phase: np.ndarray) -> np.ndarray:
    """A cosine S-curve on [0, 1], flat outside it. Zero slope at both ends."""
    return (1 - np.cos(np.pi * np.clip(phase, 0.0, 1.0))) / 2


def lateral_profile(
    offset: float,
    ramp_s: float,
    times: np.ndarray,
    delay_s: float = 0.0,
    return_at: float | None = None,
    return_s: float = 1.5,
) -> np.ndarray:
    """Hold the lane, ease across to `offset`, and -- if there is time -- ease back.

    The first version of this started the lateral motion at t = 0 and stayed out
    there: `phase = times / ramp`, "flat after". Every tick of a closed-loop run
    then re-commanded "be fully displaced, as fast as the tyres allow, starting
    now", and a model trained on it threw the wheel over at first sight and sat at
    an angle to the road. Which is what it did.

    Two things were missing and both are here. `delay_s` is when to begin, so the
    plan can say "straight for another second, then go" -- the thing the module
    docstring claimed to solve for and did not. `return_at` is the recovery, so a
    plan ends parallel to the road instead of leaving the car committed sideways
    in a state no training tick ever shows it getting out of.

    The return is the same eased step subtracted, so the whole profile is two
    S-curves and the lateral acceleration stays bounded by each phase's own ramp.
    """
    across = offset * ease((times - delay_s) / max(ramp_s, 1e-6))
    if return_at is None:
        return across
    return across - offset * ease((times - return_at) / max(return_s, 1e-6))


def min_ramp_for(
    offset: float,
    speed: float,
    accel_max: float = LATERAL_ACCEL_MAX,
    min_radius: float = MIN_TURN_RADIUS_M,
) -> float:
    """Shortest ramp the car could actually follow, under BOTH limits.

    The cosine profile peaks at |offset| * pi^2 / (2 T^2), which gives the
    acceleration bound directly. That alone is not enough: path curvature is
    roughly that acceleration over speed squared, so the same manoeuvre at
    walking pace demands a turn no car can make. Bounding curvature adds a term
    that grows as speed falls, which is what "when to start" really depends on.

    Returns the ramp; the caller checks it against the horizon, because at low
    speed a wide offset simply is not available.
    """
    by_accel = np.pi * np.sqrt(abs(offset) / (2 * max(accel_max, 1e-6)))
    by_curvature = (np.pi / max(speed, 0.1)) * np.sqrt(abs(offset) * min_radius / 2)
    return float(max(by_accel, by_curvature))


def speed_profile(
    speed: float,
    scale: float,
    times: np.ndarray,
    decel: float = BRAKE_MAX,
    delay_s: float = 0.0,
) -> np.ndarray:
    """Distance travelled by each time: hold speed for `delay_s`, then decelerate
    AT `decel` to `scale * speed`.

    The drop used to be spread across the whole horizon, which made BRAKE_MAX dead
    below 20 m/s -- a commanded stop from 1.8 m/s came out as 0.45 m/s^2 over
    3.60 m, where a real stop is 0.32 m. Every tick labelled "brake" would have
    taught a lift-off. The rate is now the rate.

    `delay_s` is the other half of that, and it was missing. The swerve was given
    a "when to begin" and the brake was not, so a brake began at t = 0 however far
    away the hazard was. Measured on round 0, the expert first reacts at a
    time-to-contact of 3.85 s -- the edge of its 4 s horizon, which is as late as
    a 4 s planner can see -- and then slowed from the first instant, which is what
    makes a reaction to a car 63 m away look like a flinch. Coasting first is what
    a driver does with four seconds in hand.
    """
    target = speed * scale
    if speed <= target or decel <= 0:
        return speed * times
    held = np.minimum(times, max(delay_s, 0.0))
    after = np.maximum(times - max(delay_s, 0.0), 0.0)
    stop_time = (speed - target) / decel
    ramp = np.minimum(after, stop_time)
    return speed * held + speed * ramp - 0.5 * decel * ramp**2 + target * np.maximum(after - stop_time, 0.0)


def required_offset(nominal: np.ndarray, predicted: np.ndarray, actor_dims) -> tuple[float, float]:
    """How far sideways is needed to pass the actor, and by when.

    Returns (signed offset, seconds until the binding moment). Zero means the
    nominal path already clears it.
    """
    clearance = EGO_WIDTH / 2 + actor_dims[0] / 2 + MARGIN_M
    need, when = 0.0, float(HORIZON[-1])
    for k, tau in enumerate(HORIZON):
        if abs(predicted[k, 0] - nominal[k, 0]) > actor_dims[1] / 2 + 3.0:
            continue  # not alongside us at this instant
        lateral_gap = predicted[k, 1] - nominal[k, 1]
        if abs(lateral_gap) >= clearance:
            continue
        side = -1.0 if lateral_gap >= 0 else 1.0  # push away from the actor
        this = side * (clearance - abs(lateral_gap))
        if abs(this) > abs(need):
            need, when = float(this), float(tau)
    return need, when


def speed_at(waypoints: np.ndarray, tau: float) -> float:
    """The plan's own speed at time `tau`, from the spacing of its waypoints."""
    points = np.vstack([[0.0, 0.0], waypoints[:, :2]])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) / WAYPOINT_DT
    stamps = np.arange(1, len(waypoints) + 1) * WAYPOINT_DT
    return float(np.interp(tau, stamps, steps))


def rejoin(reference_ego: np.ndarray, speed: float, times: np.ndarray) -> np.ndarray:
    """The lateral correction that gets a displaced car back to the path drivably.

    `follow` returns points ON the logged path. For a car sitting on that path
    that is what you want. For a car three metres off it -- which is every tick
    after an evasion -- it means the plan's first waypoint is three metres
    sideways half a second from now: 24.5 m/s^2, six times this module's own
    comfort limit, and the car cannot do it. Taught as a target it is worse than
    useless, because it is the answer to "how do I get back in the lane" and it is
    a lie.

    Round 0 never exposed this: the released model does not swerve, so the ego was
    always already on the path and the correction was always zero. The finetuned
    model reaches these states constantly, and sits at an angle to the road with
    nothing sensible to imitate. This eases the rejoin over a ramp the car can
    actually follow instead.
    """
    drift = -float(follow(reference_ego, np.zeros(1))[0, 1])
    if abs(drift) < 1e-3:
        return np.zeros_like(times)
    ramp = min(min_ramp_for(drift, speed), HORIZON_S)
    return drift * (1 - ease(times / max(ramp, 1e-6)))


def _plan_from(
    reference_ego: np.ndarray,
    speed: float,
    scale: float,
    offset: float,
    ramp_s: float,
    delay_s: float = 0.0,
    return_at: float | None = None,
    brake_delay_s: float = 0.0,
) -> np.ndarray:
    times = np.arange(1, POSES + 1) * WAYPOINT_DT
    points = follow(reference_ego, speed_profile(speed, scale, times, delay_s=brake_delay_s))
    # Start from where the car actually is, not from where the path is.
    points[:, 1] += rejoin(reference_ego, speed, times)
    points[:, 1] += lateral_profile(offset, ramp_s, times, delay_s, return_at)
    return _trajectory(points)


def teach(
    ego2world: np.ndarray,
    t0: int,
    speed: float,
    reference_world: np.ndarray,
    targets: list[dict],
    area=None,
) -> Plan:
    """The trajectory the model should have produced at this tick.

    Drawn against a constant-velocity prediction of each target, then graded
    against their real futures. Widening and braking are tried, in that order,
    only when the grader rejects what was drawn.
    """
    reference = np.asarray(reference_world)[:, :2]
    reference_ego = to_ego(reference, ego2world)

    def judge(waypoints):
        return grade(waypoints, ego2world, t0, targets, reference, area)

    nominal = _plan_from(reference_ego, speed, 1.0, 0.0, 1.0)
    verdict = judge(nominal)
    if verdict.clear:
        return Plan(nominal, verdict, "continue", 0.0, float(nominal[5, 0]))

    # Size the evasion against what was predictable, not against the logged future.
    nominal_dense = follow(reference_ego, speed_profile(speed, 1.0, HORIZON))
    need = 0.0
    closing = 0.0
    for actor in targets:
        predicted = predict_actor(actor, t0, HORIZON)
        if predicted is None:
            continue
        ego_frame = to_ego(predicted, ego2world)
        this, _when = required_offset(nominal_dense, ego_frame, actor["dims"])
        if abs(this) > abs(need):
            need = this
        # How fast it closes on us along our own heading, which is what decides
        # whether braking can resolve this at all.
        closing = min(closing, (ego_frame[-1, 0] - ego_frame[0, 0]) / float(HORIZON[-1]))

    attempts: list[Plan] = []

    def attempt(kind: str, scale: float, offset: float, brake_delay_s: float = 0.0) -> bool:
        ramp = min_ramp_for(offset, speed) if offset else 1.0
        if ramp > HORIZON_S:
            return False  # no drivable way to reach that offset in the time available
        waypoints = _plan_from(reference_ego, speed, scale, offset, ramp, brake_delay_s=brake_delay_s)
        attempts.append(Plan(waypoints, judge(waypoints), kind, float(offset), float(waypoints[5, 0])))
        return attempts[-1].clear

    # When the ego draws level with the nearest hazard, from the same prediction
    # the offset was sized against. After this the lane is free again and the plan
    # should be coming back, not holding a metre of offset to the horizon.
    passes_at = None
    for actor in targets:
        predicted = predict_actor(actor, t0, HORIZON)
        if predicted is None:
            continue
        ahead = to_ego(predicted, ego2world)[:, 0] - nominal_dense[:, 0]
        level = np.flatnonzero(ahead <= 0.0)
        if len(level):
            when = float(HORIZON[level[0]])
            passes_at = when if passes_at is None else min(passes_at, when)

    def refine(offset: float, scale: float) -> Plan | None:
        """The same evasion, drawn as late and as gently as it can still be drawn.

        Feasibility already said this offset works; this asks how little of it the
        car has to do early. Latest start first, then longest ramp, and a recovery
        whenever one fits -- so the answer degrades gracefully to the old
        drive-hard-immediately plan only when the closing speed leaves nothing else.
        """
        floor = min_ramp_for(offset, speed)
        for delay in DELAYS_S:
            for ramp in RAMPS_S:
                if ramp < floor or delay + ramp > HORIZON_S:
                    continue
                returns = [None]
                if passes_at is not None and passes_at + RETURN_GAP_S + RETURN_S <= HORIZON_S:
                    returns.insert(0, passes_at + RETURN_GAP_S)
                for back in returns:
                    waypoints = _plan_from(reference_ego, speed, scale, offset, ramp, delay, back)
                    verdict = judge(waypoints)
                    if verdict.clear:
                        return Plan(waypoints, verdict, "swerve", float(offset), float(waypoints[5, 0]))
        return None

    # A hazard that is not coming at us -- a parked car, a car crossing ahead --
    # is one that braking genuinely resolves: slowing removes the ego's own
    # contribution to the closing speed, and the crossing car clears the lane by
    # itself. Measured on round 0, braking alone was clear on 197 of 197 sampled
    # stationary swerve ticks and 10 of 10 side ones, yet the ladder never tried
    # it, because it only reached braking when no swerve worked. So a model
    # trained on this learned to swerve round a parked car at full speed, and on
    # neuroNCAP 0103 it swerved the wrong way into it while the released model
    # scored 5.0 by simply stopping.
    #
    # An oncoming car is the opposite case and keeps the swerve: it supplies the
    # closing speed itself, so braking only changes how fast you are going when it
    # arrives. That is the one split in this ladder, and it is geometric -- the
    # expert never learns which scenario it is in.
    if closing > APPROACHING_MPS:
        # Latest start first, then gentlest -- the same order the swerve is refined
        # in, for the same reason: with four seconds in hand a driver coasts before
        # slowing, and a plan that slows from t = 0 whatever the distance is what
        # makes a reaction to a car 60 m away look like a flinch.
        for delay in DELAYS_S:
            for scale in (0.6, 0.3, 0.0):
                if attempt("brake", scale, 0.0, brake_delay_s=delay):
                    return attempts[-1]

    if need:
        # Away from the actor first, then the other side. Trying only one side was
        # fine while nothing knew where the road was; once it does, the natural
        # escape is often blocked -- on a two-lane street with the ego in the
        # right-hand lane, "away from the oncoming car" is the parked cars, and the
        # empty oncoming carriageway is on the side the actor came from.
        found = False
        for direction in (np.sign(need), -np.sign(need)):
            # Widen in half-metre steps: the prediction is approximate, so the
            # first solve can fall just short of an actor that is not quite
            # constant velocity.
            for extra in (0.0, 0.5, 1.0, 1.5):
                offset = direction * (abs(need) + extra)
                if abs(offset) > TRUST_BAND_M:
                    break
                if attempt("swerve", 1.0, offset):
                    found = True
                    break
            if found:
                break
        if found:
            # Feasibility used the harshest drawing of this offset. Take the
            # gentlest one that still clears, which is the target worth teaching.
            smoother = refine(attempts[-1].offset_m, 1.0)
            if smoother is not None:
                return smoother

    if not any(a.clear for a in attempts):
        for delay in DELAYS_S:
            if any(attempt("brake", scale, 0.0, brake_delay_s=delay) for scale in (0.6, 0.3, 0.0)):
                break

    if not any(a.clear for a in attempts) and need:
        offset = float(np.sign(need) * min(abs(need) + 1.0, TRUST_BAND_M))
        for scale in (0.6, 0.3):
            if attempt("swerve+brake", scale, offset):
                break

    clear = [a for a in attempts if a.clear]
    if clear:
        return max(clear, key=lambda a: a.progress_m)
    if attempts:
        # Nothing avoids it, so take the lowest speed we can be going at contact.
        # This used to maximise `hit_at`, which is a TIME at 0.1 s resolution and
        # says nothing about speed -- when every attempt hit in the same tenth of
        # a second it returned whichever happened to be first in the list, once
        # preferring an undrivable swerve over a full stop. Later contact only
        # breaks ties now.
        gentlest = min(
            attempts,
            key=lambda a: (speed_at(a.waypoints, a.verdict.hit_at or 0.0), -(a.verdict.hit_at or 0.0)),
        )
        return Plan(
            gentlest.waypoints, gentlest.verdict, "least-impact", gentlest.offset_m, gentlest.progress_m
        )
    return Plan(nominal, verdict, "least-impact", 0.0, float(nominal[5, 0]))
