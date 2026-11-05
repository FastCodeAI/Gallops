"""Expert v6: brake for what will be in the way; for a car coming at us, brake first, then go round it.

v5 (`draw.teach`) was built and checked one tick at a time, and it never drove. In closed loop four
things it did turned out to be what the model learned:

* it chose a fresh plan every tick with no memory, so the swerve side came from the sign of a lateral
  gap near zero and flipped from tick to tick (154 flips across round-0 frontal labels);
* its brake ladder tried the latest start first, so every tick put braking off by another 2 s;
* for an oncoming car it only ever swerved at full speed -- it never braked first;
* it graded plans against the one inserted car, never the others the benchmark checks.

This expert is stateful instead. One `Teacher` drives one episode, called tick by tick in time order,
and once it acts it COMMITS to a manoeuvre anchored in the world: a speed schedule in clock time
(when braking began, how hard, down to what speed) and a sideways offset laid out by distance along
a fixed route (the lane the episode started in). Every later tick continues that manoeuvre from
wherever the car actually is; nothing is redrawn from "now". It changes the manoeuvre only when the
grader says the committed one no longer clears -- and then commits to the change.

What it does depends on how the hazard moves, read from the hazard's real future (which the
labelling has), never from the scenario's name:

* parked on the route, or crossing it: brake to a stop short of the conflict, hold the stop while
  blocked (no creep), and drive on once the way is clear;
* coming at us: brake at once, but only down to a floor speed, because a stopped car cannot steer and
  neuro-ncap's tracker ignores the path once the plan ends within a metre; then move across just far
  enough to clear, on whichever side has room, and come back once past.

It acts when two things hold: the plan of simply carrying on would hit something within `trigger_s`,
and the hazard is wide enough in the front camera for the model to have any chance of seeing it
(`min_px`). Teaching a reaction to something below one encoder patch teaches a reflex, not a reaction.

Every plan it returns still goes through `grader.grade`, against every collision-class actor, the
drivable raster and the trust band, exactly as the model's own candidates are judged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .draw import (
    HARDEST_BRAKE,
    LATERAL_ACCEL_MAX,
    MIN_TURN_RADIUS_M,
    POSES,
    WAYPOINT_DT,
    ease,
    min_length_for,
    pursue,
    to_ego,
    undrivable,
)
from .grader import (
    EGO_LENGTH,
    EGO_WIDTH,
    HORIZON,
    REAR_AXLE_TO_CENTRE,
    TRUST_BAND_M,
    Verdict,
    actor_at,
    grade,
    leaves_road,
)

#: Classes neuro-ncap's CollisionChecker checks (`evaluation/collision.py`); a hit on any of them ends
#: the benchmark episode, so the expert has to avoid all of them, not only the inserted car.
COLLISION_CLASSES = frozenset(
    {"car", "bus", "construction_vehicle", "bicycle", "motorcycle", "truck", "trailer"}
)

#: Act when carrying on would hit something within this many seconds. Swept in Gate B.
TRIGGER_S = 3.5
#: ...and the hazard spans at least this many pixels of the 512-px-wide crop the encoder sees
#: (16-px patches): at v5's first reaction a frontal car was a median 15 px.
MIN_PX = 16.0
#: Below this time to contact, act on any part of it in frame: too late to wait for 16 px. A hazard
#: wholly outside the frame is not acted on (user, 2026-10-13: brake when it appears), unless the car
#: is out of its lane (`OFF_LANE_M`), which is rare: 10 of 6,901 reactive ticks on rounds 0 and 2. A
#: crossing car on a collision course holds a constant bearing, 36-45 deg in our scenes, outside the
#: camera's 32.6 deg half-field until the last second; reacting earlier would teach the unseen.
EMERGENCY_S = 1.5
#: ...or the car is already this far out of its lane, i.e. partway through a manoeuvre the camera
#: shows by the road itself; the hazard can be at the frame's edge by then.
OFF_LANE_M = 0.75
#: The front camera, as `label.py` places it: 1.72 m ahead of the rear axle, half field 32.6 deg,
#: and 256 px from the axis to the crop's edge.
CAMERA_AHEAD_M = 1.72
CAMERA_HALF_FOV_DEG = 32.6
CROP_HALF_PX = 256.0

#: Braking for a car coming at us, and the speed it stops at: still enough to steer (5 m radius
#: at the tracker's resolution) and to keep the plan's last waypoint beyond neuro-ncap's 1 m
#: "standing still" radius, where its tracker stops following the path at all.
ONCOMING_DECEL = 4.0
ONCOMING_FLOOR_MPS = 3.0
#: Braking for a parked or crossing car is sized to stop STOP_GAP_M short of the contact point,
#: within these limits; 6 m/s^2 is neuro-ncap's vehicle-model clamp. The rate is worked out again
#: every tick from the fixed stop point, so it is the one a fresh look at the same state gives; a
#: rate fixed at the first tick and clipped up to a minimum stopped the car 8 m short instead.
BRAKE_MIN = 0.5
BRAKE_MAX = HARDEST_BRAKE
STOP_GAP_M = 2.0
#: Driving on once the way is clear: back to the place's speed, gently.
RESUME_ACCEL = 1.0
#: Whether the way is clear is asked of a plan that speeds back up to at least this. Asked of a
#: plan at the car's own crawl, a stopped car's 4 s horizon never reaches the car parked 14 m
#: ahead, so the way looked clear, it crept, braked again, and crept: the creep we set out to remove.
RELEASE_PROBE_MPS = 6.0
#: Hazard motion along the route below this (toward us) is "coming at us"; below `PARKED_MPS` overall, parked.
ONCOMING_MPS = -2.0
PARKED_MPS = 0.5
#: Every decision -- whether to act, whether a plan clears, when to drive on -- is made against
#: actors grown by this much on each side and each end. Without it a car parked 0.3 m clear of the
#: lane centre was "in the way" from one lateral position and "clear" from another, so the expert
#: braked, then drove past it: the label flipped on a few tenths of a metre the model cannot see.
GUARD_SIDE_M = 0.3
GUARD_END_M = 0.5
#: Clearance beside an oncoming car: guarded half-widths plus this, so 0.75 m in all -- 0.25 m more
#: than v5's 0.5, because the tracker lags the plan by a few tenths of a metre in a swerve.
PASS_MARGIN_M = 0.45
#: The swerve must be complete this far (along the route) before the contact point.
SWERVE_LEAD_M = 1.0
#: Return to the lane this far past the point the hazard is behind the ego's rear axle.
RETURN_GAP_M = 3.0
#: A sideways move is laid out over at least the distance the car covers easing across at this
#: lateral acceleration (comfortable side of v5's 4.0 m/s^2 limit).
SWERVE_LATERAL_ACCEL = 3.0
#: Easing back onto the committed path from wherever the car really is, over a drivable length.
REJOIN_MIN_M = 6.0
#: The ego's front bumper ahead of the rear axle.
EGO_FRONT_M = REAR_AXLE_TO_CENTRE + EGO_LENGTH / 2
#: Plans must keep the footprint this far inside the drivable raster: a swerve drawn to the very
#: edge ran 1-2 corners over it once the tracker's lag was added (frontal-0058).
ROAD_MARGIN_M = 0.3
#: How far around the ego actors are considered at all.
NEAR_M = 90.0


# --- the route --------------------------------------------------------------------------------------


class Route:
    """A polyline in the world, addressed by distance along it."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)[:, :2]
        steps = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arc = np.r_[0.0, np.cumsum(steps)]

    def project(self, xy: np.ndarray) -> tuple[float, float]:
        """(distance along, signed distance to the left) of a world point."""
        segment = np.diff(self.points, axis=0)
        length_sq = np.maximum((segment**2).sum(1), 1e-9)
        t = np.clip(((xy - self.points[:-1]) * segment).sum(1) / length_sq, 0.0, 1.0)
        feet = self.points[:-1] + t[:, None] * segment
        i = int(np.argmin(np.linalg.norm(feet - xy, axis=1)))
        rel = xy - feet[i]
        left = float(segment[i, 0] * rel[1] - segment[i, 1] * rel[0]) / np.sqrt(length_sq[i])
        return float(self.arc[i] + t[i] * np.sqrt(length_sq[i])), left

    def at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points and unit tangents at distances `s` (clamped to the route)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.arc[-1])
        x = np.interp(s, self.arc, self.points[:, 0])
        y = np.interp(s, self.arc, self.points[:, 1])
        ahead = np.clip(s + 0.5, 0.0, self.arc[-1])
        behind = np.clip(s - 0.5, 0.0, self.arc[-1])
        dx = np.interp(ahead, self.arc, self.points[:, 0]) - np.interp(behind, self.arc, self.points[:, 0])
        dy = np.interp(ahead, self.arc, self.points[:, 1]) - np.interp(behind, self.arc, self.points[:, 1])
        norm = np.maximum(np.hypot(dx, dy), 1e-9)
        return np.column_stack([x, y]), np.column_stack([dx / norm, dy / norm])

    def offset_points(self, s: np.ndarray, lateral: np.ndarray) -> np.ndarray:
        points, tangent = self.at(s)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        return points + normal * np.asarray(lateral, dtype=float)[:, None]


# --- what the expert commits to ------------------------------------------------------------------------


@dataclass(frozen=True)
class Manoeuvre:
    """A decision, in world terms, that later ticks continue rather than redraw."""

    hazard: str  # uuid of the actor that triggered it
    motion: str  # parked | crossing | oncoming
    started_us: int
    decel: float
    floor_mps: float  # braking stops at this speed; 0 = to a standstill
    stop_s: float | None  # route distance to stop at (parked, crossing)
    side: int = 0  # +1 left, -1 right, 0 none
    offset_m: float = 0.0  # signed, from the route
    swerve_s: float | None = None  # route distance the move across begins
    ramp_m: float = 0.0
    released: bool = False  # the hazard is behind or has cleared; driving on

    def lateral(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if not self.offset_m or self.swerve_s is None:
            return np.zeros_like(s)
        return self.offset_m * ease((s - self.swerve_s) / max(self.ramp_m, 1e-6))


@dataclass(frozen=True)
class Taught:
    """One tick's target and why."""

    waypoints: np.ndarray  # (8, 3) ego frame, x forward, y left, heading
    kind: str  # continue | brake | stop | brake+swerve | resume | least-impact
    verdict: Verdict
    manoeuvre: Manoeuvre | None
    hazard_px: float | None = None


# --- geometry helpers ----------------------------------------------------------------------------------


def hazard_px(ego2world: np.ndarray, state) -> float:
    """How wide an actor box appears in the 512-px crop, 0 if it is outside the frame or behind."""
    (centre, yaw), dims = state
    box = np.array(
        [
            centre
            + a * np.array([np.cos(yaw), np.sin(yaw)]) * dims[1] / 2
            + b * np.array([-np.sin(yaw), np.cos(yaw)]) * dims[0] / 2
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )
    local = to_ego(box, ego2world)
    local[:, 0] -= CAMERA_AHEAD_M
    if (local[:, 0] <= 0.5).any():
        in_front = local[local[:, 0] > 0.5]
        if len(in_front) < 2:
            return 0.0
        local = in_front
    edge = np.tan(np.radians(CAMERA_HALF_FOV_DEG))
    tangents = np.clip(local[:, 1] / local[:, 0], -edge, edge)
    return float((tangents.max() - tangents.min()) / edge * CROP_HALF_PX)


def _velocity(actor: dict, t0: int) -> np.ndarray:
    now, later = actor_at(actor, t0), actor_at(actor, t0 + 0.5e6)
    if now is None or later is None:
        return np.zeros(2)
    return (later[0] - now[0]) / 0.5


def travel(
    speed: float, times: np.ndarray, decel: float, floor: float, stop_after_m: float | None
) -> np.ndarray:
    """Distance covered by each time: braking at `decel` from `speed` to `floor`, never past a stop."""
    if decel <= 0 or speed <= floor:
        distance = speed * times
    else:
        braking = np.minimum(times, (speed - floor) / decel)
        distance = speed * braking - 0.5 * decel * braking**2 + floor * np.maximum(times - braking, 0.0)
    if stop_after_m is not None:
        distance = np.minimum(distance, max(stop_after_m, 0.0))
    return distance


def stopping_decel(speed: float, room: float, floor: float = BRAKE_MIN) -> float:
    """The rate that stops from `speed` in `room` metres, within the limits; the hardest if out of room."""
    if room <= 0.1:
        return BRAKE_MAX
    return float(np.clip(speed**2 / (2 * room), min(floor, BRAKE_MAX), BRAKE_MAX))


def resume(speed: float, times: np.ndarray, cruise: float, accel: float = RESUME_ACCEL) -> np.ndarray:
    """Distance covered by each time, speeding up gently to `cruise` (or holding it)."""
    if speed >= cruise:
        return speed * times
    rising = np.minimum(times, (cruise - speed) / accel)
    return speed * rising + 0.5 * accel * rising**2 + cruise * np.maximum(times - rising, 0.0)


# --- the expert ----------------------------------------------------------------------------------------


#: Hold an actor at its last logged pose this long after its trajectory ends. Scenario actors are
#: logged for the scenario's 8 s, which ends before a 4 s plan made near the end of the episode does;
#: past its last stamp the grader treats an actor as gone, and a parked car "vanished" from in front
#: of a stopped ego -- which then drove at it.
HOLD_AFTER_S = 10.0


def guarded(actor: dict) -> dict:
    """The copy decisions are made against: grown by the guard, held at its last pose, as arrays."""
    width, length, *rest = actor["dims"]
    stamps = np.asarray(actor["timestamps"], dtype=np.int64)
    poses = np.asarray(actor["poses"], dtype=float)
    return dict(
        actor,
        true_dims=list(actor["dims"]),
        dims=[width + 2 * GUARD_SIDE_M, length + 2 * GUARD_END_M, *rest],
        timestamps=np.r_[stamps, stamps[-1] + int(HOLD_AFTER_S * 1e6)],
        poses=np.concatenate([poses, poses[-1:]]),
    )


def human_cruise(route: Route, poses: list, timestamps: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The logged driver's speed along the route, as (route distance, m/s).

    What the expert drives at where nothing is in the way. It belongs to the place, not to the
    expert's history, so the same state always gets the same label.
    """
    points = np.asarray([np.asarray(p)[:2, 3] for p in poses])
    stamps = np.asarray(timestamps, dtype=float) / 1e6
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.maximum(np.diff(stamps), 1e-6)
    along = np.asarray([route.project(p)[0] for p in points[:-1]])
    order = np.argsort(along)
    return along[order], speeds[order]


class Eroded:
    """A drivable area shrunk by `margin`: a point counts only if its neighbourhood does too."""

    def __init__(self, area, margin: float) -> None:
        self.area = area
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.offsets = margin * np.column_stack([np.cos(angles), np.sin(angles)])

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        inside = np.asarray(self.area.contains(points), dtype=bool)
        for offset in self.offsets:
            inside &= np.asarray(self.area.contains(points + offset), dtype=bool)
        return inside


class Teacher:
    """Expert v6 for one episode. Call `teach` on each tick, in time order.

    Memory is used for one thing only: the side, once chosen, is kept. Everything else -- when to
    act, how hard to brake, where to steer, what speed to return to -- follows from the state and
    the actors' futures, so a tick's label is the one a fresh expert would give that state.
    """

    def __init__(
        self,
        route_world: np.ndarray,
        actors: list[dict],
        area=None,
        cruise: tuple[np.ndarray, np.ndarray] | None = None,
        trigger_s: float = TRIGGER_S,
        min_px: float = MIN_PX,
    ) -> None:
        self.route = Route(route_world)
        self.actors = [guarded(a) for a in actors if a.get("cls_name", "car") in COLLISION_CLASSES]
        self.raw_area = area
        self.eroded_area = None if area is None else Eroded(area, ROAD_MARGIN_M)
        self.area = self.eroded_area
        self.cruise = cruise
        self.trigger_s = trigger_s
        self.min_px = min_px
        self.manoeuvre: Manoeuvre | None = None

    def cruise_at(self, s: float, speed: float) -> float:
        if self.cruise is None or not len(self.cruise[0]):
            return speed
        return float(np.interp(s, self.cruise[0], self.cruise[1]))

    # -- grading ---------------------------------------------------------------------------------------

    def _near(self, ego2world: np.ndarray, t0: int) -> list[dict]:
        here = ego2world[:2, 3]
        near = []
        for actor in self.actors:
            state = actor_at(actor, t0)
            if state is not None and np.linalg.norm(state[0] - here) < NEAR_M:
                near.append(actor)
        return near

    def _judge(self, waypoints, ego2world, t0, actors) -> Verdict:
        return grade(waypoints, ego2world, t0, actors, self.route.points, self.area)

    def _first_hit(self, waypoints, ego2world, t0, actors) -> tuple[float | None, dict | None]:
        """When this plan first hits, and what -- the earliest over every actor."""
        best: tuple[float | None, dict | None] = (None, None)
        for actor in actors:
            verdict = grade(waypoints, ego2world, t0, [actor])
            if verdict.hit_at is not None and (best[0] is None or verdict.hit_at < best[0]):
                best = (verdict.hit_at, actor)
        return best

    def _clear_ahead(self, ego2world, t0, speed, near) -> bool:
        """Would driving on, back up to at least RELEASE_PROBE_MPS, clear everything for the horizon?"""
        times = np.arange(1, POSES + 1) * WAYPOINT_DT
        s_ego, _ = self.route.project(ego2world[:2, 3])
        probe = resume(speed, times, max(self.cruise_at(s_ego, speed), RELEASE_PROBE_MPS))
        return self._judge(self._draw(ego2world, probe, speed, None), ego2world, t0, near).clear

    def _actor(self, uuid: str) -> dict:
        return next(a for a in self.actors if a["uuid"] == uuid)

    # -- drawing ---------------------------------------------------------------------------------------

    def _draw(self, ego2world: np.ndarray, distances: np.ndarray, speed: float, man: Manoeuvre | None):
        """The car's path to the committed one: route + manoeuvre offset, eased on from where it is."""
        s_ego, d_ego = self.route.project(ego2world[:2, 3])
        target_here = float(man.lateral(np.array([s_ego]))[0]) if man else 0.0
        gap = d_ego - target_here
        rejoin_m = max(
            REJOIN_MIN_M, min_length_for(gap), speed * np.pi * np.sqrt(abs(gap) / (2 * LATERAL_ACCEL_MAX))
        )

        def desired(along: np.ndarray) -> np.ndarray:
            s = s_ego + along
            lateral = (man.lateral(s) if man else np.zeros_like(s)) + gap * (1 - ease(along / rejoin_m))
            return to_ego(self.route.offset_points(s, lateral), ego2world)

        return pursue(desired, distances, speed)

    def _plan(self, ego2world, speed, man: Manoeuvre | None) -> np.ndarray:
        """No manoeuvre, or one that is over: back to the lane at the place's speed. Else continue it."""
        times = np.arange(1, POSES + 1) * WAYPOINT_DT
        s_ego, _ = self.route.project(ego2world[:2, 3])
        if man is None or man.released:
            return self._draw(ego2world, resume(speed, times, self.cruise_at(s_ego, speed)), speed, None)
        stop_after = None if man.stop_s is None else man.stop_s - s_ego
        if stop_after is not None:
            # Too late for the stop point, or already past it: the hardest stop there is. Capped at
            # the room left, this was an all-zero plan at full speed, which grades clear because it
            # never moves and which no car can follow (side-0070/run_8: 11.24 m/s, 0.0 m throughout).
            stop_after = max(stop_after, speed**2 / (2 * BRAKE_MAX))
        decel = man.decel if stop_after is None else stopping_decel(speed, stop_after)
        distances = travel(speed, times, decel, man.floor_mps, stop_after)
        return self._draw(ego2world, distances, speed, man)

    # -- deciding --------------------------------------------------------------------------------------

    def _motion(self, actor: dict, t0: int) -> str:
        velocity = _velocity(actor, t0)
        if np.linalg.norm(velocity) < PARKED_MPS:
            return "parked"
        state = actor_at(actor, t0)
        s_h, _ = self.route.project(state[0])
        _, tangent = self.route.at(np.array([s_h]))
        return "oncoming" if float(velocity @ tangent[0]) < ONCOMING_MPS else "crossing"

    def _commit_brake(self, actor, motion, t0, ego2world, speed, hit_at) -> Manoeuvre:
        """Stop STOP_GAP_M short of the car's guarded box, where it will be when we would reach it.

        Anchored to the car, not to the ego: "current speed x time to contact" moved the stop point
        every tick as the car slowed, so the same state was labelled with different stops.
        """
        s_ego, _ = self.route.project(ego2world[:2, 3])
        position, yaw = actor_at(actor, t0 + hit_at * 1e6)
        s_c, _ = self.route.project(position)
        _, tangent = self.route.at(np.array([s_c]))
        relative = yaw - float(np.arctan2(tangent[0, 1], tangent[0, 0]))
        width, length = actor["dims"][0], actor["dims"][1]
        extent = abs(np.cos(relative)) * length / 2 + abs(np.sin(relative)) * width / 2
        stop_s = s_c - extent - EGO_FRONT_M - STOP_GAP_M
        return Manoeuvre(actor["uuid"], motion, t0, stopping_decel(speed, stop_s - s_ego), 0.0, stop_s)

    def _commit_swerve(self, actor, t0, ego2world, speed, near, sides=(1, -1)) -> Manoeuvre | None:
        """A clear swerve with the full margins; failing that, a tight one (see `_swerve`)."""
        found = self._swerve(actor, t0, ego2world, speed, near, sides, PASS_MARGIN_M)
        if found is not None or self.raw_area is None:
            return found
        # On a narrow road (frontal-0058: 6.6 m across) no pass keeps 0.75 m from the car AND 0.3 m
        # from the kerb, and the expert braked into the car instead. A tight pass -- the guard's 0.3 m
        # beside the car, the raw road edge -- beats a crash; it still never leaves the road.
        wide, self.area = self.area, self.raw_area
        try:
            return self._swerve(actor, t0, ego2world, speed, near, sides, 0.0)
        finally:
            self.area = wide

    def _swerve(self, actor, t0, ego2world, speed, near, sides, margin) -> Manoeuvre | None:
        """Brake, then pass on the side that needs the least further move and stays on the road and clear.

        None while the meeting point is still beyond the horizon: until then braking is the whole
        answer, and the swerve is planned on the first tick that can see where it has to happen.
        """
        s_ego, d_ego = self.route.project(ego2world[:2, 3])
        base = Manoeuvre(actor["uuid"], "oncoming", t0, ONCOMING_DECEL, ONCOMING_FLOOR_MPS, None)
        ego_s = s_ego + travel(speed, HORIZON, base.decel, base.floor_mps, None)
        meet = None
        for k, tau in enumerate(HORIZON):
            state = actor_at(actor, t0 + tau * 1e6)
            if state is None:
                continue
            s_h, d_h = self.route.project(state[0])
            if s_h - ego_s[k] <= EGO_FRONT_M + actor["dims"][1] / 2 + 1.0:
                meet = (k, s_h, d_h)
                break
        if meet is None:
            return None
        k, s_h, d_h = meet
        clearance = EGO_WIDTH / 2 + actor["dims"][0] / 2 + margin
        options = []
        for side in sides:
            # Pass on `side` of it: our distance from the route must exceed its own by the clearance.
            offset = d_h + side * clearance
            if side * offset > TRUST_BAND_M - 0.1:
                # Just past the trust band: try the widest move it allows; the grader says if it clears.
                offset = side * (TRUST_BAND_M - 0.1)
            if side * offset < 0 and abs(offset) > TRUST_BAND_M:
                continue
            ramp = max(
                min_length_for(offset, MIN_TURN_RADIUS_M),
                ONCOMING_FLOOR_MPS * np.pi * np.sqrt(abs(offset) / (2 * SWERVE_LATERAL_ACCEL)),
            )
            swerve_s = max(s_ego, ego_s[k] - SWERVE_LEAD_M - ramp)
            # Already partway across toward this offset: lay the move out so it is that far along
            # here. The car's own position then carries the commitment, and a fresh look at the
            # state gives the same plan as the expert that started the move.
            done = d_ego / offset if offset else 0.0
            if 0.05 < done < 1.0:
                swerve_s = min(swerve_s, s_ego - ramp * float(np.arccos(1 - 2 * done) / np.pi))
            elif done >= 1.0:
                swerve_s = min(swerve_s, s_ego - ramp)
            man = replace(
                base, side=side, offset_m=float(offset), swerve_s=float(swerve_s), ramp_m=float(ramp)
            )
            plan = self._plan(ego2world, speed, man)
            if undrivable(plan):
                continue
            verdict = self._judge(plan, ego2world, t0, near)
            options.append((not verdict.clear, abs(offset - d_ego), man))
        if not options:
            return None
        # Only a swerve that clears is committed to. With none, braking continues and the escalation
        # (and in the end least-impact) decides; v6 once took a swerve that left the road because it
        # was the only one drawn (frontal-0058).
        clear = sorted((o for o in options if not o[0]), key=lambda o: o[1])
        return clear[0][2] if clear else None

    def _escalate(
        self, man: Manoeuvre, ego2world, t0, speed, near
    ) -> tuple[Manoeuvre, np.ndarray, Verdict] | None:
        """The committed manoeuvre no longer clears: the smallest change that does, or None."""
        s_ego, _ = self.route.project(ego2world[:2, 3])
        tries = []
        if man.motion == "oncoming" and man.side:
            # Same side first: start across now if it has not begun, then a little wider.
            start = s_ego if man.swerve_s is None else min(man.swerve_s, s_ego)
            for extra in (0.0, 0.5, 1.0):
                tries.append(replace(man, offset_m=man.offset_m + man.side * extra, swerve_s=start))
        if man.motion == "oncoming":
            # Then both sides again from here; this is the only way the side ever changes.
            fresh = self._commit_swerve(self._actor(man.hazard), t0, ego2world, speed, near)
            if fresh is not None:
                tries.append(replace(fresh, started_us=man.started_us))
        if man.stop_s is not None:
            # A stop that no longer clears: stop sooner (the rate follows from the stop point).
            for sooner in (1.0, 2.0, 4.0):
                tries.append(replace(man, stop_s=man.stop_s - sooner))
            tries.append(replace(man, stop_s=s_ego))
        else:
            for decel in (man.decel + 1.0, BRAKE_MAX):
                tries.append(replace(man, decel=min(decel, BRAKE_MAX)))
        for candidate in tries:
            plan = self._plan(ego2world, speed, candidate)
            if undrivable(plan):
                continue
            verdict = self._judge(plan, ego2world, t0, near)
            if verdict.clear:
                return candidate, plan, verdict
        return None

    def teach(self, ego2world: np.ndarray, t0: int, speed: float, ego2world_before=None) -> Taught:
        """`ego2world_before` is the pose 0.5 s earlier: the model reads that frame too, so a car in
        either frame counts as seen."""
        ego2world = np.asarray(ego2world, dtype=float)
        near = self._near(ego2world, t0)
        # The margin is for where plans go, not where the car already is: a car starting inside it
        # would fail every plan from the first step, and the choice between them becomes arbitrary.
        self.area = self.eroded_area
        if (
            self.raw_area is not None
            and leaves_road(np.zeros((1, 3)), ego2world, self.eroded_area) is not None
        ):
            self.area = self.raw_area
        man = self.manoeuvre
        px = None

        if man is None or man.released:
            nominal = self._plan(ego2world, speed, None)
            hit_at, actor = self._first_hit(nominal, ego2world, t0, near)
            if actor is not None:
                state = actor_at(actor, t0)
                px = hazard_px(ego2world, (state, actor["true_dims"])) if state is not None else 0.0
                before = actor_at(actor, t0 - 0.5e6)
                if ego2world_before is not None and before is not None:
                    px = max(px, hazard_px(np.asarray(ego2world_before), (before, actor["true_dims"])))
            # Seen well enough to react to, or too close to wait, or already out of the lane: a car
            # mid-swerve has the hazard at the edge of the frame, but its own position shows the move.
            off_lane = abs(self.route.project(ego2world[:2, 3])[1]) > OFF_LANE_M
            act = (
                hit_at is not None
                and hit_at <= self.trigger_s
                and (px >= self.min_px or (px > 0 and hit_at <= EMERGENCY_S) or off_lane)
            )
            if not act:
                kind = "continue" if man is None else "resume"
                return Taught(nominal, kind, self._judge(nominal, ego2world, t0, near), man, px)
            motion = self._motion(actor, t0)
            swerve = self._commit_swerve(actor, t0, ego2world, speed, near) if motion == "oncoming" else None
            if swerve is not None:
                man = swerve
            elif motion == "oncoming":
                # The meeting point is beyond the horizon: brake toward the floor; steer once it is in view.
                man = Manoeuvre(actor["uuid"], "oncoming", t0, ONCOMING_DECEL, ONCOMING_FLOOR_MPS, None)
            else:
                man = self._commit_brake(actor, motion, t0, ego2world, speed, hit_at)
        elif man.motion == "oncoming" and not man.side:
            # Braking so far; plan the swerve as soon as the meeting point comes into the horizon.
            swerve = self._commit_swerve(self._actor(man.hazard), t0, ego2world, speed, near)
            if swerve is not None:
                man = replace(swerve, started_us=man.started_us)
        self.manoeuvre = man

        plan = self._plan(ego2world, speed, man)
        verdict = self._judge(plan, ego2world, t0, near)
        if not verdict.clear and verdict.hit_at is not None:
            changed = self._escalate(man, ego2world, t0, speed, near)
            if changed is not None:
                man, plan, verdict = changed
                self.manoeuvre = man

        # Drive on: an oncoming car once it is behind us, anything else once carrying on is clear.
        behind = True
        if man.motion == "oncoming":
            state = actor_at(self._actor(man.hazard), t0)
            s_ego, _ = self.route.project(ego2world[:2, 3])
            behind = state is None or self.route.project(state[0])[0] < s_ego + EGO_FRONT_M
        if behind and self._clear_ahead(ego2world, t0, speed, near):
            man = replace(man, released=True)
            plan = self._plan(ego2world, speed, man)
            verdict = self._judge(plan, ego2world, t0, near)
            self.manoeuvre = man

        if not verdict.clear and verdict.hit_at is not None:
            # Nothing clears: the gentlest contact on the committed side. For an oncoming car that is
            # the hardest braking. A parked or crossing car keeps its stop point, which sets the rate
            # in `_plan`, so `decel` changes nothing there. Left so, because Gate B measured that.
            hardest = replace(man, decel=BRAKE_MAX, floor_mps=0.0, released=False)
            plan = self._plan(ego2world, speed, hardest)
            return Taught(plan, "least-impact", self._judge(plan, ego2world, t0, near), man, px)
        if man.released:
            kind = "resume"
        elif man.motion == "oncoming" and man.offset_m:
            kind = "brake+swerve"
        elif speed < 0.2 and man.stop_s is not None:
            kind = "stop"
        else:
            kind = "brake"
        return Taught(plan, kind, verdict, man, px)
