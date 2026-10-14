"""Expert v6, one test per behaviour it promises, each checked by driving it.

A straight road along +x, actors scripted as neuro-ncap scripts them (they never react), and a
follower that drives 0.1 s of every plan the way the plan's own spacing says -- not neuro-ncap's
tracker, which needs torch; `rollout expert` drives it through the real engine. What these pin down
is the expert's decisions: brake, hold the stop, brake before steering, keep the side.
"""

from __future__ import annotations

import numpy as np

from gallops.expert.grader import EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE, actor_at, corners, overlap
from gallops.expert.teacher import Route, Teacher

DT = 0.1
ROUTE = np.column_stack([np.arange(-20.0, 400.0, 1.0), np.zeros(420)])


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[:2, 3] = (x, y)
    return pose


def _actor(uuid, start_xy, velocity, yaw, dims=(1.9, 4.5, 1.6), seconds=20.0, target=True) -> dict:
    stamps = np.arange(0.0, seconds + 1e-9, 0.5)
    poses = [
        _pose(start_xy[0] + velocity[0] * t, start_xy[1] + velocity[1] * t, yaw).tolist() for t in stamps
    ]
    return {
        "uuid": uuid,
        "timestamps": (stamps * 1e6).astype(np.int64).tolist(),
        "poses": poses,
        "dims": list(dims),
        "is_target_actor": target,
        "cls_name": "car",
    }


def _advance(ego2world: np.ndarray, plan: np.ndarray) -> tuple[np.ndarray, float]:
    """Move 0.1 s along the plan: arc length from a quadratic through its first two waypoints."""
    points = np.vstack([[0.0, 0.0], plan[:, :2]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    a, b, _ = np.polyfit([0.0, 0.5, 1.0], arc[:3], 2)
    travelled = max(a * DT**2 + b * DT, 0.0)
    speed = max(2 * a * DT + b, 0.0)
    x, y = np.interp(travelled, arc, points[:, 0]), np.interp(travelled, arc, points[:, 1])
    ahead = min(travelled + 0.5, arc[-1])
    hx, hy = np.interp(ahead, arc, points[:, 0]) - x, np.interp(ahead, arc, points[:, 1]) - y
    heading = np.arctan2(hy, hx) if np.hypot(hx, hy) > 0.05 else 0.0
    return ego2world @ _pose(x, y, heading), float(speed)


def _hits(ego2world: np.ndarray, t_us: int, actors: list[dict]) -> bool:
    yaw = np.arctan2(ego2world[1, 0], ego2world[0, 0])
    centre = ego2world[:2, 3] + REAR_AXLE_TO_CENTRE * np.array([np.cos(yaw), np.sin(yaw)])
    ego = corners(centre, yaw, EGO_LENGTH, EGO_WIDTH)
    for actor in actors:
        state = actor_at(actor, t_us)
        if state is not None and overlap(
            ego, corners(state[0], state[1], actor["dims"][1], actor["dims"][0])
        ):
            return True
    return False


def _drive(actors, seconds=8.0, speed=8.0, area=None):
    teacher = Teacher(ROUTE, actors, area, cruise=(np.array([0.0, 400.0]), np.array([speed, speed])))
    ego, log = _pose(0.0, 0.0, 0.0), []
    for k in range(int(seconds / DT)):
        t_us = int(k * DT * 1e6)
        taught = teacher.teach(ego, t_us, speed)
        log.append(
            {
                "t": k * DT,
                "kind": taught.kind,
                "side": 0 if taught.manoeuvre is None else taught.manoeuvre.side,
                "x": float(ego[0, 3]),
                "y": float(ego[1, 3]),
                "speed": speed,
                "plan": np.asarray(taught.waypoints),
                "ego": ego.copy(),
                "hit": _hits(ego, t_us, actors),
            }
        )
        ego, speed = _advance(ego, np.asarray(taught.waypoints))
    return log


def test_it_brakes_for_a_parked_car_stops_short_and_stays_stopped():
    log = _drive([_actor("parked", (40.0, 0.0), (0.0, 0.0), 0.0)])
    assert not any(t["hit"] for t in log)
    assert {t["kind"] for t in log} <= {"continue", "brake", "stop"}
    assert "stop" in {t["kind"] for t in log}
    assert max(abs(t["y"]) for t in log) < 0.5
    stopped = [t for t in log if t["speed"] < 0.2]
    gap = 40.0 - 4.5 / 2 - (stopped[0]["x"] + REAR_AXLE_TO_CENTRE + EGO_LENGTH / 2)
    assert gap >= 2.0
    assert max(t["x"] for t in stopped) - stopped[0]["x"] < 0.3  # holds: no creep


def test_it_still_stops_for_a_car_whose_logged_trajectory_ends_early():
    # A scenario actor is logged for the scenario's duration only; past its last stamp the
    # grader would call it gone, and the old release check drove at it.
    log = _drive([_actor("parked", (40.0, 0.0), (0.0, 0.0), 0.0, seconds=5.0)])
    assert "stop" in {t["kind"] for t in log}
    assert max(t["x"] for t in log) < 40.0 - 4.5 / 2 - EGO_LENGTH


def test_it_brakes_for_a_crossing_car_and_drives_on_once_it_has_passed():
    crossing = _actor("crossing", (35.0, -33.0), (0.0, 8.0), np.pi / 2)
    log = _drive([crossing], seconds=10.0)
    assert not any(t["hit"] for t in log)
    kinds = [t["kind"] for t in log]
    assert "brake" in kinds and kinds[-1] in ("resume", "continue")
    assert max(abs(t["y"]) for t in log) < 0.5


def test_for_an_oncoming_car_it_brakes_first_then_steers_round_it_on_one_side():
    oncoming = _actor("oncoming", (90.0, 0.5), (-6.0, 0.0), np.pi)
    log = _drive([oncoming])
    assert not any(t["hit"] for t in log)
    acted = next(t for t in log if t["kind"] != "continue")
    moved = next(t for t in log if abs(t["y"] - acted["y"]) > 0.3)
    assert moved["speed"] < acted["speed"] - 0.5  # slower before it steers
    sides = {t["side"] for t in log if t["kind"] == "brake+swerve"}
    assert len(sides) == 1  # never flips
    assert min(t["speed"] for t in log) >= 2.5  # keeps rolling: a stopped car cannot steer


class _LeftBlocked:
    """Drivable everywhere except more than 2.0 m left of the route: the kerb of a 4 m lane."""

    def contains(self, points):
        return np.atleast_2d(points)[:, 1] <= 2.0


def test_with_the_near_side_blocked_it_passes_on_the_other():
    # Slightly right of centre, so passing on its left is the smaller move -- but the left is kerb.
    oncoming = _actor("oncoming", (90.0, -0.5), (-6.0, 0.0), np.pi)
    log = _drive([oncoming], area=_LeftBlocked())
    assert not any(t["hit"] for t in log)
    assert {t["side"] for t in log if t["kind"] == "brake+swerve"} == {-1}
    assert max(t["y"] for t in log) <= 2.0 - 0.925  # the car body stays on the road


def _at(plan: np.ndarray, times: np.ndarray) -> np.ndarray:
    stamps = np.r_[0.0, np.arange(1, len(plan) + 1) * 0.5]
    points = np.vstack([[0.0, 0.0], plan[:, :2]])
    return np.column_stack([np.interp(times, stamps, points[:, k]) for k in (0, 1)])


def test_each_tick_continues_the_last_ticks_plan():
    """The plan at tick k+1 is tick k's plan 0.1 s further on -- nothing is redrawn from "now"."""
    oncoming = _actor("oncoming", (90.0, 0.5), (-6.0, 0.0), np.pi)
    log = _drive([oncoming])
    worst, compared = 0.0, 0
    for before, after in (log[i : i + 2] for i in range(len(log) - 1)):  # py3.9: no zip(strict=)
        if before["kind"] not in ("brake", "brake+swerve") or after["kind"] != before["kind"]:
            continue
        times = np.arange(1, 7) * 0.5
        expected_world = before["ego"][:2, :2] @ _at(before["plan"], times + DT).T + before["ego"][:2, 3:4]
        got_world = after["ego"][:2, :2] @ _at(after["plan"], times).T + after["ego"][:2, 3:4]
        worst = max(worst, float(np.abs(expected_world - got_world).max()))
        compared += 1
    assert compared >= 20  # the manoeuvre ran long enough to test
    assert worst < 0.2


def test_the_route_reads_distance_along_and_to_the_left():
    route = Route(ROUTE)
    s, left = route.project(np.array([12.3, 1.5]))
    assert abs(s - 32.3) < 1e-9 and abs(left - 1.5) < 1e-9
    assert route.project(np.array([5.0, -2.0]))[1] == -2.0
