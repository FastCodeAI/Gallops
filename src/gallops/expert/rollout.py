"""neuro-ncap's closed loop, offline: the engine's own step, vehicle model and scoring, with no renderer.

Every expert so far was checked one tick at a time -- each plan graded as if it were followed for four
seconds -- and never by driving it. The failures that mattered only exist in closed loop: braking put off
again on every tick, a swerve side that flips between ticks. They surfaced after training, in neuroNCAP.
This drives a plan source the way the benchmark does instead: re-planned every 0.1 s, the ego moved by the
engine's LQR tracker and kinematic bicycle, collisions and the NCAP score from the engine's own evaluator.
The other cars are scripted and never react to the ego, so the paths the engine logged (`actors.json`) are
exactly the paths they drive whatever the ego does.

How faithful this is, is a measurement rather than an argument: `replay` feeds back the plans a recorded
episode actually drove and must reproduce its logged ego poses and metrics. A new plan source -- the
expert -- is only worth driving through it once that holds.

Nothing here re-derives what the engine reads. The scene's logged data (keyframe poses, CAN bus, steering,
actor classes) is what neuro-ncap's own NuScenesAPI returns, saved once per scene by `pack`, and the loop
calls `Engine.step`, `Engine._compute_collision_reference_speed` and `Evaluator.accumulate` themselves.

Runs in the ncap environment (torch, nuplan-devkit, nuscenes-devkit), as `lanes.py` does, and from the
neuro-ncap checkout with it on PYTHONPATH: importing neuro_ncap opens fonts by a path relative to the
working directory, so it fails anywhere else.

    <ncap python> -m gallops.expert.rollout pack --scene 0053 --scene 0058 --nuscenes <root> --out <packs>
    <ncap python> -m gallops.expert.rollout replay --collect <dir> --dumps <dir> --packs <dir> \\
        --scenarios <dir> --out <report.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

#: The waypoint spacing paddock's harness and the training collection hand the engine
#: (`--engine.waypoint-dt`). The scenarios step at 0.1 s; plans stay on the 0.5 s grid.
WAYPOINT_DT = 0.5
#: neuroNCAP follows exactly six waypoints.
N_WAYPOINTS = 6
#: Gate A: replayed poses must land this close to the logged ones.
POSE_TOLERANCE_M = 0.01
#: ...and the score this close (the evaluator's arithmetic is float32 in places).
SCORE_TOLERANCE = 1e-3


# --- the scene, as neuro-ncap's NuScenesAPI sees it ------------------------------------------------------


def pack_scene(scene: str, nusc, out_dir: Path) -> Path:
    """Save what the engine reads from NuScenesAPI for one scene, so a rollout needs no devkit load.

    `nusc` is a loaded `nuscenes.NuScenes`; loading v1.0-trainval costs minutes and several GB,
    so one process packs every scene it is given.
    """
    from neuro_ncap.components.nuscenes_api import NuScenesConfig

    api = NuScenesConfig(sequence=int(scene)).setup(nusc=nusc)
    api.get_actors()  # fills uuid_cls_map, the classes the collision checker filters on
    frames = range(api.max_frame_idx + 1)
    pack = {
        "scene": f"scene-{int(scene):04d}",
        "timestamps": [int(api.get_timestamp(i)) for i in frames],
        "ego_poses": [api.get_ego_pose(i).tolist() for i in frames],
        # float32 in the engine; tolist() keeps each value exactly and load() casts back
        "canbus": [api.get_canbus(i).tolist() for i in frames],
        "steering": [api.get_steering_angle(i).item() for i in frames],
        "reference_trajectory": api.get_reference_trajectory().tolist(),
        "uuid_cls": api.uuid_cls_map,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pack['scene']}.json"
    out.write_text(json.dumps(pack))
    return out


#: How far ahead an episode's route reaches: 8 s at 15 m/s is 120 m, plus a 4 s plan past the end.
ROUTE_AHEAD_M = 200.0


def add_routes(pack_path: Path, nuscenes_root: Path) -> int:
    """Add, for every keyframe, the lane centreline ahead of the logged pose (`lanes.lane_path`).

    The expert measures an episode against ONE route, taken from the pose the closed loop starts
    at, so a swerve is an offset from the lane the car was in rather than from whichever lane it
    has drifted into. Needs the map API, hence the ncap environment; stored in the pack so the
    labeller, which runs elsewhere, reads the same route.
    """
    from nuscenes.map_expansion.map_api import NuScenesMap

    from .lanes import lane_path

    pack = json.loads(pack_path.read_text())
    tables = nuscenes_root / "v1.0-trainval"
    logs = {entry["token"]: entry["location"] for entry in json.loads((tables / "log.json").read_text())}
    city = {s["name"]: logs[s["log_token"]] for s in json.loads((tables / "scene.json").read_text())}
    nusc_map = NuScenesMap(dataroot=str(nuscenes_root), map_name=city[pack["scene"]])
    guide = np.asarray(pack["reference_trajectory"], dtype=float)[:, :2]
    routes = {}
    for frame, pose in enumerate(pack["ego_poses"]):
        pose = np.asarray(pose)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        path = lane_path(nusc_map, float(pose[0, 3]), float(pose[1, 3]), yaw, guide, ahead_m=ROUTE_AHEAD_M)
        if path is not None:
            routes[str(frame)] = np.round(path, 2).tolist()
    pack["routes"] = routes
    pack["location"] = city[pack["scene"]]
    pack_path.write_text(json.dumps(pack))
    return len(routes)


class PackDataset:
    """The part of NuScenesAPI that `Engine.step` and its helpers call, served from a pack.

    Each getter returns a fresh tensor of the dtype NuScenesAPI returns, because the engine
    writes into the poses it is handed.
    """

    def __init__(self, pack: dict) -> None:
        self.scene = pack["scene"]
        self.timestamps = [int(t) for t in pack["timestamps"]]
        self._poses = pack["ego_poses"]
        self._canbus = pack["canbus"]
        self._steering = pack["steering"]
        self._reference = pack["reference_trajectory"]
        self.uuid_cls_map = pack["uuid_cls"]
        self.routes = {int(k): np.asarray(v, dtype=float) for k, v in pack.get("routes", {}).items()}

    @classmethod
    def load(cls, path: Path) -> PackDataset:
        return cls(json.loads(path.read_text()))

    def get_timestamp(self, frame: int) -> int:
        return self.timestamps[frame]

    def get_ego_pose(self, frame: int):
        import torch

        return torch.tensor(self._poses[frame], dtype=torch.float64)

    def get_canbus(self, frame: int):
        import torch

        return torch.tensor(self._canbus[frame], dtype=torch.float32)

    def get_steering_angle(self, frame: int):
        import torch

        return torch.tensor(self._steering[frame], dtype=torch.float32)

    def get_reference_trajectory(self):
        import torch

        return torch.tensor(self._reference, dtype=torch.float64)


# --- one episode -------------------------------------------------------------------------------------------


def start_frame_for(general: dict, seed: int) -> int:
    """The engine's start frame for run `seed` (Engine._setup_scenario: one draw from a fresh rng)."""
    start = int(general["start-frame"])
    jitter = int(general.get("start-jitter", 0) or 0)
    if jitter:
        rng = np.random.default_rng(seed)
        start = max(0, start + int(rng.integers(-jitter, jitter + 1)))
    return start


def make_engine(dataset: PackDataset, general: dict, start_frame: int, waypoint_dt: float = WAYPOINT_DT):
    """An Engine with only what `step` and the reference speed read: no renderer, model or logger."""
    from neuro_ncap.engine import Engine

    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace(waypoint_dt=waypoint_dt)
    engine.dataset = dataset
    engine.scenario = SimpleNamespace(general=SimpleNamespace(priming_steps=int(general["priming-steps"])))
    engine.timestep = float(general["time-step"])
    engine._vehicle_model_discretization_time = 0.1
    sim_steps = engine.timestep / engine._vehicle_model_discretization_time
    if not float(sim_steps).is_integer():
        raise ValueError(f"vehicle model steps per tick must be an integer, got {sim_steps}")
    engine._vehicle_model_sim_steps = int(sim_steps)
    engine.vehicle_model = None
    engine.start_frame = start_frame
    engine.end_time = dataset.get_timestamp(start_frame) + int(float(general["duration"]) * 1e6)
    engine._bev_reference_trajectory = dataset.get_reference_trajectory()
    return engine


def load_actors(path: Path, uuid_cls: dict[str, str]) -> list:
    """`actors.json` back into the engine's ActorTrajectory list; classes from the scene's annotations."""
    from neuro_ncap.structures import ActorTrajectory

    actors = []
    for actor in json.loads(path.read_text()):
        actors.append(
            ActorTrajectory(
                timestamps=np.asarray(actor["timestamps"], dtype=np.int64),
                poses=np.asarray(actor["poses"], dtype=np.float64),
                dims=np.asarray(actor["dims"], dtype=np.float64),
                cls_name=uuid_cls.get(actor["uuid"], "unknown"),
                uuid=actor["uuid"],
                is_target_actor=bool(actor["is_target_actor"]),
            )
        )
    return actors


def _finish(coroutine):
    """Run a coroutine that never awaits (Engine.step) without an event loop per tick."""
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    coroutine.close()
    raise RuntimeError("Engine.step awaited something; this rollout assumes it does not")


#: plan_fn(state) -> the (6, 2) float32 plan the engine is handed at that state.
PlanFn = Callable[[object], object]


def drive(engine, actors: list, plan_fn: PlanFn) -> dict:
    """The engine's run loop with `plan_fn` in place of the model and the renderer.

    Same order as `Engine.run`: a state is scored (`Evaluator.accumulate`) before the ego is
    stepped from it; the run ends at the first state whose check collided, or once the clock
    reaches the scenario's duration -- that last state is never scored, as in the engine.
    """
    from neuro_ncap.components.evaluator import Evaluator, EvaluatorConfig
    from neuro_ncap.components.model_api import ModelOutput
    from neuro_ncap.structures import State

    targets = [actor for actor in actors if actor.is_target_actor]
    target = targets[0] if targets else None
    reference_speed = engine._compute_collision_reference_speed(target)
    evaluator = Evaluator(EvaluatorConfig(), target_actor=target, reference_speed=reference_speed)
    dataset, start = engine.dataset, engine.start_frame
    state = State(
        ego2world=dataset.get_ego_pose(start),
        timestamp=dataset.get_timestamp(start),
        canbus=dataset.get_canbus(start),
        current_step=0,
        is_in_auto_mode=False,
    )
    ticks = []
    collided = evaluator.collision_at_times[0]
    while state.timestamp < engine.end_time and not (collided and collided[-1] is not None):
        plan = plan_fn(state)
        snapshot = state.copy()
        evaluator.accumulate(snapshot, ModelOutput(trajectory=plan, aux_output=None), actors)
        ticks.append(
            {
                "timestamp": int(snapshot.timestamp),
                "auto": bool(snapshot.is_in_auto_mode),
                "ego2world": snapshot.ego2world.numpy().copy(),
                "speed": float(snapshot.speed),
                "accel": float(snapshot.acceleration[0]),
                "plan": np.asarray(plan, dtype=np.float32),
            }
        )
        _finish(engine.step(state, plan))
    hit = next((c for c in collided if c is not None), None)
    return {
        "ticks": ticks,
        "metrics": evaluator.compute_metrics(),
        "collision": None if hit is None else hit.to_json_dict(),
        "reference_speed": reference_speed,
    }


# --- replaying a recorded episode (Gate A) -----------------------------------------------------------------


@dataclass
class Episode:
    directory: Path
    shape: str
    scene: str
    seed: int

    @classmethod
    def from_dir(cls, directory: Path) -> Episode:
        shape, scene = directory.parent.name.split("-", 1)
        return cls(directory, shape, scene, int(directory.name.removeprefix("run_")))


def _dump_index(dumps: Path) -> list[dict]:
    """Every dump's rows, reduced to what replay needs, read once."""
    index = []
    for path in sorted(dumps.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        index.append(
            {r["timestamp"]: np.asarray(r["trajectory"], dtype=np.float64)[:N_WAYPOINTS, :2] for r in rows}
        )
    return index


def match_rows(plans: dict[int, list], index: list[dict]) -> dict | None:
    """The dump that holds this episode's ticks -- `label.match_dump`'s rule, on rows already read."""
    stamps = set(plans)
    for rows in index:
        if stamps and stamps <= set(rows):
            drift = max(np.abs(rows[t] - np.asarray(plans[t])).max() for t in stamps)
            if drift < 1e-3:
                return rows
    return None


def recorded_plans(directory: Path, dump_rows: dict) -> dict[int, np.ndarray]:
    """Each state's plan as the engine received it.

    `trajectories.json` holds the engine's own float32 copy, but only for states in auto mode;
    the first closed-loop step follows the plan made at the last priming state, which only the
    dump recorded (rounded to 1e-4 m).
    """
    plans = {t: np.asarray(p, dtype=np.float32) for t, p in dump_rows.items()}
    for stamp, plan in json.loads((directory / "trajectories.json").read_text()).items():
        plans[int(stamp)] = np.asarray(plan, dtype=np.float32)
    return plans


def _yaw_deg(pose: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))


def replay_episode(episode: Episode, packs: Path, scenarios: Path, dump_rows: dict) -> dict:
    import torch
    import yaml

    general = yaml.safe_load((scenarios / episode.shape / f"{episode.scene}.yaml").read_text())["general"]
    dataset = PackDataset.load(packs / f"scene-{episode.scene}.json")
    start = start_frame_for(general, episode.seed)
    engine = make_engine(dataset, general, start)
    actors = load_actors(episode.directory / "actors.json", dataset.uuid_cls_map)
    plans = recorded_plans(episode.directory, dump_rows)

    def plan_fn(state):
        stamp = int(state.timestamp)
        if state.current_step < engine.scenario.general.priming_steps and stamp not in plans:
            return torch.zeros(N_WAYPOINTS, 2)  # priming states: the engine ignores their plan
        return torch.from_numpy(plans[stamp])

    result = drive(engine, actors, plan_fn)
    logged = {
        int(t): np.asarray(p)
        for t, p in json.loads((episode.directory / "ego_poses.json").read_text()).items()
    }
    driven = {tick["timestamp"]: tick["ego2world"] for tick in result["ticks"] if tick["auto"]}
    shared = sorted(set(logged) & set(driven))
    position = max(
        (float(np.linalg.norm(driven[t][:2, 3] - logged[t][:2, 3])) for t in shared), default=np.inf
    )
    heading = max((abs(_yaw_deg(driven[t]) - _yaw_deg(logged[t])) for t in shared), default=np.inf)
    recorded = json.loads((episode.directory / "metrics.json").read_text())
    replayed = result["metrics"]
    same_ticks = set(logged) == set(driven)
    same_collision = bool(recorded["any_collide@0.0s"]) == bool(replayed["any_collide@0.0s"])
    score_gap = abs(float(recorded["ncap_score"]) - float(replayed["ncap_score"]))
    return {
        "episode": f"{episode.shape}-{episode.scene}/run_{episode.seed}",
        "start_frame": start,
        "ticks_logged": len(logged),
        "ticks_replayed": len(driven),
        "max_position_error_m": position,
        "max_heading_error_deg": heading,
        "recorded": {"collided": bool(recorded["any_collide@0.0s"]), "ncap_score": recorded["ncap_score"]},
        "replayed": {"collided": bool(replayed["any_collide@0.0s"]), "ncap_score": replayed["ncap_score"]},
        "passes": bool(
            same_ticks and position < POSE_TOLERANCE_M and same_collision and score_gap < SCORE_TOLERANCE
        ),
    }


def replay(collect: Path, dumps: Path, packs: Path, scenarios: Path, limit: int = 0) -> dict:
    from .label import episode_dirs

    index = _dump_index(dumps)
    rows = []
    for count, directory in enumerate(episode_dirs(collect)):
        if limit and count >= limit:
            break
        plans = {int(t): p for t, p in json.loads((directory / "trajectories.json").read_text()).items()}
        dump_rows = match_rows(plans, index)
        if dump_rows is None:
            rows.append({"episode": str(directory), "passes": False, "problem": "no dump matches"})
            continue
        row = replay_episode(Episode.from_dir(directory), packs, scenarios, dump_rows)
        rows.append(row)
        flag = "ok  " if row["passes"] else "FAIL"
        print(
            f"  {flag} {row['episode']}: {row['ticks_replayed']}/{row['ticks_logged']} ticks, "
            f"pose {row['max_position_error_m']:.4f} m / {row['max_heading_error_deg']:.3f} deg, "
            f"score {row['recorded']['ncap_score']:.3f} -> {row['replayed']['ncap_score']:.3f}",
            flush=True,
        )
    passed = sum(1 for r in rows if r["passes"])
    finite = [r["max_position_error_m"] for r in rows if "max_position_error_m" in r]
    return {
        "episodes": len(rows),
        "passed": passed,
        "worst_position_error_m": max(finite, default=None),
        "rows": rows,
    }


# --- driving the expert (Gate B) --------------------------------------------------------------------------

#: Kinds that mean the same thing to a model that only sees the present: it cannot tell "resume"
#: from "continue", or a held stop from braking to one.
KIND_GROUP = {
    "continue": "go",
    "resume": "go",
    "brake": "brake",
    "stop": "brake",
    "brake+swerve": "swerve",
    "least-impact": "least-impact",
}


def teacher_actors(path: Path, uuid_cls: dict[str, str]) -> list[dict]:
    """`actors.json` as the grader reads actors, with the class the collision checker filters on."""
    actors = json.loads(path.read_text())
    for actor in actors:
        actor["cls_name"] = uuid_cls.get(actor["uuid"], "unknown")
    return actors


def drive_expert(
    episode: Episode, packs: Path, scenarios: Path, drivable: Path, trigger_s: float, min_px: float
) -> dict:
    """Drive expert v6 through the engine on one recorded episode's scripted actors."""
    import torch
    import yaml

    from .drivable import DrivableArea
    from .teacher import Route, Teacher, human_cruise

    general = yaml.safe_load((scenarios / episode.shape / f"{episode.scene}.yaml").read_text())["general"]
    dataset = PackDataset.load(packs / f"scene-{episode.scene}.json")
    start = start_frame_for(general, episode.seed)
    engine = make_engine(dataset, general, start)
    route = dataset.routes[start + int(general["priming-steps"])]
    area = DrivableArea.load(drivable / f"scene-{episode.scene}.npz")
    actors_dicts = teacher_actors(episode.directory / "actors.json", dataset.uuid_cls_map)
    cruise = human_cruise(Route(route), dataset._poses, dataset.timestamps)
    teacher = Teacher(route, actors_dicts, area, cruise, trigger_s=trigger_s, min_px=min_px)
    log: list[dict] = []
    poses: dict[int, np.ndarray] = {}

    def plan_fn(state):
        if state.current_step < engine.scenario.general.priming_steps:
            return torch.zeros(N_WAYPOINTS, 2)  # the engine replays the log while priming
        ego2world = state.ego2world.numpy()
        poses[int(state.timestamp)] = ego2world
        before = poses.get(int(state.timestamp) - 500_000)
        taught = teacher.teach(ego2world, int(state.timestamp), float(state.speed), before)
        man = taught.manoeuvre
        s, lateral = teacher.route.project(ego2world[:2, 3])
        log.append(
            {
                "timestamp": int(state.timestamp),
                "kind": taught.kind,
                "side": 0 if man is None else man.side,
                "offset_m": 0.0 if man is None else man.offset_m,
                "motion": None if man is None else man.motion,
                "speed": float(state.speed),
                "route_s": s,
                "lateral_m": lateral,
                "hazard_px": taught.hazard_px,
                "ego2world": ego2world.tolist(),
                "plan": np.asarray(taught.waypoints).tolist(),
            }
        )
        return torch.from_numpy(np.asarray(taught.waypoints, dtype=np.float32)[:N_WAYPOINTS, :2].copy())

    result = drive(engine, load_actors(episode.directory / "actors.json", dataset.uuid_cls_map), plan_fn)
    return {
        "episode": f"{episode.shape}-{episode.scene}/run_{episode.seed}",
        "shape": episode.shape,
        "metrics": result["metrics"],
        "collision": result["collision"],
        "ticks": log,
        "route": route,
        "area": area,
        "actors": actors_dicts,
        "cruise": cruise,
    }


def _first(seq, predicate):
    return next((i for i, x in enumerate(seq) if predicate(x)), None)


#: A car first in frame less than this before contact could not have been braked for by a camera driver.
UNSEEN_S = 1.0


def judge_drive(drive_result: dict, trigger_s: float, min_px: float) -> dict:
    """Gate B's measurements on one expert-driven episode."""
    from .grader import EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE, actor_at, corners
    from .teacher import Teacher

    ticks, shape, area = drive_result["ticks"], drive_result["shape"], drive_result["area"]
    collided = bool(drive_result["metrics"]["any_collide@0.0s"])
    sides = [t["side"] for t in ticks if t["kind"] == "brake+swerve" and t["side"]]
    flips = sum(1 for a, b in zip(sides, sides[1:], strict=False) if a != b)
    off_road = 0
    for t in ticks:
        pose = np.asarray(t["ego2world"])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        centre = pose[:2, 3] + REAR_AXLE_TO_CENTRE * np.array([np.cos(yaw), np.sin(yaw)])
        if not area.contains(corners(centre, yaw, EGO_LENGTH, EGO_WIDTH)).all():
            off_road += 1
    widest = max((abs(t["lateral_m"]) for t in ticks), default=0.0)
    row = {
        "episode": drive_result["episode"],
        "shape": shape,
        "collided": collided,
        "ncap_score": float(drive_result["metrics"]["ncap_score"]),
        "collision": drive_result["collision"],
        "kinds": sorted({t["kind"] for t in ticks}),
        "side_flips": flips,
        "off_road_ticks": off_road,
        "widest_lateral_m": widest,
    }
    # A collision with a car the camera could not see in time is the camera's limit, not the expert's:
    # recorded separately (user, 2026-10-13: side cases brake when the car appears).
    row["unseen_collision"] = False
    if collided and drive_result["collision"]:
        from .teacher import hazard_px

        hit_uuid = drive_result["collision"]["colliding_actor"]
        hit = next((a for a in drive_result["actors"] if a["uuid"] == hit_uuid), None)
        hit_at = drive_result["collision"]["collision_time"]
        seen = None
        if hit is not None:
            for t in ticks:
                state = actor_at(hit, t["timestamp"])
                if state is not None and hazard_px(np.asarray(t["ego2world"]), (state, hit["dims"])) > 0:
                    seen = t["timestamp"]
                    break
        row["seen_before_contact_s"] = None if seen is None else (hit_at - seen) / 1e6
        row["unseen_collision"] = seen is None or (hit_at - seen) / 1e6 < UNSEEN_S
    acted = _first(ticks, lambda t: t["kind"] != "continue")
    row["acted_at_s"] = None if acted is None else (ticks[acted]["timestamp"] - ticks[0]["timestamp"]) / 1e6
    row["acted_px"] = None if acted is None else ticks[acted]["hazard_px"]
    if shape == "frontal":
        # Sideways movement counted from where the car was when the expert acted, not from the route:
        # some episodes start half a metre off the lane centre.
        # Only movement TOWARD the swerve side counts: easing back to the lane centre while braking is
        # not a swerve. None when the episode never needed one.
        side = next((t["side"] for t in ticks if t["kind"] == "brake+swerve" and t["side"]), 0)
        start_lateral = None if acted is None else ticks[acted]["lateral_m"]
        moved = (
            None
            if acted is None or not side
            else _first(ticks, lambda t: side * (t["lateral_m"] - start_lateral) > 0.3)
        )
        row["brakes_before_moving"] = (
            None if moved is None else bool(ticks[moved]["speed"] < ticks[acted]["speed"] - 0.5)
        )
    else:
        target = next((a for a in drive_result["actors"] if a["is_target_actor"]), None)
        gaps = []
        for t in ticks:
            if t["speed"] > 0.3 or target is None:
                continue
            state = actor_at(target, t["timestamp"])
            if state is not None:
                centres = float(np.linalg.norm(state[0] - np.asarray(t["ego2world"])[:2, 3]))
                # bumper to bumper, roughly: the ego's front overhang and half the other car
                gaps.append(centres - (REAR_AXLE_TO_CENTRE + EGO_LENGTH / 2) - target["dims"][1] / 2)
        row["stopped_gap_m"] = min(gaps) if gaps else None
        # Sideways excursion beyond the band between where it started and the lane centre: easing back
        # to the centre is not a swerve round the car.
        low, high = min(0.0, ticks[0]["lateral_m"]), max(0.0, ticks[0]["lateral_m"])
        row["sideways_excursion_m"] = max(
            (max(t["lateral_m"] - high, low - t["lateral_m"], 0.0) for t in ticks), default=0.0
        )
    # The label must be predictable from the present: relabel every state with no memory. Kinds
    # are compared in groups the model could tell apart; plans by their first six waypoints, which
    # is all the engine follows.
    kinds = plans = 0
    by_stamp = {t["timestamp"]: np.asarray(t["ego2world"]) for t in ticks}
    for t in ticks:
        fresh = Teacher(
            drive_result["route"],
            drive_result["actors"],
            area,
            drive_result["cruise"],
            trigger_s=trigger_s,
            min_px=min_px,
        )
        before = by_stamp.get(t["timestamp"] - 500_000)
        again = fresh.teach(np.asarray(t["ego2world"]), t["timestamp"], t["speed"], before)
        same_side = (
            t["kind"] != "brake+swerve" or again.manoeuvre is None or again.manoeuvre.side == t["side"]
        )
        kinds += int(KIND_GROUP[again.kind] == KIND_GROUP[t["kind"]] and same_side)
        gap = np.abs(np.asarray(again.waypoints)[:N_WAYPOINTS, :2] - np.asarray(t["plan"])[:N_WAYPOINTS, :2])
        plans += int(gap.max() < 0.5)
    row["memoryless_kind_agreement"] = kinds / max(len(ticks), 1)
    row["memoryless_agreement"] = plans / max(len(ticks), 1)
    return row


def gate_b(
    collect: Path,
    packs: Path,
    scenarios: Path,
    drivable: Path,
    trigger_s: float,
    min_px: float,
    limit: int = 0,
    scenes: list[str] | None = None,
) -> dict:
    from .label import episode_dirs

    rows, ticks_out = [], {}
    directories = [d for d in episode_dirs(collect) if not scenes or Episode.from_dir(d).scene in scenes]
    for count, directory in enumerate(directories):
        if limit and count >= limit:
            break
        result = drive_expert(Episode.from_dir(directory), packs, scenarios, drivable, trigger_s, min_px)
        row = judge_drive(result, trigger_s, min_px)
        rows.append(row)
        ticks_out[row["episode"]] = result["ticks"]
        print(
            f"  {'HIT ' if row['collided'] else 'ok  '}{row['episode']}: {row['ncap_score']:.2f} "
            f"kinds {','.join(row['kinds'])} acted {row['acted_at_s']} s ({row['acted_px']} px) "
            f"flips {row['side_flips']} off-road {row['off_road_ticks']} "
            f"widest {row['widest_lateral_m']:.2f} m agree {row['memoryless_agreement']:.2f}",
            flush=True,
        )
    return {**summarise(rows, trigger_s, min_px), "ticks": ticks_out}


def summarise(rows: list[dict], trigger_s: float, min_px: float) -> dict:
    """Gate B's figures over episodes; collisions with a car never seen in time are counted apart."""
    by_shape = {}
    for shape in sorted({r["shape"] for r in rows}):
        group = [r for r in rows if r["shape"] == shape]
        by_shape[shape] = {
            "episodes": len(group),
            "collision_free": sum(not r["collided"] for r in group) / len(group),
            "unseen_collisions": sum(bool(r.get("unseen_collision")) for r in group),
            "collision_free_seen": sum(not r["collided"] or bool(r.get("unseen_collision")) for r in group)
            / len(group),
            "mean_ncap": float(np.mean([r["ncap_score"] for r in group])),
            "side_flips": sum(r["side_flips"] for r in group),
            "off_road_ticks": sum(r["off_road_ticks"] for r in group),
            "widest_lateral_m": max(r["widest_lateral_m"] for r in group),
            "memoryless_agreement": float(np.mean([r["memoryless_agreement"] for r in group])),
            "memoryless_kind_agreement": float(np.mean([r["memoryless_kind_agreement"] for r in group])),
        }
        if shape == "frontal":
            swerved = [r for r in group if r.get("brakes_before_moving") is not None]
            by_shape[shape]["brakes_before_moving"] = sum(bool(r["brakes_before_moving"]) for r in swerved)
            by_shape[shape]["swerving_episodes"] = len(swerved)
        else:
            stops = [r["stopped_gap_m"] for r in group if r.get("stopped_gap_m") is not None]
            by_shape[shape]["closest_stop_m"] = min(stops) if stops else None
            by_shape[shape]["sideways_excursion_m"] = max(r.get("sideways_excursion_m", 0.0) for r in group)
    return {
        "trigger_s": trigger_s,
        "min_px": min_px,
        "episodes": len(rows),
        "collision_free": sum(not r["collided"] for r in rows) / max(len(rows), 1),
        "collision_free_seen": sum(not r["collided"] or bool(r.get("unseen_collision")) for r in rows)
        / max(len(rows), 1),
        "mean_ncap": float(np.mean([r["ncap_score"] for r in rows])) if rows else None,
        "by_shape": by_shape,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="save each scene's NuScenesAPI data")
    pack.add_argument("--scene", action="append", required=True, help="e.g. 0053; repeatable")
    pack.add_argument("--nuscenes", type=Path, required=True)
    pack.add_argument("--version", default="v1.0-trainval")
    pack.add_argument("--out", type=Path, required=True)
    routes = commands.add_parser("routes", help="add each keyframe's lane route to existing packs")
    routes.add_argument("--packs", type=Path, required=True)
    routes.add_argument("--nuscenes", type=Path, required=True)
    rep = commands.add_parser("replay", help="Gate A: recorded plans must reproduce recorded episodes")
    rep.add_argument("--collect", type=Path, required=True)
    rep.add_argument("--dumps", type=Path, required=True)
    rep.add_argument("--packs", type=Path, required=True)
    rep.add_argument("--scenarios", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--limit", type=int, default=0, help="first N episodes only (a smoke)")
    exp = commands.add_parser("expert", help="Gate B: drive expert v6 on recorded episodes' actors")
    exp.add_argument(
        "--collect", type=Path, required=True, help="episodes whose scripted actors to drive among"
    )
    exp.add_argument("--packs", type=Path, required=True)
    exp.add_argument("--scenarios", type=Path, required=True)
    exp.add_argument("--drivable", type=Path, required=True)
    exp.add_argument("--trigger-s", type=float, default=3.5)
    exp.add_argument("--min-px", type=float, default=16.0)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--limit", type=int, default=0)
    exp.add_argument("--scene", action="append", default=None, help="only these scenes; repeatable")
    merge = commands.add_parser("merge", help="join Gate B reports run per scene into one")
    merge.add_argument("parts", type=Path, nargs="+")
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "merge":
        parts = [json.loads(p.read_text()) for p in args.parts]
        rows = [r for part in parts for r in part["rows"]]
        ticks = {k: v for part in parts for k, v in part["ticks"].items()}
        report = {**summarise(rows, parts[0]["trigger_s"], parts[0]["min_px"]), "ticks": ticks}
        args.out.write_text(json.dumps(report, default=float))
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("rows", "ticks")}, indent=1, default=float
            )
        )
        return

    if args.command == "expert":
        report = gate_b(
            args.collect,
            args.packs,
            args.scenarios,
            args.drivable,
            args.trigger_s,
            args.min_px,
            args.limit,
            args.scene,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, default=float))
        summary = {k: v for k, v in report.items() if k not in ("rows", "ticks")}
        print(json.dumps(summary, indent=1, default=float))
        return

    if args.command == "pack":
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes), verbose=False)
        for scene in args.scene:
            print(f"  packed {pack_scene(scene, nusc, args.out)}", flush=True)
        return
    if args.command == "routes":
        for path in sorted(args.packs.glob("scene-*.json")):
            print(f"  {path.name}: a route from {add_routes(path, args.nuscenes)} keyframes", flush=True)
        return

    report = replay(args.collect, args.dumps, args.packs, args.scenarios, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(
        f"Gate A: {report['passed']}/{report['episodes']} episodes reproduced; "
        f"worst position error {report['worst_position_error_m']} m"
    )
    sys.exit(0 if report["passed"] == report["episodes"] else 1)


if __name__ == "__main__":
    main()
