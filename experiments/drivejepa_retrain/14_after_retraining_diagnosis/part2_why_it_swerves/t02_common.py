#!/usr/bin/env python
"""T02 shared helpers: read a rollout.asl once into a small cache, and drive the AlpaSim nonlinear MPC + vehicle model standalone.

Interpreter: {runs}/alpasim_bulk/nurec-venv/bin/python (do_mpc 5.1.2 / casadi 3.8.0). Controller sources are imported
from the read-only source tree alpasim/src/controller; nothing is installed or modified. CPU only.

Controller input format (verified in the 0.1.6 rollout logs, see README): every 50 ms tick the runtime sends one
RunControllerAndVehicleModelRequest covering [now, now + 50 ms] with planned_trajectory_in_rig = poses 0.5 s apart, first pose = rig origin
at t = now. On a replan tick (every 10th) that is 9 poses (origin + the model's 8 waypoints); on the 9 ticks in between it is 8 poses
(the same plan re-sampled at now + 0.5 k s and re-expressed in the current rig frame by adapters/drivejepa016/divider.py). The MPC itself
solves every 100 ms on the plan most recently received.
"""
from __future__ import annotations
import asyncio, csv, glob, math, os, pathlib, pickle, sys, tempfile, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parents[2]
sys.path[:0] = [str(BASE / "alpasim/src/controller"), str(BASE / "alpasim/src/utils/src"), str(BASE / "alpasim/src/grpc"), str(BASE / "alpasim/src/plugins")]
from alpasim_utils.logs import async_read_pb_log  # noqa: E402
from alpasim_grpc.v0 import common_pb2, controller_pb2  # noqa: E402
from alpasim_controller.mpc_controller import MPCGains  # noqa: E402
from alpasim_controller.mpc_impl.nonlinear_mpc import NonlinearMPC  # noqa: E402
from alpasim_controller.system import System  # noqa: E402

HDD = pathlib.Path("{runs}/alpasim_bulk/baseline_014b/runs")
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
TMP = pathlib.Path(tempfile.mkdtemp(prefix="t02_"))
TICK = 50_000; STEP = 500_000
MENUS = {"0.1.6": lambda s: BASE / "analysis_3.0/Retrain-data/07_closed_loop_test/menus" / f"d016_{s}", "0.1.4b": lambda s: BASE / "analysis_3.0/Re-ranker/baseline_014b/menus" / f"b014b_{s}"}


def run_dir(ver, s):
    return BASE / "runs" / "drivejepa" / f"drivejepa_016_{s}" if ver == "0.1.6" else HDD / f"b014b_{s}_on"


def yaw_q(q): return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def se2(pose):
    """grpc Pose -> (x, y, yaw) (planar part)."""
    return np.array([pose.vec.x, pose.vec.y, yaw_q(pose.quat)])


def to_rig(world_xyyaw, ego):
    """World (x, y, yaw) rows -> rig frame of ego = (x, y, yaw)."""
    w = np.atleast_2d(world_xyyaw); c, s = math.cos(ego[2]), math.sin(ego[2]); d = w[:, :2] - ego[:2]
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.arctan2(np.sin(w[:, 2] - ego[2]), np.cos(w[:, 2] - ego[2]))]


def to_world(rig_xyyaw, ego):
    r = np.atleast_2d(rig_xyyaw); c, s = math.cos(ego[2]), math.sin(ego[2])
    return np.c_[ego[0] + c * r[:, 0] - s * r[:, 1], ego[1] + s * r[:, 0] + c * r[:, 1], r[:, 2] + ego[2]]


async def _read(asl, frame_at_rel_s):
    out = dict(reqs={}, gt=[], frames={}); rs = None
    async for e in async_read_pb_log(asl):
        k = e.WhichOneof("log_entry")
        if k == "rollout_metadata":
            m = e.rollout_metadata; rs = int(m.session_metadata.render_start_timestamp_us); out["render_start_us"] = rs
            out["force_gt_us"] = int(m.force_gt_duration); out["egodriver"] = m.version_ids.egodriver_version.version_id
            out["gt"] = np.array([[int(p.timestamp_us), *se2(p.pose)] for p in m.ego_rig_recorded_ground_truth_trajectory.poses], dtype=np.float64)
        elif k == "controller_request":
            out["reqs"][int(e.controller_request.state.timestamp_us)] = e.controller_request.SerializeToString()
        elif k == "driver_camera_image" and rs is not None:
            ci = e.driver_camera_image.camera_image
            for rel in frame_at_rel_s:
                if abs((int(ci.frame_end_us) - rs) / 1e6 - rel) < 0.026 and "front_wide" in ci.logical_id: out["frames"][rel] = (ci.logical_id, int(ci.frame_start_us), int(ci.frame_end_us), bytes(ci.image_bytes))
    return out


def load(ver, s, frame_at_rel_s=()):
    """Cached read of one run: serialized controller requests keyed by tick, GT trajectory (t_us, x, y, yaw in the local frame), controller CSV."""
    f = CACHE / f"{ver}_{s}.pkl"
    if f.is_file():
        d = pickle.load(open(f, "rb"))
        if all(r in d["frames"] for r in frame_at_rel_s): return _finish(d)
    run = run_dir(ver, s); (asl,) = glob.glob(str(run / "rollouts" / f"clipgt-{s}*" / "*" / "rollout.asl"))
    d = asyncio.run(_read(asl, frame_at_rel_s)); d["run"] = str(run); d["ver"] = ver; d["scene"] = s
    cc = open(run / "controller-config.yaml").read()
    assert all(x in cc for x in ("mpc_implementation: nonlinear", "n_horizon: 20", "dt_mpc: 0.1", "idx_start_penalty: 10", "long_position_weight: 2.0", "lat_position_weight: 1.0", "heading_weight: 1.0",
                                 "acceleration_weight: 0.1", "rel_front_steering_angle_weight: 5.0", "rel_acceleration_weight: 1.0")), cc
    (cs,) = glob.glob(str(run / "controller" / "alpasim_controller_*.csv")); d["csv"] = {k: np.array(v, dtype=np.float64) for k, v in _cols(cs).items()}
    pickle.dump(d, open(f, "wb")); return _finish(d)


def _cols(path):
    rows = list(csv.DictReader(open(path))); return {k: [float(r[k]) for r in rows] for k in rows[0]}


def _finish(d):
    d["ho"] = d["render_start_us"] + d["force_gt_us"]; d["ticks"] = np.array(sorted(d["reqs"])); return d


def req_at(d, ts):
    r = controller_pb2.RunControllerAndVehicleModelRequest(); r.ParseFromString(d["reqs"][int(ts)]); return r


def plan_arr(traj):
    """grpc Trajectory -> (t_rel_s, x, y, yaw) rows."""
    t0 = traj.poses[0].timestamp_us
    return np.array([[(p.timestamp_us - t0) / 1e6, p.pose.vec.x, p.pose.vec.y, yaw_q(p.pose.quat)] for p in traj.poses])


def plan_pb(t0_us, xyyaw, step_us=STEP):
    """(N, 3) rig-frame x, y, yaw rows (row 0 = the pose at t0) -> grpc Trajectory with poses step_us apart."""
    traj = common_pb2.Trajectory()
    for k, (x, y, yaw) in enumerate(np.asarray(xyyaw, dtype=float)):
        p = traj.poses.add(); p.timestamp_us = int(t0_us) + k * step_us; p.pose.vec.x = x; p.pose.vec.y = y; p.pose.vec.z = 0.0
        p.pose.quat.w = math.cos(yaw / 2); p.pose.quat.z = math.sin(yaw / 2)
    return traj


GAINS = dict(long_position_weight=2.0, lat_position_weight=1.0, heading_weight=1.0, acceleration_weight=0.1, rel_front_steering_angle_weight=5.0, rel_acceleration_weight=1.0, idx_start_penalty=10)


class Sim:
    """One System (vehicle model + nonlinear MPC), seeded from a recorded controller_request.state."""

    def __init__(self, d, seed_ts):
        self.d = d; self.ctrl = NonlinearMPC(gains=MPCGains(**GAINS), n_horizon=20, dt_mpc=0.1); self.status = []
        orig = self.ctrl.compute_control
        def wrapped(inp):
            o = orig(inp); self.status.append((int(inp.timestamp_us), str(o.status))); return o
        self.ctrl.compute_control = wrapped
        fd, self.logf = tempfile.mkstemp(prefix="mpc_", suffix=".csv", dir=TMP); os.close(fd)
        self.sys = System(self.logf, req_at(d, seed_ts).state, self.ctrl); self.ts = int(seed_ts)

    def feed_recorded(self, until_ts):
        """Teacher-forced replay of the recorded requests from the current tick up to (not including) until_ts: recorded plan, recorded
        pose override and recorded coerce flag, exactly what the runtime sent. The velocities/steering state stay the simulator's own
        wherever the recorded request has coerce_dynamic_state False (every tick after handover)."""
        while self.ts < until_ts:
            r = req_at(self.d, self.ts); self.sys.run_controller_and_vehicle_model(r); self.ts = int(r.future_time_us)
        return self

    def hold(self, plan, dur_us=1_000_000, coerce=None):
        """One request [now, now + dur] with `plan` held fixed (as STEP2 R4), 50 ms reporting. State/pose of the request = the recorded
        request at this tick (pose) -- the pose is what the simulator already has when it was fed recorded requests up to here."""
        r = req_at(self.d, self.ts); r.planned_trajectory_in_rig.CopyFrom(plan); r.future_time_us = self.ts + dur_us; r.pose_reporting_interval_us = TICK
        if coerce is not None: r.coerce_dynamic_state = coerce
        self.sys.run_controller_and_vehicle_model(r); self.ts += dur_us; return self

    def rows(self):
        self.sys._log_file_handle.flush(); c = _cols(self.logf); return {k: np.array(v) for k, v in c.items()}


def sim_at(d, ts, warm=True):
    """A simulator standing at tick ts. warm=True: replayed (teacher-forced) from the first recorded request, so the MPC warm start, the
    100 ms solve phase and the steering/acceleration actuator states are those of the run. warm=False: a fresh System seeded with the
    recorded state at ts (steering and acceleration states start from the seed constructor, MPC cold) -- the STEP2 R4 seeding."""
    if not warm: return Sim(d, ts)
    return Sim(d, d["ticks"][0]).feed_recorded(ts)


def metrics(rows, t0_us, ego0, thr=0.3):
    """Summary of a held-plan run. rows = simulator CSV rows after t0. Lateral displacement is in the rig frame at t0 (ego0)."""
    t = (rows["timestamp_us"] - t0_us) / 1e6; m = t > 1e-9; t = t[m]; fs = rows["front_steering_angle"][m]; us = rows["u_steering_angle"][m]; ul = rows["u_longitudinal_actuation"][m]
    q = np.c_[rows["x"][m], rows["y"][m], np.zeros(m.sum())]; lat = to_rig(q, ego0)[:, 1]; over = np.abs(fs) > thr
    sg = np.sign(fs[np.abs(fs) > 0.02]); v = np.hypot(rows["vx"][m], rows["vy"][m])
    return dict(peak_abs_steer_rad=float(np.abs(fs).max()), peak_abs_steer_cmd_rad=float(np.abs(us).max()), first_steer_gt_0p3_s=(float(t[np.argmax(over)]) if over.any() else np.nan),
                steer_sign_changes=int((np.diff(sg) != 0).sum()) if len(sg) > 1 else 0, peak_abs_lateral_m=float(np.abs(lat).max()), lateral_end_m=float(lat[-1]),
                u_long_min_mps2=float(ul.min()), u_long_max_mps2=float(ul.max()), u_long_first_mps2=float(ul[0]), speed_end_mps=float(v[-1])), t, fs, ul
