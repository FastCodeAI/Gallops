#!/usr/bin/env python
"""T03 shared helpers. Reuses part2_why_it_swerves/t02_common.py (log reader + standalone System) and t02_run.py (Path, tangent, ...) by import;
nothing under alpasim/ or adapters/ is edited. Adds: a no-disk-cache loader, a fork-based "branch the warm simulator" hold, the Fix A plan
constructions, the Fix B menu filter, and a Sim whose MPC is constructed with other gains (Fix C).

Interpreter: {runs}/alpasim_bulk/nurec-venv/bin/python, CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import asyncio, glob, math, pathlib, pickle, sys, tempfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
T02 = HERE.parent / "part2_why_it_swerves"
sys.path.insert(0, str(T02))
import t02_common as C  # noqa: E402
import t02_run as R  # noqa: E402  (Path, tangent, step_speeds, human_plan, ego_state, lateral_from)
from alpasim_controller.mpc_controller import MPCGains  # noqa: E402
from alpasim_controller.mpc_impl.nonlinear_mpc import NonlinearMPC  # noqa: E402
from alpasim_controller.system import System  # noqa: E402
from alpasim_utils.geometry import pose_to_grpc  # noqa: E402

BASE = C.BASE
MENU = BASE / "analysis_3.0/Retrain-data/07_closed_loop_test/menus"
T8 = 0.5 * np.arange(9)
HIGHWAY = ["04749bb9", "acd22430"]
CAP = 4.0          # m/s^2, the deceleration / a1 limit of A2, A3, B1
V_GATE = 10.0      # m/s, speed above which the limit applies


def scenes():
    return sorted(p.name[5:] for p in MENU.glob("d016_*") if p.is_dir())


def load(s):
    """t02's C.load without writing a cache file (t02's cache dir is not ours to write). Reads t02's cache when it has the scene."""
    if (C.CACHE / f"0.1.6_{s}.pkl").is_file(): return C.load("0.1.6", s)
    run = C.run_dir("0.1.6", s); (asl,) = glob.glob(str(run / "rollouts" / f"clipgt-{s}*" / "*" / "rollout.asl"))
    d = asyncio.run(C._read(asl, ())); d["run"] = str(run); d["ver"] = "0.1.6"; d["scene"] = s
    cc = open(run / "controller-config.yaml").read()
    assert all(x in cc for x in ("mpc_implementation: nonlinear", "n_horizon: 20", "dt_mpc: 0.1", "idx_start_penalty: 10", "long_position_weight: 2.0", "lat_position_weight: 1.0", "heading_weight: 1.0",
                                 "acceleration_weight: 0.1", "rel_front_steering_angle_weight: 5.0", "rel_acceleration_weight: 1.0")), cc
    (cs,) = glob.glob(str(run / "controller" / "alpasim_controller_*.csv")); d["csv"] = {k: np.array(v, dtype=np.float64) for k, v in C._cols(cs).items()}
    return C._finish(d)


def menus(s):
    """Menu npz files of one scene keyed by controller tick (us), in time order."""
    out = {}
    for f in sorted(glob.glob(str(MENU / f"d016_{s}" / "replan_*.npz"))):
        z = np.load(f, allow_pickle=True); out[int(z["tn"])] = {k: z[k] for k in z.files}
    return dict(sorted(out.items()))


class SimG(C.Sim):
    """C.Sim with the MPC constructed from other settings (Fix C). gains/n_horizon/dt_mpc are the constructor arguments NonlinearMPC already has."""

    def __init__(self, d, seed_ts, gains=None, n_horizon=20, dt_mpc=0.1):
        g = dict(C.GAINS); g.update(gains or {})
        self.d = d; self.ctrl = NonlinearMPC(gains=MPCGains(**g), n_horizon=n_horizon, dt_mpc=dt_mpc); self.status = []
        orig = self.ctrl.compute_control
        def wrapped(inp):
            o = orig(inp); self.status.append((int(inp.timestamp_us), str(o.status))); return o
        self.ctrl.compute_control = wrapped
        fd, self.logf = tempfile.mkstemp(prefix="mpc_", suffix=".csv", dir=C.TMP); os.close(fd)
        self.sys = System(self.logf, C.req_at(d, seed_ts).state, self.ctrl); self.ts = int(seed_ts)


def branch(sim, fn):
    """Run fn(sim) in a forked copy of the process and return its (picklable) result; the parent's simulator is left exactly where it was.
    This is how one warm (teacher-forced from the session start) simulator serves every variant at every replan tick. The child gets its own
    controller log file so the parent's log stays clean."""
    sim.sys._log_file_handle.flush(); rfd, wfd = os.pipe(); pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(rfd); fd, lf = tempfile.mkstemp(prefix="br_", suffix=".csv", dir=C.TMP); os.close(fd)
            sim.sys._log_file_handle = open(lf, "w", encoding="utf-8"); sim.sys._log_header(); sim.logf = lf
            out = fn(sim); os.unlink(lf)
            with os.fdopen(wfd, "wb") as w: pickle.dump(out, w)
            code = 0
        except BaseException as e:  # noqa: BLE001
            try:
                with os.fdopen(wfd, "wb") as w: pickle.dump(RuntimeError(repr(e)), w)
            except Exception: pass
        finally:
            os._exit(code)
    os.close(wfd)
    with os.fdopen(rfd, "rb") as r: out = pickle.load(r)
    os.waitpid(pid, 0)
    if isinstance(out, Exception): raise out
    return out


def dense_path(plan, reach):
    """The plan's own curve as a dense polyline (continued straight past its last pose), long enough to cover `reach` metres."""
    P = R.Path(plan); return P.xy(np.linspace(0.0, max(P.s[-1], reach) + 5.0, 600)), P


def hold_metrics(rows, ts, ego, plan, n_bad, keep_series=False):
    """Summary of one held-plan run (rows = this branch's controller log, all after ts)."""
    t = (rows["timestamp_us"] - ts) / 1e6; fs = rows["front_steering_angle"]; ul = rows["u_longitudinal_actuation"]; v = np.hypot(rows["vx"], rows["vy"])
    q = C.to_rig(np.c_[rows["x"], rows["y"], np.zeros(len(t))], ego); over = np.abs(fs) > 0.3; trav = float(np.hypot(*np.diff(np.r_[np.zeros((1, 2)), q[:, :2]], axis=0).T).sum())
    dp, P = dense_path(plan, trav); latp = R.lateral_from(dp, q[:, :2]) if P.s[-1] > 0.5 else np.full(len(t), np.nan)
    m = dict(peak_abs_steer_rad=float(np.abs(fs).max()), peak_abs_steer_cmd_rad=float(np.abs(rows["u_steering_angle"]).max()), first_steer_gt_0p3_s=(float(t[np.argmax(over)]) if over.any() else np.nan),
             peak_abs_lateral_from_plan_path_m=(float(np.nanmax(np.abs(latp))) if P.s[-1] > 0.5 else np.nan), peak_abs_lateral_rig_m=float(np.abs(q[:, 1]).max()),
             u_long_min_mps2=float(ul.min()), u_long_mean_mps2=float(ul.mean()), speed_end_mps=float(v[-1]), dist_travelled_m=trav, n_solves_not_solved=int(n_bad))
    if keep_series: m["_series"] = dict(t=t, steer=fs, ulong=ul, speed=v, lat_plan=latp, lat_rig=q[:, 1], x=q[:, 0])
    return m


def held(sim, ts, plan, ego, dur_us=1_000_000, keep_series=False, also_2s=False):
    """Branch `sim` (standing at ts), hold `plan` (9 x 3 rig-frame poses 0.5 s apart) for dur, return the metrics. also_2s: the hold runs
    2.0 s in one request (its first 1.0 s is the same sequence of solves as a 1.0 s hold); the main metrics are still those of the first
    1.0 s and the 2.0 s window is added as *_2s columns. 2.0 s is the longest hold whose 2 s MPC horizon stays inside the 4 s plan."""
    def fn(s):
        n0 = len(s.status); s.hold(C.plan_pb(ts, plan), dur_us=2_000_000 if also_2s else dur_us); bad = sum(1 for x in s.status[n0:] if "solved" not in x[1].lower() and "success" not in x[1].lower())
        rows = s.rows()
        if not also_2s: return hold_metrics(rows, ts, ego, plan, bad, keep_series)
        k = rows["timestamp_us"] <= ts + 1_000_000; m = hold_metrics({c: v[k] for c, v in rows.items()}, ts, ego, plan, bad); m2 = hold_metrics(rows, ts, ego, plan, bad)
        m.update(peak_abs_steer_2s_rad=m2["peak_abs_steer_rad"], first_steer_gt_0p3_2s_s=m2["first_steer_gt_0p3_s"], peak_abs_lateral_from_plan_path_2s_m=m2["peak_abs_lateral_from_plan_path_m"], u_long_min_2s_mps2=m2["u_long_min_mps2"], speed_after_2s_mps=m2["speed_end_mps"]); return m
    return branch(sim, fn)


# ---------------------------------------------------------------- Fix A
def heading_tangent(xy, fallback):
    """Path tangent per pose (np.gradient, 3-point moving average as t02's tangent()); where neighbouring poses are < 0.05 m apart (the
    plan stands still there) the tangent is undefined and the path's own re-sampled heading `fallback` is used instead."""
    dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1]); th = np.arctan2(dy, dx); bad = np.hypot(dx, dy) < 0.05; th = np.where(bad, fallback, th); th = np.unwrap(th)
    return np.r_[th[0], (th[:-2] + th[1:-1] + th[2:]) / 3, th[-1]]


def const_profile(v0, L, T, cap=None):
    """Arc length at T8 for a constant acceleration from v0 that reaches L at time T. If that acceleration would reverse the car before T,
    use the deceleration that stops exactly at L and stay there. cap (m/s^2, > 0): the deceleration is limited to cap; the car then goes
    farther than L. Returns (s at T8, acceleration used, acceleration before the cap)."""
    a = 2.0 * (L - v0 * T) / (T * T)
    if v0 + a * T < 0: a = -v0 * v0 / (2.0 * max(L, 1e-3))
    a0 = a
    if cap is not None and a < -cap: a = -cap
    t_stop = v0 / -a if a < -1e-12 else np.inf; tt = np.minimum(T8, t_stop); return v0 * tt + 0.5 * a * tt * tt, float(a), float(a0)


def a1_of(plan, v0):
    """t02's ordering variable: the constant acceleration that reaches the plan's 1.0 s arc length, 2 (s(1 s) - v0)."""
    return float(2.0 * (R.Path(plan).s[2] - v0 * 1.0))


def fix_a(plan, v0):
    """plan (9, 3) -> dict of A1, A2, A3 plans plus bookkeeping."""
    P = R.Path(plan); L = float(P.s[-1]); gate = v0 >= V_GATE; out = {}
    s1, a_1, _ = const_profile(v0, L, 4.0); xy = P.xy(s1); out["A1"] = np.c_[xy, heading_tangent(xy, P.yaw(s1))]
    s2, a_2, a_raw = const_profile(v0, L, 4.0, cap=CAP if gate else None); xy = P.xy(s2); out["A2"] = np.c_[xy, heading_tangent(xy, P.yaw(s2))]
    a1 = a1_of(plan, v0); trig = bool(gate and a1 < -CAP); p3 = np.array(plan, float); a_3 = np.nan
    if trig:
        s3, a_3, _ = const_profile(v0, float(P.s[4]), 2.0); s3 = np.minimum(s3, P.s[4]); p3[1:4] = P.at(s3[1:4])    # poses at 0.5, 1.0, 1.5 s re-spaced on the path, heading re-sampled; 2.0 s .. 4.0 s untouched
    out["A3"] = p3
    info = dict(plan_arc_4s_m=L, a1_mps2=a1, A1_accel_mps2=a_1, A2_accel_mps2=a_2, A2_cap_bites=bool(gate and a_raw < -CAP), A2_end_beyond_model_end_m=float(s2[-1] - L), A3_triggered=trig, A3_accel_mps2=a_3,
                A1_stops_before_4s=bool(a_1 < 0 and v0 / -a_1 < 4.0 - 1e-9))
    return out, info


# ---------------------------------------------------------------- Fix B
def fix_b(z, v0):
    """Menu filter B1. z = one replan's menu. Returns pick index, survivors mask and per-proposal a1 / mean 4 s speed. The adapter's pick is
    argmax of `pdm_score` (the score after the momentum recalibration; verified equal to `chosen` on all 1,170 replans), so the filtered
    pick is argmax of the same score over the survivors."""
    Pr = z["proposals"].astype(float); sc = z["pdm_score"].astype(float); ch = int(z["chosen"]); n = len(Pr); a1 = np.empty(n); mv = np.empty(n)
    for i in range(n):
        P = R.Path(np.r_[np.zeros((1, 3)), Pr[i]]); a1[i] = 2.0 * (P.s[2] - v0); mv[i] = P.s[-1] / 4.0
    drop = ((a1 < -CAP) & (v0 >= V_GATE)) | (a1 > CAP) | (mv > 1.5 * v0 + 5.0); keep = ~drop
    pick = ch if not keep.any() else int(np.flatnonzero(keep)[np.argmax(sc[keep])])
    return pick, keep, a1, mv
