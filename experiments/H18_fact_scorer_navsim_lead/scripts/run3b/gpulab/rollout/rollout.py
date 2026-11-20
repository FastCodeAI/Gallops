"""gpulab/rollout: the ego rollout of run 3A's scorer (cns.before_score_proposals, rows 1..) as batched torch float64 on CUDA.

    extract_rollout(kind, ctx) -> dict      CPU, numpy, pose-independent: start state, start pose and its cos / sin (numpy's
                                            own values), motion-model wheelbase, and the live simulator's settings (checked
                                            against this module's per-kind constants, so a patched process cannot drift)
    simulate(kind, scenes, poses)           poses (B, P, 8, 3) rear axle, ego frame at t, 0.5 .. 4 s -> states (B, P, 41, 11)
                                            float64 cuda (StateIndex order; index 0 = the start state)
    ego_coords(kind, states)                (..., 41, 11) -> (..., 41, 5, 2) = PDMScorer._ego_coords (BBCoordsIndex order:
                                            FL, RL, RR, FR, centre)

The CPU steps it restates (read, never edited):
  navsim/evaluate/pdm_score.fast_transform_trajectory    ego frame -> world, np.unwrap, interp1d (linear) 0.5 s -> 0.1 s
  pdm_planner/simulation_v2/pdm_simulator.PDMSimulator   40 steps of tracker + motion model
  simulation_v2/batch_lqr.BatchLQRTracker + batch_lqr_utils   velocity / curvature least-squares fits (pinv), stopping
                                            P controller (<= 0.2 m/s), one-step longitudinal LQR, lateral LQR over the horizon
  simulation_v2/batch_kinematic_bicycle.BatchKinematicBicycleModel   first-order accel / steering lag, Euler step,
                                            steering clip at 60 deg, heading principal value
  carla_v2/fixes.no_reverse_model (CARLA)   speed clamped at 0 where it was >= 0 before the step
  utils/pdm_array_representation.state_array_to_coords_array   the scorer's ego boxes
kind 'nav': NAVSIM's simulator (Pacifica, LQR horizon 10, q_lon 10, q_lat [1, 10, 0], steering lag 0.05 s).
kind 'carla': carla_full_v1's (MKZ, LQR horizon 5, q_lon 100, q_lat [100, 10, 0], steering lag 0, no-reverse bicycle).
Elementwise steps keep the CPU's operation order (true IEEE division, no FMA contraction, sequential cumulative sums); the
least-squares fits use a batched Cholesky solve where the CPU uses scipy's SVD pinv, and the trig is CUDA's, so values agree to
rounding (verify_rollout.json: max 4.4e-9 on any state field), not bit for bit. The three loop implementations give the same bits."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEV = torch.device("cuda")
F64 = torch.float64
PI = math.pi
TWO_PI = 2 * math.pi
HALF_PI = math.pi / 2.0

# the CPU objects' values, per kind (extract_rollout asserts the live process holds exactly these)
CFG = {
    "nav": dict(q_lon=10.0, r_lon=1.0, q_lat=(1.0, 10.0, 0.0), r_lat=1.0, disc_dt=0.1, horizon=10, wb_lqr=3.089,
                jerk=0.0001, curv_pen=0.01, init_curv_pen=1e-10, stop_gain=0.5, stop_v=0.2,
                max_steer=1.0471975511965976, tau_accel=0.2, tau_steer=0.05, no_reverse=False,
                half_length=2.588, half_width=1.1485, rear_axle_to_center=1.461),
    "carla": dict(q_lon=100.0, r_lon=1.0, q_lat=(100.0, 10.0, 0.0), r_lat=1.0, disc_dt=0.1, horizon=5, wb_lqr=2.8604,
                  jerk=0.0001, curv_pen=0.01, init_curv_pen=1e-10, stop_gain=0.5, stop_v=0.2,
                  max_steer=1.0471975511965976, tau_accel=0.2, tau_steer=0.0, no_reverse=True,
                  half_length=2.44619083404541, half_width=0.9183566570281982, rear_axle_to_center=1.382306031702459),
}
N_POSES = 40           # proposal_sampling.num_poses (41 states at 0.1 s)
SOLVE = "chol_solve"   # least-squares solve: 'chol_solve' | 'chol_inv' | 'inv' (LU) | 'pinv' (SVD; very slow on CUDA)
LOOP = "triton"        # the 40-step loop: 'triton' (one kernel, tl_loop.py) | 'graph' (torch ops replayed as a CUDA graph per
                       # (kind, rows)) | 'eager' (torch ops); all three give the same bits (verify_rollout.py checks)


# ------------------------------------------------------------------------------------------------ time grids (numpy, once)
def _grids():
    """fast_transform_trajectory's source / target times and interp1d's (lo, hi, x_hi - x_lo, x_new - x_lo), computed with
    numpy exactly as the CPU does (Trajectory's default sampling 4 s / 0.5 s; proposal sampling 4 s / 0.1 s)."""
    t_src = np.arange(0.0, 4.0 + 0.5, 0.5)
    t_tgt = np.arange(0.0, 4.0 + 0.1, 0.1)
    assert len(t_src) == 9 and len(t_tgt) == N_POSES + 1
    idx = np.searchsorted(t_src, t_tgt).clip(1, len(t_src) - 1).astype(int)
    lo, hi = idx - 1, idx
    return dict(t_src=t_src, t_tgt=t_tgt, lo=lo, hi=hi, dxs=t_src[hi] - t_src[lo], dxn=t_tgt - t_src[lo])


GRID = _grids()
# simulation step: nuPlan TimePoint difference of TimeDuration.from_s(0.1) -> int(0.1 * int(1e6)) us -> time_us * 1e-6
DT_SIM = int(0.1 * int(1e6)) * 1e-6
_CONST = {}


def _consts(kind):
    """per-kind GPU constants (built once)."""
    if kind in _CONST:
        return _CONST[kind]
    c = CFG[kind]
    M = N_POSES
    dt = c["disc_dt"]
    # velocity fit regulariser: jerk * R^T R (R = [0 | banded difference (M-2) x (M-1)]), numpy as the CPU builds it
    nb = M - 2
    band = np.zeros((nb, nb + 1))
    eye = np.eye(nb)
    band[:, 1:] = eye
    band[:, :-1] = -eye
    Rm = np.block([np.zeros((len(band), 1)), band])
    jRTR = c["jerk"] * (Rm.T @ Rm)
    Q = c["curv_pen"] * np.eye(M)
    Q[0, 0] = c["init_curv_pen"]
    up = np.repeat(np.triu(np.ones((M, M), dtype=bool), k=1), 2, axis=0)  # (2M, M) zeroed entries of the velocity fit's A
    tri = np.tri(M, dtype=np.float64)
    Bl = c["horizon"] * dt
    inv_lon = -1 / (Bl * c["q_lon"] * Bl + c["r_lon"])
    out = dict(
        jRTR=torch.as_tensor(jRTR, dtype=F64, device=DEV), Qc=torch.as_tensor(Q, dtype=F64, device=DEV),
        up=torch.as_tensor(up, device=DEV), tri=torch.as_tensor(tri, dtype=F64, device=DEV),
        lo=torch.as_tensor(GRID["lo"], device=DEV), hi=torch.as_tensor(GRID["hi"], device=DEV),
        dxs=torch.as_tensor(GRID["dxs"], dtype=F64, device=DEV), dxn=torch.as_tensor(GRID["dxn"], dtype=F64, device=DEV),
        k_lon=(inv_lon * Bl) * c["q_lon"],            # lqr_input = inverse * B * q * err, left to right
        a_gain=DT_SIM / (DT_SIM + c["tau_accel"]), s_gain=DT_SIM / (DT_SIM + c["tau_steer"]),
        dt=dt, dt2=dt ** 2, wb=c["wb_lqr"],
        # divisors as 0-dim CUDA tensors: torch divides by a Python scalar as a * (1 / b) (one bit off IEEE division)
        wb_t=torch.tensor(c["wb_lqr"], dtype=F64, device=DEV), dtsim_t=torch.tensor(DT_SIM, dtype=F64, device=DEV))
    _CONST[kind] = out
    return out


# ------------------------------------------------------------------------------------------------ extract (CPU)
def extract_rollout(kind, ctx):
    """pose-independent rollout inputs from a labeller context (labeller._nav_env / _carla_env context(meta)): plain numpy.
    Must run in that kind's labeller process (it reads the process's simulator to check the settings)."""
    from navsim.agents.drive_jepa_run3a.score_module import compute_navsim_score as cns
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    _check_process(kind, cns)
    es = ctx["mc"].ego_state
    ra = es.rear_axle
    x0, y0, h0 = ra.x, ra.y, ra.heading
    return dict(kind=kind, s0=ego_state_to_state_array(es).astype(np.float64),
                origin=np.array([x0, y0, h0], np.float64),
                cs=np.array([np.cos(h0), np.sin(h0)], np.float64),       # fast_transform_trajectory's R, numpy's values
                wb_model=np.float64(es.car_footprint.vehicle_parameters.wheel_base))


_CHECKED = set()


def _check_process(kind, cns):
    if kind in _CHECKED:
        return
    from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
    c = CFG[kind]
    sim, sc = cns.simulator, cns.scorer
    tr, mm = sim._tracker, sim._motion_model
    ps = sim.proposal_sampling
    assert type(tr).__name__ == "BatchLQRTracker" and "simulation_v2" in type(tr).__module__, type(tr)
    assert ps.num_poses == N_POSES and ps.interval_length == c["disc_dt"], ps
    td = TimeDuration.from_s(ps.interval_length)
    assert TimePoint(td.time_us).time_s == DT_SIM
    got = dict(q_lon=tr._q_longitudinal, r_lon=tr._r_longitudinal, q_lat=tuple(np.diag(tr._q_lateral)),
               r_lat=float(tr._r_lateral[0, 0]), horizon=tr._tracking_horizon, wb_lqr=tr._wheel_base, jerk=tr._jerk_penalty,
               curv_pen=tr._curvature_rate_penalty, stop_gain=tr._stopping_proportional_gain, stop_v=tr._stopping_velocity,
               max_steer=mm._max_steering_angle, tau_accel=mm._accel_time_constant,
               tau_steer=mm._steering_angle_time_constant, no_reverse=type(mm).__name__ == "NoReverseBicycle",
               half_length=sc._vehicle_parameters.half_length, half_width=sc._vehicle_parameters.half_width,
               rear_axle_to_center=sc._vehicle_parameters.rear_axle_to_center)
    assert np.count_nonzero(tr._q_lateral - np.diag(np.diag(tr._q_lateral))) == 0 and tr._r_lateral.shape == (1, 1)
    for k, v in got.items():
        assert v == c[k], (kind, k, v, c[k])
    if got["no_reverse"]:
        import inspect
        src = inspect.getsource(type(mm).propagate_state)
        assert "out[m, SI.VELOCITY_X] = 0.0" in src and "out[m, SI.ANGULAR_VELOCITY] = 0.0" in src
    else:
        assert type(mm).__name__ == "BatchKinematicBicycleModel"
    import navsim.planning.simulation.planner.pdm_planner.simulation_v2.batch_lqr_utils as U
    assert U.INITIAL_CURVATURE_PENALTY == c["init_curv_pen"]
    _CHECKED.add(kind)


# ------------------------------------------------------------------------------------------------ helpers (GPU)
def _np_mod(a, b):
    """numpy's float remainder (npy_divmod): fmod, then the divisor's sign."""
    m = torch.fmod(a, b)
    fix = (m != 0) & ((m < 0) != (b < 0))
    m = torch.where(fix, m + b, m)
    return torch.where(m == 0, torch.zeros_like(m).copysign(torch.full_like(m, b)), m)


def _wrap(h):
    """principal_value / fast_transform's wrap: (h - (-pi)) % 2 pi + (-pi)."""
    return _np_mod(h - (-PI), TWO_PI) + (-PI)


def _normalize(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _seq_cumsum(x):
    """np.cumsum along the last axis (sequential adds)."""
    if LOOP == "triton" and x.dim() == 2:
        import tl_loop
        return tl_loop.seq_cumsum(x)
    out = torch.empty_like(x)
    acc = x[..., 0]
    out[..., 0] = acc
    for k in range(1, x.shape[-1]):
        acc = acc + x[..., k]
        out[..., k] = acc
    return out


def _unwrap(p):
    """np.unwrap along the last axis (period 2 pi)."""
    dd = p[..., 1:] - p[..., :-1]
    ddmod = _np_mod(dd - (-PI), TWO_PI) + (-PI)
    ddmod = torch.where((ddmod == -PI) & (dd > 0), torch.full_like(ddmod, PI), ddmod)
    ph = ddmod - dd
    ph = torch.where(dd.abs() < PI, torch.zeros_like(ph), ph)
    out = p.clone()
    out[..., 1:] = p[..., 1:] + _seq_cumsum(ph)
    return out


def _solve_ls(M, At, y):
    """x = (pinv(M) @ A^T) @ y for the SPD normal matrix M (the CPU: scipy.linalg.pinv per matrix)."""
    if SOLVE == "chol_solve":
        L, _ = torch.linalg.cholesky_ex(M)
        X = torch.cholesky_solve(At, L)
    elif SOLVE == "chol_solve1":  # one right-hand side: solve M x = (A^T y)
        L, _ = torch.linalg.cholesky_ex(M)
        return torch.cholesky_solve(At @ y[..., None], L)[..., 0]
    elif SOLVE == "chol_inv":
        L, _ = torch.linalg.cholesky_ex(M)
        X = torch.cholesky_inverse(L) @ At
    elif SOLVE == "inv":
        X = torch.linalg.inv_ex(M)[0] @ At
    else:
        X = torch.linalg.pinv(M) @ At
    return (X @ y[..., None])[..., 0]


def _scene_rows(scenes, P):
    """the per-scene inputs in one host -> device copy, repeated per path: s0 (R, 11), origin (R, 3), cs (R, 2), wb (R,)."""
    h = np.empty((len(scenes), 17), np.float64)
    for b, s in enumerate(scenes):
        h[b, :11], h[b, 11:14], h[b, 14:16], h[b, 16] = s["s0"], s["origin"], s["cs"], s["wb_model"]
    d = torch.from_numpy(h).pin_memory().to(DEV, non_blocking=True).repeat_interleave(P, 0)
    return d[:, :11], d[:, 11:14], d[:, 14:16], d[:, 16].contiguous()


# ------------------------------------------------------------------------------------------------ the steps
def _transform(p, org, cs, K):
    """fast_transform_trajectory: (R, 8, 3) ego-frame poses -> (R, 41, 3) world reference states."""
    rx, ry, rh = p[..., 0], p[..., 1], p[..., 2]
    c, s = cs[:, 0:1], cs[:, 1:2]
    x0, y0, h0 = org[:, 0:1], org[:, 1:2], org[:, 2:3]
    ax = (rx * c + ry * (-s)) + x0          # rel_xy @ R.T + (x0, y0), R = [[c, -s], [s, c]]
    ay = (rx * s + ry * c) + y0
    ah = rh + h0
    Y = torch.stack([torch.cat([x0, ax], 1), torch.cat([y0, ay], 1), _unwrap(torch.cat([h0, ah], 1))], -1)  # (R, 9, 3)
    ylo, yhi = Y[:, K["lo"]], Y[:, K["hi"]]
    out = (yhi - ylo) / K["dxs"][None, :, None] * K["dxn"][None, :, None] + ylo
    out[..., 2] = _wrap(out[..., 2])
    return out


def _profiles(ref, K):
    """batch_lqr_utils.get_velocity_curvature_profiles_with_derivatives_from_poses: velocity and curvature (R, 40)."""
    R, M = ref.shape[0], N_POSES
    dt, dt2 = K["dt"], K["dt2"]
    dxy = ref[:, 1:, :2] - ref[:, :-1, :2]
    dh = _normalize(ref[:, 1:, 2] - ref[:, :-1, 2])
    hd = ref[:, :-1, 2]
    y = dxy.reshape(R, 2 * M)
    col = torch.stack([torch.cos(hd), torch.sin(hd)], -1).reshape(R, 2 * M)
    A = (col * dt2)[..., None].expand(R, 2 * M, M).clone()
    A[..., 0] = col * dt
    A = torch.where(K["up"], 0.0, A)
    At = A.transpose(1, 2)
    x = _solve_ls(At @ A + K["jRTR"], At, y)
    v0, acc = x[:, 0], x[:, 1:]
    vel = torch.cat([v0[:, None], v0[:, None] + _seq_cumsum(acc * dt)], 1)            # (R, 40)
    A2 = K["tri"].expand(R, M, M).clone()
    A2[:, :, 0] = vel * dt
    A2[:, 1:, 1:] *= (vel * dt2)[:, 1:, None]
    A2t = A2.transpose(1, 2)
    x2 = _solve_ls(A2t @ A2 + K["Qc"], A2t, dh)
    k0, rate = x2[:, 0], x2[:, 1:]
    curv = torch.cat([k0[:, None], k0[:, None] + _seq_cumsum(rate * dt)], 1)
    return vel, curv


def _track(i, cur, ref, vel, curv, c, K):
    """BatchLQRTracker.track_trajectory at iteration i: (accel, steering rate) commands (R,), and the stop mask."""
    N = c["horizon"]
    dt = K["dt"]
    r = ref[:, i]
    xe = cur[:, 0] - r[:, 0]
    ye = cur[:, 1] - r[:, 1]
    hr = r[:, 2]
    lat = (-xe) * torch.sin(hr) + ye * torch.cos(hr)
    he = _normalize(cur[:, 2] - hr)
    v = cur[:, 3]
    ridx = min(i + N, N_POSES - 1)
    vref = vel[:, ridx]
    L = ridx - i
    kp = curv[:, i:ridx]
    if L < N:
        kp = torch.cat([kp, curv[:, ridx:ridx + 1].expand(-1, N - L)], 1)
    stop = (vref <= c["stop_v"]) & (v <= c["stop_v"])
    a_stop = (-c["stop_gain"]) * (v - vref)
    err = 1.0 * v + 0.0 - vref
    a_lqr = K["k_lon"] * err
    accel = torch.where(stop, a_stop, a_lqr)
    # lateral LQR on the velocity profile v, v + a dt, ... (N values)
    vp = torch.cat([v[:, None] + 0.0, v[:, None] + _seq_cumsum((accel * dt)[:, None].expand(-1, N - 1))], 1)
    a_s = vp * dt
    b_s = vp * dt / K["wb_t"]
    aff = (-vp) * kp * dt
    # A, B, g of the horizon: S_k = [[1, a_k, 0], [0, 1, b_k], [0, 0, 1]], in_matrix = (0, 0, dt), affine = (0, aff_k, 0);
    # A stays unit upper triangular (A01, A02, A12), g2 stays 0
    z = torch.zeros_like(v)
    A01, A02, A12 = z, z, z
    B0, B1, B2 = z, z, z
    g0, g1 = z, z
    for k in range(N):
        ak, bk = a_s[:, k], b_s[:, k]
        A02 = A02 + ak * A12
        A01 = A01 + ak
        A12 = A12 + bk
        B0 = B0 + ak * B1
        B1 = B1 + bk * B2
        B2 = B2 + dt
        g0 = g0 + ak * g1
        g1 = g1 + aff[:, k]
    l2 = cur[:, 7]
    se0 = ((lat + A01 * he) + A02 * l2) + g0
    se12 = _normalize(torch.stack([(he + A12 * l2) + g1, l2 + 0.0], -1))
    se1, se2 = se12[:, 0], se12[:, 1]
    q0, q1, q2 = c["q_lat"]
    t0, t1, t2 = B0 * q0, B1 * q1, B2 * q2
    den = (t0 * B0 + t1 * B1) + t2 * B2
    tail = (t0 * se0 + t1 * se1) + t2 * se2
    sr = (-1 / (den + c["r_lat"])) * tail
    sr = torch.where(stop, torch.zeros_like(sr), sr)
    return accel, sr, stop


def _propagate(s, a_cmd, sr_cmd, wb, c, K):
    """BatchKinematicBicycleModel.propagate_state (+ the CARLA no-reverse clamp); returns (new state, clip mask, norev mask)."""
    dt = DT_SIM
    accel, steer = s[:, 5], s[:, 7]
    ideal_steer = dt * sr_cmd + steer
    upd_a = K["a_gain"] * (a_cmd - accel) + accel
    upd_s = K["s_gain"] * (ideal_steer - steer) + steer
    upd_rate = (upd_s - steer) / K["dtsim_t"]
    v, h = s[:, 3], s[:, 2]
    out = s.clone()
    out[:, 0] = s[:, 0] + (v * torch.cos(h)) * dt
    out[:, 1] = s[:, 1] + (v * torch.sin(h)) * dt
    out[:, 2] = _wrap(h + (v * torch.tan(steer) / wb) * dt)
    out[:, 3] = v + upd_a * dt
    out[:, 4] = 0.0
    raw = steer + upd_rate * dt
    st = raw.clamp(-c["max_steer"], c["max_steer"])
    out[:, 7] = st
    out[:, 9] = out[:, 3] * torch.tan(st) / wb
    out[:, 5] = upd_a
    out[:, 6] = 0.0
    out[:, 10] = (out[:, 9] - s[:, 9]) / K["dtsim_t"]
    out[:, 8] = upd_rate
    clip = raw.abs() > c["max_steer"]
    if c["no_reverse"]:
        m = (v >= 0) & (out[:, 3] < 0)
        out[:, 3] = torch.where(m, torch.zeros_like(v), out[:, 3])
        out[:, 5] = torch.where(m, (-v) / K["dtsim_t"], out[:, 5])
        out[:, 9] = torch.where(m, torch.zeros_like(v), out[:, 9])
    else:
        m = torch.zeros_like(clip)
    return out, clip, m


def _loop(kind, ref, vel, curv, s0, wb):
    """the 40 tracker + motion-model steps: states (R, 41, 11) and the branch masks (R, 40) x 3."""
    c, K = CFG[kind], _consts(kind)
    Rn = ref.shape[0]
    S = torch.empty(Rn, N_POSES + 1, 11, dtype=F64, device=DEV)
    S[:, 0] = s0
    stop_m, clip_m, nr_m = (torch.empty(Rn, N_POSES, dtype=torch.bool, device=DEV) for _ in range(3))
    for i in range(N_POSES):
        a, sr, stop = _track(i, S[:, i], ref, vel, curv, c, K)
        S[:, i + 1], clip, nr = _propagate(S[:, i], a, sr, wb, c, K)
        stop_m[:, i], clip_m[:, i], nr_m[:, i] = stop, clip, nr
    return S, stop_m, clip_m, nr_m


_GRAPHS = {}


def _run(kind, p, s0, org, cs, wb):
    """transform + least-squares profiles (torch, cuSOLVER Cholesky), then the 40-step loop: one Triton kernel (LOOP='triton'),
    or torch ops eagerly (~6,000 small kernels) or replayed from a CUDA graph captured once per (kind, rows)."""
    K = _consts(kind)
    ref = _transform(p, org, cs, K)
    vel, curv = _profiles(ref, K)
    ins = (ref, vel, curv, s0, wb)
    if LOOP == "triton":
        import tl_loop
        return (*tl_loop.loop(CFG[kind], K, *ins, PI, TWO_PI, DT_SIM), ref, vel, curv)
    if LOOP == "eager":
        return (*_loop(kind, *ins), ref, vel, curv)
    key = (kind, p.shape[0])
    if key not in _GRAPHS:
        static = [t.clone() for t in ins]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                _loop(kind, *static)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = _loop(kind, *static)
        _GRAPHS[key] = (g, static, outs)
    g, static, outs = _GRAPHS[key]
    for d, x in zip(static, ins):
        d.copy_(x)
    g.replay()
    return (*(o.clone() for o in outs), ref, vel, curv)


def release_graphs():
    """free the captured graphs (and their memory pools)."""
    _GRAPHS.clear()
    torch.cuda.empty_cache()


@torch.no_grad()
def simulate(kind, scenes, poses, return_branches=False, return_profiles=False):
    """poses (B, P, 8, 3) (numpy or torch) -> states (B, P, 41, 11) float64 cuda = cns.before_score_proposals rows 1..
    return_branches: also {stop, clip, norev} (B, P, 40) bool per step (stopping controller used, steering clipped,
    no-reverse clamp); return_profiles: also the reference states and the fitted velocity / curvature (diagnostics)."""
    assert all(s["kind"] == kind for s in scenes)
    poses = torch.as_tensor(poses, device=DEV).to(F64)
    B, P = poses.shape[:2]
    assert B == len(scenes) and poses.shape[2:] == (8, 3), poses.shape
    Rn = B * P
    s0, org, cs, wb = _scene_rows(scenes, P)
    S, stop, clip, nr, ref, vel, curv = _run(kind, poses.reshape(Rn, 8, 3).contiguous(), s0, org, cs, wb)
    out = S.reshape(B, P, N_POSES + 1, 11)
    extra = []
    if return_branches:
        extra.append(dict(stop=stop.reshape(B, P, N_POSES), clip=clip.reshape(B, P, N_POSES), norev=nr.reshape(B, P, N_POSES)))
    if return_profiles:
        extra.append(dict(ref=ref.reshape(B, P, N_POSES + 1, 3), vel=vel.reshape(B, P, N_POSES),
                          curv=curv.reshape(B, P, N_POSES)))
    return (out, *extra) if extra else out


@torch.no_grad()
def ego_coords(kind, states):
    """state_array_to_coords_array with the kind's scorer vehicle: (..., 41, 11) -> (..., 41, 5, 2), BBCoordsIndex order
    (FL, RL, RR, FR, centre). Per corner: centre + (lat cos(h + pi/2) + lon cos h, lat sin(h + pi/2) + lon sin h)."""
    c = CFG[kind]
    states = torch.as_tensor(states, device=DEV).to(F64)
    hl, hw, r2c = c["half_length"], c["half_width"], c["rear_axle_to_center"]
    key = ("coords", kind)
    if key not in _CONST:
        _CONST[key] = (torch.tensor([hl, -hl, -hl, hl, 0.0], dtype=F64, device=DEV),
                       torch.tensor([hw, hw, -hw, -hw, 0.0], dtype=F64, device=DEV))
    lon, lat = _CONST[key]
    h = states[..., 2:3]
    cs = torch.cat([torch.cos(h), torch.sin(h)], -1)                       # (..., 41, 2)
    cs90 = torch.cat([torch.cos(h + HALF_PI), torch.sin(h + HALF_PI)], -1)
    cen = states[..., :2] + r2c * cs
    off = lat[:, None] * cs90[..., None, :] + lon[:, None] * cs[..., None, :]  # (..., 41, 5, 2)
    out = cen[..., None, :] + off
    out[..., 4, :] = cen                                                    # centre row: exactly the centre (CPU: no offset)
    return out
