#!/usr/bin/env python
"""T02 Parts B and C: counterfactual plans through the standalone controller, the braking sweep, the menu tables.

Every controller run: a System replayed (teacher-forced, t02_common.sim_at warm=True) from the session's first recorded request up to the
tick, then ONE request [tick, tick + 1.0 s] with the variant plan held fixed (50 ms reporting, 10 MPC solves), as STEP2 R4 did. Gate for
this procedure: t02_gate.py (warm_tf reproduces the recorded controls exactly).

Plan format = what the controller received on a replan tick: 9 poses 0.5 s apart in the rig frame at the tick, pose 0 = (0, 0, yaw 0) at
t = tick, poses 1..8 = the model's 8 waypoints (x, y, heading).

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python "analysis_3.0/14_after_retraining_diagnosis/part2_why_it_swerves/t02_run.py"
"""
import glob, io, json, math, pickle, time
import numpy as np, pandas as pd
import t02_common as C

HW = ["04749bb9", "acd22430"]; VERS = ["0.1.6", "0.1.4b"]
SWEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]      # constant deceleration, m/s^2 (7 and 9 added to the requested list to bracket the onset)
ACC_MAX = 6.0                                        # controller's upper bound on commanded acceleration (nonlinear_mpc.py bounds, acc_cmd in [-9, +6]); used only by the supplementary "bounded" safe class
ONSET_RAD = 0.1                                      # P4's threshold: onset = smallest swept deceleration with peak |steer| > 0.1 rad
T8 = 0.5 * np.arange(9)


# ---------------------------------------------------------------- geometry
def tangent(xy, smooth=True, fallback=None):
    """Path tangent per pose by finite differences (np.gradient: central inside, one-sided at the ends), 3-point moving average when
    smooth. Poses whose neighbours are < 0.05 m apart take the previous pose's value (fallback for pose 0: 0)."""
    dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1]); th = np.arctan2(dy, dx); bad = np.hypot(dx, dy) < 0.05
    for k in range(len(th)):
        if bad[k]: th[k] = th[k - 1] if k else (0.0 if fallback is None else fallback)
    th = np.unwrap(th)
    if smooth and len(th) >= 3: th = np.r_[th[0], (th[:-2] + th[1:-1] + th[2:]) / 3, th[-1]]
    return th


class Path:
    """The plan's own curve: polyline through its 9 poses, parametrised by arc length; beyond the last pose it continues straight along the
    last segment, heading held."""
    def __init__(self, xyyaw):
        self.p = np.asarray(xyyaw, float); seg = np.hypot(*np.diff(self.p[:, :2], axis=0).T); self.seg = seg; self.s = np.r_[0, np.cumsum(np.maximum(seg, 1e-9))]
        j = np.where(seg > 0.05)[0]; a = self.p[j[-1] + 1, :2] - self.p[j[-1], :2] if len(j) else np.array([1.0, 0.0]); self.end_dir = a / np.linalg.norm(a)
    def xy(self, s):
        s = np.asarray(s, float); out = np.c_[np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])]; ex = np.maximum(s - self.s[-1], 0)
        return out + ex[:, None] * self.end_dir
    def yaw(self, s): return np.interp(s, self.s, np.unwrap(self.p[:, 2]))
    def at(self, s, heading="resampled"):
        xy = self.xy(s); return np.c_[xy, self.yaw(s) if heading == "resampled" else tangent(xy)]


STRAIGHT = Path(np.c_[np.linspace(0, 400, 9), np.zeros(9), np.zeros(9)])


def decel_s(v0, a):
    t_stop = v0 / a if a > 0 else np.inf; tt = np.minimum(T8, t_stop); return v0 * tt - 0.5 * a * tt * tt


def step_speeds(xy): return np.hypot(*np.diff(xy[:, :2], axis=0).T) / 0.5


def variants(rec, v0, human):
    """rec = recorded plan (9, 3) x, y, yaw; human = the human's future (9, 3) in the same frame. Returns {name: (9, 3)}."""
    P = Path(rec); out = {"B0_recorded": rec.copy()}
    out["B1_no_braking"] = P.at(v0 * T8)                                                       # same curve, s = v0 t, heading re-sampled along it
    b = rec.copy(); b[:, 2] = tangent(rec[:, :2]); out["B2_heading_tangent"] = b            # recorded positions and timing, heading = smoothed path tangent
    b = rec.copy(); b[:, 2] = rec[0, 2]; out["B2z_heading_first_pose"] = b                  # heading = pose 0's heading (0 rad) throughout
    out["B3_straightened"] = np.c_[P.s, np.zeros(9), rec[:, 2]]                               # x = recorded arc length, y = 0, recorded heading
    L = P.seg.copy(); short = v0 * 0.5 - L[0]; L2 = L.copy(); L2[0] = v0 * 0.5; L2[1:] = L[1:] - short / 7
    assert (L2 > 0).all(), L2
    out["B5_first_step_respaced"] = P.at(np.r_[0, np.cumsum(L2)])                             # first step at current speed, shortfall spread evenly over steps 2..8 (same end point)
    a_end = 2 * (v0 * 4.0 - P.s[-1]) / 16.0                                                    # constant deceleration that reaches the recorded plan's 4 s arc length
    out["B4e_const_decel_same_endpoint"] = P.at(decel_s(v0, a_end), heading="tangent")
    out["B6_human_future"] = human.copy()
    out["B7_cv_floor_straight"] = np.c_[v0 * T8, np.zeros(9), np.zeros(9)]
    return out


def human_plan(d, ts, ego):
    """The clip's ground-truth ego trajectory at ts + 0.5 k s, expressed in the rig frame `ego` (= the run's ego pose at ts). Pose 0 is set
    to the origin, as the adapter does for every plan it sends."""
    g = d["gt"]; tq = ts + (T8 * 1e6); w = np.c_[np.interp(tq, g[:, 0], g[:, 1]), np.interp(tq, g[:, 0], g[:, 2]), np.interp(tq, g[:, 0], np.unwrap(g[:, 3]))]
    r = C.to_rig(w, ego); r0 = r[0].copy(); r[0] = 0.0; return r, r0


def human_dense(d, ts, ego, t_lo=-3.0, t_hi=10.0):
    g = d["gt"]; tq = np.clip(ts + np.arange(t_lo, t_hi, 0.05) * 1e6, g[0, 0], g[-1, 0]); w = np.c_[np.interp(tq, g[:, 0], g[:, 1]), np.interp(tq, g[:, 0], g[:, 2]), np.zeros(len(tq))]
    return C.to_rig(w, ego)[:, :2]


def lateral_from(path_xy, pts):
    """Signed perpendicular offset (m, + = left of the path's direction of travel) of each point from the dense polyline path_xy."""
    out = []
    for p in np.atleast_2d(pts):
        i = int(np.argmin(np.hypot(*(path_xy - p[:2]).T))); a = path_xy[max(i - 1, 0)]; b = path_xy[min(i + 1, len(path_xy) - 1)]; t = b - a; n = np.linalg.norm(t)
        q = p[:2] - path_xy[i]; out.append(float((t[0] * q[1] - t[1] * q[0]) / n) if n > 1e-6 else float(np.hypot(*(p[:2] - path_xy[i]))))
    return np.array(out)


def ego_state(d, ts):
    r = C.req_at(d, ts); st = r.state.state; return C.se2(r.state.pose), math.hypot(st.linear_velocity.x, st.linear_velocity.y), r


def run_hold(d, ts, plan, ego, warm=True):
    s = C.sim_at(d, ts, warm=warm).hold(C.plan_pb(ts, plan)); m, t, fs, ul = C.metrics(s.rows(), ts, ego)
    m["n_solves_not_solved"] = sum(1 for x in s.status[-10:] if "solved" not in x[1].lower() and "success" not in x[1].lower()); return m, t, fs, ul


def props(plan, v0):
    sp = step_speeds(plan); th = tangent(plan[:, :2], smooth=False); mm = np.degrees(np.arctan2(np.sin(plan[:, 2] - th), np.cos(plan[:, 2] - th)))
    return dict(first_step_speed_mps=float(sp[0]), first_0p5s_accel_mps2=float((sp[0] - v0) / 0.5), mean_speed_4s_mps=float(sp.mean()), step_speed_spread_mps=float(sp.max() - sp.min()),
                accel_equiv_to_1s_mps2=float(2 * (Path(plan).s[2] - v0 * 1.0) / 1.0), accel_equiv_to_2s_mps2=float(2 * (Path(plan).s[4] - v0 * 2.0) / 4.0),
                lat_1s_m=float(plan[2, 1]), lat_2s_m=float(plan[4, 1]), heading_vs_tangent_max_deg=float(np.abs(mm[1:]).max()))


# ---------------------------------------------------------------- Part B
def part_b(series):
    rows, anat, jumps, relate = [], [], [], []
    for ver in VERS:
        for s in HW:
            d = C.load(ver, s); ho = d["ho"]
            for rel in ((0.0, 0.5) if ver == "0.1.6" else (0.0,)):
                ts = ho + int(rel * 1e6); ego, v0, r = ego_state(d, ts); rec = C.plan_arr(r.planned_trajectory_in_rig); assert len(rec) == 9 and np.allclose(rec[:, 0], T8), rec[:, 0]
                rec = rec[:, 1:]; hum, hum0 = human_plan(d, ts, ego); V = variants(rec, v0, hum) if ver == "0.1.6" else {"B0_recorded": rec.copy()}
                for name, plan in V.items():
                    m, t, fs, ul = run_hold(d, ts, plan, ego); rows.append(dict(version=ver, scene=s, tick_after_handover_s=rel, variant=name, ego_speed_mps=v0, const_decel_to_recorded_4s_endpoint_mps2=2 * (v0 * 4.0 - Path(rec).s[-1]) / 16.0, **m, **props(plan, v0)))
                    series[(ver, s, rel, name)] = dict(t=t, steer=fs, ulong=ul, plan=plan)
                if rel == 0.0:   # the STEP2 R4 seeding (fresh System, cold MPC) on the recorded plan, for comparison with the warm seeding used everywhere else
                    m, t, fs, ul = run_hold(d, ts, rec, ego, warm=False); rows.append(dict(version=ver, scene=s, tick_after_handover_s=rel, variant="B0_recorded_coldseed", ego_speed_mps=v0, **m, **props(rec, v0)))
                print("B", ver, s, rel, "human pose-0 offset from ego (m, m, rad):", np.round(hum0, 3), flush=True)
            # ---- anatomy of the handover plan: raw 8 waypoints (regenerated menu), the 9-pose plan the controller got, the human
            z = np.load(C.MENUS[ver](s) / "replan_060.npz", allow_pickle=True); assert abs(float(z["t_after_ho"])) < 1e-9; raw = z["proposals"][int(z["chosen"])]
            ego, v0, r = ego_state(d, ho); rec = C.plan_arr(r.planned_trajectory_in_rig)[:, 1:]; hum, _ = human_plan(d, ho, ego); hd = human_dense(d, ho, ego)
            dev = float(np.abs(rec[1:, :2] - raw[:, :2]).max()); dyaw = float(np.abs(rec[1:, 2] - raw[:, 2]).max())
            relate.append(dict(version=ver, scene=s, max_abs_pos_diff_ctrl_plan_vs_raw_waypoints_m=dev, max_abs_yaw_diff_rad=dyaw, pose0_xyyaw=str(np.round(rec[0], 6).tolist())))
            sp_raw = step_speeds(np.r_[np.zeros((1, 3)), raw]); sp = step_speeds(rec); sph = step_speeds(hum); th = tangent(rec[:, :2], smooth=False); lat_h = lateral_from(hd, rec)
            for k in range(1, 9):
                anat.append(dict(version=ver, scene=s, source="handover_replan", tick_after_handover_s=0.0, pose=k, t_s=0.5 * k, x_m=rec[k, 0], y_lateral_m=rec[k, 1], heading_deg=math.degrees(rec[k, 2]), tangent_deg=math.degrees(th[k]),
                                 heading_minus_tangent_deg=math.degrees(math.atan2(math.sin(rec[k, 2] - th[k]), math.cos(rec[k, 2] - th[k]))), lateral_from_human_path_m=lat_h[k], step_speed_raw_waypoints_mps=sp_raw[k - 1], step_speed_ctrl_plan_mps=sp[k - 1],
                                 step_speed_human_mps=sph[k - 1], implied_accel_mps2=(sp[k - 1] - (v0 if k == 1 else sp[k - 2])) / 0.5, ego_speed_mps=v0))
            # ---- the 20 Hz stream: the 8-pose plans of the 9 cached ticks after handover; worst world-frame deviation from the replan plan re-sampled at the tick's times
            w0 = C.to_world(rec, ego); worst = 0.0
            for j in range(1, 10):
                tj = ho + j * C.TICK; ej, vj, rj = ego_state(d, tj); pj = C.plan_arr(rj.planned_trajectory_in_rig); assert len(pj) == 8, len(pj)
                wj = C.to_world(pj[1:, 1:], ej); tt = j * 0.05 + pj[1:, 0]; ref = np.c_[np.interp(tt, T8, w0[:, 0]), np.interp(tt, T8, w0[:, 1])]; worst = max(worst, float(np.hypot(*(wj[:, :2] - ref).T).max()))
                spj = step_speeds(pj[:, 1:])
                for k in range(1, 8):
                    anat.append(dict(version=ver, scene=s, source="cached_tick", tick_after_handover_s=j * 0.05, pose=k, t_s=0.5 * k, x_m=pj[k, 1], y_lateral_m=pj[k, 2], heading_deg=math.degrees(pj[k, 3]), step_speed_ctrl_plan_mps=spj[k - 1],
                                     implied_accel_mps2=(spj[k - 1] - (vj if k == 1 else spj[k - 2])) / 0.5, ego_speed_mps=vj))
            relate[-1]["cached_ticks_max_world_dev_from_resampled_replan_plan_m"] = worst
            # ---- jump between consecutive plans at each replan, +0.0 .. +3.0 s
            prev = None
            for rel in np.arange(-0.5, 3.01, 0.5):
                ts = ho + int(round(rel * 1e6)); ego, v0, r = ego_state(d, ts); pl = C.plan_arr(r.planned_trajectory_in_rig)[:, 1:]; w = C.to_world(pl, ego)
                if prev is not None and len(pl) == 9 and prev["n"] == 9:
                    said = prev["w"][1]; e = C.to_rig(np.array([[ego[0], ego[1], ego[2]]]), np.array([said[0], said[1], said[2]]))[0]     # ego now, seen from where the old plan said it would be
                    nxt = float(np.hypot(*(w[1, :2] - prev["w"][2, :2]))); jumps.append(dict(version=ver, scene=s, replan_after_handover_s=rel, ego_speed_mps=v0, start_jump_m=float(np.hypot(e[0], e[1])), start_jump_along_old_plan_m=float(e[0]),
                                 start_jump_lateral_m=float(e[1]), start_heading_diff_deg=math.degrees(e[2]), jump_at_plus_0p5s_new_vs_old_m=nxt))
                prev = dict(w=w, n=len(pl)) if len(pl) == 9 else None
    return pd.DataFrame(rows), pd.DataFrame(anat), pd.DataFrame(jumps), pd.DataFrame(relate)


# ---------------------------------------------------------------- B4 braking sweep
def sweep(series):
    rows = []
    seeds = [("04749bb9", "recorded_path"), ("acd22430", "recorded_path"), ("04749bb9", "straight"), ("acd22430", "straight"), ("bd07fb91", "straight"), ("d1a25a99", "straight")]
    for s, pathkind in seeds:
        d = C.load("0.1.6", s); ts = d["ho"]; ego, v0, r = ego_state(d, ts); rec = C.plan_arr(r.planned_trajectory_in_rig)[:, 1:]; P = Path(rec) if pathkind == "recorded_path" else STRAIGHT
        for a in SWEEP:
            plan = P.at(decel_s(v0, a), heading="tangent"); m, t, fs, ul = run_hold(d, ts, plan, ego)
            rows.append(dict(seed_scene=s, seed_tick="handover", path=pathkind, ego_speed_mps=v0, const_decel_mps2=a, **m, **props(plan, v0))); series[("sweep", s, pathkind, a)] = dict(t=t, steer=fs, ulong=ul)
        print("sweep", s, pathkind, f"v0={v0:.2f}", [round(x["peak_abs_steer_rad"], 3) for x in rows[-len(SWEEP):]], flush=True)
    sw = pd.DataFrame(rows); on = []
    for (s, pk), g in sw.groupby(["seed_scene", "path"]):
        g = g.sort_values("const_decel_mps2"); over = g[g.peak_abs_steer_rad > ONSET_RAD]
        on.append(dict(seed_scene=s, path=pk, ego_speed_mps=float(g.ego_speed_mps.iloc[0]), onset_const_decel_mps2=(float(over.const_decel_mps2.iloc[0]) if len(over) else np.nan),
                       last_decel_below_onset_mps2=(float(g[g.const_decel_mps2 < over.const_decel_mps2.iloc[0]].const_decel_mps2.max()) if len(over) and over.const_decel_mps2.iloc[0] > 0 else (float(g.const_decel_mps2.max()) if not len(over) else np.nan)),
                       onset_as_first_0p5s_accel_mps2=(float(over.first_0p5s_accel_mps2.iloc[0]) if len(over) else np.nan), peak_steer_at_max_decel_rad=float(g.peak_abs_steer_rad.iloc[-1])))
    return sw, pd.DataFrame(on)


# ---------------------------------------------------------------- Part C
def safe_threshold(onsets):
    """Onset as a function of speed, from the straight-path sweeps (three seed speeds): returned as the first-0.5 s demanded acceleration of
    the first swept constant-deceleration plan that steered > 0.1 rad (same metric as the menu column; = -a/2 for constant a), linearly
    interpolated in speed and held outside the measured range. No onset in the sweep at a speed -> -inf (everything is 'safe')."""
    o = onsets[onsets.path == "straight"].groupby("seed_scene").first().sort_values("ego_speed_mps"); v = o.ego_speed_mps.values
    same = np.where(np.isnan(o.onset_as_first_0p5s_accel_mps2.values), -np.inf, o.onset_as_first_0p5s_accel_mps2.values); lit = np.where(np.isnan(o.onset_const_decel_mps2.values), -np.inf, -o.onset_const_decel_mps2.values)
    def f(vals):
        def g(x):
            i = int(np.clip(np.searchsorted(v, x), 1, len(v) - 1)); a, b = vals[i - 1], vals[i]; w = float(np.clip((x - v[i - 1]) / (v[i] - v[i - 1]), 0, 1))
            if not np.isfinite(a) or not np.isfinite(b): return a if w < 0.5 else b
            return a + w * (b - a)
        return g
    return f(same), f(lit)


def part_c(onsets, series):
    f_same, f_lit = safe_threshold(onsets); rows, summ, ctl = [], [], []
    jobs = [(v, s, list(np.arange(0.0, 3.01, 0.5))) for v in VERS for s in HW] + [(v, "bd07fb91", [5.0, 5.5, 6.0, 6.5]) for v in VERS]
    for ver, s, rels in jobs:
        d = C.load(ver, s); ho = d["ho"]; files = {round(float(np.load(f, allow_pickle=True)["t_after_ho"]), 3): f for f in glob.glob(str(C.MENUS[ver](s) / "replan_*.npz"))}
        for rel in rels:
            z = np.load(files[round(rel, 3)], allow_pickle=True); ts = int(z["tn"]); assert ts == ho + int(round(rel * 1e6)), (ts, ho, rel)
            ego, v0, r = ego_state(d, ts); hum, _ = human_plan(d, ts, ego); hd = human_dense(d, ts, ego); g = d["gt"]; tq = ts + np.arange(0, 4.0001, 0.05) * 1e6
            hxy = np.c_[np.interp(tq, g[:, 0], g[:, 1]), np.interp(tq, g[:, 0], g[:, 2])]; human_mean = float(np.hypot(*np.diff(hxy, axis=0).T).sum() / 4.0)
            Pr = z["proposals"]; S = z["S_plain"]; cal = z["pdm_score"]; Sc = z["S_c"]; ch = int(z["chosen"]); shipped = int(z["shipped"]); assert int(np.argmax(S)) == shipped, (ver, s, rel, int(np.argmax(S)), shipped)
            rank = np.empty(32, int); rank[np.argsort(-S, kind="stable")] = np.arange(1, 33); thr_same, thr_lit = f_same(v0), f_lit(v0); block = []
            for i in range(32):
                plan = np.r_[np.zeros((1, 3)), Pr[i].astype(float)]; p = props(plan, v0); sp = step_speeds(plan); lat = lateral_from(hd, plan[[4, 8]]); th = tangent(plan[:, :2], smooth=False)
                mm = np.degrees(np.arctan2(np.sin(plan[1:, 2] - th[1:]), np.cos(plan[1:, 2] - th[1:])))[sp > 0.4]
                clean = bool(abs(p["mean_speed_4s_mps"] - human_mean) <= 0.15 * human_mean and p["step_speed_spread_mps"] < 3.0 and abs(lat[0]) < 1.0)
                block.append(dict(version=ver, scene=s, replan_after_handover_s=rel, proposal=i, ego_speed_mps=v0, human_mean_speed_4s_mps=human_mean, mean_speed_4s_mps=p["mean_speed_4s_mps"], first_step_speed_over_current=p["first_step_speed_mps"] / v0 if v0 > 0.1 else np.nan,
                                  first_0p5s_accel_mps2=p["first_0p5s_accel_mps2"], accel_equiv_to_1s_mps2=p["accel_equiv_to_1s_mps2"], accel_equiv_to_2s_mps2=p["accel_equiv_to_2s_mps2"], step_speed_spread_mps=p["step_speed_spread_mps"],
                                  lat_from_human_path_2s_m=lat[0], lat_from_human_path_4s_m=lat[1], lat_rig_2s_m=plan[4, 1], lat_rig_4s_m=plan[8, 1], heading_vs_tangent_mean_abs_deg=(float(np.abs(mm).mean()) if len(mm) else np.nan),
                                  heading_vs_tangent_max_abs_deg=(float(np.abs(mm).max()) if len(mm) else np.nan), scorer_score=float(S[i]), scorer_score_calibrated=float(cal[i]), score_after_3p5_rescore=float(Sc[i]), scorer_rank=int(rank[i]),
                                  chosen=bool(i == ch), scorer_top_pick=bool(i == shipped), rescore_3p5_changed_pick=bool(z["changed"]), momentum_code=str(z["code"]), clean_human_like=clean,
                                  controller_safe=bool(p["first_0p5s_accel_mps2"] > thr_same), controller_safe_literal=bool(p["first_0p5s_accel_mps2"] > thr_lit), controller_safe_bounded=bool(thr_same < p["first_0p5s_accel_mps2"] <= ACC_MAX), safe_threshold_first_0p5s_accel_mps2=thr_same, safe_threshold_literal_mps2=thr_lit,
                                  straight_hold_speed=bool(abs(plan[4, 1]) < 0.5 and abs(plan[8, 1]) < 1.5 and p["mean_speed_4s_mps"] >= 0.85 * v0)))
            b = pd.DataFrame(block); rows.append(b); c = b[b.chosen].iloc[0]
            def best(col):
                q = b[b[col]]; return (int(q.scorer_rank.min()), int(q.sort_values("scorer_rank").proposal.iloc[0])) if len(q) else (np.nan, -1)
            (rc, ic), (rs_, is_), (rl, il), (rh, ih), (rb, ib) = best("clean_human_like"), best("controller_safe"), best("controller_safe_literal"), best("straight_hold_speed"), best("controller_safe_bounded")
            summ.append(dict(version=ver, scene=s, replan_after_handover_s=rel, ego_speed_mps=v0, human_mean_speed_4s_mps=human_mean, n_clean_human_like=int(b.clean_human_like.sum()), best_rank_clean_human_like=rc, n_controller_safe=int(b.controller_safe.sum()),
                             best_rank_controller_safe=rs_, n_controller_safe_literal=int(b.controller_safe_literal.sum()), best_rank_controller_safe_literal=rl, n_controller_safe_bounded=int(b.controller_safe_bounded.sum()), best_rank_controller_safe_bounded=rb, n_clean_and_safe=int((b.clean_human_like & b.controller_safe).sum()),
                             n_straight_hold_speed=int(b.straight_hold_speed.sum()), best_rank_straight_hold_speed=rh, safe_threshold_first_0p5s_accel_mps2=thr_same, safe_threshold_literal_mps2=thr_lit, chosen_proposal=ch, chosen_scorer_rank=int(c.scorer_rank),
                             rescore_3p5_changed_pick=bool(z["changed"]), chosen_mean_speed_4s_mps=c.mean_speed_4s_mps, chosen_first_0p5s_accel_mps2=c.first_0p5s_accel_mps2, chosen_accel_equiv_to_2s_mps2=c.accel_equiv_to_2s_mps2, chosen_step_speed_spread_mps=c.step_speed_spread_mps,
                             chosen_lat_from_human_path_2s_m=c.lat_from_human_path_2s_m, chosen_lat_rig_2s_m=c.lat_rig_2s_m, chosen_heading_vs_tangent_max_abs_deg=c.heading_vs_tangent_max_abs_deg, chosen_score=c.scorer_score, chosen_clean_human_like=bool(c.clean_human_like),
                             chosen_controller_safe=bool(c.controller_safe), max_mean_speed_4s_on_menu_mps=float(b.mean_speed_4s_mps.max()), min_step_speed_spread_on_menu_mps=float(b.step_speed_spread_mps.min())))
            series[("menu", ver, s, rel)] = dict(P=Pr, S=S, rank=rank, chosen=ch, human=hum, human_dense=hd, ego_v=v0, human_mean=human_mean, table=b)
            # ---- best clean / best safe / best straight-hold proposals through the controller (this version's own run state at this tick)
            if (s in HW and rel in (0.0, 0.5)) or (s == "bd07fb91" and rel == 6.0 and ver == "0.1.6"):
                todo = {"chosen": ch, "best_clean_human_like": ic, "best_controller_safe": is_, "best_controller_safe_literal": il, "best_straight_hold_speed": ih, "best_controller_safe_bounded": ib}
                for nm, i in todo.items():
                    if i < 0: ctl.append(dict(version=ver, scene=s, tick_after_handover_s=rel, which=nm, proposal=-1)); continue
                    plan = np.r_[np.zeros((1, 3)), Pr[i].astype(float)]; m, t, fs, ul = run_hold(d, ts, plan, ego); q = b.iloc[i]
                    ctl.append(dict(version=ver, scene=s, tick_after_handover_s=rel, which=nm, proposal=i, scorer_rank=int(q.scorer_rank), scorer_score=q.scorer_score, mean_speed_4s_mps=q.mean_speed_4s_mps, first_0p5s_accel_mps2=q.first_0p5s_accel_mps2,
                                    accel_equiv_to_2s_mps2=q.accel_equiv_to_2s_mps2, step_speed_spread_mps=q.step_speed_spread_mps, lat_from_human_path_2s_m=q.lat_from_human_path_2s_m, **m))
                    series[("menu_ctl", ver, s, rel, nm)] = dict(t=t, steer=fs, ulong=ul)
        print("C", ver, s, "done", flush=True)
    return pd.concat(rows), pd.DataFrame(summ), pd.DataFrame(ctl)


def frame():
    d = C.load("0.1.6", "bd07fb91", frame_at_rel_s=(9.0,)); lid, fs, fe, raw = d["frames"][9.0]
    from PIL import Image
    im = Image.open(io.BytesIO(raw)); im.save(C.HERE / "bd07fb91_front_camera_plus6p0s.png"); return dict(logical_id=lid, frame_end_minus_handover_s=(fe - d["ho"]) / 1e6, size=im.size)


def main():
    t0 = time.time(); series = {}
    fr = frame(); print("frame", fr, flush=True)
    cf, anat, jumps, relate = part_b(series); cf.to_csv(C.HERE / "counterfactuals.csv", index=False); anat.to_csv(C.HERE / "plan_anatomy.csv", index=False); jumps.to_csv(C.HERE / "replan_jumps.csv", index=False); relate.to_csv(C.HERE / "plan_relation.csv", index=False)
    sw, on = sweep(series); sw.to_csv(C.HERE / "braking_sweep.csv", index=False); on.to_csv(C.HERE / "braking_onset.csv", index=False)
    mp, ms, ctl = part_c(on, series); mp.to_csv(C.HERE / "menu_proposals.csv", index=False); ms.to_csv(C.HERE / "menu_summary.csv", index=False); ctl.to_csv(C.HERE / "menu_through_controller.csv", index=False)
    pickle.dump(series, open(C.HERE / "run_series.pkl", "wb")); json.dump(dict(frame=fr, secs=time.time() - t0), open(C.HERE / "run_info.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    print(cf[["version", "scene", "tick_after_handover_s", "variant", "peak_abs_steer_rad", "first_steer_gt_0p3_s", "steer_sign_changes", "peak_abs_lateral_m", "u_long_min_mps2", "u_long_max_mps2", "first_0p5s_accel_mps2", "accel_equiv_to_2s_mps2", "mean_speed_4s_mps", "n_solves_not_solved"]].round(3).to_string())
    print(on.round(3).to_string()); print(relate.to_string()); print(jumps.round(3).to_string())
    print(ms[["version", "scene", "replan_after_handover_s", "ego_speed_mps", "human_mean_speed_4s_mps", "n_clean_human_like", "best_rank_clean_human_like", "n_controller_safe", "best_rank_controller_safe", "n_controller_safe_literal", "best_rank_controller_safe_literal",
              "n_straight_hold_speed", "best_rank_straight_hold_speed", "chosen_scorer_rank", "rescore_3p5_changed_pick", "chosen_mean_speed_4s_mps", "chosen_first_0p5s_accel_mps2", "chosen_step_speed_spread_mps", "chosen_lat_from_human_path_2s_m", "max_mean_speed_4s_on_menu_mps", "min_step_speed_spread_on_menu_mps"]].round(2).to_string())
    print(ctl.round(3).to_string()); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
