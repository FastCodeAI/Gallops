#!/usr/bin/env python
"""T03 Fix C: the MPC constructed with other settings (only arguments NonlinearMPC / MPCGains already take), nothing edited on disk.

Per controller variant:
  sweep     constant-deceleration plans on a straight line (t02's B4 construction), 0..12 m/s^2, held 1.0 s from the handover tick of
            04749bb9 (23.6 m/s) and bd07fb91 (15.7 m/s);
  handover  the ORIGINAL chosen plan held 1.0 s at the handover tick of 04749bb9 and acd22430 (warm), and at bd07fb91 +6.0 s (cold seed
            from the recorded state: the warm state there belongs to the default controller's own history);
  tracking  all 30 scenes (the 5 clean scenes of the task are scene_set == "clean5"), 5 s after handover, closed loop on the simulator's own pose, recorded plans kept fixed in the world frame
            (t02_gate's cold_free loop, started from the warm simulator), compared with the recorded trajectory.
The simulator is first replayed teacher-forced from the session's first request up to the handover tick WITH THE VARIANT CONTROLLER (every
request up to and including handover has coerce_dynamic_state True, so the vehicle state at handover is the recorded one for every variant;
only the MPC's warm start differs).

  python t03_fixc.py            # all variants in VARIANTS
"""
import multiprocessing as mp, sys, time
import numpy as np, pandas as pd
import t03_common as T

C, R = T.C, T.R
SWEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
CLEAN = ["02e075b9", "16b83346", "6725df2d", "01d503d4", "048b974e"]
VARIANTS = {
    "default": {},
    "C1_track_from_step0": dict(gains=dict(idx_start_penalty=0)),
    "C2_long_weight_0p5": dict(gains=dict(long_position_weight=0.5)),
    "C3_heading_weight_10": dict(gains=dict(heading_weight=10.0)),
    "C4_steer_rate_weight_50": dict(gains=dict(rel_front_steering_angle_weight=50.0)),
    # C5 candidates (the "at most 6 further tries"), chosen after reading C1-C4
    "T1_long0p5_heading10": dict(gains=dict(long_position_weight=0.5, heading_weight=10.0)),
    "T2_long0p5_steer50": dict(gains=dict(long_position_weight=0.5, rel_front_steering_angle_weight=50.0)),
    "T3_heading10_steer50": dict(gains=dict(heading_weight=10.0, rel_front_steering_angle_weight=50.0)),
    "T4_long0p5_heading10_steer50": dict(gains=dict(long_position_weight=0.5, heading_weight=10.0, rel_front_steering_angle_weight=50.0)),
    "T5_long0p2_heading10": dict(gains=dict(long_position_weight=0.2, heading_weight=10.0)),
    "T6_lat_weight_10": dict(gains=dict(lat_position_weight=10.0)),
}


def warm(d, kw, until):
    return T.SimG(d, d["ticks"][0], **kw).feed_recorded(until)


def job_sweep(a):
    name, kw, s = a; d = T.load(s); ts = d["ho"]; sim = warm(d, kw, ts); ego, v0, r = R.ego_state(d, ts); rows = []
    for dec in SWEEP:
        plan = R.STRAIGHT.at(R.decel_s(v0, dec), heading="tangent"); m = T.held(sim, ts, plan, ego)
        rows.append(dict(kind="sweep", controller=name, seed_scene=s, car_speed_mps=v0, const_decel_mps2=dec, **m))
    return rows


def job_handover(a):
    name, kw, s, rel, mode = a; d = T.load(s); ts = d["ho"] + int(rel * 1e6); sim = warm(d, kw, ts) if mode == "warm" else T.SimG(d, ts, **kw)
    ego, v0, r = R.ego_state(d, ts); rec = C.plan_arr(r.planned_trajectory_in_rig)[:, 1:]; m = T.held(sim, ts, rec, ego)
    return [dict(kind="original_plan", controller=name, seed_scene=s, tick_after_handover_s=rel, seeding=mode, car_speed_mps=v0, **m)]


def job_track(a):
    name, kw, s = a; d = T.load(s); t0 = d["ho"]; DUR = 5_000_000; sim = warm(d, kw, t0); ts = t0; n0 = len(sim.status)
    while ts < t0 + DUR and ts in d["reqs"]:
        r = C.req_at(d, ts); rec_ego = C.se2(r.state.pose); pl = C.plan_arr(r.planned_trajectory_in_rig)
        lp = sim.sys._trajectory.last_pose; q = lp.quat; ego = np.array([lp.vec3[0], lp.vec3[1], np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))])
        w = C.to_world(pl[:, 1:], rec_ego); r.planned_trajectory_in_rig.CopyFrom(C.plan_pb(ts, C.to_rig(w, ego)))
        r.state.pose.CopyFrom(T.pose_to_grpc(lp)); r.coerce_dynamic_state = False if ts > t0 else r.coerce_dynamic_state
        sim.sys.run_controller_and_vehicle_model(r); ts += C.TICK
    rows = sim.rows(); rc = d["csv"]; m = (rows["timestamp_us"] > t0) & (rows["timestamp_us"] <= ts); mr = (rc["timestamp_us"] > t0) & (rc["timestamp_us"] <= ts)
    assert np.array_equal(rows["timestamp_us"][m], rc["timestamp_us"][mr]), (name, s)
    ryaw = np.arctan2(2 * (rc["qw"][mr] * rc["qz"][mr] + rc["qx"][mr] * rc["qy"][mr]), 1 - 2 * (rc["qy"][mr] ** 2 + rc["qz"][mr] ** 2)); dx = rows["x"][m] - rc["x"][mr]; dy = rows["y"][m] - rc["y"][mr]
    lon = np.cos(ryaw) * dx + np.sin(ryaw) * dy; lat = -np.sin(ryaw) * dx + np.cos(ryaw) * dy; bad = sum(1 for x in sim.status[n0:] if "solved" not in x[1].lower() and "success" not in x[1].lower())
    return [dict(kind="tracking", controller=name, scene=s, scene_set="clean5" if s in CLEAN else "other", handover_speed_mps=R.ego_state(d, t0)[1], duration_s=(ts - t0) / 1e6, max_abs_lateral_dev_from_recorded_m=float(np.abs(lat).max()), max_abs_longitudinal_dev_from_recorded_m=float(np.abs(lon).max()),
                 longitudinal_dev_end_m=float(lon[-1]), peak_abs_steer_rad=float(np.abs(rows["front_steering_angle"][m]).max()), recorded_peak_abs_steer_rad=float(np.abs(rc["front_steering_angle"][mr]).max()),
                 max_abs_steer_diff_from_recorded_rad=float(np.abs(rows["front_steering_angle"][m] - rc["front_steering_angle"][mr]).max()), min_u_long_mps2=float(rows["u_longitudinal_actuation"][m].min()),
                 recorded_min_u_long_mps2=float(rc["u_longitudinal_actuation"][mr].min()), speed_end_mps=float(np.hypot(rows["vx"][m][-1], rows["vy"][m][-1])), recorded_speed_end_mps=float(np.hypot(rc["vx"][mr][-1], rc["vy"][mr][-1])), n_solves_not_solved=bad)]


def run(a):
    return dict(sweep=job_sweep, handover=job_handover, track=job_track)[a[0]](a[1:])


def main():
    t0 = time.time(); jobs = []
    for name, kw in VARIANTS.items():
        jobs += [("sweep", name, kw, s) for s in ("04749bb9", "bd07fb91")] + [("handover", name, kw, s, 0.0, "warm") for s in T.HIGHWAY] + [("handover", name, kw, "bd07fb91", 6.0, "cold")] + [("track", name, kw, s) for s in T.scenes()]
    jobs.append(("handover", "default", {}, "bd07fb91", 6.0, "warm"))
    with mp.Pool(16) as pool: res = [r for x in pool.map(run, jobs, chunksize=1) for r in x]
    df = pd.DataFrame(res); sw = df[df.kind != "tracking"].dropna(axis=1, how="all"); tr = df[df.kind == "tracking"].dropna(axis=1, how="all")
    sw.to_csv(T.HERE / "fixC_sweep.csv", index=False); tr.to_csv(T.HERE / "fixC_tracking.csv", index=False)
    pd.set_option("display.width", 260)
    s = sw[sw.kind == "sweep"]; print(s.pivot_table(index=["controller"], columns=["seed_scene", "const_decel_mps2"], values="peak_abs_steer_rad").round(3).to_string())
    print(s.pivot_table(index=["controller"], columns=["seed_scene", "const_decel_mps2"], values="u_long_min_mps2").round(1).to_string())
    print(s.pivot_table(index=["controller"], columns=["seed_scene", "const_decel_mps2"], values="speed_end_mps").round(1).to_string())
    print(sw[sw.kind == "original_plan"][["controller", "seed_scene", "tick_after_handover_s", "seeding", "peak_abs_steer_rad", "u_long_min_mps2", "u_long_mean_mps2", "speed_end_mps", "n_solves_not_solved"]].round(3).to_string())
    print(tr[tr.scene_set == "clean5"].groupby("controller")[["max_abs_lateral_dev_from_recorded_m", "max_abs_longitudinal_dev_from_recorded_m", "peak_abs_steer_rad", "max_abs_steer_diff_from_recorded_rad", "n_solves_not_solved"]].max().round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
