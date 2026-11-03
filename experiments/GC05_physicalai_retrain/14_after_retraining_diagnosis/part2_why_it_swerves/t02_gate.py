#!/usr/bin/env python
"""T02 Part A -- gate: does the standalone controller reproduce the recorded 0.1.6 run?

Three replays per (scene, start tick), each fed the recorded plans at the recorded 50 ms ticks for 3 s:
  cold_tf  fresh System seeded with the recorded controller_request.state at the start tick; each tick's request is the recorded one
           (recorded plan, recorded pose override, recorded coerce flag).                                   <- the gate the task asked for
  warm_tf  the same, but the System was first replayed from the first recorded request of the session (so the MPC warm start, the 100 ms
           solve phase and the actuator states are those of the run).
  cold_free fresh System at the start tick, pose NOT overridden (the simulator's own pose is sent back), each recorded plan held fixed
           in the world (re-expressed from the recorded rig frame into the simulator's rig frame).
Compared against the run's own controller CSV (front_steering_angle = steering state, u_steering_angle / u_longitudinal_actuation = commands).

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python "analysis_3.0/14_after_retraining_diagnosis/part2_why_it_swerves/t02_gate.py"
"""
import pickle, sys, time
import numpy as np, pandas as pd
import t02_common as C

CASES = [("04749bb9", 0.0, "highway"), ("acd22430", 0.0, "highway"), ("bd07fb91", 0.0, "highway"), ("bd07fb91", 5.5, "highway"), ("02e075b9", 0.0, "control")]
DUR = 3_000_000


def free_run(d, t0):
    s = C.Sim(d, t0); ts = t0
    while ts < t0 + DUR:
        r = C.req_at(d, ts); rec_ego = C.se2(r.state.pose); pl = C.plan_arr(r.planned_trajectory_in_rig)
        lp = s.sys._trajectory.last_pose; q = lp.quat; ego = np.array([lp.vec3[0], lp.vec3[1], np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))])   # utils_rs quat order x, y, z, w
        w = C.to_world(pl[:, 1:], rec_ego); r.planned_trajectory_in_rig.CopyFrom(C.plan_pb(ts, C.to_rig(w, ego)))
        r.state.pose.CopyFrom(C.pose_to_grpc(lp)); r.coerce_dynamic_state = False if ts > t0 else r.coerce_dynamic_state
        s.sys.run_controller_and_vehicle_model(r); ts += C.TICK
    return s


def main():
    from alpasim_utils.geometry import pose_to_grpc; C.pose_to_grpc = pose_to_grpc
    rows, series = [], {}
    for scene, rel, kind in CASES:
        d = C.load("0.1.6", scene); t0 = d["ho"] + int(rel * 1e6); rec = d["csv"]; t1 = time.time()
        assert C.req_at(d, d["ho"]).coerce_dynamic_state and not C.req_at(d, d["ho"] + C.TICK).coerce_dynamic_state
        sims = dict(cold_tf=C.Sim(d, t0).feed_recorded(t0 + DUR), warm_tf=C.sim_at(d, t0).feed_recorded(t0 + DUR), cold_free=free_run(d, t0))
        mrec = (rec["timestamp_us"] > t0) & (rec["timestamp_us"] <= t0 + DUR); tr = (rec["timestamp_us"][mrec] - t0) / 1e6
        series[(scene, rel)] = dict(t=tr, rec_steer=rec["front_steering_angle"][mrec], rec_ucmd=rec["u_steering_angle"][mrec], rec_ulong=rec["u_longitudinal_actuation"][mrec])
        for mode, s in sims.items():
            r = s.rows(); m = (r["timestamp_us"] > t0) & (r["timestamp_us"] <= t0 + DUR); assert np.array_equal(r["timestamp_us"][m], rec["timestamp_us"][mrec]), (scene, mode)
            series[(scene, rel)][mode] = dict(steer=r["front_steering_angle"][m], ucmd=r["u_steering_angle"][m], ulong=r["u_longitudinal_actuation"][m])
            bad = [x for x in s.status if "solved" not in x[1].lower() and "success" not in x[1].lower()]
            row = dict(scene=scene, start_after_handover_s=rel, kind=kind, mode=mode, n_solves=len(s.status), n_solves_not_solved=len(bad))
            for name, a, b in (("steer_state", rec["front_steering_angle"][mrec], r["front_steering_angle"][m]), ("steer_cmd", rec["u_steering_angle"][mrec], r["u_steering_angle"][m]), ("long_cmd", rec["u_longitudinal_actuation"][mrec], r["u_longitudinal_actuation"][m])):
                for wn, w in (("0p5s", tr <= 0.5 + 1e-9), ("3s", tr <= 3.0 + 1e-9)):
                    e = b[w] - a[w]; pk = np.abs(a[w]).max()
                    row[f"{name}_maxdiff_{wn}"] = float(np.abs(e).max()); row[f"{name}_rmsdiff_{wn}"] = float(np.sqrt((e ** 2).mean())); row[f"{name}_rec_peak_{wn}"] = float(pk)
                    row[f"{name}_maxdiff_pct_of_rec_peak_{wn}"] = float(100 * np.abs(e).max() / pk) if pk > 1e-9 else np.nan
            for nm, a in (("rec", rec["front_steering_angle"][mrec]), ("sim", r["front_steering_angle"][m])):
                o = np.abs(a) > 0.3; row[f"first_steer_gt_0p3_{nm}_s"] = float(tr[np.argmax(o)]) if o.any() else np.nan; row[f"peak_abs_steer_3s_{nm}_rad"] = float(np.abs(a).max())
            rows.append(row)
        print(scene, rel, f"{time.time() - t1:.0f}s", flush=True)
    g = pd.DataFrame(rows); g.to_csv(C.HERE / "gate.csv", index=False); pickle.dump(series, open(C.HERE / "gate_series.pkl", "wb"))
    pd.set_option("display.width", 250)
    print(g[["scene", "start_after_handover_s", "mode", "steer_state_maxdiff_0p5s", "steer_state_rec_peak_0p5s", "steer_state_maxdiff_pct_of_rec_peak_0p5s", "long_cmd_maxdiff_0p5s", "long_cmd_rec_peak_0p5s", "steer_state_maxdiff_3s", "steer_state_rmsdiff_3s", "long_cmd_maxdiff_3s",
             "first_steer_gt_0p3_rec_s", "first_steer_gt_0p3_sim_s", "peak_abs_steer_3s_rec_rad", "peak_abs_steer_3s_sim_rad", "n_solves_not_solved"]].round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
