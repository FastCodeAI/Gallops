#!/usr/bin/env python
"""T03 test bed: every replan of every 0.1.6 scene, six variants (baseline, A1, A2, A3, B1, B2), one plan held 1.0 s in the standalone
controller branched from the warm (teacher-forced from session start) simulator at that tick. Also the 4.0 s holds at the two highway
handover ticks. Writes testbed.csv, fixA_alteration.csv, fixB_picks.csv, later_braking_cases.csv, long_hold.csv, long_hold.pkl.

  CUDA_VISIBLE_DEVICES="" {runs}/alpasim_bulk/nurec-venv/bin/python t03_testbed.py
"""
import glob, multiprocessing as mp, pickle, time
import numpy as np, pandas as pd
import t03_common as T

C, R = T.C, T.R
VARS = ["baseline", "A1", "A2", "A3", "B1", "B2"]


def group_of(s, ho_speed):
    return "highway" if s in T.HIGHWAY else "bd07fb91" if s == "bd07fb91" else "other>=10" if ho_speed >= 10 else "<10"


def tick_metrics(run):
    (f,) = glob.glob(str(run) + "/rollouts/*/*/metrics.parquet"); m = pd.read_parquet(f)
    return m.pivot_table(index="timestamps_us", columns="name", values="values").sort_index()


def one_scene(s):
    t0 = time.time(); d = T.load(s); ho = d["ho"]; M = T.menus(s); w = tick_metrics(d["run"]); rc = d["csv"]; g = d["gt"]
    ho_speed = R.ego_state(d, ho)[1]; grp = group_of(s, ho_speed); sim = C.Sim(d, d["ticks"][0]); tb, alt, picks, later, longs = [], [], [], [], {}
    wt = w.index.values; off = (w.offroad.values > 0); coll = (w.collision_any.values > 0); fault = ((w.collision_front.values > 0) | (w.collision_lateral.values > 0)); dobs = w.min_distance_to_obstacle_m.values
    for ts, z in M.items():
        if ts not in d["reqs"]: continue
        sim.feed_recorded(ts); ego, v0, r = R.ego_state(d, ts); rel = (ts - ho) / 1e6; ch = int(z["chosen"]); menu_plan = np.r_[np.zeros((1, 3)), z["proposals"][ch].astype(float)]
        post = ts >= ho; rp = C.plan_arr(r.planned_trajectory_in_rig)
        if post:
            assert len(rp) == 9 and np.allclose(rp[:, 0], T.T8), (s, rel, len(rp)); base = rp[:, 1:]; dev = float(np.abs(base[:, :2] - menu_plan[:, :2]).max()); assert dev < 1e-3, (s, rel, dev)
        else:
            base = menu_plan; dev = np.nan     # before handover the controller was fed the 151-pose ground-truth trajectory, not the model's plan
        # ---- tags
        mr = (rc["timestamp_us"] > ts - 1_000_000) & (rc["timestamp_us"] <= ts); rec_steer_1s = float(np.abs(rc["front_steering_angle"][mr]).max()) if mr.any() else 0.0
        off_yet = bool(off[wt <= ts].any()); pre_swerve = bool(rec_steer_1s <= 0.2 and not off_yet)
        tag = dict(scene=s, group=grp, handover_speed_mps=ho_speed, tick_after_handover_s=rel, phase="post_handover" if post else "pre_handover_hypothetical", car_speed_mps=v0, speed_ge_10=bool(v0 >= 10),
                   pre_swerve=pre_swerve, recorded_peak_steer_prev_1s_rad=rec_steer_1s, offroad_already=off_yet)
        # ---- plans
        A, ia = T.fix_a(base, v0); pick, keep, a1s, mvs = T.fix_b(z, v0); b1 = base if pick == ch else np.r_[np.zeros((1, 3)), z["proposals"][pick].astype(float)]; AB, ib = T.fix_a(b1, v0)
        plans = dict(baseline=base, A1=A["A1"], A2=A["A2"], A3=A["A3"], B1=b1, B2=AB["A1"]); done = []
        for name in VARS:
            p = plans[name]; same = next((m for q, m in done if q.shape == p.shape and np.abs(q - p).max() < 1e-9), None); m = same if same is not None else T.held(sim, ts, p, ego, also_2s=True); done.append((p, m))
            P = R.Path(p); sp = R.step_speeds(p); dd = np.hypot(*(p[[1, 2, 4, 8], :2] - base[[1, 2, 4, 8], :2]).T)
            tb.append(dict(**tag, variant=name, plan_changed=bool(np.abs(p - base).max() > 1e-6), plan_a1_mps2=T.a1_of(p, v0), plan_first_0p5s_accel_mps2=float((sp[0] - v0) / 0.5), plan_arc_4s_m=float(P.s[-1]), plan_end_speed_mps=float(sp[-1]), **m))
            if name in ("A1", "A2", "A3"):
                alt.append(dict(**tag, variant=name, plan_changed=bool(dd.max() > 1e-6), pos_diff_0p5s_m=dd[0], pos_diff_1s_m=dd[1], pos_diff_2s_m=dd[2], pos_diff_4s_m=dd[3], heading_diff_max_rad=float(np.abs(np.arctan2(np.sin(p[:, 2] - base[:, 2]), np.cos(p[:, 2] - base[:, 2]))).max()),
                                **{k: v for k, v in ia.items()}))
        # ---- B1 pick bookkeeping
        tq = ts + np.arange(0, 4.0001, 0.05) * 1e6; hmean = float(np.hypot(*np.diff(np.c_[np.interp(tq, g[:, 0], g[:, 1]), np.interp(tq, g[:, 0], g[:, 2])], axis=0).T).sum() / 4.0) if tq[-1] <= g[-1, 0] else np.nan
        sc = z["pdm_score"].astype(float); rk = np.empty(32, int); rk[np.argsort(-sc, kind="stable")] = np.arange(1, 33); rkp = np.empty(32, int); rkp[np.argsort(-z["S_plain"], kind="stable")] = np.arange(1, 33)
        dp, _ = T.dense_path(base, R.Path(b1).s[-1]); lat = R.lateral_from(dp, b1[[4, 8]])
        picks.append(dict(**tag, n_survive_of_32=int(keep.sum()), none_left_kept_original=bool(not keep.any()), original_pick=ch, original_pick_survives=bool(keep[ch]), new_pick=pick, pick_changed=bool(pick != ch),
                          new_pick_rank_final_score=int(rk[pick]), new_pick_rank_plain_score=int(rkp[pick]), original_pick_rank_plain_score=int(rkp[ch]), new_pick_score=float(sc[pick]), original_pick_score=float(sc[ch]),
                          original_a1_mps2=float(a1s[ch]), new_a1_mps2=float(a1s[pick]), original_mean_speed_4s_mps=float(mvs[ch]), new_mean_speed_4s_mps=float(mvs[pick]), human_mean_speed_4s_mps=hmean,
                          new_lateral_from_original_path_2s_m=float(lat[0]), new_lateral_from_original_path_4s_m=float(lat[1]), new_minus_original_pos_2s_m=float(np.hypot(*(b1[4, :2] - base[4, :2]))), new_minus_original_pos_4s_m=float(np.hypot(*(b1[8, :2] - base[8, :2]))),
                          n_dropped_a1_low=int(((a1s < -T.CAP) & (v0 >= T.V_GATE)).sum()), n_dropped_a1_high=int((a1s > T.CAP).sum()), n_dropped_too_fast=int((mvs > 1.5 * v0 + 5).sum())))
        # ---- cost side: the fix brakes later / less than the model asked
        w4 = (wt > ts) & (wt <= ts + 4_000_000); spb = R.step_speeds(base)
        rec4 = dict(recorded_collision_any_within_4s=bool(coll[w4].any()), recorded_collision_front_or_lateral_within_4s=bool(fault[w4].any()), recorded_offroad_within_4s=bool(off[w4].any()),
                    recorded_min_distance_to_obstacle_within_4s_m=(float(np.nanmin(dobs[w4])) if w4.any() and np.isfinite(dobs[w4]).any() else np.nan), recorded_ticks_in_4s=int(w4.sum()))
        common = dict(**tag, original_plan_end_speed_mps=float(spb[-1]), original_plan_first_0p5s_accel_mps2=float((spb[0] - v0) / 0.5), original_plan_a1_mps2=ia["a1_mps2"], original_plan_arc_4s_m=ia["plan_arc_4s_m"], original_const_accel_to_end_mps2=ia["A1_accel_mps2"])
        if ia["A2_cap_bites"]: later.append(dict(fix="A2", **common, fixed_plan_arc_4s_m=float(R.Path(A["A2"]).s[-1]), ends_beyond_model_end_m=ia["A2_end_beyond_model_end_m"], fixed_plan_end_speed_mps=float(R.step_speeds(A["A2"])[-1]), **rec4))
        if pick != ch and R.Path(b1).s[-1] > ia["plan_arc_4s_m"] + 0.1:
            later.append(dict(fix="B1", **common, fixed_plan_arc_4s_m=float(R.Path(b1).s[-1]), ends_beyond_model_end_m=float(R.Path(b1).s[-1] - ia["plan_arc_4s_m"]), fixed_plan_end_speed_mps=float(R.step_speeds(b1)[-1]), **rec4))
        # ---- 4 s holds at the highway handover ticks (plus bd07fb91 +6.0 s for the record)
        if (s in T.HIGHWAY and ts == ho) or (s == "bd07fb91" and ts == ho + 6_000_000):
            for name in VARS:
                m = T.held(sim, ts, plans[name], ego, dur_us=4_000_000, keep_series=True); longs[(s, rel, name)] = dict(plan=plans[name], **m.pop("_series"), summary=m)
    print(f"{s} {grp} {len(M)} replans {time.time() - t0:.0f}s", flush=True)
    return tb, alt, picks, later, longs


def main():
    t0 = time.time(); S = T.scenes(); assert len(S) == 30, len(S)
    with mp.Pool(15) as pool: res = pool.map(one_scene, S, chunksize=1)
    tb = pd.DataFrame([r for x in res for r in x[0]]); alt = pd.DataFrame([r for x in res for r in x[1]]); pk = pd.DataFrame([r for x in res for r in x[2]]); lt = pd.DataFrame([r for x in res for r in x[3]])
    longs = {k: v for x in res for k, v in x[4].items()}
    tb.to_csv(T.HERE / "testbed.csv", index=False); alt.to_csv(T.HERE / "fixA_alteration.csv", index=False); pk.to_csv(T.HERE / "fixB_picks.csv", index=False); lt.to_csv(T.HERE / "later_braking_cases.csv", index=False)
    pickle.dump(longs, open(T.HERE / "long_hold.pkl", "wb"))
    rows = [dict(scene=s, tick_after_handover_s=rel, variant=n, t_s=t, front_steering_angle_rad=a, speed_mps=v, lateral_from_plan_path_m=l, lateral_rig_m=lr, u_long_mps2=u)
            for (s, rel, n), L in longs.items() for t, a, v, l, lr, u in zip(L["t"], L["steer"], L["speed"], L["lat_plan"], L["lat_rig"], L["ulong"])]
    pd.DataFrame(rows).to_csv(T.HERE / "long_hold.csv", index=False)
    print(len(tb), "rows;", tb.groupby("variant").size().to_dict(), f"total {time.time() - t0:.0f}s")
    b = tb[(tb.variant == "baseline") & (tb.tick_after_handover_s == 0) & tb.scene.isin(T.HIGHWAY)]; print(b[["scene", "peak_abs_steer_rad"]].to_string())


if __name__ == "__main__":
    main()
