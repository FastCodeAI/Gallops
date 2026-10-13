#!/usr/bin/env python
"""GENERATOR-ONLY TRIAL scene reads, 0.1.4b / 0.1.6 / 0.1.7 (copy of 07_closed_loop_test/s5_reads.py: readout module, arms, and the worse list is 0.1.7 against 0.1.6; a steering read for bd07fb91 and the >0.2 rad list added).
Original docstring: S5 STEP 3 scene reads (mandatory): highway per-tick speeds, pull-out ranks of departing plans, the no-cue stop, why the worse scenes
lost score. Reads the regenerated menus (baseline_014b/menus/b014b_<s>, 07_closed_loop_test/menus/d016_<s>) and the runs. Writes tables/reads.json.
  cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/s5_reads.py"""
import sys, glob, json, pathlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
import importlib.util   # baseline_014b registers its own patched module under the name "s5_readout" in sys.modules: load ours under another name
_sp = importlib.util.spec_from_file_location("s017_readout", HERE / "s017_readout.py"); R = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(R)
SC = R.SC; out = {}
def menus(arm, s):
    rows = []
    for f in sorted(glob.glob(str(R.MENUS[arm](s) / "replan_*.npz"))):
        z = np.load(f, allow_pickle=True); P = z["proposals"]; S = z["pdm_score"]; ch = int(z["chosen"]); sp = np.linalg.norm(np.diff(np.concatenate([np.zeros((32, 1, 2)), P[:, :, :2]], 1), axis=1), axis=2) / 0.5
        disp = np.linalg.norm(P[:, -1, :2], axis=1); rows.append(dict(t=float(z["t_after_ho"]), ego_v=float(z["ego_v"]), ch=ch, sp=sp, disp=disp, S=S, dec=float(z["step1_decel"][ch]), changed=bool(z["changed"]), lat_end=P[:, -1, 1]))
    return rows
def human_speed(run, ts_rel):
    sc, ts, P, V, M, gt_ts, gt_xy = SC.load(run); ho = sc["asks"][0]["tn"] + 3_000_000; g = (np.asarray(gt_ts) - ho) / 1e6; v = np.r_[np.linalg.norm(np.diff(gt_xy, axis=0), axis=1) / np.maximum(np.diff(g), 1e-6), np.nan]
    return [float(np.nanmean(v[(g >= t) & (g < t + 0.5)])) if ((g >= t) & (g < t + 0.5)).any() else np.nan for t in ts_rel], (ts - ho) / 1e6, V
# ---- highway
for s in ("04749bb9", "acd22430"):
    out[s] = {}
    for arm in R.ARMS:
        m = [r for r in menus(arm, s) if -0.01 <= r["t"] <= 3.01]; hs, t, V = human_speed(R.run_dir(arm, s), [r["t"] for r in m])
        rows = [dict(t=r["t"], car_speed=round(r["ego_v"], 1), human_speed=round(h, 1), chosen_first_half_s=round(float(r["sp"][r["ch"], 0]), 1), chosen_mean_4s=round(float(r["sp"][r["ch"]].mean()), 1), chosen_end=round(float(r["sp"][r["ch"], -1]), 1),
                     fastest_of_32_mean_4s=round(float(r["sp"].mean(1).max()), 1), fastest_of_32_end=round(float(r["sp"][:, -1].max()), 1), slowest_mean_4s=round(float(r["sp"].mean(1).min()), 1), demanded_accel_first_half_s=round(r["dec"], 1),
                     chosen_ge_0p8_human=bool(r["sp"][r["ch"]].mean() >= 0.8 * h), changed_by_35=r["changed"]) for r, h in zip(m, hs)]
        out[s][arm] = dict(ticks=rows, car_speed_at_1_2_3s=[round(float(V[np.argmin(abs(t - x))]), 1) for x in (1, 2, 3)], holds_23p5=bool(min(V[(t >= 0) & (t <= 3)]) >= 0.9 * 23.5), all_chosen_ge_0p8_human_first_3s=bool(all(q["chosen_ge_0p8_human"] for q in rows)))
        print(s, arm, "| car speed at 1/2/3 s", out[s][arm]["car_speed_at_1_2_3s"]); print(pd.DataFrame(rows).to_string(index=False))
# ---- pull-outs
for s in ("d2715023", "b44a4a44"):
    out[s] = {}
    for arm in R.ARMS:
        m = [r for r in menus(arm, s) if r["t"] >= -0.01]; rk = []
        for r in m:
            dep = np.where(r["disp"] >= 2.0)[0]; order = np.argsort(-r["S"]); rank = {int(j): k + 1 for k, j in enumerate(order)}
            rk.append(dict(t=r["t"], n_departing=int(len(dep)), best_rank_of_departing=(min(rank[int(j)] for j in dep) if len(dep) else None), chosen_disp_m=round(float(r["disp"][r["ch"]]), 2), max_disp_m=round(float(r["disp"].max()), 1),
                           score_chosen=round(float(r["S"][r["ch"]]), 3), best_score_departing=(round(float(r["S"][dep].max()), 3) if len(dep) else None)))
        d = pd.DataFrame(rk); wd = d[d.n_departing > 0]
        out[s][arm] = dict(replans=len(d), replans_with_a_departing_plan=int(len(wd)), median_n_departing=float(d.n_departing.median()), best_rank_median=(float(wd.best_rank_of_departing.median()) if len(wd) else None), best_rank_min=(int(wd.best_rank_of_departing.min()) if len(wd) else None),
                           departing_picked=int((d.chosen_disp_m >= 2.0).sum()), median_score_chosen=float(d.score_chosen.median()), median_best_score_departing=(float(wd.best_score_departing.median()) if len(wd) else None), max_disp_any_plan_m=float(d.max_disp_m.max()))
        print(s, arm, out[s][arm])
# ---- no-cue stop
s = "89f2eeb8"; out[s] = {}
for arm in R.ARMS:
    m = [r for r in menus(arm, s) if -0.01 <= r["t"] <= 6.01]; _, t, V = human_speed(R.run_dir(arm, s), [0]); hs, _, _ = human_speed(R.run_dir(arm, s), [r["t"] for r in m])
    out[s][arm] = dict(min_speed=round(float(V[t >= 0].min()), 2), ticks=[dict(t=r["t"], car=round(r["ego_v"], 1), human=round(h, 1), chosen_mean_4s=round(float(r["sp"][r["ch"]].mean()), 1), demanded_accel=round(r["dec"], 1), fastest_mean_4s=round(float(r["sp"].mean(1).max()), 1)) for r, h in zip(m, hs)])
    print(s, arm, "min speed", out[s][arm]["min_speed"]); print(pd.DataFrame(out[s][arm]["ticks"]).to_string(index=False))
# ---- worse scenes: what ended scoring / speed
per = pd.read_csv(HERE / "tables" / "per_scene.csv").set_index("scene"); out["worse"] = {}
for s in per.index[(per["017_scene_score"] - per["016_scene_score"]) <= -0.02]:
    row = {}
    for arm in R.ARMS:
        run = R.run_dir(arm, s); sc, ts, P, V, M, gt_ts, gt_xy = SC.load(run); ho = sc["asks"][0]["tn"] + 3_000_000; t = (ts - ho) / 1e6; lat = np.abs(SC.lateral(P, gt_xy)); ex = per.loc[s, f"{arm}_exit_s"]
        over = t[(t >= 0) & (lat > 4.0)]; row[arm] = dict(exit=per.loc[s, f"{arm}_exit_kind"], exit_s=float(ex), first_collision_any_s=(None if np.isnan(per.loc[s, f"{arm}_first_collision_any_s"]) else float(per.loc[s, f"{arm}_first_collision_any_s"])),
                                                         first_time_4m_off_path_s=(float(over.min()) if len(over) else None), abs_lat_at_exit=round(float(lat[np.argmin(abs(t - ex))]), 2), mean_speed_to_exit=round(float(V[(t >= 0) & (t <= ex)].mean()), 2),
                                                         mean_speed_whole_clip=round(float(V[t >= 0].mean()), 2), dist_at_exit_m=round(float(np.linalg.norm(np.diff(P[(t >= 0) & (t <= ex)], axis=0), axis=1).sum()), 1), longest_below_1ms=float(per.loc[s, f"{arm}_longest_slow_s"]),
                                                         human_dist_to_exit_m=round(float(np.linalg.norm(np.diff(gt_xy[((np.asarray(gt_ts) - ho) / 1e6 >= 0) & ((np.asarray(gt_ts) - ho) / 1e6 <= ex)], axis=0), axis=1).sum()), 1))
    out["worse"][s] = row; print(s, json.dumps(row))
# ---- steering: every scene/arm with peak steer above 0.2 rad after handover, when, the car's speed then, and whether scoring had already ended
out["steer"] = {}
for s in per.index:
    row = {}
    for arm in R.ARMS:
        run = R.run_dir(arm, s); sc, ts, P, V, M, gt_ts, gt_xy = SC.load(run); ho = sc["asks"][0]["tn"] + 3_000_000; t = (ts - ho) / 1e6; st = SC.steering(run); tt = (st.timestamp_us.to_numpy() - ho) / 1e6; a = np.abs(st.front_steering_angle.to_numpy()); post = tt >= 0
        over = tt[post & (a > 0.2)]; ex = float(per.loc[s, f"{arm}_exit_s"]); fc = per.loc[s, f"{arm}_first_collision_any_s"]; fo = per.loc[s, f"{arm}_first_offroad_s"]
        row[arm] = dict(peak=round(float(a[post].max()), 3), peak_t=round(float(tt[post][a[post].argmax()]), 2), first_above_0p2_s=(round(float(over.min()), 2) if len(over) else None), seconds_above_0p2=round(float(len(over)) * float(np.median(np.diff(tt))), 2),
                        speed_at_first_above=(round(float(V[np.argmin(abs(t - over.min()))]), 1) if len(over) else None), scoring_ends_s=ex, exit=str(per.loc[s, f"{arm}_exit_kind"]), first_collision_any_s=(None if np.isnan(fc) else float(fc)), first_offroad_s=(None if np.isnan(fo) else float(fo)),
                        above_0p2_only_after_scoring=(bool(over.min() > ex) if len(over) else None))
    out["steer"][s] = row
    if any(v["peak"] > 0.2 for v in row.values()): print("steer", s, json.dumps(row))
json.dump(out, open(HERE / "tables" / "reads.json", "w"), indent=1, default=float)
