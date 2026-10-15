#!/usr/bin/env python
"""T01: read-only inventory of the Drive-JEPA 0.1.4 / 0.1.4b / 0.1.6 closed-loop sets and a per-scene table on honest metrics.

Reads every number from each run's own files (rollouts/*/*/metrics.parquet per tick, aggregate/metrics_results.parquet,
controller/*.csv, rollout.asl through the S5 loader, run logs). Writes CSVs next to this file. Touches no run dir.

  cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" alpasim/.venv/bin/python "analysis_3.0/14_after_retraining_diagnosis/part1_inventory/t01_build.py"
"""
from __future__ import annotations
import asyncio, csv, glob, json, pathlib, pickle, re, sys, warnings
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]
B = ALP / "analysis_3.0" / "Re-ranker" / "baseline_014b"; S5 = ALP / "analysis_3.0" / "Retrain-data" / "07_closed_loop_test"
sys.path.insert(0, str(B)); import b014b_readout as BR  # noqa: E402  (the S5 readers: X.replay, SC.scene_score, X.scene_map)
X, SC = BR.X, BR.SC
HDD = pathlib.Path("{runs}/alpasim_bulk/baseline_014b/runs"); RUNS = ALP / "runs"
SCENES = [l.split() for l in open(S5 / "tables" / "scenes_to_run.txt")]  # short, clipgt id, lane
SM = X.scene_map()
MENUS = {"0.1.4b": lambda s: B / "menus" / f"b014b_{s}", "0.1.6": lambda s: S5 / "menus" / f"d016_{s}"}
MAX_DGT = 4.0  # eval-config.yaml aggregation_modifiers.max_dist_to_gt_trajectory


def run_dir(ver, s):
    if ver == "0.1.6": return RUNS / "drivejepa" / f"drivejepa_016_{s}"
    if ver == "0.1.4b": return HDD / f"b014b_{s}_on"
    if ver == "0.1.4": return pathlib.Path(SM[s]["run"])
    if ver == "cv_0.2.3":
        return RUNS / "baselines" / "cv_023_20hz_pilot10" if SM[s]["lane"] == "pilot10" else RUNS / "baselines" / "cv_023_20hz_ext20" / f"cv_023_20hz_ext20_{s}"


def load(run, s):
    asl = glob.glob(str(run / "rollouts" / f"clipgt-{s}*" / "*" / "rollout.asl")); assert len(asl) == 1, (run, s, asl)
    sc = asyncio.run(X.replay(asl[0])); ts = np.array(sorted(sc["poses"])); P = np.array([sc["poses"][t][0][:2, 3] for t in ts])
    V = np.zeros(len(ts)); V[1:] = np.hypot(*np.diff(P, axis=0).T) / (np.diff(ts.astype(float)) / 1e6); V[0] = V[1]
    gt_ts = np.array([t for t, _ in sc["gt"]], float); gt_xy = np.array([m[:2, 3] for _, m in sc["gt"]])
    m = pd.read_parquet(pathlib.Path(asl[0]).parent / "metrics.parquet"); w = m.pivot_table(index="timestamps_us", columns="name", values="values").sort_index()
    return sc, ts, P, V, gt_ts, gt_xy, w, pathlib.Path(asl[0]).parent


def window(w):
    """Re-implementation of eval/aggregation: start at first eval_relevant tick; cut after the first tick with dist_to_gt_trajectory >= 4 m
    (main.py:136-139) and after the first tick with collision_any or offroad (processing.py:66). Returns kept index and the reason."""
    rel = w[w.eval_relevant > 0]; t = rel.index.values
    ev = {"dist_to_gt>=4m": rel.index[rel.dist_to_gt_trajectory >= MAX_DGT], "offroad": rel.index[rel.offroad > 0],
          "collision_at_fault": rel.index[(rel.collision_front > 0) | (rel.collision_lateral > 0)], "collision_not_at_fault": rel.index[(rel.collision_any > 0) & ~((rel.collision_front > 0) | (rel.collision_lateral > 0))]}
    ends = {}
    for k, ix in ev.items():
        if len(ix): ends[k] = ix[0]   # last kept tick: the event tick itself is kept (checked against every aggregate file)
    if not ends: return rel, "end_of_clip"
    why = min(ends, key=lambda k: (ends[k], k)); same = sorted(k for k in ends if ends[k] == ends[why])
    return rel[rel.index <= ends[why] + 1], "+".join(same)


def inventory(ver, s, run, sc, rd):
    wz = yaml.safe_load(open(run / "wizard-config.yaml")); log = (run / "run_ext20_scene.log"); log = log.read_text(errors="replace") if log.is_file() else ""
    drvf = [f for f in (run / "driver.log", run / "cv_driver.log") if f.is_file()]; drv = drvf[0].read_text(errors="replace") if drvf else ""
    md = yaml.safe_load(open(run / "run_metadata.yaml")) if (run / "run_metadata.yaml").is_file() else {}
    pv = re.search(r"adapter_version=(\S+) version_id=(\S+) planner_module=(\S+)", drv); pok = re.findall(r"prov OK: (.*)", log); sha = re.findall(r"\b([0-9a-f]{64})\b", pok[-1]) if pok else []
    gates = json.load(open(run / "ext20_gates.json")) if (run / "ext20_gates.json").is_file() else {}
    flat = json.dumps(wz, sort_keys=True, default=str)
    def find(key):
        r = re.findall(rf'"{key}": ([^,}}]+)', flat); return "|".join(sorted(set(r)))
    return dict(version=ver, scene=s, folder=str(run), run_time=md.get("run_time"), complete_marker=(rd / "_complete").is_file(), n_rollout_dirs=len(glob.glob(str(run / "rollouts" / f"clipgt-{s}*" / "*" / "rollout.asl"))),
                run_end_rc=";".join(re.findall(r"RUN END rc=(\d+)", log)), launches_in_log=len(re.findall(r"scenes csv sha256", log)), gates_ok=gates.get("ok"),
                adapter_version_logged=pv.group(1) if pv else None, version_id_logged=pv.group(2) if pv else None, planner_logged=pv.group(3) if pv else None, weights_sha_logged=";".join(sha), prov_ok_line=(pok[-1][:400] if pok else None),
                scenes_csv_sha=";".join(sorted(set(re.findall(r"scenes csv sha256 (\w{8})", log)))), egodriver_asl=sc["meta"].get("egodriver"), nre_asl=sc["meta"].get("nre_version"), n_sim_steps_asl=sc["meta"].get("n_sim_steps"),
                control_timestep_us=find("control_timestep_us"), force_gt_duration_us=find("force_gt_duration_us"), skip_driver_during_force_gt=find("skip_driver_during_force_gt"), n_sim_steps_cfg=find("n_sim_steps"), base_image=find("base_image"),
                driver_device=";".join(sorted(set(re.findall(r"(?:--device[ =]|device=)['\"]?(cpu|cuda[:\d]*)", drv + log)))))


def flat_cfg(run, s):
    d = yaml.safe_load(open(run / "wizard-config.yaml")); out = {}
    def rec(p, v):
        if isinstance(v, dict):
            for k, x in v.items(): rec(f"{p}.{k}" if p else str(k), x)
        elif isinstance(v, list):
            for i, x in enumerate(v): rec(f"{p}[{i}]", x)
        else: out[p] = re.sub(r"/\S*?(" + re.escape(run.name) + r")", "<RUNDIR>", str(v)).replace(run.name, "<RUN>")
    rec("", d); return out


def one(ver, s, lane):
    run = run_dir(ver, s); sc, ts, P, V, gt_ts, gt_xy, w, rd = load(run, s)
    rel = w[w.eval_relevant > 0]; ho = float(rel.index[0]); fg = int(sc["meta"]["force_gt_duration_us"]); ho_meta = sc["meta"]["render_start_us"] + fg
    kept, why = window(w); t = (ts - ho) / 1e6; aft = t >= -1e-6; tw = (w.index.values - ho) / 1e6; wa = w[tw >= -1e-6]; ta = tw[tw >= -1e-6]
    def first(mask): return float(ta[np.argmax(mask)]) if mask.any() else np.nan
    fault = ((wa.collision_front > 0) | (wa.collision_lateral > 0)).values; anyc = (wa.collision_any > 0).values; off = (wa.offroad > 0).values
    i0 = int(np.argmax(anyc)) if anyc.any() else None   # at fault = the FIRST contact tick is front or lateral (the evaluator's rule, processing.py:46-50, applied to the first contact)
    ctype = "+".join(k for k in ("front", "lateral", "rear") if i0 is not None and wa[f"collision_{k}"].values[i0] > 0); fault_any_tick = fault.copy(); fault = np.zeros_like(anyc) if i0 is None or not fault_any_tick[i0] else anyc
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(gt_xy, axis=0).T))]; t_end = float(w.index[-1]); hd = float(np.interp(t_end, gt_ts, seg) - np.interp(ho, gt_ts, seg))
    d_ho = float(w.dist_traveled_m[rel.index[0]]); row = dict(scene=s, set=lane, version=ver, handover_tick_us=int(ho), handover_eq_render_start_plus_force_gt=bool(abs(ho - ho_meta) < 1), handover_eq_first_ask_plus_3s=bool(abs(ho - (sc["asks"][0]["tn"] + fg)) < 1) if sc["asks"] else None,
               handover_speed_mps=float(V[np.argmin(np.abs(t))]), human_mean_speed_mps=hd / ((t_end - ho) / 1e6), human_dist_traveled_m=hd,
               progress_rel_to_total=float(kept.progress_rel_to_total.iloc[-1]), progress_rel_to_total_full_clip=float(w.progress_rel_to_total.iloc[-1]), progress_rel_to_total_at_handover=float(w.progress_rel_to_total[rel.index[0]]),
               collision_any=int(anyc.any()), collision_at_fault=int(fault.any()), collision_time_s=first(anyc), first_collision_type=ctype, collision_front_or_lateral_any_tick=int(fault_any_tick.any()), collision_front_or_lateral_first_time_s=first(fault_any_tick), collision_in_scored_window=int((kept.collision_any > 0).any()),
               offroad=int(off.any()), offroad_first_time_s=first(off), offroad_total_s=float(off.sum() * 0.05), offroad_in_scored_window=int((kept.offroad > 0).any()),
               dist_traveled_m=float(w.dist_traveled_m.iloc[-1] - d_ho), dist_traveled_scored_window_m=float(kept.dist_traveled_m.iloc[-1] - d_ho), dist_before_handover_m=d_ho, dist_traveled_metric_agg_m=float(kept.dist_traveled_m.iloc[-1]),
               scored_ticks=len(kept), ticks_after_handover=len(rel), scored_window_fraction=len(kept) / len(rel), scored_window_end_s=float((kept.index[-1] - ho) / 1e6), clip_end_s=float((t_end - ho) / 1e6), why_window_ended=why,
               mean_speed_mps=float(V[aft].mean()), mean_speed_scored_window_mps=float(V[aft & (ts <= kept.index[-1] + 1)].mean()), min_speed_after_handover_mps=float(V[aft].min()),
               first_move_gt_0p5_s=(float(t[aft][np.argmax(V[aft] > 0.5)]) if (V[aft] > 0.5).any() else np.nan), max_dist_to_gt_m=float(wa.dist_to_gt_trajectory.max()))
    # aggregate file (single-scene runs only): the values S5 read, and a check of window()
    af = run / "aggregate" / "metrics_results.parquet"; a = pd.read_parquet(af).iloc[0] if af.is_file() else None
    if a is not None and int(a.get("n_clips", 1)) == 1:
        row.update(agg_progress_rel_to_total=float(a.progress_rel_to_total), agg_duration_frac_20s=float(a.duration_frac_20s), agg_dist_traveled_m=float(a.dist_traveled_m), agg_collision_any=float(a.collision_any), agg_collision_at_fault=float(a.collision_at_fault),
                   agg_offroad=float(a.offroad), s5_headline_scene_score=float(SC.scene_score(a)), gt_dist_traveled_total_m=float(a.gt_dist_traveled_m))
    else:   # multi-scene aggregate (cv pilot10): same rule on the window() values
        k = kept.iloc[-1]; ps = min(max(min(float(k.progress_rel_to_total), 1.0), 0.0) / 0.8, 1.0); ps = 1.0 if float(k.gt_dist_traveled_m) < 5 else ps
        row.update(s5_headline_scene_score=(ps if not (((kept.collision_front > 0) | (kept.collision_lateral > 0)).any() or (kept.offroad > 0).any()) else 0.0), gt_dist_traveled_total_m=float(k.gt_dist_traveled_m))
    # steering (controller csv)
    cs = sorted(glob.glob(str(run / "controller" / "alpasim_controller_*.csv"))); st = None
    if len(cs) == 1 or (cs and ver != "cv_0.2.3"):
        c = pd.read_csv(cs[0]); c = c[c.timestamp_us >= ho - 1]; st = np.c_[(c.timestamp_us.values - ho) / 1e6, c.front_steering_angle.values, np.hypot(c.vx.values, c.vy.values), c.acceleration.values]
        i = int(np.argmax(np.abs(st[:, 1]))); over = np.abs(st[:, 1]) > 0.3
        row.update(peak_abs_steer_rad=float(abs(st[i, 1])), peak_abs_steer_time_s=float(st[i, 0]), first_steer_gt_0p3_s=(float(st[np.argmax(over), 0]) if over.any() else np.nan))
    # driven plan (regenerated menus; 0.1.4b and 0.1.6 only): step1_decel = demanded longitudinal acceleration over the plan's first 0.5 s
    men = []
    if ver in MENUS:
        for f in sorted(glob.glob(str(MENUS[ver](s) / "replan_*.npz"))):
            z = np.load(f, allow_pickle=True); ch = int(z["chosen"]); pr = z["proposals"][ch]
            men.append(dict(t=float(z["t_after_ho"]), ego_v=float(z["ego_v"]), acc=float(z["step1_decel"][ch]), lat1=float(pr[1, 1]), lat2=float(pr[3, 1]), x1=float(pr[1, 0]), x2=float(pr[3, 0]), changed=bool(z["changed"]), gate_dev_m=float(z["gate_dev_m"])))
        post = [m for m in men if m["t"] >= -1e-6]
        if post:
            j = int(np.argmin([m["acc"] for m in post])); row.update(peak_demanded_decel_mps2=float(-post[j]["acc"]), peak_demanded_decel_time_s=post[j]["t"], n_menus_after_handover=len(post), menu_max_gate_dev_m=max(m["gate_dev_m"] for m in post))
    series = dict(t=t, V=V, P=P, tw=tw, dist=w.dist_traveled_m.values - d_ho, offroad=(w.offroad > 0).values, coll=(w.collision_any > 0).values, steer=st, menus=men, gt_t=(gt_ts - ho) / 1e6, gt_d=seg - np.interp(ho, gt_ts, seg), win_end=row["scored_window_end_s"])
    return row, series, (inventory(ver, s, run, sc, rd) if ver != "cv_0.2.3" else None), run


def at_event(series, te):
    """Speed at te and the driven plan of the replan immediately before te."""
    if series is None or not np.isfinite(te): return {}
    t, V = series["t"], series["V"]; out = dict(speed_at_event_mps=float(V[np.argmin(np.abs(t - te))])); prev = [m for m in series["menus"] if m["t"] <= te + 1e-6]
    if prev: m = prev[-1]; out.update(replan_before_t_s=m["t"], plan_demanded_accel_first_0p5s_mps2=m["acc"], plan_lateral_offset_1s_m=m["lat1"], plan_lateral_offset_2s_m=m["lat2"], plan_forward_2s_m=m["x2"])
    return out


def main():
    rows, inv, ser, cfgs = [], [], {}, {}
    for s, _, lane in SCENES:
        lane = "pilot10" if lane == "P10" else "ext20"; assert SM[s]["lane"] == lane, (s, lane)
        for ver in ("0.1.4", "0.1.4b", "0.1.6", "cv_0.2.3"):
            try: r, se, iv, run = one(ver, s, lane)
            except Exception as e: print("MISSING/FAILED", ver, s, repr(e)[:200]); continue
            rows.append(r); ser[(ver, s)] = se; cfgs[(ver, s)] = flat_cfg(run, s)
            if iv: inv.append(iv)
        print(s, "done", flush=True)
    d = pd.DataFrame(rows); d.to_csv(HERE / "honest_table_long_all_columns.csv", index=False); pd.DataFrame(inv).to_csv(HERE / "inventory.csv", index=False); pickle.dump(ser, open(HERE / "series.pkl", "wb"))
    # --- window() against the aggregate files
    chk = d.dropna(subset=["agg_duration_frac_20s"]); bad = chk[(np.abs(chk.scored_window_end_s - chk.agg_duration_frac_20s * 20) > 1e-6) | (np.abs(chk.progress_rel_to_total - chk.agg_progress_rel_to_total) > 1e-9) | (np.abs(chk.dist_traveled_metric_agg_m - chk.agg_dist_traveled_m) > 1e-6)
                                                 | (chk.collision_in_scored_window != (chk.agg_collision_any > 0)) | (chk.offroad_in_scored_window != (chk.agg_offroad > 0))]
    print(f"window() vs aggregate: {len(chk) - len(bad)} of {len(chk)} runs agree"); print(bad[["scene", "version", "scored_window_end_s", "agg_duration_frac_20s", "progress_rel_to_total", "agg_progress_rel_to_total", "why_window_ended"]].to_string())
    # --- config differences
    with open(HERE / "config_diff.txt", "w") as fh:
        for a, b in (("0.1.4b", "0.1.6"), ("0.1.4", "0.1.6"), ("0.1.4", "0.1.4b"), ("cv_0.2.3", "0.1.6")):
            keys = {}
            for s, _, _ in SCENES:
                if (a, s) not in cfgs or (b, s) not in cfgs: continue
                ca, cb = cfgs[(a, s)], cfgs[(b, s)]
                for k in set(ca) | set(cb):
                    if ca.get(k) != cb.get(k): keys.setdefault(re.sub(r"\[\d+\]", "[]", k), []).append((s, ca.get(k), cb.get(k)))
            fh.write(f"=== wizard-config.yaml, {a} vs {b}: {len(keys)} differing keys (run-dir paths normalised)\n")
            for k, v in sorted(keys.items()): fh.write(f"  {k}: {len(v)} scene-entries; e.g. {v[0][0]}: {str(v[0][1])[:110]!r} -> {str(v[0][2])[:110]!r}\n")
    # --- honest_table.csv
    cols = ["scene", "set", "handover_speed_mps", "human_mean_speed_mps", "version", "progress_rel_to_total", "progress_rel_to_total_full_clip", "progress_rel_to_total_at_handover", "collision_any", "collision_at_fault", "collision_time_s", "first_collision_type", "collision_front_or_lateral_any_tick", "collision_front_or_lateral_first_time_s",
            "offroad", "offroad_first_time_s", "offroad_total_s", "dist_traveled_m", "dist_traveled_scored_window_m", "dist_before_handover_m", "human_dist_traveled_m", "scored_window_fraction", "scored_ticks", "ticks_after_handover", "scored_window_end_s", "clip_end_s", "why_window_ended",
            "mean_speed_mps", "mean_speed_scored_window_mps", "min_speed_after_handover_mps", "peak_abs_steer_rad", "peak_abs_steer_time_s", "peak_demanded_decel_mps2", "peak_demanded_decel_time_s", "s5_headline_scene_score", "collision_in_scored_window", "offroad_in_scored_window",
            "agg_dist_traveled_m", "first_move_gt_0p5_s", "max_dist_to_gt_m"]
    ho_speed = d[d.version == "0.1.6"].set_index("scene").handover_speed_mps; d["handover_speed_mps_per_run"] = d.handover_speed_mps; d["handover_speed_mps"] = d.scene.map(ho_speed)
    d["lane_note"] = np.where(d.version == "cv_0.2.3", "reference only: skip_driver_during_force_gt=true", "same lane")
    d[cols + ["handover_speed_mps_per_run", "lane_note"]].to_csv(HERE / "honest_table.csv", index=False, float_format="%.4f")
    D = {v: d[d.version == v].set_index("scene") for v in d.version.unique()}
    # --- swerve table (rule on the AFTER runs)
    sw = []
    for s in D["0.1.6"].index:
        a = D["0.1.6"].loc[s]
        if not (a.peak_abs_steer_rad > 0.3 or a.offroad): continue
        row = dict(scene=s, set=a.set, handover_speed_mps=a.handover_speed_mps)
        for tag, ver in (("after", "0.1.6"), ("before", "0.1.4b")):
            r = D[ver].loc[s]; te = np.nanmin([r.first_steer_gt_0p3_s, r.offroad_first_time_s]) if (np.isfinite(r.first_steer_gt_0p3_s) or np.isfinite(r.offroad_first_time_s)) else np.nan
            e = at_event(ser[(ver, s)], te); es = at_event(ser[(ver, s)], r.first_steer_gt_0p3_s)
            row.update({f"{tag}_event_time_s": te, f"{tag}_event_kind": ("none" if not np.isfinite(te) else "steer>0.3" if te == r.first_steer_gt_0p3_s and te != r.offroad_first_time_s else "offroad" if te != r.first_steer_gt_0p3_s else "steer>0.3+offroad"),
                        f"{tag}_first_steer_gt_0p3_s": r.first_steer_gt_0p3_s, f"{tag}_peak_abs_steer_rad": r.peak_abs_steer_rad, f"{tag}_peak_abs_steer_time_s": r.peak_abs_steer_time_s, f"{tag}_offroad": int(r.offroad), f"{tag}_offroad_first_time_s": r.offroad_first_time_s,
                        f"{tag}_offroad_total_s": r.offroad_total_s, f"{tag}_scored_window_end_s": r.scored_window_end_s, f"{tag}_why_window_ended": r.why_window_ended})
            row.update({f"{tag}_{k}": v for k, v in e.items()}); row.update({f"{tag}_steer_event_{k}": v for k, v in es.items() if k != "speed_at_event_mps"}); row[f"{tag}_speed_at_steer_event_mps"] = es.get("speed_at_event_mps", np.nan)
            row[f"{tag}_high_speed"] = bool(e.get("speed_at_event_mps", 0) >= 10.0)
        sw.append(row)
    sw = pd.DataFrame(sw).sort_values("after_speed_at_event_mps", ascending=False); sw.to_csv(HERE / "swerve_table.csv", index=False, float_format="%.4f")
    # --- stuck table
    stk = []
    for s in D["0.1.6"].index:
        a = D["0.1.6"].loc[s]
        if not (a.handover_speed_mps < 1.0 or a.dist_traveled_m < 5.0): continue
        row = dict(scene=s, set=a.set, handover_speed_mps=a.handover_speed_mps, rule=("handover<1m/s" if a.handover_speed_mps < 1.0 else "") + ("+dist<5m" if a.dist_traveled_m < 5.0 else ""), human_dist_traveled_m=a.human_dist_traveled_m, human_mean_speed_mps=a.human_mean_speed_mps)
        for tag, ver in (("after_016", "0.1.6"), ("before_014b", "0.1.4b"), ("before_014", "0.1.4"), ("cv_ref", "cv_0.2.3")):
            if s not in D.get(ver, pd.DataFrame()).index: continue
            r = D[ver].loc[s]; row.update({f"{tag}_dist_traveled_m": r.dist_traveled_m, f"{tag}_first_move_gt_0p5_s": r.first_move_gt_0p5_s, f"{tag}_collision_any": int(r.collision_any), f"{tag}_collision_at_fault": int(r.collision_at_fault), f"{tag}_collision_time_s": r.collision_time_s,
                                           f"{tag}_progress_rel_to_total": r.progress_rel_to_total, f"{tag}_scored_window_fraction": r.scored_window_fraction, f"{tag}_s5_score": r.s5_headline_scene_score})
        stk.append(row)
    pd.DataFrame(stk).to_csv(HERE / "stuck_table.csv", index=False, float_format="%.4f")
    # --- totals + comparison with S5's own table
    tot = {}
    for v, x in D.items():
        tot[v] = dict(n=len(x), mean_progress_rel_to_total=float(x.progress_rel_to_total.mean()), mean_progress_full_clip=float(x.progress_rel_to_total_full_clip.mean()), mean_progress_at_handover=float(x.progress_rel_to_total_at_handover.mean()), mean_scored_window_fraction=float(x.scored_window_fraction.mean()),
                      median_scored_window_fraction=float(x.scored_window_fraction.median()), n_full_window=int((x.why_window_ended == "end_of_clip").sum()), why_counts=x.why_window_ended.value_counts().to_dict(), mean_s5_score=float(x.s5_headline_scene_score.mean()),
                      at_fault=sorted(x.index[x.collision_at_fault > 0]), any_collision=sorted(x.index[x.collision_any > 0]), front_or_lateral_any_tick=sorted(x.index[x.collision_front_or_lateral_any_tick > 0]), collisions_after_window=sorted(x.index[(x.collision_any > 0) & (x.collision_in_scored_window == 0)]), offroad=sorted(x.index[x.offroad > 0]), at_fault_in_window=sorted(x.index[(x.get("agg_collision_at_fault", pd.Series(0, x.index)) > 0)]),
                      offroad_in_window=sorted(x.index[x.offroad_in_scored_window > 0]), collision_in_window=sorted(x.index[x.collision_in_scored_window > 0]), mean_dist_traveled_after_handover_m=float(x.dist_traveled_m.mean()), mean_dist_metric_agg_m=float(x.dist_traveled_metric_agg_m.mean()),
                      mean_dist_before_handover_m=float(x.dist_before_handover_m.mean()), mean_human_dist_m=float(x.human_dist_traveled_m.mean()), steer_gt_0p3=sorted(x.index[x.get("peak_abs_steer_rad", pd.Series(0, x.index)) > 0.3]),
                      median_peak_demanded_decel=(float(x.peak_demanded_decel_mps2.median()) if "peak_demanded_decel_mps2" in x and x.peak_demanded_decel_mps2.notna().any() else None), mean_speed=float(x.mean_speed_mps.mean()))
    p5 = pd.read_csv(S5 / "tables" / "per_scene.csv").set_index("scene"); cmp = []
    for arm, ver in (("014b", "0.1.4b"), ("016", "0.1.6")):
        x = D[ver]
        for mine, theirs in (("progress_rel_to_total", "progress_clipped_rel"), ("s5_headline_scene_score", "scene_score"), ("dist_traveled_metric_agg_m", "dist_traveled_m"), ("peak_abs_steer_rad", "peak_steer"), ("offroad_first_time_s", "first_offroad_s"), ("offroad_total_s", "offroad_s"),
                             ("collision_time_s", "first_collision_any_s"), ("peak_demanded_decel_mps2", "peak_demanded_decel"), ("mean_speed_scored_window_mps", "mean_speed_scored"), ("scored_window_end_s", "exit_s"), ("min_speed_after_handover_mps", "min_speed_after_ho")):
            th = p5[f"{arm}_{theirs}"].reindex(x.index); th = -th if theirs == "peak_demanded_decel" else th; df = (x[mine] - th).abs(); nanmis = int((x[mine].isna() != th.isna()).sum())
            cmp.append(dict(arm=ver, mine=mine, s5_column=f"{arm}_{theirs}", max_abs_diff=float(df.max()), n_diff_gt_0p01=int((df > 0.01).sum()), nan_mismatch=nanmis, scenes=";".join(df.index[df > 0.01][:8])))
    pd.DataFrame(cmp).to_csv(HERE / "compare_with_s5_per_scene.csv", index=False); json.dump(tot, open(HERE / "totals.json", "w"), indent=1, default=float)
    print(pd.DataFrame(cmp).to_string()); print(json.dumps(tot, indent=1, default=float))


if __name__ == "__main__":
    main()
