#!/usr/bin/env python
"""S5 STEP 3: per-scene comparison, recorded 0.1.4b vs 0.1.6 (new runs). Read-only on all runs.

Same code on both arms: baseline_014b/b014b_readout.py's own functions (arm_block, momentum_rows, stall), which call the S5 scripts.
Adds the fields the S5 brief names: mean speed, collision at fault / not and its time, off-road time, peak steer, peak demanded
deceleration of the driven pick (from the regenerated menus, when they exist), first-plan speed ratio at handover (mode1_fields a3_ratio),
number of §3.5 pick changes. Writes tables/per_scene.csv, tables/replans_016.csv, tables/insight_facts.csv, tables/totals.json.

  cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/s5_readout.py [--scenes a,b]
"""
from __future__ import annotations
import argparse, glob, json, pathlib, re, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent; ALP = HERE.parents[2]; T = HERE / "tables"; B = ALP / "analysis_3.0" / "Re-ranker" / "baseline_014b"
sys.path.insert(0, str(B)); import b014b_readout as BR  # noqa: E402
SC, RO = BR.SC, BR.RO; RUNS16 = ALP / "runs" / "drivejepa"; ARMS = ("014b", "016")
MENUS = {"014b": lambda s: B / "menus" / f"b014b_{s}", "016": lambda s: HERE / "menus" / f"d016_{s}"}
def run_dir(arm, s): return BR.RUNS / f"b014b_{s}_on" if arm == "014b" else RUNS16 / f"drivejepa_016_{s}"

def extra(run, keep, blk, arm, s):
    sc, ts, V, ho, P, gt_xy = (keep[k] for k in ("sc", "ts", "V", "ho", "P", "gt_xy")); t = (ts - ho) / 1e6; ex = blk["exit_s"]
    m = pd.read_parquet(glob.glob(str(run / "rollouts" / "clipgt-*" / "*" / "metrics.parquet"))[0])
    def first(nm): g = m[(m.name == nm) & (m["values"] > 0)]; return (float(g.timestamps_us.min()) - ho) / 1e6 if len(g) else np.nan
    def secs(nm): g = m[(m.name == nm) & (m.timestamps_us >= ho)]; return float((g["values"] > 0).sum()) * 0.05 if len(g) else np.nan
    st = SC.steering(run); stp = st[st.timestamp_us >= ho]; i_pk = stp.front_steering_angle.abs().idxmax(); lat = SC.lateral(P, gt_xy)
    out = dict(mean_speed_scored=float(V[(t >= 0) & (t <= max(ex, 0.05))].mean()), mean_speed_after_ho=float(V[t >= 0].mean()), first_collision_any_s=first("collision_any"), first_collision_at_fault_s2=first("collision_at_fault"),
               first_offroad_s=first("offroad"), offroad_s=secs("offroad"), peak_steer=float(stp.front_steering_angle.abs().max()), peak_steer_t=(float(st.timestamp_us[i_pk]) - ho) / 1e6,
               peak_steer_first_1p5s=float(stp[stp.timestamp_us <= ho + 1_500_000].front_steering_angle.abs().max()), abs_lat_at_exit=float(abs(lat[int(np.argmin(np.abs(t - ex)))])), max_abs_lat=float(np.abs(lat[t >= 0]).max()))
    md = MENUS[arm](s); fs = sorted(glob.glob(str(md / "replan_*.npz"))); dec = []; rows = []
    for f in fs:
        z = np.load(f, allow_pickle=True); ta = float(z["t_after_ho"])
        if ta < -1e-6: continue
        P32 = z["proposals"]; ch = int(z["chosen"]); sp = np.linalg.norm(np.diff(np.concatenate([np.zeros((32, 1, 2)), P32[:, :, :2]], 1), axis=1), axis=2) / 0.5
        dec.append(float(z["step1_decel"][ch])); rows.append(dict(t=ta, ego_v=float(z["ego_v"]), chosen_step1=float(sp[ch, 0]), chosen_end=float(sp[ch, -1]), chosen_mean=float(sp[ch].mean()), fastest_end=float(sp[:, -1].max()), fastest_mean=float(sp.mean(1).max()),
                                                                 slowest_mean=float(sp.mean(1).min()), chosen=ch, changed=bool(z["changed"])))
    out.update(peak_demanded_decel=(float(min(dec)) if dec else np.nan), peak_demanded_accel=(float(max(dec)) if dec else np.nan), menus=len(rows))
    return out, pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", default=""); ap.add_argument("--print-only", action="store_true"); a = ap.parse_args()
    scenes = a.scenes.split(",") if a.scenes else [l.split()[0] for l in open(T / "scenes_to_run.txt")]
    per, mom, facts, skipped = [], [], [], []; (T / "ticks").mkdir(exist_ok=True)
    for s in scenes:
        if not (run_dir("016", s) / "aggregate" / "metrics_results.parquet").is_file(): skipped.append(s); continue
        _, keep014 = BR.arm_block(SC.ref_dir(s), s); row = dict(scene=s); fact = dict(scene=s)
        for arm in ARMS:
            run = run_dir(arm, s); blk, keep = BR.arm_block(run, s); fl, mv = BR.stall(blk["longest_slow_s"], blk["slow_t_start"], blk["slow_t_end"], keep014)
            ex, ticks = extra(run, keep, blk, arm, s); m1 = RO.mode1_fields(keep["sc"], keep["ts"], run); mr = BR.momentum_rows(keep["sc"], s); post = [r for r in mr if r["post_ho"]]
            if len(ticks) and not a.print_only: ticks.to_csv(T / "ticks" / f"{arm}_{s}.csv", index=False)
            row.update({f"{arm}_{k}": v for k, v in (blk | ex | m1).items()}); row[f"{arm}_stall"] = fl; row[f"{arm}_stall_014_mean_v_then"] = mv
            row[f"{arm}_momentum_changed_n"] = int(sum(r["changed"] for r in mr)); row[f"{arm}_momentum_changed_post_ho"] = int(sum(r["changed"] for r in post)); row[f"{arm}_n_replans"] = len(mr)
            if arm == "016":
                mom += mr; log = (run / "run_ext20_scene.log").read_text(errors="replace"); drv = (run / "driver.log").read_text(errors="replace"); rc = re.findall(r"RUN END rc=(\d+)", log)
                gates = json.load(open(run / "ext20_gates.json")) if (run / "ext20_gates.json").is_file() else {}
                row.update(wizard_rc=int(rc[-1]) if rc else None, gates_ok=gates.get("ok"), gates_failed=";".join(k for k, g in gates.get("gates", {}).items() if not g["ok"]),
                           prov_version=bool(re.search(r"adapter_version=0\.1\.6 ", drv)), prov_planner=("adapters/drivejepa016/planner.py" in drv), prov_line=("drivejepa-0.1.6 | S4 pass-9 weights" in drv and "switch=on" in drv),
                           codes=json.dumps(pd.Series([r["code"] for r in mr]).value_counts().to_dict()), fields_ok=int(sum(r["fields_ok"] for r in mr)))
            fact.update({f"{arm}_exit": blk["exit_kind"], f"{arm}_exit_s": blk["exit_s"], f"{arm}_first_coll_s": round(ex["first_collision_any_s"], 2), f"{arm}_abs_lat_at_exit": round(ex["abs_lat_at_exit"], 2), f"{arm}_max_abs_lat": round(ex["max_abs_lat"], 2),
                         f"{arm}_v_mean": round(ex["mean_speed_scored"], 2), f"{arm}_min_v": round(float(keep["V"][keep["ts"] >= keep["ho"]].min()), 2), f"{arm}_peak_steer": round(ex["peak_steer"], 3), f"{arm}_peak_steer_t": round(ex["peak_steer_t"], 2),
                         f"{arm}_progress": round(float(blk["progress_clipped_rel"]), 3), f"{arm}_dist": round(float(blk["dist_traveled_m"]), 1)})
        fact.update(score_014b=row["014b_scene_score"], score_016=row["016_scene_score"]); per.append(row); facts.append(fact)
        print(f"{s}: rc {row['wizard_rc']} gates {row['gates_ok']} | score 014b {row['014b_scene_score']:.2f} -> 016 {row['016_scene_score']:.2f} | progress {row['014b_progress_clipped_rel']:.2f} -> {row['016_progress_clipped_rel']:.2f} | "
              f"at-fault {int(row['014b_collision_at_fault'] > 0)}/{int(row['016_collision_at_fault'] > 0)} offroad {int(row['014b_offroad'] > 0)}/{int(row['016_offroad'] > 0)} | peak steer {row['014b_peak_steer']:.2f}/{row['016_peak_steer']:.2f} | §3.5 changed {row['014b_momentum_changed_n']}/{row['016_momentum_changed_n']}", flush=True)
    if a.print_only: return
    d = pd.DataFrame(per); d.to_csv(T / "per_scene.csv", index=False); pd.DataFrame(mom).to_csv(T / "replans_016.csv", index=False); pd.DataFrame(facts).to_csv(T / "insight_facts.csv", index=False); print("skipped (no 0.1.6 run yet):", skipped)
    if not len(d): return
    tot = dict(n=len(d), skipped=skipped)
    for arm in ARMS:
        tot[arm] = dict(mean_score=float(d[f"{arm}_scene_score"].mean()), at_fault=int((d[f"{arm}_collision_at_fault"] > 0).sum()), at_fault_scenes=sorted(d[d[f"{arm}_collision_at_fault"] > 0].scene), any_collision=int((d[f"{arm}_collision_any"] > 0).sum()),
                        offroad=int((d[f"{arm}_offroad"] > 0).sum()), offroad_scenes=sorted(d[d[f"{arm}_offroad"] > 0].scene), stalls=int(d[f"{arm}_stall"].sum()), stall_scenes=sorted(d[d[f"{arm}_stall"]].scene),
                        mean_progress_clipped_rel=float(d[f"{arm}_progress_clipped_rel"].mean()), mean_progress=float(d[f"{arm}_progress"].mean()), mean_dist_traveled=float(d[f"{arm}_dist_traveled_m"].mean()), full_score=sorted(d[d[f"{arm}_scene_score"] >= 0.999].scene))
    dl = d["016_scene_score"] - d["014b_scene_score"]
    tot["016_vs_014b"] = dict(improved=sorted(d[dl >= 0.02].scene), decreased=sorted(d[dl <= -0.02].scene), unchanged=sorted(d[dl.abs() < 0.02].scene), mean_delta=float(dl.mean()),
                              new_at_fault=sorted(d[(d["016_collision_at_fault"] > 0) & (d["014b_collision_at_fault"] == 0)].scene), removed_at_fault=sorted(d[(d["016_collision_at_fault"] == 0) & (d["014b_collision_at_fault"] > 0)].scene))
    tot["runs"] = dict(complete=int((d.wizard_rc == 0).sum()), gates_ok=int(d.gates_ok.fillna(False).astype(bool).sum()), provenance_ok=int((d.prov_version & d.prov_planner & d.prov_line & (d["016_egodriver"] == "drivejepa-v2-driver-0.1.6")).sum()), nre=sorted(d["016_nre"].astype(str).unique()))
    json.dump(tot, open(T / "totals.json", "w"), indent=1); print(json.dumps(tot, indent=1))

if __name__ == "__main__":
    main()
