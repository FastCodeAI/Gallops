#!/usr/bin/env python
"""S9 part C.3: merge the pull-out dump shards into one file with a manifest (NAS), check the alignment of the map-free scene against the
stored human path, and report the share of pull-out menus on which a plan leaves and passes collision (definitions: PREDICTIONS.md, P-C).
Map-free grade = collision gate x (5 x time buffer + 2 x comfort) / 7. Leave = the plan ends >= 2 m to the side or >= 5 m forward.
Usage: s9_c_analyze.py [--tag _timing]"""
import sys, glob, hashlib, pathlib, numpy as np, pandas as pd
from s7_common import boot, RD
HERE = pathlib.Path(__file__).resolve().parent; TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""; NAS = pathlib.Path("{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/pullout"); fs = sorted(f for f in glob.glob(str(NAS / ("dump" + TAG) / "shard_*.npz")) if ".tmp." not in f)
Z = [np.load(f, allow_pickle=True) for f in fs]; D = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}; n = len(D["token"]); print(len(fs), "shards,", n, "examples,", len(set(D["clip"])), "clips")
for s in ("strict", "lenient"): D[f"mapfree_grade_{s}"] = (D[f"nc_{s}"] * (5 * D[f"ttc_{s}"] + 2 * D["hc"]) / 7).astype(np.float32)
if not TAG:
    np.savez(NAS / "pullout_all.npz", **D); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rows = []; o = 0
    for f, z in zip(fs, Z):
        for r, t in enumerate(z["token"]): rows.append(dict(sample_id=t, clip_uuid=D["clip"][o], anchor_t=float(D["anchor_t"][o]), shard=pathlib.Path(f).name, row=r, row_in_pullout_all=o)); o += 1
    pd.DataFrame(rows).to_csv(NAS / "pullout_manifest.csv", index=False)
    with open(NAS / "files.sha256", "w") as fh:
        for p in [NAS / "pullout_all.npz", NAS / "pullout_manifest.csv"] + [pathlib.Path(f) for f in fs]: fh.write(f"{sha(p)}  {p.relative_to(NAS)}\n")
    (HERE / "c_pullout_files.sha256").write_text((NAS / "files.sha256").read_text())
clip = D["clip"]; P = D["menu_9"]; x, y = P[:, :, -1, 0], P[:, :, -1, 1]; leave = (np.abs(y) >= 2.0) | (x >= 5.0); hx, hy = D["human"][:, -1, 0], D["human"][:, -1, 1]; human_leaves = (np.abs(hy) >= 2.0) | (hx >= 5.0)
mo = pd.read_csv(RD / "01_find_clips/r3_raw_moveoffs.csv"); first = mo.groupby("uuid").t_move.min(); tmove = np.array([first.get(c, np.nan) for c in clip]); sp = D["car_speed"].astype(float); score = 1 / (1 + np.exp(-D["logit_9"][..., -1].astype(np.float64))); pick = score.argmax(1); ar = np.arange(n)
# ---- alignment and sanity of the map-free scene
dev = np.linalg.norm(D["human"][:, :, :2] - D["human_from_ego"][:, :, :2], axis=-1).max(1); A = dict(examples=n, clips=len(set(clip)), human_rebuilt_vs_stored_max_dev_m_median=float(np.median(dev)), p99=float(np.percentile(dev, 99)), max=float(dev.max()), share_over_0p5m=float((dev > 0.5).mean()),
    real_vehicle_box_share=float(D["real_vehicle_box"].mean()), has_past_share=float(D["has_past"].mean()), full_4s_future_share=float((D["future_frames"] >= 40).mean()), boxes_at_anchor_median=float(np.median(D["boxes_at_anchor"])), examples_with_no_box=float((D["boxes_at_anchor"] == 0).mean()),
    human_passes_collision_strict=float((D["human_nc_strict"] >= 1).mean()), human_passes_collision_lenient=float((D["human_nc_lenient"] >= 1).mean()), human_time_buffer_strict=float(D["human_ttc_strict"].mean()), human_time_buffer_lenient=float(D["human_ttc_lenient"].mean()), human_comfort=float(D["human_hc"].mean()),
    car_under_0p5_mps_share=float((sp < 0.5).mean()), before_first_moveoff_share=float((D["anchor_t"] < tmove).mean()), clips_with_a_moveoff_time=int(pd.Series(clip).drop_duplicates().isin(first.index).sum()), human_leaves_within_4s_share=float(human_leaves.mean()))
pd.Series(A).to_csv(HERE / f"c_alignment{TAG}.csv", header=["value"]); print(pd.Series(A).round(4).to_string())
# ---- per plan
PP = []
for lab, k in (("all menus", np.ones(n, bool)), ("car under 0.5 m/s", sp < 0.5), ("before the clip's first move-off", D["anchor_t"] < tmove), ("car at 0.5 m/s or more", sp >= 0.5)):
    if not k.any(): continue
    r = dict(subset=lab, menus=int(k.sum()), clips=int(len(set(clip[k]))), plans_that_leave=float(leave[k].mean()), leave_plans_per_menu_median=float(np.median(leave[k].sum(1))))
    for s in ("strict", "lenient"):
        nc = D[f"nc_{s}"][k]; r[f"plans_passing_collision_{s}"] = float((nc >= 1).mean()); r[f"plans_collision_0p5_{s}"] = float((nc == 0.5).mean()); r[f"plans_collision_0_{s}"] = float((nc == 0).mean()); r[f"leave_plans_passing_collision_{s}"] = float((nc >= 1)[leave[k]].mean()) if leave[k].any() else np.nan
        r[f"stay_plans_passing_collision_{s}"] = float((nc >= 1)[~leave[k]].mean()) if (~leave[k]).any() else np.nan; r[f"plans_time_buffer_ok_{s}"] = float(D[f"ttc_{s}"][k].mean()); r[f"mapfree_grade_mean_{s}"] = float(D[f"mapfree_grade_{s}"][k].mean())
    r["plans_comfort_ok"] = float(D["hc"][k].mean()); PP.append(r)
pd.DataFrame(PP).to_csv(HERE / f"c_per_plan{TAG}.csv", index=False); pd.set_option("display.width", 300, "display.max_rows", 300); print(pd.DataFrame(PP).set_index("subset").T.round(4).to_string())
# ---- per menu: a plan leaves and passes collision
MM = []; PER = dict(token=D["token"], clip=clip, anchor_t=D["anchor_t"], car_speed=sp, before_first_moveoff=D["anchor_t"] < tmove, leave_plans=leave.sum(1), human_leaves=human_leaves, scorer_pick_leaves=leave[ar, pick], scorer_pick_end_x=x[ar, pick])
for s in ("strict", "lenient"):
    ok = leave & (D[f"nc_{s}"] >= 1); PER[f"a_plan_leaves_and_passes_collision_{s}"] = ok.any(1); PER[f"leave_plans_passing_collision_{s}"] = ok.sum(1); PER[f"a_plan_leaves_with_mapfree_grade_1_{s}"] = (leave & (D[f"mapfree_grade_{s}"] >= 1 - 1e-6)).any(1)
    PER[f"scorer_pick_passes_collision_{s}"] = D[f"nc_{s}"][ar, pick] >= 1; bl = np.where(leave, score, -1).argmax(1); PER[f"best_scored_leave_plan_passes_collision_{s}"] = np.where(leave.any(1), D[f"nc_{s}"][ar, bl] >= 1, False)
per = pd.DataFrame(PER); per.to_parquet(HERE / f"c_per_menu{TAG}.parquet")
for lab, k in (("all menus", np.ones(n, bool)), ("car under 0.5 m/s", sp < 0.5), ("before the clip's first move-off", (D["anchor_t"] < tmove)), ("car under 0.5 m/s and the human leaves within 4 s", (sp < 0.5) & human_leaves), ("car at 0.5 m/s or more", sp >= 0.5)):
    if not k.any(): continue
    q = per[k]; cl = q["clip"].to_numpy(); r = dict(subset=lab, menus=int(k.sum()), clips=int(q["clip"].nunique()))
    for c in ["a_plan_leaves_and_passes_collision_strict", "a_plan_leaves_and_passes_collision_lenient", "a_plan_leaves_with_mapfree_grade_1_strict", "a_plan_leaves_with_mapfree_grade_1_lenient", "best_scored_leave_plan_passes_collision_strict", "scorer_pick_leaves", "scorer_pick_passes_collision_strict", "human_leaves"]:
        r[c], r[c + "_lo"], r[c + "_hi"] = boot(cl, {"x": q[c].to_numpy(float)}, lambda z: z["x"].mean())
    r["a_plan_leaves"] = float((q.leave_plans > 0).mean()); r["leave_plans_passing_collision_strict_median"] = float(q.leave_plans_passing_collision_strict.median())
    bc = q.groupby("clip").a_plan_leaves_and_passes_collision_strict.mean(); r["by_clip_median"] = float(bc.median()); r["by_clip_p10"] = float(bc.quantile(0.1)); r["clips_at_100pct"] = float((bc >= 1).mean()); r["clips_at_0pct"] = float((bc <= 0).mean()); r["clips_below_60pct"] = float((bc < 0.6).mean()); MM.append(r)
pd.DataFrame(MM).to_csv(HERE / f"c_per_menu_summary{TAG}.csv", index=False); print(pd.DataFrame(MM).set_index("subset").T.round(4).to_string())
per.groupby("clip").agg(menus=("token", "size"), stationary_menus=("car_speed", lambda v: int((v < 0.5).sum())), a_plan_leaves_and_passes_collision_strict=("a_plan_leaves_and_passes_collision_strict", "mean"), a_plan_leaves_and_passes_collision_lenient=("a_plan_leaves_and_passes_collision_lenient", "mean"),
                            scorer_pick_leaves=("scorer_pick_leaves", "mean"), human_leaves=("human_leaves", "mean")).round(4).to_csv(HERE / f"c_by_clip{TAG}.csv")
m = MM[0]; print("P-C first half: a leaving plan passes collision (strict) on %.4f (%.4f-%.4f) of all pull-out menus" % (m["a_plan_leaves_and_passes_collision_strict"], m["a_plan_leaves_and_passes_collision_strict_lo"], m["a_plan_leaves_and_passes_collision_strict_hi"]))
