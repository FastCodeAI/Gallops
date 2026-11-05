#!/usr/bin/env python
"""S10 task 3: tables and plots from the arrays s10_diag3.py left in NAS/12_second_training_run/diag3. Usage: s10_diag3_report.py REPORT_DIR"""
import sys, json, pathlib, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = pathlib.Path("{runs}/drivejepa_retrain/12_second_training_run/diag3"); R = pathlib.Path(sys.argv[1]); SEED = 20261008
ho = pd.read_parquet(D / "ho_meta.parquet"); tg = np.load(D / "ho_targets.npy").astype(np.float64); N = len(ho); assert N == 4400
def load(p, bt, fx): return np.load(D / f"hrun_p{p}_{bt}_{fx.replace('+', '_')}.npz")
SL = {"highway, car >= 20 m/s": ((ho["slice"] == "highway") & (ho.speed >= 20)).to_numpy(), "highway": (ho["slice"] == "highway").to_numpy(), "close-pass": (ho["slice"] == "close-pass").to_numpy(), "ordinary": (ho["slice"] == "ordinary").to_numpy(), "all": np.ones(N, bool)}
def metrics(P):
    """P (N,32,8,3) final-stage plans -> per-example human term, swing (N,32), spacing term; as s10_gen_train.py:131-133 and drive_jepa_agent_s10.py spacing_rows"""
    P = P.astype(np.float64); hum = np.abs(P - tg[:, None]).sum(-1).mean(-1).min(1)
    xy = P[..., :2]; prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], -2); d = np.linalg.norm(xy - prev, axis=-1); sp = d / 0.5; sw = sp.max(-1) - sp.min(-1)
    s = d[..., 2:] - 2 * d[..., 1:-1] + d[..., :-2]; return hum, sw, (s ** 2).mean((-1, -2))
RUNS = [(0, "one", "nofix"), (0, "clip", "nofix"), (0, "shuffle", "nofix"), (9, "one", "nofix"), (9, "clip", "nofix"), (9, "shuffle", "nofix"), (9, "clip", "fix1+fix2"), (9, "shuffle", "fix1+fix2"), (9, "clip", "fix1+fix2+fix3"), (9, "shuffle", "fix1+fix2+fix3"), (0, "clip", "fix1+fix2+fix3"), (0, "shuffle", "fix1+fix2+fix3")]
BT = {"one": "batch of 1", "clip": "batch 32, clip order", "shuffle": "batch 32, shuffled"}; M = {}; rows = []
for p, bt, fx in RUNS:
    z = load(p, bt, fx); hum, sw, spc = metrics(z["plans"][:, 3]); M[(p, bt, fx)] = dict(z=z, hum=hum, sw=sw, spc=spc)
    for sl, k in SL.items(): rows.append(dict(checkpoint=f"pass {p}", batching=BT[bt], fix=fx, slice=sl, examples=int(k.sum()), human_path_term=hum[k].mean(), swing_median_all_32_plans_mps=np.median(sw[k]), share_plans_swing_above_5mps=(sw[k] > 5).mean(), spacing_sq_mean_m2=spc[k].mean(), spacing_sq_median_m2=np.median(spc[k])))
q3 = pd.DataFrame(rows); q3.to_csv(R / "q3_q6_readout_by_pass_batching_fix.csv", index=False)

# ---- Q6b / Q7: batch 32 against batch of 1
rows = []; gr = ho.has_label2.to_numpy()
for p, bt, fx in [(9, "clip", "nofix"), (9, "shuffle", "nofix"), (9, "clip", "fix1+fix2"), (9, "shuffle", "fix1+fix2"), (9, "clip", "fix1+fix2+fix3"), (9, "shuffle", "fix1+fix2+fix3"), (0, "clip", "nofix"), (0, "shuffle", "nofix"), (0, "clip", "fix1+fix2+fix3"), (0, "shuffle", "fix1+fix2+fix3")]:
    a = M[(p, bt, fx)]["z"]; b = M[(p, "one", "nofix")]["z"]; dp = np.abs(a["plans"][..., :2] - b["plans"][..., :2]).max((1, 2, 3, 4)); dl = np.abs(a["logits"] - b["logits"]).reshape(N, -1).max(1); ds = np.abs(a["scores"] - b["scores"]); pk = a["scores"].argmax(1) != b["scores"].argmax(1)
    for sl, k in SL.items():
        g = k & gr; rows.append(dict(checkpoint=f"pass {p}", batching=BT[bt] + " vs batch of 1", fix=fx, slice=sl, examples=int(k.sum()), plan_xy_diff_max_m=dp[k].max(), plan_xy_diff_median_over_rows_m=np.median(dp[k]), logit_diff_max=dl[k].max(), graded_examples=int(g.sum()),
                                     share_graded_examples_pick_changes=pk[g].mean(), score_abs_change_median=np.median(ds[g]), score_abs_change_mean=ds[g].mean(), score_abs_change_p90=np.quantile(ds[g], 0.9), per_example_max_score_change_median=np.median(ds[g].max(1))))
q7 = pd.DataFrame(rows); q7.to_csv(R / "q6_q7_batch32_vs_batch1.csv", index=False)

# ---- Q2: reference points outside 0..1 at batch 1 (pass 9)
z1 = M[(9, "one", "nofix")]["z"]; zs = M[(9, "shuffle", "nofix")]["z"]; hw = np.where(SL["highway, car >= 20 m/s"])[0]; od = np.sort(np.random.default_rng(SEED + 2).choice(np.where(SL["ordinary"])[0], 575, replace=False)); sel = np.concatenate([hw, od]); kind = np.array(["highway, car >= 20 m/s"] * 575 + ["ordinary"] * 575)
e = pd.DataFrame(dict(token=ho.sample_id.to_numpy()[sel], kind=kind, speed=ho.speed.to_numpy()[sel], swing_median_32_plans_batch1=np.median(M[(9, "one", "nofix")]["sw"][sel], 1), spacing_sq_batch1=M[(9, "one", "nofix")]["spc"][sel],
                       out_share_batch1=z1["own_out"][sel].mean((1, 2)), x_above_1_share_batch1=z1["own_x_hi"][sel].mean((1, 2)), x_below_0_share_batch1=z1["own_x_lo"][sel].mean((1, 2)), y_out_share_batch1=z1["own_y_out"][sel].mean((1, 2)),
                       attention_mass_on_zero_padding_batch1=z1["zero_mass"][sel].mean((1, 2)), sampling_points_fully_outside_batch1=z1["loc_zero"][sel].mean((1, 2)),
                       out_share_own_points_in_shuffled_batch=zs["own_out"][sel].mean((1, 2)), out_share_points_used_in_shuffled_batch_nofix=zs["used_out"][sel].mean((1, 2)), attention_mass_on_zero_padding_shuffled_nofix=zs["zero_mass"][sel].mean((1, 2)),
                       swing_median_32_plans_shuffled_nofix=np.median(M[(9, "shuffle", "nofix")]["sw"][sel], 1)))
for s in range(4): e[f"out_share_batch1_stage{s+1}"] = z1["own_out"][sel][:, s].mean(1)
e.to_csv(R / "q2_per_example.csv", index=False); rows = []
for kd, ii in (("highway, car >= 20 m/s", hw), ("ordinary", od)):
    for nm, zz, key in (("batch of 1: own reference points outside 0..1", z1, "own_out"), ("batch of 1: x above 1 (more than 32 m ahead)", z1, "own_x_hi"), ("batch of 1: x below 0", z1, "own_x_lo"), ("batch of 1: y outside 0..1", z1, "own_y_out"), ("batch of 1: attention mass on sampling points with zero return", z1, "zero_mass"),
                        ("batch 32 shuffled, no fix: points the block used for this row, outside 0..1", zs, "used_out"), ("batch 32 shuffled, no fix: the row's own points outside 0..1", zs, "own_out"), ("batch 32 shuffled, no fix: attention mass on sampling points with zero return", zs, "zero_mass")):
        for s in range(4): rows.append(dict(rows=kd, quantity=nm, stage=s + 1, **{f"waypoint_{w+1}": float(zz[key][ii][:, s, w].mean()) for w in range(8)}, all_waypoints=float(zz[key][ii][:, s].mean())))
q2 = pd.DataFrame(rows); q2.to_csv(R / "q2_out_of_range_by_stage_waypoint.csv", index=False); sp = []
for kd in ("highway, car >= 20 m/s", "ordinary", "both"):
    d = e if kd == "both" else e[e.kind == kd]
    for x in ("out_share_batch1", "attention_mass_on_zero_padding_batch1"):
        sp.append(dict(rows=kd, n=len(d), share=x, spearman_with_speed=d[x].corr(d.speed, method="spearman"), spearman_with_median_swing=d[x].corr(d.swing_median_32_plans_batch1, method="spearman"), share_mean=d[x].mean(), share_min=d[x].min(), share_max=d[x].max()))
sp = pd.DataFrame(sp); sp.to_csv(R / "q2_spearman.csv", index=False)
for f in ("q1_summary.csv", "q1_per_row.csv", "q1_extra.json", "q1_trace_row5_batch8_vs_alone.csv", "pass0_vs_shipped.json"): shutil.copy(D / f, R / f)

# ---- plots
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#e6e5df", "grid.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(8, 4.8))
for kd, c in (("ordinary", BLUE), ("highway, car >= 20 m/s", ORANGE)):
    d = e[e.kind == kd]; ax.scatter(d.speed, 100 * d.out_share_batch1, s=14, color=c, edgecolors="white", linewidths=0.4, alpha=0.85, label=f"{kd} (n = {len(d)})")
ax.set_xlabel("car speed, m/s"); ax.set_ylabel("plan reference points outside 0..1, %\n(mean over 4 stages x 32 plans x 8 waypoints)"); ax.set_ylim(-2, 102); ax.grid(True, axis="y"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Pass 9, batch of 1, SEEN held-out: share of self-attention reference points\noutside the 0..1 range (beyond +/-32 m), against car speed", loc="left", fontsize=11, color=INK); fig.tight_layout(); fig.savefig(R / "plot1_out_of_range_share_vs_speed.png", dpi=150); plt.close(fig)
pr = pd.read_csv(D / "q1_per_row.csv"); fx_l = ["nofix", "fix1", "fix1+fix2", "fix1+fix3", "fix1+fix2+fix3"]; bn_l = [("b_batch32_clip_order", "batch 32, clip order"), ("c_batch32_shuffled_with_others", "batch 32, shuffled with other rows"), ("d_batch32_reversed", "batch 32, clip order reversed")]
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True); rg = np.random.default_rng(0)
for ax, (bn, bt) in zip(axs, bn_l):
    for xi, fx in enumerate(fx_l):
        for kd, c, off in (("ordinary", BLUE, -0.18), ("highway >= 20 m/s", ORANGE, 0.18)):
            d = pr[(pr.fix == fx) & (pr.batching == bn) & (pr.kind == kd)]; ax.scatter(xi + off + rg.uniform(-0.13, 0.13, len(d)), np.maximum(d.plan_xy_max_abs_diff_m, 1e-7), s=12, color=c, edgecolors="white", linewidths=0.3, alpha=0.85, label=f"{kd} (64 rows)" if xi == 0 else None)
    ax.set_yscale("log"); ax.set_xticks(range(len(fx_l))); ax.set_xticklabels(["no fix", "fix1", "fix1+fix2", "fix1+fix3", "fix1+fix2\n+fix3"]); ax.axhline(1e-3, color=MUTED, lw=0.8, ls="--"); ax.grid(True, axis="y"); ax.set_title(bt, loc="left", fontsize=10, color=INK); ax.set_xlabel("correction applied")
axs[0].set_ylabel("largest plan difference to the batch-of-1 plans, m\n(per row, over 32 plans x 4 stages x 8 waypoints, x and y)"); axs[0].legend(frameon=False, loc="lower left"); axs[0].text(-0.4, 1.25e-3, "1 mm", color=MUTED, fontsize=8)
fig.suptitle("Pass 9, 128 SEEN held-out rows, CPU float32: per-row largest plan difference, batch 32 against batch of 1, by correction", x=0.01, ha="left", fontsize=12, color=INK); fig.tight_layout(); fig.savefig(R / "plot2_plan_diff_batch32_vs_batch1_by_fix.png", dpi=150); plt.close(fig)
pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
print("Q3/Q6 readout\n", q3[q3.slice != "all"].round(3).to_string()); print("Q6/Q7 vs batch 1\n", q7.round(5).to_string()); print("Q2 spearman\n", sp.round(3).to_string()); print("Q2 table\n", q2.round(3).to_string())
print("Q1\n", pd.read_csv(D / "q1_summary.csv").round(6).to_string()); print(open(D / "q1_extra.json").read())
t = pd.read_csv(D / "q1_trace_row5_batch8_vs_alone.csv"); print(t[(t.fix == "fix1+fix2") & t.module.str.contains("layers.0.attentions.0") & (t.call == 0)].to_string()); print(t[(t.fix == "fix1+fix3") & t.module.str.contains("layers.0.attentions") & (t.call == 0)].to_string())
