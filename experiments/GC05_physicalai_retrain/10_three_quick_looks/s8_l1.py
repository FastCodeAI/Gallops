#!/usr/bin/env python
"""S8 L1: what the jagged highway plans look like, and whether the step-speed swing has structure (definitions: PREDICTIONS.md, P-L1).
Read-only on S6's UNSEEN dump, S7's h1_instants.parquet and the catalogue. No model run.
Three versions of one plan:
  raw8    the 8 waypoints as the generator emits them (menu_9; 0.5 s apart, car frame)
  interp  the calculator's input: car position prepended, linear interpolation to 41 poses at 0.1 s (navsim/evaluate/pdm_score.py:86-124)
  sim     the 41 simulated states the calculator graded (states_main from S6's dump; tracker + vehicle model), moved into the car frame
Writes l1_structure_per_instant.parquet, l1_structure.csv, l1_step_profile.csv, l1_sample_instants.csv, l1_sample_plans.csv, plots/*.png."""
import pathlib, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import s7_common as C7
from s7_common import load, boot, step_speeds, RD
HERE = pathlib.Path(__file__).resolve().parent; PL = HERE / "plots"; PL.mkdir(exist_ok=True); SEED = 20261009
VOC40 = np.load(RD / "04_training_set/vocab_10240.npy"); VOC = VOC40[:, 4::5]
D = load("unseen", ["human", "menu_9", "grade_9"]); n = len(D["token"]); ar = np.arange(n); gt = D["grade_9"][..., -1].astype(np.float64); hum = D["human"]
d4 = np.linalg.norm(D["menu_9"][:, :, -1, :2] - hum[:, None, -1, :2], axis=2); best = np.where(gt >= gt.max(1, keepdims=True) - 1e-9, d4, np.inf).argmin(1)           # S7's rule: highest grade, closest end to the human
h1 = pd.read_parquet(RD / "09_three_hypotheses/h1_instants.parquet"); h1 = h1[h1.set == "UNSEEN"].set_index("token"); pos = {t: i for i, t in enumerate(D["token"])}
assert all(best[pos[t]] == s for t, s in h1.best_slot.items()), "best plan differs from S7"
group = np.where(gt.max(1) >= 0.95, "good plan on menu", "no good plan")

def structure(sp):
    """sp (m, 8) step speeds -> dict of per-plan measures; lag-1 autocorrelation is NaN where the spread is under 0.5 m/s"""
    m = sp.mean(1, keepdims=True); dv = sp - m; den = (dv ** 2).sum(1); spread = sp.max(1) - sp.min(1); ok = spread >= 0.5
    r1 = np.where(ok, (dv[:, :-1] * dv[:, 1:]).sum(1) / np.where(den > 0, den, 1), np.nan); r2 = np.where(ok, (dv[:, :-2] * dv[:, 2:]).sum(1) / np.where(den > 0, den, 1), np.nan)
    df = np.diff(sp, axis=1); sc = np.where(ok, ((df[:, :-1] * df[:, 1:]) < 0).mean(1), np.nan); ad = np.abs(dv); first = np.where(ok, ad[:, :2].sum(1) / np.where(ad.sum(1) > 0, ad.sum(1), 1), np.nan)
    return dict(lag1=r1, lag2=r2, sign_changes=sc, odd_minus_even=sp[:, 0::2].mean(1) - sp[:, 1::2].mean(1), first_second_share=first, swing=spread, mean_speed=m[:, 0], swing_over_mean=spread / np.maximum(m[:, 0], 0.5), argmax_step=sp.argmax(1) + 1, argmin_step=sp.argmin(1) + 1)

SPM = step_speeds(D["menu_9"]); SPB = SPM[ar, best]; SPH = step_speeds(hum); per = pd.DataFrame(dict(token=D["token"], clip=D["clip"], slice=D["slice"], group=group, v0=D["speed"], best_slot=best, best_grade=gt[ar, best]))
for k, v in structure(SPB).items(): per["menu_" + k] = v
for k, v in structure(SPH).items(): per["human_" + k] = v
a32 = structure(SPM.reshape(-1, 8)); per["all32_lag1_median"] = np.nanmedian(a32["lag1"].reshape(n, 32), axis=1); per["all32_swing_median"] = np.median(a32["swing"].reshape(n, 32), axis=1); per["all32_share_lag1_below_m0p5"] = np.nanmean(a32["lag1"].reshape(n, 32) < -0.5, axis=1)
ci = np.full(n, -1); ci[[pos[t] for t in h1.index[h1.cat_curves_ge095 > 0]]] = h1.cat_best_curve[h1.cat_curves_ge095 > 0].astype(int).to_numpy(); hc = ci >= 0; SPC = np.full((n, 8), np.nan); SPC[hc] = step_speeds(VOC[ci[hc]])
for k, v in structure(SPC[hc]).items(): per["cat_" + k] = np.nan; per.loc[hc, "cat_" + k] = v
# the 41 simulated states, car frame (only the best plan of each instant is kept)
SIM = np.empty((n, 41, 3)); SIMV = np.empty((n, 41)); o = 0
for f in D["_files"]:
    st = np.load(f)["states_main"]; m = len(st); b = best[o:o + m]; s = st[np.arange(m), b]; x0, y0, h0 = s[:, 0, 0], s[:, 0, 1], s[:, 0, 2]; dx = s[:, :, 0] - x0[:, None]; dy = s[:, :, 1] - y0[:, None]; c, sn = np.cos(h0)[:, None], np.sin(h0)[:, None]
    SIM[o:o + m, :, 0] = c * dx + sn * dy; SIM[o:o + m, :, 1] = -sn * dx + c * dy; SIM[o:o + m, :, 2] = s[:, :, 2] - h0[:, None]; SIMV[o:o + m] = np.hypot(s[:, :, 3], s[:, :, 4]); o += m
per["sim_speed_swing"] = SIMV.max(1) - SIMV.min(1); per["sim_speed_start"] = SIMV[:, 0]; simstep = np.linalg.norm(np.diff(SIM[:, ::5, :2], axis=1), axis=-1) / 0.5
for k, v in structure(simstep).items(): per["sim_" + k] = v
per["sim_end_gap_to_raw_m"] = np.linalg.norm(SIM[:, -1, :2] - D["menu_9"][ar, best, -1, :2], axis=1); per["sim_accel_abs_max"] = np.abs(np.diff(SIMV, axis=1) / 0.1).max(1)
per.to_parquet(HERE / "l1_structure_per_instant.parquet")

# ---- L1.3 / L1.4 tables
ROWS = []; PROF = []
def add(label, x, sp_b):
    cl = x["clip"].to_numpy(); r = dict(group=label, instants=len(x), clips=x["clip"].nunique())
    for src in ("menu", "sim", "cat", "human"):
        q = x.dropna(subset=[src + "_swing"]) if src == "cat" else x; c2 = q["clip"].to_numpy(); l = q[src + "_lag1"].to_numpy(float); r[f"{src}_plans"] = len(q); r[f"{src}_left_out_spread_under_0p5"] = int(np.isnan(l).sum())
        if len(q) == 0 or np.isnan(l).all(): continue
        r[f"{src}_lag1_median"], r[f"{src}_lag1_median_lo"], r[f"{src}_lag1_median_hi"] = boot(c2, {"x": l}, lambda c: np.nanmedian(c["x"])); r[f"{src}_lag1_mean"] = float(np.nanmean(l)); r[f"{src}_lag1_p25"], r[f"{src}_lag1_p75"] = np.nanpercentile(l, [25, 75])
        r[f"{src}_share_lag1_below_m0p5"], r[f"{src}_s_lo"], r[f"{src}_s_hi"] = boot(c2, {"x": l}, lambda c: np.mean(c["x"][~np.isnan(c["x"])] < -0.5)); r[f"{src}_share_lag1_above_0"] = float(np.mean(l[~np.isnan(l)] > 0))
        r[f"{src}_lag2_median"] = float(np.nanmedian(q[src + "_lag2"])); r[f"{src}_sign_changes_mean"] = float(np.nanmean(q[src + "_sign_changes"])); r[f"{src}_odd_minus_even_mean"] = float(q[src + "_odd_minus_even"].mean()); r[f"{src}_odd_minus_even_abs_median"] = float(q[src + "_odd_minus_even"].abs().median())
        r[f"{src}_first_second_share_median"] = float(np.nanmedian(q[src + "_first_second_share"])); r[f"{src}_swing_median"], r[f"{src}_sw_lo"], r[f"{src}_sw_hi"] = boot(c2, {"x": q[src + "_swing"].to_numpy(float)}, lambda c: np.median(c["x"])); r[f"{src}_swing_over_mean_median"] = float(q[src + "_swing_over_mean"].median())
        r[f"{src}_swing_over_5_share"] = float((q[src + "_swing"] > 5).mean())
    sw = x.menu_swing.to_numpy(float); v0 = x.v0.to_numpy(float); r["spearman_swing_vs_car_speed"] = float(spearmanr(sw, v0)[0]); r["spearman_relative_swing_vs_car_speed"] = float(spearmanr(x.menu_swing_over_mean, v0)[0])
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 99)):
        k = (v0 >= lo) & (v0 < hi); r[f"swing_median_car_{lo}_{hi}_mps"] = float(np.median(sw[k])) if k.sum() >= 20 else np.nan; r[f"n_car_{lo}_{hi}_mps"] = int(k.sum())
    r["all32_lag1_median_of_medians"] = float(x.all32_lag1_median.median()); r["all32_swing_median_of_medians"] = float(x.all32_swing_median.median()); r["all32_share_lag1_below_m0p5_mean"] = float(x.all32_share_lag1_below_m0p5.mean())
    r["sim_speed_swing_median"] = float(x.sim_speed_swing.median()); r["sim_end_gap_to_raw_m_median"] = float(x.sim_end_gap_to_raw_m.median()); r["sim_accel_abs_max_median"] = float(x.sim_accel_abs_max.median())
    for kk in range(1, 9): r[f"share_fastest_step_{kk}"] = float((x.menu_argmax_step == kk).mean()); r[f"share_slowest_step_{kk}"] = float((x.menu_argmin_step == kk).mean())
    dv = sp_b - sp_b.mean(1, keepdims=True); mp = dv.mean(0); nz = (dv.std(1) > 0); cc = np.array([np.corrcoef(dv[i], mp)[0, 1] for i in np.where(nz)[0]]); r["corr_with_group_mean_profile_median"] = float(np.median(cc)); r["corr_with_group_mean_profile_share_above_0p5"] = float((cc > 0.5).mean()); ROWS.append(r)
    for kk in range(8): PROF.append(dict(group=label, step=kk + 1, t_s=0.5 * (kk + 1), menu_dev_from_plan_mean_mean=float(mp[kk]), menu_dev_median=float(np.median(dv[:, kk])), menu_dev_p25=float(np.percentile(dv[:, kk], 25)), menu_dev_p75=float(np.percentile(dv[:, kk], 75)), menu_abs_dev_median=float(np.median(np.abs(dv[:, kk])))))
for sl in ("highway", "ordinary", "close-pass"):
    for g in ("no good plan", "good plan on menu"):
        k = ((per.slice == sl) & (per.group == g)).to_numpy(); add(f"{sl}, {g}", per[k], SPB[k])
    k = (per.slice == sl).to_numpy(); add(f"{sl}, all", per[k], SPB[k])
S = pd.DataFrame(ROWS); S.to_csv(HERE / "l1_structure.csv", index=False); pd.DataFrame(PROF).to_csv(HERE / "l1_step_profile.csv", index=False); pd.set_option("display.width", 320, "display.max_rows", 500); print(S.set_index("group").T.round(3).to_string()); print(pd.DataFrame(PROF).round(2).to_string(index=False))

# ---- L1.1 sample: 20 + 20 UNSEEN highway instants, table and plots
rng = np.random.default_rng(SEED); hw = per[per.slice == "highway"]; samp = pd.concat([hw[hw.group == g].sort_values("token").iloc[np.sort(rng.choice((hw.group == g).sum(), 20, replace=False))] for g in ("no good plan", "good plan on menu")]); LONG = []; INST = []
tt8 = 0.5 * np.arange(1, 9); t41 = 0.1 * np.arange(41)
def panel_xy(ax, i, title=True):
    raw = D["menu_9"][i, best[i]]; src = np.vstack([[0, 0, 0], raw]); itp = np.stack([np.interp(t41, 0.5 * np.arange(9), src[:, c]) for c in range(2)], 1)
    if ci[i] >= 0: ax.plot(VOC40[ci[i], :, 1], VOC40[ci[i], :, 0], "-", color="tab:green", lw=1, alpha=.6); ax.plot(VOC[ci[i], :, 1], VOC[ci[i], :, 0], "s", color="tab:green", ms=5, mfc="none", label="best catalogue curve (8 poses graded; line = its 40 stored poses)")
    ax.plot(hum[i][:, 1], hum[i][:, 0], "x", color="0.4", ms=5, label="human (8 poses)"); ax.plot(itp[:, 1], itp[:, 0], "-", color="tab:orange", lw=1, label="calculator input: linear interpolation, 41 poses")
    ax.plot(SIM[i, :, 1], SIM[i, :, 0], "-", color="tab:blue", lw=1.6, label="graded: 41 simulated states"); ax.plot(SIM[i, 5::5, 1], SIM[i, 5::5, 0], "|", color="tab:blue", ms=9, mew=1.6, label="simulated position at each 0.5 s")
    ax.plot(raw[:, 1], raw[:, 0], "o", color="tab:red", ms=6, mfc="none", mew=1.6, label="raw generator output: 8 waypoints"); ax.invert_xaxis(); ax.set_xlabel("left  <-  y, m  ->  right"); ax.set_ylabel("forward x, m"); m = max(1.5, np.abs(ax.get_xlim()).max()); ax.set_xlim(m, -m); ax.grid(alpha=.3)
def panel_v(ax, i):
    raw = D["menu_9"][i, best[i]]; ax.step(np.r_[0, tt8], np.r_[SPB[i][0], SPB[i]], where="pre", color="tab:red", lw=1.6, label="raw waypoints: speed per 0.5 s step (= calculator input speed)")
    ax.plot(t41, SIMV[i], "-", color="tab:blue", lw=1.6, label="graded: simulated speed, 0.1 s")
    if ci[i] >= 0: ax.step(np.r_[0, tt8], np.r_[SPC[i][0], SPC[i]], where="pre", color="tab:green", lw=1.2, label="best catalogue curve, per step")
    ax.step(np.r_[0, tt8], np.r_[SPH[i][0], SPH[i]], where="pre", color="0.4", lw=1, ls="--", label="human, per step"); ax.axhline(D["speed"][i], color="k", lw=.6, ls=":", label="car speed now"); ax.set_xlabel("time, s"); ax.set_ylabel("speed, m/s"); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
for j, q in enumerate(samp.itertuples()):
    i = pos[q.token]; tag = "nogood" if q.group == "no good plan" else "good"; fn = f"{tag}_{j % 20 + 1:02d}_{q.token}.png"; fig, axs = plt.subplots(1, 2, figsize=(13, 5.2)); panel_xy(axs[0], i); panel_v(axs[1], i)
    g = D["grade_9"][i, best[i]]; fig.suptitle(f"{q.token}  |  UNSEEN highway, {q.group}  |  best menu plan: slot {best[i]}, grade {g[-1]:.3f} (comfort {g[7]:.0f}, lane keeping {g[6]:.0f}, progress {g[4]:.2f}, time buffer {g[5]:.0f})  |  step-speed swing {q.menu_swing:.1f} m/s, lag-1 {q.menu_lag1:.2f}", fontsize=9)
    axs[0].legend(fontsize=6.5, loc="best"); axs[1].legend(fontsize=6.5, loc="best"); fig.tight_layout(); fig.savefig(PL / fn, dpi=110); plt.close(fig)
    raw = D["menu_9"][i, best[i]]; src = np.vstack([[0, 0, 0], raw]); itp = np.stack([np.interp(t41, 0.5 * np.arange(9), src[:, c]) for c in range(3)], 1)
    INST.append(dict(token=q.token, clip=q.clip, group=q.group, plot=f"plots/{fn}", car_speed=q.v0, best_slot=int(best[i]), grade=float(g[-1]), **{C7.TERMS[k]: float(g[k]) for k in range(8)}, swing=q.menu_swing, lag1=q.menu_lag1, sign_changes=q.menu_sign_changes, fastest_step=int(q.menu_argmax_step), slowest_step=int(q.menu_argmin_step),
                     sim_speed_swing=q.sim_speed_swing, sim_end_gap_to_raw_m=q.sim_end_gap_to_raw_m, catalogue_curve=int(ci[i]), catalogue_swing=q.cat_swing, human_swing=q.human_swing, **{f"raw_step_speed_{k + 1}": float(SPB[i][k]) for k in range(8)}))
    for nm, P, t, v in (("raw8", raw, tt8, SPB[i]), ("interp41", itp, t41, np.r_[np.nan, np.linalg.norm(np.diff(itp[:, :2], axis=0), axis=1) / 0.1]), ("sim41", SIM[i], t41, SIMV[i]), ("human8", hum[i], tt8, SPH[i])) + ((("catalogue8", VOC[ci[i]], tt8, SPC[i]), ("catalogue40", VOC40[ci[i]], 0.1 * np.arange(1, 41), np.r_[np.nan, np.linalg.norm(np.diff(VOC40[ci[i], :, :2], axis=0), axis=1) / 0.1])) if ci[i] >= 0 else ()):
        for k in range(len(P)): LONG.append(dict(token=q.token, group=q.group, version=nm, k=k, t_s=round(float(t[k]), 2), x_m=float(P[k, 0]), y_m=float(P[k, 1]), heading_rad=float(P[k, 2]), speed_mps=float(v[k])))
pd.DataFrame(INST).to_csv(HERE / "l1_sample_instants.csv", index=False); pd.DataFrame(LONG).to_csv(HERE / "l1_sample_plans.csv", index=False)
for tag, g in (("nogood", "no good plan"), ("good", "good plan on menu")):
    for kind, fn_ in (("speed", panel_v), ("topdown", panel_xy)):
        fig, axs = plt.subplots(4, 5, figsize=(22, 14))
        for ax, q in zip(axs.ravel(), samp[samp.group == g].itertuples()): fn_(ax, pos[q.token]); ax.set_title(f"{q.token}  swing {q.menu_swing:.1f} m/s, lag-1 {q.menu_lag1:.2f}, grade {q.best_grade:.2f}", fontsize=8)
        h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=9); fig.suptitle(f"UNSEEN highway, {g}: best menu plan of 20 random instants ({kind})", fontsize=13); fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(PL / f"sheet_{tag}_{kind}.png", dpi=80); plt.close(fig)
# one figure for the structure question
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8)); bins = np.linspace(-1, 1, 41)
for lab, k, c in (("highway, no good plan", (per.slice == "highway") & (per.group == "no good plan"), "tab:red"), ("highway, good plan", (per.slice == "highway") & (per.group != "no good plan"), "tab:orange"), ("ordinary, all", per.slice == "ordinary", "tab:blue")):
    axs[0].hist(per[k].menu_lag1.dropna(), bins=bins, histtype="step", density=True, color=c, label=f"menu best plan: {lab}"); axs[2].scatter(per[k].v0, per[k].menu_swing, s=3, alpha=.25, color=c, label=lab)
axs[0].hist(per[per.slice == "highway"].cat_lag1.dropna(), bins=bins, histtype="step", density=True, color="tab:green", label="best catalogue curve: highway"); axs[0].axvline(-0.5, color="k", ls=":", lw=1); axs[0].set_xlabel("lag-1 autocorrelation of the 8 step speeds"); axs[0].set_ylabel("density"); axs[0].legend(fontsize=7)
pf = pd.DataFrame(PROF)
for lab, c in (("highway, no good plan", "tab:red"), ("highway, good plan on menu", "tab:orange"), ("ordinary, all", "tab:blue")):
    q = pf[pf["group"] == lab]; tq = q.t_s.to_numpy(float); axs[1].plot(tq, q.menu_dev_median.to_numpy(float), "o-", color=c, label=lab); axs[1].fill_between(tq, q.menu_dev_p25.to_numpy(float), q.menu_dev_p75.to_numpy(float), color=c, alpha=.15)
axs[1].axhline(0, color="k", lw=.6); axs[1].set_xlabel("step end time, s"); axs[1].set_ylabel("step speed minus the plan's mean, m/s (median, 25-75 %)"); axs[1].legend(fontsize=7); axs[2].set_xlabel("car speed now, m/s"); axs[2].set_ylabel("swing: fastest minus slowest step, m/s"); axs[2].set_ylim(0, 80); axs[2].legend(fontsize=7)
fig.suptitle("UNSEEN, best menu plan: is the step-speed swing structured?"); fig.tight_layout(); fig.savefig(PL / "structure_overview.png", dpi=110); plt.close(fig); print("plots:", len(list(PL.glob("*.png"))))
