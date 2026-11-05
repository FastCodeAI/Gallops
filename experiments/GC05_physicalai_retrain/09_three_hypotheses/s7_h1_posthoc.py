#!/usr/bin/env python
"""S7 H1, ADDED AFTER THE PRE-REGISTERED TABLES WERE READ (not part of P-H1; descriptive). The pre-registered "max speed" (largest of the 8
half-second step speeds) came out at 52 m/s for best menu plans on no-good-plan highway instants while their end speed (27 m/s) and
the calculator's simulated max speed (32 m/s) did not. This looks at where along the plan the step speeds depart: the mean step-speed
profile of the best menu plan, the best catalogue curve and the human, which step holds the maximum, how uneven the steps are, and
P-H1's speed condition re-read with the calculator's simulated max speed. Reads h1_instants.parquet and S6's dumps."""
import numpy as np, pandas as pd
from s7_common import *
d = pd.read_parquet(HERE / "h1_instants.parquet"); VOC = np.load(RD / "04_training_set/vocab_10240.npy")[:, 4::5]; VSP = step_speeds(VOC); PROF = []; OUT = []
for name in ("unseen", "seen"):
    D = load(name, ["human", "menu_9"]); pos = {t: i for i, t in enumerate(D["token"])}; x = d[d.set == name.upper()].copy(); ii = np.array([pos[t] for t in x.token]); b = x.best_slot.to_numpy()
    sp = step_speeds(D["menu_9"][ii, b]); hs = step_speeds(D["human"][ii]); v0 = x.v0.to_numpy()[:, None]; has = (x.cat_curves_ge095 > 0).to_numpy(); cs = np.full_like(sp, np.nan); cs[has] = VSP[x.cat_best_curve.to_numpy()[has].astype(int)]
    x["menu_step_spread"] = sp.max(1) - sp.min(1); x["cat_step_spread"] = np.nanmax(cs, 1) - np.nanmin(cs, 1) if has.any() else np.nan; x["human_step_spread"] = hs.max(1) - hs.min(1); x["menu_argmax_step"] = sp.argmax(1) + 1; x["menu_max_jump_between_steps"] = np.abs(np.diff(sp, axis=1)).max(1)
    x["menu_lat_end_m"] = D["menu_9"][ii, b, -1, 1]; x["human_lat_end_m"] = D["human"][ii, -1, 1]
    for grp, m in (("no good plan", (x.group == "no good plan").to_numpy()), ("good plan on menu", (x.group != "no good plan").to_numpy())):
        for k in range(8): PROF.append(dict(set=name.upper(), group=grp, step=k + 1, t_s=0.5 * (k + 1), menu_minus_v0_mean=float((sp - v0)[m, k].mean()), menu_minus_v0_median=float(np.median((sp - v0)[m, k])), cat_minus_v0_mean=float(np.nanmean((cs - v0)[m, k])), human_minus_v0_mean=float((hs - v0)[m, k].mean()),
                                            menu_abs_dev_from_human_median=float(np.median(np.abs(sp - hs)[m, k])), cat_abs_dev_from_human_median=float(np.nanmedian(np.abs(cs - hs)[m, k])), share_menu_max_is_this_step=float((sp.argmax(1) == k)[m].mean())))
        q = x[m]; h = q[q.cat_curves_ge095 > 0]; cl = q["clip"].to_numpy(); r = dict(set=name.upper(), group=grp, instants=int(m.sum()))
        for k in ("menu_step_spread", "cat_step_spread", "human_step_spread", "menu_max_jump_between_steps"): r[k + "_median_mps"] = float(np.nanmedian(q[k])); r[k + "_mean_mps"] = float(np.nanmean(q[k]))
        r["menu_step_spread_over_5mps_share"], r["s5_lo"], r["s5_hi"] = boot(cl, {"x": (q.menu_step_spread > 5).to_numpy(float)}, lambda c: c["x"].mean()); r["cat_step_spread_over_5mps_share"] = float((h.cat_step_spread > 5).mean()); r["human_step_spread_over_5mps_share"] = float((q.human_step_spread > 5).mean())
        sim = ((q.cat_curves_ge095 > 0) & (q.menu_vmax_simulated > 1.15 * q.cat_vmax)).to_numpy(); lim = q.lim_is_comfort_or_ttc.to_numpy(bool)
        r["simulated_vmax_over_1p15_cat_share"], r["sv_lo"], r["sv_hi"] = boot(cl, {"x": sim.astype(float)}, lambda c: c["x"].mean()); r["both_with_simulated_vmax"], r["bsim_lo"], r["bsim_hi"] = boot(cl, {"x": (sim & lim).astype(float)}, lambda c: c["x"].mean())
        r["simulated_vmax_over_cat_vmax_median"] = float((h.menu_vmax_simulated / h.cat_vmax)[h.cat_vmax > 0.5].median()); r["end_speed_over_1p15_cat_share"] = float(((q.cat_curves_ge095 > 0) & (q.menu_vend > 1.15 * q.cat_vend)).mean())
        r["comfort_fails_share"] = float((q.best_HC < 1).mean()); r["lane_keeping_fails_share"] = float((q.best_LK < 1).mean()); r["progress_below_1_share"] = float((q.best_EP < 1 - 1e-9).mean()); r["time_buffer_fails_share"] = float((q.best_TTC < 1).mean())
        r["comfort_fails_and_spread_over_5"] = float(((q.best_HC < 1) & (q.menu_step_spread > 5)).mean()); r["of_comfort_fails_spread_over_5"] = float((q.menu_step_spread > 5)[q.best_HC < 1].mean()) if (q.best_HC < 1).any() else np.nan
        r["menu_end_within_1p5m_lateral_of_human_share"] = float(((q.menu_lat_end_m - q.human_lat_end_m).abs() <= 1.5).mean()); r["menu_plans_ge095_mean"] = float(q.menu_plans_ge095.mean()); OUT.append(r)
pd.DataFrame(PROF).to_csv(HERE / "h1_posthoc_step_profile.csv", index=False); pd.DataFrame(OUT).to_csv(HERE / "h1_posthoc_summary.csv", index=False); pd.set_option("display.width", 250, "display.max_rows", 300)
print(pd.DataFrame(OUT).round(3).T.to_string()); print(pd.DataFrame(PROF).round(2).to_string(index=False))
