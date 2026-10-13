#!/usr/bin/env python
"""S7 H1: highway instants, pass-9 menu. What is wrong with the menu where no plan grades >= 0.95 (definitions: PREDICTIONS.md, P-H1).
Reads S6's dumps and S3's stored catalogue grades; no model run. Writes h1_instants.parquet (one row per highway instant) and h1_*.csv."""
import sys, pathlib, numpy as np, pandas as pd
from s7_common import *
from navsim.agents.drive_jepa_perception_based.score_module import compute_navsim_score as CNS
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex as W
C = pathlib.Path("{runs}/drivejepa_retrain/04_training_set/s1/cache"); VOC = np.load(RD / "04_training_set/vocab_10240.npy")[:, 4::5]; VSP = step_speeds(VOC); VEND = VOC[:, -1, :2].astype(np.float64)
wa = np.asarray(CNS.config.weighted_metrics_array, float); WT = np.array([wa[W.PROGRESS], wa[W.TTC], wa[W.LANE_KEEPING], wa[W.HISTORY_COMFORT]]); print("weights EP, TTC, LK, HC:", WT, flush=True)
total = lambda c: c[..., :4].prod(-1) * (c[..., 4:8] * WT).sum(-1) / WT.sum()
def decel(v0, sp):
    a1 = (v0 - sp[..., 0]) / 0.5; a2 = (sp[..., 0] - sp[..., 1]) / 0.5; return np.maximum(0, np.maximum(a1, a2)), np.maximum(0, a1)
ROWS = []; TABLES_ONLY = "--tables-only" in sys.argv   # reuse h1_instants.parquet (written before the tables) instead of recomputing it
for name in (() if TABLES_ONLY else ("unseen", "seen")):
    D = load(name, ["human", "menu_9", "grade_9", "maxspeed_9"]); g = D["grade_9"].astype(np.float64); gt = g[..., -1]
    err = np.abs(total(g) - gt); print(name.upper(), "formula vs stored total, all plans: max abs error %.6f, share within 1e-4 %.5f" % (err.max(), (err < 1e-4).mean()), flush=True)
    hw = np.where(D["slice"] == "highway")[0]; hum = D["human"]; hsp = step_speeds(hum)
    for n_done, i in enumerate(hw):
        t = D["token"][i]; G = gt[i]; d4 = np.linalg.norm(D["menu_9"][i][:, -1, :2] - hum[i][-1, :2], axis=1); b = int(np.where(G >= G.max() - 1e-9, d4, np.inf).argmin()); c = g[i, b, :8]
        gains = np.array([total(np.where(np.arange(8) == j, 1.0, c)) for j in range(8)]) - G[b]; top = np.where(gains >= gains.max() - 1e-6)[0]
        if gains.max() <= 1e-6: lim = "none (grade is 1)" if G[b] >= 1 - 1e-6 else "more than one gate failed"
        elif len(top) == 1: lim = TERMS[top[0]]
        else: lim = "tie: " + "+".join(TERMS[j] for j in top)
        sp = step_speeds(D["menu_9"][i][b]); d1, ds = decel(D["speed"][i], sp)
        r = dict(set=name.upper(), token=t, clip=D["clip"][i], group="good plan on menu" if G.max() >= 0.95 else "no good plan", best_grade=G[b], best_slot=b, limiting=lim, limiting_gain=float(gains.max()), lim_is_comfort_or_ttc=bool(gains.max() > 1e-6 and all(TERMS[j] in ("HC", "TTC") for j in top)),
                 **{f"best_{k}": c[j] for j, k in enumerate(TERMS[:8])}, v0=D["speed"][i], menu_vmax=sp.max(), menu_vend=sp[-1], menu_vmax_simulated=float(D["maxspeed_9"][i, b]), menu_end_m=float(np.linalg.norm(D["menu_9"][i][b, -1, :2])), menu_decel1s=float(d1), menu_decel_step1=float(ds),
                 human_vmax=hsp[i].max(), human_vend=hsp[i, -1], human_end_m=float(np.linalg.norm(hum[i][-1, :2])), menu_plans_ge095=int((G >= 0.95).sum()), menu_vmax_fastest_of_32=float(step_speeds(D["menu_9"][i]).max()))
        fin = np.load(C / D["clip"][i] / t / "anchors_scores_final.npy").astype(np.float64); ok = np.where(np.isfinite(fin) & (fin >= 0.95))[0]; r["cat_curves_ge095"] = len(ok); r["cat_curves_graded"] = int(np.isfinite(fin).sum())
        if len(ok):
            dh = np.linalg.norm(VEND[ok] - hum[i][-1, :2], axis=1); k = int(ok[np.where(fin[ok] >= fin[ok].max() - 1e-9, dh, np.inf).argmin()]); d1c, dsc = decel(D["speed"][i], VSP[k])
            r.update(cat_best_curve=k, cat_best_grade=fin[k], cat_vmax=VSP[k].max(), cat_vend=VSP[k, -1], cat_end_m=float(np.linalg.norm(VEND[k])), cat_decel1s=float(d1c), cat_decel_step1=float(dsc), cat_vmax_median_of_good=float(np.median(VSP[ok].max(1))), cat_vend_median_of_good=float(np.median(VSP[ok][:, -1])),
                     cat_decel1s_median_of_good=float(np.median(decel(D["speed"][i], VSP[ok])[0])), menu_to_cat_best_dist_m=float(np.linalg.norm(D["menu_9"][i][b, :, :2] - VOC[k, :, :2], axis=-1).mean()))
        ROWS.append(r)
        if n_done % 500 == 0: print(name, n_done, "of", len(hw), flush=True)
d = pd.read_parquet(HERE / "h1_instants.parquet") if TABLES_ONLY else pd.DataFrame(ROWS)
if not TABLES_ONLY: d.to_parquet(HERE / "h1_instants.parquet"); print("rows", len(d), flush=True)

# ---- tables
def rows_for(x, cols):
    return {k: x[k].to_numpy(float) for k in cols}
T1 = []; T2 = []; T3 = []; T4 = []; PH = []
for s in ("UNSEEN", "SEEN"):
    ds = d[d.set == s]; v, lo, hi = boot(ds["clip"].to_numpy(), {"x": (ds.group == "no good plan").to_numpy(float)}, lambda c: c["x"].mean())
    T1.append(dict(set=s, highway_instants=len(ds), clips=ds["clip"].nunique(), no_good_plan=int((ds.group == "no good plan").sum()), share=v, lo=lo, hi=hi, catalogue_has_ge095_share=float((ds.cat_curves_ge095 > 0).mean()),
                   of_no_good_plan_catalogue_has_ge095=float((ds[ds.group == "no good plan"].cat_curves_ge095 > 0).mean())))
    for grp in ("no good plan", "good plan on menu"):
        x = ds[ds.group == grp]; cl = x["clip"].to_numpy(); n = len(x)
        if not n: continue
        # H1.2 limiting check
        for lim, cnt in x.limiting.value_counts().items():
            v, lo, hi = boot(cl, {"x": (x.limiting == lim).to_numpy(float)}, lambda c: c["x"].mean()); T2.append(dict(set=s, group=grp, instants=n, clips=x["clip"].nunique(), limiting=NAMES.get(lim, lim), n=int(cnt), share=v, lo=lo, hi=hi, mean_gain=float(x[x.limiting == lim].limiting_gain.mean())))
        r = dict(set=s, group=grp, instants=n, best_grade_mean=float(x.best_grade.mean()))
        for k in TERMS[:8]: r[f"best_plan_{k}_mean"] = float(x[f"best_{k}"].mean()); r[f"best_plan_{k}_below_1_share"] = float((x[f"best_{k}"] < 1 - 1e-9).mean())
        T2.append(dict(r, limiting="(values of the best plan's checks)"))
        # H1.3 speeds
        h = x[x.cat_curves_ge095 > 0]; r = dict(set=s, group=grp, instants=n, with_catalogue_curve_ge095=len(h))
        defs = dict(menu_over_cat_vmax=(h, "menu_vmax", "cat_vmax"), menu_over_cat_vend=(h, "menu_vend", "cat_vend"), menu_over_catmedian_vmax=(h, "menu_vmax", "cat_vmax_median_of_good"), menu_over_human_vend=(x, "menu_vend", "human_vend"), menu_over_human_vmax=(x, "menu_vmax", "human_vmax"), cat_over_human_vend=(h, "cat_vend", "human_vend"))
        for k, (q, a, b) in defs.items():
            ok = (q[b] > 0.5).to_numpy(); ra = (q[a] / q[b]).to_numpy()[ok]; c2 = q["clip"].to_numpy()[ok]; r[k + "_n"] = int(ok.sum()); r[k + "_excluded_denominator_below_0p5"] = int((~ok).sum())
            if ok.sum() == 0: continue
            r[k + "_mean"], r[k + "_mean_lo"], r[k + "_mean_hi"] = boot(c2, {"x": ra}, lambda c: c["x"].mean()); r[k + "_median"], r[k + "_median_lo"], r[k + "_median_hi"] = boot(c2, {"x": ra}, lambda c: np.median(c["x"]))
            for p in (10, 25, 75, 90): r[f"{k}_p{p}"] = float(np.percentile(ra, p))
            r[k + "_share_above_1p15"] = float((ra > 1.15).mean()); r[k + "_share_below_0p85"] = float((ra < 0.85).mean())
        for k in ("menu_vmax", "menu_vend", "cat_vmax", "cat_vend", "cat_vmax_median_of_good", "human_vmax", "human_vend", "v0", "menu_vmax_simulated", "menu_vmax_fastest_of_32"): r[k + "_mean_mps"] = float((h if k.startswith("cat") else x)[k].mean())
        if len(h):
            for k, f in dict(menu_faster_than_cat_vmax=lambda q: (q.menu_vmax > q.cat_vmax), menu_slower_than_cat_vmax=lambda q: (q.menu_vmax < q.cat_vmax), menu_faster_by_15pct=lambda q: (q.menu_vmax > 1.15 * q.cat_vmax), menu_slower_by_15pct=lambda q: (q.menu_vmax < 0.85 * q.cat_vmax),
                             menu_faster_than_cat_vend=lambda q: (q.menu_vend > q.cat_vend), menu_faster_by_15pct_vs_catmedian=lambda q: (q.menu_vmax > 1.15 * q.cat_vmax_median_of_good)).items():
                r[k], r[k + "_lo"], r[k + "_hi"] = boot(h["clip"].to_numpy(), {"x": f(h).to_numpy(float)}, lambda c: c["x"].mean())
        T3.append(r)
        # H1.4 handover deceleration
        r = dict(set=s, group=grp, instants=n, with_catalogue_curve_ge095=len(h))
        for k, q in (("menu_decel1s", x), ("menu_decel_step1", x), ("cat_decel1s", h), ("cat_decel_step1", h), ("cat_decel1s_median_of_good", h)):
            if not len(q): continue
            a = q[k].to_numpy(float); r[k + "_mean"], r[k + "_mean_lo"], r[k + "_mean_hi"] = boot(q["clip"].to_numpy(), {"x": a}, lambda c: c["x"].mean()); r[k + "_median"] = float(np.median(a)); r[k + "_p90"] = float(np.percentile(a, 90)); r[k + "_share_above_3"] = float((a > 3).mean()); r[k + "_share_above_6"] = float((a > 6).mean())
        if len(h): r["menu_minus_cat_decel1s_mean"], r["diff_lo"], r["diff_hi"] = boot(h["clip"].to_numpy(), {"x": (h.menu_decel1s - h.cat_decel1s).to_numpy(float)}, lambda c: c["x"].mean()); r["menu_decel_larger_share"] = float((h.menu_decel1s > h.cat_decel1s + 0.1).mean())
        T4.append(r)
        # P-H1 joint condition (denominator: every instant of the group; instants without a catalogue curve cannot satisfy it)
        fast = ((x.cat_curves_ge095 > 0) & (x.menu_vmax > 1.15 * x.cat_vmax)).to_numpy(); lim = x.lim_is_comfort_or_ttc.to_numpy(bool); has = (x.cat_curves_ge095 > 0).to_numpy()
        r = dict(set=s, group=grp, instants=n, clips=x["clip"].nunique())
        for k, a in dict(limiting_is_comfort_or_time_buffer=lim, menu_vmax_over_1p15_cat=fast, both=lim & fast, menu_slower_than_cat=((x.cat_curves_ge095 > 0) & (x.menu_vmax < x.cat_vmax)).to_numpy()).items(): r[k], r[k + "_lo"], r[k + "_hi"] = boot(cl, {"x": a.astype(float)}, lambda c: c["x"].mean())
        if has.any(): r["both_among_instants_with_catalogue_curve"], r["bh_lo"], r["bh_hi"] = boot(cl[has], {"x": (lim & fast)[has].astype(float)}, lambda c: c["x"].mean())
        fast2 = ((x.cat_curves_ge095 > 0) & (x.menu_vmax > 1.15 * x.cat_vmax_median_of_good)).to_numpy(); r["both_secondary_cat_median"], r["bs_lo"], r["bs_hi"] = boot(cl, {"x": (lim & fast2).astype(float)}, lambda c: c["x"].mean()); PH.append(r)
for nm, t in (("h1_1_counts", T1), ("h1_2_limiting_check", T2), ("h1_3_speed", T3), ("h1_4_handover_decel", T4), ("h1_prediction", PH)):
    o = pd.DataFrame(t); o.to_csv(HERE / f"{nm}.csv", index=False); pd.set_option("display.width", 250, "display.max_rows", 500); print("\n==", nm); print(o.round(3).T.to_string() if nm in ("h1_3_speed", "h1_4_handover_decel", "h1_prediction", "h1_1_counts") else o.round(3).dropna(axis=1, how="all").to_string(index=False))
