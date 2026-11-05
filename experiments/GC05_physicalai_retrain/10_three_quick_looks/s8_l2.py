#!/usr/bin/env python
"""S8 L2: where the regret lives, UNSEEN, raw pick (definitions: PREDICTIONS.md, P-L2). Scorer = its own output (logit_9[..., -1]);
ranking head = S7's chosen head, picks read from 09_three_hypotheses/h3_unseen_per_instant.parquet; random = expected value over the 32 plans.
Read-only on S6's dump and S7's outputs. Writes l2_regret.csv, l2_gates.csv, l2_zero_picks_best_available.csv, l2_per_instant.parquet."""
import pathlib, numpy as np, pandas as pd
from s7_common import load, boot, RD, SL, TERMS
HERE = pathlib.Path(__file__).resolve().parent
D = load("unseen", ["grade_9", "logit_9"]); g = D["grade_9"].astype(np.float64); gt = g[..., -1]; n = len(gt); ar = np.arange(n); best = gt.max(1)
per = pd.read_parquet(RD / "09_three_hypotheses/h3_unseen_per_instant.parquet").set_index("token").loc[D["token"]]; rk = [c for c in per.columns if c.startswith("pick|") and "the ranking head" in c][0]; sk = [c for c in per.columns if c.startswith("pick|") and "scorer's own" in c][0]
ps = D["logit_9"][..., -1].argmax(1); assert (ps == per[sk].to_numpy()).all(), "scorer pick differs from S7"; pr = per[rk].to_numpy().astype(int); print("ranking head column:", rk, flush=True)
GATES = ["NC", "DAC", "TLC", "DDC"]; GN = dict(NC="collision", DAC="drivable area", TLC="traffic light", DDC="driving direction")
def onehot(p): w = np.zeros((n, 32)); w[ar, p] = 1.0; return w
W = {"scorer": onehot(ps), "ranking head": onehot(pr), "random pick (expected)": np.full((n, 32), 1 / 32)}
reg = best[:, None] - gt; zero = gt <= 1e-9; gz = g[..., :4] <= 1e-9                       # per plan
def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, q * c[-1])])
ROWS = []; GR = []; BA = []; PI = dict(token=D["token"], clip=D["clip"], slice=D["slice"], best_grade=best)
for name, w in W.items():
    PI[f"regret|{name}"] = (w * reg).sum(1); PI[f"pick_graded_0|{name}"] = (w * zero).sum(1)
    for sl in SL:
        k = np.ones(n, bool) if sl == "all" else D["slice"] == sl; cl = D["clip"][k]; wk, rg, zk, bk = w[k], reg[k], zero[k], best[k]; r = dict(pick=name, slice=sl, instants=int(k.sum()), clips=int(len(set(cl))))
        cols = dict(tot=(wk * rg).sum(1), z=(wk * zk).sum(1), zr=(wk * zk * rg).sum(1), b0=(wk * (rg <= 1e-9)).sum(1), b1=(wk * ((rg > 1e-9) & (rg <= 0.1))).sum(1), b2=(wk * ((rg > 0.1) & (rg <= 0.5))).sum(1), b3=(wk * (rg > 0.5)).sum(1),
                    r1=(wk * rg * ((rg > 1e-9) & (rg <= 0.1))).sum(1), r2=(wk * rg * ((rg > 0.1) & (rg <= 0.5))).sum(1), r3=(wk * rg * (rg > 0.5)).sum(1))
        for nm, f in dict(regret_mean=lambda c: c["tot"].mean(), share_regret_0=lambda c: c["b0"].mean(), share_regret_0_to_0p1=lambda c: c["b1"].mean(), share_regret_0p1_to_0p5=lambda c: c["b2"].mean(), share_regret_over_0p5=lambda c: c["b3"].mean(), share_picks_graded_0=lambda c: c["z"].mean(),
                          share_of_total_regret_from_picks_graded_0=lambda c: c["zr"].sum() / c["tot"].sum(), share_of_total_regret_from_band_0_to_0p1=lambda c: c["r1"].sum() / c["tot"].sum(), share_of_total_regret_from_band_0p1_to_0p5=lambda c: c["r2"].sum() / c["tot"].sum(),
                          share_of_total_regret_from_band_over_0p5=lambda c: c["r3"].sum() / c["tot"].sum(), regret_mean_without_picks_graded_0=lambda c: (c["tot"] - c["zr"]).sum() / np.maximum((1 - c["z"]).sum(), 1e-9)).items():
            r[nm], r[nm + "_lo"], r[nm + "_hi"] = boot(cl, cols, f)
        ww = wk.ravel(); xx = rg.ravel(); nzw = ww > 0
        for q in (50, 90, 99): r[f"regret_p{q}"] = wq(xx[nzw], ww[nzw], q / 100)
        ROWS.append(r)
        # which gate zeroed the picks graded 0 (weights = pick weights on zero-graded plans)
        wz = wk * zk; tot = wz.sum(); gzk = gz[k]; ng = gzk.sum(-1); row = dict(pick=name, slice=sl, picks_graded_0=float(tot), share_of_instants=float(tot / k.sum()))
        for j, a in enumerate(GATES): row[f"{GN[a]} at 0 (alone or with others)"] = float((wz * gzk[..., j]).sum() / tot) if tot else np.nan; row[f"{GN[a]} alone"] = float((wz * (gzk[..., j] & (ng == 1))).sum() / tot) if tot else np.nan
        row["more than one gate at 0"] = float((wz * (ng > 1)).sum() / tot) if tot else np.nan; row["no gate at 0"] = float((wz * (ng == 0)).sum() / tot) if tot else np.nan
        if tot and (wz * (ng == 0)).sum() > 0:
            m0 = (wz * (ng == 0)); row["no gate at 0: mean of progress, time buffer, lane keeping, comfort"] = " / ".join("%.2f" % ((m0 * g[k][..., j]).sum() / m0.sum()) for j in (4, 5, 6, 7)); row["no gate at 0: smallest gate product"] = float(np.where(m0 > 0, g[k][..., :4].prod(-1), np.inf).min())
        GR.append(row)
        # best available grade on instants where the pick is graded 0 (instant weight = probability that the pick is graded 0)
        wi = wz.sum(1); has = wi > 0
        if has.any():
            b = bk[has]; ww2 = wi[has]; s = ww2.sum(); ng9 = (g[k][..., -1] >= 0.9).sum(1)[has]; nz0 = zk.sum(1)[has]
            BA.append(dict(pick=name, slice=sl, instants_weighted=float(s), best_available_mean=float((ww2 * b).sum() / s), best_available_median=wq(b, ww2, 0.5), best_available_p10=wq(b, ww2, 0.1), share_best_ge_0p95=float((ww2 * (b >= 0.95)).sum() / s), share_best_ge_0p9=float((ww2 * (b >= 0.9)).sum() / s),
                           share_best_below_0p5=float((ww2 * (b < 0.5)).sum() / s), share_best_is_0_too=float((ww2 * (b <= 1e-9)).sum() / s), plans_graded_ge_0p9_on_menu_mean=float((ww2 * ng9).sum() / s), plans_graded_0_on_menu_mean=float((ww2 * nz0).sum() / s)))
        else: BA.append(dict(pick=name, slice=sl, instants_weighted=0.0))
pd.DataFrame(ROWS).to_csv(HERE / "l2_regret.csv", index=False); pd.DataFrame(GR).to_csv(HERE / "l2_gates.csv", index=False); pd.DataFrame(BA).to_csv(HERE / "l2_zero_picks_best_available.csv", index=False); pd.DataFrame(PI).to_parquet(HERE / "l2_per_instant.parquet")
# menu context: how many plans of a menu are graded 0, and instants where every plan is
MC = []
for sl in SL:
    k = np.ones(n, bool) if sl == "all" else D["slice"] == sl; MC.append(dict(slice=sl, instants=int(k.sum()), plans_graded_0_share=float(zero[k].mean()), menus_with_a_plan_graded_0=float(zero[k].any(1).mean()), menus_all_graded_0=float(zero[k].all(1).mean()), plans_graded_0_per_menu_mean=float(zero[k].sum(1).mean()),
                                                                           both_scorer_and_ranking_pick_graded_0=float((zero[ar, ps] & zero[ar, pr])[k].mean()), same_pick=float((ps == pr)[k].mean())))
pd.DataFrame(MC).to_csv(HERE / "l2_menu_context.csv", index=False); pd.set_option("display.width", 330, "display.max_rows", 500, "display.max_colwidth", 60)
o = pd.DataFrame(ROWS); print(o[[c for c in o.columns if not c.endswith(("_lo", "_hi"))]].round(4).T.to_string()); x = o[(o.slice == "all")]; print(x[["pick", "share_picks_graded_0", "share_picks_graded_0_lo", "share_picks_graded_0_hi", "share_of_total_regret_from_picks_graded_0", "share_of_total_regret_from_picks_graded_0_lo", "share_of_total_regret_from_picks_graded_0_hi"]].round(4).to_string(index=False))
print(pd.DataFrame(GR).round(3).T.to_string()); print(pd.DataFrame(BA).round(3).to_string(index=False)); print(pd.DataFrame(MC).round(4).to_string(index=False))
