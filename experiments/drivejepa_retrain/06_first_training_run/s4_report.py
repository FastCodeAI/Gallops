#!/usr/bin/env python
"""S4 readouts: per-pass tables R1-R5, pre-registered V1-V5, curves. Reads run/ only (curve.csv, evals.csv, readouts/, state.json).
Run: alpasim/.venv/bin/python analysis_3.0/Retrain-data/06_first_training_run/s4_report.py [set]   (set defaults to val_full when every pass has it)"""
import sys, json, pathlib, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore'); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; R = HERE / "run"; out = []; W = lambda s="": (out.append(s), print(s))
def tbl(df):
    f = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    W("| " + " | ".join(map(str, df.columns)) + " |"); W("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): W("| " + " | ".join(f(v) for v in r) + " |")
    W()
pct = lambda x: f"{100*x:.1f} %"
state = json.load(open(R / "state.json")); cur = pd.read_csv(R / "curve.csv"); ev = pd.read_csv(R / "evals.csv"); P = int(state["pass_done"])
full = set(ev[ev.set == "val_full"].passno); SET = sys.argv[1] if len(sys.argv) > 1 else ("val_full" if full >= set(range(P + 1)) else "val")
NAME = {"A": "highway", "B_raw": "pull-out", "B_pass": "pass", "ordinary": "ordinary"}; ORDER = ["highway", "pull-out", "pass", "ordinary"]
def rd(s, p): d = pd.read_parquet(R / "readouts" / f"{s}_pass{p:02d}.parquet"); d["T"] = d.type.map(NAME); d["G"] = np.where(d.has_label2, "graded", "not graded"); return d
def bl(s, p): d = pd.read_parquet(R / "readouts" / f"{s}_pass{p:02d}_batchloss.parquet"); return d
passes = sorted(set(ev[ev.set == SET].passno)); E = ev[ev.set == SET].set_index("passno").sort_index(); TP = ev[ev.set == "train_probe"].set_index("passno").sort_index(); assert list(E.index) == passes; RD = {p: rd(SET, p) for p in passes}
W(f"Unseen set for these tables: **{SET}** ({int(E.n.iloc[0]):,} examples, {int(E.n_label2.iloc[0]):,} graded); passes with numbers on it: {passes[0]}–{passes[-1]}. Pass 0 = the shipped weights."); W()
best = int(E.loc[[p for p in passes if p > 0], "loss"].idxmin()) if len(passes) > 1 else 0
# ---------------- R1
W("### R1 — error (loss) per pass"); W()
r1 = pd.DataFrame({"pass": passes, "training, as trained": [cur.set_index("passno").train_loss.get(p, np.nan) for p in passes], "training probe": [TP.loss.get(p, np.nan) for p in passes], "unseen total": E.loss.values, "unseen generator": E.trajectory_loss.values,
                   "unseen scorer (graded)": E.final_score_loss_graded.values}); tbl(r1)
g = pd.concat([bl(SET, p) for p in passes]); g = g[g.group != "mixed"]; g["T"] = g.group.str.split("|").str[0].map(NAME); g["G"] = g.group.str.split("|").str[1]
gl = g.groupby(["passno", "T", "G"]).apply(lambda d: pd.Series({k: np.average(d[k], weights=d.n) for k in ("loss", "trajectory_loss", "final_score_loss")})).reset_index()
W("Unseen total error by type (graded / not graded):"); W(); pv = gl.pivot_table(index="passno", columns=["T", "G"], values="loss"); pv.columns = [f"{a} {b}" for a, b in pv.columns]; tbl(pv.reset_index().rename(columns={"passno": "pass"}))
# ---------------- R2
W("### R2 — generator"); W(); W("Chosen plan's distance from the human's path at 4 s, metres (mean):"); W()
d4 = pd.concat([RD[p].groupby(["T", "G"]).dist4.mean().rename(p) for p in passes], axis=1).T; d4.columns = [f"{a} {b}" for a, b in d4.columns]; tbl(d4.reset_index().rename(columns={"index": "pass"}))
hw = pd.DataFrame([dict(**{"pass": p}, examples=len(d), **{"fastest of 32 / human speed (median)": float((d.fastest_end / d.human_end.clip(lower=0.1)).median()), "fastest ≥ 0.9 × human": pct(float((d.fastest_end >= 0.9 * d.human_end).mean())),
                                                               "fastest of 32, m/s (median)": float(d.fastest_end.median()), "human, m/s (median)": float(d.human_end.median()), "chosen / human speed (median)": float((d.chosen_end / d.human_end.clip(lower=0.1)).median())}) for p in passes for d in [RD[p][RD[p]["T"] == "highway"]] if len(d)])
W("Highway, fastest of the 32 proposals vs the human (speed over the last half second of the 4 s):"); W(); tbl(hw)
# ---------------- R3
W("### R3 — scorer (graded examples)"); W()
r3 = []
for p in passes:
    d = RD[p][RD[p].has_label2]; row = {"pass": p, "pick = true best, all": pct(float(d.top1.mean())), "rank agreement, all": float(d.rank_corr.mean())}
    for T in ORDER:
        q = d[d["T"] == T]
        if len(q): row[f"pick = best, {T}"] = pct(float(q.top1.mean()))
    r3.append(row)
tbl(pd.DataFrame(r3)); W("'pick = true best' counts a pick whose true grade equals the best grade among the 32 (ties count). Rank agreement = Spearman correlation between the scorer's 32 scores and the true grades, mean over examples."); W()
ps = pd.DataFrame([dict(**{"pass": p}, examples=len(d), **{"pick: distance from human at 4 s, m (median)": float(d.dist4.median()), "closest of 32, m (median)": float(d.best_dist4.median()), "pick within 1.5 m": pct(float((d.dist4 <= 1.5).mean())), "closest of 32 within 1.5 m": pct(float((d.best_dist4 <= 1.5).mean()))}) for p in passes for d in [RD[p][RD[p]["T"] == "pass"]] if len(d)])
W("Pass examples: the scorer's pick vs the closest of the 32 proposals:"); W(); tbl(ps)
# ---------------- R4 / R5
o = pd.Series({p: float(RD[p][RD[p]["T"] == "ordinary"].dist4.mean()) for p in passes}); W("### R4 — ordinary (forgetting tripwire)"); W(); tbl(pd.DataFrame({"pass": passes, "ordinary 4 s distance, m": o.values, "vs pass 0": [f"{100*(v/o.iloc[0]-1):+.1f} %" for v in o.values]}))
W("### R5 — share seen in training"); W(); sc = [c for c in cur.columns if c.startswith("seen_share_")]; t5 = cur[["passno", "draws", "graded_share_seen"] + sc].rename(columns={"passno": "pass", **{c: NAME[c[11:]] for c in sc}}); tbl(t5)
# ---------------- pre-registered
b = RD[best]; b0 = RD[0]; h = b[b["T"] == "highway"]; v1 = float((h.fastest_end >= 0.9 * h.human_end).mean()) if len(h) else np.nan; gd = b[b.has_label2]; v2 = float(gd.top1.mean()); pe = b[b["T"] == "pass"]; v3 = float((pe.dist4 <= 1.5).mean()) if len(pe) else np.nan
v4 = o[best] / o.iloc[0] - 1; trip = state.get("tripwires", []); v5 = (not trip) and P >= 10
V = [("V1", "highway: fastest of 32 ≥ 0.9 × human speed on ≥ 70 %", f"{pct(v1)} of {len(h):,}", v1 >= 0.70), ("V2", "scorer: pick = true best on ≥ 50 % of graded", f"{pct(v2)} of {len(gd):,}", v2 >= 0.50),
     ("V3", "pass examples: pick within 1.5 m of the human on ≥ 60 %", f"{pct(v3)} of {len(pe):,}", v3 >= 0.60), ("V4", "ordinary: 4 s distance not worse than pass 0 by more than 5 %", f"{100*v4:+.1f} % ({o.iloc[0]:.2f} → {o[best]:.2f} m)", v4 <= 0.05),
     ("V5", "no tripwire before pass 10", "none, 10 passes done" if v5 else (f"{[t['code'] for t in trip]} at pass {[t['at_pass'] for t in trip]}" if trip else f"{P} passes done"), v5)]
W(f"### Pre-registered (on {SET}, best pass = {best} by unseen total error)"); W(); tbl(pd.DataFrame([(a, s, r, "holds" if ok else "fails") for a, s, r, ok in V], columns=["id", "statement", "result", "verdict"]))
fails = [a for a, _, _, ok in V if not ok]; verdict = ("STOPPED (" + ", ".join(t["code"] for t in trip) + ")") if trip else ("RUN-OK" if not fails else "PARTIAL (failed: " + ", ".join(fails) + ")")
W(f"**VERDICT: {verdict}**" + ("" if SET == "val_full" else f"  — computed on `{SET}`, NOT on val_full")); W()
W("### Time and GPU"); W(); tbl(cur[["passno", "steps", "sec_per_step", "train_secs", "gpu_reserved_gb", "elapsed_h", "eval_set"]].assign(train_secs=(cur.train_secs / 60).round(1)).rename(columns={"passno": "pass", "train_secs": "train min", "eval_set": "evaluated on (during run)"}))
json.dump(dict(set=SET, best_pass=best, V1=v1, V2=v2, V3=v3, V4=float(v4), V5=bool(v5), verdict=verdict, tripwires=trip, passes_done=P), open(R / "checks.json", "w"), indent=1); open(R / "tables.md", "w").write("\n".join(out))
# ---------------- curves
C_TR, C_UN, C_AS = "#2563eb", "#d97706", "#9ca3af"
def style(ax, t):
    ax.set_title(t, fontsize=10, loc="left", color="#1f2933"); ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.tick_params(labelsize=8, colors="#6b7280"); ax.set_xticks(passes)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d1d5db")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
for a_, key, t in zip(ax, ["loss", "trajectory_loss", "final_score_loss_graded"], ["Total error", "Generator part", "Scorer part (graded examples)"]):
    a_.plot(passes, [TP[key].get(p, np.nan) for p in passes], color=C_TR, lw=2, marker="o", ms=5, label="training probe (815 training examples, evaluated like unseen)")
    a_.plot(passes, E[key].values, color=C_UN, lw=2, marker="o", ms=5, label=f"unseen clips ({SET})")
    if key == "loss": a_.plot(cur.passno, cur.train_loss, color=C_AS, lw=1.5, ls="--", marker="o", ms=4, label="training, as trained (jitter on, sample weights)")
    a_.axvline(best, color="#6b7280", lw=0.8, ls=":"); style(a_, t); a_.set_xlabel("pass", fontsize=8, color="#6b7280")
ax[0].legend(fontsize=7.5, frameon=False); ax[0].text(best, ax[0].get_ylim()[1], f" best pass {best}", fontsize=7.5, color="#6b7280", va="top"); fig.tight_layout(); fig.savefig(HERE / "curves_total.png"); plt.close(fig)
gp = pd.concat([bl("train_probe", p) for p in passes if (R / "readouts" / f"train_probe_pass{p:02d}_batchloss.parquet").exists()]); gp = gp[gp.group != "mixed"]; gp["T"] = gp.group.str.split("|").str[0].map(NAME)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), dpi=120)
for a_, T in zip(ax, ORDER):
    u = g[g["T"] == T].groupby("passno").apply(lambda d: np.average(d.loss, weights=d.n)); q = gp[gp["T"] == T].groupby("passno").apply(lambda d: np.average(d.loss, weights=d.n))
    if len(q): a_.plot(q.index, q.values, color=C_TR, lw=2, marker="o", ms=5, label="training probe")
    if len(u): a_.plot(u.index, u.values, color=C_UN, lw=2, marker="o", ms=5, label=f"unseen clips ({SET})")
    style(a_, f"{T}: total error" + ("" if len(u) else "  (none in the unseen set)")); a_.set_xlabel("pass", fontsize=8, color="#6b7280")
ax[0].legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(HERE / "curves_by_type.png"); plt.close(fig)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), dpi=120)
for a_, T in zip(ax, ORDER):
    u = pd.Series({p: RD[p][RD[p]["T"] == T].dist4.mean() for p in passes}).dropna(); q = pd.Series({p: rd("train_probe", p).query("T == @T").dist4.mean() for p in passes if (R / "readouts" / f"train_probe_pass{p:02d}.parquet").exists()}).dropna()
    if len(q): a_.plot(q.index, q.values, color=C_TR, lw=2, marker="o", ms=5, label="training probe")
    if len(u): a_.plot(u.index, u.values, color=C_UN, lw=2, marker="o", ms=5, label=f"unseen clips ({SET})")
    style(a_, f"{T}: chosen plan vs human at 4 s, m"); a_.set_xlabel("pass", fontsize=8, color="#6b7280")
ax[0].legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(HERE / "curves_distance_by_type.png"); plt.close(fig)
print("wrote tables.md, checks.json, curves_total.png, curves_by_type.png, curves_distance_by_type.png | best pass", best, "| verdict", verdict)
