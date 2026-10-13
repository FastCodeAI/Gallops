"""S3.5 readouts R1-R4, checks S1-S6, projection -> run/tables.md + PNG curves."""
import json, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; R = HERE / "run"; cv = pd.read_csv(R / "curve.csv"); rd = pd.read_parquet(R / "readouts.parquet"); sm = json.load(open(R / "summary.json")); out = []; W = lambda s="": (out.append(s), print(s))
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    W("| " + " | ".join(map(str, df.columns)) + " |"); W("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): W("| " + " | ".join(f(v) for v in r) + " |")
    W()
pct = lambda x: f"{100*x:.1f} %"; last = int(cv.epoch.max()); best = int(cv.loc[cv.heldout_loss.idxmin(), "epoch"]); E = sorted({0, best, last}); lab = {0: "epoch 0", best: f"best ({best})", last: f"last ({last})"}
if best == last: lab[last] = f"best = last ({last})"
# ---- curves
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(cv.epoch, cv.eval_train_loss, label="train slice (eval mode)", c="#2563eb"); ax[0].plot(cv.epoch[1:], cv.train_loss[1:], label="train slice (training mode, running)", c="#93c5fd", ls="--"); ax[0].plot(cv.epoch, cv.heldout_loss, label="held-out", c="#dc2626"); ax[0].axvline(best, c="#9ca3af", ls=":"); ax[0].set_title("total loss"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(cv.epoch, cv.eval_train_trajectory_loss, c="#2563eb", label="train slice"); ax[1].plot(cv.epoch, cv.heldout_trajectory_loss, c="#dc2626", label="held-out"); ax[1].set_title("generator head (trajectory loss)"); ax[1].legend(frameon=False, fontsize=8)
ax[2].plot(cv.epoch, cv.eval_train_final_score_loss_labelled, c="#2563eb", label="train slice"); ax[2].plot(cv.epoch, cv.heldout_final_score_loss_labelled, c="#dc2626", label="held-out"); ax[2].set_title("scorer head (final-score loss, labelled samples)"); ax[2].legend(frameon=False, fontsize=8)
for a_ in ax: a_.set_xlabel("epoch"); a_.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(R / "curves_loss.png", dpi=110); plt.close(fig)
g = rd.groupby(["epoch", "set", "type"]).agg(dist4=("dist4", "mean"), fastest=("fastest_end", "mean")).reset_index(); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); col = {"A": "#dc2626", "B_raw": "#16a34a", "B_pass": "#9333ea", "ordinary": "#2563eb"}
for (s, t), d in g.groupby(["set", "type"]): ax[0].plot(d.epoch, d.dist4, c=col[t], ls="-" if s == "train" else "--", label=f"{t} {s}")
ax[0].set_title("R2: chosen plan vs human path at 4 s (m)"); ax[0].set_yscale("log"); ax[0].legend(frameon=False, fontsize=7, ncol=2); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.25)
A = rd[rd.type == "A"].groupby(["epoch", "set"]).agg(fastest=("fastest_end", "mean"), human=("human_end", "mean"), chosen=("chosen_end", "mean")).reset_index()
for s, d in A.groupby("set"): ax[1].plot(d.epoch, d.fastest, c="#dc2626", ls="-" if s == "train" else "--", label=f"fastest of 32, {s}"); ax[1].plot(d.epoch, d.chosen, c="#2563eb", ls="-" if s == "train" else "--", label=f"chosen, {s}"); ax[1].plot(d.epoch, d.human, c="#111827", ls="-" if s == "train" else "--", lw=0.8, label=f"human, {s}")
ax[1].set_title("R4: highway samples, end speed (m/s)"); ax[1].legend(frameon=False, fontsize=7, ncol=2); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.25); fig.tight_layout(); fig.savefig(R / "curves_readouts.png", dpi=110); plt.close(fig)
# ---- R1
W("### R1 — loss per epoch (total and per head)"); W()
c = cv[["epoch", "train_loss", "train_trajectory_loss", "train_final_score_loss", "eval_train_loss", "eval_train_trajectory_loss", "eval_train_final_score_loss_labelled", "heldout_loss", "heldout_trajectory_loss", "heldout_final_score_loss_labelled"]].copy()
c.columns = ["epoch", "train total (training mode)", "train generator", "train scorer", "slice total (eval mode)", "slice generator", "slice scorer (labelled)", "held-out total", "held-out generator", "held-out scorer (labelled)"]
tbl(c[c.epoch.isin(sorted(set([0, 1, 2, 3, 5, 10, 15, 20, 25, last, best])))].round(3)); W(f"Full curve: `run/curve.csv`; plots `run/curves_loss.png`, `run/curves_readouts.png`. Training-mode loss includes the GridMask augmentation; eval-mode numbers do not. Best held-out epoch: **{best}**."); W()
# ---- R2
W("### R2 — generator: distance between the chosen plan and the human path at 4 s (m), mean per type"); W()
p = rd[rd.epoch.isin(E)].groupby(["set", "type", "epoch"]).agg(n=("token", "size"), dist4=("dist4", "mean"), best_of_32=("best_dist4", "mean")).reset_index()
rows = []
for (s, t), d in p.groupby(["set", "type"]):
    r = dict(set=s, type=t, samples=int(d.n.iloc[0]))
    for e in E: x = d[d.epoch == e]; r[f"{lab[e]} chosen"] = float(x.dist4.iloc[0]); r[f"{lab[e]} closest of 32"] = float(x.best_of_32.iloc[0])
    rows.append(r)
tbl(pd.DataFrame(rows).round(2)); W("Held-out has no pull-out (B_raw) clips, so the pull-out readout exists on the train slice only."); W()
# ---- R3
W("### R3 — scorer on samples with label 2"); W()
l2 = rd[rd.has_label2 & rd.epoch.isin(E)]; rows = []
for (s, e), d in l2.groupby(["set", "epoch"]): rows.append(dict(set=s, epoch=lab[e], samples=len(d), top1_agreement=pct(d.top1.mean()), top1_strict=pct(d.top1_strict.mean()), rank_correlation_mean=d.rank_corr.mean(), anchors_with_flat_referee=int((d.ref_std < 1e-6).sum()), referee_grade_of_chosen=d.ref_chosen.mean(), referee_best_of_32=d.ref_best.mean()))
tbl(pd.DataFrame(rows).round(3)); W("top-1 agreement = the scorer's pick has the referee's top grade among the 32 (ties count); strict = same index as the referee's first maximum. Rank correlation is Spearman per sample, averaged over samples where both vary."); W()
rows = []
for (s, t, e), d in l2.groupby(["set", "type", "epoch"]): rows.append(dict(set=s, type=t, epoch=lab[e], samples=len(d), top1=pct(d.top1.mean()), rank_corr=d.rank_corr.mean(), ref_chosen=d.ref_chosen.mean()))
tbl(pd.DataFrame(rows).round(3))
# ---- R4
W("### R4 — highway (type A) samples: fastest of the 32 proposals vs the human"); W()
a = rd[(rd.type == "A") & rd.epoch.isin(E)]; rows = []
for (s, e), d in a.groupby(["set", "epoch"]): rows.append(dict(set=s, epoch=lab[e], anchors=len(d), human_end_speed=d.human_end.mean(), fastest_mean=d.fastest_end.mean(), fastest_max=d.fastest_end.max(), chosen_mean=d.chosen_end.mean(), fastest_over_human=(d.fastest_end / d.human_end).median(), chosen_over_human=(d.chosen_end / d.human_end).median()))
tbl(pd.DataFrame(rows).round(2))
# ---- checks
tA0 = rd[(rd.type == "A") & (rd.set == "train") & (rd.epoch == 0)].set_index("token"); tAb = rd[(rd.type == "A") & (rd.set == "train") & (rd.epoch == best)].set_index("token"); ceil0 = float(tA0.fastest_end.max())
s5_glob = float((tAb.fastest_end > ceil0).mean()); s5_per = float((tAb.fastest_end > tA0.fastest_end.reindex(tAb.index)).mean())
l0, lN = float(cv.eval_train_loss.iloc[0]), float(cv.eval_train_loss.iloc[-1]); h0, hmin, hN = float(cv.heldout_loss.iloc[0]), float(cv.heldout_loss.min()), float(cv.heldout_loss.iloc[-1])
S1 = lN <= 0.25 * l0; S2 = (best < last) and (hN >= 1.05 * hmin); S3 = hmin < h0; s4 = sm["s4_post_training_raw_steps"]; S4 = len(s4) > 0 and all(x["scorer_unchanged"] and x["scorer_grad_none"] for x in s4); S5 = s5_glob >= 0.5
ro = sm["raw_only_batches_in_training"]
v = [("S1", "train loss at the last epoch ≤ 25 % of epoch 0", f"slice eval loss {l0:.3f} → {lN:.3f} = {pct(lN / l0)} (training-mode loss epoch 1 → last: {cv.train_loss.iloc[1]:.3f} → {cv.train_loss.iloc[-1]:.3f} = {pct(cv.train_loss.iloc[-1] / cv.train_loss.iloc[1])})", "holds" if S1 else "fails"),
     ("S2", "held-out loss bottoms before the last epoch and then rises ≥ 5 %", f"minimum {hmin:.3f} at epoch {best}; last {hN:.3f} = {100*(hN/hmin-1):+.1f} %", "holds" if S2 else "fails"),
     ("S3", "held-out minimum < held-out at epoch 0", f"{hmin:.3f} vs {h0:.3f}", "holds" if S3 else "fails"),
     ("S4", "scorer weights unchanged by raw-only steps", f"{sum(x['scorer_unchanged'] for x in s4)} of {len(s4)} raw-only optimiser steps after training leave the scorer bit-identical (generator moved on {sum(x['generator_moved'] for x in s4)}, encoder unchanged on {sum(x['encoder_unchanged'] for x in s4)}); raw-only batches met during training: {ro['batches']} ({ro['scorer_unchanged']} unchanged)", "holds" if S4 else "fails"),
     ("S5", "A samples: fastest proposal at the best epoch exceeds the epoch-0 ceiling on ≥ 50 % of train-slice A anchors", f"epoch-0 ceiling (fastest proposal on any train-slice A anchor) {ceil0:.1f} m/s; exceeded on {pct(s5_glob)} of {len(tAb)} anchors at epoch {best} (above the same anchor's own epoch-0 fastest: {pct(s5_per)})", "holds" if S5 else "fails"),
     ("S6", "held-out and train-slice drives disjoint", "0 shared chunks, clips or samples (asserted in `s35_slice.py` and again at training start)", "holds")]
W("### Pre-registered"); W(); tbl(pd.DataFrame(v, columns=["id", "statement", "result", "verdict"]))
verdict = "PIPELINE-OK" if (S1 and S2 and S3 and S4) else ("BROKEN" if not S1 else ("LEAK" if (not S2 and abs(hN - lN) / max(lN, 1e-9) < 0.2) else "not PIPELINE-OK: " + ", ".join(k for k, ok in (("S2", S2), ("S3", S3), ("S4", S4)) if not ok)))
if not S5: verdict += " + GENERATOR-STUCK"
W(f"**VERDICT: {verdict}**"); W()
# ---- usage and projection
sps = sm["sec_per_step_median"]; tr = pd.read_parquet(HERE.parent / "04_training_set" / "manifest.parquet"); draws = len(tr) + int((tr.type == "A").sum()); steps = int(np.ceil(draws / sm["batch_size"]))
W("### GPU, time, projection"); W()
tbl(pd.DataFrame([dict(item="epochs completed", value=f"{sm['epochs_done']} of 30" + (f" (stopped: {sm['stopped']})" if sm["stopped"] else "")), dict(item="wall time", value=f"{sm['wall_h']:.2f} h of the 2 h cap"), dict(item="GPU memory, peak reserved / allocated", value=f"{sm['gpu_max_reserved_gb']:.1f} / {sm['gpu_max_alloc_gb']:.1f} GiB of the 24 GiB cap"),
                  dict(item="seconds per step (batch 32), median / mean", value=f"{sps:.2f} / {sm['sec_per_step_mean']:.2f}"), dict(item="steps", value=sm["steps"]), dict(item="per-epoch draws / samples", value=f"{sm['draws_per_epoch']} / {sm['slice_samples']}"),
                  dict(item="evaluation per epoch (slice + held-out)", value=f"{cv.eval_secs.iloc[1:].median():.0f} s"), dict(item="device / torch", value=f"{sm['device']} / {sm['torch']}"), dict(item="vocabulary loaded", value=str(tuple(sm["vocab_shape"])))]))
seen = cv[[c for c in cv.columns if c.startswith("seen_share_")]].iloc[1:]; W("Effective per-type share actually seen per epoch (type A drawn twice): " + ", ".join(f"{c.replace('seen_share_', '')} {pct(seen[c].mean())}" for c in seen.columns) + f" (identical every epoch; sampler plan {sm['sampler_share']})."); W()
tbl(pd.DataFrame([dict(epochs=n, steps=steps * n, hours_training_only=steps * n * sps / 3600) for n in (5, 10, 20)]).round(1))
W(f"Projection basis: full train manifest {len(tr):,} samples + {int((tr.type == 'A').sum()):,} repeated highway samples = {draws:,} draws = {steps:,} steps per epoch at batch 32, at the measured {sps:.2f} s per step. It excludes per-epoch evaluation and assumes the same labelled share per batch; the full set is {pct(tr.has_label2.mean())} labelled against {pct(pd.read_parquet(HERE / 'slice_train.parquet').has_label2.mean())} in the slice, and the referee call on labelled rows is part of the step time.")
json.dump(dict(S1=bool(S1), S2=bool(S2), S3=bool(S3), S4=bool(S4), S5=bool(S5), S6=True, verdict=verdict, best_epoch=best, last_epoch=last), open(R / "checks.json", "w"), indent=1); open(R / "tables.md", "w").write("\n".join(out))
