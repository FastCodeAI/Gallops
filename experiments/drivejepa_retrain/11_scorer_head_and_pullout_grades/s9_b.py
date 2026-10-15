#!/usr/bin/env python
"""S9 part B: two-stage head on the frozen 256 features (definitions: PREDICTIONS.md, P-B; pieces in s9_lib.py).
Phase 1 uses SEEN only (S7's train / validation clip split): gate head, one ranker per threshold t, t chosen on the validation clips,
all written to b_selection.json. UNSEEN is opened only after that file exists.
Usage: s9_b.py            part B
       s9_b.py --c4       part C.4: the gate head is retrained with the pull-out examples added (collision label only), the ranker is
                          retrained on its P with the same recipe, t is NOT re-chosen (read from b_selection.json); outputs prefixed c4_."""
import sys, json, time, datetime, numpy as np, pandas as pd, torch
from s9_lib import *
C4 = "--c4" in sys.argv; PRE = "c4_" if C4 else "b_"; T0 = time.time(); now = lambda: datetime.datetime.now().astimezone().isoformat(timespec="seconds"); LOG = []
D = load("seen", ["slot_9", "grade_9", "logit_9"]); tr, va = seen_split(D); X = D["slot_9"].astype(np.float32); g = D["grade_9"].astype(np.float64); G = g[..., -1]; mu = X[tr].reshape(-1, 256).mean(0); sd = X[tr].reshape(-1, 256).std(0); sd = np.where(sd < 1e-6, 1.0, sd)
s7 = torch.load(RD / "09_three_hypotheses/h3_models.pt", weights_only=False); assert np.allclose(mu, s7["mu"], atol=1e-5) and np.allclose(sd, s7["sd"], atol=1e-5), "split or scaling differs from S7"
Z = (X - mu) / sd; Y = fails(g); M = np.ones_like(Y); print(f"SEEN train {tr.sum()} instants, validation {va.sum()} | plans failing [kerb, oncoming, collision], train: {Y[tr].mean((0, 1)).round(4)} | any: {Y[tr].any(-1).mean():.4f}", flush=True)
Ztr, Ytr, Mtr = Z[tr], Y[tr], M[tr]; extra = None
if C4:
    import pathlib; pz = np.load(pathlib.Path("{runs}/drivejepa_retrain/11_scorer_head_and_pullout_grades/pullout/pullout_all.npz"), allow_pickle=True); pc = pz["clip"]; cl = np.unique(pc); perm = np.random.default_rng(SEED).permutation(len(cl)); nv = int(round(0.2 * len(cl))); pv = np.isin(pc, cl[perm[:nv]])
    Zp = (pz["slot_9"].astype(np.float32) - mu) / sd; Yp = np.zeros((len(Zp), 32, 3), np.float32); Yp[..., 2] = (pz["nc_strict"] < 1.0); Mp = np.zeros_like(Yp); Mp[..., 2] = 1.0
    Ztr = np.concatenate([Z[tr], Zp[~pv]]); Ytr = np.concatenate([Y[tr], Yp[~pv]]); Mtr = np.concatenate([M[tr], Mp[~pv]]); extra = (Zp[pv], Yp[pv], Mp[pv])
    print(f"C.4: pull-out examples added: train {int((~pv).sum())} ({len(cl) - nv} clips), validation {int(pv.sum())} ({nv} clips) | plans failing collision (strict, map-free) {Yp[..., 2].mean():.4f}", flush=True)
Zva_g, Yva_g, Mva_g = (Z[va], Y[va], M[va]) if extra is None else (np.concatenate([Z[va], extra[0]]), np.concatenate([Y[va], extra[1]]), np.concatenate([M[va], extra[2]]))
# ---- gate head
gates = {wd: train_gate(Ztr, Ytr, Mtr, Zva_g, Yva_g, Mva_g, wd, LOG) for wd in WDS}; gwd = min(WDS, key=lambda w: (gates[w][1], w)); gate = gates[gwd][0]; print("gate head:", [(w, round(v[1], 5)) for w, v in gates.items()], "-> weight decay", gwd, f"{time.time() - T0:.0f}s", flush=True)
pg_tr, P_tr = gate_probs(gate, Z[tr]); pg_va, P_va = gate_probs(gate, Z[va]); Gtr, Gva = G[tr], G[va]
# ---- one ranker per t; t chosen on the SEEN validation clips
s7net = rank_net(); s7net.load_state_dict(s7["nets"]["rank-mlp64|0.01"]); s7_va = fwd(s7net, Z[va]).squeeze(-1); RK = {}; CURVE = []
tsel = json.load(open(HERE / "b_selection.json"))["t"] if C4 else None
for t in (TGRID if not C4 else [tsel, 1.0]):
    RK[t], rv = train_ranker(Z[tr], Gtr, P_tr, Z[va], Gva, P_va, t, True, LOG); ov = fwd(RK[t], Z[va]).squeeze(-1); pk, v = select(ov, P_va, t); gp = Gva[np.arange(len(Gva)), pk]; pk7, _ = select(s7_va, P_va, t)
    CURVE.append(dict(t=t, val_regret=float(regret_of(pk, Gva).mean()), val_picks_graded_0=float((gp <= 1e-9).mean()), val_true_best=float((gp >= Gva.max(1) - 1e-9).mean()), val_plans_vetoed=float(v.mean()), val_instants_all_vetoed=float(v.all(1).mean()),
                      val_best_plan_vetoed=float(v[np.arange(len(Gva)), Gva.argmax(1)].mean()), val_regret_veto_plus_s7_ranker=float(regret_of(pk7, Gva).mean()), val_regret_gate_only=float(regret_of(P_va.argmin(1), Gva).mean())))
    print(CURVE[-1], f"{time.time() - T0:.0f}s", flush=True)
cv = pd.DataFrame(CURVE); cv.to_csv(HERE / f"{PRE}t_curve.csv", index=False)
if not C4: tsel = float(cv.sort_values(["val_regret", "t"], ascending=[True, False]).iloc[0].t)
SEL = dict(written=now(), note="written before UNSEEN is opened", seed=SEED, t=tsel, t_rechosen=not C4, gate_weight_decay=gwd, gate_val_loss=gates[gwd][1], ranker_weight_decay=RANK_WD, train_instants=int(len(Ztr)), val_instants=int(len(Zva_g)),
           val_regret_at_t=float(cv[cv.t == tsel].val_regret.iloc[0]), val_regret_scorer=float(regret_of(D["logit_9"][va][..., -1].argmax(1), Gva).mean()), val_regret_s7_head=float(regret_of(s7_va.argmax(1), Gva).mean()))
json.dump(SEL, open(HERE / f"{PRE}selection.json", "w"), indent=1); pd.DataFrame(LOG).to_csv(HERE / f"{PRE}training_runs.csv", index=False); print("SELECTION", json.dumps(SEL), flush=True)
torch.save(dict(mu=mu, sd=sd, gate=gate.state_dict(), rankers={str(t): n.state_dict() for t, n in RK.items()}, selection=SEL), HERE / f"{PRE}models.pt")
# gate AUROC on the SEEN validation clips (the model itself trained on them; the gate head did not)
AU = []
def au_rows(setname, pg, P, Yx, clip):
    for j, gn in enumerate(GATES): v, lo, hi = auroc_boot(pg[..., j], Yx[..., j], clip); AU.append(dict(set=setname, gate=gn, auroc=v, lo=lo, hi=hi, plans_failing=float(Yx[..., j].mean())))
    v, lo, hi = auroc_boot(P, Yx.any(-1), clip); AU.append(dict(set=setname, gate="any of the three (P)", auroc=v, lo=lo, hi=hi, plans_failing=float(Yx.any(-1).mean())))
au_rows("SEEN validation clips", pg_va, P_va, Y[va], D["clip"][va])
if C4: pgp, Pp = gate_probs(gate, extra[0]); v, lo, hi = auroc_boot(pgp[..., 2], extra[1][..., 2], pc[pv]); AU.append(dict(set="pull-out validation clips (map-free strict label)", gate="collision", auroc=v, lo=lo, hi=hi, plans_failing=float(extra[1][..., 2].mean())))
del D, X, Z

# ---------------- UNSEEN, opened here for the first time in this run
assert (HERE / f"{PRE}selection.json").is_file(); U = load("unseen", ["slot_9", "grade_9", "logit_9"]); print("UNSEEN opened at", now(), flush=True)
ZU = (U["slot_9"].astype(np.float32) - mu) / sd; gU = U["grade_9"].astype(np.float64); GU = gU[..., -1]; YU = fails(gU); clip = U["clip"]; slc = U["slice"]; n = len(GU); pgU, PU = gate_probs(gate, ZU); au_rows("UNSEEN", pgU, PU, YU, clip)
sc = U["logit_9"][..., -1].astype(np.float64); v, lo, hi = auroc_boot(-sc, YU.any(-1), clip); AU.append(dict(set="UNSEEN", gate="any of the three, by the scorer's own output (reference)", auroc=v, lo=lo, hi=hi, plans_failing=float(YU.any(-1).mean())))
pd.DataFrame(AU).to_csv(HERE / f"{PRE}gate_auroc.csv", index=False); print(pd.DataFrame(AU).round(4).to_string(index=False), flush=True)
o_two = fwd(RK[tsel], ZU).squeeze(-1); o_nov = fwd(RK[1.0], ZU).squeeze(-1); o_s7 = fwd(s7net, ZU).squeeze(-1); pk_two, vU = select(o_two, PU, tsel); st = np.concatenate([np.load(f)["states_main"] for f in U["_files"]]); at = U["anchor_t"]
base = regret_of(o_s7.argmax(1), GU); ROWS = []; PER = dict(token=U["token"], clip=clip, slice=slc, best_grade=GU.max(1), gate_P_min=PU.min(1), plans_vetoed=vU.sum(1))
ex = dict(plans_vetoed_share=vU.mean(1), all_vetoed=vU.all(1).astype(float), best_plan_vetoed=vU[np.arange(n), GU.argmax(1)].astype(float))
def both(name, raw_pick, order_out, s01, fallback, extra=None):
    ROWS.extend(table_rows("UNSEEN", name, "raw pick", raw_pick, gU, clip, slc, order_out, base, extra)); PER[f"pick|{name}"] = raw_pick; PER[f"regret|{name}"] = regret_of(raw_pick, GU)
    mp, used = momentum_pick(clip, at, s01, st, fallback); ROWS.extend(table_rows("UNSEEN", name, "with the momentum nudge", mp, gU, clip, slc, None, base, dict(nudge_applied=used.astype(float), pick_changed_by_nudge=(mp != raw_pick).astype(float))))
lab = "two-stage head" if not C4 else "two-stage head, gate head retrained with pull-outs (C.4)"
both(f"{lab} (t = {tsel})", pk_two, o_two, np.where(vU, -np.inf, sig(o_two)), PU.argmin(1), ex)
both("scorer's own output", sc.argmax(1), sc, sig(sc), sc.argmax(1)); both("S7 ranking head", o_s7.argmax(1), o_s7, sig(o_s7), o_s7.argmax(1))
pk, v2 = select(o_s7, PU, tsel); both(f"veto (t = {tsel}) + S7's ranker", pk, o_s7, np.where(v2, -np.inf, sig(o_s7)), PU.argmin(1)); both("weighted ranker, no veto (t = 1.0)", o_nov.argmax(1), o_nov, sig(o_nov), o_nov.argmax(1))
ROWS.extend(table_rows("UNSEEN", "gate head alone (lowest P)", "raw pick", PU.argmin(1), gU, clip, slc, -PU, base))
if C4:
    b = torch.load(HERE / "b_models.pt", weights_only=False); rb = rank_net(); rb.load_state_dict(b["rankers"][str(tsel)]); ob = fwd(rb, ZU).squeeze(-1); pk, v3 = select(ob, PU, tsel); both(f"C.4 gate head + B's ranker (t = {tsel})", pk, ob, np.where(v3, -np.inf, sig(ob)), PU.argmin(1))
    gb = gate_net(); gb.load_state_dict(b["gate"]); _, PB = gate_probs(gb, ZU); PER["gate_P_B_head_of_two_stage_pick"] = PB[np.arange(n), pk_two]; print("gate P on UNSEEN plans, C.4 head vs B head: correlation %.4f, mean abs difference %.4f, veto decisions changed %.4f" % (np.corrcoef(PU.ravel(), PB.ravel())[0, 1], np.abs(PU - PB).mean(), ((PU > tsel) != (PB > tsel)).mean()))
gt = GU; w = np.full((n, 32), 1 / 32)
for sl in SL:
    k = np.ones(n, bool) if sl == "all" else slc == sl; reg = (gt.max(1, keepdims=True) - gt)[k]; z = (gt <= 1e-9)[k]
    ROWS.append(dict(set="UNSEEN", head="random pick (expected)", selection="-", slice=sl, instants=int(k.sum()), regret_mean=float(reg.mean()), picks_graded_0=float(z.mean()), true_best=float((gt >= gt.max(1, keepdims=True) - 1e-9)[k].mean()), ge09=float((gt >= 0.9)[k].mean()), share_of_regret_from_picks_graded_0=float((reg * z).sum() / reg.sum())))
o = pd.DataFrame(ROWS); o.to_csv(HERE / f"{PRE}table.csv", index=False); pd.DataFrame(PER).to_parquet(HERE / f"{PRE}unseen_per_instant.parquet")
# veto statistics at the chosen t
VS = []
for sl in SL:
    k = np.ones(n, bool) if sl == "all" else slc == sl; f = YU.any(-1)[k]; v = vU[k]; z = (gt <= 1e-9)[k]
    VS.append(dict(slice=sl, t=tsel, plans_vetoed=float(v.mean()), plans_failing_a_gate=float(f.mean()), failing_plans_vetoed=float(v[f].mean()), passing_plans_vetoed=float(v[~f].mean()), plans_graded_0_vetoed=float(v[z].mean()), plans_graded_0_left_in=float((z & ~v).sum() / max((~v).sum(), 1)), instants_all_vetoed=float(v.all(1).mean()),
                   instants_best_plan_vetoed=float(v[np.arange(k.sum()), gt[k].argmax(1)].mean()), instants_every_best_plan_vetoed=float((v | (gt[k] < gt[k].max(1, keepdims=True) - 1e-9)).all(1).mean()), best_grade_left_after_veto_mean=float(np.where(v.all(1), 0, np.where(v, -1, gt[k]).max(1)).mean()), best_grade_on_menu_mean=float(gt[k].max(1).mean())))
pd.DataFrame(VS).to_csv(HERE / f"{PRE}veto_stats.csv", index=False); pd.set_option("display.width", 330, "display.max_rows", 500, "display.max_colwidth", 60)
show = ["head", "selection", "slice", "instants", "regret_mean", "regret_mean_lo", "regret_mean_hi", "regret_p90", "picks_graded_0", "picks_graded_0_lo", "picks_graded_0_hi", "true_best", "ge09", "rank_agreement_median", "share_of_regret_from_picks_graded_0", "zeroed_by_drivable_area", "zeroed_by_driving_direction", "zeroed_by_collision", "regret_minus_s7_head", "regret_minus_s7_head_lo", "regret_minus_s7_head_hi"]
print(o[[c for c in show if c in o]].round(3).to_string(index=False)); print(pd.DataFrame(VS).round(4).T.to_string()); print(cv.round(4).to_string(index=False))
x = o[(o.slice == "all") & (o.selection == "raw pick")].set_index("head"); h = x[x.index.str.startswith("two-stage")].iloc[0]; s0 = x.loc["scorer's own output"]; s7r = x.loc["S7 ranking head"]
print("CHECK baselines reproduce S7/S8: scorer regret %.4f (0.115), zero picks %.4f (0.081); S7 head regret %.4f (0.110), zero picks %.4f (0.095)" % (s0.regret_mean, s0.picks_graded_0, s7r.regret_mean, s7r.picks_graded_0))
print("P-B: picks graded 0 %.4f (%.4f-%.4f), regret %.4f (%.4f-%.4f) | wall %.0f s" % (h.picks_graded_0, h.picks_graded_0_lo, h.picks_graded_0_hi, h.regret_mean, h.regret_mean_lo, h.regret_mean_hi, time.time() - T0))
