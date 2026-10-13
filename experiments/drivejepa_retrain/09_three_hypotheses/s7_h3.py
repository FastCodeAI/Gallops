#!/usr/bin/env python
"""S7 H3: does a ranking lesson on the SAME frozen 256 features order the 32 plans better than the scorer's own output?
Definitions: PREDICTIONS.md, P-H3. CPU only. Phase 1 uses SEEN only (train 80 % / validation 20 % of clips) and writes every choice to
h3_selection.json; UNSEEN is opened only after that file exists (phase 2). No model forward pass of Drive-JEPA: features, scorer outputs
and grades come from S6's dumps.
Loss of baseline (iii) reproduces 05_small_overfit_check/train_patch/drive_jepa_agent_s35.py:248
    final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])      with bce_logit_loss = nn.BCEWithLogitsLoss() (:99)
i.e. cross-entropy on the logit with the grade in [0, 1] as a soft target, mean over instants x 32 plans."""
import os, json, time, datetime, numpy as np, pandas as pd, torch, torch.nn as nn
from s7_common import *
torch.set_num_threads(int(os.environ.get("S7_THREADS", "4"))); T0 = time.time()
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, np.float64)))
WDS = [0.0, 1e-4, 1e-3, 1e-2]; ALPHAS = [0.1, 1.0, 10.0, 100.0]; LR = 1e-3; BATCH = 256; MAXEP = 60; PATIENCE = 8; MARGIN = 0.1; MINDIFF = 0.05

# ---------------- phase 1: SEEN only
D = load("seen", ["slot_9", "grade_9", "logit_9"]); X = D["slot_9"].astype(np.float32); G = D["grade_9"][..., -1].astype(np.float32); clips = np.unique(D["clip"]); rng = np.random.default_rng(SEED); perm = rng.permutation(len(clips)); nva = int(round(0.2 * len(clips)))
va_clips = set(clips[perm[:nva]]); va = np.array([c in va_clips for c in D["clip"]]); tr = ~va; mu = X[tr].reshape(-1, 256).mean(0); sd = X[tr].reshape(-1, 256).std(0); sd = np.where(sd < 1e-6, 1.0, sd); Z = (X - mu) / sd
print(f"SEEN: {len(G)} instants, {len(clips)} clips | train {tr.sum()} instants / {len(clips) - nva} clips | validation {va.sum()} instants / {nva} clips", flush=True)
Ztr, Gtr, Zva, Gva = torch.from_numpy(Z[tr]), torch.from_numpy(G[tr]), torch.from_numpy(Z[va]), G[va]
npairs = int((np.abs(G[tr][:, :, None] - G[tr][:, None, :]) >= MINDIFF).sum() // 2); print("training pairs with grade difference >= 0.05:", npairs, "| instants with none:", int(((np.abs(G[tr][:, :, None] - G[tr][:, None, :]) >= MINDIFF).sum((1, 2)) == 0).sum()), flush=True)
regret = lambda o, g: float((g.max(1) - g[np.arange(len(g)), o.argmax(1)]).mean())
def make(arch): return nn.Linear(256, 1) if arch == "linear" else nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
def rank_loss(o, g):
    d = g[:, :, None] - g[:, None, :]; m = d >= MINDIFF; return torch.relu(MARGIN - (o[:, :, None] - o[:, None, :]))[m].mean() if m.any() else o.sum() * 0
bce = nn.BCEWithLogitsLoss()
def train(arch, loss_name, wd):
    torch.manual_seed(SEED); net = make(arch); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=wd); r2 = np.random.default_rng(SEED); best = (np.inf, -1, None); hist = []
    for ep in range(MAXEP):
        net.train(); p = r2.permutation(len(Ztr)); tot = 0.0
        for k in range(0, len(p), BATCH):
            ii = torch.from_numpy(p[k:k + BATCH]); o = net(Ztr[ii]).squeeze(-1); l = rank_loss(o, Gtr[ii]) if loss_name == "rank" else bce(o, Gtr[ii]); opt.zero_grad(); l.backward(); opt.step(); tot += float(l) * len(ii)
        net.eval()
        with torch.no_grad(): ov = net(Zva).squeeze(-1).numpy()
        rv = regret(ov, Gva); hist.append((ep, tot / len(p), rv))
        if rv < best[0] - 1e-9: best = (rv, ep, {k: v.clone() for k, v in net.state_dict().items()})
        if ep - best[1] >= PATIENCE: break
    net.load_state_dict(best[2]); net.eval(); return net, best[0], best[1], hist
def out(net, Zx):
    with torch.no_grad(): return net(torch.from_numpy(Zx)).squeeze(-1).numpy().astype(np.float64)
RUNS = []; NETS = {}
for arch in ("linear", "mlp64"):
    for wd in WDS:
        net, rv, ep, hist = train(arch, "rank", wd); RUNS.append(dict(head=f"rank-{arch}", weight_decay=wd, val_regret=rv, best_pass=ep, passes_run=len(hist), val_rank_agreement_median=float(np.nanmedian(spearman_rows(out(net, Z[va]), Gva))))); NETS[(f"rank-{arch}", wd)] = net
        print(RUNS[-1], f"{time.time() - T0:.0f}s", flush=True)
R = pd.DataFrame(RUNS); pick_wd = {h: float(x.sort_values(["val_regret", "weight_decay"]).iloc[0].weight_decay) for h, x in R.groupby("head")}; pick_val = {h: float(x.val_regret.min()) for h, x in R.groupby("head")}
chosen = min(pick_val, key=lambda h: (pick_val[h], h)); arch_c = chosen.split("-")[1]; other = [h for h in pick_val if h != chosen][0]
for wd in WDS:
    net, rv, ep, hist = train(arch_c, "bce", wd); RUNS.append(dict(head=f"bce-{arch_c}", weight_decay=wd, val_regret=rv, best_pass=ep, passes_run=len(hist), val_rank_agreement_median=float(np.nanmedian(spearman_rows(out(net, Z[va]), Gva))))); NETS[(f"bce-{arch_c}", wd)] = net; print(RUNS[-1], flush=True)
R = pd.DataFrame(RUNS); xb = R[R["head"] == f"bce-{arch_c}"].sort_values(["val_regret", "weight_decay"]).iloc[0]; bce_wd = float(xb.weight_decay)
# ridge on the grade (closed form, intercept not penalised)
A = Z[tr].reshape(-1, 256).astype(np.float64); y = G[tr].reshape(-1).astype(np.float64); ym = y.mean(); AtA = A.T @ A; Aty = A.T @ (y - ym); RID = {}
for a in ALPHAS:
    w = np.linalg.solve(AtA + a * np.eye(256), Aty); RID[a] = w; ov = Z[va].astype(np.float64) @ w + ym; RUNS.append(dict(head="ridge", weight_decay=a, val_regret=regret(ov, Gva), best_pass=-1, passes_run=0, val_rank_agreement_median=float(np.nanmedian(spearman_rows(ov, Gva))))); print(RUNS[-1], flush=True)
R = pd.DataFrame(RUNS); alpha = float(R[R["head"] == "ridge"].sort_values(["val_regret", "weight_decay"]).iloc[0].weight_decay); R.to_csv(HERE / "h3_training_runs.csv", index=False)
sv = D["logit_9"][va][..., -1].astype(np.float64)
SEL = dict(written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), note="written before UNSEEN is opened", seed=SEED, seen_instants=int(len(G)), train_instants=int(tr.sum()), val_instants=int(va.sum()), train_clips=int(len(clips) - nva), val_clips=int(nva), training_pairs=npairs,
           weight_decay_by_head=pick_wd, val_regret_by_head=pick_val, the_ranking_head=chosen, other_ranking_head=other, bce_head=f"bce-{arch_c}", bce_weight_decay=bce_wd, bce_val_regret=float(xb.val_regret), ridge_alpha=alpha, ridge_val_regret=float(R[(R['head'] == 'ridge') & (R.weight_decay == alpha)].val_regret.iloc[0]),
           scorer_val_regret=regret(sv, Gva), scorer_val_rank_agreement_median=float(np.nanmedian(spearman_rows(sv, Gva))), random_val_regret=float((Gva.max(1) - Gva.mean(1)).mean()))
json.dump(SEL, open(HERE / "h3_selection.json", "w"), indent=1); print("SELECTION", json.dumps(SEL, indent=1), flush=True)
M = {"scorer's own output": None, "ridge on the grade": ("ridge", alpha), f"scorer's loss form ({arch_c})": (f"bce-{arch_c}", bce_wd), f"ranking head ({arch_c}) [the ranking head]": (chosen, pick_wd[chosen]), f"ranking head ({other.split('-')[1]}) [not chosen]": (other, pick_wd[other])}
torch.save(dict(mu=mu, sd=sd, ridge_w=RID[alpha], ridge_b=ym, nets={f"{k[0]}|{k[1]}": v.state_dict() for k, v in NETS.items()}, selection=SEL), HERE / "h3_models.pt")
def outputs(Zx, logit):
    """per method: (ordering output, score on the 0-1 scale used for the momentum step and the top-2 score gap)"""
    O = {}
    for m, k in M.items():
        if k is None: o = logit.astype(np.float64); O[m] = (o, sig(o))
        elif k[0] == "ridge": o = Zx.astype(np.float64) @ RID[alpha] + ym; O[m] = (o, np.clip(o, 0, 1))
        else: o = out(NETS[k], Zx); O[m] = (o, sig(o))
    return O
del Ztr, A

# ---------------- phase 2: evaluation (SEEN validation clips first, then UNSEEN, opened here for the first time)
assert (HERE / "h3_selection.json").is_file()
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
def momentum_pick(E, s, st):
    """s6_analyze.momentum_pick, unchanged: score <- (14 score + 2 comfort) / 16, previous plan = the previous graded instant of the clip and ITS momentum pick"""
    n = len(s); pick = s.argmax(1).copy(); used = np.zeros(n, bool); o = 5; tp = np.arange(41 - o) * 0.1; order = np.lexsort((E["anchor_t"], E["clip"])); prev = None
    for i in order:
        if prev is not None and E["clip"][prev] == E["clip"][i] and abs(E["anchor_t"][i] - E["anchor_t"][prev] - 0.5) < 1e-3:
            cur = np.repeat(st[prev, pick[prev]][None, o:], 32, 0); nxt = st[i][:, :-o]; c = ego_is_two_frame_extended_comfort(cur, nxt, tp).astype(np.float64); pick[i] = int(((14.0 * s[i] + 2.0 * c) / 16.0).argmax()); used[i] = True
        prev = i
    return pick, used
def evaluate(setname, E, Zx, mask=None, momentum=True):
    idx = np.arange(len(E["token"])) if mask is None else np.where(mask)[0]; gt = E["grade_9"][idx][..., -1].astype(np.float64); clip = E["clip"][idx]; slc = E["slice"][idx]; n = len(idx); ar = np.arange(n); best = gt.max(1); O = outputs(Zx[idx], E["logit_9"][idx][..., -1]); rows = []; marg = []; per = dict(token=E["token"][idx], clip=clip, slice=slc, best_grade=best)
    st = np.concatenate([np.load(f)["states_main"] for f in E["_files"]])[idx] if momentum else None; Em = dict(clip=clip, anchor_t=E["anchor_t"][idx]); base_reg = None
    for m, (o, s) in O.items():
        sels = [("raw pick", o.argmax(1), None)]
        if momentum: mp, used = momentum_pick(Em, s, st); sels.append(("with the momentum nudge", mp, used))
        ra = spearman_rows(o, gt)
        for sel, pk, used in sels:
            gp = gt[ar, pk]; reg = best - gp; c = dict(strict=(gp >= best - 1e-9).astype(float), ge09=(gp >= 0.9).astype(float), within005=(reg <= 0.05).astype(float), regret=reg, ra=ra)
            if sel == "raw pick": per[f"pick|{m}"] = pk; per[f"regret|{m}"] = reg; per[f"rank_agreement|{m}"] = ra
            if sel == "raw pick" and m == "scorer's own output": base_reg = reg
            for sl in SL:
                k = np.ones(n, bool) if sl == "all" else slc == sl
                if not k.any(): continue
                r = dict(set=setname, head=m, selection=sel, slice=sl, instants=int(k.sum()), clips=int(len(set(clip[k])))); ck = {a: b[k] for a, b in c.items()}; ck["base"] = base_reg[k]
                for nm, f in dict(regret_mean=lambda q: q["regret"].mean(), regret_p90=lambda q: np.quantile(q["regret"], 0.9), ge09=lambda q: q["ge09"].mean(), true_best=lambda q: q["strict"].mean(), within005=lambda q: q["within005"].mean(), rank_agreement_median=lambda q: np.nanmedian(q["ra"]), rank_agreement_mean=lambda q: np.nanmean(q["ra"]),
                                  regret_minus_scorer=lambda q: q["regret"].mean() - q["base"].mean()).items(): r[nm], r[nm + "_lo"], r[nm + "_hi"] = boot(clip[k], ck, f)
                if used is not None: r["nudge_applied_share"] = float(used[k].mean()); r["pick_changed_by_nudge_share"] = float((pk != o.argmax(1))[k].mean())
                rows.append(r)
        o2 = np.argsort(-s, axis=1, kind="stable")[:, :2]; sg = s[ar, o2[:, 0]] - s[ar, o2[:, 1]]; gg = np.abs(gt[ar, o2[:, 0]] - gt[ar, o2[:, 1]]); worse = (gt[ar, o2[:, 0]] < gt[ar, o2[:, 1]] - 0.1).astype(float)
        for sl in SL:
            k = np.ones(n, bool) if sl == "all" else slc == sl
            if not k.any(): continue
            r = dict(set=setname, head=m, slice=sl, instants=int(k.sum())); c = {"sg": sg[k], "gg": gg[k], "w": worse[k]}; near = lambda q: q["sg"] < 0.01
            for nm, f in dict(score_gap_median=lambda q: np.median(q["sg"]), share_score_gap_lt_0p01=lambda q: np.mean(near(q)), of_near_ties_share_grade_gap_gt_0p1=lambda q: np.mean(q["gg"][near(q)] > 0.1) if near(q).any() else np.nan, share_grade_gap_gt_0p1=lambda q: np.mean(q["gg"] > 0.1),
                              share_top_is_the_worse_by_gt_0p1=lambda q: np.mean(q["w"])).items(): r[nm], r[nm + "_lo"], r[nm + "_hi"] = boot(clip[k], c, f)
            marg.append(r)
    # context rows on the same instants: random pick (expected) and the menu's limit
    for sl in SL:
        k = np.ones(n, bool) if sl == "all" else slc == sl
        if k.any(): v, lo, hi = boot(clip[k], {"x": (best - gt.mean(1))[k]}, lambda q: q["x"].mean()); rows.append(dict(set=setname, head="random pick (expected)", selection="-", slice=sl, instants=int(k.sum()), regret_mean=v, regret_mean_lo=lo, regret_mean_hi=hi, ge09=float((gt >= 0.9).mean(1)[k].mean()), true_best=float((gt >= best[:, None] - 1e-9).mean(1)[k].mean())))
    return rows, marg, pd.DataFrame(per)
show = ["set", "head", "selection", "slice", "instants", "regret_mean", "regret_mean_lo", "regret_mean_hi", "regret_p90", "ge09", "true_best", "rank_agreement_median", "rank_agreement_median_lo", "rank_agreement_median_hi", "rank_agreement_mean", "regret_minus_scorer", "regret_minus_scorer_lo", "regret_minus_scorer_hi"]; pd.set_option("display.width", 300, "display.max_rows", 500)
rv, mv, _ = evaluate("SEEN validation clips", D, Z, mask=va, momentum=False); print(pd.DataFrame(rv)[[c for c in show if c in pd.DataFrame(rv)]].round(3).to_string(index=False), flush=True)
del D, X, Z
U = load("unseen", ["slot_9", "grade_9", "logit_9"]); ZU = (U["slot_9"].astype(np.float32) - mu) / sd; print("UNSEEN opened at", datetime.datetime.now().astimezone().isoformat(timespec="seconds"), len(U["token"]), "instants", flush=True)
ru, mu_, per = evaluate("UNSEEN", U, ZU); o = pd.DataFrame(rv + ru); o.to_csv(HERE / "h3_table.csv", index=False); pd.DataFrame(mv + mu_).to_csv(HERE / "h3_top2_margin.csv", index=False); per.to_parquet(HERE / "h3_unseen_per_instant.parquet")
print(o[o.set == "UNSEEN"][show].round(3).to_string(index=False)); print(pd.DataFrame(mu_).round(4).to_string(index=False))
x = o[(o.set == "UNSEEN") & (o.slice == "all") & (o.selection == "raw pick")].set_index("head"); s0 = x.loc["scorer's own output"]; print("CHECK scorer reproduces S6 (0.115, 0.39): regret %.4f, rank agreement median %.3f" % (s0.regret_mean, s0.rank_agreement_median))
h = x[x.index.str.contains("the ranking head")].iloc[0]; print("P-H3: ranking head regret %.4f (%.4f-%.4f), rank agreement median %.3f (%.3f-%.3f)" % (h.regret_mean, h.regret_mean_lo, h.regret_mean_hi, h.rank_agreement_median, h.rank_agreement_median_lo, h.rank_agreement_median_hi)); print("wall %.0f s" % (time.time() - T0))
