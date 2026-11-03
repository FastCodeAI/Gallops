#!/usr/bin/env python
"""S6 part b from the dumps (no model run): can each calculator check be told apart, plan by plan,
  (1) by the scorer's own outputs (the trained total output, and the per-check output where one exists), per checkpoint, and
  (2) by a fresh linear probe (logistic regression) trained on SEEN pass-9 features: the 256 numbers per plan the scorer reads ("slot")
      and the 1024 numbers just before its final layer ("feat"). Train = 60 % of SEEN clips; tested on the other 40 % of SEEN clips
      and on UNSEEN.
Separation = AUROC over plans pooled across examples (0.5 = coin flip). A plan "passes" a check when its column equals 1 (for progress, EP: full progress; a median split was tried first and dropped,
because over half of the pass-1 plans sit exactly at 1); "score" passes at >= 0.9. A check is skipped when either class has < 200 plans.
Intervals: clip-level bootstrap, 95 %, 200 resamples, seed 20261007.
Usage: s6_partb.py [--heads-only] [--set=unseen]"""
import sys, glob, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = pathlib.Path(__file__).resolve().parent; NAS = pathlib.Path("{runs}/drivejepa_retrain/08_scorer_diagnosis/dump"); SEED = 20261007; NB = 200
TERMS = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]; HEAD = {"DDC": 0, "EP": 1, "TTC": 2, "LK": 3, "HC": 4, "score": 5}   # scorer outputs = last 6 calculator columns
MAPFREE = {"HC"}; SL = ["all", "highway", "close-pass", "ordinary"]; HEADS_ONLY = "--heads-only" in sys.argv; ONLY = [x.split("=")[1] for x in sys.argv if x.startswith("--set=")]

def load(name, keys):
    fs = [f for f in sorted(glob.glob(str(NAS / name / "shard_*.npz"))) if ".tmp." not in f]; man = pd.read_parquet(HERE / f"set_{name}.parquet").set_index("sample_id")
    D = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}; assert len(D["token"]) == len(man), f"{name} dump incomplete"; m = man.loc[D["token"]]; D["slice"] = m["slice"].to_numpy(); D["clip"] = m.clip_uuid.to_numpy(); return D

def auc(score, y):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 < 1 or n0 < 1: return np.nan
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def auc_boot(score, y, clip_of_row):
    v = auc(score, y); u, inv = np.unique(clip_of_row, return_inverse=True); order = np.argsort(inv, kind="stable"); cnt = np.bincount(inv, minlength=len(u)); start = np.r_[0, np.cumsum(cnt)[:-1]]; rng = np.random.default_rng(SEED); out = []
    for _ in range(NB):
        ii = np.concatenate([order[start[k]:start[k] + cnt[k]] for k in rng.integers(0, len(u), len(u))]); out.append(auc(score[ii], y[ii]))
    lo, hi = np.nanpercentile(out, [2.5, 97.5]); return v, float(lo), float(hi)

def labels(g, t):
    x = g[:, :, TERMS.index(t)].reshape(-1); return (x >= 1.0 - 1e-6) if t != "score" else (x >= 0.9)

ROWS = []; DATA = {}
for name in (ONLY or ("unseen", "seen")):
    D = load(name, ["token"] + [f"grade_{p}" for p in (0, 1, 9)] + [f"logit_{p}" for p in (0, 1, 9)]); DATA[name] = D; n = len(D["token"]); clip_row = np.repeat(D["clip"], 32); sl_row = np.repeat(D["slice"], 32)
    for p, pname in [(0, "shipped"), (1, "pass 1"), (9, "pass 9")]:
        g = D[f"grade_{p}"]; lg = D[f"logit_{p}"].reshape(-1, 6).astype(np.float64)
        for t in TERMS:
            y = labels(g, t)
            for sl in (SL if p == 9 else ["all"]):
                m = np.ones(len(y), bool) if sl == "all" else sl_row == sl
                if min(y[m].sum(), (~y[m]).sum()) < 200: ROWS.append(dict(set=name.upper(), scorer=pname, check=t, slice=sl, source="skipped: a class has < 200 plans", plans=int(m.sum()), pass_share=float(y[m].mean()))); continue
                for src, col in [("total output (trained)", 5)] + ([(f"own output for this check (weight {'1' if t == 'score' else '0'})", HEAD[t])] if t in HEAD and t != "score" else []):
                    v, lo, hi = auc_boot(lg[m][:, col], y[m], clip_row[m]); ROWS.append(dict(set=name.upper(), scorer=pname, check=t, slice=sl, source=src, plans=int(m.sum()), pass_share=float(y[m].mean()), separation=v, lo=lo, hi=hi, needs_map=t not in MAPFREE))
    print(name, "heads done", flush=True)
pd.DataFrame(ROWS).to_csv(HERE / ("table_b_heads" + ("_" + "_".join(ONLY) if ONLY else "") + ".csv"), index=False)
h = pd.DataFrame(ROWS); pd.set_option("display.width", 250); print(h[(h.slice == "all")].drop(columns=["slice", "plans"]).round(3).to_string(index=False))
if HEADS_ONLY or ONLY: sys.exit(0)

# ---------------------------------------------------------------- fresh probes on pass-9 features, trained on SEEN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
P = []; S, U = DATA["seen"], DATA["unseen"]; rng = np.random.default_rng(SEED); cl = np.unique(S["clip"]); tr_clips = set(rng.permutation(cl)[: int(0.6 * len(cl))]); tr_ex = np.array([c in tr_clips for c in S["clip"]]); tr = np.repeat(tr_ex, 32)
for feat in ("slot", "feat"):
    XS = load("seen", ["token", f"{feat}_9"])[f"{feat}_9"].reshape(len(S["token"]) * 32, -1).astype(np.float32); XU = load("unseen", ["token", f"{feat}_9"])[f"{feat}_9"].reshape(len(U["token"]) * 32, -1).astype(np.float32)
    sc = StandardScaler().fit(XS[tr]); XS = sc.transform(XS); XU = sc.transform(XU); sub = np.where(tr)[0]; sub = rng.permutation(sub)[:120000]     # 120,000 training plans, drawn from the training clips only
    for t in TERMS:
        yS = labels(S["grade_9"], t); yU = labels(U["grade_9"], t)
        if min(yS[sub].sum(), (~yS[sub]).sum(), yU.sum(), (~yU).sum(), yS[~tr].sum(), (~yS[~tr]).sum()) < 200: P.append(dict(features=feat, check=t, note="skipped: a class has < 200 plans")); continue
        m = LogisticRegression(C=0.1, max_iter=300).fit(XS[sub], yS[sub]); r = dict(features=feat, check=t, needs_map=t not in MAPFREE, train_plans=len(sub), train_clips=len(tr_clips))
        r["train"], _, _ = auc(m.decision_function(XS[sub]), yS[sub]), 0, 0
        r["seen_heldout"], r["seen_heldout_lo"], r["seen_heldout_hi"] = auc_boot(m.decision_function(XS[~tr]), yS[~tr], np.repeat(S["clip"], 32)[~tr]); r["unseen"], r["unseen_lo"], r["unseen_hi"] = auc_boot(m.decision_function(XU), yU, np.repeat(U["clip"], 32))
        P.append(r); print(feat, t, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    del XS, XU
pd.DataFrame(P).to_csv(HERE / "table_b_probes.csv", index=False); print(pd.DataFrame(P).round(3).to_string(index=False))
