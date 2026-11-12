"""Ceiling test, amendment (information only, reports/CEILING_PREREG.md "Amendment 1"): the same MLP reader as s11_ceiling.py with a longer
training budget (at most 400 epochs, patience 30 instead of 60 / 6), because the training log of fold 0 showed the speed targets still
improving at the 60-epoch cap. Everything else (sample, folds, directions, inputs, seeds, per-target early stopping) is s11_ceiling's, by import.
  s11_mlp_long.py run       -> runs/ceiling/probe/mlplong_fold<k>.npz
  s11_mlp_long.py summary   -> out/ceiling_results_mlp_long.json"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s11_ceiling as C
L2 = C.L2
C.MAX_EP, C.PATIENCE = 400, 30
PAIRS = [("PH+V24", "PH"), ("PH+V24shuf", "PH"), ("PHf", "PH"), ("PHf+Vf24", "PHf")]


def run():
    D = C.load(); rows = D["rows"]; outer = L2.fold_of(D["chunk"], C.N_OUTER, C.FOLD_SEED); shuf = L2.shuffle_across_clips(D["clip"], C.SHUF_SEED)
    vrows = {"now": rows, "v2": rows - 4, "v4": rows - 8, "v2s": rows[shuf] - 4, "v4s": rows[shuf] - 8}
    X = {"coarse": np.load(C.RUN / "coarse.npy"), "fine": np.load(C.RUN / "fine.npy")}; low = np.concatenate([D["ego"], D["H"]], 1)
    for fo in range(C.N_OUTER):
        dst = C.PDIR / f"mlplong_fold{fo}.npz"
        if dst.exists():
            continue
        if C.avail_gb() < 20:
            raise SystemExit("available RAM below 20 GB: stopping")
        t0 = time.time(); te = np.flatnonzero(outer == fo); tr = np.flatnonzero(outer != fo); val = L2.fold_of(D["chunk"][tr], C.N_INNER, C.FOLD_SEED + 1 + fo) == 0
        inp = {}
        for fam, exact in (("coarse", True), ("fine", False)):
            S, _ = C.pca_scores(X[fam], rows[tr], C.K_PCA, exact, seed=fo); sdk = S[rows[tr], :C.K_MLP].std(0, keepdims=True)
            for v in (vrows if fam == "coarse" else ("now", "v2", "v4")):
                inp[(fam, v)] = (S[vrows[v][tr], :C.K_MLP] / sdk, S[vrows[v][te], :C.K_MLP] / sdk)
            del S
        ym = np.stack([D["T"][t][tr] for t in C.REG], 1); mu, sd = ym.mean(0), ym.std(0)
        Ym = torch.from_numpy(np.concatenate([(ym - mu) / sd, D["T"]["T4"][tr][:, None]], 1).astype(np.float32)); ltr, lte = C.std_low(low[tr], low[te]); res = {"te": te}; info = {}
        for cfg in C.MLP:
            t1 = time.time(); _, fam, views = C.LIN[cfg]
            Xa = torch.from_numpy(np.concatenate([ltr] + [inp[(fam, v)][0] for v in views], 1).astype(np.float32)); Xb = torch.from_numpy(np.concatenate([lte] + [inp[(fam, v)][1] for v in views], 1).astype(np.float32))
            outs, eps = [], []
            for s in range(C.N_SEEDS):
                o, eb, ne = C.mlp_fold(Xa, Ym, Xb, val, 1000 * fo + s); outs.append(o); eps.append((eb, ne))
            o = np.mean(outs, 0)
            for i, t in enumerate(C.REG):
                res[f"{cfg}|{t}"] = o[:, i] * sd[i] + mu[i]
            res[f"{cfg}|T4"] = o[:, 5]; info[cfg] = eps; C.log(f"fold {fo} long mlp {cfg} {time.time() - t1:.0f} s best epochs / epochs run", eps)
        np.savez(dst, info=json.dumps(info), **res); C.log(f"fold {fo} done {time.time() - t0:.0f} s")
    print("MLP LONG DONE", flush=True)


def summary():
    D = C.load(); n = len(D["rows"]); T = D["T"]; P = {}; infos = []
    for fo in range(C.N_OUTER):
        z = np.load(C.PDIR / f"mlplong_fold{fo}.npz"); infos.append(json.loads(str(z["info"])))
        for k in z.files:
            if "|" in k:
                P.setdefault(k, np.full(n, np.nan))[z["te"]] = z[k]
    def err(cfg, t):
        p = P[f"{cfg}|{t}"]
        if t == "T4":
            p = np.clip(p, 1e-7, 1 - 1e-7); return -(T[t] * np.log(p) + (1 - T[t]) * np.log(1 - p))
        return (p - T[t]) ** 2
    masks = {"all": np.ones(n, bool), "event": D["event"]}
    res = {"made": time.strftime("%F %T"), "max_epochs": C.MAX_EP, "patience": C.PATIENCE, "folds": infos,
           "sets": {c: {t: float(err(c, t).mean()) for t in C.ALL_T} for c in C.MLP},
           "pairs": {f"{A} vs {B}": {mk: {t: L2.gain_ci(err(A, t), err(B, t), D["chunk"], m, np.random.RandomState(L2.BOOT_SEED)) for t in C.ALL_T} for mk, m in masks.items()} for A, B in PAIRS}}
    (C.OUT / "ceiling_results_mlp_long.json").write_text(json.dumps(res, indent=1) + "\n"); print("written")


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1]]()
