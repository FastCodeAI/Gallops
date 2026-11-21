"""Link-2 test, the probes (CPU only: taskset -c 0-1, nice 19 via run_model.sh; no model is loaded here). Reads the replays written by
s10_link2.py (runs/link2/feats_<slice>.npz, mem_<checkpoint>_<slice>.npz). reports/LINK2_PREREG.md fixes everything below before any probe ran.

MOMENTS  those with >= 4 earlier moments in their chain (the 4-moment warm-up); on the picking slice also in_picking.
TARGETS  from the moment's human future traj (8, 3) = (x, y, heading) at 0.5 s steps in the ego frame, waypoint j at (j + 1) * 0.5 s: targets().
FEATURES P = [ego row (accelerations zeroed) | 32 block means]; H = ego history over the past 4 s (history()); memory blocks W / C (16 x 256,
         written after the update at t, wiped / carried), Cs (256, the scorer's summary of the memory read at t in carry mode).
PROBE    one setup for every target and feature set (fit_fold):
         * low-dimensional features (ego row, ego history, intercept) enter UNPENALISED, so the ego-history baseline is as strong as a linear
           fit allows and is not shrunk away beside 8,192 image features;
         * high-dimensional blocks (block means, memory) are standardised with the training fold's mean / sd, each block divided by
           sqrt(its dimension) (equal weight per block), and penalised with one ridge strength lambda;
         * regression targets: ridge (kernel form, exact); T4: L2-penalised logistic regression on the same features (top R_LOGIT principal
           directions of the same residualised kernel + the unpenalised block);
         * outer 5 folds by clip; lambda chosen inside each training fold by 5-fold cross-validation over clips (ridge: exact
           leave-clips-out formula for a linear smoother; logistic: refits), per target;
         * measure: gain = 1 - error(A) / error(B) on the out-of-fold predictions, error = mean squared error (regression, recall control)
           or log-loss (T4); AUC for T4 also; 95 % interval = percentiles of a clip-clustered bootstrap, 2,000 draws.
CONTROLS positive: target = the standardised 32 block means of the moment 2 s earlier (the recall lesson's 2 s target), PH+C vs PH+W.
         negative: the carried memories shuffled across clips (shuffle_across_clips), PH+Cshuf vs PH+W."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "out"
RUN = Path("{experiment_store}/runs/link2"); PDIR = RUN / "probe"
WARM, N_OUTER, N_INNER, FOLD_SEED, BOOT, BOOT_SEED, SHUF_SEED = 4, 5, 5, 20261111, 2000, 20261112, 20261113
LAMS = [10.0 ** e for e in np.arange(-3, 3.01, 0.5)]                 # ridge strengths (kernel diagonal ~ number of penalised blocks)
LAMS_LOGIT = [10.0 ** e for e in np.arange(-2, 3.01, 1.0)]; R_LOGIT = 600
REG = ("T1", "T2", "T3", "T5", "T6"); ALL_T = ("T1", "T2", "T3", "T4", "T5", "T6")
TNAME = {"T1": "speed at +1 s (m/s)", "T2": "speed at +2 s (m/s)", "T3": "speed at +3 s (m/s)", "T4": "braking onset within 3 s (0/1)",
         "T5": "lateral offset at +3 s (m)", "T6": "heading change at +3 s (rad)"}
DT = 0.5
f64 = torch.float64


# ---------------------------------------------------------------------------------------------------------------- targets and features
def targets(traj: np.ndarray) -> dict:
    """traj (n, 8, 3). Waypoint index j is the pose at (j + 1) * 0.5 s in the ego frame of the moment."""
    xy = traj[:, :, :2]
    speed_at = lambda k: np.linalg.norm(xy[:, 2 * k] - xy[:, 2 * k - 2], axis=1) / (2 * DT)   # centred difference: poses at k + 0.5 s and k - 0.5 s
    seg = np.concatenate([xy[:, :1], np.diff(xy, axis=1)], 1)                                 # displacement of each 0.5 s segment (first from the origin)
    s = np.linalg.norm(seg[:, :6], axis=2) / DT                                               # mean speed of the 6 segments up to 3.0 s (centres 0.25 ... 2.75 s)
    dec = (s[:, :-1] - s[:, 1:]) / DT                                                         # mean deceleration between consecutive segment centres (5 spans of 0.5 s)
    head = (traj[:, 5, 2] + np.pi) % (2 * np.pi) - np.pi
    return {"T1": speed_at(1), "T2": speed_at(2), "T3": speed_at(3), "T4": (dec >= 2.0).any(1).astype(np.float64),
            "T5": traj[:, 5, 1].astype(np.float64), "T6": head.astype(np.float64)}


def history(chain: np.ndarray, ego: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """ego history over the past 4 s for every moment (rows are the clip's moments in order, chain = moments since the chain start).
    For k = 1 ... 8 (t - 0.5 k s): the pose-derived mean speed and yaw rate over [t - 0.5 k, t - 0.5 (k - 1)] (the earlier moment's
    logged future waypoint 0 = the next logged pose in its frame), the logged speed at t - 0.5 k (ego row), and a flag "this lag exists".
    Lags before the chain start (chain < k, only k = 5 ... 8 after the warm-up) are zero with the flag 0: the memory cannot have seen them either."""
    n = len(chain); H = np.zeros((n, 32)); idx = np.arange(n)
    for k in range(1, 9):
        ok = chain >= k; j = idx[ok] - k
        H[ok, k - 1] = np.linalg.norm(traj[j, 0, :2], axis=1) / DT
        H[ok, 8 + k - 1] = traj[j, 0, 2] / DT
        H[ok, 16 + k - 1] = np.linalg.norm(ego[j, 3:5], axis=1)
        H[ok, 24 + k - 1] = 1.0
    return H


def shuffle_across_clips(clip: np.ndarray, seed: int = SHUF_SEED) -> np.ndarray:
    """index map for the negative control: every moment takes the memory of a moment of ANOTHER clip (clips permuted in one cycle; the
    j-th kept moment of a clip takes the (j mod n)-th kept moment of its partner). P and H stay the moment's own."""
    clips = list(dict.fromkeys(clip)); rng = np.random.RandomState(seed); order = rng.permutation(len(clips))
    partner = {clips[order[i]]: clips[order[(i + 1) % len(clips)]] for i in range(len(clips))}
    rows = {c: np.flatnonzero(clip == c) for c in clips}; out = np.empty(len(clip), dtype=np.int64)
    for c in clips:
        src = rows[partner[c]]; out[rows[c]] = src[np.arange(len(rows[c])) % len(src)]
    assert (clip[out] != clip).all()
    return out


def fold_of(clip: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    clips = list(dict.fromkeys(clip)); rng = np.random.RandomState(seed); order = rng.permutation(len(clips))
    f = {clips[order[i]]: i % n_folds for i in range(len(clips))}
    return np.array([f[c] for c in clip])


# ---------------------------------------------------------------------------------------------------------------- the probe
def std_block(Xtr, Xte):
    """standardise with the training fold's mean / sd; constant columns -> 0; divide by sqrt(dimension)."""
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True); keep = (sd > 1e-6).to(Xtr.dtype); sd = sd.clamp_min(1e-6)
    d = Xtr.shape[1] ** 0.5
    return (Xtr - mu) / sd * keep / d, (Xte - mu) / sd * keep / d


def block_kernel(Xtr, Xte):
    a, b = std_block(Xtr, Xte); a = a.to(f64); b = b.to(f64)
    return a @ a.T, b @ a.T


def low_dim(Ztr, Zte):
    """unpenalised block: standardise, add the intercept, orthonormal basis of the training columns (rank-revealing); same map for test."""
    mu = Ztr.mean(0, keepdim=True); sd = Ztr.std(0, keepdim=True); ok = sd[0] > 1e-8
    A = torch.cat([torch.ones(len(Ztr), 1, dtype=f64), ((Ztr - mu) / sd.clamp_min(1e-8))[:, ok]], 1)
    B = torch.cat([torch.ones(len(Zte), 1, dtype=f64), ((Zte - mu) / sd.clamp_min(1e-8))[:, ok]], 1)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False); r = S > S[0] * 1e-8
    T = Vh[r].T / S[r]                                         # A @ T = U[:, r] (orthonormal)
    return A @ T, B @ T


def logistic_fit(F, y, pen, w0=None, iters=30):
    """Newton / IRLS for sum log-loss + 0.5 * sum(pen * w^2)."""
    w = torch.zeros(F.shape[1], dtype=f64) if w0 is None else w0.clone()
    for _ in range(iters):
        p = torch.sigmoid(F @ w); g = F.T @ (p - y) + pen * w
        Hs = (F * (p * (1 - p)).clamp_min(1e-10)[:, None]).T @ F + torch.diag(pen)
        step = torch.linalg.solve(Hs, g); w = w - step
        if float(step.abs().max()) < 1e-7:
            break
    return w


def fit_fold(Ztr, Zte, Ktr, Kte, Y: dict, groups: np.ndarray, Yrec=None) -> dict:
    """one outer fold. Ztr / Zte low-dimensional (float64), Ktr (n, n) and Kte (m, n) the summed block kernels, Y {target: (n,) train values},
    groups (n,) inner fold of every training moment (by clip), Yrec (n, D) the recall-control target or None.
    Returns test predictions per target, the chosen lambdas, and (recall) the per-test-moment prediction (m, D)."""
    n = Ktr.shape[0]; Q, Qte = low_dim(Ztr, Zte)
    KQ = Ktr @ Q; Kt = Ktr - Q @ KQ.T - KQ @ Q.T + Q @ (Q.T @ KQ) @ Q.T                       # P K P, P = I - Q Q^T
    kte = Kte - Qte @ KQ.T; kte = kte - (kte @ Q) @ Q.T                                        # test rows residualised on Z with the training fit, then P
    s, U = torch.linalg.eigh((Kt + Kt.T) / 2); keep = s > s.max() * 1e-10; s, U = s[keep], U[:, keep]
    gi = [torch.from_numpy(np.flatnonzero(groups == g)) for g in range(N_INNER)]
    Ug, Qg = [U[i] for i in gi], [Q[i] for i in gi]
    out = {"pred": {}, "lam": {}, "cv": {}}
    Yreg = torch.stack([torch.as_tensor(Y[t], dtype=f64) for t in REG], 1)                    # (n, 5)
    UtY, QtY = U.T @ Yreg, Q.T @ Yreg
    cv = torch.zeros(len(LAMS), Yreg.shape[1], dtype=f64); cvr = torch.zeros(len(LAMS), dtype=f64)
    if Yrec is not None:
        Yrec = Yrec.to(f64); UtR, QtR = U.T @ Yrec, Q.T @ Yrec
    for li, lam in enumerate(LAMS):
        sh = s / (s + lam)
        for g in range(N_INNER):                               # exact leave-group-out residuals of a linear smoother: (I - H_gg)^-1 (y - H y)_g
            Hgg = Qg[g] @ Qg[g].T + (Ug[g] * sh) @ Ug[g].T; A = torch.eye(len(gi[g]), dtype=f64) - Hgg
            res = Yreg[gi[g]] - Qg[g] @ QtY - (Ug[g] * sh) @ UtY
            cv[li] += (torch.linalg.solve(A, res) ** 2).sum(0)
            if Yrec is not None:
                resr = Yrec[gi[g]] - Qg[g] @ QtR - (Ug[g] * sh) @ UtR
                cvr[li] += float((torch.linalg.solve(A, resr) ** 2).sum())
    best = cv.argmin(0)
    for ti, t in enumerate(REG):
        lam = LAMS[int(best[ti])]; alpha = U @ (UtY[:, ti] / (s + lam))
        out["pred"][t] = (Qte @ QtY[:, ti] + kte @ alpha).numpy(); out["lam"][t] = lam; out["cv"][t] = (cv[:, ti] / n).tolist()
    if Yrec is not None:
        lam = LAMS[int(cvr.argmin())]; out["lam"]["recall"] = lam
        out["pred_recall"] = (Qte @ QtR + kte @ (U @ (UtR / (s + lam)[:, None]))).to(torch.float32).numpy()
    # T4: logistic on [Q | top principal directions of the residualised kernel]
    r = min(R_LOGIT, len(s)); sr, Ur = s[-r:], U[:, -r:]
    Ftr = torch.cat([Q, Ur * sr.sqrt()], 1); Fte = torch.cat([Qte, (kte @ Ur) / sr.sqrt()], 1); y = torch.as_tensor(Y["T4"], dtype=f64)
    nq = Q.shape[1]; cvl = []
    for lam in LAMS_LOGIT:
        pen = torch.cat([torch.full((nq,), 1e-6, dtype=f64), torch.full((r,), lam, dtype=f64)]); tot = 0.0
        for g in range(N_INNER):
            m = torch.ones(n, dtype=torch.bool); m[gi[g]] = False
            w = logistic_fit(Ftr[m], y[m], pen); p = torch.sigmoid(Ftr[gi[g]] @ w).clamp(1e-7, 1 - 1e-7)
            tot += float(-(y[gi[g]] * p.log() + (1 - y[gi[g]]) * (1 - p).log()).sum())
        cvl.append(tot / n)
    lam = LAMS_LOGIT[int(np.argmin(cvl))]; pen = torch.cat([torch.full((nq,), 1e-6, dtype=f64), torch.full((r,), lam, dtype=f64)])
    out["pred"]["T4"] = torch.sigmoid(Fte @ logistic_fit(Ftr, y, pen)).numpy(); out["lam"]["T4"] = lam; out["cv"]["T4"] = cvl
    return out


# ---------------------------------------------------------------------------------------------------------------- data
def load_slice(sl: str) -> dict:
    f = np.load(RUN / f"feats_{sl}.npz"); keep = f["chain"] >= WARM
    if sl == "picking":
        keep &= f["in_picking"]
    clip = f["clip"]; ev = pd.read_parquet(OUT / ("memory_moments_picking.parquet" if sl == "picking" else "memory_moments_train.parquet"))
    evset = set(zip(ev.clip_id, np.round(ev.anchor_t.to_numpy().astype(float), 3)))
    is_ev = np.array([(c, round(float(t), 3)) in evset for c, t in zip(clip, f["anchor_t"])])
    H = history(f["chain"], f["ego"].astype(np.float64), f["traj"]); T = targets(f["traj"])
    st = torch.load(OUT / "recall_target_stats.pt", map_location="cpu", weights_only=False)
    mu = st["mu"].reshape(1, -1).float().numpy(); sd = st["sd"].reshape(1, -1).float().numpy()
    z = (f["bm"] - mu) / sd; rows = np.flatnonzero(keep); assert (f["chain"][rows] >= 4).all()
    return {"keep": keep, "clip": clip[keep], "event": is_ev[keep], "ego": f["ego"][keep].astype(np.float64), "H": H[keep], "bm": f["bm"][keep],
            "T": {k: v[keep] for k, v in T.items()}, "recall_2s": z[rows - 4].astype(np.float32), "chain": f["chain"][keep], "n_all": int(len(keep))}


def config_blocks(D: dict, sl: str, cfg: str, shuf: np.ndarray):
    """cfg -> (Z low-dimensional, list of extra penalised blocks beside the block means)."""
    parts = cfg.split("+"); base = parts[0]; Z = D["ego"] if base == "P" else np.concatenate([D["ego"], D["H"]], 1); extra = []
    if len(parts) > 1:
        kind, _, ck = parts[1].partition("@"); ck = ck or "build5"
        m = np.load(RUN / f"mem_{ck}_{sl}.npz")
        key = {"W": "W", "C": "C", "Cs": "Cs_read", "Cshuf": "C"}[kind]; X = m[key][D["keep"]]
        if kind == "Cshuf":
            X = X[shuf]
        extra.append(X)
    return Z, extra


def run_slice(sl: str, cfgs: list, recall_cfgs: set):
    PDIR.mkdir(parents=True, exist_ok=True); D = load_slice(sl); n = len(D["clip"]); print(time.strftime("%F %T"), sl, "moments", n, "clips", len(set(D["clip"])), flush=True)
    outer = fold_of(D["clip"], N_OUTER, FOLD_SEED); shuf = shuffle_across_clips(D["clip"])
    todo = [c for c in cfgs if not (PDIR / f"{sl}__{c.replace('+', '_').replace('@', '-')}.npz").exists()]
    if not todo:
        return
    preds = {c: {t: np.full(n, np.nan) for t in ALL_T} for c in todo}; rec_err = {c: np.full(n, np.nan) for c in todo if c in recall_cfgs}
    lams = {c: [] for c in todo}; bm = torch.from_numpy(D["bm"])
    for fo in range(N_OUTER):
        te = np.flatnonzero(outer == fo); tr = np.flatnonzero(outer != fo); t0 = time.time()
        inner = fold_of(D["clip"][tr], N_INNER, FOLD_SEED + 1 + fo)
        Kimg, Kimg_te = block_kernel(bm[tr], bm[te]); Ytr = {t: D["T"][t][tr] for t in ALL_T}
        for c in todo:
            t1 = time.time(); Z, extra = config_blocks(D, sl, c, shuf); Ktr, Kte = Kimg.clone(), Kimg_te.clone()
            for X in extra:
                X = torch.from_numpy(np.ascontiguousarray(X)); a, b = block_kernel(X[tr], X[te]); Ktr += a; Kte += b
            Yrec = torch.from_numpy(D["recall_2s"][tr]) if c in recall_cfgs else None
            r = fit_fold(torch.from_numpy(Z[tr]).to(f64), torch.from_numpy(Z[te]).to(f64), Ktr, Kte, Ytr, inner, Yrec)
            for t in ALL_T:
                preds[c][t][te] = r["pred"][t]
            if Yrec is not None:
                rec_err[c][te] = ((r["pred_recall"] - D["recall_2s"][te]) ** 2).mean(1)
            lams[c].append(r["lam"]); print(time.strftime("%F %T"), sl, f"fold {fo}", c, f"{time.time() - t1:.0f} s", {k: v for k, v in r["lam"].items()}, flush=True)
        print(time.strftime("%F %T"), sl, f"fold {fo} done {time.time() - t0:.0f} s", flush=True)
    for c in todo:
        assert all(np.isfinite(preds[c][t]).all() for t in ALL_T)
        np.savez(PDIR / f"{sl}__{c.replace('+', '_').replace('@', '-')}.npz", lams=json.dumps(lams[c]), **{f"pred_{t}": preds[c][t] for t in ALL_T},
                 **({"recall_err": rec_err[c]} if c in rec_err else {}))


# ---------------------------------------------------------------------------------------------------------------- measures
def errors(pred: dict, T: dict) -> dict:
    e = {t: (pred[t] - T[t]) ** 2 for t in REG}; p = np.clip(pred["T4"], 1e-7, 1 - 1e-7)
    e["T4"] = -(T["T4"] * np.log(p) + (1 - T["T4"]) * np.log(1 - p))
    return e


def wauc(score, y, w):
    o = np.argsort(score, kind="mergesort"); y, w = y[o], w[o]; neg = np.cumsum(w * (1 - y)); pos = w * y
    d = pos.sum() * neg[-1]
    return float((pos * neg).sum() / d) if d > 0 else float("nan")


def gain_ci(eA, eB, clip, mask, draws):
    """gain of A over B = 1 - sum(eA) / sum(eB) on the masked moments; clip-clustered bootstrap percentiles."""
    cl, inv = np.unique(clip[mask], return_inverse=True); a = np.bincount(inv, eA[mask]); b = np.bincount(inv, eB[mask]); k = len(cl)
    if k < 2 or b.sum() <= 0:
        return {"gain": None, "clips": int(k), "moments": int(mask.sum())}
    idx = draws.randint(0, k, size=(BOOT, k)); g = 1 - a[idx].sum(1) / b[idx].sum(1)
    q = lambda p: float(np.percentile(g, p))
    return {"gain": float(1 - a.sum() / b.sum()), "lo": q(2.5), "hi": q(97.5), "lo_bonf6": q(100 * 0.025 / 6), "hi_bonf6": q(100 * (1 - 0.025 / 6)),
            "err_A": float(a.sum() / mask.sum()), "err_B": float(b.sum() / mask.sum()), "clips": int(k), "moments": int(mask.sum())}


def auc_ci(pA, pB, y, clip, mask, draws):
    cl, inv = np.unique(clip[mask], return_inverse=True); k = len(cl); yA = y[mask]
    if k < 2 or yA.min() == yA.max():
        return {"auc_A": None, "auc_B": None}
    one = np.ones(mask.sum()); out = {"auc_A": wauc(pA[mask], yA, one), "auc_B": wauc(pB[mask], yA, one)}; d = []
    for _ in range(BOOT):
        w = np.bincount(draws.randint(0, k, size=k), minlength=k)[inv].astype(float); d.append(wauc(pA[mask], yA, w) - wauc(pB[mask], yA, w))
    d = np.array(d); d = d[np.isfinite(d)]
    out.update(diff=out["auc_A"] - out["auc_B"], diff_lo=float(np.percentile(d, 2.5)), diff_hi=float(np.percentile(d, 97.5)))
    return out


def summarise(sl: str, pairs: list, recall_pairs: list) -> dict:
    D = load_slice(sl); T = D["T"]; n = len(D["clip"]); res = {"moments": n, "clips": len(set(D["clip"])), "event_moments": int(D["event"].sum()),
                                                                 "event_clips": len(set(D["clip"][D["event"]])), "T4_rate": float(T["T4"].mean()),
                                                                 "T4_rate_event": float(T["T4"][D["event"]].mean()) if D["event"].any() else None,
                                                                 "target_sd": {t: float(np.std(T[t])) for t in REG}, "pairs": {}, "sets": {}}
    ld = lambda c: np.load(PDIR / f"{sl}__{c.replace('+', '_').replace('@', '-')}.npz")
    masks = {"all": np.ones(n, bool), "event": D["event"]}; E = {}
    def get(c):
        if c not in E:
            z = ld(c); pred = {t: z[f"pred_{t}"] for t in ALL_T}; E[c] = (pred, errors(pred, T), z)
            res["sets"][c] = {"lambdas": json.loads(str(z["lams"])), **{t: {"mse" if t != "T4" else "log_loss": float(E[c][1][t].mean()),
                              **({"r2": float(1 - E[c][1][t].mean() / np.var(T[t]))} if t != "T4" else {"auc": wauc(pred["T4"], T["T4"], np.ones(n))})} for t in ALL_T}}
        return E[c]
    for A, B in pairs:
        pa, ea, _ = get(A); pb, eb, _ = get(B); r = {}
        for mk, m in masks.items():
            r[mk] = {}
            for t in ALL_T:
                r[mk][t] = gain_ci(ea[t], eb[t], D["clip"], m, np.random.RandomState(BOOT_SEED))
            r[mk]["T4_auc"] = auc_ci(pa["T4"], pb["T4"], T["T4"], D["clip"], m, np.random.RandomState(BOOT_SEED))
        res["pairs"][f"{A} vs {B}"] = r
    res["recall"] = {}
    for A, B in recall_pairs:
        za, zb = ld(A), ld(B)
        res["recall"][f"{A} vs {B}"] = {mk: gain_ci(za["recall_err"], zb["recall_err"], D["clip"], m, np.random.RandomState(BOOT_SEED)) for mk, m in masks.items()}
    return res


# ---------------------------------------------------------------------------------------------------------------- self-test (synthetic)
def selftest():
    """synthetic data only: (1) the kernel solution equals the primal ridge with unpenalised Z; (2) the leave-group-out formula equals
    explicit refits; (3) a planted memory signal is found and vanishes when the memory is shuffled across clips."""
    rng = np.random.RandomState(0); n, m, dz, dx = 400, 100, 6, 900
    Z = rng.randn(n + m, dz); X = rng.randn(n + m, dx); y = Z @ rng.randn(dz) + X @ rng.randn(dx) * 0.05 + rng.randn(n + m) * 0.5
    Zt, Xt = torch.from_numpy(Z), torch.from_numpy(X); Ktr, Kte = block_kernel(Xt[:n], Xt[n:]); groups = np.arange(n) % N_INNER
    Y = {t: y[:n] for t in REG}; Y["T4"] = (y[:n] > 0).astype(float)
    r = fit_fold(Zt[:n], Zt[n:], Ktr, Kte, Y, groups); lam = r["lam"]["T1"]
    a, b = std_block(Xt[:n], Xt[n:]); Q, Qte = low_dim(Zt[:n], Zt[n:]); F = torch.cat([Q, a], 1); Fte = torch.cat([Qte, b], 1)
    pen = torch.cat([torch.zeros(Q.shape[1], dtype=f64), torch.full((dx,), lam, dtype=f64)]); yt = torch.from_numpy(y[:n])
    w = torch.linalg.solve(F.T @ F + torch.diag(pen), F.T @ yt); d1 = float((Fte @ w - torch.from_numpy(r["pred"]["T1"])).abs().max())
    li = LAMS.index(lam); tot = 0.0
    for g in range(N_INNER):
        mk = torch.from_numpy(groups != g); wg = torch.linalg.solve(F[mk].T @ F[mk] + torch.diag(pen), F[mk].T @ yt[mk]); tot += float(((F[~mk] @ wg - yt[~mk]) ** 2).sum())
    d2 = abs(tot / n - r["cv"]["T1"][li])
    # planted signal: 60 clips x 20 moments, the "memory" holds a clip-level latent that drives the target
    nc, per = 60, 20; clip = np.repeat(np.arange(nc), per).astype(str); lat = np.repeat(rng.randn(nc, 8), per, 0) + 0.3 * rng.randn(nc * per, 8)
    Mem = lat @ rng.randn(8, 200) + rng.randn(nc * per, 200); Wm = rng.randn(nc * per, 200); Xi = rng.randn(nc * per, 300); Zz = rng.randn(nc * per, 4)
    yy = Zz @ rng.randn(4) + lat @ rng.randn(8) + rng.randn(nc * per) * 0.5; outer = fold_of(clip, N_OUTER, 1); sh = shuffle_across_clips(clip, 5)
    P = {k: np.zeros(nc * per) for k in ("W", "C", "S")}
    for fo in range(N_OUTER):
        te, tr = np.flatnonzero(outer == fo), np.flatnonzero(outer != fo); inner = fold_of(clip[tr], N_INNER, 2 + fo); Ki, Kit = block_kernel(torch.from_numpy(Xi[tr]), torch.from_numpy(Xi[te]))
        for k, Xm in (("W", Wm), ("C", Mem), ("S", Mem[sh])):
            aa, bb = block_kernel(torch.from_numpy(Xm[tr]), torch.from_numpy(Xm[te])); Yd = {t: yy[tr] for t in REG}; Yd["T4"] = (yy[tr] > 0).astype(float)
            P[k][te] = fit_fold(torch.from_numpy(Zz[tr]), torch.from_numpy(Zz[te]), Ki + aa, Kit + bb, Yd, inner)["pred"]["T1"]
    e = {k: (v - yy) ** 2 for k, v in P.items()}; al = np.ones(nc * per, bool)
    g1 = gain_ci(e["C"], e["W"], clip, al, np.random.RandomState(1)); g2 = gain_ci(e["S"], e["W"], clip, al, np.random.RandomState(1))
    ok = d1 < 1e-6 and d2 < 1e-6 and g1["lo"] > 0.2 and g2["hi"] < 0.1
    print(json.dumps({"kernel_vs_primal_max_abs_diff": d1, "leave_group_out_formula_vs_refit_abs_diff": d2, "planted_gain_C_over_W": g1, "shuffled_gain": g2, "ok": bool(ok)}, indent=1))
    (OUT / "link2_selftest.json").write_text(json.dumps({"at": time.strftime("%F %T"), "kernel_vs_primal_max_abs_diff": d1, "leave_group_out_formula_vs_refit_abs_diff": d2,
                                                          "planted_gain_C_over_W": g1, "shuffled_gain": g2, "ok": bool(ok)}, indent=1) + "\n")
    assert ok


CFG_MAIN = ["P", "PH", "PH+W", "PH+C", "PH+Cs", "PH+Cshuf"]; CFG_OLD = ["PH+W@build4", "PH+C@build4", "PH+W@pilot", "PH+C@pilot"]
RECALL = {"PH", "PH+W", "PH+C", "PH+W@build4", "PH+C@build4", "PH+W@pilot", "PH+C@pilot"}
PAIRS = [("PH", "P"), ("PH+W", "PH"), ("PH+C", "PH+W"), ("PH+C", "PH"), ("PH+Cs", "PH+W"), ("PH+Cs", "PH"), ("PH+Cshuf", "PH+W"), ("PH+Cshuf", "PH"),
         ("PH+C@build4", "PH+W@build4"), ("PH+C@pilot", "PH+W@pilot")]
RECALL_PAIRS = [("PH+C", "PH+W"), ("PH+W", "PH"), ("PH+C@build4", "PH+W@build4"), ("PH+C@pilot", "PH+W@pilot")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "stats", "run", "summary"]); a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "stats":                                   # target statistics only (no probe): written into the pre-registration
        o = {}
        for sl in ("primary", "picking"):
            D = load_slice(sl); o[sl] = {"moments_replayed": D["n_all"], "moments_after_warm_up": len(D["clip"]), "clips": len(set(D["clip"])), "event_moments": int(D["event"].sum()),
                                         "event_clips": len(set(D["clip"][D["event"]])), "T4_rate": float(D["T"]["T4"].mean()),
                                         "moments_with_full_4s_history": int((D["chain"] >= 8).sum()),
                                         "target_mean_sd": {t: [float(D["T"][t].mean()), float(D["T"][t].std())] for t in ALL_T}}
        (OUT / "link2_stats.json").write_text(json.dumps(o, indent=1) + "\n"); print(json.dumps(o, indent=1))
    elif a.cmd == "run":
        for sl in ("picking", "primary"):
            run_slice(sl, CFG_MAIN + CFG_OLD, RECALL)
        print("PROBES DONE", flush=True)
    else:
        res = {"made": time.strftime("%F %T"), **{sl: summarise(sl, PAIRS, RECALL_PAIRS) for sl in ("primary", "picking")}}
        (OUT / "link2_results.json").write_text(json.dumps(res, indent=1) + "\n"); print("written", OUT / "link2_results.json")
