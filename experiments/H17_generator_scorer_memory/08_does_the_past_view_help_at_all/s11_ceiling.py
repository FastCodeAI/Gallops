"""Ceiling test (Sid, 2026-11-12): does the TRUE past camera view help predict what the driver does next, beyond the present view and the
car's own motion history? No model, no memory, no GPU. CPU only (taskset -c 0-1, nice 19 via run_model.sh). reports/CEILING_PREREG.md fixes
everything below before any probe ran. Reads the views written by s11_extract.py (runs/ceiling/{coarse.npy, fine.npy, meta.npz}).

MOMENTS  every training moment with >= 8 earlier consecutive moments in its clip (4 s of history).
TARGETS  T1-T6 = s10_probe.targets (imported: the link-2 definitions). Control target: the car's logged speed 3 s ago, |ego[t - 6, 3:5]|.
LOW-DIM  ego row (accelerations zeroed) and the 4 s ego history = s10_probe.history (imported), unpenalised, as in link 2.
VIEWS    coarse = 4 x 8 block means (8,192 values), fine = 8 x 16 block means (32,768 values); now, 1 s, 2 s and 4 s ago (rows t - 2, t - 4, t - 8).
         Inside every outer training fold: each value standardised with the fold's mean / sd of the CURRENT view, principal directions of
         the standardised current view (coarse: exact, top K_PCA; fine: randomised, top K_PCA), the same map applied to the past views,
         scores divided by sqrt(number of raw values). So the linear reader is the link-2 ridge restricted to the top K_PCA directions.
LINEAR   ridge with unpenalised low-dim block (normal equations; lambda per target by 5-fold cross-validation over chunks inside the
         training fold, the link-2 grid); T4: L2-penalised logistic on [low-dim | top R_LOGIT principal directions of the residualised
         penalised block], lambda on one inner hold-out fold (the link-2 grid).
MLP      one hidden layer (256, ReLU) plus a linear skip, inputs = standardised low-dim + first K_MLP whitened scores per view, six outputs
         (five standardised regression targets, T4 logit), Adam, early stopping per target on one inner hold-out fold, mean of N_SEEDS seeds.
FOLDS    5 outer folds by CHUNK; bootstrap clustered by chunk, 2,000 draws; gain = 1 - error(A) / error(B) (s10_probe.gain_ci)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s10_probe as L2                                                   # targets, history, fold_of, shuffle_across_clips, low_dim, gain_ci, wauc, grids

torch.set_num_threads(2)
OUT = HERE.parent / "out"; RUN = Path("{experiment_store}/runs/ceiling"); PDIR = RUN / "probe"
HIST, N_OUTER, N_INNER, FOLD_SEED, SHUF_SEED, SUB_SEED = 8, 5, 5, 20261114, 20261115, 20261116
K_PCA, K_MLP, R_LOGIT, OVER = 1024, 256, 600, 64
LAMS, LAMS_LOGIT = L2.LAMS, L2.LAMS_LOGIT
REG, ALL_T = L2.REG, L2.ALL_T; YCOLS = REG + ("past_speed",)
HID, LR, WD, BATCH, MAX_EP, PATIENCE, N_SEEDS = 256, 1e-3, 1e-4, 512, 60, 6, 2
f64 = torch.float64
# name -> (low-dim, family, views); views index into VIEW_NAMES of the family
VIEW_NAMES = {"coarse": ["now", "v1", "v2", "v4", "v2s", "v4s"], "fine": ["now", "v2", "v4"]}
LIN = {"P": ("P", "coarse", ["now"]), "PH": ("PH", "coarse", ["now"]), "PH+V2": ("PH", "coarse", ["now", "v2"]), "PH+V24": ("PH", "coarse", ["now", "v2", "v4"]),
       "PH+V124": ("PH", "coarse", ["now", "v1", "v2", "v4"]), "PH+V24shuf": ("PH", "coarse", ["now", "v2s", "v4s"]), "P+V24": ("P", "coarse", ["now", "v2", "v4"]),
       "PHf": ("PH", "fine", ["now"]), "PHf+Vf24": ("PH", "fine", ["now", "v2", "v4"])}
MLP = ["PH", "PH+V24", "PH+V24shuf", "PHf", "PHf+Vf24"]


def avail_gb():
    return [int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable")][0] / 2 ** 20


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- data
def load(subsample: int = 0) -> dict:
    m = np.load(RUN / "meta.npz"); chain = m["chain"]; clip = m["clip"]; chunk = m["chunk"]; ego = m["ego"].astype(np.float64); traj = m["traj"]
    keep = chain >= HIST
    ev = pd.read_parquet(OUT / "memory_moments_train.parquet"); evset = set(zip(ev.clip_id, np.round(ev.anchor_t.to_numpy().astype(float), 3)))
    is_ev = np.array([(c, round(float(t), 3)) in evset for c, t in zip(clip, m["anchor_t"])])
    if subsample:                                             # by clip, fixed seed: every clip holding a kept event moment + random others
        clips = np.array(sorted(set(clip[keep]))); evc = set(clip[keep & is_ev]); others = np.array([c for c in clips if c not in evc])
        rng = np.random.RandomState(SUB_SEED); take = set(evc) | set(rng.choice(others, max(0, subsample - len(evc)), replace=False))
        keep &= np.isin(clip, list(take))
    rows = np.flatnonzero(keep); H = L2.history(chain, ego, traj); T = L2.targets(traj)
    past_speed = np.linalg.norm(ego[rows - 6, 3:5], axis=1)
    assert (clip[rows - HIST] == clip[rows]).all()
    cl = clip[rows]; assert len(pd.DataFrame({"c": cl, "k": chunk[rows]}).drop_duplicates()) == len(set(cl))      # a clip lies in one chunk
    return {"rows": rows, "clip": cl, "chunk": chunk[rows], "event": is_ev[rows], "ego": ego[rows], "H": H[rows], "T": {k: v[rows] for k, v in T.items()},
            "past_speed": past_speed, "type": m["type"][rows], "source": m["source"][rows], "n_all": int(len(chain)), "clips_all": int(len(set(clip)))}


# ---------------------------------------------------------------------------------------------------------------- principal directions
def _stats(X, rows, step=4096):
    s = torch.zeros(X.shape[1], dtype=f64); ss = torch.zeros(X.shape[1], dtype=f64)
    for a in range(0, len(rows), step):
        t = torch.from_numpy(X[rows[a:a + step]].astype(np.float32)).to(f64); s += t.sum(0); ss += (t * t).sum(0)
    n = len(rows); mu = s / n; sd = ((ss - n * mu * mu) / (n - 1)).clamp_min(0).sqrt(); keep = (sd > 1e-6)
    return mu.float(), sd.clamp_min(1e-6).float(), keep.float()


def _std_chunks(X, rows, mu, sd, keep, step=4096):
    for a in range(0, len(rows), step):
        yield a, (torch.from_numpy(X[rows[a:a + step]].astype(np.float32)) - mu) / sd * keep


def pca_scores(X, tr_rows, k, exact: bool, seed=0):
    """standardise with the training rows' mean / sd, top-k principal directions of the standardised training rows, scores of ALL rows of X
    divided by sqrt(number of raw values). Returns scores (N, k) float32, the variance share of every kept direction."""
    d = X.shape[1]; n = len(tr_rows); mu, sd, keep = _stats(X, tr_rows); total = float(keep.sum())
    if exact:
        G = torch.zeros(d, d, dtype=f64)
        for _, t in _std_chunks(X, tr_rows, mu, sd, keep):
            G += (t.T @ t).to(f64)
        ev, V = torch.linalg.eigh(G / (n - 1)); V = V[:, -k:].flip(1).float(); share = (ev[-k:].flip(0) / total).numpy()
    else:                                                     # randomised: range of A, one power step, then the small problem (Halko et al. 2011)
        g = torch.Generator().manual_seed(seed); l = k + OVER; Om = torch.randn(d, l, generator=g)
        Y = torch.cat([t @ Om for _, t in _std_chunks(X, tr_rows, mu, sd, keep)]); Q, _ = torch.linalg.qr(Y); del Y
        Z = torch.zeros(d, l)
        for a, t in _std_chunks(X, tr_rows, mu, sd, keep):
            Z += t.T @ Q[a:a + len(t)]
        Qz, _ = torch.linalg.qr(Z); del Z, Q
        Tm = torch.cat([t @ Qz for _, t in _std_chunks(X, tr_rows, mu, sd, keep)])
        ev, W = torch.linalg.eigh((Tm.T @ Tm).to(f64) / (n - 1)); V = (Qz @ W[:, -k:].flip(1).float()); share = (ev[-k:].flip(0) / total).numpy(); del Tm
    allr = np.arange(X.shape[0]); S = torch.cat([t @ V for _, t in _std_chunks(X, allr, mu, sd, keep)]) / d ** 0.5
    return S.numpy(), share


# ---------------------------------------------------------------------------------------------------------------- linear reader
def gram_groups(X, Y, gi):
    """per inner group: X^T X, X^T Y, Y^T Y (float64 sums of float32 chunk products)."""
    out = []
    for i in gi:
        XX = torch.zeros(X.shape[1], X.shape[1], dtype=f64); Xy = torch.zeros(X.shape[1], Y.shape[1], dtype=f64)
        for a in range(0, len(i), 8192):
            j = i[a:a + 8192]; t = X[j]; XX += (t.T @ t).to(f64); Xy += t.to(f64).T @ Y[j]
        out.append({"XX": XX, "Xy": Xy, "yy": (Y[i] ** 2).sum(0)})
    return out


def low_groups(Q, X, Y, gi):
    return [{"QQ": Q[i].T @ Q[i], "QX": torch.cat([Q[i[a:a + 8192]].T @ X[i[a:a + 8192]].to(f64) for a in range(0, len(i), 8192)], 0).reshape(-1, Q.shape[1], X.shape[1]).sum(0),
             "Qy": Q[i].T @ Y[i]} for i in gi]


def assemble(G, Lo, cols):
    Gs, bs = [], []
    for g, lo in zip(G, Lo):
        QX = lo["QX"][:, cols]; Gs.append(torch.cat([torch.cat([lo["QQ"], QX], 1), torch.cat([QX.T, g["XX"][cols][:, cols]], 1)], 0)); bs.append(torch.cat([lo["Qy"], g["Xy"][cols]], 0))
    return Gs, bs, [g["yy"] for g in G]


def ridge_cv(Gs, bs, yys, q, lams=LAMS):
    """weights (d, m) of the ridge with the first q columns unpenalised; lambda per target column by leave-one-group-out error computed from
    the group Gram matrices: |y_g - X_g w|^2 = y_g'y_g - 2 w'X_g'y_g + w'X_g'X_g w."""
    Gt, bt = sum(Gs), sum(bs); d, m = bt.shape; cv = torch.zeros(len(lams), m, dtype=f64)
    def solve(Gm, b, lam):
        A = Gm.clone(); dg = A.diagonal(); dg[:q] += 1e-8; dg[q:] += lam
        return torch.cholesky_solve(b, torch.linalg.cholesky(A))
    for g in range(len(Gs)):
        Gtr, btr = Gt - Gs[g], bt - bs[g]
        for li, lam in enumerate(lams):
            W = solve(Gtr, btr, lam); cv[li] += yys[g] - 2 * (W * bs[g]).sum(0) + (W * (Gs[g] @ W)).sum(0)
    best = cv.argmin(0); W = torch.zeros(d, m, dtype=f64)
    for li in best.unique().tolist():
        W[:, best == li] = solve(Gt, bt, lams[li])[:, best == li]
    return W, [lams[int(b)] for b in best], cv


def logistic(F, y, pen, w0=None, iters=25, tol=1e-6):
    w = torch.zeros(F.shape[1], dtype=f64) if w0 is None else w0.clone()
    for _ in range(iters):
        p = torch.sigmoid(F @ w); g = F.T @ (p - y) + pen * w
        Hs = (F * (p * (1 - p)).clamp_min(1e-10)[:, None]).T @ F + torch.diag(pen)
        step = torch.linalg.solve(Hs, g); w = w - step
        if float(step.abs().max()) < tol:
            break
    return w


def logit_fold(Q, Qte, Xp, Xpte, Gpp, C, y, val, R=R_LOGIT):
    """T4. Q orthonormal low-dim (train), Xp penalised block (n, dp) float32, Gpp = Xp'Xp, C = Q'Xp. Features [Q | top-R principal scores of the
    block residualised on Q]; lambda on the hold-out rows `val` (bool mask), then a refit on all training rows."""
    nq = Q.shape[1]
    if Xp.shape[1] == 0:
        Ftr, Fte = Q, Qte; r = 0
    else:
        s, E = torch.linalg.eigh(Gpp - C.T @ C); r = min(R, int((s > s.max() * 1e-10).sum())); E = E[:, -r:]
        Ftr = torch.cat([Q, Xp.to(f64) @ E - Q @ (C @ E)], 1); Fte = torch.cat([Qte, Xpte.to(f64) @ E - Qte @ (C @ E)], 1)
    yt = torch.as_tensor(y, dtype=f64); v = torch.from_numpy(val); best, w = None, None; cvl = []
    lams = LAMS_LOGIT[::-1] if r else [1.0]
    for lam in lams:                                          # strongest penalty first, warm starts
        pen = torch.cat([torch.full((nq,), 1e-6, dtype=f64), torch.full((r,), lam, dtype=f64)]); w = logistic(Ftr[~v], yt[~v], pen, w)
        p = torch.sigmoid(Ftr[v] @ w).clamp(1e-7, 1 - 1e-7); ll = float(-(yt[v] * p.log() + (1 - yt[v]) * (1 - p).log()).mean()); cvl.append(ll)
        if best is None or ll < best[0]:
            best = (ll, lam, w.clone())
    pen = torch.cat([torch.full((nq,), 1e-6, dtype=f64), torch.full((r,), best[1], dtype=f64)])
    return torch.sigmoid(Fte @ logistic(Ftr, yt, pen, best[2])).numpy(), best[1], cvl


# ---------------------------------------------------------------------------------------------------------------- MLP reader
class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.lin = torch.nn.Linear(d, 6); self.h = torch.nn.Linear(d, HID); self.o = torch.nn.Linear(HID, 6); torch.nn.init.zeros_(self.o.weight); torch.nn.init.zeros_(self.o.bias)

    def forward(self, x):
        return self.lin(x) + self.o(torch.relu(self.h(x)))


def mlp_fold(Xtr, Ytr, Xte, val, seed):
    """Xtr (n, d) float32 standardised inputs, Ytr (n, 6) float32 = five standardised regression targets + T4 (0/1); val bool mask (hold-out rows).
    Returns test outputs (m, 6) taken, per target, at the epoch with the best hold-out error (regression: standardised units; T4: probability)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); v = torch.from_numpy(val); Xa, Ya, Xv, Yv = Xtr[~v], Ytr[~v], Xtr[v], Ytr[v]
    net = Net(Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD); bce = torch.nn.functional.binary_cross_entropy_with_logits
    best = torch.full((6,), float("inf")); out = torch.zeros(len(Xte), 6); ep_best = [0] * 6; since = 0
    for ep in range(MAX_EP):
        net.train(); perm = torch.randperm(len(Xa), generator=g)
        for a in range(0, len(perm), BATCH):
            j = perm[a:a + BATCH]; o = net(Xa[j]); loss = ((o[:, :5] - Ya[j, :5]) ** 2).mean(0).sum() + bce(o[:, 5], Ya[j, 5])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = net(Xv); e = torch.cat([((o[:, :5] - Yv[:, :5]) ** 2).mean(0), bce(o[:, 5], Yv[:, 5])[None]]); better = e < best
            if better.any():
                ot = net(Xte); ot[:, 5] = torch.sigmoid(ot[:, 5]); out[:, better] = ot[:, better]; best[better] = e[better]
                for t in better.nonzero().flatten().tolist():
                    ep_best[t] = ep
                since = 0
            else:
                since += 1
        if since >= PATIENCE:
            break
    return out.numpy(), ep_best, ep + 1


# ---------------------------------------------------------------------------------------------------------------- run
def std_low(Ztr, Zte):
    mu = Ztr.mean(0, keepdims=True); sd = Ztr.std(0, keepdims=True); ok = sd[0] > 1e-8
    return ((Ztr - mu) / np.maximum(sd, 1e-8))[:, ok].astype(np.float32), ((Zte - mu) / np.maximum(sd, 1e-8))[:, ok].astype(np.float32)


def run(subsample: int = 0):
    PDIR.mkdir(parents=True, exist_ok=True); D = load(subsample); n = len(D["rows"]); rows = D["rows"]
    log("moments", n, "clips", len(set(D["clip"])), "chunks", len(set(D["chunk"])), "event moments", int(D["event"].sum()), "subsample", subsample)
    outer = L2.fold_of(D["chunk"], N_OUTER, FOLD_SEED); shuf = L2.shuffle_across_clips(D["clip"], SHUF_SEED)
    vrows = {"now": rows, "v1": rows - 2, "v2": rows - 4, "v4": rows - 8, "v2s": rows[shuf] - 4, "v4s": rows[shuf] - 8}
    coarse = np.load(RUN / "coarse.npy"); fine = np.load(RUN / "fine.npy"); log("views loaded, ram", round(avail_gb()), "GB")
    Zlow = {"P": D["ego"], "PH": np.concatenate([D["ego"], D["H"]], 1)}
    Yall = torch.from_numpy(np.stack([D["T"][t] for t in REG] + [D["past_speed"]], 1)).to(f64)
    for fo in range(N_OUTER):
        dst = PDIR / f"fold{fo}.npz"
        if dst.exists():
            log("have", dst.name); continue
        if avail_gb() < 20:
            raise SystemExit("available RAM below 20 GB: stopping")
        t0 = time.time(); te = np.flatnonzero(outer == fo); tr = np.flatnonzero(outer != fo); inner = L2.fold_of(D["chunk"][tr], N_INNER, FOLD_SEED + 1 + fo)
        gi = [torch.from_numpy(np.flatnonzero(inner == g)) for g in range(N_INNER)]; val = inner == 0; res = {"te": te}; info = {"n_train": len(tr), "n_test": len(te)}
        Qs = {k: L2.low_dim(torch.from_numpy(z[tr]).to(f64), torch.from_numpy(z[te]).to(f64)) for k, z in Zlow.items()}; Ytr = Yall[tr]
        mlp_in = {}
        for fam, X, exact in (("coarse", coarse, True), ("fine", fine, False)):
            t1 = time.time(); S, share = pca_scores(X, rows[tr], K_PCA, exact, seed=fo)
            info[f"{fam}_variance_kept_{K_PCA}"] = float(share.sum()); info[f"{fam}_variance_kept_{K_MLP}"] = float(share[:K_MLP].sum())
            names = VIEW_NAMES[fam]; Xtr = torch.from_numpy(np.concatenate([S[vrows[v][tr]] for v in names], 1)); Xte = torch.from_numpy(np.concatenate([S[vrows[v][te]] for v in names], 1))
            sdk = S[rows[tr], :K_MLP].std(0, keepdims=True)
            for v in names:                                   # MLP inputs: first K_MLP scores per view, whitened with the current view's training sd
                mlp_in[(fam, v)] = (S[vrows[v][tr], :K_MLP] / sdk, S[vrows[v][te], :K_MLP] / sdk)
            del S; log(f"fold {fo} {fam} directions {time.time() - t1:.0f} s, variance kept {share.sum():.3f} (first {K_MLP}: {share[:K_MLP].sum():.3f})")
            t1 = time.time(); G = gram_groups(Xtr, Ytr, gi); Lo = {k: low_groups(Qs[k][0], Xtr, Ytr, gi) for k in (("P", "PH") if fam == "coarse" else ("PH",))}
            log(f"fold {fo} {fam} gram {time.time() - t1:.0f} s")
            for cfg, (low, f2, views) in LIN.items():
                if f2 != fam:
                    continue
                t1 = time.time(); cols = torch.cat([torch.arange(K_PCA) + K_PCA * names.index(v) for v in views]); Q, Qte = Qs[low]; q = Q.shape[1]
                Gs, bs, yys = assemble(G, Lo[low], cols); W, lam, _ = ridge_cv(Gs, bs, yys, q)
                pred = (torch.cat([Qte, Xte[:, cols].to(f64)], 1) @ W).numpy()
                for i, t in enumerate(YCOLS):
                    res[f"lin|{cfg}|{t}"] = pred[:, i]
                lams = dict(zip(YCOLS, lam))
                if cfg != "P+V24":
                    Gt = sum(Gs); p4, l4, _ = logit_fold(Q, Qte, Xtr[:, cols], Xte[:, cols], Gt[q:, q:], Gt[:q, q:], D["T"]["T4"][tr], val)
                    res[f"lin|{cfg}|T4"] = p4; lams["T4"] = l4
                info[f"lin|{cfg}"] = lams; log(f"fold {fo} linear {cfg} {time.time() - t1:.0f} s", lams)
            del G, Lo, Xtr, Xte
        ym = np.stack([D["T"][t][tr] for t in REG], 1); mu, sd = ym.mean(0), ym.std(0)
        Ym = torch.from_numpy(np.concatenate([(ym - mu) / sd, D["T"]["T4"][tr][:, None]], 1).astype(np.float32)); ltr, lte = std_low(Zlow["PH"][tr], Zlow["PH"][te])
        for cfg in MLP:
            t1 = time.time(); _, fam, views = LIN[cfg]
            Xa = torch.from_numpy(np.concatenate([ltr] + [mlp_in[(fam, v)][0] for v in views], 1).astype(np.float32)); Xb = torch.from_numpy(np.concatenate([lte] + [mlp_in[(fam, v)][1] for v in views], 1).astype(np.float32))
            outs, eps = [], []
            for s in range(N_SEEDS):
                o, eb, ne = mlp_fold(Xa, Ym, Xb, val, 1000 * fo + s); outs.append(o); eps.append((eb, ne))
            o = np.mean(outs, 0)
            for i, t in enumerate(REG):
                res[f"mlp|{cfg}|{t}"] = o[:, i] * sd[i] + mu[i]
            res[f"mlp|{cfg}|T4"] = o[:, 5]; info[f"mlp|{cfg}"] = eps; log(f"fold {fo} mlp {cfg} {time.time() - t1:.0f} s inputs {Xa.shape[1]} best epochs / epochs run", eps)
        info["wall_s"] = round(time.time() - t0, 1); np.savez(dst, info=json.dumps(info), **res); log(f"fold {fo} done {time.time() - t0:.0f} s")
    print("PROBES DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------------- summary
PAIRS = [("lin", "PH", "P"), ("lin", "PH+V2", "PH"), ("lin", "PH+V24", "PH"), ("lin", "PH+V124", "PH"), ("lin", "PH+V24shuf", "PH"), ("lin", "PHf", "PH"), ("lin", "PHf+Vf24", "PHf"),
         ("mlp", "PH+V24", "PH"), ("mlp", "PH+V24shuf", "PH"), ("mlp", "PHf", "PH"), ("mlp", "PHf+Vf24", "PHf")]


def summary(subsample: int = 0):
    D = load(subsample); n = len(D["rows"]); T = dict(D["T"]); T["past_speed"] = D["past_speed"]; P = {}; infos = []
    for fo in range(N_OUTER):
        z = np.load(PDIR / f"fold{fo}.npz"); infos.append(json.loads(str(z["info"]))); te = z["te"]
        for k in z.files:
            if "|" in k:
                P.setdefault(k, np.full(n, np.nan))[te] = z[k]
    assert all(np.isfinite(v).all() for v in P.values())
    def err(reader, cfg, t):
        p = P[f"{reader}|{cfg}|{t}"]
        if t == "T4":
            p = np.clip(p, 1e-7, 1 - 1e-7); return -(T[t] * np.log(p) + (1 - T[t]) * np.log(1 - p))
        return (p - T[t]) ** 2
    masks = {"all": np.ones(n, bool), "event": D["event"]}; one = np.ones(n)
    res = {"made": time.strftime("%F %T"), "subsample": subsample, "moments": n, "clips": len(set(D["clip"])), "chunks": len(set(D["chunk"])), "event_moments": int(D["event"].sum()),
           "event_clips": len(set(D["clip"][D["event"]])), "event_chunks": len(set(D["chunk"][D["event"]])), "training_moments_total": D["n_all"], "training_clips_total": D["clips_all"],
           "T4_rate": float(T["T4"].mean()), "T4_rate_event": float(T["T4"][D["event"]].mean()), "target_sd": {t: float(np.std(T[t])) for t in YCOLS}, "folds": infos, "sets": {}, "pairs": {}}
    for k in sorted({k.rsplit("|", 1)[0] for k in P}):
        r, c = k.split("|"); res["sets"][k] = {t: ({"auc": L2.wauc(P[f"{k}|T4"], T["T4"], one), "log_loss": float(err(r, c, t).mean())} if t == "T4" else
                                                   {"mse": float(err(r, c, t).mean()), "r2": float(1 - err(r, c, t).mean() / np.var(T[t]))}) for t in ALL_T + ("past_speed",) if f"{k}|{t}" in P}
    for r, A, B in PAIRS:
        res["pairs"][f"{r}: {A} vs {B}"] = {mk: {t: L2.gain_ci(err(r, A, t), err(r, B, t), D["chunk"], m, np.random.RandomState(L2.BOOT_SEED)) for t in ALL_T} for mk, m in masks.items()}
    res["positive_control"] = {mk: L2.gain_ci(err("lin", "P+V24", "past_speed"), err("lin", "P", "past_speed"), D["chunk"], m, np.random.RandomState(L2.BOOT_SEED)) for mk, m in masks.items()}
    res["mlp_vs_linear_PH"] = {t: L2.gain_ci(err("mlp", "PH", t), err("lin", "PH", t), D["chunk"], masks["all"], np.random.RandomState(L2.BOOT_SEED)) for t in ALL_T}
    (OUT / "ceiling_results.json").write_text(json.dumps(res, indent=1) + "\n"); print("written", OUT / "ceiling_results.json")


# ---------------------------------------------------------------------------------------------------------------- self-test (synthetic)
def selftest():
    rng = np.random.RandomState(0); n, dz, dx = 3000, 5, 60; groups = np.arange(n) % N_INNER; gi = [torch.from_numpy(np.flatnonzero(groups == g)) for g in range(N_INNER)]
    Z = rng.randn(n + 500, dz); X = rng.randn(n + 500, dx).astype(np.float32) / dx ** 0.5; y = Z @ rng.randn(dz) + X @ rng.randn(dx) * 3 + rng.randn(n + 500)
    Q, Qte = L2.low_dim(torch.from_numpy(Z[:n]), torch.from_numpy(Z[n:])); Xt = torch.from_numpy(X[:n]); Y = torch.from_numpy(y[:n, None]); q = Q.shape[1]
    G = gram_groups(Xt, Y, gi); Lo = low_groups(Q, Xt, Y, gi); cols = torch.arange(dx); Gs, bs, yys = assemble(G, Lo, cols); W, lam, cv = ridge_cv(Gs, bs, yys, q)
    F = torch.cat([Q, Xt.to(f64)], 1); li = LAMS.index(lam[0]); pen = torch.cat([torch.full((q,), 1e-8, dtype=f64), torch.full((dx,), lam[0], dtype=f64)]); tot = 0.0
    for g in range(N_INNER):
        mk = torch.from_numpy(groups != g); wg = torch.linalg.solve(F[mk].T @ F[mk] + torch.diag(pen), F[mk].T @ Y[mk, 0]); tot += float(((F[~mk] @ wg - Y[~mk, 0]) ** 2).sum())
    d1 = abs(tot - float(cv[li, 0])) / tot; wd = torch.linalg.solve(F.T @ F + torch.diag(pen), F.T @ Y[:, 0]); d2 = float((wd - W[:, 0]).abs().max())
    # kernel form of link 2 on the same data = this primal form (same lambda)
    Ktr = Xt.to(f64) @ Xt.to(f64).T; Kte = torch.from_numpy(X[n:]).to(f64) @ Xt.to(f64).T; KQ = Ktr @ Q; Kt = Ktr - Q @ KQ.T - KQ @ Q.T + Q @ (Q.T @ KQ) @ Q.T
    kte = Kte - Qte @ KQ.T; kte = kte - (kte @ Q) @ Q.T; al = torch.linalg.solve(Kt + lam[0] * torch.eye(n, dtype=f64), Y[:, 0] - Q @ (Q.T @ Y[:, 0]))
    pk = Qte @ (Q.T @ Y[:, 0]) + kte @ al; pp = torch.cat([Qte, torch.from_numpy(X[n:]).to(f64)], 1) @ W[:, 0]; d3 = float((pk - pp).abs().max())
    # randomised directions vs exact on a low-rank + noise matrix
    A = (rng.randn(4000, 20) @ rng.randn(20, 300) + 0.3 * rng.randn(4000, 300)).astype(np.float32); r = np.arange(4000)
    _, s1 = pca_scores(A, r, 20, True); _, s2 = pca_scores(A, r, 20, False); d4 = float(abs(s1.sum() - s2.sum()))
    # MLP: a planted NON-linear signal in the "past view" is found, and vanishes when the view is shuffled across clips
    nc, per = 300, 30; m = nc * per; clip = np.repeat(np.arange(nc), per).astype(str); low = rng.randn(m, 6).astype(np.float32); pv = rng.randn(m, 20).astype(np.float32)
    yy = low @ rng.randn(6) + 2.0 * np.abs(pv[:, 0]) + 0.5 * rng.randn(m); outer = L2.fold_of(clip, 5, 1); sh = L2.shuffle_across_clips(clip, 5); pr = {k: np.zeros(m) for k in "BVS"}
    for fo in range(5):
        te, tr = np.flatnonzero(outer == fo), np.flatnonzero(outer != fo); val = L2.fold_of(clip[tr], 5, 2) == 0; mu, sd = yy[tr].mean(), yy[tr].std()
        Yt = torch.from_numpy(np.stack([(yy[tr] - mu) / sd] * 5 + [(yy[tr] > mu).astype(float)], 1).astype(np.float32))
        for k, Xm in (("B", low), ("V", np.concatenate([low, pv], 1)), ("S", np.concatenate([low, pv[sh]], 1))):
            o, _, _ = mlp_fold(torch.from_numpy(Xm[tr]), Yt, torch.from_numpy(Xm[te]), val, fo); pr[k][te] = o[:, 0] * sd + mu
    e = {k: (v - yy) ** 2 for k, v in pr.items()}; allm = np.ones(m, bool)
    g1 = L2.gain_ci(e["V"], e["B"], clip, allm, np.random.RandomState(1)); g2 = L2.gain_ci(e["S"], e["B"], clip, allm, np.random.RandomState(1))
    ok = d1 < 1e-6 and d2 < 1e-6 and d3 < 1e-5 and d4 < 0.01 and g1["lo"] > 0.3 and g2["hi"] < 0.05
    o = {"at": time.strftime("%F %T"), "cv_from_gram_vs_refits_rel_diff": d1, "weights_vs_direct_solve_max_abs_diff": d2, "primal_vs_link2_kernel_form_max_abs_diff": d3,
         "randomised_vs_exact_variance_share_abs_diff": d4, "mlp_planted_nonlinear_gain": g1, "mlp_shuffled_gain": g2, "ok": bool(ok)}
    (OUT / "ceiling_selftest.json").write_text(json.dumps(o, indent=1) + "\n"); print(json.dumps(o, indent=1)); assert ok


def stats(subsample: int = 0):
    """sample statistics only (no probe): written into the pre-registration."""
    D = load(subsample); ev = D["event"]
    o = {"subsample": subsample, "training_clips_total": D["clips_all"], "training_moments_total": D["n_all"], "moments_with_4s_history": len(D["rows"]), "clips": len(set(D["clip"])),
         "chunks": len(set(D["chunk"])), "event_moments": int(ev.sum()), "event_clips": len(set(D["clip"][ev])), "event_chunks": len(set(D["chunk"][ev])),
         "T4_rate": float(D["T"]["T4"].mean()), "by_type": {k: int(v) for k, v in pd.Series(D["type"]).value_counts().items()},
         "clips_by_type": {k: int(v) for k, v in pd.DataFrame({"c": D["clip"], "t": D["type"]}).drop_duplicates()["t"].value_counts().items()},
         "target_mean_sd": {t: [float(D["T"][t].mean()), float(D["T"][t].std())] for t in ALL_T}, "past_speed_mean_sd": [float(D["past_speed"].mean()), float(D["past_speed"].std())]}
    (OUT / f"ceiling_stats{'_sub' if subsample else ''}.json").write_text(json.dumps(o, indent=1) + "\n"); print(json.dumps(o, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "stats", "run", "summary"]); ap.add_argument("--subsample", type=int, default=0); a = ap.parse_args()
    {"selftest": selftest, "stats": lambda: stats(a.subsample), "run": lambda: run(a.subsample), "summary": lambda: summary(a.subsample)}[a.cmd]()
