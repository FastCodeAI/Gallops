"""S9 shared pieces for parts B and C.4 (definitions: PREDICTIONS.md). CPU only; reads S6's dumps through s7_common.load.
Gate order everywhere: [drivable area ("kerb"), driving direction ("oncoming"), collision]; a plan fails a gate when its stored value < 1."""
import os, pathlib, numpy as np, pandas as pd, torch, torch.nn as nn
from s7_common import load, boot, spearman_rows, RD, SL, TERMS
HERE = pathlib.Path(__file__).resolve().parent; SEED = 20261008; torch.set_num_threads(int(os.environ.get("S9_THREADS", "4")))
GATES = ["drivable area", "driving direction", "collision"]; GCOL = [1, 3, 0]            # columns of grade_9 (NC DAC TLC DDC ...)
WDS = [0.0, 1e-4, 1e-3, 1e-2]; LR = 1e-3; BATCH = 256; MAXEP = 60; PATIENCE = 8; MARGIN = 0.1; MINDIFF = 0.05; RANK_WD = 1e-2
TGRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, np.float64)))

def seen_split(D):
    """S7's split of SEEN: 80 % / 20 % of clips, seed 20261008 (s7_h3.py). Returns boolean masks train, validation."""
    clips = np.unique(D["clip"]); rng = np.random.default_rng(SEED); perm = rng.permutation(len(clips)); nva = int(round(0.2 * len(clips))); vc = set(clips[perm[:nva]]); va = np.array([c in vc for c in D["clip"]]); return ~va, va
def fails(g): return (g[..., GCOL] < 1.0).astype(np.float32)                                # (n, 32, 3)
def gate_net(): return nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3))
def rank_net(): return nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
def fwd(net, Z, bs=4096):
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(Z), bs): out.append(net(torch.from_numpy(Z[k:k + bs])).numpy().astype(np.float64))
    return np.concatenate(out)
def gate_probs(net, Z):
    """-> per-gate fail probability (n, 32, 3) and P(any gate fails) (n, 32)"""
    p = sig(fwd(net, Z)); return p, 1 - np.prod(1 - p, axis=-1)

def train_gate(Ztr, Ytr, Mtr, Zva, Yva, Mva, wd, log=None):
    """BCE per gate on 'fails', masked where the label is unknown (mask 0). Early stop on masked validation loss."""
    torch.manual_seed(SEED); net = gate_net(); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=wd); r = np.random.default_rng(SEED); bce = nn.BCEWithLogitsLoss(reduction="none")
    Zt, Yt, Mt, Zv, Yv, Mv = (torch.from_numpy(np.ascontiguousarray(a)) for a in (Ztr, Ytr, Mtr, Zva, Yva, Mva)); best = (np.inf, -1, None)
    for ep in range(MAXEP):
        net.train(); p = r.permutation(len(Zt))
        for k in range(0, len(p), BATCH):
            ii = torch.from_numpy(p[k:k + BATCH]); l = (bce(net(Zt[ii]), Yt[ii]) * Mt[ii]).sum() / Mt[ii].sum().clamp(min=1); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): lv = float((bce(net(Zv), Yv) * Mv).sum() / Mv.sum().clamp(min=1))
        if lv < best[0] - 1e-9: best = (lv, ep, {k: v.clone() for k, v in net.state_dict().items()})
        if ep - best[1] >= PATIENCE: break
    net.load_state_dict(best[2]); net.eval()
    if log is not None: log.append(dict(head="gate", weight_decay=wd, val_loss=best[0], best_pass=best[1], passes_run=ep + 1))
    return net, best[0]

def select(out, P, t):
    """veto plans with P > t, take the ranker's top among the rest; all vetoed -> lowest P. Returns pick (n,), vetoed (n, 32)"""
    v = P > t; s = np.where(v, -np.inf, out); pick = s.argmax(1); allv = v.all(1); pick[allv] = P[allv].argmin(1); return pick, v
def regret_of(pick, g): return g.max(1) - g[np.arange(len(g)), pick]

def train_ranker(Ztr, Gtr, Ptr, Zva, Gva, Pva, t, weighted=True, log=None, tag="ranker"):
    """S7's ranking head with pairs weighted by |grade gap| and limited to pairs where both plans have gate P <= t.
    Early stop on validation mean regret of the two-stage selection at the same t."""
    torch.manual_seed(SEED); net = rank_net(); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=RANK_WD); r = np.random.default_rng(SEED)
    Zt, Gt, At = torch.from_numpy(Ztr), torch.from_numpy(Gtr.astype(np.float32)), torch.from_numpy(Ptr <= t); best = (np.inf, -1, None); npairs = 0
    for ep in range(MAXEP):
        net.train(); p = r.permutation(len(Zt))
        for k in range(0, len(p), BATCH):
            ii = torch.from_numpy(p[k:k + BATCH]); o = net(Zt[ii]).squeeze(-1); g = Gt[ii]; a = At[ii]; d = g[:, :, None] - g[:, None, :]; m = (d >= MINDIFF) & a[:, :, None] & a[:, None, :]
            if ep == 0: npairs += int(m.sum())
            if not m.any(): continue
            h = torch.relu(MARGIN - (o[:, :, None] - o[:, None, :])); w = d if weighted else torch.ones_like(d); l = (h * w)[m].sum() / w[m].sum(); opt.zero_grad(); l.backward(); opt.step()
        ov = fwd(net, Zva).squeeze(-1); rv = float(regret_of(select(ov, Pva, t)[0], Gva).mean())
        if rv < best[0] - 1e-9: best = (rv, ep, {k: v.clone() for k, v in net.state_dict().items()})
        if ep - best[1] >= PATIENCE: break
    net.load_state_dict(best[2]); net.eval()
    if log is not None: log.append(dict(head=tag, t=t, weighted=weighted, val_regret=best[0], best_pass=best[1], passes_run=ep + 1, training_pairs=npairs))
    return net, best[0]

def auroc(score, y):
    from scipy.stats import rankdata
    y = y.astype(bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def auroc_boot(score, y, clip, nb=200):
    """pooled over plans; score, y (n, 32); clip-level bootstrap as S6's probe table (200 resamples)"""
    v = auroc(score.ravel(), y.ravel()); u, inv = np.unique(clip, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(u))]; rng = np.random.default_rng(20261007); o = []
    for _ in range(nb):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))]); o.append(auroc(score[ii].ravel(), y[ii].ravel()))
    lo, hi = np.nanpercentile(o, [2.5, 97.5]); return v, float(lo), float(hi)

def momentum_pick(clip, anchor_t, s, st, fallback):
    """s6_analyze.momentum_pick with a veto: s (n, 32) on the 0-1 scale, -inf for vetoed plans; rows with every plan vetoed keep
    `fallback`. score <- (14 s + 2 comfort) / 16, previous plan = previous graded instant of the clip and ITS momentum pick."""
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
    n = len(s); pick = np.where(np.isfinite(s).any(1), np.where(np.isfinite(s), s, -np.inf).argmax(1), fallback); used = np.zeros(n, bool); o = 5; tp = np.arange(41 - o) * 0.1; order = np.lexsort((anchor_t, clip)); prev = None
    for i in order:
        if prev is not None and clip[prev] == clip[i] and abs(anchor_t[i] - anchor_t[prev] - 0.5) < 1e-3 and np.isfinite(s[i]).any():
            cur = np.repeat(st[prev, pick[prev]][None, o:], 32, 0); nxt = st[i][:, :-o]; c = ego_is_two_frame_extended_comfort(cur, nxt, tp).astype(np.float64); pick[i] = int(((14.0 * s[i] + 2.0 * c) / 16.0).argmax()); used[i] = True
        prev = i
    return pick, used

def table_rows(setname, name, sel, pick, g, clip, slc, order_out=None, base_regret=None, extra=None):
    """one row per slice for one pick; g (n, 32, 9). Zero-graded picks and the gate that zeroed them follow S8 (value <= 1e-9)."""
    gt = g[..., -1]; n = len(gt); ar = np.arange(n); best = gt.max(1); gp = gt[ar, pick]; reg = best - gp; z = gp <= 1e-9; gz = g[ar, pick][:, GCOL] <= 1e-9; ra = spearman_rows(order_out, gt) if order_out is not None else None; rows = []
    for sl in SL:
        k = np.ones(n, bool) if sl == "all" else slc == sl
        if not k.any(): continue
        c = dict(reg=reg[k], z=z[k].astype(float), tb=(gp >= best - 1e-9)[k].astype(float), g9=(gp >= 0.9)[k].astype(float), zr=(reg * z)[k]); r = dict(set=setname, head=name, selection=sel, slice=sl, instants=int(k.sum()), clips=int(len(set(clip[k]))))
        if base_regret is not None: c["base"] = base_regret[k]
        if ra is not None: c["ra"] = ra[k]
        F = dict(regret_mean=lambda q: q["reg"].mean(), regret_p90=lambda q: np.quantile(q["reg"], 0.9), picks_graded_0=lambda q: q["z"].mean(), true_best=lambda q: q["tb"].mean(), ge09=lambda q: q["g9"].mean(), share_of_regret_from_picks_graded_0=lambda q: q["zr"].sum() / max(q["reg"].sum(), 1e-12))
        if ra is not None: F["rank_agreement_median"] = lambda q: np.nanmedian(q["ra"])
        if base_regret is not None: F["regret_minus_s7_head"] = lambda q: q["reg"].mean() - q["base"].mean()
        for nm, f in F.items(): r[nm], r[nm + "_lo"], r[nm + "_hi"] = boot(clip[k], c, f)
        nz = int(z[k].sum()); r["n_picks_graded_0"] = nz
        for j, gn in enumerate(GATES): r[f"zeroed_by_{gn.replace(' ', '_')}"] = float(gz[k][z[k], j].mean()) if nz else np.nan
        r["zeroed_by_more_than_one"] = float((gz[k][z[k]].sum(1) > 1).mean()) if nz else np.nan
        for a, b in (extra or {}).items(): r[a] = float(np.mean(b[k]))
        rows.append(r)
    return rows
