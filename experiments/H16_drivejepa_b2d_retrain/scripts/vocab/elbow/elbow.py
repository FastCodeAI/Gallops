"""Which vocabulary size? (Navvrat 05 Nov: an elbow check, and whether more clusters only split speed or feed a mode collapse.)
Same paths and k-means settings as ../build_vocab.py, at sizes between 256 and 8,192, three seeds each. Analysis files only: nothing here replaces ../b2d_<K>.npy.
Per run, in the expert's own terms (8 points, 0.5 .. 4.0 s):
  fit         inertia, iterations
  coverage    nearest entry by ADE on training and held-out paths; all paths, and moving paths (> 0.1 m in 4 s); by speed band; on turns (> 30 deg end heading)
  error split at the nearest entry, each waypoint's error along the expert's heading (speed) and across it (shape), held-out moving paths
  entries     how many entries sit in each speed band; straight entries (|end heading| < 5 deg, |end y| < 1 m) and the median gap between their mean speeds
  memorising  entries built from one path, entries built from one clip
  use         held-out moving paths: entries used, effective entries exp(entropy); share of all training paths on the biggest entry
  teachers    held-out moving paths: entries within 1 m ADE (median, share with none), the pool a teacher list would draw from
  spacing     nearest other entry (median ADE), moving entries
   $PY elbow.py <K> <seed>   -> runs/K_seed.json, runs/centres_K_seed.npy"""
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
K, SEED = int(sys.argv[1]), int(sys.argv[2])
OUT = f"{HERE}/runs/{K}_{SEED}.json"
if os.path.exists(OUT):
    sys.exit(0)
z = np.load(f"{HERE}/../paths_cache.npz")
rear, split, clip = z["rear"], z["split"], z["clip_id"]
P8 = rear[:, 4::5, :]
tr, ho = split == 0, split == 1
X = np.ascontiguousarray(P8[tr][:, :, :2].reshape(tr.sum(), 16))
km = KMeans(n_clusters=K, init="k-means++", n_init=1, algorithm="lloyd", max_iter=10000, tol=0.0, random_state=SEED).fit(X)
C = km.cluster_centers_.reshape(K, 8, 2).astype(np.float32)
lab = km.labels_


def speeds(p):
    """(N, 8, 2) -> mean speed over 4 s, path length."""
    q = np.concatenate([np.zeros_like(p[:, :1]), p], 1)
    step = np.linalg.norm(np.diff(q, axis=1), axis=-1)
    return step.sum(1) / 4.0, step.sum(1)


BANDS = [("stationary", -1, None), ("0-3", 0, 3), ("3-6", 3, 6), ("6-9", 6, 9), ("9+", 9, 99)]


def band(v, length):
    out = np.zeros(len(v), int)
    mov = length > 0.1
    out[mov & (v < 3)] = 1
    out[mov & (v >= 3) & (v < 6)] = 2
    out[mov & (v >= 6) & (v < 9)] = 3
    out[mov & (v >= 9)] = 4
    return out


def nearest(p, c, chunk=128):
    idx = np.empty(len(p), np.int32)
    ade = np.empty(len(p), np.float32)
    within = np.empty(len(p), np.int32)
    for s in range(0, len(p), chunk):
        d = np.zeros((min(chunk, len(p) - s), len(c)), np.float32)
        for j in range(8):
            dx = p[s:s + chunk, j, 0, None].astype(np.float32) - c[None, :, j, 0]
            dy = p[s:s + chunk, j, 1, None].astype(np.float32) - c[None, :, j, 1]
            d += np.sqrt(dx * dx + dy * dy)
        d /= 8
        idx[s:s + chunk] = d.argmin(1)
        ade[s:s + chunk] = d.min(1)
        within[s:s + chunk] = (d <= 1.0).sum(1)
    return idx, ade, within


def summ(a):
    return dict(n=int(len(a)), mean=float(a.mean()) if len(a) else None, median=float(np.median(a)) if len(a) else None, p95=float(np.percentile(a, 95)) if len(a) else None, max=float(a.max()) if len(a) else None)


res = dict(K=K, seed=SEED, iters=int(km.n_iter_), inertia=float(km.inertia_), inertia_per_path=float(km.inertia_ / len(X)))
vp, lp = speeds(P8[:, :, :2])
bp = band(vp, lp)
turn = np.abs(np.arctan2(np.sin(P8[:, -1, 2]), np.cos(P8[:, -1, 2]))) > np.radians(30)
moving = lp > 0.1
for name, m in (("train", tr), ("heldout", ho)):
    if name == "train" and SEED != 20261105:
        continue
    i, a, w = nearest(P8[m][:, :, :2], C)
    res[f"{name}_all"] = summ(a)
    res[f"{name}_moving"] = summ(a[moving[m]])
    res[f"{name}_turn"] = summ(a[turn[m] & moving[m]])
    res[f"{name}_band"] = {b[0]: summ(a[bp[m] == k]) for k, b in enumerate(BANDS)}
    if name == "heldout":
        e = C[i] - P8[m][:, :, :2]
        hd = P8[m][:, :, 2]
        along = e[..., 0] * np.cos(hd) + e[..., 1] * np.sin(hd)
        across = -e[..., 0] * np.sin(hd) + e[..., 1] * np.cos(hd)
        mv = moving[m]
        res["heldout_err_along_m"] = float(np.abs(along[mv]).mean())
        res["heldout_err_across_m"] = float(np.abs(across[mv]).mean())
        res["heldout_err_across_turn_m"] = float(np.abs(across[mv & turn[m]]).mean())
        im = i[mv]
        cnt = np.bincount(im, minlength=K).astype(float)
        p = cnt[cnt > 0] / cnt.sum()
        res["use_entries"] = int((cnt > 0).sum())
        res["use_share_of_entries"] = float((cnt > 0).mean())
        res["use_effective"] = float(np.exp(-(p * np.log(p)).sum()))
        res["teachers_within_1m_median"] = float(np.median(w[mv]))
        res["teachers_none_share"] = float((w[mv] == 0).mean())
        res["teachers_p90"] = float(np.percentile(w[mv], 90))
        res["per_clip"] = {int(c): [float(a[mv & (clip[m] == c)].sum()), int((mv & (clip[m] == c)).sum())] for c in np.unique(clip[m])}
vc, lc = speeds(C.astype(np.float64))
bc = band(vc, lc)
res["entries_band"] = {b[0]: int((bc == k).sum()) for k, b in enumerate(BANDS)}
res["paths_band_train"] = {b[0]: int((bp[tr] == k).sum()) for k, b in enumerate(BANDS)}
g = np.bincount(lab, minlength=K)
res["groups"] = dict(min=int(g.min()), median=float(np.median(g)), max=int(g.max()), single=int((g == 1).sum()), under5=int((g < 5).sum()))
res["biggest_share"] = float(g.max() / g.sum())
clips_tr = clip[tr]
ncl = np.zeros(K, int)
order = np.lexsort((clips_tr, lab))
ls, cs = lab[order], clips_tr[order]
first = np.r_[True, (ls[1:] != ls[:-1]) | (cs[1:] != cs[:-1])]
np.add.at(ncl, ls[first], 1)
res["entries_one_clip"] = int((ncl == 1).sum())
res["entries_one_clip_share"] = float((ncl == 1).mean())
res["paths_in_one_clip_entries_share"] = float(g[ncl == 1].sum() / g.sum())
# heading of a centre from its last segment; straight entries and the speed gap between them
end = C[:, -1] - C[:, -2]
hd_end = np.arctan2(end[:, 1], end[:, 0])
straight = (lc > 0.1) & (np.abs(hd_end) < np.radians(5)) & (np.abs(C[:, -1, 1]) < 1.0)
sv = np.sort(vc[straight])
res["straight_entries"] = int(straight.sum())
res["straight_share_of_moving"] = float(straight.sum() / max(1, (lc > 0.1).sum()))
res["straight_speed_gap_median"] = float(np.median(np.diff(sv))) if len(sv) > 1 else None
turning = (lc > 0.1) & (np.abs(hd_end) > np.radians(30))
res["turning_entries"] = int(turning.sum())
mvc = C[lc > 0.1]
i2, _, _ = None, None, None
nn = np.empty(len(mvc), np.float32)
for s in range(0, len(mvc), 128):
    d = np.zeros((min(128, len(mvc) - s), len(mvc)), np.float32)
    for j in range(8):
        d += np.hypot(mvc[s:s + 128, j, 0, None] - mvc[None, :, j, 0], mvc[s:s + 128, j, 1, None] - mvc[None, :, j, 1])
    d /= 8
    d[np.arange(d.shape[0]), s + np.arange(d.shape[0])] = np.inf
    nn[s:s + 128] = d.min(1)
res["spacing_median_m"] = float(np.median(nn))
res["spacing_p10_m"] = float(np.percentile(nn, 10))
np.save(f"{HERE}/runs/centres_{K}_{SEED}.npy", C)
json.dump(res, open(OUT + ".tmp", "w"), indent=0)
os.replace(OUT + ".tmp", OUT)
print("DONE", K, SEED, res["iters"], round(res["heldout_moving"]["mean"], 3), flush=True)
