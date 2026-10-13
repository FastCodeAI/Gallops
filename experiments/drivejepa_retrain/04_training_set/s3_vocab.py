"""Part C: extended vocabulary. Original = navsim_v2/data/8192.npy, (8192, 40, 3) float32, pose k at (k+1)*0.1 s, columns x, y, heading
in the rear-axle frame at t (training takes poses[:, 4::5], drive_jepa_agent.py:49-50). Output written ONLY to this folder."""
import hashlib, numpy as np, pandas as pd, pathlib
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
import s3_common as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ORIG = pathlib.Path("{runs}/paddock_sid/Alpasim/adapters/drivejepa/src/navsim_v2/data/8192.npy")
SEED, KMAX, VMIN = 20261006, 2048, 14.0
v0 = np.load(ORIG); assert v0.shape == (8192, 40, 3) and v0.dtype == np.float32
idx = pd.read_parquet(C.HERE / "labels_index.parquet"); lab = np.load(C.HERE / "labels.npy"); C.assert_train(idx.uuid.unique())      # train split only
sel = (idx.speed >= VMIN).to_numpy(); L = lab[sel]; src = idx[sel]
# 8 poses at 0.5 s -> 40 poses at 0.1 s: cubic spline through (t=0: 0,0,0) and the 8 label poses (x, y, unwrapped heading), evaluated at
# 0.1 ... 4.0 s. Poses 4, 9, ..., 39 are the label poses themselves, so poses[:, 4::5] returns the 8-pose future unchanged.
tk = np.arange(0, 4.01, 0.5); tq = np.arange(1, 41) * 0.1
K8 = np.concatenate([np.zeros((len(L), 1, 3)), L], axis=1); K8[:, :, 2] = np.unwrap(K8[:, :, 2], axis=1)
dense = CubicSpline(tk, K8, axis=1)(tq); assert np.abs(dense[:, 4::5] - K8[:, 1:]).max() < 1e-9
k = min(KMAX, len(dense) // 4); km = KMeans(n_clusters=k, random_state=SEED, n_init=3, max_iter=100).fit(dense.reshape(len(dense), -1))
cen = km.cluster_centers_.reshape(k, 40, 3).astype(np.float32); cnt = np.bincount(km.labels_, minlength=k)
order = np.argsort(np.linalg.norm(np.diff(np.concatenate([np.zeros((k, 1, 2)), cen[:, 4::5, :2]], 1), axis=1), axis=2).mean(1) / 0.5); cen = cen[order]; cnt = cnt[order]   # appended in order of mean speed
out = np.concatenate([v0, cen], axis=0); p = C.HERE / f"vocab_{len(out)}.npy"; np.save(p, out)
chk = np.load(p); assert chk.dtype == np.float32 and chk[:8192].tobytes() == v0.tobytes(), "Z6: first 8,192 rows differ"
def speeds(a8): st = np.diff(np.concatenate([np.zeros((len(a8), 1, 2)), a8[:, :, :2]], 1), axis=1); return np.linalg.norm(st, axis=2) / 0.5
s8 = speeds(cen[:, 4::5]); mx, mean = s8.max(1), s8.mean(1)
d = np.diff(np.concatenate([np.zeros((k, 1, 3)), cen], 1), axis=1); ds = np.linalg.norm(d[:, :, :2], axis=2); curv = np.where(ds.sum(1) > 1, (cen[:, -1, 2]) / np.maximum(ds.sum(1), 1e-6), np.nan)   # mean signed curvature = heading change / path length
pd.DataFrame({"row": 8192 + np.arange(k), "members": cnt, "mean_speed": mean, "max_speed": mx, "end_x": cen[:, -1, 0], "end_y": cen[:, -1, 1], "heading_change_deg": np.degrees(cen[:, -1, 2]), "mean_curvature_1pm": curv}).to_csv(C.HERE / "vocab_added.csv", index=False)
bins = [0, 14, 16, 20, 24, 28, 32, 36, 40, 99]; h = pd.cut(mean, bins, right=False).value_counts().sort_index(); hs = pd.cut(src.speed, bins, right=False).value_counts().sort_index()
print("source futures (human speed >= 14 m/s):", len(dense), "from", src.uuid.nunique(), "clips | clusters", k, "| members per cluster min/med/max", cnt.min(), int(np.median(cnt)), cnt.max())
print("added shapes by mean speed:", {str(a): int(b) for a, b in h.items()}); print("source anchors by human speed:", {str(a): int(b) for a, b in hs.items()})
print("added: mean speed min/max", mean.min().round(2), mean.max().round(2), "| max step speed", mx.max().round(2), "| curvature 1/m min/max", np.nanmin(curv).round(5), np.nanmax(curv).round(5), "| heading change deg min/max", np.degrees(cen[:, -1, 2]).min().round(1), np.degrees(cen[:, -1, 2]).max().round(1), "| end lateral m min/max", cen[:, -1, 1].min().round(1), cen[:, -1, 1].max().round(1))
print("file", p.name, out.shape, out.dtype, "sha256", C.sha256_file(p), "| original sha256", C.sha256_file(ORIG), "| Z6 bit-identical first 8192: True")
fig, ax = plt.subplots(figsize=(8, 4)); b = np.arange(0, 42, 1); ax.hist(speeds(v0[:, 4::5]).mean(1), bins=b, color="#9ca3af", label="original 8,192 (mean speed)"); ax.hist(mean, bins=b, color="#2563eb", alpha=0.85, label=f"added {k} fast shapes")
ax.set_yscale("log"); ax.set_xlabel("mean speed over 4 s, m/s"); ax.set_ylabel("shapes"); ax.legend(frameon=False); ax.set_title(f"Vocabulary: original vs extended ({len(out)} shapes)", fontsize=10); fig.tight_layout(); fig.savefig(C.HERE / "vocab_hist.png", dpi=110)
