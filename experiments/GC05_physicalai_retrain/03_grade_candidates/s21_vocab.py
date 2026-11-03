"""S2.1 Part C: vocabulary speed profile (read-only on navsim_v2/data/8192.npy) + anchor-index feasibility on our metric caches."""
import sys, time, hashlib, numpy as np, pandas as pd
import s2_common as C, s2_grade as G
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = "./data/8192.npy"; v = np.load(P); print("file", P, "sha256", hashlib.sha256(open(P, "rb").read()).hexdigest()[:16], "shape", v.shape, v.dtype)
step = np.diff(np.concatenate([np.zeros((len(v), 1, 2)), v[:, :, :2]], axis=1), axis=1); sp = np.linalg.norm(step, axis=2) / 0.1          # 40 steps of 0.1 s
a8 = v[:, 4::5]; st8 = np.diff(np.concatenate([np.zeros((len(v), 1, 2)), a8[:, :, :2]], axis=1), axis=1); sp8 = np.linalg.norm(st8, axis=2) / 0.5   # what training uses (agent.py:50)
mx, fin, mean = sp8.max(1), sp8[:, -1], sp8.mean(1)
print("first-step dt check: |pose0| median", np.median(np.linalg.norm(v[:, 0, :2], axis=1)).round(3), "m; pose39 x max", v[:, -1, 0].max().round(1), "min", v[:, -1, 0].min().round(1), "| end lateral |y| max", np.abs(v[:, -1, 1]).max().round(1), "| heading range deg", np.degrees(v[:, -1, 2].min()).round(0), np.degrees(v[:, -1, 2].max()).round(0))
rows = [dict(threshold_mps=t, max_speed_above=int((mx > t).sum()), final_speed_above=int((fin > t).sum()), mean_speed_above=int((mean > t).sum()), share_max_pct=round(100 * (mx > t).mean(), 2)) for t in (16, 20, 25, 30, 35)]
pd.DataFrame(rows).to_csv(C.HERE / "s21_vocab_speed.csv", index=False); print(pd.DataFrame(rows).to_string()); print("max speed quantiles 50/90/99/100:", np.quantile(mx, [.5, .9, .99, 1]).round(2), "| final speed 50/90/99/100:", np.quantile(fin, [.5, .9, .99, 1]).round(2), "| stationary anchors (length < 0.5 m):", int((np.linalg.norm(st8, axis=2).sum(1) < 0.5).sum()))
pd.DataFrame({"anchor": np.arange(len(v)), "max_speed": mx, "final_speed": fin, "mean_speed": mean, "end_x": a8[:, -1, 0], "end_y": a8[:, -1, 1]}).to_csv(C.HERE / "s21_vocab_anchors.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.2)); b = np.arange(0, 42, 1); ax.hist(mx, bins=b, color="#2563eb", alpha=0.85, label="max speed over 4 s"); ax.hist(fin, bins=b, histtype="step", color="#111827", lw=1.6, label="final speed (3.5–4.0 s)")
for t in (16, 20, 30): ax.axvline(t, c="#9ca3af", lw=0.8, ls="--")
ax.set_xlabel("speed, m/s (0.5 s steps)"); ax.set_ylabel("anchors (of 8,192)"); ax.set_yscale("log"); ax.legend(frameon=False); ax.set_title("Drive-JEPA trajectory vocabulary: speed per anchor", fontsize=10); fig.tight_layout(); fig.savefig(C.HERE / "s21_png" / "vocab_speed_hist.png", dpi=110)
# ---- anchor-index feasibility: upstream = calc_anchors_scores.py (score all 8,192) + convert_to_index.py (final >= 0.95, at most 256)
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.drive_jepa_perception_based.score_module.compute_navsim_score import load_metric_cache
mcl = MetricCacheLoader(C.NAS / "s1" / "cache"); g = pd.read_csv(C.HERE / "pilot_grades.csv", low_memory=False); h = g[g.plan == "human"]
demo = [h[h.type == t].iloc[k].token for t, k in (("A", 40), ("A", 100), ("T", 40), ("O", 20))]; out = []
for tok in demo:
    mc = load_metric_cache(mcl.metric_cache_paths[tok]); t0 = time.time(); sc, ex = G.grade(mc, a8); dt = time.time() - t0; ok = np.where(sc[:, -1] >= 0.95)[0]; r = h[h.token == tok].iloc[0]
    out.append(dict(token=tok, type=r.type, human_speed=round(r.speed, 1), secs=round(dt, 1), anchors_ge_0_95=len(ok), best=round(float(sc[:, -1].max()), 3), max_speed_among_ok=round(float(mx[ok].max()), 1) if len(ok) else np.nan, median_speed_among_ok=round(float(np.median(mean[ok])), 1) if len(ok) else np.nan, human_grade=round(r.score, 2)))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(C.HERE / "s21_anchor_index_demo.csv", index=False)
