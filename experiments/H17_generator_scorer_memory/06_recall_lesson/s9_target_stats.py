"""Build 5: the standardisation statistics of the recall targets. Mean and standard deviation per (block position, dimension) of the 32 block
means (pooled[:, 1:], BUILD2_CACHE.md) over ALL moments of the TRAINING slice's pooled array -> out/recall_target_stats.pt
{mu (32, 256), sd (32, 256), n, source, sha-free provenance}. CPU only. Usage: run_model.sh <abs>/s9_target_stats.py"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
SRC = Path("{experiment_store}/cache/training/pooled_training.npy"); OUT = Path(__file__).resolve().parents[1] / "out"
t0 = time.time(); a = np.load(SRC, mmap_mode="r"); assert a.ndim == 2 and a.shape[1] == 33 * 256, a.shape       # stored flat: (moments, 33 x 256)
# layout check against one clip file: row `pooled_row` of the array = that moment's pooled (33, 256), row-major
import pandas as pd
idx = pd.read_parquet(SRC.parent / "index_training.parquet"); r0 = idx.iloc[len(idx) // 2]
clip = torch.load(SRC.parent / r0.file, map_location="cpu", weights_only=False); j = clip["sample_id"].index(r0.sample_id)
check = float(np.abs(np.asarray(a[int(r0.pooled_row)]).reshape(33, 256) - clip["pooled"][j].float().numpy()).max()); assert check < 1e-5, check
n = a.shape[0]; s1 = np.zeros((32, 256), np.float64); s2 = np.zeros((32, 256), np.float64)
for i in range(0, n, 4096):
    x = np.asarray(a[i:i + 4096], np.float64).reshape(-1, 33, 256)[:, 1:]; s1 += x.sum(0); s2 += (x ** 2).sum(0)
mu = s1 / n; var = s2 / n - mu ** 2; sd = np.sqrt(np.maximum(var, 0)) + 1e-6
info = {"source": str(SRC), "moments": int(n), "array_shape": list(a.shape), "dtype": str(a.dtype), "sd_floor": 1e-6, "layout_check_max_abs_diff_vs_clip_file": check, "layout_check_sample": str(r0.sample_id), "computed": time.strftime("%F %T"),
        "mu_abs_mean": float(np.abs(mu).mean()), "sd_mean": float(sd.mean()), "sd_min": float(sd.min()), "sd_max": float(sd.max()), "wall_s": round(time.time() - t0, 1)}
torch.save({"mu": torch.from_numpy(mu).float(), "sd": torch.from_numpy(sd).float(), **info}, OUT / "recall_target_stats.pt")
(OUT / "recall_target_stats.json").write_text(json.dumps(info, indent=1) + "\n"); print(json.dumps(info))
