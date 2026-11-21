"""Ceiling test (Sid, 2026-11-12), step 1: pooled camera views of every training moment, straight from the build-2 cache. No model, no GPU.

Per clip one read of cache/training/<clip>.pt (tokens (N, 512, 256) of the 16 x 32 grid, ego_status, trajectory, anchor_t). Kept per moment:
  coarse  the 4 x 8 block means (32 x 256 = 8,192), the same reshape as s6_model.block_means (s6_model.py:182-185), float32
  fine    the 8 x 16 block means (128 x 256 = 32,768): tokens.reshape(N, 8, 2, 16, 2, 256).mean((2, 4)), float16
  ego     ego_status with the acceleration columns 5:7 zeroed (as s6_data.py:73), traj (8, 3), anchor_t, chain (moments since the start of the
          run of consecutive 0.5 s anchors), clip, chunk (index column "group"), type, source
-> {experiment_store}/runs/ceiling/{coarse.npy, fine.npy, meta.npz, extract.json}"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
CACHE = Path("{experiment_store}/cache/training"); RUN = Path("{experiment_store}/runs/ceiling"); STEP_S = 0.5


def avail_gb():
    return [int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable")][0] / 2 ** 20


def read(job):
    clip, f, n = job
    rec = torch.load(CACHE / f, map_location="cpu", weights_only=False); tok = rec["tokens"].float(); N = tok.shape[0]; assert N == n and tok.shape[1:] == (512, 256), clip
    coarse = tok.reshape(N, 4, 4, 8, 4, 256).mean((2, 4)).reshape(N, -1).numpy()
    fine = tok.reshape(N, 8, 2, 16, 2, 256).mean((2, 4)).reshape(N, -1).numpy().astype(np.float16)
    ego = rec["ego_status"].float().clone(); ego[:, 5:7] = 0.0
    at = np.asarray(rec["anchor_t"], np.float64); chain = np.zeros(N, np.int64)
    for i in range(1, N):
        chain[i] = chain[i - 1] + 1 if abs(at[i] - at[i - 1] - STEP_S) < 1e-6 else 0
    return coarse, fine, ego.numpy(), np.asarray(rec["trajectory"], np.float64), at, chain, list(rec["sample_id"])


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(CACHE / "index_training.parquet").sort_values(["clip", "row"]).reset_index(drop=True)
    g = idx.groupby("clip", sort=False); jobs = [(c, d["file"].iloc[0], len(d)) for c, d in g]; n = len(idx)
    coarse = np.zeros((n, 8192), np.float32); fine = np.zeros((n, 32768), np.float16); ego = np.zeros((n, 11), np.float32); traj = np.zeros((n, 8, 3)); at = np.zeros(n)
    chain = np.zeros(n, np.int64); pos = 0; t0 = time.time()
    with ThreadPoolExecutor(4) as pool:                       # NAS reads; closed on exit
        for ci, (job, r) in enumerate(zip(jobs, pool.map(read, jobs))):
            m = len(r[4]); sl = slice(pos, pos + m); assert r[6] == list(idx["sample_id"].iloc[sl]), job[0]
            coarse[sl], fine[sl], ego[sl], traj[sl], at[sl], chain[sl] = r[:6]; pos += m
            if (ci + 1) % 200 == 0:
                print(time.strftime("%F %T"), f"{ci + 1}/{len(jobs)} clips {time.time() - t0:.0f} s ram {avail_gb():.0f} GB", flush=True)
                if avail_gb() < 20:
                    raise SystemExit("available RAM below 20 GB: stopping")
    assert pos == n and np.isfinite(coarse).all()
    np.save(RUN / "coarse.npy", coarse); np.save(RUN / "fine.npy", fine)
    np.savez(RUN / "meta.npz", clip=idx["clip"].to_numpy().astype(str), chunk=idx["group"].to_numpy().astype(str), type=idx["type"].to_numpy().astype(str),
             source=idx["source"].to_numpy().astype(str), sample_id=idx["sample_id"].to_numpy().astype(str), anchor_t=at, ego=ego, traj=traj, chain=chain)
    info = {"at": time.strftime("%F %T"), "clips": len(jobs), "moments": int(n), "chunks": int(idx["group"].nunique()), "moments_with_4s_history": int((chain >= 8).sum()),
            "clips_with_such_moments": int(pd.Series(idx["clip"].to_numpy()[chain >= 8]).nunique()), "wall_s": round(time.time() - t0, 1)}
    (RUN / "extract.json").write_text(json.dumps(info, indent=1) + "\n"); print(json.dumps(info)); print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
