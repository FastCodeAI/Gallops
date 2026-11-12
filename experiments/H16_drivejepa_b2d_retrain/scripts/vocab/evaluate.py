"""Step 3 of the Bench2Drive path vocabularies (b2d_drivejepa/vocab/CLAUDE.md): groups, coverage and speed tables. Numbers only.

Coverage: for each expert path, the smallest ADE to any vocabulary path; ADE = mean over the 8 points (0.5 .. 4.0 s) of the (x, y)
distance in metres. Exact: every path against every vocabulary path, on the files as saved (float32). The six sizes and Drive-JEPA's
NAVSIM 8,192, separately for the training paths and the held-out paths.

Speed, one definition for every row: the 8 steps from (0, 0) at t through the 8 points, each step's distance / 0.5 s. Path mean speed =
mean of the 8; path peak speed = largest of the 8. A vocabulary's top = largest over its paths; the expert's p99 and max = over its paths.

Writes (refuses to overwrite): coverage.csv, coverage_per_scenario.csv, groups.csv, speed.csv, nearest_<vocab>.npz."""
import csv
import json
import os
import resource
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (256, 512, 1024, 2048, 4096, 8192)
NAVSIM = "{workspace_root}/wa-jepa/Drive-JEPA-upstream/navsim_v2/data/8192.npy"
VOCABS = [(f"b2d_{k}", f"{HERE}/b2d_{k}.npy") for k in SIZES] + [("navsim_8192", NAVSIM)]
SPLIT_NAMES = {0: "train", 1: "heldout"}
CHUNK = 128


def nearest_ade(paths, vocab):
    """paths (N, 8, 2), vocab (K, 8, 2) float32 -> smallest ADE (N,) and its vocabulary index (N,)."""
    p = torch.from_numpy(np.ascontiguousarray(paths))
    v = torch.from_numpy(np.ascontiguousarray(vocab))
    ade, idx = torch.empty(len(p)), torch.empty(len(p), dtype=torch.int64)
    for i in range(0, len(p), CHUNK):
        c = p[i:i + CHUNK, None]                                         # (B, 1, 8, 2)
        d = torch.linalg.vector_norm(c - v[None], dim=-1).mean(-1)       # (B, K)
        ade[i:i + CHUNK], idx[i:i + CHUNK] = d.min(1)
    return ade.numpy(), idx.numpy()


def path_speeds(p8):
    """(N, 8, 2) -> mean speed (N,), peak speed (N,), m/s, 0.5 s steps from the origin."""
    q = np.concatenate([np.zeros((len(p8), 1, 2)), np.asarray(p8, np.float64)], 1)
    v = np.linalg.norm(np.diff(q, axis=1), axis=-1) / 0.5
    return v.mean(1), v.max(1)


def stats(a):
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()))


def main():
    outs = [f"{HERE}/{n}" for n in ("coverage.csv", "coverage_per_scenario.csv", "groups.csv", "speed.csv")]
    outs += [f"{HERE}/nearest_{n}.npz" for n, _ in VOCABS]
    for t in outs:
        if os.path.exists(t):
            sys.exit(f"STOP: {t} exists, not overwriting")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    z = np.load(f"{HERE}/paths_cache.npz")
    rear, split, clip_id, names = z["rear"], z["split"], z["clip_id"], z["clip_names"]
    p8 = rear[:, 4::5, :2].astype(np.float32)
    scen_of_clip = np.asarray([c.split("_Town")[0] for c in names])
    scen = scen_of_clip[clip_id]

    t0 = time.time()
    cov, per_scen = [], []
    for name, path in VOCABS:
        poses = np.load(path)
        ade, idx = nearest_ade(p8, poses[:, 4::5, :2])
        with open(f"{HERE}/nearest_{name}.npz", "xb") as fh:
            np.savez(fh, ade=ade, idx=idx.astype(np.int32))
        for sid, sn in SPLIT_NAMES.items():
            m = split == sid
            cov.append(dict(vocab=name, K=len(poses), split=sn, n_paths=int(m.sum()), **stats(ade[m])))
            for s in sorted(set(scen[m])):
                ms = m & (scen == s)
                per_scen.append(dict(scenario=s, split=sn, vocab=name, K=len(poses), n_paths=int(ms.sum()),
                                     n_clips=len(set(clip_id[ms])), **stats(ade[ms])))
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)

    with open(outs[0], "x", newline="") as fh:
        w = csv.DictWriter(fh, list(cov[0]))
        w.writeheader()
        w.writerows(cov)
    with open(outs[1], "x", newline="") as fh:
        w = csv.DictWriter(fh, list(per_scen[0]))
        w.writeheader()
        w.writerows(per_scen)

    tr = split == 0
    train_len = path_speeds(p8[tr])[0] * 4.0                              # polyline length of each training path, metres
    g = []
    for k in SIZES:
        lab = np.load(f"{HERE}/groups_b2d_{k}.npy")
        assert len(lab) == tr.sum()
        c = np.bincount(lab, minlength=k)
        big = int(c.argmax())
        vb = np.load(f"{HERE}/b2d_{k}.npy")[big:big + 1, 4::5, :2]
        g.append(dict(K=k, min=int(c.min()), median=float(np.median(c)), max=int(c.max()), single_path_groups=int((c == 1).sum()),
                      largest_group_centre_length_m=float(path_speeds(vb)[0][0] * 4.0),
                      largest_group_share_under_0p1m=float((train_len[lab == big] < 0.1).mean())))
    with open(outs[2], "x", newline="") as fh:
        w = csv.DictWriter(fh, list(g[0]))
        w.writeheader()
        w.writerows(g)

    sp = []
    for name, path in VOCABS:
        poses = np.load(path)
        mean_v, peak_v = path_speeds(poses[:, 4::5, :2])
        sp.append(dict(row=name, n_paths=len(poses), kind="vocabulary_top", mean_speed=float(mean_v.max()), peak_speed=float(peak_v.max())))
    for sid, sn in SPLIT_NAMES.items():
        mean_v, peak_v = path_speeds(rear[split == sid, 4::5, :2])
        sp.append(dict(row=f"expert_{sn}", n_paths=len(mean_v), kind="p99", mean_speed=float(np.percentile(mean_v, 99)),
                       peak_speed=float(np.percentile(peak_v, 99))))
        sp.append(dict(row=f"expert_{sn}", n_paths=len(mean_v), kind="max", mean_speed=float(mean_v.max()), peak_speed=float(peak_v.max())))
    with open(outs[3], "x", newline="") as fh:
        w = csv.DictWriter(fh, list(sp[0]))
        w.writeheader()
        w.writerows(sp)
    print(json.dumps(dict(seconds=round(time.time() - t0, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))))


if __name__ == "__main__":
    main()
