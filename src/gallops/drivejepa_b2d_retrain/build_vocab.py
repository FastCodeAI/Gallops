"""Step 2 of the Bench2Drive path vocabularies (specs/vocab.md in H16's folder): k-means on the training
paths, one run per size.

Same paths and settings at every size: sklearn KMeans on the 8 (x, y) points at 0.5 .. 4.0 s (16 numbers per
path, metres, no scaling,
every training path weight 1), k-means++ start, n_init 1, Lloyd, tol 0 so a run stops only when no label
changes.

A centre's 40 poses are the mean of its members at each of the 40 steps, with heading the circular mean, so
the 8 clustered points of
the saved file are the k-means centre. File layout is Drive-JEPA's data/8192.npy: (K, 40, 3) float32, 0.1 ..
4.0 s, so that
poses[:, 4::5] gives (K, 8, 3).

Writes per size (refuses to overwrite): b2d_<K>.npy, groups_b2d_<K>.npy (the group of each training path, in
cache order),
kmeans_<K>.json."""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
from sklearn import __version__ as sk_version
from sklearn.cluster import KMeans

SIZES = (256, 512, 1024, 2048, 4096, 8192)
SEED = 20261105
MAX_ITER = 10000
TOL = 0.0
SETTINGS = dict(init="k-means++", n_init=1, algorithm="lloyd", max_iter=MAX_ITER, tol=TOL, random_state=SEED)


def member_means(poses, labels, k):
    """poses (N, 40, 3), labels (N,) -> (k, 40, 3): mean x, y and circular-mean heading per group and step;
    min resultant length."""
    n = np.bincount(labels, minlength=k).astype(np.float64)
    out = np.empty((k, poses.shape[1], 3))
    r_min = 1.0
    for j in range(poses.shape[1]):
        out[:, j, 0] = np.bincount(labels, poses[:, j, 0], k) / n
        out[:, j, 1] = np.bincount(labels, poses[:, j, 1], k) / n
        s = np.bincount(labels, np.sin(poses[:, j, 2]), k) / n
        c = np.bincount(labels, np.cos(poses[:, j, 2]), k) / n
        out[:, j, 2] = np.arctan2(s, c)
        r_min = min(r_min, float(np.hypot(s, c).min()))
    return out, r_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    a = ap.parse_args()
    for k in a.sizes:
        for n in (f"b2d_{k}.npy", f"groups_b2d_{k}.npy", f"kmeans_{k}.json"):
            if os.path.exists(f"{a.out}/{n}"):
                sys.exit(f"STOP: {a.out}/{n} exists, not overwriting")

    z = np.load(a.cache)
    poses = z["rear"][z["split"] == 0]  # training paths only
    X = np.ascontiguousarray(poses[:, 4::5, :2].reshape(len(poses), 16))
    print(
        f"{len(X)} training paths, sklearn {sk_version}, threads OMP={os.environ.get('OMP_NUM_THREADS')}",
        flush=True,
    )

    for k in a.sizes:
        t0 = time.time()
        km = KMeans(n_clusters=k, **SETTINGS).fit(X)
        fit_s = time.time() - t0
        labels = km.labels_.astype(np.int32)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < 1:
            sys.exit(
                f"STOP: K={k} has {int((counts == 0).sum())} empty groups, nothing written for this size"
            )
        vocab, r_min = member_means(poses, labels, k)
        centres = km.cluster_centers_.reshape(k, 8, 2)
        saved = np.ascontiguousarray(vocab.astype(np.float32))
        info = dict(
            K=k,
            seed=SEED,
            settings={n: v for n, v in SETTINGS.items()},
            sklearn=sk_version,
            n_paths=len(X),
            n_iter=int(km.n_iter_),
            converged=bool(km.n_iter_ < MAX_ITER),
            inertia=float(km.inertia_),
            fit_seconds=round(fit_s, 1),
            labels_unchanged_on_final_centres=float((km.predict(X) == labels).mean()),
            max_gap_member_mean_vs_kmeans_centre_m=float(np.abs(vocab[:, 4::5, :2] - centres).max()),
            max_gap_saved_float32_vs_kmeans_centre_m=float(
                np.abs(saved[:, 4::5, :2].astype(np.float64) - centres).max()
            ),
            min_heading_resultant_length=r_min,
            peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        )
        with open(f"{a.out}/b2d_{k}.npy", "xb") as fh:
            np.save(fh, saved)
        with open(f"{a.out}/groups_b2d_{k}.npy", "xb") as fh:
            np.save(fh, labels)
        with open(f"{a.out}/kmeans_{k}.json", "x") as fh:
            json.dump(info, fh, indent=1)
        print(json.dumps({n: v for n, v in info.items() if n != "settings"}), flush=True)


if __name__ == "__main__":
    main()
