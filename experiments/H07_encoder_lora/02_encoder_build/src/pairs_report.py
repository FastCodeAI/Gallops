"""Part B.3: pair counts and read speed from NAS (CPU only).

    ./run.sh src/pairs_report.py [--workers 1 4 8] [--n 48]
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

import common as C
import splits as S
from pairs import PairDataset


def throughput(ds: PairDataset, idx: list[int], workers: int, batch: int = 8) -> float:
    sub = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(sub, batch_size=batch, num_workers=workers, shuffle=False)
    t0 = time.perf_counter()
    n = 0
    for pairs, _, _ in loader:
        n += pairs.shape[0]
    return n / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--n", type=int, default=48, help="pairs per throughput measurement")
    args = ap.parse_args()

    data = S.load_splits()
    t0 = time.perf_counter()
    train = PairDataset(data["train"], data["scene_paths"])
    selection = PairDataset(data["selection"], data["scene_paths"])
    index_s = time.perf_counter() - t0
    out = {
        "pairs_train": len(train),
        "pairs_selection": len(selection),
        "frames_train": sum(train.frames.values()),
        "frames_selection": sum(selection.frames.values()),
        "index_build_seconds": round(index_s, 1),
        "frames_per_scene_min_max": [min(train.frames.values()), max(train.frames.values())],
    }
    print(json.dumps(out))

    sample = train[0][0]
    out["pair_shape"] = list(sample.shape)
    out["pair_dtype"] = str(sample.dtype)

    # Different random pairs per measurement, so no run reads from a page cache
    # warmed by an earlier one.
    rng = random.Random(0)
    pool = rng.sample(range(len(train)), args.n * len(args.workers))
    rates = {}
    for k, w in enumerate(args.workers):
        idx = pool[k * args.n:(k + 1) * args.n]
        rates[w] = round(throughput(train, idx, w), 2)
        print(f"workers {w}: {rates[w]} pairs/s")
    out["pairs_per_second_by_workers"] = rates
    (C.OUT / "pairs_report.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
