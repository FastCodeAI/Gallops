"""Build 2, step 1b re-measure: the reader benchmark on the local token copy. 20 batches x 32 random rows (default_rng(0)), cold: the whole
tokens.npy and pooled.npy get posix_fadvise(DONTNEED) before every batch. Fetch = np.load(mmap_mode="r") once, then a copy of each row
(tokens and pooled), exactly what a4_data.TokenSet.item does. Output MH/out/local_reader_bench.json."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import a4_config as CFG

OUT = Path(__file__).resolve().parents[1] / "out"


def drop(p):
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    tok = np.load(CFG.TOKENS / "tokens.npy", mmap_mode="r"); poo = np.load(CFG.TOKENS / "pooled.npy", mmap_mode="r")
    rng = np.random.default_rng(0); per = []
    for b in range(20):
        rows = rng.choice(len(tok), 32, replace=False)
        for p in ("tokens.npy", "pooled.npy"):
            drop(CFG.TOKENS / p)
        t0 = time.perf_counter()
        x = [(np.array(tok[r]), np.array(poo[r])) for r in rows]
        per.append(time.perf_counter() - t0)
        assert x[0][0].shape == (512, 256)
    res = {"batches": 20, "batch": 32, "s_per_batch_mean": float(np.mean(per)), "s_per_batch_median": float(np.median(per)),
           "s_per_batch_max": float(np.max(per)), "per_batch_s": [round(v, 4) for v in per]}
    OUT.mkdir(exist_ok=True)
    (OUT / "local_reader_bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_batch_s"}))


if __name__ == "__main__":
    main()
