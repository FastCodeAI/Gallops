"""Build 2, step 1: three ways of fetching the tokens of 32 random moments from 32 different clips, 20 batches each, cold.

Cache = {experiment_store_h17}/cache/training/<clip>.pt (torch zip, stored uncompressed; tokens (N, 512, 256) fp32 = archive/data/1 in
every file checked here, located by name through the pickle, not assumed). Moments come from index_training.parquet (file, row).
  (i)   torch.load(whole file) and take tokens[row]
  (ii)  torch.load(mmap=True) and take tokens[row].clone()
  (iii) zipfile: read only the tokens entry (whole entry), then take the row
Cold: every method draws its own 640 clips (no clip shared between methods or batches, default_rng(0)), and each file gets
posix_fadvise(DONTNEED) right before it is read. MB read = the NFS mount's READ bytes from /proc/self/mountstats (whole mount, so other
users' NAS reads in the same seconds are included) and, beside it, the bytes this process asked for (rchar delta; mmap page-ins are not
in rchar). Correctness: every fetched row equals method (i)'s row for the same moment bit for bit (method (i) re-reads those 640 x 2 files
afterwards, warm, for the comparison only). Output: MH/out/reader_bench.json.
"""
from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CACHE = Path("{experiment_store_h17}/cache/training")
OUT = Path(__file__).resolve().parents[1] / "out"
NB, BS = 20, 32


def nfs_read_bytes() -> int:
    """READ bytes for the {nas} nfs4 mount (per-op line 'READ: ops trans timeouts bytes_sent bytes_recv ...')."""
    on = False
    for line in open("/proc/self/mountstats"):
        if line.startswith("device "):
            on = " mounted on {nas} with fstype nfs" in line
        elif on and line.strip().startswith("READ:"):
            return int(line.split()[5])
    raise RuntimeError("no READ line for {nas}")


def rchar() -> int:
    for line in open("/proc/self/io"):
        if line.startswith("rchar"):
            return int(line.split()[1])
    return 0


def drop(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def m_whole(path, row):
    return torch.load(path, map_location="cpu", weights_only=False)["tokens"][row].clone()


def m_mmap(path, row):
    r = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return r["tokens"][row].clone()


_ENTRY = {}


def tokens_entry(path) -> tuple[str, tuple]:
    """name of the zip entry holding `tokens` and its shape, read from the pickle (persistent ids) of this file."""
    import pickle
    with zipfile.ZipFile(path) as z:
        pk = next(n for n in z.namelist() if n.endswith("data.pkl"))
        prefix = pk[: -len("data.pkl")]
        found = {}

        class U(pickle.Unpickler):
            def persistent_load(self, pid):
                return ("PID", pid[2])                      # ('storage', storage_type, key, location, numel)

            def find_class(self, mod, name):
                if name == "_rebuild_tensor_v2":
                    def rb(storage, offset, size, stride, *a):
                        return ("T", storage[1], offset, tuple(size), tuple(stride))
                    return rb
                return super().find_class(mod, name)
        d = U(z.open(pk)).load()
        _, key, off, size, stride = d["tokens"]
        assert off == 0 and stride == (size[1] * size[2], size[2], 1), (off, stride)
        return prefix + "data/" + key, size


def m_zip(path, row):
    name, size = tokens_entry(path)
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        assert info.compress_type == zipfile.ZIP_STORED
        buf = z.read(name)
    a = np.frombuffer(buf, dtype=np.float32).reshape(size)
    return torch.from_numpy(a[row].copy())


METHODS = {"i_torch_load_whole": m_whole, "ii_torch_load_mmap": m_mmap, "iii_zip_tokens_entry": m_zip}


def main():
    idx = pd.read_parquet(CACHE / "index_training.parquet")
    clips = sorted(idx["clip"].unique())
    rng = np.random.default_rng(0)
    order = rng.permutation(len(clips))
    need = len(METHODS) * NB * BS
    assert need <= len(clips)
    by_clip = {c: g for c, g in idx.groupby("clip")}
    res = {"setup": {"batches": NB, "batch": BS, "files_total": need, "loadavg_start": open("/proc/loadavg").read().split()[:3]}}
    fetched = {}
    for mi, (mname, fn) in enumerate(METHODS.items()):
        mine = [clips[i] for i in order[mi * NB * BS:(mi + 1) * NB * BS]]
        per_batch, mb_nfs, mb_req, out = [], [], [], []
        for b in range(NB):
            items = []
            for c in mine[b * BS:(b + 1) * BS]:
                g = by_clip[c]
                r = g.iloc[int(rng.integers(len(g)))]
                items.append((CACHE / r["file"], int(r["row"]), r["sample_id"]))
            for p, _, _ in items:
                drop(p)
            n0, r0, t0 = nfs_read_bytes(), rchar(), time.perf_counter()
            toks = [fn(p, row) for p, row, _ in items]
            dt = time.perf_counter() - t0
            per_batch.append(dt); mb_nfs.append((nfs_read_bytes() - n0) / 1e6); mb_req.append((rchar() - r0) / 1e6)
            out += [(p, row, sid, t) for (p, row, sid), t in zip(items, toks)]
        fetched[mname] = out
        res[mname] = {"s_per_batch_mean": float(np.mean(per_batch)), "s_per_batch_median": float(np.median(per_batch)),
                      "s_per_batch_max": float(np.max(per_batch)), "s_first_batch": per_batch[0],
                      "nfs_MB_per_batch_mean": float(np.mean(mb_nfs)), "requested_MB_per_batch_mean": float(np.mean(mb_req)),
                      "per_batch_s": [round(x, 3) for x in per_batch]}
        print(mname, json.dumps({k: v for k, v in res[mname].items() if k != "per_batch_s"}), flush=True)
    # correctness: each fetched row vs method (i) on the same file and row
    for mname, out in fetched.items():
        eq = sum(int(torch.equal(t, m_whole(p, row))) for p, row, _, t in out)
        res[mname]["bit_equal_to_i"] = f"{eq}/{len(out)}"
        res[mname]["dtype_shape"] = [str(out[0][3].dtype), list(out[0][3].shape)]
    res["setup"]["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    OUT.mkdir(exist_ok=True)
    (OUT / "reader_bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("s_per_batch_mean"), v.get("nfs_MB_per_batch_mean"), v.get("bit_equal_to_i")) for k, v in res.items() if k != "setup"}))


if __name__ == "__main__":
    main()
