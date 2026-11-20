"""Build 2, step 1b: one sequential pass over approach 3's training token cache into local memory-mapped arrays (Sid, 2026-11-17, option (b);
destination exception for this lane: {approach4_store}/tokens/).

Source: {experiment_store_h17}/cache/training/<clip>.pt for the 4,015 clips of the training manifest (pai_common.MANIFEST, sha-checked
by pai_train.manifests). Each file is read once as bytes, its sha256 checked against index_training.parquet, then torch.load'ed from memory.
Order of rows: clips sorted by uuid, moments in the file's own (time) order; asserted equal to the index's `row` order.

Written (temp names *.tmp, renamed only when every clip is in and the checks pass):
  tokens.npy   (M, 512, 256) fp32, np.lib.format (np.load(mmap_mode="r") reads it)
  pooled.npy   (M, 33, 256) fp32
  index.parquet  sample_id, clip, anchor_t, row (into the two arrays), has_label2, and the per-moment values the cache file carries beside
                 the tokens, copied as stored: ego_status (11, fp32, the converter's accelerations kept; the loader zeroes slots 5:7 as
                 dataset_s35._finish does) and trajectory (8 x 3, float64 as stored), so the loader never opens a NAS clip file.
  copy.json      timing, sizes, per-file sha check, the 640-row check
Check: 640 random rows (default_rng(1)) of tokens and pooled equal torch.load of the source file bit for bit; row count = manifest moments.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path("{experiment_store_h17}/cache/training")
DST = Path("{approach4_store}/tokens")
LOG_EVERY = 200


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def main():
    import pai_train as T
    tr, _, _ = T.manifests()                                        # sha-checked manifest, 126,981 moments / 4,015 clips
    idx = pd.read_parquet(SRC / "index_training.parquet")
    assert set(idx.sample_id) == set(tr.sample_id) and len(idx) == len(tr), "cache index != manifest"
    idx = idx.sort_values(["clip", "row"]).reset_index(drop=True)
    clips = sorted(idx["clip"].unique())
    assert set(clips) == set(tr.clip_uuid) and len(clips) == 4015
    M = len(idx)
    DST.mkdir(parents=True, exist_ok=True)
    tmp = {k: DST / f"{k}.npy.tmp" for k in ("tokens", "pooled")}
    tok = np.lib.format.open_memmap(tmp["tokens"], mode="w+", dtype=np.float32, shape=(M, 512, 256))
    poo = np.lib.format.open_memmap(tmp["pooled"], mode="w+", dtype=np.float32, shape=(M, 33, 256))
    rows, ego, traj = [], np.zeros((M, 11), np.float32), np.zeros((M, 8, 3), np.float64)
    t0 = time.time(); nbytes = 0; r0 = 0; sha_ok = 0
    by_clip = {c: g for c, g in idx.groupby("clip", sort=False)}
    for k, c in enumerate(clips):
        g = by_clip[c]
        b = (SRC / g.file.iloc[0]).read_bytes(); nbytes += len(b)
        assert hashlib.sha256(b).hexdigest() == g.sha256.iloc[0], f"sha mismatch {c}"; sha_ok += 1
        rec = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        n = len(g)
        assert rec["sample_id"] == list(g.sample_id) and list(g.row) == list(range(n)) and rec["tokens"].shape == (n, 512, 256), c
        assert rec["tokens"].dtype == torch.float32 and rec["pooled"].shape == (n, 33, 256) and rec["pooled"].dtype == torch.float32
        tok[r0:r0 + n] = rec["tokens"].numpy(); poo[r0:r0 + n] = rec["pooled"].numpy()
        ego[r0:r0 + n] = rec["ego_status"].numpy(); traj[r0:r0 + n] = rec["trajectory"].numpy()
        assert np.allclose(np.asarray(rec["anchor_t"], np.float64), g.anchor_t.to_numpy())
        rows.append(pd.DataFrame({"sample_id": rec["sample_id"], "clip": c, "anchor_t": g.anchor_t.to_numpy(), "row": np.arange(r0, r0 + n),
                                  "has_label2": torch.as_tensor(rec["has_label2"]).bool().numpy()}))
        r0 += n
        if (k + 1) % LOG_EVERY == 0 or k + 1 == len(clips):
            dt = time.time() - t0
            log(f"{k + 1}/{len(clips)} clips, {r0}/{M} rows, {nbytes / 1e9:.1f} GB read, {dt:.0f} s, {nbytes / 1e6 / dt:.1f} MB/s, "
                f"load {open('/proc/loadavg').read().split()[0]}")
    assert r0 == M
    tok.flush(); poo.flush(); del tok, poo
    t_copy = time.time() - t0
    out = pd.concat(rows, ignore_index=True)
    out["ego_status"] = list(ego); out["trajectory"] = list(traj.reshape(M, 24))
    assert out.sample_id.is_unique and len(out) == len(tr)
    # check: 640 random rows vs torch.load of the source file
    tk = np.load(tmp["tokens"], mmap_mode="r"); po = np.load(tmp["pooled"], mmap_mode="r")
    rng = np.random.default_rng(1); pick = np.sort(rng.choice(M, 640, replace=False))
    eq_t = eq_p = eq_e = 0; cache = {}
    for r in pick:
        c = out["clip"].iloc[r]
        if c not in cache:
            cache = {c: torch.load(SRC / f"{c}.pt", map_location="cpu", weights_only=False)}
        rec = cache[c]; j = rec["sample_id"].index(out.sample_id.iloc[r])
        eq_t += int(np.array_equal(tk[r], rec["tokens"][j].numpy())); eq_p += int(np.array_equal(po[r], rec["pooled"][j].numpy()))
        eq_e += int(np.array_equal(out.ego_status.iloc[r], rec["ego_status"][j].numpy()))
    check = {"rows_checked": 640, "tokens_bit_equal": eq_t, "pooled_bit_equal": eq_p, "ego_equal": eq_e, "rows": M, "manifest_moments": len(tr)}
    log(f"check {check}")
    assert eq_t == eq_p == eq_e == 640 and M == len(tr)
    out.to_parquet(DST / "index.parquet.tmp")
    for k in ("tokens", "pooled"):
        os.replace(tmp[k], DST / f"{k}.npy")
    os.replace(DST / "index.parquet.tmp", DST / "index.parquet")
    rec = {"source": str(SRC), "clips": len(clips), "rows": M, "source_bytes_read": nbytes, "source_sha_checked": sha_ok,
           "copy_wall_s": round(t_copy, 1), "total_wall_s": round(time.time() - t0, 1),
           "sizes_bytes": {p.name: p.stat().st_size for p in sorted(DST.iterdir()) if p.suffix in (".npy", ".parquet")}, "check": check,
           "finished": time.strftime("%F %T")}
    (DST / "copy.json").write_text(json.dumps(rec, indent=1))
    log(json.dumps(rec))


def finish():
    """2026-11-18: the first run wrote every clip (log: 4015/4015 clips, 126981 rows, then tok/poo flushed) and crashed in the check on
    `out.clip` (pandas resolves it to DataFrame.clip, the method; fixed above). The in-memory index was lost, the two .tmp arrays were
    complete. This rebuilds the index from the source files with mmap reads (sample_id, ego_status, trajectory, has_label2, anchor_t only;
    the tokens are not re-read), in the same row order as main(), then checks: (1) the whole pooled array equals approach 3's
    pooled_training.npy (every moment, mapped by sample_id); (2) no all-zero token row; (3) the 640-row bit-for-bit check of main()."""
    from concurrent.futures import ThreadPoolExecutor
    import pai_train as T
    tr, _, _ = T.manifests()
    t0 = time.time()
    idx = pd.read_parquet(SRC / "index_training.parquet").sort_values(["clip", "row"]).reset_index(drop=True)
    clips = sorted(idx["clip"].unique()); M = len(idx)
    by_clip = {c: g for c, g in idx.groupby("clip", sort=False)}
    tmp = {k: DST / f"{k}.npy.tmp" for k in ("tokens", "pooled")}
    tok = np.load(tmp["tokens"], mmap_mode="r"); poo = np.load(tmp["pooled"], mmap_mode="r")
    assert tok.shape == (M, 512, 256) and poo.shape == (M, 33, 256)

    def meta(c):
        g = by_clip[c]
        rec = torch.load(SRC / g.file.iloc[0], map_location="cpu", weights_only=False, mmap=True)
        assert rec["sample_id"] == list(g.sample_id), c
        return c, list(rec["sample_id"]), np.asarray(rec["anchor_t"], np.float64), rec["ego_status"].numpy().copy(), \
            rec["trajectory"].numpy().copy(), torch.as_tensor(rec["has_label2"]).bool().numpy().copy()
    with ThreadPoolExecutor(8) as ex:
        metas = list(ex.map(meta, clips))
    log(f"metadata read {time.time() - t0:.0f} s")
    rows, ego, traj, r0 = [], np.zeros((M, 11), np.float32), np.zeros((M, 8, 3), np.float64), 0
    for c, sids, at, e, tj, lab in metas:
        n = len(sids)
        assert np.allclose(at, by_clip[c].anchor_t.to_numpy())
        ego[r0:r0 + n] = e; traj[r0:r0 + n] = tj
        rows.append(pd.DataFrame({"sample_id": sids, "clip": c, "anchor_t": at, "row": np.arange(r0, r0 + n), "has_label2": lab}))
        r0 += n
    assert r0 == M
    out = pd.concat(rows, ignore_index=True)
    out["ego_status"] = list(ego); out["trajectory"] = list(traj.reshape(M, 24))
    assert out.sample_id.is_unique and len(out) == len(tr) and set(out.sample_id) == set(tr.sample_id)
    # (1) pooled, every row, vs approach 3's pooled array (index order = pooled_row)
    ref = np.load(SRC / "pooled_training.npy", mmap_mode="r")
    src_row = idx.set_index("sample_id").pooled_row.reindex(out.sample_id).to_numpy()
    pooled_equal = 0
    for s in range(0, M, 4096):
        a = np.asarray(poo[s:s + 4096]).reshape(-1, 33 * 256); b = np.asarray(ref[np.sort(src_row[s:s + 4096])])
        order = np.argsort(np.argsort(src_row[s:s + 4096]))
        pooled_equal += int((a == b[order]).all(1).sum())
    log(f"pooled equal {pooled_equal}/{M} ({time.time() - t0:.0f} s)")
    # (2) no all-zero token row
    zero_rows = 0
    for s in range(0, M, 2048):
        zero_rows += int((~np.asarray(tok[s:s + 2048]).reshape(-1, 512 * 256).any(1)).sum())
    log(f"all-zero token rows {zero_rows} ({time.time() - t0:.0f} s)")
    # (3) 640 random rows vs torch.load of the source file (main()'s check)
    rng = np.random.default_rng(1); pick = np.sort(rng.choice(M, 640, replace=False))
    eq_t = eq_p = eq_e = 0; cache = {}
    for r in pick:
        c = out["clip"].iloc[r]
        if c not in cache:
            cache = {c: torch.load(SRC / f"{c}.pt", map_location="cpu", weights_only=False)}
        rec = cache[c]; j = rec["sample_id"].index(out.sample_id.iloc[r])
        eq_t += int(np.array_equal(tok[r], rec["tokens"][j].numpy())); eq_p += int(np.array_equal(poo[r], rec["pooled"][j].numpy()))
        eq_e += int(np.array_equal(out.ego_status.iloc[r], rec["ego_status"][j].numpy()))
    check = {"rows_checked": 640, "tokens_bit_equal": eq_t, "pooled_bit_equal": eq_p, "ego_equal": eq_e, "rows": M, "manifest_moments": len(tr),
             "pooled_all_rows_equal_approach3_pooled_array": f"{pooled_equal}/{M}", "all_zero_token_rows": zero_rows}
    log(f"check {check}")
    assert eq_t == eq_p == eq_e == 640 and M == len(tr) and pooled_equal == M and zero_rows == 0
    del tok, poo
    out.to_parquet(DST / "index.parquet.tmp")
    for k in ("tokens", "pooled"):
        os.replace(tmp[k], DST / f"{k}.npy")
    os.replace(DST / "index.parquet.tmp", DST / "index.parquet")
    rec = {"source": str(SRC), "clips": len(clips), "rows": M, "source_bytes_read_first_run": 70.9e9, "copy_wall_s_first_run": 1480,
           "first_run_note": "every clip written and sha-checked; crashed in its own check (out.clip bug); finish() rebuilt the index",
           "finish_wall_s": round(time.time() - t0, 1),
           "sizes_bytes": {p.name: p.stat().st_size for p in sorted(DST.iterdir()) if p.suffix in (".npy", ".parquet")}, "check": check,
           "finished": time.strftime("%F %T")}
    (DST / "copy.json").write_text(json.dumps(rec, indent=1))
    log(json.dumps(rec))


if __name__ == "__main__":
    sys.exit(finish() if sys.argv[1:] == ["finish"] else main())
