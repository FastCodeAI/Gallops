"""Part A: build the frame cache (resumable) and check it.

    ./run.sh src/a_cache.py build     # NAS -> local shards, <= 16 parallel readers
    ./run.sh src/a_cache.py gate      # Gate A: 300 random pairs vs s1 pairs.build_pair

Only train (482) and selection (50) scenes from s1 splits.json are read.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import cache as K
import pairs as PR
import splits as S
from s2common import CACHE, OUT, SEED, write_json

READERS = 16
SHARD_FRAMES = 2000  # ~786 MB per shard


def plan(data: dict) -> tuple[list[tuple[str, int]], dict[str, int]]:
    scenes = sorted(data["train"]) + sorted(data["selection"])
    with ThreadPoolExecutor(READERS) as pool:
        n = dict(zip(scenes, pool.map(lambda s: PR.scene_num_frames(data["scene_paths"][s]), scenes)))
    frames = [(s, t) for s in scenes for t in K.frames_for_scene(n[s])]
    return frames, n


def build() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    data = S.load_splits()
    t0 = time.time()
    frames, n = plan(data)
    shards = [frames[i:i + SHARD_FRAMES] for i in range(0, len(frames), SHARD_FRAMES)]
    print(f"{len(frames)} frames, {len(shards)} shards; plan {time.time() - t0:.0f}s", flush=True)
    log = {"started": time.strftime("%F %T"), "frames": len(frames), "shards": len(shards), "per_shard": []}
    paths = data["scene_paths"]
    with ThreadPoolExecutor(READERS) as pool:
        for k, shard in enumerate(shards):
            meta_path = CACHE / f"shard_{k:03d}.json"
            if meta_path.exists() and json.loads(meta_path.read_text())["frames"] == [list(f) for f in shard]:
                print(f"shard {k} done, skip", flush=True)
                continue
            ts = time.time()
            tmp = CACHE / f"shard_{k:03d}.bin.tmp"
            digest = hashlib.sha256()
            with open(tmp, "wb") as fh:
                for arr in pool.map(lambda f: K.frame_array(paths[f[0]], f[1]), shard):
                    b = np.ascontiguousarray(arr).tobytes()
                    digest.update(b)
                    fh.write(b)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(CACHE / f"shard_{k:03d}.bin")
            write_json(meta_path, {"frames": [list(f) for f in shard], "sha256": digest.hexdigest(),
                                   "bytes": len(shard) * K.FRAME_BYTES})
            dt = time.time() - ts
            log["per_shard"].append({"shard": k, "frames": len(shard), "s": round(dt, 1)})
            print(f"shard {k}/{len(shards)}: {len(shard)} frames in {dt:.0f}s ({len(shard) / dt:.2f} frames/s)", flush=True)
            PR.scene_adapter.cache_clear()

    index, scene_frames = {}, {}
    for k in range(len(shards)):
        meta = json.loads((CACHE / f"shard_{k:03d}.json").read_text())
        for slot, (s, t) in enumerate(meta["frames"]):
            index[f"{s}:{t}"] = [k, slot]
    for s in n:
        scene_frames[s] = n[s]
    write_json(CACHE / "index.json", {"index": index, "scene_frames": scene_frames,
                                      "frame_shape": [K.H, K.W, 3], "dtype": "uint8"})
    log["finished"] = time.strftime("%F %T")
    log["wall_s"] = round(time.time() - t0, 1)
    write_json(OUT / "a_build_log.json", log)
    print("done", log["wall_s"], "s")


def gate() -> None:
    import common as C
    import lora as L

    data = S.load_splits()
    meta = json.loads((CACHE / "index.json").read_text())
    nf = meta["scene_frames"]
    checks = {}
    for meta_path in sorted(CACHE.glob("shard_*.json")):
        m = json.loads(meta_path.read_text())
        checks[meta_path.stem] = C.sha256_file(meta_path.with_suffix(".bin")) == m["sha256"]
    print("shard checksums", checks, flush=True)

    counts = {}
    for sub in ("train", "selection"):
        counts[sub] = {"scenes": len(data[sub]),
                       "frames": sum(len(K.frames_for_scene(nf[s])) for s in data[sub]),
                       "pairs": sum(len(K.pair_ends(nf[s])) for s in data[sub])}
    print(counts, flush=True)

    rng = random.Random(SEED)
    picks = []
    for sub, n in (("train", 250), ("selection", 50)):
        items = [(s, t) for s in data[sub] for t in K.pair_ends(nf[s])]
        picks += [(sub, *x) for x in rng.sample(items, n)]
    fc = K.FrameCache()

    def one(p):
        sub, s, t = p
        a = PR.build_pair(data["scene_paths"][s], t)
        b = fc.pair(s, t)
        return {"subset": sub, "scene": s, "t": t, "bit_equal": L.bit_equal(a, b),
                "max_abs_diff": float((a - b).abs().max())}

    t0 = time.time()
    with ThreadPoolExecutor(READERS) as pool:
        rows = list(pool.map(one, picks))
    n_eq = sum(r["bit_equal"] for r in rows)
    sizes = sum((CACHE / f"{k}.bin").stat().st_size for k in checks)
    out = {"gate_A_pass": n_eq == len(rows) and all(checks.values()),
           "pairs_checked": len(rows), "bit_equal": n_eq,
           "by_subset": {sub: sum(r["bit_equal"] for r in rows if r["subset"] == sub) for sub in ("train", "selection")},
           "worst_max_abs_diff": max(r["max_abs_diff"] for r in rows),
           "shard_checksums_ok": all(checks.values()), "shards": len(checks),
           "cache_bytes": sizes, "cache_GB": round(sizes / 1e9, 2), "counts": counts,
           "check_s": round(time.time() - t0, 1), "failures": [r for r in rows if not r["bit_equal"]][:10]}
    write_json(OUT / "a_gate.json", {**out, "rows": rows})
    print(json.dumps(out, indent=1))
    if not out["gate_A_pass"]:
        print("STOP: Gate A failed")
        sys.exit(2)


if __name__ == "__main__":
    {"build": build, "gate": gate}[sys.argv[1]]()
