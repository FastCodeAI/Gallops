"""Build step 3: data for the in-order trainer and the in-order check, read from the build-2 cache.

Per clip ONE read of A3/cache/<slice>/<clip>.pt = {sample_id, anchor_t, tokens (N, 512, 256) fp32, pooled, ego_status (N, 11),
trajectory (N, 8, 3), has_label2} in time order (BUILD2_CACHE.md, "What is cached and how"). The acceleration columns of ego_status
(slots 5:7) are ZEROED here exactly as the model's loader does (dataset_s35.py:_finish, "[S35-1c] ... features['ego_status'][..., 5:7] = 0.0";
the cache kept the converter's values, s5_cache_full.py docstring). Not in the cache and read from their existing places:
  lidar2img (2, 4, 4) and img_shape (2, 3): the slim npz of the clip's FIRST moment, PAI/s4_train/slim/<clip>/<sid>.npz (keys written by
    dataset_s35.py SLIM_KEYS, :190-196); constant within a clip (out/s6_lidar2img_constancy.json), so one read per clip
  metric cache (grading) and anchor index (vocabulary term): PAI/s3_assembly/s1/cache/<clip>/<sid>/{metric_cache.pkl,
    anchors_scores_index_ext.npy} for map moments (dataset_s35.py:_finish sets the path; drive_jepa_agent_s10._index_path :176-179)
Windows: 16 consecutive moments (8 s) of one clip, anchors 0.5 s apart; a window never crosses a missing anchor (runs_of). Per clip and
pass: each run of n >= 16 consecutive moments gives n // 16 windows, each with a random start inside the run (seeded by seed and pass);
clips without a run of 16 are skipped (counted). Event over-sampling (flag, default on): a window holding any moment of
out/memory_moments_train.parquet (clip_id, anchor_t) is drawn twice. Batches of 2 windows (32 moments) in a seeded shuffle."""
from __future__ import annotations
import json, threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch
import pai_common as C

A3 = Path("{experiment_store}"); CACHE = A3 / "cache"; OUT = Path(__file__).resolve().parents[1] / "out"
WIN, WARM, BATCH_WINDOWS = 16, 4, 2
STEP_S = 0.5


def runs_of(anchor_t: np.ndarray) -> list[tuple[int, int]]:
    """(start_row, length) of the maximal runs of consecutive anchors (0.5 s apart) in a clip sorted by time."""
    out = []; st = 0
    for i in range(1, len(anchor_t) + 1):
        if i == len(anchor_t) or abs(anchor_t[i] - anchor_t[i - 1] - STEP_S) > 1e-6:
            out.append((st, i - st)); st = i
    return out


class ClipStore:
    """index of one cached slice + per-clip reads (tokens, ego with acceleration zeroed, human future, flags, camera matrices, paths)."""

    def __init__(self, which: str, slim_root=C.SLIM, cache_root=C.CACHE, keep_clips: int = 160):
        """keep_clips: clip records held in RAM (LRU, ~17 MB each; 160 = 2.7 GB) so a clip read twice (the check's replays, a smoke run's two
        modes) costs one NAS read: on 2026-11-09 the NAS delivered 0.6-4 MB/s while a disk-relief move ran (report flag)."""
        self.which = which; self.dir = CACHE / which; self.keep_clips = keep_clips; self._lru = OrderedDict(); self._lock = threading.Lock()
        self.idx = pd.read_parquet(self.dir / f"index_{which}.parquet").sort_values(["clip", "row"]).reset_index(drop=True)
        self.clips = list(dict.fromkeys(self.idx["clip"]))
        self.by_clip = {c: g for c, g in self.idx.groupby("clip", sort=False)}
        self.slim_root, self.cache_root = Path(slim_root), Path(cache_root)
        self._sidecar = {}

    def camera(self, clip: str):
        """lidar2img (2, 4, 4) float32 and img_shape (2, 3) from the slim npz of the clip's first moment (one read per clip, cached)."""
        if clip not in self._sidecar:
            sid = self.by_clip[clip].sample_id.iloc[0]
            with np.load(self.slim_root / clip / f"{sid}.npz") as z:
                self._sidecar[clip] = (torch.from_numpy(z["lidar2img"].astype(np.float32)), torch.from_numpy(z["img_shape"].astype(np.float32)))
        return self._sidecar[clip]

    def load(self, clip: str) -> dict:
        with self._lock:
            if clip in self._lru:
                self._lru.move_to_end(clip); return self._lru[clip]
        rec = self._read(clip)
        with self._lock:
            self._lru[clip] = rec
            while len(self._lru) > self.keep_clips:
                self._lru.popitem(last=False)
        return rec

    def _read(self, clip: str) -> dict:
        g = self.by_clip[clip]
        rec = torch.load(self.dir / g.file.iloc[0], map_location="cpu", weights_only=False)
        assert rec["sample_id"] == list(g.sample_id) and rec["tokens"].shape[0] == len(g), clip
        ego = rec["ego_status"].float().clone(); ego[..., 5:7] = 0.0                                      # dataset_s35.py _finish, zero_accel
        lab = torch.as_tensor(rec["has_label2"]).bool()
        mc = [str(self.cache_root / clip / s / "metric_cache.pkl") if f else "" for s, f in zip(g.sample_id, lab.tolist())]
        # the anchor index of every map moment (drive_jepa_agent_s10._index_path :176-179, read_scores_index :359-371), read here in the
        # prefetch thread rather than in the step (2026-11-09: 24 small NAS reads cost ~12 s inside a step, s6_all.log T2 warm step 2)
        scores_index = [np.load(Path(m).parent / "anchors_scores_index_ext.npy") if m else np.zeros(0, dtype=np.int64) for m in mc]
        l2i, shp = self.camera(clip)
        return {"clip": clip, "sample_id": list(g.sample_id), "anchor_t": np.asarray(rec["anchor_t"], np.float64), "tokens": rec["tokens"].float(),
                "ego": ego, "traj": rec["trajectory"].float(), "has_label2": lab, "metric_cache_path": mc, "scores_index": scores_index, "lidar2img": l2i, "img_shape": shp,
                "in_picking": g.in_picking.to_numpy() if "in_picking" in g else np.zeros(len(g), bool)}


def memory_moment_set(path=OUT / "memory_moments_train.parquet") -> set:
    mm = pd.read_parquet(path)
    return set(zip(mm.clip_id, np.round(mm.anchor_t.to_numpy().astype(float), 3)))


def plan_pass(store: ClipStore, seed: int, pass_idx: int, oversample: bool, mem: set | None) -> dict:
    """windows of one pass (module docstring) -> {"batches": [[(clip, start_row), (clip, start_row)], ...], counts}."""
    rng = np.random.RandomState(seed * 1000 + pass_idx)
    windows, dup, skipped = [], 0, 0
    for c in store.clips:
        g = store.by_clip[c]; at = g.anchor_t.to_numpy().astype(float)
        rs = [r for r in runs_of(at) if r[1] >= WIN]
        if not rs:
            skipped += 1; continue
        for st, n in rs:
            for s in rng.randint(0, n - WIN + 1, size=n // WIN):
                w = (c, int(st + s)); windows.append(w)
                if oversample and mem is not None and any((c, round(float(t), 3)) in mem for t in at[st + s: st + s + WIN]):
                    windows.append(w); dup += 1
    order = rng.permutation(len(windows)); windows = [windows[i] for i in order]
    if len(windows) % BATCH_WINDOWS:
        windows = windows[:-(len(windows) % BATCH_WINDOWS)]
    batches = [windows[i:i + BATCH_WINDOWS] for i in range(0, len(windows), BATCH_WINDOWS)]
    return {"batches": batches, "windows": len(windows), "duplicated_event_windows": dup, "clips_skipped_no_16_run": skipped,
            "clips_used": len(store.clips) - skipped, "steps": len(batches), "moments_per_pass": len(windows) * WIN}


class Prefetcher:
    """loads the clips of the next DEPTH batches in threads while the current step runs (one .pt read per clip); records the wait."""
    DEPTH, THREADS = 4, 4

    def __init__(self, store: ClipStore, batches: list):
        self.store, self.batches = store, batches; self.pool = ThreadPoolExecutor(self.THREADS); self.i = 0; self.q = []
        self.last_wait_s = 0.0
        while len(self.q) < self.DEPTH and self.i + len(self.q) < len(self.batches):
            self._submit(self.i + len(self.q))

    def _submit(self, j):
        self.q.append([self.pool.submit(self.store.load, c) for c, _ in self.batches[j]])

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.batches):
            raise StopIteration
        import time
        t0 = time.time(); futs = self.q.pop(0); recs = [f.result() for f in futs]; self.last_wait_s = time.time() - t0
        b = self.batches[self.i]; self.i += 1
        if self.i + len(self.q) < len(self.batches):
            self._submit(self.i + len(self.q))
        return [(r, s) for r, (_, s) in zip(recs, b)]

    def close(self):
        self.pool.shutdown(wait=False)


def window_tensors(items: list, device="cuda") -> dict:
    """[(clip record, start_row)] x B -> tensors (B, 16, ...) on the device plus per-moment lists (sample_id, metric cache paths)."""
    B = len(items)
    stack = lambda k: torch.stack([r[k][s:s + WIN] for r, s in items]).to(device, non_blocking=True)
    out = {"tokens": stack("tokens"), "ego": stack("ego"), "traj": stack("traj"), "has_label2": torch.stack([r["has_label2"][s:s + WIN] for r, s in items]),
           "lidar2img": torch.stack([r["lidar2img"] for r, _ in items]).to(device), "img_shape": torch.stack([r["img_shape"] for r, _ in items]).to(device),
           "sample_id": [[r["sample_id"][s + t] for t in range(WIN)] for r, s in items],
           "metric_cache_path": [[r["metric_cache_path"][s + t] for t in range(WIN)] for r, s in items],
           "scores_index": [[r["scores_index"][s + t] for t in range(WIN)] for r, s in items],
           "clip": [r["clip"] for r, _ in items], "start": [s for _, s in items]}
    for r, s in items:
        at = r["anchor_t"][s:s + WIN]; assert np.allclose(np.diff(at), STEP_S), (r["clip"], s)
    return out
