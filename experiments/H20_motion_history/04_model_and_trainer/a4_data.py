"""Approach 4 loader (build 2, step 2): 0.1.8b's sample list and sampler with the frozen encoder's cached tokens and the history side file.

Sample list and sampler = 0.1.8b's, imported unchanged: pai_train.manifests() (sha-checked manifest, 126,981 moments) and
pai_data.sampler_017 / batches_017 (pai_data.py:82-99: every moment once, type A or label-2 moments twice, permuted by torch.Generator(seed +
pass); batches of 32, a tail shorter than 4 merged into the previous batch). Random single moments across clips.

Per moment (what planner_dataset gave 0.1.8b, minus the images, plus the cached tokens and the history):
  tokens (512, 256) fp32, pooled (33, 256) fp32    local copy (a4_config.TOKENS, copy_tokens.py), np.load(mmap_mode="r"), one row each
  ego_status (1, 11) fp32                          the cache file's newest row, copied into TOKENS/index.parquet; slots 5:7 zeroed here
                                                   exactly as dataset_s35._finish does (zero_accel=True, pai_data.planner_dataset)
  lidar2img (2, 4, 4), img_shape (2, 3) fp32       the slim npz of the clip's first moment (approach 3's ClipStore.camera; constant within
                                                   a clip, A3/out/s6_lidar2img_constancy.json), tabulated once in TOKENS/camera.npz
  trajectory (8, 3) float64                        the cache file's human future, float64 as stored and as dataset_s35 serves it
                                                   (data_check.py: dtype and values equal on 100; a first version cast to fp32 and failed)
  has_label2, metric_cache_path, token             as dataset_s35._finish: path = CACHE/<clip>/<token>/metric_cache.pkl when has_label2
  ego_history (8, 6) fp32                          the side file, a4_config.SIDE/<clip>.npz
Any moment missing from the token index, the side file or the camera table raises KeyError (no fallback).
Batches are built in a background thread (DEPTH batches ahead); the collate matches torch's default_collate for these keys.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import a4_config as CFG


def load_side(root: Path, sample_ids=None) -> dict:
    """{sample_id: (8, 6) fp32} from every npz under root."""
    out = {}
    for f in sorted(root.glob("*.npz")):
        with np.load(f) as z:
            for s, h in zip(z["sample_id"], z["history"]):
                out[str(s)] = h
    if sample_ids is not None:
        miss = [s for s in sample_ids if s not in out]
        if miss:
            raise KeyError(f"{len(miss)} moments have no history in {root}, e.g. {miss[:3]}")
    return out


def build_camera(clip_first_sid: dict, out: Path, threads: int = 8) -> dict:
    """lidar2img / img_shape of each clip's first moment from the slim npz (read only), saved as one npz table."""
    import pai_common as C

    def one(item):
        clip, sid = item
        with np.load(C.SLIM / clip / f"{sid}.npz") as z:
            return clip, z["lidar2img"].astype(np.float32), z["img_shape"].astype(np.float32)
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, sorted(clip_first_sid.items())))
    np.savez(out, clip=np.array([r[0] for r in res]), lidar2img=np.stack([r[1] for r in res]), img_shape=np.stack([r[2] for r in res]))
    return load_camera(out)


def load_camera(path: Path) -> dict:
    with np.load(path) as z:
        return {str(c): (torch.from_numpy(l.copy()), torch.from_numpy(s.copy())) for c, l, s in zip(z["clip"], z["lidar2img"], z["img_shape"])}


class TokenSet:
    """moments of one split: local token arrays (training) or a dict of per-moment arrays (picking checks)."""

    def __init__(self, man: pd.DataFrame, history: dict, camera: dict, tokens=None, pooled=None, index: pd.DataFrame | None = None):
        import pai_common as C
        self.man = man.reset_index(drop=True)
        ix = index.set_index("sample_id")
        miss = [s for s in self.man.sample_id if s not in ix.index]
        if miss:
            raise KeyError(f"{len(miss)} moments missing from the token index, e.g. {miss[:3]}")
        ix = ix.loc[list(self.man.sample_id)]
        self.row = ix["row"].to_numpy()
        self.clip = ix["clip"].to_numpy()
        self.ego = np.stack(ix["ego_status"].to_numpy()).astype(np.float32)
        self.traj = np.stack(ix["trajectory"].to_numpy()).reshape(-1, 8, 3)
        self.lab = ix["has_label2"].to_numpy().astype(bool)
        assert (self.lab == self.man.has_label2.to_numpy().astype(bool)).all(), "has_label2 differs between manifest and token index"
        self.tokens, self.pooled = tokens, pooled
        self.sid = list(self.man.sample_id)
        self.hist = np.stack([history[s] for s in self.sid]).astype(np.float32)          # KeyError = missing history
        for c in set(self.clip):
            camera[c]                                                                     # KeyError = missing camera row
        self.camera = camera
        self.cache_root = C.CACHE

    def __len__(self):
        return len(self.sid)

    def item(self, i: int) -> tuple[dict, dict]:
        r = int(self.row[i]); clip = str(self.clip[i]); sid = self.sid[i]
        ego = torch.from_numpy(self.ego[i][None].copy()); ego[..., 5:7] = 0.0                   # dataset_s35._finish zero_accel
        l2i, shp = self.camera[clip]
        f = {"tokens": torch.from_numpy(np.array(self.tokens[r])), "pooled": torch.from_numpy(np.array(self.pooled[r])),
             "ego_status": ego, "lidar2img": l2i.clone(), "img_shape": shp.clone(), "ego_history": torch.from_numpy(self.hist[i].copy())}
        t = {"trajectory": torch.from_numpy(self.traj[i].astype(np.float64)), "token": sid, "has_label2": torch.tensor(bool(self.lab[i])),
             "metric_cache_path": str(self.cache_root / clip / sid / "metric_cache.pkl") if self.lab[i] else ""}
        return f, t

    def batch(self, idx: list[int]):
        from torch.utils.data.dataloader import default_collate
        return default_collate([self.item(i) for i in idx])


def training_set(tr: pd.DataFrame) -> TokenSet:
    idx = pd.read_parquet(CFG.TOKENS / "index.parquet")
    tok = np.load(CFG.TOKENS / "tokens.npy", mmap_mode="r"); poo = np.load(CFG.TOKENS / "pooled.npy", mmap_mode="r")
    assert tok.shape == (len(idx), 512, 256) and poo.shape == (len(idx), 33, 256) and tok.dtype == np.float32
    cam_path = CFG.TOKENS / "camera.npz"
    if not cam_path.exists():
        first = {c: g.sort_values("row").sample_id.iloc[0] for c, g in idx.groupby("clip")}
        build_camera(first, cam_path)
    return TokenSet(tr, load_side(CFG.SIDE, list(tr.sample_id)), load_camera(cam_path), tok, poo, idx)


def picking_set(pk: pd.DataFrame) -> TokenSet:
    """the 2,000 picking moments: approach 3's picking cache files (84 clips, read whole once), the picking history, the camera of each
    clip's first cached moment (same rule as training)."""
    pidx = pd.read_parquet(CFG.PICK_CACHE / "index_picking.parquet")
    pidx = pidx[pidx.sample_id.isin(set(pk.sample_id))]
    toks, pools, rows = [], [], []
    for clip, g in pidx.groupby("clip"):
        rec = torch.load(CFG.PICK_CACHE / g.file.iloc[0], map_location="cpu", weights_only=False)
        pos = {s: j for j, s in enumerate(rec["sample_id"])}
        for s in g.sample_id:
            j = pos[s]
            rows.append({"sample_id": s, "clip": clip, "row": len(rows), "has_label2": bool(rec["has_label2"][j]),
                         "ego_status": rec["ego_status"][j].numpy(), "trajectory": rec["trajectory"][j].numpy().reshape(24)})
            toks.append(rec["tokens"][j].numpy()); pools.append(rec["pooled"][j].numpy())
    index = pd.DataFrame(rows)
    full = pd.read_parquet(CFG.PICK_CACHE / "index_picking.parquet")
    first = {c: g.sort_values("row").sample_id.iloc[0] for c, g in full.groupby("clip") if c in set(index["clip"])}
    cam_path = CFG.TOKENS / "camera_picking.npz"
    cam = load_camera(cam_path) if cam_path.exists() else build_camera(first, cam_path)
    return TokenSet(pk, load_side(CFG.PICK_HISTORY, list(pk.sample_id)), cam, np.stack(toks), np.stack(pools), index)


class Prefetch:
    """batches built in one background thread, DEPTH ahead (the GPU step and the next batch's reads overlap); records the wait."""
    DEPTH = 4

    def __init__(self, ds: TokenSet, batches: list):
        self.ds, self.batches = ds, batches
        self.q: queue.Queue = queue.Queue(self.DEPTH)
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
        self.last_wait_s = 0.0

    def _run(self):
        for b in self.batches:
            try:
                self.q.put(("ok", self.ds.batch(b)))
            except Exception as e:                                       # surfaced in the consumer, never swallowed
                self.q.put(("err", e)); return
        self.q.put(("end", None))

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.time(); kind, v = self.q.get(); self.last_wait_s = time.time() - t0
        if kind == "err":
            raise v
        if kind == "end":
            raise StopIteration
        return v
