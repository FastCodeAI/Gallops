"""Training side file: car-state history for every moment of the 0.1.8b training manifest.

  taskset -c 0-1 nice -n 19 <alpasim-venv>/bin/python -I build_side_file.py time    # one chunk end to end, prints the estimate
  taskset -c 0-1 nice -n 19 <alpasim-venv>/bin/python -I build_side_file.py full    # every clip; resumable (existing npz skipped)

Inputs (read only):
  manifest   RD/s3_assembly/manifest.parquet (sample_id, clip_uuid, anchor_t)       126,981 moments / 4,015 clips
  chunk map  RD/s3_assembly/clips.csv (uuid -> chunk)
  egomotion  {dataset_store}/s0_census/raw/labels/egomotion.offline/egomotion.offline.chunk_XXXX.zip
             = s1_common.RAW0 (s0_census/common.py:47), the zips the converter read (s1_convert.py:86; s3_convert.py keeps RAW0)
Outputs:
  OUT/ego_history/<clip_uuid>.npz   sample_id (N,) str, anchor_t (N,) float64 s, history (N, 8, 6) float32
  OUT/diag/<clip_uuid>.parquet      per moment: n_valid, gap rows, frame offset vs a drive-style reference (report only)
The anchor time is anchor_t * 1e6 us and must reproduce the sample id (s1_convert.py:162 `f"{uuid[:8]}_{t // 1000:05d}"`).
Reference frame = the converter's Ego.T(t) (ego_history.py, LEAK RULE).
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ego_history as H  # noqa: E402

RD = Path("{paddock_sid}/Alpasim/analysis_3.0/Retrain-data")
RAW0 = Path("{dataset_store}/s0_census/raw")
# The brief puts outputs here. On 2026-11-17 the NAS refused every write from this host (EROFS on all folders, reads fine), so the
# run was staged on the HDD via MH_OUT and must be copied to this path once the NAS accepts writes (BUILD1_HISTORY_DATA.md).
OUT = Path(os.environ.get("MH_OUT", "{nas_store}/memory/4_motion_history"))
LOGS = Path(__file__).resolve().parent.parent / "logs"
GAP_US = 150_000                                          # 1.5 x nominal 0.1 s, the recon's gap rule (APPROACH3_RECON.md:149)


def log(msg, f):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    f.write(line + "\n"); f.flush()


def plan():
    m = pd.read_parquet(RD / "s3_assembly/manifest.parquet", columns=["sample_id", "clip_uuid", "anchor_t"])
    clips = pd.read_csv(RD / "s3_assembly/clips.csv")[["uuid", "chunk"]].set_index("uuid").chunk
    assert set(m.clip_uuid) <= set(clips.index)
    m = m.assign(chunk=m.clip_uuid.map(clips).astype(int))
    return m


def do_clip(uuid, rows, df):
    ts, T = H.poses_from_egomotion(df)
    hist, diag = [], []
    for sid, at in zip(rows.sample_id, rows.anchor_t):
        t = int(round(float(at) * 1e6))
        assert sid == f"{uuid[:8]}_{t // 1000:05d}", (sid, t)
        ref = H.converter_reference(ts, T, t)
        h = H.build_history(ts, T, t, T_ref=ref)
        hist.append(h)
        # diagnostics (not in the side file)
        br = H.row_brackets(ts, t)
        gap_rows = sum(any(a >= 0 and b >= 0 and b - a > GAP_US for a, b in row) for row in br)
        held_rows = sum(any(b < 0 for a, b in row) for i, row in enumerate(br) if h[i, 5] == 1.0)
        win = ts[(ts >= t - 9 * H.STEP_US) & (ts <= t)]
        max_step = int(np.diff(win).max()) if len(win) > 1 else 0
        drive_ref = H._Poses(ts[ts <= t], T[ts <= t]).at(t)          # newest pose <= t, the driver's frame
        off = H._relative_pose(drive_ref, ref)
        diag.append(dict(sample_id=sid, anchor_t=float(at), n_valid=int(h[:, 5].sum()), gap_rows=int(gap_rows), held_rows=int(held_rows),
                         max_pose_step_us_in_window=max_step, ref_offset_m=float(np.hypot(off[0], off[1])), ref_offset_yaw=float(off[2]),
                         first_pose_us=int(ts[0])))
    return np.stack(hist).astype(np.float32), pd.DataFrame(diag)


def write_clip(uuid, rows, hist, diag):
    d1, d2 = OUT / "ego_history", OUT / "diag"
    d1.mkdir(parents=True, exist_ok=True); d2.mkdir(parents=True, exist_ok=True)
    tmp = d1 / f".{uuid}.{os.getpid()}.tmp.npz"
    np.savez(tmp, sample_id=np.array(list(rows.sample_id)), anchor_t=rows.anchor_t.to_numpy(np.float64), history=hist)
    with np.load(tmp) as z:                                           # round trip before the rename
        assert z["history"].shape == hist.shape and np.array_equal(z["history"], hist)
    os.replace(tmp, d1 / f"{uuid}.npz")
    diag.to_parquet(d2 / f"{uuid}.parquet", index=False)


def run(m, chunks, logf, every=100):
    done = n = 0; t0 = time.time()
    total = m.clip_uuid.nunique()
    for ch in chunks:
        sub = m[m.chunk == ch]
        todo = [u for u in dict.fromkeys(sub.clip_uuid) if not (OUT / "ego_history" / f"{u}.npz").exists()]
        done += sub.clip_uuid.nunique() - len(todo)
        if not todo:
            continue
        z = zipfile.ZipFile(RAW0 / f"labels/egomotion.offline/egomotion.offline.chunk_{ch:04d}.zip")
        for u in todo:
            rows = sub[sub.clip_uuid == u].sort_values("anchor_t")
            df = pd.read_parquet(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet")))
            hist, diag = do_clip(u, rows, df)
            write_clip(u, rows, hist, diag)
            done += 1; n += 1
            if done % every == 0:
                el = time.time() - t0
                log(f"progress {done}/{total} clips (this run {n} in {el:.0f} s, {el / max(n, 1):.3f} s/clip)", logf)
    return n, time.time() - t0


def main():
    mode = sys.argv[1]
    LOGS.mkdir(exist_ok=True)
    m = plan()
    with open(LOGS / f"side_file_{mode}.log", "a") as logf:
        log(f"start mode={mode}: {len(m)} moments, {m.clip_uuid.nunique()} clips, {m.chunk.nunique()} chunks", logf)
        if mode == "time":
            per = m.groupby("chunk").clip_uuid.nunique()
            ch = int(per[per == per.max()].index[0])                     # the largest chunk: a conservative per-clip time
            n, el = run(m, [ch], logf, every=10**9)
            per_clip = el / max(n, 1)
            est = per_clip * m.clip_uuid.nunique()
            log(f"TIMING chunk {ch}: {n} clips, {int((m.chunk == ch).sum())} moments in {el:.1f} s = {per_clip:.3f} s/clip; "
                f"estimate all {m.clip_uuid.nunique()} clips {est:.0f} s = {est / 3600:.2f} h", logf)
        elif mode == "full":
            n, el = run(m, sorted(m.chunk.unique()), logf)
            files = len(list((OUT / "ego_history").glob("*.npz")))
            log(f"FULL done: {n} clips written this run in {el:.0f} s ({el / 3600:.2f} h); files on disk {files}", logf)


if __name__ == "__main__":
    main()
