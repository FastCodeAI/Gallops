"""Car-state history for the 2,000 picking-set moments (the 0.1.8b checks' set; val_full clips, so not in the training side file).

Same builder and frame as the training side file: build_side_file.do_clip (ego_history.build_history with the converter's reference
frame), same egomotion zips (RAW0, chunk from the picking set's `chunk` column). anchor_t from approach 3's picking-cache index (the
sample id is re-derived from it and asserted, as build_side_file does). Output beside the training side file, one npz per clip:
  <STAGE>/ego_history_picking/<clip>.npz  (sample_id, anchor_t, history (N, 8, 6) fp32), only the picking moments.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_side_file as B  # noqa: E402
import a4_config as CFG  # noqa: E402


def main():
    import pai_train as T
    _, pk, _ = T.manifests()
    idx = pd.read_parquet("{experiment_store_h17}/cache/picking/index_picking.parquet").set_index("sample_id")
    pk = pk.assign(anchor_t=pk.sample_id.map(idx.anchor_t))
    assert pk.anchor_t.notna().all()
    out = CFG.PICK_HISTORY; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for ch, sub in pk.groupby("chunk"):
        z = zipfile.ZipFile(B.RAW0 / f"labels/egomotion.offline/egomotion.offline.{ch}.zip")
        for u, rows in sub.groupby("clip_uuid"):
            rows = rows.sort_values("anchor_t")
            df = pd.read_parquet(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet")))
            hist, _ = B.do_clip(u, rows, df)
            tmp = out / f".{u}.tmp.npz"
            np.savez(tmp, sample_id=np.array(list(rows.sample_id)), anchor_t=rows.anchor_t.to_numpy(np.float64), history=hist)
            os.replace(tmp, out / f"{u}.npz"); n += len(rows)
    print(f"picking history: {n} moments, {pk.clip_uuid.nunique()} clips -> {out}")
    assert n == 2000


if __name__ == "__main__":
    sys.exit(main())
