#!/usr/bin/env python
"""S10 task 5: X99 = 99th percentile over the training manifest (S4 manifest minus the 145 SEEN held-out clips) of the human path's
forward distance (x, car frame) at 4 s = trajectory[7, 0] of each example's slim copy. Writes ref_range.json. CPU only."""
import sys, json, pathlib, hashlib, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent; sys.path.insert(0, str(RD / "05_small_overfit_check" / "train_patch")); import s35_common as C
SLIM = C.NAS / "06_first_training_run" / "slim"; man = pd.read_parquet(C.S3 / "manifest.parquet"); ho = set(pd.read_csv(HERE / "seen_heldout_clips.csv").clip_uuid); assert len(ho) == 145
def one(r):
    with np.load(SLIM / r[1] / f"{r[0]}.npz") as z: t = z["trajectory"]
    assert t.shape == (8, 3); return t[7, 0], t[7, 1]
with ThreadPoolExecutor(8) as ex: xy = np.array(list(ex.map(one, zip(man.sample_id, man.clip_uuid))), dtype=np.float64)
man["x4"] = xy[:, 0]; man["y4"] = xy[:, 1]; tr = man[~man.clip_uuid.isin(ho)]; assert len(tr) == 122581, len(tr)
q = lambda v, p: float(np.percentile(v, p)); hw20 = tr[(tr.type == "A") & (tr.speed >= 20)]
out = dict(definition="x_hi = 99th percentile (numpy linear interpolation) of trajectory[7, 0] (human path, forward metres at 4 s) over the training examples; the forward reference axis maps -32 m .. x_hi onto 0 .. 1, lateral axis unchanged (-32 .. 32 m)",
           forward_axis="last-axis index 0 of ref_2d / reference_points (x); see train_patch/ref_range.py and AGENT/bevformer/encoder.py:235", examples=int(len(tr)), clips=int(tr.clip_uuid.nunique()),
           x_hi=q(tr.x4, 99), X99=q(tr.x4, 99), scale_on_upstream_value=64.0 / (q(tr.x4, 99) + 32.0),
           forward_4s_percentiles_m={str(p): q(tr.x4, p) for p in (0, 1, 50, 90, 95, 99, 99.9, 100)}, share_above_32m=float((tr.x4 > 32).mean()), share_above_x_hi=float((tr.x4 > q(tr.x4, 99)).mean()),
           highway_ge20=dict(examples=int(len(hw20)), forward_4s_percentiles_m={str(p): q(hw20.x4, p) for p in (1, 50, 99, 100)}, share_above_x_hi=float((hw20.x4 > q(tr.x4, 99)).mean())),
           lateral_4s_abs_percentiles_m={str(p): q(tr.y4.abs(), p) for p in (50, 99, 100)}, full_s4_manifest=dict(examples=int(len(man)), X99=q(man.x4, 99)),
           manifest_sha256=hashlib.sha256(open(C.S3 / "manifest.parquet", "rb").read()).hexdigest())
json.dump(out, open(HERE / "ref_range.json", "w"), indent=1); print(json.dumps(out, indent=1))
