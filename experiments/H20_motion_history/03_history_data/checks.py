"""Checks 3a-3e on the side file. taskset -c 0-1 nice -n 19 <alpasim-venv>/bin/python -I checks.py  (MH_OUT as for build_side_file)

3a/3b use the same 41 four-row slim samples as READONLY_CARSTATE_HISTORY.md 2a: default_rng(0), 40 clip folders chosen from the sorted
slim root, each folder's first and middle file; the four-row ones are kept. Stored rows 0, 1, 2 are t-1.5, t-1.0, t-0.5 (s1_convert.py:121-125,
oldest first); history rows 5, 6, 7 are the same times.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("MH_OUT", "{nas_store}/memory/4_motion_history"))
SLIM = Path("{dataset_store}/s4_train/slim")
HERE = Path(__file__).resolve().parent.parent
RD = Path("{paddock_sid}/Alpasim/analysis_3.0/Retrain-data")
COLS = ("x", "y", "heading", "speed", "turn_rate", "valid")


def q(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(n=int(a.size), min=float(a.min()), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()))


def main():
    res = {}
    # ---- completeness
    m = pd.read_parquet(RD / "s3_assembly/manifest.parquet", columns=["sample_id", "clip_uuid", "anchor_t"])
    files = sorted((OUT / "ego_history").glob("*.npz"))
    H, SID, AT, CL = [], [], [], []
    for f in files:
        with np.load(f) as z:
            H.append(z["history"]); SID += list(z["sample_id"]); AT.append(z["anchor_t"]); CL += [f.stem] * len(z["sample_id"])
    H = np.concatenate(H); AT = np.concatenate(AT); SID = np.array(SID)
    res["completeness"] = dict(files=len(files), clips_in_manifest=int(m.clip_uuid.nunique()), moments=int(len(SID)),
                               manifest_moments=len(m), same_sample_ids=bool(set(SID) == set(m.sample_id)),
                               dtype=str(H.dtype), shape=list(H.shape), all_finite=bool(np.isfinite(H).all()))

    # ---- 3a / 3b
    sub = sorted(os.listdir(SLIM)); rng = np.random.default_rng(0)
    picked = []
    for d in rng.choice(sub, 40, replace=False):
        fs = sorted(os.listdir(SLIM / d))
        for f in [fs[0], fs[len(fs) // 2]]:
            with np.load(SLIM / d / f) as z:
                e = z["ego_status"]
            if e.shape[0] == 4:
                picked.append((d, f[:-4], e))
    idx = {s: i for i, s in enumerate(SID)}
    # 11 of the 41 are validation clips (slim holds val/val_full too): history built on the fly, same builder, same egomotion zips.
    import io, zipfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ego_history as EH
    vchunk = pd.concat([pd.read_csv(RD / f"s3_assembly/{v}/clips.csv")[["uuid", "chunk"]] for v in ("val", "val_full")]).drop_duplicates("uuid").set_index("uuid").chunk
    RAW0 = Path("{dataset_store}/s0_census/raw")

    def hist_for(d, sid):
        if sid in idx:
            return H[idx[sid]], "train side file"
        z = zipfile.ZipFile(RAW0 / f"labels/egomotion.offline/egomotion.offline.chunk_{int(vchunk[d]):04d}.zip")
        ts, T = EH.poses_from_egomotion(pd.read_parquet(io.BytesIO(z.read(f"{d}.egomotion.offline.parquet"))))
        t = int(sid.split("_")[1]) * 1000
        return EH.build_history(ts, T, t, T_ref=EH.converter_reference(ts, T, t)), "val, built on the fly"

    dpos, dhead, dvx, dsp = [], [], [], []
    per = []
    src_count = {}
    for d, sid, e in picked:
        h, src = hist_for(str(d), sid)
        src_count[src] = src_count.get(src, 0) + 1
        for srow, hrow in ((0, 5), (1, 6), (2, 7)):
            dpos += [abs(float(e[srow, 0]) - float(h[hrow, 0])), abs(float(e[srow, 1]) - float(h[hrow, 1]))]
            dhead.append(abs(float(e[srow, 2]) - float(h[hrow, 2])))
        dvx.append(abs(float(e[2, 3]) - float(h[7, 3])))
        dsp.append(abs(float(np.hypot(e[2, 3], e[2, 4])) - float(h[7, 3])))
        per.append(dict(sample_id=sid, stored_vx=float(e[2, 3]), hist_speed=float(h[7, 3]), stored_ax=float(e[2, 5])))
    res["3a"] = dict(samples=len(picked), sources=src_count, rows_compared=3 * len(picked), xy_abs_diff=q(dpos), heading_abs_diff=q(dhead))
    res["3b"] = dict(samples=len(picked), abs_diff_vs_stored_vx=q(dvx), abs_diff_vs_stored_planar_speed=q(dsp),
                     worst=sorted(per, key=lambda r: -abs(r["stored_vx"] - r["hist_speed"]))[:5])

    # ---- diag
    D = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "diag").glob("*.parquet"))], ignore_index=True)
    nv = H[:, :, 5].sum(1).astype(int)
    res["3c"] = dict(counts={int(k): int((nv == k).sum()) for k in range(8, -1, -1)}, fewer_than_8=int((nv < 8).sum()),
                     fewer_than_8_share=float((nv < 8).mean()),
                     anchor_lt_4p0=int((AT < 4.0).sum()), anchor_lt_4p5=int((AT < 4.5).sum()),
                     first_pose_us=q(D.first_pose_us), diag_agrees=bool((D.set_index("sample_id").n_valid.reindex(SID).to_numpy() == nv).all()))
    gap_m = D.max_pose_step_us_in_window > 150_000
    res["3d"] = dict(moments_with_gap_over_0p15s_in_window=int(gap_m.sum()), rows_interpolated_across_gap=int(D.gap_rows.sum()),
                     moments_with_such_rows=int((D.gap_rows > 0).sum()), valid_rows_held_past_last_pose=int(D.held_rows.sum()),
                     largest_steps_us=sorted(D.max_pose_step_us_in_window.nlargest(5).astype(int).tolist(), reverse=True),
                     clips_with_gap=int(D[gap_m].sample_id.str[:8].nunique()))

    # ---- 3e
    V = H[H[:, :, 5] == 1.0]
    res["3e"] = {c: dict(min=float(V[:, i].min()), median=float(np.median(V[:, i])), max=float(V[:, i].max())) for i, c in enumerate(COLS[:5])}
    res["3e"]["valid_rows"] = int(len(V))
    fast = np.unique(SID[(H[:, :, 3] > 40.0).any(1)]); turn = np.unique(SID[(np.abs(H[:, :, 4]) > 2.0).any(1)])
    res["3e"]["speed_gt_40"] = dict(moments=int(len(fast)), sample_ids=fast[:30].tolist())
    res["3e"]["abs_turn_rate_gt_2"] = dict(moments=int(len(turn)), sample_ids=turn[:30].tolist())
    hi = np.argsort(-np.abs(H[:, :, 4]).max(1))[:5]
    res["3e"]["top_turn_rate"] = [dict(sample_id=str(SID[i]), max_abs=float(np.abs(H[i, :, 4]).max()), max_speed=float(H[i, :, 3].max())) for i in hi]

    # ---- frame note: converter frame vs a drive-style frame (newest pose <= t)
    res["ref_frame_offset"] = dict(m=q(D.ref_offset_m), yaw_rad=q(np.abs(D.ref_offset_yaw)))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
