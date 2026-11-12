"""Check D: does every candidate training moment have a full 4 s human path on the
generator's 0.5 s grid, in the generator's frame?  Read-only, CPU only.

    ./run.sh src/d_horizon.py

Scenes: s1 splits.json train (482) + selection (50).  Val / overlap / test are not read.
Poses and timestamps are read with the evaluation wrapper's own functions
(kitscenes_adapter.load_poses, to_relative_poses, constants), the same calls
KITScenesSceneAdapter.__init__ / build_sample make.

Moment rule tested (every 5th frame): t % 5 == 0, t >= 15 (build_sample indexes the
history t-15, t-10, t-5, t; a smaller t would wrap to the end of the array), and
t + 40 <= N - 1 (8 future waypoints at 5-frame steps, which is also the horizon the
'gt' command reads, kitscenes_adapter.py _driving_command).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import common as C
import splits as S

ADP = C.kit_adapter_module()
OUT = Path(__file__).resolve().parent.parent / "out"
STEP = ADP.HISTORY_STRIDE  # 5 frames
N_FUT = 8
MIN_T = ADP.HISTORY_STRIDE * (ADP.NUM_HISTORY_FRAMES - 1)  # 15, as valid_keyframes' minimum
TOL_S = 0.02


def scene(path: str) -> dict:
    p = Path(path)
    ref = np.loadtxt(p / ADP.REFERENCE_TIMESTAMP_FILENAME)
    pts, poses = ADP.load_poses(p)
    n = len(ref)
    r = {"frames": n, "pose_rows": len(pts)}
    r["pose_ts_max_abs_diff_s"] = float(np.max(np.abs(pts - ref))) if len(pts) == n else None
    d = np.diff(ref)
    r["step_min_s"], r["step_max_s"] = float(d.min()), float(d.max())
    ts = list(range(MIN_T, n, STEP))
    ok, bad_time, fwd, yaw_err = [], 0, [], []
    for t in ts:
        if t + STEP * N_FUT > n - 1:
            continue
        fut = [t + STEP * (k + 1) for k in range(N_FUT)]
        dt = ref[fut] - ref[t]
        if np.max(np.abs(dt - 0.5 * np.arange(1, N_FUT + 1))) > TOL_S:
            bad_time += 1
            continue
        ok.append(t)
        rel = ADP.to_relative_poses(poses[t], poses[fut])
        if np.linalg.norm(rel[0, :2]) > 1.0:  # moving > 2 m/s over the first 0.5 s
            fwd.append(rel[0, 0] > 0)
            # heading of travel vs recorded yaw, in the moment's own frame
            yaw_err.append(abs(np.arctan2(rel[0, 1], rel[0, 0]) - rel[0, 2] / 2))
    r.update(moments=len(ok), dropped_time=bad_time, ts=ok,
             moving=len(fwd), moving_forward=int(np.sum(fwd)),
             yaw_err_med_deg=float(np.degrees(np.median(yaw_err))) if yaw_err else None)
    return r


def main() -> None:
    t0 = time.time()
    data = S.load_splits()
    out = {"rule": f"t%5==0, t>={MIN_T}, t+{STEP * N_FUT}<=N-1; future times within {TOL_S}s of k*0.5s"}
    per = {}
    for name in ("train", "selection"):
        ids = sorted(data[name])
        with ThreadPoolExecutor(16) as pool:
            res = dict(zip(ids, pool.map(lambda s: scene(data["scene_paths"][s]), ids)))
        per[name] = res
        v = list(res.values())
        out[name] = {
            "scenes": len(v),
            "scenes_with_pose_rows_ne_frames": sum(r["pose_rows"] != r["frames"] for r in v),
            "pose_ts_vs_frame_ts_max_abs_diff_s": max(r["pose_ts_max_abs_diff_s"] or 0 for r in v),
            "frame_step_min_max_s": [min(r["step_min_s"] for r in v), max(r["step_max_s"] for r in v)],
            "moments": sum(r["moments"] for r in v),
            "moments_dropped_bad_timing": sum(r["dropped_time"] for r in v),
            "scenes_with_zero_moments": sum(r["moments"] == 0 for r in v),
            "moving_moments": sum(r["moving"] for r in v),
            "moving_with_x_forward": sum(r["moving_forward"] for r in v),
            "median_of_scene_median_heading_error_deg": float(np.nanmedian(
                [r["yaw_err_med_deg"] for r in v if r["yaw_err_med_deg"] is not None])),
        }
    # Frame cache coverage: every image a moment needs (t-5, t) is a multiple of 5.
    idx = json.loads((Path("{runs}/drivejepa_kitscenes/encoder_lora_kit/s2/frame_cache") / "index.json").read_text())
    have = idx.get("index") or idx.get("frames") or {}
    nf = idx["scene_frames"]
    miss = 0
    for name in ("train", "selection"):
        for s, r in per[name].items():
            miss += int(nf.get(s) != r["frames"])
    out["frame_cache_scene_frame_count_mismatches"] = miss
    out["frame_cache_index_keys"] = sorted(idx.keys())
    out["wall_s"] = round(time.time() - t0, 1)
    (OUT / "d_horizon.json").write_text(json.dumps(out, indent=1) + "\n")
    (OUT / "d_moments.json").write_text(json.dumps(
        {n: {s: r["ts"] for s, r in per[n].items()} for n in per}) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
