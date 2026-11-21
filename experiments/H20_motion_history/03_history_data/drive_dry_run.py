"""Drive-side dry run (no simulation): the pose stream AlpaSim pushed to the 0.1.8b driver in a recorded rollout, fed to build_history.

  taskset -c 0-1 nice -n 19 <alpasim-venv>/bin/python -I drive_dry_run.py <rollout.asl> [more.asl ...]

Reads `driver_ego_trajectory` entries the way the scaffold accumulates them (adapters/scaffold.py:558-578: extend + sort by timestamp;
the replay loop of paddock-sid/dj014_accel_counterfactual.py:144-151 does the same). Pose quaternions are w-first on the wire
(AD/planner.py:192-202 `_pose_to_matrix`). Only timestamps + poses go to the builder; dynamic states are read only to count the
force-GT zero velocities. The reference frame is the driver's: the newest pose at or before t (T_ref=None).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ego_history as H  # noqa: E402
from alpasim_utils.logs import async_read_pb_log  # noqa: E402


def mat(pose) -> np.ndarray:
    q, v = pose.quat, pose.vec
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([float(q.x), float(q.y), float(q.z), float(q.w)]).as_matrix()   # proto is w-first
    M[:3, 3] = [float(v.x), float(v.y), float(v.z)]
    return M


async def read(asl):
    poses, dyn, msgs, asks = [], [], [], []
    async for e in async_read_pb_log(asl):
        k = e.WhichOneof("log_entry")
        if k == "driver_ego_trajectory":
            g = e.driver_ego_trajectory
            ps = list(g.trajectory.poses)
            msgs.append(len(ps))
            for p, d in zip(ps, g.dynamic_states):
                poses.append((int(p.timestamp_us), mat(p.pose)))
                dyn.append((int(p.timestamp_us), float(d.linear_velocity.x), float(d.linear_velocity.y)))
        elif k == "driver_request":
            asks.append(int(e.driver_request.time_now_us))
    poses.sort(key=lambda x: x[0]); dyn.sort(key=lambda x: x[0])
    return poses, dyn, msgs, asks


def one(asl: str) -> dict:
    poses, dyn, msgs, asks = asyncio.run(read(asl))
    ts = np.array([p[0] for p in poses], dtype=np.int64)
    T = np.stack([p[1] for p in poses])
    uniq = np.unique(ts)
    dt = np.diff(uniq) / 1e6
    v = np.array([np.hypot(d[1], d[2]) for d in dyn])
    t0 = int(ts[0])
    rel = [0.0, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0, 3.9, 4.0, 4.5, 6.0, 10.0]
    end = (int(ts[-1]) - t0) / 1e6
    rel += [round(end / 2, 3), round(end - 0.05, 3)]
    checks = []
    example = None
    for r in rel:
        t = t0 + int(round(r * 1e6))
        h = H.build_history(ts, T, t)
        Tp = T.copy(); Tp[ts > t] = np.nan
        hp = H.build_history(ts, Tp, t)
        exp_valid = [(t - (8 - i + 1) * H.STEP_US) >= t0 for i in range(8)]
        checks.append(dict(t_after_first_pose_s=r, shape=list(h.shape), dtype=str(h.dtype), finite=bool(np.isfinite(h).all()),
                           valid=[int(x) for x in h[:, 5]], valid_as_expected=[int(x) for x in exp_valid] == [int(x) for x in h[:, 5]],
                           invalid_rows_all_zero=bool(np.all(h[~h[:, 5].astype(bool)] == 0)),
                           identical_with_future_nan=bool(np.array_equal(h, hp))))
        if r == round(end / 2, 3):
            example = dict(t_after_first_pose_s=r, rows_oldest_first=[[round(float(x), 4) for x in row] for row in h])
    return dict(asl=asl, poses=len(ts), unique_timestamps=len(uniq), duplicate_timestamps=int(len(ts) - len(uniq)),
                egomotion_messages=len(msgs), poses_per_message=dict(zip(*[x.tolist() for x in np.unique(msgs, return_counts=True)])),
                spacing_s=dict(median=float(np.median(dt)), min=float(dt.min()), p99=float(np.percentile(dt, 99)), max=float(dt.max())),
                span_s=float((ts[-1] - ts[0]) / 1e6), driver_requests=len(asks),
                ask_spacing_median_s=float(np.median(np.diff(asks)) / 1e6) if len(asks) > 1 else None,
                reported_speed_exactly_zero=dict(
                    first_3p05_s=int(sum(1 for (tt, *_), x in zip(dyn, v) if tt - t0 <= 3_050_000 and x == 0.0)),
                    poses_in_first_3p05_s=int(sum(1 for d in dyn if d[0] - t0 <= 3_050_000)),
                    after_3p05_s=int(sum(1 for (tt, *_), x in zip(dyn, v) if tt - t0 > 3_050_000 and x == 0.0)),
                    last_zero_at_s=float(max(((tt - t0) / 1e6 for (tt, *_), x in zip(dyn, v) if x == 0.0), default=-1.0)),
                    first_pose_speed=float(v[0])), checks=checks, all_checks_ok=all(
                    c["shape"] == [8, 6] and c["dtype"] == "float32" and c["finite"] and c["valid_as_expected"] and c["invalid_rows_all_zero"]
                    and c["identical_with_future_nan"] for c in checks), example=example)


def main():
    out = [one(a) for a in sys.argv[1:]]
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
