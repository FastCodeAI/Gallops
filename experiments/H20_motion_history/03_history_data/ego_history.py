"""Car-state history for Drive-JEPA approach 4: one builder for the training side file and the AlpaSim driver.

History = 8 past rows at t-4.0, t-3.5, ..., t-0.5 s, OLDEST FIRST. The current moment is not a row.
Row = [x, y, heading, speed, turn_rate, valid], float32.
  x, y, heading : the pose at the row's time, in the rig frame at t (`_relative_pose`, below).
  speed         : |xy(row) - xy(row - 0.5 s)| / 0.5, both in the rig frame at t (planar, z dropped as the ego_status pose slots do).
  turn_rate     : wrap(heading(row) - heading(row - 0.5 s)) to (-pi, pi], / 0.5.
  valid         : 1.0 if both times are at or after the first pose; otherwise 0.0 and the whole row is 0.0.
Poses only: never reported velocity, never stored vx / ax.

Interpolation is the converter's (RD/s1_converter/s1_convert.py:73-75, `Ego.T`): scipy `Slerp` on the rotations, `np.interp` per
position axis, time clipped to the pose range. Rotation slerp between two keyframes depends only on those two, so evaluating it over
the poses at or before t gives exactly what the converter's all-pose Slerp gives for any time whose bracket ends at or before t.

LEAK RULE. `build_history` reads only poses with timestamp <= t_us; it drops the rest before anything else (see `_kept`). The one
input that may legitimately come from elsewhere is the reference frame, `T_ref`:
  * training: the converter's frame for the moment, `Ego.T(t)` (s1_convert.py:109), which the current ego row and the labels are
    already in. Physical AI AV poses sit ~91 ms off the 0.5 s grid, so that frame interpolates with the first pose AFTER t (<= 0.1 s).
    Passing it keeps history in the labels' frame; it adds nothing the existing path does not already use. `converter_reference`
    computes it exactly as the converter does.
  * driving: None -> the newest pose at or before t, which is what the AlpaSim driver anchors on (AD/planner.py:822).

No import of drivejepa.planner or the converter: both pull grpc / alpasim / the model harness. `_yaw_of` and `_relative_pose` are
copied verbatim from adapters/drivejepa014/planner.py:219-227 (identical in drivejepa018bp06); test_ego_history.py checks them
against the real functions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

N_ROWS = 8
STEP_US = 500_000
ROW_WIDTH = 6
COLUMNS = ("x", "y", "heading", "speed", "turn_rate", "valid")


# ---- verbatim copies of adapters/drivejepa014/planner.py:219-227 ------------------------------------------------------------------
def _yaw_of(matrix: np.ndarray) -> float:
    """Yaw (rotation about z) of a 4x4 or 3x3 transform, radians."""
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(dx, dy, dyaw)`` of ``pose`` expressed in ``reference``'s frame (§5.1)."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], _yaw_of(delta)], dtype=np.float64)


# ---- interpolation (s1_convert.py:73-75) ------------------------------------------------------------------------------------------
class _Poses:
    def __init__(self, ts_us: np.ndarray, T: np.ndarray):
        self.ts = np.asarray(ts_us, dtype=np.int64)
        self.T = np.asarray(T, dtype=np.float64)
        self.p = self.T[:, :3, 3]
        R = Rotation.from_matrix(self.T[:, :3, :3])
        self.slerp = Slerp(self.ts.astype(np.float64), R) if len(self.ts) >= 2 else None
        self.R0 = R

    def at(self, t_us: float) -> np.ndarray:
        t = float(np.clip(t_us, self.ts[0], self.ts[-1]))
        M = np.eye(4)
        M[:3, :3] = self.slerp([t]).as_matrix()[0] if self.slerp is not None else self.R0.as_matrix()[0]
        M[:3, 3] = [np.interp(t, self.ts, self.p[:, i]) for i in range(3)]
        return M


def converter_reference(ts_us, T_list, t_us: int) -> np.ndarray:
    """The converter's frame at t, Ego.T(t) over ALL poses (s1_convert.py:73-75, 109). Training side only (see LEAK RULE)."""
    return _Poses(ts_us, T_list).at(t_us)


def _kept(ts_us, T_list, t_us: int):
    ts = np.asarray(ts_us, dtype=np.int64)
    T = np.asarray(T_list, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    ts, T = ts[order], T[order]
    keep = ts <= int(t_us)
    return ts[keep], T[keep]


def _wrap(a: float) -> float:
    """to (-pi, pi]."""
    w = (a + np.pi) % (2 * np.pi) - np.pi
    return float(np.pi) if w == -np.pi else float(w)


def build_history(ts_us, T_list, t_us: int, T_ref: np.ndarray | None = None) -> np.ndarray:
    """(8, 6) float32 history at t_us, oldest row first. Poses: timestamps (us) + 4x4 poses in any one fixed frame."""
    ts, T = _kept(ts_us, T_list, t_us)
    out = np.zeros((N_ROWS, ROW_WIDTH), dtype=np.float32)
    if len(ts) == 0:
        return out
    P = _Poses(ts, T)
    ref = P.at(t_us) if T_ref is None else np.asarray(T_ref, dtype=np.float64)
    for i in range(N_ROWS):
        k = N_ROWS - i                                   # i = 0 -> t - 4.0 s ... i = 7 -> t - 0.5 s
        tj, tp = int(t_us) - k * STEP_US, int(t_us) - (k + 1) * STEP_US
        if tp < ts[0]:                                   # a needed pose time precedes the first pose: row invalid, all zero
            continue
        a, b = _relative_pose(P.at(tj), ref), _relative_pose(P.at(tp), ref)
        out[i] = (a[0], a[1], a[2], np.hypot(a[0] - b[0], a[1] - b[1]) / 0.5, _wrap(a[2] - b[2]) / 0.5, 1.0)
    return out


def row_brackets(ts_us, t_us: int) -> list[list[tuple[int, int]]]:
    """For diagnostics: per row, the (before, after) pose timestamps bracketing its two times, from poses <= t only."""
    ts = np.sort(np.asarray(ts_us, dtype=np.int64))
    ts = ts[ts <= int(t_us)]
    res = []
    for i in range(N_ROWS):
        k = N_ROWS - i
        row = []
        for tau in (int(t_us) - k * STEP_US, int(t_us) - (k + 1) * STEP_US):
            j = int(np.searchsorted(ts, tau, side="right")) - 1
            row.append((int(ts[j]) if j >= 0 else -1, int(ts[j + 1]) if j + 1 < len(ts) else -1))
        res.append(row)
    return res


def poses_from_egomotion(df) -> tuple[np.ndarray, np.ndarray]:
    """Physical AI AV egomotion.offline parquet -> (ts_us, (N, 4, 4)), as s1_convert.Ego reads it (:68-69): sorted, xyzw quats."""
    df = df.sort_values("timestamp")
    ts = df.timestamp.values.astype(np.int64)
    T = np.tile(np.eye(4), (len(ts), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values).as_matrix()
    T[:, :3, 3] = df[["x", "y", "z"]].values.astype(np.float64)
    return ts, T
