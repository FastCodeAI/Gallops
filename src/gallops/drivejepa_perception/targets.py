"""The side-car's targets and the camera geometry they rest on (numpy only, so tests run anywhere).

**The geometry.** The vendor's `lidar2img` maps ego points onto a 768x448 padded frame (scale 0.4 of
1920x1080). The tokens come from rows 28 to 1052 of the 1080, resized to 512x256. `TO_512` maps one
onto the other exactly: x_512 = (0.2667 / 0.4) x_pad = 2/3 x_pad, and y_512 = 0.25 (y - 28) =
0.625 y_pad - 7 (H04's matrix; docs/drivejepa-perception-sidecar.md, S0.4).

**Targets** (`build`): agents of the three classes the side-car names, within RANGE_M, whose centre
projects inside the 512x256 image in front of the camera (H04's "seen" rule), nearest first, at most
QUERIES. An agent the camera cannot see is not a target: neither a positive nor an "empty" label.
Paths are offsets from the agent's own centre, kept only where the label's track lasted.
"""

from __future__ import annotations

import numpy as np

#: vehicle, pedestrian, cyclist (navsim_data.CLASSES[:3]); index 3 is "no agent".
NUM_CLASSES = 3
NONE = NUM_CLASSES
QUERIES = 64
RANGE_M = 50.0
PATH_STEPS = 2
#: The vendor's padded-frame projection -> the 512x256 token image (see the module docstring).
TO_512 = np.array([[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
H_TOK, W_TOK = 16, 32
#: The release's lidar2img slot for CAM_F0 (drive_jepa_model.py keeps [:, 1:2]).
FRONT_SLOT = 1


def front_projection(lidar2img: np.ndarray) -> np.ndarray:
    """(..., 4, 4, 4) cached lidar2img -> (..., 4, 4) ego -> 512x256 token-image projection."""
    return TO_512 @ np.asarray(lidar2img, dtype=np.float64)[..., FRONT_SLOT, :, :]


def seen(projection: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """(N,) bool: the point projects inside the 512x256 image, in front of the camera (H04's rule)."""
    h = np.c_[xyz, np.ones(len(xyz))] @ np.asarray(projection, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = h[:, 0] / h[:, 2], h[:, 1] / h[:, 2]
    return (h[:, 2] > 0) & (u >= 0) & (u < 512) & (v >= 0) & (v < 256)


def build(agents: dict, projection: np.ndarray) -> dict:
    """One sample's side-car targets, padded to QUERIES, from `navsim_data.agents_of`."""
    box = agents["box"].astype(np.float64)
    keep = (agents["cls"] < NUM_CLASSES) & (np.hypot(box[:, 0], box[:, 1]) <= RANGE_M)
    keep &= seen(projection, box[:, :3]) if len(box) else keep
    index = np.flatnonzero(keep)
    index = index[np.argsort(np.hypot(box[index, 0], box[index, 1]), kind="stable")][:QUERIES]
    n = len(index)
    out = {
        "cls": np.zeros(QUERIES, np.int64),
        "box": np.zeros((QUERIES, 7), np.float32),
        "vel": np.zeros((QUERIES, 2), np.float32),
        "path": np.zeros((QUERIES, PATH_STEPS, 2), np.float32),
        "path_ok": np.zeros((QUERIES, PATH_STEPS), bool),
        "mask": np.zeros(QUERIES, bool),
    }
    out["cls"][:n] = agents["cls"][index]
    out["box"][:n] = agents["box"][index]
    out["vel"][:n] = agents["velocity"][index]
    ok = agents["path_ok"][index]
    offsets = agents["path"][index] - agents["box"][index][:, None, :2]
    out["path"][:n] = np.where(ok[..., None], offsets, 0.0)
    out["path_ok"][:n] = ok
    out["mask"][:n] = True
    return out
