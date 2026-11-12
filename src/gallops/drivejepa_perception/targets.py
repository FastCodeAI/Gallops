"""The side-car's targets and the camera geometry they rest on (numpy only, so tests run anywhere).

**The geometry.** The vendor's `lidar2img` maps ego points onto a 768x448 padded frame (scale 0.4 of
1920x1080). The tokens come from rows 28 to 1052 of the 1080, resized to 512x256. `TO_512` maps one
onto the other exactly: x_512 = (0.2667 / 0.4) x_pad = 2/3 x_pad, and y_512 = 0.25 (y - 28) =
0.625 y_pad - 7 (H04's matrix; docs/drivejepa-perception-sidecar/first-pass-plan.md, S0.4).

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
#: The vendor's lidar2img scales a camera's pixels by this (tokens.SCALE), and the camera feature
#: drops this many rows top and bottom before its resize to 512x256 (drive_jepa_features.py).
VENDOR_SCALE, CROP_ROWS = 0.4, 28


def to_512(width: int, height: int) -> np.ndarray:
    """TO_512 for a `width` x `height` camera: x by 512 / (0.4 width), y by 256 / (0.4 (height - 56))
    after the 28-row crop. NAVSIM's 1920x1080 gives TO_512; nuScenes' 1600x900 does not."""
    rows = height - 2 * CROP_ROWS
    out = np.eye(4)
    out[0, 0] = 512 / (VENDOR_SCALE * width)
    out[1, 1] = 256 / (VENDOR_SCALE * rows)
    out[1, 2] = -CROP_ROWS * 256 / rows
    return out


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


def target_index(agents: dict, projection: np.ndarray) -> np.ndarray:
    """The rows of `agents` that are targets, in the order `build` lays them out (nearest first)."""
    box = agents["box"].astype(np.float64)
    keep = (agents["cls"] < NUM_CLASSES) & (np.hypot(box[:, 0], box[:, 1]) <= RANGE_M)
    keep &= seen(projection, box[:, :3]) if len(box) else keep
    index = np.flatnonzero(keep)
    return index[np.argsort(np.hypot(box[index, 0], box[index, 1]), kind="stable")][:QUERIES]


def build(agents: dict, projection: np.ndarray) -> dict:
    """One sample's side-car targets, padded to QUERIES, from `navsim_data.agents_of`."""
    index = target_index(agents, projection)
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


#: The two path steps' times, seconds after t.
PATH_DT = (0.5, 1.0)


def truth_agents(labels: dict, projections: np.ndarray, fill: str = "zero") -> dict:
    """Arm O's agents: the labels' targets the camera sees (`build`), in the side-car's decoded format,
    confidence 1, with `path_ok` beside them.

    `fill` says what a path step the label's track lacks becomes:
    - "zero", the first pass's arm O (`train_scorer.truth_agents`): offset 0, so the agent stands at
      its centre for that step;
    - "constant_velocity", B-1-2's: the step continues at the label's velocity from the step before
      (0.5 s from the centre, then 0.5 s more). A track that ends means the agent left the annotated
      range or was hidden, not that it stopped; and the side-car always predicts both steps, so B-1-2
      differs from B-1-3 only in perception (docs/drivejepa-perception-sidecar/part-b/B-1-veto.md).
    """
    from gallops.drivejepa_perception import navsim_data

    if fill not in ("zero", "constant_velocity"):
        raise ValueError(f"fill is zero or constant_velocity, not {fill!r}")
    n = len(labels["count"])
    out = {
        "cls": np.zeros((n, QUERIES), np.int64),
        "conf": np.zeros((n, QUERIES), np.float32),
        "xyz": np.zeros((n, QUERIES, 3), np.float32),
        "lwh": np.ones((n, QUERIES, 3), np.float32),
        "yaw": np.zeros((n, QUERIES), np.float32),
        "vel": np.zeros((n, QUERIES, 2), np.float32),
        "path": np.zeros((n, QUERIES, PATH_STEPS, 2), np.float32),
        "mask": np.zeros((n, QUERIES), bool),
        "path_ok": np.zeros((n, QUERIES, PATH_STEPS), bool),
    }
    for row in range(n):
        t = build(navsim_data.agents_of(labels, row), projections[row])
        m = t["mask"]
        out["cls"][row], out["conf"][row] = t["cls"], m.astype(np.float32)
        out["xyz"][row], out["lwh"][row] = t["box"][:, :3], np.where(m[:, None], t["box"][:, 3:6], 1.0)
        out["yaw"][row], out["vel"][row], out["mask"][row] = t["box"][:, 6], t["vel"], m
        out["path_ok"][row] = t["path_ok"]
        path = t["path"].copy()
        if fill == "constant_velocity":
            ok, vel = t["path_ok"], t["vel"]
            step0 = np.where(ok[:, 0, None], path[:, 0], vel * PATH_DT[0])
            step1 = np.where(ok[:, 1, None], path[:, 1], step0 + vel * (PATH_DT[1] - PATH_DT[0]))
            path = np.stack([step0, step1], axis=1) * m[:, None, None]
        out["path"][row] = path
    return out
