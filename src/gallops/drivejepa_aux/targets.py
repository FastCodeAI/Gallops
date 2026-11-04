"""The rules that turn one keyframe's boxes and the logged future into auxiliary targets.

Pure functions of arrays in the ego frame (x forward, y left, metres), so each rule is tested
on hand-built scenes in `make check`. What reads nuScenes and the map is `nusc`.

Three targets, each with a mask, because a target that does not exist must cost nothing:

- **agents**: up to 30 vehicle boxes (x, y, heading, length, width), upstream's format for its
  box head, restricted to boxes whose centre the front camera sees and that lie on the BEV grid.
- **lead vehicle**: the nearest vehicle ahead whose footprint overlaps the band the ego sweeps
  along its logged path: present, gap, closing speed.
- **lane change**: keep / left / right over the next 4 s, from which lane the ego occupies now
  and 4 s later.
"""

from __future__ import annotations

import math

import numpy as np

from gallops.drivejepa_aux import bev

MAX_AGENTS = 30  # DriveJEPAConfig.num_bounding_boxes

#: How far ahead a lead vehicle is looked for, along the path.
LEAD_RANGE_M = 50.0
#: Half the ego's width. A vehicle is in the ego's way when their footprints overlap laterally:
#: |offset from the path| < its half width + this.
EGO_HALF_WIDTH_M = 0.9
#: The path is extended straight past its last point to LEAD_RANGE_M; a path shorter than this
#: (the ego is stopped) is replaced by the ego's own heading.
MIN_PATH_M = 1.0

LANE_CHANGE_CLASSES = ("keep", "left", "right")
#: A lane change moves the car about a lane's width across its starting lane. Ending in another
#: lane polygon after moving less than this is riding the line between two lanes -- on scene-0001
#: the ego drives along a divider past parked trucks and the polygon holding it flips keyframe to
#: keyframe -- and is labelled keep.
LANE_CHANGE_MIN_SHIFT_M = 2.0


def agent_targets(boxes: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's box-head target: (MAX_AGENTS, 5) states and (MAX_AGENTS,) valid, nearest first.

    `boxes` (N, 5) are (x, y, heading, length, width); `seen` (N,) says whether each box's centre
    lands inside the part of the image the model reads. Upstream keeps every vehicle within
    +-32 m, behind the car included; this model sees one front camera, so a box is kept only if
    it is on the BEV grid (0 <= x < 32 m, |y| < 32 m) and seen.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    keep = np.asarray(seen, dtype=bool) & (boxes[:, 0] >= 0) & (boxes[:, 0] < bev.X_MAX)
    keep &= np.abs(boxes[:, 1]) < bev.Y_HALF
    kept = boxes[keep]
    kept = kept[np.argsort(np.hypot(kept[:, 0], kept[:, 1]), kind="stable")][:MAX_AGENTS]
    states = np.zeros((MAX_AGENTS, 5), dtype=np.float32)
    valid = np.zeros(MAX_AGENTS, dtype=bool)
    states[: len(kept)] = kept
    valid[: len(kept)] = True
    return states, valid


def ego_path(future_xy: np.ndarray, length: float = LEAD_RANGE_M) -> np.ndarray:
    """The ego's path from its origin through the logged future, extended straight to `length`."""
    points = np.vstack([[0.0, 0.0], np.asarray(future_xy, dtype=float)[:, :2]])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if steps.sum() < MIN_PATH_M:
        return np.array([[0.0, 0.0], [length, 0.0]])
    # keep only real steps, so the extension's direction is defined
    points = np.vstack([points[:1], points[1:][steps > 1e-3]])
    travelled = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if travelled < length:
        direction = points[-1] - points[-2]
        direction = direction / np.linalg.norm(direction)
        points = np.vstack([points, points[-1] + direction * (length - travelled)])
    return points


def path_coordinates(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance along `path` (s), signed offset from it (d, left positive) and the path's unit
    tangent there, for each point (N, 2)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    a, b = path[:-1], path[1:]
    seg = b - a
    seg_len = np.linalg.norm(seg, axis=1)
    start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    rel = points[:, None, :] - a[None]  # (N, S, 2)
    t = np.clip((rel * seg[None]).sum(-1) / np.maximum(seg_len**2, 1e-12)[None], 0.0, 1.0)
    foot = a[None] + t[..., None] * seg[None]
    dist = np.linalg.norm(points[:, None, :] - foot, axis=-1)
    nearest = dist.argmin(1)
    rows = np.arange(len(points))
    tangent = seg[nearest] / np.maximum(seg_len[nearest], 1e-12)[:, None]
    offset = points - foot[rows, nearest]
    d = tangent[:, 0] * offset[:, 1] - tangent[:, 1] * offset[:, 0]
    s = start[nearest] + t[rows, nearest] * seg_len[nearest]
    return s, d, tangent


def lead_vehicle(
    boxes: np.ndarray,
    velocities: np.ndarray,
    seen: np.ndarray,
    future_xy: np.ndarray,
    ego_speed: float,
) -> dict:
    """The nearest vehicle ahead in the ego's way, along its logged path.

    `boxes` (N, 5) vehicles as (x, y, heading, length, width); `velocities` (N, 2) their ground
    velocity in ego axes (NaN where unknown); `seen` (N,) whether the camera sees each centre.
    A vehicle qualifies when it is seen, ahead along the path (0 < s <= LEAD_RANGE_M), its
    footprint overlaps the ego's swept band (|d| < width / 2 + EGO_HALF_WIDTH_M), and it is not
    oncoming (its heading within 90 degrees of the path's).

    Returns present (0 or 1), gap (m, along the path from the ego's origin -- nuScenes' ego
    frame sits at the rear axle -- to the lead's rear edge), closing speed (m/s, positive when
    the ego gains on it; NaN when its velocity is unknown) and the index of the box.
    """
    none = {"present": 0, "gap": float("nan"), "closing": float("nan"), "index": -1}
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
    if not len(boxes):
        return none
    path = ego_path(future_xy)
    s, d, tangent = path_coordinates(boxes[:, :2], path)
    facing = np.cos(boxes[:, 2]) * tangent[:, 0] + np.sin(boxes[:, 2]) * tangent[:, 1]
    candidate = np.asarray(seen, dtype=bool) & (s > 0) & (s <= LEAD_RANGE_M) & (facing > 0)
    candidate &= np.abs(d) < boxes[:, 4] / 2 + EGO_HALF_WIDTH_M
    if not candidate.any():
        return none
    index = int(np.flatnonzero(candidate)[np.argmin(s[candidate])])
    lead_speed = float(np.dot(np.asarray(velocities, dtype=float)[index], tangent[index]))
    return {
        "present": 1,
        "gap": float(s[index] - boxes[index, 3] / 2),
        "closing": float(ego_speed - lead_speed) if math.isfinite(lead_speed) else float("nan"),
        "index": index,
    }


def lane_change(lanes_now: set, lanes_later: set, reachable: set, shift: float) -> int:
    """keep (0) / left (1) / right (2), or -1 when either end is off every lane.

    `lanes_now` and `lanes_later` are the lanes (and lane connectors) whose polygon holds the
    ego now and 4 s later; `reachable` is every lane the lane graph leads to from `lanes_now`
    within the horizon; `shift` is how far the ego moved across its starting lane's centreline
    in those 4 s (left positive). A change is ending in a lane the graph does not lead to AND
    having moved at least LANE_CHANGE_MIN_SHIFT_M sideways.
    """
    if not lanes_now or not lanes_later:
        return -1
    if lanes_later & (set(lanes_now) | set(reachable)) or abs(shift) < LANE_CHANGE_MIN_SHIFT_M:
        return 0
    return 1 if shift > 0 else 2
