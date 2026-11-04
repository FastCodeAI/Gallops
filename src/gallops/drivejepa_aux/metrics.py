"""Held-out measurements of an aux run: the plan, the heads against a trivial baseline, and their
uncertainty. numpy only, so `make check` tests every one; the trainer and `report` call them.

**The plan** (the picked one of 32, against the human's next seconds):
- L2 at 1, 2 and 3 s. Waypoint k is at (k + 1) x 0.5 s, so those are waypoints 1, 3 and 5.
- collision within 3 s: the ego's footprint at a waypoint overlaps a box logged at that time.
  The footprint and the separating-axis test are Gallops' grader's (`expert.grader`, the
  neuroNCAP vendor's numbers), imported. The boxes are every annotation, as paddock's nuScenes
  benchmark counts them. Open loop: the logged boxes do not react to the plan.
- off road within 3 s: a waypoint inside the BEV grid (32 m ahead, 32 m either side) on a cell
  the map does not call drivable. Waypoints beyond the grid are not judged.
- best of 32: the proposal nearest the human over 3 s (minADE), which is what the generator
  learns and the scorer cannot fix.

**The heads**, each beside what a model with no camera would score:
- BEV: IoU per class over the labelled cells, against the per-cell majority of the train labels.
- boxes: F1 of confident boxes (p > 0.5) matched to labelled ones within 2 m, against no boxes.
- lead: present accuracy against "never a lead"; gap error against the train mean gap.
- lane change: F1 of left and of right, against "always keep".

**Uncertainty is clustered by scene.** A scene's 30-odd keyframes are 0.5 s apart and far from
independent; a per-keyframe standard error would be several times too small.
"""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_aux import bev
from gallops.expert.grader import EGO_LENGTH, EGO_WIDTH, REAR_AXLE_TO_CENTRE, corners, overlap

#: Waypoint index of each L2 horizon: waypoint k is at (k + 1) x 0.5 s.
L2_WAYPOINTS = {"1s": 1, "2s": 3, "3s": 5}
#: Collision, off-road and minADE look this many waypoints ahead: 3 s, the nuScenes protocol's horizon.
HORIZON_WAYPOINTS = 6
#: A box is "predicted" above this probability, and matched within this distance of a label.
BOX_THRESHOLD = 0.5
BOX_MATCH_M = 2.0
#: Lane-change classes scored: a change, either way ("keep" is the majority any model gets).
LANE_CHANGES = {"left": 1, "right": 2}


# --- the plan -------------------------------------------------------------------------------------


def l2(plans: np.ndarray, expert: np.ndarray) -> np.ndarray:
    """(N, 8) distance between each plan's waypoint and the human's."""
    return np.linalg.norm(plans[..., :2] - expert[..., :2], axis=-1)


def min_ade(proposals: np.ndarray, expert: np.ndarray, steps: int = HORIZON_WAYPOINTS) -> np.ndarray:
    """(N,) the best of each row's proposals, (N, P, 8, 3), by mean distance over `steps` waypoints."""
    gap = np.linalg.norm(proposals[..., :steps, :2] - expert[:, None, :steps, :2], axis=-1)
    return gap.mean(-1).min(-1)


def ego_footprint(pose: np.ndarray) -> np.ndarray:
    """The ego's corners (4, 2) at a waypoint (x, y, heading) of its rear axle."""
    heading = float(pose[2])
    centre = pose[:2] + REAR_AXLE_TO_CENTRE * np.array([np.cos(heading), np.sin(heading)])
    return corners(centre, heading, EGO_LENGTH, EGO_WIDTH)


def collides(plan: np.ndarray, boxes: list, steps: int = HORIZON_WAYPOINTS) -> bool:
    """Whether the plan's footprint overlaps a box at the same step. `boxes[k]` is (M, 5)
    [x, y, heading, length, width] at waypoint k's time, in the plan's frame."""
    reach = 0.5 * np.hypot(EGO_LENGTH, EGO_WIDTH) + REAR_AXLE_TO_CENTRE
    for k in range(min(steps, len(plan), len(boxes))):
        near = boxes[k]
        if not len(near):
            continue
        # a cheap circle test first: most boxes are nowhere near the ego
        span = reach + 0.5 * np.hypot(near[:, 3], near[:, 4])
        near = near[np.hypot(near[:, 0] - plan[k, 0], near[:, 1] - plan[k, 1]) < span]
        if not len(near):
            continue
        ego = ego_footprint(plan[k])
        for box in near:
            if overlap(ego, corners(box[:2], float(box[2]), float(box[3]), float(box[4]))):
                return True
    return False


def collided(plans: np.ndarray, boxes: list, steps: int = HORIZON_WAYPOINTS) -> np.ndarray:
    """(N,) `collides` for each row; `boxes[i]` is row i's list of per-step boxes."""
    return np.array([collides(p, b, steps) for p, b in zip(plans, boxes)], dtype=bool)  # noqa: B905 -- 3.9


def grid_values(xy: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The grid's value under each point (..., 2) and whether the point is on the grid at all."""
    rows, cols = bev.xy_to_cell(xy[..., 0], xy[..., 1])
    inside = bev.in_grid(rows, cols)
    values = np.zeros(inside.shape, dtype=grid.dtype)
    values[inside] = grid[rows[inside], cols[inside]]
    return values, inside


def off_road(plans: np.ndarray, road: np.ndarray, steps: int = HORIZON_WAYPOINTS) -> np.ndarray:
    """(N,) 1.0 where a waypoint within `steps` lies on the grid and off the drivable area, 0.0
    where every one on the grid is on it, NaN where none is on the grid. `road` is (N, ROWS, COLS)."""
    out = np.full(len(plans), np.nan)
    for i, (plan, grid) in enumerate(zip(plans, road)):  # noqa: B905 -- 3.9
        on, inside = grid_values(plan[:steps, :2], grid)
        if inside.any():
            out[i] = float((~on[inside].astype(bool)).any())
    return out


def on_road(path: np.ndarray, road: np.ndarray, view: np.ndarray) -> tuple[int, int]:
    """Of a path's waypoints on the grid and in the camera's view: how many, and how many on road.

    The labels' sanity check: the human drove on the road, so a map that disagrees on more than a
    sliver of their path is misplaced (a pose or a map error), and so is every label made from it.
    """
    on, inside = grid_values(path[:, :2], road)
    seen, _ = grid_values(path[:, :2], view)
    judged = inside & seen.astype(bool)
    return int(judged.sum()), int(on[judged].astype(bool).sum())


# --- uncertainty ----------------------------------------------------------------------------------


def clustered(values, clusters) -> tuple[float, float, int]:
    """Mean, its standard error clustered by `clusters` (CR1), and the number of clusters. NaN
    values are left out; a single cluster has no standard error (NaN)."""
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]
    if not len(values):
        return float("nan"), float("nan"), 0
    mean = float(values.mean())
    names, index = np.unique(clusters, return_inverse=True)
    sums = np.bincount(index, weights=values - mean)
    g, n = len(names), len(values)
    se = float(np.sqrt(g / (g - 1) * (sums**2).sum()) / n) if g > 1 else float("nan")
    return mean, se, g


def paired(a, b, clusters) -> tuple[float, float, int]:
    """Mean of a - b over the rows both have, with its scene-clustered standard error."""
    return clustered(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), clusters)


# --- the heads ------------------------------------------------------------------------------------


def bev_counts(predicted: np.ndarray, label: np.ndarray, classes: int = len(bev.CLASSES)) -> np.ndarray:
    """(N, classes, 2) intersection and union per row and class, over the labelled cells only."""
    seen = label != bev.IGNORE
    out = np.zeros((len(label), classes, 2), dtype=np.int64)
    for c in range(classes):
        p, t = (predicted == c) & seen, (label == c) & seen
        out[:, c, 0] = (p & t).reshape(len(label), -1).sum(1)
        out[:, c, 1] = (p | t).reshape(len(label), -1).sum(1)
    return out


def iou(counts: np.ndarray) -> np.ndarray:
    """(classes,) IoU over every row's cells, from `bev_counts`."""
    total = counts.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total[:, 0] / total[:, 1]


def prior_bev(labels: np.ndarray, classes: int = len(bev.CLASSES)) -> np.ndarray:
    """The no-camera baseline: each cell's most frequent labelled class over `labels` (N, ROWS, COLS);
    background where no row labels the cell."""
    votes = np.stack([(labels == c).sum(0) for c in range(classes)])
    return votes.argmax(0).astype(np.uint8)


def box_matches(
    states: np.ndarray, logits: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> tuple[int, int, int, list]:
    """One row's predicted boxes against its labelled ones: true positives, false positives, false
    negatives and each match's centre error. Confident boxes, most confident first, each take the
    nearest unmatched label within BOX_MATCH_M."""
    keep = 1.0 / (1.0 + np.exp(-logits)) > BOX_THRESHOLD
    order = np.argsort(-logits[keep])
    predicted = states[keep][order]
    labelled = truth[valid]
    free = np.ones(len(labelled), dtype=bool)
    errors = []
    for box in predicted:
        if not free.any():
            break
        gap = np.hypot(labelled[:, 0] - box[0], labelled[:, 1] - box[1])
        gap[~free] = np.inf
        j = int(gap.argmin())
        if gap[j] <= BOX_MATCH_M:
            free[j] = False
            errors.append(float(gap[j]))
    tp = len(errors)
    return tp, len(predicted) - tp, len(labelled) - tp, errors


def f1(tp, fp, fn) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def lane_f1(predicted: np.ndarray, label: np.ndarray) -> dict:
    """F1 of each lane change, over the rows whose label is known (>= 0)."""
    known = label >= 0
    p, t = predicted[known], label[known]
    out = {}
    for name, c in LANE_CHANGES.items():
        tp, fp, fn = ((p == c) & (t == c)).sum(), ((p == c) & (t != c)).sum(), ((p != c) & (t == c)).sum()
        out[name] = f1(int(tp), int(fp), int(fn))
    return out
