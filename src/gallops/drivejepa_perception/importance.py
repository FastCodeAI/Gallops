"""Which agents matter to the ego: the derived labels H15's plan proposes (§3.8 of
docs/drivejepa-perception-sidecar-A.md), for review before any of them changes a target, a loss weight or
a metric.

Simple on purpose: each is a rule a person can check on a picture, not a learned importance.

- **visible**: the share of an agent's box, as drawn on the 512x256 image, that no nearer agent's box
  covers. Rectangles, not silhouettes, and only labelled agents occlude (buildings and unlabelled objects
  do not): a line-of-sight proxy, not a visibility label. OpenScene's annotations carry no visibility flag.
- **in_path**: some position of the agent (now, at +0.5 s, at +1.0 s where its track lasted) lies in the
  ego's corridor: the ego's logged next 4 s (the labels' `ego_future`), as wide as both half-widths plus
  CORRIDOR_MARGIN_M. It reads the human's own future, so it is a label for training and grading, never an
  input.
- **causal** (in the simple sense the plan asks for): visible (at least VISIBLE_MIN) and either in_path or
  within NEAR_M of the ego. Hidden agents and visible ones far from the ego's path are not causal.
- **pedestrian groups**: pedestrians chained within GROUP_M of each other (single linkage). One detection
  anywhere in a group can stand for all of it: to the planner, one pedestrian or three at the same spot are
  the same obstacle.
- **rank**: nearest first, by distance from the ego (targets.build already orders targets this way).

numpy only.
"""

from __future__ import annotations

import numpy as np

from gallops.drivejepa_perception import targets as tg
from gallops.drivejepa_perception import views

VISIBLE_MIN = 0.5
NEAR_M = 10.0
CORRIDOR_MARGIN_M = 0.5
GROUP_M = 0.5
PEDESTRIAN = 1
#: Pixels per cell of the coverage grid on the 512x256 image.
GRID = 4


def image_rects(projection: np.ndarray, boxes: np.ndarray) -> tuple:
    """(N, 7) boxes -> (N, 4) u0, v0, u1, v1 of each box's corners in front of the camera, clipped to the
    512x256 image, and (N,) True where the rectangle is not empty."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    if not len(boxes):
        return np.zeros((0, 4)), np.zeros(0, bool)
    uv, depth = views.project(projection, views.box_corners(boxes))
    front = depth > 0.1
    uv = np.nan_to_num(uv)
    low = np.where(front[..., None], uv, np.inf).min(axis=1)  # corners behind the camera do not count
    high = np.where(front[..., None], uv, -np.inf).max(axis=1)
    rects = np.clip(np.concatenate([low, high], axis=1), 0, [views.MODEL_W, views.MODEL_H] * 2)
    ok = front.any(1) & (rects[:, 2] > rects[:, 0]) & (rects[:, 3] > rects[:, 1])
    return np.where(ok[:, None], rects, 0.0), ok


def visibility(projection: np.ndarray, boxes: np.ndarray, grid: int = GRID) -> np.ndarray:
    """(N,) the share of each box's image rectangle that the rectangles of boxes nearer the camera leave
    uncovered; NaN for a box with nothing in view. Nearer means a smaller camera depth of the centre."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    out = np.full(len(boxes), np.nan)
    rects, ok = image_rects(projection, boxes)
    if not len(boxes):
        return out
    _, depth = views.project(projection, boxes[:, :3])
    painted = np.zeros((views.MODEL_H // grid, views.MODEL_W // grid), bool)
    for i in np.argsort(np.where(ok, depth, np.inf), kind="stable"):
        if not ok[i]:
            continue
        u0, v0 = int(rects[i, 0] // grid), int(rects[i, 1] // grid)
        u1 = max(int(np.ceil(rects[i, 2] / grid)), u0 + 1)
        v1 = max(int(np.ceil(rects[i, 3] / grid)), v0 + 1)
        cell = painted[v0:v1, u0:u1]
        out[i] = 1.0 - float(cell.mean())
        painted[v0:v1, u0:u1] = True
    return out


def path_distance(ego_future: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(M, 2) points -> (M,) distance to the ego's path: the polyline from the origin (now) through the
    x, y of `ego_future` (its poses at 0.5 s steps)."""
    path = np.vstack([[0.0, 0.0], np.asarray(ego_future, dtype=np.float64)[:, :2]])
    a, ab = path[:-1], np.diff(path, axis=0)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    length2 = (ab**2).sum(-1)
    t = np.where(length2 > 0, ((p - a) * ab).sum(-1) / np.where(length2 > 0, length2, 1.0), 0.0)
    nearest = a + np.clip(t, 0.0, 1.0)[..., None] * ab
    return np.linalg.norm(p - nearest, axis=-1).min(axis=1)


def in_path(
    agents: dict, ego_future: np.ndarray, ego_half_width: float | None = None, horizon_s: float = 0.0
) -> np.ndarray:
    """(N,) whether any of the agent's labelled positions (now, +0.5 s, +1.0 s where known) is within the
    corridor: closer to the ego's path than its half-width, the ego's and CORRIDOR_MARGIN_M together.

    With `horizon_s`, also its positions at constant velocity every 0.5 s up to that horizon: a pedestrian
    walking towards the road is in the path before its labelled second shows it.
    """
    if ego_half_width is None:
        from gallops.drivejepa_perception.geometry import EGO_WIDTH  # nuPlan's Pacifica

        ego_half_width = EGO_WIDTH / 2
    box = np.asarray(agents["box"], dtype=np.float64).reshape(-1, 7)
    if not len(box):
        return np.zeros(0, bool)
    positions = [box[:, :2]]
    known = [np.ones(len(box), bool)]
    for step in range(np.asarray(agents["path"]).shape[1]):
        positions.append(np.asarray(agents["path"], dtype=np.float64)[:, step])
        known.append(np.asarray(agents["path_ok"])[:, step])
    velocity = np.asarray(agents.get("velocity", np.zeros((len(box), 2))), dtype=np.float64)
    for t in np.arange(0.5, horizon_s + 1e-9, 0.5):
        positions.append(box[:, :2] + t * velocity)
        known.append(np.ones(len(box), bool))
    reach = ego_half_width + box[:, 4] / 2 + CORRIDOR_MARGIN_M
    inside = [
        k & (path_distance(ego_future, np.nan_to_num(p)) <= reach)
        for p, k in zip(positions, known)  # noqa: B905
    ]
    return np.any(inside, axis=0)


def groups(xy: np.ndarray, cls: np.ndarray, radius: float = GROUP_M) -> np.ndarray:
    """(N,) a group id per pedestrian, chaining pedestrians within `radius` of each other (single linkage);
    -1 for every other class. A group's id is the index of its member nearest the ego."""
    xy, cls = np.asarray(xy, dtype=np.float64).reshape(-1, 2), np.asarray(cls)
    out = np.full(len(xy), -1, dtype=np.int64)
    members = np.flatnonzero(cls == PEDESTRIAN)
    if not len(members):
        return out
    parent = {int(m): int(m) for m in members}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.linalg.norm(xy[members, None] - xy[None, members], axis=-1)
    for a, b in zip(*np.nonzero(np.triu(d <= radius, k=1))):  # noqa: B905
        parent[find(int(members[a]))] = find(int(members[b]))
    roots = np.array([find(int(m)) for m in members])
    rng = np.hypot(xy[:, 0], xy[:, 1])
    for root in np.unique(roots):
        group = members[roots == root]
        out[group] = int(group[np.argmin(rng[group])])
    return out


def target_importance(agents: dict, projection: np.ndarray, ego_future: np.ndarray, **kwargs) -> dict:
    """The labels above for one sample's targets, in `targets.build`'s order (its first n rows).

    Every labelled agent occludes (static objects included); only targets are labelled.
    """
    index = tg.target_index(agents, projection)
    seen_share = visibility(projection, agents["box"])[index]
    chosen = {key: np.asarray(agents[key])[index] for key in ("box", "cls", "velocity", "path", "path_ok")}
    xy = chosen["box"][:, :2].astype(np.float64)
    path = in_path(chosen, ego_future, kwargs.get("ego_half_width"), kwargs.get("horizon_s", 0.0))
    near = np.hypot(xy[:, 0], xy[:, 1]) <= NEAR_M
    visible = np.nan_to_num(seen_share, nan=0.0) >= VISIBLE_MIN
    return {
        "visible": seen_share,
        "in_path": path,
        "causal": visible & (path | near),
        "group": groups(xy, chosen["cls"], kwargs.get("radius", GROUP_M)),
        "rank": np.arange(len(index)),
    }
