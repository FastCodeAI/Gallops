"""Grade many trajectories at once, and the sub-scores the scorer's six logits learn.

`grader.grade` judges one trajectory at a time in a Python loop: about 2-3 s per
tick for 64 plans against every vehicle, which is a day of CPU for the 24k ticks
the scorer cache holds. This is the same judgement written over arrays. It is held
to `grader.grade` exactly -- `tests/test_fastgrade.py` compares the two on random
scenes, and `cache_labels` re-checks a sample of real ticks before it writes --
so a verdict here means what a verdict there means.

Beyond the grader's three answers it computes what Drive-JEPA's scorer has slots
for but was never trained on (the vendor's `sub_score_weight` is 0):

* `hit5_at` -- a hit within 5 s when the plan carries on past its last waypoint at
  its final velocity, so a plan clear at 4 s but closing fast on a car still counts.
* `ttc_at` -- the first step (up to 3 s) where the ego box, carried on at its
  current velocity for 0.3, 0.6 or 0.9 s, overlaps a vehicle ahead of it. Shaped
  after PDM's time-to-collision check; not PDM's code, and with neuroNCAP's rule
  that a stopped ego can still be hit.
* `lane_run_s` -- the longest continuous time the plan's centre is more than
  0.5 m from the lane centre (PDM's lane-keeping fails at 2 s).
* `travel` -- path length over 4 s, for the progress answer.
"""

from __future__ import annotations

import numpy as np

from gallops.expert.grader import EGO_LENGTH, EGO_WIDTH, HORIZON, REAR_AXLE_TO_CENTRE, TRUST_BAND_M

WAYPOINT_DT = 0.5
#: Offsets PDM's TTC check projects the ego forward by.
TTC_OFFSETS = (0.3, 0.6, 0.9)
TTC_UNTIL_S = 3.0
LANE_KEEP_M = 0.5
EXTENDED_S = 5.0


def densify_many(plans: np.ndarray, horizon: np.ndarray = HORIZON):
    """(P, 8, 3) waypoints at 0.5 s -> (x, y, heading), each (P, len(horizon)), as `grader.densify`."""
    plans = np.asarray(plans, dtype=float)
    p, n = plans.shape[:2]
    pts = np.concatenate([np.zeros((p, 1, 3)), plans], axis=1)  # stamps 0, 0.5, ..., 0.5 n
    heading = np.unwrap(pts[:, :, 2], axis=1)
    stamps = np.arange(n + 1) * WAYPOINT_DT
    # np.interp holds the end value past the last stamp, exactly as grader.densify does.
    i = np.clip(np.searchsorted(stamps, horizon, side="right") - 1, 0, n - 1)
    w = np.clip((horizon - stamps[i]) / WAYPOINT_DT, 0.0, 1.0)
    out = []
    for column in (pts[:, :, 0], pts[:, :, 1], heading):
        out.append(column[:, i] * (1 - w) + column[:, i + 1] * w)
    return out[0], out[1], out[2]


def _corners(centre: np.ndarray, yaw: np.ndarray, length, width) -> np.ndarray:
    """(..., 2) centres and (...,) yaws -> (..., 4, 2), corner order as grader.corners."""
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    left = np.stack([-fwd[..., 1], fwd[..., 0]], axis=-1)
    length = np.asarray(length)[..., None, None] if np.ndim(length) else length
    width = np.asarray(width)[..., None, None] if np.ndim(width) else width
    signs = np.array([(1, 1), (1, -1), (-1, -1), (-1, 1)], dtype=float)
    return (
        centre[..., None, :]
        + signs[:, 0, None] * fwd[..., None, :] * (length / 2)
        + signs[:, 1, None] * left[..., None, :] * (width / 2)
    )


def ego_boxes(x, y, heading, ego2world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World footprint (P, T, 4, 2), rear-axle positions (P, T, 2) and yaws (P, T)."""
    rotation, origin = ego2world[:2, :2], ego2world[:2, 3]
    yaw0 = np.arctan2(rotation[1, 0], rotation[0, 0])
    local = np.stack([x, y], axis=-1)
    position = origin + local @ rotation.T
    yaw = yaw0 + heading
    centre = position + REAR_AXLE_TO_CENTRE * np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    return _corners(centre, yaw, EGO_LENGTH, EGO_WIDTH), position, yaw


def actor_states(actor: dict, times_us: np.ndarray):
    """Position (T, 2), yaw (T,), active (T,) at absolute times, as `grader.actor_at`."""
    stamps = np.asarray(actor["timestamps"], dtype=float)
    poses = np.asarray(actor["poses"], dtype=float)
    active = (stamps[0] <= times_us) & (times_us <= stamps[-1])
    if len(stamps) < 2:
        return np.zeros((len(times_us), 2)), np.zeros(len(times_us)), np.zeros(len(times_us), bool)
    i = np.clip(np.searchsorted(stamps, times_us) - 1, 0, len(stamps) - 2)
    w = (times_us - stamps[i]) / (stamps[i + 1] - stamps[i])
    position = poses[i, :2, 3] * (1 - w)[:, None] + poses[i + 1, :2, 3] * w[:, None]
    y0 = np.arctan2(poses[i, 1, 0], poses[i, 0, 0])
    y1 = np.arctan2(poses[i + 1, 1, 0], poses[i + 1, 0, 0])
    return position, y0 + np.angle(np.exp(1j * (y1 - y0))) * w, active


def actor_boxes(actors: list[dict], times_us: np.ndarray):
    """(A, T, 4, 2) boxes, (A, T) active, (A, T, 2) centres."""
    if not actors:
        t = len(times_us)
        return np.zeros((0, t, 4, 2)), np.zeros((0, t), bool), np.zeros((0, t, 2))
    boxes, active, centres = [], [], []
    for actor in actors:
        position, yaw, on = actor_states(actor, times_us)
        boxes.append(_corners(position, yaw, actor["dims"][1], actor["dims"][0]))
        active.append(on)
        centres.append(position)
    return np.stack(boxes), np.stack(active), np.stack(centres)


def overlap_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Separating-axis test on broadcastable (..., 4, 2) rectangles, as `grader.overlap`.

    Touching counts as overlapping, as it does there.
    """
    separated = np.zeros(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]), dtype=bool)
    for polygon in (a, b):
        for i in range(4):
            edge = polygon[..., (i + 1) % 4, :] - polygon[..., i, :]
            normal = np.stack([-edge[..., 1], edge[..., 0]], axis=-1)[..., None, :]
            pa = (a * normal).sum(-1)
            pb = (b * normal).sum(-1)
            separated |= (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~separated


def _first(mask: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """(P, T) booleans -> first tau where True, NaN where never."""
    any_ = mask.any(1)
    out = np.full(mask.shape[0], np.nan)
    out[any_] = taus[mask[any_].argmax(1)]
    return out


def first_hit_many(boxes: np.ndarray, actors: list[dict], t0: int, taus: np.ndarray) -> np.ndarray:
    """(P,) first tau where a plan's footprint overlaps any actor, NaN if none."""
    abox, active, _ = actor_boxes(actors, t0 + taus * 1e6)
    if len(abox) == 0:
        return np.full(boxes.shape[0], np.nan)
    hit = overlap_many(boxes[:, None], abox[None]) & active[None]  # (P, A, T)
    return _first(hit.any(1), taus)


def leaves_road_many(boxes: np.ndarray, area, taus: np.ndarray) -> np.ndarray:
    """(P,) first tau where any footprint corner is off the drivable raster, NaN if none."""
    inside = area.contains(boxes.reshape(-1, 2)).reshape(boxes.shape[:-1])  # (P, T, 4)
    return _first(~inside.all(-1), taus)


def polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each (..., 2) point to the polyline's segments, as `grader._to_polyline`."""
    polyline = np.asarray(polyline, dtype=float)[:, :2]
    flat = points.reshape(-1, 2)
    if len(polyline) < 2:
        d = np.linalg.norm(flat[:, None] - polyline[None], axis=-1).min(1)
        return d.reshape(points.shape[:-1])
    seg = np.diff(polyline, axis=0)
    length_sq = np.maximum((seg**2).sum(1), 1e-9)
    best = np.full(len(flat), np.inf)
    for start in range(0, len(flat), 4096):  # bound memory: points x segments
        chunk = flat[start : start + 4096]
        along = np.clip(((chunk[:, None] - polyline[None, :-1]) * seg[None]).sum(-1) / length_sq, 0.0, 1.0)
        feet = polyline[None, :-1] + along[..., None] * seg[None]
        best[start : start + 4096] = np.linalg.norm(feet - chunk[:, None], axis=-1).min(1)
    return best.reshape(points.shape[:-1])


def _longest_run(mask: np.ndarray, dt: float) -> np.ndarray:
    """(P, T) booleans -> longest run of consecutive Trues, in seconds."""
    best = np.zeros(mask.shape[0])
    run = np.zeros(mask.shape[0])
    for k in range(mask.shape[1]):
        run = np.where(mask[:, k], run + 1, 0)
        best = np.maximum(best, run)
    return best * dt


def extend(plans: np.ndarray, until_s: float = EXTENDED_S) -> np.ndarray:
    """Waypoints continued past 4 s at each plan's final velocity and heading."""
    plans = np.asarray(plans, dtype=float)
    last = plans[:, -1]
    prev = plans[:, -2]
    step = last[:, :2] - prev[:, :2]
    extra = []
    for k in range(1, int(round((until_s - plans.shape[1] * WAYPOINT_DT) / WAYPOINT_DT)) + 1):
        xy = last[:, :2] + k * step
        extra.append(np.concatenate([xy, last[:, 2:3]], axis=1))
    return np.concatenate([plans, np.stack(extra, axis=1)], axis=1) if extra else plans


def grade_many(
    plans: np.ndarray,
    ego2world: np.ndarray,
    t0: int,
    vehicles: list[dict],
    reference: np.ndarray | None = None,
    area=None,
    targets: list[dict] | None = None,
) -> dict[str, np.ndarray]:
    """Every verdict and sub-score for (P, 8, 3) plans planned at t0 from ego2world.

    `vehicles` is what a plan can hit (pass every vehicle, as neuroNCAP counts any);
    `targets`, if given, gets its own hit time so the inserted car can be told apart.
    The first four keys are `grader.grade`'s, with NaN for None.
    """
    plans = np.asarray(plans, dtype=float)
    x, y, heading = densify_many(plans)
    boxes, position, yaw = ego_boxes(x, y, heading, ego2world)
    out = {"hit_at": first_hit_many(boxes, vehicles, t0, HORIZON)}

    if reference is not None:
        gap = polyline_distance(position, reference)
        out["max_deviation"] = gap.max(1)
        out["off_path_at"] = _first(gap > TRUST_BAND_M, HORIZON)
        out["lane_run_s"] = _longest_run(gap > LANE_KEEP_M, 0.1)
    else:
        nan = np.full(len(plans), np.nan)
        out.update(max_deviation=np.zeros(len(plans)), off_path_at=nan, lane_run_s=nan)
    out["off_road_at"] = (
        leaves_road_many(boxes, area, HORIZON) if area is not None else np.full(len(plans), np.nan)
    )

    long_h = np.round(np.arange(0.1, EXTENDED_S + 1e-4, 0.1), 1)
    ex, ey, eh = densify_many(extend(plans), long_h)
    long_boxes, _, _ = ego_boxes(ex, ey, eh, ego2world)
    out["hit5_at"] = first_hit_many(long_boxes, vehicles, t0, long_h)
    if targets is not None:
        out["hit_target_at"] = first_hit_many(boxes, targets, t0, HORIZON)

    # TTC: carry the box on at its current velocity and look for a vehicle ahead.
    steps = HORIZON <= TTC_UNTIL_S + 1e-9
    taus = HORIZON[steps]
    pos = np.concatenate([ego2world[None, :2, 3].repeat(len(plans), 0)[:, None], position], axis=1)
    velocity = np.diff(pos, axis=1)[:, : len(HORIZON)] / 0.1  # (P, T, 2), step ending at each tau
    speed = np.linalg.norm(velocity, axis=-1)
    violation = np.zeros((len(plans), steps.sum()), dtype=bool)
    cbox = boxes[:, steps]
    for offset in TTC_OFFSETS:
        moved = cbox + (velocity[:, steps] * offset)[:, :, None, :]
        abox, active, centres = actor_boxes(vehicles, t0 + (taus + offset) * 1e6)
        if len(abox) == 0:
            break
        hit = overlap_many(moved[:, None], abox[None]) & active[None]  # (P, A, T)
        fwd = np.stack([np.cos(yaw[:, steps]), np.sin(yaw[:, steps])], axis=-1)  # (P, T, 2)
        rel = centres[None] - position[:, None, steps]  # (P, A, T, 2)
        ahead = (rel * fwd[:, None]).sum(-1) > 0
        violation |= (hit & ahead).any(1)
    violation &= speed[:, steps] > 5e-3
    out["ttc_at"] = _first(violation, taus)

    seg = np.diff(np.concatenate([np.zeros((len(plans), 1, 2)), plans[:, :, :2]], axis=1), axis=1)
    out["travel"] = np.linalg.norm(seg, axis=-1).sum(1)
    return out
