"""Provisional vocabulary for smoke-testing the cache: versions A (pick's speed) and B (current speed), 64 paths."""
import numpy as np

OFFSETS = (-3.5, -2.5, -1.5, -1.0, 1.0, 1.5, 2.5, 3.5)
SPEEDS = (1.0, 0.7)
SHAPES = ("hold", "return")
DT = 0.5
T = np.arange(1, 9) * DT


def ease(u):
    return (1 - np.cos(np.pi * np.clip(u, 0.0, 1.0))) / 2


def lateral(shape, t):
    if shape == "hold":
        return ease(t / 2.0)
    out = np.where(t <= 1.5, ease(t / 1.5), 1.0)
    return np.where(t > 2.5, ease((4.0 - t) / 1.5), out)


def retime(base, factor):
    if factor == 1.0:
        return base[:, :2].copy()
    k = (1 - factor) / 2.0
    poly = np.vstack([[0.0, 0.0], base[:, :2]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] < 1e-6:
        return base[:, :2].copy()
    new_arc = np.cumsum(seg * (1 - k * (T - DT / 2)))
    return np.stack([np.interp(new_arc, arc, poly[:, 0]), np.interp(new_arc, arc, poly[:, 1])], axis=1)


def at_speed(pick, v0, factor):
    k = (1 - factor) / 2.0
    dist = v0 * (T - k * T**2 / 2)
    poly = np.vstack([[0.0, 0.0], pick[:, :2]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] < 0.5:
        poly, arc = np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.0, 1.0])
    d = poly[-1] - poly[-2]
    d = d / max(np.linalg.norm(d), 1e-6)
    extra = max(0.0, dist.max() - arc[-1]) + 1.0
    poly = np.vstack([poly, poly[-1] + extra * d])
    arc = np.r_[arc, arc[-1] + extra]
    return np.stack([np.interp(dist, arc, poly[:, 0]), np.interp(dist, arc, poly[:, 1])], axis=1)


def vocabulary(natives, pick_index, speed, accel):
    pick = np.asarray(natives[pick_index], dtype=float)
    paths, meta = [], []
    for base in ("pick", "current"):
        for factor in SPEEDS:
            along = retime(pick, factor) if base == "pick" else at_speed(pick, speed, factor)
            for shape in SHAPES:
                frac = lateral(shape, T)
                for off in OFFSETS:
                    xy = along.copy()
                    xy[:, 1] += off * frac
                    step = np.diff(np.vstack([[0.0, 0.0], xy]), axis=0)
                    paths.append(np.column_stack([xy, np.arctan2(step[:, 1], step[:, 0])]))
                    meta.append({"base": base, "speed": factor, "shape": shape, "offset": off})
    return np.stack(paths), meta
