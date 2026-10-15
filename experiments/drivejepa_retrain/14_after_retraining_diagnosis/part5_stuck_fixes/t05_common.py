#!/usr/bin/env python
"""T05 shared. Reuses part4_why_it_stays_stopped/t04_common.py (cached run reader, box check, dense plan, ranks) by import; adds the rectangle-to-rectangle
clearance, the standing clock and the per-proposal table for all 30 scenes."""
import math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part4_why_it_stays_stopped"))
import t04_common as T  # noqa: E402
C = T.C
V_STAND = 0.5; STUCK = T.STUCK


def corners(c, yaw, h):
    """c (N,2), yaw (N,), h (N,2) half sizes -> (N,4,2)."""
    cs, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]; sx = np.array([1, 1, -1, -1.0])[None] * h[:, :1]; sy = np.array([1, -1, -1, 1.0])[None] * h[:, 1:]
    return np.stack([c[:, :1] + cs * sx - sn * sy, c[:, 1:] + sn * sx + cs * sy], -1)


def _pt_seg(p, a, b):
    """p (N,P,2) against segments a->b (N,S,2) -> (N,P,S) distances."""
    ab = (b - a)[:, None]; ap = p[:, :, None] - a[:, None]; t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1)


def rect_dist(c1, yaw1, h1, c2, yaw2, h2):
    """One ego rectangle (c1 (2,), yaw1, h1 (2,)) against N actor rectangles -> (N,) distance, 0 where they overlap."""
    n = len(c2); hit = T.sat(c1[None], np.array([yaw1]), h1, c2, yaw2, h2)[0]
    k1 = np.repeat(corners(c1[None], np.array([yaw1]), h1[None]), n, 0); k2 = corners(c2, yaw2, h2)
    d = np.minimum(_pt_seg(k1, k2, np.roll(k2, -1, 1)).min((1, 2)), _pt_seg(k2, k1, np.roll(k1, -1, 1)).min((1, 2)))
    return np.where(hit, 0.0, d)


def clearance(o, t0_us, world_rig, times_s):
    """Minimum distance (m) between the ego box and any recorded actor box at the same time stamps (0 = overlap). Times beyond the actor
    log are not checked (as t04 box_check). -> (min_clearance_m, time_s of the minimum, actor id)."""
    ap_t = o["ap_t"]; h1 = o["ego_dims"] / 2; best = (np.inf, np.nan, "")
    for k, t in enumerate(times_s):
        tq = t0_us + int(round(t * 1e6)); i = min(int(np.searchsorted(ap_t, tq)), len(ap_t) - 1)
        if abs(int(ap_t[i]) - tq) > 1000:
            if i > 0 and abs(int(ap_t[i - 1]) - tq) <= 1000: i -= 1
            else: continue
        a = o["A"][i]; x, y, yaw = world_rig[k]; c = np.array([x + o["rig_to_aabb_x"] * math.cos(yaw), y + o["rig_to_aabb_x"] * math.sin(yaw)])
        near = np.isfinite(a[:, 0]) & (np.hypot(a[:, 0] - c[0], a[:, 1] - c[1]) < 25.0)
        if not near.any(): continue
        d = rect_dist(c, float(yaw), h1, a[near, :2].astype(float), a[near, 2].astype(float), o["dims"][near] / 2); j = int(np.argmin(d))
        if d[j] < best[0]: best = (float(d[j]), float(t), o["ids"][np.flatnonzero(near)[j]])
    return best


def label(o, actor): return o["actors"][actor][2] if actor in o["actors"] else ""


def standing_since(o):
    """For the 50 ms controller-state rows of the run (pre-handover rows included: there the state is the simulator-reported replayed
    recording): seconds the car has continuously been below 0.5 m/s up to and including each row. -> (t_us, speed, standing_s)."""
    e = o["ego"]; v = np.hypot(e[:, 4], e[:, 5]); t = e[:, 0]; st = np.zeros(len(t)); start = None
    for i in range(len(t)):
        if v[i] < V_STAND:
            if start is None: start = t[i]
            st[i] = (t[i] - start) / 1e6
        else: start = None; st[i] = -1.0
    return t, v, st


def plan9(P): return np.r_[np.zeros((1, 3)), np.asarray(P, float)]


def classify(o, tn, ego, P):
    """One (8,3) plan in the car frame at tick tn -> dict of the t04 classes plus clearance."""
    p9 = plan9(P); arc = np.r_[0, np.cumsum(np.hypot(*np.diff(p9[:, :2], axis=0).T))]; w = C.to_world(T.dense(p9), ego)
    f0, t0, a0, hz = T.box_check(o, tn, w, T.TS, 0.0); f3, t3, a3, _ = T.box_check(o, tn, w, T.TS, 0.3); cl, ct, ca = clearance(o, tn, w, T.TS)
    return dict(leave=bool(np.hypot(*p9[-1, :2]) >= 2.0), prompt=bool(arc[2] >= 0.5 and arc[4] >= 1.5), box_free=f0, box_free_margin0p3=f3, first_overlap_s=t0, first_overlap_actor=a0, first_overlap_label=label(o, a0),
                first_overlap_margin0p3_s=t3, first_overlap_margin0p3_actor=a3, first_overlap_margin0p3_label=label(o, a3), box_checked_horizon_s=hz, min_clearance_m=cl, min_clearance_t_s=ct, min_clearance_actor=ca,
                min_clearance_label=label(o, ca), arc_1s_m=arc[2], arc_2s_m=arc[4], arc_4s_m=arc[8], end_x_fwd_m=p9[-1, 0], end_y_left_m=p9[-1, 1])
