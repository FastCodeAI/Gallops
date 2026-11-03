#!/usr/bin/env python
"""T04 shared: cached run reader, oriented-box overlap, menu classification. See README.md for the definitions."""
import glob, math, pathlib, pickle, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part2_why_it_swerves"))
import t02_common as C  # noqa: E402

STUCK = ["d2715023", "b44a4a44"]; MOVE = ["5f064b02", "e3ed554f"]; FOUR = STUCK + MOVE; VERS = ["0.1.6", "0.1.4b"]
TS = np.round(np.arange(0, 4.0001, 0.1), 1)      # box-check sample times (s)
T8 = np.arange(0, 4.0001, 0.5)


def scenes(): return sorted(p.name[5:] for p in (C.BASE / "analysis_3.0/Retrain-data/07_closed_loop_test/menus").glob("d016_*") if p.is_dir())


def load(ver, s):
    o = pickle.load(open(HERE / "cache" / f"{ver}_{s}.pkl", "rb")); ids = [a for a in o["actors"] if a != "EGO"]; n = len(o["ap_t"])
    A = np.full((n, len(ids), 3), np.nan)
    for i, d in enumerate(o["ap"]):
        for j, a in enumerate(ids):
            if a in d: A[i, j] = d[a]
    o["ids"] = ids; o["A"] = A; o["dims"] = np.array([o["actors"][a][:2] for a in ids]); o["labels"] = [o["actors"][a][2] for a in ids]
    o["ego_dims"] = np.array(o["actors"]["EGO"][:2]); g = o["gt"]; tg = (g[:, 0] - o["ho"]) / 1e6
    v = np.hypot(np.gradient(g[:, 1], tg), np.gradient(g[:, 2], tg)); o["gt_t"] = tg; o["gt_v"] = v; o["egot"] = {int(r[0]): r for r in o["ego"]}
    return o


def menus(ver, s):
    out = {}
    for f in sorted(glob.glob(str(C.MENUS[ver](s) / "replan_*.npz"))):
        z = np.load(f, allow_pickle=True); out[int(z["tn"])] = {k: z[k] for k in z.files}
    return dict(sorted(out.items()))


def gt_at(o, t_us):
    g = o["gt"]; t_us = np.asarray(t_us, dtype=float)
    return np.c_[np.interp(t_us, g[:, 0], g[:, 1]), np.interp(t_us, g[:, 0], g[:, 2]), np.interp(t_us, g[:, 0], np.unwrap(g[:, 3]))]


def human_start(o):
    """First time (s after handover, searched from the first ground-truth sample at about -3.15 s) the human's speed exceeds 0.5 m/s."""
    k = np.argmax(o["gt_v"] > 0.5); return float(o["gt_t"][k]) if (o["gt_v"] > 0.5).any() else np.nan


def dense(plan9):
    """(9, 3) rig poses 0.5 s apart -> (41, 3) at 0.1 s, linear in x, y and (unwrapped) yaw."""
    return np.c_[np.interp(TS, T8, plan9[:, 0]), np.interp(TS, T8, plan9[:, 1]), np.interp(TS, T8, np.unwrap(plan9[:, 2]))]


def sat(c1, yaw1, h1, c2, yaw2, h2):
    """Oriented rectangles overlap. c1 (M,2), yaw1 (M,), h1 (2,) half sizes; c2 (N,2), yaw2 (N,), h2 (N,2). -> (M,N) bool."""
    u1 = np.stack([np.c_[np.cos(yaw1), np.sin(yaw1)], np.c_[-np.sin(yaw1), np.cos(yaw1)]], 1)      # (M,2,2)
    u2 = np.stack([np.c_[np.cos(yaw2), np.sin(yaw2)], np.c_[-np.sin(yaw2), np.cos(yaw2)]], 1)      # (N,2,2)
    d = c2[None] - c1[:, None]; dots = np.abs(np.einsum("mik,njk->mnij", u1, u2)); sep = np.zeros(d.shape[:2], bool)
    for i in range(2):
        sep |= np.abs(np.einsum("mnk,mk->mn", d, u1[:, i])) > h1[i] + dots[:, :, i, 0] * h2[None, :, 0] + dots[:, :, i, 1] * h2[None, :, 1]
        sep |= np.abs(np.einsum("mnk,nk->mn", d, u2[:, i])) > h2[None, :, i] + dots[:, :, 0, i] * h1[0] + dots[:, :, 1, i] * h1[1]
    return ~sep


def box_check(o, t0_us, world_rig, times_s, margin=0.0):
    """world_rig (K,3) rig-origin poses (local frame) at t0 + times_s. The ego box (run's EGO aabb, centred rig_to_aabb_x ahead of the rig
    origin, grown by `margin` on every side) against every recorded actor box at the same time. Times beyond the recorded actor log are not
    checked. -> (free, first_overlap_s, actor_id, checked_horizon_s)."""
    ap_t = o["ap_t"]; h1 = o["ego_dims"] / 2 + margin; last = np.nan
    for k, t in enumerate(times_s):
        tq = t0_us + int(round(t * 1e6)); i = int(np.searchsorted(ap_t, tq)); i = min(i, len(ap_t) - 1)
        if abs(int(ap_t[i]) - tq) > 1000:
            if i > 0 and abs(int(ap_t[i - 1]) - tq) <= 1000: i -= 1
            else: continue
        a = o["A"][i]; ok = np.isfinite(a[:, 0]); x, y, yaw = world_rig[k]; c = np.array([[x + o["rig_to_aabb_x"] * math.cos(yaw), y + o["rig_to_aabb_x"] * math.sin(yaw)]])
        near = ok & (np.hypot(a[:, 0] - c[0, 0], a[:, 1] - c[0, 1]) < 12.0); last = float(t)
        if near.any():
            hit = sat(c, np.array([yaw]), h1, a[near, :2].astype(float), a[near, 2].astype(float), o["dims"][near] / 2)[0]
            if hit.any(): return False, float(t), o["ids"][np.flatnonzero(near)[np.argmax(hit)]], last
    return True, np.nan, "", last


def ranks(score):
    r = np.empty(len(score), int); r[np.argsort(-np.asarray(score, float), kind="stable")] = np.arange(1, len(score) + 1); return r
