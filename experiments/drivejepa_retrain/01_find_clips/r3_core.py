"""Run 3 detector core, shared by the NuRec (clipgt, world frame) and raw (offline labels, rig frame) loaders.
Scene S: t, xy, yaw, speed, acc on a 10 Hz grid; tracks {tid: t, xy, yaw, v, L, W, cat} in the ego world frame.
Ego box B: F (rig origin -> front bumper), C (rig origin -> box centre), L, W."""
import math, numpy as np, pandas as pd
DT = 0.1; ANCH = np.round(np.arange(0.5, 16.0 + 1e-9, 0.5), 1); HIST, FUT = 0.5, 4.0
def yaw_q(x, y, z, w): return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
def smooth(a, k=5): return np.convolve(np.pad(a, k//2, mode="edge"), np.ones(k)/k, mode="valid")
def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def build_ego(ts_s, xy, yaw):
    o = np.argsort(ts_s); tt = ts_s[o] - ts_s[o][0]; xy = xy[o]; yaw = np.unwrap(yaw[o])
    g = np.round(np.arange(0, tt[-1] + 1e-6, DT), 1)
    X = np.c_[np.interp(g, tt, xy[:, 0]), np.interp(g, tt, xy[:, 1])]; Y = np.interp(g, tt, yaw)
    sp = np.linalg.norm(np.gradient(X, DT, axis=0), axis=1)
    return dict(t=g, xy=X, yaw=Y, speed=sp, acc=np.gradient(smooth(sp), DT), dur=float(tt[-1]), t0=float(ts_s[o][0]))

def build_tracks(S, ob):
    """ob: DataFrame tid, ts (s, same clock as ego grid), x, y, yaw (world), L, W, cat."""
    g = S["t"]; T = {}
    for tid, r in ob.sort_values("ts").groupby("tid", sort=False):
        a = r["ts"].to_numpy(); keep = np.r_[True, np.diff(a) > 1e-4]; a = a[keep]
        if len(a) < 5 or a[-1] - a[0] < 1.0: continue
        gi = g[(g >= a[0]) & (g <= a[-1])]
        if len(gi) < 11: continue
        P = np.c_[np.interp(gi, a, r["x"].to_numpy()[keep]), np.interp(gi, a, r["y"].to_numpy()[keep])]
        yw = np.interp(gi, a, np.unwrap(r["yaw"].to_numpy()[keep]))
        v = np.zeros(len(gi)); v[5:-5] = np.linalg.norm(P[10:] - P[:-10], axis=1); v[:5] = v[5]; v[-5:] = v[-6]
        mdir = np.zeros(len(gi)); d = P[10:] - P[:-10]; mdir[5:-5] = np.arctan2(d[:, 1], d[:, 0]); mdir[:5] = mdir[5]; mdir[-5:] = mdir[-6]
        T[tid] = dict(t=gi, xy=P, yaw=yw, v=v, mdir=mdir, L=float(r["L"].median()), W=float(r["W"].median()), cat=r["cat"].iloc[0])
    S["tracks"] = T; return S

def ego_frame(p, o, yaw):
    d = np.asarray(p) - o; c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c*d[..., 0] + s*d[..., 1], -s*d[..., 0] + c*d[..., 1]], -1)
def idx_at(tr, tq):
    j = int(round((tq - tr["t"][0]) / DT)); return j if 0 <= j < len(tr["t"]) else None
def dist_poly(p, P):
    A, B = P[:-1], P[1:]; AB = B - A; AP = p - A
    u = np.clip((AP * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(A + u[:, None]*AB - p, axis=1)))
def rect(c, yaw, L, W):
    ca, sa = math.cos(yaw), math.sin(yaw); R = np.array([[ca, -sa], [sa, ca]])
    return np.asarray(c) + np.array([[L/2, W/2], [L/2, -W/2], [-L/2, -W/2], [-L/2, W/2]]) @ R.T
def rect_dist(P, Q):
    """0 if the convex quads overlap (SAT), else the minimum distance between them."""
    sep = False
    for A in (P, Q):
        for i in range(4):
            e = A[(i + 1) % 4] - A[i]; n = np.array([-e[1], e[0]]); a, b = P @ n, Q @ n
            if a.max() < b.min() or b.max() < a.min(): sep = True; break
        if sep: break
    if not sep: return 0.0
    best = 1e9
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            for i in range(4):
                s0, s1 = B[i], B[(i + 1) % 4]; d = s1 - s0; u = np.clip(np.dot(p - s0, d) / max(np.dot(d, d), 1e-9), 0, 1)
                best = min(best, float(np.linalg.norm(s0 + u*d - p)))
    return best

# ---------------- D-A (Run 2 definition) with the 1c heading filter ----------------
def anchors(S, uid):
    sp, n = S["speed"], len(S["t"]); out = []
    for ta in ANCH:
        i = int(round(ta / DT)); i0, i1 = i - 5, i + 40
        if i1 >= n: break
        wmin, wmax = float(sp[i0:i1 + 1].min()), float(sp[i0:i1 + 1].max())
        row = dict(uuid=uid, t=float(ta), v=float(sp[i]), wmin=wmin, wmax=wmax, A=wmin >= 16.0)
        if row["A"]:
            path = S["xy"][i:i1 + 1]; o, yw = S["xy"][i], S["yaw"][i]
            row["brake"] = float(S["acc"][i:i1 + 1].min()) < -1.0; row["curve"] = abs(math.degrees(S["yaw"][i1] - S["yaw"][i])) > 10.0
            f_old = f_new = f_mot = False
            for tr in S["tracks"].values():
                j = idx_at(tr, ta)
                if j is None or tr["v"][j] < 0.5: continue
                e = ego_frame(tr["xy"][j], o, yw)
                if e[0] > 0 and e[0] - tr["L"]/2 <= 50.0 and dist_poly(tr["xy"][j], path) <= 2.0:
                    f_old = True
                    if abs(wrap(tr["yaw"][j] - yw)) <= math.radians(30): f_new = True
                    if abs(wrap(tr["mdir"][j] - yw)) <= math.radians(30): f_mot = True
            row.update(follow_old=f_old, follow=f_new, follow_motion=f_mot)
            row["cruise_old"] = not (row["brake"] or f_old or row["curve"]); row["cruise"] = not (row["brake"] or f_new or row["curve"])
        out.append(row)
    return out

# ---------------- move-offs: ego-only events, then actor attachment ----------------
def moveoffs(S, uid):
    sp, t, n = S["speed"], S["t"], len(S["t"]); ev = []; still = sp < 0.3; i = 0
    while i < n:
        if not still[i]: i += 1; continue
        j = i
        while j < n and still[j]: j += 1
        if (j - i) * DT >= 2.0 and j < n:
            k = next((k for k in range(j, min(n, j + 61)) if sp[k] > 2.0), None)
            if k is not None:
                o, yw = S["xy"][j - 1], S["yaw"][j - 1]; b = min(n - 1, j - 1 + 40); e = ego_frame(S["xy"][j - 1:b + 1], o, yw)
                m = int(np.abs(e[:, 1]).argmax())
                ev.append(dict(uuid=uid, t_stop_start=float(t[i]), from_clip_start=bool(i == 0), t_move=float(t[j - 1]), t_depart=float(t[k]), lat4=float(abs(e[m, 1])), lat4_signed=float(e[m, 1]),
                               span4=float((b - j + 1) * DT), dyaw4_deg=float(math.degrees(S["yaw"][b] - S["yaw"][j - 1])), _i=i, _j=j, _k=k))
        i = j
    return ev

def attach_actor(S, e, F):
    """Nearest stationary actor (speed < 0.5 m/s from stop start to depart + 4 s) within ±2 m laterally and 0-8 m ahead.
    rig_*: gap from the rig origin (Run 2); bum_*: gap from the front bumper (Run 3, 1b)."""
    t, n = S["t"], len(S["t"]); i, j, k = e["_i"], e["_j"], e["_k"]; o, yw = S["xy"][j - 1], S["yaw"][j - 1]
    lo, hi = t[i], t[min(n - 1, k + 40)]; best = {"rig": None, "bum": None}
    for tr in S["tracks"].values():
        q = idx_at(tr, t[j - 1])
        if q is None: continue
        m = (tr["t"] >= lo - 1e-6) & (tr["t"] <= hi + 1e-6)
        if tr["v"][m].max() >= 0.5: continue
        p = ego_frame(tr["xy"][q], o, yw)
        if abs(p[1]) > 2.0: continue
        for key, off in (("rig", 0.0), ("bum", F)):
            gap = p[0] - tr["L"]/2 - off
            if 0.0 <= gap <= 8.0 and (best[key] is None or gap < best[key][0]): best[key] = (float(gap), float(p[1]), tr["cat"])
    for key in ("rig", "bum"):
        b = best[key]; e[f"actor_{key}"] = b is not None; e[f"gap_{key}"] = b[0] if b else None; e[f"alat_{key}"] = b[1] if b else None; e[f"acat_{key}"] = b[2] if b else None
    return e

# ---------------- Part 2: around a still obstacle ----------------
def around(S, uid, B):
    F, C, Le, We = B["F"], B["C"], B["L"], B["W"]; hw = We/2 + 0.3; n = len(S["t"]); seen = {}
    stat = {tid: tr for tid, tr in S["tracks"].items() if tr["v"].min() < 0.5}
    for ta in ANCH:
        i = int(round(ta / DT)); i0, i1 = i - 5, i + 40
        if i1 >= n: break
        o, yw = S["xy"][i], S["yaw"][i]
        for tid, tr in stat.items():
            if tid in seen: continue
            j = idx_at(tr, ta)
            if j is None: continue
            m = (tr["t"] >= S["t"][i0] - 1e-6) & (tr["t"] <= S["t"][i1] + 1e-6)
            if tr["v"][m].max() >= 0.5: continue
            Q = rect(tr["xy"][j], tr["yaw"][j], tr["L"], tr["W"]); q = ego_frame(Q, o, yw)
            near = q[:, 0].min() - F
            if not (2.0 <= near <= 25.0) or q[:, 1].min() > hw or q[:, 1].max() < -hw: continue
            ax = float(ego_frame(tr["xy"][j], o, yw)[0])
            cyaw = S["yaw"][i:i1 + 1]; cen = S["xy"][i:i1 + 1] + C * np.c_[np.cos(cyaw), np.sin(cyaw)]; ce = ego_frame(cen, o, yw)
            past = np.nonzero(ce[:, 0] > ax)[0]
            if not len(past): continue
            kp = int(past[0]); dmin = 1e9
            for s in range(0, len(cen)):
                if abs(ce[s, 0] - ax) > Le/2 + tr["L"]/2 + tr["W"]/2 + 3.0: continue      # far steps cannot be the minimum
                dmin = min(dmin, rect_dist(rect(cen[s], cyaw[s], Le, We), Q))
            if dmin <= 0.0: seen[tid] = None; continue                                     # box overlap: not a clean pass (counted separately)
            seen[tid] = dict(uuid=uid, track=tid, t=float(ta), ego_speed=float(S["speed"][i]), clearance=float(dmin), cat=tr["cat"], near_gap=float(near),
                             actor_lat=float(ego_frame(tr["xy"][j], o, yw)[1]), path_lat_at_pass=float(ce[kp, 1]), t_pass=float(ta + kp * DT),
                             dyaw_at_pass_deg=float(math.degrees(cyaw[kp] - cyaw[0])), actor_L=tr["L"], actor_W=tr["W"])
    ev = [v for v in seen.values() if v is not None]
    return ev, sum(1 for v in seen.values() if v is None)
