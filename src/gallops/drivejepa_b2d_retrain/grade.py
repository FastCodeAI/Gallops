"""Step A grade (specs/phase2_v2.md in H16's folder). One path over 4 s:
  S = P x 0.5^n_ped x 0.6^n_veh x 0.65^n_static x
0.7^n_red x 0.8^n_stop x (1 - outside).
Geometry, the follower model and the context come from scorer.py (paddock-ops mtd/scorer.py, unchanged).
New here, all ported from Bench2Drive's atomic_criteria.py:
  collisions   CollisionTest: counted by type, one per actor (MAX_ID_TIME 5 s > the window), none within
               COLLISION_RADIUS 5 m of the last one, none while the ego is under EPSILON 0.1 m/s.
               Scorer boxes and margins as they are. Followers: the scorer's counted followers are vehicle
               infractions.
  red          the scorer's port, counted per light (stop segments sharing a trigger centre are one light).
  stop         RunningStopTest over the 40 steps, starting from the logged state at t
               (build_context.stop_replay). Stopped = 0.5 s segment speed under 1.0 m/s, see SPEED_T.
  outside      OutsideRouteLanesTest: nearest of Driving and Parking, then distance > lane half width + 0.5;
               or the scorer's oncoming rule. Share of path distance, not time. 0 for a path that doesn't
               move.
No expert mask on collisions or lanes: the expert is graded by the same rules (the scorer's own expert masks
inside the follower rule stay).
v2 (spec v2): P = min(1, d / d_expert), P = 1 when the expert moves under 0.5 m: pass the expert's distance
to grade(). outside = metres off lanes or oncoming / route length (the expert's distance over the clip): set
g['route_len'].
v1: d_ref from the frame's clean feasible paths, outside a share of the 4 s path.
scripts/phase2/grade2_v1_copy.py in H16's folder is the v1 file."""

import numpy as np
import torch

from gallops.drivejepa_b2d_retrain import scorer as SC

K = SC.K
COEF = dict(ped=0.5, veh=0.6, static=0.65, red=0.7, stop=0.8)
COL_RADIUS, COL_EPS, OUT_TOL = 5.0, 0.1, 0.5
PROX, STEP, NW, MULT, BACK = 4.0, 0.5, 9, 1.2, -0.17
# Bench2Drive's 0.1 m/s is an instantaneous speed; a path only has 0.5 s average speeds. Measured
# (probe_stop.py): on 9,698 stop-clip frames the expert's 8 poses give 289 failures at 0.1, 11 at 1.0, its
# 10 Hz log 12
SPEED_T = 1.0
XK = (
    "park_pts",
    "park_hw",
    "park_valid",
    "ss_c",
    "ss_ext",
    "ss_valid",
    "ss_ax",
    "ss_ay",
    "ss_target",
    "ss_done",
    "ss_lastfail",
)
OFF_REAR = 1.3886


def to_centre_t(p):
    """rear-axle poses (..., 3) -> actor centre, in the frame whose origin is the centre at t. As
    grade.to_centre."""
    q = p.clone()
    q[..., 0] = p[..., 0] + OFF_REAR * torch.cos(p[..., 2]) - OFF_REAR
    q[..., 1] = p[..., 1] + OFF_REAR * torch.sin(p[..., 2])
    return q


def load_clip(ctx_dir, clip, frame_ids=None):
    c = SC.load_ctx(f"{ctx_dir}/{clip}.npz", frame_ids)
    z = np.load(f"{ctx_dir}/{clip}.x.npz")
    pos = {str(f): i for i, f in enumerate(z["frame_id"])}
    idx = np.array([pos[str(f)] for f in c["frame_id"]], np.int64)
    for k in XK:
        c[k] = z[k][idx]
    return c


def collate2(ctxs, device="cuda"):
    out = SC.collate(ctxs, device)
    for k in XK:
        a = [np.asarray(c[k]) for c in ctxs]
        if a[0].ndim > 1:
            shp = tuple(max(x.shape[i] for x in a) for i in range(1, a[0].ndim))
            pad = []
            for x in a:
                b = np.zeros((x.shape[0],) + shp, x.dtype)
                b[(slice(None),) + tuple(slice(0, s) for s in x.shape[1:])] = x
                pad.append(b)
            a = pad
        v = np.concatenate(a)
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        if v.dtype == np.int16:
            v = v.astype(np.int64)
        out[k] = torch.from_numpy(v).to(device)
    return out


def prepare2(ctx, device="cuda"):
    g = SC.prepare(ctx, device)
    g["lane_p"] = ctx["lane_pts"].to(device).float()
    cls = ctx["agent_cls"].to(device).long()
    g["ctype"] = torch.where(
        cls == 1, torch.zeros_like(cls), torch.where(cls == 3, torch.full_like(cls, 2), torch.ones_like(cls))
    )  # 0 pedestrian, 1 vehicle (incl. cyclist), 2 static
    pp = ctx["park_pts"].to(device).float()
    g.update(
        park_p=pp,
        park_hw=ctx["park_hw"].to(device).float(),
        park_v=ctx["park_valid"].to(device).bool(),
        any_park=bool(ctx["park_valid"].any()),
    )
    g.update(
        ss_c=ctx["ss_c"].to(device).float(),
        ss_ext=ctx["ss_ext"].to(device).float(),
        ss_v=ctx["ss_valid"].to(device).bool(),
        ss_ax=ctx["ss_ax"].to(device).float(),
        ss_ay=ctx["ss_ay"].to(device).float(),
        ss_t0=ctx["ss_target"].to(device).long(),
        ss_d0=ctx["ss_done"].to(device).bool(),
        ss_f0=ctx["ss_lastfail"].to(device).long(),
    )
    # stop segments of one light share its trigger centre
    if g["st_c"] is not None and g["st_c"].shape[1]:
        same = (g["st_c"][:, :, None] - g["st_c"][:, None]).norm(dim=-1) < 1e-3
        g["st_same"] = same
        g["st_first"] = ~torch.tril(same, -1).any(-1)
    return g


def sub(g, i):
    """g restricted to frames i (LongTensor): only per-frame tensors."""
    o = {}
    for k, v in g.items():
        o[k] = v[i] if torch.is_tensor(v) and v.dim() and v.shape[0] == g["F"] else v
    o["F"] = len(i)
    return o


def count_collisions(g, hit, spd, xy, first=None):
    """hit (F, C, K, A), spd (F, C, K), xy (F, C, K, 2) -> (F, C, 3) by type.
    CollisionTest._count_collisions, on the candidates that hit anything."""
    F, C, _, A = hit.shape
    n = torch.zeros(F, C, 3, device=hit.device, dtype=torch.long)
    idx = hit.flatten(2).any(-1).nonzero()
    if not len(idx):
        return n
    f, c = idx[:, 0], idx[:, 1]
    h = hit[f, c] & (spd[f, c] >= COL_EPS)[..., None]
    p = xy[f, c]
    ty = g["ctype"][f]  # (M, K, A), (M, K, 2), (M, A)
    M = len(f)
    last_id = torch.full((M,), -1, device=hit.device)
    has = torch.zeros(M, dtype=torch.bool, device=hit.device)
    loc = torch.zeros(M, 2, device=hit.device)
    ar = torch.arange(A, device=hit.device)
    cnt = torch.zeros(M, 3, device=hit.device, dtype=torch.long)
    row = torch.arange(M, device=hit.device)
    fa = torch.full((M,), -1, device=hit.device)
    fk = torch.zeros(M, dtype=torch.long, device=hit.device)
    for k in range(K):
        has = has & ((p[:, k] - loc).norm(dim=-1) <= COL_RADIUS)
        hk = h[:, k] & (ar[None] != last_id[:, None])
        ev = hk.any(-1) & ~has
        a = hk.float().argmax(-1)
        cnt[row[ev], ty[row[ev], a[ev]]] += 1
        new = ev & (fa < 0)
        fa = torch.where(new, a, fa)
        fk = torch.where(new, torch.full_like(fk, k), fk)
        last_id = torch.where(ev, a, last_id)
        loc = torch.where(ev[:, None], p[:, k], loc)
        has = has | ev
    n[f, c] = cnt
    if first is not None:
        first[0][f, c] = fa
        first[1][f, c] = fk
    return n


def lanes2(g, xy, yaw, budget=6e7):
    """-> off, onc (F, C, K) and the nearest Driving lane point index + along-lane offset, for the stop
    test."""
    F, C = xy.shape[:2]
    L = g["lane_t"].shape[1]
    N = C * K
    dev = xy.device
    off = torch.zeros(F, C, K, dtype=torch.bool, device=dev)
    onc = torch.zeros_like(off)
    near = torch.zeros(F, C, K, dtype=torch.long, device=dev)
    nlon = torch.zeros(F, C, K, device=dev)
    if L == 0:
        return off, onc, near, nlon
    fb = max(1, int(budget // max(1, N * L)))
    inf = torch.tensor(float("inf"), device=dev)
    for i in range(0, F, fb):
        j = min(F, i + fb)
        p = xy[i:j].reshape(j - i, N, 2)
        h = yaw[i:j].reshape(j - i, N)
        t, nrm = g["lane_t"][i:j], g["lane_n"][i:j]
        lon = torch.bmm(p, t.transpose(1, 2)) - g["lane_qt"][i:j, None]
        lat = torch.bmm(p, nrm.transpose(1, 2)) - g["lane_qn"][i:j, None]
        d = torch.sqrt(lat * lat + (lon.abs() - SC.LANE_SEG_HALF).clamp(min=0) ** 2)
        hw = g["lane_hw"][i:j, None]
        dD = torch.where(g["lane_valid"][i:j, None], d, inf)
        dmin, ix = dD.min(-1)
        hwD = g["lane_hw"][i:j].gather(1, ix)
        dist, wid = dmin, hwD
        if g["any_park"]:
            q = ((p[:, :, None] - g["park_p"][i:j, None]) ** 2).sum(-1).sqrt()
            q = torch.where(g["park_v"][i:j, None], q, inf)
            dP, ip = q.min(-1)
            hwP = g["park_hw"][i:j].gather(1, ip)
            useP = dmin >= dP
            dist = torch.where(useP, dP, dmin)
            wid = torch.where(useP, hwP, hwD)
        inside = (d <= hw + SC.INSIDE_TOL) & (d < SC.INSIDE_DMAX)
        cosd = torch.bmm(torch.stack([torch.cos(h), torch.sin(h)], -1), t.transpose(1, 2))
        same = (inside & (cosd >= 0)).any(-1)
        opp = (inside & (cosd < 0)).any(-1)
        judged = p.norm(dim=-1) < (g["lane_r"][i:j, None] - SC.LANE_EDGE)
        off[i:j] = ((dist > wid + OUT_TOL) & judged).reshape(j - i, C, K)
        onc[i:j] = (opp & ~same & judged).reshape(j - i, C, K)
        near[i:j] = ix.reshape(j - i, C, K)
        nlon[i:j] = lon.gather(-1, ix[..., None])[..., 0].reshape(j - i, C, K)
        del lon, lat, d, inside, cosd, dD
    return off, onc, near, nlon


def red_count(g, xy, yaw):
    F, C = xy.shape[:2]
    S = g["st_q1"].shape[1]
    dev = xy.device
    if S == 0:
        return torch.zeros(F, C, dtype=torch.long, device=dev)
    P = torch.cat([torch.zeros(F, C, 1, 2, device=dev), xy], 2)
    H = torch.cat([torch.zeros(F, C, 1, device=dev), yaw], 2)
    f = torch.stack([torch.cos(H), torch.sin(H)], -1)
    tc = (P - 0.8 * SC.HL * f)[:, :, :, None]
    tf = (P - (SC.HL + 1.0) * f)[:, :, 1:, None]
    ex = lambda x: x[:, None, None]  # noqa: E731
    fire = (
        SC._cross(tc[:, :, :-1], tc[:, :, 1:], ex(g["st_q1"]), ex(g["st_q2"]))
        & (((tf - ex(g["st_wp"])) * ex(g["st_n"])).sum(-1).abs() <= 0.5 * ex(g["st_w"]))
        & ((f[:, :, 1:, None] * ex(g["st_d"])).sum(-1) > 0)
        & g["st_red"][:, :, 1 : K + 1].transpose(1, 2)[:, None]
    )
    if g["st_c"] is not None:
        fire = fire & ((P[:, :, 1:, None] - ex(g["st_c"])).norm(dim=-1) <= SC.RED_NEAR)
    seg = fire.any(2)  # (F, C, S)
    if g.get("st_same") is None:
        return seg.long().sum(-1)
    return (
        (torch.einsum("fcs,fts->fct", seg.float(), g["st_same"].float()) > 0)
        .logical_and(g["st_first"][:, None])
        .long()
        .sum(-1)
    )


def stop_count(g, xy, yaw, vel, near, nlon):
    """RunningStopTest.update over the path; only frames with a stop sign are simulated. -> (F, C)."""
    F, C = xy.shape[:2]
    dev = xy.device
    out = torch.zeros(F, C, dtype=torch.long, device=dev)
    fi = g["ss_v"].any(-1).nonzero()[:, 0]
    if not len(fi):
        return out
    c, ext, sv, ax, ay = (
        g["ss_c"][fi, None],
        g["ss_ext"][fi, None],
        g["ss_v"][fi, None],
        g["ss_ax"][fi, None, None],
        g["ss_ay"][fi, None, None],
    )
    n = len(fi)
    tgt = g["ss_t0"][fi, None].expand(n, C).clone()
    done = g["ss_d0"][fi, None].expand(n, C).clone()
    lastf = g["ss_f0"][fi, None].expand(n, C).clone()
    cnt = torch.zeros(n, C, dtype=torch.long, device=dev)
    lp, lt = g["lane_p"][fi], g["lane_t"][fi]
    ar = torch.arange(n, device=dev)[:, None]
    spd = vel[fi].norm(dim=-1)
    done = done | ((tgt >= 0) & (g["v0"][fi, None] < 0.1))  # the next tick after t, in the log
    for k in range(K):
        ix = near[fi, :, k]
        t = lt[ar, ix]
        wp0 = lp[ar, ix] + nlon[fi, :, k, None].clamp(-0.5, 0.5) * t  # (n, C, 2)
        h = yaw[fi, :, k]
        f = torch.stack([torch.cos(h), torch.sin(h)], -1)
        close = (c - wp0[:, :, None]).norm(dim=-1) <= PROX
        ins = torch.zeros_like(close)
        for j in range(NW):
            rel = (wp0 + STEP * j * t)[:, :, None] - c
            ins |= ((rel * ax).sum(-1).abs() < MULT * ext[..., 0]) & (
                (rel * ay).sum(-1).abs() < MULT * ext[..., 1]
            )
        aff = close & ins & sv
        has = tgt >= 0
        ok = ((vel[fi, :, k] * f).sum(-1) >= BACK) & ((f * t).sum(-1) >= BACK)
        new = ~has & ok & aff.any(-1)
        first = aff.float().argmax(-1)
        donen = done | (has & (spd[:, :, k] < SPEED_T))
        at = aff.gather(-1, tgt.clamp(min=0)[..., None])[..., 0]
        leave = has & ~at
        fail = leave & ~donen & (lastf != tgt)
        cnt += fail.long()
        lastf = torch.where(fail, tgt, lastf)
        tgt = torch.where(leave, torch.full_like(tgt, -1), tgt)
        done = torch.where(leave, torch.zeros_like(donen), donen)
        tgt = torch.where(new, first, tgt)
    out[fi] = cnt
    return out


def along(g, cand):
    """metres gained along the expert's 4 s path, capped at the expert's own. As scorer._progress without
    the ratio."""
    E = g["expert"][:, :, :2].double()
    cand = cand.double()
    F, C = cand.shape[:2]
    dev = cand.device
    path = torch.cat([torch.zeros(F, 1, 2, device=dev, dtype=E.dtype), E], 1)
    v = path[:, 1:] - path[:, :-1]
    L2 = (v * v).sum(-1)
    seg = L2.sqrt()
    cum = torch.cat([torch.zeros(F, 1, device=dev, dtype=E.dtype), torch.cumsum(seg, -1)], -1)
    p = cand[:, :, -1, :2]
    rel = p[:, :, None] - path[:, None, :-1]
    tt = torch.where(
        L2[:, None] < 1e-12,
        torch.zeros_like(rel[..., 0]),
        ((rel * v[:, None]).sum(-1) / L2[:, None].clamp(min=1e-12)).clamp(0, 1),
    )
    q = path[:, None, :-1] + tt[..., None] * v[:, None]
    best = (p[:, :, None] - q).norm(dim=-1).argmin(-1, keepdim=True)
    return torch.minimum(
        (cum[:, None, :-1] + tt * seg[:, None]).gather(-1, best)[..., 0], cum[:, -1, None]
    ).float()


def _terms(g, cand, steps=False):
    xy, yaw, vel = SC.densify(cand)
    spd = vel.norm(dim=-1)
    first = (
        (
            torch.full(cand.shape[:2], -1, device=cand.device, dtype=torch.long),
            torch.zeros(cand.shape[:2], device=cand.device, dtype=torch.long),
        )
        if steps
        else None
    )
    hit = (
        SC.overlap(*SC._cand_box(xy, yaw), *SC._agent_box(g, 1))
        & g["valid"][:, :, 1:].transpose(1, 2)[:, None]
    )
    soft, fs = SC._followers(g, xy, yaw)
    hitn = hit & ~fs["active"]
    del hit
    counted = (soft & ~g["exp_soft"][:, None]) | fs["hard"]
    n3 = count_collisions(g, hitn, spd, xy, first)
    n3[..., 1] += (counted & ~hitn.any(2)).long().sum(-1)
    del hitn, fs, soft, counted
    off, onc, near, nlon = lanes2(g, xy, yaw)
    bad = off | (onc & ~g["blocked"][:, None, None])
    seg = (xy - torch.cat([torch.zeros_like(xy[:, :, :1]), xy[:, :, :-1]], 2)).norm(dim=-1)
    tot = seg.sum(-1)
    outside = torch.where(tot > 1e-6, (seg * bad).sum(-1) / tot.clamp(min=1e-6), torch.zeros_like(tot))
    route = g.get(
        "route_len"
    )  # v2: a share of the route, as OutsideRouteLanesTest. Without it: v1's share of the 4 s path
    share = (
        (lambda x: ((seg * x).sum(-1) / route[:, None].clamp(min=1e-3)).clamp(max=1.0))
        if route is not None
        else (
            lambda x: torch.where(tot > 1e-6, (seg * x).sum(-1) / tot.clamp(min=1e-6), torch.zeros_like(tot))
        )
    )
    if route is not None:
        outside = share(bad)
    o = dict(
        n_ped=n3[..., 0],
        n_veh=n3[..., 1],
        n_static=n3[..., 2],
        n_red=red_count(g, xy, yaw),
        n_stop=stop_count(g, xy, yaw, vel, near, nlon),
        outside=outside,
        off_share=share(off),
        d=along(g, cand),
        bad_m=(seg * bad).sum(-1),
        path_m=tot,
    )
    if steps:
        onb = onc & ~g["blocked"][:, None, None]
        o.update(onc_share=share(onb), off_step=off, onc_step=onb, first_agent=first[0], first_step=first[1])
    return o


@torch.no_grad()
def terms(g, cand, max_elems=4e7, steps=False):
    """cand (F, C, 8, 3), actor centre frame -> dict of (F, C); steps adds per-step areas and the first
    agent hit."""
    cand = torch.as_tensor(cand, dtype=torch.float32, device=g["expert"].device)
    F, C = cand.shape[:2]
    cc = max(1, int(max_elems // max(1, F * K * max(g["A"], 1))))
    parts = [_terms(g, cand[:, i : i + cc], steps) for i in range(0, C, cc)]
    return {k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}


def penalty(t):
    return (
        COEF["ped"] ** t["n_ped"].float()
        * COEF["veh"] ** t["n_veh"].float()
        * COEF["static"] ** t["n_static"].float()
        * COEF["red"] ** t["n_red"].float()
        * COEF["stop"] ** t["n_stop"].float()
        * (1 - t["outside"])
    )


def clean(t):
    return (t["n_ped"] + t["n_veh"] + t["n_static"] + t["n_red"] + t["n_stop"] == 0) & (t["outside"] == 0)


def grade(t, d_ref):
    """t: terms (F, C); d_ref (F,) -> S (F, C)."""
    P = torch.where(
        d_ref[:, None] < 0.5,
        torch.ones_like(t["d"]),
        (t["d"] / d_ref[:, None].clamp(min=1e-6)).clamp(max=1.0),
    )
    return P * penalty(t), P


def d_ref_of(t, valid):
    """valid (F, C): feasible, not padding."""
    return torch.where(clean(t) & valid, t["d"], torch.zeros_like(t["d"])).amax(1)
