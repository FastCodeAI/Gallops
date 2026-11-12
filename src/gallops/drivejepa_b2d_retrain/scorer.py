"""MTD Stage-2 rule scorer (SPEC.md, 5 Oct). Torch on CUDA, vectorised over frames x candidates x 40 steps x
agents.
Source-agnostic: works from the context-file arrays only; the per-frame mask (F, 7) decides which rules apply.

    from scorer import load_ctx, collate, score
    ctx = collate([load_ctx(path, frame_ids)], device='cuda')        # dict of padded tensors, F frames
    out = score(ctx, cand, device='cuda')                            # cand (F, C, 8, 3) poses at 0.5 .. 4.0
    s, ego frame
    out['total'], out['collision_reactive'], ... (F, C); out['mask'] (F, 7)

Definitions (v6 labeller = b2d-labels v5 per-step rules + v6 oncoming/blocked + v4 red; v3 progress/comfort):
  densify    8 poses + origin -> 40 steps of 0.1 s, linear in x, y, unwrapped yaw (label_v5.interp); step
             velocity = the
             velocity of the 0.5 s segment containing the step.
  collision  ego footprint (half 2.446 x 0.918 m at the step pose) overlaps a valid agent box at the same
             step (separating
             axes). Static margin 0.3 m on every side for static objects (cls 3, incl. poles), agent_static,
             and vehicles /
             cyclists below 0.1 m/s at that step (agent_speed if present, else finite differences).
             Non-reactive (log replay).
  collision_reactive
             as collision, but a moving vehicle / cyclist that is a FOLLOWER of the candidate (behind it:
             centre behind the
             candidate's centre, in line with it: |lateral| < min(1.75 m, half widths), heading within 45
             deg, bumper gap 0-40 m) is replaced by an IDM driver on
             the candidate's path: it keeps its logged speed for 0.5 s after the candidate becomes its
             leader, then follows IDM
             (a 1.5, b 2.0 m/s^2, s0 2 m, T 1.0 s, v_des = its max logged speed, decel capped at 9 m/s^2)
             until it leaves the
             candidate's lane. Its log-replay hits are excused; instead the candidate collides with it if
             the simulated gap
             reaches 0, or, once the follower reacts, it needs > 4 m/s^2 braking to stay behind (closing
             speed^2 / 2 gap) or
             its time gap (gap / its speed) is below 1 s -- for a follower already behind the ego at t = 0
             only when below its
             starting time gap too (a logged tail-gater is not the candidate's doing). The braking /
             time-gap violations are
             expert-masked per follower (they do not count when the expert's own trajectory produces one
             with that follower).
             follower_brake / follower_tgap / follower_gap = max needed braking, min time gap, min gap over
             the followers.
  ttc        the footprint pushed 0.5 s and 1.0 s along the step velocity overlaps an agent (same boxes and
             margins);
             masked at steps where the expert's pushed footprint (10 Hz log velocity) overlaps one (label_v5).
  lane       the centre more than 0.5 m beyond the nearest Driving lane's half width at some step (distance
             to the lane
             points taken as short segments along the lane direction). Poses beyond lane_radius - 3 m are
             not judged.
  direction  oncoming at some step: the centre inside (within half width + 0.125 m, < 2.0 m) a lane whose
             direction differs
             by > 90 deg from the step heading and inside no lane within 90 deg of it (v6 onc rule, junction
             connectors
             included); never in a frame marked blocked (v6 own-lane-blocked exception).
  red        Bench2Drive RunningRedLightTest (label_red_v4): the rear point 0.8 x half-length behind the
             centre crosses a stop
             segment between two consecutive 0.1 s points while that light is red at that step, heading the
             lane's way, the
             far tail (half-length + 1 m behind) within the lane's half width of the stop waypoint, and the
             centre within
             15 m of the light's trigger-volume centre (stop_center, if given). stop_seg = (left end, right
             end) w.r.t. travel.
  progress   label_candidates_v3: the candidate's last pose projected on the expert's polyline (origin +
             expert poses), past
             its end -> + distance to the expert's end; ratio (min(d, d_exp) + 0.01) / (max(d, d_exp) +
             0.01), so the
             candidate's progress counts at most up to the expert's own and overshooting is penalised
             symmetrically.
  comfort    label_candidates_v3: every |acceleration| (0.5 s finite differences) and |yaw rate| (steps
             moving > 0.3 m)
             within the expert's own maxima in the frame.
  total      collision_reactive * lane * direction * red * (5 ttc + 5 progress + 2 comfort) / 12, masked
             rules = 1.
horizon (poses, default 8 = 4 s) shortens every rule to the first `horizon` poses (the v4 labels used 6 = 3
s)."""

import math

import numpy as np
import torch

HL, HW = 2.446, 0.918
K = 40
DT = 0.1
MARGIN = 0.3
TTC_DT = (0.5, 1.0)
OFFROAD_TOL = 0.5
INSIDE_TOL, INSIDE_DMAX = 0.125, 2.0
LANE_SEG_HALF = (
    0.5  # each lane point stands for a lane piece +-0.5 m along its direction (points 0.5-1 m apart)
)
LANE_EDGE = 3.0  # poses within 3 m of the lane-coverage radius are not judged by lane / direction
RED_NEAR = 15.0
F_LAT, F_COS, F_RANGE, REACT = 1.75, math.cos(math.radians(45)), 40.0, 5
BRAKE_MAX, TGAP_MIN = 4.0, 1.0
IDM_A, IDM_B, IDM_S0, IDM_T, IDM_AMIN = 1.5, 2.0, 2.0, 1.0, -9.0
RULES = ("collision", "ttc", "lane", "direction", "red", "progress", "comfort")
REQUIRED = (
    "frame_id",
    "v0",
    "expert",
    "expert_ok",
    "agent_xyyaw",
    "agent_valid",
    "agent_lw",
    "agent_cls",
    "agent_static",
    "lane_pts",
    "lane_hw",
    "lane_yaw",
    "lane_valid",
    "stop_seg",
    "stop_red",
    "stop_valid",
    "mask",
    "blocked",
)
OPTIONAL = ("agent_speed", "expert40", "lane_radius", "stop_center")


# -----------------------------------------------------------------------------------------------------------
# loading
def load_ctx(path, frame_ids=None):
    """one context npz -> dict of numpy arrays (all frames, or the given frame_ids in that order)."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k in REQUIRED + OPTIONAL}
    if frame_ids is not None:
        pos = {str(f): i for i, f in enumerate(d["frame_id"])}
        idx = np.array([pos[str(f)] for f in frame_ids], np.int64)
        d = {k: v[idx] for k, v in d.items()}
    return d


def collate(ctxs, device="cuda"):
    """list of context dicts (each with a leading F axis) -> one dict of torch tensors padded to the largest
    A, L, S.
    Optional fields missing from any input are dropped (the scorer then uses its fallbacks)."""
    keys = [k for k in REQUIRED + OPTIONAL + ("agent_id",) if all(k in c for c in ctxs)]
    out = {"frame_id": np.concatenate([np.asarray(c["frame_id"]).astype(str) for c in ctxs])}
    for k in keys:
        if k == "frame_id":
            continue
        arrs = [np.asarray(c[k]) for c in ctxs]
        if arrs[0].ndim > 1:
            shp = tuple(max(a.shape[i] for a in arrs) for i in range(1, arrs[0].ndim))
            fill = (
                -1
                if k in ("agent_cls", "agent_id") and arrs[0].dtype.kind in "iu"
                else (np.zeros((), arrs[0].dtype) if arrs[0].dtype.kind in "USO" else 0)
            )
            pad = []
            for a in arrs:
                b = np.full((a.shape[0],) + shp, fill, a.dtype)
                b[(slice(None),) + tuple(slice(0, s) for s in a.shape[1:])] = a
                pad.append(b)
            arrs = pad
        a = np.concatenate(arrs)
        if a.dtype.kind in "USO":
            out[k] = a
            continue  # string ids stay numpy
        if a.dtype == np.float16 or a.dtype == np.float64:
            a = a.astype(np.float32)
        if a.dtype == np.int8:
            a = a.astype(np.int64)
        out[k] = torch.from_numpy(a).to(device)
    return out


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


# -----------------------------------------------------------------------------------------------------------
# geometry
def densify(P8):
    """(..., 8, 3) poses at 0.5 s -> xy (..., 40, 2), yaw (..., 40), segment velocity (..., 40, 2)."""
    shp = P8.shape[:-2]
    dev = P8.device
    P = torch.cat([torch.zeros(*shp, 1, 3, device=dev, dtype=P8.dtype), P8], -2)
    yaw = torch.cat(
        [P[..., :1, 2], P[..., :1, 2] + torch.cumsum(_wrap(torch.diff(P[..., 2], dim=-1)), -1)], -1
    )
    k = torch.arange(1, K + 1, device=dev)
    s = (k - 1) // 5 + 1
    frac = (k - 5 * (s - 1)).to(P8.dtype) / 5.0
    P0, P1 = P[..., s - 1, :2], P[..., s, :2]
    xy = P0 + frac[:, None] * (P1 - P0)
    yw = yaw[..., s - 1] + frac * (yaw[..., s] - yaw[..., s - 1])
    return xy, yw, (P1 - P0) / 0.5


def overlap(cx, cy, cc, cs, hl, hw, ax, ay, ac, as_, ahl, ahw):
    """oriented rectangles (centre, cos/sin yaw, half length/width), broadcastable -> bool overlap
    (separating axes)."""
    dx, dy = ax - cx, ay - cy
    cd = (ac * cc + as_ * cs).abs()
    sd = (as_ * cc - ac * cs).abs()
    sep = (dx * cc + dy * cs).abs() > hl + ahl * cd + ahw * sd
    sep |= (dy * cc - dx * cs).abs() > hw + ahl * sd + ahw * cd
    sep |= (dx * ac + dy * as_).abs() > ahl + hl * cd + hw * sd
    sep |= (dy * ac - dx * as_).abs() > ahw + hl * sd + hw * cd
    return ~sep


def prepare(ctx, device="cuda"):
    """per-frame geometry shared by every candidate of the frame (agents, inflation, lanes, stop lines, the
    expert's own
    TTC mask and follower results). Cache and reuse it when scoring several candidate chunks of the same
    frames."""
    g = {}
    f32 = lambda k: ctx[k].to(device).float()  # noqa: E731
    xyz = f32("agent_xyyaw")
    valid = ctx["agent_valid"].to(device).bool()
    lw = f32("agent_lw")
    cls = ctx["agent_cls"].to(device).long()
    static = ctx["agent_static"].to(device).bool()
    F, A = valid.shape[:2]
    if "agent_speed" in ctx:
        spd = f32("agent_speed")
    else:
        d = torch.zeros_like(xyz[..., 0])
        dd = (xyz[:, :, 2:, :2] - xyz[:, :, :-2, :2]).norm(dim=-1) / (2 * DT)
        d[:, :, 1:-1] = dd
        d[:, :, 0] = (xyz[:, :, 1, :2] - xyz[:, :, 0, :2]).norm(dim=-1) / DT
        d[:, :, -1] = (xyz[:, :, -1, :2] - xyz[:, :, -2, :2]).norm(dim=-1) / DT
        ok = valid[:, :, 2:] & valid[:, :, :-2]
        d[:, :, 1:-1] = torch.where(ok, dd, torch.full_like(dd, 1.0))
        spd = torch.where(valid, d, torch.zeros_like(d))
    veh = (cls == 0) | (cls == 2)
    infl = (cls == 3)[..., None] | static[..., None] | (veh[..., None] & (spd < 0.1))  # (F, A, 41)
    g.update(
        F=F,
        A=A,
        ax=xyz[..., 0],
        ay=xyz[..., 1],
        ac=torch.cos(xyz[..., 2]),
        as_=torch.sin(xyz[..., 2]),
        valid=valid,
        ahl=lw[..., 0, None] / 2 + MARGIN * infl,
        ahw=lw[..., 1, None] / 2 + MARGIN * infl,
        alen=lw[..., 0],
        awid=lw[..., 1],
        spd=spd,
        fol=veh & ~static & (cls >= 0),
        vdes=torch.clamp(torch.where(valid, spd, torch.zeros_like(spd)).amax(-1), min=1.0),
    )
    g["v0"] = f32("v0")
    g["mask"] = ctx["mask"].to(device).bool()
    g["blocked"] = ctx["blocked"].to(device).bool()
    g["expert"] = f32("expert")
    g["expert_ok"] = ctx["expert_ok"].to(device).bool()
    # lanes: point, direction, half width, constants for the matmul form of along / across distances
    lp = f32("lane_pts")
    ly = f32("lane_yaw")
    lv = ctx["lane_valid"].to(device).bool()
    t = torch.stack([torch.cos(ly), torch.sin(ly)], -1)
    g.update(
        lane_t=t,
        lane_n=torch.stack([t[..., 1], -t[..., 0]], -1),
        lane_qt=(lp * t).sum(-1),
        lane_qn=lp[..., 0] * t[..., 1] - lp[..., 1] * t[..., 0],
        lane_hw=torch.where(lv, f32("lane_hw"), torch.full_like(ly, -1e4)),
        lane_valid=lv,
        lane_r=f32("lane_radius") if "lane_radius" in ctx else torch.full((F,), 60.0, device=device),
    )
    # stop lines
    ss = f32("stop_seg")
    q1, q2 = ss[:, :, 0], ss[:, :, 1]
    nv = q1 - q2
    wl = nv.norm(dim=-1).clamp(min=1e-6)
    nv = nv / wl[..., None]
    g.update(
        st_q1=q1,
        st_q2=q2,
        st_wp=(q1 + q2) / 2,
        st_n=nv,
        st_d=torch.stack([nv[..., 1], -nv[..., 0]], -1),
        st_w=wl / 1.2,
        st_red=ctx["stop_red"].to(device).bool() & ctx["stop_valid"].to(device).bool()[..., None],
        st_c=f32("stop_center") if "stop_center" in ctx else None,
    )
    # the expert as an internal candidate. TTC mask = label_v5's (10 Hz log, 0.5 s backward velocity) OR the
    # densified
    # 8-pose expert pushed the way candidates are (so a candidate equal to the expert is always masked);
    # follower mask from
    # the densified expert (exactly what a candidate equal to the expert produces).
    dxy, dyaw, dvel = densify(g["expert"])
    kk = torch.arange(1, K + 1, device=device)
    if "expert40" in ctx:
        e40 = f32("expert40")
        exy, eyaw = e40[..., :2], e40[..., 2]
        P = torch.cat([torch.zeros(F, 1, 2, device=device), exy], 1)
        ev = torch.where(
            (kk >= 6)[None, :, None],
            (P[:, kk] - P[:, (kk - 5).clamp(min=0)]) / 0.5,
            P[:, kk] / (DT * kk)[None, :, None],
        )
        pushes = [(exy, eyaw, ev), (dxy, dyaw, dvel)]
    else:
        pushes = [(dxy, dyaw, dvel)]
    eh = torch.zeros(F, K, dtype=torch.bool, device=device)
    av = g["valid"][:, None, :, 1:].transpose(-1, -2)
    for exy_, eyaw_, ev_ in pushes:
        for dt in TTC_DT:
            o = overlap(
                *_cand_box(exy_[:, None] + ev_[:, None] * dt, eyaw_[:, None]), *_agent_box(g, 1)
            )  # (F, 1, K, A)
            eh |= (o & av).any(-1)[:, 0]
    g["ttc_expert"] = eh
    soft, _ = _followers(g, dxy[:, None], dyaw[:, None], need_steps=False)
    g["exp_soft"] = soft[:, 0]  # (F, A)
    # the densified expert's own per-step hits / offroad / oncoming, for the optional expert masks
    g["exp_hit"] = (overlap(*_cand_box(dxy[:, None], dyaw[:, None]), *_agent_box(g, 1)) & av)[
        :, 0
    ]  # (F, K, A)
    eo, en = _lanes(g, dxy[:, None], dyaw[:, None])
    g["exp_off"], g["exp_onc"] = eo[:, 0], en[:, 0]
    return g


def _cand_box(xy, yaw):
    """candidate centres (F, C, K, 2), yaw (F, C, K) -> broadcast against agents on a trailing A axis."""
    return xy[..., 0, None], xy[..., 1, None], torch.cos(yaw)[..., None], torch.sin(yaw)[..., None], HL, HW


def _agent_box(g, k0):
    """agents at steps k0 .. k0+K-1 -> (F, 1, K, A) broadcastable."""
    sl = lambda x: x[:, :, k0 : k0 + K].transpose(1, 2)[:, None]  # noqa: E731
    return sl(g["ax"]), sl(g["ay"]), sl(g["ac"]), sl(g["as_"]), sl(g["ahl"]), sl(g["ahw"])


def _followers(g, xy, yaw, need_steps=True):
    """IDM followers on the candidate path. xy (F, C, K, 2), yaw (F, C, K) -> soft (F, C, A) braking /
    time-gap
    violations, plus (if need_steps) hard (F, C, A) gap <= 0, active (F, C, K, A) follower steps and stats."""
    F, C = xy.shape[:2]
    A = g["A"]
    dev = xy.device
    P = torch.cat([torch.zeros(F, C, 1, 2, device=dev), xy], 2)
    H = torch.cat([torch.zeros(F, C, 1, device=dev), yaw], 2)  # (F, C, 41)
    s = torch.cat(
        [torch.zeros(F, C, 1, device=dev), torch.cumsum((P[:, :, 1:] - P[:, :, :-1]).norm(dim=-1), -1)], -1
    )
    vl = torch.cat(
        [g["v0"].abs()[:, None, None].expand(F, C, 1), (s[:, :, 1:] - s[:, :, :-1]) / DT], -1
    )  # lead speed at each step
    z = lambda v=0.0, dt=torch.float32: torch.full((F, C, A), v, device=dev, dtype=dt)  # noqa: E731
    started, active, soft, hard = z(dt=torch.bool), z(dt=torch.bool), z(dt=torch.bool), z(dt=torch.bool)
    k0, xf, vf, tg0 = z(), z(), z(), z()
    brake, mintg, mingap = z(), z(99.0), z(99.0)
    act_steps = torch.zeros(F, C, K, A, dtype=torch.bool, device=dev) if need_steps else None
    fol = g["fol"][:, None]
    alen = g["alen"][:, None]
    vdes = g["vdes"][:, None]
    sq = 2 * math.sqrt(IDM_A * IDM_B)
    flat = torch.clamp(HW + g["awid"][:, None] / 2, max=F_LAT)  # in line with the candidate
    for k in range(K + 1):
        cx, cy, ch = P[:, :, k, 0, None], P[:, :, k, 1, None], H[:, :, k, None]
        c, sn = torch.cos(ch), torch.sin(ch)
        rx, ry = g["ax"][:, None, :, k] - cx, g["ay"][:, None, :, k] - cy
        lon, lat = rx * c + ry * sn, -rx * sn + ry * c
        cdh = g["ac"][:, None, :, k] * c + g["as_"][:, None, :, k] * sn
        rel = g["valid"][:, None, :, k] & fol & (lat.abs() < flat) & (cdh > F_COS)
        gapk = -lon - HL - alen / 2
        geo = rel & (lon < 0) & (gapk > 0) & (gapk < F_RANGE)
        sk = s[:, :, k, None]
        if k > 0:
            react = active & ((k - 1 - k0) >= REACT)
            gp = (s[:, :, k - 1, None] - HL - xf).clamp(min=0.1)
            dv = vf - vl[:, :, k - 1, None]
            sstar = IDM_S0 + (vf * IDM_T + vf * dv / sq).clamp(min=0)
            a = (IDM_A * (1 - (vf / vdes) ** 4 - (sstar / gp) ** 2)).clamp(min=IDM_AMIN)
            a = torch.where(react, a, torch.zeros_like(a))
            vn = (vf + a * DT).clamp(min=0)
            xf = torch.where(active, xf + 0.5 * (vf + vn) * DT, xf)
            vf = torch.where(active, vn, vf)
            active = active & rel
            gap = sk - HL - xf
            tg = gap / vf.clamp(min=0.1)
            drac = (vf - vl[:, :, k, None]).clamp(min=0) ** 2 / (
                2 * gap.clamp(min=0.1)
            )  # braking needed to stay behind
            hard |= active & (gap <= 0)
            soft |= (
                active
                & react
                & ((drac > BRAKE_MAX) | ((tg < TGAP_MIN) & ((tg < tg0) | (k0 > 0)) & (vf > 1.0)))
            )
            if need_steps:
                brake = torch.maximum(
                    brake, torch.where(active & react, drac.clamp(max=99.0), torch.zeros_like(drac))
                )
                mintg = torch.minimum(mintg, torch.where(active & (vf > 1.0), tg, torch.full_like(tg, 99.0)))
                mingap = torch.minimum(mingap, torch.where(active, gap, torch.full_like(gap, 99.0)))
        newly = ~started & geo
        xf = torch.where(newly, sk - HL - gapk, xf)
        vf = torch.where(newly, g["spd"][:, None, :, k], vf)
        k0 = torch.where(newly, torch.full_like(k0, float(k)), k0)
        tg0 = torch.where(newly, gapk / vf.clamp(min=0.1), tg0)
        started |= newly
        active |= newly
        if need_steps and k > 0:
            act_steps[:, :, k - 1] = active
    if not need_steps:
        return soft, None
    return soft, dict(hard=hard, active=act_steps, brake=brake, mintg=mintg, mingap=mingap)


# -----------------------------------------------------------------------------------------------------------
# scoring
@torch.no_grad()
def score(
    ctx,
    cand,
    device="cuda",
    horizon=8,
    reactive=True,
    steps=False,
    prepared=None,
    max_elems=4e7,
    expert_mask=(),
):
    """ctx: collated context (dict of tensors / arrays, F frames); cand (F, C, 8, 3). Returns a dict of (F,
    C) tensors
    (rules in [0, 1] with the mask applied, total, follower_brake / follower_tgap / follower_gap) + 'mask'
    (F, 7);
    steps=True adds the per-step flags (F, C, 40) col / colr / ttc_s / off / onc.
    expert_mask: rules among ('collision', 'lane', 'direction') whose violations do not count where the
    expert's own
    (densified) trajectory has the same one: a hit on the same agent at the same step, offroad / oncoming at
    the same step.
    Off by default (the v6 labeller has no expert mask on these rules); TTC is always expert-masked
    (label_v5)."""
    g = prepared if prepared is not None else prepare(ctx, device)
    cand = torch.as_tensor(cand, dtype=torch.float32, device=device)
    F, C = cand.shape[:2]
    cc = max(1, int(max_elems // max(1, F * K * max(g["A"], 1))))
    parts = [
        _score_chunk(g, cand[:, i : i + cc], horizon, reactive, steps, tuple(expert_mask))
        for i in range(0, C, cc)
    ]
    out = {k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}
    m = g["mask"]
    one = torch.ones(F, C, device=device)
    for j, r in enumerate(RULES):
        out[r] = torch.where(m[:, j, None], out[r], one)
    out["collision_reactive"] = torch.where(m[:, 0, None], out["collision_reactive"], one)
    out["total"] = (
        out["collision_reactive"]
        * out["lane"]
        * out["direction"]
        * out["red"]
        * (5 * out["ttc"] + 5 * out["progress"] + 2 * out["comfort"])
        / 12
    )
    out["mask"] = m
    return out


def _score_chunk(g, cand, H, reactive, steps, emask=()):
    F, C = cand.shape[:2]
    dev = cand.device
    Kh = 5 * H
    xy, yaw, vel = densify(cand)
    sm = torch.arange(K, device=dev) < Kh  # steps inside the horizon
    av = g["valid"][:, :, 1:].transpose(1, 2)[:, None]  # (F, 1, K, A)
    cb = _cand_box(xy, yaw)
    ab = _agent_box(g, 1)
    hit = overlap(*cb, *ab) & av  # (F, C, K, A)
    if "collision" in emask:
        hit = hit & ~g["exp_hit"][:, None]
    th = torch.zeros_like(hit)
    for dt in TTC_DT:
        th |= overlap(*_cand_box(xy + vel * dt, yaw), *ab)
    th &= av
    col_step = hit.any(-1) & sm
    ttc_step = th.any(-1) & ~g["ttc_expert"][:, None] & sm
    o = dict(collision=(~col_step.any(-1)).float(), ttc=(~ttc_step.any(-1)).float())
    # reactive
    if reactive:
        soft, fs = _followers(g, xy, yaw)
        act = fs["active"]
        colr_step = (hit & ~act).any(-1) & sm
        counted = (soft & ~g["exp_soft"][:, None]) | fs["hard"]  # (F, C, A)
        # note: the follower simulation always spans the full 4 s (horizon only shortens the per-step rules)
        o["collision_reactive"] = (~(colr_step.any(-1) | counted.any(-1))).float()
        cm = counted | (fs["active"].any(2) & ~g["exp_soft"][:, None])
        o["follower_brake"] = (
            torch.where(cm, fs["brake"], torch.zeros_like(fs["brake"])).amax(-1)
            if g["A"]
            else torch.zeros(F, C, device=dev)
        )
        o["follower_tgap"] = (
            torch.where(cm, fs["mintg"], torch.full_like(fs["mintg"], 99.0)).amin(-1)
            if g["A"]
            else torch.full((F, C), 99.0, device=dev)
        )
        o["follower_gap"] = (
            torch.where(cm, fs["mingap"], torch.full_like(fs["mingap"], 99.0)).amin(-1)
            if g["A"]
            else torch.full((F, C), 99.0, device=dev)
        )
    else:
        colr_step = col_step
        o["collision_reactive"] = o["collision"].clone()
        for k_, v_ in (("follower_brake", 0.0), ("follower_tgap", 99.0), ("follower_gap", 99.0)):
            o[k_] = torch.full((F, C), v_, device=dev)
    # lane + direction
    off_step, onc_step = _lanes(g, xy, yaw)
    off_step &= sm
    onc_step &= sm & ~g["blocked"][:, None, None]
    if "lane" in emask:
        off_step &= ~g["exp_off"][:, None]
    if "direction" in emask:
        onc_step &= ~g["exp_onc"][:, None]
    o["lane"] = (~off_step.any(-1)).float()
    o["direction"] = (~onc_step.any(-1)).float()
    o["red"] = _red(g, xy, yaw, Kh).float()
    o["progress"] = _progress(g, cand, H)
    o["comfort"] = _comfort(g, cand, H).float()
    if steps:
        o.update(col=col_step, colr=colr_step, ttc_s=ttc_step, off=off_step, onc=onc_step)
    return o


def _lanes(g, xy, yaw, budget=6e7):
    """offroad (F, C, K) and oncoming (F, C, K) per step, chunked over frames to bound the (poses x lane
    points) tensors."""
    F, C = xy.shape[:2]
    L = g["lane_t"].shape[1]
    N = C * K
    off = torch.zeros(F, C, K, dtype=torch.bool, device=xy.device)
    onc = torch.zeros_like(off)
    if L == 0:
        return off, onc
    fb = max(1, int(budget // max(1, N * L)))
    for i in range(0, F, fb):
        j = min(F, i + fb)
        p = xy[i:j].reshape(j - i, N, 2)
        h = yaw[i:j].reshape(j - i, N)
        t, nrm = g["lane_t"][i:j], g["lane_n"][i:j]
        lon = torch.bmm(p, t.transpose(1, 2)) - g["lane_qt"][i:j, None]
        lat = torch.bmm(p, nrm.transpose(1, 2)) - g["lane_qn"][i:j, None]
        d = torch.sqrt(lat * lat + (lon.abs() - LANE_SEG_HALF).clamp(min=0) ** 2)
        hw = g["lane_hw"][i:j, None]
        margin = (d - hw).amin(-1)  # (fb, N)
        inside = (d <= hw + INSIDE_TOL) & (d < INSIDE_DMAX)
        cosd = torch.bmm(torch.stack([torch.cos(h), torch.sin(h)], -1), t.transpose(1, 2))
        same = (inside & (cosd >= 0)).any(-1)
        opp = (inside & (cosd < 0)).any(-1)
        judged = p.norm(dim=-1) < (g["lane_r"][i:j, None] - LANE_EDGE)
        off[i:j] = ((margin > OFFROAD_TOL) & judged).reshape(j - i, C, K)
        onc[i:j] = (opp & ~same & judged).reshape(j - i, C, K)
        del lon, lat, d, inside, cosd
    return off, onc


def _cross(p1, p2, q1, q2):
    """segments p1-p2 vs q1-q2 (broadcastable (..., 2)) -> bool, as label_red_v4.cross."""

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
            c[..., 0] - a[..., 0]
        )

    d1, d2, d3, d4 = orient(q1, q2, p1), orient(q1, q2, p2), orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 <= 0) & (d3 * d4 <= 0) & ~((d1 == 0) & (d2 == 0))


def _red(g, xy, yaw, Kh):
    """True = no red-light infraction within the first Kh steps. (F, C)"""
    F, C = xy.shape[:2]
    S = g["st_q1"].shape[1]
    dev = xy.device
    if S == 0:
        return torch.ones(F, C, dtype=torch.bool, device=dev)
    P = torch.cat([torch.zeros(F, C, 1, 2, device=dev), xy[:, :, :Kh]], 2)
    H = torch.cat([torch.zeros(F, C, 1, device=dev), yaw[:, :, :Kh]], 2)
    f = torch.stack([torch.cos(H), torch.sin(H)], -1)
    tc = (P - 0.8 * HL * f)[:, :, :, None]
    tf = (P - (HL + 1.0) * f)[:, :, 1:, None]  # (F, C, Kh+1, 1, 2)
    ex = lambda x: x[:, None, None]  # noqa: E731   (F, S, ..) -> (F, 1, 1, S, ..)
    hit = _cross(tc[:, :, :-1], tc[:, :, 1:], ex(g["st_q1"]), ex(g["st_q2"]))  # (F, C, Kh, S)
    lat = ((tf - ex(g["st_wp"])) * ex(g["st_n"])).sum(-1).abs() <= 0.5 * ex(g["st_w"])
    same = (f[:, :, 1:, None] * ex(g["st_d"])).sum(-1) > 0
    red = g["st_red"][:, :, 1 : Kh + 1].transpose(1, 2)[:, None]  # (F, 1, Kh, S)
    fire = hit & lat & same & red
    if g["st_c"] is not None:
        fire &= (P[:, :, 1:, None] - ex(g["st_c"])).norm(dim=-1) <= RED_NEAR
    return ~fire.flatten(2).any(-1)


def _progress(g, cand, H):
    # float64: a stopped expert jitters by microns, and which end of that jitter is nearest decides the v3
    # ratio
    E = g["expert"][:, :H, :2].double()
    cand = cand.double()
    F, C = cand.shape[:2]
    path = torch.cat([torch.zeros(F, 1, 2, device=cand.device), E], 1)  # (F, H+1, 2)
    v = path[:, 1:] - path[:, :-1]
    L2 = (v * v).sum(-1)
    seg = L2.sqrt()
    cum = torch.cat([torch.zeros(F, 1, device=cand.device), torch.cumsum(seg, -1)], -1)
    dexp = cum[:, -1]
    p = cand[:, :, H - 1, :2]  # (F, C, 2)
    rel = p[:, :, None] - path[:, None, :-1]  # (F, C, H, 2)
    tt = torch.where(
        L2[:, None] < 1e-12,
        torch.zeros_like(rel[..., 0]),
        ((rel * v[:, None]).sum(-1) / L2[:, None].clamp(min=1e-12)).clamp(0, 1),
    )
    q = path[:, None, :-1] + tt[..., None] * v[:, None]
    dd = (p[:, :, None] - q).norm(dim=-1)
    best = dd.argmin(-1, keepdim=True)
    raw = (cum[:, None, :-1] + tt * seg[:, None]).gather(-1, best)[..., 0]
    endm = (raw - dexp[:, None]).abs() < 1e-6
    raw = torch.where(endm, raw + (p - path[:, None, -1]).norm(dim=-1), raw)
    return ((torch.minimum(raw, dexp[:, None]) + 0.01) / (torch.maximum(raw, dexp[:, None]) + 0.01)).float()


def _kin(A):
    """(..., H, 3) poses -> acc (..., H-1), yaw rate (..., H) as label_candidates (0.5 s steps)."""
    prev = torch.cat([torch.zeros_like(A[..., :1, :]), A[..., :-1, :]], -2)
    vel = (A[..., :2] - prev[..., :2]) / 0.5
    acc = (vel[..., 1:, :] - vel[..., :-1, :]).norm(dim=-1) / 0.5
    tr = _wrap(A[..., 2] - prev[..., 2]).abs() / 0.5
    tr = torch.where((A[..., :2] - prev[..., :2]).norm(dim=-1) > 0.3, tr, torch.zeros_like(tr))
    return acc, tr


def _comfort(g, cand, H):
    acc, tr = _kin(cand[:, :, :H])
    eacc, etr = _kin(g["expert"][:, :H])
    am = eacc.amax(-1)[:, None, None] if H > 1 else torch.zeros(cand.shape[0], 1, 1, device=cand.device)
    return (acc <= am + 1e-6).all(-1) & (tr <= etr.amax(-1)[:, None, None] + 1e-6).all(-1)
