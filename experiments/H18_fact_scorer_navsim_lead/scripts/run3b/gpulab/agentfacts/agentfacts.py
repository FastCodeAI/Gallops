"""gpulab/agentfacts: run 3B's road-user and comfort live labels as batched torch (float64, CUDA).

    extract_agents(kind, ctx) -> dict      CPU, numpy, pose-independent. kind 'nav' | 'carla'; ctx = labeller._nav_env() /
                                           _carla_env() context(meta). Road users from the CPU code itself (facts.tracks,
                                           facts.behind_in_lane0, run3a_labels.vehicle_tracks / in_ego_lane0), plus what the
                                           pose-dependent rules need: lanes holding each vehicle centre (cut-in lane gate),
                                           NAVSIM's object centroids / tree order / heading / stopped flag / intersection
                                           polygons (run 3A's key agents), the 1.5 s logged comfort history.
    label_agents(kind, scenes, states, coords, ego_areas) -> dict of cuda tensors, rows = the P paths:
        hit           (B, P, 8) bool     first counted contact (moving > 0.05 m/s, not excluded; CARLA: not debris), latched
        ttc           (B, P, 8) float16  facts.closeness min per step, inf = none (the CPU's float16 values, rounded exactly)
        cutin_brake   (B, P, 8) float32  facts.cutin per-step max, nan -> 0
        comfort_step  (B, P, 8) bool     NAVSIM: float16(comfort ratio per step) max over the 6 >= 1; CARLA: all False
        NAVSIM only: key_corners (B, P, 40, 2, 4, 2) float32, key_valid (B, P, 40, 2) bool  = run 3A's
                     after_score_proposals(...)[1:3] (at-fault collision agent, TTC agent; ego frame at t)
        _diag         only with diag=True: unrounded intermediates (ttc10 (B, P, 41), brake10, comfort10 (B, P, 41, 6), first
                      counted contact index, key-agent index / object, key_err = rows on which the CPU raises)
    states (B, P, 41, 11) float64 = cns.before_score_proposals rows 1..; coords (B, P, 41, 5, 2) = PDMScorer._ego_coords;
    ego_areas (B, P, 41, >= 2) bool = PDMScorer._ego_areas (EgoAreaIndex: 0 MULTIPLE_LANES, 1 NON_DRIVABLE_AREA).

CPU code restated (read, never edited): run3a/facts/facts.py (tracks, behind_in_lane0, contacts, time_to_contact,
closeness, cutin + idm_wanted, comfort_ratios, per_step, latch); run 3A's run3a_labels.py (vehicle_tracks, in_ego_lane0,
ego_path_lane_pairs), train_pdm_scorer.py (_calculate_no_at_fault_collision, _calculate_ttc), compute_navsim_score.py
(after_score_proposals), pdm_scorer_utils.get_collision_type, nuplan idm utils (is_agent_ahead / behind / is_track_stopped),
pdm_comfort_metrics (savgol-based values), BatchIDMPolicy.propagate; carla_full_v1/full.py apply_rules (debris).
Geometry: separating-axis tests on the boxes (touching counts, as shapely 'intersects'); even-odd ray casting for points in
polygons (strict interior, as shapely contains_xy / 'within'); a point exactly on an edge may go either way."""
import math
import sys

import numpy as np
import torch

F64 = torch.float64
K, NS, DT = 41, 8, 0.1
STEP_OF = np.r_[0, np.repeat(np.arange(NS), 5)]
STOPPED = 5e-2          # facts.STOPPED (NAVSIM get_collision_type's ego-stopped threshold)
TTC_CAP = 4.0           # facts.TTC_CAP
SC_STOPPED = 5e-3       # PDMScorerConfig.stopped_speed_threshold (scorer TTC)
FUT = (0, 3, 6, 9)      # PDMScorer._calculate_ttc future_time_idcs (1 s window, every 3rd index)
N_TTC_T = 40 - 9 + 1    # scorer TTC time indices 0 .. 31
AHEAD_RAD, BEHIND_RAD = float(np.deg2rad(30)), float(np.deg2rad(150))
CHUNK = 12_000_000      # pair elements per dense chunk (peak memory ~1-2 GB at B = 64)
SENT = 2 ** 62          # 'none' in int64 order codes
_WIN = np.stack([np.r_[np.arange(0, 6)]] + [np.r_[5 * k + 1, np.arange(5 * k + 1, 5 * k + 6)] for k in range(1, NS)])  # (8, 6)


# ============================================================================================================== extract (CPU)
def _mods():
    """the live fact modules of this process (NAVSIM or CARLA: carla_facts has already patched run 3A's car constants)."""
    import facts as F
    return F, F.r3


def _rings(geom):
    """segments (S, 4) [x1, y1, x2, y2] of all rings of a (multi)polygon."""
    out = []
    for g in getattr(geom, "geoms", [geom]):
        if g.is_empty:
            continue
        for r in [g.exterior, *g.interiors]:
            c = np.asarray(r.coords, np.float64)[:, :2]
            out.append(np.concatenate([c[:-1], c[1:]], 1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def extract_agents(kind, ctx, meta=None, aux=True):
    import shapely
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    F, r3 = _mods()
    mc = ctx["mc"]
    obs, dam, map_name = mc.observation, mc.drivable_area_map, mc.map_parameters.map_name
    s0 = ego_state_to_state_array(mc.ego_state)  # = every row's state 0 (PDMSimulator.simulate_proposals)
    toks, corners, present, vel, cls = F.tracks(obs)
    M = len(toks)
    coll = set(obs.collided_track_ids)
    init = np.array([t in coll for t in toks], bool)
    behind = (F.behind_in_lane0(s0[None, None], corners[:, 0].mean(1), present[:, 0], dam, map_name) if M
              else np.zeros(0, bool))
    out = dict(kind=kind, M=M, s0=s0, corners=corners, present=present, vel=vel, cls=cls,
               init=init, behind=behind,
               debris=np.array([t in ctx["deb"] for t in toks], bool) if kind == "carla" else np.zeros(M, bool),
               const=dict(r2c=float(r3.REAR_AXLE_TO_CENTER), half_length=float(r3.HALF_LENGTH), ego_rear=float(r3.EGO_REAR),
                          strip=float(r3.STRIP), same_dir=float(r3.SAME_DIR), brake=float(r3.BRAKE), dt=float(r3.DT),
                          lane_gate=bool(r3.REAR_LANE_GATE), stopped=float(F.STOPPED), ttc_cap=float(F.TTC_CAP)))
    # ---- cut-in: vehicle tracks, lane chain at t = 0, lane polygons holding each vehicle centre
    xyh, pv, length, speed = r3.vehicle_tracks(obs)
    Mv = len(length)
    lane0 = r3.in_ego_lane0(dam, map_name, s0[:3], xyh[:, 0, :2]) if Mv else np.zeros(0, bool)
    lane_ids = dam.get_indices_of_map_type(r3.LANES)
    hold = []  # (lane local id, mask (Mv, 41))
    segs, seg_off = [], [0]
    if Mv:
        px, py = xyh[..., 0].ravel(), xyh[..., 1].ravel()
        for i in lane_ids:
            g = dam._geometries[i]
            m = shapely.contains_xy(g, px, py).reshape(Mv, K) & pv
            if m.any():
                hold.append(m)
                s = _rings(g)
                segs.append(s)
                seg_off.append(seg_off[-1] + len(s))
    L = len(hold)
    nper = np.sum(hold, 0) if L else np.zeros((Mv, K), int)
    Lmax = max(1, int(nper.max()) if L else 1)
    vl = np.full((Mv, K, Lmax), -1, np.int32)
    if L:
        cnt = np.zeros((Mv, K), int)
        for l, m in enumerate(hold):
            mi, ki = np.where(m)
            vl[mi, ki, cnt[mi, ki]] = l
            cnt[mi, ki] += 1
    out.update(veh_xyh=xyh, veh_present=pv, veh_length=length, veh_speed=speed, veh_lane0=lane0, veh_lanes=vl,
               lane_segs=np.concatenate(segs) if segs else np.zeros((0, 4)), lane_seg_off=np.asarray(seg_off, np.int64))
    if kind != "nav":
        if aux:
            out.update(_extract_aux_carla(ctx, meta))                          # run 3A key agents on LEAD (lead_aux v2)
        return out
    # ---- NAVSIM: run 3A's key agents (scorer at-fault collision / TTC) and the comfort history
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.observation.idm.utils import is_track_stopped
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
    tix = {t: i for i, t in enumerate(toks)}
    cent = np.zeros((M, K, 2))
    rank = np.full((M, K), 2 ** 30, np.int32)
    giant = shapely.box(-1e12, -1e12, 1e12, 1e12)
    for k in range(K):
        om = obs[k]
        order = om._str_tree.query(giant)  # STRtree traversal order: the order the scorer's query loops meet the objects
        mm, gg = [], []
        for pos, j in enumerate(order.tolist()):
            t = om.tokens[j]
            if obs.red_light_token in t:
                continue
            m = tix[t]
            rank[m, k] = pos
            mm.append(m)
            gg.append(om._geometries[j])
        if mm:
            c = shapely.centroid(np.array(gg, dtype=object))
            cent[mm, k, 0], cent[mm, k, 1] = shapely.get_x(c), shapely.get_y(c)
    uo = obs.unique_objects
    inter = dam.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])
    isegs = [_rings(dam._geometries[i]) for i in inter]
    hp = mc.past_human_trajectory
    t0, t1 = hp.start_time.time_us, hp.end_time.time_us
    tus = np.clip(np.arange(t0, t1, int(0.1 * 1e6), dtype=np.int64), t0, t1)
    hist = ego_states_to_state_array(hp.get_state_at_times([TimePoint(t) for t in tus[:-1]])) if len(tus) > 1 else np.zeros((0, 11))
    ra = mc.ego_state.rear_axle
    out.update(centroid=cent, rank=rank,
               obj_heading=np.array([uo[t].box.center.heading for t in toks], np.float64),
               track_stopped=np.array([is_track_stopped(uo[t]) for t in toks], bool),
               init_order=np.array([tix[t] for t in obs.collided_track_ids if t in tix], np.int64),
               inter_segs=np.concatenate(isegs) if isegs else np.zeros((0, 4)),
               inter_poly=np.concatenate([np.full(len(s), j) for j, s in enumerate(isegs)]).astype(np.int64) if isegs else np.zeros(0, np.int64),
               origin=np.array([ra.x, ra.y, ra.heading], np.float64),
               hist=np.asarray(hist, np.float64), comfort_r2c=float(F.get_pacifica_parameters().rear_axle_to_center))
    return out


# ============================================================================================================== helpers (GPU)
_CONST = {}


def _consts(dev):
    """small index tables, uploaded once per device (a pageable upload mid-stream would wait for the queue)."""
    key = str(dev)
    if key not in _CONST:
        _CONST[key] = dict(win=torch.as_tensor(_WIN.reshape(-1), device=dev),
                           latch=torch.as_tensor(np.r_[STEP_OF, np.full(64, 99)], device=dev),
                           fut=torch.as_tensor(FUT, device=dev),
                           dts=torch.as_tensor([float(f) * 0.1 for f in FUT], device=dev, dtype=F64),
                           tf=torch.arange(N_TTC_T, device=dev)[:, None] + torch.as_tensor(FUT, device=dev)[None],
                           ns=torch.arange(NS, device=dev), t40=torch.arange(1, K, device=dev))
    return _CONST[key]


class _Upload:
    """every host array of one call in three flat buffers (float64, int64, bool): three host-to-device copies, all made
    before the first kernel; per-object arrays are concatenated over the samples and scattered to (B, M, ...) on the GPU."""

    def __init__(self):
        self.parts = {np.float64: [], np.int64: [], np.bool_: []}
        self.index = {}

    def put(self, name, a, dt):
        a = np.ascontiguousarray(a, dtype=dt)
        lst = self.parts[dt]
        off = sum(x.size for x in lst)
        lst.append(a.ravel())
        self.index[name] = (dt, off, a.shape)

    def go(self, dev):
        tdt = {np.float64: F64, np.int64: torch.int64, np.bool_: torch.bool}
        bufs = {}
        for dt, lst in self.parts.items():
            flat = np.concatenate(lst) if lst else np.zeros(0, dt)
            bufs[dt] = torch.from_numpy(flat).to(dev, tdt[dt])
        return {n: bufs[dt][off:off + int(np.prod(shp))].view(shp) for n, (dt, off, shp) in self.index.items()}


def _scatter_pad(cat, sid, lid, B, M, fill):
    out = torch.full((B, M) + tuple(cat.shape[1:]), fill, dtype=cat.dtype, device=cat.device)
    out[sid, lid] = cat
    return out


def f16_round(x):
    """float64 -> the nearest float16 value (round half to even, overflow -> inf), as numpy's astype(float16); returns
    float16. (A plain .half() of a double rounds through float32 and can double-round.)"""
    ax = x.abs()
    fin = torch.isfinite(x)
    axf = torch.where(fin, ax, torch.zeros_like(ax))
    _, e = torch.frexp(axf)
    q_exp = torch.clamp(e.to(torch.int64) - 11, min=-24)
    y = torch.round(axf * _pow2(-q_exp)) * _pow2(q_exp)                       # exact power-of-two scalings
    y = torch.where(y > 65504.0, torch.full_like(y, math.inf), y)
    y = torch.where(fin, torch.copysign(y, x), x)
    return y.to(torch.float32).to(torch.float16)


def _pow2(n):
    """2.0 ** n exactly (n int64 tensor, normal range), from the bits."""
    return ((n + 1023) << 52).view(F64)


def _wrap(a):
    """(a + pi) % (2 pi) - pi with numpy's float modulo (fmod, then + divisor where the sign differs)."""
    d = 2 * np.pi
    m = torch.fmod(a + np.pi, d)
    m = torch.where((m != 0) & (m < 0), m + d, m)
    return m - np.pi


def _per_step(a, how, win):
    """(..., 41) -> (..., 8) over facts.WIN (index 0 joins step 0)."""
    g = a[..., win].reshape(*a.shape[:-1], NS, 6)
    return g.amax(-1) if how == "max" else g.amin(-1)


def _sat_intersects(A, Bq):
    """A, Bq (..., 4, 2) convex quads -> (...,) bool: closed sets intersect (touching counts), all 8 edge normals."""
    ok = torch.ones(A.shape[:-2], dtype=torch.bool, device=A.device)
    for C in (A, Bq):
        for i in range(4):
            e = C[..., (i + 1) % 4, :] - C[..., i, :]
            a0, a1 = -e[..., 1:2], e[..., 0:1]
            pa = A[..., 0] * a0 + A[..., 1] * a1
            pb = Bq[..., 0] * a0 + Bq[..., 1] * a1
            ok &= (pb.amin(-1) <= pa.amax(-1)) & (pb.amax(-1) >= pa.amin(-1))
    return ok


def _seg_intersects_quad(s0, s1, Q):
    """segments s0 -> s1 (..., 2) and convex quads Q (..., 4, 2) -> (...,) bool, touching counts (axes: the segment's
    normal and the quad's edge normals)."""
    S = torch.stack([s0, s1], -2)
    ok = torch.ones(Q.shape[:-2], dtype=torch.bool, device=Q.device)
    axes = [torch.stack([-(s1 - s0)[..., 1], (s1 - s0)[..., 0]], -1)]
    for i in range(4):
        e = Q[..., (i + 1) % 4, :] - Q[..., i, :]
        axes.append(torch.stack([-e[..., 1], e[..., 0]], -1))
    for ax in axes:
        a0, a1 = ax[..., 0:1], ax[..., 1:2]
        pa = S[..., 0] * a0 + S[..., 1] * a1
        pb = Q[..., 0] * a0 + Q[..., 1] * a1
        ok &= (pb.amin(-1) <= pa.amax(-1)) & (pb.amax(-1) >= pa.amin(-1))
    return ok


def _time_to_contact(ego_c, ego_v, obj_c, obj_v, cap):
    """facts.time_to_contact, same operations: (Q, 4, 2), (Q, 2) -> (Q,) s (0 touching now, inf none within cap)."""
    rel = obj_v - ego_v
    lo = torch.full(rel.shape[:1], -math.inf, dtype=F64, device=rel.device)
    hi = torch.full_like(lo, math.inf)
    ninf, pinf = torch.full_like(lo, -math.inf), torch.full_like(lo, math.inf)
    for c in (ego_c, obj_c):
        e1, e2 = c[:, 1] - c[:, 0], c[:, 2] - c[:, 1]
        for e in (e1, e2):
            n0, n1 = -e[:, 1], e[:, 0]
            nn = torch.sqrt(n0 * n0 + n1 * n1).clamp_min(1e-9)
            a0, a1 = n0 / nn, n1 / nn
            pa = ego_c[..., 0] * a0[:, None] + ego_c[..., 1] * a1[:, None]
            pb = obj_c[..., 0] * a0[:, None] + obj_c[..., 1] * a1[:, None]
            amin, amax, bmin, bmax = pa.amin(1), pa.amax(1), pb.amin(1), pb.amax(1)
            w = rel[:, 0] * a0 + rel[:, 1] * a1
            still = w.abs() < 1e-9
            now = (bmin <= amax) & (bmax >= amin)
            t1, t2 = (amin - bmax) / w, (amax - bmin) / w
            enter, leave = torch.minimum(t1, t2), torch.maximum(t1, t2)
            enter = torch.where(still, torch.where(now, ninf, pinf), enter)
            leave = torch.where(still, torch.where(now, pinf, ninf), leave)
            lo, hi = torch.maximum(lo, enter), torch.minimum(hi, leave)
    hit = (lo <= hi) & (hi >= 0) & (lo <= cap)
    return torch.where(hit, lo.clamp_min(0.0), pinf)


def _crossings(px, py, seg):
    """points x segments [x1, y1, x2, y2] (broadcast) -> bool: the edge crosses the ray from the point to +x (even-odd)."""
    x1, y1, x2, y2 = seg[..., 0], seg[..., 1], seg[..., 2], seg[..., 3]
    up = (y1 > py) != (y2 > py)
    xi = (x2 - x1) * (py - y1) / torch.where(y2 == y1, torch.ones_like(y2), y2 - y1) + x1
    return up & (px < xi)


def _nonzero(m):
    return torch.nonzero(m, as_tuple=True)  # one host sync


# ============================================================================================================== label (GPU)
def _upload(kind, scenes, dev):
    nav = kind == "nav"
    B = len(scenes)
    U = _Upload()
    Ms = np.array([s["M"] for s in scenes])
    Mvs = np.array([len(s["veh_length"]) for s in scenes])
    sid = np.repeat(np.arange(B), Ms)
    lid = np.concatenate([np.arange(m) for m in Ms]) if Ms.sum() else np.zeros(0, np.int64)
    U.put("sid", sid, np.int64)
    U.put("lid", lid, np.int64)
    cat = lambda k: np.concatenate([s[k] for s in scenes])  # noqa: E731
    U.put("corners", cat("corners").reshape(-1, K, 4, 2), np.float64)
    U.put("present", cat("present").reshape(-1, K), np.bool_)
    U.put("vel", cat("vel").reshape(-1, K, 2), np.float64)
    U.put("init", cat("init"), np.bool_)
    U.put("behind", cat("behind"), np.bool_)
    U.put("debris", cat("debris"), np.bool_)
    U.put("s0", np.stack([s["s0"] for s in scenes]), np.float64)
    # vehicles (cut-in) and the lanes holding them (lane gate: global lane ids, local-frame segments)
    U.put("vsid", np.repeat(np.arange(B), Mvs), np.int64)
    U.put("vlid", np.concatenate([np.arange(m) for m in Mvs]) if Mvs.sum() else np.zeros(0, np.int64), np.int64)
    U.put("veh_xyh", cat("veh_xyh").reshape(-1, K, 3), np.float64)
    U.put("veh_present", cat("veh_present").reshape(-1, K), np.bool_)
    U.put("veh_length", cat("veh_length"), np.float64)
    U.put("veh_speed", cat("veh_speed").reshape(-1, K), np.float64)
    U.put("veh_lane0", cat("veh_lane0"), np.bool_)
    Lmax = max(s["veh_lanes"].shape[2] for s in scenes)
    vls, segs, starts, nl, base = [], [], [], 0, 0
    for s in scenes:
        v = s["veh_lanes"].astype(np.int64)
        v = np.where(v >= 0, v + nl, -1)
        vls.append(np.pad(v, ((0, 0), (0, 0), (0, Lmax - v.shape[2])), constant_values=-1))
        off = s["lane_seg_off"]
        segs.append(s["lane_segs"] - np.r_[s["s0"][:2], s["s0"][:2]])         # local frame (exact for UTM-sized values)
        starts.append(off[:-1] + base)
        base += int(off[-1])
        nl += len(off) - 1
    U.put("veh_lanes", np.concatenate(vls).reshape(-1, K, Lmax), np.int64)
    U.put("lane_segs", np.concatenate(segs).reshape(-1, 4), np.float64)
    st = np.concatenate(starts).astype(np.int64) if nl else np.zeros(0, np.int64)
    U.put("lane_start", st, np.int64)
    U.put("lane_end", np.r_[st[1:], base].astype(np.int64) if nl else np.zeros(0, np.int64), np.int64)
    if nav:
        U.put("centroid", cat("centroid").reshape(-1, K, 2), np.float64)
        U.put("rank", cat("rank").reshape(-1, K), np.int64)
        U.put("track_stopped", cat("track_stopped"), np.bool_)
        U.put("origin", np.stack([s["origin"] for s in scenes]), np.float64)
        E = max(1, max(len(s["inter_segs"]) for s in scenes))
        isg = np.full((B, E, 4), np.nan)
        ipo = np.full((B, E), 0, np.int64)
        NPm = 1
        for i, s in enumerate(scenes):
            n = len(s["inter_segs"])
            if n:
                isg[i, :n] = s["inter_segs"] - np.r_[s["s0"][:2], s["s0"][:2]]
                ipo[i, :n] = s["inter_poly"]
                NPm = max(NPm, int(s["inter_poly"].max()) + 1)
        U.put("inter_segs", isg, np.float64)
        U.put("inter_poly", ipo, np.int64)
        Hs = np.array([len(s["hist"]) for s in scenes])
        Hm = max(1, int(Hs.max()))
        hist = np.zeros((B, Hm, 11))
        for i, s in enumerate(scenes):
            hist[i, :len(s["hist"])] = s["hist"]
        U.put("hist", hist, np.float64)
        groups = [(int(H), np.where(Hs == H)[0]) for H in np.unique(Hs)]
        for H, idx in groups:
            U.put(f"hgroup{H}", idx, np.int64)
    g = U.go(dev)
    M, Mv = max(1, int(Ms.max())), max(1, int(Mvs.max()))
    T = dict(M=M, Mv=Mv, Lmax=Lmax, nl=nl, n_segs=base, s0=g["s0"], N=int(Ms.sum()))
    for k in ("sid", "lid", "corners", "present", "vel", "init", "behind", "debris"):
        T["f_" + k] = g[k]                                                     # flat over all objects of the batch
    for k in ("corners", "vel"):
        T[k] = _scatter_pad(g[k], g["sid"], g["lid"], B, M, 0.0)
    T["present"] = _scatter_pad(g["present"], g["sid"], g["lid"], B, M, False)
    for k in ("init", "behind", "debris"):
        T[k] = _scatter_pad(g[k], g["sid"], g["lid"], B, M, False)
    vs, vl = g["vsid"], g["vlid"]
    T["veh_xyh"] = _scatter_pad(g["veh_xyh"], vs, vl, B, Mv, 0.0)
    T["veh_present"] = _scatter_pad(g["veh_present"], vs, vl, B, Mv, False)
    T["veh_length"] = _scatter_pad(g["veh_length"], vs, vl, B, Mv, 0.0)
    T["veh_speed"] = _scatter_pad(g["veh_speed"], vs, vl, B, Mv, math.nan)
    T["veh_lane0"] = _scatter_pad(g["veh_lane0"], vs, vl, B, Mv, False)
    T["veh_lanes"] = _scatter_pad(g["veh_lanes"], vs, vl, B, Mv, -1)
    T["lane_segs"], T["lane_start"], T["lane_end"] = g["lane_segs"], g["lane_start"], g["lane_end"]
    if nav:
        T["centroid"] = _scatter_pad(g["centroid"], g["sid"], g["lid"], B, M, 0.0)
        T["rank"] = _scatter_pad(g["rank"], g["sid"], g["lid"], B, M, 2 ** 30)
        T["track_stopped"] = _scatter_pad(g["track_stopped"], g["sid"], g["lid"], B, M, True)
        T["origin"], T["inter_segs"], T["inter_poly"], T["NPm"] = g["origin"], g["inter_segs"], g["inter_poly"], NPm
        T["hist"] = g["hist"]
        T["hgroups"] = [(H, g[f"hgroup{H}"]) for H, _ in groups]
        T["comfort_r2c"] = scenes[0]["comfort_r2c"]
        T["init_first"] = [int(s["init_order"][0]) if len(s["init_order"]) else -1 for s in scenes]
    return T


@torch.no_grad()
def label_agents(kind, scenes, states, coords, ego_areas, diag=False):
    dev = states.device
    B, P = states.shape[:2]
    assert len(scenes) == B and all(s["kind"] == kind for s in scenes)
    C = scenes[0]["const"]
    assert all(s["const"] == C for s in scenes) and C["lane_gate"], "mixed or unexpected fact constants"
    if kind == "nav":
        assert len({s["comfort_r2c"] for s in scenes}) == 1
    cs = _consts(dev)
    T = _upload(kind, scenes, dev)                                             # all host-to-device copies, first
    TA = _upload_aux(scenes, dev) if kind == "carla" and "aux_M" in scenes[0] else None
    states, coords = states.to(F64), coords.to(F64)
    ego_areas = ego_areas.bool()
    M = T["M"]
    corners, present, init = T["corners"], T["present"], T["init"]
    excl = init | T["behind"]
    s0 = T["s0"]
    sp = torch.hypot(states[..., 3], states[..., 4])                           # (B, P, 41)
    hd = states[..., 2]
    ev = torch.stack([torch.cos(hd) * sp, torch.sin(hd) * sp], -1)            # (B, P, 41, 2)
    ego4 = coords[..., :4, :]
    ego4l = ego4 - s0[:, None, None, None, :2]                                # local frame (exact translation for UTM)
    corl = corners - s0[:, None, None, None, :2]                              # (B, M, 41, 4, 2)
    out, dg = {}, {}

    # ---------------------------------------------------------------- contacts (hit; NAVSIM key collision agent)
    # dense prefilters run over the batch's real objects (flat list j -> sample ob[j], slot om[j]), not the padded M
    ob, om, N = T["f_sid"], T["f_lid"], T["N"]
    fcl = T["f_corners"] - s0[ob][:, None, None, :2]                          # (N, 41, 4, 2) local frame
    fpres = T["f_present"]
    ecl = ego4l.mean(-2)                                                      # bounding circles (prefilter only)
    erad = (ego4l - ecl[..., None, :]).pow(2).sum(-1).amax(-1).sqrt()
    focl = fcl.mean(-2)
    forad = (fcl - focl[..., None, :]).pow(2).sum(-1).amax(-1).sqrt()          # (N, 41)
    t_hit = torch.full((B * P,), 99, dtype=torch.int64, device=dev)
    f_excl = T["f_init"] | T["f_behind"]
    f_excl_hit = f_excl | T["f_debris"]
    first_any = torch.full((B * P * M,), 99, dtype=torch.int64, device=dev)  # first contact per (path, object), any speed
    nc = max(1, CHUNK // (P * K))
    for j0 in range(0, N, nc):
        j1 = min(N, j0 + nc)
        bj = ob[j0:j1]
        dx = ecl[bj, :, :, 0] - focl[j0:j1, None, :, 0]                        # (n, P, 41)
        dy = ecl[bj, :, :, 1] - focl[j0:j1, None, :, 1]
        r = erad[bj] + forad[j0:j1, None] + 1e-6
        cand = (dx * dx + dy * dy <= r * r) & fpres[j0:j1, None]
        jj, pi, ti = _nonzero(cand)
        if not len(jj):
            continue
        jj = jj + j0
        bi, mi = ob[jj], om[jj]
        hitq = _sat_intersects(ego4l[bi, pi, ti], fcl[jj, ti])
        cnt = hitq & ~f_excl_hit[jj] & (sp[bi, pi, ti] > C["stopped"])
        t_hit.scatter_reduce_(0, bi * P + pi, torch.where(cnt, ti, 99), "amin")
        first_any.scatter_reduce_(0, (bi * P + pi) * M + mi, torch.where(hitq, ti, 99), "amin")
    t_hit = t_hit.view(B, P)
    first_any = first_any.view(B, P, M)
    out["hit"] = cs["ns"] >= cs["latch"][t_hit.clamp(0, 104)][..., None]
    dg["hit_first10"] = t_hit

    # ---------------------------------------------------------------- closeness: facts.closeness (fact ttc)
    ttc10 = torch.full((B * P * K,), math.inf, dtype=F64, device=dev)
    ecen_g = ego4.mean(-2)
    erad_g = (ego4 - ecen_g[..., None, :]).pow(2).sum(-1).amax(-1).sqrt()
    fcor, fvel = T["f_corners"], T["f_vel"]
    fcen = fcor.mean(-2)                                                      # prefilter only (facts: corners.mean(2))
    forad_g = (fcor - fcen[..., None, :]).pow(2).sum(-1).amax(-1).sqrt()      # (N, 41)
    moving = sp > C["stopped"]
    for j0 in range(0, N, nc):
        j1 = min(N, j0 + nc)
        bj = ob[j0:j1]
        dx = ecen_g[bj, :, :, 0] - fcen[j0:j1, None, :, 0]
        dy = ecen_g[bj, :, :, 1] - fcen[j0:j1, None, :, 1]
        rx = fvel[j0:j1, None, :, 0] - ev[bj, :, :, 0]
        ry = fvel[j0:j1, None, :, 1] - ev[bj, :, :, 1]
        reach = torch.sqrt(rx * rx + ry * ry) * C["ttc_cap"] + erad_g[bj] + forad_g[j0:j1, None] + 0.5
        cand = (dx * dx + dy * dy <= reach * reach) & fpres[j0:j1, None] & ~f_excl[j0:j1, None, None] & moving[bj]
        jj, pi, ti = _nonzero(cand)
        if not len(jj):
            continue
        jj = jj + j0
        step = CHUNK // 8
        for q0 in range(0, len(jj), step):
            j_, p_, t_ = jj[q0:q0 + step], pi[q0:q0 + step], ti[q0:q0 + step]
            b_ = ob[j_]
            tt = _time_to_contact(ego4[b_, p_, t_], ev[b_, p_, t_], fcor[j_, t_], fvel[j_, t_], C["ttc_cap"])
            ttc10.scatter_reduce_(0, (b_ * P + p_) * K + t_, tt, "amin")
    ttc10 = ttc10.view(B, P, K)
    out["ttc"] = f16_round(_per_step(ttc10, "min", cs["win"]))
    dg["ttc10"] = ttc10

    # ---------------------------------------------------------------- cut-in: facts.cutin (rear_ok's IDM brake)
    brake10 = _cutin(T, states, ego_areas, C, dev)
    out["cutin_brake"] = torch.nan_to_num(_per_step(brake10, "max", cs["win"]).to(torch.float32), nan=0.0)
    dg["brake10"] = brake10

    # ---------------------------------------------------------------- comfort
    if kind == "nav":
        ratios = _comfort(T, states, dev)                                      # (B, P, 41, 6)
        rs = _per_step(ratios.transpose(-1, -2), "max", cs["win"])            # (B, P, 6, 8)
        out["comfort_step"] = f16_round(rs).to(torch.float32).amax(-2) >= 1.0
        dg["comfort10"] = ratios
    else:
        out["comfort_step"] = torch.zeros((B, P, NS), dtype=torch.bool, device=dev)

    # ---------------------------------------------------------------- NAVSIM: run 3A's key agents
    if kind == "nav":
        kc, kv, kd = _key_agents(T, cs, states, coords, ego_areas, first_any, sp, corl, ecl, erad, fcl, focl, forad, dev)
        out["key_corners"], out["key_valid"] = kc, kv
        dg.update(kd)
        out["key_mask"] = _key_mask(kd, kv, T["init_first"])
    elif TA is not None:
        fa_aux = _first_contacts(TA, ego4l, ecl, erad, B, P, dev)
        kc, kv, kd = _key_agents(TA, cs, states, coords, ego_areas, fa_aux, sp, TA["corl"], ecl, erad, TA["fcl"], TA["focl"],
                                 TA["forad"], dev)
        out["key_corners"], out["key_valid"] = kc, kv
        out["key_mask"] = _key_mask(kd, kv, TA["init_first"])
        dg.update(kd)
    if diag:
        out["_diag"] = dg
    return out


# ------------------------------------------------------------------------------------------------------------- cut-in
def _cutin(T, states, ego_areas, C, dev):
    B, P = states.shape[:2]
    brake = torch.zeros((B * P, K), dtype=F64, device=dev)
    lane_change = ego_areas[..., 0].any(-1)                                    # (B, P)
    xyh, pv, lane0 = T["veh_xyh"], T["veh_present"], T["veh_lane0"]
    ex, ey, eh = states[..., 0], states[..., 1], states[..., 2]               # (B, P, 41)
    dx = xyh[:, None, :, :, 0] - ex[:, :, None]                               # (B, P, Mv, 41)
    dy = xyh[:, None, :, :, 1] - ey[:, :, None]
    c, s = torch.cos(eh)[:, :, None], torch.sin(eh)[:, :, None]
    lon, lat = c * dx + s * dy, -s * dx + c * dy
    lat0 = lat[..., :1]
    same_dir = _wrap(xyh[:, None, :, :, 2] - eh[:, :, None]).abs() <= C["same_dir"]
    out0 = (lat0.abs() > C["strip"]) & ~lane0[:, None, :, None]
    cond = (pv[:, None] & pv[:, None, :, :1] & same_dir & (lon < -C["ego_rear"]) & (lat.abs() <= C["strip"]) & out0
            & lane_change[:, :, None, None])
    bi, pi, mi = _nonzero(cond.any(-1))
    Q = len(bi)
    if not Q:
        return brake.view(B, P, K)
    k0 = cond[bi, pi, mi].to(torch.uint8).argmax(-1)                          # first True
    # lane gate (run3a_labels.ego_path_lane_pairs): the vehicle's centre at k0 is in a lane polygon the ego's centre enters
    keep = _lane_gate(T, states, bi, pi, mi, k0, C, dev)
    v0 = T["veh_speed"][bi, mi, k0]
    v0 = torch.where(torch.isnan(v0), torch.zeros_like(v0), v0)
    S = states[bi, pi]                                                         # (Q, 41, 11)
    sx, sy = S[..., 0], S[..., 1]
    arc = torch.cat([torch.zeros_like(sx[:, :1]), torch.cumsum(torch.hypot(sx[:, 1:] - sx[:, :-1], sy[:, 1:] - sy[:, :-1]), 1)], 1)
    evq = torch.hypot(S[..., 3], S[..., 4])
    v0t = torch.clamp(v0, min=0.5) * 1.0                                       # speed_limit_fraction 1 x fallback max(v, 0.5)
    g0, Th, a, b = 1.0, 1.5, 1.5, C["brake"]
    sq = 2 * math.sqrt(a * b)
    x = arc[torch.arange(Q, device=dev), k0] + lon[bi, pi, mi, k0] + T["veh_length"][bi, mi] / 2
    v = v0
    need = torch.zeros((Q, K), dtype=F64, device=dev)
    dtc = C["dt"]
    for k in range(K - 1):                                                     # pairs act from their k0 on
        act = keep & (k0 <= k)
        x_lead, v_lead = arc[:, k] + C["r2c"], evq[:, k]
        s_star = g0 + v * Th + (v * (v - v_lead)) / sq
        s_alpha = torch.clamp(x_lead - x - C["half_length"], min=g0)
        want = a * (1 - (v / v0t) ** 10 - (s_star / s_alpha) ** 2)
        need[:, k] = torch.where(act, torch.clamp(-want, min=0.0), need[:, k])
        nx = x + dtc * v
        nv = torch.clamp(v + dtc * torch.clamp(want, -b, a), min=0.0)
        x, v = torch.where(act, nx, x), torch.where(act, nv, v)
    brake.scatter_reduce_(0, (bi * P + pi)[:, None].expand(Q, K), need, "amax")
    return brake.view(B, P, K)


def _lane_gate(T, states, bi, pi, mi, k0, C, dev):
    """(Q,) bool: some lane polygon holding vehicle mi's centre at k0 also holds path pi's centre at some of the 41 states
    (shapely contains_xy, strict interior; lanes from extract, per vehicle and index)."""
    Q = len(bi)
    if T["nl"] == 0:
        return torch.zeros(Q, dtype=torch.bool, device=dev)
    lanes = T["veh_lanes"][bi, mi, k0]                                         # (Q, Lmax), -1 = none
    L = lanes.shape[1]
    lid = lanes.clamp_min(0).reshape(-1)
    n_e = torch.where(lanes.reshape(-1) >= 0, T["lane_end"][lid] - T["lane_start"][lid], torch.zeros_like(lid))
    R = Q * L
    tot = int(n_e.sum())                                                       # one host sync
    if tot == 0:
        return torch.zeros(Q, dtype=torch.bool, device=dev)
    r2c = C["r2c"]
    s0 = T["s0"]
    S = states[bi, pi]                                                         # (Q, 41, 11)
    cx = (S[..., 0] + r2c * torch.cos(S[..., 2])) - s0[bi, 0:1]                # run3a_labels: rear axle + REAR_AXLE_TO_CENTER
    cy = (S[..., 1] + r2c * torch.sin(S[..., 2])) - s0[bi, 1:2]
    cum = torch.cumsum(n_e, 0)
    rows_all = torch.repeat_interleave(torch.arange(R, device=dev), n_e, output_size=tot)
    eidx_all = (torch.arange(tot, device=dev) - torch.repeat_interleave(cum - n_e, n_e, output_size=tot)
                + torch.repeat_interleave(T["lane_start"][lid], n_e, output_size=tot))
    par = torch.zeros((R, K), dtype=torch.int64, device=dev)
    step = max(1, CHUNK // K)
    segs = T["lane_segs"]
    for e0 in range(0, tot, step):
        rr, ee = rows_all[e0:e0 + step], eidx_all[e0:e0 + step]
        q = rr // L
        cr = _crossings(cx[q], cy[q], segs[ee][:, None, :])                    # (n, 41)
        par.index_add_(0, rr, cr.to(torch.int64))
    inside = ((par % 2) == 1).any(1).view(Q, L)
    return inside.any(1)


# ------------------------------------------------------------------------------------------------------------- comfort
_SAVGOL = {}


def _savgol_mats(n, dev):
    """the linear maps of pdm_comfort_metrics' savgol filters (mode 'interp') for n padded states, from scipy itself:
    accel (window 8, order 2), jerk (deriv 1, window 15, order 2), yaw rate (deriv 1, window 5, order 2), yaw accel
    (deriv 2, window 5, order 3); delta = the mean step of the padded time stamps (as _approximate_derivatives)."""
    key = (n, str(dev))
    if key not in _SAVGOL:
        from scipy.signal import savgol_filter
        ts = np.arange(n).astype(np.float64) * 0.1
        dx = np.diff(ts, axis=-1).mean()
        I = np.eye(n)
        mats = dict(acc=savgol_filter(I, polyorder=2, window_length=min(8, n), axis=-1),
                    jerk=savgol_filter(I, polyorder=2, window_length=min(15, n), deriv=1, delta=dx, axis=-1),
                    yawr=savgol_filter(I, polyorder=2, window_length=min(5, n), deriv=1, delta=dx, axis=-1),
                    yawa=savgol_filter(I, polyorder=3, window_length=min(5, n), deriv=2, delta=dx, axis=-1))
        _SAVGOL[key] = {k: torch.as_tensor(v, device=dev, dtype=F64) for k, v in mats.items()}  # row j = impulse at j
    return _SAVGOL[key]


def _r8(x):
    return torch.round(x * 100000000.0) / 100000000.0                         # np.round(x, 8): rint(x * 1e8) / 1e8


def _comfort(T, states, dev):
    """facts.comfort_ratios: the six history-comfort values over the logged history + the 41 states, / their limits,
    for the 41 rollout indices -> (B, P, 41, 6)."""
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as cm
    B, P = states.shape[:2]
    out = torch.empty((B, P, K, 6), dtype=F64, device=dev)
    r2c = T["comfort_r2c"]
    for H, it in T["hgroups"]:
        A = _savgol_mats(H + K, dev)
        hist = T["hist"][it, :H]                                               # (b, H, 11)
        pad = torch.cat([hist[:, None].expand(-1, P, -1, -1), states[it]], 2)  # (b, P, n, 11)
        ax, ay, w, al = pad[..., 5], pad[..., 6], pad[..., 9], pad[..., 10]
        cax = (ax + r2c * (w * w)) + r2c * al                                  # get_acceleration_shifted, displacement (r2c, 0)
        cay = (ay + 0.0 * (w * w)) + 0.0 * al
        mag = torch.hypot(ax, ay)
        lon = _r8(cax @ A["acc"])                                              # out[i] = sum_j y[j] A[i, j] = (y @ rows)[i]
        lat = _r8(cay @ A["acc"])
        jerk = _r8(_r8(mag @ A["acc"]) @ A["jerk"])
        lonj = _r8(lon @ A["jerk"])
        hh = pad[..., 2]
        two_pi = 2.0 * np.pi
        adj = torch.zeros_like(hh)
        adj[..., 1:] = torch.cumsum(torch.round((hh[..., 1:] - hh[..., :-1]) / two_pi), -1)
        unw = hh - two_pi * adj
        yawa = _r8(unw @ A["yawa"])
        yawr = _r8(unw @ A["yawr"])
        r = torch.stack([torch.where(lon >= 0, lon / cm.MAX_LON_ACCEL, lon / cm.MIN_LON_ACCEL), lat.abs() / cm.MAX_ABS_LAT_ACCEL,
                         jerk.abs() / cm.MAX_ABS_MAG_JERK, lonj.abs() / cm.MAX_ABS_LON_JERK, yawa.abs() / cm.MAX_ABS_YAW_ACCEL,
                         yawr.abs() / cm.MAX_ABS_YAW_RATE], -1)
        out[it] = r[:, :, H:]
    return out


# ------------------------------------------------------------------------------------------------------------- key agents
def _rel_angle(ex, ey, eh, ax, ay):
    """nuplan get_agent_relative_angle: arccos(dot(ego heading unit, (agent - ego) / norm))."""
    vx, vy = ax - ex, ay - ey
    nrm = torch.sqrt(vx * vx + vy * vy)
    return torch.arccos(torch.cos(eh) * (vx / nrm) + torch.sin(eh) * (vy / nrm))


def _key_agents(T, cs, states, coords, ego_areas, first_any, sp, corl, ecl, erad, fcl, focl, forad, dev):
    """compute_navsim_score.after_score_proposals(test=False)[1:3] for the P paths: per path the first at-fault collision
    agent (PDMScorer._calculate_no_at_fault_collision) and the first TTC agent (_calculate_ttc), their corners at indices
    1 .. the infraction index (if present there), in the ego frame at t."""
    B, P = states.shape[:2]
    M = T["M"]
    s0, cent, rank, tstop, init, present, corners = (T["s0"], T["centroid"], T["rank"], T["track_stopped"], T["init"],
                                                     T["present"], T["corners"])
    ml_nd = ego_areas[..., 0] | ego_areas[..., 1]                              # (B, P, 41)
    ex, ey, eh = states[..., 0], states[..., 1], states[..., 2]
    BIG = 2 ** 31
    # ---- at-fault collision: each (path, object)'s first contact decides (later contacts of a non-fault token skipped)
    fa = torch.where(init[:, None, :], torch.full_like(first_any, 99), first_any)   # collided_track_ids: skipped
    bi, pi, mi = _nonzero(fa < 99)
    ti = fa[bi, pi, mi]
    stopped_ego = sp[bi, pi, ti] <= STOPPED
    behind = _rel_angle(ex[bi, pi, ti], ey[bi, pi, ti], eh[bi, pi, ti], cent[bi, mi, ti, 0], cent[bi, mi, ti, 1]) > BEHIND_RAD
    fl, fr = coords[bi, pi, ti, 0] - s0[bi, :2], coords[bi, pi, ti, 3] - s0[bi, :2]
    front = _seg_intersects_quad(fl, fr, corl[bi, mi, ti])
    fault = ~stopped_ego & (tstop[bi, mi] | (~behind & (front | ml_nd[bi, pi, ti])))
    code = torch.full((B, P, M), SENT, dtype=torch.int64, device=dev)
    code[bi, pi, mi] = torch.where(fault, ti * BIG + rank[bi, mi, ti], torch.full_like(ti, SENT))
    cmin, col_m = code.min(-1)
    col_t = torch.where(cmin < SENT, cmin // BIG, torch.full_like(cmin, -1))  # -1: no at-fault collision
    # ---- TTC: extruded boxes at time t, look-ahead f, against the objects at t + f
    dxy = torch.stack([torch.cos(eh) * sp, torch.sin(eh) * sp], -1)            # (B, P, 41, 2)
    NT, NF = N_TTC_T, len(FUT)
    tf, dts = cs["tf"], cs["dts"]
    ev_first = torch.full((B * P * M,), SENT, dtype=torch.int64, device=dev)  # first event code (t * 4 + s) per (path, obj)
    ok_speed = sp >= SC_STOPPED
    ob, om, N = T["f_sid"], T["f_lid"], T["N"]
    finit, fpres = T["f_init"], T["f_present"]
    # swept prefilter: the 4 extruded ego boxes of (t) lie in a circle around ecl + dxy * 0.45 of radius erad + |v| * 0.45;
    # the object's 4 boxes at t + f in a circle around their mean centre (present ones only)
    ec_sw = ecl[:, :, :NT] + dxy[:, :, :NT] * 0.45                            # (B, P, T, 2)
    er_sw = erad[:, :, :NT] + sp[:, :, :NT] * 0.45
    oc4, pr4, or4 = focl[:, tf], fpres[:, tf], forad[:, tf]                    # (N, T, 4, ...)
    w = pr4.to(F64)
    omid = (oc4 * w[..., None]).sum(-2) / w.sum(-1).clamp_min(1.0)[..., None]   # (N, T, 2)
    or_sw = torch.where(pr4, or4 + (oc4 - omid[:, :, None]).pow(2).sum(-1).sqrt(), torch.zeros_like(or4)).amax(-1)
    any4 = pr4.any(-1) & ~finit[:, None]
    nc = max(1, CHUNK // (P * NT))
    for j0 in range(0, N, nc):
        j1 = min(N, j0 + nc)
        bj = ob[j0:j1]
        dx = ec_sw[bj, :, :, 0] - omid[j0:j1, None, :, 0]                      # (n, P, T)
        dy = ec_sw[bj, :, :, 1] - omid[j0:j1, None, :, 1]
        r = er_sw[bj] + or_sw[j0:j1, None] + 1e-6
        cand = (dx * dx + dy * dy <= r * r) & any4[j0:j1, None] & ok_speed[bj, :, :NT]
        jj, p_, t_ = _nonzero(cand)
        if not len(jj):
            continue
        jj = jj + j0
        n = len(jj)
        jj, p_, t_ = jj.repeat_interleave(NF), p_.repeat_interleave(NF), t_.repeat_interleave(NF)
        s_ = torch.arange(NF, device=dev).repeat(n)
        b_ = ob[jj]
        # the scorer's extruded polygon: coords (FL RL RR FR) + dxy * delta_t (its own rounding), then the local frame
        E = (coords[b_, p_, t_, :4] + dxy[b_, p_, t_][:, None, :] * dts[s_][:, None, None]) - s0[b_, None, :2]
        cur = tf[t_, s_]
        hitq = fpres[jj, cur] & _sat_intersects(E, fcl[jj, cur])
        ev_first.scatter_reduce_(0, (b_ * P + p_) * M + om[jj], torch.where(hitq, t_ * NF + s_, SENT), "amin")
    ev_first = ev_first.view(B, P, M)
    bi, pi, mi = _nonzero(ev_first < SENT)
    evc = ev_first[bi, pi, mi]
    ti, si = evc // NF, evc % NF
    cur = tf[ti, si]
    ang = _rel_angle(ex[bi, pi, ti], ey[bi, pi, ti], eh[bi, pi, ti], cent[bi, mi, cur, 0], cent[bi, mi, cur, 1])
    ahead, behind = ang < AHEAD_RAD, ang > BEHIND_RAD
    # rear axle strictly inside an INTERSECTION polygon (drivable map is_in_layer, STRtree 'within')
    inter = torch.zeros_like(ahead)
    E = T["inter_segs"].shape[1]
    step = max(1, CHUNK // E)
    for q0 in range(0, len(bi), step):
        b = bi[q0:q0 + step]
        px = ex[b, pi[q0:q0 + step], ti[q0:q0 + step]] - s0[b, 0]
        py = ey[b, pi[q0:q0 + step], ti[q0:q0 + step]] - s0[b, 1]
        sg = T["inter_segs"][b]
        cr = _crossings(px[:, None], py[:, None], sg) & ~torch.isnan(sg[..., 0])
        cnt = torch.zeros((len(b), T["NPm"]), dtype=torch.int64, device=dev)
        cnt.scatter_add_(1, T["inter_poly"][b], cr.to(torch.int64))
        inter[q0:q0 + step] = ((cnt % 2) == 1).any(1)
    fault = ahead | ((ml_nd[bi, pi, ti] | inter) & ~behind)
    code = torch.full((B, P, M), SENT, dtype=torch.int64, device=dev)
    code[bi, pi, mi] = torch.where(fault, evc * BIG + rank[bi, mi, cur], torch.full_like(evc, SENT))
    cmin, ttc_m = code.min(-1)
    ttc_t = torch.where(cmin < SENT, (cmin // BIG) // NF, torch.full_like(cmin, -1))
    # ---- run 3A's quirk: a non-empty collided_track_ids list makes its first token the key agent of both kinds
    err = torch.zeros((B, P), dtype=torch.bool, device=dev)
    for b, m0 in enumerate(T["init_first"]):
        if m0 >= 0:
            err[b] = (col_t[b] < 0) | (ttc_t[b] < 0)                           # the CPU raises (int(inf)) on these rows
            col_m[b], ttc_m[b] = m0, m0
    # ---- corners at indices 1 .. t_key (present there), ego frame
    tt = cs["t40"]                                                             # time_idx 1..40 -> slot 0..39
    bb = torch.arange(B, device=dev)[:, None, None]
    kcs, kvs = [], []
    for kt, km in ((col_t, col_m), (ttc_t, ttc_m)):
        use = (tt <= kt[..., None]) & present[bb, km[..., None], tt]          # (B, P, 40)
        cc = corners[bb, km[..., None], tt]                                    # (B, P, 40, 4, 2)
        kcs.append(torch.where(use[..., None, None], cc, torch.zeros_like(cc)))
        kvs.append(use)
    kc, kv = torch.stack(kcs, 3), torch.stack(kvs, 3)                          # (B, P, 40, 2, 4, 2), (B, P, 40, 2)
    o = T["origin"]                                                            # rear axle x, y, heading at t
    c, s_ = torch.cos(o[:, 2]), torch.sin(o[:, 2])
    shp = (B, 1, 1, 1, 1)
    X = kc[..., 0] - o[:, 0].view(shp)
    Y = kc[..., 1] - o[:, 1].view(shp)
    kc = torch.stack([X * c.view(shp) + Y * s_.view(shp), X * (-s_).view(shp) + Y * c.view(shp)], -1)
    return kc.to(torch.float32), kv, dict(key_col_t=col_t, key_col_m=col_m, key_ttc_t=ttc_t, key_ttc_m=ttc_m, key_err=err)


# ============================================================================================================== LEAD key agents
# run 3A's key-agent aux labels on LEAD samples (lead_aux/keyagents/keyagents_v2.py, Navvrat 19 Nov): the same selection as
# NAVSIM's (_key_agents: CARLA scorer rules = the same PDMScorer config; MKZ boxes come in as coords; the CARLA drivable map's
# INTERSECTION polygons for the TTC rear-axle test; the STRtree order of this observation; collided_track_ids empty) on the
# key-agent observation built by keyagents_v2.observation_v2b (v2 boxes present at t + carla_full_v1's later objects, no
# full.DEBRIS objects). Additive: hit / ttc / cutin_brake / comfort_step and NAVSIM's keys are computed as before.
KA_DIR = "{runs}/run3b/lead_aux/keyagents"


def _key_obs_arrays(obs, toks):
    """centroid (M, 41, 2) and STRtree traversal rank (M, 41) of an observation's objects (as the NAVSIM extract)."""
    import shapely
    M = len(toks)
    tix = {t: i for i, t in enumerate(toks)}
    cent = np.zeros((M, K, 2))
    rank = np.full((M, K), 2 ** 30, np.int32)
    giant = shapely.box(-1e12, -1e12, 1e12, 1e12)
    for k in range(K):
        om = obs[k]
        order = om._str_tree.query(giant)
        mm, gg = [], []
        for pos, j in enumerate(order.tolist()):
            t = om.tokens[j]
            if obs.red_light_token in t:
                continue
            rank[tix[t], k] = pos
            mm.append(tix[t])
            gg.append(om._geometries[j])
        if mm:
            c = shapely.centroid(np.array(gg, dtype=object))
            cent[mm, k, 0], cent[mm, k, 1] = shapely.get_x(c), shapely.get_y(c)
    return cent, rank


def _extract_aux_carla(ctx, meta=None):
    """CARLA extract, additive keys aux_*: the LEAD key-agent observation's objects (corners, presence, centroids, tree
    rank, stopped flag), the CARLA map's INTERSECTION rings and the rear-axle origin at t."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.observation.idm.utils import is_track_stopped
    if KA_DIR not in sys.path:
        sys.path.insert(0, KA_DIR)
    import keyagents_v2 as KV2
    F, _ = _mods()
    if "v2" not in ctx:
        ctx["v2"] = KV2.v2_record_for_ctx(ctx, meta)
    if "rt" not in ctx:
        ctx["rt"] = KV2.route_for_ctx(ctx)
    obs, info = KV2.observation_v2b(ctx, ctx["v2"])
    toks, corners, present, _, _ = F.tracks(obs)
    cent, rank = _key_obs_arrays(obs, toks)
    uo = obs.unique_objects
    dam = ctx["mc"].drivable_area_map
    isegs = [_rings(dam._geometries[i]) for i in dam.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])]
    ra = ctx["mc"].ego_state.rear_axle
    return dict(aux_M=len(toks), aux_tokens=np.array(toks, object), aux_corners=corners, aux_present=present, aux_centroid=cent, aux_rank=rank,
                aux_track_stopped=np.array([is_track_stopped(uo[t]) for t in toks], bool),
                aux_inter_segs=np.concatenate(isegs) if isegs else np.zeros((0, 4)),
                aux_inter_poly=np.concatenate([np.full(len(s), j) for j, s in enumerate(isegs)]).astype(np.int64) if isegs else np.zeros(0, np.int64),
                aux_origin=np.array([ra.x, ra.y, ra.heading], np.float64))


def _upload_aux(scenes, dev):
    """the aux observation of a CARLA batch as the dict _key_agents reads (one upload per dtype, before any kernel)."""
    B = len(scenes)
    U = _Upload()
    Ms = np.array([s["aux_M"] for s in scenes])
    U.put("sid", np.repeat(np.arange(B), Ms), np.int64)
    U.put("lid", np.concatenate([np.arange(m) for m in Ms]) if Ms.sum() else np.zeros(0, np.int64), np.int64)
    cat = lambda k: np.concatenate([s[k] for s in scenes])  # noqa: E731
    U.put("corners", cat("aux_corners").reshape(-1, K, 4, 2), np.float64)
    U.put("present", cat("aux_present").reshape(-1, K), np.bool_)
    U.put("centroid", cat("aux_centroid").reshape(-1, K, 2), np.float64)
    U.put("rank", cat("aux_rank").reshape(-1, K), np.int64)
    U.put("track_stopped", cat("aux_track_stopped"), np.bool_)
    U.put("s0", np.stack([s["s0"] for s in scenes]), np.float64)
    U.put("origin", np.stack([s["aux_origin"] for s in scenes]), np.float64)
    E = max(1, max(len(s["aux_inter_segs"]) for s in scenes))
    isg = np.full((B, E, 4), np.nan)
    ipo = np.zeros((B, E), np.int64)
    NPm = 1
    for i, s in enumerate(scenes):
        n = len(s["aux_inter_segs"])
        if n:
            isg[i, :n] = s["aux_inter_segs"] - np.r_[s["s0"][:2], s["s0"][:2]]
            ipo[i, :n] = s["aux_inter_poly"]
            NPm = max(NPm, int(s["aux_inter_poly"].max()) + 1)
    U.put("inter_segs", isg, np.float64)
    U.put("inter_poly", ipo, np.int64)
    g = U.go(dev)
    M = max(1, int(Ms.max()))
    s0 = g["s0"]
    TA = dict(M=M, N=int(Ms.sum()), s0=s0, origin=g["origin"], inter_segs=g["inter_segs"], inter_poly=g["inter_poly"], NPm=NPm,
              init_first=[-1] * B, f_sid=g["sid"], f_lid=g["lid"], f_corners=g["corners"], f_present=g["present"])
    TA["corners"] = _scatter_pad(g["corners"], g["sid"], g["lid"], B, M, 0.0)
    TA["present"] = _scatter_pad(g["present"], g["sid"], g["lid"], B, M, False)
    TA["centroid"] = _scatter_pad(g["centroid"], g["sid"], g["lid"], B, M, 0.0)
    TA["rank"] = _scatter_pad(g["rank"], g["sid"], g["lid"], B, M, 2 ** 30)
    TA["track_stopped"] = _scatter_pad(g["track_stopped"], g["sid"], g["lid"], B, M, True)
    TA["init"] = torch.zeros((B, M), dtype=torch.bool, device=dev)              # collided_track_ids = [] (as NAVSIM's cache)
    TA["f_init"] = torch.zeros(TA["N"], dtype=torch.bool, device=dev)
    TA["corl"] = TA["corners"] - s0[:, None, None, None, :2]
    TA["fcl"] = g["corners"] - s0[g["sid"]][:, None, None, :2]
    TA["focl"] = TA["fcl"].mean(-2)
    TA["forad"] = (TA["fcl"] - TA["focl"][..., None, :]).pow(2).sum(-1).amax(-1).sqrt()
    return TA


def _first_contacts(TA, ego4l, ecl, erad, B, P, dev):
    """(B, P, M) first 10 Hz index at which the path's box touches each object of TA (any speed; 99 = never)."""
    M, N = TA["M"], TA["N"]
    ob, om, fcl, focl, forad, fpres = TA["f_sid"], TA["f_lid"], TA["fcl"], TA["focl"], TA["forad"], TA["f_present"]
    first_any = torch.full((B * P * M,), 99, dtype=torch.int64, device=dev)
    nc = max(1, CHUNK // (P * K))
    for j0 in range(0, N, nc):
        j1 = min(N, j0 + nc)
        bj = ob[j0:j1]
        dx = ecl[bj, :, :, 0] - focl[j0:j1, None, :, 0]
        dy = ecl[bj, :, :, 1] - focl[j0:j1, None, :, 1]
        r = erad[bj] + forad[j0:j1, None] + 1e-6
        jj, pi, ti = _nonzero((dx * dx + dy * dy <= r * r) & fpres[j0:j1, None])
        if not len(jj):
            continue
        jj = jj + j0
        bi, mi = ob[jj], om[jj]
        hitq = _sat_intersects(ego4l[bi, pi, ti], fcl[jj, ti])
        first_any.scatter_reduce_(0, (bi * P + pi) * M + mi, torch.where(hitq, ti, 99), "amin")
    return first_any.view(B, P, M)


def _key_mask(kd, kv, init_first):
    """(B, P, 2) bool: False where a key agent exists (token) but none of its 40 slots is valid."""
    has = torch.stack([kd["key_col_t"] >= 0, kd["key_ttc_t"] >= 0], -1)
    if any(m0 >= 0 for m0 in init_first):                                     # NAVSIM collided_track_ids quirk only
        has = has | torch.as_tensor([m0 >= 0 for m0 in init_first], device=has.device)[:, None, None]
    return ~(has & ~kv.any(2))
