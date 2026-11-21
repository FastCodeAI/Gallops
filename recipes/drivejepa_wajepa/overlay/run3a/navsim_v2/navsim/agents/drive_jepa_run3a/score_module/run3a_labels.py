"""Run 3A labels (paddock-ops/run3a/CLAUDE.md, "New labels"), per candidate path, from the training metric cache and the
training scorer's state right after score_proposals (simulated states, ego corners, ego areas).

rear_ok     only paths the scorer flags MULTIPLE_LANES at some step can fail. A logged vehicle (VEHICLE type, heading within
            45 deg of the ego's at t) fails the path if at some step t its centre is behind the ego's rear bumper and inside
            the ego's lane strip (|lateral| <= 1.75 m in the ego frame at t), while at t = 0 it was outside the strip
            (|lateral| > 1.75 m in the ego frame at t = 0) - so a vehicle already behind in the ego's lane never counts.
            On curves the straight t = 0 strip misses vehicles far back in the ego's own lane, so a vehicle whose t = 0
            centre is in the ego's lane or its predecessors / successors (3 links, lane graph from lane_centrelines.py)
            also never counts. Option (a) (user, 24 Oct): the vehicle must also be, at t, in a lane the ego's centre
            enters at some step of that path (REAR_LANE_GATE).
            From the first such t, IDM (BatchIDMPolicy, PDM-Closed's min gap 1 m / headway 1.5 s / accel 1.5) runs the
            vehicle from its logged speed (also its target speed) behind the ego's simulated path to 4 s; the path fails if
            the vehicle must brake harder than 4 m/s^2 (decel_max = 4: a step at the clip means the IDM wanted more).
            Tracks come from the observation's 41 occupancy maps (0.1 s): centre = polygon centroid, heading = corner 1 ->
            corner 0 (checked against the agent's own heading), speed = centroid travel over +-0.2 s.
edge        worst ego corner's signed distance to the drivable boundary (ROADBLOCK, INTERSECTION, DRIVABLE_AREA,
            CARPARK_AREA union, as the scorer's drivable check) over the 41 simulated steps; + inside; clip [-1, 2] m.
route       share of the 8 waypoint steps (0.5 .. 4.0 s) whose centre is on a route lane (the scorer's ONCOMING_TRAFFIC
            flag off; route lanes = every lane of the route roadblocks).
map aux     per waypoint (the path's own 8 poses, rear axle): the 4 ego-box corners' signed distance to the drivable
            boundary (clip [-1, 2] m), signed offset from the lane centre (left +, clip +-3.5 m) and heading minus lane
            heading (rad, wrapped). Lane = the LANE / LANE_CONNECTOR polygon containing the waypoint (several: best heading
            match), else the nearest centreline. Centrelines from lane_centrelines.py (one pickle per map).
"""
import os
import pickle
from typing import Dict, Optional

import numpy as np
import shapely
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex

DT = 0.1
STRIP = 1.75
SAME_DIR = np.radians(45.0)
BRAKE = 4.0
# pacifica (the scorer's vehicle): rear axle -> centre 1.461 m, half length 2.588 m, half width 1.1485 m
REAR_AXLE_TO_CENTER, HALF_LENGTH, HALF_WIDTH = 1.461, 2.588, 1.1485
EGO_REAR = HALF_LENGTH - REAR_AXLE_TO_CENTER  # rear axle -> rear bumper
DRIVABLE = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA,
            SemanticMapLayer.CARPARK_AREA]
LANES = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
CENTRELINE_DIR = os.environ.get("RUN3A_CENTRELINES", "")  # set by gallops' recipe; as run, a machine-path default
WAYPOINT_STEPS = np.arange(5, 41, 5)  # simulated steps of the 8 waypoints
# rear_ok option (a), chosen by the user 24 Oct: a vehicle also needs its centre (at the in-strip step) in a LANE /
# LANE_CONNECTOR polygon that the ego's centre enters at some step of that path (the straight strip alone sweeps over
# other lanes in turns). RUN3A_REAR_LANE_GATE=0 gives the spec's literal rule.
REAR_LANE_GATE = os.environ.get("RUN3A_REAR_LANE_GATE", "1") == "1"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def vehicle_tracks(observation):
    """VEHICLE tracks over the observation's steps: xyh (M, K, 3) global, present (M, K), length (M,), speed (M, K)."""
    toks = [t for t, a in observation._unique_objects.items() if a.tracked_object_type == TrackedObjectType.VEHICLE]
    K = len(observation._occupancy_maps)
    xyh, present = np.zeros((len(toks), K, 3)), np.zeros((len(toks), K), bool)
    for k in range(K):
        om = observation[k]
        for i, t in enumerate(toks):
            j = om._token_to_idx.get(t)
            if j is None:
                continue
            c = np.asarray(om._geometries[j].exterior.coords)
            xyh[i, k, :2] = c[:4].mean(0)
            xyh[i, k, 2] = np.arctan2(c[0, 1] - c[1, 1], c[0, 0] - c[1, 0])
            present[i, k] = True
    length = np.array([observation._unique_objects[t].box.length for t in toks], dtype=np.float64)
    speed = np.full((len(toks), K), np.nan)
    for k in range(K):
        a, b = max(k - 2, 0), min(k + 2, K - 1)
        ok = present[:, a] & present[:, b]
        speed[ok, k] = np.linalg.norm(xyh[ok, b, :2] - xyh[ok, a, :2], axis=-1) / ((b - a) * DT)
    return xyh, present, length, speed


def lane_at(drivable_area_map, lines, x, y, h):
    """token of the LANE / LANE_CONNECTOR polygon holding (x, y) (several: best heading match), or None."""
    best, best_d = None, np.inf
    for i in drivable_area_map.get_indices_of_map_type(LANES):
        tk = drivable_area_map.tokens[i]
        if tk not in lines or not shapely.contains_xy(drivable_area_map._geometries[i], x, y):
            continue
        pl = lines[tk]
        dh = abs(wrap(h - pl[np.hypot(pl[:, 0] - x, pl[:, 1] - y).argmin(), 2]))
        if dh < best_d:
            best, best_d = tk, dh
    return best


def lane_chain(graph, lane, hops=3):
    """lane plus its predecessors and successors up to `hops` links each way (no siblings)."""
    out = {lane}
    for key in ("pred", "succ"):
        front = {lane}
        for _ in range(hops):
            front = {n for f in front for n in graph[key].get(f, [])} - out
            out |= front
    return out


def in_ego_lane0(drivable_area_map, map_name, ego_xyh, pts):
    """pts (M, 2) global at t = 0 -> (M,) bool: centre inside the ego's lane or its lane chain (lane_chain) at t = 0."""
    graph = centrelines(map_name)
    lane = lane_at(drivable_area_map, graph["lines"], *ego_xyh)
    inside = np.zeros(len(pts), bool)
    if lane is None or not len(pts):
        return inside
    chain = lane_chain(graph, lane)
    for i in drivable_area_map.get_indices_of_map_type(LANES):
        if drivable_area_map.tokens[i] in chain:
            inside |= shapely.contains_xy(drivable_area_map._geometries[i], pts[:, 0], pts[:, 1])
    return inside


def ego_path_lane_pairs(drivable_area_map, S, pi, veh_xy):
    """S (n, K, >=3) simulated rear-axle states, pi (Q,) path of each pair, veh_xy (Q, 2) vehicle centres at the in-strip
    step -> (Q,) bool: the vehicle is in a lane polygon the ego's centre enters at some step of that path."""
    paths = np.unique(pi)
    cx = S[paths, :, 0] + REAR_AXLE_TO_CENTER * np.cos(S[paths, :, 2])
    cy = S[paths, :, 1] + REAR_AXLE_TO_CENTER * np.sin(S[paths, :, 2])
    row = np.searchsorted(paths, pi)
    pts = np.concatenate([np.stack([cx.ravel(), cy.ravel()], -1), veh_xy])
    cand = near(drivable_area_map, pts, 1.0)
    keep = np.zeros(len(pi), bool)
    for i in drivable_area_map.get_indices_of_map_type(LANES):
        if i not in cand:
            continue
        g = drivable_area_map._geometries[i]
        ego_in = shapely.contains_xy(g, cx, cy).any(1)  # (n_paths,)
        keep |= ego_in[row] & shapely.contains_xy(g, veh_xy[:, 0], veh_xy[:, 1])
    return keep


def rear_ok(states, ego_areas, observation, drivable_area_map=None, map_name=None, return_detail=False):
    """states (N, 41, 11) simulated (rear axle, global), ego_areas (N, 41, 3) bool -> (N,) bool (True = passes).
    With drivable_area_map and map_name, a vehicle whose t = 0 centre is in the ego's lane chain (in_ego_lane0) also
    counts as already in the ego's lane (on a curve the straight t = 0 strip misses it)."""
    N, K = states.shape[:2]
    ok = np.ones(N, bool)
    lane_change = ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES].any(1)
    detail = {"lane_change": lane_change, "pairs": 0}
    if not lane_change.any():
        return (ok, detail) if return_detail else ok
    xyh, present, length, speed = vehicle_tracks(observation)
    if not len(length):
        return (ok, detail) if return_detail else ok
    K = min(K, xyh.shape[1])
    idx = np.where(lane_change)[0]
    S = states[idx, :K]
    ex, ey, eh = S[..., 0], S[..., 1], S[..., 2]
    dx, dy = xyh[None, :, :K, 0] - ex[:, None], xyh[None, :, :K, 1] - ey[:, None]
    c, s = np.cos(eh)[:, None], np.sin(eh)[:, None]
    lon, lat = c * dx + s * dy, -s * dx + c * dy  # (n, M, K) in the ego frame at each step
    lat0 = lat[:, :, :1]  # the ego's start pose is the same for every path
    same_dir = np.abs(wrap(xyh[None, :, :K, 2] - eh[:, None])) <= SAME_DIR
    out0 = np.abs(lat0) > STRIP
    if drivable_area_map is not None and map_name is not None:
        lane0 = in_ego_lane0(drivable_area_map, map_name, states[0, 0, :3], xyh[:, 0, :2])
        out0 = out0 & ~lane0[None, :, None]
        detail["lane0_vehicles"] = int((lane0 & present[:, 0]).sum())
    cond = (present[None, :, :K] & present[None, :, :1] & same_dir & (lon < -EGO_REAR) & (np.abs(lat) <= STRIP) & out0)
    hit = cond.any(-1)
    if not hit.any():
        return (ok, detail) if return_detail else ok
    pi, mi = np.where(hit)  # (path, vehicle) pairs
    k0 = cond[pi, mi].argmax(-1)
    if REAR_LANE_GATE and drivable_area_map is not None:
        keep = ego_path_lane_pairs(drivable_area_map, S, pi, xyh[mi, k0, :2])
        detail["lane_gate_dropped"] = int((~keep).sum())
        pi, mi, k0 = pi[keep], mi[keep], k0[keep]
        if not len(pi):
            return (ok, detail) if return_detail else ok
    v0 = speed[mi, k0]
    v0 = np.where(np.isnan(v0), 0.0, v0)
    # 1-D along the ego's simulated path: arc length of the rear axle; the vehicle's front at the first step k0
    arc = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(np.hypot(np.diff(ex, axis=1), np.diff(ey, axis=1)), 1)], 1)
    ev = np.hypot(S[..., 3], S[..., 4])
    Q = len(pi)
    policy = BatchIDMPolicy(fallback_target_velocity=[float(max(v, 0.5)) for v in v0], speed_limit_fraction=[1.0] * Q,
                            min_gap_to_lead_agent=1.0, headway_time=1.5, accel_max=1.5, decel_max=BRAKE)
    policy.update(None)
    state = np.stack([arc[pi, k0] + lon[pi, mi, k0] + length[mi] / 2, v0], -1)
    fail = np.zeros(Q, bool)
    for k in range(int(k0.min()), K - 1):
        act = k0 <= k
        if not act.any():
            continue
        lead = np.stack([arc[pi, k] + REAR_AXLE_TO_CENTER, ev[pi, k], np.full(Q, HALF_LENGTH)], -1)
        nxt = policy.propagate(state, lead, list(range(Q)), DT)
        at_clip = state[:, 1] - nxt[:, 1] >= BRAKE * DT - 1e-9
        fail |= act & at_clip
        nxt[:, 1] = np.maximum(nxt[:, 1], 0.0)
        state = np.where(act[:, None], nxt, state)
    bad = np.zeros(len(idx), bool)
    np.logical_or.at(bad, pi, fail)
    ok[idx[bad]] = False
    detail.update(pairs=Q, failing_pairs=int(fail.sum()), failing=(idx[pi[fail]], mi[fail], k0[fail]))
    return (ok, detail) if return_detail else ok


def near(drivable_area_map, pts, margin):
    """indices of the map polygons whose bounding boxes come within margin of the points' bounding box."""
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    return set(drivable_area_map.query(shapely.box(x0, y0, x1, y1)).tolist())


def drivable_union(drivable_area_map, pts=None, margin=5.0):
    """union of the drivable polygons; with pts (n, 2) only those within margin of their bounding box (distances are
    clipped at 2 m, so polygons farther than 5 m cannot change a clipped distance)."""
    idx = drivable_area_map.get_indices_of_map_type(DRIVABLE)
    if pts is not None:
        cand = near(drivable_area_map, pts, margin)
        idx = [i for i in idx if i in cand]
    geoms = [drivable_area_map._geometries[i] for i in idx]
    union = shapely.union_all(geoms)
    shapely.prepare(union)
    return union


def signed_distance(union, pts):
    """pts (..., 2) -> signed distance (...) to the union's boundary, + inside."""
    flat = pts.reshape(-1, 2)
    d = shapely.distance(union.boundary, shapely.points(flat))
    inside = shapely.contains_xy(union, flat[:, 0], flat[:, 1])
    return np.where(inside, d, -d).reshape(pts.shape[:-1])


def edge_distance(ego_coords, union):
    """ego_coords (N, 41, 5, 2) the scorer's corners (+ centre last) -> (N,) worst corner, clip [-1, 2]."""
    return np.clip(signed_distance(union, ego_coords[:, :, :4]).min((1, 2)), -1.0, 2.0)


def route_share(ego_areas):
    """ego_areas (N, 41, 3) -> (N,) share of the 8 waypoint steps on a route lane."""
    return 1.0 - ego_areas[:, WAYPOINT_STEPS, EgoAreaIndex.ONCOMING_TRAFFIC].mean(1)


_CENTRELINES: Dict[str, dict] = {}


def centrelines(map_name):
    if map_name not in _CENTRELINES:
        with open(f"{CENTRELINE_DIR}/{map_name}.pkl", "rb") as f:
            _CENTRELINES[map_name] = pickle.load(f)
    return _CENTRELINES[map_name]


def box_corners(x, y, h):
    """rear-axle poses (...) -> the ego box's 4 corners (..., 4, 2): front left, rear left, rear right, front right."""
    cx, cy = x + REAR_AXLE_TO_CENTER * np.cos(h), y + REAR_AXLE_TO_CENTER * np.sin(h)
    lo = np.array([HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH, HALF_LENGTH])
    la = np.array([HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, -HALF_WIDTH])
    c, s = np.cos(h)[..., None], np.sin(h)[..., None]
    return np.stack([cx[..., None] + lo * c - la * s, cy[..., None] + lo * s + la * c], -1)


def map_aux(poses, ego_pose, drivable_area_map, union, map_name, lines: Optional[dict] = None):
    """poses (N, 8, 3) ego frame (rear axle at t = 0), ego_pose (x, y, h) global -> (N, 8, 6):
    4 corner distances to the drivable boundary, lane-centre offset, heading difference."""
    X0, Y0, H0 = ego_pose
    c0, s0 = np.cos(H0), np.sin(H0)
    gx = X0 + c0 * poses[..., 0] - s0 * poses[..., 1]
    gy = Y0 + s0 * poses[..., 0] + c0 * poses[..., 1]
    gh = wrap(H0 + poses[..., 2])
    out = np.zeros(poses.shape[:2] + (6,))
    out[..., :4] = np.clip(signed_distance(union, box_corners(gx, gy, gh)), -1.0, 2.0)
    lines = centrelines(map_name)["lines"] if lines is None else lines
    px, py, ph = gx.reshape(-1), gy.reshape(-1), gh.reshape(-1)
    lane_idx = [i for i in drivable_area_map.get_indices_of_map_type(LANES) if drivable_area_map.tokens[i] in lines]
    cand = near(drivable_area_map, np.stack([px, py], -1), 10.0)  # lanes within 10 m of the waypoints' bounding box
    lane_idx = [i for i in lane_idx if i in cand] or lane_idx
    if not lane_idx:
        return out
    P = len(px)
    # every candidate lane's centreline as segments
    seg_a, seg_b, seg_lane = [], [], []
    for li, i in enumerate(lane_idx):
        pl = lines[drivable_area_map.tokens[i]]
        if len(pl) < 2:
            continue
        seg_a.append(pl[:-1, :2]); seg_b.append(pl[1:, :2]); seg_lane.append(np.full(len(pl) - 1, li))
    A, B, SL = np.concatenate(seg_a), np.concatenate(seg_b), np.concatenate(seg_lane)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(-1), 1e-9)
    Pt = np.stack([px, py], -1)
    t = np.clip((((Pt[:, None] - A[None]) * AB[None]).sum(-1)) / L2[None], 0, 1)  # (P, S)
    proj = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(Pt[:, None] - proj, axis=-1)
    seg_h = np.arctan2(AB[:, 1], AB[:, 0])
    cross = AB[None, :, 0] * (Pt[:, None, 1] - A[None, :, 1]) - AB[None, :, 1] * (Pt[:, None, 0] - A[None, :, 0])
    nl = len(lane_idx)
    # nearest segment per lane
    dl = np.full((P, nl), np.inf); best_seg = np.zeros((P, nl), np.int64)
    for li in range(nl):
        m = np.where(SL == li)[0]
        if not len(m):
            continue
        j = d[:, m].argmin(1)
        dl[:, li] = d[np.arange(P), m[j]]; best_seg[:, li] = m[j]
    contains = np.stack([shapely.contains_xy(drivable_area_map._geometries[i], px, py) for i in lane_idx], 1)  # (P, nl)
    hdiff_l = np.abs(wrap(ph[:, None] - seg_h[best_seg]))
    hdiff_l[~np.isfinite(dl)] = np.inf
    pick = np.where(contains.any(1), np.where(contains, hdiff_l, np.inf).argmin(1), dl.argmin(1))
    sg = best_seg[np.arange(P), pick]
    off = np.sign(cross[np.arange(P), sg]) * d[np.arange(P), sg]
    out[..., 4] = np.clip(off, -3.5, 3.5).reshape(poses.shape[:2])
    out[..., 5] = wrap(ph - seg_h[sg]).reshape(poses.shape[:2])
    return out
