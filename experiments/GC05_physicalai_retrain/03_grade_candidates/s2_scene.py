"""S2 scene model in the clipgt frame (the usdz's own metric frame, origin = first ego pose): ego, agents, map.
Run in the conda drive-jepa env (nuplan + navsim importable). Read-only on S0/S1 code (imported)."""
import json, math, pathlib, numpy as np, pandas as pd, pyarrow.parquet as pq, shapely
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import s2_common as C
import r3_core as K          # S0 Run 3 core (build_ego, build_tracks): same 10 Hz grid and 1 s-baseline actor speed
POOL = C.S0C.NAS / "pool"; SC = C.NAS / "scenes"
def _xy(lst): return np.array([[p["x"], p["y"]] for p in lst], dtype=float)
def _resample(P, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] <= 0: return np.repeat(P[:1], n, 0)
    u = np.linspace(0, s[-1], n); return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]

def boundary_strips(lanes, d, assoc, max_gap=8.0):
    """S2.1 A1. Each road boundary (clipgt/road_boundary.parquet: curb / road edge polyline, the table AlpaSim loads as RoadEdge,
    trajdata mads_utils.py:302,367) is an OPEN line. It is closed into a polygon against the nearer rail of every lane the map
    associates with it (association kind ROAD_BOUNDARY_TO_LANE): rail section + boundary section reversed. The strip between a lane
    edge and its curb (shoulder, parking strip) thereby becomes drivable. Pairs more than max_gap apart are skipped."""
    from shapely.ops import substring
    rb = {k["map_id"]: _xy(a["location"]) for k, a in zip(*[pq.read_table(d / "clipgt__road_boundary.parquet").to_pandas()[c] for c in ("key", "road_boundary")]) if a["location"] is not None and len(a["location"]) >= 2}
    strips = []; stats = dict(boundaries=len(rb), pairs=0, strips=0, skipped_far=0, skipped_degenerate=0)
    for bid, lids in assoc.get("ROAD_BOUNDARY_TO_LANE", {}).items():
        if bid not in rb: continue
        bl = LineString(rb[bid])
        for lid in lids:
            if lid not in lanes: continue
            stats["pairs"] += 1; l = lanes[lid]; rails = [LineString(l["left"]), LineString(l["right"])]
            rail = min(rails, key=lambda r: r.distance(bl) + 0.01 * np.mean([bl.distance(Point(p)) for p in r.coords]))
            sa, sb = sorted([bl.project(Point(rail.coords[0])), bl.project(Point(rail.coords[-1]))])
            if sb - sa < 1.0: stats["skipped_degenerate"] += 1; continue
            bs = substring(bl, sa, sb); ra, rb_ = sorted([rail.project(Point(bs.coords[0])), rail.project(Point(bs.coords[-1]))])
            if rb_ - ra < 1.0: stats["skipped_degenerate"] += 1; continue
            rs = substring(rail, ra, rb_); gap = max(rs.distance(Point(p)) for p in bs.coords)
            if gap > max_gap: stats["skipped_far"] += 1; continue
            if gap < 0.15: continue                                   # boundary lies on the rail: nothing to add
            R = np.asarray(rs.coords)[:, :2]; B = np.asarray(bs.coords)[:, :2]
            if np.linalg.norm(R[0] - B[0]) + np.linalg.norm(R[-1] - B[-1]) > np.linalg.norm(R[0] - B[-1]) + np.linalg.norm(R[-1] - B[0]): B = B[::-1]
            p = Polygon(np.vstack([R, B[::-1]])); p = p if p.is_valid else p.buffer(0)
            if p.is_empty or p.area < 0.2: stats["skipped_degenerate"] += 1; continue
            strips.append(p); stats["strips"] += 1
    return strips, stats

def bridge_route_gaps(S, max_gap=1.0):
    """S2.1 A2. Gaps of at most max_gap between on-route lane polygons, where the ego path crosses them, are filled: morphological
    closing of the on-route union by max_gap/2, minus the union, kept where it touches the ego-centre path (1.5 m buffer)."""
    U = unary_union([S["lanes"][l]["poly"] for l in S["route_ids"] if l in S["lanes"]]); closed = U.buffer(max_gap / 2).buffer(-max_gap / 2)
    diff = closed.difference(U); parts = [q for q in (diff.geoms if hasattr(diff, "geoms") else [diff]) if (not q.is_empty) and q.geom_type == "Polygon" and q.area >= 0.02]
    ctr = S["xy"] + 1.461 * np.c_[np.cos(S["yaw"]), np.sin(S["yaw"])]; path = LineString(ctr).buffer(1.5) if len(ctr) > 1 and np.ptp(ctr, axis=0).max() > 0.5 else Point(ctr[0]).buffer(1.5)
    out = []
    for q in parts:
        if not q.intersects(path): continue
        w = 2 * q.area / max(q.length, 1e-9)                      # mean width of a thin sliver (area / half-perimeter)
        out.append(dict(poly=q, width=float(w), length=float(q.length / 2), area=float(q.area), on_centre_line=bool(q.intersects(LineString(ctr))) if len(ctr) > 1 else False))
    S["bridges"] = out; return out

def load_scene(u, fix=False):
    d0 = POOL / u; d = SC / u
    t = pq.read_table(d0 / "clipgt__egomotion_estimate.parquet").flatten().flatten(); g = lambda n: t.column(n).to_numpy(zero_copy_only=False)
    ts = g("key.timestamp_micros").astype(np.int64); o = np.argsort(ts)
    xy = np.c_[g("egomotion_estimate.location.x"), g("egomotion_estimate.location.y")]; yaw = K.yaw_q(*[g(f"egomotion_estimate.orientation.{k}") for k in "xyzw"])
    S = K.build_ego(ts / 1e6, xy, yaw); S["ts_abs_us"] = ts[o]; S["raw_xy"] = xy[o]; S["raw_yaw"] = np.unwrap(yaw[o])
    ob = pq.read_table(d0 / "clipgt__obstacle.parquet").flatten().flatten(); h = lambda n: ob.column(n).to_numpy(zero_copy_only=False)
    S = K.build_tracks(S, pd.DataFrame({"tid": ob.column("obstacle.trackline_id").to_pylist(), "ts": h("key.timestamp_micros").astype(np.int64) / 1e6 - S["t0"], "x": h("obstacle.center.x"), "y": h("obstacle.center.y"),
                       "yaw": K.yaw_q(*[h(f"obstacle.orientation.{k}") for k in "xyzw"]), "L": h("obstacle.size.x"), "W": h("obstacle.size.y"), "cat": ob.column("obstacle.category").to_pylist()}))
    # ---- map
    lanes = {}
    for k, l in zip(*[pq.read_table(d / "clipgt__lane.parquet").to_pandas()[c] for c in ("key", "lane")]):
        if l["left_rail"] is None or len(l["left_rail"]) < 2 or len(l["right_rail"]) < 2: continue
        L, R = _xy(l["left_rail"]), _xy(l["right_rail"]); n = max(len(L), len(R), 2) * 4; cen = (_resample(L, n) + _resample(R, n)) / 2
        poly = Polygon(np.vstack([L, R[::-1]]))
        if not poly.is_valid: poly = poly.buffer(0)
        if poly.is_empty or poly.area < 0.5: continue
        if poly.geom_type != "Polygon": poly = max(poly.geoms, key=lambda q: q.area)
        lanes[k["map_id"]] = dict(left=L, right=R, center=cen, poly=poly, direction=l["lane_direction"], speed_limit=float(l["speed_limit"] or 0), vehicle_types=list(l["vehicle_types"]), length=float(np.linalg.norm(np.diff(cen, axis=0), axis=1).sum()))
    assoc = {}
    for k, a in zip(*[pq.read_table(d / "clipgt__association.parquet").to_pandas()[c] for c in ("key", "association")]):
        if k["kind"] is None: continue
        assoc.setdefault(k["kind"], {}).setdefault(k["map_id"], []).extend([x for x in a["objects"] if x != "-1"])
    inter = []
    for k, a in zip(*[pq.read_table(d / "clipgt__intersection_area.parquet").to_pandas()[c] for c in ("key", "intersection_area")]):
        if a["location"] is None or len(a["location"]) < 3: continue
        p = Polygon(_xy(a["location"])); p = p if p.is_valid else p.buffer(0)
        if not p.is_empty: inter.append((k["map_id"], p if p.geom_type == "Polygon" else max(p.geoms, key=lambda q: q.area), a["category"]))
    wait = [(k["map_id"], _xy(a["location"]), a["category"], a["intersection_subtype"], a["is_implicit"]) for k, a in zip(*[pq.read_table(d / "clipgt__wait_line.parquet").to_pandas()[c] for c in ("key", "wait_line")]) if a["location"] is not None and len(a["location"]) >= 2]
    tl = [(a["center"]["x"], a["center"]["y"]) for a in pq.read_table(d / "clipgt__traffic_light.parquet").to_pandas()["traffic_light"] if a["center"]["x"] is not None]
    conn = set(x for v in assoc.get("INTERSECTION_AREA_TO_LANE", {}).values() for x in v)
    S.update(uuid=u, lanes=lanes, assoc=assoc, inter=inter, wait=wait, tl=tl, connectors=conn)
    # drivable area: union of all lane polygons and intersection areas, morphologically closed by 0.3 m (fills hairline gaps between
    # neighbouring lane polygons; closing does not grow the outline)
    base = [l["poly"] for l in lanes.values()] + [p for _, p, _ in inter]; S["fix"] = fix; S["bridges"] = []; S["strip_stats"] = {}
    U0 = unary_union(base).buffer(0.3, join_style=2).buffer(-0.3, join_style=2); S["drivable_s2_area"] = float(U0.area)
    if fix:
        strips, S["strip_stats"] = boundary_strips(lanes, d, assoc); S["strips"] = strips
        U = unary_union(base + strips).buffer(0.3, join_style=2).buffer(-0.3, join_style=2)
    else: U = U0
    S["drivable"] = [q for q in (U.geoms if U.geom_type == "MultiPolygon" else [U]) if q.area > 1.0]
    S["lane_tree"] = shapely.STRtree([l["poly"] for l in lanes.values()]); S["lane_ids"] = list(lanes.keys())
    return S

def lane_heading_at(l, p):
    c = l["center"]; i = int(np.argmin(np.linalg.norm(c - p, axis=1))); i = min(max(i, 0), len(c) - 2); v = c[i + 1] - c[i]
    return math.atan2(v[1], v[0]), float(LineString(c).distance(Point(p)))

def lane_at(S, p, yaw, max_dist=6.0):
    """Lane the ego is on: contains p (else nearest within max_dist), heading within 45 deg of the ego, smallest centre distance."""
    best = None; P = Point(p)
    cand = [S["lane_ids"][i] for i in S["lane_tree"].query(P.buffer(max_dist))]
    for inside_only in (True, False):
        for lid in cand:
            l = S["lanes"][lid]
            if inside_only and not l["poly"].contains(P): continue
            h, dc = lane_heading_at(l, p)
            if abs(K.wrap(h - yaw)) > math.radians(45): continue
            if best is None or dc < best[1]: best = (lid, dc)
        if best: return best[0]
    return None

def route_lanes(S):
    """Human route for the whole clip: lanes the human occupies in time order, plus their same-direction road-segment siblings
    (ROAD_SEGMENT_SIBLING_LANE = the roadblock analogue), plus successors beyond the clip end."""
    seq = []
    for i in range(0, len(S["t"]), 2):
        lid = lane_at(S, S["xy"][i], S["yaw"][i])
        if lid and (not seq or seq[-1] != lid): seq.append(lid)
    sib = S["assoc"].get("ROAD_SEGMENT_SIBLING_LANE", {}); nxt = S["assoc"].get("NEXT_LANE", {})
    ids = set(seq); tail = seq[-1] if seq else None; ext = 0.0
    while tail and ext < 150 and nxt.get(tail):
        tail = [x for x in nxt[tail] if x in S["lanes"]][:1]; tail = tail[0] if tail else None
        if tail is None or tail in ids: break
        ids.add(tail); ext += S["lanes"][tail]["length"]
    for lid in list(ids): ids.update(x for x in sib.get(lid, []) if x in S["lanes"])
    S["route_seq"] = seq; S["route_ids"] = ids
    if S.get("fix"):
        for k, br in enumerate(bridge_route_gaps(S)): br["token"] = f"bridge_{k}"; ids.add(br["token"])
        if S["bridges"]:
            U = unary_union(S["drivable"] + [b["poly"] for b in S["bridges"]]); S["drivable"] = [q for q in (U.geoms if U.geom_type == "MultiPolygon" else [U]) if q.area > 1.0]
    return seq, ids

def centerline_at(S, i, back=30.0, ahead=200.0):
    """Per-anchor centreline, nuPlan-style: the ego's current lane from its start, then successors that stay on the human route
    (preferring the lane the human actually took), independent of later lane changes."""
    seq, ids = S["route_seq"], S["route_ids"]; cur = lane_at(S, S["xy"][i], S["yaw"][i])
    if cur is None: return None, None
    nxt = S["assoc"].get("NEXT_LANE", {}); prv = S["assoc"].get("PREVIOUS_LANE", {}); chain = [cur]; length = S["lanes"][cur]["length"]; seen = {cur}
    order = {l: k for k, l in enumerate(seq)}
    while length < ahead:
        c = [x for x in nxt.get(chain[-1], []) if x in S["lanes"] and x not in seen]
        if not c: break
        c.sort(key=lambda x: (0 if x in order else 1, 0 if x in ids else 1)); chain.append(c[0]); seen.add(c[0]); length += S["lanes"][c[0]]["length"]
    pre = [x for x in prv.get(cur, []) if x in S["lanes"]]; pre.sort(key=lambda x: 0 if x in order else 1)
    pts = ([S["lanes"][pre[0]]["center"]] if pre else []) + [S["lanes"][l]["center"] for l in chain]; P = np.vstack(pts)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-3]; P = P[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; u = np.arange(0, s[-1], 0.5); P = np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]
    return P, cur
