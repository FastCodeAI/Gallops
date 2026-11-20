"""gpulab/mapfacts: run 3B's map-based live labels on the GPU (gpulab/CONTRACT.md), batched over B samples x P paths, float64.

CPU reference (read, never edited): facts.scene_facts (offroad, lane_flags -> wrong_way / left_route, run 3A's map_aux on the
centre poses -> lane_off, metres, stop lines -> red / stop, per_step / latch), the scorer's _calculate_ego_area (ego areas),
carla_full_v1.apply_rules (CARLA: off road counts from the first state with all four corners inside), run 3A's get_sub_score
(NAVSIM aux: map_aux on the poses, ego_areas[:, 1:, 1:]).

  extract_map(kind, ctx) -> dict of numpy arrays            CPU, in the process that holds ctx (= labeller._nav_env() /
                                                              _carla_env() context(meta)); pose-independent
  collate_map(scenes, device) -> tables on the GPU            (label_map calls it when T is not given)
  label_map(kind, scenes, states, coords, poses=None, T=None) -> dict of cuda tensors
        states (B, P, 41, 11), coords (B, P, 41, 5, 2) float64 cuda (= before_score_proposals rows 1.., scorer._ego_coords);
        poses (B, P, 8, 3) float64 (the paths as the labeller got them: float64 of the float32 sent) are needed only for the
        aux map_aux; given poses, both kinds return aux (CARLA aux added 19 Nov, see 'CARLA aux' below).
        -> offroad, wrong_way, left_route, stop (B, P, 8) bool; red (B, P, 8) int8; metres, lane_off (B, P, 8) float64;
           ego_areas (B, P, 41, 3) bool (EgoAreaIndex order: MULTIPLE_LANES, NON_DRIVABLE_AREA, ONCOMING_TRAFFIC);
           with poses: aux = dict(map_aux (B, P, 8, 6) float64, ego_areas (B, P, 40, 2) bool) (NAVSIM; CARLA too since 19 Nov);
           diag = int tensor counters (all 0 = no case outside what is matched; see DIAG).
  as_labeller_dtypes(out) -> numpy in the labeller's dtypes (metres float32, lane_off float16, aux map_aux float32).

How each CPU predicate is matched (GPU work in Triton kernels, float64, fp fusion off so a*b+c rounds as numpy does; no host
synchronisation inside label_map except the NAVSIM aux boxes)
  point in polygon   shapely contains_xy / vectorized.contains on GEOS 3.11 (non-prepared: PointLocator per ring with GEOS's
                     RayCrossingCounter: half-open straddle, sign of orientationIndexFilter's determinant, a point on a ring
                     = boundary = not contained). Same rule per ring (shell odd, no hole odd, no boundary hit), after the
                     envelope test GEOS also makes. Candidates: a 4 m grid of polygon envelopes; edges: per-polygon horizontal
                     bands of BAND_H m (an edge is listed in every band its y-range touches). GEOS decides near-collinear
                     orientation exactly (double-double); here double: a point within ~1e-12 m of an edge may differ.
                     Horizontal edges are not listed (GEOS only uses them to call a point lying exactly on one 'boundary').
  centrelines        run 3A's map_aux / lane_flags formulas op for op, first minimum on ties, numpy's segment headings
                     (arctan2 precomputed on the CPU); candidate lanes = envelope within 10 m of the box of all the call's
                     points (facts: PDM-Closed row 0's 8 centre points included, precomputed in extract).
  route projection   GEOS LengthIndexOfPoint (shapely project): first segment at the minimum Distance::pointToSegment,
                     measure = start + projectionFactor x length (start = sequential sum of the GEOS segment lengths).
  drivable distance  NAVSIM aux only (run3a_labels.drivable_union + signed_distance): the union of the ROADBLOCK /
                     INTERSECTION / DRIVABLE_AREA / CARPARK_AREA polygons whose envelope meets the box of all ego corners
                     (rows 0 .. P, 41 states) grown by 5 m. That union depends on the paths, and GEOS's union_all leaves
                     or drops zero-area sliver holes along shared edges depending on which polygons go in, so it is not
                     precomputed: label_map builds the same union on the CPU per NAVSIM sample (shapely.union_all of the same
                     polygons in map order, rebuilt from the scene's vertices; ~0.5 ms each; one host synchronisation for the
                     boxes), then the GPU takes the distance to its rings (GEOS pointToSegment, minimum over all ring
                     segments) and the inside test (GEOS RayCrossingCounter over all rings, as the prepared contains),
                     clipped [-1, 2].
  CARLA aux          (19 Nov, lead_aux/map_ego/build.aux_from_mc = get_sub_score's lines 65-84 in a CARLA labeller process:
                     MKZ corners, CARLA map conversion, row 0 = LEAD's logged future through the car model.) Same union rule
                     as NAVSIM; the CARLA union boundary is long (lane strips at 0.1-0.5 m), so: corners in numpy exactly as
                     run3a_labels.map_aux / box_corners, inside = shapely contains_xy on the prepared union (the CPU's own
                     call), distance on the GPU over the union edges whose envelope comes within 2 m of the sample's corners
                     (same clipped value: a nearer edge would be within 2 m). One host synchronisation (boxes + poses).
  stop lines         facts.stop_lines (extract), facts.crossings / light_colour (colour per line and 2 Hz frame j = 0..8
                     precomputed), latch rules of scene_facts.
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import triton
import triton.language as tl

F64 = torch.float64
K, NS = 41, 8
WP = np.arange(5, 41, 5)
STEP_OF = np.r_[0, np.repeat(np.arange(NS), 5)]
BAND_H = 0.25
PGRID = 4.0
KMAX = 32
RB = 32  # route segments per pruning block
REACH_C = 2.0 + 1e-6  # CARLA aux: union edges kept within this of the corners (distances are clipped at 2 m)
AUX_THREADS = int(os.environ.get("MAPFACTS_AUX_THREADS", "1"))  # CARLA aux CPU unions: threads (GEOS releases the GIL)
_POOL = []
WRONG_WAY = float(np.radians(135.0))
COMPAT = float(np.radians(90.0))
STOP_SPEED, LAT_MARGIN = 0.1, 0.5
FL_LANE, FL_DRIVE, FL_ROUTE, FL_START = 1, 2, 4, 8
DIAG = ("pip_overflow", "lane_off_no_segment_pick", "aux_no_segment_pick", "pip_max_count")
BLOCK = 64


# ================================================================================================================ extract (CPU)
def _facts():
    F = sys.modules["facts"]  # facts.py (NAVSIM) or the same module after carla_facts' patches (CARLA process)
    return F, F.r3, F.cns


def _rings(geoms):
    """polygons -> ring vertices (nv, 2) incl. each ring's closing point, ring index within its polygon per vertex (nv,),
    owner polygon per vertex (nv,), edge-start flag (nv,) (False on each ring's last vertex), rings per polygon."""
    import shapely
    geoms = np.asarray(geoms, dtype=object)
    rings, rown = shapely.get_rings(geoms, return_index=True)
    nring = np.bincount(rown, minlength=len(geoms))
    first = np.r_[0, np.cumsum(nring)[:-1]]
    rloc = np.arange(len(rings)) - first[rown]
    xy, ci = shapely.get_coordinates(rings, return_index=True)
    start = np.r_[ci[:-1] == ci[1:], False]
    return xy.astype(np.float64), rloc[ci].astype(np.int32), rown[ci], start, nring


def _bands(vxy, owner, start, bbox):
    """per-polygon horizontal bands: band b of polygon i covers y in [ymin_i + b h, ymin_i + (b + 1) h); an edge (vertex e ->
    e + 1, not horizontal) is listed in every band from its lowest to its highest y. -> band0 (n_poly,), off, entries
    (edge-start vertex ids, sorted by band then ring order)."""
    y0 = bbox[:, 1]
    nb = np.floor((bbox[:, 3] - y0) / BAND_H).astype(np.int64) + 1 if len(bbox) else np.zeros(0, np.int64)
    band0 = np.r_[0, np.cumsum(nb)[:-1]].astype(np.int64) if len(bbox) else np.zeros(0, np.int64)
    e = np.nonzero(start)[0]
    e = e[vxy[e, 1] != vxy[e + 1, 1]]
    o = owner[e]
    ya, yb = vxy[e, 1], vxy[e + 1, 1]
    lo = np.floor((np.minimum(ya, yb) - y0[o]) / BAND_H).astype(np.int64)
    hi = np.floor((np.maximum(ya, yb) - y0[o]) / BAND_H).astype(np.int64)
    cnt = hi - lo + 1
    tot = int(cnt.sum())
    rep = np.repeat(e, cnt)
    band = np.repeat(band0[o] + lo, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    order = np.argsort(band, kind="stable")
    off = np.searchsorted(band[order], np.arange(int(nb.sum()) + 1))
    return band0, off.astype(np.int64), rep[order].astype(np.int64)


def _grid(x0, y0, x1, y1, g, ids=None):
    """boxes (n,) -> grid (origin, n cells, CSR cell -> box ids ascending) registering each box in every cell it touches."""
    n = len(x0)
    if not n:
        return np.zeros(2), np.zeros(2, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int64)
    ox, oy = float(x0.min()) - g, float(y0.min()) - g
    nx = int(np.floor((x1.max() - ox) / g)) + 2
    ny = int(np.floor((y1.max() - oy) / g)) + 2
    ix0, ix1 = np.floor((x0 - ox) / g).astype(np.int64), np.floor((x1 - ox) / g).astype(np.int64)
    iy0, iy1 = np.floor((y0 - oy) / g).astype(np.int64), np.floor((y1 - oy) / g).astype(np.int64)
    cx, cy = ix1 - ix0 + 1, iy1 - iy0 + 1
    cnt = cx * cy
    tot = int(cnt.sum())
    ids = np.arange(n) if ids is None else ids
    sid = np.repeat(ids, cnt)
    w = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cell = (np.repeat(iy0, cnt) + w // np.repeat(cx, cnt)) * nx + np.repeat(ix0, cnt) + w % np.repeat(cx, cnt)
    order = np.lexsort((sid, cell))
    off = np.searchsorted(cell[order], np.arange(nx * ny + 1))
    return np.array([ox, oy]), np.array([nx, ny], np.int64), off.astype(np.int64), sid[order].astype(np.int64)


def extract_map(kind, ctx):
    """ctx = labeller context (mc, smp, frames, ...) -> compact numpy dict, independent of the paths."""
    import shapely
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    F, r3, cns = _facts()
    mc = ctx["mc"]
    dam = mc.drivable_area_map
    ra = mc.ego_state.rear_axle
    out = dict(kind=kind, x0=float(ra.x), y0=float(ra.y), h0=float(ra.heading), c0=float(np.cos(ra.heading)),
               s0=float(np.sin(ra.heading)), r2c=float(r3.REAR_AXLE_TO_CENTER), front=float(F.FRONT),
               hl=float(r3.HALF_LENGTH), hw=float(r3.HALF_WIDTH))
    # ------------------------------------------------------------------ PDM-Closed / logged row 0 (pose-independent)
    st0 = cns.before_score_proposals(mc, np.zeros((0, 8, 3)), mc.ego_state)
    co0 = state_array_to_coords_array(st0, cns.scorer._vehicle_parameters)
    hwp = st0[:, WP, 2]
    cg = st0[:, WP, :2] + r3.REAR_AXLE_TO_CENTER * np.stack([np.cos(hwp), np.sin(hwp)], -1)
    ce = F.EgoFrame(ra.x, ra.y, ra.heading).to_ego(cg)
    X0, Y0, H0 = ra.x, ra.y, ra.heading
    c0, s0 = np.cos(H0), np.sin(H0)
    gx = X0 + c0 * ce[..., 0] - s0 * ce[..., 1]
    gy = Y0 + s0 * ce[..., 0] + c0 * ce[..., 1]
    out["row0_wpbox"] = np.array([gx.min(), gy.min(), gx.max(), gy.max()])
    c4 = co0[0, :, :4].reshape(-1, 2)
    out["row0_cbox"] = np.r_[c4.min(0), c4.max(0)]
    out["row0_coords"] = co0[0]
    start = co0[0, 0, 4]
    # ------------------------------------------------------------------ polygons
    geoms = list(dam._geometries)
    lanes_t = (L.LANE, L.LANE_CONNECTOR)
    drive_t = (L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA)
    route = set(mc.route_lane_ids)
    lines = r3.centrelines(mc.map_parameters.map_name)["lines"]
    n = len(geoms)
    flag = np.zeros(n, np.int32)
    pline = np.full(n, -1, np.int32)
    lv, tok_line = [], {}
    for i, (tok, ty) in enumerate(zip(dam.tokens, dam.map_types)):
        if ty in lanes_t:
            flag[i] |= FL_LANE
            if tok in route:
                flag[i] |= FL_ROUTE
            if tok in lines:
                if tok not in tok_line:
                    tok_line[tok] = len(lv)
                    lv.append(np.asarray(lines[tok], np.float64)[:, :3])
                pline[i] = tok_line[tok]
            if shapely.contains_xy(geoms[i], start[0], start[1]):  # lane_flags' 'start': lanes holding row 0's centre at t = 0
                flag[i] |= FL_START
        if ty in drive_t:
            flag[i] |= FL_DRIVE
    assert set(r3.DRIVABLE) == set(drive_t)  # run 3A's drivable union uses the scorer's drivable layers
    if geoms:
        bbox = shapely.bounds(np.asarray(geoms, dtype=object)).astype(np.float64)
        vxy, vring, vown, vstart, nring = _rings(geoms)
    else:
        bbox, vxy, vring, vown = np.zeros((0, 4)), np.zeros((0, 2)), np.zeros(0, np.int32), np.zeros(0, np.int64)
        vstart, nring = np.zeros(0, bool), np.zeros(0, np.int64)
    band0, boff, bent = _bands(vxy, vown, vstart, bbox)
    go, gn, goff, gent = _grid(bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3], PGRID)
    voff = np.searchsorted(vown, np.arange(len(geoms) + 1)).astype(np.int32)
    out["poly_token"] = list(dam.tokens)
    out.update(poly_bbox=bbox, poly_flag=flag, poly_line=pline, poly_band0=band0.astype(np.int32), vxy=vxy, vring=vring, poly_voff=voff,
               band_off=boff.astype(np.int32), band_ent=bent.astype(np.int32),
               pg_origin=go, pg_n=gn.astype(np.int32), pg_off=goff.astype(np.int32), pg_ent=gent.astype(np.int32))
    # ------------------------------------------------------------------ centrelines (x, y, heading) + numpy segment headings
    if lv:
        loff = np.r_[0, np.cumsum([len(v) for v in lv])].astype(np.int32)
        V = np.concatenate(lv)
        segh = np.zeros(len(V))
        for a, b in zip(loff[:-1], loff[1:]):
            if b - a >= 2:
                AB = V[a + 1:b, :2] - V[a:b - 1, :2]
                segh[a:b - 1] = np.arctan2(AB[:, 1], AB[:, 0])
    else:
        loff, V, segh = np.zeros(1, np.int32), np.zeros((0, 3)), np.zeros(0)
    out.update(line_v=V, line_segh=segh, line_off=loff)
    # ------------------------------------------------------------------ route centreline (PDMPath linestring), GEOS lengths
    rc = shapely.get_coordinates(mc.centerline.linestring).astype(np.float64)
    d = rc[:-1] - rc[1:]
    rlen = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
    rstart = np.r_[0.0, np.add.accumulate(rlen)[:-1]] if len(rlen) else np.zeros(0)
    nseg = max(len(rc) - 1, 0)
    rbb = np.array([np.r_[rc[a:min(a + RB, nseg) + 1].min(0), rc[a:min(a + RB, nseg) + 1].max(0)] for a in range(0, nseg, RB)]).reshape(-1, 4)
    out.update(route=rc, route_len=np.r_[rlen, 0.0], route_start=np.r_[rstart, 0.0], route_bb=rbb)
    # ------------------------------------------------------------------ stop lines, colour per 2 Hz frame j
    smp, frames = ctx.get("smp"), ctx.get("frames")
    lines_ = F.stop_lines(smp)[0] if smp is not None else []
    nl = len(lines_)
    sl = dict(sl_c=np.zeros((nl, 2)), sl_u=np.zeros((nl, 2)), sl_n=np.zeros((nl, 2)), sl_uh=np.zeros(nl), sl_smax=np.zeros(nl),
              sl_lmin=np.zeros(nl), sl_lmax=np.zeros(nl), sl_sign=np.zeros(nl, bool), sl_col=np.zeros((nl, 9), np.int8))
    for k, ln in enumerate(lines_):
        sl["sl_c"][k], sl["sl_u"][k], sl["sl_n"][k] = ln["c"], ln["u"], ln["n"]
        sl["sl_uh"][k] = np.arctan2(ln["u"][1], ln["u"][0])
        sl["sl_smax"][k], sl["sl_lmin"][k], sl["sl_lmax"][k] = ln["s_max"], ln["l_min"], ln["l_max"]
        sl["sl_sign"][k] = ln["type"] == "STOP_SIGN" and not ln["governing"]
        if not sl["sl_sign"][k]:
            for j in range(9):
                col = F.light_colour(ln["id"], j, smp, frames)[0]
                sl["sl_col"][k, j] = 1 if col == "red" else (-1 if col == "unknown" else 0)
    out.update(sl)
    out["stats"] = dict(n_poly=len(geoms),
                        n_vertices=len(vxy), n_band_entries=len(bent), n_grid_entries=len(gent), n_lines=len(lv),
                        n_short_lines=int(sum(len(v) < 2 for v in lv)), n_stop_lines=nl, n_holes=int((nring > 1).sum()))
    return out


# ================================================================================================================ collate
class _Packer:
    """many small numpy arrays -> one float64 and one int32 buffer, one host-to-device copy each, then views."""

    def __init__(self):
        self.f, self.i, self.fn, self.inn = [], [], 0, 0
        self.views = {}

    def add(self, name, a, kind):
        a = np.ascontiguousarray(a, np.float64 if kind == "f" else np.int32)
        if kind == "f":
            self.views[name] = ("f", self.fn, a.shape)
            self.f.append(a.reshape(-1)); self.fn += a.size
        else:
            self.views[name] = ("i", self.inn, a.shape)
            self.i.append(a.reshape(-1)); self.inn += a.size

    def to(self, device, T):
        fb = torch.from_numpy(np.concatenate(self.f) if self.f else np.zeros(0)).to(device)
        ib = torch.from_numpy(np.concatenate(self.i) if self.i else np.zeros(0, np.int32)).to(device)
        for name, (k, o, shp) in self.views.items():
            n = int(np.prod(shp)) if len(shp) else 1
            setattr(T, name, (fb if k == "f" else ib)[o:o + n].view(shp))


def _pad(arrs, fill, dtype, tail=()):
    n = max([len(a) for a in arrs] + [1])
    out = np.full((len(arrs), n) + tail, fill, dtype)
    for i, a in enumerate(arrs):
        if len(a):
            out[i, :len(a)] = a
    return out


def collate_map(scenes, device):
    """list of extract_map dicts -> concatenated tables on device."""
    B = len(scenes)
    T = SimpleNamespace(B=B, kind=scenes[0]["kind"])
    pk = _Packer()

    def base(key, f=len):
        n = np.array([f(s[key]) for s in scenes], np.int64)
        return np.r_[0, np.cumsum(n)[:-1]], n
    pb, npoly = base("poly_flag")
    vb, _ = base("vxy")
    bb, _ = base("band_off", lambda a: len(a) - 1)
    eb, _ = base("band_ent")
    cb, _ = base("pg_off", lambda a: len(a) - 1)
    gb, _ = base("pg_ent")
    lb, _ = base("line_off", lambda a: len(a) - 1)
    lvb, nlv = base("line_v")
    rb, nr = base("route")
    cat = np.concatenate
    pk.add("pstart", pb, "i"); pk.add("npoly", npoly, "i")
    bbox = cat([s["poly_bbox"] for s in scenes])
    for j, nm in enumerate(("bx0", "by0", "bx1", "by1")):
        pk.add(nm, bbox[:, j], "f")
    pk.add("pflag", cat([s["poly_flag"] for s in scenes]), "i")
    pk.add("pline", cat([np.where(s["poly_line"] >= 0, s["poly_line"] + o, -1) for s, o in zip(scenes, lb)]), "i")
    pk.add("band0", cat([s["poly_band0"] + o for s, o in zip(scenes, bb)]), "i")
    pk.add("boff", np.r_[cat([s["band_off"][:-1] + o for s, o in zip(scenes, eb)]), sum(len(s["band_ent"]) for s in scenes)], "i")
    pk.add("bent", cat([s["band_ent"] + o for s, o in zip(scenes, vb)]), "i")
    vxy = cat([s["vxy"] for s in scenes])
    pk.add("vx", vxy[:, 0], "f"); pk.add("vy", vxy[:, 1], "f")
    pk.add("vring", cat([s["vring"] for s in scenes]), "i")
    pk.add("g_ox", np.array([s["pg_origin"][0] for s in scenes]), "f"); pk.add("g_oy", np.array([s["pg_origin"][1] for s in scenes]), "f")
    pk.add("g_nx", np.array([s["pg_n"][0] for s in scenes]), "i"); pk.add("g_ny", np.array([s["pg_n"][1] for s in scenes]), "i")
    pk.add("g_base", cb, "i")
    pk.add("g_off", np.r_[cat([s["pg_off"][:-1] + o for s, o in zip(scenes, gb)]), sum(len(s["pg_ent"]) for s in scenes)], "i")
    pk.add("g_ent", cat([s["pg_ent"] + o for s, o in zip(scenes, pb)]), "i")
    lv = cat([s["line_v"] for s in scenes]) if nlv.sum() else np.zeros((1, 3))
    pk.add("lvx", lv[:, 0], "f"); pk.add("lvy", lv[:, 1], "f"); pk.add("lvh", lv[:, 2], "f")
    pk.add("lsegh", cat([s["line_segh"] for s in scenes]) if nlv.sum() else np.zeros(1), "f")
    pk.add("loff", np.r_[cat([s["line_off"][:-1] + o for s, o in zip(scenes, lvb)]), int(nlv.sum())], "i")
    rv = cat([s["route"] for s in scenes])
    pk.add("rvx", rv[:, 0], "f"); pk.add("rvy", rv[:, 1], "f")
    pk.add("rlen", cat([s["route_len"] for s in scenes]), "f"); pk.add("rst", cat([s["route_start"] for s in scenes]), "f")
    pk.add("r0", rb, "i"); pk.add("r1", rb + nr - 1, "i")  # segments: vertex k -> k + 1 for k in [r0, r1)
    qb, nq = base("route_bb")
    rbb = cat([s["route_bb"] for s in scenes]) if nq.sum() else np.zeros((1, 4))
    pk.add("q0", qb, "i"); pk.add("q1", qb + nq, "i")
    for j, nm in enumerate(("qx0", "qy0", "qx1", "qy1")):
        pk.add(nm, rbb[:, j], "f")
    for k_, fill, dt, tail in (("sl_c", np.nan, "f", (2,)), ("sl_u", np.nan, "f", (2,)), ("sl_n", np.nan, "f", (2,)),
                               ("sl_uh", np.nan, "f", ()), ("sl_smax", np.nan, "f", ()), ("sl_lmin", np.nan, "f", ()),
                               ("sl_lmax", np.nan, "f", ()), ("sl_sign", 0, "i", ()), ("sl_col", 0, "i", (9,))):
        pk.add(k_, _pad([s[k_] for s in scenes], fill, np.float64 if dt == "f" else np.int32, tail), dt)
    pk.add("sl_valid", _pad([np.ones(len(s["sl_smax"])) for s in scenes], 0, np.int32), "i")
    for k_ in ("x0", "y0", "h0", "c0", "s0"):
        pk.add(k_, np.array([s[k_] for s in scenes]), "f")
    pk.add("row0_wpbox", np.stack([s["row0_wpbox"] for s in scenes]), "f")
    pk.add("row0_cbox", np.stack([s["row0_cbox"] for s in scenes]), "f")
    T.r2c, T.front, T.hl, T.hw = scenes[0]["r2c"], scenes[0]["front"], scenes[0]["hl"], scenes[0]["hw"]
    pk.to(device, T)
    T.NP = int(npoly.sum())
    T.sl_sign, T.sl_valid = T.sl_sign.bool(), T.sl_valid.bool()
    return T


# ================================================================================================================ kernels
@triton.jit
def _wrap_t(a):
    r = (a + 3.141592653589793) % 6.283185307179586
    r = tl.where(r < 0, r + 6.283185307179586, r)
    r = tl.where(r == 0, 0.0, r)
    return r - 3.141592653589793


@triton.jit
def _k_pip(px, py, psamp, pset, N, g_ox, g_oy, g_nx, g_ny, g_base, g_off, g_ent, bx0, by0, bx1, by1, setmask, NP,
           band0, boff, bent, vx, vy, vring, out, ocnt, KM: tl.constexpr, BH: tl.constexpr, G: tl.constexpr,
           BLOCK: tl.constexpr):
    """per point: every polygon of its set mask that contains it (GEOS contains), in polygon order, up to KM."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(px + i, mask=m, other=0.0)
    y = tl.load(py + i, mask=m, other=0.0)
    b = tl.load(psamp + i, mask=m, other=0)
    s = tl.load(pset + i, mask=m, other=0)
    ox = tl.load(g_ox + b, mask=m, other=0.0)
    oy = tl.load(g_oy + b, mask=m, other=0.0)
    nx = tl.load(g_nx + b, mask=m, other=0)
    ny = tl.load(g_ny + b, mask=m, other=0)
    cxf = tl.floor((x - ox) / G)
    cyf = tl.floor((y - oy) / G)
    inb = m & (cxf >= 0) & (cyf >= 0) & (cxf < nx.to(tl.float64)) & (cyf < ny.to(tl.float64))
    cell = tl.load(g_base + b, mask=m, other=0) + tl.where(inb, cyf, 0.0).to(tl.int32) * nx + tl.where(inb, cxf, 0.0).to(tl.int32)
    c0 = tl.load(g_off + cell, mask=inb, other=0)
    c1 = tl.load(g_off + cell + 1, mask=inb, other=0)
    nc = tl.where(inb, c1 - c0, 0)
    maxc = tl.max(nc, axis=0)
    cnt = tl.zeros([BLOCK], dtype=tl.int32)
    for c in range(0, maxc):
        act = c < nc
        gp = tl.load(g_ent + c0 + c, mask=act, other=0)
        x0 = tl.load(bx0 + gp, mask=act, other=0.0)
        y0 = tl.load(by0 + gp, mask=act, other=0.0)
        x1 = tl.load(bx1 + gp, mask=act, other=0.0)
        y1 = tl.load(by1 + gp, mask=act, other=0.0)
        sm = tl.load(setmask + s * NP + gp, mask=act, other=0)
        ok = act & (sm != 0) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        band = tl.floor((y - y0) / BH)
        bid = tl.load(band0 + gp, mask=ok, other=0) + tl.where(ok, band, 0.0).to(tl.int32)
        e0 = tl.load(boff + bid, mask=ok, other=0)
        e1 = tl.load(boff + bid + 1, mask=ok, other=0)
        ne = tl.where(ok, e1 - e0, 0)
        maxe = tl.max(ne, axis=0)
        ext = tl.zeros([BLOCK], dtype=tl.int32)   # crossing parity of the exterior ring (ring 0)
        hol = tl.zeros([BLOCK], dtype=tl.int32)   # crossing parity summed over the holes: holes of a valid polygon are disjoint,
        bnd = tl.zeros([BLOCK], dtype=tl.int32)   # so a point is in a hole iff exactly one hole's parity is odd = XOR odd
        for k in range(0, maxe):
            a2 = k < ne
            ev = tl.load(bent + e0 + k, mask=a2, other=0)
            ax = tl.load(vx + ev, mask=a2, other=0.0)
            ay = tl.load(vy + ev, mask=a2, other=0.0)
            bxv = tl.load(vx + ev + 1, mask=a2, other=0.0)
            byv = tl.load(vy + ev + 1, mask=a2, other=0.0)
            r = tl.load(vring + ev, mask=a2, other=0)
            # GEOS RayCrossingCounter::countSegment(p1 = edge end, p2 = edge start)
            notleft = (bxv >= x) | (ax >= x)
            vert = notleft & (x == ax) & (y == ay)
            strad = ((byv > y) & (ay <= y)) | ((ay > y) & (byv <= y))
            det = (bxv - x) * (ay - y) - (byv - y) * (ax - x)
            act2 = a2 & notleft & strad & (vert == 0)
            cross = act2 & (det != 0) & ((det > 0) != (ay < byv))
            ext = ext ^ tl.where(cross & (r == 0), 1, 0)
            hol = hol ^ tl.where(cross & (r != 0), 1, 0)
            bnd = bnd | tl.where(a2 & (vert | (act2 & (det == 0))), 1, 0)
        inside = ok & (ext == 1) & (hol == 0) & (bnd == 0)
        tl.store(out + i * KM + cnt, gp, mask=inside & (cnt < KM))
        cnt = cnt + inside.to(tl.int32)
    tl.store(ocnt + i, cnt, mask=m)


@triton.jit
def _seg_scan(x, y, v0, ns, act, lvx, lvy, BLOCK: tl.constexpr):
    """run3a_labels.map_aux's point-to-segment distance over segments v0 .. v0 + ns - 1 of a centreline: first minimum."""
    dmin = tl.full([BLOCK], float("inf"), tl.float64)
    dj = tl.full([BLOCK], -1, tl.int32)
    dcr = tl.zeros([BLOCK], dtype=tl.float64)
    maxs = tl.max(tl.where(act, ns, 0), axis=0)
    for k in range(0, maxs):
        a2 = act & (k < ns)
        j = v0 + k
        ax = tl.load(lvx + j, mask=a2, other=0.0)
        ay = tl.load(lvy + j, mask=a2, other=0.0)
        bxv = tl.load(lvx + j + 1, mask=a2, other=1.0)
        byv = tl.load(lvy + j + 1, mask=a2, other=1.0)
        abx = bxv - ax
        aby = byv - ay
        l2 = tl.maximum(abx * abx + aby * aby, 1e-9)
        t = ((x - ax) * abx + (y - ay) * aby) / l2
        t = tl.minimum(tl.maximum(t, 0.0), 1.0)
        prx = ax + t * abx
        pry = ay + t * aby
        d = tl.sqrt((x - prx) * (x - prx) + (y - pry) * (y - pry))
        cr = abx * (y - ay) - aby * (x - ax)
        better = a2 & (d < dmin)
        dmin = tl.where(better, d, dmin)
        dj = tl.where(better, j, dj)
        dcr = tl.where(better, cr, dcr)
    return dmin, dj, dcr


@triton.jit
def _k_laneaux(px, py, ph, psamp, N, lst, lcnt, pstart, npoly, pline, cmask, loff, lvx, lvy, lsegh,
               o_j, o_d, o_cr, o_flag, KM: tl.constexpr, BLOCK: tl.constexpr):
    """run3a_labels.map_aux columns 4 / 5 per point: the containing candidate lane with the best heading match (first in
    polygon order), else the nearest centreline segment over all candidate lanes (polygon order, then segment order)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(px + i, mask=m, other=0.0)
    y = tl.load(py + i, mask=m, other=0.0)
    h = tl.load(ph + i, mask=m, other=0.0)
    b = tl.load(psamp + i, mask=m, other=0)
    craw = tl.load(lcnt + i, mask=m, other=0)
    nc = tl.minimum(craw, KM)
    best_hd = tl.full([BLOCK], float("inf"), tl.float64)
    best_j = tl.full([BLOCK], -1, tl.int32)
    best_d = tl.zeros([BLOCK], dtype=tl.float64)
    best_cr = tl.zeros([BLOCK], dtype=tl.float64)
    maxc = tl.max(nc, axis=0)
    for c in range(0, maxc):
        act = c < nc
        gp = tl.load(lst + i * KM + c, mask=act, other=0)
        ln = tl.load(pline + gp, mask=act, other=0)
        act = act & (ln >= 0)
        v0 = tl.load(loff + ln, mask=act, other=0)
        v1 = tl.load(loff + ln + 1, mask=act, other=0)
        ns = tl.where(act, tl.maximum(v1 - v0 - 1, 0), 0)
        dmin, dj, dcr = _seg_scan(x, y, v0, ns, act, lvx, lvy, BLOCK)
        sh = tl.load(lsegh + dj, mask=dj >= 0, other=0.0)
        hd = tl.where(dj >= 0, tl.abs(_wrap_t(h - sh)), float("inf"))
        better = act & (hd < best_hd)
        best_hd = tl.where(better, hd, best_hd)
        best_j = tl.where(better, dj, best_j)
        best_d = tl.where(better, dmin, best_d)
        best_cr = tl.where(better, dcr, best_cr)
    fb = m & (nc == 0)
    p0 = tl.load(pstart + b, mask=m, other=0)
    pn = tl.where(fb, tl.load(npoly + b, mask=m, other=0), 0)
    fd = tl.full([BLOCK], float("inf"), tl.float64)
    fj = tl.full([BLOCK], -1, tl.int32)
    fcr = tl.zeros([BLOCK], dtype=tl.float64)
    maxp = tl.max(pn, axis=0)
    for q in range(0, maxp):
        aq = q < pn
        gp = p0 + q
        aq = aq & (tl.load(cmask + gp, mask=aq, other=0) != 0)
        ln = tl.load(pline + gp, mask=aq, other=0)
        aq = aq & (ln >= 0)
        v0 = tl.load(loff + ln, mask=aq, other=0)
        v1 = tl.load(loff + ln + 1, mask=aq, other=0)
        ns = tl.where(aq, tl.maximum(v1 - v0 - 1, 0), 0)
        dmin, dj, dcr = _seg_scan(x, y, v0, ns, aq, lvx, lvy, BLOCK)
        better = aq & (dmin < fd)
        fd = tl.where(better, dmin, fd)
        fj = tl.where(better, dj, fj)
        fcr = tl.where(better, dcr, fcr)
    j = tl.where(nc > 0, best_j, fj)
    tl.store(o_j + i, j, mask=m)
    tl.store(o_d + i, tl.where(nc > 0, best_d, fd), mask=m)
    tl.store(o_cr + i, tl.where(nc > 0, best_cr, fcr), mask=m)
    tl.store(o_flag + i, tl.where((j < 0) & ((nc > 0) | (fb & (maxp > 0))), 1, 0) + tl.where(craw > KM, 2, 0), mask=m)


@triton.jit
def _k_lanedir(px, py, ph, N, lst, lcnt, pflag, pline, loff, lvx, lvy, lvh, o_in, o_min, o_ok,
               KM: tl.constexpr, COMPAT_: tl.constexpr, BLOCK: tl.constexpr):
    """facts.lane_flags per centre point: in a lane with a centreline, the smallest heading difference to the nearest
    centreline vertex of a lane holding it, and 'held by an off-route lane that is not a start lane, within COMPAT'."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(px + i, mask=m, other=0.0)
    y = tl.load(py + i, mask=m, other=0.0)
    h = tl.load(ph + i, mask=m, other=0.0)
    nc = tl.minimum(tl.load(lcnt + i, mask=m, other=0), KM)
    inl = tl.zeros([BLOCK], dtype=tl.int32)
    okk = tl.zeros([BLOCK], dtype=tl.int32)
    mind = tl.full([BLOCK], float("inf"), tl.float64)
    maxc = tl.max(nc, axis=0)
    for c in range(0, maxc):
        act = c < nc
        gp = tl.load(lst + i * KM + c, mask=act, other=0)
        ln = tl.load(pline + gp, mask=act, other=-1)
        fl = tl.load(pflag + gp, mask=act, other=0)
        act = act & (ln >= 0) & ((fl & 1) != 0)
        v0 = tl.load(loff + ln, mask=act, other=0)
        v1 = tl.load(loff + ln + 1, mask=act, other=0)
        nv = tl.where(act, v1 - v0, 0)
        dmin = tl.full([BLOCK], float("inf"), tl.float64)
        vj = tl.full([BLOCK], -1, tl.int32)
        maxv = tl.max(nv, axis=0)
        for k in range(0, maxv):
            a2 = act & (k < nv)
            j = v0 + k
            dx = x - tl.load(lvx + j, mask=a2, other=0.0)
            dy = y - tl.load(lvy + j, mask=a2, other=0.0)
            d2 = dx * dx + dy * dy
            better = a2 & (d2 < dmin)
            dmin = tl.where(better, d2, dmin)
            vj = tl.where(better, j, vj)
        vh = tl.load(lvh + vj, mask=vj >= 0, other=0.0)
        dp = tl.where(vj >= 0, tl.abs(_wrap_t(h - vh)), float("inf"))
        inl = inl | act.to(tl.int32)
        mind = tl.where(act, tl.minimum(mind, dp), mind)
        okk = okk | (act & ((fl & 4) == 0) & ((fl & 8) == 0) & (dp < COMPAT_)).to(tl.int32)
    tl.store(o_in + i, inl, mask=m)
    tl.store(o_min + i, mind, mask=m)
    tl.store(o_ok + i, okk, mask=m)


@triton.jit
def _geos_pt_seg(x, y, ax, ay, bxv, byv):
    """GEOS Distance::pointToSegment."""
    dx = bxv - ax
    dy = byv - ay
    len2 = dx * dx + dy * dy
    r = ((x - ax) * dx + (y - ay) * dy) / len2
    da = tl.sqrt((x - ax) * (x - ax) + (y - ay) * (y - ay))
    db = tl.sqrt((x - bxv) * (x - bxv) + (y - byv) * (y - byv))
    s = ((ay - y) * dx - (ax - x) * dy) / len2
    dp = tl.abs(s) * tl.sqrt(len2)
    degen = (ax == bxv) & (ay == byv)
    return tl.where(degen | (r <= 0.0), da, tl.where(r >= 1.0, db, dp))


@triton.jit
def _k_ring(px, py, psamp, N, e0, e1, ax_, ay_, bx_, by_, o_d, o_in, BLOCK: tl.constexpr):
    """per point, over the ring segments of its sample: the smallest GEOS pointToSegment distance and the GEOS
    RayCrossingCounter location over all rings together (prepared contains): o_in 1 = interior."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(px + i, mask=m, other=0.0)
    y = tl.load(py + i, mask=m, other=0.0)
    b = tl.load(psamp + i, mask=m, other=0)
    s0 = tl.load(e0 + b, mask=m, other=0)
    ns = tl.where(m, tl.load(e1 + b, mask=m, other=0) - s0, 0)
    dmin = tl.full([BLOCK], float("inf"), tl.float64)
    ncr = tl.zeros([BLOCK], dtype=tl.int32)
    bnd = tl.zeros([BLOCK], dtype=tl.int32)
    maxs = tl.max(ns, axis=0)
    for k in range(0, maxs):
        a2 = k < ns
        j = s0 + k
        ax = tl.load(ax_ + j, mask=a2, other=0.0)
        ay = tl.load(ay_ + j, mask=a2, other=0.0)
        bxv = tl.load(bx_ + j, mask=a2, other=1.0)
        byv = tl.load(by_ + j, mask=a2, other=1.0)
        d = _geos_pt_seg(x, y, ax, ay, bxv, byv)
        dmin = tl.where(a2 & (d < dmin), d, dmin)
        notleft = (bxv >= x) | (ax >= x)
        vert = notleft & (x == ax) & (y == ay)
        hor = notleft & (vert == 0) & (ay == y) & (byv == y) & (x >= tl.minimum(ax, bxv)) & (x <= tl.maximum(ax, bxv))
        strad = ((byv > y) & (ay <= y)) | ((ay > y) & (byv <= y))
        det = (bxv - x) * (ay - y) - (byv - y) * (ax - x)
        act2 = a2 & notleft & strad & (vert == 0)
        ncr = ncr + (act2 & (det != 0) & ((det > 0) != (ay < byv))).to(tl.int32)
        bnd = bnd | tl.where(a2 & (vert | hor | (act2 & (det == 0))), 1, 0)
    tl.store(o_d + i, dmin, mask=m)
    tl.store(o_in + i, tl.where(((ncr & 1) == 1) & (bnd == 0), 1, 0), mask=m)


@triton.jit
def _k_project(px, py, psamp, N, r0, r1, q0, q1, qx0, qy0, qx1, qy1, rvx, rvy, rlen, rst, o_m, RB_: tl.constexpr,
               BLOCK: tl.constexpr):
    """GEOS LengthIndexOfPoint::indexOf over the sample's route segments (first segment at the minimum distance). Blocks of
    RB_ segments whose envelope is farther than an upper bound of the minimum (the nearest block's farthest envelope corner)
    are skipped: none of their segments can reach or tie the minimum."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(px + i, mask=m, other=0.0)
    y = tl.load(py + i, mask=m, other=0.0)
    b = tl.load(psamp + i, mask=m, other=0)
    s0 = tl.load(r0 + b, mask=m, other=0)
    ns = tl.where(m, tl.load(r1 + b, mask=m, other=0) - s0, 0)
    b0 = tl.load(q0 + b, mask=m, other=0)
    nb = tl.where(m, tl.load(q1 + b, mask=m, other=0) - b0, 0)
    maxb = tl.max(nb, axis=0)
    ub = tl.full([BLOCK], float("inf"), tl.float64)
    for kb in range(0, maxb):
        a = kb < nb
        fx = tl.maximum(tl.abs(x - tl.load(qx0 + b0 + kb, mask=a, other=0.0)), tl.abs(x - tl.load(qx1 + b0 + kb, mask=a, other=0.0)))
        fy = tl.maximum(tl.abs(y - tl.load(qy0 + b0 + kb, mask=a, other=0.0)), tl.abs(y - tl.load(qy1 + b0 + kb, mask=a, other=0.0)))
        ub = tl.where(a, tl.minimum(ub, tl.sqrt(fx * fx + fy * fy)), ub)
    dmin = tl.full([BLOCK], float("inf"), tl.float64)
    meas = tl.full([BLOCK], -1.0, tl.float64)
    for kb in range(0, maxb):
        a = kb < nb
        lx = tl.maximum(tl.maximum(tl.load(qx0 + b0 + kb, mask=a, other=0.0) - x, x - tl.load(qx1 + b0 + kb, mask=a, other=0.0)), 0.0)
        ly = tl.maximum(tl.maximum(tl.load(qy0 + b0 + kb, mask=a, other=0.0) - y, y - tl.load(qy1 + b0 + kb, mask=a, other=0.0)), 0.0)
        need = a & (tl.sqrt(lx * lx + ly * ly) <= ub + 1e-6)
        if tl.max(need.to(tl.int32), axis=0) > 0:
            for k in range(0, RB_):
                jr = kb * RB_ + k
                a2 = need & (jr < ns)
                j = s0 + jr
                ax = tl.load(rvx + j, mask=a2, other=0.0)
                ay = tl.load(rvy + j, mask=a2, other=0.0)
                bxv = tl.load(rvx + j + 1, mask=a2, other=1.0)
                byv = tl.load(rvy + j + 1, mask=a2, other=1.0)
                d = _geos_pt_seg(x, y, ax, ay, bxv, byv)
                dx = bxv - ax
                dy = byv - ay
                pf = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
                pf = tl.where((x == ax) & (y == ay), 0.0, tl.where((x == bxv) & (y == byv), 1.0, pf))
                st = tl.load(rst + j, mask=a2, other=0.0)
                ln = tl.load(rlen + j, mask=a2, other=0.0)
                mm = tl.where(pf <= 0.0, st, tl.where(pf <= 1.0, st + pf * ln, st + ln))
                better = a2 & (d < dmin)
                dmin = tl.where(better, d, dmin)
                meas = tl.where(better, mm, meas)
    tl.store(o_m + i, meas, mask=m)


def _grid1(n):
    return (triton.cdiv(max(n, 1), BLOCK),)


def _polygon(scene, i):
    """polygon i of a scene rebuilt from its ring vertices (same coordinates, same ring order)."""
    import shapely
    a, b = scene["poly_voff"][i], scene["poly_voff"][i + 1]
    v, r = scene["vxy"][a:b], scene["vring"][a:b]
    cut = np.r_[0, np.nonzero(np.diff(r))[0] + 1, len(r)]
    rings = [v[c0:c1] for c0, c1 in zip(cut[:-1], cut[1:])]
    return shapely.Polygon(rings[0], rings[1:])


def _aux_union_rings(scene, box):
    """run3a_labels.drivable_union(dam, ego corners, 5 m) for one sample: the polygons of the drivable layers whose envelope
    meets box, in map order, unioned by GEOS as run 3A does -> its ring segments (n, 4) [ax ay bx by]."""
    import shapely
    bb = scene["poly_bbox"]
    drv = (scene["poly_flag"] & FL_DRIVE) > 0
    near = (bb[:, 0] <= box[2]) & (bb[:, 2] >= box[0]) & (bb[:, 1] <= box[3]) & (bb[:, 3] >= box[1])
    union = shapely.union_all([_polygon(scene, i) for i in np.nonzero(drv & near)[0]])
    rings = shapely.get_rings(np.asarray(getattr(union, "geoms", [union]), dtype=object))
    xy, ci = shapely.get_coordinates(rings, return_index=True)
    same = ci[:-1] == ci[1:]
    return np.concatenate([xy[:-1][same], xy[1:][same]], 1)


def _aux_union_carla(scene, poses, box, reach=REACH_C):
    """CARLA aux on the CPU for one sample (19 Nov): run3a_labels.drivable_union(dam, ego corners, 5 m) as _aux_union_rings,
    the map_aux corners in numpy exactly as run3a_labels.map_aux / box_corners, inside = shapely contains_xy on the prepared
    union (run3a_labels.signed_distance's own call), and the union's ring edges whose envelope comes within reach of the
    corners. -> inside (P*8*4,) bool, corners (P*8*4, 2), edges (n, 4)."""
    import shapely
    bb = scene["poly_bbox"]
    drv = (scene["poly_flag"] & FL_DRIVE) > 0
    near = (bb[:, 0] <= box[2]) & (bb[:, 2] >= box[0]) & (bb[:, 1] <= box[3]) & (bb[:, 3] >= box[1])
    union = shapely.union_all([_polygon(scene, i) for i in np.nonzero(drv & near)[0]])
    shapely.prepare(union)
    X0, Y0, H0, c0, s0 = scene["x0"], scene["y0"], scene["h0"], scene["c0"], scene["s0"]
    gx = X0 + c0 * poses[..., 0] - s0 * poses[..., 1]
    gy = Y0 + s0 * poses[..., 0] + c0 * poses[..., 1]
    gh = (H0 + poses[..., 2] + np.pi) % (2 * np.pi) - np.pi
    cx, cy = gx + scene["r2c"] * np.cos(gh), gy + scene["r2c"] * np.sin(gh)
    hl, hw = scene["hl"], scene["hw"]
    lo, la = np.array([hl, -hl, -hl, hl]), np.array([hw, hw, -hw, -hw])
    c, s = np.cos(gh)[..., None], np.sin(gh)[..., None]
    flat = np.stack([cx[..., None] + lo * c - la * s, cy[..., None] + lo * s + la * c], -1).reshape(-1, 2)
    inside = shapely.contains_xy(union, flat[:, 0], flat[:, 1])
    rings = shapely.get_rings(np.asarray(getattr(union, "geoms", [union]), dtype=object))
    xy, ci = shapely.get_coordinates(rings, return_index=True)
    same = ci[:-1] == ci[1:]
    e = np.concatenate([xy[:-1][same], xy[1:][same]], 1)
    (x0, y0), (x1, y1) = flat.min(0) - reach, flat.max(0) + reach
    keep = ((np.minimum(e[:, 0], e[:, 2]) <= x1) & (np.maximum(e[:, 0], e[:, 2]) >= x0)
            & (np.minimum(e[:, 1], e[:, 3]) <= y1) & (np.maximum(e[:, 1], e[:, 3]) >= y0))
    return inside, flat, e[keep]


# ================================================================================================================ GPU helpers
PROF = None  # set to a dict to time the sections of label_map (synchronising; development only)
_T0 = [0.0]


def _tick(name):
    if PROF is None:
        return
    import time
    torch.cuda.synchronize()
    t = time.time()
    if name is not None:
        PROF[name] = PROF.get(name, 0.0) + t - _T0[0]
    _T0[0] = t


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * np.pi) - math.pi


def _per_step_any(x):
    """(..., 41) bool -> (..., 8): step 0 = indices 0..5, step k = 5k+1 .. 5k+5."""
    return torch.cat([x[..., :6].any(-1, keepdim=True), x[..., 6:].reshape(*x.shape[:-1], 7, 5).any(-1)], -1)


def _latch(first):
    """first (...,) 10 Hz index (int64, >= 99 = never) -> (..., 8) bool from its step on."""
    so = torch.as_tensor(STEP_OF, device=first.device)
    st = torch.where(first < 99, so[first.clamp(0, 40)], torch.full_like(first, 99))
    return torch.arange(NS, device=first.device) >= st[..., None]


def _first_true(x):
    """(..., T) bool -> first index or 99."""
    return torch.where(x, torch.arange(x.shape[-1], device=x.device), torch.full(x.shape, 99, dtype=torch.int64, device=x.device)).amin(-1)


def _lane_cand(T, gx, gy, extra_box=None):
    """run3a_labels.map_aux's candidate lanes: lanes with a centreline whose envelope meets the box of all the points
    (+ extra_box) grown by 10 m; none -> every lane with a centreline. -> (NP,) bool over the batch's polygons."""
    x0, y0, x1, y1 = gx.amin(1), gy.amin(1), gx.amax(1), gy.amax(1)
    if extra_box is not None:
        x0, y0 = torch.minimum(x0, extra_box[:, 0]), torch.minimum(y0, extra_box[:, 1])
        x1, y1 = torch.maximum(x1, extra_box[:, 2]), torch.maximum(y1, extra_box[:, 3])
    x0, y0, x1, y1 = x0 - 10.0, y0 - 10.0, x1 + 10.0, y1 + 10.0
    samp = torch.repeat_interleave(torch.arange(T.B, device=gx.device), T.npoly.long(), output_size=T.NP)
    lanes = T.pline >= 0
    c = lanes & (T.bx0 <= x1[samp]) & (T.bx1 >= x0[samp]) & (T.by0 <= y1[samp]) & (T.by1 >= y0[samp])
    anyc = torch.zeros(T.B, dtype=torch.int32, device=gx.device).index_add_(0, samp, c.int()) > 0
    return torch.where(anyc[samp], c, lanes), samp


# ================================================================================================================ label_map
def label_map(kind, scenes, states, coords, poses=None, T=None):
    """see the module docstring."""
    dev = states.device
    B, P = states.shape[:2]
    _tick(None)
    T = collate_map(scenes, dev) if T is None else T
    _tick("collate")
    out = {}
    diag = torch.zeros(len(DIAG), dtype=torch.int64, device=dev)
    i32 = torch.int32
    X0, Y0, H0, C0, S0 = (getattr(T, k)[:, None, None] for k in ("x0", "y0", "h0", "c0", "s0"))
    # ------------------------------------------------------------------ the point sets
    # 0: scorer corners + centre (41 states); 1: facts' lane_off centre poses; 2: NAVSIM aux poses; 3: NAVSIM aux corners
    hwp = states[:, :, WP, 2]
    cgx = states[:, :, WP, 0] + T.r2c * torch.cos(hwp)
    cgy = states[:, :, WP, 1] + T.r2c * torch.sin(hwp)
    dx, dy = cgx - X0, cgy - Y0
    ex, ey = C0 * dx + S0 * dy, -S0 * dx + C0 * dy
    lgx = X0 + C0 * ex - S0 * ey
    lgy = Y0 + S0 * ex + C0 * ey
    lgh = _wrap(H0 + _wrap(hwp - H0))
    sets = [coords.reshape(B, -1, 2), torch.stack([lgx, lgy], -1).reshape(B, -1, 2)]
    lane_off_cand, samp = _lane_cand(T, lgx.reshape(B, -1), lgy.reshape(B, -1), T.row0_wpbox)
    masks = [((T.pflag & (FL_LANE | FL_DRIVE)) > 0), lane_off_cand]
    aux = kind == "nav" and poses is not None
    aux_c = kind == "carla" and poses is not None  # CARLA aux (19 Nov)
    if aux or aux_c:
        pgx = X0 + C0 * poses[..., 0] - S0 * poses[..., 1]
        pgy = Y0 + S0 * poses[..., 0] + C0 * poses[..., 1]
        pgh = _wrap(H0 + poses[..., 2])
        ccx = pgx + T.r2c * torch.cos(pgh)
        ccy = pgy + T.r2c * torch.sin(pgh)
        lo = torch.tensor([T.hl, -T.hl, -T.hl, T.hl], dtype=F64, device=dev)
        la = torch.tensor([T.hw, T.hw, -T.hw, -T.hw], dtype=F64, device=dev)
        cc, ss = torch.cos(pgh)[..., None], torch.sin(pgh)[..., None]
        kx = ccx[..., None] + lo * cc - la * ss
        ky = ccy[..., None] + lo * ss + la * cc  # (B, P, 8, 4)
        aux_cand, _ = _lane_cand(T, pgx.reshape(B, -1), pgy.reshape(B, -1))
        sets += [torch.stack([pgx, pgy], -1).reshape(B, -1, 2)]
        masks += [aux_cand]
    sizes = [s.shape[1] for s in sets]
    pts = torch.cat(sets, 1)  # (B, Ntot, 2), sets side by side per sample
    Nt = pts.shape[1]
    setid = torch.cat([torch.full((n,), k, dtype=i32, device=dev) for k, n in enumerate(sizes)])
    flat = pts.reshape(-1, 2)
    px, py = flat[:, 0].contiguous(), flat[:, 1].contiguous()
    psamp = torch.arange(B, device=dev, dtype=i32).repeat_interleave(Nt)
    pset = setid.repeat(B)
    setmask = torch.stack(masks).to(torch.int8).contiguous()
    N = B * Nt
    lst = torch.empty((N, KMAX), dtype=i32, device=dev)
    lcnt = torch.empty(N, dtype=i32, device=dev)
    _k_pip[_grid1(N)](px, py, psamp, pset, N, T.g_ox, T.g_oy, T.g_nx, T.g_ny, T.g_base, T.g_off, T.g_ent, T.bx0, T.by0, T.bx1,
                      T.by1, setmask, T.NP, T.band0, T.boff, T.bent, T.vx, T.vy, T.vring, lst, lcnt, KM=KMAX, BH=BAND_H,
                      G=PGRID, BLOCK=BLOCK, enable_fp_fusion=False)
    diag[0] += (lcnt > KMAX).sum()
    diag[3] = lcnt.max()
    _tick("pip")
    lst = lst.view(B, Nt, KMAX)
    lcnt = lcnt.view(B, Nt)
    offs = np.r_[0, np.cumsum(sizes)]

    def part(k):
        return lst[:, offs[k]:offs[k + 1]], lcnt[:, offs[k]:offs[k + 1]].clamp(max=KMAX)
    # ------------------------------------------------------------------ the scorer's ego areas
    L0, C0n = part(0)
    S = B * P * K
    L0 = L0.reshape(S, 5, KMAX)
    valid = torch.arange(KMAX, device=dev) < C0n.reshape(S, 5, 1)
    L0 = torch.where(valid, L0, 0)  # slots past the count are uninitialised
    fl = torch.where(valid, T.pflag[L0.long()], 0)
    nondrv = ((fl[:, :4] & FL_DRIVE) > 0).any(-1).sum(-1) < 4
    big = torch.iinfo(torch.int32).max
    lanes = torch.where((fl[:, :4] & FL_LANE) > 0, L0[:, :4], big).reshape(S, 4 * KMAX)
    lanes = torch.sort(lanes, -1).values
    isl = lanes != big
    nl = (isl[:, :1].int().sum(-1) + ((lanes[:, 1:] != lanes[:, :-1]) & isl[:, 1:]).int().sum(-1))
    full4 = ((lanes[:, 3:] == lanes[:, :-3]) & isl[:, 3:]).any(-1)
    multi = (nl > 1) & ~full4
    on_route = ((fl[:, 4] & FL_ROUTE) > 0).any(-1)
    ego_areas = torch.stack([multi, nondrv, ~on_route], -1).view(B, P, K, 3)
    out["ego_areas"] = ego_areas
    _tick("ego_areas")
    # ------------------------------------------------------------------ wrong way, left the route (facts.lane_flags)
    cen = coords[..., 4, :].reshape(S, 2)
    ccx_, ccy_ = cen[:, 0].contiguous(), cen[:, 1].contiguous()
    hb = states[..., 2].reshape(S).contiguous()
    Lc = L0[:, 4].contiguous()
    Cc = C0n.reshape(S, 5)[:, 4].contiguous()
    o_in = torch.empty(S, dtype=i32, device=dev)
    o_min = torch.empty(S, dtype=F64, device=dev)
    o_ok = torch.empty(S, dtype=i32, device=dev)
    _k_lanedir[_grid1(S)](ccx_, ccy_, hb, S, Lc, Cc, T.pflag, T.pline, T.loff, T.lvx, T.lvy, T.lvh, o_in, o_min, o_ok,
                          KM=KMAX, COMPAT_=COMPAT, BLOCK=BLOCK, enable_fp_fusion=False)
    ww = ((o_in > 0) & (o_min > WRONG_WAY)).view(B, P, K)
    out["wrong_way"] = _per_step_any(ww)
    left = (~on_route & (o_ok > 0)).view(B, P, K)
    out["left_route"] = _latch(_first_true(left))
    _tick("lane_flags")
    # ------------------------------------------------------------------ off road
    nd = nondrv.view(B, P, K)
    if kind == "carla":  # carla_full_v1.apply_rules: from the first state with all four corners inside
        nd = nd & (torch.arange(K, device=dev) >= _first_true(~nd)[..., None])
    out["offroad"] = _per_step_any(nd)

    # ------------------------------------------------------------------ lane offset (facts: map_aux on the centre poses)
    def lane_aux(k, gx_, gy_, gh_, cand, slot):
        Lk, Ck = lst[:, offs[k]:offs[k + 1]].reshape(-1, KMAX).contiguous(), lcnt[:, offs[k]:offs[k + 1]].reshape(-1).contiguous()
        n = Lk.shape[0]
        oj = torch.empty(n, dtype=i32, device=dev)
        od = torch.empty(n, dtype=F64, device=dev)
        ocr = torch.empty(n, dtype=F64, device=dev)
        ofl = torch.empty(n, dtype=i32, device=dev)
        _k_laneaux[_grid1(n)](gx_.reshape(-1).contiguous(), gy_.reshape(-1).contiguous(), gh_.reshape(-1).contiguous(),
                              torch.arange(B, device=dev, dtype=i32).repeat_interleave(n // B), n, Lk, Ck, T.pstart, T.npoly,
                              T.pline, cand.to(torch.int8).contiguous(), T.loff, T.lvx, T.lvy, T.lsegh, oj, od, ocr, ofl,
                              KM=KMAX, BLOCK=BLOCK, enable_fp_fusion=False)
        diag[slot] += (ofl > 0).sum()
        good = oj >= 0
        off = torch.where(good, torch.sign(ocr) * od, torch.zeros_like(od)).clamp(-3.5, 3.5)
        dh = torch.where(good, _wrap(gh_.reshape(-1) - T.lsegh[oj.clamp_min(0).long()]), torch.zeros_like(od))
        return off, dh
    off, _ = lane_aux(1, lgx, lgy, lgh, lane_off_cand, 1)
    out["lane_off"] = off.view(B, P, NS)
    _tick("lane_off")
    # ------------------------------------------------------------------ metres along the route (GEOS project)
    q = coords[:, :, np.r_[0, WP], 4, :].reshape(-1, 2)
    nq = q.shape[0]
    meas = torch.empty(nq, dtype=F64, device=dev)
    _k_project[_grid1(nq)](q[:, 0].contiguous(), q[:, 1].contiguous(), torch.arange(B, device=dev, dtype=i32).repeat_interleave(P * 9),
                           nq, T.r0, T.r1, T.q0, T.q1, T.qx0, T.qy0, T.qx1, T.qy1, T.rvx, T.rvy, T.rlen, T.rst, meas, RB_=RB,
                           BLOCK=BLOCK, enable_fp_fusion=False)
    out["metres"] = torch.diff(meas.view(B, P, 9), dim=-1)
    _tick("metres")
    # ------------------------------------------------------------------ stop lines: red light, stop sign
    hd = states[..., 2]
    fx = states[..., 0] + T.front * torch.cos(hd)
    fy = states[..., 1] + T.front * torch.sin(hd)
    dx, dy = fx - X0, fy - Y0
    ex, ey = C0 * dx + S0 * dy, -S0 * dx + C0 * dy  # (B, P, 41) front bumper, ego frame at t
    he = _wrap(hd - H0)
    spd = torch.sqrt(states[..., 3] * states[..., 3] + states[..., 4] * states[..., 4])
    Lsl = T.sl_valid.shape[1]

    def sc(a):
        return a[:, None, :, None]  # (B, L) -> (B, 1, L, 1)
    rx, ry = ex[:, :, None, :] - sc(T.sl_c[..., 0]), ey[:, :, None, :] - sc(T.sl_c[..., 1])
    s_ = rx * sc(T.sl_u[..., 0]) + ry * sc(T.sl_u[..., 1])  # (B, P, L, 41)
    lat = rx * sc(T.sl_n[..., 0]) + ry * sc(T.sl_n[..., 1])
    a_, b_ = s_[..., :-1], s_[..., 1:]
    smax = sc(T.sl_smax)
    f_ = torch.clamp((smax - a_) / (b_ - a_), 0, 1)
    lc = lat[..., :-1] + f_ * (lat[..., 1:] - lat[..., :-1])
    along = torch.cos(he[:, :, None, 1:] - sc(T.sl_uh)) > 0
    xo = (a_ <= smax) & (b_ > smax) & (lc >= sc(T.sl_lmin) - LAT_MARGIN) & (lc <= sc(T.sl_lmax) + LAT_MARGIN) & along
    xo = xo & sc(T.sl_valid)
    ci = torch.where(xo.any(-1), xo.int().argmax(-1) + 1, torch.full(xo.shape[:-1], 99, dtype=torch.int64, device=dev))
    cic = ci.clamp(max=40)
    win = (cic[..., None] - torch.arange(5, -1, -1, device=dev)).clamp_min(0)  # indices i-5 .. i (clipped at 0)
    mv = torch.gather(spd[:, :, None, :].expand(-1, -1, Lsl, -1), 3, win).amin(-1)
    sign = T.sl_sign[:, None, :]
    hit = ci < 99
    nev = torch.full_like(ci, 99)
    col = torch.gather(T.sl_col[:, None, :, :].expand(-1, P, -1, -1), 3, (cic // 5)[..., None])[..., 0]
    stop_i = torch.where(hit & sign & (mv > STOP_SPEED), ci, nev).amin(-1)
    red_i = torch.where(hit & ~sign & (col == 1), ci, nev).amin(-1)
    unk_i = torch.where(hit & ~sign & (col == -1), ci, nev).amin(-1)
    out["red"] = torch.where(_latch(red_i), 1, torch.where(_latch(unk_i), -1, 0)).to(torch.int8)
    out["stop"] = _latch(stop_i)
    _tick("stop_lines")
    # ------------------------------------------------------------------ NAVSIM aux (run 3A's get_sub_score)
    if aux:
        aoff, adh = lane_aux(2, pgx, pgy, pgh, aux_cand, 2)
        # run 3A's drivable union of the polygons near all ego corners (rows 0 .. P, 41 states): built on the CPU (see top)
        c4 = coords[..., :4, :].reshape(B, -1, 2)
        box = torch.stack([torch.minimum(c4[..., 0].amin(1), T.row0_cbox[:, 0]) - 5.0,
                           torch.minimum(c4[..., 1].amin(1), T.row0_cbox[:, 1]) - 5.0,
                           torch.maximum(c4[..., 0].amax(1), T.row0_cbox[:, 2]) + 5.0,
                           torch.maximum(c4[..., 1].amax(1), T.row0_cbox[:, 3]) + 5.0], -1).cpu().numpy()  # host sync
        _tick("aux_box")
        segs = [_aux_union_rings(scenes[b], box[b]) for b in range(B)]
        ne = np.array([len(x) for x in segs])
        e0 = np.r_[0, np.cumsum(ne)[:-1]]
        U = torch.as_tensor(np.concatenate(segs) if ne.sum() else np.zeros((1, 4)), device=dev)
        E = torch.as_tensor(np.stack([e0, e0 + ne]).astype(np.int32), device=dev)
        _tick("aux_union_cpu")
        kq = torch.stack([kx, ky], -1).reshape(-1, 2)
        nk = kq.shape[0]
        ud = torch.empty(nk, dtype=F64, device=dev)
        uin = torch.empty(nk, dtype=i32, device=dev)
        _k_ring[_grid1(nk)](kq[:, 0].contiguous(), kq[:, 1].contiguous(), torch.arange(B, device=dev, dtype=i32).repeat_interleave(nk // B),
                            nk, E[0].contiguous(), E[1].contiguous(), U[:, 0].contiguous(), U[:, 1].contiguous(), U[:, 2].contiguous(),
                            U[:, 3].contiguous(), ud, uin, BLOCK=BLOCK, enable_fp_fusion=False)
        sd = torch.clamp(torch.where(uin > 0, ud, -ud), -1.0, 2.0).view(B, P, NS, 4)
        out["aux"] = dict(map_aux=torch.cat([sd, aoff.view(B, P, NS, 1), adh.view(B, P, NS, 1)], -1), ego_areas=ego_areas[:, :, 1:, 1:])
        _tick("aux")
    # ------------------------------------------------------------------ CARLA aux (19 Nov; lead_aux/map_ego/build.aux_from_mc)
    if aux_c:
        aoff, adh = lane_aux(2, pgx, pgy, pgh, aux_cand, 2)
        c4 = coords[..., :4, :].reshape(B, -1, 2)
        box = torch.stack([torch.minimum(c4[..., 0].amin(1), T.row0_cbox[:, 0]) - 5.0,
                           torch.minimum(c4[..., 1].amin(1), T.row0_cbox[:, 1]) - 5.0,
                           torch.maximum(c4[..., 0].amax(1), T.row0_cbox[:, 2]) + 5.0,
                           torch.maximum(c4[..., 1].amax(1), T.row0_cbox[:, 3]) + 5.0], -1)
        hb = torch.cat([box.reshape(-1), poses.reshape(-1)]).cpu().numpy()  # one host sync
        box_np, poses_np = hb[:B * 4].reshape(B, 4), hb[B * 4:].reshape(poses.shape)
        _tick("aux_box")
        if AUX_THREADS > 1:
            if not _POOL:
                from concurrent.futures import ThreadPoolExecutor
                _POOL.append(ThreadPoolExecutor(AUX_THREADS))
            got = list(_POOL[0].map(lambda b: _aux_union_carla(scenes[b], poses_np[b], box_np[b]), range(B)))
        else:
            got = [_aux_union_carla(scenes[b], poses_np[b], box_np[b]) for b in range(B)]
        ins_np = np.stack([g[0] for g in got])
        kq_np = np.stack([g[1] for g in got])  # (B, P*8*4, 2), numpy corners
        ne = np.array([len(g[2]) for g in got])
        e0 = np.r_[0, np.cumsum(ne)[:-1]]
        U = torch.as_tensor(np.concatenate([g[2] for g in got]) if ne.sum() else np.zeros((1, 4)), device=dev)
        E = torch.as_tensor(np.stack([e0, e0 + ne]).astype(np.int32), device=dev)
        kq = torch.as_tensor(kq_np.reshape(-1, 2), device=dev)
        ins = torch.as_tensor(ins_np.reshape(-1), device=dev)
        _tick("aux_union_cpu")
        nk = kq.shape[0]
        ud = torch.empty(nk, dtype=F64, device=dev)
        uin = torch.empty(nk, dtype=i32, device=dev)  # not used: the edge set is cut to the corners' reach
        _k_ring[_grid1(nk)](kq[:, 0].contiguous(), kq[:, 1].contiguous(), torch.arange(B, device=dev, dtype=i32).repeat_interleave(nk // B),
                            nk, E[0].contiguous(), E[1].contiguous(), U[:, 0].contiguous(), U[:, 1].contiguous(), U[:, 2].contiguous(),
                            U[:, 3].contiguous(), ud, uin, BLOCK=BLOCK, enable_fp_fusion=False)
        sd = torch.clamp(torch.where(ins, ud, -ud), -1.0, 2.0).view(B, P, NS, 4)
        out["aux"] = dict(map_aux=torch.cat([sd, aoff.view(B, P, NS, 1), adh.view(B, P, NS, 1)], -1), ego_areas=ego_areas[:, :, 1:, 1:])
        _tick("aux")
    out["diag"] = diag
    return out


def diag_dict(out):
    """the diag counters as a dict (one host synchronisation)."""
    return dict(zip(DIAG, out["diag"].tolist()))


def as_labeller_dtypes(out):
    """the CPU labeller's dtypes (facts: metres float32, lane_off float16; NAVSIM aux map_aux float32), numpy."""
    r = {k: out[k].cpu().numpy() for k in ("offroad", "wrong_way", "left_route", "stop", "red")}
    r["metres"] = out["metres"].cpu().numpy().astype(np.float32)
    r["lane_off"] = out["lane_off"].cpu().numpy().astype(np.float16)
    if "aux" in out:
        r["aux"] = dict(map_aux=out["aux"]["map_aux"].cpu().numpy().astype(np.float32),
                        ego_areas=out["aux"]["ego_areas"].cpu().numpy())
    return r
