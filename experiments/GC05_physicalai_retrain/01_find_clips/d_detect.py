"""STEP D detectors. Reads NAS/<stage>/<uuid>/ (fetched by c_pilot.py or d_fetch.py). Fixed thresholds from the brief."""
import sys, json, math, pathlib, argparse, collections, common
import numpy as np, pandas as pd, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="pilot"); ap.add_argument("--out", default="d_pilot"); a = ap.parse_args()
excl = common.load_exclusions()
ANCHORS = np.arange(0.5, 16.0 + 1e-9, 0.5); HIST = -0.5; FUT = 4.0
ratings = pd.read_csv(common.NAS / "root_files" / "clip_ratings_26.04.csv").set_index("clip_id")

def yaw_of(q):  # quaternion dict x,y,z,w -> yaw
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def load_scene(d):
    eg = pq.read_table(d / "clipgt__egomotion_estimate.parquet").to_pandas()
    ts = np.array([k["timestamp_micros"] for k in eg["key"]], dtype=np.int64); t0 = ts[0]
    t = (ts - t0) / 1e6
    xy = np.array([[e["location"]["x"], e["location"]["y"]] for e in eg["egomotion_estimate"]])
    yaw = np.array([yaw_of(e["orientation"]) for e in eg["egomotion_estimate"]])
    v = np.gradient(xy, t, axis=0); speed = np.linalg.norm(v, axis=1)
    ob = pq.read_table(d / "clipgt__obstacle.parquet").to_pandas()
    tracks = collections.defaultdict(list)
    for k, o in zip(ob["key"], ob["obstacle"]):
        tracks[o["trackline_id"]].append(((k["timestamp_micros"] - t0) / 1e6, o["center"]["x"], o["center"]["y"], o["size"]["x"], o["size"]["y"], yaw_of(o["orientation"]), o["category"]))
    T = {}
    for tid, rows in tracks.items():
        rows.sort(); arr = np.array([r[:6] for r in rows], dtype=float)
        if len(arr) < 2: continue
        tt = arr[:, 0]; cxy = arr[:, 1:3]
        sp = np.linalg.norm(np.gradient(cxy, tt, axis=0), axis=1) if len(tt) > 2 and np.all(np.diff(tt) > 0) else np.zeros(len(tt))
        T[tid] = dict(t=tt, xy=cxy, L=np.median(arr[:, 3]), W=np.median(arr[:, 4]), yaw=arr[:, 5], speed=sp, cat=rows[0][6])
    ln = pq.read_table(d / "clipgt__lane.parquet").to_pandas()
    centres = []
    for l in ln["lane"]:
        L = np.array([[p["x"], p["y"]] for p in l["left_rail"]]); R = np.array([[p["x"], p["y"]] for p in l["right_rail"]])
        if len(L) < 2 or len(R) < 2: continue
        n = max(len(L), len(R), 8)
        def rs(P):
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            if s[-1] == 0: return None
            u = np.linspace(0, s[-1], n); return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]
        l2, r2 = rs(L), rs(R)
        if l2 is None or r2 is None: continue
        centres.append((l2 + r2) / 2)
    return dict(t=t, xy=xy, yaw=yaw, speed=speed, tracks=T, centres=centres, dur=t[-1])

def interp_ego(S, tq):
    return np.c_[np.interp(tq, S["t"], S["xy"][:, 0]), np.interp(tq, S["t"], S["xy"][:, 1])], np.interp(tq, S["t"], S["speed"]), np.interp(tq, S["t"], np.unwrap(S["yaw"]))

def to_ego(p, origin, yaw):
    d = p - origin; c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c*d[0] - s*d[1], s*d[0] + c*d[1]])

def track_at(tr, tq):
    if tq < tr["t"][0] - 0.15 or tq > tr["t"][-1] + 0.15: return None
    return np.array([np.interp(tq, tr["t"], tr["xy"][:, 0]), np.interp(tq, tr["t"], tr["xy"][:, 1])]), np.interp(tq, tr["t"], tr["speed"]), np.interp(tq, tr["t"], np.unwrap(tr["yaw"]))

def dist_to_polyline(p, P):
    A, B = P[:-1], P[1:]; AB = B - A; AP = p - A
    tt = np.clip((AP * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-9), 0, 1)
    return np.min(np.linalg.norm(A + tt[:, None] * AB - p, axis=1))

def mode1(S):
    hits = []
    for ta in ANCHORS:
        if ta + FUT > S["dur"]: break
        tq = np.arange(ta + HIST, ta + FUT + 1e-9, 0.1); _, sp, _ = interp_ego(S, tq)
        if sp.min() >= 17.0: hits.append((ta, float(np.interp(ta, S["t"], S["speed"])), float(sp.min())))
    return hits

def mode2(S):
    ev = []; t, sp = S["t"], S["speed"]; still = sp < 0.3; i = 0
    while i < len(t):
        if not still[i]: i += 1; continue
        j = i
        while j < len(t) and still[j]: j += 1
        t_start, t_end = t[i], t[j-1]
        if t_end - t_start >= 2.0:
            o_xy, _, o_yaw = interp_ego(S, np.array([t_end])); o_xy, o_yaw = o_xy[0], o_yaw[0]
            k = j; t_dep = None
            while k < len(t) and t[k] <= t_end + 6.0:
                if sp[k] > 2.0: t_dep = t[k]; break
                k += 1
            if t_dep is not None:
                m = (t >= t_end) & (t <= t_end + 6.0)
                lat = max(abs(to_ego(p, o_xy, o_yaw)[1]) for p in S["xy"][m])
                if lat >= 1.0:
                    for tid, tr in S["tracks"].items():
                        r = track_at(tr, t_end)
                        if r is None: continue
                        mask = (tr["t"] >= t_start) & (tr["t"] <= t_dep)
                        if mask.sum() and tr["speed"][mask].max() >= 0.5: continue
                        e = to_ego(r[0], o_xy, o_yaw); gap = e[0] - tr["L"] / 2
                        if 0 <= gap <= 8 and abs(e[1]) <= 2.0:
                            ev.append(dict(t_start=round(t_start,1), t_end=round(t_end,1), t_depart=round(t_dep,1), obstacle=tr["cat"], track=tid, gap_m=round(float(gap),2), lateral_m=round(float(e[1]),2), ego_lat_m=round(float(lat),2)))
        i = j
    return ev

def mode3(S):
    hits = []; n_fallback = 0
    for ta in ANCHORS:
        if ta + FUT > S["dur"]: break
        o_xy, o_sp, o_yaw = interp_ego(S, np.array([ta])); o_xy, o_sp, o_yaw = o_xy[0], o_sp[0], o_yaw[0]
        if o_sp <= 2.0: continue
        tq = np.arange(ta, ta + FUT + 1e-9, 0.1); path, _, yaws = interp_ego(S, tq)
        near = []
        for tid, tr in S["tracks"].items():
            r = track_at(tr, ta)
            if r is None: continue
            e = to_ego(r[0], o_xy, o_yaw)
            if 0 <= e[0] <= 15 and dist_to_polyline(r[0], path) <= 3.5:
                near.append((tid, tr, r, e))
        if not near: continue
        # lane-keeping test
        dists = [min((dist_to_polyline(p, C) for C in S["centres"]), default=np.inf) for p in path]
        if dists[0] <= 1.5 and np.isfinite(max(dists)):
            keeps = max(dists) <= 0.5; how = "lane"
        else:
            if abs(yaws[-1] - yaws[0]) > math.radians(5): continue   # fallback only on low-curvature anchors
            n_fallback += 1; how = "heading"
            keeps = max(abs(to_ego(p, o_xy, o_yaw)[1]) for p in path) <= 0.5
        if not keeps: continue
        # obstacle type
        types = []
        for tid, tr, r, e in near:
            asp = r[1]; rel_yaw = abs((r[2] - o_yaw + math.pi) % (2*math.pi) - math.pi)
            if asp < 0.5: types.append("parked_static_beside" if abs(e[1]) > 1.0 else "static_in_path")
            elif rel_yaw > math.radians(45) and rel_yaw < math.radians(135): types.append("crossing")
            elif asp < o_sp and abs(e[1]) <= 1.5: types.append("slow_lead")
            else: types.append("other_moving")
        ahead_slow = sum(1 for tid, tr in S["tracks"].items() if (rr := track_at(tr, ta)) is not None and 0 <= to_ego(rr[0], o_xy, o_yaw)[0] <= 30 and abs(to_ego(rr[0], o_xy, o_yaw)[1]) <= 1.5 and rr[1] < 2.0)
        if ahead_slow >= 2: types.append("queue")
        hits.append(dict(t=float(ta), ego_speed=round(float(o_sp),1), n_near=len(near), types=sorted(set(types)), lane_test=how, max_dev=round(float(max(dists)) if how=="lane" else -1, 2)))
    return hits, n_fallback

if __name__ == "__main__":
    root = common.NAS / a.stage; res = {}
    for d in sorted(root.iterdir()):
        u = d.name; assert u not in excl, f"exam scene present in {root}: {u}"
        if not (d / "clipgt__egomotion_estimate.parquet").exists(): continue
        S = load_scene(d); m1 = mode1(S); m2 = mode2(S); m3, nfb = mode3(S)
        res[u] = dict(dur=round(float(S["dur"]),1), max_speed=round(float(S["speed"].max()),2), rating=ratings.loc[u, "ego_speed"] if u in ratings.index else None,
                      n_tracks=len(S["tracks"]), n_lanes=len(S["centres"]), mode1=m1, mode2=m2, mode3=m3, mode3_fallback=nfb)
        print(f"{u[:8]} dur {res[u]['dur']} vmax {res[u]['max_speed']:5.2f} rating={res[u]['rating']:<9} tracks {len(S['tracks']):3d} lanes {len(S['centres']):3d} | m1 {len(m1):2d} anchors | m2 {len(m2)} events | m3 {len(m3):2d} anchors (fallback {nfb})")
        for e in m2: print("    m2", e)
        for h in m3[:3]: print("    m3", h)
    json.dump(res, open(common.HERE / f"{a.out}_results.json", "w"), indent=1)
    print("scenes", len(res))
