"""Run 2 detectors (D-0, D-A, D-B) on NAS/pool/<uuid>/. Thresholds are the brief's; implementation choices are in REPORT.md.
Usage: d2_detect.py [--limit N] [--procs K]   -> d2_anchors.csv, d2_scenes.csv, d2_moveoffs.csv"""
import sys, json, math, argparse, collections, multiprocessing as mp, common
import numpy as np, pandas as pd, pyarrow.parquet as pq
ANCH = np.round(np.arange(0.5, 16.0 + 1e-9, 0.5), 1); HIST, FUT, DT = 0.5, 4.0, 0.1

def yaw_q(x, y, z, w): return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
def smooth(a, k=5): return np.convolve(np.pad(a, k//2, mode="edge"), np.ones(k)/k, mode="valid")

def load(d):
    t = pq.read_table(d / "clipgt__egomotion_estimate.parquet").flatten().flatten()
    c = {n: t.column(n).to_numpy(zero_copy_only=False) for n in t.column_names if n.startswith(("key.timestamp", "egomotion_estimate.location", "egomotion_estimate.orientation"))}
    ts = c["key.timestamp_micros"].astype(np.int64); o = np.argsort(ts); t0 = ts[o][0]
    tt = (ts[o] - t0) / 1e6
    xy = np.c_[c["egomotion_estimate.location.x"], c["egomotion_estimate.location.y"]][o]
    yaw = np.unwrap(yaw_q(*[c[f"egomotion_estimate.orientation.{k}"][o] for k in "xyzw"]))
    g = np.round(np.arange(0, tt[-1] + 1e-6, DT), 1)                       # uniform 10 Hz grid
    X = np.c_[np.interp(g, tt, xy[:, 0]), np.interp(g, tt, xy[:, 1])]; Y = np.interp(g, tt, yaw)
    sp = np.linalg.norm(np.gradient(X, DT, axis=0), axis=1); acc = np.gradient(smooth(sp), DT)
    S = dict(t=g, xy=X, yaw=Y, speed=sp, acc=acc, dur=float(tt[-1]), n_pose=len(tt), max_dt=float(np.diff(tt).max()))
    # actors, resampled to the same grid
    o_ = pq.read_table(d / "clipgt__obstacle.parquet").flatten().flatten()
    ob = pd.DataFrame({"tid": o_.column("obstacle.trackline_id").to_pylist(), "ts": (o_.column("key.timestamp_micros").to_numpy().astype(np.int64) - t0) / 1e6,
                       "x": o_.column("obstacle.center.x").to_numpy(), "y": o_.column("obstacle.center.y").to_numpy(),
                       "L": o_.column("obstacle.size.x").to_numpy(), "W": o_.column("obstacle.size.y").to_numpy(), "cat": o_.column("obstacle.category").to_pylist()})
    T = {}
    for tid, r in ob.sort_values("ts").groupby("tid", sort=False):
        a = r["ts"].to_numpy(); keep = np.r_[True, np.diff(a) > 1e-4]; a = a[keep]
        if len(a) < 5 or a[-1] - a[0] < 1.0: continue
        gi = g[(g >= a[0]) & (g <= a[-1])]
        if len(gi) < 11: continue
        px = np.interp(gi, a, r["x"].to_numpy()[keep]); py = np.interp(gi, a, r["y"].to_numpy()[keep]); P = np.c_[px, py]
        v = np.zeros(len(gi)); v[5:-5] = np.linalg.norm(P[10:] - P[:-10], axis=1) / 1.0   # 1 s-baseline speed (robust to cuboid jitter)
        v[:5] = v[5]; v[-5:] = v[-6]
        T[tid] = dict(t=gi, xy=P, v=v, L=float(r["L"].median()), W=float(r["W"].median()), cat=r["cat"].iloc[0])
    S["tracks"] = T
    return S

def ego_frame(p, o, yaw):
    d = p - o; c, s = math.cos(yaw), math.sin(yaw)
    return np.c_[c*d[..., 0] + s*d[..., 1], -s*d[..., 0] + c*d[..., 1]] if d.ndim == 2 else np.array([c*d[0] + s*d[1], -s*d[0] + c*d[1]])

def dist_poly(p, P):
    A, B = P[:-1], P[1:]; AB = B - A; AP = p - A
    u = np.clip((AP * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(A + u[:, None]*AB - p, axis=1)))

def actor_at(tr, i_t):
    j = int(round((i_t - tr["t"][0]) / DT))
    return (tr["xy"][j], tr["v"][j]) if 0 <= j < len(tr["t"]) else None

def scene(u):
    d = common.NAS / "pool" / u
    try: S = load(d)
    except Exception as e: return dict(uuid=u, error=repr(e)), [], []
    t, sp = S["t"], S["speed"]; n = len(t); w = int(round((HIST + FUT) / DT))
    held = max((sp[i:i + w + 1].min() for i in range(0, n - w)), default=float("nan"))
    anchors = []
    for ta in ANCH:
        i = int(round(ta / DT)); i0, i1 = i - int(HIST / DT), i + int(FUT / DT)
        if i1 >= n: break
        wmin, wmax = float(sp[i0:i1 + 1].min()), float(sp[i0:i1 + 1].max())
        row = dict(uuid=u, t=float(ta), v=float(sp[i]), wmin=wmin, wmax=wmax, A=wmin >= 16.0)
        if row["A"]:
            path = S["xy"][i:i1 + 1]; o, yw = S["xy"][i], S["yaw"][i]
            row["min_acc"] = float(S["acc"][i:i1 + 1].min()); row["brake"] = row["min_acc"] < -1.0
            row["dyaw_deg"] = float(np.degrees(S["yaw"][i1] - S["yaw"][i])); row["curve"] = abs(row["dyaw_deg"]) > 10.0
            best = None
            for tid, tr in S["tracks"].items():
                r = actor_at(tr, ta)
                if r is None or r[1] < 0.5: continue
                e = ego_frame(r[0], o, yw); gap = e[0] - tr["L"] / 2
                if e[0] > 0 and gap <= 50.0 and dist_poly(r[0], path) <= 2.0 and (best is None or gap < best[0]): best = (float(gap), tr["cat"], float(r[1]))
            row["follow"] = best is not None
            if best: row["lead_gap"], row["lead_cat"], row["lead_v"] = best
            row["cruise"] = not (row["brake"] or row["follow"] or row["curve"])
        anchors.append(row)
    # D-B move-offs
    ev = []; still = sp < 0.3; i = 0
    while i < n:
        if not still[i]: i += 1; continue
        j = i
        while j < n and still[j]: j += 1
        if (j - i) * DT >= 2.0 and j < n:
            k = next((k for k in range(j, min(n, j + int(6.0 / DT) + 1)) if sp[k] > 2.0), None)
            if k is not None:
                o, yw = S["xy"][j - 1], S["yaw"][j - 1]
                def lat(a):
                    b = min(n - 1, a + int(4.0 / DT)); e = ego_frame(S["xy"][a:b + 1], o, yw); return float(np.abs(e[:, 1]).max()), float(e[np.abs(e[:, 1]).argmax(), 1]), (b - a) * DT
                l_on, l_on_signed, span_on = lat(j - 1); l_dep, _, span_dep = lat(k)
                best = None; e_span = (t[i], t[min(n - 1, k + int(4.0 / DT))])
                for tid, tr in S["tracks"].items():
                    r = actor_at(tr, t[j - 1])
                    if r is None: continue
                    m = (tr["t"] >= e_span[0] - 1e-6) & (tr["t"] <= e_span[1] + 1e-6)
                    if tr["v"][m].max() >= 0.5: continue
                    e = ego_frame(r[0], o, yw); gap = e[0] - tr["L"] / 2
                    if 0.0 <= gap <= 8.0 and abs(e[1]) <= 2.0 and (best is None or gap < best[0]): best = (float(gap), float(e[1]), tr["cat"])
                ev.append(dict(uuid=u, t_stop_start=float(t[i]), stop_from_clip_start=bool(i == 0), t_move=float(t[j - 1]), t_depart=float(t[k]),
                               lat4_from_move=l_on, lat4_signed=l_on_signed, span_from_move=span_on, lat4_from_depart=l_dep, span_from_depart=span_dep,
                               actor=best is not None, actor_gap=best[0] if best else None, actor_lat=best[1] if best else None, actor_cat=best[2] if best else None))
        i = j
    A = [a for a in anchors if a["A"]]
    sc = dict(uuid=u, dur=S["dur"], n_pose=S["n_pose"], max_dt=S["max_dt"], n_tracks=len(S["tracks"]), n_anchors=len(anchors), vmax=float(sp.max()), vheld=float(held),
              A_anchors=len(A), A_min=min((a["wmin"] for a in A), default=None), A_max=max((a["wmax"] for a in A), default=None), moveoffs=len(ev))
    return sc, anchors, ev

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    nurec, raw, pool = common.load_views()
    todo = sorted(u for u in pool if (common.NAS / "pool" / u / "DONE").exists())
    assert not (set(todo) & raw) and not (set(todo) & nurec), "exam scene in pool dir"
    present = {p.name for p in (common.NAS / "pool").iterdir()}; assert not (present & raw), "exam uuid folder on NAS"
    if a.limit: todo = todo[:a.limit]
    print("pool", len(pool), "fetched", len(todo), flush=True)
    with mp.Pool(a.procs) as p: res = p.map(scene, todo, chunksize=8)
    sc = pd.DataFrame([r[0] for r in res]); an = pd.DataFrame([x for r in res for x in r[1]]); ev = pd.DataFrame([x for r in res for x in r[2]])
    sfx = f"_n{a.limit}" if a.limit else ""
    sc.to_csv(common.HERE / f"d2_scenes{sfx}.csv", index=False); an.to_csv(common.HERE / f"d2_anchors{sfx}.csv", index=False); ev.to_csv(common.HERE / f"d2_moveoffs{sfx}.csv", index=False)
    print("scenes", len(sc), "errors", int(sc["error"].notna().sum()) if "error" in sc else 0, "anchors", len(an), "A anchors", int(an["A"].sum()), "move-offs", len(ev))
