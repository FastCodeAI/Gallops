#!/usr/bin/env python
"""S2 Steps 3-5. Run:  cd adapters/drivejepa/src/navsim_v2 && CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:<03_grade_candidates>:<01_find_clips> <conda drive-jepa python> <this> [align|pilot|btest|all]"""
import sys, io, gzip, lzma, math, pickle, zipfile, time, numpy as np, pandas as pd
import s2_common as C, s2_scene as SC, s2_grade as G
K = SC.K; HERE = C.HERE; S1NAS = C.NAS / "s1"; CACHE = S1NAS / "cache"
pilot = pd.read_csv(HERE / "pilot_clips.csv"); scenes = pd.read_csv(HERE / "scenes.csv"); C.assert_nurec_train(scenes.uuid)
import os
FIX = os.environ.get("S2_MAPFIX", "1") == "1"      # S2.1 map fixes (boundary strips + on-route gap bridging); S2_MAPFIX=0 reproduces S2
_S = {}
def scene(u):
    if u not in _S:
        if len(_S) > 3: _S.clear()
        S = SC.load_scene(u, fix=FIX); SC.route_lanes(S); _S[u] = S
    return _S[u]

# ---------------------------------------------------------------- Step 3: alignment
def raw_ego(u, chunk):
    z = zipfile.ZipFile(C.S0C.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{chunk:04d}.zip")
    d = pd.read_parquet(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet"))).sort_values("timestamp")
    return d.timestamp.values / 1e6, d[["x", "y"]].values, np.unwrap(K.yaw_q(d.qx.values, d.qy.values, d.qz.values, d.qw.values))
def kabsch(A, B):
    """rigid 2D transform (R, t) minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, d]) @ U.T
    return R, cb - R @ ca
def align(u, chunk):
    S = scene(u); tr, pr, yr = raw_ego(u, chunk); tc = (S["ts_abs_us"] - S["ts_abs_us"][0]) / 1e6; pc, yc = S["raw_xy"], S["raw_yaw"]; best = None
    # Stage 1: clock offset with NO spatial transform (both frames have their origin at the first pose; a free fit cannot separate a
    # time shift from a translation on a straight road). Stage 2: rigid fit at that offset.
    ident = None
    for dlt in np.arange(-0.30, 0.3001, 0.01):
        m = (tr + dlt >= tc[0]) & (tr + dlt <= tc[-1]); q = np.c_[np.interp(tr[m] + dlt, tc, pc[:, 0]), np.interp(tr[m] + dlt, tc, pc[:, 1])]
        r0 = float(np.median(np.linalg.norm(pr[m] - q, axis=1)))
        if ident is None or r0 < ident[0] - 1e-6: ident = (r0, dlt)
    dlt = ident[1]; m = (tr + dlt >= tc[0]) & (tr + dlt <= tc[-1]); q = np.c_[np.interp(tr[m] + dlt, tc, pc[:, 0]), np.interp(tr[m] + dlt, tc, pc[:, 1])]
    res_ident0 = float(np.median(np.linalg.norm(pr - np.c_[np.interp(np.clip(tr, tc[0], tc[-1]), tc, pc[:, 0]), np.interp(np.clip(tr, tc[0], tc[-1]), tc, pc[:, 1])], axis=1)))
    R, t = kabsch(pr[m], q); res = np.linalg.norm((R @ pr[m].T).T + t - q, axis=1); best = (np.median(res), dlt, R, t, res, m)
    med, dlt, R, t, res, m = best; rot = math.atan2(R[1, 0], R[0, 0]); hres = np.degrees(np.abs(K.wrap(yr[m] + rot - np.interp(tr[m] + dlt, tc, yc))))
    # usdz rig_trajectories.json cross-check (same poses in the reconstruction world frame)
    import json
    rj = json.load(open(SC.SC / u / "rig_trajectories.json"))["rig_trajectories"][0]; Tw = np.array(rj["T_rig_worlds"]); tw = (np.array(rj["T_rig_world_timestamps_us"]) - S["ts_abs_us"][0]) / 1e6
    mm = (tw >= tc[0]) & (tw <= tc[-1]); q = np.c_[np.interp(tw[mm], tc, pc[:, 0]), np.interp(tw[mm], tc, pc[:, 1])]; R2, t2 = kabsch(Tw[mm][:, :2, 3], q); res2 = np.linalg.norm((R2 @ Tw[mm][:, :2, 3].T).T + t2 - q, axis=1)
    return dict(uuid=u, dt_s=round(float(dlt), 2), rot_deg=math.degrees(rot), tx=float(t[0]), ty=float(t[1]), pos_med=float(np.median(res)), pos_max=float(res.max()), head_med=float(np.median(hres)), head_max=float(hres.max()),
                n=int(m.sum()), ident_dt0_pos_med=res_ident0, ident_best_pos_med=ident[0], usdz_ts0_minus_raw0_us=int(S["ts_abs_us"][0]), rigjson_pos_med=float(np.median(res2)), rigjson_pos_max=float(res2.max()), rigjson_n=int(mm.sum())), (dlt, R, t)

def scene_png(u, typ, i):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    S = scene(u); fig, ax = plt.subplots(figsize=(11, 11)); c = S["xy"][i]; r = 70
    from matplotlib.path import Path as MP; from matplotlib.patches import PathPatch
    for p in S["drivable"]:
        v, cd = [], []
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords); v += xy.tolist(); cd += [MP.MOVETO] + [MP.LINETO] * (len(xy) - 2) + [MP.CLOSEPOLY]
        ax.add_patch(PathPatch(MP(v, cd), fc="#e5e7eb", ec="none", zorder=0))
    for lid, l in S["lanes"].items():
        x, y = l["poly"].exterior.xy; ax.plot(x, y, c="#3b82f6" if lid in S["route_ids"] else "#9ca3af", lw=0.7, zorder=1); ax.plot(l["center"][:, 0], l["center"][:, 1], c="#9ca3af", lw=0.4, ls=":", zorder=1)
    for _, p, _ in S["inter"]:
        x, y = p.exterior.xy; ax.fill(x, y, fc="#fde68a", ec="#d97706", alpha=0.4, lw=0.6, zorder=1)
    for _, w, cat, _, imp in S["wait"]:
        if not imp: ax.plot(w[:, 0], w[:, 1], c="#dc2626", lw=1.6, zorder=3)
    for x, y in S["tl"]: ax.plot(x, y, "s", c="#16a34a", ms=4, zorder=3)
    for tid, tr in S["tracks"].items():
        j = K.idx_at(tr, S["t"][i])
        if j is None: continue
        q = K.rect(tr["xy"][j], tr["yaw"][j], tr["L"], tr["W"]); ax.fill(q[:, 0], q[:, 1], fc="#f97316" if tr["v"][j] >= 0.5 else "#6b7280", ec="k", lw=0.4, alpha=0.8, zorder=4)
    ax.plot(S["xy"][:, 0], S["xy"][:, 1], c="#111827", lw=1.8, zorder=5, label="human path (usdz)"); P, _ = SC.centerline_at(S, i)
    if P is not None: ax.plot(P[:, 0], P[:, 1], c="#7c3aed", lw=1.0, ls="--", zorder=4, label="centreline at t")
    e = K.rect(c + 1.46 * np.array([math.cos(S["yaw"][i]), math.sin(S["yaw"][i])]), S["yaw"][i], 5.176, 2.297); ax.fill(e[:, 0], e[:, 1], fc="#ef4444", ec="k", lw=0.6, zorder=6, label="ego at t (scorer box)")
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{typ} {u[:8]} t={S['t'][i]:.1f}s — blue: on-route lanes, grey fill: drivable area, yellow: intersections, red: explicit stop/yield lines, boxes: agents (orange moving)", fontsize=8)
    (HERE / "scene_png").mkdir(exist_ok=True); fn = HERE / "scene_png" / f"{typ}_{u[:8]}.png"; fig.savefig(fn, dpi=90, bbox_inches="tight"); plt.close(fig); return fn.name

def run_align():
    rows = []; 
    for r in pilot.itertuples():
        a, _ = align(r.uuid, r.chunk); a["type"] = r.type; S = scene(r.uuid); a["lanes"] = len(S["lanes"]); a["intersections"] = len(S["inter"]); a["stop_lines_explicit"] = sum(1 for w in S["wait"] if not w[4]); a["traffic_lights"] = len(S["tl"])
        a["tracks"] = len(S["tracks"]); a["route_lanes"] = len(S["route_ids"]); a["map_fix"] = FIX; a["drivable_m2"] = round(sum(p.area for p in S["drivable"])); a["png"] = "scene_png/" + scene_png(r.uuid, r.type, 80); rows.append(a); print(a, flush=True)
    pd.DataFrame(rows).to_csv(HERE / "align.csv", index=False)

# ---------------------------------------------------------------- Step 4: pilot grading
def run_pilot():
    import torch
    sm = pd.read_csv(HERE / "samples.csv"); al = pd.read_csv(HERE / "align.csv").set_index("uuid"); rows = []; meta = []; t0 = time.time()
    for r in pilot.itertuples():
        S = scene(r.uuid); dlt = float(al.loc[r.uuid, "dt_s"])
        for q in sm[sm.uuid == r.uuid].itertuples():
            i = int(round((q.t_us / 1e6 + dlt) / K.DT)); tokdir = CACHE / r.uuid / q.token; fp = tokdir / "metric_cache.pkl"
            mc, info = G.build_cache(S, i, q.token, fp)
            base = dict(type=r.type, uuid=r.uuid, token=q.token, t=q.t_us / 1e6, speed=q.speed)
            if mc is None: rows.append(dict(base, plan="-", status=info)); continue
            z = np.load(S1NAS / "menus" / r.uuid / f"{q.token}.npz"); props = z["proposals"]; chosen = int(z["pdm_score"].argmax()); human = z["label"]
            poses = np.concatenate([props, human[None], np.zeros((1, 8, 3))]); sc, ex = G.grade(mc, poses)
            with lzma.open(fp, "wb", preset=0) as f: pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)                    # same container as MetricCache.dump (metric_cache.py:57-61)
            with gzip.open(tokdir / "pad_label2.gz", "wb", compresslevel=1) as f:
                pickle.dump({"label2_scores": torch.tensor(sc, dtype=torch.float32), "label2_terms": G.TERMS, "label2_plans": [f"proposal_{k}" for k in range(32)] + ["human", "stay"], "metric_cache_path": str(fp), "has_label2": True}, f)
            meta.append(str(fp))
            for k in range(34):
                name = f"p{k:02d}" if k < 32 else ("human" if k == 32 else "stay")
                rows.append(dict(base, plan=name, status="ok", chosen=bool(k == chosen), model_pdm=float(z["pdm_score"][k]) if k < 32 else np.nan, **{t_: float(sc[k, j]) for j, t_ in enumerate(G.TERMS)},
                                 progress_m=float(ex["progress_raw"][k]), multi_lane=float(ex["areas"][k, 0]), nondrivable=float(ex["areas"][k, 1]), offroute=float(ex["areas"][k, 2]), cur_lane=info["cur_lane"], n_agents=info["n_agents0"], has_past=info["has_past"]))
        print(r.uuid[:8], r.type, "anchors", int((sm.uuid == r.uuid).sum()), round(time.time() - t0), "s", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "pilot_grades.csv", index=False)
    (CACHE / "metadata").mkdir(exist_ok=True); open(CACHE / "metadata" / "s2_metric_cache_metadata.csv", "w").write("file_name\n" + "\n".join(meta) + "\n")

# ---------------------------------------------------------------- Step 5: B formula test (human vs stay, usdz human path)
def run_btest():
    ev = pd.read_csv(HERE / "b_tight_events.csv"); po = pd.read_csv(HERE / "b_pullout_events.csv"); rows = []
    jobs = [("tight", e.uuid, max(0.5, math.floor((e.t_pass - 0.5) / 0.5) * 0.5), dict(track=e.track, clearance=e.clearance, t_first=e.t, t_pass=e.t_pass, ego_speed_s0=e.ego_speed, cat=e.cat, path_lat=e.path_lat_at_pass, dyaw_pass=e.dyaw_at_pass_deg)) for e in ev.itertuples()]
    jobs += [("pullout", e.uuid, math.floor(e.t_move / 0.5) * 0.5, dict(clearance=np.nan, t_pass=e.t_depart, t_first=e.t_move, cat=e.acat_bum, path_lat=e.lat4_signed, dyaw_pass=e.dyaw4_deg)) for e in po.itertuples()]
    for kind, u, ta, info0 in jobs:
        S = scene(u); ta = min(ta, 16.0); i = int(round(ta / K.DT)); mc, info = G.build_cache(S, i, f"{u[:8]}_{int(ta*1000):05d}")
        base = dict(kind=kind, uuid=u, t_anchor=ta, speed=float(S["speed"][i]), **info0)
        if mc is None: rows.append(dict(base, status=info)); continue
        sc, ex = G.grade(mc, np.stack([info["human_rel"], np.zeros((8, 3))]), diag_no_ddc=True)
        row = dict(base, status="ok", cur_lane=info["cur_lane"], n_agents=info["n_agents0"], human_score_no_ddc=float(ex["score_no_ddc"][0]), stay_score_no_ddc=float(ex["score_no_ddc"][1]))
        # seconds the human's box centre spends in an opposite-direction lane (inside a lane polygon, not on-route, lane heading > 135 deg from the ego's)
        from shapely.geometry import Point
        opp = oth = gap = 0; vopp = []
        for j in range(i, i + 41):
            ctr = S["xy"][j] + 1.461 * np.array([math.cos(S["yaw"][j]), math.sin(S["yaw"][j])]); P = Point(ctr)
            if any(br["poly"].contains(P) for br in S["bridges"]): continue
            inl = [lid for lid in (S["lane_ids"][q] for q in S["lane_tree"].query(P)) if S["lanes"][lid]["poly"].contains(P)]
            if any(l in S["route_ids"] for l in inl): continue
            if not inl: gap += 1
            elif all(abs(K.wrap(SC.lane_heading_at(S["lanes"][l], ctr)[0] - S["yaw"][j])) > math.radians(135) for l in inl): opp += 1; vopp.append(S["speed"][j])
            else: oth += 1
        row.update(opp_lane_s=opp * K.DT, offroute_other_s=oth * K.DT, no_lane_s=gap * K.DT, opp_lane_speed=float(np.mean(vopp)) if vopp else np.nan)
        if "track" in info0 and str(info0["track"]) in S["tracks"]:
            tr = S["tracks"][str(info0["track"])]; m = (tr["t"] >= ta - 1e-6) & (tr["t"] <= ta + 4.0 + 1e-6); row["obstacle_max_speed"] = float(tr["v"][m].max()) if m.any() else np.nan; row["obstacle_stationary"] = bool(m.any() and tr["v"][m].max() < 0.5)
        for k, name in enumerate(("human", "stay")):
            row.update({f"{name}_{t_}": float(sc[k, j]) for j, t_ in enumerate(G.TERMS)}); row[f"{name}_progress_m"] = float(ex["progress_raw"][k]); row[f"{name}_multi_lane"] = float(ex["areas"][k, 0]); row[f"{name}_nondrivable"] = float(ex["areas"][k, 1]); row[f"{name}_offroute"] = float(ex["areas"][k, 2])
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "btest.csv", index=False); print("btest rows", len(rows))

def png_compare(u, typ, tag, i=80):
    """S2.1 A3: before/after drivable area for one scene."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.path import Path as MP; from matplotlib.patches import PathPatch
    def fill(ax, p, **kw):
        v, cd = [], []
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords); v += xy.tolist(); cd += [MP.MOVETO] + [MP.LINETO] * (len(xy) - 2) + [MP.CLOSEPOLY]
        ax.add_patch(PathPatch(MP(v, cd), **kw))
    S0_ = SC.load_scene(u, fix=False); SC.route_lanes(S0_); S1_ = SC.load_scene(u, fix=True); SC.route_lanes(S1_)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10)); c = S1_["xy"][i]; r = 45
    for ax, S, name in ((axs[0], S0_, "S2: lanes + intersections"), (axs[1], S1_, "S2.1: + road-boundary strips (green) + bridged on-route gaps (red)")):
        for p in S["drivable"]: fill(ax, p, fc="#e5e7eb", ec="none", zorder=0)
        if S is S1_:
            from shapely.ops import unary_union
            add = unary_union(S["drivable"]).difference(unary_union(S0_["drivable"]))
            for q in (add.geoms if hasattr(add, "geoms") else [add]):
                if q.geom_type == "Polygon" and q.area > 0.05: fill(ax, q, fc="#86efac", ec="none", zorder=1)
            for b in S["bridges"]: fill(ax, b["poly"].buffer(0.15), fc="#ef4444", ec="none", zorder=3)
        for lid, l in S["lanes"].items():
            x, y = l["poly"].exterior.xy; ax.plot(x, y, c="#3b82f6" if lid in S["route_ids"] else "#9ca3af", lw=0.6, zorder=2)
        import pyarrow.parquet as pq
        for a_ in pq.read_table(SC.SC / u / "clipgt__road_boundary.parquet").to_pandas()["road_boundary"]:
            if a_["location"] is not None and len(a_["location"]) >= 2: P = SC._xy(a_["location"]); ax.plot(P[:, 0], P[:, 1], c="#92400e", lw=1.0, zorder=3)
        for tid, tr in S["tracks"].items():
            j = K.idx_at(tr, S["t"][i])
            if j is None: continue
            q = K.rect(tr["xy"][j], tr["yaw"][j], tr["L"], tr["W"]); ax.fill(q[:, 0], q[:, 1], fc="#f97316" if tr["v"][j] >= 0.5 else "#6b7280", ec="k", lw=0.4, alpha=0.8, zorder=4)
        ax.plot(S["xy"][:, 0], S["xy"][:, 1], c="#111827", lw=1.6, zorder=5)
        e = K.rect(c + 1.46 * np.array([math.cos(S["yaw"][i]), math.sin(S["yaw"][i])]), S["yaw"][i], 5.176, 2.297); ax.fill(e[:, 0], e[:, 1], fc="#ef4444", ec="k", lw=0.6, zorder=6)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_aspect("equal"); ax.set_title(f"{typ} {u[:8]} t={S['t'][i]:.1f}s — {name}; brown: road boundaries; blue: on-route lanes", fontsize=9)
    (HERE / "s21_png").mkdir(exist_ok=True); fn = HERE / "s21_png" / f"{tag}_{u[:8]}.png"; fig.savefig(fn, dpi=80, bbox_inches="tight"); plt.close(fig); print(fn.name)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "png":
        png_compare("3259c4fe-f663-401d-b0a8-c7e944bcab8b", "T", "parking_strip", 80); png_compare("ad134e5b-118c-4dba-9217-2438bb0a747b", "A", "highspeed_gap", 80); png_compare("0771fd01-94d4-43e9-bf48-781882888352", "T", "tight_pass", 60); sys.exit(0)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("align", "all"): run_align()
    if what in ("pilot", "all"): run_pilot()
    if what in ("btest", "all"): run_btest()
