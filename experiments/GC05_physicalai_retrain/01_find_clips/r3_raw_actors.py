"""3c stage 2b: for every raw_train move-off with lateral >= 1.0 m, range-read that clip's obstacle.offline member (never an exam member),
transform rig-frame boxes to the clip world frame with egomotion.offline, attach the stationary-actor test with the bumper gap.
Bumper offset F = rear_axle_to_bbox_center + length/2 from calibration/vehicle_dimensions. -> r3_raw_moveoffs.csv"""
import os, io, sys, hashlib, zipfile, concurrent.futures as cf, numpy as np, pandas as pd, pyarrow.parquet as pq, common, hfrange, r3_core as K, r3_rawio as rio, r3_raw_scan as SC
V = common.load_split_views(); _, EXAM, _ = common.load_views()
def obstacles_world(S_raw_ego, b):
    """S_raw_ego: (ts_s, xy, yaw_unwrapped) raw samples; b: obstacle parquet bytes (reference_frame == 'rig')."""
    ts, xy, yaw = S_raw_ego; d = pq.read_table(io.BytesIO(b)).to_pandas()
    assert (d.reference_frame == "rig").all()
    t = d.reference_frame_timestamp_us.to_numpy().astype(np.float64) / 1e6
    ex, ey, eyaw = np.interp(t, ts, xy[:, 0]), np.interp(t, ts, xy[:, 1]), np.interp(t, ts, yaw); c, s = np.cos(eyaw), np.sin(eyaw)
    x, y = d.center_x.to_numpy(), d.center_y.to_numpy()
    return pd.DataFrame({"tid": d.track_id.astype(str), "ts": d.timestamp_us.to_numpy().astype(np.float64) / 1e6 - ts[0], "x": ex + c*x - s*y, "y": ey + s*x + c*y,
                         "yaw": K.yaw_q(d.orientation_x.to_numpy(), d.orientation_y.to_numpy(), d.orientation_z.to_numpy(), d.orientation_w.to_numpy()) + eyaw,
                         "L": d.size_x.to_numpy(), "W": d.size_y.to_numpy(), "cat": d.label_class})
def raw_ego(b):
    d = pq.read_table(io.BytesIO(b)).to_pandas().sort_values("timestamp"); ts = d["timestamp"].to_numpy() / 1e6
    return ts, d[["x", "y"]].to_numpy(), np.unwrap(K.yaw_q(d.qx.to_numpy(), d.qy.to_numpy(), d.qz.to_numpy(), d.qw.to_numpy()))
def vehdims(c):
    p = common.RAW_NAS / f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet"
    return pd.read_parquet(p) if p.exists() else None
def chunk(args):
    c, ev = args; out = []; need = sorted(set(ev.uuid)); assert not (set(need) & EXAM)
    dst = common.RAW_NAS / f"labels/obstacle.offline.members/chunk_{c:04d}"; dst.mkdir(parents=True, exist_ok=True)
    path = f"labels/obstacle.offline/obstacle.offline.chunk_{c:04d}.zip"; zf = f = None; ob = {}
    for u in need:
        name = f"{u}.obstacle.offline.parquet"; p = dst / name
        if not p.exists():
            for a in range(4):
                try:
                    if zf is None: f, zf = hfrange.open_zip(rio.url(path), os.environ["HF_TOKEN"])
                    data, _ = hfrange.fetch_member(zf, f, name); break
                except KeyError: data = None; break
                except Exception as e:
                    zf = None
                    if a == 3: raise
            if data is None: continue
            p.write_bytes(data); common.manifest_add(u, f"{path}::{name}", len(data), str(p), hashlib.sha256(data).hexdigest())
        ob[u] = p.read_bytes()
    egos = {u: b for u, b in SC.members(c) if u in ob}; vd = vehdims(c)
    for u, g in ev.groupby("uuid"):
        if u not in ob: 
            for e in g.to_dict("records"): e["actor_status"] = "no_obstacle_member"; out.append(e)
            continue
        S = SC.ego_from_bytes(egos[u]); S = K.build_tracks(S, obstacles_world(raw_ego(egos[u]), ob[u]))
        if vd is not None and u in vd.index: F = float(vd.loc[u, "rear_axle_to_bbox_center"] + vd.loc[u, "length"] / 2); fsrc = "vehicle_dimensions"
        else: F = 4.07; fsrc = "fallback_4.07"
        me = {round(e["t_move"], 1): e for e in K.moveoffs(S, u)}
        for e in g.to_dict("records"):
            e2 = K.attach_actor(S, dict(me[round(e["t_move"], 1)]), F); e.update({k: e2[k] for k in e2 if k.startswith(("actor_", "gap_", "alat_", "acat_"))}); e["F"] = F; e["F_src"] = fsrc; e["actor_status"] = "ok"; e["n_tracks"] = len(S["tracks"]); out.append(e)
    return out
if __name__ == "__main__":
    ev = pd.read_csv(common.HERE / "r3_raw_moveoffs_ego.csv"); cand = ev[ev.lat4 >= 1.0]
    if len(sys.argv) > 1: cand = cand[cand.chunk.isin([int(x) for x in sys.argv[1].split(",")])]
    jobs = [(c, g) for c, g in cand.groupby("chunk")]; print("candidate events", len(cand), "clips", cand.uuid.nunique(), "chunks", len(jobs), flush=True)
    res = []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(chunk, jobs)):
            res += r
            if i % 100 == 0: print(i, len(jobs), flush=True)
    out = pd.DataFrame(res).drop(columns=["_i", "_j", "_k"], errors="ignore"); out.to_csv(common.HERE / ("r3_raw_moveoffs.csv" if len(sys.argv) == 1 else "r3_raw_moveoffs_test.csv"), index=False)
    print("events", len(out), out.actor_status.value_counts().to_dict(), "actor_bum", int((out.actor_bum == True).sum()))
