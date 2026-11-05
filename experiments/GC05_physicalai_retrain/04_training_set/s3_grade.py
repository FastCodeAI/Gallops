#!/usr/bin/env python
"""Part B: grade every nurec_train anchor (S2.1 grader + fixed map, D2 real box + human drivable-area drop, D3 exemption), write the
metric cache and the anchor index for the original and the extended vocabulary. Resumable per scene.
Run via run_s3_grade.sh (conda drive-jepa python, cwd = vendored navsim_v2, CPU)."""
import os, sys, io, gzip, lzma, math, time, pickle, zipfile, zlib, multiprocessing as mp, numpy as np, pandas as pd
import s3_common as C, s2_scene as SC, s2_grade as G, s3_scorer as SS
SC.SC = C.SCENES            # S2 scene loader: read usdz map members from the S3 NAS root
K = SC.K; OUT = C.HERE / os.environ.get("S3_ROWS", "grade_rows"); OUT.mkdir(exist_ok=True)
VOC = np.load(C.HERE / "vocab_10240.npy")[:, 4::5].astype(np.float32); NORIG = 8192; assert len(VOC) == 10240
DISP = np.linalg.norm(VOC[:, -1, :2], axis=1)                    # straight-line length of each shape
LAB_I = pd.read_parquet(C.HERE / "labels_index.parquet"); LAB = np.load(C.HERE / "labels.npy"); LAB_ROW = {s: k for k, s in enumerate(LAB_I.sample_id)}
FULL = os.environ.get("S3_FULL_VOCAB", "0") == "1"               # 1 = score all 10,240 shapes (used to validate the prefilter)

def raw_ego(u, ch):
    z = zipfile.ZipFile(C.S0C.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{ch:04d}.zip"); d = pd.read_parquet(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet"))).sort_values("timestamp")
    return d.timestamp.values / 1e6, d[["x", "y"]].values
def clock_offset(S, u, ch):
    """raw clip clock -> usdz clock: offset minimising the position residual with no spatial transform (s2_run.align, stage 1)."""
    tr, pr = raw_ego(u, ch); tc = (S["ts_abs_us"] - S["ts_abs_us"][0]) / 1e6; pc = S["raw_xy"]; best = None
    for dlt in np.arange(-0.30, 0.3001, 0.01):
        m = (tr + dlt >= tc[0]) & (tr + dlt <= tc[-1]); r = float(np.median(np.linalg.norm(pr[m] - np.c_[np.interp(tr[m] + dlt, tc, pc[:, 0]), np.interp(tr[m] + dlt, tc, pc[:, 1])], axis=1)))
        if best is None or r < best[0] - 1e-6: best = (r, float(dlt))
    return best[1], best[0]
def stationary_ahead(S, i, F):
    o, yw = S["xy"][i], S["yaw"][i]
    for tr in S["tracks"].values():
        j = K.idx_at(tr, S["t"][i])
        if j is None or tr["v"][j] >= 0.5: continue
        e = K.ego_frame(tr["xy"][j], o, yw); gap = e[0] - tr["L"] / 2 - F
        if 0.0 <= gap <= SS.D3_AHEAD_M and abs(e[1]) <= SS.D3_LAT_M: return True
    return False

def scene(args):
    u, ch = args; fn = OUT / f"{u}.parquet"
    if fn.exists(): return u, 0, 0.0
    C.assert_train([u], nurec=True); t0 = time.time(); rows = []
    try:
        S = SC.load_scene(u, fix=True); SC.route_lanes(S); dlt, res0 = clock_offset(S, u, ch)
        vd = pd.read_parquet(C.S0C.RAW_NAS / f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{ch:04d}.parquet").loc[u]
        vp = SS.vehicle_params(vd["length"], vd["width"], vd["rear_axle_to_bbox_center"], vd["wheelbase"], vd["height"]); F = float(vd["rear_axle_to_bbox_center"] + vd["length"] / 2)
    except Exception as e:
        pd.DataFrame([dict(uuid=u, sample_id="-", status="scene error: " + repr(e)[:150])]).to_parquet(fn); return u, 0, time.time() - t0
    for q in LAB_I[LAB_I.uuid == u].itertuples():
        base = dict(uuid=u, sample_id=q.sample_id, t_us=q.t_us, speed=q.speed, clock_offset_s=dlt, align_res_m=res0, veh_len=float(vd["length"]), veh_wid=float(vd["width"]))
        try:
            i = int(round((q.t_us / 1e6 + dlt) / K.DT)); tokdir = C.CACHE / u / q.sample_id; fp = tokdir / "metric_cache.pkl"
            mc, info = G.build_cache(S, i, q.sample_id, fp, vp=vp)
            if mc is None: rows.append(dict(base, status=info)); continue
            mc.d3_stationary_ahead = stationary_ahead(S, i, F); mc.real_vehicle = dict(length=float(vd["length"]), width=float(vd["width"]), rear_axle_to_bbox_center=float(vd["rear_axle_to_bbox_center"]), wheelbase=float(vd["wheelbase"]))
            human = LAB[LAB_ROW[q.sample_id]]; sc, ex = SS.score(mc, np.stack([human, np.zeros((8, 3))]))
            row = dict(base, status="ok", stationary_ahead=bool(mc.d3_stationary_ahead), n_agents=info["n_agents0"], ref_progress=ex["ref_progress"],
                       **{f"human_{t}": float(sc[0, k]) for k, t in enumerate(SS.TERMS)}, **{f"stay_{t}": float(sc[1, k]) for k, t in enumerate(SS.TERMS)},
                       human_exempt=bool(ex["exempt"][0]), stay_exempt=bool(ex["exempt"][1]), human_ddc_standard=float(ex["ddc_standard"][0]), human_score_standard=float(ex["score_standard"][0]),
                       human_opp_s=float(ex["opp_s"][0]), human_max_speed=float(ex["max_speed"][0]), human_nondrivable=float(ex["nondrivable"][0]))
            if sc[0, 1] < 1.0:                                   # D2: the human path fails the drivable-area gate -> anchor dropped, nothing written
                rows.append(dict(row, status="dropped_d2")); continue
            # vocabulary scoring. Exact prefilter for the >= 0.95 rule: a shape needs EP >= 0.86 even with every other term at 1, i.e.
            # progress >= 0.86 x max(own, reference); shapes whose straight-line length is below 0.6 x reference progress - 3 m cannot
            # qualify and are not simulated (validated against full scoring, see REPORT). No pruning when the reference moves <= 5 m.
            cand = np.arange(len(VOC)) if (FULL or ex["ref_progress"] <= 5.0) else np.where(DISP >= 0.6 * ex["ref_progress"] - 3.0)[0]
            fin = np.full(len(VOC), np.nan, dtype=np.float32); n_ex = 0
            for b in range(0, len(cand), 2048):
                ii = cand[b:b + 2048]; s_, e_ = SS.score(mc, VOC[ii]); fin[ii] = s_[:, -1]; n_ex += int(e_["exempt"].sum())
            ok = np.where(np.nan_to_num(fin, nan=-1.0) >= 0.95)[0]; rng = np.random.default_rng(zlib.crc32(q.sample_id.encode()))
            pick = lambda a: np.sort(rng.choice(a, size=256, replace=False)) if len(a) > 256 else a            # convert_to_index.py:17-22 (seeded here)
            io_, ie_ = ok[ok < NORIG], ok; tokdir.mkdir(parents=True, exist_ok=True)
            with lzma.open(fp, "wb", preset=0) as f: pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(tokdir / "anchors_scores_index_orig.npy", pick(io_)); np.save(tokdir / "anchors_scores_index_ext.npy", pick(ie_)); np.save(tokdir / "anchors_scores_final.npy", fin.astype(np.float16))
            import torch
            with gzip.open(tokdir / "pad_label2.gz", "wb", compresslevel=1) as f:
                pickle.dump({"label2_scores": torch.tensor(sc, dtype=torch.float32), "label2_terms": SS.TERMS, "label2_plans": ["human", "stay"], "has_label2": True, "direction_exempt": bool(ex["exempt"][0]), "metric_cache_path": str(fp)}, f)
            rows.append(dict(row, n_scored=len(cand), n_ok_orig=len(io_), n_ok_ext=len(ie_), n_ok_added=int((ok >= NORIG).sum()), best_orig=float(np.nanmax(fin[:NORIG])) if np.isfinite(fin[:NORIG]).any() else np.nan,
                             best_ext=float(np.nanmax(fin)) if np.isfinite(fin).any() else np.nan, vocab_exemptions=n_ex))
        except Exception as e:
            rows.append(dict(base, status="error: " + repr(e)[:150]))
    pd.DataFrame(rows).to_parquet(fn); return u, len(rows), time.time() - t0

if __name__ == "__main__":
    df = pd.read_csv(C.HERE / "clips.csv"); df = df[df.source == "nurec"]; C.assert_train(df.uuid, nurec=True)
    todo = [(r.uuid, r.chunk) for r in df.itertuples() if (C.SCENES / r.uuid / "DONE").exists() and not (OUT / f"{r.uuid}.parquet").exists()]
    if len(sys.argv) > 2: todo = [x for x in todo if x[0] in set(sys.argv[2].split(","))]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10; print("scenes to grade", len(todo), "procs", n, "full vocab", FULL, flush=True); t0 = time.time(); k = 0
    with mp.Pool(n, maxtasksperchild=4) as p:
        for u, nr, s in p.imap_unordered(scene, todo):
            k += 1; print(f"{k}/{len(todo)} {u[:8]} rows {nr} {s:.0f} s | {(time.time()-t0)/60:.1f} min", flush=True)
