#!/usr/bin/env python
"""A2/A3: convert every S3 clip with the S1 converter's own functions (load_clip, build_maps, decode, rectify, sample; bonnet mask
default ON, N = 37), in parallel and resumable, without the model. Writes pad_feature.gz / pad_target.gz per anchor.
Run: cd paddock-sid/Alpasim && CUDA_VISIBLE_DEVICES="" nice -n 10 alpasim/.venv/bin/python analysis_3.0/Retrain-data/04_training_set/s3_convert.py [procs]"""
import sys, gzip, pickle, time, pathlib, multiprocessing as mp
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s3_common as C
import s1_common as C1; C1.NAS = C.S1ROOT          # S1 module object: only its input root is repointed
import s1_convert as S
import cv2, numpy as np, pandas as pd
assert S.BONNET_MASK_ROWS == 37, S.BONNET_MASK_ROWS
ROWS = C.HERE / "convert_rows"; ROWS.mkdir(exist_ok=True)
def clip(args):
    u, ch = args; fn = ROWS / f"{u}.csv"
    if fn.exists(): return u, 0, 0.0
    import torch; cv2.setNumThreads(1); torch.set_num_threads(1)
    C.assert_train([u]); t0 = time.time(); rows = []
    try:
        d, ts, ego, ft, R_N, t_N = S.load_clip(u, ch); tA = S.anchors_for(ego); near = lambda t: int(np.argmin(np.abs(ts - t)))
        pairs = {t: (near(t - S.STEP_US), near(t)) for t in tA}; need = sorted({j for p in pairs.values() for j in p})
        maps = S.build_maps(ft, R_N); frames = S.decode(d / f"{u}.{C.FRONT}.mp4", need)
        rect = {j: cv2.cvtColor(S.rectify(f, maps), cv2.COLOR_BGR2RGB) for j, f in frames.items()}; frames = None
        for t in tA:
            jo, jn = pairs[t]; feats, label, command, route, lag, cmd_rg = S.sample(ego, t, None, rect[jo], rect[jn])
            tok = f"{u[:8]}_{t // 1000:05d}"; cd = C.CACHE / u / tok; cd.mkdir(parents=True, exist_ok=True)
            with gzip.open(cd / "pad_feature.gz", "wb", compresslevel=1) as f: pickle.dump({k_: torch.tensor(v) for k_, v in feats.items()}, f)
            with gzip.open(cd / "pad_target.gz", "wb", compresslevel=1) as f: pickle.dump({"trajectory": torch.tensor(label), "token": tok}, f)
            rows.append(dict(uuid=u, sample_id=tok, t_us=t, dt_older_us=int(ts[jo] - (t - S.STEP_US)), dt_newer_us=int(ts[jn] - t), pair_gap_us=int(ts[jn] - ts[jo]), n_status_rows=len(feats["ego_status"]),
                             speed=float(np.hypot(feats["ego_status"][-1, 3], feats["ego_status"][-1, 4])), command="LSRU"[int(np.argmax(command))], bonnet_rows=S.BONNET_MASK_ROWS,
                             masked_ok=bool(float(feats["camera_feature_1"][:, -S.BONNET_MASK_ROWS:].max()) == 0.0), mount_offset_m=float(np.linalg.norm(S.t_V - t_N)), status="ok"))
    except Exception as e:
        rows = [dict(uuid=u, sample_id="-", status="error: " + repr(e)[:160])]
    pd.DataFrame(rows).to_csv(fn, index=False); return u, len(rows), time.time() - t0
if __name__ == "__main__":
    df = pd.read_csv(C.HERE / "clips.csv"); C.assert_train(df.uuid)
    todo = [(r.uuid, r.chunk) for r in df.itertuples() if (C.S1ROOT / "clips" / r.uuid / "DONE").exists() and not (ROWS / f"{r.uuid}.csv").exists()]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8; print("clips to convert", len(todo), "procs", n, "mask rows", S.BONNET_MASK_ROWS, flush=True); t0 = time.time(); k = 0
    with mp.Pool(n, maxtasksperchild=20) as p:
        for u, nr, s in p.imap_unordered(clip, todo):
            k += 1
            if k % 50 == 0 or k == len(todo): print(f"{k}/{len(todo)} last {s:.0f} s | {(time.time()-t0)/60:.1f} min", flush=True)
