#!/usr/bin/env python
"""Label-1 futures for every anchor of every S3 clip, from egomotion only (no images): the same Ego class, anchor rule and
P._relative_pose call as s1_convert.sample (s1_convert.py: Ego, anchors_for, `label = ...`), so they equal the pad_target labels.
Run: cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/04_training_set/s3_labels.py"""
import sys, io, zipfile, pathlib, multiprocessing as mp, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s3_common as C
import s1_convert as S
P, K = S.P, S.K
def clip(args):
    u, ch = args; z = zipfile.ZipFile(C.S0C.RAW_NAS / f"labels/egomotion.offline/egomotion.offline.chunk_{ch:04d}.zip")
    ego = S.Ego(pd.read_parquet(io.BytesIO(z.read(f"{u}.egomotion.offline.parquet")))); out = []
    for t in S.anchors_for(ego):
        Tn = ego.T(t); lab = np.array([P._relative_pose(ego.T(t + k * S.STEP_US), Tn) for k in range(1, K.NUM_POSES + 1)], dtype=np.float64)
        vx, vy, ax, ay, _ = ego.dyn(t); out.append((f"{u[:8]}_{t // 1000:05d}", u, t, float(np.hypot(vx, vy)), lab))
    return out
if __name__ == "__main__":
    df = pd.read_csv(C.HERE / "clips.csv"); C.assert_train(df.uuid)
    with mp.Pool(8) as p: res = [x for r in p.imap_unordered(clip, list(zip(df.uuid, df.chunk)), chunksize=16) for x in r]
    res.sort(key=lambda x: (x[1], x[2])); m = pd.DataFrame([(a, b, c, d) for a, b, c, d, _ in res], columns=["sample_id", "uuid", "t_us", "speed"])
    assert m.sample_id.is_unique, "8-hex uuid prefix collision"; m.to_parquet(C.HERE / "labels_index.parquet"); np.save(C.HERE / "labels.npy", np.stack([x[4] for x in res]))
    print("clips", m.uuid.nunique(), "anchors", len(m), "speed >= 14:", int((m.speed >= 14).sum()), "in clips", m[m.speed >= 14].uuid.nunique())
