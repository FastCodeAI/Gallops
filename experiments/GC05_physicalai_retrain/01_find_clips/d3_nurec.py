"""Run 3 Parts 1+2 on nurec_train / nurec_val from the Run 2 NAS cache (+ rig_bbox.json). -> d3_anchors.csv, d3_moveoffs.csv, d3_around.csv, d3_scenes.csv"""
import json, multiprocessing as mp, numpy as np, pandas as pd, pyarrow.parquet as pq, common, r3_core as K
V = common.load_split_views(); _, EXAM, _ = common.load_views()
def load(u):
    d = common.NAS / "pool" / u
    t = pq.read_table(d / "clipgt__egomotion_estimate.parquet").flatten().flatten(); g = lambda n: t.column(n).to_numpy(zero_copy_only=False)
    ts = g("key.timestamp_micros").astype(np.int64)
    S = K.build_ego(ts / 1e6, np.c_[g("egomotion_estimate.location.x"), g("egomotion_estimate.location.y")], K.yaw_q(*[g(f"egomotion_estimate.orientation.{k}") for k in "xyzw"]))
    o = pq.read_table(d / "clipgt__obstacle.parquet").flatten().flatten(); h = lambda n: o.column(n).to_numpy(zero_copy_only=False)
    ob = pd.DataFrame({"tid": o.column("obstacle.trackline_id").to_pylist(), "ts": h("key.timestamp_micros").astype(np.int64) / 1e6 - S["t0"], "x": h("obstacle.center.x"), "y": h("obstacle.center.y"),
                       "yaw": K.yaw_q(*[h(f"obstacle.orientation.{k}") for k in "xyzw"]), "L": h("obstacle.size.x"), "W": h("obstacle.size.y"), "cat": o.column("obstacle.category").to_pylist()})
    return K.build_tracks(S, ob)
def scene(u):
    assert u not in EXAM
    bb = json.load(open(common.NAS / "pool" / u / "rig_bbox.json")); B = dict(C=bb["centroid"][0], L=bb["dim"][0], W=bb["dim"][1]); B["F"] = B["C"] + B["L"]/2
    S = load(u); an = K.anchors(S, u); ev = [K.attach_actor(S, e, B["F"]) for e in K.moveoffs(S, u)]; ar, nover = K.around(S, u, B)
    for e in ev: [e.pop(k) for k in ("_i", "_j", "_k")]
    return dict(uuid=u, F=B["F"], C=B["C"], L=B["L"], W=B["W"], cy=bb["centroid"][1], rot=str(bb["rot"]), n_tracks=len(S["tracks"]), around_overlap_dropped=nover), an, ev, ar
if __name__ == "__main__":
    todo = sorted(V["nurec_train"] | V["nurec_val"])
    with mp.Pool(10) as p: res = p.map(scene, todo, chunksize=4)
    sp = {u: "train" for u in V["nurec_train"]}; sp.update({u: "val" for u in V["nurec_val"]})
    for name, k in [("scenes", 0), ("anchors", 1), ("moveoffs", 2), ("around", 3)]:
        df = pd.DataFrame([r[0] for r in res]) if k == 0 else pd.DataFrame([x for r in res for x in r[k]])
        df["split"] = df.uuid.map(sp); df.to_csv(common.HERE / f"d3_{name}.csv", index=False); print(name, len(df))
