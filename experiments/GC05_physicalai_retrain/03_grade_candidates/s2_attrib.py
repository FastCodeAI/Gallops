"""Attribution of human DDC/DAC failures: is the ego centre in a gap between lane polygons, in an off-route same-direction lane, or in an opposite-direction lane?"""
import math, numpy as np, pandas as pd
from shapely.geometry import Point
import s2_common as C, s2_scene as SC
K = SC.K; rows = []
g = pd.read_csv(C.HERE / "pilot_grades.csv", low_memory=False); h = g[(g.plan == "human") & (g.DDC < 1)][["uuid", "t"]].assign(src="pilot"); al = pd.read_csv(C.HERE / "align.csv").set_index("uuid")
b = pd.read_csv(C.HERE / "btest.csv"); hb = b[(b.kind == "tight") & (b.status == "ok") & (b.human_DDC < 1)][["uuid", "t_anchor", "clearance"]].rename(columns={"t_anchor": "t"}).assign(src="btest")
for src, u, t, cl in [(r.src, r.uuid, r.t, np.nan) for r in h.itertuples()] + [(r.src, r.uuid, r.t, r.clearance) for r in hb.itertuples()]:
    S = SC.load_scene(u) if not rows or rows[-1]["uuid"] != u else S; SC.route_lanes(S)
    i = int(round((t + (float(al.loc[u, "dt_s"]) if src == "pilot" else 0)) / K.DT)); c = dict(gap=0, offroute_same_dir=0, opposite=0, onroute=0)
    for j in range(i, i + 41):
        ctr = S["xy"][j] + 1.461 * np.array([math.cos(S["yaw"][j]), math.sin(S["yaw"][j])]); P = Point(ctr)
        inl = [lid for lid in (S["lane_ids"][q] for q in S["lane_tree"].query(P)) if S["lanes"][lid]["poly"].contains(P)]
        if any(l in S["route_ids"] for l in inl): c["onroute"] += 1
        elif not inl: c["gap"] += 1
        elif any(abs(K.wrap(SC.lane_heading_at(S["lanes"][l], ctr)[0] - S["yaw"][j])) < math.radians(45) for l in inl): c["offroute_same_dir"] += 1
        else: c["opposite"] += 1
    rows.append(dict(src=src, uuid=u, t=t, clearance=cl, **c))
d = pd.DataFrame(rows); d.to_csv(C.HERE / "ddc_attribution.csv", index=False)
for src, x in d.groupby("src"):
    dom = x[["gap", "offroute_same_dir", "opposite"]].idxmax(axis=1).value_counts().to_dict()
    print(src, "human DDC<1 anchors", len(x), "| dominant off-route cause:", dom, "| steps:", x[["gap", "offroute_same_dir", "opposite", "onroute"]].sum().to_dict(), "| clearance of btest events:", sorted(x.clearance.dropna().round(2).tolist()))
