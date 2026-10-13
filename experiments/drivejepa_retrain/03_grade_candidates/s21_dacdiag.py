"""Why does 'stay still' / the human still fail the drivable-area gate after A1? At the anchor pose (t), where is the scorer's ego box?"""
import math, numpy as np, pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import s2_common as C, s2_scene as SC
K = SC.K; b = pd.read_csv(C.HERE / "btest.csv"); g = pd.read_csv(C.HERE / "pilot_grades.csv", low_memory=False); al = pd.read_csv(C.HERE / "align.csv").set_index("uuid")
jobs = [("btest", r.uuid, r.t_anchor, r.stay_DAC, r.human_DAC) for r in b[b.status == "ok"].itertuples()]
h = g[g.plan == "human"].set_index("token"); jobs += [("pilot", r.uuid, r.t + float(al.loc[r.uuid, "dt_s"]), r.DAC, h.loc[r.token, "DAC"]) for r in g[g.plan == "stay"].itertuples()]
rows = []; cur = None
for src, u, t, sd, hd in sorted(jobs, key=lambda x: x[1]):
    if cur != u: S = SC.load_scene(u, fix=True); SC.route_lanes(S); D = unary_union(S["drivable"]); cur = u
    i = int(round(t / K.DT)); ctr = S["xy"][i] + 1.461 * np.array([math.cos(S["yaw"][i]), math.sin(S["yaw"][i])])
    box = K.rect(ctr, S["yaw"][i], 5.176, 2.297); real = K.rect(ctr, S["yaw"][i], 5.176, 2.0)
    exc = max(D.distance(Point(p)) for p in box); excr = max(D.distance(Point(p)) for p in real)
    rows.append(dict(src=src, uuid=u, t=t, stay_DAC=sd, human_DAC=hd, centre_outside=not D.contains(Point(ctr)), centre_dist=D.distance(Point(ctr)), corner_exc_m=exc, corner_exc_2m_wide=excr, speed=float(S["speed"][i])))
d = pd.DataFrame(rows); d.to_csv(C.HERE / "s21_dac_diag.csv", index=False)
for src, x in d.groupby("src"):
    f = x[x.stay_DAC < 1]; print(src, "anchors", len(x), "stay DAC fail", len(f), "| box already outside at the anchor", int((f.corner_exc_m > 0).sum()), "| centre outside", int(f.centre_outside.sum()), "| corner excursion at anchor m: quantiles", f.corner_exc_m.quantile([.25, .5, .75, 1]).round(2).tolist(), "| would be inside with a 2.0 m wide box", int(((f.corner_exc_m > 0) & (f.corner_exc_2m_wide == 0)).sum()), "| scenes", f.uuid.nunique())
    hf = x[(x.human_DAC < 1)]; print("   human DAC fail", len(hf), "of which box outside at the anchor", int((hf.corner_exc_m > 0).sum()))
