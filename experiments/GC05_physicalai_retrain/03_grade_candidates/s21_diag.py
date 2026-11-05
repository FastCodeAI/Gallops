import math, numpy as np, pandas as pd
from shapely.geometry import Point
import s2_common as C, s2_scene as SC
K = SC.K; g = pd.read_csv(C.HERE / "pilot_grades.csv", low_memory=False); al = pd.read_csv(C.HERE / "align.csv").set_index("uuid")
h = g[(g.plan == "human") & (g.DDC < 1)]
for u, x in h.groupby("uuid"):
    S = SC.load_scene(u, fix=True); SC.route_lanes(S); rows = []
    for t in x.t:
        i = int(round((t + float(al.loc[u, "dt_s"])) / K.DT))
        for j in range(i, i + 41):
            ctr = S["xy"][j] + 1.461 * np.array([math.cos(S["yaw"][j]), math.sin(S["yaw"][j])]); P = Point(ctr)
            inl = [lid for lid in (S["lane_ids"][q] for q in S["lane_tree"].query(P)) if S["lanes"][lid]["poly"].contains(P)]
            if any(l in S["route_ids"] for l in inl) or any(b["poly"].contains(P) for b in S["bridges"]): continue
            near = min((S["lanes"][l]["poly"].distance(P), l) for l in S["route_ids"] if l in S["lanes"])
            rows.append((round(S["t"][j], 1), "nolane" if not inl else "offroute:" + ",".join(inl), round(near[0], 2), [S["lanes"][l]["direction"] for l in inl], [round(math.degrees(K.wrap(SC.lane_heading_at(S["lanes"][l], ctr)[0] - S["yaw"][j]))) for l in inl]))
    d = pd.DataFrame(sorted(set(map(str, rows)))); print(u[:8], "DDC-fail anchors", len(x), "distinct off-route steps", len(d)); print(d.head(12).to_string(header=False))
