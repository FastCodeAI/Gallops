import sys, time, numpy as np, pandas as pd, s2_common as C, s2_scene as SC
sc = pd.read_csv(C.HERE / "scenes.csv"); C.assert_nurec_train(sc.uuid); rows = []; br = []
for r in sc.itertuples():
    S = SC.load_scene(r.uuid, fix=True); SC.route_lanes(S); a1 = sum(p.area for p in S["drivable"])
    rows.append(dict(uuid=r.uuid, pilot=r.pilot, area_s2=round(S["drivable_s2_area"]), area_s21=round(a1), added_m2=round(a1 - S["drivable_s2_area"]), **S["strip_stats"], bridges=len(S["bridges"]), bridges_on_centre=sum(b["on_centre_line"] for b in S["bridges"])))
    br += [dict(uuid=r.uuid, width=b["width"], length=b["length"], area=b["area"], on_centre_line=b["on_centre_line"]) for b in S["bridges"]]
d = pd.DataFrame(rows); d.to_csv(C.HERE / "s21_map_fix.csv", index=False); b = pd.DataFrame(br); b.to_csv(C.HERE / "s21_bridges.csv", index=False)
print(d.drop(columns=["uuid"]).describe().round(1).loc[["min", "50%", "max", "mean"]].to_string()); print("bridges", len(b), "in", b.uuid.nunique() if len(b) else 0, "scenes; width quantiles", b.width.quantile([0, .25, .5, .75, 1]).round(3).tolist() if len(b) else None, "on centre line", int(b.on_centre_line.sum()) if len(b) else 0)
