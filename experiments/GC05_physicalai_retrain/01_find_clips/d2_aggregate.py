"""Run 2 aggregation: D-0, D-A, D-B, D-D, D-E tables -> d2_tables.md (pasted into REPORT.md)."""
import sys, numpy as np, pandas as pd, common
sfx = sys.argv[1] if len(sys.argv) > 1 else ""
nurec, raw, pool = common.load_views()
sc = pd.read_csv(common.HERE / f"d2_scenes{sfx}.csv"); an = pd.read_csv(common.HERE / f"d2_anchors{sfx}.csv", low_memory=False); ev = pd.read_csv(common.HERE / f"d2_moveoffs{sfx}.csv")
assert not (set(sc.uuid) & raw) and set(sc.uuid) <= pool
rt = pd.read_csv(common.NAS / "root_files" / "clip_ratings_26.04.csv").set_index("clip_id")
sc["rating"] = sc.uuid.map(rt.ego_speed); an["rating"] = an.uuid.map(rt.ego_speed)
for c in ["A", "brake", "follow", "curve", "cruise"]: an[c] = an[c] == True
ev["actor"] = ev["actor"] == True
out = []; P = lambda s="": out.append(s)
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:g}" if isinstance(v, float) else str(v))
    P("| " + " | ".join(map(str, df.columns)) + " |"); P("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): P("| " + " | ".join(f(v) for v in r) + " |")
    P()
err = int(sc["error"].notna().sum()) if "error" in sc else 0
P(f"Scenes analysed: {len(sc)} of {len(pool)} pool scenes (load errors {err}); anchors {len(an)}; clip duration min/median/max {sc.dur.min():.1f}/{sc.dur.median():.1f}/{sc.dur.max():.1f} s; scenes with < 32 anchors: {(sc.n_anchors < 32).sum()}."); P()
# ---- D-0
P("### D-0 — speed envelope"); P()
an["bin"] = (an.wmin // 2 * 2).astype(int)
h = an.groupby("bin").agg(anchors=("uuid", "size"), scenes=("uuid", "nunique")).reindex(range(0, 50, 2), fill_value=0).reset_index()
top = int(h[h.anchors > 0]["bin"].max()); h = h[h["bin"] <= top]
h.insert(0, "window-min speed m/s", [f"{b}–{b+2}" for b in h["bin"]]); tbl(h.drop(columns="bin"))
P(f"Top bin reached: {top}–{top+2} m/s. Samples > 50 m/s: scenes with vmax > 50 = {(sc.vmax > 50).sum()}; anchors with window-min ≥ 50 = {(an.wmin >= 50).sum()}."); P()
q = sc.vheld.quantile([.5, .9, .99]).round(1).tolist(); P(f"Per scene: max speed median {sc.vmax.median():.1f}, max {sc.vmax.max():.1f} m/s; max speed held 4.5 s median/p90/p99 {q[0]}/{q[1]}/{q[2]}, max {sc.vheld.max():.1f} m/s."); P()
f10 = sc.sort_values("vheld", ascending=False).head(10)[["uuid", "vheld", "vmax", "rating"]].round(2); f10.columns = ["uuid", "max held 4.5 s m/s", "max m/s", "clip_ratings ego_speed"]; tbl(f10)
# ---- D-A
P("### D-A — high speed (whole 4.5 s window ≥ 16 m/s, no cap)"); P()
A = an[an.A].copy(); As = sc[sc.A_anchors > 0]
P(f"Distinct scenes {A.uuid.nunique()}; anchors {len(A)}; anchors per qualifying scene median {As.A_anchors.median():.0f} (min {As.A_anchors.min()}, max {As.A_anchors.max()}); scenes qualifying on every anchor {int((As.A_anchors == As.n_anchors).sum())}; per-scene window min/max speed range {As.A_min.min():.1f}–{As.A_max.max():.1f} m/s (per-scene values in d2_scenes.csv)."); P()
b = A.groupby("bin").agg(scenes=("uuid", "nunique"), anchors=("uuid", "size"), brake=("brake", "sum"), follow=("follow", "sum"), curve=("curve", "sum"), cruise=("cruise", "sum")).reindex(range(16, top + 2, 2), fill_value=0).reset_index()
for t in ["brake", "follow", "curve", "cruise"]:
    b[t + "_scenes"] = b["bin"].map(A[A[t]].groupby("bin").uuid.nunique()).fillna(0).astype(int)
b.insert(0, "bin m/s", [f"{x}–{x+2}" for x in b["bin"]]); bb = b.drop(columns="bin"); bb.columns = ["bin m/s", "scenes", "anchors", "brake a", "follow a", "curve a", "cruise a", "brake sc", "follow sc", "curve sc", "cruise sc"]; tbl(bb)
tot = {t: (int(A[t].sum()), A[A[t]].uuid.nunique()) for t in ["brake", "follow", "curve", "cruise"]}
P("Tags, pool total (anchors / scenes): " + "; ".join(f"{k} {v[0]} / {v[1]}" for k, v in tot.items()) + f". Multi-tag anchors {(A[['brake','follow','curve']].sum(axis=1) > 1).sum()}."); P()
nb = int((b.scenes > 0).sum()); ceil = b[b.scenes >= 5]["bin"].max()
P(f"Bins with ≥ 1 scene: {nb}. Highest bin with ≥ 5 distinct scenes: {ceil}–{ceil+2} m/s."); P()
ct = pd.crosstab(sc.rating, sc.A_anchors > 0); ct.columns = ["no D-A anchor", "has D-A anchor"]; ct["share with D-A"] = (ct["has D-A anchor"] / ct.sum(axis=1)).round(2); tbl(ct.reset_index())
P(f"Share of D-A scenes rated 'high': {(As.rating == 'high').mean():.2f}."); P()
# ---- D-B
P("### D-B — move-off from standstill"); P()
ev["lateral"] = (ev.lat4_from_move >= 1.0) & ev.actor; ev["lat_only"] = ev.lat4_from_move >= 1.0; ev["lat_dep"] = ev.lat4_from_depart >= 1.0
rows = [("all move-offs", len(ev), ev.uuid.nunique()),
        ("lateral ≥ 1.0 m (4 s from move onset), no actor condition", int(ev.lat_only.sum()), ev[ev.lat_only].uuid.nunique()),
        ("stationary actor 0–8 m ahead, ±2 m", int(ev.actor.sum()), ev[ev.actor].uuid.nunique()),
        ("**lateral subset** (lateral ≥ 1.0 m AND actor)", int(ev.lateral.sum()), ev[ev.lateral].uuid.nunique()),
        ("alt: lateral ≥ 1.0 m measured 4 s from t_depart, no actor condition", int(ev.lat_dep.sum()), ev[ev.lat_dep].uuid.nunique()),
        ("alt: lateral from t_depart AND actor", int((ev.lat_dep & ev.actor).sum()), ev[ev.lat_dep & ev.actor].uuid.nunique())]
tbl(pd.DataFrame(rows, columns=["set", "events", "distinct scenes"]))
P(f"Standstill observed from clip start (stop onset censored): {int(ev.stop_from_clip_start.sum())} of {len(ev)}. Stop-to-depart delay median {(ev.t_depart - ev.t_move).median():.1f} s. Lateral (from move onset) median {ev.lat4_from_move.median():.2f} m, p90 {ev.lat4_from_move.quantile(.9):.2f} m. 4 s window truncated by clip end: {int((ev.span_from_move < 3.95).sum())}. Actor classes: {ev[ev.actor].actor_cat.value_counts().to_dict()}. Rating of move-off scenes: {sc[sc.moveoffs > 0].rating.value_counts().to_dict()}."); P()
L = ev[ev.lateral].sort_values("lat4_from_move", ascending=False)[["uuid", "t_stop_start", "t_move", "t_depart", "lat4_signed", "actor_cat", "actor_gap", "actor_lat"]].round(2)
if len(L): P("Lateral-subset events:"); P(); tbl(L)
# ---- D-D
P("### D-D — overlap"); P()
sa, sb, sbl = set(As.uuid), set(ev.uuid), set(ev[ev.lateral].uuid)
tbl(pd.DataFrame([("A (D-A anchors)", len(sa)), ("B (any move-off)", len(sb)), ("B lateral subset", len(sbl)), ("A ∩ B", len(sa & sb)), ("neither A nor B (ordinary candidates)", len(set(sc.uuid) - sa - sb))], columns=["set", "scenes"]))
# ---- D-E
P("### D-E — spot-check list (front mp4 at `sample_set/26.04_release/<uuid>/camera_front_wide_120fov.mp4`, t = seconds from clip start)"); P()
rng = np.random.default_rng(20261002); rows = []
for x in range(16, top + 2, 2):
    g = A[A["bin"] == x]
    if not len(g): continue
    us = rng.permutation(sorted(g.uuid.unique()))[:3]
    for u in us:
        r = g[g.uuid == u].iloc[len(g[g.uuid == u]) // 2]; rows.append((f"{x}–{x+2}", u, r.t, round(r.wmin, 1), "+".join(t for t in ["brake", "follow", "curve", "cruise"] if r[t])))
tbl(pd.DataFrame(rows, columns=["D-A bin", "uuid", "t s", "window-min m/s", "tags"]))
E = ev[ev.lateral].sort_values("lat4_from_move", ascending=False).head(5); note = "lateral subset"
if len(E) < 5:
    E = pd.concat([E, ev[~ev.lateral].sort_values("lat4_from_move", ascending=False).head(5 - len(E))]); note = "lateral subset first, then filled with the largest-lateral move-offs without a qualifying actor"
P(f"D-B ({note}):"); P(); tbl(E[["uuid", "t_stop_start", "t_move", "t_depart", "lat4_signed", "actor", "actor_cat", "actor_gap"]].round(2))
# ---- verdicts
P("### Pre-registered"); P()
v = [("P-A", "≥ 30 distinct scenes with D-A anchors, spanning ≥ 4 bins", f"{len(sa)} scenes, {nb} bins", "ENOUGH" if len(sa) >= 30 and nb >= 4 else "THIN"),
     ("P-A2", "highest 2 m/s bin with ≥ 5 scenes (descriptive)", f"{ceil}–{ceil+2} m/s", "—"),
     ("P-B", "≥ 30 move-off events", f"{len(ev)} events in {len(sb)} scenes", "ENOUGH" if len(ev) >= 30 else "THIN"),
     ("P-BL", "≥ 15 lateral-subset events", f"{int(ev.lateral.sum())} events in {len(sbl)} scenes", "ENOUGH" if ev.lateral.sum() >= 15 else "THIN")]
tbl(pd.DataFrame(v, columns=["id", "statement", "result", "verdict"]))
open(common.HERE / f"d2_tables{sfx}.md", "w").write("\n".join(out)); print("\n".join(out))
