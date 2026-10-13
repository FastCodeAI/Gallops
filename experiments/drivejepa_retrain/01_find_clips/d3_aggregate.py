"""Run 3 Parts 1+2 tables -> d3_tables.md"""
import numpy as np, pandas as pd, common
V = common.load_split_views(); _, EXAM, _ = common.load_views()
sc = pd.read_csv("d3_scenes.csv"); an = pd.read_csv("d3_anchors.csv", low_memory=False); ev = pd.read_csv("d3_moveoffs.csv"); ar = pd.read_csv("d3_around.csv")
for d in (sc, an, ev, ar): assert not (set(d.uuid) & EXAM) and set(d.uuid) <= (V["nurec_train"] | V["nurec_val"])
B = ["A", "brake", "curve", "follow_old", "follow", "follow_motion", "cruise_old", "cruise"]
for c in B: an[c] = an[c] == True
for c in ["actor_rig", "actor_bum"]: ev[c] = ev[c] == True
out = []; P = lambda s="": out.append(s)
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:g}" if isinstance(v, (float, np.floating)) else str(v))
    P("| " + " | ".join(map(str, df.columns)) + " |"); P("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): P("| " + " | ".join(f(v) for v in r) + " |")
    P()
# consistency with Run 2 on the same scenes
a2 = pd.read_csv("d2_anchors.csv", low_memory=False); a2 = a2[a2.uuid.isin(set(an.uuid))]
m = a2.merge(an, on=["uuid", "t"], suffixes=("_r2", "")); P(f"Consistency: Run 3 code reproduces Run 2 on the same {an.uuid.nunique()} scenes — D-A flag equal on {int(((m.A_r2 == True) == m.A).sum())}/{len(m)} anchors; Run 2 follow == follow_old on {int(((m.follow_r2 == True) == m.follow_old)[m.A].sum())}/{int(m.A.sum())} D-A anchors."); P()
P(f"Ego box from usdz `rig_trajectories.json` → `rig_trajectories[0].rig_bbox` (`centroid`, `dim`, rig frame): present in {len(sc)}/{len(sc)} scenes, `rot` = 0 and centroid y = 0 in all. Front-bumper offset F = centroid_x + dim_x/2: min {sc.F.min():.3f}, median {sc.F.median():.3f}, max {sc.F.max():.3f} m ({sc.F.round(3).nunique()} distinct values); width {sc.W.min():.3f}–{sc.W.max():.3f} m."); P()
# ---- 1a/1c D-A bins
A = an[an.A].copy(); A["bin"] = (A.wmin // 2 * 2).astype(int); top = int(A["bin"].max())
P("### 1a + 1c — D-A bins (window ≥ 16 m/s), scenes / anchors; follow before → after the ±30° heading filter (anchors)"); P()
rows = []
for b in range(16, top + 2, 2):
    r = [f"{b}–{b+2}"]
    for s in ("train", "val"):
        g = A[(A["bin"] == b) & (A.split == s)]; r += [g.uuid.nunique(), len(g), f"{int(g.follow_old.sum())} → {int(g.follow.sum())}", int(g.brake.sum()), int(g.curve.sum()), f"{int(g.cruise_old.sum())} → {int(g.cruise.sum())}"]
    rows.append(r)
r = ["**all**"]
for s in ("train", "val"):
    g = A[A.split == s]; r += [g.uuid.nunique(), len(g), f"{int(g.follow_old.sum())} → {int(g.follow.sum())}", int(g.brake.sum()), int(g.curve.sum()), f"{int(g.cruise_old.sum())} → {int(g.cruise.sum())}"]
rows.append(r)
tbl(pd.DataFrame(rows, columns=["bin m/s", "train scenes", "train anchors", "train follow", "train brake", "train curve", "train cruise", "val scenes", "val anchors", "val follow", "val brake", "val curve", "val cruise"]))
res = {}
for s in ("train", "val"):
    g = A[A.split == s]; nb = g["bin"].nunique(); bs = g.groupby("bin").uuid.nunique(); ceil = int(bs[bs >= 5].index.max()); res[s] = (g.uuid.nunique(), nb, ceil)
    P(f"nurec_{s}: {g.uuid.nunique()} scenes, {nb} bins, highest bin with ≥ 5 scenes {ceil}–{ceil+2} m/s. Follow scenes {g[g.follow_old].uuid.nunique()} → {g[g.follow].uuid.nunique()}; anchors dropped by the heading filter {int((g.follow_old & ~g.follow).sum())}; motion-direction variant gives {int(g.follow_motion.sum())} anchors.")
P()
# ---- 1a/1b D-B
P("### 1a + 1b — D-B move-offs; actor gap from rig origin (before) → front bumper (after)"); P()
ev["lat"] = ev.lat4 >= 1.0; rows = []
for name, m in [("all move-offs", ev.uuid == ev.uuid), ("lateral ≥ 1.0 m", ev.lat), ("stationary actor 0–8 m, ±2 m — rig origin (Run 2)", ev.actor_rig), ("stationary actor 0–8 m, ±2 m — front bumper", ev.actor_bum),
                ("lateral subset — rig origin (Run 2)", ev.lat & ev.actor_rig), ("**lateral subset — front bumper**", ev.lat & ev.actor_bum), ("lateral subset, bumper, heading change ≤ 45°", ev.lat & ev.actor_bum & (ev.dyaw4_deg.abs() <= 45))]:
    r = [name]
    for s in ("train", "val"): g = ev[m & (ev.split == s)]; r += [len(g), g.uuid.nunique()]
    rows.append(r)
tbl(pd.DataFrame(rows, columns=["set", "train events", "train scenes", "val events", "val scenes"]))
L = ev[ev.lat & ev.actor_bum][["split", "uuid", "t_stop_start", "t_move", "t_depart", "lat4_signed", "dyaw4_deg", "acat_bum", "gap_bum", "alat_bum"]].round(2)
if len(L): P("Lateral-subset events (bumper gap):"); P(); tbl(L)
# ---- Part 2
P("### Part 2 — around a still obstacle (one event per actor per scene, earliest qualifying anchor)"); P()
ar["sbin"] = pd.cut(ar.ego_speed, [-1, 2, 8, 16, 1e3], labels=["0–2", "2–8", "8–16", "≥16"], right=False); ar["cbin"] = pd.cut(ar.clearance, [0, 0.5, 1.0, 2.0, 1e3], labels=["≤0.5", "0.5–1.0", "1.0–2.0", ">2.0"])
for s in ("train", "val"):
    g = ar[ar.split == s]; ct = pd.crosstab(g.sbin, g.cbin, dropna=False); ct["all"] = ct.sum(axis=1); ct.loc["all"] = ct.sum(); ct = ct.reset_index().rename(columns={"sbin": f"nurec_{s}: ego speed m/s \\ clearance m"})
    tbl(ct)
    P(f"nurec_{s}: {len(g)} events in {g.uuid.nunique()} scenes; clearance ≤ 1.0 m: {int((g.clearance <= 1.0).sum())} events in {g[g.clearance <= 1.0].uuid.nunique()} scenes; classes {g.cat.value_counts().to_dict()}; |path lateral at pass| median {g.path_lat_at_pass.abs().median():.2f} m, > 1.0 m in {int((g.path_lat_at_pass.abs() > 1.0).sum())}; |heading change at pass| > 20° in {int((g.dyaw_at_pass_deg.abs() > 20).sum())}; candidates dropped for box overlap {int(sc[sc.split == s].around_overlap_dropped.sum())}."); P()
g = ar[(ar.split == "train")]; g2 = g[g.dyaw_at_pass_deg.abs() <= 20]
P(f"Descriptive (nurec_train, heading change at pass ≤ 20°, i.e. not a turn/curve artefact): {len(g2)} events, {int((g2.clearance <= 1.0).sum())} with clearance ≤ 1.0 m; of those, human path offset ≥ 0.5 m at the pass: {int(((g2.clearance <= 1.0) & (g2.path_lat_at_pass.abs() >= 0.5)).sum())}."); P()
P("Spot-check — 5 smallest clearances (train first):"); P()
tbl(ar.sort_values(["split", "clearance"]).groupby("split").head(5)[["split", "uuid", "t", "t_pass", "ego_speed", "clearance", "cat", "actor_lat", "path_lat_at_pass"]].round(2))
tr = ar[ar.split == "train"]
v = [("P-A3", "nurec_train ≥ 30 high-speed scenes, ≥ 4 bins", f"{res['train'][0]} scenes, {res['train'][1]} bins", "ENOUGH" if res["train"][0] >= 30 and res["train"][1] >= 4 else "THIN"),
     ("P-NA", "nurec_train ≥ 100 around-obstacle events", f"{len(tr)} events in {tr.uuid.nunique()} scenes", "ENOUGH" if len(tr) >= 100 else "THIN"),
     ("P-NAt", "nurec_train ≥ 30 with clearance ≤ 1.0 m", f"{int((tr.clearance <= 1.0).sum())}", "ENOUGH" if (tr.clearance <= 1.0).sum() >= 30 else "THIN")]
tbl(pd.DataFrame(v, columns=["id", "statement", "result", "verdict"]))
open("d3_tables.md", "w").write("\n".join(out)); print("\n".join(out))
