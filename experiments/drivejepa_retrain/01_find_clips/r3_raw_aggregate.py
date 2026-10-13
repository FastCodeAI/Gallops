"""Run 3 Part 3 tables -> r3_raw_tables.md"""
import numpy as np, pandas as pd, common
V = common.load_split_views(); _, EXAM, _ = common.load_views()
cl = pd.read_csv("r3_raw_clips.csv"); eg = pd.read_csv("r3_raw_moveoffs_ego.csv"); ev = pd.read_csv("r3_raw_moveoffs.csv")
for d in (cl, eg, ev): assert not (set(d.uuid) & EXAM) and set(d.uuid) <= V["raw_train"]
ev["actor_bum"] = ev.actor_bum == True; ev["actor_rig"] = ev.actor_rig == True
out = []; P = lambda s="": out.append(s)
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:g}" if isinstance(v, (float, np.floating)) else str(v))
    P("| " + " | ".join(map(str, df.columns)) + " |"); P("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): P("| " + " | ".join(f(v) for v in r) + " |")
    P()
N = len(cl); NT = len(V["raw_train"])
P(f"Clips scanned: {N} of {NT} raw_train clips ({100*N/NT:.1f} %; the other {NT-N} have no `egomotion.offline`/`obstacle.offline` per `feature_presence.parquet`), {cl.chunk.nunique()} chunks; scan errors {(cl.status != 'ok').sum()}; clip duration median {cl.dur.median():.1f} s."); P()
P(f"Actor attachment: {len(ev)} lateral events, status {ev.actor_status.value_counts().to_dict()}; bumper offset source {ev.F_src.value_counts().to_dict()}; F min/median/max {ev.F.min():.2f}/{ev.F.median():.2f}/{ev.F.max():.2f} m."); P()
turn = ev.dyaw4_deg.abs() > 45; lat = ev.lat4 >= 1.0
rows = [("all move-offs", len(eg), eg.uuid.nunique()), ("lateral ≥ 1.0 m", int((eg.lat4 >= 1.0).sum()), eg[eg.lat4 >= 1.0].uuid.nunique()),
        ("lateral ≥ 1.0 m, heading change ≤ 45°", int((lat & ~turn).sum()), ev[lat & ~turn].uuid.nunique()),
        ("lateral + stationary box 0–8 m ahead of bumper, ±2 m (before turn filter)", int(ev.actor_bum.sum()), ev[ev.actor_bum].uuid.nunique()),
        ("  of which dropped as turns (> 45°)", int((ev.actor_bum & turn).sum()), ev[ev.actor_bum & turn].uuid.nunique()),
        ("**pull-out events (after removing turns)**", int((ev.actor_bum & ~turn).sum()), ev[ev.actor_bum & ~turn].uuid.nunique()),
        ("for reference: same with the rig-origin gap", int((ev.actor_rig & ~turn).sum()), ev[ev.actor_rig & ~turn].uuid.nunique())]
d = pd.DataFrame(rows, columns=["set", "events", "distinct clips"]); d["per 1,000 clips scanned"] = (d.events / N * 1000).round(2); tbl(d)
fin = ev[ev.actor_bum & ~turn]; n = len(fin); rate = n / N * 1000
P(f"Rate {rate:.2f} pull-out events per 1,000 scanned clips. Whole raw_train was scanned (no sampling): actual total {n}; extrapolated to all {NT} raw_train clips incl. those without offline labels: {rate*NT/1000:.0f}."); P()
P(f"Pull-out events: actor classes {fin.acat_bum.value_counts().to_dict()}; bumper gap median {fin.gap_bum.median():.2f} m; lateral median {fin.lat4.median():.2f} m; |heading change| median {fin.dyaw4_deg.abs().median():.1f}°; stop observed from clip start {int(fin.from_clip_start.sum())}; lateral toward left {int((fin.lat4_signed > 0).sum())} / right {int((fin.lat4_signed < 0).sum())}."); P()
dc = pd.read_parquet(common.RAW_SNAP / "metadata/data_collection.parquet"); P(f"Countries of pull-out clips: {dc.loc[fin.uuid.unique()].country.value_counts().head(6).to_dict()}; in nurec_train: {len(set(fin.uuid) & V['nurec_train'])} clips."); P()
# cross-check raw vs clipgt on nurec_train
d3 = pd.read_csv("d3_moveoffs.csv"); d3 = d3[d3.split == "train"]; r = eg[eg.uuid.isin(V["nurec_train"])]
k3 = set(zip(d3.uuid, d3.t_move.round(0))); kr = set(zip(r.uuid, r.t_move.round(0)))
P(f"Cross-check on the {len(V['nurec_train'] & set(cl.uuid))} nurec_train clips also scanned from raw: move-offs from clipgt (usdz) {len(d3)}, from raw egomotion.offline {len(r)}, same (uuid, t_move ± 0.5 s) {len(k3 & kr)}."); P()
P("Spot-check — 10 pull-out events (seed 20261003):"); P()
tbl(fin.sample(min(10, n), random_state=20261003).sort_values("uuid")[["uuid", "chunk", "t_stop_start", "t_move", "t_depart", "lat4_signed", "dyaw4_deg", "acat_bum", "gap_bum", "alat_bum"]].round(2))
P("| id | statement | result | verdict |"); P("|---|---|---|---|"); P(f"| P-RBL | raw_train ≥ 100 pull-out events after removing turns | {n} events in {fin.uuid.nunique()} clips | {'ENOUGH' if n >= 100 else 'THIN'} |")
open("r3_raw_tables.md", "w").write("\n".join(out)); print("\n".join(out))
