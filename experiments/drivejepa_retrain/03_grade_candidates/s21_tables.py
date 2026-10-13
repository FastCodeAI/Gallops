"""S2.1 tables -> s21_tables.md (conda drive-jepa python)."""
import json, numpy as np, pandas as pd, s2_common as C
HERE = C.HERE; out = []; W = lambda s="": (out.append(s), print(s))
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    W("| " + " | ".join(map(str, df.columns)) + " |"); W("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): W("| " + " | ".join(f(v) for v in r) + " |")
    W()
T = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]; pct = lambda x: f"{100*x:.1f} %"
# ---------------- A
m = pd.read_csv(HERE / "s21_map_fix.csv"); b = pd.read_csv(HERE / "s21_bridges.csv"); C.assert_nurec_train(m.uuid)
W("### Part A — map fixes (40 scenes)"); W()
tbl(pd.DataFrame([dict(quantity="drivable area, m² per scene (median)", S2=m.area_s2.median(), S2_1=m.area_s21.median()), dict(quantity="drivable area, m² (sum over 40 scenes)", S2=m.area_s2.sum(), S2_1=m.area_s21.sum()),
                  dict(quantity="road boundaries read (sum)", S2=0, S2_1=m.boundaries.sum()), dict(quantity="boundary–lane pairs from `ROAD_BOUNDARY_TO_LANE` (sum)", S2=0, S2_1=m.pairs.sum()),
                  dict(quantity="strips added (sum)", S2=0, S2_1=m.strips.sum()), dict(quantity="pairs skipped: more than 8 m apart / degenerate", S2=0, S2_1=f"{m.skipped_far.sum()} / {m.skipped_degenerate.sum()}"),
                  dict(quantity="on-route gaps bridged (scenes)", S2=0, S2_1=f"{len(b)} ({b.uuid.nunique()})")]))
W(f"Area added by the boundary strips: +{100*(m.area_s21.sum()/m.area_s2.sum()-1):.1f} % overall (per scene {m.added_m2.min():.0f}–{m.added_m2.max():.0f} m², median {m.added_m2.median():.0f}).")
bw = pd.cut(b.width, [0, 0.05, 0.1, 0.25, 0.5, 1.0], labels=["≤0.05", "0.05–0.1", "0.1–0.25", "0.25–0.5", "0.5–1.0"]).value_counts().sort_index()
W(f"Bridged gaps, mean width (m): " + "; ".join(f"{k} m: {v}" for k, v in bw.items()) + f". Median width {b.width.median():.2f} m, max {b.width.max():.2f} m; {int(b.on_centre_line.sum())} of {len(b)} lie under the ego-centre path. Most are long hairline seams between neighbouring lanes (median length {b.length.median():.0f} m), not end-to-end gaps."); W()
W("Before/after PNGs: `s21_png/parking_strip_3259c4fe.png`, `s21_png/highspeed_gap_ad134e5b.png`, `s21_png/tight_pass_0771fd01.png`."); W()
# ---------------- B
def load(tag):
    g = pd.read_csv(HERE / f"pilot_grades{tag}.csv", low_memory=False); g = g[g.status == "ok"]; bt = pd.read_csv(HERE / f"btest{tag}.csv"); return g, bt
res = {}
for name, tag in (("S2", "_s2"), ("S2.1", "")):
    g, bt = load(tag); h = g[g.plan == "human"]; st = g[g.plan == "stay"]; hao = h[h.type.isin(["A", "O"])]; p = g[g.plan.str.startswith("p")]; t = bt[(bt.kind == "tight") & (bt.status == "ok")]
    sdA = p[p.type == "A"].groupby("token").score.std(ddof=0)
    res[name] = dict(anchors=h.token.nunique(), stay_dac_pilot=(st.DAC < 1).mean(), stay_dac_btest=(t.stay_DAC < 1).mean(), stay_dac_all=(pd.concat([st.DAC, t.stay_DAC]) < 1).mean(), n_all=len(st) + len(t),
                     p2=(hao.score >= 0.8).mean(), p2A=(h[h.type == "A"].score >= 0.8).mean(), p2O=(h[h.type == "O"].score >= 0.8).mean(), p2T=(h[h.type == "T"].score >= 0.8).mean(), p3=(sdA > 0.05).mean(), p3med=sdA.median(),
                     n_t=len(t), p4a=(t.human_score >= 0.8).mean(), p4b=(t.human_score > t.stay_score).mean(), **{f"h_{k}": t[f"human_{k}"].mean() for k in T}, **{f"hp_{k}": h[k].mean() for k in T},
                     chosen=g[g.chosen == True].score.mean(), best=p.groupby("token").score.max().mean())
r2, r21 = res["S2"], res["S2.1"]
W("### Part B — re-grade, same anchors and plans"); W()
rows = [("pilot anchors graded", r2["anchors"], r21["anchors"]), ("Q1 stay-still fails drivable area: pilot anchors", pct(r2["stay_dac_pilot"]), pct(r21["stay_dac_pilot"])), ("Q1 … tight-pass anchors", pct(r2["stay_dac_btest"]), pct(r21["stay_dac_btest"])),
        ("Q1 … all graded anchors", pct(r2["stay_dac_all"]), pct(r21["stay_dac_all"]) + f" of {r21['n_all']}"),
        ("Q2 / P2 human ≥ 0.8, high-speed + ordinary", pct(r2["p2"]), pct(r21["p2"])), ("  high-speed", pct(r2["p2A"]), pct(r21["p2A"])), ("  ordinary", pct(r2["p2O"]), pct(r21["p2O"])), ("  tight-pass scenes (all anchors)", pct(r2["p2T"]), pct(r21["p2T"])),
        ("P3 (informational) std of 32 grades > 0.05, high-speed", pct(r2["p3"]) + f" (median std {r2['p3med']:.3f})", pct(r21["p3"]) + f" (median std {r21['p3med']:.3f})"),
        ("Q3 / P4 tight passes graded", r2["n_t"], r21["n_t"]), ("Q3 human ≥ 0.8", pct(r2["p4a"]), pct(r21["p4a"])), ("Q3 human > stay", pct(r2["p4b"]), pct(r21["p4b"])),
        ("tight-pass human path, mean per term (NC / DAC / DDC / TTC / LK / HC)", " / ".join(f"{r2['h_'+k]:.2f}" for k in ["NC", "DAC", "DDC", "TTC", "LK", "HC"]), " / ".join(f"{r21['h_'+k]:.2f}" for k in ["NC", "DAC", "DDC", "TTC", "LK", "HC"])),
        ("pilot human path, mean per term (NC / DAC / DDC / TTC / LK / HC)", " / ".join(f"{r2['hp_'+k]:.2f}" for k in ["NC", "DAC", "DDC", "TTC", "LK", "HC"]), " / ".join(f"{r21['hp_'+k]:.2f}" for k in ["NC", "DAC", "DDC", "TTC", "LK", "HC"])),
        ("model's chosen plan, mean grade / best of its 32", f"{r2['chosen']:.3f} / {r2['best']:.3f}", f"{r21['chosen']:.3f} / {r21['best']:.3f}")]
tbl(pd.DataFrame(rows, columns=["quantity", "S2", "S2.1"]))
g, bt = load(""); t = bt[(bt.kind == "tight") & (bt.status == "ok")].copy(); d = pd.read_csv(HERE / "s21_dac_diag.csv")
f = d[d.stay_DAC < 1]; W(f"Why stay-still still fails the drivable-area gate ({len(f)} anchors, `s21_dac_diag.csv`): the box centre is inside the drivable area on all {int((~f.centre_outside).sum())}; on {int((f.corner_exc_m > 0).sum())} a corner of the scorer's 2.30 m-wide box is already over the mapped road edge at the anchor (median overhang {f[f.corner_exc_m > 0].corner_exc_m.median():.2f} m, max {f.corner_exc_m.max():.2f} m; {int(((f.corner_exc_m > 0) & (f.corner_exc_2m_wide == 0)).sum())} would fit with a 2.0 m box), on the other {int((f.corner_exc_m == 0).sum())} a corner crosses it while the plan brakes to a stop. They sit in {f.uuid.nunique()} scenes, all narrow streets."); W()
low = t[t.human_score < 0.8]; W(f"Tight passes below 0.8 in S2.1 ({len(low)}): terms below 1 — " + ", ".join(f"{k} {int((low[f'human_{k}'] < 1).sum())}" for k in T[:-1]) + "."); W()
# ---- B2
W("### B2 — driving-direction gate, measured (diagnostic only; not written to any cache)"); W()
rows = []
for name, hs, ss in (("standard formula", "human_score", "stay_score"), ("direction gate disabled", "human_score_no_ddc", "stay_score_no_ddc")):
    rows.append(dict(formula=name, events=len(t), human_mean=t[hs].mean(), stay_mean=t[ss].mean(), human_ge_0_8=pct((t[hs] >= 0.8).mean()), human_gt_stay=pct((t[hs] > t[ss]).mean())))
tbl(pd.DataFrame(rows))
dd = t[t.human_DDC < 1]; W(f"Events where the gate bites the human pass: {len(dd)} of {len(t)} (gate 0.5 on {int((dd.human_DDC == 0.5).sum())}, 0 on {int((dd.human_DDC == 0).sum())}). In these the human spends {dd.opp_lane_s.min():.1f}–{dd.opp_lane_s.max():.1f} s of the 4 s (median {dd.opp_lane_s.median():.1f} s) in an opposite-direction lane at {dd.opp_lane_speed.min():.1f}–{dd.opp_lane_speed.max():.1f} m/s; the obstacle is stationary in {int(dd.obstacle_stationary.sum())} of {len(dd)}. Disabling the gate lifts {int(((dd.human_score < 0.8) & (dd.human_score_no_ddc >= 0.8)).sum())} of them to ≥ 0.8; the others still fail on " + ", ".join(f"{k} {int((dd[dd.human_score_no_ddc < 0.8][f'human_{k}'] < 1).sum())}" for k in ["NC", "DAC", "TTC", "LK"]) + "."); W()
oo = t[(t.opp_lane_s > 0)]; W(f"Any time in an opposite-direction lane: {len(oo)} of {len(t)} events; gate untouched (DDC = 1) on {int((oo.human_DDC == 1).sum())} of them (less than 2 m of travel there within 1 s)."); W()
cols = ["uuid", "t_anchor", "speed", "clearance", "cat", "obstacle_stationary", "opp_lane_s", "opp_lane_speed", "human_DDC", "human_score", "human_score_no_ddc", "stay_score", "stay_score_no_ddc"]
tbl(dd.sort_values("clearance")[cols].assign(uuid=lambda x: x.uuid.str[:8]).round(2))
po = bt[bt.kind == "pullout"]; W("NuRec pull-outs:"); W(); tbl(po[["uuid", "t_anchor", "speed", "opp_lane_s", "opp_lane_speed", "human_DDC", "human_DAC", "human_NC", "human_score", "human_score_no_ddc", "stay_score", "stay_score_no_ddc"]].assign(uuid=lambda x: x.uuid.str[:8]).round(2))
W("Per-event table for all 72: `btest.csv` (columns `opp_lane_s`, `opp_lane_speed`, `obstacle_stationary`, `*_score`, `*_score_no_ddc`, per-term)."); W()
# ---------------- C
v = pd.read_csv(HERE / "s21_vocab_speed.csv"); an = pd.read_csv(HERE / "s21_vocab_anchors.csv"); de = pd.read_csv(HERE / "s21_anchor_index_demo.csv")
W("### Part C — vocabulary"); W(); tbl(v.rename(columns={"threshold_mps": "speed above, m/s", "max_speed_above": "anchors by max speed", "final_speed_above": "anchors by final speed", "mean_speed_above": "anchors by mean speed", "share_max_pct": "% of 8,192"}))
W(f"Max speed per anchor: median {an.max_speed.median():.1f}, p99 {an.max_speed.quantile(.99):.1f}, **maximum {an.max_speed.max():.2f} m/s**; final speed maximum {an.final_speed.max():.2f} m/s; farthest end point {an.end_x.max():.1f} m ahead in 4 s. Histogram: `s21_png/vocab_speed_hist.png`. Q4: {100*(an.max_speed > 20).mean():.2f} % of anchors exceed 20 m/s → **VOCAB-CAPPED**."); W()
W("Anchor-index generation on our own metric caches (upstream recipe, 8,192 anchors per token):"); W(); tbl(de)
# ---------------- D
o = pd.read_csv(HERE / "s21_bonnet.csv"); ok = o[o.bonnet_rows_static < 60]; W("### Part D — bonnet mask"); W()
W(f"Bonnet top edge found on {len(ok)} of {len(o)} clips: bonnet height {int(ok.bonnet_rows_static.min())}–{int(ok.bonnet_rows_static.max())} input rows (median {ok.bonnet_rows_static.median():.0f}); N = {int(ok.bonnet_rows_static.max())} + 2 = **{int(ok.bonnet_rows_static.max()) + 2} rows** of 256. Failed detections: {o[o.bonnet_rows_static >= 60].uuid.str[:8].tolist()} (edge placed at the top of the search window; see `s21_png/bonnet_edges.png`). The dataset's own ego mask is not usable for this: it is a template reaching {int(o.bonnet_rows_mask.median())} rows (median) and covers road above the bonnet (`s21_png/bonnet_mask_check.png`)."); W()
try:
    s0 = pd.read_csv(C.S1 / "samples.csv"); s1 = pd.read_csv(HERE / "s21_masked" / "samples.csv"); mm = s0.merge(s1, on="token", suffixes=("", "_m")); assert len(mm) == len(s0) == len(s1)
    a = mm[(mm.type == "A") & (mm.speed >= 20)]; o_ = mm[(mm.type == "O") & (mm.speed >= 5) & (mm.speed <= 15)]
    c3u, c3m = (a.fastest_end <= 18.8).mean(), (a.fastest_end_m <= 18.8).mean(); c4u, c4m = (o_.chosen_step1 / o_.speed).median(), (o_.chosen_step1_m / o_.speed).median()
    tbl(pd.DataFrame([dict(check="C3 fastest of 32 ≤ 18.8 m/s (high-speed anchors ≥ 20 m/s)", n=len(a), unmasked=pct(c3u), masked=pct(c3m), change=f"{100*(c3m-c3u):+.1f} points"),
                      dict(check="C4 median chosen first-step speed ÷ human speed (ordinary, 5–15 m/s)", n=len(o_), unmasked=f"{c4u:.3f}", masked=f"{c4m:.3f}", change=f"{c4m-c4u:+.3f}"),
                      dict(check="fastest-of-32 end speed, median over all anchors (m/s)", n=len(mm), unmasked=f"{mm.fastest_end.median():.2f}", masked=f"{mm.fastest_end_m.median():.2f}", change=f"{mm.fastest_end_m.median()-mm.fastest_end.median():+.2f}"),
                      dict(check="chosen plan end speed, mean |masked − unmasked| (m/s)", n=len(mm), unmasked="", masked=f"{(mm.chosen_end_m - mm.chosen_end).abs().mean():.2f}", change=f"same proposal index on {pct((mm.chosen_idx_m == mm.chosen_idx).mean())}")]))
    q5 = (0.8 <= c4m <= 1.2) and abs(c3m - c3u) <= 0.05; W(f"Q5: C4 in [0.8, 1.2] {'yes' if 0.8 <= c4m <= 1.2 else 'no'}; C3 within 5 points {'yes' if abs(c3m - c3u) <= 0.05 else 'no'} → **{'holds' if q5 else 'fails'}**."); W()
except FileNotFoundError: W("masked samples not ready"); q5 = None
json.dump(dict(Q1=float(r21["stay_dac_all"]), Q2=float(r21["p2"]), Q3a=float(r21["p4a"]), Q3b=float(r21["p4b"]), P3=float(r21["p3"]), Q4_share_gt20=float((an.max_speed > 20).mean()), Q5=(None if q5 is None else bool(q5)), N=int(ok.bonnet_rows_static.max()) + 2), open(HERE / "s21_checks.json", "w"))
open(HERE / "s21_tables.md", "w").write("\n".join(out))
