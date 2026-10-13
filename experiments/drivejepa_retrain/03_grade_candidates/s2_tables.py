"""S2 tables + loader check. Run through run_s2-style env: see bottom of REPORT. (conda drive-jepa python, cwd navsim_v2)"""
import sys, numpy as np, pandas as pd, pathlib, torch
import s2_common as C
HERE = C.HERE; out = []; W = lambda s="": (out.append(s), print(s))
def tbl(df):
    f = lambda v: "" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    W("| " + " | ".join(map(str, df.columns)) + " |"); W("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False): W("| " + " | ".join(f(v) for v in r) + " |")
    W()
T = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]
# ---- Step 3
a = pd.read_csv(HERE / "align.csv"); W("### Step 1–3 — map, agents, alignment (pilot)"); W()
tbl(a[["type", "uuid", "lanes", "route_lanes", "intersections", "stop_lines_explicit", "traffic_lights", "tracks", "drivable_m2", "dt_s", "pos_med", "pos_max", "head_med", "head_max", "png"]].assign(uuid=lambda d: d.uuid.str[:8]))
okA = ((a.pos_med <= 0.2) & (a.head_med <= 0.5)).sum(); W(f"P1: median residual ≤ 0.2 m and ≤ 0.5° on **{okA} of {len(a)}** pilot scenes (position median max {a.pos_med.max():.3f} m, heading median max {a.head_med.max():.3f}°; worst single-pose residual {a.pos_max.max():.2f} m). Clock offset {a.dt_s.min():.2f}–{a.dt_s.max():.2f} s. `rig_trajectories.json` vs clipgt egomotion after a rigid fit: max residual {a.rigjson_pos_max.max():.1e} m (same trajectory)."); W()
# ---- Step 4
g = pd.read_csv(HERE / "pilot_grades.csv", low_memory=False); ok = g[g.status == "ok"]; bad = g[g.status != "ok"]
W("### Step 4 — pilot grades"); W(); W(f"Anchors graded {ok.token.nunique()} of {g.token.nunique()}; not graded: {bad.status.value_counts().to_dict()}."); W()
rows = []
for typ, d in ok.groupby("type"):
    for name, sel in [("human", d.plan == "human"), ("stay still", d.plan == "stay"), ("model chosen", d.chosen == True), ("32 proposals (all)", d.plan.str.startswith("p"))]:
        x = d[sel]; rows.append(dict(type=typ, plan=name, n=len(x), **{t: x[t].mean() for t in T}, share_ge_0_8=(x.score >= 0.8).mean()))
tbl(pd.DataFrame(rows))
h = ok[ok.plan == "human"]; hao = h[h.type.isin(["A", "O"])]; P2 = (hao.score >= 0.8).mean()
W(f"P2: human EPDMS ≥ 0.8 on {int((hao.score >= 0.8).sum())} of {len(hao)} A and ordinary anchors = **{100*P2:.1f} %** (A {100*(h[h.type=='A'].score >= 0.8).mean():.0f} %, ordinary {100*(h[h.type=='O'].score >= 0.8).mean():.0f} %, tight-pass scenes {100*(h[h.type=='T'].score >= 0.8).mean():.0f} %). Terms below 1 on those human paths: " + ", ".join(f"{t} {100*(hao[t] < 1).mean():.0f} %" for t in T[:-1]) + "."); W()
p = ok[ok.plan.str.startswith("p")]; sd = p.groupby(["type", "token"]).score.std(ddof=0).reset_index(); sdA = sd[sd.type == "A"]; P3 = (sdA.score > 0.05).mean()
W(f"P3: std of the 32 grades > 0.05 on {int((sdA.score > 0.05).sum())} of {len(sdA)} A anchors = **{100*P3:.1f} %** (median std A {sdA.score.median():.3f}, T {sd[sd.type=='T'].score.median():.3f}, O {sd[sd.type=='O'].score.median():.3f}). On A anchors the terms that vary across the 32: " + ", ".join(f"{t} {100*(p[p.type=='A'].groupby('token')[t].std(ddof=0) > 1e-6).mean():.0f} %" for t in T[:-1]) + "."); W()
ch = ok[ok.chosen == True].set_index("token"); best = p.groupby("token").score.max(); W(f"Model's chosen plan vs the best of its 32 by this grader: mean chosen {ch.score.mean():.3f}, mean best {best.mean():.3f}; chosen is the top-graded plan on {100*(ch.score >= best.reindex(ch.index) - 1e-9).mean():.0f} % of anchors. Correlation (Spearman, per anchor, model's own score vs grade): median {p.groupby('token').apply(lambda d: d.model_pdm.corr(d.score, method='spearman')).median():.2f}."); W()
# ---- loader check
sys.path.insert(0, "."); from navsim.planning.training.dataset import CacheOnlyDataset; from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.drive_jepa_perception_based.score_module.compute_navsim_score import load_metric_cache
class B:
    def __init__(s, n): s.n = n
    def get_unique_name(s): return s.n
root = C.NAS / "s1" / "cache"; ds = CacheOnlyDataset(str(root), feature_builders=[B("pad_feature")], target_builders=[B("pad_target"), B("pad_label2")], log_names=[u for u in pd.read_csv(HERE / "pilot_clips.csv").uuid])
f, t = ds[len(ds) // 2]; mcl = MetricCacheLoader(root); mc = load_metric_cache(mcl.metric_cache_paths[t["token"]])
W("### Complete sample through the upstream loaders"); W()
W(f"`CacheOnlyDataset` (`training/dataset.py:33`) over `{root}` with builders `pad_feature`, `pad_target`, `pad_label2`: {len(ds)} samples. One sample (`{t['token']}`):"); W()
tbl(pd.DataFrame([dict(part="images + ego_status (pad_feature.gz)", content="; ".join(f"{k} {tuple(v.shape)}" for k, v in f.items())),
                  dict(part="label 1 (pad_target.gz)", content=f"trajectory {tuple(t['trajectory'].shape)} {t['trajectory'].dtype}"),
                  dict(part="label 2 grades (pad_label2.gz)", content=f"label2_scores {tuple(t['label2_scores'].shape)} = 34 plans × {t['label2_terms']}; has_label2 {t['has_label2']}"),
                  dict(part="label 2 source (metric_cache.pkl, lzma)", content=f"`MetricCacheLoader` finds {len(mcl.metric_cache_paths)} tokens; `load_metric_cache` → {type(mc).__name__}: observation {len(mc.observation._occupancy_maps)} steps, centerline {mc.centerline.length:.0f} m, {len(mc.route_lane_ids)} route lanes, drivable map {len(mc.drivable_area_map.tokens)} polygons, ego v {mc.ego_state.dynamic_car_state.speed:.1f} m/s")]))
# ---- Step 5
b = pd.read_csv(HERE / "btest.csv"); W("### Step 5 — B formula test (human path from the usdz vs stay still)"); W()
t5 = b[b.kind == "tight"]; tk = t5[t5.status == "ok"].copy(); W(f"Tight passes graded: {len(tk)} of {len(t5)} ({t5[t5.status != 'ok'].status.value_counts().to_dict()}). Anchor = the last 0.5 s anchor at least 0.5 s before the pass."); W()
tk["cb"] = pd.cut(tk.clearance, [0, 0.3, 0.5, 0.75, 1.0], labels=["≤0.3", "0.3–0.5", "0.5–0.75", "0.75–1.0"])
rows = []
for cb, d in list(tk.groupby("cb", observed=True)) + [("all", tk)]:
    rows.append(dict(clearance_m=cb, events=len(d), human_score=d.human_score.mean(), stay_score=d.stay_score.mean(), human_ge_0_8=(d.human_score >= 0.8).mean(), human_gt_stay=(d.human_score > d.stay_score).mean(), **{f"h_{t}": d[f"human_{t}"].mean() for t in T[:-1]}))
tbl(pd.DataFrame(rows))
P4a = (tk.human_score >= 0.8).mean(); P4b = (tk.human_score > tk.stay_score).mean()
W(f"P4: human EPDMS ≥ 0.8 on {int((tk.human_score >= 0.8).sum())} of {len(tk)} = **{100*P4a:.1f} %**; human > stay on {int((tk.human_score > tk.stay_score).sum())} of {len(tk)} = **{100*P4b:.1f} %** (both need ≥ 80 %)."); W()
low = tk[tk.human_score < 0.8]; W(f"Why the {len(low)} human passes below 0.8 fall short (a term counts when it is below 1): " + ", ".join(f"{t} {int((low[f'human_{t}'] < 1).sum())}" for t in T[:-1]) + f". Gate terms (score forced to 0 or halved): NC {int((low.human_NC < 1).sum())}, DAC {int((low.human_DAC < 1).sum())}, DDC {int((low.human_DDC < 1).sum())}. Lane keeping alone caps the score at {12/14:.3f}; LK is the only failing term on {int(((low.human_LK < 1) & (low[[f'human_{t}' for t in T[:-1] if t != 'LK']] >= 1).all(axis=1)).sum())} of them."); W()
W(f"Human ≤ stay on {int((tk.human_score <= tk.stay_score).sum())} events: both 0 on {int(((tk.human_score == 0) & (tk.stay_score == 0)).sum())}, tie above 0 on {int(((tk.human_score == tk.stay_score) & (tk.human_score > 0)).sum())}, stay strictly higher on {int((tk.human_score < tk.stay_score).sum())}. Stay still loses mainly on comfort (HC mean {tk.stay_HC.mean():.2f}, a hard stop from speed) and progress (EP mean {tk.stay_EP.mean():.2f})."); W()
lk = tk.sort_values("clearance"); W("Lane-keeping failure rate of the human pass by clearance: " + "; ".join(f"{cb} {100*(d.human_LK < 1).mean():.0f} %" for cb, d in tk.groupby("cb", observed=True)) + f". Share of the 4 s the scorer's ego box straddles two lanes: mean {tk.human_multi_lane.mean():.2f}."); W()
W("Per-event table: `btest.csv` (70 tight + 2 pull-outs; per-term columns for human and stay). The 10 lowest human grades:"); W()
tbl(tk.sort_values(["human_score", "clearance"]).head(10)[["uuid", "t_anchor", "speed", "clearance", "cat", "human_score", "stay_score", "human_NC", "human_DAC", "human_DDC", "human_TTC", "human_LK", "human_EP", "human_HC"]].assign(uuid=lambda d: d.uuid.str[:8]).round(2))
po = b[b.kind == "pullout"]; W("NuRec pull-outs (anchor at standstill just before move-off):"); W()
tbl(po[["uuid", "t_anchor", "speed", "human_score", "stay_score"] + [f"human_{t}" for t in T[:-1]] + [f"stay_{t}" for t in T[:-1]] + ["human_nondrivable", "stay_nondrivable"]].assign(uuid=lambda d: d.uuid.str[:8]).round(2))
import json; json.dump(dict(P1=int(okA), P2=float(P2), P3=float(P3), P4a=float(P4a), P4b=float(P4b)), open(HERE / "checks.json", "w"))
open(HERE / "tables.md", "w").write("\n".join(out))
