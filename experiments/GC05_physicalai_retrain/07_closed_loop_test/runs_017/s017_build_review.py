#!/usr/bin/env python
"""GENERATOR-ONLY TRIAL review page: recorded 0.1.6 (left) vs 0.1.7 (right), 0.1.4b as a third value in the tables, two highway columns added. Copy of 07_closed_loop_test/s5_build_review.py (diff: tables/s017_build_review_diff.txt); page folder on the NAS.
Original docstring: S5 TRIAL review page: recorded 0.1.4b (left) vs 0.1.6 (right), the same page as the 0.1.4 vs 0.1.4b review
(baseline_014b/build_014b_artifact.py on top of build_014b_site.py; both are imported / re-applied here, neither is edited).

Everything that is wording or grouping comes from the parent's tables, nothing is hard-coded here:
  tables/per_scene.csv        one row per scene, columns 014b_<metric> / 016_<metric> (metric names of baseline_014b/tables/per_scene.csv)
  tables/replans_016.csv      per-replan log of the 0.1.6 runs (columns of baseline_014b/tables/replans_014b.csv); 0.1.4b's is read there
  tables/review_groups.json   {"improved": {"title","lead","points":[..],"scenes":[..]}, "decreased": {..},
                               "same": {"title","lead","points":[..],"buckets":[{"title","note"?,"scenes":[..]}]},
                               optional: "why": {scene: one sentence}, "header": str, "key": [str, ..]}
Group membership is the JSON's; a scene whose score change (|delta| < 0.02 = same) disagrees with its JSON group is listed on stdout.
Scenes without a finished 0.1.6 run + menus + video are left out and counted in the header.

  cd paddock-sid/Alpasim && alpasim/.venv/bin/python analysis_3.0/Retrain-data/07_closed_loop_test/s5_build_review.py [--out DIR] [--tables DIR] [--scenes a,b] [--standin]
  --standin : development only. The right arm is the recorded 0.1.4b run again and the groups are generated; writes review_standin/.
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, sys
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent; S5T = HERE.parent; T = HERE / "tables"; B14 = HERE.parents[2] / "Re-ranker" / "baseline_014b"; ALP = HERE.parents[3]; NAS = pathlib.Path("{runs}/drivejepa_retrain/07_closed_loop_test_017")
sys.path.insert(0, str(B14)); import build_014b_site as BS  # noqa: E402   (TEMPLATE, arm_series; read-only)
BR, f = BS.BR, BS.f
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--standin", action="store_true"); ap.add_argument("--scenes", default=""); ap.add_argument("--tables", default=""); a = ap.parse_args()
if a.tables: T = pathlib.Path(a.tables)   # another tables folder (the one-scene proof writes a provisional one)
assert not a.standin; OUT = pathlib.Path(a.out) if a.out else NAS / "review"; (OUT / "videos").mkdir(parents=True, exist_ok=True)
RUN16 = lambda s: ALP / "runs" / "drivejepa" / f"drivejepa_016_{s}"; RUN17 = lambda s: NAS / "runs" / f"drivejepa_017_{s}"
rep14 = pd.read_csv(S5T / "tables" / "replans_016.csv")   # LEFT arm = 0.1.6 (names of the original kept: "old"/rep14 = left, "new"/rep16 = right)
if a.standin:
    base = pd.read_csv(B14 / "tables" / "per_scene.csv"); keep = [c for c in base.columns if c.startswith("014b_")]
    per = pd.concat([base[["scene", "lane"]], base[keep], base[keep].rename(columns=lambda c: "016_" + c[5:])], axis=1).set_index("scene"); rep16 = rep14.copy()
    scenes = (a.scenes.split(",") if a.scenes else list(per.index)[:6])
    G = dict(improved=dict(title="Score improved", lead="STAND-IN", points=["stand-in"], scenes=[]), decreased=dict(title="Score decreased", lead="STAND-IN", points=["stand-in"], scenes=[]),
             same=dict(title="No change in score", lead="STAND-IN: right arm = the 0.1.4b run again.", points=["stand-in point"], buckets=[dict(title="Stand-in bucket A", note="first half", scenes=scenes[: len(scenes) // 2]), dict(title="Stand-in bucket B", note="second half", scenes=scenes[len(scenes) // 2:])]),
             why={s: f"stand-in sentence for {s}" for s in scenes})
else:
    per = pd.read_csv(T / "per_scene.csv").set_index("scene"); rep16 = pd.read_csv(T / "replans_017.csv"); G = json.load(open(T / "review_groups.json"))
    scenes = a.scenes.split(",") if a.scenes else list(per.index)
def src(s, arm):
    """run dir, menu dir, video file of one arm"""
    if arm == "016": return RUN16(s), S5T / "menus" / f"d016_{s}", S5T / "videos" / f"d016_{s}.mp4"
    return RUN17(s), NAS / "menus" / f"d017_{s}", NAS / "videos" / f"d017_{s}.mp4"
def ready(s): r, m, v = src(s, "017"); return (r / "aggregate" / "metrics_results.parquet").is_file() and (pathlib.Path(str(m) + ".json")).is_file() and v.is_file() and s in per.index and np.isfinite(per.loc[s].get("017_scene_score", np.nan))
missing = [s for s in scenes if not ready(s)]; scenes = [s for s in scenes if ready(s)]; assert scenes, f"no scene is ready (missing: {missing})"
BUCKETS = G["same"].get("buckets", []); BUCKET_OF = {s: i for i, b in enumerate(BUCKETS) for s in b["scenes"]}; GROUP_OF = {s: g for g in ("improved", "decreased") for s in G[g].get("scenes", [])} | {s: "same" for s in BUCKET_OF} | {s: "same" for s in G["same"].get("scenes", [])}
WHY = G.get("why", {}); rows = []; disagree = []
for s in scenes:
    r = per.loc[s]; ra, ma, va = src(s, "016"); rb, mb, vb = src(s, "017"); A, pa = BS.arm_series(ra, ma, "016"); Bs, pb = BS.arm_series(rb, mb, "017")
    def arm(pre, peak, rp):
        q = rp[rp.scene == s]
        return dict(score=f(r[f"{pre}_scene_score"], 3), progress=f(r[f"{pre}_progress_clipped_rel"], 3), at_fault=int(r[f"{pre}_collision_at_fault"] > 0), coll_any=int(r[f"{pre}_collision_any"] > 0), offroad=int(r[f"{pre}_offroad"] > 0),
                    dist_gt=f(r[f"{pre}_dist_to_gt_trajectory"], 2), dist=f(r[f"{pre}_dist_traveled_m"], 1), min_v=f(r[f"{pre}_min_speed_after_ho"], 2), slow_s=f(r[f"{pre}_longest_slow_s"], 1), slow_t=f(r[f"{pre}_slow_t_start"], 1),
                    headway=f(r[f"{pre}_min_headway_after_ho"], 2), exit_kind=str(r[f"{pre}_exit_kind"]), exit_s=f(r[f"{pre}_exit_s"], 2), peak_steer=f(peak, 3), stall=bool(r.get(f"{pre}_stall", False)) if str(r.get(f"{pre}_stall", "")) not in ("", "nan") else False,
                    changed_n=int(q.changed.sum()), replans=int(len(q)), changed_share=f(100 * float(q.changed.mean()) if len(q) else np.nan, 1), hstep=f(r[f"{pre}_handover_speed_step"], 1), hbrake=f(r[f"{pre}_max_brake_first_1s"], 1))
    o, n = arm("016", pa, rep14), arm("017", pb, rep16); hw = bool(r["016_a1"] >= 20.0) if "016_a1" in r else False
    b3 = dict(score=f(r["014b_scene_score"], 3), progress=f(r["014b_progress_clipped_rel"], 3), at_fault=int(r["014b_collision_at_fault"] > 0), coll_any=int(r["014b_collision_any"] > 0), offroad=int(r["014b_offroad"] > 0), peak_steer=f(r["014b_peak_steer"], 3), changed_n=int(r["014b_momentum_changed_n"]), hstep=f(r["014b_handover_speed_step"], 1), hbrake=f(r["014b_max_brake_first_1s"], 1)); delta = f(n["score"] - o["score"], 3); by = "improved" if delta >= 0.02 else ("decreased" if delta <= -0.02 else "same"); out = GROUP_OF.get(s, by)
    if out != by or (s not in GROUP_OF): disagree.append((s, by, GROUP_OF.get(s)))
    rows.append(dict(scene=s, lane=r.get("lane", ""), old=o, new=n, b3=b3, b3s=b3["score"], hw=hw, hstep_new=(n["hstep"] if hw else None), hbrake_new=(n["hbrake"] if hw else None), delta=delta, a=A, b=Bs, video_old=f"videos/d016_{s}.mp4", video_new=f"videos/d017_{s}.mp4", score_old=o["score"], score_new=n["score"], changed_share=n["changed_share"], changed_old=o["changed_share"],
                     peak_old=o["peak_steer"], peak_new=n["peak_steer"], prog_old=o["progress"], prog_new=n["progress"], dist_old=o["dist"], dist_new=n["dist"], hw_old=o["headway"], hw_new=n["headway"], slow_old=o["slow_s"], slow_new=n["slow_s"],
                     outcome=out, why=WHY.get(s, ""), bucket=(BUCKET_OF.get(s, -1) if out == "same" else -1)))
    for v, name in ((va, f"d016_{s}.mp4"), (vb, f"d017_{s}.mp4")):
        d = OUT / "videos" / name
        if not d.is_file() or d.stat().st_size != v.stat().st_size: shutil.copy2(v, d)
rows.sort(key=lambda x: ({"improved": 0, "decreased": 1, "same": 2}[x["outcome"]], x["bucket"] if x["bucket"] >= 0 else 99, -x["delta"], x["scene"]))
N = len(rows); so = [x["score_old"] for x in rows]; sn = [x["score_new"] for x in rows]; up = sum(x["outcome"] == "improved" for x in rows); dn = sum(x["outcome"] == "decreased" for x in rows)
co, ro = sum(x["old"]["changed_n"] for x in rows), sum(x["old"]["replans"] for x in rows); cn, rn = sum(x["new"]["changed_n"] for x in rows), sum(x["new"]["replans"] for x in rows)
key = G.get("key") or [f"Mean score over the {N} scenes: {sum(so) / N:.3f} (0.1.6) → {sum(sn) / N:.3f} (0.1.7); 0.1.4b {sum(x['b3']['score'] for x in rows) / N:.3f}.",
       f"Score up on {up}, down on {dn}, unchanged on {N - up - dn} of the {N} scenes.",
       f"At-fault collisions: {sum(x['old']['at_fault'] for x in rows)} (0.1.6) → {sum(x['new']['at_fault'] for x in rows)} (0.1.7), 0.1.4b {sum(x['b3']['at_fault'] for x in rows)}; any collision {sum(x['old']['coll_any'] for x in rows)} → {sum(x['new']['coll_any'] for x in rows)}, 0.1.4b {sum(x['b3']['coll_any'] for x in rows)}.",
       f"Off-road exits: {sum(x['old']['offroad'] for x in rows)} → {sum(x['new']['offroad'] for x in rows)}, 0.1.4b {sum(x['b3']['offroad'] for x in rows)}.",
       f"The nudge (§3.5) changed the pick on {co} of {ro} replans ({100 * co / max(ro, 1):.0f} %) under 0.1.6 and on {cn} of {rn} ({100 * cn / max(rn, 1):.0f} %) under 0.1.7."]
header = G.get("header") or ((f"All {N} census scenes, one closed-loop run each." if not missing else f"{N} of {N + len(missing)} scenes finished so far.") +
          " Generator only: 0.1.7 = 0.1.6 + the S10 generator weights + the batch fix on; the scorer is the pass-9 scorer of 0.1.6, unchanged. Scenes are grouped by score change, 0.1.6 → 0.1.7; 0.1.4b is shown in brackets.")
data = dict(groups={k: dict(title=G[k]["title"], lead=G[k]["lead"], points=G[k]["points"]) for k in ("improved", "decreased", "same")}, buckets=[dict(title=b["title"], note=b.get("note", ""), n=sum(x["bucket"] == i for x in rows)) for i, b in enumerate(BUCKETS)],
            built=pd.Timestamp.now().isoformat(timespec="minutes"), header=header, rows=rows, key=key, n=N, rule=[], missing=missing, standin=bool(a.standin))
# ---- three-run summary (Metrics tab, top): initial run / first training / last training, every number from tables/per_scene.csv + the replans tables
def _run3(pre, rp):
    d = per.loc[scenes]; g = lambda k: d[f"{pre}_{k}"]; names = lambda m: ", ".join(sorted(d.index[m])) or "none"; af = g("collision_at_fault") > 0; off = g("offroad") > 0; q = rp[rp.scene.isin(scenes)]
    return dict(score=f"{g('scene_score').mean():.3f}", progress=f"{g('progress_clipped_rel').mean():.3f}", dist=f"{g('dist_traveled_m').mean():.0f} m", speed=f"{g('mean_speed_after_ho').mean():.1f} m/s", full=f"{int((g('scene_score') >= 0.999).sum())} of {N}", zero=f"{int((g('scene_score') <= 0.05).sum())} of {N}",
                at_fault=f"{int(af.sum())} ({names(af)})", coll=str(int((g('collision_any') > 0).sum())), offroad=f"{int(off.sum())} ({names(off)})", stalls=str(int((g('stall').astype(str) == 'True').sum())),
                steer=f"{int((g('peak_steer') > 0.3).sum())} of {N}", brake=f"{float((-g('peak_demanded_decel')).median()):.1f} m/s²", nudge=f"{int(q.changed.sum())} of {len(q)} ({100 * q.changed.mean():.0f} %)")
RUNS3 = [dict(name="0.1.4b", role="Initial run", when="recorded 2026-10-01 / 02", rp=pd.read_csv(B14 / "tables" / "replans_014b.csv"), pre="014b",
              what="The model as shipped: upstream Drive-JEPA weights, no retraining. The only addition is the paper's momentum nudge (§3.5), which re-scores the 32 plans against the plan chosen half a second earlier.",
              changed="Baseline. Nothing trained by us.",
              weights="Shipped checkpoint (drive_jepa_perception_based_agent_vitl_v2), generator and scorer as published.",
              got="Reference point for both trainings.",
              wrong="After handover the chosen plan often starts far slower than the car, which asks for a hard brake (median 7.5 m/s²) and makes the controller swerve: 10 scenes with a full steering swing. Both highway scenes leave the road in the first half second. Neither pull-out moves."),
         dict(name="0.1.6", role="First training", when="run 2026-10-07 / 08", rp=rep14, pre="016",
              what="First retrain (S4, pass 9): generator and scorer both retrained on our driving clips, encoder kept frozen. Two input changes that the new weights were trained with: the bonnet rows of the image are blacked out and the acceleration input is zeroed.",
              changed="Against the initial run: new generator weights, new scorer weights, bonnet mask (37 rows), acceleration zeroed in the model input and in the nudge. Three changes at once.",
              weights="S4 pass 9 (sha256 8a71c3c0…). Trained in batches; three upstream attention lines made each plan depend on its batch neighbours, which was not known yet.",
              got="The hard brake after handover is gone on ordinary roads (median demand 7.5 → 1.4 m/s²). Two of four at-fault collisions gone, none new. Five more scenes at full score.",
              wrong="Highway not fixed: both scenes still leave the road with a full steering swing, and the menu holds absurd plans (up to 68 m/s). Pull-outs unchanged. Five scenes lost score by drifting off the human path or driving slower."),
         dict(name="0.1.7", role="Last training", when="run 2026-10-10", rp=rep16, pre="017",
              what="Generator-only retrain (S10): the generator is trained further from pass 9 with the batch fix on, so a plan no longer depends on what else is in the batch. The scorer is frozen and is bit for bit the 0.1.6 scorer. The batch fix also runs in the car.",
              changed="Against the first training: generator weights only (155 of 171 generator tensors differ; 30 of 30 scorer tensors identical) + the batch fix switched on. Mask, zeroed acceleration and nudge unchanged.",
              weights="S10 final choice, pass +2 (sha256 29d5f951…): pass 9 → 1 pass at learning rate 1e-4 → 2 passes at 2.5e-5.",
              got="Both highway scenes stay on the road while scored (handover speed step −3.3 → −0.1 and −2.7 → −1.2 m/s). Menus are sane (fastest plan 32–35 m/s). 9 scenes improve, 21 at full score, collisions while scored 7 → 3, steering swings 7 → 1.",
              wrong="The car now drives faster than the human and the unchanged scorer does not pick the slower plans on the menu: acd22430 has a new at-fault collision at +15.4 s, 04749bb9 crashes and leaves the road just after scoring closes, bd07fb91's crash now falls inside the scored window, d1a25a99 runs 1.7 m wider through a turn and is flagged off the road. Pull-outs unchanged.")]
for _r in RUNS3: _r.update(_run3(_r.pop("pre"), _r.pop("rp")))
data["runs3"] = RUNS3
data["runs3_rows"] = [["what", "What this run is", 1], ["changed", "What changed from the run before", 1], ["weights", "Weights", 1], ["when", "Closed-loop run", 1], ["score", "Mean score, 30 scenes", 0], ["progress", "Mean progress", 0], ["dist", "Mean distance driven", 0], ["speed", "Mean speed after handover", 0],
                      ["full", "Scenes at full score", 0], ["zero", "Scenes scoring zero (≤ 0.05)", 0], ["at_fault", "At-fault collisions", 0], ["coll", "Any collision while scored", 0], ["offroad", "Off-road exits", 0], ["steer", "Scenes with peak steer above 0.3 rad", 0], ["brake", "Hardest braking asked by a driven plan, median over scenes", 0],
                      ["stalls", "Stalls (S5 rule)", 0], ["nudge", "Replans where the nudge changed the pick", 0], ["got", "What it fixed", 1], ["wrong", "What is still wrong", 1]]
t = BS.TEMPLATE
def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:70]); t = t.replace(old, new)
# ---- 1. the structural changes of build_014b_artifact.py (groups, Insights first, buckets, extra table columns), re-applied on the same template
rep('<title>Drive-JEPA 0.1.4 vs 0.1.4b (paper §3.5) — 5 scenes</title>', '<title>generator only: scorer unchanged — Drive-JEPA 0.1.4 vs 0.1.4b</title>')
rep(':root{--bg:#111;', ':root{color-scheme:dark;--bg:#111;')
rep('<h1>Drive-JEPA adapter 0.1.4 (recorded) vs 0.1.4b (paper §3.5 momentum-aware selection) · 5 scenes · 20 Hz referee</h1>', f'<h1>generator only: scorer unchanged · Drive-JEPA adapter 0.1.4 (recorded) vs 0.1.4b (S10 generator + batch fix; same pass-9 scorer, bonnet mask, zeroed acceleration, §3.5 momentum) · {N} scenes · 20 Hz referee</h1>')
rep('<div class="tab on" data-p="videos">Tab 1 — videos</div><div class="tab" data-p="metrics">Tab 2 — summary</div>', '<div class="tab on" data-p="metrics">Metrics</div><div class="tab" data-p="videos">Videos</div>')
rep('<div class="panel on" id="videos"></div><div class="panel" id="metrics"></div>', '<div class="panel on" id="metrics"></div><div class="panel" id="videos"></div>')
rep('''onclick="document.querySelector('.tab').click()"''', '''onclick="document.querySelector('.tab[data-p=videos]').click();showSub('${r.outcome}','vids')"''')
rep("scene order = rank under the rule", "scenes grouped by score change")
rep(".up{color:var(--good)}", ".sec{margin:22px 0 30px}.sec>h2{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px}.sub2{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto;justify-content:flex-end}.sub2 button{background:#161616;color:var(--mut);border:1px solid var(--line);border-radius:14px;padding:4px 14px;font-size:13px;cursor:pointer}.sub2 button.on{background:#2b3a4d;color:var(--fg);border-color:#4a6a8f}.sub2 button:focus-visible{outline:2px solid var(--acc)}.ins{max-width:none}.ins .lead,.ins ul{max-width:980px}.ins table th{white-space:normal;max-width:150px;vertical-align:bottom}.ins .lead{font-size:15px;color:var(--fg);margin:6px 0 10px}.ins ul{margin:0 0 14px 18px;padding:0;font-size:14px}.ins li{margin:5px 0}.ins table td{white-space:normal;text-align:left;vertical-align:top}.ins table td.k{white-space:nowrap}.ins table td.sc{white-space:nowrap;font-family:ui-monospace,Menlo,monospace}.bkt{margin:18px 0 26px;padding-top:4px;border-top:1px solid var(--line)}.bkt h3{margin:10px 0 2px;font-size:14px;color:var(--fg)}tr.bh td{background:#161616;color:var(--acc);font-weight:600;text-align:left}.why1{font-size:13px;color:#cfcfcf;margin-top:10px}.b3{color:var(--mut);font-size:12px}.up{color:var(--good)}")
rep("// ---- Tab 1\nconst V=document.getElementById('videos');", """// ---- Tab 1
const V=document.getElementById('videos');const GV={};
window.showSub=(o,w)=>{const s=document.getElementById('sec-'+o);if(!s)return;s.querySelectorAll('.sub2 button').forEach(b=>b.classList.toggle('on',b.dataset.w===w));s.querySelector('.ins').hidden=(w==='ins')?false:true;s.querySelector('.vids').hidden=(w==='ins');s.querySelectorAll('.bkt').forEach(d=>{d.hidden=(w!=='vids'&&d.id!=='bkt-'+w.slice(1));});};
for(const o of ['improved','decreased','same']){const rows=D.rows.filter(r=>r.outcome===o);const G=D.groups[o];const s=document.createElement('div');s.className='sec';s.id='sec-'+o;
 s.innerHTML=`<h2>${esc(G.title)} · ${rows.length} scene${rows.length===1?'':'s'}<span class="sub2"><button data-w="ins" class="on">Insights</button><button data-w="vids">${o==='same'?'All videos':'Videos'} (${rows.length})</button>${o==='same'?D.buckets.map((b,i)=>b.n?`<button data-w="b${i}" title="${esc(b.note)}">${esc(b.title)} (${b.n})</button>`:'').join(''):''}</span></h2>
 <div class="ins"><p class="lead">${esc(G.lead)}</p><ul>${G.points.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><div class="tw"><table class="gt" data-g="${o}"><thead><tr><th>scene</th><th>score 0.1.4 → 0.1.4b <span class="b3">(@@B3@@)</span></th><th>progress</th><th>at-fault</th><th>any collision</th><th>off-road</th><th>peak steer (rad)</th><th>nudge-changed picks (§3.5)</th><th title="highway scenes only (car at 20 m/s or more at handover): the chosen plan's start speed minus the car's speed, at the handover replan">highway: speed step at handover (m/s)</th><th title="highway scenes only: largest braking a driven plan asks for in its first half second, on the replans of the first second after handover">highway: max braking demanded, first second (m/s²)</th><th style="text-align:left">what the runs show</th></tr></thead><tbody>${rows.map((r,j)=>`${(r.bucket>=0&&(j===0||rows[j-1].bucket!==r.bucket))?`<tr class="bh"><td colspan="11">${esc(D.buckets[r.bucket].title)} · ${rows.filter(q=>q.bucket===r.bucket).length}</td></tr>`:''}<tr><td class="k"><a href="#card-${r.scene}" style="color:#8cc8ff" onclick="showSub('${o}','vids')">${r.scene}</a></td><td class="sc">${fmt(r.score_old)} → ${fmt(r.score_new)} <span class="b3">(${fmt(r.b3.score)})</span></td><td class="sc">${fmt(r.prog_old)} → ${fmt(r.prog_new)} <span class="b3">(${fmt(r.b3.progress)})</span></td><td class="sc">${r.old.at_fault} → ${r.new.at_fault} <span class="b3">(${r.b3.at_fault})</span></td><td class="sc">${r.old.coll_any} → ${r.new.coll_any} <span class="b3">(${r.b3.coll_any})</span></td><td class="sc">${r.old.offroad} → ${r.new.offroad} <span class="b3">(${r.b3.offroad})</span></td><td class="sc">${fmt(r.peak_old,3)} → ${fmt(r.peak_new,3)} <span class="b3">(${fmt(r.b3.peak_steer,3)})</span></td><td class="sc">${r.old.changed_n} → ${r.new.changed_n} <span class="b3">(${r.b3.changed_n})</span></td><td class="sc">${r.hw?`${fmt(r.old.hstep,1)} → ${fmt(r.new.hstep,1)} <span class="b3">(${fmt(r.b3.hstep,1)})</span>`:'–'}</td><td class="sc">${r.hw?`${fmt(r.old.hbrake,1)} → ${fmt(r.new.hbrake,1)} <span class="b3">(${fmt(r.b3.hbrake,1)})</span>`:'–'}</td><td>${esc(r.why)}</td></tr>`).join('')}</tbody></table></div></div><div class="vids" hidden></div>`;
 V.appendChild(s);GV[o]=s.querySelector('.vids');
 if(o==='same'&&rows.length&&rows.every(r=>r.bucket>=0)){D.buckets.forEach((b,i)=>{const n=rows.filter(r=>r.bucket===i).length;if(!n)return;const d=document.createElement('div');d.className='bkt';d.id='bkt-'+i;d.innerHTML=`<h3>${esc(b.title)} · ${n} scene${n>1?'s':''}</h3><p class="sub">${esc(b.note)}</p>`;GV[o].appendChild(d);GV[o+'|'+i]=d;});}s.querySelectorAll('.sub2 button').forEach(b=>b.onclick=()=>showSub(o,b.dataset.w));}""")
rep(" V.appendChild(card);const [a,b]=card.querySelectorAll('video');", " (GV[r.outcome+'|'+r.bucket]||GV[r.outcome]).appendChild(card);const [a,b]=card.querySelectorAll('video');")
rep("<div class=\"strip\">${mrow('score',", "<div class=\"why1\">${esc(r.why)}</div><div class=\"strip\">${mrow('score',")
rep("One play / scrub control per pair drives both videos.", "Scenes are grouped by outcome; each group opens on its Insights (with the per-scene table), the videos are one click away. One play / scrub control per pair drives both videos.")
rep("let sortKey='rank',sortDir=1;", "let sortKey='delta',sortDir=-1;")
rep("['changed_share','§3.5 changed pick (% of replans)'],", "['b3s','score, @@B3@@'],['prog','progress'],['hstep_new','speed step at handover, highway (m/s)'],['hbrake_new','max braking demanded, first second, highway (m/s²)'],['coll','any collision'],['off','off-road'],['dist','distance travelled (m)'],['slow','longest < 1 m/s (s)'],['hw','min headway (s)'],['changed_old','§3.5 changed pick, 0.1.4 (% of replans)'],['changed_share','§3.5 changed pick, 0.1.4b (% of replans)'],")
rep(" default:return `<td>${r[k]==null?'–':esc(r[k])}</td>`;}}", " case 'b3s':return `<td>${fmt(r.b3.score)}</td>`;case 'hstep_new':return `<td>${r.hw?fmt(r.old.hstep,1)+' → '+fmt(r.new.hstep,1):'–'}</td>`;case 'hbrake_new':return `<td>${r.hw?fmt(r.old.hbrake,1)+' → '+fmt(r.new.hbrake,1):'–'}</td>`;case 'prog':return `<td>${fmt(r.prog_old)} → ${fmt(r.prog_new)}</td>`;case 'coll':return `<td>${r.old.coll_any} → ${r.new.coll_any}</td>`;case 'off':return `<td>${r.old.offroad} → ${r.new.offroad}</td>`;case 'dist':return `<td>${fmt(r.dist_old,1)} → ${fmt(r.dist_new,1)}</td>`;case 'slow':return `<td>${fmt(r.slow_old,1)} → ${fmt(r.slow_new,1)}</td>`;case 'hw':return `<td>${fmt(r.hw_old)} → ${fmt(r.hw_new)}</td>`;case 'changed_old':return `<td>${fmt(r.changed_old,1)}</td>`;\n default:return `<td>${r[k]==null?'–':esc(r[k])}</td>`;}}")
rep("<h2>What the five runs show</h2>", f"<h2>What the {N} runs show</h2>"); rep("<h2>The five scenes ", f"<h2>The {N} scenes, 0.1.4 → 0.1.4b ")
rep("M.innerHTML=`<div class=\"note\">${esc(D.header)}</div>", "M.innerHTML=`<div class=\"note\">${esc(D.header)}</div><h2>The three runs at a glance</h2><div class=\"tw\"><table class=\"r3\"><thead><tr><th></th>${D.runs3.map(r=>`<th>${esc(r.role)}<br><span class=\"b3\">adapter ${esc(r.name)}</span></th>`).join('')}</tr></thead><tbody>${D.runs3_rows.map(([k,l,txt])=>`<tr><td class=\"k\">${esc(l)}</td>${D.runs3.map(r=>`<td class=\"${txt?'tx':'sc'}\">${esc(r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class=\"sub\">Same 30 scenes, same referee and settings, one run each. Numbers are read from the three runs with the same code; the text rows are context.</p>")
rep(".b3{color:var(--mut);font-size:12px}", ".b3{color:var(--mut);font-size:12px}table.r3{max-width:1400px}table.r3 th{text-align:left;white-space:normal;vertical-align:bottom;width:29%}table.r3 th:first-child{width:13%}table.r3 td{white-space:normal;text-align:left;vertical-align:top;line-height:1.45}table.r3 td.k{color:var(--acc);font-weight:600}table.r3 td.sc{font-family:ui-monospace,Menlo,monospace}table.r3 td.tx{font-size:13px}")
# ---- 2. what does not exist in this comparison: the "rule" (offline §3.5 effect) table, column and chip
i0 = t.index("\n<h2>How the five were chosen "); i1 = t.index("</table></div>`;\nrenderTable();"); t = t[:i0] + t[i1 + len("</table></div>"):]
rep("['rank','rule rank'],", ""); rep('<span class="chip">rule rank ${r.rank} · offline score ${fmt(r.rule_score,1)}</span>', "")
# ---- 3. both arms run §3.5 here: the left arm shows its own changed picks too
rep("<span>§3.5 changed the pick on <b class=\"mono\" style=\"font-size:14px\">${n.changed_n} / ${n.replans}</b> replans</span>", "<span>§3.5 changed the pick on <b class=\"mono\" style=\"font-size:14px\">${o.changed_n} / ${o.replans} → ${n.changed_n} / ${n.replans}</b> replans</span>")
rep("<div class=\"m\">replans where §3.5 changed the pick<b>${fmt(n.changed_share,1)} %</b></div>", "<div class=\"m\">replans where §3.5 changed the pick<b>${fmt(o.changed_share,1)} % → ${fmt(n.changed_share,1)} %</b></div>")
rep("const ticks=B.replans.filter(p=>p.ch).map(p=>`<rect x=\"${X(p.t)-1.5}\" y=\"${PH-MB-6}\" width=\"3\" height=\"6\" fill=\"#00dcff\"/>`).join('');", "const ticks=B.replans.filter(p=>p.ch).map(p=>`<rect x=\"${X(p.t)-1.5}\" y=\"${PH-MB-6}\" width=\"3\" height=\"6\" fill=\"#00dcff\"/>`).join('')+A.replans.filter(p=>p.ch).map(p=>`<rect x=\"${X(p.t)-1.5}\" y=\"${MT}\" width=\"3\" height=\"5\" fill=\"#5f8f9a\"/>`).join('');")
rep("<i style=\"background:#00dcff\"></i>replan where §3.5 changed the pick", "<i style=\"background:#00dcff\"></i>replan where §3.5 changed the pick (bottom: 0.1.4b; top, dim: 0.1.4)")
rep("pick without momentum, when it differs (0.1.4b)", "pick without momentum, when it differs")
rep("Left video = recorded 0.1.4 run, right = 0.1.4b (momentum on).", "Left video = recorded 0.1.4 run, right = 0.1.4b; §3.5 momentum is on in both.")
rep("On 0.1.4b the proposals are shaded by the recalibrated score.", "The proposals are shaded by the recalibrated score.")
rep('<div class="lbl">0.1.4b — momentum on</div>', '<div class="lbl">0.1.4b — S10 generator + batch fix, same scorer</div>')
rep("<div class=\"m\">stall (S5 rule), 0.1.4b<b>${n.stall?'<span class=\"dn\">yes</span>':'no'}</b></div>", "<div class=\"m\">stall (S5 rule)<b>${o.stall?'yes':'no'} → ${n.stall?'<span class=\"dn\">yes</span>':'no'}</b></div>")
# ---- 3b. the page is served by `python -m http.server`, which does not answer Range requests, and Chrome cannot seek such a video.
# Each video is therefore fetched whole as a blob when its card comes near the viewport (about 1.5 MB each) and played from memory.
rep('<video preload="metadata" playsinline muted src="${r.video_old}"></video>', '<video preload="metadata" playsinline muted data-src="${r.video_old}"></video>')
rep('<video preload="metadata" playsinline muted src="${r.video_new}"></video>', '<video preload="metadata" playsinline muted data-src="${r.video_new}"></video>')
rep("const V=document.getElementById('videos');const GV={};", "const V=document.getElementById('videos');const GV={};\nconst ldv=v=>{if(v.dataset.l)return;v.dataset.l=1;fetch(v.dataset.src).then(r=>r.blob()).then(b=>{v.src=URL.createObjectURL(b);});};const _io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){_io.unobserve(e.target);ldv(e.target);}}),{rootMargin:'900px'});window.__loadAll=()=>document.querySelectorAll('video[data-src]').forEach(ldv);")
rep("const st={scene:r.scene,a,b,rng,playing:false,drift:0,pending:false};", "_io.observe(a);_io.observe(b);const st={scene:r.scene,a,b,rng,playing:false,drift:0,pending:false};")
# ---- 3c. tab "Models": every model's final score on the 10 pilot scenes and on the 30 (tables/models_compare.json, written by s017_models_table.py)
data["models"] = json.load(open(T / "models_compare.json")) if (T / "models_compare.json").is_file() else None
rep('<div class="tab" data-p="videos">Videos</div>', '<div class="tab" data-p="videos">Videos</div><div class="tab" data-p="models">Models</div>')
rep('<div class="panel" id="videos"></div>', '<div class="panel" id="videos"></div><div class="panel" id="models"></div>')
rep(".b3{color:var(--mut);font-size:12px}", ".b3{color:var(--mut);font-size:12px}table.mc{max-width:1200px}table.mc td,table.mc th{text-align:right;white-space:nowrap}table.mc td.k,table.mc th.k{text-align:left}table.mc td.k b{color:var(--fg)}table.mc tr.best td.v{color:var(--good);font-weight:600}table.mc td.na{color:var(--mut)}table.ms td,table.ms th{text-align:right;white-space:nowrap;font-family:ui-monospace,Menlo,monospace;font-size:12px}table.ms td.k,table.ms th.k{text-align:left}table.ms th{white-space:normal;max-width:90px;vertical-align:bottom;font-family:inherit}table.ms tr.sep td{border-top:2px solid var(--line)}")
rep("renderTable();\n</script></body></html>", """renderTable();
// ---- Tab "Models"
(()=>{const P=document.getElementById('models');const MC=D.models;if(!MC){P.innerHTML='<div class="note">No model comparison table was built.</div>';return;}
const R=MC.rows;const best=k=>Math.max(...R.filter(r=>r[k]!=null&&!/floor/i.test(r.name)).map(r=>r[k]));const bp=best('pilot'),ba=best('all30');
const cell=(r,k,b,n,of)=>r[k]==null?`<td class="na">${n?`not complete (${n} of ${of})`:'not run'}</td>`:`<td class="v"${r[k]===b?' style="color:var(--good);font-weight:600"':''}>${fmt(r[k],4)}</td>`;
P.innerHTML=`<div class="note">Final score of every model run on the shared 20 Hz referee. Same scenes, same referee settings, one run each. A mean is shown only where all scenes of that set are finished.</div>
<h2>All models: mean score on the 10 pilot scenes and on all 30</h2><div class="tw"><table class="mc"><thead><tr><th class="k">model</th><th>10 pilot scenes</th><th>20 extension scenes</th><th>all 30 scenes</th><th>scenes with any collision</th><th>scenes with an at-fault collision</th><th>scenes run</th></tr></thead><tbody>${R.map(r=>`<tr><td class="k"><b>${esc(r.name)}</b><br><span class="b3">${esc(r.note)}</span></td>${cell(r,'pilot',bp,r.n_pilot,10)}${r.n_ext===20?cell(r,'ext',null):`<td class="na">${r.n_ext?fmt(r.ext,4)+` (${r.n_ext} of 20 so far)`:'not run'}</td>`}${cell(r,'all30',ba,r.n_pilot+r.n_ext,30)}<td>${r.coll} of ${r.n_pilot+r.n_ext}</td><td>${r.at_fault} of ${r.n_pilot+r.n_ext}</td><td>${r.n_pilot+r.n_ext}</td></tr>`).join('')}</tbody></table></div>
<p class="sub">Green = best model in the column (the constant-velocity floor is not a model and is left out of that). Collision counts cover the scenes that model has run.</p>
<h2>Score per scene</h2><div class="tw"><table class="ms"><thead><tr><th class="k">scene</th><th class="k">set</th>${R.map(r=>`<th>${esc(r.name)}</th>`).join('')}</tr></thead><tbody>${[...MC.pilot.map(s=>[s,'pilot']),...MC.ext.map(s=>[s,'extension'])].map(([s,g],i)=>`<tr${i===MC.pilot.length?' class="sep"':''}><td class="k">${s}</td><td class="k">${g}</td>${R.map(r=>`<td${r.scores[s]==null?' class="na"':''}>${fmt(r.scores[s])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
<p class="sub">Sources: ${R.map(r=>`${esc(r.name)}: ${esc(r.src)}`).join(' · ')}</p>`;})();
</script></body></html>""")
# ---- 4. arm names: every remaining "0.1.4" is the LEFT arm (now 0.1.4b) and every "0.1.4b" the RIGHT arm (now 0.1.6)
t = t.replace("0.1.4b", "@@NEW@@").replace("0.1.4", "0.1.6").replace("@@NEW@@", "0.1.7").replace("@@B3@@", "0.1.4b")
if missing: t = t.replace('<div class="note" id="note"></div>', '<div class="note" id="note"></div><div class="note">Not shown yet (no finished 0.1.7 run, menus or video): ' + ", ".join(missing) + "</div>", 1)
(OUT / "index.html").write_text(t.replace("__DATA__", json.dumps(data, default=float)))
tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()); print(f"{OUT}: {N} scenes ({len(missing)} missing: {missing}), {tot / 2**20:.1f} MiB"); [print("  ", k) for k in key]
if disagree and not a.standin: print("GROUP vs SCORE-CHANGE disagreements (scene, by |delta| rule, in JSON):", disagree)
json.dump(dict(out=str(OUT), n=N, missing=missing, key=key, groups={g: [x["scene"] for x in rows if x["outcome"] == g] for g in ("improved", "decreased", "same")}, disagree=disagree, bytes=tot), open(T / "review_build.json", "w"), indent=1)
