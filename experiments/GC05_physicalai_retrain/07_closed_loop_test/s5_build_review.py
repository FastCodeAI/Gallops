#!/usr/bin/env python
"""S5 TRIAL review page: recorded 0.1.4b (left) vs 0.1.6 (right), the same page as the 0.1.4 vs 0.1.4b review
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

HERE = pathlib.Path(__file__).resolve().parent; T = HERE / "tables"; B14 = HERE.parents[1] / "Re-ranker" / "baseline_014b"; ALP = HERE.parents[2]
sys.path.insert(0, str(B14)); import build_014b_site as BS  # noqa: E402   (TEMPLATE, arm_series; read-only)
BR, f = BS.BR, BS.f
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--standin", action="store_true"); ap.add_argument("--scenes", default=""); ap.add_argument("--tables", default=""); a = ap.parse_args()
if a.tables: T = pathlib.Path(a.tables)   # another tables folder (the one-scene proof writes a provisional one)
OUT = pathlib.Path(a.out) if a.out else HERE / ("review_standin" if a.standin else "review"); (OUT / "videos").mkdir(parents=True, exist_ok=True)
RUN16 = lambda s: ALP / "runs" / "drivejepa" / f"drivejepa_016_{s}"
rep14 = pd.read_csv(B14 / "tables" / "replans_014b.csv")
if a.standin:
    base = pd.read_csv(B14 / "tables" / "per_scene.csv"); keep = [c for c in base.columns if c.startswith("014b_")]
    per = pd.concat([base[["scene", "lane"]], base[keep], base[keep].rename(columns=lambda c: "016_" + c[5:])], axis=1).set_index("scene"); rep16 = rep14.copy()
    scenes = (a.scenes.split(",") if a.scenes else list(per.index)[:6])
    G = dict(improved=dict(title="Score improved", lead="STAND-IN", points=["stand-in"], scenes=[]), decreased=dict(title="Score decreased", lead="STAND-IN", points=["stand-in"], scenes=[]),
             same=dict(title="No change in score", lead="STAND-IN: right arm = the 0.1.4b run again.", points=["stand-in point"], buckets=[dict(title="Stand-in bucket A", note="first half", scenes=scenes[: len(scenes) // 2]), dict(title="Stand-in bucket B", note="second half", scenes=scenes[len(scenes) // 2:])]),
             why={s: f"stand-in sentence for {s}" for s in scenes})
else:
    per = pd.read_csv(T / "per_scene.csv").set_index("scene"); rep16 = pd.read_csv(T / "replans_016.csv"); G = json.load(open(T / "review_groups.json"))
    scenes = a.scenes.split(",") if a.scenes else list(per.index)
def src(s, arm):
    """run dir, menu dir, video file of one arm"""
    if arm == "014b" or a.standin: return BR.RUNS / f"b014b_{s}_on", B14 / "menus" / f"b014b_{s}", B14 / "videos" / f"b014b_{s}.mp4"
    return RUN16(s), HERE / "menus" / f"d016_{s}", HERE / "videos" / f"d016_{s}.mp4"
def ready(s): r, m, v = src(s, "016"); return (r / "aggregate" / "metrics_results.parquet").is_file() and (pathlib.Path(str(m) + ".json")).is_file() and v.is_file() and s in per.index and np.isfinite(per.loc[s].get("016_scene_score", np.nan))
missing = [s for s in scenes if not ready(s)]; scenes = [s for s in scenes if ready(s)]; assert scenes, f"no scene is ready (missing: {missing})"
BUCKETS = G["same"].get("buckets", []); BUCKET_OF = {s: i for i, b in enumerate(BUCKETS) for s in b["scenes"]}; GROUP_OF = {s: g for g in ("improved", "decreased") for s in G[g].get("scenes", [])} | {s: "same" for s in BUCKET_OF} | {s: "same" for s in G["same"].get("scenes", [])}
WHY = G.get("why", {}); rows = []; disagree = []
for s in scenes:
    r = per.loc[s]; ra, ma, va = src(s, "014b"); rb, mb, vb = src(s, "016"); A, pa = BS.arm_series(ra, ma, "014b"); Bs, pb = BS.arm_series(rb, mb, "016")
    def arm(pre, peak, rp):
        q = rp[rp.scene == s]
        return dict(score=f(r[f"{pre}_scene_score"], 3), progress=f(r[f"{pre}_progress_clipped_rel"], 3), at_fault=int(r[f"{pre}_collision_at_fault"] > 0), coll_any=int(r[f"{pre}_collision_any"] > 0), offroad=int(r[f"{pre}_offroad"] > 0),
                    dist_gt=f(r[f"{pre}_dist_to_gt_trajectory"], 2), dist=f(r[f"{pre}_dist_traveled_m"], 1), min_v=f(r[f"{pre}_min_speed_after_ho"], 2), slow_s=f(r[f"{pre}_longest_slow_s"], 1), slow_t=f(r[f"{pre}_slow_t_start"], 1),
                    headway=f(r[f"{pre}_min_headway_after_ho"], 2), exit_kind=str(r[f"{pre}_exit_kind"]), exit_s=f(r[f"{pre}_exit_s"], 2), peak_steer=f(peak, 3), stall=bool(r.get(f"{pre}_stall", False)) if str(r.get(f"{pre}_stall", "")) not in ("", "nan") else False,
                    changed_n=int(q.changed.sum()), replans=int(len(q)), changed_share=f(100 * float(q.changed.mean()) if len(q) else np.nan, 1))
    o, n = arm("014b", pa, rep14), arm("016", pb, rep16); delta = f(n["score"] - o["score"], 3); by = "improved" if delta >= 0.02 else ("decreased" if delta <= -0.02 else "same"); out = GROUP_OF.get(s, by)
    if out != by or (s not in GROUP_OF): disagree.append((s, by, GROUP_OF.get(s)))
    rows.append(dict(scene=s, lane=r.get("lane", ""), old=o, new=n, delta=delta, a=A, b=Bs, video_old=f"videos/b014b_{s}.mp4", video_new=f"videos/d016_{s}.mp4", score_old=o["score"], score_new=n["score"], changed_share=n["changed_share"], changed_old=o["changed_share"],
                     peak_old=o["peak_steer"], peak_new=n["peak_steer"], prog_old=o["progress"], prog_new=n["progress"], dist_old=o["dist"], dist_new=n["dist"], hw_old=o["headway"], hw_new=n["headway"], slow_old=o["slow_s"], slow_new=n["slow_s"],
                     outcome=out, why=WHY.get(s, ""), bucket=(BUCKET_OF.get(s, -1) if out == "same" else -1)))
    for v, name in ((va, f"b014b_{s}.mp4"), (vb, f"d016_{s}.mp4")):
        d = OUT / "videos" / name
        if not d.is_file() or d.stat().st_size != v.stat().st_size: shutil.copy2(v, d)
rows.sort(key=lambda x: ({"improved": 0, "decreased": 1, "same": 2}[x["outcome"]], x["bucket"] if x["bucket"] >= 0 else 99, -x["delta"], x["scene"]))
N = len(rows); so = [x["score_old"] for x in rows]; sn = [x["score_new"] for x in rows]; up = sum(x["outcome"] == "improved" for x in rows); dn = sum(x["outcome"] == "decreased" for x in rows)
co, ro = sum(x["old"]["changed_n"] for x in rows), sum(x["old"]["replans"] for x in rows); cn, rn = sum(x["new"]["changed_n"] for x in rows), sum(x["new"]["replans"] for x in rows)
key = G.get("key") or [f"Mean score over the {N} scenes: {sum(so) / N:.3f} (0.1.4b) → {sum(sn) / N:.3f} (0.1.6).",
       f"Score up on {up}, down on {dn}, unchanged on {N - up - dn} of the {N} scenes.",
       f"At-fault collisions: {sum(x['old']['at_fault'] for x in rows)} (0.1.4b) → {sum(x['new']['at_fault'] for x in rows)} (0.1.6); any collision {sum(x['old']['coll_any'] for x in rows)} → {sum(x['new']['coll_any'] for x in rows)}.",
       f"Off-road exits: {sum(x['old']['offroad'] for x in rows)} → {sum(x['new']['offroad'] for x in rows)}; scenes with minimum headway under 0.8 s: {sum((x['hw_old'] or 9) < 0.8 for x in rows)} → {sum((x['hw_new'] or 9) < 0.8 for x in rows)}.",
       f"§3.5 changed the pick on {co} of {ro} replans ({100 * co / max(ro, 1):.0f} %) under 0.1.4b and on {cn} of {rn} ({100 * cn / max(rn, 1):.0f} %) under 0.1.6."]
header = G.get("header") or ((f"All {N} census scenes, one closed-loop run each." if not missing else f"{N} of {N + len(missing)} scenes finished so far.") +
          " 0.1.4b is the recorded baseline (shipped weights + §3.5 momentum); 0.1.6 = 0.1.4b + the S4 pass-9 weights + bonnet mask + zeroed acceleration. Scenes are grouped by score change.")
data = dict(groups={k: dict(title=G[k]["title"], lead=G[k]["lead"], points=G[k]["points"]) for k in ("improved", "decreased", "same")}, buckets=[dict(title=b["title"], note=b.get("note", ""), n=sum(x["bucket"] == i for x in rows)) for i, b in enumerate(BUCKETS)],
            built=pd.Timestamp.now().isoformat(timespec="minutes"), header=header, rows=rows, key=key, n=N, rule=[], missing=missing, standin=bool(a.standin))
t = BS.TEMPLATE
def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:70]); t = t.replace(old, new)
# ---- 1. the structural changes of build_014b_artifact.py (groups, Insights first, buckets, extra table columns), re-applied on the same template
rep('<title>Drive-JEPA 0.1.4 vs 0.1.4b (paper §3.5) — 5 scenes</title>', '<title>Drive-JEPA 0.1.4 vs 0.1.4b — S5 trial</title>')
rep(':root{--bg:#111;', ':root{color-scheme:dark;--bg:#111;')
rep('<h1>Drive-JEPA adapter 0.1.4 (recorded) vs 0.1.4b (paper §3.5 momentum-aware selection) · 5 scenes · 20 Hz referee</h1>', f'<h1>Drive-JEPA adapter 0.1.4 (recorded) vs 0.1.4b (S4 pass-9 weights, bonnet mask, zeroed acceleration; §3.5 momentum on in both) · {N} scenes · 20 Hz referee</h1>')
rep('<div class="tab on" data-p="videos">Tab 1 — videos</div><div class="tab" data-p="metrics">Tab 2 — summary</div>', '<div class="tab on" data-p="metrics">Metrics</div><div class="tab" data-p="videos">Videos</div>')
rep('<div class="panel on" id="videos"></div><div class="panel" id="metrics"></div>', '<div class="panel on" id="metrics"></div><div class="panel" id="videos"></div>')
rep('''onclick="document.querySelector('.tab').click()"''', '''onclick="document.querySelector('.tab[data-p=videos]').click();showSub('${r.outcome}','vids')"''')
rep("scene order = rank under the rule", "scenes grouped by score change")
rep(".up{color:var(--good)}", ".sec{margin:22px 0 30px}.sec>h2{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px}.sub2{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto;justify-content:flex-end}.sub2 button{background:#161616;color:var(--mut);border:1px solid var(--line);border-radius:14px;padding:4px 14px;font-size:13px;cursor:pointer}.sub2 button.on{background:#2b3a4d;color:var(--fg);border-color:#4a6a8f}.sub2 button:focus-visible{outline:2px solid var(--acc)}.ins{max-width:980px}.ins .lead{font-size:15px;color:var(--fg);margin:6px 0 10px}.ins ul{margin:0 0 14px 18px;padding:0;font-size:14px}.ins li{margin:5px 0}.ins table td{white-space:normal;text-align:left;vertical-align:top}.ins table td.k{white-space:nowrap}.ins table td.sc{white-space:nowrap;font-family:ui-monospace,Menlo,monospace}.bkt{margin:18px 0 26px;padding-top:4px;border-top:1px solid var(--line)}.bkt h3{margin:10px 0 2px;font-size:14px;color:var(--fg)}tr.bh td{background:#161616;color:var(--acc);font-weight:600;text-align:left}.why1{font-size:13px;color:#cfcfcf;margin-top:10px}.up{color:var(--good)}")
rep("// ---- Tab 1\nconst V=document.getElementById('videos');", """// ---- Tab 1
const V=document.getElementById('videos');const GV={};
window.showSub=(o,w)=>{const s=document.getElementById('sec-'+o);if(!s)return;s.querySelectorAll('.sub2 button').forEach(b=>b.classList.toggle('on',b.dataset.w===w));s.querySelector('.ins').hidden=(w==='ins')?false:true;s.querySelector('.vids').hidden=(w==='ins');s.querySelectorAll('.bkt').forEach(d=>{d.hidden=(w!=='vids'&&d.id!=='bkt-'+w.slice(1));});};
for(const o of ['improved','decreased','same']){const rows=D.rows.filter(r=>r.outcome===o);const G=D.groups[o];const s=document.createElement('div');s.className='sec';s.id='sec-'+o;
 s.innerHTML=`<h2>${esc(G.title)} · ${rows.length} scene${rows.length===1?'':'s'}<span class="sub2"><button data-w="ins" class="on">Insights</button><button data-w="vids">${o==='same'?'All videos':'Videos'} (${rows.length})</button>${o==='same'?D.buckets.map((b,i)=>b.n?`<button data-w="b${i}" title="${esc(b.note)}">${esc(b.title)} (${b.n})</button>`:'').join(''):''}</span></h2>
 <div class="ins"><p class="lead">${esc(G.lead)}</p><ul>${G.points.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><div class="tw"><table class="gt" data-g="${o}"><thead><tr><th>scene</th><th>score 0.1.4 → 0.1.4b</th><th>progress</th><th>at-fault</th><th>any collision</th><th>off-road</th><th>peak steer (rad)</th><th>§3.5 changed picks</th><th style="text-align:left">what the runs show</th></tr></thead><tbody>${rows.map((r,j)=>`${(r.bucket>=0&&(j===0||rows[j-1].bucket!==r.bucket))?`<tr class="bh"><td colspan="9">${esc(D.buckets[r.bucket].title)} · ${rows.filter(q=>q.bucket===r.bucket).length}</td></tr>`:''}<tr><td class="k"><a href="#card-${r.scene}" style="color:#8cc8ff" onclick="showSub('${o}','vids')">${r.scene}</a></td><td class="sc">${fmt(r.score_old)} → ${fmt(r.score_new)}</td><td class="sc">${fmt(r.prog_old)} → ${fmt(r.prog_new)}</td><td class="sc">${r.old.at_fault} → ${r.new.at_fault}</td><td class="sc">${r.old.coll_any} → ${r.new.coll_any}</td><td class="sc">${r.old.offroad} → ${r.new.offroad}</td><td class="sc">${fmt(r.peak_old,3)} → ${fmt(r.peak_new,3)}</td><td class="sc">${r.old.changed_n} → ${r.new.changed_n}</td><td>${esc(r.why)}</td></tr>`).join('')}</tbody></table></div></div><div class="vids" hidden></div>`;
 V.appendChild(s);GV[o]=s.querySelector('.vids');
 if(o==='same'&&rows.length&&rows.every(r=>r.bucket>=0)){D.buckets.forEach((b,i)=>{const n=rows.filter(r=>r.bucket===i).length;if(!n)return;const d=document.createElement('div');d.className='bkt';d.id='bkt-'+i;d.innerHTML=`<h3>${esc(b.title)} · ${n} scene${n>1?'s':''}</h3><p class="sub">${esc(b.note)}</p>`;GV[o].appendChild(d);GV[o+'|'+i]=d;});}s.querySelectorAll('.sub2 button').forEach(b=>b.onclick=()=>showSub(o,b.dataset.w));}""")
rep(" V.appendChild(card);const [a,b]=card.querySelectorAll('video');", " (GV[r.outcome+'|'+r.bucket]||GV[r.outcome]).appendChild(card);const [a,b]=card.querySelectorAll('video');")
rep("<div class=\"strip\">${mrow('score',", "<div class=\"why1\">${esc(r.why)}</div><div class=\"strip\">${mrow('score',")
rep("One play / scrub control per pair drives both videos.", "Scenes are grouped by outcome; each group opens on its Insights (with the per-scene table), the videos are one click away. One play / scrub control per pair drives both videos.")
rep("let sortKey='rank',sortDir=1;", "let sortKey='delta',sortDir=-1;")
rep("['changed_share','§3.5 changed pick (% of replans)'],", "['prog','progress'],['coll','any collision'],['off','off-road'],['dist','distance travelled (m)'],['slow','longest < 1 m/s (s)'],['hw','min headway (s)'],['changed_old','§3.5 changed pick, 0.1.4 (% of replans)'],['changed_share','§3.5 changed pick, 0.1.4b (% of replans)'],")
rep(" default:return `<td>${r[k]==null?'–':esc(r[k])}</td>`;}}", " case 'prog':return `<td>${fmt(r.prog_old)} → ${fmt(r.prog_new)}</td>`;case 'coll':return `<td>${r.old.coll_any} → ${r.new.coll_any}</td>`;case 'off':return `<td>${r.old.offroad} → ${r.new.offroad}</td>`;case 'dist':return `<td>${fmt(r.dist_old,1)} → ${fmt(r.dist_new,1)}</td>`;case 'slow':return `<td>${fmt(r.slow_old,1)} → ${fmt(r.slow_new,1)}</td>`;case 'hw':return `<td>${fmt(r.hw_old)} → ${fmt(r.hw_new)}</td>`;case 'changed_old':return `<td>${fmt(r.changed_old,1)}</td>`;\n default:return `<td>${r[k]==null?'–':esc(r[k])}</td>`;}}")
rep("<h2>What the five runs show</h2>", f"<h2>What the {N} runs show</h2>"); rep("<h2>The five scenes ", f"<h2>The {N} scenes, 0.1.4 → 0.1.4b ")
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
rep('<div class="lbl">0.1.4b — momentum on</div>', '<div class="lbl">0.1.4b — S4 pass-9 weights, bonnet mask, zeroed acceleration</div>')
rep("<div class=\"m\">stall (S5 rule), 0.1.4b<b>${n.stall?'<span class=\"dn\">yes</span>':'no'}</b></div>", "<div class=\"m\">stall (S5 rule)<b>${o.stall?'yes':'no'} → ${n.stall?'<span class=\"dn\">yes</span>':'no'}</b></div>")
# ---- 3b. the page is served by `python -m http.server`, which does not answer Range requests, and Chrome cannot seek such a video.
# Each video is therefore fetched whole as a blob when its card comes near the viewport (about 1.5 MB each) and played from memory.
rep('<video preload="metadata" playsinline muted src="${r.video_old}"></video>', '<video preload="metadata" playsinline muted data-src="${r.video_old}"></video>')
rep('<video preload="metadata" playsinline muted src="${r.video_new}"></video>', '<video preload="metadata" playsinline muted data-src="${r.video_new}"></video>')
rep("const V=document.getElementById('videos');const GV={};", "const V=document.getElementById('videos');const GV={};\nconst ldv=v=>{if(v.dataset.l)return;v.dataset.l=1;fetch(v.dataset.src).then(r=>r.blob()).then(b=>{v.src=URL.createObjectURL(b);});};const _io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){_io.unobserve(e.target);ldv(e.target);}}),{rootMargin:'900px'});window.__loadAll=()=>document.querySelectorAll('video[data-src]').forEach(ldv);")
rep("const st={scene:r.scene,a,b,rng,playing:false,drift:0,pending:false};", "_io.observe(a);_io.observe(b);const st={scene:r.scene,a,b,rng,playing:false,drift:0,pending:false};")
# ---- 4. arm names: every remaining "0.1.4" is the LEFT arm (now 0.1.4b) and every "0.1.4b" the RIGHT arm (now 0.1.6)
t = t.replace("0.1.4b", "@@NEW@@").replace("0.1.4", "0.1.4b").replace("@@NEW@@", "0.1.6")
if missing: t = t.replace('<div class="note" id="note"></div>', '<div class="note" id="note"></div><div class="note">Not shown yet (no finished 0.1.6 run, menus or video): ' + ", ".join(missing) + "</div>", 1)
(OUT / "index.html").write_text(t.replace("__DATA__", json.dumps(data, default=float)))
tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()); print(f"{OUT}: {N} scenes ({len(missing)} missing: {missing}), {tot / 2**20:.1f} MiB"); [print("  ", k) for k in key]
if disagree and not a.standin: print("GROUP vs SCORE-CHANGE disagreements (scene, by |delta| rule, in JSON):", disagree)
json.dump(dict(out=str(OUT), n=N, missing=missing, key=key, groups={g: [x["scene"] for x in rows if x["outcome"] == g] for g in ("improved", "decreased", "same")}, disagree=disagree, bytes=tot), open(T / ("review_build_standin.json" if a.standin else "review_build.json"), "w") if not a.tables else open(T / "review_build.json", "w"), indent=1)
