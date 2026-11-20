"""Run 3B v2 training report page (run3b/CLAUDE.md section 6 items 1-7) from the run's eval_step*.json + train_log.jsonl.
One column group per report (step 2,000, then every pass). Numbers only; run 3A's numbers only where the spec quotes them.
    python report_v2.py RUN_DIR OUT_HTML"""
import glob
import html
import json
import os
import sys

RUN, OUT = sys.argv[1], sys.argv[2]
EV = sorted(glob.glob(os.path.join(RUN, "eval_step*.json")), key=lambda p: json.load(open(p))["step"])
EV = [json.load(open(p)) for p in EV]
LOG = [json.loads(x) for x in open(os.path.join(RUN, "train_log.jsonl"))]
LEN = json.load(open(os.path.join(RUN, "length.json")))
SET = json.load(open(os.path.join(RUN, "setup.json")))
E = lambda s: html.escape(str(s))


def f(x, d=3):
    return "–" if x is None else (f"{x:,.{d}f}" if isinstance(x, float) else f"{x:,}")


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def label(e):
    return f"step {e['step']:,}" + ("" if e["tag"] == "smoke" else f" · {e['tag'].replace('pass', 'pass ')}")


HEADS = [label(e) for e in EV]


def table(cols, rows, sub=None, first="", cls=""):
    """cols: list of group names; sub: list of sub-column names per group; rows: (name, [cells...]) cells flattened."""
    sub = sub or [""]
    h = f'<div class="tw"><table class="{cls}"><thead><tr><th rowspan="{2 if sub != [""] else 1}">{first}</th>'
    h += "".join(f'<th colspan="{len(sub)}" class="g">{E(c)}</th>' for c in cols) + "</tr>"
    if sub != [""]:
        h += "<tr>" + "".join(f'<th class="s">{E(s)}</th>' for _ in cols for s in sub) + "</tr>"
    h += "</thead><tbody>"
    for name, cells in rows:
        if name.startswith("§"):
            h += f'<tr class="sec"><td colspan="{1 + len(cols) * len(sub)}">{E(name[1:])}</td></tr>'
            continue
        h += f"<tr><td>{E(name)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    return h + "</tbody></table></div>"


# ---------- item 1: step time and every loss (mean over the 100 steps before each report)
def window(step, n=100):
    return [r for r in LOG if step - n <= r["step"] < step]


loss_keys = list(LOG[-1]["losses"].keys())
rows = []
wins = [("steps 0–49", [r for r in LOG if r["step"] < 50])] + [(label(e), window(e["step"])) for e in EV] + \
       [(f"latest (steps {LOG[-1]['step'] - 99:,}–{LOG[-1]['step']:,})", window(LOG[-1]["step"] + 1))]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def wall(w):
    return (w[-1]["time"] - w[0]["time"]) / (len(w) - 1) if len(w) > 1 else None


rows.append(("§Step time (s)", None))
rows.append(("wall time per step", [f(wall(w), 2) for _, w in wins]))
for k in ["loader_wait", "encoder", "heads", "label_wait", "loss_backward"]:
    rows.append((f"  {k.replace('_', ' ')}", [f(mean([r['timing'].get(k) for r in w]), 2) for _, w in wins]))
rows.append(("§Losses (weighted sum = total)", None))
rows.append(("total", [f(mean([r["total"] for r in w]), 3) for _, w in wins]))
for k in loss_keys:
    rows.append((f"{k}  (w {SET['weights'].get(k, '–')})", [f(mean([r['losses'].get(k) for r in w]), 4) for _, w in wins]))
rows.append(("§Other", None))
rows.append(("teacher-forcing share", [f(mean([r["tf_share"] for r in w]), 3) for _, w in wins]))
rows.append(("lr heads", [f"{mean([r['lr'][0] for r in w]):.2e}" for _, w in wins]))
rows.append(("label errors (sum)", [f(sum(r["label_errors"] for r in w)) for _, w in wins]))
rows.append(("GPU memory (GB)", [f(max(r["gpu_mem_gb"] for r in w), 1) for _, w in wins]))
item1 = table([n for n, _ in wins], rows, first="mean over the window")

# ---------- item 2: agents
RING = ["0-10 m", "10-20 m", "20-32 m", "32-46 m"]


def agent_rows(key, extra=()):
    keys = ["all"] + list(extra) + [f"ring {r}" for r in RING] + ["dir front", "dir beside", "dir behind", "dir left", "dir right",
                                                                  "class vehicle", "class pedestrian", "class bicycle"]
    out = []
    for k in keys:
        cells = []
        for e in EV:
            v = e["agents"][key].get(k)
            cells += [f(v["recall"]) if v else "–", f(v["precision"]) if v else "–"]
        v = EV[-1]["agents"][key].get(k)
        cells.append(f(v["n_true"]) if v else "–")
        out.append((k, cells))
    return out


def agent_table(key, extra=()):
    rs = agent_rows(key, extra)
    sub = ["recall", "precision"]
    h = f'<div class="tw"><table><thead><tr><th rowspan="2">{E(key)}</th>' + "".join(
        f'<th colspan="2" class="g">{E(c)}</th>' for c in HEADS) + '<th rowspan="2">n true</th></tr><tr>' + "".join(
        f'<th class="s">{s}</th>' for _ in HEADS for s in sub) + "</tr></thead><tbody>"
    for name, cells in rs:
        h += f"<tr><td>{E(name)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    return h + "</tbody></table></div>"


def ring_dir(key):
    rows = []
    for r in RING:
        for dname in ["front", "beside", "behind"]:
            k = f"ring {r} | dir {dname}"
            cells = []
            for e in EV:
                v = e["agents"][key].get(k)
                cells += [f(v["recall"]) if v else "–", f(v["precision"]) if v else "–"]
            v = EV[-1]["agents"][key].get(k)
            rows.append((k, cells + [f(v["n_true"]) if v else "–"]))
    h = f'<div class="tw"><table><thead><tr><th rowspan="2">ring × direction</th>' + "".join(
        f'<th colspan="2" class="g">{E(c)}</th>' for c in HEADS) + '<th rowspan="2">n true</th></tr><tr>' + "".join(
        f'<th class="s">{s}</th>' for _ in HEADS for s in ["recall", "precision"]) + "</tr></thead><tbody>"
    for name, cells in rows:
        h += f"<tr><td>{E(name)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    return h + "</tbody></table></div>"


Q = EV[-1]["quoted_run3a"]
qk = {k.split(" (")[0]: (k, v) for k, v in Q.items()}
rule = EV[-1]["agents_rule"]

# ---------- item 3: facts
YN = ["hit", "offroad", "left_route", "red", "stop", "wrong_way", "comfort"]
NUM = ["metres_sofar", "lane_off", "ttc", "cutin_brake"]


def facts_table(ds):
    sub = ["true agents", "decoded agents"]
    rows = [("§Yes / no heads: within-scene pairwise accuracy, all scored paths (expert + drawn paths + the planner's 32)", None)]
    for h in YN:
        k = f"{ds} | {h}"
        if k not in EV[-1]["facts"]["true agents"]:
            continue
        cells = [f(e["facts"][a][k]["pairwise_all"]) for e in EV for a in sub]
        v = EV[-1]["facts"]["true agents"][k]
        rows.append((f"{h}  (scenes {v['scenes_all']:,}, pairs {v['pairs_all']:,}, base rate {v['base_rate']:.3f})", cells))
    rows.append(("§Yes / no heads: within-scene pairwise accuracy, the 32 proposals only", None))
    for h in YN:
        k = f"{ds} | {h}"
        if k not in EV[-1]["facts"]["true agents"]:
            continue
        cells = [f(e["facts"][a][k]["pairwise_planner32"]) for e in EV for a in sub]
        v = EV[-1]["facts"]["true agents"][k]
        rows.append((f"{h}  (scenes {v['scenes_planner32']:,}, pairs {v['pairs_planner32']:,})", cells))
    rows.append(("§Number heads: abs error per waypoint, mean (p50 / p90)", None))
    for h in NUM:
        k = f"{ds} | {h} (abs err per waypoint)"
        cells = []
        for e in EV:
            for a in sub:
                v = e["facts"][a][k]
                cells.append(f"{v['mean']:.3f} <span class=q>({v['p50']:.2f} / {v['p90']:.2f})</span>")
        rows.append((f"{h}  (n {EV[-1]['facts']['true agents'][k]['n']:,})", cells))
    return table(HEADS, rows, sub, first=ds)


# ---------- item 4: expert metres head
def metres_table():
    sub = ["true agents", "decoded agents"]
    rows = []
    for g in ["NAVSIM", "LEAD", "LEAD expert stops", "LEAD rest"]:
        n = EV[-1]["expert_metres_head"]["true agents"][g]["err_2s"]["n"]
        rows.append((f"§{g}  (n {n:,})", None))
        for t in ["err_2s", "err_4s"]:
            cells = []
            for e in EV:
                for a in sub:
                    v = e["expert_metres_head"][a][g][t]
                    cells.append(f"{v['mean']:.2f} <span class=q>({v['p50']:.2f} / {v['p90']:.2f})</span>")
            rows.append((f"abs error at {t[-2:]} (m), mean (p50 / p90)", cells))
    return table(HEADS, rows, sub, first="expert's-metres head")


# ---------- item 5: the pick
def pick_table():
    sub = ["true agents", "decoded agents"]
    rows = []
    for ds in ["NAVSIM", "LEAD"]:
        for m in ["expert metres true", "expert metres head"]:
            k = f"{ds} | {m}"
            v0 = EV[-1]["pick"]["true agents"][k]
            rows.append((f"§{ds}, pick with {m.replace('expert metres', 'expert’s metres')}  (n {v0['n']:,}; expert moves in {v0['n_expert_moves']:,})", None))
            rows.append(("true score: best of 32 minus pick, mean (p50 / p90)",
                         [f"{e['pick'][a][k]['gap_to_best_of_32']['mean']:.3f} <span class=q>({e['pick'][a][k]['gap_to_best_of_32']['p50']:.3f} / {e['pick'][a][k]['gap_to_best_of_32']['p90']:.3f})</span>"
                          for e in EV for a in sub]))
            rows.append(("share where the pick is a true top", [pct(e["pick"][a][k]["share_true_top"]) for e in EV for a in sub]))
            rows.append(("share of picks that stop while the expert moves", [pct(e["pick"][a][k]["share_stops_while_expert_moves"]) for e in EV for a in sub]))
    return table(HEADS, rows, sub, first="the pick")


# ---------- item 6: planner + EPDMS
def planner_table():
    sub = ["true agents", "decoded agents"]
    rows = []
    for ds in ["NAVSIM", "LEAD"]:
        rows.append((f"§{ds}  (n {EV[-1]['planner']['true agents'][ds]['min_dist']['n']:,})", None))
        for k, nm in [("min_dist", "nearest proposal to the expert (m), mean (p50 / p90)"), ("spread", "spread of the 32 (m), mean (p50 / p90)")]:
            rows.append((nm, [f"{e['planner'][a][ds][k]['mean']:.2f} <span class=q>({e['planner'][a][ds][k]['p50']:.2f} / {e['planner'][a][ds][k]['p90']:.2f})</span>"
                              for e in EV for a in sub]))
    rows.append((f"§NAVSIM held-out EPDMS  (n {EV[-1]['epdms_navsim']['true agents']['best of 32']['n']:,})", None))
    for k in ["pick (expert metres head)", "pick (expert metres true)", "best of 32", "mean of 32"]:
        rows.append((k.replace("expert metres", "expert’s metres"), [f(e["epdms_navsim"][a][k]["mean"], 4) for e in EV for a in sub]))
    return table(HEADS, rows, sub, first="planner")


def teachers_table():
    rows = []
    for k, nm in [("teachers_4", "4 teachers"), ("teachers_1_3", "1–3 teachers"), ("teachers_0", "0 teachers")]:
        rows.append((nm, [f"{e['lead_teachers'][k]:,} <span class=q>({100 * e['lead_teachers'][k] / e['lead_teachers']['frames']:.1f}%)</span>" for e in EV]))
    rows.append(("frames", [f(e["lead_teachers"]["frames"]) for e in EV]))
    return table(HEADS, rows, first="LEAD frames")


last = LOG[-1]
eta_s = wall(window(last["step"] + 1, 500)) * (LEN["total_steps"] - last["step"])
import datetime as _dt
eta = _dt.datetime.fromtimestamp(last["time"] + eta_s).strftime("%d %b %H:%M")
started = _dt.datetime.fromtimestamp(LOG[0]["time"]).strftime("%d %b %H:%M")
now = _dt.datetime.fromtimestamp(last["time"]).strftime("%d %b %H:%M")

r3a_lead = qk["agents LEAD run 3A epoch 13"][1]
r3a_nav = qk["agents NAVSIM run 3A epoch 13"][1]
r3a_ep = qk["EPDMS run 3A epoch 13 tuned pick"][1]
r3a_f = qk["facts run 3A heads NAVSIM"][1]

page = f"""<title>Run 3B v2 reports</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; each spec item is a section with its tables in their own scroll boxes */
:root {{
  --bg: #f6f7f8; --panel: #ffffff; --fg: #1b2127; --muted: #5d6873; --rule: #d9dee3; --accent: #1f6f8b; --sec: #eef2f5;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12171c; --panel: #192027; --fg: #e3e8ec; --muted: #94a0ab; --rule: #2c3640; --accent: #6cb6d1; --sec: #1f2830; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #12171c; --panel: #192027; --fg: #e3e8ec; --muted: #94a0ab; --rule: #2c3640; --accent: #6cb6d1; --sec: #1f2830; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 14px; line-height: 1.5; }}
main {{ max-width: 1180px; margin: 0 auto; padding-block: 28px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 28px; }}
h1 {{ font-size: 22px; font-weight: 600; margin: 0; text-wrap: balance; }}
h2 {{ font-size: 16px; font-weight: 600; margin: 0 0 6px; color: var(--accent); }}
h3 {{ font-size: 13px; font-weight: 600; margin: 14px 0 6px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
p {{ margin: 0 0 8px; max-width: 75ch; }}
.meta {{ color: var(--muted); }}
.kv {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
.kv div {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; min-width: 0; }}
.kv b {{ display: block; font-family: var(--mono); font-size: 18px; font-weight: 500; font-variant-numeric: tabular-nums; }}
.kv span {{ color: var(--muted); font-size: 12px; }}
section {{ min-width: 0; }}
.tw {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; margin: 6px 0 4px; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }}
th:first-child, td:first-child {{ text-align: left; }}
td {{ font-family: var(--mono); font-size: 12.5px; }}
td:first-child {{ font-family: var(--sans); font-size: 13px; }}
th {{ font-weight: 600; font-size: 12px; color: var(--muted); background: var(--sec); }}
th.g {{ color: var(--fg); text-align: center; border-left: 1px solid var(--rule); }}
tr.sec td {{ background: var(--sec); font-family: var(--sans); font-weight: 600; font-size: 12px; color: var(--muted); }}
.q {{ color: var(--muted); }}
.quote {{ border-left: 3px solid var(--accent); padding: 4px 10px; background: var(--panel); color: var(--fg); max-width: none; }}
details summary {{ cursor: pointer; color: var(--accent); font-size: 13px; margin-top: 6px; }}
details summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<main>
<header>
  <h1>Run 3B v2 — run3b_v2b_19nov</h1>
  <p class="meta">Spec run3b/CLAUDE.md section 6, items 1–7. Started {started}; page built from the run's files at {now}. Every report is on
  {EV[-1]['n_nav']:,} NAVSIM + {EV[-1]['n_lead']:,} LEAD held-out samples (of {SET['n_heldout_nav']:,} + {SET['n_heldout_lead']:,}). Run 3A numbers appear only as the spec quotes them, each with its own n.</p>
</header>
<div class="kv">
  <div><b>{last['step']:,} / {LEN['total_steps']:,}</b><span>step / total ({LEN['passes']:.2f} passes of {SET['pass_steps']:,} steps)</span></div>
  <div><b>{wall(window(last['step'] + 1, 500)):.2f} s</b><span>wall time per step, last 500 steps (length was set from {LEN['measured_step_s']:.2f} s at step 49)</span></div>
  <div><b>{eta}</b><span>end at the last-500-step rate, reports excluded</span></div>
  <div><b>{sum(r['label_errors'] for r in LOG):,}</b><span>label errors, all steps</span></div>
</div>

<section><h2>1 · Step time and every loss</h2>
<p class="meta">Means over the 100 steps before each report (the first column is steps 0–49). Batch 64 = 32 NAVSIM + 32 LEAD.</p>
{item1}</section>

<section><h2>2 · Agents: recall / precision by range ring and direction</h2>
<p class="meta">Decoder's last layer, all slots; a detection = 1 − P(none) ≥ {rule['conf']}, class = argmax; greedy by confidence, a hit when the nearest still-free true agent of the same class is within {rule['hit_m']} m (x-y centre). Ring by the true agent's range (recall) or the detection's (precision); front |bearing| ≤ 45°, behind ≥ 135°. Classes {', '.join(rule['classes'])}.
'as Bench2Drive' = run 3A's LEAD key (|x|, |y| ≤ 32 m, above ground); 'camera-visible' = also inside a camera view with visible pixels.</p>
<p class="quote">Spec quotes for run 3A: LEAD {r3a_lead['all recall']:.3f} recall / {r3a_lead['all precision']:.3f} precision overall, {r3a_lead['ring 10-20 m recall']:.3f} at 10–20 m (4,990 trial frames, n true {r3a_lead['n_true']:,}); NAVSIM {r3a_nav['ring 10-20 m recall']:.2f} at 10–20 m (4,457 held-out samples, n true {r3a_nav['n_true']:,}).</p>
<h3>NAVSIM</h3>{agent_table('NAVSIM')}
<details><summary>NAVSIM ring × direction</summary>{ring_dir('NAVSIM')}</details>
<h3>LEAD as Bench2Drive</h3>{agent_table('LEAD as Bench2Drive', ('normal | all', 'shifted | all'))}
<details><summary>LEAD as Bench2Drive ring × direction</summary>{ring_dir('LEAD as Bench2Drive')}</details>
<h3>LEAD camera-visible</h3>{agent_table('LEAD camera-visible', ('normal | all', 'shifted | all'))}
<details><summary>LEAD camera-visible ring × direction</summary>{ring_dir('LEAD camera-visible')}</details>
</section>

<section><h2>3 · Facts</h2>
<p class="meta">Pairwise accuracy: within a scene, every (yes, no) pair of paths; the share where the yes path gets the higher predicted probability (ties count half); mean over scenes that have both. Path label = the fact at any step; prediction = max over steps. Red leaves out paths of unknown colour; comfort is NAVSIM only. Number heads: abs error per waypoint (ttc clipped to [0, 4] s, cut-in braking to [0, 8]). 'true agents' = the looks are fed the true agents (no noise); 'decoded agents' = the decoder's agents.</p>
<p class="quote">Spec quote, run 3A's heads on NAVSIM: collision {r3a_f['collision']}, drivable {r3a_f['drivable']}, direction {r3a_f['direction']}, light {r3a_f['light']}, progress {r3a_f['progress']}.</p>
{facts_table('NAVSIM')}
{facts_table('LEAD')}
</section>

<section><h2>4 · Expert's-metres head</h2>
<p class="meta">|head − true expert metres so far| at 2 s (step 4) and 4 s (step 8). LEAD 'expert stops' = the expert gains under 0.25 m in its last 0.5 s step.</p>
{metres_table()}</section>

<section><h2>5 · The pick</h2>
<p class="meta">Pick = the proposal with the best score from the predicted facts; its true score uses live facts and the true expert metres. 'True top' = true score ≥ best of 32 − 1e-6. 'Stops' = under 0.5 m in 4 s, counted among samples whose expert moves at least 0.5 m.</p>
{pick_table()}</section>

<section><h2>6 · Planner</h2>
<p class="meta">Distance = mean over the 8 waypoints of the x-y distance; spread = mean pairwise distance of the 32.</p>
<p class="quote">Spec quote, run 3A on NAVSIM held-out (4,457 samples): pick EPDMS {r3a_ep['pick']:.3f}, best of 32 {r3a_ep['best of 32']:.3f}.</p>
{planner_table()}</section>

<section><h2>7 · LEAD teachers</h2>
{teachers_table()}</section>
</main>
"""
open(OUT, "w").write(page)
print("wrote", OUT, len(page))
