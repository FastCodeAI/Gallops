#!/usr/bin/env python
"""S5 TRIAL review page headless check (after baseline_014b/check_014b_site.py; Playwright Chromium). Serves the given folder itself with
the SAME server the page will be served with (`python -m http.server`, no Range support) on 127.0.0.1, then checks: both tabs render,
the three groups and their Insights tables exist, every video loads, a scrub keeps each pair within one frame, a pair plays in sync,
the summary table sorts, the plots have lines, no page errors. Prints a JSON verdict; screenshots into logs/.
  PLAYWRIGHT_BROWSERS_PATH=<scratch>/pw-browsers <scratch>/pw/bin/python s5_check_review.py [DIR=review] [--url URL]
"""
import json, os, pathlib, socket, subprocess, sys, time
from playwright.sync_api import sync_playwright
HERE = pathlib.Path(__file__).resolve().parent; args = [x for x in sys.argv[1:] if not x.startswith("--")]; d = pathlib.Path(args[0]) if args else HERE / "review"; srv = None
if "--url" in sys.argv: url = sys.argv[sys.argv.index("--url") + 1]
else:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(d)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5); url = f"http://127.0.0.1:{port}/"
out = {"url": url, "dir": str(d), "failures": []}
try:
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1500, "height": 1000}); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url, wait_until="load"); pg.wait_for_function("window.__s5 && window.__s5.cards.length > 0", timeout=60000)
        n = pg.evaluate("window.__s5.data.n"); ncards = pg.evaluate("window.__s5.cards.length"); out.update(n=n, cards=ncards, title=pg.title(), h1=pg.evaluate("document.querySelector('h1').textContent"))
        out["opens_on"] = pg.evaluate("document.querySelector('.panel.on').id"); out["key_bullets"] = pg.evaluate("document.querySelectorAll('ul.key li').length")
        if ncards != n: out["failures"].append(f"cards {ncards} != scenes {n}")
        # summary table sorts
        out["tab2_tables"] = pg.evaluate("document.querySelectorAll('#tables table').length"); pg.click("#tables th[data-k=score_new]"); col = pg.evaluate("Array.from(document.querySelectorAll('#tables thead th')).findIndex(th=>th.dataset.k==='score_new')+1")
        get = lambda: pg.evaluate(f"Array.from(document.querySelectorAll('#tables table tbody tr td:nth-child({col})')).map(td=>parseFloat(td.textContent))")
        v1 = get(); pg.click("#tables th[data-k=score_new]"); v2 = get(); asc = all(v1[i] <= v1[i + 1] for i in range(len(v1) - 1)); desc = all(v2[i] >= v2[i + 1] for i in range(len(v2) - 1)); out["sort"] = dict(rows=len(v1), ascending_ok=asc, descending_ok=desc)
        if not (asc and desc and len(v1) == n): out["failures"].append("summary table sort wrong")
        # videos tab: groups open on Insights, videos one click away
        pg.click(".tab[data-p=videos]"); grp = pg.evaluate("Array.from(document.querySelectorAll('.sec')).map(s=>({id:s.id,ins_visible:!s.querySelector('.ins').hidden,vids_hidden:s.querySelector('.vids').hidden,table_rows:s.querySelectorAll('table.gt tbody tr:not(.bh)').length,buttons:Array.from(s.querySelectorAll('.sub2 button')).map(b=>b.textContent),cards:s.querySelectorAll('.card').length}))")
        out["groups"] = grp
        if len(grp) != 3 or not all(g["ins_visible"] and g["vids_hidden"] for g in grp): out["failures"].append("groups do not open on Insights")
        if sum(g["table_rows"] for g in grp) != n or sum(g["cards"] for g in grp) != n: out["failures"].append("per-group tables / cards do not cover every scene")
        for o in ("improved", "decreased", "same"): pg.evaluate("(o)=>showSub(o,'vids')", o)
        out["cards_visible_after_click"] = pg.evaluate("Array.from(document.querySelectorAll('#videos .card')).filter(c=>c.offsetParent!==null).length")
        if out["cards_visible_after_click"] != n: out["failures"].append("not every card is visible after opening the videos")
        pg.evaluate("window.__loadAll()"); t0 = time.time(); pg.wait_for_function("Array.from(document.querySelectorAll('#videos video')).every(v=>v.readyState>=1 && v.duration>0)", timeout=300000)
        vids = pg.evaluate("Array.from(document.querySelectorAll('#videos video')).map(v=>({src:v.dataset.src,dur:v.duration,seekable:v.seekable.length?v.seekable.end(0):0}))"); out["videos_loaded"] = sum(1 for v in vids if v["dur"] > 0); out["videos_total"] = len(vids); out["load_secs"] = round(time.time() - t0, 1)
        out["videos_not_seekable"] = [v["src"] for v in vids if v["seekable"] < v["dur"] - 0.2]
        if out["videos_loaded"] != 2 * n: out["failures"].append(f"videos loaded {out['videos_loaded']} of {2 * n}")
        worst = 0.0; bad = []; stuck = []
        for i in range(ncards):
            for frac in (0.4, 0.7):
                pg.evaluate("([i,f])=>{const c=window.__s5.cards[i];c.rng.value=Math.round(f*1000);c.rng.dispatchEvent(new Event('input'));}", [i, frac])
                try: pg.wait_for_function("(i)=>{const c=window.__s5.cards[i];return !c.a.seeking && !c.b.seeking;}", arg=i, timeout=30000)
                except Exception: stuck.append(i); continue
                r = pg.evaluate("(i)=>{const c=window.__s5.cards[i];return [c.a.currentTime,c.b.currentTime,Math.max(c.a.duration,c.b.duration),c.scene];}", i); dv = abs(r[0] - r[1]); worst = max(worst, dv)
                if dv > 0.05 or abs(r[0] - frac * r[2]) > 0.2: bad.append(r[3])
        out["scrub_worst_diff_s"] = round(worst, 4); out["scrub_pairs_ok"] = ncards - len(set(bad)) - len(set(stuck))
        if bad or stuck: out["failures"].append(f"scrub failed on {sorted(set(bad))} (stuck: {sorted(set(stuck))})")
        pg.evaluate("()=>{const c=window.__s5.cards[0];c.rng.value=0;c.rng.dispatchEvent(new Event('input'));document.querySelector('#videos .card .pp').click();}"); time.sleep(2.5)
        r = pg.evaluate("()=>{const c=window.__s5.cards[0];document.querySelector('#videos .card .pp').click();return [c.a.currentTime,c.b.currentTime];}"); out["play_advanced_s"] = round(r[0], 2); out["play_drift_s"] = round(abs(r[0] - r[1]), 4)
        if r[0] < 0.5 or abs(r[0] - r[1]) > 0.1: out["failures"].append("pair did not play in sync")
        out["plots"] = pg.evaluate("document.querySelectorAll('#videos .plot svg').length"); out["plot_paths_nonempty"] = pg.evaluate("Array.from(document.querySelectorAll('#videos .plot svg path')).filter(p=>p.getAttribute('d').length>20).length"); out["metric_tiles"] = pg.evaluate("document.querySelectorAll('#videos .strip .m').length")
        if out["plots"] != 3 * n: out["failures"].append(f"plots {out['plots']} != {3 * n}")
        if out["plot_paths_nonempty"] < 5 * n: out["failures"].append(f"non-empty plot lines {out['plot_paths_nonempty']} < {5 * n}")
        out["old_arm_names_left"] = pg.evaluate("(document.body.innerText.match(/0\\.1\\.4(?!b)/g)||[]).length")
        if out["old_arm_names_left"]: out["failures"].append("the page still names 0.1.4 (without b) somewhere")
        out["page_errors"] = errs
        if errs: out["failures"].append(f"page errors: {errs}")
        (HERE / "logs").mkdir(exist_ok=True); pg.screenshot(path=str(HERE / "logs" / f"review_check_videos_{d.name}.png")); pg.click(".tab[data-p=metrics]"); pg.screenshot(path=str(HERE / "logs" / f"review_check_metrics_{d.name}.png"))
        pg.click(".tab[data-p=videos]"); pg.evaluate("()=>{for(const o of ['improved','decreased','same'])showSub(o,'ins');}"); pg.screenshot(path=str(HERE / "logs" / f"review_check_insights_{d.name}.png")); b.close()
finally:
    if srv: srv.terminate()
out["ok"] = not out["failures"]; print(json.dumps(out, indent=1))
