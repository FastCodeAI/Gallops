"""Load the rebuilt page headless: console errors, metrics tab per block, chart sizes, replay filter, scene link, video sync."""
import json, sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1]; SHOT = sys.argv[2]
errs, res = [], {}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL + "#metrics", wait_until="domcontentloaded"); pg.wait_for_timeout(1500)
    for blk in ("18", "12", "30"):
        pg.click(f"#mblk button[data-b='{blk}']"); pg.wait_for_timeout(300)
        res[blk] = pg.evaluate("""()=>({verdict: document.querySelector('.verdict .big').textContent,
            tiles: [...document.querySelectorAll('.tile .v')].map(x=>x.textContent),
            gap_bars: document.querySelectorAll('#v_metrics .card:nth-of-type(4) svg g').length,
            svgs: [...document.querySelectorAll('#v_metrics svg')].map(s=>[Math.round(s.getBoundingClientRect().width), Math.round(s.getBoundingClientRect().height)]),
            rows: document.querySelectorAll('#v_metrics table.m')[1].rows.length - 1,
            hscroll: document.documentElement.scrollWidth > innerWidth})""")
        pg.screenshot(path=f"{SHOT}/metrics_{blk}.png", full_page=True)
    pg.click("#mblk button[data-b='18']"); pg.wait_for_timeout(200)
    # hover a gap bar -> tooltip
    g = pg.locator("#v_metrics svg g").first; g.hover(); pg.wait_for_timeout(200)
    res["tooltip"] = pg.evaluate("document.getElementById('tip').style.display + ' | ' + document.getElementById('tip').textContent")
    # scene link -> replays tab
    pg.click("a.sc[data-s='0245ff75']"); pg.wait_for_timeout(1500)
    res["after_link"] = pg.evaluate("""()=>({view: document.querySelector('.view.on').id, hash: location.hash,
        tags: document.getElementById('s_0245ff75').querySelector('h2').textContent,
        cols: [...document.getElementById('s_0245ff75').querySelectorAll('.col h3')].map(h=>h.textContent)})""")
    pg.screenshot(path=f"{SHOT}/replay_0245ff75.png")
    for blk in ("18", "12", "all"):
        pg.click(f"#rblk button[data-b='{blk}']"); pg.wait_for_timeout(200)
        res[f"replay_visible_{blk}"] = pg.evaluate("[...document.querySelectorAll('section.scene')].filter(s=>s.style.display!=='none').length")
    # sync: seek 0245ff75 to carry crash (sim t = 2.95 + 15.75) and compare video times
    pg.evaluate("()=>{ const g = window.__cmp.groups.find(g=>g.sc.scene==='0245ff75'); window.__cmp.seek(g, 18.70, true); }"); pg.wait_for_timeout(2500)
    res["sync_0245ff75"] = pg.evaluate("window.__cmp.groups.find(g=>g.sc.scene==='0245ff75').vids.map(v=>[v.dataset.arm, v.currentTime, v.readyState])")
    pg.locator("#s_0245ff75").screenshot(path=f"{SHOT}/replay_0245ff75_crash.png")
    # phone width
    m = b.new_page(viewport={"width": 390, "height": 844}); m.goto(URL + "#metrics"); m.wait_for_timeout(1200)
    res["phone_hscroll"] = m.evaluate("document.documentElement.scrollWidth > innerWidth")
    m.screenshot(path=f"{SHOT}/metrics_phone.png")
    b.close()
res["errors"] = errs
print(json.dumps(res, indent=1))
