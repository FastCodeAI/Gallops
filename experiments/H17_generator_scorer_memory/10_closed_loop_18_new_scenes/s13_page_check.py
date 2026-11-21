#!/usr/bin/env python
"""Load the compare page in headless Chromium and check it.

  1. page screenshot (top of the page, reset column on)
  2. scene screenshots paused at a given sim time, reset column on (both bars visible: the one burnt into each video and the HTML one)
  3. sync check, per scene, 3 points: (a) seek to handover, (b) seek to the first event in any arm (else handover + 6 s),
     (c) play from (b) at 1x for 2.0 s wall time, then pause. At each point every video's decoded frame is drawn to a canvas at native
     size and the burnt-in "t X.XXX s / tick k" region is cropped; currentTime of every video is recorded. Crops are montaged per
     point for reading by eye (no OCR on shakaal).

    <playwright venv>/bin/python s13_page_check.py <url> <outdir>
"""
import base64, io, json, pathlib, sys

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

URL, OUTD = sys.argv[1], pathlib.Path(sys.argv[2]); OUTD.mkdir(parents=True, exist_ok=True)
SHOTS = [("bae463d3", None), ("04749bb9", None)]   # time filled below (reset collision of 04749bb9; see report for bae463d3)
CROP = (20, 1008, 330, 1062)   # x0, y0, x1, y1 in the 900-wide frame: "t X.XXX s" and "tick k/n"

JS_WAIT = """async (sid) => { const g = window.__cmp.groups.find(g => g.sc.scene === sid);
  await Promise.all(g.vids.map(v => new Promise(res => { const ok = () => v.readyState >= 2 && !v.seeking;
     if (ok()) return res(); const f = () => { if (ok()) { v.removeEventListener('seeked', f); v.removeEventListener('canplay', f); res(); } };
     v.addEventListener('seeked', f); v.addEventListener('canplay', f); setTimeout(res, 15000); })));
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); return true; }"""
JS_GRAB = """(args) => { const [sid, crop] = args; const g = window.__cmp.groups.find(g => g.sc.scene === sid);
  return g.vids.map(v => { const c = document.createElement('canvas'); c.width = crop[2]-crop[0]; c.height = crop[3]-crop[1];
    c.getContext('2d').drawImage(v, crop[0], crop[1], c.width, c.height, 0, 0, c.width, c.height);
    return {arm: v.dataset.arm, ct: v.currentTime, dur: v.duration, paused: v.paused, png: c.toDataURL('image/png')}; }); }"""


def grab(page, sid):
    page.evaluate(JS_WAIT, sid); return page.evaluate(JS_GRAB, [sid, list(CROP)])


def main():
    data = json.load(open("{experiment_store}/compare_page/data.json"))["scenes"]
    res = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        page = b.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(URL, wait_until="networkidle"); page.check("#showReset"); page.wait_for_timeout(1500)
        page.screenshot(path=str(OUTD / "compare_page_top.png"))
        for sc in data:
            sid = sc["scene"]; arms = sc["arms"]; ho = next(iter(arms.values()))["handover_t"]
            ev = sorted(e["t"] for a in arms.values() for e in a["events"])
            tb = round((ev[0] if ev else ho + 6.0) / 0.05) * 0.05
            page.locator(f"#s_{sid}").scroll_into_view_if_needed()
            pts = []
            for name, t in (("a_seek_handover", ho), ("b_seek_event", tb)):
                page.evaluate("(a) => { const g = window.__cmp.groups.find(g => g.sc.scene === a[0]); window.__cmp.seek(g, a[1], true); }", [sid, t])
                pts.append((name, t, grab(page, sid)))
            page.evaluate("(sid) => { const g = window.__cmp.groups.find(g => g.sc.scene === sid); g.sec.querySelector('.spd').value = '1'; window.__cmp.play(g); }", sid)
            page.wait_for_timeout(2000)
            page.evaluate("(sid) => { const g = window.__cmp.groups.find(g => g.sc.scene === sid); window.__cmp.pause(g); }", sid)
            pts.append(("c_after_2s_play", None, grab(page, sid)))
            for name, t, shots in pts:
                ims = [Image.open(io.BytesIO(base64.b64decode(s["png"].split(",")[1]))).convert("RGB") for s in shots]
                W = sum(i.width for i in ims) + 10 * len(ims); m = Image.new("RGB", (W, ims[0].height + 16), "#000"); dr = ImageDraw.Draw(m); x = 0
                for s, i in zip(shots, ims):
                    m.paste(i, (x, 16)); dr.text((x + 2, 2), f"{sid} {s['arm']} ct={s['ct']:.3f}", fill="white"); x += i.width + 10
                fn = OUTD / "sync" / f"{sid}_{name}.png"; fn.parent.mkdir(exist_ok=True); m.save(fn)
                cts = [s["ct"] for s in shots]
                res.append(dict(scene=sid, point=name, target_t=t, current_times={s["arm"]: round(s["ct"], 4) for s in shots},
                                spread_s=round(max(cts) - min(cts), 4), crops=str(fn)))
                print(res[-1], flush=True)
        # scene screenshots, reset on, paused
        for sid, t in (("bae463d3", float(sys.argv[3]) if len(sys.argv) > 3 else 9.25), ("04749bb9", 9.25)):
            page.evaluate("(a) => { const g = window.__cmp.groups.find(g => g.sc.scene === a[0]); window.__cmp.seek(g, a[1], true); }", [sid, t])
            page.evaluate(JS_WAIT, sid); loc = page.locator(f"#s_{sid}"); loc.scroll_into_view_if_needed(); page.wait_for_timeout(500)
            loc.screenshot(path=str(OUTD / f"compare_{sid}_t{t:.2f}.png"))
        b.close()
    json.dump(res, open(OUTD / "sync_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
