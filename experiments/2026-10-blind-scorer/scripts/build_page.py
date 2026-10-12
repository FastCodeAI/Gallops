"""Write swerve_check.html from the grading reports in swerves/."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SW = HERE / "swerves"
ORDER = ("clear", "off-lane", "off-road", "hit")
EPISODES = ["frontal-0053", "frontal-0064", "frontal-0070", "frontal-0075", "stationary-0070", "side-0070"]
NOTES = {
    "frontal-0053": "Held out of every training run so far. The inserted car is the white car down the road.",
    "frontal-0064": "Kept at your request. It was dropped from s2 and s3 because an earlier check found its inserted car does not render properly, so its hit labels may refer to a car the camera barely shows. Check the frame.",
    "frontal-0070": "The best-rendered car of the five scenes. No tick met the rule, because the released model is below 3 m/s by the time the car is close, so this is the fallback tick. In 0070 the released model drifts off the human's path (2 to 2.6 m by 3.5 s, heading 8 to 10 degrees away where the road bends), so every plan built on its pick is graded off-lane.",
    "frontal-0075": "Oncoming truck; its front was well observed, so it renders cleanly.",
    "stationary-0070": 'Parked car ahead; braking alone is enough here. Run 3 is one where the released model crept into the car. The same 0070 drift puts the released pick about 2 m from the lane centre, so it is graded off-lane.',
    "side-0070": 'Car crossing from the side. A straight path never meets it, so this is the fallback tick. All 32 paths miss the car; they are graded off-lane only because of the 0070 drift.',
}

reports = {}
for f in sorted(SW.glob("report_*.json")):
    for r in json.loads(f.read_text()):
        reports[r["episode"]] = r


def esc(s) -> str:
    return html.escape(str(s))


def counts_cells(c: dict) -> str:
    return "".join(f'<td class="num">{c.get(k, 0)}</td>' for k in ORDER)


def fmt_s(v) -> str:
    return "none in 4 s" if v is None else f"{v:.1f} s"


def profile_svg() -> str:
    """Lateral fraction over time for the two shapes, and distance over time for the two speeds."""
    t = np.linspace(0, 4, 161)

    def ease(u):
        return (1 - np.cos(np.pi * np.clip(u, 0, 1))) / 2

    hold = ease(t / 2.0)
    ret = np.where(t <= 1.5, ease(t / 1.5), 1.0)
    ret = np.where(t > 2.5, ease((4.0 - t) / 1.5), ret)
    held = t / 4
    brake = (t - 0.075 * t**2) / 4  # fraction of the held-speed distance at 4 s

    W, H, L, B, Tp, Rp = 300, 170, 40, 30, 14, 12

    def chart(series, title, ylab, ymax, yt):
        def X(v):
            return L + v / 4 * (W - L - Rp)

        def Y(v):
            return Tp + (1 - v / ymax) * (H - Tp - B)

        out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="profile">']
        for v in yt:
            out.append(f'<line x1="{L}" x2="{W - Rp}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
            out.append(f'<text x="{L - 6}" y="{Y(v) + 3.5:.1f}" class="tick" text-anchor="end">{v:g}</text>')
        for v in (0, 1, 2, 3, 4):
            out.append(f'<text x="{X(v):.1f}" y="{H - B + 15}" class="tick" text-anchor="middle">{v} s</text>')
        out.append(f'<line x1="{L}" x2="{W - Rp}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="axis"/>')
        for ys, cls, label, lx in series:
            d = "M" + " L".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t, ys))
            out.append(f'<path d="{d}" class="{cls}" fill="none"/>')
            i = int(lx * 40)
            out.append(f'<text x="{X(t[i]) + 4:.1f}" y="{Y(ys[i]) - 6:.1f}" class="lbl">{esc(label)}</text>')
        out.append(f'<text x="{L}" y="{Tp - 3}" class="ylab">{esc(ylab)}</text>')
        out.append("</svg>")
        return "".join(out)

    a = chart([(hold, "s1", "move over and stay", 2.6), (ret, "s2", "pass and come back", 0.35)],
              "Share of the sideways offset reached over time", "share of offset", 1.0, (0, 0.5, 1))
    b = chart([(held, "s1", "own speed", 2.9), (brake, "s2", "braking", 3.2)],
              "Distance covered over time, as a share of 4 s at the current speed", "share of distance", 1.0, (0, 0.5, 1))
    return a, b


def scene_section(name: str) -> str:
    r = reports.get(name)
    if r is None:
        return f'<section class="scene"><h3>{esc(name)}</h3><p class="muted">No tick qualified (hazard in view, at least 3 m/s, and a straight path at the current speed hits it within 4 s) in runs 0 to 9.</p></section>'
    d = r["tick_detail"]
    shape, scene = name.split("-")
    vb, va = d["vocab"], d["vocab_a"]
    n = r["pick_hit_ticks_in_view"]
    cover = ""
    if n:
        cover = (f'<p class="cover">Across this whole episode, the released pick hits on <b>{n}</b> ticks with the hazard in view. '
                 f'A plan that misses everything and stays on the road exists among the 32 native plans on <b>{r["of_those_native_ok"]}</b>, '
                 f'among the first version of the vocabulary on <b>{r["of_those_vocab_a_ok"]}</b>, '
                 f'and among the current version on <b>{r["of_those_vocab_ok"]}</b> '
                 f'({r["of_those_vocab_clear"]} of those also stay in the lane).</p>')
    else:
        cover = '<p class="cover">The released pick never hits in this episode with the hazard in view.</p>'
    return f"""
<section class="scene" id="{esc(name)}">
  <div class="scene-head">
    <h3>{esc(shape)} · scene {esc(scene)}</h3>
    <span class="tag">run {r['run']} · t = {r['t_in_episode']:.1f} s{' · fallback tick' if r.get('rule') == 'fallback' else ''}</span>
  </div>
  <p class="note">{esc(NOTES.get(name, ''))}</p>
  <dl class="facts">
    <div><dt>speed</dt><dd>{d['speed']:.1f} m/s</dd></div>
    <div><dt>straight at this speed hits in</dt><dd>{fmt_s(d['cv_hit'])}</dd></div>
    <div><dt>hazard bearing</dt><dd>{'' if d['bearing'] is None else f"{d['bearing']:.0f}°"} off axis</dd></div>
    <div><dt>released pick</dt><dd>{esc(d['pick'])}{'' if d['pick_hit'] is None else f" at {d['pick_hit']:.1f} s"} · {d['pick_travel']:.1f} m in 4 s</dd></div>
  </dl>
  <figure class="cam">
    <img src="{esc(r['stem'])}_cam.jpg" alt="Front camera at this tick with the released pick and the eight move-over paths at the current speed" loading="lazy" width="1200" height="675">
    <figcaption>Front camera, the frame Drive-JEPA read. White: the released pick. Coloured: the 8 "move over and stay" paths at the current speed. Faint: the 32 native plans.</figcaption>
  </figure>
  <figure class="bev">
    <img src="{esc(r['stem'])}_bev.png" alt="Top-down view of all 32 vocabulary paths in four panels" loading="lazy" width="1600" height="740">
    <figcaption>Top-down, ego at the bottom facing up, left is left. Light ground: drivable map. Dashed line: lane centre. Filled box: the inserted car now; dotted boxes: where it will be at +1 to +4 s. Grey boxes: other vehicles.</figcaption>
  </figure>
  <div class="table-wrap"><table>
    <thead><tr><th>32 paths at this tick</th>{''.join(f'<th class="num"><span class="chip {k}"></span>{k}</th>' for k in ORDER)}</tr></thead>
    <tbody>
      <tr><td>current version (current speed)</td>{counts_cells(vb)}</tr>
      <tr><td>first version (pick's own speed)</td>{counts_cells(va)}</tr>
      <tr><td>32 native plans</td><td class="num">{d['native_clear']}</td><td class="num" colspan="3">{32 - d['native_clear']} not clear</td></tr>
    </tbody>
  </table></div>
  {cover}
</section>"""


def totals() -> str:
    rows = []
    agg = {"n": 0, "nat": 0, "a": 0, "b": 0, "bc": 0}
    for name in EPISODES:
        r = reports.get(name)
        if not r:
            continue
        n = r["pick_hit_ticks_in_view"]
        agg["n"] += n
        agg["nat"] += r["of_those_native_ok"]
        agg["a"] += r["of_those_vocab_a_ok"]
        agg["b"] += r["of_those_vocab_ok"]
        agg["bc"] += r["of_those_vocab_clear"]
        rows.append(f'<tr><td>{esc(name)}</td><td class="num">{n}</td><td class="num">{r["of_those_native_ok"]}</td>'
                    f'<td class="num">{r["of_those_vocab_a_ok"]}</td><td class="num">{r["of_those_vocab_ok"]}</td><td class="num">{r["of_those_vocab_clear"]}</td></tr>')
    fr = [reports[n] for n in EPISODES if n in reports and n.startswith("frontal")]
    rows.append(f'<tr class="total"><td>frontal episodes</td><td class="num">{sum(r["pick_hit_ticks_in_view"] for r in fr)}</td>'
                f'<td class="num">{sum(r["of_those_native_ok"] for r in fr)}</td><td class="num">{sum(r["of_those_vocab_a_ok"] for r in fr)}</td>'
                f'<td class="num">{sum(r["of_those_vocab_ok"] for r in fr)}</td><td class="num">{sum(r["of_those_vocab_clear"] for r in fr)}</td></tr>')
    rows.append(f'<tr class="total"><td>all shown</td><td class="num">{agg["n"]}</td><td class="num">{agg["nat"]}</td>'
                f'<td class="num">{agg["a"]}</td><td class="num">{agg["b"]}</td><td class="num">{agg["bc"]}</td></tr>')
    return "".join(rows), agg


def variant_rows() -> str:
    names = {(1.0, "hold"): "own speed · move over and stay", (1.0, "return"): "own speed · pass and come back",
             (0.7, "hold"): "braking · move over and stay", (0.7, "return"): "braking · pass and come back"}
    rows = []
    shown = [reports[n] for n in EPISODES if n in reports]
    for key, label in names.items():
        cells = []
        for r in shown:
            v = [x for x in r["vocab_at_tick"] if (x["speed"], x["shape"]) == key]
            ok = sum(x["verdict"] in ("clear", "off-lane") for x in v)
            cells.append(f'<td class="num">{ok} / {len(v)}</td>')
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    head = "".join(f'<th class="num">{esc(r["episode"])}</th>' for r in shown)
    return f"<thead><tr><th>paths with no hit, on the road</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"


def main() -> None:
    svg_a, svg_b = profile_svg()
    trows, agg = totals()
    a_stem = reports["frontal-0053"]["stem"] if "frontal-0053" in reports else ""
    scenes = "".join(scene_section(n) for n in EPISODES)
    fr = [reports[n] for n in EPISODES if n in reports and n.startswith("frontal")]
    fr_n = sum(r["pick_hit_ticks_in_view"] for r in fr)
    fr_nat = sum(r["of_those_native_ok"] for r in fr)
    fr_a = sum(r["of_those_vocab_a_ok"] for r in fr)
    fr_b = sum(r["of_those_vocab_ok"] for r in fr)
    assert fr_a == fr_b, (fr_a, fr_b)  # the sentence says "either version"; rewrite it if they ever differ
    v2 = json.loads((SW / "v2_summary.json").read_text())
    figs = []
    for name in EPISODES:
        r = reports.get(name)
        if not r or name not in v2:
            continue
        c = v2[name]["counts"]
        figs.append(
            f'<figure class="bev"><img src="{esc(r["stem"])}_v2_bev.png" alt="Revised vocabulary on {esc(name)}" '
            f'loading="lazy" width="1260" height="740"><figcaption><b>{esc(name)}</b>: '
            + ", ".join(f"{k} {c[k]}" for k in ORDER)
            + f"; native plans safe {v2[name]['native_safe']} of 32.</figcaption></figure>"
        )
    TEMPLATE_V2 = "".join(figs)
    page = TEMPLATE.replace("{{V2}}", TEMPLATE_V2).replace("{{FR_N}}", str(fr_n)).replace("{{FR_NAT}}", str(fr_nat)).replace("{{FR_A}}", str(fr_a))
    for k, v in {"{{SVG_A}}": svg_a, "{{SVG_B}}": svg_b, "{{TOTAL_ROWS}}": trows, "{{SCENES}}": scenes,
                 "{{A_STEM}}": a_stem, "{{VARIANTS}}": variant_rows(), "{{AGG_N}}": str(agg["n"]), "{{AGG_NAT}}": str(agg["nat"]),
                 "{{AGG_A}}": str(agg["a"]), "{{AGG_B}}": str(agg["b"]), "{{AGG_BC}}": str(agg["bc"])}.items():
        page = page.replace(k, v)
    (SW / "swerve_check.html").write_text(page)
    print("wrote", SW / "swerve_check.html", len(page), "bytes")


TEMPLATE = (HERE / "page_template.html").read_text()

if __name__ == "__main__":
    main()
