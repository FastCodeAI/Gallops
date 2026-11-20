#!/usr/bin/env python
"""Comparison page on data8t (port 8786), now 50 scenes: the 12 diagnosis (CL12), the 18 (CL18) and the 20 new (CL20, reports/CL20_PREREG.md).
Per-run content code is s13_compare_page.py unchanged (run_info, render, draw_bar, short_exit). Changed here:
  * output under {local_store}/compare_page/ (NAS read-only); a video already rendered on the NAS whose stamp matches the
    run exactly is used from there read-only (symlink media_nas), anything else is rendered into compare_page/media/ on data8t;
  * scene discovery adds the 20 new scenes (out/cl20_scenes_20_v2.txt; runs under {local_store}/runs/new40);
  * the Metrics tab answers the CL20 rule: score and exit per run are checked against out/cl20_table.json (s20_table.py), and the
    verdict is recomputed and asserted equal to it. At-fault for the rule = front/side collision inside the score window.
Template: s20_compare_page.html (copy of s13_compare_page.html with the blocks 20 / 18 / 12 / 50).

    alpasim/.venv/bin/python -B s20_compare_page.py [--no-render]
"""
import argparse, json, os, pathlib, statistics as st, sys

ARGV = list(sys.argv)          # s13 / s12 helpers below reset sys.argv on import
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s13_compare_page as P                                                   # noqa: E402

OUT = pathlib.Path("{local_store}/compare_page")
NAS_MEDIA = pathlib.Path("{experiment_store}/compare_page/media")
N40 = "{local_store}/runs/new40"
P.OUT = OUT; P.MEDIA = OUT / "media"
OUT.mkdir(parents=True, exist_ok=True)
link = OUT / "media_nas"
if not link.is_symlink(): link.symlink_to(NAS_MEDIA)
_render = P.render
T12 = (HERE / "s12_table.py").read_text(); MC = {"__file__": str(HERE / "s12_table.py")}; exec(compile(T12.split("\nrows = []")[0], "s12_table.py", "exec"), MC)


def render(r, H):
    name = pathlib.Path(r["video"]).stem.replace("_annotated", "")
    key = json.dumps({k: r[k] for k in ("events", "handover_t", "score", "exit", "video")}, sort_keys=True)
    old, stamp = NAS_MEDIA / f"{name}_cmpbar.mp4", NAS_MEDIA / f"{name}_cmpbar.json"
    if old.is_file() and stamp.is_file() and stamp.read_text() == key and (NAS_MEDIA / f"{name}_bar.png").is_file():
        r["media"] = f"media_nas/{old.name}"; r["bar_png"] = f"media_nas/{name}_bar.png"
        return "kept (NAS, read-only)"
    return _render(r, H)                                                       # writes only under OUT/media (data8t)


P.render = render
_draw_bar = P.draw_bar
MAX_LABELS = 6                    # s13's bar has room for 7 label rows; handover + 5 events, then a "+N more" note


def draw_bar(r, path):
    """s13 draw_bar unchanged unless the run has more events than the strip can label: then every event is still drawn on the
    track, only the first MAX_LABELS-1 events get a label, and a '+N more events' note is added (the page lists them all)."""
    if len(r["events"]) + 1 <= MAX_LABELS:
        return _draw_bar(r, path)
    from PIL import Image, ImageDraw, ImageFont
    keep = dict(r, events=r["events"][:MAX_LABELS - 1])
    _draw_bar(keep, path)
    im = Image.open(path); d = ImageDraw.Draw(im); ty = 52
    for e in r["events"][MAX_LABELS - 1:]:
        d.rectangle([P.tx(e["t"]), ty, max(P.tx(e["t_end"]), P.tx(e["t"]) + 2), ty + 8], fill=e["color"])
    n = len(r["events"]) - (MAX_LABELS - 1); f = ImageFont.truetype(P.FONT, 11); txt = f"+{n} more events (listed on the page)"
    d.text((P.BAR_W - 10 - d.textlength(txt, font=f), P.BAR_H - 16), txt, fill="#9da7b3", font=f)
    im.save(path)


P.draw_bar = draw_bar
S20 = [l.split()[0] for l in open(HERE.parent / "out/cl20_scenes_20_v2.txt") if l.strip()]
S18 = [l.split()[0] for l in open(HERE.parent / "out/cl18_scenes_18.txt") if l.strip()]
CL18_NAS = "{experiment_store}/runs/cl18"


def discover():
    scenes = []
    for s in P.S12:
        runs = {"carry": (f"{P.CL12}/runs/memory_carry_{s}", "CL12"), "reset": (f"{P.CL12}/runs/memory_reset_{s}", "CL12")}
        ic = P.R018_IC.format(s); runs["018b"] = (ic, "0.1.8b image-check re-run (cl12/imagecheck_018bp06)") if os.path.isdir(ic) else (P.R018.format(s), "0.1.8b closed-loop run (018b/closed_loop)")
        scenes.append(dict(scene=s, source="CL12", runs=runs))
    for s in S18:
        runs = {"carry": (f"{CL18_NAS}/runs/memory_carry_{s}", "CL18"), "018b": (f"{CL18_NAS}/018bp06/runs/drivejepa_018bp06_{s}", "0.1.8b CL18 run (cl18/018bp06)")}
        scenes.append(dict(scene=s, source="CL18", runs=runs))
    for s in S20:
        c, b = f"{N40}/runs/memory_carry_{s}", f"{N40}/018bp06/runs/drivejepa_018bp06_{s}"
        if all(os.path.isfile(f"{d}/aggregate/results-summary.json") for d in (c, b)):
            scenes.append(dict(scene=s, source="CL20", runs={"carry": (c, "CL20"), "018b": (b, "0.1.8b CL20 run (new40/018bp06)")}))
    return scenes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-render", action="store_true"); a = ap.parse_args(ARGV[1:])
    assert str(P.OUT).startswith("{local_disk}/") and str(P.MEDIA).startswith("{local_disk}/")
    P.MEDIA.mkdir(parents=True, exist_ok=True); P.head_png(P.MEDIA / "playhead.png")
    T = json.load(open(HERE.parent / "out/cl20_table.json")); TR = {(r["block"], r["scene"]): r for r in T["rows"]}
    scenes = discover(); order = {"CL20": 0, "CL18": 1, "CL12": 2}
    scenes.sort(key=lambda x: (order[x["source"]], x["scene"]))
    log = []
    for sc in scenes:
        sc["arms"] = {}
        blk = {"CL12": "12", "CL18": "18", "CL20": "20"}[sc["source"]]
        for arm, (run, src) in sc["runs"].items():
            r = P.run_info(run, arm, sc["scene"]); r["source"] = src
            if arm in ("carry", "018b"):                       # same score and exit as out/cl20_table.json (s20_table.py)
                t = TR[(blk, sc["scene"])][arm]
                assert t and abs(t["score"] - r["score"]) < 1e-9 and t["exit"] == r["exit"], (sc["scene"], arm, t and t["exit"], r["exit"])
                r["atfault_window"] = t["atfault_window"]
            sc["arms"][arm] = r
        t0s = {r["t0_us"] for r in sc["arms"].values()}; assert len(t0s) == 1, (sc["scene"], t0s)
        hos = {round(r["handover_t"], 6) for r in sc["arms"].values()}; assert len(hos) == 1, (sc["scene"], hos)
        for r in sc["arms"].values():
            w, h = P.probe_h(r["video"]); assert w == P.BAR_W, (r["video"], w)
            r["video_h"] = h
            stt = "skipped (--no-render)" if a.no_render else render(r, h)
            log.append(dict(scene=sc["scene"], arm=r["arm"], video=r["video"], out=r.get("media"), action=stt, n_events=len(r["events"])))
            print(sc["scene"], r["arm"], stt, r["score"], r["exit"], flush=True)
        del sc["runs"]
    met = metrics(scenes, T)
    if a.no_render:
        dry = OUT / "s20_dry_metrics.json"; json.dump(met, open(dry, "w"), indent=1, default=str); print("dry run ->", dry); return
    json.dump(dict(scenes=scenes, metrics=met, render_log=log), open(OUT / "data.json", "w"), indent=1, default=str)
    (OUT / "index.html").write_text(page(scenes, met))
    print("wrote", OUT / "index.html", len(scenes), "scenes")


def page(scenes, met):
    data = json.dumps([dict(scene=s["scene"], source=s["source"], arms={k: dict(media=v.get("media"), score=v["score"], exit=v["exit"],
            short=P.short_exit(v), handover=v["handover_t"], events=v["events"], n_ticks=v["n_ticks"], progress=v["progress_uncapped"],
            run=v["run"], source=v["source"]) for k, v in s["arms"].items()}) for s in scenes])
    tmpl = (HERE / "s20_compare_page.html").read_text()
    return tmpl.replace("/*DATA*/[]", data).replace("/*MET*/{}", json.dumps(met, default=str))


AF = ("collision_front", "collision_lateral")


def metrics(scenes, T):
    C18 = {r["scene"]: r for r in __import__("csv").DictReader(open(HERE.parent / "out/cl18_table.csv"))}
    C12 = {r["scene"]: r for r in __import__("csv").DictReader(open(HERE.parent / "out/cl12_table.csv"))}
    rows = []
    for sc in scenes:
        c, b = sc["arms"]["carry"], sc["arms"]["018b"]
        blk = {"CL12": "12", "CL18": "18", "CL20": "20"}[sc["source"]]
        arm = lambda r: dict(score=r["score"], exit=r["exit"], short=P.short_exit(r), window_end_rel=r["window_end_rel"],   # noqa: E731
                             events=[dict(kind=e["kind"], label=e["label"], rel=e["rel"], after_window=e["after_window"]) for e in r["events"]],
                             summary=r["summary"], atfault=r["exit"].startswith("at-fault"), atfault_window=r["atfault_window"],
                             offroad=r["exit"].startswith("off-road"))
        t = C18.get(sc["scene"])
        row = dict(scene=sc["scene"], block=blk, carry=arm(c), b018=arm(b), diff=c["score"] - b["score"],
                   s017=float(t["score_017"]) if t else None, exit017=t["exit_017"] if t else None)
        if blk == "12":
            row["shadow"] = dict(calls=int(float(C12[sc["scene"]]["calls_carry"])), differ_final=int(float(C12[sc["scene"]]["n_differ_final_carry"])))
        elif blk == "18":
            row["shadow"] = dict(calls=int(float(t["calls"])), differ_final=int(float(t["n_differ_final"])))
        else:
            calls = MC["memory_calls"](c["run"]); assert calls and all(d["mode"] == "carry" for d in calls), sc["scene"]
            row["shadow"] = dict(calls=len(calls), differ_final=sum(bool(d["picks_differ_final"]) for d in calls))
        rows.append(row)

    def agg(R):
        out = {}
        for k, key in (("carry", "carry"), ("018b", "b018")):
            A = [r[key] for r in R]
            out[k] = dict(mean=st.mean(a["score"] for a in A), atfault=[r["scene"] for r in R if r[key]["atfault"]],
                          atfault_window=[r["scene"] for r in R if r[key]["atfault_window"]],
                          offroad=[r["scene"] for r in R if r[key]["offroad"]],
                          exits_after_window=[r["scene"] for r in R if r[key]["events"] and r[key]["events"][0]["rel"] > 0 and r[key]["events"][0]["after_window"]],
                          window_end_median=st.median([a["window_end_rel"] for a in A if a["window_end_rel"] > 0] or [0.0]),
                          perfect=sum(a["score"] >= 1 - 1e-9 for a in A), zero=sum(a["score"] <= 1e-9 for a in A),
                          dist=st.mean(a["summary"]["dist_traveled_m"] for a in A),
                          progress=st.mean(min(a["summary"]["progress"], 1.0) for a in A),
                          dist_gt=st.mean(a["summary"]["dist_to_gt_trajectory"] for a in A),
                          min_obst=st.median(a["summary"]["min_distance_to_obstacle_m"] for a in A),
                          wrong_lane=sum(a["summary"]["wrong_lane"] > 0 for a in A),
                          plan_dev=st.mean(a["summary"]["plan_deviation"] for a in A))
        R17 = [r for r in R if r["s017"] is not None]
        out["017"] = dict(n=len(R17), mean=st.mean(r["s017"] for r in R17) if R17 else None,
                          atfault=[r["scene"] for r in R17 if r["exit017"].startswith("at-fault")],
                          offroad=[r["scene"] for r in R17 if r["exit017"].startswith("off-road")])
        D = sorted(R, key=lambda r: -r["diff"]); top2 = [r["scene"] for r in D[:2]]
        out.update(n=len(R), d=st.mean(r["diff"] for r in R), top2=top2, d_wo_top2=st.mean(r["diff"] for r in R if r["scene"] not in top2),
                   wins=[r["scene"] for r in R if r["diff"] > 0.1], losses=[r["scene"] for r in R if r["diff"] < -0.1],
                   ties=sum(abs(r["diff"]) <= 0.1 for r in R), identical=sum(abs(r["diff"]) < 1e-9 for r in R),
                   both_perfect=sum(r["carry"]["score"] >= 1 - 1e-9 and r["b018"]["score"] >= 1 - 1e-9 for r in R))
        sh = [r["shadow"] for r in R if "shadow" in r]
        if sh: out["shadow_pct"] = 100.0 * sum(s["differ_final"] for s in sh) / sum(s["calls"] for s in sh)
        return out
    B = {k: agg([r for r in rows if k == "50" or r["block"] == k]) for k in ("20", "18", "12", "50")}
    b = B["20"]; added = [s for s in b["carry"]["atfault_window"] if s not in b["018b"]["atfault_window"]]
    W, L = len(b["wins"]), len(b["losses"])
    verdict = "AGAINST" if (b["d"] <= -0.05 or L > W or added) else ("SUPPORTS memory" if (b["d"] >= 0.05 and W > L) else "NO DETECTABLE EFFECT")
    J = T["result"]
    assert b["n"] == 20 and verdict == J["verdict"] and abs(b["d"] - J["primary_20"]["d"]) < 1e-12 and added == J["primary_20"]["atfault_window_carry_not_018b"], (verdict, J["verdict"])
    assert abs(B["50"]["d"] - J["all_50"]["d"]) < 1e-12

    R = {r["scene"]: r for r in rows}
    reasons = []
    if b["d"] <= -0.05: reasons.append(f"the mean gap is {b['d']:+.3f}, at or below −0.05")
    if L > W: reasons.append(f"memory loses more scenes than it wins ({L} against {W})")
    if added: reasons.append("memory has at-fault crashes inside the score window that 0.1.8b doesn't (" + ", ".join(added) + ")")
    ins = []
    ins.append(dict(h=f"On the 20 new scenes memory scores {'lower' if b['d'] < 0 else 'higher'} than 0.1.8b",
        p=f"Mean score memory {b['carry']['mean']:.4f} vs 0.1.8b {b['018b']['mean']:.4f}, a gap of {b['d']:+.3f}. Memory wins {W} scenes "
          f"({', '.join(b['wins']) or 'none'}), loses {L} ({', '.join(b['losses']) or 'none'}), and ties {b['ties']}. Without the two scenes most in "
          f"memory's favour ({', '.join(b['top2'])}) the gap is {b['d_wo_top2']:+.3f}. {b['both_perfect']} of the 20 score 1.000 for both drivers."))
    ins.append(dict(h="Crashes inside the score window",
        p=f"At-fault (front or side) crashes inside the score window on the 20: memory {len(b['carry']['atfault_window'])} "
          f"({', '.join(b['carry']['atfault_window']) or 'none'}), 0.1.8b {len(b['018b']['atfault_window'])} ({', '.join(b['018b']['atfault_window']) or 'none'}). "
          + (f"Only memory crashes on {', '.join(added)}, which by itself makes the answer AGAINST under the rule." if added else "Memory adds none.")))
    xo = [s for s in b["carry"]["offroad"] if s not in b["018b"]["offroad"]]
    ins.append(dict(h="Off-road exits",
        p=f"Off-road exits on the 20: memory {len(b['carry']['offroad'])} ({', '.join(b['carry']['offroad']) or 'none'}), 0.1.8b {len(b['018b']['offroad'])} "
          f"({', '.join(b['018b']['offroad']) or 'none'}). 7747986c is off-road before handover in both arms (the scene start, not the driver)."
          + (f" Off-road only with memory: {', '.join(xo)}." if xo else "")))
    ins.append(dict(h="All 50 scenes",
        p=f"Over all 50, memory {B['50']['carry']['mean']:.4f} vs 0.1.8b {B['50']['018b']['mean']:.4f} ({B['50']['d']:+.3f}); wins / losses / ties "
          f"{len(B['50']['wins'])} / {len(B['50']['losses'])} / {B['50']['ties']}. By block the gap is {B['12']['d']:+.3f} on the 12 diagnosis scenes, "
          f"{B['18']['d']:+.3f} on the 18 and {b['d']:+.3f} on the 20 new ones. The 12 diagnosis scenes are where memory was looked at most, so the 18 and the 20 are the fairer tests."))
    ins.append(dict(h="This compares two models, not memory alone",
        p="The memory model is 0.1.8b plus the memory and 930 more training steps. This test cannot tell which of the two causes a difference."))
    if "shadow_pct" in b:
        ins.append(dict(h="How often memory changes the pick",
            p=f"In memory runs the shadow branch, which runs the reset memory on the same input, picks a different final plan on {b['shadow_pct']:.1f} % of calls "
              f"on the 20 ({B['18']['shadow_pct']:.1f} % on the 18, {B['12']['shadow_pct']:.1f} % on the 12)."))
    return dict(rows=rows, blocks=B, verdict=verdict, added=added, reasons=reasons,
                rule="on the 20 new scenes, SUPPORTS memory if memory − 0.1.8b ≥ +0.05 AND memory wins more scenes than it loses AND memory adds no at-fault crash inside the score window; "
                     "AGAINST if ≤ −0.05 OR it loses more than it wins OR it adds such a crash; otherwise NO DETECTABLE EFFECT",
                insights=ins)


if __name__ == "__main__":
    main()
