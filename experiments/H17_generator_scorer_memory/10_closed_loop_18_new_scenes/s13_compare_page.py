#!/usr/bin/env python
"""Side-by-side replay page: memory carried vs 0.1.8b (and memory reset, toggle), per scene, synchronised by simulated time.

Re-run to add scenes: the 12 CL12 scenes always; any CL18 scene whose carry run is complete (results-summary.json + annotated
video) under $CL18_ROOT (default {experiment_store}/runs/cl18/runs/memory_<arm>_<scene>) is added automatically.

Per run:
  score, exit  = s5_diagnosis/src/stepb.py `score`, `exit_of` (imported unchanged, as s12_table.py does)
  events       = every episode (onset .. last tick) of collision_front / collision_lateral / offroad / collision_rear > 0 in the run's
                 metrics.parquet: the same rows exit_of reads; exit_of's exit must equal the first onset found here (asserted)
  handover     = gates G-SENT handover_us in the run's dj_extra_gates.json (= first driver_request in the run's own rollout.asl
                 + 3 s), put on the controller-csv clock with the replay trace's t0; asserted equal to the trace's handover_t_s
  sim clock    = controller-csv t (t=0 at the csv's first timestamp_us); the annotated video's frame k is tick k, t = k * 0.05 s
                 (burnt into the frame as "t X.XXX s"); the arms of a scene share t0 (asserted), so equal t = equal sim time
Bar overlay: the original annotated video is never modified. A new file <run>_cmpbar.mp4 is written under compare_page/media/ =
original frames + a 192 px strip below with the event bar (ffmpeg libx264 -preset veryfast, 2 threads, pinned to 2 cores).

    alpasim/.venv/bin/python -B s13_compare_page.py [--no-render]
"""
import argparse, glob, html, json, os, pathlib, re, subprocess, sys

import polars as pl
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
G = pathlib.Path("{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim")
S5 = G / "s5_diagnosis/src"; argv = sys.argv; sys.argv = sys.argv[:1]
_src = (S5 / "stepb.py").read_text().split("\nrows = []")[0]
B = {"__file__": str(S5 / "stepb.py")}; exec(compile(_src, str(S5 / "stepb.py"), "exec"), B)
sys.argv = argv

OUT = pathlib.Path("{experiment_store}/compare_page"); MEDIA = OUT / "media"
CL12 = "{experiment_store}/runs/cl12"
CL18 = os.environ.get("CL18_ROOT", "{experiment_store}/runs/cl18")
R018 = "{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_{}"
R018_IC = f"{CL12}/imagecheck_018bp06/runs/drivejepa_018bp06_{{}}"      # image-checked re-runs (4 scenes), preferred where present
R018_CL18 = f"{CL18}/018bp06/runs/drivejepa_018bp06_{{}}"                 # 0.1.8b on the 18 new scenes (CL18 arm)
SUMMARY_KEYS = ["dist_traveled_m", "gt_dist_traveled_m", "progress", "dist_to_gt_trajectory", "min_distance_to_obstacle_m",
                "min_distance_to_lane_boundary_m", "wrong_lane", "plan_deviation", "duration_frac_20s"]
FF = "{paddock_sid}/Alpasim/alpasim/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
FLIPPED = ["bae463d3", "5f064b02", "04749bb9"]; NEXT = ["048b974e"]
S12 = [l.split()[0] for l in open(G / "s10_018b_closed_loop/tables/scenes_12.txt") if l.strip()]
EVENTS = [("collision_front", "front collision", "#ff3b30"), ("collision_lateral", "side collision", "#d946ef"),
          ("offroad", "off-road", "#ffd60a"), ("collision_rear", "hit from behind", "#38bdf8")]
HO_COL = "#ff9f0a"; T_MAX = 20.0; DT = 0.05
BAR_W, BAR_H, X0, X1 = 900, 192, 40, 860
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ARM_TITLE = {"carry": "Memory carried", "018b": "0.1.8b", "reset": "Memory reset"}


def tx(t):
    return X0 + (X1 - X0) * min(max(t, 0.0), T_MAX) / T_MAX


def run_info(run, arm, scene):
    run = pathlib.Path(run)
    (tr,) = glob.glob(str(run / "replay/*_trace.json")); h = json.load(open(tr))["header"]
    (vid,) = glob.glob(str(run / "replay/*_annotated.mp4"))
    t0 = int(re.search(r"timestamp_us (\d+)", h["t_s_basis"])[1])
    ho_us = json.load(open(run / "dj_extra_gates.json"))["gates"]["G-SENT"]["handover_us"]
    ho = (ho_us - t0) / 1e6
    assert abs(ho - h["handover_t_s"]) < 1e-6, (run, ho, h["handover_t_s"])
    m = pl.read_parquet(glob.glob(str(run / "rollouts/clipgt-*/*/metrics.parquet"))[0]).with_columns(pl.col("timestamps_us").cast(pl.Float64))
    ev = []
    for nm, lab, col in EVENTS:
        rows = m.filter(pl.col("name") == nm).sort("timestamps_us")
        ts, vals = rows["timestamps_us"].to_list(), rows["values"].to_list()
        start = prev = None
        for t, v in zip(ts, vals):
            if v > 0:
                if start is None: start = t
                prev = t
            elif start is not None:
                ev.append((start, prev, nm, lab, col)); start = None
        if start is not None: ev.append((start, prev, nm, lab, col))
    ev.sort()
    events = [dict(kind=nm, label=lab, color=col, t=(a - t0) / 1e6, t_end=(b - t0) / 1e6, rel=(a - t0) / 1e6 - ho) for a, b, nm, lab, col in ev]
    sc = B["score"](run); ex = B["exit_of"](run)
    if events:   # exit_of's first exit must be the first onset found here (same rows, same handover)
        e0 = events[0]; lab0 = {"collision_front": "at-fault collision (front)", "collision_lateral": "at-fault collision (side)",
                                "offroad": "off-road", "collision_rear": "hit from behind"}[e0["kind"]]
        assert ex == f"{lab0} at +{e0['rel']:.2f} s", (run, ex, e0)
    else:
        assert " at +" not in ex, (run, ex)
    summ = json.load(open(run / "aggregate/results-summary.json"))["metrics_results"][0]
    win_end = h["scored_window"]["t_csv_s"][1]                        # end of the window the score reads (sim t)
    for e in events: e["after_window"] = e["t"] > win_end + 1e-6
    return dict(arm=arm, scene=scene, run=str(run), video=vid, summary={k: summ.get(k) for k in SUMMARY_KEYS},
                window_end_t=win_end, window_end_rel=win_end - ho, t0_us=t0, handover_us=ho_us, handover_t=ho, n_ticks=h["n_ticks"],
                score=float(sc["score"]), status=sc.get("status"), exit=ex, events=events,
                progress_uncapped=h.get("aggregate", {}).get("progress"))


def short_exit(r):
    if not r["events"]:
        return r["exit"]
    e = r["events"][0]; return f"{e['label']} +{e['rel']:.2f} s"


def draw_bar(r, path):
    im = Image.new("RGB", (BAR_W, BAR_H), "#0d1117"); d = ImageDraw.Draw(im)
    f, fs, fb = ImageFont.truetype(FONT, 11), ImageFont.truetype(FONT, 10), ImageFont.truetype(FONTB, 12)
    d.text((10, 4), f"{ARM_TITLE[r['arm']]}  ·  score {r['score']:.3f}  ·  exit: {short_exit(r)}", fill="#e6edf3", font=fb)
    lx = 10; ly = 22                                                   # legend
    for lab, col in [("handover", HO_COL)] + [(l, c) for _, l, c in EVENTS]:
        d.rectangle([lx, ly + 3, lx + 9, ly + 12], fill=col); d.text((lx + 13, ly), lab, fill="#9da7b3", font=fs)
        lx += 18 + d.textlength(lab, font=fs) + 12
    d.text((BAR_W - 10 - d.textlength("times: s after handover (sim t)", font=fs), ly), "times: s after handover (sim t)", fill="#9da7b3", font=fs)
    ty = 52                                                            # track
    d.rectangle([X0, ty, X1, ty + 8], fill="#30363d")
    for e in r["events"]:
        d.rectangle([tx(e["t"]), ty, max(tx(e["t_end"]), tx(e["t"]) + 2), ty + 8], fill=e["color"])
    for s in range(0, int(T_MAX) + 1):
        x = tx(s); d.line([x, ty + 9, x, ty + (14 if s % 2 == 0 else 11)], fill="#6e7681")
        if s % 2 == 0: d.text((x - d.textlength(str(s), font=fs) / 2, ty + 14), str(s), fill="#6e7681", font=fs)
    marks = [dict(t=r["handover_t"], color=HO_COL, text=f"handover (t {r['handover_t']:.2f})")] + \
            [dict(t=e["t"], color=e["color"], text=f"{e['label']} +{e['rel']:.2f} s") for e in r["events"]]
    rows_end, placed = [], []                                          # labels stacked in rows so none overlap
    for mk in marks:
        x = tx(mk["t"]); w = d.textlength(mk["text"], font=f)
        lx = min(x + 4, BAR_W - w - 4)                                  # label starts just right of its own line
        row = next((i for i, e in enumerate(rows_end) if lx > e + 8), len(rows_end))
        if row == len(rows_end): rows_end.append(-1e9)
        rows_end[row] = lx + w; placed.append((mk, x, lx, w, 82 + 15 * row))
    for mk, x, lx, w, y in placed:                                     # all lines first, then labels on a dark backing
        d.line([x, ty - 4, x, y + 13], fill=mk["color"], width=2)
    for mk, x, lx, w, y in placed:
        d.rectangle([lx - 1, y, lx + w + 1, y + 13], fill="#0d1117")
        d.text((lx, y), mk["text"], fill=mk["color"], font=f)
    assert len(rows_end) <= 7, (r["run"], marks)
    im.save(path)


def head_png(path):
    im = Image.new("RGBA", (9, 30), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(0, 0), (8, 0), (4, 6)], fill="white"); d.line([4, 0, 4, 29], fill="white", width=2); im.save(path)


def render(r, H):
    name = pathlib.Path(r["video"]).stem.replace("_annotated", "")
    bar = MEDIA / f"{name}_bar.png"; out = MEDIA / f"{name}_cmpbar.mp4"; draw_bar(r, bar)
    r["media"] = f"media/{out.name}"; r["bar_png"] = f"media/{bar.name}"
    stamp = MEDIA / f"{name}_cmpbar.json"; key = json.dumps({k: r[k] for k in ("events", "handover_t", "score", "exit", "video")}, sort_keys=True)
    if out.is_file() and stamp.is_file() and stamp.read_text() == key:
        return "kept"
    w, h = 900, H
    # playhead x = X0 + (X1-X0) * t / T_MAX - 4 ; t = the frame's timestamp = tick * 0.05
    fc = (f"[0:v]pad={w}:{h + BAR_H}:0:0:color=#0d1117[p];[p][1:v]overlay=0:{h}[q];"
          f"[q][2:v]overlay=x='{X0}+{X1 - X0}*min(t,{T_MAX})/{T_MAX}-4':y={h + 46}:eval=frame[o]")
    tmp = out.with_suffix(".tmp.mp4")
    cmd = ["nice", "-n", "19", "taskset", "-c", "0,1", FF, "-hide_banner", "-loglevel", "error", "-y", "-i", r["video"], "-i", str(bar), "-i", str(MEDIA / "playhead.png"),
           "-filter_complex", fc, "-map", "[o]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-threads", "2", "-filter_threads", "1", "-x264-params", "threads=2", "-movflags", "+faststart", "-an", str(tmp)]
    subprocess.run(cmd, check=True); tmp.rename(out); stamp.write_text(key)
    return "rendered"


def probe_h(video):
    o = subprocess.run([FF, "-hide_banner", "-i", video], capture_output=True, text=True).stderr
    m = re.search(r"Video: .*?, (\d+)x(\d+)", o); return int(m[1]), int(m[2])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-render", action="store_true"); a = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True); head_png(MEDIA / "playhead.png")
    table = {row.split(",")[0]: row.split(",") for row in open(HERE.parent / "out/cl12_table.csv").read().splitlines()}
    hdr = table.pop("scene")
    scenes = []
    for s in S12:
        runs = {"carry": (f"{CL12}/runs/memory_carry_{s}", "CL12"), "reset": (f"{CL12}/runs/memory_reset_{s}", "CL12")}
        ic = R018_IC.format(s); runs["018b"] = (ic, "0.1.8b image-check re-run (cl12/imagecheck_018bp06)") if os.path.isdir(ic) else (R018.format(s), "0.1.8b closed-loop run (018b/closed_loop)")
        scenes.append(dict(scene=s, source="CL12", runs=runs))
    for d in sorted(glob.glob(f"{CL18}/runs/memory_carry_*")):
        s = d.rsplit("_", 1)[1]
        if s in S12 or not os.path.isfile(f"{d}/aggregate/results-summary.json") or not glob.glob(f"{d}/replay/*_annotated.mp4"):
            continue
        runs = {"carry": (d, "CL18")}
        if os.path.isfile(R018_CL18.format(s) + "/aggregate/results-summary.json"): runs["018b"] = (R018_CL18.format(s), "0.1.8b CL18 run (cl18/018bp06)")
        rd = f"{CL18}/runs/memory_reset_{s}"
        if os.path.isfile(f"{rd}/aggregate/results-summary.json") and glob.glob(f"{rd}/replay/*_annotated.mp4"): runs["reset"] = (rd, "CL18")
        scenes.append(dict(scene=s, source="CL18", runs=runs))
    order = {s: i for i, s in enumerate(FLIPPED + NEXT)}
    scenes.sort(key=lambda x: (order.get(x["scene"], 99), 0 if x["source"] == "CL12" else 1))
    log = []
    for sc in scenes:
        sc["arms"] = {}
        for arm, (run, src) in sc["runs"].items():
            r = run_info(run, arm, sc["scene"]); r["source"] = src
            if sc["source"] == "CL12":                 # same score/exit as the CL12 table
                col = {"carry": "carry", "reset": "reset", "018b": "018bp06"}[arm]; row = dict(zip(hdr, table[sc["scene"]]))
                assert abs(float(row[f"score_{col}"]) - r["score"]) < 1e-9 and row[f"exit_{col}"] == r["exit"], (sc["scene"], arm, row[f"exit_{col}"], r["exit"])
            sc["arms"][arm] = r
        t0s = {r["t0_us"] for r in sc["arms"].values()}; assert len(t0s) == 1, (sc["scene"], t0s)
        hos = {round(r["handover_t"], 6) for r in sc["arms"].values()}; assert len(hos) == 1, (sc["scene"], hos)
        for r in sc["arms"].values():
            w, h = probe_h(r["video"]); assert w == BAR_W, (r["video"], w)
            r["video_h"] = h
            st = "skipped (--no-render)" if a.no_render else render(r, h)
            log.append(dict(scene=sc["scene"], arm=r["arm"], video=r["video"], out=r.get("media"), action=st, n_events=len(r["events"])))
            print(sc["scene"], r["arm"], st, r["score"], r["exit"], [(e["label"], round(e["rel"], 2)) for e in r["events"]], flush=True)
        del sc["runs"]
    met = metrics(scenes)
    if a.no_render:                                    # dry run: never overwrite the live page
        dry = pathlib.Path(os.environ.get("DRY_OUT", "/tmp")) / "s13_dry_metrics.json"; json.dump(met, open(dry, "w"), indent=1); print("dry run ->", dry); return
    json.dump(dict(scenes=scenes, metrics=met, render_log=log), open(OUT / "data.json", "w"), indent=1)
    (OUT / "index.html").write_text(page(scenes, met))
    print("wrote", OUT / "index.html", len(scenes), "scenes")


def page(scenes, met):
    data = json.dumps([dict(scene=s["scene"], source=s["source"], arms={k: dict(media=v.get("media"), score=v["score"], exit=v["exit"],
            short=short_exit(v), handover=v["handover_t"], events=v["events"], n_ticks=v["n_ticks"], progress=v["progress_uncapped"],
            run=v["run"], source=v["source"]) for k, v in s["arms"].items()}) for s in scenes])
    tmpl = (HERE / "s13_compare_page.html").read_text()
    return tmpl.replace("/*DATA*/[]", data).replace("/*MET*/{}", json.dumps(met))


AF = ("collision_front", "collision_lateral")


def metrics(scenes):
    """Metrics tab. Per-scene numbers from the runs themselves (run_info); cross-checked against out/cl18_table.csv (score + exit, all
    30 scenes, carry and 0.1.8b) and out/cl18_table.json (18-scene verdict). 0.1.7 score/exit read from out/cl18_table.csv (earlier image)."""
    import statistics as st
    T = {r["scene"]: r for r in __import__("csv").DictReader(open(HERE.parent / "out/cl18_table.csv"))}
    J = json.load(open(HERE.parent / "out/cl18_table.json"))
    T12 = {r["scene"]: r for r in __import__("csv").DictReader(open(HERE.parent / "out/cl12_table.csv"))}
    rows = []
    for sc in scenes:
        c, b = sc["arms"]["carry"], sc["arms"].get("018b")
        assert b is not None, sc["scene"]
        t = T[sc["scene"]]
        for r, col in ((c, "carry"), (b, "018bp06")):
            assert abs(float(t[f"score_{col}"]) - r["score"]) < 1e-9 and t[f"exit_{col}"] == r["exit"], (sc["scene"], col)
        arm = lambda r: dict(score=r["score"], exit=r["exit"], short=short_exit(r), window_end_rel=r["window_end_rel"],   # noqa: E731
                             events=[dict(kind=e["kind"], label=e["label"], rel=e["rel"], after_window=e["after_window"]) for e in r["events"]],
                             summary=r["summary"], atfault=r["exit"].startswith("at-fault"), offroad=r["exit"].startswith("off-road"))
        row = dict(scene=sc["scene"], block="12" if sc["source"] == "CL12" else "18", carry=arm(c), b018=arm(b),
                   s017=float(t["score_017"]), exit017=t["exit_017"], diff=c["score"] - b["score"])
        if t.get("calls"):                                    # 18 new: out/cl18_table.csv
            row["shadow"] = dict(calls=int(float(t["calls"])), differ_final=int(float(t["n_differ_final"])))
        elif sc["scene"] in T12:                              # 12 diagnosis: out/cl12_table.csv (carry run)
            row["shadow"] = dict(calls=int(float(T12[sc["scene"]]["calls_carry"])), differ_final=int(float(T12[sc["scene"]]["n_differ_final_carry"])))
        rows.append(row)

    def agg(R):
        out = {}
        for k, key in (("carry", "carry"), ("018b", "b018")):
            A = [r[key] for r in R]
            out[k] = dict(mean=st.mean(a["score"] for a in A), atfault=[r["scene"] for r in R if r[key]["atfault"]],
                          offroad=[r["scene"] for r in R if r[key]["offroad"]],
                          # runs whose exit (first event after handover) falls after the score window closed: invisible to the score
                          exits_after_window=[r["scene"] for r in R if r[key]["events"] and r[key]["events"][0]["rel"] > 0 and r[key]["events"][0]["after_window"]],
                          window_end_median=st.median(a["window_end_rel"] for a in A if a["window_end_rel"] > 0),
                          perfect=sum(a["score"] >= 1 - 1e-9 for a in A), zero=sum(a["score"] <= 1e-9 for a in A),
                          dist=st.mean(a["summary"]["dist_traveled_m"] for a in A),
                          progress=st.mean(min(a["summary"]["progress"], 1.0) for a in A),
                          dist_gt=st.mean(a["summary"]["dist_to_gt_trajectory"] for a in A),
                          min_obst=st.median(a["summary"]["min_distance_to_obstacle_m"] for a in A),
                          wrong_lane=sum(a["summary"]["wrong_lane"] > 0 for a in A),
                          plan_dev=st.mean(a["summary"]["plan_deviation"] for a in A))
        out["017"] = dict(mean=st.mean(r["s017"] for r in R), atfault=[r["scene"] for r in R if r["exit017"].startswith("at-fault")],
                          offroad=[r["scene"] for r in R if r["exit017"].startswith("off-road")])
        D = sorted(R, key=lambda r: -r["diff"]); top2 = [r["scene"] for r in D[:2]]
        out.update(n=len(R), d=st.mean(r["diff"] for r in R), top2=top2, d_wo_top2=st.mean(r["diff"] for r in R if r["scene"] not in top2),
                   wins=[r["scene"] for r in R if r["diff"] > 0.1], losses=[r["scene"] for r in R if r["diff"] < -0.1],
                   ties=sum(abs(r["diff"]) <= 0.1 for r in R), identical=sum(abs(r["diff"]) < 1e-9 for r in R))
        sh = [r["shadow"] for r in R if "shadow" in r]
        if sh: out["shadow_pct"] = 100.0 * sum(s["differ_final"] for s in sh) / sum(s["calls"] for s in sh)
        return out
    B = {k: agg([r for r in rows if k == "30" or r["block"] == k]) for k in ("12", "18", "30")}
    b18 = B["18"]; added = [s for s in b18["carry"]["atfault"] if s not in b18["018b"]["atfault"]]
    verdict = "HURTS" if (b18["d"] <= -0.08 or added) else ("HELPS" if b18["d"] >= 0.08 else "NO DETECTABLE EFFECT")
    assert verdict == J["verdict"] and abs(b18["d"] - J["d"]) < 1e-12 and added == J["atfault_carry_not_018bp06"], (verdict, J["verdict"])
    assert abs(B["30"]["carry"]["mean"] - J["mean_30"]["carry"]) < 1e-12 and abs(B["30"]["018b"]["mean"] - J["mean_30"]["018bp06"]) < 1e-12

    R = {r["scene"]: r for r in rows}
    def crash(s, key):
        e = next(e for e in R[s][key]["events"] if e["kind"] in AF)
        return e["rel"], R[s][key]["window_end_rel"]
    late = [(s, *crash(s, "carry")) for s in added]
    ins = []
    ins.append(dict(h="On the 18 new scenes the two drivers score almost the same",
        p=f"Mean score carry {b18['carry']['mean']:.4f} vs 0.1.8b {b18['018b']['mean']:.4f}, a gap of {b18['d']:+.3f}, well inside the ±0.08 the rule "
          f"needs to call an effect. {b18['identical']} of 18 scenes score exactly the same for both. Carry is ahead by more than 0.1 on "
          f"{len(b18['wins'])} ({', '.join(b18['wins'])}) and behind on {len(b18['losses'])}. Without the two scenes most in carry's favour "
          f"({', '.join(b18['top2'])}) the gap is {b18['d_wo_top2']:+.3f}."))
    ins.append(dict(h="The HURTS verdict comes from two crashes the score never sees",
        p="Carry has at-fault front collisions that 0.1.8b doesn't on " + "; ".join(
            f"{s} at +{rel:.2f} s, {rel - we:.2f} s after its score window closed (+{we:.2f} s)" for s, rel, we in late) +
          ". Both runs still score 1.000. The pre-registered rule counts any at-fault exit, with no time limit, so the verdict is HURTS. "
          f"0.1.8b's own crash on the 18 ({', '.join(b18['018b']['atfault'])}) also comes after its score window closed. On the 18, crash or off-road "
          f"exits that come after the score window closed: carry {len(b18['carry']['exits_after_window'])} ({', '.join(b18['carry']['exits_after_window'])}), "
          f"0.1.8b {len(b18['018b']['exits_after_window'])} ({', '.join(b18['018b']['exits_after_window'])}). Under a rule that only counted crashes inside the window, the verdict would be "
          "NO DETECTABLE EFFECT. That reading came after the results, so it does not replace the verdict."))
    xo = [s for s in b18["carry"]["offroad"] if s not in b18["018b"]["offroad"]]
    xo_after = all(R[s]["carry"]["events"][0]["after_window"] for s in xo)
    ins.append(dict(h="Carry leaves the road more often, but late" if xo_after else "Carry leaves the road more often",
        p=f"Off-road exits on the 18: carry {len(b18['carry']['offroad'])} ({', '.join(b18['carry']['offroad'])}), 0.1.8b {len(b18['018b']['offroad'])} "
          f"({', '.join(b18['018b']['offroad'])}). 791d68ce is off-road before handover in every arm (scene start, not the driver). " +
          (f"Carry's extra off-road exits ({', '.join(xo)}) all happen after its score window closes, so they cost nothing in the score. " if xo_after else
           f"Carry's extra off-road exits: {', '.join(xo)}. ") + "They are worth watching in the Replays tab."))
    ins.append(dict(h="The 30-scene lead comes from the 12 diagnosis scenes",
        p=f"30-scene mean carry {B['30']['carry']['mean']:.4f} vs 0.1.8b {B['30']['018b']['mean']:.4f} ({B['30']['d']:+.3f}). The gap is {B['12']['d']:+.3f} "
          f"on the 12 diagnosis scenes, mostly from {', '.join(B['12']['wins'])}, but only {b18['d']:+.3f} on the 18 "
          f"new scenes. This run was the first time memory was tested on the 18, so they are the fairer test, and there the gap almost disappears. At-fault exits over 30: carry {len(B['30']['carry']['atfault'])}, "
          f"0.1.8b {len(B['30']['018b']['atfault'])}."))
    ins.append(dict(h="Memory changes the pick on about 1 call in 8, and the outcome rarely",
        p=f"In carry runs the shadow branch, which runs the reset memory on the same input, picks a different final plan on "
          f"{B['18']['shadow_pct']:.1f} % of calls on the 18 ({B['12']['shadow_pct']:.1f} % on the 12). Yet {b18['identical']} of the 18 scenes end with "
          "exactly the same score, so the memory nudges the driving but seldom decides how a scene ends."))
    ins.append(dict(h="The score saturates, so look past it",
        p=f"On the 18, carry scores 1.000 on {b18['carry']['perfect']} and 0.1.8b on {b18['018b']['perfect']}. The score window is cut by the first "
          f"collision or by drifting 4 m from the human path. That often happens early: the median window ends {b18['carry']['window_end_median']:.1f} s "
          f"after handover for carry and {b18['018b']['window_end_median']:.1f} s for 0.1.8b, out of about 16.8 s. Carry's window closes sooner, and its mean distance from the human path is "
          f"{b18['carry']['dist_gt']:.2f} m against {b18['018b']['dist_gt']:.2f} m: it drives its own line more. The other numbers in the table (distance, closest approach to obstacles, lateral error) "
          "come from the simulator's own summary and are a rough guide to driving quality, not part of the rule."))
    return dict(rows=rows, blocks=B, verdict=verdict, added=added,
                rule="carry − 0.1.8b ≥ +0.08 with no extra at-fault crash = HELPS; ≤ −0.08 or an extra at-fault crash = HURTS; otherwise NO DETECTABLE EFFECT",
                insights=ins)


if __name__ == "__main__":
    main()
