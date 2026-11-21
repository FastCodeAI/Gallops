#!/usr/bin/env python
"""Full-drive re-score (reports/FULLDRIVE_RESCORE_PREREG.md). Read-only on the runs; no simulator.

Old score  = AlpaSim eval's own aggregation, exactly as the run did it: DEFAULT_MODIFIERS (window start at eval_relevant, cut after the first
             collision/off-road tick) + RemoveTimestepsAfterEvent(dist_to_gt_trajectory >= max_dist_to_gt_trajectory) (eval/aggregation/main.py:135),
             then scene_score.score_rollout. Must equal results-summary.json rollouts[0].score to 1e-9 (else the drive gets no full-drive number).
Full drive = the same two functions with both window-END cuts removed (the start cut and everything else unchanged).
Events     = every episode (onset..last tick) of offroad / collision_front / collision_lateral / collision_rear > 0 over the whole metrics.parquet,
             times in s after handover (replay trace: csv t0 + handover_t_s, as s13_compare_page.py / stepb.exit_of).
Thresholds = read from each run's own eval-config.yaml (asserted equal to the code defaults used here).

    cd <alpasim> && nice -n 19 taskset -c 0,1 .venv/bin/python -B <this> [--only 017]
"""
import argparse, glob, json, os, pathlib, re, sys, time

import polars as pl
import yaml

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / "out"
ALPA = pathlib.Path("{paddock_sid}/Alpasim/alpasim/src/eval/src")
sys.path.insert(0, str(ALPA))
import eval.aggregation.processing as P                                         # noqa: E402
from eval.aggregation.modifiers import RemoveTimestepsAfterEvent                # noqa: E402
from eval.aggregation.scene_score import score_rollout                          # noqa: E402
from eval.schema import SceneScoreConfig                                        # noqa: E402

G = pathlib.Path("{working_notes}/LORA Refiner/Generator-Joint-PAI-alpasim")
CL12, CL18 = "{experiment_store}/runs/cl12", "{experiment_store}/runs/cl18"
R018 = "{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_{}"
R018_IC = f"{CL12}/imagecheck_018bp06/runs/drivejepa_018bp06_{{}}"
R17 = "{dataset_store}/s5_trial_017/runs/drivejepa_017_{}"
S12 = [l.split()[0] for l in open(G / "s10_018b_closed_loop/tables/scenes_12.txt") if l.strip()]
S18 = [l.split()[0] for l in open(OUT / "cl18_scenes_18.txt") if l.strip()]
EVENTS = [("offroad", "off-road"), ("collision_front", "front collision"), ("collision_lateral", "side collision"), ("collision_rear", "hit from behind")]
CODE_MAX_DIST = 4.0                                                             # wizard/configs/base_config.yaml:402
ORIG_DEFAULT = list(P.DEFAULT_MODIFIERS)


def drives():
    for s in S12:
        yield "carry", s, "12", f"{CL12}/runs/memory_carry_{s}"
        yield "reset", s, "12", f"{CL12}/runs/memory_reset_{s}"
        yield "018b", s, "12", R018_IC.format(s) if os.path.isdir(R018_IC.format(s)) else R018.format(s)
    for s in S18:
        yield "carry", s, "18", f"{CL18}/runs/memory_carry_{s}"
        yield "018b", s, "18", f"{CL18}/018bp06/runs/drivejepa_018bp06_{s}"
    for s in S12 + S18:
        yield "017", s, "12" if s in S12 else "18", R17.format(s)


def run_cfg(run):
    """Thresholds the run itself used (its eval-config.yaml), checked against what this script applies."""
    c = yaml.safe_load(open(run / "eval-config.yaml"))
    c = c.get("eval", c)
    md = float(c["aggregation_modifiers"]["max_dist_to_gt_trajectory"])
    ss = c.get("scene_score", {}) or {}
    cfg = SceneScoreConfig(**{k: ss[k] for k in ("enabled", "progress_saturation_threshold", "min_gt_distance_for_full_score_m") if k in ss})
    return md, cfg


def aggregate(df, full, max_dist, cfg):
    if full:   # both window-END cuts removed: drop RemoveTimestepsAfterEvent from the defaults, no dist cut
        P.DEFAULT_MODIFIERS[:] = [m for m in ORIG_DEFAULT if not isinstance(m, RemoveTimestepsAfterEvent)]
        mods = []
    else:
        P.DEFAULT_MODIFIERS[:] = ORIG_DEFAULT
        mods = [RemoveTimestepsAfterEvent(pl.col("dist_to_gt_trajectory") >= max_dist)]
    try:
        r = P.aggregate_and_write_metrics_results_txt(df, force_same_run=True, output_path=None, additional_modifiers=mods, scene_score_config=cfg)
    finally:
        P.DEFAULT_MODIFIERS[:] = ORIG_DEFAULT
    (row,) = r.df_wide_avg_t.to_dicts()
    res = score_rollout(row, cfg)
    win = r.df_wide_modified["timestamps_us"]
    return dict(score=res.score, failure=res.failure_reason, progress=row.get("progress_clipped_rel"),
                collision_at_fault=row.get("collision_at_fault"), offroad=row.get("offroad"), win_start_us=int(win.min()), win_end_us=int(win.max()))


def one(arm, s, block, run):
    run = pathlib.Path(run)
    rec = dict(arm=arm, scene=s, block=block, run=str(run))
    summ = json.load(open(run / "aggregate/results-summary.json"))
    (ro,) = summ["rollouts"]; rec["old_score_file"] = float(ro["score"])
    df = pl.read_parquet(glob.glob(str(run / "rollouts/clipgt-*/*/metrics.parquet")))
    if df["timestamps_us"].dtype != pl.Int64:      # Int128 in newer files (memory note make-replays-int128); value-preserving cast
        rec["ts_cast"] = str(df["timestamps_us"].dtype); df = df.with_columns(pl.col("timestamps_us").cast(pl.Int64))
    max_dist, cfg = run_cfg(run); rec["max_dist_run_cfg"] = max_dist
    assert max_dist == CODE_MAX_DIST, (run, max_dist)
    old = aggregate(df, False, max_dist, cfg); full = aggregate(df, True, max_dist, cfg)
    rec["reproduces"] = abs(old["score"] - rec["old_score_file"]) < 1e-9
    rec.update({f"old_{k}": v for k, v in old.items()}); rec.update({f"full_{k}": v for k, v in full.items()})
    # clock: replay trace (csv t0 + handover)
    (tr,) = glob.glob(str(run / "replay/*_trace.json")); h = json.load(open(tr))["header"]
    t0 = int(re.search(r"timestamp_us (\d+)", h["t_s_basis"])[1]); ho = h["handover_t_s"]
    rel = lambda us: (us - t0) / 1e6 - ho                                       # noqa: E731
    rec["handover_t"] = ho; rec["last_tick_rel"] = rel(int(df["timestamps_us"].max()))
    rec["old_win_end_rel"] = rel(old["win_end_us"]); rec["full_win_end_rel"] = rel(full["win_end_us"]); rec["win_start_rel"] = rel(old["win_start_us"])
    w = df.pivot(values="values", index="timestamps_us", on="name").sort("timestamps_us")
    d4 = w.filter(pl.col("dist_to_gt_trajectory") >= max_dist)["timestamps_us"]
    rec["first_dist4_rel"] = rel(int(d4.min())) if len(d4) else None
    ev = []
    for nm, lab in EVENTS:
        if nm not in w.columns: continue
        ts, vs = w["timestamps_us"].to_list(), w[nm].to_list(); start = prev = None
        for t, v in zip(ts, vs):
            if v is not None and v > 0:
                start = t if start is None else start; prev = t
            elif start is not None:
                ev.append((start, prev, nm, lab)); start = None
        if start is not None: ev.append((start, prev, nm, lab))
    ev.sort()
    rec["events"] = [dict(kind=nm, label=lab, rel=rel(a), rel_end=rel(b), in_old_window=a <= old["win_end_us"]) for a, b, nm, lab in ev]
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
    out, t = [], time.time()
    for arm, s, block, run in drives():
        if a.only and arm != a.only: continue
        try:
            r = one(arm, s, block, run)
        except Exception as e:                      # reported, never silently dropped
            r = dict(arm=arm, scene=s, block=block, run=run, error=f"{type(e).__name__}: {e}")
        out.append(r)
        print(arm, s, "ERR " + r["error"] if "error" in r else f"old {r['old_score_file']:.4f} repro {r['reproduces']} full {r['full_score']:.4f} "
              f"win_end +{r['old_win_end_rel']:.2f} last +{r['last_tick_rel']:.2f} ev {[(e['label'], round(e['rel'], 2)) for e in r['events']]}", flush=True)
    tag = f"_{a.only}" if a.only else ""
    json.dump(out, open(OUT / f"fulldrive_rescore{tag}.json", "w"), indent=1, default=str)
    print(f"{len(out)} drives, {sum('error' in r for r in out)} errors, {sum(not r.get('reproduces', False) for r in out if 'error' not in r)} not reproduced, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
