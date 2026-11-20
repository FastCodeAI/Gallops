#!/usr/bin/env python
"""Closed loop, step 2 (PREREG_APPROACH4.md section 2, image precondition): p06 on the 8 named scenes vs the recorded runs.
Rule = approach 3's s20_imagecheck_cmp.py, unchanged: score equal to 3 decimals AND every per-tick metrics.parquet row bit-identical
(run identifiers excluded, rows sorted by name and timestamp, timestamps cast to float64). PASS only if all 8 match.
  reused (cl20ic, Sid's terms 2026-11-20): e3ed554f 8023dce5 d2715023 04749bb9 from approach 3's image check
  driven here: 791d68ce 026d6a39 02e075b9 89f2eeb8
Recorded: the 12-scene set from NASG/018b/closed_loop/runs, the 18-scene set from approach3/runs/cl18/018bp06/runs (PREREG table).
Run with the alpasim venv (polars). Usage: cl_imagecheck_cmp.py [scene ...]  (default all 8). Writes MH/out/cl_imagecheck.json."""
import glob, json, pathlib, subprocess, sys
import numpy as np, polars as pl

HERE = pathlib.Path(__file__).resolve().parent
REC12 = "{retrain_store}/018b/closed_loop/runs"
REC18 = "{experiment_store_h17}/runs/cl18/018bp06/runs"
NEW_A3 = "{h17_bulk}/runs/new40/imagecheck/018bp06/runs"
NEW_A4 = "{approach4_store}/cl/imagecheck/runs"
SCENES = {"e3ed554f": (REC12, NEW_A3), "8023dce5": (REC12, NEW_A3), "d2715023": (REC12, NEW_A3), "04749bb9": (REC12, NEW_A3),
          "791d68ce": (REC18, NEW_A4), "026d6a39": (REC18, NEW_A4), "02e075b9": (REC18, NEW_A4), "89f2eeb8": (REC18, NEW_A4)}
RUN_IDS = ("rollout_id", "run_uuid", "run_name")


def load(run):
    f = glob.glob(f"{run}/rollouts/clipgt-*/*/metrics.parquet")
    assert len(f) == 1, (run, f)
    m = pl.read_parquet(f[0]).with_columns(pl.col("timestamps_us").cast(pl.Float64)).drop(list(RUN_IDS))
    return m.select(sorted(m.columns)).sort(["name", "timestamps_us"])


def col_eq(a, b, c):
    return np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) if a[c].dtype.is_numeric() else a[c].to_list() == b[c].to_list()


which = sys.argv[1:] or list(SCENES)
out, ok = {}, True
for s in which:
    rec_root, new_root = SCENES[s]
    o, n = f"{rec_root}/drivejepa_018bp06_{s}", f"{new_root}/drivejepa_018bp06_{s}"
    if not pathlib.Path(f"{n}/aggregate/results-summary.json").is_file():
        out[s] = {"rerun": "missing", "new": n}; ok = False; continue
    so = json.load(open(f"{o}/aggregate/results-summary.json"))["rollouts"][0]
    sn = json.load(open(f"{n}/aggregate/results-summary.json"))["rollouts"][0]
    mo, mn = load(o), load(n)
    same_cols = mo.columns == mn.columns
    bit = bool(same_cols and mo.height == mn.height and all(col_eq(mo, mn, c) for c in mo.columns))
    r = dict(recorded=o, new=n, orig=so["score"], rerun=sn["score"], equal_3dp=round(so["score"], 3) == round(sn["score"], 3),
             metric_rows=(mo.height, mn.height), columns_equal=same_cols, per_tick_bit_identical=bit)
    if not bit and same_cols and mo.height == mn.height:
        r["columns_differing"] = [c for c in mo.columns if not col_eq(mo, mn, c)]
    out[s] = r; ok &= bool(r["equal_3dp"] and bit)
img = lambda ref: subprocess.run(["docker", "image", "inspect", ref, "--format", "{{.Id}} {{.Created}}"], capture_output=True, text=True).stdout.strip()  # noqa: E731
res = {"rule": "score equal to 3 dp AND per-tick metrics bit-identical, on every scene listed", "scenes": which,
       "image": img("alpasim-base:0.134.0"), "renderer": img("nvcr.io/nvidia/nre/nre-ga:26.04"), "drives": out, "pass": ok}
name = "cl_imagecheck.json" if len(which) == len(SCENES) else f"cl_imagecheck_{'_'.join(which)}.json"
json.dump(res, open(HERE.parent / "out" / name, "w"), indent=1, default=str)
for k, r in out.items():
    print(k, {x: r.get(x) for x in ("orig", "rerun", "equal_3dp", "metric_rows", "per_tick_bit_identical", "columns_differing", "rerun")})
print("IMAGE CHECK", "PASS" if ok else "FAIL", res["image"])
