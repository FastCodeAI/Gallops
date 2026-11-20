#!/usr/bin/env python
"""Closed-loop check of the memory: simulator-image check (pre-registration, amendment 1). Derived from LORA Refiner/Generator-Joint-PAI-alpasim/
s11_018c/src/imagecheck_cmp.py; changed: the model (0.1.8b pass 0.6 instead of 0.1.7, whose runner cannot start since 2026-11-09), the four
scenes (e3ed554f, 8023dce5, d2715023, 04749bb9: all among the 12), the folders, the output file. Rule unchanged, fixed before looking: the
re-runs on the rebuilt simulator image must equal the original runs, scores to 3 decimals AND per-tick metrics bit-identical (run
identifiers excluded, timestamps cast to float64). Writes out/cl12_imagecheck.json.
"""
import glob, json, pathlib, subprocess
import numpy as np, polars as pl

HERE = pathlib.Path(__file__).resolve().parent
ORIG = "{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_{}"
NEW = "{experiment_store}/runs/cl12/imagecheck_018bp06/runs/drivejepa_018bp06_{}"


RUN_IDS = ("rollout_id", "run_uuid", "run_name")   # identify the run itself, so they differ between any two runs; not metrics.
# The first compare (2026-11-03 05:32) included them and reported FAIL on every scene; every metric column was equal
# (kept as tables/imagecheck_1103_first_compare_with_run_ids.json).


def load(run):
    f = glob.glob(f"{run}/rollouts/clipgt-*/*/metrics.parquet")
    m = pl.read_parquet(f[0]).with_columns(pl.col("timestamps_us").cast(pl.Float64)).drop(list(RUN_IDS))
    return m.select(sorted(m.columns)).sort(["name", "timestamps_us"])


out, ok = {}, True
for s in ("e3ed554f", "8023dce5", "d2715023", "04749bb9"):
    o, n = ORIG.format(s), NEW.format(s)
    so = json.load(open(f"{o}/aggregate/results-summary.json"))["rollouts"][0]
    if not pathlib.Path(f"{n}/aggregate/results-summary.json").is_file():
        out[s] = {"rerun": "missing"}; ok = False; continue
    sn = json.load(open(f"{n}/aggregate/results-summary.json"))["rollouts"][0]
    mo, mn = load(o), load(n)
    same_cols = mo.columns == mn.columns
    bit = bool(same_cols and mo.height == mn.height and all(
        np.array_equal(mo[c].to_numpy(), mn[c].to_numpy(), equal_nan=True) if mo[c].dtype.is_numeric() else mo[c].to_list() == mn[c].to_list() for c in mo.columns))
    r = dict(orig=so["score"], rerun=sn["score"], equal_3dp=round(so["score"], 3) == round(sn["score"], 3), orig_status=so.get("status"),
             rerun_status=sn.get("status"), metric_rows=(mo.height, mn.height), columns_equal=same_cols, per_tick_bit_identical=bit)
    out[s] = r; ok &= bool(r["equal_3dp"] and bit)
res = {"rule": "0.1.8b pass 0.6 re-run: scores equal to 3 decimals and per-tick metrics bit-identical -> image accepted, existing 0.1.8b runs reused", "image": subprocess.run(["docker", "image", "inspect", "alpasim-base:0.134.0", "--format", "{{.Id}} {{.Created}}"], capture_output=True, text=True).stdout.strip(),
       "scenes": out, "accepted": ok}
json.dump(res, open(HERE.parent / "out/cl12_imagecheck.json", "w"), indent=1, default=str)
for s, r in out.items():
    print(s, r)
print("IMAGE CHECK", "PASS -> image accepted" if ok else "FAIL -> rerun 0.1.8b on the other 8 scenes; flag the 0.1.7 figure 'different simulator build'")
