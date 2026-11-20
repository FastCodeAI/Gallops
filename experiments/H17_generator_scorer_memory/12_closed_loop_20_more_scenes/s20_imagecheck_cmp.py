#!/usr/bin/env python
"""CL20 simulator-image check (reports/CL20_PREREG.md, step 0b). Copy of s12_imagecheck_cmp.py with the rule unchanged, extended to the
two carry drives as pre-registered:
  0.1.8b pass 0.6 on e3ed554f 8023dce5 d2715023 04749bb9  vs  generator_joint_pai/018b/closed_loop/runs/drivejepa_018bp06_<s>
  carry on bae463d3 vs cl12/runs/memory_carry_bae463d3;  carry on 0245ff75 vs cl18/runs/memory_carry_0245ff75
Rule: score equal to 3 decimals AND every per-tick metrics.parquet row bit-identical (run identifiers excluded, rows sorted by name and
timestamp, timestamps cast to float64). PASS only if all 6 match. Writes out/cl20_imagecheck.json (local disk)."""
import glob, json, pathlib, subprocess
import numpy as np, polars as pl

HERE = pathlib.Path(__file__).resolve().parent
IC = "{local_store}/runs/new40/imagecheck"
PAIRS = [(s, "0.1.8b", f"{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_{s}",
          f"{IC}/018bp06/runs/drivejepa_018bp06_{s}") for s in ("e3ed554f", "8023dce5", "d2715023", "04749bb9")] + [
    ("bae463d3", "carry", "{experiment_store}/runs/cl12/runs/memory_carry_bae463d3", f"{IC}/runs/memory_carry_bae463d3"),
    ("0245ff75", "carry", "{experiment_store}/runs/cl18/runs/memory_carry_0245ff75", f"{IC}/runs/memory_carry_0245ff75")]
RUN_IDS = ("rollout_id", "run_uuid", "run_name")   # identify the run itself, so they differ between any two runs; not metrics


def load(run):
    f = glob.glob(f"{run}/rollouts/clipgt-*/*/metrics.parquet")
    m = pl.read_parquet(f[0]).with_columns(pl.col("timestamps_us").cast(pl.Float64)).drop(list(RUN_IDS))
    return m.select(sorted(m.columns)).sort(["name", "timestamps_us"])


def diffs(mo, mn):   # where they differ, for the report if FAIL
    if mo.columns != mn.columns or mo.height != mn.height:
        return {"columns_orig": mo.columns, "columns_new": mn.columns, "rows": (mo.height, mn.height)}
    bad = [c for c in mo.columns if not (np.array_equal(mo[c].to_numpy(), mn[c].to_numpy(), equal_nan=True) if mo[c].dtype.is_numeric() else mo[c].to_list() == mn[c].to_list())]
    out = {}
    for c in bad:
        a, b = mo[c].to_numpy(), mn[c].to_numpy()
        idx = np.where(~((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b))[0]
        out[c] = dict(n_rows=int(len(idx)), first_names=mo["name"][idx[:5].tolist()].to_list() if len(idx) else [])
    return out


out, ok = {}, True
for s, arm, o, n in PAIRS:
    k = f"{arm} {s}"
    so = json.load(open(f"{o}/aggregate/results-summary.json"))["rollouts"][0]
    if not pathlib.Path(f"{n}/aggregate/results-summary.json").is_file():
        out[k] = {"rerun": "missing"}; ok = False; continue
    sn = json.load(open(f"{n}/aggregate/results-summary.json"))["rollouts"][0]
    mo, mn = load(o), load(n)
    same_cols = mo.columns == mn.columns
    bit = bool(same_cols and mo.height == mn.height and all(
        np.array_equal(mo[c].to_numpy(), mn[c].to_numpy(), equal_nan=True) if mo[c].dtype.is_numeric() else mo[c].to_list() == mn[c].to_list() for c in mo.columns))
    r = dict(orig=so["score"], rerun=sn["score"], equal_3dp=round(so["score"], 3) == round(sn["score"], 3), orig_status=so.get("status"),
             rerun_status=sn.get("status"), metric_rows=(mo.height, mn.height), columns_equal=same_cols, per_tick_bit_identical=bit)
    if not bit:
        r["differences"] = diffs(mo, mn)
    out[k] = r; ok &= bool(r["equal_3dp"] and bit)
img = lambda ref: subprocess.run(["docker", "image", "inspect", ref, "--format", "{{.Id}} {{.Created}} {{.RepoDigests}}"], capture_output=True, text=True).stdout.strip()  # noqa: E731
res = {"rule": "all 6: score equal to 3 dp AND per-tick metrics bit-identical -> PASS; else STOP, drive nothing",
       "image": img("alpasim-base:0.134.0"), "renderer": img("nvcr.io/nvidia/nre/nre-ga:26.04"), "drives": out, "pass": ok}
json.dump(res, open(HERE.parent / "out/cl20_imagecheck.json", "w"), indent=1, default=str)
for k, r in out.items():
    print(k, r)
print("IMAGE CHECK", "PASS" if ok else "FAIL -> STOP, drive nothing")
