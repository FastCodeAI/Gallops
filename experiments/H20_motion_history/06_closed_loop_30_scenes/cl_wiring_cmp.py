#!/usr/bin/env python
"""Closed loop, step 3, wiring proof: arm A's step-0 snapshot (runs/arm_A/ckpt/p00.0.pt) through adapters/drivejepa_a4 on e3ed554f vs
p06's recorded run there (NASG/018b/closed_loop/runs/drivejepa_018bp06_e3ed554f) and vs today's p06 re-run on the same image (approach 3's
image check). Rule: score equal AND every per-tick metrics row bit-identical (same rule as the image check). Writes
adapters/drivejepa_a4/launch/tables/wiring_proof.json (the runner refuses the arm lanes without "pass": true) and MH/out/wiring_proof.json."""
import glob, json, pathlib
import numpy as np, polars as pl

MH = pathlib.Path(__file__).resolve().parents[1]
NEW = "{approach4_store}/cl/runs/a4_step0A_e3ed554f"
REFS = {"recorded_p06": "{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_e3ed554f",
        "p06_rerun_this_image": "{h17_bulk}/runs/new40/imagecheck/018bp06/runs/drivejepa_018bp06_e3ed554f"}
RUN_IDS = ("rollout_id", "run_uuid", "run_name")


def load(run):
    f = glob.glob(f"{run}/rollouts/clipgt-*/*/metrics.parquet"); assert len(f) == 1, (run, f)
    m = pl.read_parquet(f[0]).with_columns(pl.col("timestamps_us").cast(pl.Float64)).drop(list(RUN_IDS))
    return m.select(sorted(m.columns)).sort(["name", "timestamps_us"])


def eq(a, b):
    return a.columns == b.columns and a.height == b.height and all(
        np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) if a[c].dtype.is_numeric() else a[c].to_list() == b[c].to_list()
        for c in a.columns)


res = {"new": NEW}
ok = pathlib.Path(f"{NEW}/aggregate/results-summary.json").is_file()
if ok:
    mn = load(NEW); sn = json.load(open(f"{NEW}/aggregate/results-summary.json"))["rollouts"][0]["score"]
    for k, ref in REFS.items():
        mo = load(ref); so = json.load(open(f"{ref}/aggregate/results-summary.json"))["rollouts"][0]["score"]
        bit = eq(mo, mn)
        res[k] = {"ref": ref, "score_ref": so, "score_new": sn, "rows": (mo.height, mn.height), "per_tick_bit_identical": bit}
        ok &= bool(bit and so == sn)
    wl = open(f"{NEW}/driver_worker.log").read()
    res["worker_logged_step0_snapshot"] = "arm A eval_hides False step 0" in wl and "p00.0.pt" in wl
    ok &= res["worker_logged_step0_snapshot"]
res["pass"] = bool(ok)
for p in (pathlib.Path("{paddock_sid}/Alpasim/adapters/drivejepa_a4/launch/tables/wiring_proof.json"),
          MH / "out" / "wiring_proof.json"):
    p.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, default=str))
print("WIRING PROOF", "PASS" if ok else "FAIL")
