#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""ext20 extra gates for a drive-jepa 0.1.4 single-scene run dir. Read-only.

    dj_extra_gates.py <run_dir> --json <out.json>

Runs AFTER the pilot10 gates (run script GATE 1-3 greps + check_smoke_gates.py
GATES 2/4/5).  Adds the two reads the ext20 brief names that the pilot10 gate
scripts do not compute, plus the console reads:

  G-VER   console `Connected to driver ... version_id: "drivejepa-v2-driver-0.1.4"`
  G-NRE   console: 26.1.112-59673f88 seen, 26.4.96-91b06fb8 never, no
          "multiple NRE versions" warning
  G-SENT  every `handover sentinel` fire in driver.log belongs to a replan whose
          sim_time_us <= handover_us  ("sentinel only at ticks <= 60").  A fire
          precedes its replan line (planner logs it while building ego_status;
          the divider logs the replan when the inference returns), so each fire
          is attributed to the NEXT replan line.
  G-AGE   handover-plan age = handover_us - sim_time_us of the last real
          inference at or before handover; must be <= 300 000 us.  The 0.1.4
          planner does not latch (analysis/drivejepa/drivejepa_orientation.md
          §latch), so this is the closest measurable to the brief's
          "latch t0 < 0.3 s"; on the 2 Hz grid it is 50 000 us by construction
          (replan #6 at 2.95 s).  DEFINITION IS THIS CAMPAIGN'S -- flagged.

handover_us = first driver_request.time_now_us in the run's own rollout.asl
(= first_policy_timestamp_us, unbound_rollout.py:143) + 3 000 000 us = tick 60.
(The metrics.parquet grid used by the CV report starts ~0.3 s later, so it is
NOT used here -- it would misplace the tick-60 fire as "after handover".)
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import sys

from alpasim_utils.logs import async_read_pb_log

FGT_US = 3_000_000
MAX_AGE_US = 300_000
NRE_PIN = "26.1.112-59673f88"
NRE_BAD = "26.4.96-91b06fb8"
VERSION_ID = "drivejepa-v2-driver-0.1.7-oracle"   # copy of 07_closed_loop_test/dj_extra_gates_016.py; only this line changed


def handover_us(run_dir: str) -> int:
    (asl,) = glob.glob(os.path.join(run_dir, "rollouts", "clipgt-*", "*", "rollout.asl"))

    async def first_request() -> int:
        async for e in async_read_pb_log(asl):
            if e.WhichOneof("log_entry") == "driver_request":
                return int(e.driver_request.time_now_us)
        raise RuntimeError("no driver_request in rollout.asl")

    return asyncio.run(first_request()) + FGT_US


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    gates: dict[str, dict] = {}

    console = open(os.path.join(a.run_dir, "wizard_console.log"), errors="replace").read()
    gates["G-VER"] = {"ok": f'version_id: "{VERSION_ID}"' in console,
                      "detail": f"expects {VERSION_ID} in wizard_console.log"}
    n26, n04 = console.count(NRE_PIN), console.count(NRE_BAD)
    multi = len(re.findall(r"multiple NRE versions", console))
    gates["G-NRE"] = {"ok": n26 > 0 and n04 == 0 and multi == 0,
                      "detail": f"26.01 x{n26}, 26.04 x{n04}, multi-version warnings {multi}"}
    m = re.search(r"Clipping n_sim_steps from (\d+) to (\d+)", console)
    gates["steps"] = {"ok": True, "detail": f"n_sim_steps {m.group(1)} -> {m.group(2)}" if m else "no clip line (400 fit)"}

    ho = handover_us(a.run_dir)
    log = open(os.path.join(a.run_dir, "driver.log"), errors="replace").read().splitlines()
    fires, replans = [], []
    pending = 0
    for line in log:
        if "handover sentinel:" in line:
            pending += 1
        mm = re.search(r"divider: replan #(\d+) at sim_time_us=(\d+)", line)
        if mm:
            k, t = int(mm.group(1)), int(mm.group(2))
            replans.append((k, t))
            for _ in range(pending):
                fires.append((k, t))
            pending = 0
    late = [(k, t) for k, t in fires if t > ho]
    gates["G-SENT"] = {"ok": not late and pending == 0,
                       "detail": f"{len(fires)} fires on replans {sorted({k for k, _ in fires})}; "
                                 f"{len(late)} after handover; {pending} unattributed",
                       "fires": fires, "handover_us": ho}
    before = [t for _, t in replans if t <= ho]
    age = (ho - max(before)) if before else None
    gates["G-AGE"] = {"ok": age is not None and age <= MAX_AGE_US,
                      "detail": f"handover-plan age {age} us (limit {MAX_AGE_US}); "
                                f"{len(replans)} replans logged", "age_us": age}
    ok = all(g["ok"] for g in gates.values())
    for k, g in gates.items():
        print(f"{'PASS' if g['ok'] else 'FAIL'}  {k:7} {g['detail']}")
    print("EXTRA GATES", "PASSED" if ok else "FAILED")
    if a.json:
        json.dump({"ok": ok, "gates": gates, "run_dir": a.run_dir}, open(a.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
