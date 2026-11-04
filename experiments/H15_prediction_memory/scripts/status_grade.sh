#!/bin/bash
# One line: graded moments per batch, rate, NAS read rate, ETA (Pacific), running or not, last log line.
WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$WT"
exec "$WT/.venv/bin/python" - <<'PY'
import json, os, re, subprocess, time
import numpy as np
from gallops.config import SiteConfig
cache = SiteConfig.load().resolve("{data.navtrain.cache}")
out = f"{cache}/grades"
running = subprocess.run(["pgrep", "-f", r"^[^ ]*/bin/python -m gallops.memory.grade build"],
                         capture_output=True).returncode == 0
logs = sorted((f for f in os.listdir(f"{cache}/logs") if f.startswith("grade_")),
              key=lambda f: os.path.getmtime(f"{cache}/logs/{f}"))
last = open(f"{cache}/logs/{logs[-1]}").read().strip().splitlines()[-1] if logs else ""
if not os.path.exists(f"{out}/done.npy"):
    print(f"grades: not started ({'running' if running else 'stopped'})"); raise SystemExit
done = np.load(f"{out}/done.npy")
row = np.load(f"{out}/row.npy")
t = np.load(f"{cache}/ticks/moments.npz")
sel = set(open("experiments/H15_prediction_memory/selection_logs.txt").read().split())
logs_of = t["log"][row]
is_val = t["split"][row] == "val"
is_sel = np.isin(logs_of, list(sel)) & ~is_val
parts = []
for name, m in (("val", is_val), ("selection", is_sel), ("rest", ~is_val & ~is_sel)):
    parts.append(f"{name} {int(done[m].sum()):,}/{int(m.sum()):,}")
p = json.load(open(f"{out}/progress.json")) if os.path.exists(f"{out}/progress.json") else {}
secs, n = p.get("seconds_this_run", 0), p.get("graded_this_run", 0)
rate = n / secs if secs else 0
mbps = p.get("metric_cache_bytes_this_run", 0) / 1e6 / secs if secs else 0
os.environ["TZ"] = "America/Los_Angeles"; time.tzset()
batch = p.get("batch", "?")
left = {"val": is_val, "selection": is_sel}.get(batch, ~is_val & ~is_sel)
todo = int((~done[left]).sum())
eta = time.strftime("%H:%M %Z", time.localtime(time.time() + todo / rate)) if rate and running else "-"
print(f"grades {' | '.join(parts)} | batch {batch} {rate:.1f}/s, NAS {mbps:.1f} MB/s, ETA {eta} | "
      f"{'running' if running else 'stopped'} | {last[:70]}")
PY
