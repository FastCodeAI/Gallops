"""What a stage needs from the shared host before it starts, checked before it takes it.

Stdlib only and valid Python 3.9: the generator stage runs in Drive-JEPA's own interpreter.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

#: Beyond the feature cache itself: the 5 Oct generator runs measured 26 GB RSS each with 22 GB of features.
OVERHEAD_BYTES = 4 * 10**9


def available_bytes(meminfo: str = "/proc/meminfo") -> int:
    with open(meminfo) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise SystemExit(f"no MemAvailable in {meminfo}: cannot check free RAM before loading the cache")


def cache_bytes(cache: Path) -> int:
    """What the generator stage holds in RAM: every feat_*.npy of the cache, train and held-out."""
    return sum(os.path.getsize(f) for f in glob.glob(str(cache / "feat_*.npy")))


def require_free_ram(need: int, what: str, meminfo: str = "/proc/meminfo") -> None:
    """Refuse to start below `need` bytes of available RAM. Two cache-in-RAM trainers at once pushed this box
    into full swap on 5 Oct, with both stalled for ~13 minutes and everyone else's jobs with them. A check at
    start cannot see a job that starts later, so it is a floor, not a reservation."""
    have = available_bytes(meminfo)
    if have < need:
        raise SystemExit(
            f"refusing: {what} needs about {need / 1e9:.0f} GB of RAM and {have / 1e9:.0f} GB is available"
        )
