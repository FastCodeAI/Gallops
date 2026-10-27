"""Stage navtest-score: the navtest numbers docs/drivejepa_wajepa.md reports, from run_pdm_score CSVs.

Two scores per run, over the valid scene rows (the three summary rows dropped):

- **CSV**: the mean of the CSV's own `score` column (no human filter).
- **EPDMS, human filter**: the CSV's per-metric columns, which navsim has already human-filtered, recombined
  per scene as the product of the four gates times the weighted mean of the soft terms (progress 5, TTC 5,
  lane keeping 2, history comfort 2, two-frame extended comfort 2), with a blank term (EC is blank where the
  previous frame was not scored) left out of that scene's mean.

This is how the numbers in the doc were made (Run 2 85.27, the released Drive-JEPA 87.78, Run 3A 87.00 /
87.30). Every sub-score mean is printed beside them. Stdlib only.

Ported from paddock-ops run3a/build/navtest_table.py, plus --json.

    python -m gallops.drivejepa_wajepa.navtest_score NAME=CSV [NAME=CSV ...] [--json out.json]

CSV may be a glob (navsim names each run's directory by its start time); the last match in sorted order is
read, and the JSON records which file that was."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math

SUMMARY = {"average_all_frames", "pseudo_closed_loop", "average_expert_frames"}
GATES = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
]
SOFT = [
    ("ego_progress", 5),
    ("time_to_collision_within_bound", 5),
    ("lane_keeping", 2),
    ("history_comfort", 2),
    ("two_frame_extended_comfort", 2),
]
SHORT = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def num(x: str | None) -> float:
    return float(x) if x not in ("", None) else float("nan")


def filtered(r: dict) -> float:
    g = math.prod(num(r[c]) for c in GATES)
    sv = [(num(r[c]), w) for c, w in SOFT if not math.isnan(num(r[c]))]
    return g * sum(v * w for v, w in sv) / sum(w for _, w in sv)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs)


def resolve(pattern: str) -> str:
    """A CSV path, or a glob over navsim's timestamped run directories: the last match in sorted order."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no CSV matches {pattern}")
    return matches[-1]


def score(path: str) -> dict:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["token"] not in SUMMARY and r["valid"] == "True"]
    cols = GATES + [c for c, _ in SOFT]
    return {
        "scenes": len(rows),
        "epdms_human_filter": 100 * mean([filtered(r) for r in rows]),
        "csv_score": 100 * mean([num(r["score"]) for r in rows]),
        "sub_scores": {s: mean([num(r[c]) for r in rows]) for s, c in zip(SHORT, cols, strict=True)},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=CSV")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = {}
    print(f"{'run':<22}{'scenes':>7}{'EPDMS(hf)':>10}{'CSV':>8}" + "".join(f"{s:>7}" for s in SHORT))
    for arg in a.runs:
        name, pattern = arg.split("=", 1)
        path = resolve(pattern)
        s = out[name] = {"csv": path, **score(path)}
        print(
            f"{name:<22}{s['scenes']:>7}{s['epdms_human_filter']:>10.2f}{s['csv_score']:>8.2f}"
            + "".join(f"{s['sub_scores'][k]:>7.3f}" for k in SHORT)
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
