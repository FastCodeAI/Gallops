"""navtest EPDMS table: run_pdm_score CSVs -> CSV score column (no human filter) and EPDMS with NAVSIM's human filter
(the CSV's per-metric columns, which are human-filtered, recombined: gates product x weighted mean of the soft terms,
EC dropped where blank) = the 19 Oct number (run 2 85.27, released 87.78), plus every sub-score mean. Numbers only.
usage: python3 navtest_table.py name=csv [name=csv ...]"""
import csv
import math
import sys

SUMMARY = {"average_all_frames", "pseudo_closed_loop", "average_expert_frames"}
GATES = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
SOFT = [("ego_progress", 5), ("time_to_collision_within_bound", 5), ("lane_keeping", 2), ("history_comfort", 2),
        ("two_frame_extended_comfort", 2)]
SHORT = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def num(x):
    return float(x) if x not in ("", None) else float("nan")


def filtered(r):
    g = math.prod(num(r[c]) for c in GATES)
    sv = [(num(r[c]), w) for c, w in SOFT if not math.isnan(num(r[c]))]
    return g * sum(v * w for v, w in sv) / sum(w for _, w in sv)


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs)


print(f"{'run':<22}{'scenes':>7}{'EPDMS(hf)':>10}{'CSV':>8}" + "".join(f"{s:>7}" for s in SHORT))
for arg in sys.argv[1:]:
    name, p = arg.split("=", 1)
    rows = [r for r in csv.DictReader(open(p)) if r["token"] not in SUMMARY and r["valid"] == "True"]
    cols = GATES + [c for c, _ in SOFT]
    print(f"{name:<22}{len(rows):>7}{100 * mean([filtered(r) for r in rows]):>10.2f}{100 * mean([num(r['score']) for r in rows]):>8.2f}"
          + "".join(f"{mean([num(r[c]) for r in rows]):>7.3f}" for c in cols))
