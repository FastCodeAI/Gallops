#!/usr/bin/env python
"""Closed loop, step 5: score tables for MH/reports/CL_ARM_A.md (PREREG_APPROACH4.md section 2).
Score = windowed, rollouts[0].score of each run's aggregate/results-summary.json. p06's per-scene baseline = the PREREG table.
  cl_scores.py preview   arm A on the 12 diagnosis scenes vs p06 (recorded, never judged)
  cl_scores.py final     arm A and arm B on all 30, means, p06's 0.7527, per-scene flips, and the gate:
                         PASS iff mean(A) >= 0.8194 AND mean(A) > mean(B)   (PREREG section 2; the line = 0.7527 + 0.0667)
A "flip" here = a scene whose score moves between 0 and 1 against p06 (|change| = 1); other changes are listed as partial.
Appends to the report and prints the table. Run with any python 3 (json only)."""
import json
import sys
from pathlib import Path

MH = Path(__file__).resolve().parents[1]
RUNS = Path("{approach4_store}/cl/runs")
REPORT = MH / "reports" / "CL_ARM_A.md"
P06 = {"e3ed554f": 1.0, "8023dce5": 0.0, "5f064b02": 1.0, "e121e37d": 1.0, "048b974e": 1.0, "cdd9448c": 1.0, "d2715023": 1.0,
       "bd07fb91": 0.0, "bae463d3": 0.379891, "d1a25a99": 0.0, "b1b170d0": 0.723395, "04749bb9": 0.0,
       "01d503d4": 1.0, "023b7fcc": 1.0, "0245ff75": 1.0, "026d6a39": 1.0, "02e075b9": 1.0, "02eadd92": 1.0, "032b6f21": 1.0,
       "04394343": 1.0, "16b83346": 1.0, "1acf2f5d": 0.834999, "2e1ce314": 1.0, "6725df2d": 1.0, "71d7d03b": 1.0, "791d68ce": 0.0,
       "89f2eeb8": 1.0, "98dd56b2": 0.791024, "acd22430": 0.0, "b44a4a44": 0.851624}
S12, S18 = list(P06)[:12], list(P06)[12:]
P06_MEAN, LINE = 0.7527, 0.8194


def score(tag, s):
    f = RUNS / f"a4_{tag}_{s}" / "aggregate" / "results-summary.json"
    if not f.is_file():
        return None
    r = json.load(open(f))["rollouts"][0]
    return float(r["score"])


def kind(new, old):
    if new is None:
        return "missing"
    d = new - old
    if abs(d) < 1e-9:
        return "same"
    if abs(abs(d) - 1.0) < 1e-9:
        return "FLIP up" if d > 0 else "FLIP down"
    return f"partial {d:+.3f}"


def mean(xs):
    return sum(xs) / len(xs)


def table(scenes, tags):
    head = "| # | scene | set | p06 | " + " | ".join(f"arm {t[-1]} | vs p06" for t in tags) + " |"
    sep = "|---" * (4 + 2 * len(tags)) + "|"
    rows = [head, sep]
    for s in scenes:
        n = list(P06).index(s) + 1
        cells = []
        for t in tags:
            v = score(t, s)
            cells += ["—" if v is None else f"{v:.6f}", kind(v, P06[s])]
        rows.append(f"| {n} | {s} | {'12' if s in S12 else '18'} | {P06[s]:.6f} | " + " | ".join(cells) + " |")
    return rows


def main(mode):
    L = []
    if mode == "preview":
        a = [score("armA", s) for s in S12]
        L += ["", "## Preview: arm A on the 12 diagnosis scenes (recorded, never judged)", ""] + table(S12, ["armA"])
        if None not in a:
            fl = [kind(v, P06[s]) for v, s in zip(a, S12)]
            L += ["", f"Arm A mean over the 12: **{mean(a):.4f}** vs p06 0.5919. Flips up {sum(k == 'FLIP up' for k in fl)}, "
                  f"flips down {sum(k == 'FLIP down' for k in fl)}, partial {sum(k.startswith('partial') for k in fl)}, same {fl.count('same')}."]
    else:
        L += ["", "## Results: arms A and B on all 30", ""] + table(S12 + S18, ["armA", "armB"])
        res = {}
        for t in ("armA", "armB"):
            v = [score(t, s) for s in P06]
            res[t] = None if None in v else {"mean": mean(v), "mean12": mean(v[:12]), "mean18": mean(v[12:]),
                                              "kinds": [kind(x, P06[s]) for x, s in zip(v, P06)]}
        L.append("")
        L.append("| | mean 30 | mean 12 | mean 18 | flips up | flips down | partial | same |")
        L.append("|---|---|---|---|---|---|---|---|")
        L.append("| p06 (baseline) | 0.7527 | 0.5919 | 0.8599 | | | | |")
        for t, r in res.items():
            if r is None:
                L.append(f"| arm {t[-1]} | incomplete | | | | | | |"); continue
            k = r["kinds"]
            L.append(f"| arm {t[-1]} | **{r['mean']:.4f}** | {r['mean12']:.4f} | {r['mean18']:.4f} | {k.count('FLIP up')} | {k.count('FLIP down')} | "
                     f"{sum(x.startswith('partial') for x in k)} | {k.count('same')} |")
        if res["armA"] and res["armB"]:
            A, B = res["armA"]["mean"], res["armB"]["mean"]
            c1, c2 = A >= LINE, A > B
            L += ["", f"**Gate (PREREG section 2): PASS iff mean(A) ≥ 0.8194 AND mean(A) > mean(B).** mean(A) = {A:.4f} "
                  f"({'≥' if c1 else '<'} 0.8194: {'yes' if c1 else 'no'}); mean(A) {'>' if c2 else '≤'} mean(B) = {B:.4f}: {'yes' if c2 else 'no'}. "
                  f"**Gate: {'PASS' if c1 and c2 else 'FAIL'}.**",
                  f"Difference vs p06: arm A {A - P06_MEAN:+.4f} ({(A - P06_MEAN) * 30:+.1f} scene-equivalents), arm B {B - P06_MEAN:+.4f}."]
    with open(REPORT, "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
