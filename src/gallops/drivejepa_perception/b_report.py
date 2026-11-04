"""Phase B's verdicts, by the pre-registered rule (PREDICTIONS.md; plan section 5.0).

    python -m gallops.drivejepa_perception.b_report --labels L --pdm P --runs {out}/b1 \
        --out {out}/b1/report.json

Reads every `<mode>-<arm>` run directory under --runs (train_scorer's output: report_picks.npz and
summary.json). For each mode, on the report logs:
- **F - C, per seed:** the mean paired difference in the pick's EPDMS on the same rows, and its SE
  clustered by log (navsim's independent unit; paddock's gate clusters NAVSIM by log too);
- **C's seed range:** max - min of C's mean pick EPDMS over its seeds;
- **the rule:** F beats C when, on every seed, F - C exceeds both C's seed range and twice its SE.
  NEUTRAL otherwise, meaning "not resolved at this n";
- **O and S,** where they ran, as shares of F - C (H-info: S - C <= 20% and O >= F);
- **the subsets:** F - C on each pre-registered subset (P8).

Seed k of F is paired with seed k of C, at the kept epoch, which each run chose on the select logs.
The last epoch is reported beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUBSETS = ("lead_30m", "slowing_or_stopped", "agent_within_10m")


def clustered_se(diff: np.ndarray, clusters: np.ndarray) -> float:
    """SE of mean(diff), clustered: sqrt(G / (G - 1) * sum_g (sum_i in g d_i - n_g mean d)^2) / N."""
    n = len(diff)
    groups = np.unique(clusters)
    if len(groups) < 2:
        return float("nan")
    centred = diff - diff.mean()
    sums = np.array([centred[clusters == g].sum() for g in groups])
    return float(np.sqrt(len(groups) / (len(groups) - 1) * (sums**2).sum()) / n)


def pick_scores(picks: dict, key: str, pdm: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return pdm[rows, picks[key], -1]


def compare(f: dict, c: dict, pdm, rows, logs, subsets, which: str) -> dict:
    """F - C per seed at `which` (kept / last), with SE, C's seed range, and the rule's verdict."""
    seeds = sorted(
        {k.split("_")[0] for k in f if k.startswith("seed")}
        & {k.split("_")[0] for k in c if k.startswith("seed")}
    )
    c_means = [float(pick_scores(c, f"{s}_{which}", pdm, rows).mean()) for s in seeds]
    c_range = float(max(c_means) - min(c_means)) if c_means else float("nan")
    per_seed = []
    for s in seeds:
        d = pick_scores(f, f"{s}_{which}", pdm, rows) - pick_scores(c, f"{s}_{which}", pdm, rows)
        entry = {"seed": s, "f_minus_c": float(d.mean()), "se": clustered_se(d, logs)}
        for name in SUBSETS:
            m = subsets[name]
            entry[f"f_minus_c_{name}"] = float(d[m].mean()) if m.any() else float("nan")
            entry[f"se_{name}"] = clustered_se(d[m], logs[m]) if m.sum() > 1 else float("nan")
        per_seed.append(entry)
    beats = bool(per_seed) and all(e["f_minus_c"] > max(c_range, 2 * e["se"]) for e in per_seed)
    loses = bool(per_seed) and all(-e["f_minus_c"] > max(c_range, 2 * e["se"]) for e in per_seed)
    return {
        "seeds": seeds,
        "c_means": c_means,
        "c_seed_range": c_range,
        "per_seed": per_seed,
        "verdict": "F beats C" if beats else ("C beats F" if loses else "NEUTRAL"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="gallops.drivejepa_perception.b_report")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pdm", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    labels = np.load(Path(args.labels) / "labels.npz")
    logs_all = np.array(json.loads((Path(args.labels) / "manifest.json").read_text())["logs"])[
        labels["log_id"]
    ]
    pdm = np.nan_to_num(np.load(Path(args.pdm) / "pdm.npz")["candidates"], nan=0.0)
    runs = {p.name: p for p in sorted(Path(args.runs).iterdir()) if (p / "report_picks.npz").exists()}
    picks = {name: dict(np.load(p / "report_picks.npz")) for name, p in runs.items()}
    out = {"runs": sorted(runs)}
    for mode in sorted({name.rsplit("-", 1)[0] for name in runs}):
        arms = {name.rsplit("-", 1)[1]: picks[name] for name in runs if name.rsplit("-", 1)[0] == mode}
        if "C" not in arms:
            continue
        rows = arms["C"]["rows"]
        logs = logs_all[rows]
        masks = {name: arms["C"][f"subset_{name}"].astype(bool) for name in SUBSETS}
        result = {"release_pick_epdms": float(pdm[rows, arms["C"]["release"], -1].mean())}
        for arm in ("F", "O", "S"):
            if arm in arms:
                for which in ("kept", "last"):
                    result[f"{arm}_vs_C_{which}"] = compare(
                        arms[arm], arms["C"], pdm, rows, logs, masks, which
                    )
        if "F" in arms:
            gain = np.mean([e["f_minus_c"] for e in result["F_vs_C_kept"]["per_seed"]])
            for arm in ("O", "S"):
                if arm in arms:
                    other = np.mean([e["f_minus_c"] for e in result[f"{arm}_vs_C_kept"]["per_seed"]])
                    result[f"{arm}_minus_C_share_of_F_minus_C"] = (
                        float(other / gain) if gain else float("nan")
                    )
        out[mode] = result
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
