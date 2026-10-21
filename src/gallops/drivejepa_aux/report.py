"""The aux runs side by side: what the plan's comparisons say, at what uncertainty.

    python -m gallops.drivejepa_aux.report {runs}/drivejepa_aux/runs/<run> ... -o report.md

Reads each run's summary.json, heldout.npz and metrics.jsonl (`train.py`). numpy only.

What it puts side by side (docs/drivejepa-aux-heads.md, step 4):
- every run's kept epoch on the held-out scenes, and the release's (R), which every run measures
  first and must agree on;
- **T vs C** for each seed (H-gain: do the heads help?) and **every arm vs R** (H-safe, in-domain
  only: the gate is paddock's), as paired differences over the same held-out rows, each with its
  scene-clustered standard error, beside **C's range across its seeds** -- the yardstick the plan's
  decision rule holds a T-vs-C difference to;
- the heads against their no-camera baselines (do they learn at all?), and how hard they pulled on
  the refiner over the run, and which way.

It reports; it does not decide. The plan's H-gain rule also needs the shuffled-label arm S and the
sight check, and H-safe needs paddock's gate: none of that is here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gallops.drivejepa_aux import metrics

#: The plan's measurements, each as (key in summary.json, row function on heldout.npz, label, unit
#: scale for printing). Lower is better for every one.
PLANNING = (
    ("l2_1s", lambda m: m["l2"][:, metrics.L2_WAYPOINTS["1s"]], "L2 @1 s (m)", 1.0),
    ("l2_2s", lambda m: m["l2"][:, metrics.L2_WAYPOINTS["2s"]], "L2 @2 s (m)", 1.0),
    ("l2_3s", lambda m: m["l2"][:, metrics.L2_WAYPOINTS["3s"]], "L2 @3 s (m)", 1.0),
    ("min_ade_3s", lambda m: m["min_ade"], "best of 32, ADE 3 s (m)", 1.0),
    ("collision_3s", lambda m: m["collided"].astype(float), "collision 3 s (%)", 100.0),
    ("off_road_3s", lambda m: m["off_road"], "off road 3 s (%)", 100.0),
)
ORDER = {"C": 0, "T1": 1, "T2": 2}


def load_run(directory: Path) -> dict:
    directory = Path(directory)
    npz = np.load(directory / "heldout.npz", allow_pickle=False)
    measured = {"release": {}, "kept": {}}
    for key in npz.files:
        which, _, name = key.partition("_")
        measured[which][name] = npz[key]
    pulls = []
    for line in (directory / "metrics.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["kind"] == "pull":
            pulls.append(entry)
    summary = json.loads((directory / "summary.json").read_text())
    return {"name": directory.name, "summary": summary, "measured": measured, "pulls": pulls}


def _fmt(mean, se=None, scale: float = 1.0, digits: int = 3) -> str:
    if mean is None or not np.isfinite(mean):
        return "--"
    text = f"{mean * scale:.{digits}f}"
    return text if se is None or not np.isfinite(se) else f"{text} ± {se * scale:.{digits}f}"


def _signed(mean, se, scale: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    return (
        f"{mean * scale:+.{digits}f} ± {se * scale:.{digits}f}"
        if np.isfinite(se)
        else f"{mean * scale:+.{digits}f}"
    )


def check_release(runs: list) -> list[str]:
    """Every run measured the same release on the same rows; say where one did not."""
    problems = []
    first = runs[0]["measured"]["release"]
    for run in runs[1:]:
        other = run["measured"]["release"]
        if not np.array_equal(first["rows"], other["rows"]):
            problems.append(f"{run['name']} measured different held-out rows")
        elif float(np.abs(first["plan"] - other["plan"]).max()) > 1e-3:
            problems.append(f"{run['name']} measured a different release plan")
    return problems


def paired_row(a: dict, b: dict) -> list:
    """Each planning measurement's mean a - b over the rows both measured, scene-clustered."""
    if not np.array_equal(a["rows"], b["rows"]):
        raise SystemExit("paired comparison over different rows")
    return [metrics.paired(f(a), f(b), a["scene"]) for _, f, _, _ in PLANNING]


def seed_range(runs: list, arm: str) -> list:
    """Max - min across an arm's seeds of each planning measurement's kept mean (NaN under 2 seeds)."""
    means = [
        [
            metrics.clustered(f(r["measured"]["kept"]), r["measured"]["kept"]["scene"])[0]
            for _, f, _, _ in PLANNING
        ]
        for r in runs
        if r["summary"]["arm"] == arm
    ]
    if len(means) < 2:
        return [float("nan")] * len(PLANNING)
    means = np.asarray(means)
    return list(means.max(0) - means.min(0))


def verdict(per_seed: list, span: float) -> str:
    """The plan's bar for one measurement: every seed's T - C difference beyond C's seed range and
    beyond twice its own standard error, all the same way. Lower is better."""
    if not np.isfinite(span):
        return "not resolved (C has fewer than two seeds)"
    if all(m < 0 and -m > span and -m > 2 * se for m, se, _ in per_seed):
        return "better, clears the bar"
    if all(m > 0 and m > span and m > 2 * se for m, se, _ in per_seed):
        return "worse, clears the bar"
    return "not resolved"


def render(runs: list) -> str:
    runs = sorted(runs, key=lambda r: (ORDER.get(r["summary"]["arm"], 9), r["summary"]["seed"]))
    out = ["# Drive-JEPA auxiliary heads: held-out results", ""]
    problems = check_release(runs)
    release = runs[0]["measured"]["release"]
    n_rows, n_scenes = len(release["rows"]), len(np.unique(release["scene"]))
    out += [
        f"Held-out: {n_rows} keyframes in {n_scenes} nuScenes train scenes the runs never trained on "
        "(recipes/drivejepa_aux/heldout_scenes.json). Every value is a mean over keyframes ± its standard "
        "error clustered by scene. The same scenes chose each run's epoch, so these numbers carry that "
        "selection; paddock's benchmarks are the clean test.",
        "",
    ]
    if problems:
        out += ["**The runs disagree about the release:** " + "; ".join(problems), ""]

    # every run's kept epoch, and the release
    header = (
        ["run", "arm", "seed", "kept epoch"]
        + [label for _, _, label, _ in PLANNING]
        + ["sight: max move (m)"]
    )
    out += [
        "## Planning on the held-out scenes",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    summary0 = runs[0]["summary"]["release"]
    cells = ["release (R)", "R", "--", "--"]
    cells += [_fmt(summary0[k]["mean"], summary0[k]["se"], scale) for k, _, _, scale in PLANNING]
    cells.append(_fmt(summary0["sight"].get("max_move_m")))
    out.append("| " + " | ".join(cells) + " |")
    for run in runs:
        s = run["summary"]
        kept = s["kept"]
        cells = [run["name"], s["arm"], str(s["seed"]), str(s["kept_epoch"])]
        cells += [_fmt(kept[k]["mean"], kept[k]["se"], scale) for k, _, _, scale in PLANNING]
        cells.append(_fmt(kept["sight"].get("max_move_m")))
        out.append("| " + " | ".join(cells) + " |")
    human = runs[0]["summary"]["baselines"]
    collision = _fmt(human["human_collision_3s"]["mean"], None, 100, 2)
    off_road = _fmt(human["human_off_road_3s"]["mean"], None, 100, 2)
    out += [
        "",
        f"The human's own logged path scores collision {collision}% and off road {off_road}% on these "
        "metrics: their floor (annotation and map error, and open-loop boxes that do not react).",
        "",
    ]

    # paired differences
    header = ["comparison", "seed"] + [label for _, _, label, _ in PLANNING]
    out += [
        "## Paired differences (negative: the first is better)",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    by_arm_seed = {(r["summary"]["arm"], r["summary"]["seed"]): r for r in runs}
    seeds = sorted({r["summary"]["seed"] for r in runs})
    lines = []
    for seed in seeds:
        control = by_arm_seed.get(("C", seed))
        for arm in ("T1", "T2"):
            treated = by_arm_seed.get((arm, seed))
            if treated and control:
                diffs = paired_row(treated["measured"]["kept"], control["measured"]["kept"])
                lines.append((f"{arm} − C", seed, diffs))
        for arm in ("C", "T1", "T2"):
            run = by_arm_seed.get((arm, seed))
            if run:
                diffs = paired_row(run["measured"]["kept"], run["measured"]["release"])
                lines.append((f"{arm} − R", seed, diffs))
    scales = [scale for _, _, _, scale in PLANNING]
    for name, seed, diffs in lines:
        cells = [name, str(seed)] + [_signed(m, se, scales[k]) for k, (m, se, _) in enumerate(diffs)]
        out.append("| " + " | ".join(cells) + " |")
    span = seed_range(runs, "C")
    ranges = " | ".join(_fmt(v, None, scales[k]) for k, v in enumerate(span))
    out += [f"| C's range across seeds | -- | {ranges} |", ""]

    # the plan's H-gain test, per measurement: beats C by more than C's seed range and 2 SE
    out += [
        "The plan's bar for a T-vs-C difference: larger than C's range across its seeds **and** than twice "
        "its scene-clustered standard error, in the same direction on every seed.",
        "",
    ]
    for arm in ("T1", "T2"):
        verdicts = []
        for k, (_, _, label, _) in enumerate(PLANNING):
            per_seed = [d[k] for name, _, d in lines if name == f"{arm} − C"]
            if per_seed:
                verdicts.append(f"{label}: {verdict(per_seed, span[k])}")
        if verdicts:
            out.append(f"- **{arm} vs C:** " + "; ".join(verdicts) + ".")
    out.append("")

    # the heads
    heads = [r for r in runs if r["summary"]["arm"] != "C"]
    if heads:
        base = heads[0]["summary"]["baselines"]
        out += [
            "## Do the heads learn? (held-out, against a model that never looks)",
            "",
            "| run | BEV mIoU | BEV IoU: lane divider / vehicle / road | box F1 | box centre error (m) "
            "| lead present acc. | lead gap MAE (m) | lane change F1 left / right |",
            "|---|---|---|---|---|---|---|---|",
        ]
        prior = base["bev_iou_prior"]
        miou = np.nanmean([v for v in prior.values() if v is not None])
        out.append(
            f"| no camera | {miou:.3f} | {_fmt(prior.get('lane_divider'))} / {_fmt(prior.get('vehicle'))} / "
            f"{_fmt(prior.get('road'))} | 0 | -- | {_fmt(base['lead_present_accuracy_never'])} | "
            f"{_fmt(base['lead_gap_mae_train_mean_m'], None, 1, 2)} | "
            f"{_fmt(base['lane_f1_always_keep']['left'])} / {_fmt(base['lane_f1_always_keep']['right'])} |"
        )
        for run in heads:
            k = run["summary"]["kept"]
            iou = k.get("bev_iou", {})
            m = np.nanmean([v for v in iou.values() if v is not None]) if iou else float("nan")
            lane = k.get("lane_f1", {})
            # a run that kept the release kept heads that never trained: say so, not a score
            name = run["name"] + (
                " (kept epoch 0: untrained heads)" if run["summary"]["kept_epoch"] == 0 else ""
            )
            out.append(
                f"| {name} | {_fmt(m)} | {_fmt(iou.get('lane_divider'))} / {_fmt(iou.get('vehicle'))} / "
                f"{_fmt(iou.get('road'))} | {_fmt(k.get('box_f1'))} | "
                f"{_fmt(k.get('box_centre_error_m'), None, 1, 2)} | "
                f"{_fmt(k.get('lead_present_accuracy'))} | {_fmt(k.get('lead_gap_mae_m'), None, 1, 2)} | "
                f"{_fmt(lane.get('left'))} / {_fmt(lane.get('right'))} |"
            )
        out += ["", "## How hard the heads pulled on the refiner, and which way", ""]
        out += [
            "| run | measurements | aux / planning gradient: first, median, last "
            "| cosine: first, median, last |",
            "|---|---|---|---|",
        ]
        for run in heads:
            pulls = run["pulls"]
            if not pulls:
                continue
            ratio = np.array([p["ratio"] for p in pulls], dtype=float)
            cosine = np.array([p["cosine"] for p in pulls], dtype=float)
            ratios = f"{ratio[0]:.3f}, {np.nanmedian(ratio):.3f}, {ratio[-1]:.3f}"
            cosines = f"{cosine[0]:+.3f}, {np.nanmedian(cosine):+.3f}, {cosine[-1]:+.3f}"
            out.append(f"| {run['name']} | {len(pulls)} | {ratios} | {cosines} |")
        out.append("")

    # the selection objective itself, per arm: no per-row values are kept, so a spread across seeds only
    out += [
        "## The selection objective: held-out planning loss at the kept epoch",
        "",
        "| arm | runs | kept epochs | loss at the kept epoch: each seed | mean | range |",
        "|---|---|---|---|---|---|",
    ]
    release_loss = runs[0]["summary"]["history"][0]["held_out"]["planning"]
    out.append(f"| R | -- | -- | {release_loss:.4f} | {release_loss:.4f} | -- |")
    for arm in ("C", "T1", "T2"):
        mine = [r for r in runs if r["summary"]["arm"] == arm]
        if not mine:
            continue
        kept = [r["summary"]["history"][r["summary"]["kept_epoch"]]["held_out"]["planning"] for r in mine]
        epochs = ", ".join(str(r["summary"]["kept_epoch"]) for r in mine)
        each = ", ".join(f"{v:.4f}" for v in kept)
        spread = f"{max(kept) - min(kept):.4f}" if len(kept) > 1 else "--"
        out.append(f"| {arm} | {len(mine)} | {epochs} | {each} | {np.mean(kept):.4f} | {spread} |")
    out.append("")

    # the curves
    out += ["## Held-out planning loss by epoch (epoch 0 is the release)", ""]
    epochs = max(len(r["summary"]["history"]) for r in runs)
    out += ["| run | " + " | ".join(str(e) for e in range(epochs)) + " |", "|" + "---|" * (epochs + 1)]
    for run in runs:
        values = [h["held_out"]["planning"] for h in run["summary"]["history"]]
        kept = run["summary"]["kept_epoch"]
        cells = [f"**{v:.4f}**" if e == kept else f"{v:.4f}" for e, v in enumerate(values)]
        out.append(f"| {run['name']} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories ({runs}/drivejepa_aux/runs/<name>)")
    ap.add_argument("-o", "--out", type=Path, help="write the markdown here (default: print it)")
    args = ap.parse_args()
    text = render([load_run(d) for d in args.runs])
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
