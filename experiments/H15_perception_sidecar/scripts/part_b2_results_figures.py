"""H15 Part B, B-2's results: what the runs measured, drawn for the owner's review.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/part_b2_results_figures.py

For docs/drivejepa-perception-sidecar/part-b/B-2-hidden-refiner.md, "What has run". Nothing trains and nothing
is scored here: it reads B-2's records under {runs}/drivejepa_perception/part-b/b2 (the navtest reads,
R-clean, nuScenes, the swaps and each run's telemetry) and draws
- b2r-chain.png: the pre-registered contrasts, per seed, ± 2 SE, over their predicted ranges;
- b2r-navtest.png: every run minus R on navtest, per seed;
- b2r-subscores.png: each arm's difference from R, sub-score by sub-score;
- b2r-training.png: the training telemetry, every run;
- b2r-health.png: the health reads, which choose nothing: the pick's vendored final on select, and B-1's value
  on R-clean with C's seed range (the noise floor);
- b2r-readouts.png: the vendored readouts of each run's 32 on navtest, against R's;
- b2r-nusc.png: nuScenes, each run against the release.
Each draws what has been read so far (phase 1 alone, or both phases). Every number behind a chart is written
to ../figures/part-b2-results-numbers.json; the numbers are copied from the records, never re-derived.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("GALLOPS_CONFIG", str(REPO / "gallops.toml"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import part_b_explained_figures as px  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gallops.drivejepa_perception import b2_read, views  # noqa: E402

OUT, FIGURES = px.OUT, px.FIGURES
B2 = OUT / "part-b/b2"
RUNS = B2 / "runs"
NUMBERS: dict = {}
#: The arms in the reference palette's first six categorical slots, in this fixed order (validated together on
#: the light surface with the dataviz skill's validator: every check passes; three slots are under 3:1 against
#: the surface, so every line is labelled directly and every chart has a legend).
ARMS = ("C", "O", "D", "HD", "H6", "H3")
ARM_COLOURS = dict(zip(ARMS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))  # noqa: B905
VERSIONS = {"1": "B-2-1 · joint", "2": "B-2-2 · two stages, LoRA", "3": "B-2-3 · two stages, full"}
#: PREDICTIONS.md, Part B · B-2 (committed before any run): (row label, the pairs' key prefix, low, high).
CHAIN = (
    ("K0 · the gate: O − C", "O-C", 0.0, 0.4),
    ("K1 · the primary: HD − D", "HD-D", -0.1, 0.2),
    ("K2 · HD − C", "HD-C", -0.1, 0.3),
    ("K3 · HD − R", "vs_r:1:HD", -0.4, 0.3),
    ("K4 · H6 − HD", "H6-HD", -0.3, 0.1),
    ("K5 · H3 − H6 (a pilot)", "H3-H6", -0.1, 0.1),
    ("D − C (reported)", "D-C", -0.1, 0.3),
    ("C − R (reported)", "vs_r:1:C", -0.5, 0.3),
    ("O − R (the ceiling; no prediction)", "vs_r:1:O", None, None),
    ("HD, B-2-2 − B-2-1 (descriptive)", "HD B-2-2 - B-2-1", -0.3, 0.2),
    ("HD, B-2-3 − B-2-1 (descriptive)", "HD B-2-3 - B-2-1", -0.4, 0.2),
)
MEETS = -0.4
SUBSCORES = ("NC", "DAC", "DDC", "TTC", "EP", "LK", "HC", "EC")
SEED_OFFSET = 0.22


# ------------------------------------------------------------------------------------------ the data


def reads() -> dict:
    """Both phases' navtest records, merged: vs_r and pairs by name, readouts by run."""
    merged = {"vs_r": {}, "pairs": {}, "readouts": {}, "verdicts": {}, "phases": []}
    for phase in ("1", "2"):
        path = B2 / f"navtest-phase{phase}" / "navtest.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        merged["phases"].append(phase)
        for key in ("vs_r", "pairs", "readouts"):
            merged[key].update(r[key])
        merged["verdicts"][phase] = r["verdicts"]
    return merged


def parse(run: str) -> tuple:
    """b2-<v>-<arm>-s<seed> -> (v, arm, seed)."""
    _, v, arm, seed = run.split("-")
    return v, arm, int(seed[1:])


def contrast_points(r: dict, key: str) -> list:
    """[(seed, diff, se)] for a CHAIN key: a pair prefix, or vs_r:<v>:<arm>."""
    out = []
    if key.startswith("vs_r:"):
        _, v, arm = key.split(":")
        for run, p in sorted(r["vs_r"].items()):
            rv, ra, seed = parse(run)
            if (rv, ra) == (v, arm):
                out.append((seed, p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]))
        return out
    for name, p in sorted(r["pairs"].items()):
        label, seed = name.rsplit(" s", 1)
        if label == key:
            out.append((int(seed), p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]))
    return out


# ------------------------------------------------------------------------------------------ the charts


def forest(ax, rows: list, xlim: tuple) -> None:
    """rows: (label, points [(seed, diff, se)], low, high, bold). Seeds stacked within a row, top down."""
    labels = []
    y = len(rows) - 1
    for label, points, lo, hi, bold in rows:
        if lo is not None:
            ax.plot([lo, hi], [y, y], color="#86b6ef", lw=9, solid_capstyle="butt", zorder=2, alpha=0.9)
        for seed, d, se in points:
            yy = y + SEED_OFFSET * (1 - seed) if seed < 3 else y - SEED_OFFSET * (seed - 1.5)
            ax.plot([d - 2 * se, d + 2 * se], [yy, yy], color=views.INK_2, lw=1.2, zorder=3)
            ax.plot(d, yy, marker="o", ms=5.5, color=views.INK, mec=views.SURFACE, mew=1.2, zorder=4)
        if points:
            right = max(d + 2 * se for _, d, se in points)
            text = "  ".join(f"{d:+.2f}" for _, d, _ in points)
            ax.text(min(right + 0.03, xlim[1] - 0.02), y, text, va="center", fontsize=7, color=views.INK_2)
        else:
            ax.text(xlim[0] + 0.02, y, "not run", va="center", fontsize=7, color=views.GREY, style="italic")
        labels.append((y, label, bold))
        y -= 1
    ax.axvline(0, color=views.INK_2, lw=1, zorder=1)
    ax.set_yticks([yy for yy, _, _ in labels])
    ax.set_yticklabels([name for _, name, _ in labels])
    for tick, (_, _, bold) in zip(ax.get_yticklabels(), labels):  # noqa: B905 -- Python 3.9
        tick.set_fontweight("bold" if bold else "normal")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def fig_chain(r: dict) -> None:
    rows, numbers = [], {}
    for label, key, lo, hi in CHAIN:
        points = contrast_points(r, key)
        numbers[label] = {
            "predicted": None if lo is None else [lo, hi],
            "per_seed": {f"s{s}": {"diff": d, "se": se} for s, d, se in points},
        }
        rows.append((label, points, lo, hi, label.startswith("K1")))
    NUMBERS["chain"] = numbers
    NUMBERS["verdicts"] = r["verdicts"]
    lows = [d - 2 * se for _, p, _, _, _ in rows for _, d, se in p] + [-0.6]
    highs = [d + 2 * se for _, p, _, _, _ in rows for _, d, se in p] + [0.5]
    xlim = (min(lows) - 0.1, max(highs) + 0.45)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.6))
    forest(ax, rows, xlim)
    ax.axvline(MEETS, color=views.INK_2, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.text(MEETS - 0.01, len(rows) - 0.35, "meets R", fontsize=7.5, color=views.INK_2, ha="right", va="top")
    ax.legend(
        handles=[
            Line2D(
                [], [], color="#86b6ef", lw=8, label="predicted, a range (PREDICTIONS.md, before any run)"
            ),
            Line2D([], [], color=views.INK, marker="o", ms=5, lw=1.2, label="measured, one seed, ± 2 SE"),
        ],
        loc="lower right",
    )
    ax.set_xlabel(
        "navtest EPDMS (human filter), points: the named difference, paired per scene, SE clustered by log"
    )
    ax.set_title(
        "B-2's pre-registered contrasts: seeds 0, 1, 2 top to bottom within a row (B-2-1 unless named)",
        loc="left",
    )
    fig.tight_layout()
    px.save(fig, "b2r-chain.png")


def fig_navtest(r: dict) -> None:
    order = [(v, arm) for v in ("1", "2", "3") for arm in ARMS]
    rows, numbers = [], {}
    for v, arm in order:
        points = []
        for run, p in sorted(r["vs_r"].items()):
            rv, ra, seed = parse(run)
            if (rv, ra) == (v, arm):
                points.append((seed, p["epdms_hf"]["a_minus_b"], p["epdms_hf"]["se"]))
                numbers[run] = {"epdms_hf": p["epdms_hf"]["a"], "minus_R": p["epdms_hf"]["a_minus_b"]}
                numbers[run]["se"] = p["epdms_hf"]["se"]
        if points:
            rows.append((f"{arm} · {VERSIONS[v]}", points, None, None, arm == "HD"))
    NUMBERS["navtest_minus_R"] = numbers
    if not rows:
        return
    lows = [d - 2 * se for _, p, _, _, _ in rows for _, d, se in p] + [MEETS]
    highs = [d + 2 * se for _, p, _, _, _ in rows for _, d, se in p] + [0.2]
    xlim = (min(lows) - 0.1, max(highs) + 0.45)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.5))
    forest(ax, rows, xlim)
    ax.axvline(MEETS, color=views.INK_2, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.text(MEETS - 0.01, len(rows) - 0.35, "meets R", fontsize=7.5, color=views.INK_2, ha="right", va="top")
    some = next(iter(r["vs_r"].values()))
    ax.set_xlabel(
        f"navtest EPDMS (human filter) minus R's {some['epdms_hf']['b']:.3f}, points, one seed ± 2 SE"
    )
    ax.set_title("Every B-2 run against the release, on navtest (seeds 0, 1, 2 top to bottom)", loc="left")
    fig.tight_layout()
    px.save(fig, "b2r-navtest.png")


def fig_subscores(r: dict) -> None:
    groups = [(v, arm) for v in ("1", "2", "3") for arm in ARMS]
    present = [(v, arm) for v, arm in groups if any(parse(run)[:2] == (v, arm) for run in r["vs_r"])]
    if not present:
        return
    numbers = {}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharey=False)
    for ax, key in zip(axes.ravel(), SUBSCORES):  # noqa: B905 -- Python 3.9
        numbers[key] = {}
        for i, (v, arm) in enumerate(present):
            vals = []
            for run, p in sorted(r["vs_r"].items()):
                if parse(run)[:2] == (v, arm):
                    d, se = p[key]["a_minus_b"] * 100, p[key]["se"] * 100
                    vals.append(d)
                    ax.plot([i, i], [d - 2 * se, d + 2 * se], color=ARM_COLOURS[arm], lw=1.2, alpha=0.8)
                    ax.plot(i, d, marker="o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
                    numbers[key][run] = {"minus_R_pp": d, "se_pp": se}
            if vals:
                ax.plot([i - 0.3, i + 0.3], [np.mean(vals)] * 2, color=views.INK, lw=1.5)
        ax.axhline(0, color=views.INK_2, lw=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{arm}\nB-2-{v}" for v, arm in present], fontsize=7)
        ax.set_title(key, loc="left")
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("minus R, percentage points")
    NUMBERS["subscores_minus_R"] = numbers
    fig.suptitle(
        "Where each run's difference from R comes from: navtest sub-scores, one dot a seed (± 2 SE), "
        "the bar their mean",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-subscores.png")


def telemetry(run: Path) -> dict:
    lines = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines() if x.strip()]
    return {k: np.array([x.get(k, np.nan) for x in lines], float) for k in lines[0]} if lines else {}


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """A centred running mean over k telemetry points (k x 50 steps), shorter at the ends."""
    out = np.empty_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - k // 2), min(len(x), i + k // 2 + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out


def end_labels(ax, ends: dict, gap: float = 0.06) -> None:
    """Each arm's name at the right edge, at its runs' mean last value, pushed apart to `gap` (axes fraction)
    so that close arms stay legible."""
    lo, hi = ax.get_ylim()
    items = sorted((float(np.mean(v)) - lo) / (hi - lo) for v in ends.values())
    names = [a for a, _ in sorted(ends.items(), key=lambda kv: float(np.mean(kv[1])))]
    for i in range(1, len(items)):
        items[i] = max(items[i], items[i - 1] + gap)
    for name, y in zip(names, items):  # noqa: B905 -- Python 3.9
        ax.text(1.01, y, name, transform=ax.transAxes, fontsize=8, color=views.INK, va="center")


def fig_training() -> None:
    runs = sorted(p for p in RUNS.glob("b2-*") if (p / "steps.jsonl").exists())
    if not runs:
        return
    panels = (
        ("planning", "planning loss (mtd_loss)"),
        ("final", "final-score BCE"),
        ("null_share", "attention on the null token"),
    )
    joint = [p for p in runs if parse(p.name)[0] == "1"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    numbers = {}
    for run in joint:
        v, arm, seed = parse(run.name)
        t = telemetry(run)
        numbers[run.name] = {
            k: float(np.nanmean(t[k][-10:])) for k, _ in panels if k in t and np.isfinite(t[k][-10:]).any()
        }
        for ax, (key, _) in zip(axes, panels):  # noqa: B905 -- Python 3.9
            if key in t and np.isfinite(t[key]).any():
                ax.plot(t["step"], smooth(t[key]), color=ARM_COLOURS[arm], lw=1.2, alpha=0.85)
    for ax, (key, title) in zip(axes, panels):  # noqa: B905 -- Python 3.9
        ax.set_title(title, loc="left")
        ax.set_xlabel("step (batch 64; 3 epochs)")
        ends = {}
        for run in joint:
            v, arm, seed = parse(run.name)
            if key in numbers[run.name]:
                ends.setdefault(arm, []).append(numbers[run.name][key])
        if ends:
            end_labels(ax, ends)
    axes[0].legend(
        handles=[
            Line2D([], [], color=ARM_COLOURS[a], lw=2, label=a)
            for a in ARMS
            if any(parse(p.name)[1] == a for p in joint)
        ],
        loc="upper right",
        title="arm (every seed)",
    )
    NUMBERS["training_last_500_steps"] = numbers
    fig.suptitle(
        "B-2-1's training telemetry, every run: a running mean over 250 steps (each point of the log is 50)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-training.png")


def fig_health() -> None:
    """Select (vendored pick final, each run minus R on the same rows) and R-clean (B-1's value, minus the
    release's), one dot a run."""
    splits = json.loads((REPO / "recipes/drivejepa_perception/splits.json").read_text())
    labels = np.load(OUT / "labels-v1" / "labels.npz")
    logs = np.array(json.loads((OUT / "labels-v1" / "manifest.json").read_text())["logs"])[labels["log_id"]]
    select = np.flatnonzero(np.isin(logs, splits["select"]))
    cand = np.load(OUT / "pdm-v1" / "pdm.npz")["candidates"]
    pick = np.load(OUT / "tokens-v1" / "pick.npy").astype(np.int64)
    r_select = float(cand[select, pick[select], -1].mean() * 100)
    sel = {}
    for run in sorted(RUNS.glob("b2-*")):
        d = run / "dump-select"
        if (d / "labels.npz").exists():
            dump = np.load(d / "dump.npz")
            if not np.array_equal(dump["rows"], select):
                raise SystemExit(f"{d} does not hold the select rows in order")
            mine = np.load(d / "labels.npz")["candidates"][np.arange(len(select)), dump["pick"], -1]
            sel[run.name] = float(mine.mean() * 100) - r_select
    # The final read over every run, else phase 1's (which decided the noise floor; it is kept as written).
    rclean_path = next((p for p in (B2 / "rclean-all.json", B2 / "rclean.json") if p.exists()), None)
    rc = json.loads(rclean_path.read_text()) if rclean_path else None
    if not sel and rc is None:
        return
    NUMBERS["health"] = {
        "select_R_pick_final": r_select,
        "select_minus_R": sel,
        "select_rows": int(len(select)),
    }
    groups = [(v, arm) for v in ("1", "2", "3") for arm in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    panels = (
        (axes[0], sel, f"select ({len(select):,} rows): the pick's vendored final minus R's {r_select:.2f}"),
        (
            axes[1],
            {} if rc is None else {k: v["minus_release"] for k, v in rc["runs"].items()},
            "R-clean (2,624 known-command windows, city-weighted): B-1's value minus the release's",
        ),
    )
    for ax, values, title in panels:
        present = [(v, arm) for v, arm in groups if any(parse(k)[:2] == (v, arm) for k in values)]
        for i, (v, arm) in enumerate(present):
            ys = [x for k, x in sorted(values.items()) if parse(k)[:2] == (v, arm)]
            ax.plot([i] * len(ys), ys, "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
        ax.axhline(0, color=views.INK_2, lw=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{arm}\nB-2-{v}" for v, arm in present], fontsize=7)
        ax.set_xlim(-0.6, max(len(present), 1) - 0.4)
        ax.set_ylabel("points")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    if rc is not None and rc.get("noise_floor"):
        floor = rc["noise_floor"]
        axes[1].text(
            0.01,
            0.02,
            f"C's seed range {floor['c_seed_range']:.2f} (limit {floor['limit']}): "
            + ("seeds 3 and 4 run" if floor["more_seeds"] else "three seeds suffice"),
            transform=axes[1].transAxes,
            fontsize=7.5,
            color=views.INK_2,
        )
    fig.suptitle("Health reads, which choose nothing: one dot a seed", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    px.save(fig, "b2r-health.png")


def fig_readouts(r: dict) -> None:
    if not r["readouts"]:
        return
    release_pick = np.load(OUT / "tokens-navtest-v1" / "pick.npy").astype(np.int64)
    pdm = np.load(OUT / "pdm-navtest-v1" / "pdm.npz")["candidates"]
    proposals = np.load(OUT / "tokens-navtest-v1" / "proposals.npy", mmap_mode="r")
    n = np.arange(len(release_pick))
    lateral = np.asarray(proposals[:, :, -1, 1])
    reference = {
        "pick_final": float(pdm[n, release_pick, -1].mean() * 100),
        "best_of_32": float(pdm[..., -1].max(1).mean() * 100),
        "share_safe": float(((pdm[..., 0] == 1) & (pdm[..., 5] == 1)).mean() * 100),
        "lateral_spread_4s_m": float((lateral.max(1) - lateral.min(1)).mean()),
        "switch_from_release": 0.0,
    }
    NUMBERS["readouts"] = {"R": reference, **r["readouts"]}
    panels = (
        ("pick_final", "the pick's vendored final, ×100"),
        ("best_of_32", "best of the 32, vendored, ×100"),
        ("share_safe", "share of the 32 passing NC and TTC, %"),
        ("lateral_spread_4s_m", "the 32's lateral spread at 4 s, m"),
        ("switch_from_release", "share of scenes whose pick differs from R's"),
    )
    groups = [(v, arm) for v in ("1", "2", "3") for arm in ARMS]
    present = [(v, arm) for v, arm in groups if any(parse(run)[:2] == (v, arm) for run in r["readouts"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.0))
    for ax, (key, title) in zip(axes, panels):  # noqa: B905 -- Python 3.9
        for i, (v, arm) in enumerate(present):
            vals = [x[key] for run, x in sorted(r["readouts"].items()) if parse(run)[:2] == (v, arm)]
            ax.plot([i] * len(vals), vals, "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
        ax.axhline(reference[key], color=views.INK_2, lw=1, ls=(0, (3, 2)))
        ax.text(
            len(present) - 0.5, reference[key], " R", fontsize=7.5, color=views.INK_2, va="bottom", ha="right"
        )
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{arm}\nB-2-{v}" for v, arm in present], fontsize=7)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "The 32 on navtest, by the vendored labels: one dot a seed; the dashed line is the release (R)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-readouts.png")


def fig_nusc() -> None:
    path = next((p for p in (B2 / "nusc.json", B2 / "nusc-phase1.json") if p.exists()), None)
    if path is None:
        return
    r = json.loads(path.read_text())
    NUMBERS["nusc"] = r
    keys = (
        ("l2_3s", "the pick's L2 at 3 s, m"),
        ("collision_3s", "collisions within 3 s, share"),
        ("min_ade_3s", "best of 32 by ADE over 3 s, m"),
    )
    groups = [(v, arm) for v in ("1", "2", "3") for arm in ARMS]
    present = [(v, arm) for v, arm in groups if any(parse(run)[:2] == (v, arm) for run in r["runs"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (key, title) in zip(axes, keys):  # noqa: B905 -- Python 3.9
        for i, (v, arm) in enumerate(present):
            for run, x in sorted(r["runs"].items()):
                if parse(run)[:2] != (v, arm):
                    continue
                d, se = x["minus_release"][key]["mean"], x["minus_release"][key]["se"]
                ax.plot([i, i], [d - 2 * se, d + 2 * se], color=ARM_COLOURS[arm], lw=1.2, alpha=0.8)
                ax.plot(i, d, "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
        ax.axhline(0, color=views.INK_2, lw=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{arm}\nB-2-{v}" for v, arm in present], fontsize=7)
        ax.set_title(f"{title}, minus the release's {r['release'][key]:.3f}", loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        f"nuScenes, out of domain: {r['rows']:,} keyframes of {r['scenes']} scenes, one dot a seed ± 2 SE "
        "(clustered by scene); lower is better",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-nusc.png")


def main() -> None:
    views.use_style()
    r = reads()
    NUMBERS["phases_read"] = r["phases"]
    for name in ("rclean", "rclean-all"):
        if (B2 / f"{name}.json").exists():
            NUMBERS[name.replace("-", "_")] = json.loads((B2 / f"{name}.json").read_text())
    swaps = B2 / "swaps.json"
    if swaps.exists():
        NUMBERS["swaps"] = json.loads(swaps.read_text())
    if r["phases"]:
        fig_chain(r)
        fig_navtest(r)
        fig_subscores(r)
        fig_readouts(r)
    fig_training()
    fig_health()
    fig_nusc()
    NUMBERS["noise_limit"] = b2_read.NOISE_LIMIT
    (FIGURES / "part-b2-results-numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float) + "\n")
    print(f"wrote {FIGURES / 'part-b2-results-numbers.json'}")


if __name__ == "__main__":
    main()
