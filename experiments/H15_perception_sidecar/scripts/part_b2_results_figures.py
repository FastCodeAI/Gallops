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
- b2r-nusc.png: nuScenes, each run against the release;
- b2r-where.png: where each contrast's difference comes from, scene by scene: by the ego's speed, by the
  number of agents the side-car decodes, and by whether R's pick fails NC or TTC (official, per scene);
- b2r-failures.png: R's 275 navtest scenes whose pick fails NC or TTC: how many each run fixes, by the object
  behind the failure (B-0-3's rule), and how many new failures it makes among R's passing scenes;
- b2r-hidden-scenes.jpg: the primary contrast scene by scene: where HD beats D on every paired seed, and where
  it loses on every one, with the fold side-car's slots and what HD's read attends to;
- b2r-scenes.jpg: three scenes, each the extreme of a rule in SCENE_RULES: where the fine-tune gains on
  navtest, where it loses, and an R-clean window it newly drives against the traffic direction.
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
from matplotlib.patches import Patch  # noqa: E402

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
#: What an empty row of the chain says: K0 stopped B-2 after phase 1 (PREDICTIONS.md; D29 (b)).
NOT_RUN = "not run: the phase-1 gate (K0) stopped B-2"
SUBSCORES = ("NC", "DAC", "DDC", "TTC", "EP", "LK", "HC", "EC")
SEED_OFFSET = 0.22
#: Seeds side by side within an arm's column (strip charts): seed k at x + JITTER * (k - 2).
JITTER = 0.09
BACKING = {"fc": views.SURFACE, "ec": "none", "pad": 0.8, "alpha": 0.9}


#: The scenes b2r-scenes draws, each the extreme of its rule (official per-scene scores on navtest; vendored
#: labels on R-clean, which has no official replay). C's seeds must agree, so a scene shows the fine-tune, not
#: one seed's noise.
SCENE_RULES = {
    "gain": "among navtest scenes where every C seed's official EPDMS (human filter) is above R's, the one "
    "with the largest gain on its worst seed",
    "loss": "among navtest scenes where every C seed's official EPDMS is below R's, the one with the largest "
    "loss on its best seed",
    "ddc": "among R-clean's known-command windows where the release's pick passes DDC (vendored, the human "
    "passing too) and C seeds 0 and 1 both fail it, the one with the largest city weight, then the largest "
    "loss on seed 0",
}
#: A proposal drawn red fails a gate the grader multiplies by (NC, DAC, DDC) or TTC (vendored).
GATES_DRAWN = (0, 1, 2, 5)


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
    """b2-<v>-<arm>-s<seed>, or its stage 1's b2-<v>-<arm>-s<seed>-stage1 -> (v, arm, seed)."""
    _, v, arm, seed = run.replace("-stage1", "").split("-")
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
            yy = y + SEED_OFFSET * (2 - seed) * 0.6  # seeds 0-4 top to bottom, inside the row
            ax.plot([d - 2 * se, d + 2 * se], [yy, yy], color=views.INK_2, lw=1.2, zorder=3)
            ax.plot(d, yy, marker="o", ms=5.5, color=views.INK, mec=views.SURFACE, mew=1.2, zorder=4)
        if points:
            right = max(d + 2 * se for _, d, se in points)
            text = "  ".join(f"{d:+.2f}" for _, d, _ in points)
            ax.text(
                min(right + 0.03, xlim[1] - 0.02),
                y,
                text,
                va="center",
                fontsize=7,
                color=views.INK_2,
                bbox=BACKING,
            )
        else:
            ax.text(
                xlim[0] + 0.02,
                y,
                NOT_RUN,
                va="center",
                fontsize=7,
                color=views.INK_2,
                style="italic",
                zorder=5,
                bbox=BACKING,
            )
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
    fig, axes = plt.subplots(2, 4, figsize=(max(12, 1.1 * len(present) * 4), 6.6), sharey=False)
    for ax, key in zip(axes.ravel(), SUBSCORES):  # noqa: B905 -- Python 3.9
        numbers[key] = {}
        for i, (v, arm) in enumerate(present):
            vals = []
            for run, p in sorted(r["vs_r"].items()):
                if parse(run)[:2] == (v, arm):
                    d, se = p[key]["a_minus_b"] * 100, p[key]["se"] * 100
                    vals.append(d)
                    x = i + JITTER * (parse(run)[2] - 2)
                    ax.plot([x, x], [d - 2 * se, d + 2 * se], color=ARM_COLOURS[arm], lw=1.2, alpha=0.8)
                    ax.plot(x, d, marker="o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
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
    """Row 1, B-2-1: the planning loss, the final-score BCE and the read's null share. Row 2, the two-stage
    recipes: stage 1's planning loss, stage 3's final-score BCE and its pass-4 read's null share."""
    runs = sorted(p for p in RUNS.glob("b2-*") if (p / "steps.jsonl").exists())
    if not runs:
        return
    rows = (
        (
            "B-2-1, joint",
            lambda p: parse(p.name)[0] == "1",
            (
                ("planning", "planning loss (mtd_loss)"),
                ("final", "final-score BCE"),
                ("null_share", "null share"),
            ),
        ),
        (
            "two stages: stage 1 (B-2-2 solid, B-2-3 dashed)",
            lambda p: p.name.endswith("-stage1"),
            (("planning", "stage 1: planning loss"),),
        ),
        (
            "two stages: stage 3",
            lambda p: parse(p.name)[0] in ("2", "3") and not p.name.endswith("-stage1"),
            (("final", "stage 3: final-score BCE"), ("null_share", "stage 3: pass 4's null share")),
        ),
    )
    two_stage = any(parse(p.name)[0] in ("2", "3") for p in runs)
    fig, axes = plt.subplots(2 if two_stage else 1, 3, figsize=(14, 8.4 if two_stage else 4.2), squeeze=False)
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    numbers, c = {}, 0
    for _, pick, panels in rows if two_stage else rows[:1]:
        chosen = [p for p in runs if pick(p)]
        for key, label in panels:
            ax = axes[cells[c]]
            c += 1
            ends = {}
            for run in chosen:
                v, arm, seed = parse(run.name)
                t = telemetry(run)
                if key not in t or not np.isfinite(t[key]).any():
                    continue
                ls = (0, (4, 2)) if v == "3" else "-"
                ax.plot(t["step"], smooth(t[key]), color=ARM_COLOURS[arm], lw=1.2, alpha=0.85, ls=ls)
                last = float(np.nanmean(t[key][-10:]))
                numbers.setdefault(run.name, {})[key] = last
                ends.setdefault(arm, []).append(last)
            ax.set_title(label, loc="left")
            ax.set_xlabel("step (batch 64; 3 epochs)")
            if ends:
                end_labels(ax, ends)
    for ax in axes.ravel()[c:]:
        ax.axis("off")
    present = [a for a in ARMS if any(parse(p.name)[1] == a for p in runs)]
    axes[0, 0].legend(
        handles=[Line2D([], [], color=ARM_COLOURS[a], lw=2, label=a) for a in present],
        loc="upper right",
        title="arm (every seed)",
    )
    NUMBERS["training_last_500_steps"] = numbers
    fig.suptitle(
        "Training telemetry, every run: a running mean over 250 steps (each point of the log is 50); "
        "B-2-1 on top, the two-stage recipes below",
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
    shown = {
        (v, arm)
        for v, arm in groups
        for k in list(sel) + list(rc["runs"] if rc else [])
        if parse(k)[:2] == (v, arm)
    }
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.1 * len(shown) * 2), 4.2))
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
            for k, x in sorted(values.items()):
                if parse(k)[:2] == (v, arm):
                    xx = i + JITTER * (parse(k)[2] - 2)
                    ax.plot(xx, x, "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
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
    fig, axes = plt.subplots(1, len(panels), figsize=(max(15, 0.75 * len(present) * len(panels)), 4.2))
    for ax, (key, title) in zip(axes, panels):  # noqa: B905 -- Python 3.9
        for i, (v, arm) in enumerate(present):
            for run, x in sorted(r["readouts"].items()):
                if parse(run)[:2] == (v, arm):
                    xx = i + JITTER * (parse(run)[2] - 2)
                    ax.plot(xx, x[key], "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
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
    fig, axes = plt.subplots(1, 3, figsize=(max(13, 0.95 * len(present) * 3), 4.2))
    for ax, (key, title) in zip(axes, keys):  # noqa: B905 -- Python 3.9
        for i, (v, arm) in enumerate(present):
            for run, x in sorted(r["runs"].items()):
                if parse(run)[:2] != (v, arm):
                    continue
                d, se = x["minus_release"][key]["mean"], x["minus_release"][key]["se"]
                xx = i + JITTER * (parse(run)[2] - 2)
                ax.plot([xx, xx], [d - 2 * se, d + 2 * se], color=ARM_COLOURS[arm], lw=1.2, alpha=0.8)
                ax.plot(xx, d, "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
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


def scene_rows(r: dict) -> dict:
    """Each SCENE_RULES example: its split, row, token and the numbers that chose it."""
    from gallops.drivejepa_perception import b2_read as reads_module
    from gallops.drivejepa_perception import veto

    navtest = px.Split("navtest")
    official_r = px.official(OUT / "part-b/b0/navtest-R")
    runs = [reads_module.name("1", "C", s) for s in (0, 1, 2)]
    official_c = [px.official(B2 / "navtest-phase1" / f"score-{run}") for run in runs]
    gains = np.array(
        [[c[t]["epdms_hf"] - official_r[t]["epdms_hf"] for c in official_c] for t in navtest.tokens], float
    )
    found = {}
    for key, ok, worst in (
        ("gain", (gains > 0).all(1), gains.min(1)),
        ("loss", (gains < 0).all(1), -gains.max(1)),
    ):
        score = np.where(ok & np.isfinite(worst), worst, -np.inf)
        row = int(np.argmax(score))  # the first token in navtest's order on a tie
        found[key] = {
            "split": "navtest",
            "row": row,
            "token": navtest.tokens[row],
            "R_epdms_hf": float(official_r[navtest.tokens[row]]["epdms_hf"]),
            "C_epdms_hf": [float(c[navtest.tokens[row]]["epdms_hf"]) for c in official_c],
            "terms": {
                k: [float(official_r[navtest.tokens[row]][k])]
                + [float(c[navtest.tokens[row]][k]) for c in official_c]
                for k in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC")
            },
        }
    ctx = reads_module.rclean_context(OUT)
    data = veto.load_split(OUT, "fresh")
    n = np.arange(len(data["rows"]))
    r_pick = data["pdm"][n, data["pick"]]
    ok = (r_pick[:, 2] == 1) & (ctx["human"][:, 2] == 1)
    values = {}
    for seed in (0, 1):
        d = RUNS / reads_module.name("1", "C", seed) / "dump-fresh"
        dump = np.load(d / "dump.npz")
        at = {int(x): i for i, x in enumerate(dump["rows"])}
        idx = np.array([at[int(x)] for x in ctx["rows"]])
        chosen = np.load(d / "labels.npz")["candidates"][idx, dump["pick"][idx]]
        ok &= chosen[:, 2] == 0
        values[seed] = veto.filtered_final(chosen, ctx["human"])
    loss = veto.filtered_final(r_pick, ctx["human"]) - values[0]
    order = np.lexsort((-loss, -ctx["weights"]))
    k = int(next(i for i in order if ok[i]))
    found["ddc"] = {
        "split": "fresh",
        "row": int(ctx["rows"][k]),
        "token": str(data["token"][k]),
        "city": str(data["city"][k]),
        "city_weight_share": float(ctx["weights"][k] / ctx["weights"].sum()),
        "windows_matching": int(ok.sum()),
        "R_value": float(veto.filtered_final(r_pick, ctx["human"])[k]),
        "C_value_s0_s1": [float(values[0][k]), float(values[1][k])],
    }
    for key in found:
        found[key]["rule"] = SCENE_RULES[key]
    return found


def run_32(split: str, run: str, row: int) -> tuple:
    """A run's 32, its pick and each proposal's gate verdict (GATES_DRAWN) on one row of its dump."""
    d = RUNS / run / f"dump-{split}"
    dump = np.load(d / "dump.npz")
    at = int(np.flatnonzero(dump["rows"] == row)[0])
    cand = np.load(d / "labels.npz")["candidates"][at]
    failing = (cand[:, list(GATES_DRAWN)] < 1).any(1)
    return dump["proposals"][at], int(dump["pick"][at]), failing


def fig_scenes(r: dict) -> None:
    import textwrap

    from gallops.drivejepa_perception import env

    env.enter(px.CFG)  # navsim's own classes, for unpickling the metric caches the scenes draw
    scenes = scene_rows(r)
    NUMBERS["scenes"] = scenes
    splits = {"navtest": px.Split("navtest"), "fresh": px.Split("fresh")}
    fig, axes = plt.subplots(2, 3, figsize=(17, 12.5), gridspec_kw={"height_ratios": [1.0, 2.1]})
    heads = {
        "gain": "Where the fine-tune gains (navtest)",
        "loss": "Where the fine-tune loses (navtest)",
        "ddc": "A new direction failure (R-clean)",
    }
    for i, key in enumerate(("gain", "loss", "ddc")):
        ex = scenes[key]
        s, row = splits[ex["split"]], ex["row"]
        cam, ax = axes[0, i], axes[1, i]
        cam.imshow(views.model_view(s.image(row)))
        cam.axis("off")
        if key == "ddc":
            detail = (
                f"{ex['city']}, a window carrying {ex['city_weight_share'] * 100:.2f}% of R-clean's city "
                "weight. "
                f"B-1's value: R {ex['R_value']:.2f}; C seed 0 {ex['C_value_s0_s1'][0]:.2f}, seed 1 "
                f"{ex['C_value_s0_s1'][1]:.2f}. {ex['windows_matching']} windows meet the rule."
            )
        else:
            t = ex["terms"]
            changed = [k for k in t if len({round(v, 4) for v in t[k] if np.isfinite(v)}) > 1]
            detail = f"Official EPDMS (human filter): R {ex['R_epdms_hf']:.2f}; C seeds " + ", ".join(
                f"{v:.2f}" for v in ex["C_epdms_hf"]
            )
            detail += ". Terms that moved (R; C seeds 0, 1, 2): " + "; ".join(
                f"{k} " + ", ".join("–" if not np.isfinite(v) else f"{v:.2f}" for v in t[k]) for k in changed
            )
        cam.set_title(f"{heads[key]} · {ex['token']}", loc="left", fontsize=10.5, fontweight="bold")
        cam.text(
            0,
            -0.03,
            textwrap.fill(detail, 78),
            transform=cam.transAxes,
            fontsize=8,
            color=views.INK_2,
            va="top",
        )
        cache = s.cache(row)
        px.draw_road(ax, cache)
        px.grader_agents(ax, cache)
        props, pick_c, failing = run_32(ex["split"], b2_read.name("1", "C", 0), row)
        o_props, o_pick, _ = run_32(ex["split"], b2_read.name("1", "O", 0), row)
        picks = {
            "R": np.asarray(s.proposals[row, s.pick[row], :, :2]),
            "C": props[pick_c, :, :2],
            "O": o_props[o_pick, :, :2],
            "human": np.asarray(s.labels["ego_future"][row][:, :2]),
        }
        for k in range(32):
            u, v = px.xy_to_plot(props[k, :, :2])
            ax.plot(
                u,
                v,
                color=px.FAIL if failing[k] else px.PASS,
                lw=0.9,
                alpha=0.9 if failing[k] else 0.5,
                zorder=5,
            )
        # Each pick with its eight poses marked, 0.5 s apart: their spacing is the plan's speed.
        for name, colour, lw, ls, marker in (
            ("human", views.INK_2, 1.5, (0, (1, 2)), "x"),
            ("R", views.GREY, 3.2, "-", "s"),
            ("O", ARM_COLOURS["O"], 1.8, (0, (4, 2)), "^"),
            ("C", ARM_COLOURS["C"], 2.2, "-", "o"),
        ):
            u, v = px.xy_to_plot(picks[name])
            ax.plot(np.r_[0, u], np.r_[0, v], color=colour, lw=lw, ls=ls, zorder=8, solid_capstyle="round")
            ax.plot(
                u, v, ls="none", marker=marker, ms=4.5, color=colour, mec=views.SURFACE, mew=0.6, zorder=9
            )
        ax.plot([0], [0], marker="^", color=views.INK, ms=9, zorder=10)
        travel = {k: float(np.linalg.norm(v[-1])) for k, v in picks.items()}
        same = float(np.abs(picks["O"] - picks["C"]).max()) < 0.2
        ex["travel_4s_m"] = travel
        ex["O_pick_equals_C"] = same
        ahead = max(float(np.max(props[..., 0])), *(float(v[:, 0].max()) for v in picks.values()))
        side = max(
            float(np.abs(props[..., 1]).max()), *(float(np.abs(v[:, 1]).max()) for v in picks.values())
        )
        top = max(14.0, ahead + 5.0)
        half = max(8.0, side + 4.0, 0.4 * (top + 3.0))
        px.bev(ax, (-half, half), (-3.0, top), "from above (zoomed to the plans)")
        ax.text(
            0.02,
            0.98,
            "travel in 4 s: R {R:.1f} m, C {C:.1f} m, O {O:.1f} m, human {human:.1f} m".format(**travel)
            + ("\nO's pick is C's" if same else ""),
            transform=ax.transAxes,
            fontsize=8,
            color=views.INK,
            va="top",
            bbox=BACKING,
            zorder=11,
        )
    handles = [
        Patch(color=views.GRID, label="drivable area (the grader's map)"),
        Patch(color=views.INK_2, alpha=0.25, label="agents at t, as the grader reads them"),
        Line2D([], [], color=px.PASS, lw=1.2, label="C seed 0's proposal, passing (vendored)"),
        Line2D([], [], color=px.FAIL, lw=1.2, label="C seed 0's proposal failing NC, DAC, DDC or TTC"),
        Line2D([], [], color=views.GREY, lw=3.2, marker="s", ms=4.5, label="the release's pick (R)"),
        Line2D([], [], color=ARM_COLOURS["C"], lw=2.2, marker="o", ms=4.5, label="C seed 0's pick"),
        Line2D(
            [],
            [],
            color=ARM_COLOURS["O"],
            lw=1.8,
            ls=(0, (4, 2)),
            marker="^",
            ms=4.5,
            label="O seed 0's pick",
        ),
        Line2D(
            [], [], color=views.INK_2, lw=1.5, ls=(0, (1, 2)), marker="x", ms=4.5, label="the human's plan"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "What the fine-tune (C) changes, scene by scene: three scenes, each the extreme of a rule "
        '(figures/part-b2-results-numbers.json, "scenes"); the marks are a plan\'s poses, 0.5 s apart',
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    px.save(fig, "b2r-scenes.jpg")


# ------------------------------------------------------------------------------- reads, swaps, recipes


def fig_attention() -> None:
    """What each trained read attends to (scripts/b2_attention.py): the null share by pass, and attention per
    token against uniform by the token's confidence and distance, and near each query's own pose."""
    path = B2 / "attention.json"
    if not path.exists():
        return
    att = json.loads(path.read_text())
    NUMBERS["attention"] = att
    runs = [(name, e) for name, e in sorted(att.items()) if e["recipe"] == "joint"]
    if not runs:
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [2.4, 3, 3, 2.2]})
    for name, e in runs:
        arm, seed = e["arm"], parse(name)[2]
        passes = sorted(e["passes"], key=int)
        axes[0].plot(
            [int(k) for k in passes],
            [e["passes"][k]["null"] for k in passes],
            "-o",
            color=ARM_COLOURS[arm],
            ms=4,
            lw=1.2,
            alpha=0.85,
        )
        p4 = e["passes"]["4"]
        for ax, key in ((axes[1], "conf"), (axes[2], "dist")):
            ratio = [m / u if u > 0 else np.nan for m, u in zip(p4[f"{key}_mass"], p4[f"{key}_uniform"])]  # noqa: B905
            xs = np.arange(len(ratio)) + JITTER * (seed - 2) + 0.04 * (ARMS.index(arm) - 3)
            ax.plot(xs, ratio, "o", color=ARM_COLOURS[arm], ms=4.5, mec=views.SURFACE, mew=0.8)
        x = ARMS.index(arm) + JITTER * (seed - 2)
        axes[3].plot(x, p4["near_share"] / max(p4["near_uniform"], 1e-9), "o", color=ARM_COLOURS[arm], ms=5)
    some = runs[0][1]
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xlabel("pass")
    axes[0].set_title("attention on the null token", loc="left", fontsize=9)
    for ax, key, title in (
        (axes[1], "conf_bins", "pass 4: attention per token ÷ uniform, by confidence"),
        (axes[2], "dist_bins", "pass 4: attention per token ÷ uniform, by distance (m)"),
    ):
        edges = some[key]
        names = [f"{lo:g}–{hi:g}" if hi < 1e8 else f"> {lo:g}" for lo, hi in zip(edges[:-1], edges[1:])]  # noqa: B905
        if key == "conf_bins":
            names = [n.replace("1.0001", "1") for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7.5)
        ax.axhline(1.0, color=views.INK_2, lw=1, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axes[1].axvline(2.5, color=views.GREY, lw=1)
    axes[1].text(
        2.55, axes[1].get_ylim()[1], " the decoded list's cut", fontsize=7, color=views.INK_2, va="top"
    )
    present = [a for a in ARMS if any(e["arm"] == a for _, e in runs)]
    axes[3].set_xticks([ARMS.index(a) for a in present])
    axes[3].set_xticklabels(present)
    axes[3].axhline(1.0, color=views.INK_2, lw=1, ls=(0, (3, 2)))
    axes[3].set_title(
        f"pass 4: attention within {some['near_m']:g} m of a query's pose ÷ uniform", loc="left", fontsize=9
    )
    axes[3].grid(axis="x", visible=False)
    axes[0].legend(
        handles=[Line2D([], [], color=ARM_COLOURS[a], marker="o", lw=1.2, label=a) for a in present],
        loc="lower left",
        title="arm (every seed)",
    )
    fig.suptitle(
        f"What the trained reads attend to, on {some['sample_rows']:,} navtest scenes (B-2-1; the dashed "
        "line is uniform attention): O, D and HD read the agents their list keeps, H6 and H3 all 128 slots",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-attention.png")


def fig_swaps() -> None:
    """The test-time swaps (b2_read swaps), every phase's: the share of picks moving more than 1 m, and the
    pick's vendored final, swapped minus own."""
    found = {}
    for path in sorted(B2.glob("swaps*.json")):
        found.update(json.loads(path.read_text())["runs"])
    if not found:
        return
    kinds = ("none", "log", "fold", "truth")
    names = {
        "none": "no input",
        "log": "another log's",
        "fold": "another fold's side-car",
        "truth": "labelled agents",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    present = [a for a in ARMS if any(parse(run)[1] == a for run in found)]
    width = 0.8 / max(len(present), 1)
    for j, arm in enumerate(present):
        for run, e in sorted(found.items()):
            if parse(run)[1] != arm:
                continue
            seed = parse(run)[2]
            for i, kind in enumerate(kinds):
                if kind not in e:
                    continue
                x = i - 0.4 + width * (j + 0.5) + 0.02 * (seed - 2)
                axes[0].plot(
                    x, e[kind]["moved"] * 100, "o", color=ARM_COLOURS[arm], ms=4.5, mec=views.SURFACE
                )
                d, se = e[kind]["final_minus_own"], e[kind]["se"]
                axes[1].plot([x, x], [d - 2 * se, d + 2 * se], color=ARM_COLOURS[arm], lw=1.1, alpha=0.8)
                axes[1].plot(x, d, "o", color=ARM_COLOURS[arm], ms=4.5, mec=views.SURFACE)
    for ax, title, ylabel in (
        (axes[0], "the pick moves more than 1 m at 4 s", "% of navtest scenes"),
        (axes[1], "the pick's vendored final, swapped minus own", "points, ± 2 SE clustered by log"),
    ):
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([names[k] for k in kinds], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[1].axhline(0, color=views.INK_2, lw=1)
    axes[1].legend(
        handles=[Line2D([], [], color=ARM_COLOURS[a], marker="o", lw=1.1, label=a) for a in present],
        loc="lower left",
        title="arm (B-2-1, every seed)",
    )
    NUMBERS["swaps_all"] = found
    fig.suptitle(
        "Does the planner use its input? Each B-2-1 run read again on navtest with its input replaced at "
        "test (vendored)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-swaps.png")


def fig_recipes(r: dict) -> None:
    """C, D and HD under the three recipes: navtest EPDMS minus R, R-clean, nuScenes L2 at 3 s, and the 32's
    matched move from R (B-0-4); one mark a seed, the recipe by its shape."""
    marks = {"1": "o", "2": "s", "3": "^"}
    rc_path = next((p for p in (B2 / "rclean-all.json", B2 / "rclean.json") if p.exists()), None)
    rc = json.loads(rc_path.read_text())["runs"] if rc_path else {}
    nu_path = next((p for p in (B2 / "nusc.json", B2 / "nusc-phase1.json") if p.exists()), None)
    nu = json.loads(nu_path.read_text())["runs"] if nu_path else {}
    an_path = B2 / "analysis-navtest-all.json"
    an = json.loads(an_path.read_text())["runs"] if an_path.exists() else {}
    if not any(parse(run)[0] in ("2", "3") for run in r["vs_r"]):
        return
    panels = (
        ("navtest EPDMS − R, points", {k: v["epdms_hf"]["a_minus_b"] for k, v in r["vs_r"].items()}),
        ("R-clean, B-1's value − the release's", {k: v["minus_release"] for k, v in rc.items()}),
        (
            "nuScenes: the pick's L2 at 3 s − the release's, m",
            {k: v["minus_release"]["l2_3s"]["mean"] for k, v in nu.items()},
        ),
        (
            "navtest: the 32's matched move from R, m",
            {k: v["from_R"]["matched_ade_m"] for k, v in an.items()},
        ),
    )
    arms = ("C", "D", "HD")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    numbers = {}
    for ax, (title, values) in zip(axes, panels):  # noqa: B905 -- Python 3.9
        numbers[title] = {}
        for i, arm in enumerate(arms):
            for v in ("1", "2", "3"):
                ys = [(parse(k)[2], x) for k, x in sorted(values.items()) if parse(k)[:2] == (v, arm)]
                for seed, y in ys:
                    xx = i + (int(v) - 2) * 0.22 + 0.04 * (seed - 2)
                    ax.plot(xx, y, marks[v], color=ARM_COLOURS[arm], ms=5.5, mec=views.SURFACE, mew=0.8)
                    numbers[title][b2_read.name(v, arm, seed)] = y
        ax.axhline(0, color=views.INK_2, lw=1)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].legend(
        handles=[
            Line2D([], [], color=views.INK_2, marker=marks[v], ls="none", ms=6, label=VERSIONS[v])
            for v in marks
        ],
        loc="best",
        fontsize=7.5,
    )
    NUMBERS["recipes"] = numbers
    fig.suptitle(
        "The three recipes on C, D and HD: one mark a seed, the recipe by its shape (descriptive: the "
        "recipes differ in steps, parameters and labels)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-recipes.png")


# ---------------------------------------------------------------------------------------- scene by scene

#: Buckets for b2r-where: the ego's speed at t (m/s, from the cached ego status) and the agents the fold
#: side-car's decoded list keeps on the scene.
SPEED_BINS = (
    (0.0, 0.5, "stopped (< 0.5)"),
    (0.5, 5.0, "slow (0.5–5)"),
    (5.0, 10.0, "5–10"),
    (10.0, 1e9, "> 10"),
)
AGENT_BINS = ((0, 0, "none"), (1, 2, "1–2"), (3, 5, "3–5"), (6, 10**6, "6 or more"))
#: The contrasts b2r-where reads, each a run pair per seed: (label, (version, arm) A, (version, arm) B or R).
WHERE = (
    ("C − R", ("1", "C"), None),
    ("O − C", ("1", "O"), ("1", "C")),
    ("D − C", ("1", "D"), ("1", "C")),
    ("HD − D", ("1", "HD"), ("1", "D")),
    ("H6 − HD", ("1", "H6"), ("1", "HD")),
)
KIND_ORDER = (
    "in_view_50m",
    "outside_view_or_range",
    "not_in_labels",
    "static",
    "unnamed",
    "no_safe_proposal",
)
KIND_NAMES = {
    "in_view_50m": "in view,\n≤ 50 m",
    "outside_view_or_range": "out of view\nor range",
    "not_in_labels": "not in\nthe labels",
    "static": "static",
    "unnamed": "unnamed",
    "no_safe_proposal": "no safe\nproposal",
}


def official_runs() -> dict:
    """run name -> its official per-scene scores, from whichever phase read it."""
    out = {}
    for phase in ("1", "2"):
        d = B2 / f"navtest-phase{phase}"
        for score in sorted(d.glob("score-b2-*")) if d.exists() else ():
            if (score / "manifest.json").exists():
                out[score.name[len("score-") :]] = px.official(score)
    return out


def scene_table() -> dict:
    """Per navtest scene: its log, R's official EPDMS and NC/TTC verdict, the ego's speed, the decoded agents
    the fold side-car keeps, and for R's failures the object behind it (B-0-3's rule: failure_kind)."""
    from gallops.drivejepa_perception import headroom, navsim_data
    from gallops.drivejepa_perception import targets as tg

    nav = px.Split("navtest")
    official_r = px.official(OUT / "part-b/b0/navtest-R")
    tokens = nav.tokens
    fails = np.array([official_r[t]["NC"] < 1 or official_r[t]["TTC"] < 1 for t in tokens])
    safe = ((nav.pdm[..., 0] == 1) & (nav.pdm[..., 5] == 1)).any(1)
    kind = np.array([""] * len(tokens), dtype=object)
    for r in np.flatnonzero(fails):
        if not safe[r]:
            kind[r] = "no_safe_proposal"
            continue
        track = int(nav.collision_agent[r, nav.pick[r]])
        if track == headroom.NO_AGENT:
            track = int(nav.ttc_agent[r, nav.pick[r]])
        kind[r] = headroom.failure_kind(
            navsim_data.agents_of(nav.labels, int(r)),
            tg.front_projection(np.asarray(nav.lidar2img[r])),
            track,
        )[0]
    return {
        "tokens": tokens,
        "logs": nav.logs,
        "r_epdms": np.array([official_r[t]["epdms_hf"] for t in tokens], float),
        "r_fails": fails,
        "kind": kind,
        "speed": np.linalg.norm(np.asarray(nav.ego_status[:, -1, 3:5], float), axis=-1),
        "decoded": np.load(OUT / "b2-hidden-navtest-v1" / "agents.npz")["mask"].sum(1),
    }


def fig_where(table: dict, official: dict) -> None:
    from gallops.drivejepa_perception.b_report import clustered_se

    tokens, logs = table["tokens"], table["logs"]

    def epdms(run):
        return np.array([official[run][t]["epdms_hf"] for t in tokens], float)

    families = (
        (
            "the ego's speed at t, m/s",
            [(table["speed"] >= lo) & (table["speed"] < hi) for lo, hi, _ in SPEED_BINS],
            [b[2] for b in SPEED_BINS],
        ),
        (
            "agents in the decoded list (the fold side-car)",
            [(table["decoded"] >= lo) & (table["decoded"] <= hi) for lo, hi, _ in AGENT_BINS],
            [b[2] for b in AGENT_BINS],
        ),
        ("R's pick (official)", [table["r_fails"], ~table["r_fails"]], ["fails NC or TTC (275)", "passes"]),
    )
    series = []
    for label, a, b in WHERE:
        diffs = []
        for seed in range(5):
            ra = b2_read.name(*a, seed)
            rb = None if b is None else b2_read.name(*b, seed)
            if ra in official and (b is None or rb in official):
                diffs.append(epdms(ra) - (table["r_epdms"] if b is None else epdms(rb)))
        if diffs:
            series.append((label, a[1], np.mean(diffs, axis=0) * 100, len(diffs)))
    if not series:
        return
    NUMBERS["where"] = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [4, 4, 2.2]})
    width = 0.8 / len(series)
    for ax, (title, masks, names) in zip(axes, families):  # noqa: B905 -- Python 3.9
        for j, (label, arm, d, seeds) in enumerate(series):
            for i, m in enumerate(masks):
                ok = m & np.isfinite(d)
                mean, se = float(d[ok].mean()), clustered_se(d[ok], logs[ok])
                x = i - 0.4 + width * (j + 0.5)
                ax.plot([x, x], [mean - 2 * se, mean + 2 * se], color=ARM_COLOURS[arm], lw=1.4)
                ax.plot(x, mean, "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
                NUMBERS["where"].setdefault(label, {}).setdefault(title, {})[names[i]] = {
                    "mean": mean,
                    "se": se,
                    "scenes": int(ok.sum()),
                    "seeds": seeds,
                }
        ax.axhline(0, color=views.INK_2, lw=1)
        ax.set_xticks(range(len(masks)))
        ax.set_xticklabels(
            [f"{n}\n{int(m.sum()):,} scenes" for n, m in zip(names, masks)],  # noqa: B905 -- Python 3.9
            fontsize=7.5,
        )
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("EPDMS (human filter) difference, points")
    axes[-1].legend(
        handles=[
            Line2D([], [], color=ARM_COLOURS[arm], marker="o", lw=1.4, label=f"{label} ({seeds} seeds)")
            for label, arm, _, seeds in series
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="contrast (seeds averaged per scene)",
    )
    fig.suptitle(
        "Where each contrast's difference comes from: the scene's mean difference within each group, ± 2 SE "
        "clustered by log (navtest, official)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-where.png")


def fig_failures(table: dict, official: dict) -> None:
    tokens = table["tokens"]
    fails, kind = table["r_fails"], table["kind"]
    passes = ~fails
    rows = {}
    for run, scores in sorted(official.items()):
        v, arm, seed = parse(run)
        bad = np.array([scores[t]["NC"] < 1 or scores[t]["TTC"] < 1 for t in tokens])
        fixed = fails & ~bad
        rows[run] = {
            "fixed": int(fixed.sum()),
            "fixed_by_kind": {k: int((fixed & (kind == k)).sum()) for k in KIND_ORDER},
            "broken": int((passes & bad).sum()),
            "failing": int(bad.sum()),
        }
    if not rows:
        return
    totals = {k: int((fails & (kind == k)).sum()) for k in KIND_ORDER}
    NUMBERS["failures"] = {"R_failing": int(fails.sum()), "R_failing_by_kind": totals, "runs": rows}
    groups = [(v, arm) for v in ("1", "2", "3") for arm in ARMS]
    present = [(v, arm) for v, arm in groups if any(parse(run)[:2] == (v, arm) for run in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 3, 3]})
    panels = (
        (axes[0], "fixed", f"of R's {int(fails.sum())} failing scenes, how many each run passes"),
        (axes[1], "broken", f"new failures among R's {int(passes.sum()):,} passing scenes"),
    )
    for ax, key, title in panels:
        for i, (v, arm) in enumerate(present):
            for run, x in sorted(rows.items()):
                if parse(run)[:2] == (v, arm):
                    xx = i + JITTER * (parse(run)[2] - 2)
                    ax.plot(xx, x[key], "o", ms=5, color=ARM_COLOURS[arm], mec=views.SURFACE, mew=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{arm}\nB-2-{v}" for v, arm in present], fontsize=7)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_ylabel("scenes")
        ax.grid(axis="x", visible=False)
    ax = axes[2]
    shown = [k for k in KIND_ORDER if totals[k]]
    for i, (v, arm) in enumerate(present):
        runs = [x for run, x in rows.items() if parse(run)[:2] == (v, arm)]
        for j, k in enumerate(shown):
            share = np.mean([x["fixed_by_kind"][k] / totals[k] for x in runs]) * 100
            x = j + (i - (len(present) - 1) / 2) * (0.8 / max(len(present), 1))
            ax.bar(x, share, width=0.8 / max(len(present), 1) * 0.9, color=ARM_COLOURS[arm], lw=0)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"{KIND_NAMES[k]}\n({totals[k]})" for k in shown], fontsize=7)
    ax.set_ylabel("% of those failures fixed (mean over seeds)")
    ax.set_title("fixed, by the object behind R's failure (B-0-3)", loc="left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[Patch(color=ARM_COLOURS[arm], label=f"{arm} · B-2-{v}") for v, arm in present],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7.5,
    )
    fig.suptitle(
        "Collisions and time-to-collision, scene by scene (official): fixed and new failures against R, one "
        "dot a seed",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    px.save(fig, "b2r-failures.png")


#: b2r-hidden-scenes' rules: the paired seeds' official EPDMS, HD minus D, scene by scene.
HIDDEN_RULES = {
    "gain": "among navtest scenes where HD's official EPDMS (human filter) is above D's on every paired "
    "seed, the one with the largest mean gain",
    "loss": "among navtest scenes where HD's official EPDMS is below D's on every paired seed, the one with "
    "the largest mean loss",
}
SUB = 0.1


def attention_on_row(run: str, row: int) -> np.ndarray:
    """HD's read on one navtest row: pass 4's attention per token, averaged over the 256 queries and the heads
    (b2_attention's own capture), as the run's dumps restore it."""
    import torch
    from b2_attention import Capture, Spy

    from gallops.drivejepa_perception import env, train_b2, train_refiner

    entered = env.enter(px.CFG)
    device = torch.device("cuda")
    meta, model, hook = train_b2.restore(entered, RUNS / run, device)
    src = train_b2.Source(OUT, "navtest", meta["args"]["arm"])
    weights, proposals = [], []
    Capture(hook.adapter.read, weights)
    adapter = hook.adapter
    hook.adapter = Spy(adapter, proposals)
    with torch.no_grad():
        adapter.context = src.context(np.array([row]), device)
        hook.k = 0
        train_refiner.forward(model, src.data, np.array([row]), device)
    out = weights[3][0].float().cpu().numpy().mean(0)[:-1]
    del model, hook
    torch.cuda.empty_cache()
    return out


def fig_hidden_scenes(table: dict, official: dict) -> None:
    import textwrap

    from gallops.drivejepa_perception import env

    tokens = table["tokens"]
    seeds = [
        s
        for s in range(5)
        if b2_read.name("1", "D", s) in official and b2_read.name("1", "HD", s) in official
    ]
    if not seeds:
        return
    env.enter(px.CFG)

    def epdms(run):
        return np.array([official[run][t]["epdms_hf"] for t in tokens], float)

    d = np.stack([epdms(b2_read.name("1", "HD", s)) - epdms(b2_read.name("1", "D", s)) for s in seeds])
    mean = d.mean(0)
    chosen = {}
    for key, agree, sign in (("gain", (d > 0).all(0), 1.0), ("loss", (d < 0).all(0), -1.0)):
        score = np.where(agree & np.isfinite(mean), sign * mean, -np.inf)
        if not np.isfinite(score).any() or score.max() == -np.inf:
            continue
        r = int(np.argmax(score))
        chosen[key] = {
            "row": r,
            "token": tokens[r],
            "seeds": seeds,
            "hd_minus_d": [float(x) for x in d[:, r]],
            "scenes_meeting_rule": int(agree.sum()),
            "rule": HIDDEN_RULES[key],
        }
    if not chosen:
        return
    nav = px.Split("navtest")
    slots = dict(np.load(OUT / "b2-hidden-navtest-v1" / "agents.npz"))
    fold = np.load(OUT / "b2-hidden-navtest-v1" / "fold.npy")
    fig, axes = plt.subplots(
        2,
        len(chosen),
        figsize=(8.6 * len(chosen), 12.5),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 2.0]},
    )
    heads = {"gain": "Where HD beats D on every seed", "loss": "Where HD loses to D on every seed"}
    for i, (key, ex) in enumerate(chosen.items()):
        r = ex["row"]
        conf = slots["conf"][r]
        decoded = np.flatnonzero(conf >= 0.5)
        sub = np.flatnonzero((conf >= SUB) & (conf < 0.5))
        att = attention_on_row(b2_read.name("1", "HD", 0), r)
        ex["fold"] = int(fold[r])
        ex["decoded"] = int(len(decoded))
        ex["slots_0.1_to_0.5"] = int(len(sub))
        ex["hd_s0_attention_on_decoded"] = {int(q): float(att[q]) for q in decoded}
        boxes = [px.det_box(slots, q) for q in np.r_[decoded, sub]]
        colours = [views.CLASS_COLOURS[int(slots["cls"][r, q])] for q in np.r_[decoded, sub]]
        lws = [1.8] * len(decoded) + [0.8] * len(sub)
        styles = ["-"] * len(decoded) + [(0, (2, 2))] * len(sub)
        cam, ax = axes[0, i], axes[1, i]
        px.image_boxes(cam, views.model_view(nav.image(r)), nav.projection(r), boxes, colours, lws, styles)
        official_terms = {
            name: [official[b2_read.name("1", name, s)][ex["token"]]["epdms_hf"] for s in seeds]
            for name in ("D", "HD")
        }
        ex["official_epdms"] = official_terms
        cam.set_title(f"{heads[key]} · {ex['token']}", loc="left", fontsize=10.5, fontweight="bold")
        detail = (
            f"Official EPDMS (human filter), seeds {', '.join(map(str, seeds))}: D "
            + ", ".join(f"{v:.2f}" for v in official_terms["D"])
            + "; HD "
            + ", ".join(f"{v:.2f}" for v in official_terms["HD"])
            + f". Fold {ex['fold']}'s side-car: {len(decoded)} slots at ≥ 0.5 (solid: D's list, and HD's), "
            f"{len(sub)} at 0.1–0.5 (dashed). {ex['scenes_meeting_rule']} scenes meet the rule."
        )
        cam.text(
            0,
            -0.03,
            textwrap.fill(detail, 95),
            transform=cam.transAxes,
            fontsize=8,
            color=views.INK_2,
            va="top",
        )
        cache = nav.cache(r)
        px.draw_road(ax, cache)
        px.grader_agents(ax, cache)
        top = att[decoded].max() if len(decoded) else 1.0
        for q in decoded:
            u, v = px.xy_to_plot(views.footprint(px.det_box(slots, q)[None])[0])
            ax.fill(u, v, fc=views.CLASS_COLOURS[int(slots["cls"][r, q])], alpha=0.35, lw=0, zorder=4)
            ax.plot(
                np.r_[u, u[:1]],
                np.r_[v, v[:1]],
                color=views.INK,
                lw=0.6 + 3.0 * att[q] / max(top, 1e-9),
                zorder=5,
            )
        for q in sub:
            u, v = px.xy_to_plot(views.footprint(px.det_box(slots, q)[None])[0])
            ax.plot(np.r_[u, u[:1]], np.r_[v, v[:1]], color=views.GREY, lw=0.8, ls=(0, (2, 1.5)), zorder=4)
        props, pick_hd, failing = run_32("navtest", b2_read.name("1", "HD", 0), r)
        for k in range(32):
            u, v = px.xy_to_plot(props[k, :, :2])
            ax.plot(
                u,
                v,
                color=px.FAIL if failing[k] else px.PASS,
                lw=0.8,
                alpha=0.85 if failing[k] else 0.45,
                zorder=5,
            )
        d_props, pick_d, _ = run_32("navtest", b2_read.name("1", "D", 0), r)
        picks = {
            "human": np.asarray(nav.labels["ego_future"][r][:, :2]),
            "R": np.asarray(nav.proposals[r, nav.pick[r], :, :2]),
            "D": d_props[pick_d, :, :2],
            "HD": props[pick_hd, :, :2],
        }
        for name, colour, lw, ls, marker in (
            ("human", views.INK_2, 1.5, (0, (1, 2)), "x"),
            ("R", views.GREY, 3.2, "-", "s"),
            ("D", ARM_COLOURS["D"], 2.0, (0, (4, 2)), "^"),
            ("HD", ARM_COLOURS["HD"], 2.2, "-", "o"),
        ):
            u, v = px.xy_to_plot(picks[name])
            ax.plot(np.r_[0, u], np.r_[0, v], color=colour, lw=lw, ls=ls, zorder=8, solid_capstyle="round")
            ax.plot(
                u, v, ls="none", marker=marker, ms=4.5, color=colour, mec=views.SURFACE, mew=0.6, zorder=9
            )
        ax.plot([0], [0], marker="^", color=views.INK, ms=9, zorder=10)
        ahead = max(25.0, float(max(v[:, 0].max() for v in picks.values())) + 8.0)
        half = max(12.0, 0.45 * (ahead + 4.0))
        px.bev(
            ax,
            (-half, half),
            (-4.0, ahead),
            "from above: HD seed 0's 32; agent outlines weighted by HD's read",
        )
        ex["travel_4s_m"] = {k: float(np.linalg.norm(v[-1])) for k, v in picks.items()}
    handles = [
        Patch(color=views.INK_2, alpha=0.25, label="agents at t, as the grader reads them"),
        *[
            Patch(color=c, alpha=0.5, label=f"a decoded slot: {n}")
            for c, n in zip(views.CLASS_COLOURS, views.CLASS_NAMES)  # noqa: B905 -- Python 3.9
        ],
        Line2D(
            [], [], color=views.INK, lw=2.5, label="its outline: heavier = more of HD's attention (pass 4)"
        ),
        Line2D(
            [],
            [],
            color=views.GREY,
            lw=0.8,
            ls=(0, (2, 1.5)),
            label="a slot at 0.1–0.5 (H6 reads it; D, HD not)",
        ),
        Line2D([], [], color=px.PASS, lw=1.2, label="HD's proposal, passing (vendored)"),
        Line2D([], [], color=px.FAIL, lw=1.2, label="HD's proposal failing NC, DAC, DDC or TTC"),
        Line2D([], [], color=views.GREY, lw=3.2, marker="s", ms=4.5, label="R's pick"),
        Line2D(
            [],
            [],
            color=ARM_COLOURS["D"],
            lw=2.0,
            ls=(0, (4, 2)),
            marker="^",
            ms=4.5,
            label="D seed 0's pick",
        ),
        Line2D([], [], color=ARM_COLOURS["HD"], lw=2.2, marker="o", ms=4.5, label="HD seed 0's pick"),
        Line2D(
            [], [], color=views.INK_2, lw=1.5, ls=(0, (1, 2)), marker="x", ms=4.5, label="the human's plan"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    NUMBERS["hidden_scenes"] = chosen
    fig.suptitle(
        "The primary contrast, HD − D, scene by scene: each scene the extreme of a rule (the JSON's "
        '"hidden_scenes"); the marks are a plan\'s poses, 0.5 s apart',
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    px.save(fig, "b2r-hidden-scenes.jpg")


def main() -> None:
    views.use_style()
    r = reads()
    NUMBERS["phases_read"] = r["phases"]
    for name in ("rclean", "rclean-all"):
        if (B2 / f"{name}.json").exists():
            NUMBERS[name.replace("-", "_")] = json.loads((B2 / f"{name}.json").read_text())
    # The test-time swaps and B-0-4, as read (b2_read swaps, analyse): every phase's read, by name.
    for path in sorted(B2.glob("swaps*.json")) + sorted(B2.glob("analysis-*.json")):
        NUMBERS[path.stem.replace("-", "_")] = json.loads(path.read_text())
    if r["phases"]:
        fig_chain(r)
        fig_navtest(r)
        fig_subscores(r)
        fig_readouts(r)
    fig_training()
    fig_health()
    fig_nusc()
    fig_attention()
    fig_swaps()
    if r["phases"]:
        fig_recipes(r)
    if "1" in r["phases"]:
        fig_scenes(r)
        official = official_runs()
        table = scene_table()
        fig_where(table, official)
        fig_failures(table, official)
        fig_hidden_scenes(table, official)
    NUMBERS["noise_limit"] = b2_read.NOISE_LIMIT
    (FIGURES / "part-b2-results-numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float) + "\n")
    print(f"wrote {FIGURES / 'part-b2-results-numbers.json'}")


if __name__ == "__main__":
    main()
