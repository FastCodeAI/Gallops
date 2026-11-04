"""B1's and B2's verdicts as one figure: every arm against its control, per seed, beside the gain
predicted. One-off: rerun to refresh.

    {python.drivejepa} experiments/H15_perception_sidecar/scripts/verdicts_figure.py \
        --b1 {runs}/drivejepa_perception/b1/report.json --b2 {runs}/drivejepa_perception/b2/report.json \
        --out experiments/H15_perception_sidecar/figures

Writes verdicts-light.png and verdicts-dark.png (and .svg). Each row is one mode and arm; each dot
one seed's mean paired difference from the control, in EPDMS points, with its 2 SE interval
(clustered by log). The shaded band is the gain PREDICTIONS.md expected (P7 for B1, P12 for B2).
Colours are the dataviz skill's validated first three categorical slots, checked all-pairs in both
modes; each arm also has its own marker and its row label, so identity is never colour alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {  # arm -> (label, marker, light colour, dark colour)
    "F": ("side-car's agents (F)", "o", "#2a78d6", "#3987e5"),
    "O": ("ground-truth agents (O)", "s", "#eb6834", "#d95926"),
    "S": ("another log's agents (S)", "^", "#1baf7a", "#199e70"),
}
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e4e3df",
        "band": "#ecebe7",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "grid": "#33332f",
        "band": "#2a2a27",
    },
}


def rows_of(report: dict, key: str, modes: tuple) -> list:
    """(mode, arm, [(diff_points, 2se_points) per seed]) for every arm a report has."""
    out = []
    for mode in modes:
        if mode not in report:
            continue
        for arm in ARMS:
            k = key.format(arm=arm)
            if k in report[mode]:
                per = report[mode][k]["per_seed"]
                out.append((mode, arm, [(e["f_minus_c"] * 100, 2 * e["se"] * 100) for e in per]))
    return out


def panel(ax, rows, band, title, theme, colour_index):
    import numpy as np

    t = THEMES[theme]
    ax.set_facecolor(t["surface"])
    ax.axvspan(*band, color=t["band"], zorder=0, lw=0)
    ax.text(sum(band) / 2, -0.9, "predicted", ha="center", va="center", fontsize=8, color=t["secondary"])
    ax.axvline(0, color=t["secondary"], lw=1, zorder=1)
    y_labels = []
    for i, (mode, arm, seeds) in enumerate(rows):
        label, marker, light, dark = ARMS[arm]
        colour = light if colour_index == 0 else dark
        for j, (diff, two_se) in enumerate(seeds):
            y = i + (j - 1) * 0.22
            ax.plot(
                [diff - two_se, diff + two_se], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=2
            )
            ax.plot(diff, y, marker=marker, ms=7, color=colour, mec=t["surface"], mew=1.5, zorder=3)
        y_labels.append(f"{mode} · {arm}")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(y_labels, fontsize=9, color=t["primary"])
    ax.set_ylim(len(rows) - 0.4, -1.4)
    ax.set_title(title, fontsize=10, color=t["primary"], loc="left")
    ax.set_xlabel(
        "arm − control, EPDMS points (dot: one seed; line: ± 2 SE)", fontsize=9, color=t["secondary"]
    )
    ax.tick_params(axis="x", colors=t["secondary"], labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b1 = json.loads(Path(args.b1).read_text())
    b2 = json.loads(Path(args.b2).read_text())
    rows_b1 = rows_of(b1, "{arm}_vs_C_kept", ("frozen", "lora", "full", "replace"))
    rows_b2 = rows_of(b2, "{arm}_vs_C_best_of_32", ("frozen", "lora", "full"))
    out = Path(args.out)
    for index, theme in enumerate(("light", "dark")):
        t = THEMES[theme]
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(11, 6.2), dpi=200, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35}
        )
        fig.patch.set_facecolor(t["surface"])
        panel(
            left,
            rows_b1,
            (0.3, 1.0),
            "B1 · scorer: the pick's EPDMS (P7 predicted +0.3 to +1.0)",
            theme,
            index,
        )
        panel(
            right, rows_b2, (0.2, 0.5), "B2 · refiner: best of 32 (P12 predicted +0.2 to +0.5)", theme, index
        )
        for ax in (left, right):
            ax.set_xlim(-0.6, 1.1)
        handles = [
            Line2D([], [], marker=m, ls="", ms=7, color=(lc if index == 0 else dc), label=label)
            for label, m, lc, dc in ARMS.values()
        ]
        legend = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(t["primary"])
        fig.suptitle(
            "H15 · no arm beats its control on every seed, and no estimate reaches the gain predicted",
            fontsize=11,
            color=t["primary"],
            x=0.02,
            ha="left",
        )
        fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.17)
        for suffix in ("png", "svg"):
            fig.savefig(out / f"verdicts-{theme}.{suffix}", facecolor=t["surface"])
        plt.close(fig)
    print(f"wrote {out}/verdicts-light.png, verdicts-dark.png (+ svg)")


if __name__ == "__main__":
    main()
