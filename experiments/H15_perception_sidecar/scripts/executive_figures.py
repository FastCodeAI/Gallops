"""The executive update's figures (docs/drivejepa-perception-sidecar-A-update.md), from step 5's runs.

    <[python] drivejepa> experiments/H15_perception_sidecar/scripts/executive_figures.py

Every number is read from the runs' own files (each run's kept epoch in `metrics.jsonl`), except the
depth and lateral errors, which notebook 06 §7 computes and printed; they are copied here, not
re-derived. The example scene is cut from notebook 06 §4's figure. Writes into ../figures/:
part-a-exec-f1.png, part-a-exec-recall.png, part-a-exec-depth.png and part-a-exec-example.jpg.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("GALLOPS_CONFIG", str(REPO / "gallops.toml"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gallops.config import SiteConfig  # noqa: E402
from gallops.drivejepa_perception import views  # noqa: E402

FIGURES = HERE.parent / "figures"
NOTEBOOK = HERE.parent / "notebooks" / "06_predictions.ipynb"
#: The reference palette's slot 1 for the chosen side-car; context in muted grey.
CHOSEN, CONTEXT = "#2a78d6", "#b8b7b0"
#: Notebook 06 §7, v3 on 24 epochs (seed 0): median depth (along the camera's ray) and lateral (across
#: it) error of targets paired with a detection of their class within 5 m, by target range.
DEPTH = {
    "0–10 m": (0.243, 0.117),
    "10–20 m": (0.465, 0.223),
    "20–30 m": (0.885, 0.404),
    "30–50 m": (1.200, 0.554),
}


def kept(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    record = records[summary["kept_epoch"] - 1]
    return {"f1": record["f1_30m"], **record["select"]}


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def f1_chart(part_a: Path) -> None:
    rows = [
        ("v1 · 4.4 M · 12 epochs", ["5a-v1-e12-s0"]),
        ("v2 · 8.4 M · 12 epochs", ["5a-v2-e12-s0"]),
        ("v3 · 18.4 M · 12 epochs", ["5a-v3-e12-s0"]),
        ("v1 · 4.4 M · 24 epochs", ["5a-v1-e24-s0"]),
        ("v3 · 18.4 M · 24 epochs (3 seeds), chosen", ["5b-v3-e24-s0", "5b-v3-e24-s1", "5b-v3-e24-s2"]),
        ("v3 · 24 epochs · no camera", ["5b-v3-e24-blind-s0"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for y, (label, names) in enumerate(rows[::-1]):
        values = [kept(part_a / n)["f1"] for n in names]
        colour = CHOSEN if "chosen" in label else CONTEXT
        ax.barh(y, np.mean(values), height=0.62, color=colour)
        if len(values) > 1:
            ax.plot([min(values), max(values)], [y, y], color=views.INK, lw=1.4)
        text = f"{np.mean(values):.3f}" + (
            f"  ({min(values):.3f}–{max(values):.3f})" if len(values) > 1 else ""
        )
        ax.text(max(values) + 0.012, y, text, va="center", fontsize=8, color=views.INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 at 2 m, within 30 m (select logs, each run's kept epoch)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Detection quality by configuration")
    save(fig, "part-a-exec-f1.png")


def recall_chart(part_a: Path) -> None:
    r = kept(part_a / "5b-v3-e24-s0")
    rows = [
        ("moving vehicles within 20 m", r["moving_vehicles_20m_recall"]),
        ("causal agents (in or near the ego's path)", r["recall_causal_chosen"]),
        ("vehicles within 30 m", r["recall_2m_within_30m_by_class"][0]),
        ("the nearest agent", r["recall_nearest_1"]),
        ("agents in the line of sight", r["recall_visible"]),
        ("pedestrians within 30 m", r["recall_2m_within_30m_by_class"][1]),
        ("all agents, to 50 m", r["recall_2m"]),
        ("hidden agents", r["recall_hidden"]),
        ("cyclists within 30 m", r["recall_2m_within_30m_by_class"][2]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for y, (_label, value) in enumerate(rows[::-1]):
        ax.barh(y, value, height=0.62, color=CHOSEN)
        ax.text(value + 0.012, y, f"{value:.2f}", va="center", fontsize=8, color=views.INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows[::-1]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recall at 2 m (select logs; v3, 24 epochs, seed 0)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Which agents it finds")
    save(fig, "part-a-exec-recall.png")


def depth_chart() -> None:
    bands = list(DEPTH)
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for k, (name, colour) in enumerate(
        (("along the camera's ray (distance)", CHOSEN), ("across it (direction)", "#eb6834"))
    ):
        values = [DEPTH[b][k] for b in bands]
        ax.bar(x + (k - 0.5) * 0.36, values, width=0.34, color=colour, label=name)
        for xi, v in zip(x, values):  # noqa: B905 -- Python 3.9
            ax.text(xi + (k - 0.5) * 0.36, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=views.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_xlabel("distance of the agent from the camera")
    ax.set_ylabel("median position error (m)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Position error: distance, not direction (v3, 24 epochs, select)")
    save(fig, "part-a-exec-depth.png")


def example_scene() -> None:
    """Rows 2 and 4 of notebook 06 §4's figure (select samples drawn at random), and its legend."""
    from PIL import Image

    nb = json.loads(NOTEBOOK.read_text())
    cell = next(
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and "at its kept epoch, select samples drawn" in "".join(c["source"])
    )
    jpeg = next(o["data"]["image/jpeg"] for o in cell["outputs"] if "image/jpeg" in o.get("data", {}))
    image = Image.open(io.BytesIO(base64.b64decode(jpeg))).convert("RGB")
    pixels = np.asarray(image).astype(float)
    # Pixel rows of the page between its parts are the surface colour. A part taller than 150 px is a
    # sample's panels; the part just before it is that sample's title, and the last part is the legend.
    background = np.abs(pixels - np.array([252, 252, 251])).sum(-1).mean(1) < 6
    parts, start = [], None
    for y, empty in enumerate(background):
        if not empty and start is None:
            start = y
        elif empty and start is not None:
            parts.append((start, y))
            start = None
    panels = [k for k, (a, b) in enumerate(parts) if b - a > 150]
    titles = [parts[k - 1][0] - 6 for k in panels]
    ends = titles[1:] + [parts[-1][0] - 6]  # each sample runs to the next one's title, the last to the legend
    pieces = [image.crop((0, titles[k], image.width, ends[k])) for k in (1, 3)]  # the 2nd and 4th samples
    pieces.append(image.crop((0, parts[-1][0] - 6, image.width, image.height)))
    # A sample's title shares its pixel rows with the axis label of the bird's-eye view above it; on the
    # first piece that label belongs to a sample left out, so it is painted over.
    from PIL import ImageDraw

    ImageDraw.Draw(pieces[0]).rectangle((int(0.55 * image.width), 0, image.width, 24), fill=(252, 252, 251))
    bands = [parts[k] for k in panels]
    out = Image.new("RGB", (image.width, sum(p.height for p in pieces)), (252, 252, 251))
    y = 0
    for piece in pieces:
        out.paste(piece, (0, y))
        y += piece.height
    out.save(FIGURES / "part-a-exec-example.jpg", quality=85)
    print("wrote", FIGURES / "part-a-exec-example.jpg", out.size, "from bands", bands)


def main() -> None:
    views.use_style()
    part_a = Path(SiteConfig.load().resolve("{runs}")) / "drivejepa_perception" / "part-a"
    f1_chart(part_a)
    recall_chart(part_a)
    depth_chart()
    example_scene()


if __name__ == "__main__":
    main()
