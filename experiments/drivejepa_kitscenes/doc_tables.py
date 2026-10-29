"""Render the tables that docs/drivejepa-kitscenes.md quotes, from the committed result files.

The doc's numbers are copied from this script's output, never typed by hand, and
tests/test_drivejepa_kitscenes.py checks that every table below appears in the doc
character for character. Standard library only, so it runs under `make check`.

    python experiments/drivejepa_kitscenes/doc_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELDOUT_FOUR = HERE / "09_heldout_comparison" / "out" / "final_tables.json"
HELDOUT_FIVE = HERE / "10_encoder_generator_scorer" / "out" / "final_tables.json"
GATE = HERE / "11_regression_gate" / "results.json"

MODELS = (
    "Released model",
    "Encoder only, tuned on KIT",
    "Generator only, trained on KIT",
    "Generator and encoder trained together",
    "Encoder, generator and scorer trained together",
)

# Label, metric and independent unit per subset. results.json keeps the unit's name for only some
# subsets; the names here are the ones Paddock's own verdict.json for this job gives.
GATE_SUBSETS = {
    "kitscenes": ("KITScenes, open loop", "ADE (3 s), m", "scene"),
    "nuscenes": ("nuScenes, open loop", "L2 averaged over 1, 2 and 3 s, m", "log"),
    "navsim": ("NAVSIM, open loop, simulated scoring", "EPDMS", "log"),
    "neuroncap_quick": ("neuroNCAP, quick check", "NCAP score", "scenario pair"),
    "bench2drive_quick": ("Bench2Drive, crash check", "driving score", "route"),
    "alpasim_quick": ("AlpaSim, crash check", "scene score", "scene"),
    "neuroncap": ("neuroNCAP, closed loop", "NCAP score", "scenario pair"),
    "bench2drive": ("Bench2Drive, closed loop", "driving score", "route"),
    "alpasim": ("AlpaSim, closed loop", "scene score", "scene"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _ci(row: dict) -> str:
    lo, hi = row["ci95"]
    return f"{row['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def heldout_table() -> str:
    """The five models on the 223 held-out moments, ALL group."""
    models = _load(HELDOUT_FIVE)["models"]
    head = (
        "| Model | closest candidate ADE (3 s) | closest candidate FDE (3 s) | candidates within 2 m"
        " | variety (m) | pick ADE (3 s) | pick FDE (1 s) | pick FDE (2 s) | pick FDE (3 s) | pick gap |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for name in MODELS:
        g = models[name]["groups"]["ALL"]
        rows.append(
            f"| {name} | {g['m1a']:.3f} | {g['m1b']:.3f} | {g['m2_lt2']:.1f} | {g['m3_variety']:.2f}"
            f" | {g['pick_ade']:.3f} | {g['pick_fde@1s']:.3f} | {g['pick_fde@2s']:.3f}"
            f" | {g['pick_fde@3s']:.3f} | {g['pick_gap']:.3f} |"
        )
    return head + "\n".join(rows)


def groups_table() -> str:
    """SLOW and SLOW TURNS, closest candidate and pick ADE (3 s)."""
    data = _load(HELDOUT_FIVE)
    n = data["n"]
    head = (
        f"| Model | slow ({n['SLOW']}): closest ADE (3 s) | slow: pick ADE (3 s)"
        f" | slow turns ({n['SLOW_TURNS']}): closest ADE (3 s) | slow turns: pick ADE (3 s) |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    rows = []
    for name in MODELS:
        g = data["models"][name]["groups"]
        rows.append(
            f"| {name} | {g['SLOW']['m1a']:.3f} | {g['SLOW']['pick_ade']:.3f}"
            f" | {g['SLOW_TURNS']['m1a']:.3f} | {g['SLOW_TURNS']['pick_ade']:.3f} |"
        )
    return head + "\n".join(rows)


def paired_table() -> str:
    """Each model against the one it is compared with; FAST (2 moments) left out."""
    four = _load(HELDOUT_FOUR)["paired"]
    five = _load(HELDOUT_FIVE)["paired"]
    newest = MODELS[-1]
    rows = [dict(r) for r in four] + [dict(r, new=newest) for r in five]
    head = (
        "| Model | compared with | metric | all 223 | slow (221) | slow turns (37) |\n"
        "|---|---|---|---|---|---|\n"
    )
    out = []
    seen = []
    for r in rows:
        key = (r["new"], r["old"], r["metric"])
        if key not in seen:
            seen.append(key)
    for new, old, metric in seen:
        cell = {r["group"]: _ci(r) for r in rows if (r["new"], r["old"], r["metric"]) == (new, old, metric)}
        label = metric.replace("scorer's pick", "pick").replace("scorer pick", "pick")
        out.append(f"| {new} | {old} | {label} | {cell['ALL']} | {cell['SLOW']} | {cell['SLOW_TURNS']} |")
    return head + "\n".join(out)


def gate_table() -> str:
    """The regression gate's verdict per benchmark subset."""
    pairs = _load(GATE)["A"]["pairs"]
    head = (
        "| Benchmark subset | metric | released model | Encoder, generator and scorer trained together"
        " | units | t | verdict |\n"
        "|---|---|---:|---:|---|---:|---|\n"
    )
    rows = []
    for p in pairs:
        label, metric, unit = GATE_SUBSETS[p["subset"]]
        n = int(p["units"])
        t = "–" if p.get("t") is None else f"{p['t']:.2f}"
        # results.json holds the means to 4 decimals; print them as recorded.
        rows.append(
            f"| {label} | {metric} | {p['baseline_mean']:.4f} | {p['candidate_mean']:.4f}"
            f" | {n} {unit}{'' if n == 1 else 's'} | {t} | {p['verdict']} |"
        )
    return head + "\n".join(rows)


def gate_overall() -> str:
    return _load(GATE)["A"]["overall"]


TABLES = {
    "heldout": heldout_table,
    "groups": groups_table,
    "paired": paired_table,
    "gate": gate_table,
}


def main() -> None:
    for name, fn in TABLES.items():
        print(f"<!-- {name} -->\n{fn()}\n")
    print(f"gate overall: {gate_overall()}")


if __name__ == "__main__":
    main()
