"""Render the tables the docs quote, straight from the recorded JSON.

    python experiments/drivejepa_retrain/doc_tables.py            # print every block
    python experiments/drivejepa_retrain/doc_tables.py results    # print one

Two steps of this experiment have tables and no written report: the second
training run and the 30-scene closed-loop test of adapter 0.1.7. Rather than
write those reports after the fact, README.md and
docs/drivejepa-retrain.md carry these blocks, and
tests/test_drivejepa_retrain.py fails if a doc's copy drifts from what this
prints. Numbers are rounded to four decimals here; the JSON beside this file
has them unrounded.

Stdlib only, so it runs under `make check` with no torch and no data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIAL = HERE / "07_closed_loop_test" / "runs_017" / "tables"
SECOND_RUN = HERE / "12_second_training_run"

#: How each adapter version is named in a results row. 0.1.6 drove the first
#: run's checkpoint and was never shipped; it is shown once, as what it was.
#: The recorded files name checkpoints in the working notes' shorthand; the
#: tables use these instead.
CANDIDATES = {
    "pilot": ("pilot", "the first run's generator + 1 pilot pass at learning rate 1e-4"),
    "+1": ("pilot + 1 pass", "the pilot + 1 pass at learning rate 2.5e-5"),
    "+2": ("pilot + 2 passes", "the pilot + 2 passes at learning rate 2.5e-5"),
}
ARMS = {"P0": "A", "P1": "B", "P2": "C", "P3": "D", "P4": "E"}
RANGES = {"std": "standard", "wide": "wide"}
HELD_OUT = "4,400 examples from 145 training clips held out of the second run"

VERSIONS = (
    ("014b", "0.1.4b", "released checkpoint, momentum-aware selection"),
    ("016", "0.1.6", "intermediate checkpoint, not shipped (first training run)"),
    ("017", "0.1.7", "retrained checkpoint (second training run, generator only)"),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _f(value: float) -> str:
    return f"{value:.4f}"


def _scenes(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _yes(flag: bool, bold: bool = False) -> str:
    return ("**yes**" if bold else "yes") if flag else "no"


def results() -> str:
    """30 scenes, one rollout each: the headline, with the intermediate row."""
    totals = _load(TRIAL / "totals.json")
    n = totals["n"]
    rows = [
        _row(
            "Adapter",
            "Weights",
            "Mean `scene_score`",
            "At-fault collisions",
            "Any collision",
            "Off-road",
            "Stalls",
            "Mean progress",
            "Mean distance (m)",
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name, weights in VERSIONS:
        t = totals[key]
        rows.append(
            _row(
                name,
                weights,
                _f(t["mean_score"]),
                f"{t['at_fault']} / {n}",
                f"{t['any_collision']} / {n}",
                f"{t['offroad']} / {n}",
                f"{t['stalls']} / {n}",
                _f(t["mean_progress"]),
                f"{t['mean_dist_traveled']:.1f}",
            )
        )
    return "\n".join(rows)


def trial_017() -> str:
    """The 0.1.7 closed-loop test as recorded. There is no written report for it."""
    totals = _load(TRIAL / "totals.json")
    gate = _load(TRIAL / "gate_017.json")
    t, runs, n = totals["017"], totals["runs"], totals["n"]
    done = f"{runs['complete']} / {runs['gates_ok']} / {runs['provenance_ok']} of {n}"
    agree = f"Driver vs trainer on the same {gate['examples']} examples: plans agree within 2 cm"
    rows = [
        _row("0.1.7, 30 scenes", "Recorded value"),
        "|---|---|",
        _row("Runs complete / checks passed / provenance verified", done),
        _row("Renderer", ", ".join(runs["nre"])),
        _row("Checkpoint sha256", f"`{', '.join(runs['checkpoint_sha256'])}`"),
        _row("As-built driver diff sha256", f"`{', '.join(runs['adapter_diff_sha256'])}`"),
        _row("Mean `scene_score`", _f(t["mean_score"])),
        _row("Scenes at full score", f"{len(t['full_score'])} of {n}"),
        _row("At-fault collisions", f"{t['at_fault']}: {_scenes(t['at_fault_scenes'])}"),
        _row("Any collision", t["any_collision"]),
        _row("Off-road", f"{t['offroad']}: {_scenes(t['offroad_scenes'])}"),
        _row("Stalls", f"{t['stalls']}: {_scenes(t['stall_scenes'])}"),
        _row("Mean progress", _f(t["mean_progress"])),
        _row("Mean distance travelled (m)", f"{t['mean_dist_traveled']:.1f}"),
        _row(agree, _yes(gate["gate_1_under_2cm"])),
        _row("Batch fix changes a plan at batch 1", _yes(gate["gate_2_fix_off_differs_at_batch_1"])),
        _row("Batch fix changes plans at batch 2", _yes(gate["fix_live_at_batch_2"])),
    ]
    return "\n".join(rows)


def second_run_pilot() -> str:
    """The five pilot arms of the second training run, and which one was continued."""
    pilot = _load(SECOND_RUN / "pilot_choice.json")
    rows = [
        _row("Pilot arm", "Run", "Spacing weight (`lam`)", "Refiner reference range", "Trained", "Continued"),
        "|---|---|---:|---|---|---|",
    ]
    for arm, info in pilot["arms_info"].items():
        rows.append(
            _row(
                ARMS[arm],
                f"`{info['run']}`",
                info["lam"],
                RANGES[info["ref_range"]],
                _yes(info["trained"]),
                _yes(arm == pilot["chosen_arm"], bold=True),
            )
        )
    rows.append("")
    rows.append(f"Compared on {HELD_OUT}.")
    return "\n".join(rows)


def second_run_choice() -> str:
    """The rule written down beforehand, applied to the three candidates, in float32."""
    final = _load(SECOND_RUN / "final_choice.json")
    rule = final["rule_on_float32"]
    slices = ("highway", "close-pass", "ordinary")
    reference = [_f(rule["reference_pass9"][name]) for name in slices]
    limits = [_f(rule["limits"][name]) for name in slices]
    rows = [
        _row(
            "Checkpoint",
            "What it is",
            "Human-path term: highway",
            "close-pass",
            "ordinary",
            "Highway speed swing, car ≥ 20 m/s (m/s)",
            "95 % interval",
            "Qualifies",
            "Chosen",
        ),
        "|---|---|---:|---:|---:|---:|---|---|---|",
        _row(
            "reference",
            "the first run's checkpoint",
            *reference,
            _f(rule["pass9_swing_hw20"]),
            "—",
            "—",
            "—",
        ),
        _row("limit", "reference × 1.05", *limits, "—", "—", "—", "—"),
    ]
    for name, cand in rule["candidates"].items():
        low, high = cand["swing_hw20_ci"]
        rows.append(
            _row(
                CANDIDATES[name][0],
                CANDIDATES[name][1],
                *[_f(cand["human_path_term"][part]) for part in slices],
                _f(cand["swing_hw20"]),
                f"{_f(low)} to {_f(high)}",
                _yes(cand["qualifies"]),
                _yes(name == rule["chosen"], bold=True),
            )
        )
    state = final["final_run"]["state"]
    rows.append("")
    rows.append(f"Compared on {HELD_OUT}, in float32.")
    rows.append("")
    rows.append(
        f"The two further passes, run `{state['run']}`: learning rate {state['lr']}, "
        f"spacing weight (`lam`) {state['lam']}, {state['pass_done']} passes, {state['steps']} steps."
    )
    return "\n".join(rows)


def second_run_checkpoint() -> str:
    """What the merged 0.1.7 checkpoint is made of, tensor by tensor."""
    ckpt = _load(SECOND_RUN / "gen_ckpt.json")
    same = ckpt["same_plans_check"]
    encoder = f"{ckpt['encoder_tensors_bit_identical_to_shipped']} of {ckpt['tensors_from_shipped_encoder']}"
    scorer = f"{ckpt['scorer_tensors_bit_identical_to_pass9_in_merged_file']} of {ckpt['scorer_tensors']}"
    generator = f"{ckpt['generator_tensors_that_differ_from_pass9']} of {ckpt['generator_tensors']}"
    plans = f"Merged file vs the trainer's own file, {same['examples']} examples: max plan difference"
    rows = [
        _row("Merged checkpoint", "Recorded value"),
        "|---|---|",
        _row("sha256", f"`{ckpt['sha256']}`"),
        _row("Bytes", ckpt["bytes"]),
        _row("Tensors", ckpt["tensors_total"]),
        _row("Encoder tensors, bit-identical to the released checkpoint", encoder),
        _row("Scorer tensors, bit-identical to the first run's scorer", scorer),
        _row("Generator tensors that differ from the first run's generator", generator),
        _row("Pass merged", f"{ckpt['passno']} of the two further passes, learning rate {ckpt['lr']}"),
        _row("Spacing weight (`lam`) of that run", ckpt["lam"]),
        _row("Released base checkpoint sha256", f"`{ckpt['base_sha256']}`"),
        _row(plans, same["plan_max_abs_diff_merged_vs_pass_file"]),
    ]
    return "\n".join(rows)


BLOCKS = {
    "results": results,
    "trial_017": trial_017,
    "second_run_pilot": second_run_pilot,
    "second_run_choice": second_run_choice,
    "second_run_checkpoint": second_run_checkpoint,
}


def block(name: str) -> str:
    """One table wrapped in the markers the docs carry, so a test can find it."""
    return f"<!-- table:{name} -->\n{BLOCKS[name]()}\n<!-- /table:{name} -->"


if __name__ == "__main__":
    for wanted in sys.argv[1:] or list(BLOCKS):
        print(block(wanted))
        print()
