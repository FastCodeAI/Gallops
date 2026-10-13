"""The Drive-JEPA retrain record: what can be checked with no GPU and no data.

The experiment under experiments/drivejepa_retrain/ needs a GPU, a 1.3 GB
checkpoint and an access-restricted dataset to run, so nothing here runs it. What is tested
is the part of its claims that the committed files can prove on their own:

    splits     no training clip is an evaluation scene, and the 30 test scenes
               are on the exclusion list every step's scripts assert against
    record     the docs quote the recorded tables exactly; the checkpoint named
               by the closed-loop test, the merge and the pointer file is one
               checkpoint
    facts      the second run's spacing weight was 0 and its scorer was frozen
    hygiene    no driver source, no weights, no frames; every script tokenizes

The experiment's own wiring checks (05_small_overfit_check/test_wiring.py) are
a script that builds the real agent and reads 640 samples; they are part of the
record and cannot run here.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "drivejepa_retrain"
TRIAL = EXP / "07_closed_loop_test" / "runs_017" / "tables"
SECOND_RUN = EXP / "12_second_training_run"
CHECKPOINT_SHA256 = "29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _uuids(path: Path) -> set[str]:
    return {
        line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
    }


@pytest.fixture(scope="module")
def tables():
    spec = importlib.util.spec_from_file_location("doc_tables", EXP / "doc_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- splits


@pytest.fixture(scope="module")
def training_clips() -> list[dict[str, str]]:
    with open(EXP / "04_training_set" / "clips.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_training_set_is_the_train_split_and_nothing_else(training_clips) -> None:
    """Select on one split, report on another: the set that trained is `train`."""
    # clips.csv lists the 4,018 clips that were converted. Three of them yielded
    # no sample, so the training manifest holds 4,015 (04_training_set/checks.json),
    # and 4,015 is the figure the docs quote.
    assert len(training_clips) == 4018
    assert {row["split"] for row in training_clips} == {"train"}
    assert len({row["uuid"] for row in training_clips}) == 4018
    checks = _json(EXP / "04_training_set" / "checks.json")
    assert checks["clips"] == 4015
    for name in ("README.md", "docs/data.md", "docs/drivejepa-retrain.md"):
        text = (REPO_ROOT / name).read_text()
        assert "4,015 clips" in text, name
        assert "from 4,018 clips" not in text, name


def test_no_training_clip_is_an_evaluation_scene(training_clips) -> None:
    """The exclusion list is the simulator's 913 public scenes, which hold the 30 test scenes."""
    excluded = _uuids(EXP / "01_find_clips" / "exam_exclusions.txt")
    assert len(excluded) == 913
    assert not ({row["uuid"] for row in training_clips} & excluded)


def test_the_30_test_scenes_are_on_the_exclusion_list_and_not_in_training(training_clips) -> None:
    with open(TRIAL / "per_scene.csv", newline="") as handle:
        scenes = [row["scene"] for row in csv.DictReader(handle)]
    assert len(scenes) == 30 and len(set(scenes)) == 30
    excluded = _uuids(EXP / "01_find_clips" / "exam_exclusions.txt")
    trained = {row["uuid"] for row in training_clips}
    for scene in scenes:
        assert any(uuid.startswith(scene) for uuid in excluded), f"{scene} is not on the exclusion list"
        assert not any(uuid.startswith(scene) for uuid in trained), f"{scene} is in the training set"


# ---------------------------------------------------------------- record


TABLES = ["results", "trial_017", "second_run_pilot", "second_run_choice", "second_run_checkpoint"]


@pytest.mark.parametrize("name", TABLES)
def test_the_experiment_doc_quotes_each_recorded_table_exactly(tables, name: str) -> None:
    """Two steps have tables and no report, so the doc's copy is the report."""
    doc = (REPO_ROOT / "docs" / "drivejepa-retrain.md").read_text()
    assert tables.block(name) in doc, f"docs/drivejepa-retrain.md is out of step with doc_tables.py {name}"


def test_the_readme_quotes_the_results_table_exactly(tables) -> None:
    assert tables.block("results") in (REPO_ROOT / "README.md").read_text()


def test_the_headline_is_what_totals_json_says() -> None:
    totals = _json(TRIAL / "totals.json")
    assert totals["n"] == 30
    assert round(totals["014b"]["mean_score"], 4) == 0.5433 and totals["014b"]["at_fault"] == 4
    assert round(totals["017"]["mean_score"], 4) == 0.7697 and totals["017"]["at_fault"] == 3
    # The retrained model has fewer at-fault collisions, but not a subset of them.
    assert set(totals["017"]["at_fault_scenes"]) - set(totals["014b"]["at_fault_scenes"]) == {
        "acd22430",
        "bd07fb91",
    }


def test_totals_json_is_what_the_per_scene_table_adds_up_to() -> None:
    """The doc tables are rendered from totals.json; this ties totals.json to the 30 rows under it."""
    totals = _json(TRIAL / "totals.json")
    with open(TRIAL / "per_scene.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == totals["n"] == 30
    for arm in ("014b", "016", "017"):
        scores = [float(row[f"{arm}_scene_score"]) for row in rows]
        assert sum(scores) / len(scores) == pytest.approx(totals[arm]["mean_score"], abs=1e-9), arm
        at_fault = sorted(row["scene"][:8] for row in rows if float(row[f"{arm}_collision_at_fault"]) > 0)
        assert len(at_fault) == totals[arm]["at_fault"], arm
        assert at_fault == sorted(totals[arm]["at_fault_scenes"]), arm


def test_the_figures_the_docs_state_outside_the_rendered_tables_are_the_recorded_ones() -> None:
    """Prose numbers are not covered by the table tests; these are the ones a reader quotes."""
    with open(TRIAL / "per_scene.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def paired(a: str, b: str) -> tuple[float, float]:
        diffs = [float(r[f"{a}_scene_score"]) - float(r[f"{b}_scene_score"]) for r in rows]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        return round(mean, 3), round((var / len(diffs)) ** 0.5, 3)

    assert paired("017", "016") == (0.066, 0.068)
    assert paired("017", "014b") == (0.226, 0.069)
    whole_clip = {
        arm: (
            sum(bool(r[f"{arm}_first_collision_any_s"]) for r in rows),
            sum(bool(r[f"{arm}_first_offroad_s"]) for r in rows),
        )
        for arm in ("014b", "016", "017")
    }
    assert whole_clip == {"014b": (11, 6), "016": (9, 3), "017": (6, 4)}
    doc = (REPO_ROOT / "docs" / "drivejepa-retrain.md").read_text()
    for needle in (
        "0.066 above 0.1.6 with a standard error of 0.068",
        "0.226 above with a standard error of 0.069",
        "6 scenes with a collision and 4 off-road",
        "(0.1.4b: 11 and 6; 0.1.6: 9 and 3)",
        "scores 0.4876 on the same 30 scenes",
    ):
        assert needle in doc, needle
    compare = _json(TRIAL / "models_compare.json")
    assert any(round(v, 4) == 0.4876 for v in _floats(compare)), "0.4876 is not in models_compare.json"
    evaluation = (SECOND_RUN / "unseen_clip_evaluation.md").read_text()
    assert "| pick = true best, % | all | 11842 | 54.6 (52.1 to 56.9) | 38.3 (35.6 to 41.5) |" in evaluation
    assert "| T-G1 highway instants with a plan graded >= 0.95, % | 78.9 (74.1 to 82.9) |" in evaluation
    assert "54.6 % of the time (52.1 to 56.9)" in doc and "78.9 % (74.1 to 82.9)" in doc


def _floats(node):
    if isinstance(node, float):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _floats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _floats(value)


def test_one_checkpoint_is_named_by_the_merge_the_closed_loop_test_and_the_pointers() -> None:
    assert _json(SECOND_RUN / "gen_ckpt.json")["sha256"] == CHECKPOINT_SHA256
    assert _json(TRIAL / "totals.json")["runs"]["checkpoint_sha256"] == [CHECKPOINT_SHA256]
    assert _json(TRIAL / "gate_017.json")["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert CHECKPOINT_SHA256 in (EXP / "POINTERS.md").read_text()
    assert CHECKPOINT_SHA256 in (EXP / "07_closed_loop_test" / "adapter_0.1.7_as_built.diff").read_text()


def test_every_pointer_row_carries_a_size_and_a_full_sha256() -> None:
    rows = [
        line
        for line in (EXP / "POINTERS.md").read_text().splitlines()
        if "{weights}" in line or "{runs}" in line
    ]
    hashed = [line for line in rows if line.startswith("|") and re.search(r"\| \d+ \|", line)]
    assert len(hashed) >= 40
    for line in hashed:
        assert re.search(r"`[0-9a-f]{64}` \|$", line), line


# ---------------------------------------------------------------- facts


def test_the_spacing_term_was_trained_with_weight_zero() -> None:
    """It is implemented (train_patch/drive_jepa_agent_s10.diff); the shipped generator did not use it."""
    assert _json(SECOND_RUN / "gen_ckpt.json")["lam"] == 0.0
    assert _json(SECOND_RUN / "final_choice.json")["final_run"]["state"]["lam"] == 0.0
    assert "--lam 0" in (SECOND_RUN / "final_launch.json").read_text()
    assert "spacing" in (SECOND_RUN / "train_patch" / "drive_jepa_agent_s10.diff").read_text()


def test_the_scorer_in_the_shipped_file_is_the_first_runs_untouched_by_the_second() -> None:
    ckpt = _json(SECOND_RUN / "gen_ckpt.json")
    assert ckpt["scorer_tensors"] == ckpt["scorer_tensors_bit_identical_to_pass9_in_merged_file"] == 30
    assert ckpt["encoder_tensors_bit_identical_to_shipped"] == ckpt["tensors_from_shipped_encoder"] == 294
    trainer = (SECOND_RUN / "s10_gen_train.py").read_text()
    assert "for p_ in m.scorer.parameters(): p_.requires_grad_(False)" in trainer


# ---------------------------------------------------------------- hygiene


def test_no_driver_source_and_nothing_binary_is_in_the_record() -> None:
    """The driver lives in paddock; weights, arrays and frames live nowhere in git."""
    driver_files = {"planner.py", "bridge.py", "divider.py", "momentum.py", "real_model.py", "batch_fix.py"}
    banned = {
        ".ckpt",
        ".pt",
        ".pth",
        ".npy",
        ".npz",
        ".pkl",
        ".parquet",
        ".png",
        ".jpg",
        ".mp4",
        ".usdz",
        ".asl",
    }
    for path in EXP.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(EXP).as_posix()
        if path.name == "batch_fix.py":
            # The training-side original the generator was trained under: a patch
            # to the upstream refiner, not driver code.
            assert rel == "12_second_training_run/train_patch/batch_fix.py", rel
            continue
        assert path.name not in driver_files, f"driver source in the record: {rel}"
        assert path.suffix not in banned, f"binary or derived data in the record: {rel}"


def test_upstream_training_files_are_diffs_not_copies() -> None:
    for rel in (
        "05_small_overfit_check/train_patch/dataset.diff",
        "05_small_overfit_check/train_patch/drive_jepa_agent.diff",
        "12_second_training_run/train_patch/drive_jepa_agent_s10.diff",
    ):
        assert (EXP / rel).is_file(), rel
    for rel in (
        "05_small_overfit_check/train_patch/dataset_s35.py",
        "05_small_overfit_check/train_patch/drive_jepa_agent_s35.py",
        "12_second_training_run/train_patch/drive_jepa_agent_s10.py",
        "05_small_overfit_check/train_patch/upstream_ref",
    ):
        assert not (EXP / rel).exists(), f"a full copy of an upstream file is committed: {rel}"


def test_every_script_still_tokenizes_after_its_paths_were_replaced() -> None:
    """Tokenized, not parsed: the scripts target two interpreters, neither of them this one.

    The trainers ran under python 3.9 and the closed-loop test tooling under 3.12, and one
    of the latter uses an f-string form 3.11 rejects. A substitution that broke
    a string literal or a bracket would still fail here.
    """
    scripts = sorted(EXP.rglob("*.py"))
    assert len(scripts) >= 160
    for path in scripts:
        with tokenize.open(path) as handle:
            list(tokenize.generate_tokens(handle.readline))


def _original_folders() -> dict[str, str]:
    """Original working folder -> folder here, read from the one table that records it."""
    doc = (REPO_ROOT / "docs" / "drivejepa-retrain.md").read_text()
    table = doc.split("## Where this came from", 1)[1]
    pairs = re.findall(r"^\| `([^`{]+)/` \| `([^`]+)/` \|$", table, flags=re.M)
    assert len(pairs) == 19
    return {old: new for new, old in pairs}


def test_as_run_lists_original_paths_that_map_to_real_files_here() -> None:
    """AS_RUN.sha256 keeps each file's original path; the doc's table says where that is now."""
    folders = _original_folders()
    header, *lines = (EXP / "AS_RUN.sha256").read_text().splitlines()
    assert header.startswith("# ")
    assert len(lines) == 207
    for line in lines:
        digest, original = line.split("  ", 1)
        assert re.fullmatch(r"[0-9a-f]{64}", digest), line
        match = max((old for old in folders if original.startswith(old + "/")), key=len, default=None)
        assert match, f"no folder in the table for {original}"
        assert (EXP / folders[match] / original[len(match) + 1 :]).is_file(), original


def test_the_working_folder_names_appear_only_in_that_table() -> None:
    """Folder names are plain; the names the work was done under live in one table."""
    old_names = {old.split("/")[-1] for old in _original_folders()}
    for path in EXP.iterdir():
        assert path.name not in old_names, path.name
    for name in ("README.md", "docs/data.md", "experiments/drivejepa_retrain/README.md"):
        text = (REPO_ROOT / name).read_text()
        for old in old_names:
            assert old not in text, f"{old} in {name}"
