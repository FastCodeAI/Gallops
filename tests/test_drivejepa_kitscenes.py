"""The Drive-JEPA KITScenes record: what can be checked with no GPU and no data.

The three experiments under experiments/H07_encoder_lora/ need a GPU, the
released 3.7 GB checkpoint and the KITScenes Multimodal scenes to run, so nothing
here runs them. What is tested is the part of their claims the committed files
can prove on their own:

    splits     the 50 picking scenes and the 482 training scenes are disjoint, and
               no held-out moment comes from either
    record     the doc quotes the recorded tables exactly, and states the gate's
               overall verdict as recorded
    pointers   every checkpoint the doc names has a size and sha256 in POINTERS.md
    hygiene    no weights, pickles, logs or caches; no location table; every
               script tokenizes
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
EXP = REPO_ROOT / "experiments" / "H07_encoder_lora"
DOC = EXP / "README.md"
SPLITS = EXP / "02_encoder_build" / "out" / "splits.json"
HELDOUT = EXP / "09_heldout_comparison" / "heldout_moments.csv"

# The checkpoints the doc names, by sha256 (H07 README.md, "Checkpoints").
CHECKPOINTS = {
    "released": "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05",
    "encoder only": "2568101fb234cc25f4e6781834c72659283cb8c5df5569cc9e42f248b9e2aaf0",
    "generator only": "e66e505e1a166c019bd80a4f8c5e02581323a36e4a0ec80b1239d1c21fe019c5",
    "generator and encoder": "c27aae6e392d98e849463445391490dab94ef9414e920cb874b251183adca915",
    "all three": "dc6c99d5b0298d32bf26f77f14f228969a7efb53822fcdd5feecac32b096eb45",
    "all three, merged export": "91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98",
}


@pytest.fixture(scope="module")
def tables():
    spec = importlib.util.spec_from_file_location("kit_doc_tables", EXP / "doc_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- splits


@pytest.fixture(scope="module")
def splits() -> dict:
    return json.loads(SPLITS.read_text())


@pytest.fixture(scope="module")
def heldout() -> list[dict[str, str]]:
    with HELDOUT.open() as fh:
        return list(csv.DictReader(fh))


def test_training_and_picking_scenes_are_disjoint(splits):
    train, picking = set(splits["train"]), set(splits["selection"])
    assert len(train) == 482 and len(picking) == 50
    assert not train & picking


def test_heldout_moments_come_from_neither_training_nor_picking_scenes(splits, heldout):
    used = set(splits["train"]) | set(splits["selection"])
    scenes = {row["scene_id"] for row in heldout}
    assert len(heldout) == 223 and len(scenes) == 140
    assert not scenes & used
    by_split = {s: sum(row["split"] == s for row in heldout) for s in ("val", "overlap_train_val")}
    assert by_split == {"val": 186, "overlap_train_val": 37}


# ---------------------------------------------------------------- record


@pytest.mark.parametrize("name", ["heldout", "groups", "paired", "gate"])
def test_doc_quotes_the_recorded_table(tables, name):
    assert tables.TABLES[name]() in DOC.read_text()


def test_doc_states_the_gate_verdict_as_recorded(tables):
    assert tables.gate_overall() == "FAIL"
    assert "overall verdict is **FAIL**" in DOC.read_text()


def test_the_five_model_table_reuses_the_four_model_rows_unchanged():
    def models(folder: str) -> dict:
        return json.loads((EXP / folder / "out" / "final_tables.json").read_text())["models"]

    four, five = models("09_heldout_comparison"), models("10_encoder_generator_scorer")
    # The four-model file keeps a mean and a median per value; the five-model file keeps the mean.
    for name, model in four.items():
        for group, metrics in model["groups"].items():
            for metric, value in metrics.items():
                mean = value["mean"] if isinstance(value, dict) else value
                assert five[name]["groups"][group][metric] == mean, (name, group, metric)


# ---------------------------------------------------------------- pointers


def test_every_named_checkpoint_has_a_pointer():
    pointers = (EXP / "POINTERS.md").read_text()
    doc = DOC.read_text()
    for name, sha in CHECKPOINTS.items():
        assert sha in pointers, name
        assert sha[:16] in doc, name


def test_pointer_hashes_are_well_formed():
    pointers = (EXP / "POINTERS.md").read_text()
    for cell in re.findall(r"`([0-9a-f]{20,})`", pointers):
        assert len(cell) == 64, cell
    for line in (EXP / "AS_RUN.sha256").read_text().splitlines():
        if line and not line.startswith("#"):
            assert re.fullmatch(r"[0-9a-f]{64}  \S.*", line), line


# ---------------------------------------------------------------- hygiene


def test_no_weights_arrays_logs_or_caches():
    banned = {".pt", ".ckpt", ".pkl", ".npy", ".npz", ".safetensors", ".log", ".jsonl", ".pyc", ".bin"}
    # Bytecode caches are gitignored and appear whenever doc_tables.py is imported; skip them.
    for path in EXP.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            assert path.suffix not in banned, path
            assert path.stat().st_size <= 1_000_000, path


def test_no_location_table():
    assert not list(EXP.rglob("split_locations.json"))


def test_every_script_tokenizes():
    scripts = sorted(EXP.rglob("*.py"))
    assert len(scripts) > 40
    for path in scripts:
        with path.open("rb") as fh:
            list(tokenize.tokenize(fh.readline))
