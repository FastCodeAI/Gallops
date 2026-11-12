"""What the memory picker's trainer writes about its weights, and when it refuses to train (no torch)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from gallops.config import SiteConfig
from gallops.memory import train


def _notes(mode: str, length: int = 4) -> dict:
    return train.sidecar_notes(
        mode, length, 0, 6, [], True, ["log_b", "log_a"], 878, {"grades/scores.npy": "ab" * 32}
    )


def test_the_wiped_arm_is_written_as_length_0_whatever_the_flag_says():
    assert _notes("reset", 4)["length"] == 0
    assert _notes("carry", 4)["length"] == 4


def test_the_sidecar_names_the_logs_and_inputs_the_weights_saw():
    notes = _notes("carry")
    assert notes["train_logs_used"] == 2 and notes["train_logs_in_split"] == 878
    # the digest is of the sorted names, so it does not depend on the order they were found in
    assert notes["train_logs_used_sha256"] == hashlib.sha256(b"log_a\nlog_b").hexdigest()
    assert notes["inputs_sha256"] == {"grades/scores.npy": "ab" * 32}


def test_a_run_that_holds_a_checkpoint_is_refused_before_training(tmp_path):
    cfg = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    runs = [train.run_name(mode, 4, 0) for mode in ("carry", "reset")]
    assert runs == ["h15-r3-a-L4-s0", "h15-r3-b-L0-s0"]
    train.refuse_existing_runs(cfg, runs)  # nothing there yet: no refusal
    (cfg.checkpoint_dir(train.MODEL_NAME, runs[1], 4)).mkdir(parents=True)
    with pytest.raises(SystemExit, match="h15-r3-b-L0-s0"):
        train.refuse_existing_runs(cfg, runs)


def test_input_digests_cover_every_file_the_trainer_reads(tmp_path):
    for name in train.INPUTS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(name.encode())
    digests = train.input_digests(tmp_path)
    assert set(digests) == {*train.INPUTS, "selection_logs.txt"}
    assert digests["grades/scores.npy"] == hashlib.sha256(b"grades/scores.npy").hexdigest()


def test_every_file_data_reads_is_in_the_digest_list():
    """`INPUTS` is a second list of `Data`'s reads; a read added to one and not the other is caught here."""
    source = Path(train.__file__).read_text()
    data = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == "Data")
    strings = {n.value for n in ast.walk(data) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    read = {s for s in strings if s.endswith((".npy", ".npz"))}
    assert read, "found no file names in Data"
    assert read <= {Path(name).name for name in train.INPUTS}, read


def test_every_method_the_trainer_calls_on_its_data_exists():
    """The structural guard (tests/test_expert.py) for this trainer: `make check` has no torch, so
    nothing builds a `Data`, and a deleted method would only show at the preflight."""
    tree = ast.parse(Path(train.__file__).read_text())
    data = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Data")
    methods = {n.name for n in data.body if isinstance(n, ast.FunctionDef)}
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "data"
    }
    assert called, "found no calls on `data`"
    assert called <= methods, f"the trainer calls Data.{called - methods}, which do not exist"
