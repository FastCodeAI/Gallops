"""A drivejepa_b2d checkpoint reaches {weights} whole and with a true sidecar, or not at all.

The writer itself needs torch and a 3 GB checkpoint; what it publishes and what its sidecar claims are decided
in `drivejepa_b2d.sidecar`, which needs neither, so they are tested here on synthetic files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gallops.checkpoint import Checkpoint, read
from gallops.drivejepa_b2d import sidecar

WEIGHTS = "w.ckpt"


def a_record(**overrides) -> Checkpoint:
    fields = dict(model="m", step=1, train_git_sha="abc", train_run="r", notes={"note": "x"})
    fields.update(overrides)
    return Checkpoint(**fields)


def save_ok(staging: Path) -> dict:
    (staging / WEIGHTS).write_bytes(b"weights")
    return {"probe": "ok"}


def test_a_published_checkpoint_is_whole(tmp_path: Path) -> None:
    out = tmp_path / "m" / "r" / "step_000000001"
    rec = sidecar.publish(out, a_record(), WEIGHTS, save_ok)
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint.json", WEIGHTS]
    on_disk = read(out)
    assert on_disk.weights_sha256 == sidecar.sha256(out / WEIGHTS) == rec.weights_sha256
    assert on_disk.notes == {"note": "x", "probe": "ok"}


def test_a_failed_probe_leaves_nothing_in_weights(tmp_path: Path) -> None:
    """The bug this replaced: the weights were saved into the step directory before the probe ran, so a failed
    probe left weights with no sidecar where paddock looks."""

    def save_then_fail(staging: Path) -> dict:
        (staging / WEIGHTS).write_bytes(b"weights")
        raise SystemExit("PROBE FAILED")

    out = tmp_path / "m" / "r" / "step_000000001"
    with pytest.raises(SystemExit, match="PROBE FAILED"):
        sidecar.publish(out, a_record(), WEIGHTS, save_then_fail)
    assert not (tmp_path / "m" / "r").exists()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_a_failed_first_write_does_not_block_the_retry(tmp_path: Path) -> None:
    def save_crash(staging: Path) -> dict:
        raise RuntimeError("out of disk")

    out = tmp_path / "m" / "r" / "step_000000001"
    with pytest.raises(RuntimeError):
        sidecar.publish(out, a_record(), WEIGHTS, save_crash)
    sidecar.publish(out, a_record(), WEIGHTS, save_ok)
    assert (out / "checkpoint.json").exists()


def test_one_fit_is_one_train_run(tmp_path: Path) -> None:
    """A rerun that keeps a different epoch must not land beside the first under the same train run."""
    sidecar.publish(tmp_path / "m" / "r" / "step_000000003", a_record(step=3), WEIGHTS, save_ok)
    with pytest.raises(SystemExit, match="already holds a checkpoint"):
        sidecar.publish(tmp_path / "m" / "r" / "step_000000005", a_record(step=5), WEIGHTS, save_ok)
    assert [p.name for p in (tmp_path / "m" / "r").iterdir()] == ["step_000000003"]
