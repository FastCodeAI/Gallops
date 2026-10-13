"""Publishing a drivejepa_b2d checkpoint: its step directory appears whole, sidecar included, or not at all.

Torch-free, so `make check` tests it: the writer passes in the function that saves the weights. Paddock reads
{weights} while this writes to it, so weights with no sidecar -- what a failed probe or a kill between the two
writes used to leave -- would look like a usable checkpoint.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from gallops.checkpoint import Checkpoint, staged, write


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def optimizer_step(head_metrics: dict[str, Any]) -> int:
    """The `step` of a checkpoint built from a split head: optimizer steps of that head's stage up to its kept
    epoch. Not the epoch -- `Checkpoint.step` is the optimizer step, and anything that orders or compares
    checkpoints by it would read an epoch as a step count 442 times too small (Split-v4's head)."""
    if "steps_per_epoch" not in head_metrics:
        raise SystemExit(
            "the head's metrics.json has no steps_per_epoch: it was trained before train_split_head recorded "
            "it, so its optimizer step is not known -- retrain it with this repo's train_split_head"
        )
    return int(head_metrics["best_epoch"]) * int(head_metrics["steps_per_epoch"])


def refuse_existing_run(run_dir: Path) -> None:
    """One fit is one train run. A rerun that kept a different epoch would otherwise sit beside the first
    under the same name, and paddock groups a train run as one sweep. An empty directory is not a run."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing: {run_dir} already holds a checkpoint; pass a new --train-run")


def publish(
    out_dir: Path, record: Checkpoint, weights_file: str, save: Callable[[Path], dict[str, Any]]
) -> Checkpoint:
    """Write `out_dir` whole. `save(tmp)` writes the weights into a staging directory, checks them, and
    returns notes to add (what the check found); the sidecar is written beside them and the directory renamed
    into place (`checkpoint.staged`). `record` is built, and so validated, before anything is written."""
    run_dir = out_dir.parent
    refuse_existing_run(run_dir)
    try:
        with staged(out_dir) as tmp:
            notes = save(tmp)
            record = replace(
                record, weights_sha256=sha256(tmp / weights_file), notes={**record.notes, **notes}
            )
            write(tmp, record)
    except BaseException:
        with contextlib.suppress(OSError):
            run_dir.rmdir()  # succeeds only if empty: a failed first write leaves no run behind
        raise
    return record
