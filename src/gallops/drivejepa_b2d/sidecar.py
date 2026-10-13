"""Publishing a drivejepa_b2d checkpoint: its step directory appears whole, sidecar included, or not at all.

Torch-free, so `make check` tests it: the writer passes in the function that saves the weights. Paddock reads
{weights} while this writes to it, so weights with no sidecar -- what a failed probe or a kill between the two
writes used to leave -- would look like a usable checkpoint.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
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


def parse_env(pairs: list[str]) -> dict[str, str]:
    """`KEY=VALUE` arguments as a dict. A split-head checkpoint is its weights *plus* the switches paddock's
    adapter is driven with: under the adapter's defaults it picks by column 5, whose values moved when the
    hidden layers were retrained, which is a different model. So the switches belong in its sidecar."""
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--policy-env wants KEY=VALUE, got {pair!r}")
        if key in env:
            raise SystemExit(f"--policy-env {key} given twice")
        env[key] = value
    return env


#: The Final's area head, copied into the step directory beside the weights. The policy loads it from the
#: path in DRIVEJEPA_SPLIT_AREA; the sidecar names the file, which a site resolves against the checkpoint
#: directory.
AREA_HEAD_FILE = "area_head.pt"


def with_area_head(env: dict[str, str]) -> dict[str, str]:
    """`env` plus the switch that turns the area factor on, naming the copy in the checkpoint directory.
    Without it the same weights drive with the area factor off, which is not the Final."""
    if "DRIVEJEPA_SPLIT_AREA" in env:
        raise SystemExit("DRIVEJEPA_SPLIT_AREA comes from --area, which ships the head with the checkpoint")
    return {**env, "DRIVEJEPA_SPLIT_AREA": AREA_HEAD_FILE}


def input_notes(manifests: list[Path], heldout: Path) -> dict[str, Any]:
    """What pins a checkpoint's inputs. Each tracked data manifest lists every file with its sha256, so the
    manifest's own sha256 identifies the dataset; the held-out clips are recorded by name, as read, with the
    list's sha256. Without them a checkpoint trained on a regenerated dump looks identical."""
    return {
        "dataset_manifests": {m.name: sha256(m) for m in manifests},
        "holdout_scenes": json.loads(heldout.read_text())["val"],
        "holdout_sha256": sha256(heldout),
    }


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
