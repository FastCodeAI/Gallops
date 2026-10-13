"""The checkpoint sidecar: the entire interface between gallops and paddock.

Gallops trains models. Paddock scores them. Neither imports the other, and
neither ever will — paddock's core invariant is that it never imports a model,
and gallops has no business depending on an evaluation harness.

So the two communicate through a directory on disk:

    {weights}/<model>/<train_run>/step_<n>/
        checkpoint.json      <- this module writes it, paddock reads it
        model.safetensors

`checkpoint.json` is a small JSON file -- kilobytes at most, mostly the scene
list a run records in `notes`. It is deliberately not a
pickle, not a torch object, and not anything that requires either side to share
a dependency to read it. Paddock parses it with `json.load`; this module is
stdlib-only so it can be imported anywhere, including from a training script
running in an interpreter that has nothing else in common with paddock's.

Why a sidecar at all, rather than paddock inspecting the weights:

  - **Only the trainer knows the step and its own commit.** A weights file does
    not carry the git sha of the code that produced it, or which split it was
    trained on. Those are the facts that make a result reproducible, and they
    are lost the moment they are not written down at the time.
  - **Hashing 20 GB per run is not acceptable.** Paddock computes a run id
    before it starts a node; making that wait on a checkpoint digest would add
    minutes to every run. The digest, when we want one, is computed once here
    and cached in the sidecar.
  - **It survives rsync.** Identity by (path, size, mtime) breaks the moment a
    checkpoint is copied between machines, which is exactly when two sites are
    about to disagree about what they measured.

What paddock does with it is its own business, but the fields exist because
paddock needs them: `step`/`train_git_sha` for provenance in `run.json`,
`trained_on` for the contamination declaration on the leaderboard, and
`params_m`/`macs_g`/`precision` for the model card.

**Adding a field is fine; changing the meaning of one is not.** Bump
SIDECAR_VERSION if a reader could misinterpret an old file, and keep readers
tolerant of unknown keys so a newer trainer does not break an older scorer.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Bump only when an existing field changes meaning. New optional fields do not
#: need a bump, because readers ignore what they do not recognise.
SIDECAR_VERSION = 1

#: The filename is part of the contract. Paddock looks for exactly this.
SIDECAR_NAME = "checkpoint.json"

#: Precisions we know how to talk about. `mixed` means the artifact holds more
#: than one -- say bf16 weights with an fp32 master copy -- which is a different
#: claim from either alone and should not be flattened into one.
PRECISIONS = ("fp32", "bf16", "fp16", "int8", "mixed")


class SidecarError(Exception):
    """A sidecar is missing, malformed, or claims something impossible."""


@dataclass(frozen=True)
class TrainedOn:
    """One (dataset, split) pair this checkpoint saw during training.

    `dataset` must match a key in paddock's tracked `datasets.yaml` -- nuscenes,
    kitscenes, navsim -- because that is what makes contamination detectable at
    scoring time. A dataset paddock does not know about is still worth
    recording; it simply cannot be cross-checked.

    Record every split that contributed, including ones used only for early
    stopping. A split you selected on is not a split you can report on, and the
    distinction is invisible six months later if nobody wrote it down.
    """

    dataset: str
    split: str
    #: Fraction of the split actually consumed, when it was not all of it.
    #: `None` means "all of it", which is different from 1.0 meaning "we
    #: measured it and it was all of it".
    fraction: float | None = None


@dataclass(frozen=True)
class Checkpoint:
    """Everything paddock needs to know about a set of weights.

    Required fields are the ones without which a result cannot be attributed:
    which model, at what step, from what code. Everything else is optional
    because a first training run should not be blocked on knowing its own MAC
    count -- but each optional field disables something downstream when absent,
    and the docstrings say which.
    """

    #: Model family, not the run. Becomes the paddock adapter name, so it must
    #: be a valid directory name and stable across steps.
    model: str

    #: Optimizer step. The one number that orders checkpoints within a run.
    step: int

    #: Commit of the *training* repo that produced these weights. Without it, a
    #: result cannot be reproduced even with the data and the seed.
    train_git_sha: str

    #: Label for the training run this checkpoint belongs to. Free-form, but
    #: keep it stable within a run -- it is how a sweep is grouped in a report.
    train_run: str

    created_utc: str = field(default_factory=lambda: _now())

    #: What this model was trained on. **An empty list means "we did not record
    #: it", not "it was trained on nothing"** -- a reader cannot tell those
    #: apart, so write it down. Paddock uses this to mark a row whose benchmark
    #: overlaps the training data.
    trained_on: tuple[TrainedOn, ...] = ()

    #: The split checkpoints were selected on, if any. Reporting a headline
    #: number from this split is selection bias, not a result. See
    #: docs/conventions.md.
    selection_split: str | None = None

    #: Millions of parameters. Absent means paddock cannot fill the deployment
    #: column for this model.
    params_m: float | None = None

    #: Billions of multiply-accumulates for one forward pass at the model's
    #: declared input shape. Absent means the same.
    macs_g: float | None = None

    precision: str | None = None

    #: sha256 of the weights file, when someone has paid for it. Computed once
    #: here rather than once per evaluation run.
    weights_sha256: str | None = None

    #: Anything else worth carrying. Kept free-form on purpose: a training run
    #: knows things we have not thought to make fields for yet.
    notes: dict[str, Any] = field(default_factory=dict)

    version: int = SIDECAR_VERSION

    def __post_init__(self) -> None:
        if not self.model or "/" in self.model:
            raise SidecarError(f"model must be a non-empty directory-safe name, got {self.model!r}")
        if self.step < 0:
            raise SidecarError(f"step must be non-negative, got {self.step}")
        if not self.train_git_sha:
            raise SidecarError(
                "train_git_sha is required -- a checkpoint whose code is unknown "
                "cannot be reproduced, and that is discovered far too late"
            )
        if self.precision is not None and self.precision not in PRECISIONS:
            raise SidecarError(f"precision {self.precision!r} not one of {', '.join(PRECISIONS)}")
        if self.selection_split is not None and not self.trained_on:
            raise SidecarError(
                "selection_split is set but trained_on is empty -- if you selected on a "
                "split, you know what this model saw, so record it"
            )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["trained_on"] = [_drop_none(t) for t in out["trained_on"]]
        return _drop_none(out, keep=("trained_on", "notes"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Checkpoint:
        """Build from parsed JSON, tolerating fields we do not know.

        Unknown keys go to `notes` rather than raising, so a checkpoint written
        by a newer gallops is still readable by an older one. The alternative --
        refusing to read it -- turns a forward-compatible addition into an
        outage in the middle of a sweep.
        """
        known = {f for f in cls.__dataclass_fields__ if f != "notes"}
        notes = dict(raw.get("notes") or {})
        for key, value in raw.items():
            if key not in known and key != "notes":
                notes[key] = value

        missing = [k for k in ("model", "step", "train_git_sha", "train_run") if k not in raw]
        if missing:
            raise SidecarError(f"sidecar is missing required field(s): {', '.join(missing)}")

        return cls(
            model=raw["model"],
            step=int(raw["step"]),
            train_git_sha=raw["train_git_sha"],
            train_run=raw["train_run"],
            created_utc=raw.get("created_utc") or _now(),
            trained_on=tuple(
                TrainedOn(
                    dataset=t["dataset"],
                    split=t["split"],
                    fraction=t.get("fraction"),
                )
                for t in raw.get("trained_on") or ()
            ),
            selection_split=raw.get("selection_split"),
            params_m=raw.get("params_m"),
            macs_g=raw.get("macs_g"),
            precision=raw.get("precision"),
            weights_sha256=raw.get("weights_sha256"),
            notes=notes,
            version=int(raw.get("version", SIDECAR_VERSION)),
        )


def write(directory: Path | str, checkpoint: Checkpoint) -> Path:
    """Write the sidecar beside the weights. Returns the path written.

    Write this at the same moment the weights are saved, in the same function.
    A sidecar written later is a sidecar that gets skipped when a job is killed,
    and an unlabelled checkpoint is worth much less than no checkpoint at all --
    it looks usable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SIDECAR_NAME
    # Write-then-rename: a reader must never see a half-written sidecar, and a
    # sweep has readers running while a trainer is writing.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


@contextmanager
def staged(directory: Path | str) -> Iterator[Path]:
    """Build a checkpoint directory under a temporary name, then publish it whole.

    Yields an empty sibling of `directory`. Save the weights and `write()` the
    sidecar into it; on a clean exit it is renamed to `directory` in one step,
    and on any exception it is removed. A reader therefore sees a complete
    checkpoint or nothing -- never weights without their sidecar, and never new
    weights beside an old one, which is what a job killed between two in-place
    writes leaves behind.

    Refuses a `directory` that already exists. A step is written once: rewriting
    one in place is how two fits end up described by one sidecar.
    """
    directory = Path(directory)
    if directory.exists():
        raise SidecarError(f"{directory} already exists; a checkpoint is written once, never over another")
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A dot-prefixed sibling: the same filesystem, so the rename is atomic, and
    # nothing that globs `step_*` picks up a partial one.
    tmp = directory.parent / f".{directory.name}.{uuid.uuid4().hex[:8]}.partial"
    tmp.mkdir()
    try:
        yield tmp
        if not (tmp / SIDECAR_NAME).exists():
            raise SidecarError(f"refusing to publish {directory} without its {SIDECAR_NAME}")
        tmp.rename(directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read(directory: Path | str) -> Checkpoint:
    """Read the sidecar from a checkpoint directory.

    Raises `SidecarError` naming the directory, because the caller is usually a
    training script or an evaluation harness several layers up, and "file not
    found" without the path is the least useful error message there is.
    """
    directory = Path(directory)
    path = directory / SIDECAR_NAME if directory.is_dir() else directory
    if not path.exists():
        raise SidecarError(
            f"no {SIDECAR_NAME} in {directory} -- "
            f"a checkpoint without one cannot be attributed; see docs/checkpoints.md"
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SidecarError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SidecarError(f"{path} must contain a JSON object, got {type(raw).__name__}")
    return Checkpoint.from_dict(raw)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _drop_none(d: dict[str, Any], keep: tuple[str, ...] = ()) -> dict[str, Any]:
    """Omit null fields so a sidecar says what is known, not what is absent."""
    return {k: v for k, v in d.items() if v is not None or k in keep}
