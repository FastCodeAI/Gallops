"""Stages run2-checkpoint and run3a-checkpoint: publish a run's kept epoch with its sidecar.

Copies the Lightning checkpoint the run kept (Run 2: epoch_19.ckpt, the last of 20, no selection; Run 3A:
epoch_13.ckpt, the best held-out val/score_epoch) to {weights}/<model>/<train_run>/step_<n>/ with
checkpoint.json beside it (docs/checkpoints.md). The step directory is staged and renamed into place only
once the copy is checked and the sidecar written (drivejepa_b2d.sidecar.publish), so {weights} never holds
weights without a sidecar, and a train run that already holds a checkpoint is refused. The source checkpoint
is only read.

Check: the copy's sha256 equals the source's, and its state dict reloads with the tensor count and parameter
total the sidecar records.

What the sidecar says, and where each value comes from (`record_for`, torch-free so the tests build it):
- `step`: the checkpoint's own Lightning global_step, i.e. optimizer steps (Run 2's epoch_19.ckpt: 26,580; Run
  3A's epoch_13.ckpt: 21,616). The epoch goes in `notes`.
- `params_m`: the distinct tensors in the file. The four refinement rounds share one module, so their keys
  share storage: counted once (Run 2: 292 keys over 100 storages, 7.77 M, not the 15.96 M a sum over keys
  gives). The frozen WA-JEPA encoder that runs at inference is not in the file and not counted.
- `precision`: the dtypes actually saved (fp32 when every floating tensor is float32). The training precision,
  Lightning's 16-mixed, goes in `notes`.
- `train_git_sha`: the gallops commit that published it. The run trained in a Drive-JEPA checkout at the
  overlay's upstream commit, which `notes.upstream` records.

The sidecar also records what pins the run besides its weights: the overlay manifest's sha256 (the exact
agent code, recipes/drivejepa_wajepa/overlay/<run>.json), how it was selected, and the paddock adapter
switches it is driven with (--policy-env).

    python -m gallops.drivejepa_wajepa.write_checkpoint --src <epoch_N.ckpt> --overlay <overlay/<run>>
        --model wajepa_dj --train-run run2 --trained-on navsim:navtrain [--selection-split ...]
        [--policy-env KEY=VALUE ...] [--note "..."]

--src defaults (in the recipe) to the recipe's own retrain output. To publish the checkpoint the record pins
and the gate judged, pass --src at it (experiments/<ID>/POINTERS.md names where it sits).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from gallops.checkpoint import Checkpoint, TrainedOn
from gallops.config import SiteConfig
from gallops.drivejepa_b2d.provenance import git_sha
from gallops.drivejepa_b2d.sidecar import parse_env, publish, sha256

#: The precision Lightning trained these runs at; recorded in notes, not as the sidecar's `precision`, which
#: describes the saved tensors.
TRAINING_PRECISION = "16-mixed"


def trained_on(pairs: list[str]) -> tuple[TrainedOn, ...]:
    out = []
    for pair in pairs:
        dataset, sep, split = pair.partition(":")
        if not sep or not dataset or not split:
            raise SystemExit(f"--trained-on wants DATASET:SPLIT, got {pair!r}")
        out.append(TrainedOn(dataset, split))
    return tuple(out)


def summarize(state: dict[str, Any]) -> tuple[int, set[str]]:
    """(elements in the distinct tensors, floating dtypes) of a state dict. Keys that share one tensor (tied
    weights: the shared refinement rounds) count once."""
    seen: set[tuple[int, int]] = set()
    n = 0
    dtypes: set[str] = set()
    for v in state.values():
        if not hasattr(v, "numel"):
            continue
        if v.is_floating_point():
            dtypes.add(str(v.dtype).removeprefix("torch."))
        key = (v.data_ptr(), v.numel())
        if v.numel() and key in seen:
            continue
        seen.add(key)
        n += int(v.numel())
    return n, dtypes


def precision_of(dtypes: set[str]) -> str:
    """The sidecar's precision for tensors saved with these floating dtypes."""
    return "fp32" if dtypes == {"float32"} else "mixed"


def record_for(
    *,
    global_step: int,
    epoch: int,
    n_params: int,
    dtypes: set[str],
    model: str,
    train_run: str,
    trained_on_pairs: list[str],
    selection_split: str | None,
    src_name: str,
    src_sha256: str,
    manifest_name: str,
    manifest_sha256: str,
    upstream: dict[str, Any],
    policy_env: dict[str, str],
    note: str,
    publisher_sha: str,
) -> Checkpoint:
    """The sidecar for one kept epoch (module docstring says where each value comes from)."""
    return Checkpoint(
        model=model,
        step=int(global_step),
        train_git_sha=publisher_sha,
        train_run=train_run,
        trained_on=trained_on(trained_on_pairs),
        selection_split=selection_split,
        params_m=round(n_params / 1e6, 2),
        precision=precision_of(dtypes),
        notes={
            "epoch": int(epoch),
            "step_is": "Lightning global_step (optimizer steps)",
            "params_m_is": (
                "distinct tensors in the file, tied keys once; the frozen WA-JEPA encoder is not in it"
            ),
            "training_precision": TRAINING_PRECISION,
            "train_git_sha_is": (
                "the gallops commit that published it; the run trained at notes.upstream with the overlay"
            ),
            "source_checkpoint": src_name,
            "source_sha256": src_sha256,
            "overlay_manifest": manifest_name,
            "overlay_manifest_sha256": manifest_sha256,
            "upstream": upstream,
            "policy_env": policy_env,
            "note": note,
        },
    )


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the run's kept Lightning checkpoint")
    ap.add_argument("--overlay", type=Path, required=True, help="recipes/drivejepa_wajepa/overlay/<run>")
    ap.add_argument("--model", required=True, help="the paddock adapter that drives it")
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--trained-on", action="append", default=[], help="DATASET:SPLIT, repeatable")
    ap.add_argument("--selection-split", default=None)
    ap.add_argument("--policy-env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)

    manifest = a.overlay.with_suffix(".json")
    ckpt = torch.load(a.src, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    n_params, dtypes = summarize(state)
    record = record_for(
        global_step=ckpt["global_step"],
        epoch=ckpt["epoch"],
        n_params=n_params,
        dtypes=dtypes,
        model=a.model,
        train_run=a.train_run,
        trained_on_pairs=a.trained_on,
        selection_split=a.selection_split,
        src_name=a.src.name,
        src_sha256=sha256(a.src),
        manifest_name=manifest.name,
        manifest_sha256=sha256(manifest),
        upstream=json.loads(manifest.read_text())["upstream"],
        policy_env=parse_env(a.policy_env),
        note=a.note,
        publisher_sha=git_sha(),
    )
    cfg = SiteConfig.load()
    out_dir = cfg.checkpoint_dir(a.model, a.train_run, record.step)

    def save(tmp: Path) -> dict:
        dest = tmp / a.src.name
        shutil.copyfile(a.src, dest)
        if sha256(dest) != record.notes["source_sha256"]:
            raise SystemExit("the copy differs from the source checkpoint")
        reloaded = torch.load(dest, map_location="cpu", weights_only=False)["state_dict"]
        if len(reloaded) != len(state) or summarize(reloaded)[0] != n_params:
            raise SystemExit("the copy's state dict does not reload whole")
        return {"tensors": len(state)}

    published = publish(out_dir, record, a.src.name, save)
    print(
        f"wrote {out_dir} (step {published.step}, epoch {published.notes['epoch']}, {published.params_m} M)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
