"""Stages run2-checkpoint and run3a-checkpoint: publish a run's kept epoch with its sidecar.

Copies the Lightning checkpoint the run kept (Run 2: epoch_19.ckpt, the last of 20, no selection; Run 3A:
epoch_13.ckpt, the best held-out val/score_epoch) to {weights}/<model>/<train_run>/step_<n>/ with
checkpoint.json beside it (docs/checkpoints.md). The step directory is staged and renamed into place only
once the copy is checked and the sidecar written (drivejepa_b2d.sidecar.publish), so {weights} never holds
weights without a sidecar, and a train run that already holds a checkpoint is refused. The source checkpoint
is only read.

Check: the copy's sha256 equals the source's, and its state dict loads with the tensor count and parameter
total the sidecar records. `step` is the checkpoint's own Lightning global_step.

The sidecar records what pins the run besides its weights: the overlay manifest's sha256 (the exact agent
code, recipes/drivejepa_wajepa/overlay/<run>.json), the upstream commit, the epoch, how it was selected, and
the paddock adapter switches it is driven with (--policy-env).

    python -m gallops.drivejepa_wajepa.write_checkpoint --src <epoch_N.ckpt> --overlay <overlay/<run>>
        --model wajepa_dj --train-run run2 --trained-on navsim:navtrain [--selection-split ...]
        [--policy-env KEY=VALUE ...] [--note "..."]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from gallops.checkpoint import Checkpoint, TrainedOn
from gallops.config import SiteConfig
from gallops.drivejepa_b2d.provenance import git_sha
from gallops.drivejepa_b2d.sidecar import parse_env, publish, sha256


def trained_on(pairs: list[str]) -> tuple[TrainedOn, ...]:
    out = []
    for pair in pairs:
        dataset, sep, split = pair.partition(":")
        if not sep or not dataset or not split:
            raise SystemExit(f"--trained-on wants DATASET:SPLIT, got {pair!r}")
        out.append(TrainedOn(dataset, split))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
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
    n_params = sum(int(v.numel()) for v in state.values())
    record = Checkpoint(
        model=a.model,
        step=int(ckpt["global_step"]),
        train_git_sha=git_sha(),
        train_run=a.train_run,
        trained_on=trained_on(a.trained_on),
        selection_split=a.selection_split,
        params_m=round(n_params / 1e6, 2),
        precision="16-mixed",
        notes={
            "epoch": int(ckpt["epoch"]),
            "source_checkpoint": a.src.name,
            "source_sha256": sha256(a.src),
            "overlay_manifest": manifest.name,
            "overlay_manifest_sha256": sha256(manifest),
            "upstream": json.loads(manifest.read_text())["upstream"],
            "policy_env": parse_env(a.policy_env),
            "note": a.note,
        },
    )
    cfg = SiteConfig.load()
    out_dir = cfg.checkpoint_dir(a.model, a.train_run, record.step)

    def save(tmp: Path) -> dict:
        dest = tmp / a.src.name
        shutil.copyfile(a.src, dest)
        if sha256(dest) != record.notes["source_sha256"]:
            raise SystemExit("the copy differs from the source checkpoint")
        reloaded = torch.load(dest, map_location="cpu", weights_only=False)["state_dict"]
        if len(reloaded) != len(state) or sum(int(v.numel()) for v in reloaded.values()) != n_params:
            raise SystemExit("the copy's state dict does not reload whole")
        return {"tensors": len(state)}

    published = publish(out_dir, record, a.src.name, save)
    print(
        f"wrote {out_dir} (step {published.step}, epoch {published.notes['epoch']}, {published.params_m} M)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
