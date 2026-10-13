"""Write a trained checkpoint in the form the vendor agent loads.

Training saves `_pad_model.state_dict()` -- the model, and nothing else. The
vendor's `DriveJEPAAgent.initialize` instead does

    torch.load(path)["state_dict"]

and strips an `agent._pad_model` prefix off every key, because its released
checkpoint came out of a Lightning trainer wrapping the agent wrapping the model.
So a file this repo writes cannot be handed to paddock as-is: same tensors, two
levels of naming apart.

This writes `agent.ckpt` beside `model.pt` in that shape. It exists so paddock's
drivejepa adapter needs no change at all -- point DRIVEJEPA_CHECKPOINT at the file
and the released evaluation path runs a finetuned model. Keeping the conversion on
the training side is the point: paddock stays ignorant of how the weights were made.

    python -m gallops.train.export --run <run dir>      # every step_* in it
"""

from __future__ import annotations

import argparse
from pathlib import Path

PREFIX = "_pad_model."


def export(directory: Path, torch) -> Path:
    """`model.pt` -> `agent.ckpt`, the vendor's shape. Returns the written path."""
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if any(key.startswith(PREFIX) for key in state):
        raise SystemExit(f"{directory}: model.pt is already prefixed; nothing to do")
    out = directory / "agent.ckpt"
    torch.save({"state_dict": {PREFIX + key: value for key, value in state.items()}}, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a run of step_* directories")
    parser.add_argument("--step", help="only this step directory")
    args = parser.parse_args()

    import torch

    directories = [args.run / args.step] if args.step else sorted(args.run.glob("step_*"))
    if not directories:
        raise SystemExit(f"no step_* under {args.run}")
    for directory in directories:
        if not (directory / "model.pt").exists():
            continue
        path = export(directory, torch)
        print(f"  {path}  ({path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
