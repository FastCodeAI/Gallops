"""Which nuScenes train scenes train, and which choose the epoch: the held-out split, by scene.

    python -m gallops.drivejepa_aux.split --extract <dir>     # writes HELDOUT_FILE, once

Held out by whole scene, never by keyframe: a scene's keyframes are 0.5 s apart and nearly the
same picture, so a keyframe split puts near-duplicates on both sides and a held-out loss that
measures memory (swerve's `split_by_scene`, for the same reason).

The candidates are the extract's scenes with at least one trainable keyframe, so the share held
out is a share of what trains. The file keeps them beside the draw, so `load` -- and `make check`
-- redraws the list from the file alone and refuses one that was edited by hand. numpy only.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from gallops.config import REPO_ROOT

#: 15% of scenes, drawn with seed 0: the plan's rule (docs/drivejepa-aux-heads.md, step 4).
HELDOUT_FRACTION = 0.15
HELDOUT_SEED = 0
HELDOUT_FILE = REPO_ROOT / "recipes" / "drivejepa_aux" / "heldout_scenes.json"


def draw(candidates, fraction: float = HELDOUT_FRACTION, seed: int = HELDOUT_SEED) -> list[str]:
    """`fraction` of the candidate scenes, chosen by `seed`, sorted. The same names give the same draw
    in any order."""
    names = sorted(set(candidates))
    count = int(round(fraction * len(names)))
    picked = np.random.default_rng(seed).choice(len(names), size=count, replace=False)
    return sorted(names[i] for i in picked)


def load(path: Path = HELDOUT_FILE) -> dict:
    """The committed split, refused unless its list is what its own rule draws from its candidates."""
    split = json.loads(Path(path).read_text())
    redrawn = draw(split["candidates"], split["fraction"], split["seed"])
    if split["held_out"] != redrawn:
        raise SystemExit(f"{path}: held_out is not draw(candidates, {split['fraction']}, {split['seed']})")
    return split


def partition(scenes, held_out) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the rows whose scene trains, and of those held out."""
    held = np.isin(np.asarray(scenes), np.asarray(sorted(held_out)))
    return np.flatnonzero(~held), np.flatnonzero(held)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", type=Path, required=True, help="the `nusc extract` directory")
    ap.add_argument("--out", type=Path, default=HELDOUT_FILE)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the split is drawn once, before any run")
    with open(args.extract / "extract.pkl", "rb") as f:
        ticks = pickle.load(f)["ticks"]
    candidates = sorted(name for name, rows in ticks.items() if rows)
    held = draw(candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "rule": "draw(candidates, fraction, seed) in gallops.drivejepa_aux.split",
                "candidates_are": "nuScenes v1.0-trainval train scenes with a trainable keyframe "
                "(nusc.train_scene_names: neuroNCAP's and the reconstructed scenes excluded)",
                "fraction": HELDOUT_FRACTION,
                "seed": HELDOUT_SEED,
                "held_out": held,
                "candidates": candidates,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(held)} of {len(candidates)} scenes held out -> {args.out}")


if __name__ == "__main__":
    main()
