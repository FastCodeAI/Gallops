"""Each run's weights at a FIXED epoch, measured on the held-out scenes: numbers no selection touched.

    <drive-jepa python> -m gallops.drivejepa_aux.evaluate --cache <cache> --epoch 10 <run dir> ...

A run keeps the epoch with the lowest held-out planning loss, on the same held-out scenes its summary
then reports, and that loss moves by 0.01-0.03 from epoch to epoch: the minimum of eleven noisy
values flatters every arm, and a comparison of two arms each at its own minimum carries the bias of
both. CLAUDE.md's rule is to select on one split and report on another. Every epoch's weights are
saved (`epoch_NN.pt`), so the fix needs no retraining: measure every run at one epoch fixed in
advance -- the recipe's last, 10 -- which nothing chose.

Writes, beside each run's own files, `heldout_epochNN.npz` and `summary_epochNN.json` in the layout of
`heldout.npz` and `summary.json`, the fixed epoch in the place of the kept one, so `report --epoch NN`
reads them unchanged. The release is measured again in the same process, as every run measured it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.drivejepa_aux import cache, split, train

HEADS = {"C": "none", "T1": "upstream", "T2": "all"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories ({runs}/drivejepa_aux/runs/<name>)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--extract", type=Path, required=True, help="for the sight check's live rebuild")
    ap.add_argument("--nuscenes", type=Path, help="default: gallops.toml's {data.nuscenes}")
    ap.add_argument("--epoch", type=int, required=True, help="the epoch every run is measured at")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    runs = [r.resolve() for r in args.runs]
    for run in runs:
        if not (run / f"epoch_{args.epoch:02d}.pt").exists():
            raise SystemExit(f"{run}: no epoch_{args.epoch:02d}.pt")
        if (run / f"summary_epoch{args.epoch:02d}.json").exists():
            raise SystemExit(f"{run}: already measured at epoch {args.epoch}")

    from gallops.drivejepa_aux import step

    cfg = SiteConfig.load()
    cache_dir, extract_dir = args.cache.resolve(), args.extract.resolve()
    root = (args.nuscenes or Path(cfg.resolve("{data.nuscenes}"))).resolve()
    release = step.enter(cfg)
    import torch

    from gallops.drivejepa_aux import inputs

    data = cache.load(cache_dir)
    extract = cache.load_extract(extract_dir)
    inputs.use_calibration(extract_dir / "calib")
    built = data["inputs"]["built"]
    train_rows, held_rows = split.partition(data["labels"]["scene"], split.load()["held_out"])
    train_rows, held_rows = train_rows[built[train_rows]], held_rows[built[held_rows]]
    spatial = torch.as_tensor(data["inputs"]["spatial"], device=args.device)
    level = torch.as_tensor(data["inputs"]["level"], device=args.device)
    # One model holds every arm's parts: upstream's heads switched on, and the scene heads beside it.
    # A run's state replaces the refiner path and the heads it trained; the planning outputs never
    # read the heads (eval mode), and an arm's heads are measured only if it trained them.
    with contextlib.redirect_stdout(io.StringIO()):
        _, model, scene_heads, _ = step.build_model(release, args.device, aux=True, scene=True)
    release_state = step.trainable_state(model, scene_heads)
    step.load_trainable(model, None, release_state)
    released = train.measure(model, None, data, held_rows, spatial, level, args.device, with_heads=False)
    sight_rows = held_rows[np.linspace(0, len(held_rows) - 1, train.SIGHT_ROWS).round().astype(int)]
    base = train.baselines(data, train_rows, held_rows)

    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        arm = summary["arm"]
        state = torch.load(run / f"epoch_{args.epoch:02d}.pt", map_location="cpu")
        step.load_trainable(model, None, release_state)  # nothing left over from the previous run
        step.load_trainable(model, scene_heads if arm == "T2" else None, state)
        heads = scene_heads if arm == "T2" else None
        measured = train.measure(
            model, heads, data, held_rows, spatial, level, args.device, with_heads=arm != "C"
        )
        fixed = {
            **summary,
            "kept_epoch": args.epoch,
            "kept_is": f"the fixed epoch {args.epoch}, chosen by nothing; summary.json holds the kept epoch",
            "release": {**train.summarize(released, data), "sight": summary["release"]["sight"]},
            "kept": {
                **train.summarize(measured, data),
                "sight": train.sight(model, extract, data, sight_rows, root, args.device),
            },
            "baselines": base,
        }
        np.savez(
            run / f"heldout_epoch{args.epoch:02d}.npz",
            **{f"release_{k}": v for k, v in released.items()},
            **{f"kept_{k}": v for k, v in measured.items()},
        )
        text = json.dumps(train.plain(fixed), indent=1, allow_nan=False) + "\n"
        (run / f"summary_epoch{args.epoch:02d}.json").write_text(text)
        l2 = fixed["kept"]["l2_3s"]["mean"]
        print(
            f"{run.name} ({arm}, seed {summary['seed']}) at epoch {args.epoch}: L2 @3 s {l2:.4f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
