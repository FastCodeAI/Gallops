"""Stage A: teach Drive-JEPA's refiner to propose a sideways plan at all.

Its scorer refuses a swerve when handed one, and its refiner never offers one --
the widest of its 32 candidates reaches 0.70 m on a frontal tick. Both halves need
training, and the two are taken one at a time so each ends on a measurement:

    stage A  train the refiner, scorer frozen   -> do the 32 now reach past 0.70 m?
    stage B  freeze the refiner, train the scorer -> does it rank clear above hitting?

This is stage A. The loss is the vendor's own shape (`trajectory_loss_anchors` in
`drive_jepa_agent.py`): the L1 distance from the target to the CLOSEST of the 32
proposals. Winner-takes-all is the whole reason this can work -- it pulls one
proposal toward the expert's evasion without collapsing the other 31 onto it, so
the model keeps proposing the ordinary plans it already gets right.

**Almost nothing moves.** The V-JEPA encoder is 97.7% of the parameters and the
earlier probe showed obstacles on the ego's path are already legible in what it
outputs, so it is frozen -- which also means it cannot quietly learn what a
NeuRAD-inserted car looks like, the shortcut this whole setup most invites.

**What this first run is and is not.** It trains on round-0 data only: 150
episodes in which the released model drove and never swerved. No navtrain replay
yet, so the model WILL drift on ordinary driving, and that is expected rather than
hidden -- the point of this run is to answer one question cheaply, whether the
refiner can be moved to propose an evasion at all. If it can, navtrain mixing and
stage B follow. If it cannot, nothing further is worth building.

    <drive-jepa python> -m gallops.train.swerve --labelled <dir> --epochs 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from gallops import checkpoint as sidecar
from gallops.config import SiteConfig

#: Only these move. Everything else, the encoder above all, is frozen.
TRAINABLE = ("_trajectory_head", "init_feature", "hist_encoding")


def git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_ticks(labelled: Path, blind: bool = False) -> list[dict]:
    """Every labelled tick that has a frame to read and an expert plan to copy.

    Reactive targets whose hazard sits outside the one camera's frame are dropped.
    In the side scenarios the crossing car is at a median 43.5 deg off axis and the
    half-FOV is 32.6 deg, so about 390 of the round-0 ticks would otherwise teach
    an evasion of something invisible -- a mapping from "empty road" to "swerve
    left", which is worse than no training at all. `label.py` records the verdict
    per tick; this only honours it. `--keep-blind` restores them for a comparison.
    """
    ticks, dropped = [], 0
    for path in sorted(labelled.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if not (record.get("frame") and record.get("expert")):
                continue
            # Older labels predate the visibility field; absent means unfiltered.
            if not blind and record.get("usable_for_imitation", True) is False:
                dropped += 1
                continue
            ticks.append(record)
    if dropped:
        print(f"  dropped {dropped} ticks that react to a hazard the camera cannot see")
    return ticks


def split_by_scene(ticks: list[dict], held_out: str) -> tuple[list[dict], list[dict]]:
    """Hold out a whole SCENE, not random ticks.

    Ticks inside an episode are 0.1 s apart and almost identical, so a random
    split would put near-duplicates on both sides and report a validation loss
    that means nothing.
    """
    train = [t for t in ticks if held_out not in t["episode"]]
    val = [t for t in ticks if held_out in t["episode"]]
    return train, val


def build_model(checkpoint_path: str, device: str, repo: str | None = None):
    """Load the released model and freeze everything but the refiner path.

    `repo` is the working directory the agent needs: it opens
    "./vjepa2/configs/eval/vitl/in1k.yaml" and np.load("./data/8192.npy")
    relative to cwd, so paddock's node runs from navsim_v2 and so must this.
    """
    import os

    import torch

    if repo:
        os.chdir(repo)
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=checkpoint_path)
    agent.initialize()
    model = agent._pad_model.to(device)

    frozen = trainable = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(part in name for part in TRAINABLE)
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    total = frozen + trainable
    print(
        f"  {trainable / 1e6:.2f} M trainable of {total / 1e6:.1f} M "
        f"({trainable / total:.1%}); the encoder stands still"
    )
    return agent, model, torch


def min_over_proposals(proposals, target, torch):
    """The vendor's winner-takes-all: distance to the CLOSEST of the 32.

    Anything that averages over the 32 instead would drag every proposal toward
    the same plan and destroy the diversity the scorer needs to choose from.
    """
    return torch.linalg.norm(proposals - target[:, None], dim=-1, ord=1).mean(-1).amin(1).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--held-out", default="0058", help="a whole scene kept back")
    parser.add_argument("--train-run", default="swerve-stageA-round0")
    parser.add_argument("--limit", type=int, default=0, help="a smoke run over N ticks")
    parser.add_argument(
        "--keep-blind",
        action="store_true",
        help="keep reactive ticks whose hazard is off-camera (for comparison only)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = SiteConfig.load()
    # The agent reads these at construction, exactly as paddock's model.yaml
    # supplies them to its node. NAVSIM_EXP_ROOT in particular is where it looks
    # for the V-JEPA weights the encoder is built on.
    for name, template in (
        ("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"),
        ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
        ("OPENSCENE_DATA_ROOT", "{data.navsim}"),
        ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps"),
    ):
        os.environ.setdefault(name, cfg.resolve(template))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")

    ticks = load_ticks(args.labelled, blind=args.keep_blind)
    if args.limit:
        ticks = ticks[: args.limit]
    train, val = split_by_scene(ticks, args.held_out)
    print(f"  {len(ticks)} labelled ticks: {len(train)} train, {len(val)} held out (scene {args.held_out})")
    if not train:
        raise SystemExit("no training ticks")

    from gallops.train.swerve_data import TickDataset  # local import: needs torch

    checkpoint_path = cfg.resolve(
        "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
    )
    agent, model, torch = build_model(checkpoint_path, args.device, cfg.resolve("{data.drivejepa_repo}"))
    dataset = TickDataset(train, args.device)
    validation = TickDataset(val, args.device) if val else None

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(parameters, lr=args.lr)

    out_root = Path(cfg.resolve("{weights}")) / "drivejepa_swerve" / args.train_run
    history = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(dataset)))
        random.Random(epoch).shuffle(order)
        losses = []
        started = time.time()
        for start in range(0, len(order), args.batch):
            chunk = order[start : start + args.batch]
            features, target = dataset.batch(chunk)
            if features is None:
                continue
            output = model(features)
            loss = min_over_proposals(output["proposals"], target, torch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 25 == 0:
                print(f"    epoch {epoch} step {step}: loss {np.mean(losses[-25:]):.4f}", flush=True)

        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "minutes": (time.time() - started) / 60,
        }
        if validation is not None:
            model.eval()
            with torch.no_grad():
                held = []
                for start in range(0, len(validation), args.batch):
                    stop = min(start + args.batch, len(validation))
                    features, target = validation.batch(list(range(start, stop)))
                    if features is None:
                        continue
                    held.append(float(min_over_proposals(model(features)["proposals"], target, torch)))
            entry["val_loss"] = float(np.mean(held)) if held else None
        history.append(entry)
        print(
            f"  epoch {epoch}: train {entry['train_loss']:.4f} "
            f"val {entry.get('val_loss')} in {entry['minutes']:.1f} min",
            flush=True,
        )

        directory = out_root / f"step_{step:09d}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v for k, v in model.state_dict().items()}, directory / "model.pt")
        sidecar.write(
            directory,
            sidecar.Checkpoint(
                model="drivejepa_swerve",
                step=step,
                train_git_sha=git_sha(Path(__file__).resolve().parents[3]),
                train_run=args.train_run,
                trained_on=(
                    sidecar.TrainedOn(
                        dataset="neuroncap-actor-insertion",
                        split="round0, 5 non-benchmark nuScenes train scenes",
                    ),
                ),
                selection_split=f"held-out scene {args.held_out}",
                params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                precision="fp32",
                notes={
                    "stage": "A -- refiner only, scorer and encoder frozen",
                    "loss": "min over 32 proposals of L1 to the expert trajectory",
                    "history": history,
                    "caveat": "round-0 data only, no navtrain replay: drift on ordinary "
                    "driving is expected and unmeasured here",
                    "off_camera_targets": "kept" if args.keep_blind else "dropped",
                },
            ),
        )
    print(f"  wrote checkpoints under {out_root}")


if __name__ == "__main__":
    main()
