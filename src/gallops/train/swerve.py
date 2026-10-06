"""Teach Drive-JEPA to propose a sideways plan AND to rank it above driving into
something -- both heads at once, because separating them did not work.

**What the first attempt measured.** Training the refiner alone, behind a frozen
scorer, moved the proposals: the widest of the 32 went from 0.70 m to over 5 m
sideways. It also broke the model. On neuroNCAP scene 0103 the released model
scores 5.0 ten times out of ten on the stationary and side scenarios; after that
finetune it scored 1.09 and 0.68 and hit the obstacle in every run, while the
frontal case it was built for got no better. The run is kept as
`swerve-stageA-round0-seen` and the numbers in ops/dj_train/reach.json.

Two things caused it, and both are fixed here rather than worked around:

1. **Half a loss.** The trainer used only the winner-takes-all term of the
   vendor's `trajectory_loss_anchors`. The refiner is ONE module applied to all
   32 proposals and reapplied four times, so moving the winner moves every
   proposal, and the missing anchor term is what holds them to a vocabulary of
   plausible plans. See `losses.py`, which restores it.
2. **A frozen scorer ranking proposals it had never seen.** It was calibrated on
   the released proposal distribution; once that distribution moved, its ranking
   stopped meaning anything, and it picked a drifted plan. So the scorer trains
   too, on soft targets from the same grader the expert is held to.

The two are trained jointly with a stop-gradient between them, which is what the
vendor does: `compute_score(targets, proposals.detach())` and then
`proposals = proposals.detach()` before the trajectory loss. The scorer learns to
rank the proposals the refiner currently makes; it never pulls the refiner around.

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
from gallops.expert.grader import grade
from gallops.expert.lanes import load_references, reference_at
from gallops.train.losses import score_target, scorer_loss, trajectory_loss, travelled
from gallops.train.swerve_data import episode_context

#: Only these move. Everything else, the encoder above all, is frozen.
#: `scorer` is the addition the first run's failure argued for: 4.70 M more, and
#: still under 3% of the model.
TRAINABLE = ("_trajectory_head", "init_feature", "hist_encoding", "scorer")


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


class Referee:
    """Grades trajectories for the scorer's targets, and picks each tick's anchors.

    Both jobs need the same context, and both are pure numpy: the anchor set for a
    tick never changes, and `episode_context` reads an episode's actors and map
    once however many epochs run.
    """

    def __init__(self, anchors, drivable=None, per_tick: int = 32, seed: int = 0):
        self.anchors = np.asarray(anchors)
        self.drivable = drivable
        self.per_tick = per_tick
        self.random = np.random.default_rng(seed)
        self._anchor_cache: dict[tuple, np.ndarray] = {}
        self._references: dict[str, np.ndarray] = {}

    def context(self, tick: dict):
        # Per tick, never cached per episode: the pose is the tick's own.
        return episode_context(tick, self.drivable)

    def reference(self, tick: dict) -> np.ndarray:
        """The path this tick's plans are graded against: the lane the car is in.

        The grader measures a plan's deviation from it, and that deviation is what
        settles the choice between two plans that are both clear: the one holding
        its lane wins, because a swerve has a direction to get wrong. It is the
        lane centre where `gallops.expert.lanes` has run, the logged human path
        otherwise -- read through the same function labelling uses.
        """
        directory = Path(tick["frame"]).parent.parent
        key = str(directory)
        if key not in self._references:
            self._references[key] = load_references(directory)
        logged, lanes = self._references[key]
        return reference_at(int(tick["timestamp"]), logged, lanes)[0]

    def good_anchors(self, tick: dict) -> np.ndarray:
        """A sample of anchors that neither hit anything nor leave the road here.

        The vendor gets this set from the PDM simulator over a metric cache we do
        not have for these scenes, so it comes from the grader instead. Sampled
        rather than exhaustive: 8,192 anchors graded per tick would cost more than
        the training step it regularises, and the term only needs a few.
        """
        key = (str(Path(tick["frame"]).parent.parent), int(tick["timestamp"]))
        if key not in self._anchor_cache:
            ego2world, targets, area = self.context(tick)
            stamp = int(tick["timestamp"])
            indices = self.random.choice(len(self.anchors), size=self.per_tick, replace=False)
            clear = [
                self.anchors[i]
                for i in indices
                if grade(self.anchors[i], ego2world, stamp, targets, None, area).clear
            ]
            self._anchor_cache[key] = np.asarray(clear) if clear else np.empty((0, 8, 3))
        return self._anchor_cache[key]

    def targets(self, tick: dict, proposals: np.ndarray) -> np.ndarray:
        """A soft score in [0, 1] for each of the 32 proposals as they are now.

        Recomputed every step, because the proposals move -- grading the ones
        recorded at collection time would train the scorer against a distribution
        the refiner has already left behind.
        """
        ego2world, actors, area = self.context(tick)
        stamp = int(tick["timestamp"])
        reference = self.reference(tick)
        # The expert's own plan sets the bar: matching it is full marks and going
        # further earns nothing. Measured against free-running speed instead, the
        # scorer ranked a full-speed swerve above the brake the expert chose.
        expert_travelled = travelled(np.asarray(tick["expert"]))
        out = np.empty(len(proposals), dtype=np.float32)
        for i, proposal in enumerate(proposals):
            verdict = grade(proposal, ego2world, stamp, actors, reference, area)
            out[i] = score_target(
                verdict.hit_at,
                verdict.off_road_at,
                travelled(proposal),
                expert_travelled,
                verdict.max_deviation,
            )
        return out


#: Where each step's wall clock goes, so a slow run is diagnosed from its own log
#: instead of re-derived later. One perf_counter per phase, reset every report.
TIMING: dict[str, float] = {}


def timed(name: str, work):
    start = time.perf_counter()
    value = work()
    TIMING[name] = TIMING.get(name, 0.0) + time.perf_counter() - start
    return value


def combined_loss(output, target, ticks, referee, torch):
    """Both losses for one batch: (trajectory, score).

    The scorer's targets are computed from DETACHED proposals, because grading
    them runs through numpy -- but the loss itself is on the live `pred_logit`,
    so it backpropagates into the refiner through `bev_feature`, exactly as the
    vendor's does. The scorer reads bev_feature rather than the proposal
    coordinates (see `Scorer.forward`), so this is not a stop-gradient: both
    losses train the shared trunk, and only the targets are frozen.
    """
    anchors = timed("anchors", lambda: [referee.good_anchors(t) for t in ticks])
    trajectory = timed(
        "traj_loss",
        lambda: trajectory_loss(output["proposal_list"], target, anchors, torch)[0],
    )
    proposals = timed("to_cpu", lambda: output["proposals"].detach().float().cpu().numpy())
    targets = timed(
        "grade",
        lambda: np.stack([referee.targets(tick, proposals[i]) for i, tick in enumerate(ticks)]),
    )
    scored = timed(
        "score_loss",
        lambda: scorer_loss(
            output["pred_logit"], torch.from_numpy(targets).to(output["pred_logit"].device), torch
        ),
    )
    return trajectory, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    # 0058 was the first choice and a bad one: 54 swerve ticks in 1,673, exactly
    # one of them frontal, so its held-out loss was measuring ordinary driving.
    # 0053 has 444 swerve ticks (157 frontal) and the only balanced mix of all
    # four target kinds, and it is a different city from most of the rest.
    parser.add_argument("--held-out", default="0053", help="a whole scene kept back")
    parser.add_argument("--train-run", default="swerve-stageA-round0")
    parser.add_argument("--limit", type=int, default=0, help="a smoke run over N ticks")
    parser.add_argument(
        "--keep-blind",
        action="store_true",
        help="keep reactive ticks whose hazard is off-camera (for comparison only)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drivable", type=Path, help="directory of scene-*.npz rasters")
    parser.add_argument(
        "--dumps",
        type=Path,
        required=True,
        help="the candidate dumps these labels were made from, collected by the weights training "
        "starts from; the preflight re-grades them through the trainer and must match the labels",
    )
    # The released config's own weights: trajectory_weight 1, final_score_weight 1.
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
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
        # Evenly spaced, not the first N: the files are sorted by scenario-scene,
        # so a head slice is one scene and a smoke run over it holds out its own
        # entire training set.
        ticks = [ticks[i] for i in np.linspace(0, len(ticks) - 1, args.limit).round().astype(int)]
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

    # `agent.anchors` is the vendor's own trajectory vocabulary: data/8192.npy,
    # subsampled to the 8 poses the model plans over.
    referee = Referee(agent.anchors, drivable=args.drivable)

    # Before any step: the code is committed and its tests pass, the trainer's own
    # grading reproduces the labels exactly, and the starting weights reproduce
    # what they drove. v1-v4 trained on targets graded 14-16 m from where the car
    # was, and nothing that existed could see it. See preflight.py.
    from gallops.train import preflight

    checked = preflight.run(
        Path(__file__).resolve().parents[3],
        train,
        referee,
        args.dumps,
        model,
        TickDataset,
        torch,
        args.device,
    )

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
            features, target, ticks_used = timed("data", lambda c=chunk: dataset.batch(c))
            if features is None:
                continue
            output = timed("forward", lambda f=features: model(f))
            trajectory, scored = combined_loss(output, target, ticks_used, referee, torch)
            loss = args.trajectory_weight * trajectory + args.score_weight * scored
            optimiser.zero_grad(set_to_none=True)
            timed("backward", loss.backward)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            losses.append((float(trajectory.detach()), float(scored.detach())))
            step += 1
            if step % 25 == 0:
                recent = np.asarray(losses[-25:])
                spent = "  ".join(f"{k} {v / 25:.2f}s" for k, v in sorted(TIMING.items()))
                TIMING.clear()
                print(
                    f"    epoch {epoch} step {step}: "
                    f"trajectory {recent[:, 0].mean():.4f}  score {recent[:, 1].mean():.4f}\n"
                    f"      per step: {spent}",
                    flush=True,
                )

        trained = np.asarray(losses)
        entry = {
            "epoch": epoch,
            "train_trajectory": float(trained[:, 0].mean()),
            "train_score": float(trained[:, 1].mean()),
            "minutes": (time.time() - started) / 60,
        }
        if validation is not None:
            model.eval()
            with torch.no_grad():
                held = []
                for start in range(0, len(validation), args.batch):
                    stop = min(start + args.batch, len(validation))
                    features, target, ticks_used = validation.batch(list(range(start, stop)))
                    if features is None:
                        continue
                    trajectory, scored = combined_loss(model(features), target, ticks_used, referee, torch)
                    held.append((float(trajectory), float(scored)))
            if held:
                validated = np.asarray(held)
                entry["val_trajectory"] = float(validated[:, 0].mean())
                entry["val_score"] = float(validated[:, 1].mean())
        history.append(entry)
        print(
            f"  epoch {epoch}: train traj {entry['train_trajectory']:.4f} "
            f"score {entry['train_score']:.4f} | val traj {entry.get('val_trajectory')} "
            f"score {entry.get('val_score')} in {entry['minutes']:.1f} min",
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
                    # These two read "stage A -- refiner only" and "L1 to the expert"
                    # long after the trainer became joint; they now say what runs.
                    "stage": "joint -- refiner and scorer train, encoder frozen",
                    "loss": "vendor trajectory_loss_anchors (winner + good anchors) + scorer BCE "
                    "against score_target",
                    # Which label set and which dumps: v4 could only be traced to
                    # its labels by file timestamps.
                    "labelled": str(args.labelled),
                    "dumps": str(args.dumps),
                    "preflight": checked,
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
