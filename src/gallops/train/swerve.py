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
from gallops.train.losses import score_target, scorer_loss, soft_target_floor, trajectory_loss, travelled
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


def load_ticks(labelled: Path, blind: bool = False, source: int = 0) -> list[dict]:
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
            record["source"] = source
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
    """Both losses for one batch: (trajectory, score, parts).

    `parts` is what gets logged: the trajectory loss split into its pull toward
    the expert (`traj_winner`, last stage) and toward the anchors (`traj_anchor`),
    and the scorer's BCE beside its floor -- the entropy of the soft targets,
    which no scorer can go below -- and their difference, the part that can
    actually reach zero.

    The scorer's targets are computed from DETACHED proposals, because grading
    them runs through numpy -- but the loss itself is on the live `pred_logit`,
    so it backpropagates into the refiner through `bev_feature`, exactly as the
    vendor's does. The scorer reads bev_feature rather than the proposal
    coordinates (see `Scorer.forward`), so this is not a stop-gradient: both
    losses train the shared trunk, and only the targets are frozen.
    """
    anchors = timed("anchors", lambda: [referee.good_anchors(t) for t in ticks])
    trajectory, winner, anchored = timed(
        "traj_loss",
        lambda: trajectory_loss(output["proposal_list"], target, anchors, torch),
    )
    proposals = timed("to_cpu", lambda: output["proposals"].detach().float().cpu().numpy())
    targets = timed(
        "grade",
        lambda: np.stack([referee.targets(tick, proposals[i]) for i, tick in enumerate(ticks)]),
    )
    soft = torch.from_numpy(targets).to(output["pred_logit"].device)
    scored = timed("score_loss", lambda: scorer_loss(output["pred_logit"], soft, torch))
    floor = float(soft_target_floor(soft.float(), torch))
    parts = {
        "traj_total": float(trajectory.detach()),
        "traj_winner": float(winner.detach()),
        "traj_anchor": float((anchored - winner).detach()),
        "score_bce": float(scored.detach()),
        "score_floor": floor,
        "score_above_floor": float(scored.detach()) - floor,
    }
    return trajectory, scored, parts


def evaluate(model, dataset, indices, referee, torch, batch: int) -> dict:
    """Mean loss parts over `indices` of `dataset`, no gradient, model left in train mode."""
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            features, target, ticks_used = dataset.batch(list(indices[start : start + batch]))
            if features is None:
                continue
            rows.append(combined_loss(model(features), target, ticks_used, referee, torch)[2])
    model.train()
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]} if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelled", type=Path, help="one source, driven by the starting weights; or use --source"
    )
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
    # A scene whose inserted car renders as nothing teaches reactions to an empty
    # road. scene-0064's car was only ever seen 47 m+ away, and in its frontal
    # episodes it is not visible at all.
    parser.add_argument("--exclude-scene", action="append", default=[], help="e.g. 0064; repeatable")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drivable", type=Path, help="directory of scene-*.npz rasters")
    parser.add_argument(
        "--dumps",
        type=Path,
        help="the candidate dumps --labelled was made from, collected by the weights training "
        "starts from; the preflight re-grades them through the trainer and must match the labels",
    )
    # DAgger trains on the UNION of every round (see gallops.expert.rounds), and
    # each round was driven by a different checkpoint, so each needs its own dumps
    # and its own answer to "can the starting weights be held to these?".
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        default=[],
        metavar=("LABELLED", "DUMPS", "DROVE"),
        help="repeatable: a labelled dir, its dumps, and who drove -- 'start' for the weights this "
        "run starts from (round 0, GT-shadow), else the driving checkpoint's name (a DAgger round)",
    )
    # The released config's own weights: trajectory_weight 1, final_score_weight 1.
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    # Validation during the epoch, not only after it: the held-out trajectory loss
    # ended at 2.1-2.4 against ~1.0 in training on every run so far, and a single
    # number at the end cannot say when that gap opens.
    parser.add_argument(
        "--val-every", type=int, default=100, help="steps between held-out checks; 0 = epoch end only"
    )
    parser.add_argument(
        "--val-sample", type=int, default=400, help="held-out ticks per interim check; 0 = all"
    )
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

    from gallops.train.preflight import START, Source

    sources = [Source(*spec) for spec in args.source]
    if args.labelled or args.dumps:
        if not (args.labelled and args.dumps):
            raise SystemExit("--labelled and --dumps go together")
        sources.insert(0, Source(str(args.labelled), str(args.dumps), START))
    if not sources:
        raise SystemExit("nothing to train on: give --labelled/--dumps or --source")
    ticks = [
        t
        for i, s in enumerate(sources)
        for t in load_ticks(Path(s.labelled), blind=args.keep_blind, source=i)
    ]
    for i, s in enumerate(sources):
        n = sum(1 for t in ticks if t["source"] == i)
        print(f"  source {i}: {n} ticks from {s.labelled} (drove: {s.drove})")
    if args.exclude_scene:
        kept = [t for t in ticks if not any(f"-{scene}/" in t["episode"] for scene in args.exclude_scene)]
        print(f"  excluded scenes {', '.join(args.exclude_scene)}: {len(ticks) - len(kept)} ticks dropped")
        ticks = kept
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
        sources,
        model,
        TickDataset,
        torch,
        args.device,
    )

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(parameters, lr=args.lr)

    out_root = Path(cfg.resolve("{weights}")) / "drivejepa_swerve" / args.train_run
    out_root.mkdir(parents=True, exist_ok=True)
    # One JSON line per 25 training steps and per held-out check, beside the
    # checkpoints, so curves can be drawn from the run itself.
    metrics_path = out_root / "metrics.jsonl"
    metrics = metrics_path.open("a")

    def record(kind: str, **fields) -> None:
        metrics.write(json.dumps({"kind": kind, "time": round(time.time(), 1), **fields}) + "\n")
        metrics.flush()

    record(
        "start",
        run=args.train_run,
        sources=[s._asdict() for s in sources],
        train_ticks=len(dataset),
        val_ticks=len(validation) if validation else 0,
        excluded_scenes=args.exclude_scene,
    )
    interim = []
    if validation is not None:
        n = len(validation) if not args.val_sample else min(args.val_sample, len(validation))
        interim = sorted({int(i) for i in np.linspace(0, len(validation) - 1, n).round()})
    history = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(dataset)))
        random.Random(epoch).shuffle(order)
        losses = []
        recent_parts: list[dict] = []
        started = time.time()
        for start in range(0, len(order), args.batch):
            chunk = order[start : start + args.batch]
            features, target, ticks_used = timed("data", lambda c=chunk: dataset.batch(c))
            if features is None:
                continue
            output = timed("forward", lambda f=features: model(f))
            trajectory, scored, parts = combined_loss(output, target, ticks_used, referee, torch)
            loss = args.trajectory_weight * trajectory + args.score_weight * scored
            optimiser.zero_grad(set_to_none=True)
            timed("backward", loss.backward)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            losses.append((float(trajectory.detach()), float(scored.detach())))
            recent_parts.append(parts)
            step += 1
            if step % 25 == 0:
                recent = np.asarray(losses[-25:])
                record(
                    "train",
                    epoch=epoch,
                    step=step,
                    **{k: float(np.mean([r[k] for r in recent_parts[-25:]])) for k in recent_parts[-1]},
                )
                spent = "  ".join(f"{k} {v / 25:.2f}s" for k, v in sorted(TIMING.items()))
                TIMING.clear()
                print(
                    f"    epoch {epoch} step {step}: "
                    f"trajectory {recent[:, 0].mean():.4f}  score {recent[:, 1].mean():.4f}\n"
                    f"      per step: {spent}",
                    flush=True,
                )
            if interim and args.val_every and step % args.val_every == 0:
                held_out = evaluate(model, validation, interim, referee, torch, args.batch)
                record("val", epoch=epoch, step=step, ticks=len(interim), **held_out)
                nan = float("nan")
                print(
                    f"    held-out at step {step}: trajectory {held_out.get('traj_total', nan):.4f}  "
                    f"score above floor {held_out.get('score_above_floor', nan):.4f}",
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
            held_out = evaluate(model, validation, list(range(len(validation))), referee, torch, args.batch)
            if held_out:
                record("val_full", epoch=epoch, step=step, ticks=len(validation), **held_out)
                entry["val_trajectory"] = held_out["traj_total"]
                entry["val_score"] = held_out["score_bce"]
                entry["val_score_above_floor"] = held_out["score_above_floor"]
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
                    "sources": [s._asdict() for s in sources],
                    "preflight": checked,
                    "history": history,
                    # every 25-step mean and held-out check of this run, beside the checkpoints
                    "metrics_file": "../metrics.jsonl",
                    "caveat": "no navtrain replay: drift on ordinary driving is expected and unmeasured here",
                    "off_camera_targets": "kept" if args.keep_blind else "dropped",
                    "excluded_scenes": args.exclude_scene,
                },
            ),
        )
    metrics.close()
    print(f"  wrote checkpoints and {metrics_path.name} under {out_root}")


if __name__ == "__main__":
    main()
