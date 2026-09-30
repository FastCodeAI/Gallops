"""The on-policy loop: collect with the current model, label, train, repeat.

Collecting once and training once is DAgger's first iteration and nothing more.
For most behaviours that is a reasonable approximation. It is not here, because a
swerve moves the state distribution a long way: braking keeps the car on the
logged path, while swerving puts it somewhere it has never been -- mid-evasion,
off-path, half-committed. A model trained only on states its non-swerving self
reached has no idea what to do once it starts swerving, and the failure modes are
oscillation and driving off the road on the recovery.

The loop costs no new machinery, because of one decision made earlier: **the
expert labels offline**. So the model is the only thing that ever drives, and a
round is just another collection pass with newer weights:

    round 0   the released checkpoint drives     -> D0 -> train on D0            -> M1
    round 1   M1 drives the same scenarios       -> D1 -> train on D0 + D1       -> M2
    round 2   M2 drives them                     -> D2 -> train on D0 + D1 + D2  -> M3

Two details that are easy to get wrong.

**Aggregate, never replace.** Round n trains on the union of every round so far.
Training on the newest round alone is where DAgger implementations oscillate: the
model forgets the states that made it stop swerving into things last round.

**Round 0 contains no swerve states at all**, because the released model never
swerves. That is not a defect to fix; it is what iteration 1 always looks like.

This module owns the bookkeeping -- which rounds exist, which checkpoint drove
each, and what to hand a trainer -- not the driving or the training, which are
`ops/dj_collect/collect.py` and gallops' trainer respectively.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gallops.config import SiteConfig


@dataclass(frozen=True)
class Round:
    """One pass of: this checkpoint drove, and these labelled ticks came out."""

    index: int
    checkpoint: str
    collected: str
    labelled: str
    episodes: int
    ticks: int
    note: str = ""


def manifest_path(root: Path) -> Path:
    return root / "rounds.json"


def load(root: Path) -> list[Round]:
    path = manifest_path(root)
    if not path.exists():
        return []
    return [Round(**entry) for entry in json.loads(path.read_text())]


def record(root: Path, entry: Round) -> list[Round]:
    """Add or replace a round, keeping the manifest ordered and on disk."""
    rounds = [r for r in load(root) if r.index != entry.index]
    rounds.append(entry)
    rounds.sort(key=lambda r: r.index)
    manifest_path(root).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(root).write_text(json.dumps([asdict(r) for r in rounds], indent=2))
    return rounds


def training_set(root: Path, through: int | None = None) -> list[Path]:
    """Every labelled file from round 0 through `through`, which is the point.

    Returning only the newest round would be the mistake this module exists to
    prevent.
    """
    files: list[Path] = []
    for entry in load(root):
        if through is not None and entry.index > through:
            continue
        directory = Path(entry.labelled)
        if not directory.exists():
            raise SystemExit(f"round {entry.index}: {directory} is gone; the manifest is stale")
        files.extend(sorted(directory.glob("*.jsonl")))
    return files


def next_index(root: Path) -> int:
    rounds = load(root)
    return 0 if not rounds else rounds[-1].index + 1


def describe(root: Path) -> None:
    rounds = load(root)
    if not rounds:
        print(f"  no rounds yet under {root}")
        return
    print(f"  {'round':>5}  {'episodes':>8} {'ticks':>7}  checkpoint")
    total = 0
    for entry in rounds:
        total += entry.ticks
        name = Path(entry.checkpoint).name or entry.checkpoint
        print(f"  {entry.index:5}  {entry.episodes:8} {entry.ticks:7}  {name}")
        if entry.note:
            print(f"         {entry.note}")
    print(f"  {'':5}  {'':8} {total:7}  ticks in the aggregated training set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="what rounds exist")
    show.add_argument("--root", type=Path)

    add = sub.add_parser("add", help="record a completed round")
    add.add_argument("--root", type=Path)
    add.add_argument("--index", type=int)
    add.add_argument("--checkpoint", required=True)
    add.add_argument("--collected", type=Path, required=True)
    add.add_argument("--labelled", type=Path, required=True)
    add.add_argument("--note", default="")

    files = sub.add_parser("files", help="the aggregated training set, one path per line")
    files.add_argument("--root", type=Path)
    files.add_argument("--through", type=int)

    args = parser.parse_args()
    root = args.root or Path(SiteConfig.load().resolve("{runs}")) / "dagger"

    if args.command == "show":
        describe(root)
        return
    if args.command == "files":
        for path in training_set(root, args.through):
            print(path)
        return

    labelled = sorted(args.labelled.glob("*.jsonl"))
    ticks = sum(sum(1 for _ in path.open()) for path in labelled)
    entry = Round(
        index=args.index if args.index is not None else next_index(root),
        checkpoint=str(args.checkpoint),
        collected=str(args.collected),
        labelled=str(args.labelled),
        episodes=len(labelled),
        ticks=ticks,
        note=args.note,
    )
    record(root, entry)
    print(f"  recorded round {entry.index}: {entry.episodes} episodes, {entry.ticks} ticks")
    describe(root)


if __name__ == "__main__":
    main()
