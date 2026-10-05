"""Before the first training step: is the code right, and is this data fit to train on?

The trainer runs this after it has built the model and before it builds an
optimiser. Any failure stops the run with every reason listed, and a run that
passes carries the result in its checkpoint's sidecar.

Why it exists. v1-v4 trained the scorer on targets graded from the wrong place:
`Referee` cached `episode_context` per episode, pose included, so every tick was
graded as if the ego were where the first-looked-up tick had it -- 14-16 m off.
That flipped the hit/clear verdict on 19.5% of stationary and 36.6% of frontal
candidates. Every check that existed passed, because each tested a part that was
right: the grader on its own, and the labeller, which grades at the right pose.
Nothing compared what the TRAINER computes with what the LABELLER recorded. That
comparison is the heart of this module, and it fails loudly on that bug.

Three groups, cheapest first:

    code       the tree is committed, so `train_git_sha` names the code that
               ran, and the test suite passes under the interpreter that trains
    grading    on a sample of ticks, the trainer's own grading path -- the
               Referee -- reproduces what labelling recorded, EXACTLY: the
               expert's verdict, the driven plan's hit time, and all 32
               candidate verdicts. Measured 132 of 132 ticks on round 0, so
               anything less is a disagreement, not noise. Plus the scorer's
               targets: finite, in [0, 1], and higher for clear plans than
               for ones that crash.
    fidelity   the weights training starts from, fed the trainer's rebuilt
               inputs, reproduce the plan they actually drove at collection.
               Measured over 506 round-0 ticks with their history on disk:
               the pick sits a median 0.29% of four seconds' travel from the
               driven one (p90 4.9%, where a hair of input flips the argmax
               between two close candidates). This is only meaningful when
               `--dumps` were collected by those same weights.

               It was an absolute 5 cm at first, set from one scene's late
               ticks, and it refused its first real run: the error scales with
               speed, and an episode's first half second has no real past at
               all (see `HISTORY_TOLERANCE_S` in swerve_data.py). Both were
               measured before the limit moved, and the second was a defect the
               trainer now skips rather than a reason to loosen anything.

The sample is drawn across episodes and visited LATE TICK FIRST, so anything
that caches per episode is caught by the first episode it touches.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Pick verdicts may differ on this fraction of sampled ticks before fidelity
#: fails. The measured baseline is zero; this leaves room for a tick whose plan
#: grazes the target by millimetres, where the rebuilt frame's JPEG round trip
#: (the dump saw the pixels the renderer sent) can flip a verdict.
FIDELITY_VERDICT_TOLERANCE = 0.05
#: Median gap between the rebuilt pick and the driven one, as a fraction of four
#: seconds' travel at the tick's speed. Measured 0.29%; a gap this size is three
#: times what the JPEG round trip explains.
FIDELITY_MEDIAN_PICK_FRACTION = 0.01
#: The dump adapter writes `timestamp` first on every line.
LEADING_TIMESTAMP = re.compile(r'\{"timestamp": (\d+)')


class PreflightFailed(RuntimeError):
    def __init__(self, problems: list[str]):
        super().__init__("training preflight failed:\n  - " + "\n  - ".join(problems))
        self.problems = problems


def code_checks(repo: Path) -> tuple[dict, list[str]]:
    """Committed tree, and the test suite passing under THIS interpreter."""
    problems, report = [], {}
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tests"], cwd=repo, capture_output=True, text=True
    )
    if status.returncode != 0:
        problems.append(f"not a git checkout ({repo}): the sidecar could not name the code that trained")
    elif status.stdout.strip():
        changed = [line[3:] for line in status.stdout.splitlines()]
        problems.append(
            f"uncommitted changes in {', '.join(changed[:5])}{' ...' if len(changed) > 5 else ''}: "
            "commit first, so train_git_sha names the code that actually ran"
        )
    report["git_clean"] = status.returncode == 0 and not status.stdout.strip()

    env = dict(os.environ, PYTHONPATH=str(repo / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    summary = (tests.stdout.strip().splitlines() or ["(no output)"])[-1]
    report["tests"] = summary
    if tests.returncode != 0:
        problems.append(f"the test suite fails under {sys.executable}: {summary}")
    return report, problems


def sample_ticks(ticks: list[dict], episodes: int = 12, per_episode: int = 4, seed: int = 0) -> list[dict]:
    """Ticks spread across episodes and across each episode, LATE FIRST.

    Each episode contributes ticks at fixed fractions of its length, visited from
    the end backwards: a cache keyed by episode is then filled by a late tick
    and read back for an early one, which is exactly the shape of the v1-v4 bug.
    """
    by_episode: dict[str, list[dict]] = {}
    for tick in ticks:
        by_episode.setdefault(tick["episode"], []).append(tick)
    names = sorted(by_episode)
    random.Random(seed).shuffle(names)
    chosen = []
    for name in names[:episodes]:
        rows = sorted(by_episode[name], key=lambda t: int(t["timestamp"]))
        picks = sorted(
            {round(f * (len(rows) - 1)) for f in np.linspace(0.1, 0.95, per_episode)}, reverse=True
        )
        chosen.extend(rows[i] for i in picks)
    return chosen


def _dumped(tick: dict, dumps: Path, cache: dict) -> dict | None:
    from gallops.expert.label import match_dump

    directory = Path(tick["frame"]).parent.parent
    key = str(directory)
    if key not in cache:
        # Only the files that hold this tick's timestamp can be its episode's dump.
        # Without the shortlist every episode parses every dump: 216 MB each on round 0.
        candidates = _dump_index(str(dumps)).get(int(tick["timestamp"]), [])
        rows = match_dump(directory, dumps, candidates) if candidates else None
        cache[key] = {r["timestamp"]: r for r in rows} if rows else None
    rows = cache[key]
    return None if rows is None else rows.get(int(tick["timestamp"]))


@lru_cache(maxsize=8)
def _dump_index(dumps: str) -> dict[int, list[Path]]:
    """timestamp -> the dump files holding it, from each line's leading field alone."""
    index: dict[int, list[Path]] = {}
    for path in Path(dumps).glob("*.jsonl"):
        with path.open() as handle:
            for line in handle:
                found = LEADING_TIMESTAMP.match(line)
                if found:
                    index.setdefault(int(found.group(1)), []).append(path)
    return index


def grading_checks(sample: list[dict], referee, dumps: Path) -> tuple[dict, list[str]]:
    """The trainer's grading path against what labelling recorded, exactly."""
    from gallops.expert.grader import first_hit, grade

    problems, cache = [], {}
    agree = Counter()
    clear_scores, hit_scores = [], []
    missing = 0
    for tick in sample:
        row = _dumped(tick, dumps, cache)
        if row is None:
            missing += 1
            continue
        ego2world, targets, area = referee.context(tick)
        reference = referee.reference(tick)
        stamp = int(tick["timestamp"])
        agree["expert"] += grade(
            np.asarray(tick["expert"]), ego2world, stamp, targets, reference, area
        ).clear == bool(tick["expert_clear"])
        agree["driven"] += (
            first_hit(np.asarray(row["trajectory"]), ego2world, stamp, targets) == tick["driven_hits"]
        )
        proposals = np.asarray(row["proposals"])
        verdicts = [grade(p, ego2world, stamp, targets, reference, area).clear for p in proposals]
        agree["candidates"] += verdicts == list(tick["candidate_clear"])
        scores = referee.targets(tick, proposals)
        if not np.all(np.isfinite(scores)) or scores.min() < 0 or scores.max() > 1:
            problems.append(f"{tick['episode']} @ {stamp}: scorer targets outside [0, 1] or not finite")
        # by index: this runs in the training interpreter, 3.9, where zip has no strict=
        for i, score in enumerate(scores):
            (clear_scores if tick["candidate_hits"][i] is None else hit_scores).append(float(score))
        agree["n"] += 1

    n = agree["n"]
    report = {"sampled_ticks": len(sample), "graded": n, "no_dump": missing}
    if missing:
        problems.append(f"{missing} of {len(sample)} sampled ticks have no dump under {dumps}")
    if n == 0:
        problems.append("no sampled tick could be graded; nothing about the targets was checked")
        return report, problems
    for name in ("expert", "driven", "candidates"):
        report[f"{name}_reproduced"] = f"{agree[name]}/{n}"
        if agree[name] != n:
            problems.append(
                f"the trainer's grading disagrees with labelling on {n - agree[name]} of {n} ticks ({name}): "
                "the targets it would train on are not the ones that were labelled"
            )
    if clear_scores and hit_scores:
        gap = float(np.mean(clear_scores) - np.mean(hit_scores))
        report["target_gap_clear_minus_hit"] = round(gap, 3)
        if gap <= 0:
            problems.append(
                f"scorer targets rate crashing plans at least as high as clear ones (gap {gap:+.3f})"
            )
    return report, problems


def fidelity_check(
    model, dataset_cls, sample: list[dict], referee, dumps: Path, torch, device: str
) -> tuple[dict, list[str]]:
    """The starting weights, on rebuilt inputs, against the plan they drove."""
    from gallops.expert.grader import first_hit

    problems, cache = [], {}
    rows = [(t, _dumped(t, dumps, cache)) for t in sample]
    rows = [(t, r) for t, r in rows if r is not None]
    if not rows:
        return {}, ["fidelity: no sampled tick has a dump to compare with"]
    ticks = [t for t, _ in rows]
    driven = {(t["episode"], int(t["timestamp"])): r for t, r in rows}
    dataset = dataset_cls(ticks, device)
    diffs, fractions, same = [], [], 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ticks), 8):
            features, _, used = dataset.batch(list(range(start, min(start + 8, len(ticks)))))
            if features is None:
                continue
            picked = model(features)["trajectory"].float().cpu().numpy()
            # `used`, not the indices asked for: batch skips a tick it cannot rebuild.
            for k, tick in enumerate(used):
                plan = picked[k]
                row = driven[(tick["episode"], int(tick["timestamp"]))]
                ego2world, targets, _ = referee.context(tick)
                hit = first_hit(plan, ego2world, int(tick["timestamp"]), targets)
                same += (hit is None) == (tick["driven_hits"] is None)
                gap = float(np.abs(plan[:, :2] - np.asarray(row["trajectory"])[:, :2]).max())
                diffs.append(gap)
                # as a share of four seconds' travel: the gap scales with speed
                fractions.append(gap / max(float(tick["speed"]) * 4.0, 1.0))
    model.train(was_training)
    if not diffs:
        return {}, ["fidelity: the dataset rebuilt none of the sampled ticks"]
    agreement = same / len(diffs)
    median = float(np.median(fractions))
    report = {
        "compared": len(diffs),
        "skipped_unrebuildable": len(ticks) - len(diffs),
        "pick_verdict_agrees": f"{same}/{len(diffs)}",
        "median_pick_diff_m": round(float(np.median(diffs)), 4),
        "median_pick_diff_of_travel": round(median, 4),
    }
    if agreement < 1 - FIDELITY_VERDICT_TOLERANCE:
        problems.append(
            f"fidelity: the starting weights' pick hits/clears differently from the driven plan on "
            f"{len(diffs) - same} of {len(diffs)} ticks -- the rebuilt inputs are not what the model saw "
            "(or --dumps were not collected by these weights)"
        )
    if median > FIDELITY_MEDIAN_PICK_FRACTION:
        problems.append(
            f"fidelity: rebuilt pick is {median:.2%} of four seconds' travel from the driven one "
            f"(median), limit {FIDELITY_MEDIAN_PICK_FRACTION:.0%}"
        )
    return report, problems


def describe(ticks: list[dict]) -> dict:
    """What is about to be trained on, for the log and the sidecar."""
    shapes = Counter(t["episode"].split("-", 1)[0] for t in ticks)
    kinds = Counter(t["expert_kind"] for t in ticks)
    creep = sum(1 for t in ticks if t["speed"] < 3.0 and t["driven_hits"] is not None)
    return {
        "ticks": len(ticks),
        "episodes": len({t["episode"] for t in ticks}),
        "by_shape": dict(shapes),
        "by_expert_kind": dict(kinds),
        "creep_ticks": creep,
    }


def run(
    repo: Path,
    train: list[dict],
    referee,
    dumps: Path,
    model,
    dataset_cls,
    torch,
    device: str,
    sample_size: int = 48,
) -> dict:
    """Every check; raises PreflightFailed listing all problems, or returns the report."""
    report = {"data": describe(train)}
    problems = []
    report["code"], found = code_checks(repo)
    problems += found
    sample = sample_ticks(train, episodes=max(1, sample_size // 4), per_episode=4)
    report["grading"], found = grading_checks(sample, referee, dumps)
    problems += found
    report["fidelity"], found = fidelity_check(model, dataset_cls, sample, referee, dumps, torch, device)
    problems += found
    print("  preflight: " + json.dumps(report), flush=True)
    if problems:
        raise PreflightFailed(problems)
    return report
