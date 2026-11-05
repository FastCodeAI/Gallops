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

    code       the tree is committed, and its sha -- taken here, once -- is the
               `train_git_sha` of every checkpoint the run writes; the test
               suite passes under the interpreter that trains
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

               Gated on the median AND the p90, with a bound on ticks it could
               not rebuild, and the sample gains each episode's earliest tick.
               It was an absolute 5 cm at first, set from one scene's late
               ticks, and it refused its first real run because of that design:
               the error scales with speed, and the run passed once the limit
               became a share of travel. The 506-tick measurement the refusal
               prompted is what found an episode's first half second has no
               real past (see `HISTORY_TOLERANCE_S` in swerve_data.py); the
               median gate itself could not see that, since 1 of its 48
               sampled ticks sat there. Such ticks are now dropped before
               anything is counted.
    sight      the starting weights' plan changes when the front camera is
               blanked. Fidelity compares the trainer with a recording, and a
               recording can carry the same mistake: before paddock #62 (open) both
               placed the cameras relative to nuScenes' LIDAR_TOP (yawed -90
               degrees), no image sample landed in the frame, and fidelity
               passed because the model had driven blind too. Stage A through
               s3 trained that way. A plan that a black frame does not move is
               a model that is not looking, whatever it agrees with.

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
from typing import NamedTuple

import numpy as np

#: `Source.drove` for dumps the weights a run starts from collected.
START = "start"
#: `Source.drove` for logged human driving (real nuScenes): the target is the human's path, there are
#: no dumps and no candidates to grade, so the grading and fidelity checks have nothing to compare.
HUMAN = "human"
#: `Source.drove` for a DAgger round driven by the weights `--init` loads. The start-of-run fidelity
#: check cannot cover it (the released weights never drove it), so `init_fidelity` checks it once
#: those weights are loaded: same rebuild, same limits.
INIT = "init"


class Source(NamedTuple):
    """One labelled directory, the dumps it was labelled from, and who drove.

    `drove` is START when the weights training starts from drove the collection
    (round 0, GT-shadow), INIT when the `--init` weights did (a DAgger round
    continued from the checkpoint that drove it), else a name for the checkpoint
    that did. START sources are fidelity-checked before `--init` loads, INIT
    sources after.
    """

    labelled: str
    dumps: str
    drove: str


#: Pick verdicts may differ on this fraction of sampled ticks before fidelity
#: fails. The measured baseline is zero; this leaves room for a tick whose plan
#: grazes the target by millimetres, where the rebuilt frame's JPEG round trip
#: (the dump saw the pixels the renderer sent) can flip a verdict.
FIDELITY_VERDICT_TOLERANCE = 0.05
#: Median gap between the rebuilt pick and the driven one, as a fraction of four
#: seconds' travel at the tick's speed. Measured 0.29%; a gap this size is three
#: times what the JPEG round trip explains.
FIDELITY_MEDIAN_PICK_FRACTION = 0.01
#: The same gap's 90th percentile. A median cannot see a defect confined to a
#: minority of ticks: 22 of 48 sampled ticks 10 m off still pass it. Twice the p90
#: measured over 446 round-0 ticks with a real past (4.9%), as the median limit is
#: about three times its own baseline.
FIDELITY_P90_PICK_FRACTION = 0.10
#: Sampled ticks fidelity may fail to rebuild. The trainer drops unrebuildable ticks
#: before the preflight (`swerve_data.rebuildable`), so a skip here means the filter
#: and the dataset disagree; unbounded, 42 of 48 skipped still passed on 6.
FIDELITY_MAX_SKIPPED = 0.10
#: Unrebuildable ticks the trainer may drop before it refuses the data. Measured
#: 6.2-6.7% of round 0 (positions 0-4 of each episode), most of them brake targets.
MAX_UNREBUILDABLE = 0.10
#: Blanking the front camera must move the starting weights' plan by more than
#: this somewhere in the sample. Camera-blind it moved 0.000 m on every tick;
#: with the cameras on the ego frame, 9.6 m on a frontal tick near impact.
SIGHT_MIN_MOVE_M = 0.05
#: Ticks the sight check runs, two forward passes each.
SIGHT_TICKS = 8
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
    # The sha is taken HERE, beside the check that makes it mean something, and the
    # trainer writes this one into every sidecar of the run. Read again at save time
    # it names whatever was checked out by then: v4fix's sidecar says 85e1f9c and
    # c18f488 trained it; l5's says a9ef8ba and cdfdb64 trained it.
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    report["sha"] = (
        head.stdout.strip() + ("" if report["git_clean"] else "-dirty") if head.returncode == 0 else "unknown"
    )

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


def episode_key(tick: dict) -> str:
    """The episode's directory on disk, which names it uniquely.

    `tick["episode"]` does not: a DAgger round replays round 0's scenarios and
    seeds, so `stationary-0070/run_3` exists once per round, with the same log
    timestamps. Keyed by that name, two rounds' ticks would merge into one episode.
    """
    return str(Path(tick["frame"]).parent.parent) if tick.get("frame") else tick["episode"]


def dumps_for(tick: dict, dumps) -> Path:
    """`dumps` is one directory for every tick, or a list of `Source`s indexed by `tick["source"]`."""
    return dumps if isinstance(dumps, Path) else Path(dumps[tick.get("source", 0)].dumps)


def sample_ticks(ticks: list[dict], episodes: int = 12, per_episode: int = 4, seed: int = 0) -> list[dict]:
    """Ticks spread across episodes and across each episode, LATE FIRST.

    Each episode contributes ticks at fixed fractions of its length, visited from
    the end backwards: a cache keyed by episode is then filled by a late tick
    and read back for an early one, which is exactly the shape of the v1-v4 bug.
    """
    by_episode: dict[str, list[dict]] = {}
    for tick in ticks:
        by_episode.setdefault(episode_key(tick), []).append(tick)
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


def _dumped(tick: dict, dumps, cache: dict) -> dict | None:
    from gallops.expert.label import match_dump

    dumps = dumps_for(tick, dumps)
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


def grading_checks(sample: list[dict], referee, dumps) -> tuple[dict, list[str]]:
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
        where = dumps if isinstance(dumps, Path) else ", ".join(str(d.dumps) for d in dumps)
        problems.append(f"{missing} of {len(sample)} sampled ticks have no dump under {where}")
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


def drivable_checks(ticks: list[dict]) -> tuple[dict, list[str]]:
    """Every target trained on is one a car could follow -- all of them, not a sample.

    4,347 of round 0's 9,495 targets were not, by `undrivable` (the 934 stops that
    slid 2 m sideways among them), and nothing noticed. Checking all of them costs a
    fraction of a second; a 48-tick sample misses a 1% regression 62% of the time.

    A logged human path is not a drawn target: a car did drive it, and relabelling cannot
    change it, so it is counted apart. 542 of labelled_nusc's 18,015 fail `undrivable` (a
    3.8 m radius at real-0020/run_0), which is the limit's reach, not the driver's error.
    `human_checks` is what stops a drawn target claiming this.
    """
    from gallops.expert.draw import undrivable

    usable = [t for t in ticks if t.get("usable_for_imitation", True)]
    logged = [t for t in usable if t.get("expert_kind") == HUMAN]
    usable = [t for t in usable if t.get("expert_kind") != HUMAN]

    def why(tick: dict) -> str | None:
        return undrivable(np.asarray(tick["expert"]), speed=tick.get("speed"))

    bad = [t for t in usable if why(t)]
    report = {"undrivable": f"{len(bad)}/{len(usable)}"}
    if logged:
        report["logged"] = len(logged)
    if not bad:
        return report, []
    first = bad[0]
    return report, [
        f"{len(bad)} of {len(usable)} targets are undrivable "
        f"({why(first)}, {first['episode']} @ {first['timestamp']}): "
        "relabel with the current expert"
    ]


def human_checks(ticks: list[dict], sources) -> list[str]:
    """A `human` source holds logged driving and nothing else, and no other source holds any.

    Saying `human` skips grading and fidelity, and `drivable_checks` exempts the tick, so the
    claim has to be checked: a rendered round passed as `human` was otherwise never graded,
    never fidelity-checked, and raised nothing.
    """
    problems = []
    for i, source in enumerate(sources):
        kinds = Counter(t.get("expert_kind", "?") for t in ticks if t.get("source", 0) == i)
        if source.drove == HUMAN:
            drawn = {k: n for k, n in sorted(kinds.items()) if k != HUMAN}
            if drawn:
                problems.append(
                    f"sources: {source.labelled} says a human drove it, but it holds drawn targets "
                    f"{drawn}; only logged driving skips the grading and fidelity checks"
                )
        elif kinds.get(HUMAN):
            problems.append(
                f"sources: {source.labelled} holds {kinds[HUMAN]} ticks marked {HUMAN!r} but says "
                f"{source.drove!r} drove it; name it `{HUMAN}`, or relabel it"
            )
    return problems


def fidelity_check(
    model, dataset_cls, sample: list[dict], referee, dumps, torch, device: str
) -> tuple[dict, list[str]]:
    """The starting weights, on rebuilt inputs, against the plan they drove."""
    from gallops.expert.grader import first_hit

    problems, cache = [], {}
    rows = [(t, _dumped(t, dumps, cache)) for t in sample]
    rows = [(t, r) for t, r in rows if r is not None]
    if not rows:
        return {}, ["fidelity: no sampled tick has a dump to compare with"]
    ticks = [t for t, _ in rows]
    driven = {(episode_key(t), int(t["timestamp"])): r for t, r in rows}
    dataset = dataset_cls(ticks, device)
    diffs, fractions, same = [], [], 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # One tick per pass, as paddock drives it: batched, the vendor model hands every
        # tick the first one's image mask (TICKS_PER_PASS in swerve_data), and a correct
        # loader then fails this check while the message blames the inputs.
        for index in range(len(ticks)):
            features, _, used = dataset.batch([index])
            if features is None:
                continue
            picked = model(features)["trajectory"].float().cpu().numpy()
            # `used`, not the indices asked for: batch skips a tick it cannot rebuild.
            for k, tick in enumerate(used):
                plan = picked[k]
                row = driven[(episode_key(tick), int(tick["timestamp"]))]
                ego2world, targets, _ = referee.context(tick)
                hit = first_hit(plan, ego2world, int(tick["timestamp"]), targets)
                # The model's own pick at collection. In GT-driven collection the engine
                # drove the human's path, so `trajectory` is that and `model_trajectory`
                # is what the model chose; older dumps have only `trajectory`.
                recorded = np.asarray(row.get("model_trajectory") or row["trajectory"])
                recorded_hit = first_hit(recorded, ego2world, int(tick["timestamp"]), targets)
                same += (hit is None) == (recorded_hit is None)
                gap = float(np.abs(plan[:, :2] - recorded[:, :2]).max())
                diffs.append(gap)
                # as a share of four seconds' travel: the gap scales with speed
                fractions.append(gap / max(float(tick["speed"]) * 4.0, 1.0))
    model.train(was_training)
    if not diffs:
        return {}, ["fidelity: the dataset rebuilt none of the sampled ticks"]
    agreement = same / len(diffs)
    median = float(np.median(fractions))
    tail = float(np.percentile(fractions, 90))
    skipped = len(ticks) - len(diffs)
    report = {
        "compared": len(diffs),
        "skipped_unrebuildable": skipped,
        "pick_verdict_agrees": f"{same}/{len(diffs)}",
        "median_pick_diff_m": round(float(np.median(diffs)), 4),
        "median_pick_diff_of_travel": round(median, 4),
        "p90_pick_diff_of_travel": round(tail, 4),
    }
    if skipped > FIDELITY_MAX_SKIPPED * len(ticks):
        problems.append(
            f"fidelity: {skipped} of {len(ticks)} sampled ticks could not be rebuilt "
            f"(limit {FIDELITY_MAX_SKIPPED:.0%}); what was compared is not the sample"
        )
    if tail > FIDELITY_P90_PICK_FRACTION:
        problems.append(
            f"fidelity: the rebuilt pick's p90 gap is {tail:.2%} of four seconds' travel "
            f"(limit {FIDELITY_P90_PICK_FRACTION:.0%}): some ticks are rebuilt unlike what the model saw"
        )
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


def sight_check(model, dataset_cls, sample: list[dict], torch, device: str) -> tuple[dict, list[str]]:
    """The starting weights, on rebuilt inputs, with and without the front image."""
    ticks = sample[:SIGHT_TICKS]
    dataset = dataset_cls(ticks, device)
    moves = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in range(len(ticks)):
            # One tick per pass: batched, the vendor model gives every tick the first
            # one's image mask (TICKS_PER_PASS in swerve_data). And two separate
            # builds, because its forward rewrites `lidar2img` in the dict it is
            # given, so the same dict cannot be fed to it twice.
            real, _, used = dataset.batch([index])
            if real is None:
                continue
            dark, _, _ = dataset.batch([index])
            for key in ("camera_feature_1", "camera_feature_2"):
                dark[key] = torch.zeros_like(dark[key])
            seen = model(real)["trajectory"].float().cpu().numpy()
            blind = model(dark)["trajectory"].float().cpu().numpy()
            moves.append(float(np.abs(seen[..., :2] - blind[..., :2]).max()))
    model.train(was_training)
    if not moves:
        return {}, ["sight: the dataset rebuilt none of the sampled ticks"]
    report = {
        "compared": len(moves),
        "max_move_m": round(max(moves), 3),
        "median_move_m": round(float(np.median(moves)), 3),
    }
    if max(moves) <= SIGHT_MIN_MOVE_M:
        return report, [
            f"sight: blanking the front camera moves the plan at most {max(moves):.3f} m over "
            f"{len(moves)} ticks (limit {SIGHT_MIN_MOVE_M} m) -- the image is not reaching the "
            "model; check the camera extrinsics"
        ]
    return report, []


def init_fidelity(
    train: list[dict], referee, sources, model, dataset_cls, torch, device: str, sample_size: int = 48
) -> dict:
    """The fidelity check again, once `--init` has loaded, on the sources those weights drove.

    Raises PreflightFailed on a problem; returns {} when no source is INIT.
    """
    group = [t for t in train if sources[t.get("source", 0)].drove == INIT]
    if not group:
        return {}
    sample = sample_ticks(group, episodes=max(3, sample_size // 4), per_episode=4)
    report, problems = fidelity_check(model, dataset_cls, sample, referee, sources, torch, device)
    print("  preflight, --init weights: " + json.dumps(report), flush=True)
    if problems:
        raise PreflightFailed(problems)
    return report


def keep_rebuildable(ticks: list[dict], rebuildable) -> tuple[list[dict], dict]:
    """The ticks `rebuildable` accepts, and what was dropped, by expert kind.

    Refuses when more than MAX_UNREBUILDABLE of them are dropped: the drop falls on
    an episode's first half second, where most brake targets sit (18.9% of l4fix's
    brake ticks against 1.8% of `continue`), so a bigger one changes what is taught.
    """
    kept = [t for t in ticks if rebuildable(t)]
    dropped = Counter(t.get("expert_kind", "?") for t in ticks if not rebuildable(t))
    report = {
        "dropped": sum(dropped.values()),
        "of": len(ticks),
        "by_expert_kind": {
            k: f"{n}/{sum(1 for t in ticks if t.get('expert_kind', '?') == k)}"
            for k, n in sorted(dropped.items())
        },
    }
    if ticks and report["dropped"] > MAX_UNREBUILDABLE * len(ticks):
        raise SystemExit(
            f"{report['dropped']} of {len(ticks)} ticks cannot be rebuilt (limit {MAX_UNREBUILDABLE:.0%}): "
            f"{report['by_expert_kind']}"
        )
    return kept, report


def earliest_per_episode(ticks: list[dict], episodes: set[str]) -> list[dict]:
    """Each named episode's first tick in `ticks`.

    The sample starts a tenth of the way in, so it almost never holds the ticks
    nearest an episode's start -- exactly where the priming-frame defect lived.
    """
    first: dict[str, dict] = {}
    for tick in ticks:
        key = episode_key(tick)
        if key in episodes and (key not in first or int(tick["timestamp"]) < int(first[key]["timestamp"])):
            first[key] = tick
    return [first[k] for k in sorted(first)]


def describe(ticks: list[dict]) -> dict:
    """What is about to be trained on, for the log and the sidecar."""
    shapes = Counter(t["episode"].split("-", 1)[0] for t in ticks)
    kinds = Counter(t["expert_kind"] for t in ticks)
    creep = sum(1 for t in ticks if t["speed"] < 3.0 and t.get("driven_hits") is not None)
    return {
        "ticks": len(ticks),
        "episodes": len({episode_key(t) for t in ticks}),
        "by_shape": dict(shapes),
        "by_expert_kind": dict(kinds),
        "creep_ticks": creep,
    }


def run(
    repo: Path,
    train: list[dict],
    referee,
    sources,
    model,
    dataset_cls,
    torch,
    device: str,
    sample_size: int = 48,
) -> dict:
    """Every check; raises PreflightFailed listing all problems, or returns the report.

    `sources` is a list of `Source`s, indexed by each tick's `source` (a bare
    dumps directory still works, for one source). Grading is checked on every
    source against its own dumps. Fidelity only on sources the starting weights
    drove: a DAgger round's dumps were made by a later checkpoint, and comparing
    the starting weights with them would fail for the right reason.
    """
    if isinstance(sources, Path):
        sources = [Source("", str(sources), START)]
    report = {"data": describe(train)}
    problems = []
    report["code"], found = code_checks(repo)
    problems += found
    problems += human_checks(train, sources)
    groups = [
        [t for t in train if t.get("source", 0) == i]
        for i in range(len(sources))
        if sources[i].drove != HUMAN
    ]
    groups = [g for g in groups if g]
    per = max(3, sample_size // 4 // max(1, len(groups)))
    sample = [t for g in groups for t in sample_ticks(g, episodes=per, per_episode=4)]
    report["grading"], found = grading_checks(sample, referee, sources)
    problems += found
    report["targets"], found = drivable_checks(train)
    problems += found
    own = [t for t in sample if sources[t.get("source", 0)].drove == START]
    if own:
        chosen = {(episode_key(t), int(t["timestamp"])) for t in own}
        own += [
            t
            for t in earliest_per_episode(
                [t for t in train if sources[t.get("source", 0)].drove == START],
                {episode_key(t) for t in own},
            )
            if (episode_key(t), int(t["timestamp"])) not in chosen
        ]
        report["fidelity"], found = fidelity_check(model, dataset_cls, own, referee, sources, torch, device)
        problems += found
    else:
        problems.append(
            "fidelity: no source was driven by the starting weights, so nothing checks that the "
            "trainer rebuilds the inputs the model saw; add one (round 0 always qualifies)"
        )
    if model is None:
        problems.append("sight: no model was given, so nothing checks that it reads the camera")
    else:
        report["sight"], found = sight_check(model, dataset_cls, sample, torch, device)
        problems += found
    report["sources"] = [
        {
            "labelled": s.labelled,
            "dumps": s.dumps,
            "drove": s.drove,
            "ticks": sum(1 for t in train if t.get("source", 0) == i),
        }
        for i, s in enumerate(sources)
    ]
    print("  preflight: " + json.dumps(report), flush=True)
    if problems:
        raise PreflightFailed(problems)
    return report
