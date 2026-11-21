"""The re-ranker's training data: which keyframes, and the labelled dataset.

`gallops train-rerank` fits a selector over Drive-JEPA's 32 candidates. Its
input takes three steps, and two of them are here:

1. `keyframes` lists the KITScenes train keyframes to dump: every keyframe of
   the scenes in `rerank_train_scenes.txt`.
2. Released Drive-JEPA runs on each, through paddock's own input path
   (`tools/models/drivejepa/paddock_candidates.py --rows`, which paddock held
   until #121; it is copied in experiments/GC01_rerank_kitscenes/paddock/), and
   writes one JSON row of candidates per keyframe.
3. `dataset` turns that dump into the `.npz` `train-rerank` reads: each
   candidate's features, and its ADE to the logged future, which is the label.

    python -m gallops.data.rerank keyframes -o train_samples.pkl
    python -m gallops.data.rerank dataset train.jsonl --split train -o train.npz

`dataset` records the dataset and split each row names (paddock writes them as
`tier` and `split`), and refuses a dump that mixes them; `--dataset`/`--split`
only check that record.

**The features are the adapter's.** The selector ran inside paddock's
`drivejepa_rerank` adapter, which computes them with `candidate_features` in
`adapters/models/drivejepa_rerank/ranker.py`; paddock removed the adapter in #121,
and it is copied in experiments/GC01_rerank_kitscenes/paddock/. Neither imports the other,
so `candidate_features` below is a transcription of it. Weights fitted to a
different definition would still load and pass paddock's probe, which replays
the forward pass on stored features, and then drive the wrong candidate. What
holds the copies equal:
- paddock's loader refuses weights whose `feature_names` differ from its own
- both repos test their copy against the same input and expected values
  (`GOLDEN` in tests/test_rerank_data.py here, `test_features_match_gallops`
  in the adapter's tests, which are a record now and run in no `make check`),
  and `test_features_match_the_recorded_adapter` here compares this
  definition with the recorded adapter's `ranker.py` directly
- the export keeps a feature probe, the raw inputs of its first few samples
  with their features, and the trainer copies it into the weights
  (`gallops.train.rerank.FEATURE_PROBE`), so a reader can recompute it with
  its own definition. Paddock's loader does not do so yet: until it does, the
  two tests above are the only check, and each guards only its own side.
Change one, change both.

numpy only.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import uuid
from pathlib import Path

import numpy as np

from gallops.compat import zip_strict
from gallops.config import ConfigError, SiteConfig
from gallops.train.rerank import FEATURE_PROBE


def refuse_existing(path: Path) -> None:
    # A shipped checkpoint's `dataset_sha256` names one file: overwriting it,
    # even with an export of the same rows, leaves a checkpoint whose recorded
    # input exists nowhere. Checked before the work, not only at the write.
    if path.exists():
        raise SystemExit(f"{path} exists; an output is written once, so name a new one")


def write_new(path: Path, payload: bytes) -> None:
    """Write `payload` to `path`, which must not exist, so a reader sees all of it or nothing.

    Staged under a dot-prefixed sibling and renamed into place, as
    `checkpoint.staged()` does for a step directory: a job killed mid-write
    leaves no truncated file under the name a later step would read.
    """
    refuse_existing(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.partial")
    try:
        tmp.write_bytes(payload)
        tmp.rename(path)
    finally:
        tmp.unlink(missing_ok=True)


# --- which keyframes -----------------------------------------------------------

#: The 373 train scenes the first checkpoint was trained on. They follow no rule,
#: so the list itself is the record.
SCENES = Path(__file__).with_name("rerank_train_scenes.txt")

# The Drive-JEPA devkit KITScenes loader's valid_keyframes(num_future_frames=8,
# stride=10), restated, because that loader imports navsim:
# HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1) frames of history, then
# NUM_FUTURE_FRAMES poses of future, HISTORY_STRIDE frames apart. The constants
# are the devkit's; the frame count is not (see `list_keyframes`).
HISTORY_STRIDE = 5
NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 8
STRIDE = 10


def keyframes(num_frames: int) -> list[int]:
    """Every 10th 10 Hz frame, from the first with 1.5 s of history to the last with 4 s of future."""
    first = HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1)
    last = num_frames - NUM_FUTURE_FRAMES * HISTORY_STRIDE - 1
    return list(range(first, last + 1, STRIDE)) if last >= first else []


def read_scene_ids(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
    ]


def list_keyframes(args: argparse.Namespace) -> None:
    try:
        roots = SiteConfig.load().datasets["kitscenes"].roots
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyError as exc:
        raise SystemExit("gallops.toml configures no [datasets.kitscenes]") from exc
    refuse_existing(args.output)

    ids = read_scene_ids(args.scenes)
    rows, missing = [], []
    for scene_id in ids:
        # Tiers hold different scenes, not nested copies, so look in each.
        scene = next((r / "train" / scene_id for r in roots if (r / "train" / scene_id).is_dir()), None)
        if scene is None:
            missing.append(scene_id)
            continue
        # Frames counted from poses.txt, one row per 10 Hz frame. The devkit counts
        # timestamp.reference.txt instead; paddock's harness builds each sample's
        # history and ground truth from the poses, so they are what a keyframe
        # needs. The two counts were measured equal on all 373 scenes of the
        # tracked list; a different --scenes list is not checked.
        num_frames = len(np.loadtxt(scene / "poses.txt", ndmin=2))
        rows += [{"scene_id": scene_id, "split": "train", "keyframe_index": k} for k in keyframes(num_frames)]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(ids)} listed scenes are under no kitscenes tier's train/: "
            f"{', '.join(missing[:5])} ..."
        )
    write_new(args.output, pickle.dumps(rows))
    print(f"wrote {args.output}: {len(rows)} keyframes from {len(ids)} scenes")


# --- features: transcribed from paddock's ranker.py, keep equal ---------------

#: Waypoints the features look at: 3 s at 0.5 s.
HORIZON = 6

FEATURE_NAMES = (
    "logit_0",
    "logit_1",
    "logit_2",
    "logit_3",
    "logit_4",
    "logit_5",
    "travel_1s",
    "travel_2s",
    "travel_3s",
    "lateral_3s",
    "heading_3s",
    "max_abs_lateral",
    "step_mean",
    "step_min",
    "step_max",
    "step_std",
    "total_turning",
    "ego_speed",
)


def candidate_features(proposals: np.ndarray, logits: np.ndarray, speed: float) -> np.ndarray:
    """(K, F) features for K candidates, in `FEATURE_NAMES` order. Nothing here sees the label."""
    proposals = np.asarray(proposals, dtype=np.float64)
    xy = proposals[:, :HORIZON, :2]
    heading = proposals[:, :HORIZON, 2]
    path = np.concatenate([np.zeros((len(xy), 1, 2)), xy], axis=1)
    steps = np.linalg.norm(np.diff(path, axis=1), axis=2)  # metres per 0.5 s
    dist = np.linalg.norm(xy, axis=2)  # straight-line distance from ego

    features = np.column_stack(
        [
            np.asarray(logits, dtype=np.float64),
            dist[:, 1],
            dist[:, 3],
            dist[:, 5],
            xy[:, 5, 1],
            heading[:, 5],
            np.abs(xy[:, :, 1]).max(axis=1),
            steps.mean(axis=1),
            steps.min(axis=1),
            steps.max(axis=1),
            steps.std(axis=1),
            np.abs(np.diff(heading, axis=1)).sum(axis=1),
            np.full(len(xy), float(speed)),
        ]
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"built {features.shape[1]} features, FEATURE_NAMES lists {len(FEATURE_NAMES)}")
    return features


# --- the label and the dataset -------------------------------------------------

#: The label is ADE over 3 s at 0.5 s, the horizon paddock's KITScenes metric
#: scores. `ade3` restates `paddock.metrics.trajectory.displacement_errors(pred,
#: gt, 0.5, 3.0)["ade"]`, which paddock's report also uses; that function is the
#: definition. Its 6 steps are 3 s only at 0.5 s, so `export` refuses any other
#: `interval_s` rather than label a different horizon.
INTERVAL_S = 0.5
ADE_STEPS = 6


def ade3(pred: np.ndarray, gt: np.ndarray) -> float:
    n = min(len(pred), len(gt), ADE_STEPS)
    return float(np.linalg.norm(pred[:n, :2] - gt[:n, :2], axis=-1).mean())


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def speed(row: dict) -> float:
    return float(np.linalg.norm(row["ego_velocity"]))


def geometry_of(rows: list[dict]) -> str:
    """The one DRIVEJEPA_IMAGE_GEOMETRY a dump was made with; refuses mixed or unrecorded.

    The same sample gives different candidates under `native` and `openscene`,
    so weights fitted on one only fit the model run with the same setting.
    """
    found = {(r.get("variant") or {}).get("image_geometry") for r in rows}
    if None in found or len(found) != 1:
        raise SystemExit(
            f"dump records image_geometry {sorted(map(str, found))}; expected exactly one. "
            "Re-dump with paddock_candidates.py (experiments/GC01_rerank_kitscenes/paddock/), "
            "which records it per row."
        )
    return found.pop()


def source_of(rows: list[dict]) -> tuple[str, str]:
    """The one (dataset, split) a dump's rows come from; refuses mixed or unrecorded.

    This becomes the checkpoint's `trained_on`, the contamination declaration
    paddock's report guard checks a dump's own `(tier, split)` against. Taken
    from the rows rather than typed in: a label that understates what the
    model saw lets the guard pass a report on its training data. KITScenes'
    benchmark dump mixes `val` and `overlap_train_val`, so no single split
    describes it, and it is refused rather than half-declared.
    """
    found = {(r.get("tier"), r.get("split")) for r in rows}
    if len(found) != 1 or None in next(iter(found)):
        raise SystemExit(
            f"dump records (tier, split) {sorted(map(str, found))}; expected exactly one, "
            "since it becomes the checkpoint's trained_on. Export each split on its own."
        )
    return found.pop()


def features_and_labels(rows: list[dict]):
    """(N, K, F) features, (N, K) ADE@3s, (N,) model pick, (N,) scene id."""
    x, ade, picks, scenes = [], [], [], []
    for row in rows:
        proposals = np.asarray(row["proposals"])
        human = np.asarray(row["human_poses"])[:, :2]
        x.append(candidate_features(proposals, np.asarray(row["logits"]), speed(row)))
        ade.append([ade3(p[:, :2], human) for p in proposals])
        picks.append(int(row["picked"]))
        scenes.append(row["scene_id"])
    return np.stack(x), np.array(ade), np.array(picks), np.array(scenes)


#: Samples whose raw inputs the export keeps as its feature probe.
FEATURE_PROBE_SAMPLES = 4


def feature_probe(rows: list[dict], x: np.ndarray) -> dict[str, np.ndarray]:
    """The first few samples' raw candidates, logits and speed, beside the features built from them."""
    head = rows[:FEATURE_PROBE_SAMPLES]
    inputs = (
        np.array([r["proposals"] for r in head], dtype=np.float64),
        np.array([r["logits"] for r in head], dtype=np.float64),
        np.array([speed(r) for r in head]),
        x[: len(head)],
    )
    return dict(zip_strict(FEATURE_PROBE, inputs))


def export(args: argparse.Namespace) -> None:
    refuse_existing(args.output)
    rows = load_rows(args.candidates)
    geometry = geometry_of(rows)
    dataset, split = source_of(rows)
    intervals = {r.get("interval_s") for r in rows}
    if intervals != {INTERVAL_S}:
        raise SystemExit(
            f"dump records interval_s {sorted(map(str, intervals))}; the ADE label needs {INTERVAL_S}"
        )
    for flag, given, found in (("--dataset", args.dataset, dataset), ("--split", args.split, split)):
        if given is not None and given != found:
            raise SystemExit(f"{flag} {given!r}, but the dump's rows record {found!r}")
    x, ade, picks, scenes = features_and_labels(rows)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        x=x,
        ade=ade,
        model_pick=picks,
        scene=scenes,
        sample_id=np.array([r.get("sample_id") or f"{r['scene_id']}/{r['keyframe_index']}" for r in rows]),
        feature_names=np.array(FEATURE_NAMES),
        dataset=np.array(dataset),
        split=np.array(split),
        image_geometry=np.array(geometry),
        **feature_probe(rows, x),
    )
    write_new(args.output, buffer.getvalue())
    n_scenes = len(set(scenes.tolist()))
    print(f"wrote {args.output}: {len(rows)} samples, {n_scenes} scenes, {x.shape[-1]} features")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    kf = sub.add_parser("keyframes", help="list the train keyframes for paddock to dump candidates for")
    kf.add_argument("-o", "--output", type=Path, required=True)
    kf.add_argument("--scenes", type=Path, default=SCENES, help="one scene id per line; # comments")
    kf.set_defaults(func=list_keyframes)

    ds = sub.add_parser("dataset", help="turn a paddock candidate dump into a `train-rerank` dataset")
    ds.add_argument("candidates", type=Path, nargs="+")
    # `trained_on` comes from the rows' own tier and split. These only check it.
    ds.add_argument("--dataset", help="refuse the dump unless its rows come from this dataset")
    ds.add_argument("--split", help="refuse the dump unless its rows come from this split")
    ds.add_argument("-o", "--output", type=Path, required=True)
    ds.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
