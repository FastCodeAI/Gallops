"""Train a listwise re-ranker over a frozen model's candidate trajectories.

The first thing gallops trains is not a driving model. Drive-JEPA proposes 32
trajectories and picks one with a scorer trained on NAVSIM's PDM score. Off
NAVSIM that scorer ranks close to random: on KITScenes the best of its 32 is
0.37 m ADE and its pick 2.36 m. Replacing only the choice -- frozen model, small
selector -- recovers most of that gap for the cost of a CPU minute. See
docs/STRATEGY.md.

The input is the dataset `python -m gallops.data.rerank dataset` builds from
paddock's candidate dump, not raw model output. Its features are a transcription
of paddock's, and that module says what holds the two equal. The dataset holds:

    x              (N, K, F)  features of K candidates for N samples
    ade            (N, K)     each candidate's ADE to the logged future
    scene          (N,)       scene id, so held-out selection splits by scene
    feature_names  (F,)       copied into the weights for paddock to check
    dataset, split            what the samples are, for `trained_on`
    feature_probe_*           optional; copied into the weights (`FEATURE_PROBE`)

The label is a soft target over the K, softmax(-ade / temperature); the loss is
listwise cross-entropy; the model is a one-hidden-layer MLP trained full-batch
with Adam. The epoch kept is the one whose argmax pick has the lowest mean ADE
on a held-out slice of *training* scenes -- the metric, not the loss, because a
lower cross-entropy that drives a worse trajectory is not an improvement.

Output, in `{weights}/<model>/<train_run>/step_<epoch>/`:

    reranker.npz     format, feature_names, mean, std, w1, b1, w2, b2, and a
                     probe: inputs plus the scores this forward pass gave them,
                     which paddock recomputes at load and refuses on mismatch;
                     and the export's feature probe (`FEATURE_PROBE`), if any
    checkpoint.json  the sidecar (`gallops.checkpoint`)

numpy only. `pip install -e ".[train]"` provides it.
"""

from __future__ import annotations

import hashlib
import io
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gallops.checkpoint import Checkpoint, TrainedOn, staged, write
from gallops.config import SiteConfig

#: Must equal paddock's `adapters/models/drivejepa_rerank/ranker.py::FORMAT`.
#: A contract string, not an import: neither repo imports the other.
FORMAT = "paddock-listwise-mlp/1"

WEIGHTS_NAME = "reranker.npz"

#: The feature probe: a few real candidate sets as raw model output, and the
#: features gallops' `candidate_features` computed from them. The score probe
#: replays stored features, so it cannot see the two repos' feature copies
#: part; recomputing these can. `python -m gallops.data.rerank dataset` writes
#: them into the export and the trainer carries them into the weights
#: untouched. Paddock does not check them yet (a paddock follow-up), so they are
#: additive under the same FORMAT: an older reader ignores keys it does not use.
FEATURE_PROBE = ("feature_probe_proposals", "feature_probe_logits", "feature_probe_speed", "feature_probe_x")


@dataclass(frozen=True)
class Settings:
    hidden: int = 128
    temperature: float = 1.0
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    holdout_frac: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        # Each of these fits nothing and would still write a checkpoint: zero
        # epochs saves the random initial weights.
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if not 0 < self.holdout_frac < 1:
            raise ValueError(f"holdout_frac must be strictly between 0 and 1, got {self.holdout_frac}")
        if not self.temperature > 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


@dataclass
class Result:
    directory: Path
    epoch: int
    #: Mean ADE of the re-ranker's pick on the held-out scenes -- the same scenes
    #: the epoch was chosen on, so it is optimistic (by ~0.01-0.02 m on the
    #: shipped checkpoint) and not a result to report. The two numbers below do
    #: not depend on that choice.
    selection_pick_ade: float
    holdout_model_pick_ade: float | None
    holdout_best_ade: float
    train_samples: int
    holdout_samples: int
    holdout_scenes: int


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Ranker:
    """relu(z @ w1 + b1) @ w2 + b2 over standardised features; linear if hidden == 0."""

    def __init__(self, in_dim: int, hidden: int, rng: np.random.Generator) -> None:
        self.hidden = hidden
        if hidden:
            self.w1 = rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, hidden))
            self.b1 = np.zeros(hidden)
            self.w2 = rng.normal(0, math.sqrt(2.0 / hidden), (hidden, 1))
            self.params = ["w1", "b1", "w2", "b2"]
        else:
            self.w2 = rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, 1))
            self.params = ["w2", "b2"]
        self.b2 = np.zeros(1)

    def forward(self, z: np.ndarray):
        if self.hidden:
            pre = z @ self.w1 + self.b1
            act = np.maximum(pre, 0.0)
            return (act @ self.w2 + self.b2)[..., 0], (z, pre, act)
        return (z @ self.w2 + self.b2)[..., 0], (z,)

    def grads(self, cache, dscore: np.ndarray) -> dict[str, np.ndarray]:
        g: dict[str, np.ndarray] = {}
        if self.hidden:
            z, pre, act = cache
            g["w2"] = np.einsum("nkh,nk->h", act, dscore)[:, None]
            g["b2"] = dscore.sum(keepdims=True).reshape(1)
            dpre = dscore[..., None] * self.w2[:, 0] * (pre > 0)
            g["w1"] = np.einsum("nki,nkh->ih", z, dpre)
            g["b1"] = dpre.sum(axis=(0, 1))
        else:
            (z,) = cache
            g["w2"] = np.einsum("nki,nk->i", z, dscore)[:, None]
            g["b2"] = dscore.sum(keepdims=True).reshape(1)
        return g

    def state(self) -> dict[str, np.ndarray]:
        return {p: getattr(self, p).copy() for p in self.params}


def split_by_scene(scenes: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of held-out samples: whole scenes, never a scene split across.

    Keyframes of one scene share the road, the weather and the driver, so a
    scene straddling the split would make the held-out number optimistic.
    """
    unique = np.array(sorted(set(scenes.tolist())))
    np.random.default_rng(seed).shuffle(unique)
    held = set(unique[: max(1, int(len(unique) * frac))].tolist())
    return np.array([s in held for s in scenes.tolist()])


def fit(x: np.ndarray, ade: np.ndarray, held: np.ndarray, settings: Settings):
    """Train on ~held, keep the epoch with the lowest held-out pick ADE.

    Returns (ranker, mean, std, best_epoch, best_metric). Standardisation uses
    the training samples only.
    """
    flat = x[~held].reshape(-1, x.shape[-1])
    mean, std = flat.mean(0), flat.std(0) + 1e-8
    z = (x - mean) / std
    zt, target = z[~held], softmax(-ade[~held] / settings.temperature)
    zh, ade_h = z[held], ade[held]

    rng = np.random.default_rng(settings.seed)
    ranker = Ranker(x.shape[-1], settings.hidden, rng)
    moments = {p: (np.zeros_like(v), np.zeros_like(v)) for p, v in ranker.state().items()}
    best = (math.inf, 0, ranker.state())
    for epoch in range(1, settings.epochs + 1):
        scores, cache = ranker.forward(zt)
        # d/dscore of listwise CE(target, softmax(scores)) is softmax - target.
        grads = ranker.grads(cache, (softmax(scores) - target) / len(zt))
        for p in ranker.params:
            value = getattr(ranker, p)
            grad = grads[p] + settings.weight_decay * value
            m, v = moments[p]
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            step = settings.lr * (m / (1 - 0.9**epoch)) / (np.sqrt(v / (1 - 0.999**epoch)) + 1e-8)
            setattr(ranker, p, value - step)
            moments[p] = (m, v)
        if epoch % 5 == 0 or epoch == settings.epochs:
            held_scores, _ = ranker.forward(zh)
            metric = float(ade_h[np.arange(len(zh)), held_scores.argmax(1)].mean())
            if metric < best[0] - 1e-5:
                best = (metric, epoch, ranker.state())
    for p, value in best[2].items():
        setattr(ranker, p, value)
    return ranker, mean, std, best[1], best[0]


def check_export(path: Path, x, ade, scenes, feature_names, model_pick) -> None:
    """Refuse an export that would fit into a checkpoint that looks fine and is not.

    Paddock checks the feature *names* and replays a probe of the same width as
    the weights, so a name list that does not match the columns passes both and
    fails only on the first real sample.
    """
    if x.ndim != 3 or ade.shape != x.shape[:2] or scenes.shape != (len(x),):
        raise ValueError(f"{path}: x {x.shape}, ade {ade.shape}, scene {scenes.shape} disagree")
    if np.shape(feature_names) != (x.shape[-1],):
        raise ValueError(f"{path}: {np.size(feature_names)} feature names for {x.shape[-1]} feature columns")
    if not (np.isfinite(x).all() and np.isfinite(ade).all()):
        raise ValueError(f"{path}: x or ade holds a NaN or an infinity")
    if model_pick is not None and not (
        model_pick.shape == (len(x),)
        and model_pick.dtype.kind in "iu"
        and ((model_pick >= 0) & (model_pick < x.shape[1])).all()
    ):
        raise ValueError(f"{path}: model_pick must be one candidate index in [0, {x.shape[1]}) per sample")


def check_feature_probe(path: Path, probe: dict[str, np.ndarray], x: np.ndarray) -> None:
    """Refuse a feature probe that is partial or not shaped like this export's samples.

    Absent is allowed: an export from before the probe existed still fits, and
    its weights simply carry none. Partial is not, since a reader could not
    replay it.
    """
    if not probe:
        return
    if set(probe) != set(FEATURE_PROBE):
        raise ValueError(f"{path}: feature probe has {sorted(probe)}, needs all of {list(FEATURE_PROBE)}")
    n, (k, f) = len(probe["feature_probe_speed"]), x.shape[1:]
    shapes_agree = (
        probe["feature_probe_speed"].shape == (n,)
        and probe["feature_probe_proposals"].shape[:2] == (n, k)
        and probe["feature_probe_logits"].shape[:2] == (n, k)
        and probe["feature_probe_x"].shape == (n, k, f)
    )
    if not (n and shapes_agree):
        shapes = {key: value.shape for key, value in probe.items()}
        raise ValueError(
            f"{path}: feature probe {shapes} does not fit samples of {k} candidates x {f} features"
        )


def sample_ids_digest(sample_ids: np.ndarray) -> str:
    """sha256 of the sample ids, sorted and newline-joined.

    Sorted so it can be recomputed from any manifest of the same samples, in any
    order. Sample-id overlap is the real contamination check (docs/checkpoints.md);
    the ids themselves are thousands of lines, so the sidecar carries the hash
    and the export carries the list.
    """
    return hashlib.sha256("\n".join(sorted(sample_ids.tolist())).encode()).hexdigest()


def git_sha(repo: Path) -> str:
    """HEAD of the training repo, with `-dirty` when uncommitted changes exist.

    A dirty tree still gets a checkpoint -- refusing would only teach people to
    skip the sidecar -- but the suffix says the sha alone does not reproduce it.
    """

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def train(
    dataset: Path,
    cfg: SiteConfig,
    model: str,
    train_run: str,
    settings: Settings,
    train_git_sha: str,
) -> Result:
    """Fit a re-ranker on an exported dataset and write it as a gallops checkpoint.

    One call is one run. A `train_run` that already exists is refused rather
    than added to: paddock groups a run's steps as one sweep, so a second fit
    beside the first would read as part of it.
    """
    run_dir = cfg.run_dir(model, train_run)
    # A run exists once it has a published step. An empty directory is what a
    # write that failed inside `staged()` leaves, and refusing it would block the
    # retry of a fit that never happened.
    if any(run_dir.glob("step_*")):
        raise ValueError(f"{run_dir} holds a checkpoint already; one fit is one run, name a new --train-run")
    # Read once, and hash what is read: hashing the file again when the sidecar
    # is written would record whatever is on disk by then, not what was fitted.
    raw = dataset.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        x, ade, scenes = npz["x"].astype(np.float64), npz["ade"].astype(np.float64), npz["scene"]
        feature_names = npz["feature_names"]
        feature_probe = {key: npz[key] for key in FEATURE_PROBE if key in npz.files}
        source, split = str(npz["dataset"]), str(npz["split"])
        model_pick = npz["model_pick"] if "model_pick" in npz.files else None
        sample_ids = npz["sample_id"].astype(str) if "sample_id" in npz.files else None
        if "image_geometry" not in npz.files:
            # Paddock's exporter always writes it now, so its absence means an
            # export from before it did. The same sample gives different candidates
            # under another geometry, and paddock's report checks a dump against
            # model.yaml, never against the weights: a checkpoint that cannot say
            # which one it saw can be served under the wrong one with every check
            # passing.
            raise ValueError(
                f"{dataset}: no image_geometry, so this export predates paddock recording it; "
                "re-export it with paddock's current exporter"
            )
        geometry = str(npz["image_geometry"])
    check_export(dataset, x, ade, scenes, feature_names, model_pick)
    if sample_ids is not None and sample_ids.shape != (len(x),):
        raise ValueError(f"{dataset}: {sample_ids.shape} sample ids for {len(x)} samples")
    check_feature_probe(dataset, feature_probe, x)

    held = split_by_scene(scenes, settings.holdout_frac, settings.seed)
    if held.all():
        raise ValueError(
            f"{dataset}: {len(set(scenes.tolist()))} scene(s) leave nothing to train on once the "
            "selection scenes are held out; at least two are needed"
        )
    ranker, mean, std, epoch, _ = fit(x, ade, held, settings)

    # fp32 on disk, as the sidecar declares. Everything below is computed from the
    # cast arrays, so the probe and the reported pick ADE describe the weights
    # paddock loads, not the float64 ones fit() held. Paddock's forward pass
    # promotes the features to float64, and so does this one.
    for name, value in ranker.state().items():
        setattr(ranker, name, value.astype(np.float32))
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    held_scores, _ = ranker.forward((x[held] - mean) / std)
    metric = float(ade[held][np.arange(held.sum()), held_scores.argmax(1)].mean())

    # The probe: a few real held-out samples, raw features in, scores out. Paddock
    # replays it through its own forward pass at load time. These are test
    # vectors rather than weights, so they keep float64.
    probe_x = x[held][:4]
    probe_scores, _ = ranker.forward((probe_x - mean) / std)

    arrays = {
        "format": np.array(FORMAT),
        "feature_names": np.asarray(feature_names).astype(str),
        "mean": mean,
        "std": std,
        "probe_x": probe_x,
        "probe_scores": probe_scores,
        **feature_probe,
        **ranker.state(),
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    weights = buffer.getvalue()

    held_scenes = sorted(set(scenes[held].tolist()))
    train_scenes = sorted(set(scenes[~held].tolist()))
    n_params = sum(value.size for value in ranker.state().values())
    model_pick_ade = None
    if model_pick is not None:
        model_pick_ade = float(ade[held][np.arange(held.sum()), model_pick[held]].mean())
    # Built, and so validated, before anything is written: a sidecar that cannot
    # be constructed must not leave its weights behind on disk.
    checkpoint = Checkpoint(
        model=model,
        step=epoch,
        train_git_sha=train_git_sha,
        train_run=train_run,
        trained_on=(TrainedOn(source, split),),
        selection_split=f"{source}/{split}: {len(held_scenes)} held-out scenes (seed {settings.seed})",
        params_m=n_params / 1e6,
        precision="fp32",
        weights_sha256=hashlib.sha256(weights).hexdigest(),
        notes={
            "settings": settings.__dict__,
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "samples": int(len(x)),
            "scenes": int(len(set(scenes.tolist()))),
            # The scenes actually read, not the split they came from: `trained_on`
            # says kitscenes/train, and only these say which of its scenes.
            "train_scenes": train_scenes,
            "holdout_scenes": held_scenes,
            "selection_pick_ade": metric,
            "numpy": np.__version__,
            **({"sample_ids_sha256": sample_ids_digest(sample_ids)} if sample_ids is not None else {}),
            # The camera geometry the candidates were produced under (see above).
            "image_geometry": geometry,
        },
    )
    directory = cfg.checkpoint_dir(model, train_run, epoch)
    with staged(directory) as tmp:
        (tmp / WEIGHTS_NAME).write_bytes(weights)
        write(tmp, checkpoint)
    return Result(
        directory=directory,
        epoch=epoch,
        selection_pick_ade=metric,
        holdout_model_pick_ade=model_pick_ade,
        holdout_best_ade=float(ade[held].min(1).mean()),
        train_samples=int((~held).sum()),
        holdout_samples=int(held.sum()),
        holdout_scenes=len(held_scenes),
    )
