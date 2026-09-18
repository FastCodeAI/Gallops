"""Train a listwise re-ranker over a frozen model's candidate trajectories.

The first thing gallops trains is not a driving model. Drive-JEPA proposes 32
trajectories and picks one with a scorer trained on NAVSIM's PDM score. Off
NAVSIM that scorer ranks close to random: on KITScenes the best of its 32 is
0.37 m ADE and its pick 2.36 m. Replacing only the choice -- frozen model, small
selector -- recovers most of that gap for the cost of a CPU minute. See
docs/STRATEGY.md.

The input is a dataset exported by paddock (`tools/models/drivejepa/rerank.py
dataset`), not raw model output, so that paddock alone defines the features:

    x              (N, K, F)  features of K candidates for N samples
    ade            (N, K)     each candidate's ADE to the logged future
    scene          (N,)       scene id, so held-out selection splits by scene
    feature_names  (F,)       copied into the weights for paddock to check
    dataset, split            what the samples are, for `trained_on`

The label is a soft target over the K, softmax(-ade / temperature); the loss is
listwise cross-entropy; the model is a one-hidden-layer MLP trained full-batch
with Adam. The epoch kept is the one whose argmax pick has the lowest mean ADE
on a held-out slice of *training* scenes -- the metric, not the loss, because a
lower cross-entropy that drives a worse trajectory is not an improvement.

Output, in `{weights}/<model>/<train_run>/step_<epoch>/`:

    reranker.npz     format, feature_names, mean, std, w1, b1, w2, b2, and a
                     probe: inputs plus the scores this forward pass gave them,
                     which paddock recomputes at load and refuses on mismatch
    checkpoint.json  the sidecar (`gallops.checkpoint`)

numpy only. `pip install -e ".[train]"` provides it.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gallops.checkpoint import Checkpoint, TrainedOn, write

#: Must equal paddock's `adapters/models/drivejepa_rerank/ranker.py::FORMAT`.
#: A contract string, not an import: neither repo imports the other.
FORMAT = "paddock-listwise-mlp/1"

WEIGHTS_NAME = "reranker.npz"


@dataclass(frozen=True)
class Settings:
    hidden: int = 128
    temperature: float = 1.0
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    holdout_frac: float = 0.15
    seed: int = 0


@dataclass
class Result:
    directory: Path
    epoch: int
    holdout_pick_ade: float
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
    weights_root: Path,
    model: str,
    train_run: str,
    settings: Settings,
    train_git_sha: str,
) -> Result:
    """Fit a re-ranker on an exported dataset and write it as a gallops checkpoint."""
    with np.load(dataset, allow_pickle=False) as npz:
        x, ade, scenes = npz["x"].astype(np.float64), npz["ade"].astype(np.float64), npz["scene"]
        feature_names = npz["feature_names"]
        source, split = str(npz["dataset"]), str(npz["split"])
        model_pick = npz["model_pick"] if "model_pick" in npz.files else None
    if x.shape[:2] != ade.shape or len(scenes) != len(x):
        raise ValueError(f"{dataset}: x {x.shape}, ade {ade.shape}, scene {scenes.shape} disagree")

    held = split_by_scene(scenes, settings.holdout_frac, settings.seed)
    ranker, mean, std, epoch, metric = fit(x, ade, held, settings)

    # The probe: a few real held-out samples, raw features in, scores out. Paddock
    # replays it through its own forward pass at load time.
    probe_x = x[held][:4]
    probe_scores, _ = ranker.forward((probe_x - mean) / std)

    directory = weights_root / model / train_run / f"step_{epoch:09d}"
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "format": np.array(FORMAT),
        "feature_names": np.asarray(feature_names).astype(str),
        "mean": mean,
        "std": std,
        "probe_x": probe_x,
        "probe_scores": probe_scores,
        **ranker.state(),
    }
    weights = directory / WEIGHTS_NAME
    with open(weights, "wb") as handle:
        np.savez(handle, **arrays)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    held_scenes = sorted(set(scenes[held].tolist()))
    n_params = sum(value.size for value in ranker.state().values())
    write(
        directory,
        Checkpoint(
            model=model,
            step=epoch,
            train_git_sha=train_git_sha,
            train_run=train_run,
            trained_on=(TrainedOn(source, split),),
            selection_split=f"{source}/{split}: {len(held_scenes)} held-out scenes (seed {settings.seed})",
            params_m=n_params / 1e6,
            precision="fp32",
            weights_sha256=digest,
            notes={
                "settings": settings.__dict__,
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "samples": int(len(x)),
                "scenes": int(len(set(scenes.tolist()))),
                "holdout_scenes": held_scenes,
                "holdout_pick_ade": metric,
            },
        ),
    )
    model_pick_ade = None
    if model_pick is not None:
        model_pick_ade = float(ade[held][np.arange(held.sum()), model_pick[held]].mean())
    return Result(
        directory=directory,
        epoch=epoch,
        holdout_pick_ade=metric,
        holdout_model_pick_ade=model_pick_ade,
        holdout_best_ade=float(ade[held].min(1).mean()),
        train_samples=int((~held).sum()),
        holdout_samples=int(held.sum()),
        holdout_scenes=len(held_scenes),
    )
