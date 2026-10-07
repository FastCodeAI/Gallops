"""The re-ranker trainer: learns a recoverable choice, writes a checkpoint paddock can verify."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.checkpoint import read
from gallops.train import rerank

FEATURES = ("signal", "noise_a", "noise_b")


def a_dataset(path, n=240, k=8, scenes=24, seed=0):
    """Candidates whose ADE is carried by one feature, plus two of noise.

    The model's own pick is random, so the trainer has something to beat.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k, len(FEATURES)))
    ade = np.abs(x[..., 0]) + 0.05 * rng.random((n, k))
    np.savez(
        path,
        x=x,
        ade=ade,
        scene=np.array([f"scene{i % scenes:02d}" for i in range(n)]),
        feature_names=np.array(FEATURES),
        dataset=np.array("kitscenes"),
        split=np.array("train"),
        model_pick=rng.integers(0, k, n),
    )
    return path


def test_holdout_is_whole_scenes():
    scenes = np.array([f"s{i % 10}" for i in range(100)])
    held = rerank.split_by_scene(scenes, 0.3, seed=0)
    for scene in set(scenes.tolist()):
        assert len(set(held[scenes == scene].tolist())) == 1, f"{scene} straddles the split"
    assert 0 < held.sum() < len(held)


def test_learns_the_recoverable_choice(tmp_path):
    result = rerank.train(
        a_dataset(tmp_path / "data.npz"),
        tmp_path / "weights",
        model="drivejepa_rerank",
        train_run="synthetic",
        settings=rerank.Settings(hidden=16, epochs=200),
        train_git_sha="abc1234",
    )
    # Random picks average ~0.8 here; the best candidate ~0.1.
    assert result.holdout_pick_ade < 0.3
    assert result.holdout_model_pick_ade is not None and result.holdout_model_pick_ade > 0.5


@pytest.mark.parametrize("hidden", [16, 0])
def test_writes_weights_paddock_can_verify(tmp_path, hidden):
    result = rerank.train(
        a_dataset(tmp_path / "data.npz"),
        tmp_path / "weights",
        model="drivejepa_rerank",
        train_run="synthetic",
        settings=rerank.Settings(hidden=hidden, epochs=50),
        train_git_sha="abc1234",
    )
    expected = tmp_path / "weights" / "drivejepa_rerank" / "synthetic" / f"step_{result.epoch:09d}"
    assert result.directory == expected

    with np.load(result.directory / rerank.WEIGHTS_NAME, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    assert str(arrays["format"]) == rerank.FORMAT
    assert tuple(arrays["feature_names"].tolist()) == FEATURES
    assert ("w1" in arrays) == bool(hidden)

    # Replay the probe the way paddock's reader does, from the file alone.
    z = (arrays["probe_x"] - arrays["mean"]) / arrays["std"]
    if "w1" in arrays:
        z = np.maximum(z @ arrays["w1"] + arrays["b1"], 0.0)
    np.testing.assert_allclose((z @ arrays["w2"] + arrays["b2"])[..., 0], arrays["probe_scores"])


def test_sidecar_declares_what_it_saw_and_chose_on(tmp_path):
    result = rerank.train(
        a_dataset(tmp_path / "data.npz"),
        tmp_path / "weights",
        model="drivejepa_rerank",
        train_run="synthetic",
        settings=rerank.Settings(hidden=8, epochs=20),
        train_git_sha="abc1234",
    )
    sidecar = read(result.directory)
    assert sidecar.step == result.epoch
    assert [(t.dataset, t.split) for t in sidecar.trained_on] == [("kitscenes", "train")]
    assert sidecar.selection_split and sidecar.selection_split.startswith("kitscenes/train")
    assert sidecar.weights_sha256 and len(sidecar.weights_sha256) == 64
    assert len(sidecar.notes["holdout_scenes"]) == result.holdout_scenes


def test_mismatched_arrays_are_refused(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        x=np.zeros((4, 8, 3)),
        ade=np.zeros((4, 7)),
        scene=np.array(["a"] * 4),
        feature_names=np.array(FEATURES),
        dataset=np.array("kitscenes"),
        split=np.array("train"),
    )
    with pytest.raises(ValueError, match="disagree"):
        rerank.train(path, tmp_path / "w", "m", "r", rerank.Settings(epochs=5), "abc1234")


def test_the_scorer_floor_is_the_entropy_of_the_soft_targets():
    """BCE against soft targets cannot go below their entropy, so that floor is
    logged beside it: ln 2 at y = 0.5, zero at y = 0 or 1, and BCE(y, y) itself."""
    torch = pytest.importorskip("torch")
    from gallops.train.losses import soft_target_floor

    assert float(soft_target_floor(torch.tensor([0.5, 0.5]), torch)) == pytest.approx(np.log(2), abs=1e-5)
    assert float(soft_target_floor(torch.tensor([0.0, 1.0]), torch)) == pytest.approx(0.0, abs=1e-4)
    y = torch.tensor([0.2, 0.75, 0.9])
    at_target = torch.nn.functional.binary_cross_entropy(y, y)
    assert float(soft_target_floor(y, torch)) == pytest.approx(float(at_target), abs=1e-5)
