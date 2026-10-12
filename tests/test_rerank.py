"""The re-ranker trainer: learns a recoverable choice, writes a checkpoint paddock can verify."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gallops.checkpoint import SidecarError, read
from gallops.config import SiteConfig
from gallops.train import rerank

FEATURES = ("signal", "noise_a", "noise_b")


def a_dataset(path, n=240, k=8, scenes=24, seed=0, **override):
    """Candidates whose ADE is carried by one feature, plus two of noise.

    The model's own pick is random, so the trainer has something to beat.
    `override` replaces or adds arrays, to build a broken export.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k, len(FEATURES)))
    ade = np.abs(x[..., 0]) + 0.05 * rng.random((n, k))
    arrays = dict(
        x=x,
        ade=ade,
        scene=np.array([f"scene{i % scenes:02d}" for i in range(n)]),
        feature_names=np.array(FEATURES),
        dataset=np.array("kitscenes"),
        split=np.array("train"),
        model_pick=rng.integers(0, k, n),
    )
    np.savez(path, **{**arrays, **override})
    return path


def a_site(tmp_path) -> SiteConfig:
    return SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")


def files_under(root) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


def test_holdout_is_whole_scenes():
    scenes = np.array([f"s{i % 10}" for i in range(100)])
    held = rerank.split_by_scene(scenes, 0.3, seed=0)
    for scene in set(scenes.tolist()):
        assert len(set(held[scenes == scene].tolist())) == 1, f"{scene} straddles the split"
    assert 0 < held.sum() < len(held)


def test_learns_the_recoverable_choice(tmp_path):
    result = rerank.train(
        a_dataset(tmp_path / "data.npz"),
        a_site(tmp_path),
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
        a_site(tmp_path),
        model="drivejepa_rerank",
        train_run="synthetic",
        settings=rerank.Settings(hidden=hidden, epochs=50),
        train_git_sha="abc1234",
    )
    assert result.directory == a_site(tmp_path).checkpoint_dir("drivejepa_rerank", "synthetic", result.epoch)

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
        a_site(tmp_path),
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
    # The scene list it read, split into what it trained on and what it chose on.
    train, held = set(sidecar.notes["train_scenes"]), set(sidecar.notes["holdout_scenes"])
    assert not train & held
    assert train | held == {f"scene{i:02d}" for i in range(24)}
    assert sidecar.notes["numpy"] == np.__version__


def test_sidecar_carries_the_exports_sample_ids_and_geometry(tmp_path):
    ids = np.array([f"scene{i % 24:02d}/{i}" for i in range(240)])
    result = rerank.train(
        a_dataset(tmp_path / "data.npz", sample_id=ids, image_geometry=np.array("openscene")),
        a_site(tmp_path),
        "m",
        "r",
        rerank.Settings(hidden=4, epochs=10),
        "abc1234",
    )
    notes = read(result.directory).notes
    assert notes["image_geometry"] == "openscene"
    # Order-free, so it can be recomputed from any manifest of the same samples.
    assert notes["sample_ids_sha256"] == hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    assert notes["sample_ids_sha256"] == rerank.sample_ids_digest(ids[::-1])


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
        rerank.train(path, a_site(tmp_path), "m", "r", rerank.Settings(epochs=5), "abc1234")


def test_a_run_is_one_fit(tmp_path):
    """A second fit under the same --train-run would read as part of the first's sweep."""
    data = a_dataset(tmp_path / "data.npz")
    first = rerank.train(data, a_site(tmp_path), "m", "r", rerank.Settings(hidden=4, epochs=20), "abc1234")
    before = files_under(tmp_path / "weights")
    with pytest.raises(ValueError, match="one fit is one run"):
        rerank.train(data, a_site(tmp_path), "m", "r", rerank.Settings(hidden=4, epochs=10), "abc1234")
    assert files_under(tmp_path / "weights") == before
    assert read(first.directory).step == first.epoch


@pytest.mark.parametrize(
    "model, train_run",
    [("a/b", "r"), ("m", "nested/run"), ("m", ".."), ("m", "/an/absolute/run")],
)
def test_a_name_that_is_not_one_directory_writes_nothing(tmp_path, model, train_run):
    with pytest.raises(ValueError, match="single directory name"):
        rerank.train(
            a_dataset(tmp_path / "data.npz"),
            a_site(tmp_path),
            model,
            train_run,
            rerank.Settings(hidden=4, epochs=10),
            "abc1234",
        )
    assert files_under(tmp_path) == ["data.npz"]


def test_a_sidecar_that_cannot_be_built_leaves_no_weights(tmp_path):
    with pytest.raises(SidecarError, match="train_git_sha"):
        rerank.train(
            a_dataset(tmp_path / "data.npz"), a_site(tmp_path), "m", "r", rerank.Settings(epochs=10), ""
        )
    assert files_under(tmp_path / "weights") == []


@pytest.mark.parametrize(
    "settings",
    [dict(epochs=0), dict(holdout_frac=0.0), dict(holdout_frac=1.0), dict(temperature=0.0)],
)
def test_settings_that_fit_nothing_are_refused(settings):
    with pytest.raises(ValueError):
        rerank.Settings(**settings)


NAN_X = np.where(np.arange(3) == 1, np.nan, 0.0) * np.ones((240, 8, 3))


@pytest.mark.parametrize(
    "dataset, refusal",
    [
        (dict(scenes=1), "nothing to train on"),
        (dict(feature_names=np.array(FEATURES + ("extra",))), "4 feature names for 3"),
        (dict(model_pick=np.zeros(239, dtype=int)), "model_pick"),
        (dict(model_pick=np.full(240, 8)), "model_pick"),
        (dict(x=NAN_X), "NaN"),
        (dict(x=np.zeros((240, 8)), ade=np.zeros(240)), "disagree"),
        (dict(sample_id=np.array(["a"] * 239)), "sample ids"),
    ],
)
def test_an_export_that_would_fit_badly_writes_nothing(tmp_path, dataset, refusal):
    with pytest.raises(ValueError, match=refusal):
        rerank.train(
            a_dataset(tmp_path / "data.npz", **dataset),
            a_site(tmp_path),
            "m",
            "r",
            rerank.Settings(hidden=4, epochs=10),
            "abc1234",
        )
    assert files_under(tmp_path / "weights") == []


def test_the_dataset_hash_is_of_the_bytes_trained_on(tmp_path):
    data = a_dataset(tmp_path / "data.npz")
    result = rerank.train(data, a_site(tmp_path), "m", "r", rerank.Settings(hidden=4, epochs=10), "abc1234")
    assert read(result.directory).notes["dataset_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
