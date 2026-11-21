"""The re-ranker's training data: paddock's features, the devkit's keyframes, a dataset train-rerank reads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gallops.checkpoint import read
from gallops.compat import zip_strict
from gallops.config import SiteConfig
from gallops.data import rerank as data
from gallops.train import rerank


def golden_input():
    """Three candidates of eight waypoints, six logits each, ego at 4.2 m/s."""
    t = 0.5 * np.arange(1, 9)
    proposals = np.stack(
        [
            np.column_stack([4.0 * t, np.zeros(8), np.zeros(8)]),
            np.column_stack([3.0 * t, 0.2 * t**2, np.arctan(0.4 * t / 3.0)]),
            np.column_stack([2.0 * t - 0.1 * t**2, -0.3 * t, -0.05 * t]),
        ]
    )
    logits = np.array(
        [
            [0.5, -1.0, 2.0, 0.0, 1.5, -0.5],
            [1.0, 0.25, -2.0, 3.0, 0.0, 0.75],
            [-0.5, 0.5, 1.0, -1.5, 2.5, 0.125],
        ]
    )
    return proposals, logits, 4.2


#: travel_1s .. total_turning for `golden_input()`, as the adapter's
#: `ranker.candidate_features` computes them. Paddock removed the adapter in
#: #121, so its own `test_features_match_gallops` runs nowhere now;
#: `test_features_match_the_recorded_adapter` below holds the two definitions
#: equal from this side.
GOLDEN = np.array(
    [
        [4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
        [
            3.0066592756745814,
            6.053098380168622,
            9.178235124467012,
            1.8,
            0.38050637711236496,
            1.8,
            1.5388329608336029,
            1.5008331019803633,
            1.5976545308670458,
            0.0340556209568158,
            0.31393821333654115,
        ],
        [
            1.9235384061671343,
            3.6496575181789317,
            5.1788029504896205,
            -0.8999999999999999,
            -0.15000000000000002,
            0.8999999999999999,
            0.8632635125463421,
            0.7403546447480419,
            0.9864709828474428,
            0.08407005752833539,
            0.12500000000000003,
        ],
    ]
)


def test_features_match_paddock():
    proposals, logits, speed = golden_input()
    features = data.candidate_features(proposals, logits, speed)
    assert features.shape == (3, len(data.FEATURE_NAMES))
    np.testing.assert_array_equal(features[:, :6], logits)
    np.testing.assert_allclose(features[:, 6:17], GOLDEN, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(features[:, 17], speed)


#: The adapter that ran these weights in paddock, kept as a record since paddock
#: #121. It imports only numpy, so it loads here without paddock.
RECORDED_RANKER = (
    Path(__file__).resolve().parents[1]
    / "experiments/GC01_rerank_kitscenes/paddock/src/paddock/adapters/models/drivejepa_rerank/ranker.py"
)


def test_features_match_the_recorded_adapter():
    """The transcription and the adapter it copies agree: format, names and values.

    Weights trained on a different feature definition would pass the adapter's
    name check and its probe, and then rank candidates on the wrong inputs.
    """
    spec = importlib.util.spec_from_file_location("recorded_ranker", RECORDED_RANKER)
    assert spec is not None and spec.loader is not None
    ranker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranker)
    assert ranker.FORMAT == rerank.FORMAT
    assert tuple(ranker.FEATURE_NAMES) == data.FEATURE_NAMES
    proposals, logits, speed = golden_input()
    np.testing.assert_allclose(
        data.candidate_features(proposals, logits, speed),
        ranker.candidate_features(proposals, logits, speed),
        rtol=0,
        atol=1e-12,
    )


def test_keyframes_follow_the_devkit_rule():
    # First keyframe needs 15 frames of history; the last needs 40 frames of future.
    assert data.keyframes(55) == []
    assert data.keyframes(56) == [15]
    assert data.keyframes(100) == [15, 25, 35, 45, 55]


def a_scene(root, scene_id, frames):
    scene = root / "train" / scene_id
    scene.mkdir(parents=True)
    # TUM rows: t x y z qx qy qz qw. Only the row count matters here.
    t = np.arange(frames) * 0.1
    poses = np.column_stack([t, np.arange(frames), np.zeros((frames, 5)), np.ones(frames)])
    np.savetxt(scene / "poses.txt", poses)


def a_config(tmp_path, monkeypatch, tiers):
    config = tmp_path / "gallops.toml"
    config.write_text(
        f'[roots]\nweights = "{tmp_path / "w"}"\nruns = "{tmp_path / "r"}"\n'
        f"[datasets.kitscenes]\ntiers = {json.dumps([str(t) for t in tiers])}\n"
    )
    monkeypatch.setenv("GALLOPS_CONFIG", str(config))


def test_keyframes_finds_each_scene_in_whichever_tier_holds_it(tmp_path, monkeypatch):
    fast, bulk = tmp_path / "fast", tmp_path / "bulk"
    a_scene(fast, "a", 100)
    a_scene(bulk, "b", 56)
    a_config(tmp_path, monkeypatch, [fast, bulk])
    (tmp_path / "scenes.txt").write_text("# a comment\na\nb\n")
    out = tmp_path / "rows.pkl"

    data.list_keyframes(argparse.Namespace(output=out, scenes=tmp_path / "scenes.txt"))

    rows = pickle.loads(out.read_bytes())
    expected = [("a", k) for k in (15, 25, 35, 45, 55)] + [("b", 15)]
    assert [(r["scene_id"], r["keyframe_index"]) for r in rows] == expected
    assert {r["split"] for r in rows} == {"train"}


def test_keyframes_refuses_a_listed_scene_it_cannot_find(tmp_path, monkeypatch):
    a_scene(tmp_path / "fast", "a", 100)
    a_config(tmp_path, monkeypatch, [tmp_path / "fast"])
    (tmp_path / "scenes.txt").write_text("a\nnot-downloaded\n")
    with pytest.raises(SystemExit, match="not-downloaded"):
        data.list_keyframes(argparse.Namespace(output=tmp_path / "rows.pkl", scenes=tmp_path / "scenes.txt"))
    assert not (tmp_path / "rows.pkl").exists()


def test_the_tracked_scene_list_is_the_373():
    ids = data.read_scene_ids(data.SCENES)
    assert len(ids) == len(set(ids)) == 373


def a_dump(path, scenes=6, per_scene=3, geometry="openscene", **override):
    """Rows as paddock_candidates.py writes them: the golden candidates, a logged path near the second."""
    proposals, logits, speed = golden_input()
    rows = []
    for s in range(scenes):
        for k in range(per_scene):
            rows.append(
                {
                    "scene_id": f"scene{s}",
                    "keyframe_index": 15 + 10 * k,
                    "split": "train",
                    "tier": "kitscenes",
                    "sample_id": f"kitscenes/train/scene{s}/{15 + 10 * k}",
                    "variant": {"input_path": "paddock", "image_geometry": geometry},
                    "ego_velocity": [speed, 0.0],
                    "interval_s": 0.5,
                    "human_poses": (proposals[1] + 0.1 * (s + k)).tolist(),
                    "picked": 0,
                    "logits": logits.tolist(),
                    "proposals": proposals.tolist(),
                    **override,
                }
            )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def an_export(dump, out):
    data.export(argparse.Namespace(candidates=[dump], dataset="kitscenes", split="train", output=out))
    return out


def test_dataset_holds_paddocks_features_and_the_ade_label(tmp_path):
    out = an_export(a_dump(tmp_path / "train.jsonl"), tmp_path / "train.npz")
    proposals, logits, speed = golden_input()
    with np.load(out) as npz:
        assert npz["x"].shape == (18, 3, 18)
        np.testing.assert_array_equal(npz["x"][0], data.candidate_features(proposals, logits, speed))
        # The first sample's logged path is the second candidate shifted 0 m: its label is zero.
        assert npz["ade"][0, 1] == 0.0
        # Sample 1 is that path shifted (0.1, 0.1) m: ADE is the shift's length.
        assert npz["ade"][1, 1] == pytest.approx(0.1 * np.sqrt(2))
        assert tuple(npz["feature_names"]) == data.FEATURE_NAMES
        assert str(npz["image_geometry"]) == "openscene"
        assert npz["sample_id"][0] == "kitscenes/train/scene0/15"


def test_dataset_is_one_train_rerank_reads(tmp_path):
    out = an_export(a_dump(tmp_path / "train.jsonl"), tmp_path / "train.npz")
    site = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    result = rerank.train(
        out,
        site,
        model="drivejepa_rerank",
        train_run="synthetic",
        settings=rerank.Settings(hidden=4, epochs=5),
        train_git_sha="abc1234",
    )
    assert read(result.directory).notes["image_geometry"] == "openscene"


@pytest.mark.parametrize("variant", [{"input_path": "paddock"}, None])
def test_dataset_refuses_unrecorded_geometry(tmp_path, variant):
    dump = a_dump(tmp_path / "train.jsonl", variant=variant)
    with pytest.raises(SystemExit, match="image_geometry"):
        an_export(dump, tmp_path / "train.npz")


def test_dataset_refuses_mixed_geometry(tmp_path):
    native = a_dump(tmp_path / "native.jsonl", geometry="native")
    openscene = a_dump(tmp_path / "openscene.jsonl")
    with pytest.raises(SystemExit, match="image_geometry"):
        data.export(
            argparse.Namespace(
                candidates=[native, openscene], dataset="kitscenes", split="train", output=tmp_path / "x.npz"
            )
        )


def an_export_of(dumps, out, **flags):
    data.export(argparse.Namespace(candidates=dumps, output=out, **{"dataset": None, "split": None, **flags}))
    return out


def test_trained_on_comes_from_the_rows_not_the_flags(tmp_path):
    """A dump that mixes splits must not become a checkpoint declaring one of them.

    The real benchmark dump mixes `val` and `overlap_train_val`; exported with
    `--split train`, it used to fit into a checkpoint declaring kitscenes/train,
    and paddock's report guard then passed a report on those very scenes.
    """
    train = a_dump(tmp_path / "train.jsonl")
    val = a_dump(tmp_path / "val.jsonl", split="val")
    with pytest.raises(SystemExit, match="trained_on"):
        an_export_of([train, val], tmp_path / "mixed.npz", dataset="kitscenes", split="train")
    assert not (tmp_path / "mixed.npz").exists()

    out = an_export_of([val], tmp_path / "val.npz")
    site = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    result = rerank.train(out, site, "m", "r", rerank.Settings(hidden=4, epochs=5), "abc1234")
    assert [(t.dataset, t.split) for t in read(result.directory).trained_on] == [("kitscenes", "val")]


@pytest.mark.parametrize(
    "flags, refusal",
    [(dict(split="train"), "--split 'train'"), (dict(dataset="nuscenes"), "--dataset 'nuscenes'")],
)
def test_a_flag_that_disagrees_with_the_rows_is_refused(tmp_path, flags, refusal):
    with pytest.raises(SystemExit, match=refusal):
        an_export_of([a_dump(tmp_path / "val.jsonl", split="val")], tmp_path / "x.npz", **flags)


@pytest.mark.parametrize("override", [dict(split=None), dict(tier=None)])
def test_a_dump_that_does_not_say_its_split_is_refused(tmp_path, override):
    with pytest.raises(SystemExit, match="tier, split"):
        an_export_of([a_dump(tmp_path / "train.jsonl", **override)], tmp_path / "x.npz")


def test_the_weights_carry_a_feature_probe_a_reader_can_recompute(tmp_path):
    """The score probe replays stored features, so only this one sees the feature copies part.

    Paddock's loader does not recompute it yet; this pins gallops' half: the
    raw inputs reach the weights, and a definition that drifted disagrees with them.
    """
    out = an_export(a_dump(tmp_path / "train.jsonl"), tmp_path / "train.npz")
    site = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    result = rerank.train(out, site, "m", "r", rerank.Settings(hidden=4, epochs=5), "abc1234")

    with np.load(result.directory / rerank.WEIGHTS_NAME) as npz:
        assert str(npz["format"]) == rerank.FORMAT
        probe = {key: npz[key] for key in rerank.FEATURE_PROBE}
    keys = ("feature_probe_proposals", "feature_probe_logits", "feature_probe_speed")
    inputs = zip_strict(*(probe[key] for key in keys))
    recomputed = np.stack([data.candidate_features(*sample) for sample in inputs])
    assert recomputed.shape == (data.FEATURE_PROBE_SAMPLES, 3, len(data.FEATURE_NAMES))
    np.testing.assert_array_equal(recomputed, probe["feature_probe_x"])

    # travel_1s read from the wrong waypoint: the drift the score probe passes.
    drifted = recomputed.copy()
    drifted[..., 6] = np.linalg.norm(probe["feature_probe_proposals"][:, :, 2, :2], axis=-1)
    assert not np.allclose(drifted, probe["feature_probe_x"])


@pytest.mark.parametrize(
    "probe, refusal",
    [
        (dict(feature_probe_x=np.zeros((4, 3, 18))), "needs all of"),
        ("short", "does not fit"),
    ],
)
def test_a_feature_probe_that_cannot_be_replayed_writes_nothing(tmp_path, probe, refusal):
    out = an_export(a_dump(tmp_path / "train.jsonl"), tmp_path / "train.npz")
    with np.load(out) as npz:
        arrays = {key: npz[key] for key in npz.files}
    for key in rerank.FEATURE_PROBE:
        if probe == "short":
            if arrays[key].ndim > 1:
                arrays[key] = arrays[key][:, :2]  # two candidates of the export's three
        else:
            arrays.pop(key)
    if probe != "short":
        arrays.update(probe)
    np.savez(tmp_path / "broken.npz", **arrays)
    site = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    with pytest.raises(ValueError, match=refusal):
        rerank.train(tmp_path / "broken.npz", site, "m", "r", rerank.Settings(hidden=4, epochs=5), "abc1234")
    assert not (tmp_path / "weights").exists() or not any((tmp_path / "weights").rglob("*.npz"))


@pytest.mark.parametrize("interval", [0.25, None])
def test_a_dump_at_another_interval_is_refused(tmp_path, interval):
    """The label's 6 steps are 3 s only at 0.5 s; at 0.25 s they would silently be 1.5 s."""
    with pytest.raises(SystemExit, match="interval_s"):
        an_export(a_dump(tmp_path / "train.jsonl", interval_s=interval), tmp_path / "x.npz")


def test_neither_command_writes_over_an_existing_output(tmp_path, monkeypatch):
    """The shipped checkpoint's dataset_sha256 names one train.npz; a re-export must not replace it."""
    a_scene(tmp_path / "fast", "a", 100)
    a_config(tmp_path, monkeypatch, [tmp_path / "fast"])
    (tmp_path / "scenes.txt").write_text("a\n")
    for output, run in (
        (
            tmp_path / "rows.pkl",
            lambda o: data.list_keyframes(argparse.Namespace(output=o, scenes=tmp_path / "scenes.txt")),
        ),
        (tmp_path / "train.npz", lambda o: an_export(a_dump(tmp_path / "train.jsonl"), o)),
    ):
        output.write_bytes(b"what the checkpoint was trained on")
        with pytest.raises(SystemExit, match="exists"):
            run(output)
        assert output.read_bytes() == b"what the checkpoint was trained on"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []
