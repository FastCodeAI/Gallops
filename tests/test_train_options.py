"""The trainer's data options: which ticks an epoch sees, and how often."""

from __future__ import annotations

from collections import Counter

from gallops.train.swerve import excluded, oversampled


def _ticks():
    kinds = ["continue", "swerve", "brake", "swerve", "least-impact"]
    return [{"episode": f"frontal-0058/run_{i}", "expert_kind": k} for i, k in enumerate(kinds)] + [
        {"episode": "frontal-0064/run_0", "expert_kind": "swerve"},
        {"episode": "side-0064/run_0", "expert_kind": "brake"},
    ]


def test_oversampling_repeats_only_the_named_kind():
    ticks = _ticks()
    order = oversampled(ticks, {"swerve": 4})
    counts = Counter(order)
    for i, tick in enumerate(ticks):
        assert counts[i] == (4 if tick["expert_kind"] == "swerve" else 1)


def test_no_factors_is_every_tick_once():
    ticks = _ticks()
    assert oversampled(ticks, {}) == list(range(len(ticks)))


def test_excluding_an_episode_prefix_keeps_the_rest_of_its_scene():
    kept = excluded(_ticks(), ["frontal-0064"])
    assert [t["episode"] for t in kept if "0064" in t["episode"]] == ["side-0064/run_0"]
    assert len(kept) == len(_ticks()) - 1


def test_an_exclusion_takes_either_spelling_and_refuses_to_drop_nothing():
    import pytest

    from gallops.train.swerve import exclude_scenes

    kept, names = exclude_scenes(_ticks(), ["scene-0064"])
    assert names == ["0064"] and all("0064" not in t["episode"] for t in kept) and len(kept) == 5
    assert exclude_scenes(_ticks(), ["0064"])[0] == kept
    with pytest.raises(SystemExit, match="no loaded tick"):
        exclude_scenes(_ticks(), ["0099"])


def test_trained_on_counts_the_scenes_read_and_carries_the_init_forward(tmp_path):
    from gallops import checkpoint
    from gallops.train.swerve import NUSCENES_TRAIN_SCENES, declared_training, inherited_training

    train = [t for t in _ticks() if "0064" not in t["episode"]]
    declared = declared_training(train, ["/x/labelled_gt", "/x/labelled_round1"], 1.0)
    assert declared[0] == checkpoint.TrainedOn("nuscenes", "train", round(1 / NUSCENES_TRAIN_SCENES, 6))
    assert [t.split for t in declared[1:]] == ["labelled_gt", "labelled_round1"]
    smoke = declared_training(train, ["/x/labelled_gt"], 0.01)
    assert smoke[0].fraction < declared[0].fraction and smoke[1].fraction == 0.01

    step = tmp_path / "step_000000001"
    checkpoint.write(
        step,
        checkpoint.Checkpoint(
            model="m",
            step=1,
            train_git_sha="abc",
            train_run="r",
            trained_on=(checkpoint.TrainedOn("neuroncap-actor-insertion", "labelled_round0"),),
        ),
    )
    inherited = inherited_training(step / "model.pt")
    lineage = declared_training(train, ["/x/labelled_gt"], 1.0, inherited)
    assert lineage[-1].split == "labelled_round0" and len(lineage) == 3


def test_an_init_without_a_sidecar_is_refused(tmp_path):
    import pytest

    from gallops.train.swerve import inherited_training

    with pytest.raises(SystemExit, match="cannot be carried forward"):
        inherited_training(tmp_path / "model.pt")


def test_the_label_digest_sees_a_rewrite(tmp_path):
    from gallops.train.swerve import labelled_digest

    (tmp_path / "a.jsonl").write_text('{"x": 1}\n')
    before = labelled_digest(tmp_path)
    (tmp_path / "a.jsonl").write_text('{"x": 2}\n')
    assert labelled_digest(tmp_path) != before
