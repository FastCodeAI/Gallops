"""The sidecar is the contract with paddock, so it gets tested like one."""

from __future__ import annotations

import json

import pytest

from gallops.checkpoint import (
    SIDECAR_NAME,
    Checkpoint,
    SidecarError,
    TrainedOn,
    read,
    staged,
    write,
)


def a_checkpoint(**overrides) -> Checkpoint:
    base = dict(
        model="gallops-v0",
        step=90_000,
        train_git_sha="4f2a91c",
        train_run="nuscenes-base",
    )
    base.update(overrides)
    return Checkpoint(**base)


def test_round_trips_through_disk(tmp_path):
    original = a_checkpoint(
        trained_on=(TrainedOn("nuscenes", "train"),),
        selection_split="val",
        params_m=71.4,
        precision="fp32",
    )
    write(tmp_path, original)
    assert read(tmp_path) == original


def test_written_file_is_the_name_paddock_looks_for(tmp_path):
    write(tmp_path, a_checkpoint())
    assert (tmp_path / SIDECAR_NAME).exists()
    # No .tmp left behind: a reader must never find a half-written sidecar.
    assert not list(tmp_path.glob("*.tmp"))


def test_absent_fields_are_omitted_not_nulled(tmp_path):
    """A sidecar should say what is known, not enumerate what is not."""
    write(tmp_path, a_checkpoint())
    raw = json.loads((tmp_path / SIDECAR_NAME).read_text())
    assert "params_m" not in raw
    # trained_on is kept even when empty, because "not recorded" is a fact a
    # reader needs to see rather than infer from a missing key.
    assert raw["trained_on"] == []


def test_unknown_fields_survive_rather_than_raising(tmp_path):
    """A newer trainer must not break an older reader mid-sweep."""
    (tmp_path / SIDECAR_NAME).write_text(
        json.dumps(
            {
                "model": "gallops-v0",
                "step": 1,
                "train_git_sha": "abc1234",
                "train_run": "r",
                "some_future_field": 42,
            }
        )
    )
    assert read(tmp_path).notes["some_future_field"] == 42


@pytest.mark.parametrize("missing", ["model", "step", "train_git_sha", "train_run"])
def test_required_fields_are_required(tmp_path, missing):
    body = {"model": "m", "step": 1, "train_git_sha": "abc", "train_run": "r"}
    del body[missing]
    (tmp_path / SIDECAR_NAME).write_text(json.dumps(body))
    with pytest.raises(SidecarError, match=missing):
        read(tmp_path)


def test_a_checkpoint_without_a_training_sha_is_refused():
    with pytest.raises(SidecarError, match="train_git_sha"):
        a_checkpoint(train_git_sha="")


def test_selecting_on_a_split_requires_saying_what_was_trained_on():
    with pytest.raises(SidecarError, match="selection_split"):
        a_checkpoint(selection_split="val")


def test_unknown_precision_is_refused():
    with pytest.raises(SidecarError, match="precision"):
        a_checkpoint(precision="int4")


def test_missing_sidecar_names_the_directory(tmp_path):
    with pytest.raises(SidecarError, match=str(tmp_path)):
        read(tmp_path)


def test_malformed_json_names_the_file(tmp_path):
    (tmp_path / SIDECAR_NAME).write_text("{not json")
    with pytest.raises(SidecarError, match="not valid JSON"):
        read(tmp_path)


def test_a_staged_checkpoint_appears_whole(tmp_path):
    directory = tmp_path / "m" / "r" / "step_000000001"
    with staged(directory) as tmp:
        (tmp / "model.safetensors").write_bytes(b"weights")
        write(tmp, a_checkpoint())
        assert not directory.exists(), "visible before it was complete"
    assert sorted(p.name for p in directory.iterdir()) == [SIDECAR_NAME, "model.safetensors"]
    assert [p.name for p in directory.parent.iterdir()] == [directory.name], "a partial was left behind"


def test_a_failed_save_publishes_nothing(tmp_path):
    directory = tmp_path / "step_000000001"
    with pytest.raises(RuntimeError), staged(directory) as tmp:
        (tmp / "model.safetensors").write_bytes(b"weights")
        raise RuntimeError("killed before the sidecar")
    assert list(tmp_path.iterdir()) == []


def test_weights_without_a_sidecar_are_not_published(tmp_path):
    directory = tmp_path / "step_000000001"
    with pytest.raises(SidecarError, match="without its"), staged(directory) as tmp:
        (tmp / "model.safetensors").write_bytes(b"weights")
    assert list(tmp_path.iterdir()) == []


def test_an_existing_checkpoint_is_never_written_over(tmp_path):
    directory = tmp_path / "step_000000001"
    write(directory, a_checkpoint(step=1))
    with pytest.raises(SidecarError, match="already exists"), staged(directory):
        pass
    assert read(directory).step == 1
