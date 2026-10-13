"""A drivejepa_b2d checkpoint reaches {weights} whole and with a true sidecar, or not at all.

The writer itself needs torch and a 3 GB checkpoint; what it publishes and what its sidecar claims are decided
in `drivejepa_b2d.sidecar`, which needs neither, so they are tested here on synthetic files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gallops import recipe as recipes
from gallops.checkpoint import Checkpoint, read
from gallops.drivejepa_b2d import sidecar

WEIGHTS = "w.ckpt"


def a_record(**overrides) -> Checkpoint:
    fields = dict(model="m", step=1, train_git_sha="abc", train_run="r", notes={"note": "x"})
    fields.update(overrides)
    return Checkpoint(**fields)


def save_ok(staging: Path) -> dict:
    (staging / WEIGHTS).write_bytes(b"weights")
    return {"probe": "ok"}


def test_a_published_checkpoint_is_whole(tmp_path: Path) -> None:
    out = tmp_path / "m" / "r" / "step_000000001"
    rec = sidecar.publish(out, a_record(), WEIGHTS, save_ok)
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint.json", WEIGHTS]
    on_disk = read(out)
    assert on_disk.weights_sha256 == sidecar.sha256(out / WEIGHTS) == rec.weights_sha256
    assert on_disk.notes == {"note": "x", "probe": "ok"}


def test_a_failed_probe_leaves_nothing_in_weights(tmp_path: Path) -> None:
    """The bug this replaced: the weights were saved into the step directory before the probe ran, so a failed
    probe left weights with no sidecar where paddock looks."""

    def save_then_fail(staging: Path) -> dict:
        (staging / WEIGHTS).write_bytes(b"weights")
        raise SystemExit("PROBE FAILED")

    out = tmp_path / "m" / "r" / "step_000000001"
    with pytest.raises(SystemExit, match="PROBE FAILED"):
        sidecar.publish(out, a_record(), WEIGHTS, save_then_fail)
    assert not (tmp_path / "m" / "r").exists()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_a_failed_first_write_does_not_block_the_retry(tmp_path: Path) -> None:
    def save_crash(staging: Path) -> dict:
        raise RuntimeError("out of disk")

    out = tmp_path / "m" / "r" / "step_000000001"
    with pytest.raises(RuntimeError):
        sidecar.publish(out, a_record(), WEIGHTS, save_crash)
    sidecar.publish(out, a_record(), WEIGHTS, save_ok)
    assert (out / "checkpoint.json").exists()


def test_one_fit_is_one_train_run(tmp_path: Path) -> None:
    """A rerun that keeps a different epoch must not land beside the first under the same train run."""
    sidecar.publish(tmp_path / "m" / "r" / "step_000000003", a_record(step=3), WEIGHTS, save_ok)
    with pytest.raises(SystemExit, match="already holds a checkpoint"):
        sidecar.publish(tmp_path / "m" / "r" / "step_000000005", a_record(step=5), WEIGHTS, save_ok)
    assert [p.name for p in (tmp_path / "m" / "r").iterdir()] == ["step_000000003"]


def test_step_is_the_optimizer_step_not_the_epoch() -> None:
    """Split-v4's head: kept epoch 20 of a stage with 28,266 training rows in batches of 64."""
    metrics = {"best_epoch": 20, "n_train": 28_266, "batch": 64, "steps_per_epoch": 442}
    assert sidecar.optimizer_step(metrics) == 8_840


def test_a_head_without_its_step_count_is_refused() -> None:
    with pytest.raises(SystemExit, match="steps_per_epoch"):
        sidecar.optimizer_step({"best_epoch": 20, "n_train": 28_266})


def test_the_sidecar_pins_the_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "data_x.json"
    manifest.write_text('{"files": {"a.npz": {"bytes": 1, "sha256": "00"}}}')
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"val": ["clip_a", "clip_b"], "source": "test"}))
    notes = sidecar.input_notes([manifest], heldout)
    assert notes["dataset_manifests"] == {"data_x.json": sidecar.sha256(manifest)}
    assert notes["holdout_scenes"] == ["clip_a", "clip_b"]
    assert notes["holdout_sha256"] == sidecar.sha256(heldout)


def test_policy_env_is_parsed_strictly() -> None:
    assert sidecar.parse_env(["DRIVEJEPA_PICK=split", "DRIVEJEPA_SPLIT_K=4"]) == {
        "DRIVEJEPA_PICK": "split",
        "DRIVEJEPA_SPLIT_K": "4",
    }
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        sidecar.parse_env(["DRIVEJEPA_PICK"])
    with pytest.raises(SystemExit, match="twice"):
        sidecar.parse_env(["A=1", "A=2"])


def test_every_checkpoint_stage_records_how_to_drive_it_and_what_it_read() -> None:
    """A split-head checkpoint scored under the adapter's defaults is a different model; one with no manifest
    cannot be told apart from one trained on a regenerated dump."""
    stages = [s for s in recipes.load("drivejepa_b2d").stages if s.module.endswith(".write_checkpoint")]
    assert stages
    for s in stages:
        args = list(s.args)
        flags = {a for a in args[::2] if a.startswith("--")}
        values = {flag: [args[i + 1] for i, a in enumerate(args[:-1]) if a == flag] for flag in flags}
        assert sidecar.parse_env(values["--policy-env"])["DRIVEJEPA_PICK"] == "split", s.name
        assert values["--data-manifest"] and values["--holdout-list"], s.name
        for m in values["--data-manifest"]:
            assert m.startswith("{repo}/recipes/drivejepa_b2d/data_"), m


def test_the_area_head_switch_names_the_shipped_file() -> None:
    env = sidecar.with_area_head({"DRIVEJEPA_PICK": "split"})
    assert env == {"DRIVEJEPA_PICK": "split", "DRIVEJEPA_SPLIT_AREA": sidecar.AREA_HEAD_FILE}
    with pytest.raises(SystemExit, match="--area"):
        sidecar.with_area_head({"DRIVEJEPA_SPLIT_AREA": "/some/site/path.pt"})


def test_the_final_ships_its_area_head() -> None:
    """The bug this replaced: final-area-head's output stayed in {runs}, so a site taking the Final from
    {weights} drove it with the area factor off -- not the Final."""
    rec = recipes.load("drivejepa_b2d")
    stage = rec.stage("final-checkpoint")
    args = list(stage.args)
    assert args[args.index("--area") + 1] == "{out}/final-area-head"
    assert "final-area-head" in stage.after


def test_the_generator_refuses_to_start_without_room_for_its_cache(tmp_path: Path) -> None:
    from gallops.drivejepa_b2d import host

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "feat_train_0_0000.npy").write_bytes(b"x" * 3000)
    (cache / "feat_held_0_0000.npy").write_bytes(b"x" * 1000)
    (cache / "meta_train_0_0000.npz").write_bytes(b"x" * 500)
    assert host.cache_bytes(cache) == 4000
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable:       3 kB\n")
    with pytest.raises(SystemExit, match="refusing"):
        host.require_free_ram(host.cache_bytes(cache), "test", meminfo=str(meminfo))
    meminfo.write_text("MemAvailable:       5 kB\n")
    host.require_free_ram(host.cache_bytes(cache), "test", meminfo=str(meminfo))
