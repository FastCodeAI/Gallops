"""A scorer cache is keyed by everything its tokens depend on, and nothing reads an unkeyed one.

Keyed by file name alone, a "rebuild" into an existing directory kept every
camera-blind, batch-4 file under a fresh manifest (#8 review, finding 2), and the
labels and the folds' reduced input were reused whatever cache build they came
from (#6 review, finding 6). Torch-free.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gallops.train import scorer_cache, scorer_folds, scorer_labels


def _key(tmp_path, **change):
    vocab = tmp_path / "vocab.py"
    vocab.write_text("def vocabulary(*a):\n    return [], []\n")
    fields = {"git_sha": "abc", "batch": 1, "vocab": vocab, "checkpoint": "released.ckpt", "calibration": "x"}
    return scorer_cache.cache_key(**{**fields, **change})


def _episode(out, key, name="gt/frontal-0053__run_0.npz"):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, timestamp=np.arange(3), cache_key=np.array(json.dumps(key, sort_keys=True)))
    return path


def test_a_cache_resumes_only_under_its_own_key(tmp_path):
    out, key = tmp_path / "cache", _key(tmp_path)
    scorer_cache.open_cache(out, key, {"sources": [["/l/gt", "gt"]]})
    assert scorer_cache.require_key(out)["camera_frame"] == "ego"
    path = _episode(out, key)
    assert scorer_cache.is_current(path, key)
    scorer_cache.open_cache(out, key, {"sources": [["/l/r0", "round0"]]})  # resumed, a source added
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sources"] == [["/l/gt", "gt"], ["/l/r0", "round0"]]

    other = _key(tmp_path, batch=4)
    with pytest.raises(SystemExit, match="batch 1 -> 4"):
        scorer_cache.open_cache(out, other, {"sources": []})
    with pytest.raises(SystemExit, match="another key"):
        scorer_cache.is_current(path, other)


def test_old_unkeyed_files_are_not_kept_under_a_new_manifest(tmp_path):
    out = tmp_path / "v2"
    (out / "gt").mkdir(parents=True)
    np.savez(out / "gt" / "frontal-0053__run_0.npz", timestamp=np.arange(3))
    with pytest.raises(SystemExit, match="no manifest key"):
        scorer_cache.open_cache(out, _key(tmp_path), {"sources": []})
    (out / "manifest.json").write_text(json.dumps({"sources": [], "vocab": "v.py"}))
    with pytest.raises(SystemExit, match="records no key"):
        scorer_cache.open_cache(out, _key(tmp_path), {"sources": []})
    with pytest.raises(SystemExit, match="camera-blind"):
        scorer_cache.require_key(out)


def test_labels_are_reused_only_for_the_build_they_graded(tmp_path):
    out, key = tmp_path / "cache", _key(tmp_path)
    path = _episode(out, key)
    assert not scorer_labels.labels_current(path)
    labels = path.with_name(path.name.replace(".npz", ".labels.npz"))
    np.savez(labels, cache_key=np.array(json.dumps(key, sort_keys=True)))
    assert scorer_labels.labels_current(path)
    _episode(out, _key(tmp_path, git_sha="def"))  # the cache file rebuilt
    with pytest.raises(SystemExit, match="graded another build"):
        scorer_labels.labels_current(path)


def test_the_folds_input_is_rebuilt_when_its_inputs_change(tmp_path):
    out, key = tmp_path / "cache", _key(tmp_path)
    scorer_cache.open_cache(out, key, {"sources": []})
    calls = []

    def loader(cache, labelled):
        calls.append(sorted(labelled))
        return {"scene": np.array(["0053"])}

    assert scorer_folds.reduced_input(out, {"gt": tmp_path}, loader)["scene"][0] == "0053"
    scorer_folds.reduced_input(out, {"gt": tmp_path}, loader)
    assert len(calls) == 1, "same inputs: reused"
    scorer_folds.reduced_input(out, {"gt": tmp_path, "r1": tmp_path}, loader)
    assert len(calls) == 2, "another label set: rebuilt"


def test_real_nuscenes_takes_the_command_rule_and_scene_set_from_paddock(monkeypatch):
    """#6 review, finding 7: both were restated; they now follow paddock's own."""
    import sys
    import types

    from gallops.data import nuscenes_real

    seen = {}

    def command_from_future(future):
        seen["points"] = len(future)
        return 7, None, None

    loader = types.ModuleType("loader")
    loader.command_from_future = command_from_future
    harness = types.ModuleType("harness")
    harness.OFFICIAL_SET = {"frontal": ("0103",), "side": ("0921",)}
    monkeypatch.setitem(sys.modules, "paddock.adapters.benchmarks.nuscenes.loader", loader)
    monkeypatch.setitem(sys.modules, "paddock.adapters.benchmarks.neuroncap.harness", harness)
    assert nuscenes_real.command(np.zeros((8, 2))) == 7 and seen["points"] == 6  # the 3 s point
    assert nuscenes_real.neuroncap_scenes() == {"0103", "0921", "0061"}

    monkeypatch.setitem(sys.modules, "paddock.adapters.benchmarks.nuscenes.loader", None)
    with pytest.raises(SystemExit, match="PYTHONPATH"):
        nuscenes_real.command(np.zeros((8, 2)))


def test_real_held_out_scenes_are_never_trained_on():
    """#6 review, finding 9: their scene number is never one of SCENES, so every fold took them."""
    d = {
        "scene": np.array(["0053", "0058", "1234", "1234"]),
        "episode": np.array(["a", "b", "c", "d"]),
        "usable": np.array([True, True, True, True]),
        "holdout": np.array([False, False, True, False]),
    }
    train, test = scorer_folds.fold_rows(d, "0053")
    assert list(test) == [0] and list(train) == [1, 3]
