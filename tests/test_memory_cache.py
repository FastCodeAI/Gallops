"""The release cache's build refuses to start on inputs it cannot vouch for (numpy only).

The model, the forward and the image reads are replaced by stand-ins, and torch by a stub, so
these run in `make check`'s environment. What is tested is the bookkeeping around them: the
fidelity verdict a build needs, and what a resume must share with the run that wrote the folder.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gallops.memory import cache

TREE = "a" * 40


def _ticks(tmp: Path, name: str, order: list[str]) -> Path:
    """A moments table with one moment per entry of `order` (its log name)."""
    d = tmp / name
    d.mkdir()
    n = len(order)
    np.savez(
        d / "moments.npz",
        log=np.array(order),
        token=np.array([f"{name}-{i}" for i in range(n)]),
        split=np.array(["train"] * n),
        timestamp=np.arange(n),
        frame_idx=np.arange(n, dtype=np.int32),
        chain=np.zeros(n, np.int32),
        chain_pos=np.ones(n, np.int32),
        root_now=np.zeros(n, np.int32),
        root_prev=np.zeros(n, np.int32),
        graded=np.ones(n, bool),
        inputs_ok=np.ones(n, bool),
    )
    (d / "logs.json").write_text("{}")
    (d / "manifest.json").write_text(json.dumps({"git_sha": name}))
    return d


def _verdict(path: Path, checkpoint: Path, tree: str = TREE, result: str = "PASS") -> Path:
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    verdict = {"fidelity": result, "self_consistency": "PASS", "checkpoint_sha256": digest, "code_tree": tree}
    path.write_text(json.dumps(verdict))
    return path


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Two logs of four frames, two checkpoints, and the model replaced by one that writes `mark`."""
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("logA", "logB"):
        (logs / f"{name}.pkl").write_bytes(pickle.dumps([{}] * 4))
    extra = {"logs": logs, "blobs": tmp_path, "extra_blobs": tmp_path}
    cfg = types.SimpleNamespace(datasets={"navtrain": types.SimpleNamespace(extra=extra)})
    state: dict = {"tree": TREE}

    def run(agent, builder, agent_input, device):
        return {name: np.full(shape, state["mark"], np.float32) for name, shape in cache.ARRAYS.items()}

    def log_inputs(frames, idx, roots, pool=None, ahead=None):
        yield from idx

    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(__version__="stub", set_num_threads=print)
    )
    monkeypatch.setattr(cache, "enter", lambda cfg: str(state["checkpoint"]))
    monkeypatch.setattr(cache, "build_model", lambda checkpoint, device: (None, None))
    monkeypatch.setattr(cache, "run", run)
    monkeypatch.setattr(cache, "log_inputs", log_inputs)
    monkeypatch.setattr(cache, "code_tree", lambda: state["tree"])
    released, other = tmp_path / "released.ckpt", tmp_path / "other.ckpt"
    released.write_bytes(b"released weights")
    other.write_bytes(b"other weights")
    return types.SimpleNamespace(cfg=cfg, state=state, tmp=tmp_path, released=released, other=other)


def test_a_resume_with_another_checkpoint_and_table_is_refused_and_writes_nothing(site):
    first = _ticks(site.tmp, "ticks_v1", ["logA", "logA", "logB", "logB"])
    rebuilt = _ticks(site.tmp, "ticks_v2", ["logB", "logB", "logA", "logA"])  # same size, rows moved
    out = site.tmp / "release"
    site.state.update(checkpoint=site.released, mark=1.0)
    cache.build(site.cfg, first, out, "cpu", _verdict(site.tmp / "f1.json", site.released), limit=1)
    before = np.load(out / "ego.npy").copy()

    site.state.update(checkpoint=site.other, mark=2.0)
    with pytest.raises(SystemExit) as refused:
        cache.build(site.cfg, rebuilt, out, "cpu", _verdict(site.tmp / "f2.json", site.other))
    assert "checkpoint_sha256" in str(refused.value) and "ticks_sha256" in str(refused.value)
    assert np.array_equal(np.load(out / "ego.npy"), before)
    assert json.loads((out / "manifest.json").read_text())["checkpoint_sha256"] == (
        hashlib.sha256(site.released.read_bytes()).hexdigest()
    )


def test_a_resume_with_other_code_is_refused(site):
    ticks_dir = _ticks(site.tmp, "ticks", ["logA", "logA", "logB", "logB"])
    out = site.tmp / "release"
    site.state.update(checkpoint=site.released, mark=1.0)
    cache.build(site.cfg, ticks_dir, out, "cpu", _verdict(site.tmp / "f1.json", site.released), limit=1)
    site.state["tree"] = "b" * 40
    with pytest.raises(SystemExit, match="code_tree"):
        cache.build(site.cfg, ticks_dir, out, "cpu", _verdict(site.tmp / "f2.json", site.released, "b" * 40))


def test_a_resume_with_the_same_inputs_finishes_the_cache(site):
    ticks_dir = _ticks(site.tmp, "ticks", ["logA", "logA", "logB", "logB"])
    out = site.tmp / "release"
    fidelity = _verdict(site.tmp / "f.json", site.released)
    site.state.update(checkpoint=site.released, mark=1.0)
    cache.build(site.cfg, ticks_dir, out, "cpu", fidelity, limit=1)
    cache.build(site.cfg, ticks_dir, out, "cpu", fidelity)
    assert np.load(out / "done.npy").all()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fidelity_sha256"] == hashlib.sha256(fidelity.read_bytes()).hexdigest()
    assert manifest["code_tree"] == TREE and manifest["rows"] == 4


@pytest.mark.parametrize(
    "case",
    ["no verdict", "verdict failed", "verdict for other weights", "verdict for other code", "dirty code"],
)
def test_a_build_without_a_matching_passing_fidelity_verdict_is_refused(site, case):
    ticks_dir = _ticks(site.tmp, "ticks", ["logA", "logB"])
    site.state.update(checkpoint=site.released, mark=1.0)
    fidelity = site.tmp / "fidelity.json"
    if case == "verdict failed":
        _verdict(fidelity, site.released, result="FAIL")
    elif case == "verdict for other weights":
        _verdict(fidelity, site.other)
    elif case == "verdict for other code":
        _verdict(fidelity, site.released, tree="c" * 40)
    elif case == "dirty code":
        site.state["tree"] = TREE + "-dirty"
        _verdict(fidelity, site.released, tree=TREE + "-dirty")
    with pytest.raises(SystemExit, match="refusing to build"):
        cache.build(site.cfg, ticks_dir, site.tmp / "release", "cpu", fidelity)
    assert not (site.tmp / "release" / "manifest.json").exists()
