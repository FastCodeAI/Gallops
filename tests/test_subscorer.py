"""The sub-score scorer's contract with paddock, and the checks that guard its training.

None of this needs torch. The layout test is the contract: paddock's `drivejepa_subscore`
adapter rebuilds the module from these shapes and loads `seed_<k>.pt` into it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gallops.train import scorer_train, subscorer

#: What paddock's drivejepa_subscore `build_scorer` makes, written out by hand: MLP over
#: own 256 + probe context 256 + geometry 25, a 256-wide query, 4-head attention over 256 probe
#: tokens with a learned position table.
ADAPTER_LAYOUT = {
    "mlp.0.weight": (1024, 537),
    "mlp.0.bias": (1024,),
    "mlp.1.weight": (1024,),
    "mlp.1.bias": (1024,),
    "mlp.3.weight": (1024, 1024),
    "mlp.3.bias": (1024,),
    "mlp.4.weight": (1024,),
    "mlp.4.bias": (1024,),
    "mlp.6.weight": (6, 1024),
    "mlp.6.bias": (6,),
    "pos": (256, 256),
    "q.weight": (256, 281),
    "q.bias": (256,),
    "attn.in_proj_weight": (768, 256),
    "attn.in_proj_bias": (768,),
    "attn.out_proj.weight": (256, 256),
    "attn.out_proj.bias": (256,),
}


def test_layout_is_what_the_paddock_adapter_builds():
    assert subscorer.layout() == ADAPTER_LAYOUT


def test_geometry_scales_the_plan_and_appends_speed():
    plans = np.zeros((2, 64, 8, 3), dtype=np.float32)
    plans[0, 5, :, 0] = np.arange(1, 9) * 2.0
    plans[0, 5, :, 2] = 0.1
    g = subscorer.geometry(plans, np.array([10.0, 0.0]))
    assert g.shape == (2, 64, 25) and g.dtype == np.float16
    np.testing.assert_allclose(g[0, 5, 0], 0.1, atol=1e-3)  # x of the first waypoint / 20
    np.testing.assert_allclose(g[0, 5, 16:24], 0.1, atol=1e-3)  # headings
    assert g[0, 0, -1] == 1.0 and g[1, 0, -1] == 0.0  # speed / 10


def test_real_ticks_teach_only_what_mask_allows():
    w = subscorer.channel_weights(np.array(["gt", "nusc", "nusc_holdout"]), "mask")
    assert w[0].min() == 1.0
    for row in (1, 2):
        assert list(w[row, 0]) == [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        subscorer.channel_weights(np.array(["gt"]), "some")


def _episode(cache, source, stem, t, tokens_value, rng):
    d = cache / source
    d.mkdir(parents=True, exist_ok=True)
    ts = np.arange(t, dtype=np.int64) * 500_000 + 1_000_000
    native = np.full((t, 32, 8, 256), tokens_value, dtype=np.float16)
    np.savez(
        d / f"{stem}.npz",
        native_tokens=native,
        vocab_tokens=np.full((t, 64, 8, 256), tokens_value, dtype=np.float16),
        natives=rng.normal(size=(t, 32, 8, 3)).astype(np.float32),
        vocab=rng.normal(size=(t, 64, 8, 3)).astype(np.float32),
        pdm_score=rng.random((t, 32)).astype(np.float32),
        usable=np.ones(t, bool),
        in_frame=np.ones(t, bool),
        speed=np.full(t, 5.0, np.float32),
        timestamp=ts,
    )
    labels = {k: np.full((t, 96), np.nan, np.float32) for k in ("hit5_at", "off_road_at", "ttc_at")}
    labels["max_deviation"] = np.zeros((t, 96), np.float32)
    labels["travel"] = np.full((t, 96), 10.0, np.float32)
    np.savez(d / f"{stem}.labels.npz", **labels, human_travel=np.full(t, 20.0, np.float32), timestamp=ts)


def test_reduce_reads_every_labelled_cache_file(tmp_path):
    rng = np.random.default_rng(0)
    _episode(tmp_path, "gt", "frontal-0070__run_0", 3, 0.5, rng)
    _episode(tmp_path, "round0_cam", "side-0053__run_1", 2, 0.25, rng)
    d = subscorer.reduce(tmp_path)
    assert d["own"].shape == (5, 64, 256) and d["probe"].shape == (5, 256, 256)
    assert d["plans"].shape == (5, 64, 8, 3) and d["hit5_at"].shape == (5, 64)
    assert sorted(set(d["episode"])) == ["gt/frontal-0070__run_0", "round0_cam/side-0053__run_1"]
    assert (tmp_path / "probe_input.npz").exists()
    targets = subscorer.targets(d)
    assert targets.shape == (5, 64, 6) and np.allclose(targets[..., 4], 0.5)  # progress 10 / 20


def test_a_reduction_made_mid_build_is_not_reused(tmp_path):
    rng = np.random.default_rng(1)
    _episode(tmp_path, "gt", "frontal-0070__run_0", 2, 0.5, rng)
    assert len(subscorer.reduce(tmp_path)["own"]) == 2
    _episode(tmp_path, "gt", "frontal-0070__run_1", 3, 0.5, rng)
    assert len(subscorer.reduce(tmp_path)["own"]) == 5


def _gt_pair(own_b):
    own = np.zeros((2, 64, 256), np.float16)
    own[1] = own_b
    return {
        "episode": np.array(["gt/frontal-0070__run_0", "gt/side-0070__run_0"]),
        "scene": np.array(["0070", "0070"]),
        "timestamp": np.array([5, 5]),
        "in_frame": np.array([True, True]),
        "own": own,
    }


def test_sight_refuses_features_that_ignore_the_car():
    report, problems = subscorer.sight_from_cache(_gt_pair(0.0))
    assert report["features_differ"] == 0.0 and problems and "not reaching" in problems[0]
    report, problems = subscorer.sight_from_cache(_gt_pair(0.5))
    assert problems == [] and report["features_differ"] == 1.0


def test_sight_refuses_a_cache_with_nothing_to_compare():
    d = _gt_pair(0.5)
    d["episode"] = np.array(["round0_cam/frontal-0070__run_0", "round0_cam/side-0070__run_0"])
    _, problems = subscorer.sight_from_cache(d)
    assert problems and "include a GT source" in problems[0]


def test_fidelity_compares_cached_scores_with_the_live_ones(tmp_path):
    live = np.linspace(0, 1, 32, dtype=np.float32)
    (tmp_path / "frontal-0070__run_0.jsonl").write_text(
        "\n".join(json.dumps({"timestamp": t, "pdm_score": live.tolist()}) for t in (1, 2))
    )
    d = {
        "episode": np.array(["round0_cam/frontal-0070__run_0"] * 2),
        "timestamp": np.array([1, 2]),
        "pdm": np.stack([live, live]),
    }
    report, problems = scorer_train.fidelity_check(d, {"round0_cam": tmp_path}, ["round0_cam"])
    assert problems == [] and report["round0_cam"]["pick_agrees"] == 1.0
    d["pdm"] = np.stack([live[::-1], live[::-1]])
    _, problems = scorer_train.fidelity_check(d, {"round0_cam": tmp_path}, ["round0_cam"])
    assert problems and "did not rebuild" in problems[0]
    _, problems = scorer_train.fidelity_check(d, {}, [])
    assert problems and "no --start source" in problems[0]


def test_benchmark_scenes_are_refused(monkeypatch):
    """The benchmark's scene set is paddock's own (`nuscenes_real.neuroncap_scenes`); a stand-in here."""
    import sys
    import types

    harness = types.ModuleType("harness")
    harness.OFFICIAL_SET = {"frontal": ("0103",)}
    monkeypatch.setitem(sys.modules, "paddock.adapters.benchmarks.neuroncap.harness", harness)
    d = {
        "episode": np.array(["round0_cam/frontal-0103__run_0", "round0_cam/frontal-0070__run_0"]),
        "scene": np.array(["0103", "0070"]),
        "shape": np.array(["frontal", "frontal"]),
        "usable": np.array([True, True]),
    }
    _, problems = scorer_train.data_check(d)
    assert problems and "0103" in problems[0]


def test_choose_switches_only_when_the_rule_says_so():
    s = np.full((2, 3, 6), 0.9)
    s[:, 0, 0] = [0.3, 0.8]  # released pick (plan 0): low no-collision on tick 0 only
    s[:, 2, 4] = 1.0  # plan 2 makes the most progress, so the formula prefers it
    rel = np.array([0, 0])
    assert list(subscorer.choose(s, rel, "absolute")) == [2, 0]
    assert list(subscorer.choose(s, rel, "relative", margin=0.2)) == [2, 0]
    with pytest.raises(ValueError):
        subscorer.choose(s, rel, "sometimes")


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def cpu(self):
        return self


class _FakeModel:
    """Enough of a module for `scorer_train.main`: what it writes, not what it learns."""

    def to(self, device):
        return self

    def state_dict(self):
        return {k: _FakeTensor(v) for k, v in subscorer.layout().items()}

    def parameters(self):
        return []


def _train(monkeypatch, tmp_path, argv=(), fit=lambda *a, **k: None):
    """`scorer_train.main` on a synthetic cache: torch stubbed, a temporary weights root, the real
    file handling. Returns the config and the run's first checkpoint directory."""
    import sys
    import types

    from gallops.config import SiteConfig

    cache = tmp_path / "cache"
    if not cache.exists():
        rng = np.random.default_rng(0)
        _episode(cache, "round0_cam", "frontal-0070__run_0", 3, 0.5, rng)
        _episode(cache, "nusc", "real-0020__run_0", 2, 0.5, rng)
        _episode(cache, "nusc_holdout", "real-0030__run_0", 2, 0.5, rng)
    cfg = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    torch = types.SimpleNamespace(
        from_numpy=lambda a: a,
        manual_seed=lambda s: None,
        save=lambda state, path: Path(path).write_bytes(b"weights"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(SiteConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(scorer_train, "preflight", lambda *a: {"code": {"sha": "checked-at-the-start"}})
    monkeypatch.setattr(scorer_train, "released_state", lambda *a: {})
    monkeypatch.setattr(subscorer, "build", lambda *a: _FakeModel())
    monkeypatch.setattr(subscorer, "fit", fit)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scorer_train", "--cache", str(cache), "--drivable", "d", "--checkpoint", "c"]
        + ["--train-run", "r", "--seeds", "0", "1", "--device", "cpu", *argv],
    )
    return cfg, cfg.checkpoint_dir("drivejepa_subscore", "r", 20)


def test_a_finished_run_is_whole_and_says_what_it_read(tmp_path, monkeypatch):
    """#17 review, 🟠 3: the sha is the one checked before the fits, and `trained_on` the scenes read."""
    from gallops import checkpoint as sidecar

    _, out = _train(monkeypatch, tmp_path)
    scorer_train.main()
    assert sorted(p.name for p in out.iterdir()) == [
        "checkpoint.json",
        "scorer.json",
        "seed_0.pt",
        "seed_1.pt",
    ]
    written = sidecar.read(out)
    assert written.train_git_sha == "checked-at-the-start"
    rendered, real = written.trained_on
    assert "1 " in rendered.split and "0070" in rendered.split and "0053" not in rendered.split
    assert real.dataset == "nuscenes" and real.fraction == round(1 / 700, 6)  # not the held-out scene


def test_a_killed_fit_leaves_no_checkpoint(tmp_path, monkeypatch):
    """#17 review, 🟠 3: seeds written straight into `{weights}` stayed there with no sidecar."""
    fits = []

    def dies_on_the_second_seed(*args, **kwargs):
        fits.append(kwargs["seed"])
        if len(fits) == 2:
            raise KeyboardInterrupt

    cfg, out = _train(monkeypatch, tmp_path, fit=dies_on_the_second_seed)
    with pytest.raises(KeyboardInterrupt):
        scorer_train.main()
    assert not out.exists()
    assert [p for p in out.parent.iterdir() if not p.name.startswith(".")] == []


def test_a_second_fit_into_a_scorer_run_is_refused(tmp_path, monkeypatch):
    """#17 review, 🟠 3: a rerun left the old seeds beside the new one, under a sidecar naming only it."""
    _, out = _train(monkeypatch, tmp_path)
    scorer_train.main()
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(SystemExit, match="already holds a fit"):
        scorer_train.main()
    assert sorted(p.name for p in out.iterdir()) == before
