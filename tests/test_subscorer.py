"""The sub-score scorer's contract with paddock, and the checks that guard its training.

None of this needs torch. The layout test is the contract: paddock's `drivejepa_subscore`
adapter rebuilds the module from these shapes and loads `seed_<k>.pt` into it.
"""

from __future__ import annotations

import json

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
