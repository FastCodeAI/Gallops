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
