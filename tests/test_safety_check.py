"""The safety check's pure parts: which ticks it replays, the guard, and the summary."""

from __future__ import annotations

import pytest

from gallops.train import safety_check as sc


def _tick(episode, speed, kind="brake", driven_hits=None):
    return {"episode": episode, "speed": speed, "expert_kind": kind, "driven_hits": driven_hits}


def test_groups_split_creep_from_approach_for_one_scene_and_shape():
    ticks = [
        _tick("stationary-0053/run_0", 1.5, driven_hits=2.0),  # creep: slow and the driven plan hits
        _tick("stationary-0053/run_0", 1.5),  # slow but clear: neither group
        _tick("stationary-0053/run_0", 8.0, "brake"),  # approach
        _tick("stationary-0053/run_0", 8.0, "continue"),  # wrong kind
        _tick("frontal-0053/run_0", 8.0, "brake"),  # other shape
        _tick("stationary-0070/run_0", 8.0, "brake"),  # other scene
    ]
    out = sc.groups(ticks, "0053")
    assert [t["speed"] for t in out["creep"]] == [1.5]
    assert len(out["approach"]) == 1
    assert len(sc.groups(ticks, "0053", "frontal", ("brake",))["approach"]) == 1


def test_groups_thin_the_approach_evenly():
    ticks = [_tick("stationary-0053/run_0", 8.0) for _ in range(300)]
    assert len(sc.groups(ticks, "0053")["approach"]) == sc.MAX_APPROACH


def _row(recomputed, recorded, pick_hit=None, driven_hit=None, **extra):
    row = {
        "recomputed_clear": recomputed,
        "recorded_clear": recorded,
        "pick_hit_at": pick_hit,
        "recorded_driven_hits": driven_hit,
        "pick_clear": pick_hit is None,
        "pick_off_road_at": None,
        "pick_off_path_at": None,
        "pick_hits_other_at": None,
        "pick_end_speed": 1.0,
        "pick_travel": 10.0,
        "expert_travel": 12.0,
        "n_clear": sum(recomputed),
        "best_clear_rank": 1,
    }
    row.update(extra)
    return row


def test_guard_measures_agreement_with_labelling():
    rows = [
        _row([True, False], [True, False]),
        _row([True, True], [True, False], pick_hit=1.0, driven_hit=1.0),
    ]
    check = sc.guard(rows)
    assert check["verdicts_agree"] == pytest.approx(3 / 4)
    assert check["ticks_identical"] == pytest.approx(1 / 2)
    assert check["pick_hits_iff_driven_hit"] == pytest.approx(1.0)
    assert check["verdicts_agree"] < sc.GUARD_MIN_AGREEMENT


def test_summary_counts_why_a_pick_is_unsafe():
    rows = [
        _row([True], [True]),
        _row([False], [False], pick_hit=1.5),
        _row([False], [False], pick_off_road_at=2.0, pick_clear=False),
        _row([True], [True], pick_hits_other_at=3.0),
    ]
    s = sc.summary(rows)
    assert s["n"] == 4
    assert s["safe"] == pytest.approx(2 / 4)  # the other-car hit still counts as safe to the labels
    assert s["hit"] == pytest.approx(1 / 4)
    assert s["off_road"] == pytest.approx(1 / 4)
    assert s["hits_other_car"] == pytest.approx(1 / 4)
    assert sc.summary([]) == {"n": 0}


def test_length_and_end_speed():
    plan = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    assert sc.length(plan) == pytest.approx(3.0)
    assert sc.end_speed(plan) == pytest.approx(2.0)  # 1 m over the last 0.5 s


def _rows(agree: int, of: int = 10):
    row = {"pick_hit_at": None, "recorded_driven_hits": None}
    return [{**row, "recomputed_clear": [True], "recorded_clear": [i < agree]} for i in range(of)]


def test_a_failed_guard_stops_before_any_finetune(tmp_path):
    """#5 review, finding 2: the guard ran after every model, and nothing recorded it."""

    out, calls = tmp_path / "check.json", []

    def measure_model(path):
        calls.append(path)
        return {"approach": _rows(9)}  # 90% agreement

    with pytest.raises(SystemExit, match="GUARD FAILED"):
        sc.measure_all([("released", None), ("s1", "s1.pt")], measure_model, out)
    assert calls == [None], "no finetune is measured once the guard fails"
    assert not out.exists() and (tmp_path / "check.json.guard-failed").exists()


def test_results_are_published_whole_and_carry_their_guard(tmp_path):
    import json

    out = tmp_path / "check.json"

    def dies_on_the_finetune(path):
        if path:
            raise RuntimeError("CUDA out of memory")
        return {"approach": _rows(10)}

    with pytest.raises(RuntimeError):
        sc.measure_all([("released", None), ("s1", "s1.pt")], dies_on_the_finetune, out)
    assert not out.exists(), "a half-measured file must not sit under the name a reader expects"
    sc.measure_all([("released", None)], dies_on_the_finetune, out)
    written = json.loads(out.read_text())
    assert written["guard"]["passed"] and written["guard"]["verdicts_agree"] == 1.0 and "released" in written


def test_a_drivable_directory_without_the_scenes_raster_is_refused(tmp_path):
    import json

    import numpy as np

    from gallops.train.swerve_data import episode_context

    episode = tmp_path / "stationary-0042" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    (episode / "ego_poses.json").write_text(json.dumps({"1": np.eye(4).tolist()}))
    (episode / "actors.json").write_text("[]")
    tick = {"frame": str(episode / "CAM_FRONT" / "1.jpg"), "timestamp": 1}
    assert episode_context(tick)[2] is None  # no drivable directory given: no road check, by request
    with pytest.raises(FileNotFoundError, match="scene-0042"):
        episode_context(tick, tmp_path / "rasters")
