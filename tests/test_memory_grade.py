"""gallops.memory.grade: the parts that do not need the vendor's grader."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.memory import grade


def test_metric_cache_file_is_the_vendor_layout(tmp_path):
    path = grade.metric_cache_file(tmp_path, "2021.05.12.19.36.12_veh-35_00005_00204", "0a678d2136b35b56")
    assert path == tmp_path / "2021.05.12.19.36.12_veh-35_00005_00204" / "unknown" / "0a678d2136b35b56" / (
        "metric_cache.pkl"
    )


def test_proposal_states_drops_the_reference_row_and_keeps_millimetres():
    origin = np.array([664885.123456, 3999644.987654])
    simulated = np.zeros((33, 41, 11))
    simulated[0] = -99.0  # the PDM reference path: must not survive
    simulated[1:, :, 0] = origin[0] + np.linspace(0, 40, 41)  # 40 m ahead in map coordinates
    simulated[1:, :, 1] = origin[1] + 0.0012345
    simulated[1:, :, 3] = 12.5  # another channel, stored as it comes
    states = grade.proposal_states(simulated, origin)
    assert states.shape == (32, 41, 11) and states.dtype == np.float32
    assert not (states == -99.0).any()
    # back in map coordinates, float64: well under a millimetre off
    x = states[..., 0].astype(np.float64) + origin[0]
    y = states[..., 1].astype(np.float64) + origin[1]
    assert np.abs(x - simulated[1:, :, 0]).max() < 1e-5
    assert np.abs(y - simulated[1:, :, 1]).max() < 1e-6
    assert (states[..., 3] == 12.5).all()
    # in raw map coordinates float32 would have lost about a quarter of a metre
    assert (
        np.abs(simulated[1:, :, 1].astype(np.float32).astype(np.float64) - simulated[1:, :, 1]).max() > 0.01
    )


def test_proposal_states_refuses_the_wrong_shape():
    with pytest.raises(ValueError, match="simulated states"):
        grade.proposal_states(np.zeros((33, 40, 11)), np.zeros(2))


def test_log_order_is_val_then_selection_then_rest():
    logs = np.array(["t3", "v2", "t1", "v1", "s2", "s1", "t3", "v1"])
    split = np.array(["train", "val", "train", "val", "train", "train", "train", "val"])
    order = grade.log_order(logs, split, selection=["s1", "s2", "v2", "missing"])
    assert order == ["v1", "v2", "s1", "s2", "t1", "t3"]  # each once; a val log listed as selection stays val


def test_graded_rows_need_a_grade_and_a_built_row():
    table = {"graded": np.array([1, 1, 0, 1], bool), "inputs_ok": np.array([1, 0, 1, 1], bool)}
    assert grade.graded_rows(table).tolist() == [0, 3]


def test_log_groups_are_the_three_batches():
    logs = np.array(["t3", "v2", "t1", "v1", "s2", "s1"])
    split = np.array(["train", "val", "train", "val", "train", "train"])
    groups = grade.log_groups(logs, split, selection=["s1", "s2", "v2"])
    assert groups == {"val": ["v1", "v2"], "selection": ["s1", "s2"], "rest": ["t1", "t3"]}
    assert grade.log_order(logs, split, ["s1", "s2", "v2"]) == ["v1", "v2", "s1", "s2", "t1", "t3"]


def test_forgive_sets_a_metric_to_one_only_where_the_human_fails_it():
    lane_fine, light = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    # the human passes both: nothing is forgiven
    assert grade.forgive(lane_fine, light, 1.0, 1.0).tolist() == [[0, 1], [1, 0], [0, 0]]
    # the human fails lane keeping only: every path's lane keeping becomes 1, traffic light unchanged
    assert grade.forgive(lane_fine, light, 0.0, 1.0).tolist() == [[1, 1], [1, 0], [1, 0]]
    # the human fails traffic light only
    assert grade.forgive(lane_fine, light, 1.0, 0.0).tolist() == [[0, 1], [1, 1], [0, 1]]
    assert grade.forgive(lane_fine, light, 1.0, 1.0).dtype == np.float32


def test_arrays_survive_a_restart(tmp_path):
    a = grade.open_arrays(tmp_path, 3, "w+")
    assert a["scores"].shape == (3, 32, 9) and a["states"].shape == (3, 32, 41, 11)
    assert a["forgiving"].shape == (3, 32, 2) and a["human"].shape == (3, 10)
    assert a["origin"].dtype == np.float64
    a["scores"][1] = 0.5
    a["forgiving"][1] = 1.0
    a["origin"][1] = [664885.123456, 3999644.987654]
    a["done"][1] = True
    for v in a.values():
        v.flush()
    del a
    b = grade.open_arrays(tmp_path, 3, "w+")  # a resumed build reopens, never recreates
    assert (b["scores"][1] == 0.5).all() and (b["forgiving"][1] == 1.0).all()
    assert b["done"].tolist() == [False, True, False]
    assert b["origin"][1].tolist() == [664885.123456, 3999644.987654]


def test_grade_rows_keeps_row_order_and_counts_bytes(tmp_path, monkeypatch):
    proposals = np.arange(4 * 32 * 8 * 3, dtype=np.float32).reshape(4, 32, 8, 3)
    np.save(tmp_path / "proposals.npy", proposals)
    for token in ("a", "b"):
        path = grade.metric_cache_file(tmp_path / "mc", "log", token)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * 10)
    seen = []

    def fake(path, poses):
        seen.append((path.parent.name, float(poses[0, 0, 0])))
        v = float(poses[0, 0, 0])
        return {
            "scores": np.full((32, 9), v, np.float32),
            "forgiving": np.ones((32, 2), np.float32),
            "human": np.zeros(10, np.float32),
            "states": np.zeros((32, 41, 11), np.float32),
            "origin": np.ones(2),
        }

    monkeypatch.setattr(grade, "grade_poses", fake)
    out = grade._grade_rows(("log", [3, 1], ["a", "b"], tmp_path / "mc", tmp_path / "proposals.npy"))
    assert (out["log"], out["rows"], out["bytes"]) == ("log", [3, 1], 20)
    assert seen == [("a", float(proposals[3, 0, 0, 0])), ("b", float(proposals[1, 0, 0, 0]))]
    assert out["scores"][0, 0, 0] == proposals[3, 0, 0, 0] and out["scores"][1, 0, 0] == proposals[1, 0, 0, 0]
    assert out["states"].shape == (2, 32, 41, 11) and out["origin"].shape == (2, 2)
    assert out["forgiving"].shape == (2, 32, 2) and out["human"].shape == (2, 10)


def test_score_names_match_the_vendor_columns():
    # after_score_proposals stacks the four gates, the four weighted metrics, then the final score
    assert len(grade.SCORE_NAMES) == 9 and grade.SCORE_NAMES[-1] == "final"
    assert grade.SHARED_WITH_CSV == grade.SCORE_NAMES[:8]
    assert grade.SCORE_NAMES[grade.LANE] == "lane_keeping"
    assert grade.SCORE_NAMES[grade.LIGHT] == "traffic_light_compliance"
    assert grade.HUMAN_NAMES[-1] == "lane_keeping_fine" and len(grade.HUMAN_NAMES) == 10
