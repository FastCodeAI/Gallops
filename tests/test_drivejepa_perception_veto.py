"""B-1's rule, on hand-made rows (numpy only: the rule and its bookkeeping never need torch).

docs/drivejepa-perception-sidecar/part-b/B-1-veto.md; the settings are D24's.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gallops.drivejepa_perception import b_report, veto


def test_a_horizon_reads_only_its_plan_times():
    # One proposal whose clearance dips to 0.3 m at 1.5 s, the fourth plan time.
    per_time = np.full((1, 1, 8), 10.0)
    per_time[0, 0, 2] = 0.3
    assert not veto.vetoed(per_time, margin=1.0, horizon=1.0)[0, 0]  # 0.5 s and 1.0 s only
    assert veto.vetoed(per_time, margin=1.0, horizon=2.0)[0, 0]
    assert not veto.vetoed(per_time, margin=0.3, horizon=2.0)[0, 0]  # strictly under the margin
    assert [veto.horizon_steps(h) for h in veto.HORIZONS] == [2, 4]


def test_the_pick_is_the_best_score_left_and_ties_go_first():
    scores = np.array([[0.9, 0.8, 0.8, 0.1], [0.5, 0.7, 0.6, 0.2]])
    release = scores.argmax(1)
    no_veto = np.zeros_like(scores, bool)
    pick, every = veto.pick_with_veto(scores, no_veto, release)
    assert pick.tolist() == release.tolist() and not every.any()  # an empty veto changes nothing
    ban = np.array([[True, False, False, False], [False, True, True, True]])
    pick, every = veto.pick_with_veto(scores, ban, release)
    assert pick.tolist() == [1, 0]  # 1 and 2 tie at 0.8: the first, as the release's argmax
    assert not every.any()


def test_when_every_proposal_is_vetoed_the_release_pick_stands():
    scores = np.array([[0.1, 0.9, 0.3]])
    pick, every = veto.pick_with_veto(scores, np.ones((1, 3), bool), np.array([1]))
    assert pick.tolist() == [1] and every.tolist() == [True]


def test_the_ceiling_vetoes_what_the_labels_fail():
    pdm = np.ones((1, 4, 9))
    pdm[0, 1, veto.NC] = 0.0  # an at-fault collision
    pdm[0, 2, veto.NC] = 0.5  # a static object
    pdm[0, 3, veto.TTC] = 0.0
    assert veto.ceiling_vetoed(pdm).tolist() == [[False, True, True, True]]


def test_the_human_filter_forgives_what_the_human_fails():
    plan = np.array([[1, 1, 1, 1, 0.791, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 1, 0]], dtype=float)
    human = np.ones((2, 9))
    got = veto.filtered_final(plan, human)
    assert got[0] == pytest.approx((5 * 0.791 + 9) / 14)  # the vendored formula (navsim-metric-cache.md)
    assert got[1] == 0.0  # a collision the human avoided
    human[1, [veto.NC, veto.EP, veto.TTC]] = 0.0  # the human collided too: NC, its EP and TTC forgiven
    assert veto.filtered_final(plan, human)[1] == pytest.approx(1.0)


def test_city_weights_stand_for_the_target_without_its_missing_city():
    rows = np.array(["lv"] * 6 + ["pit"] * 2 + ["bos"] * 2)
    target = np.array(["lv"] * 4 + ["pit"] * 2 + ["bos"] * 2 + ["sg"] * 2)
    w, record = veto.city_weights(rows, target)
    assert w.sum() == pytest.approx(1.0)
    assert record["dropped"] == ["sg"]
    assert record["used_share"] == pytest.approx({"bos": 0.25, "lv": 0.5, "pit": 0.25})
    assert w[0] == pytest.approx(0.5 / 6) and w[-1] == pytest.approx(0.25 / 2)
    assert record["kish_n_eff"] < len(rows)


def test_the_weighted_se_reduces_to_the_clustered_one():
    rng = np.random.default_rng(0)
    diff, clusters = rng.normal(size=200), rng.integers(0, 17, size=200)
    equal = np.full(200, 1 / 200)
    assert b_report.clustered_se_weighted(diff, clusters, equal) == pytest.approx(
        b_report.clustered_se(diff, clusters), rel=1e-12
    )
    assert np.isnan(b_report.clustered_se_weighted(diff, np.zeros(200), equal))


def test_the_confusion_counts_against_the_labels():
    v = np.array([[True, True, False, False]])
    failing = np.array([[True, False, True, False]])
    assert veto.confusion(v, failing) == {"right": 1, "wrong": 1, "missed": 1, "kept": 1}


def test_the_grid_is_d24s():
    assert veto.settings("B-1-1") == [None]
    assert len(veto.settings("B-1-2")) == 6 and len(veto.settings("B-1-3")) == 12
    assert (
        veto.setting_name((1.0, 2.0, 0.7)) == "m1-h2-c0.7"
        and veto.setting_name((0.5, 1.0, None)) == "m0.5-h1"
    )


def test_the_choice_is_the_best_value_then_the_fewer_vetoes():
    values = {"a": 0.80, "b": 0.82, "c": 0.82}
    rates = {"a": 0.1, "b": 0.3, "c": 0.2}
    assert veto.choose(values, rates) == "c"
    assert veto.choose({"x": 0.5, "y": 0.5}, {"x": 0.2, "y": 0.2}) == "x"  # then the grid's order


def _grid():
    entry = {"value": 0.8, "veto_rate": 0.1}
    return {
        "split": "fresh",
        "smoke": False,
        "rows": 11340,
        "weights": {"dropped": ["sg-one-north"]},
        "results": {
            "B-1-2": {"m1-h1": {"truth": dict(entry)}, "m2-h1": {"truth": {"value": 0.81, "veto_rate": 0.2}}},
            "B-1-3": {"m1-h1-c0.5": {"mean_over_seeds": dict(entry)}},
        },
    }


def test_a_frozen_choice_is_held_to_its_own_values(tmp_path):
    record = veto.frozen_record(_grid(), "abc")
    assert record["chosen"] == {"B-1-2": "m2-h1", "B-1-3": "m1-h1-c0.5"}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(record))
    assert veto.load_frozen(path)["chosen"] == record["chosen"]
    record["chosen"]["B-1-2"] = "m1-h1"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="not what its values give"):
        veto.load_frozen(path)


def test_only_r_cleans_grid_is_frozen_and_only_once(tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    smoke = dict(_grid(), smoke=True, split="select")
    (grid_dir / "grid.json").write_text(json.dumps(smoke))
    with pytest.raises(SystemExit, match="chooses nothing"):
        veto.freeze(grid_dir, tmp_path / "frozen.json")
    (grid_dir / "grid.json").write_text(json.dumps(_grid()))
    veto.freeze(grid_dir, tmp_path / "frozen.json")
    with pytest.raises(SystemExit, match="frozen once"):
        veto.freeze(grid_dir, tmp_path / "frozen.json")
    stored = veto.load_frozen(tmp_path / "frozen.json", grid_dir)
    assert stored["source"]["split"] == "fresh"


def test_settings_are_never_chosen_on_navtest(tmp_path):
    with pytest.raises(SystemExit, match="never chosen on navtest"):
        veto.run_grid(tmp_path, "navtest", tmp_path, tmp_path / "out", smoke=False)
    with pytest.raises(SystemExit, match="smoke run"):
        veto.run_grid(tmp_path, "select", tmp_path, tmp_path / "out", smoke=False)


def _pair(d, se=0.1, ep=0.0, ep_se=0.01):
    return {"epdms_hf": {"a_minus_b": d, "se": se}, "EP": {"a_minus_b": ep, "se": ep_se}}


def test_the_verdicts_follow_predictions():
    pairs = {
        "B-1-1": _pair(1.2),
        "B-1-2": _pair(0.5),
        "B-1-3-s0": _pair(0.25),
        "B-1-3-s1": _pair(0.30),
        "B-1-3-s2": _pair(0.21, ep=-0.002),
    }
    v = veto.verdicts(pairs)
    assert v["V1"]["supports"] and v["V2"]["supports"] and not v["V2"]["refutes"]
    assert v["V3"]["supports"] and not v["V3"]["refutes"]
    assert v["V4"]["share_kept"] == pytest.approx((0.25 + 0.30 + 0.21) / 3 / 0.5)
    assert v["V5"]["ep_lost_points"] == pytest.approx(0.2 / 3) and not v["B-1-4_runs"]
    pairs["B-1-3-s1"] = _pair(-0.3, ep=-0.03)  # one seed loses beyond 2 SE, in EPDMS and in progress
    v = veto.verdicts(pairs)
    assert v["V3"]["refutes"] and not v["V3"]["supports"] and v["B-1-4_runs"]


def test_the_agent_behind_a_veto_is_at_the_closest_time_within_the_horizon():
    per_time = np.full((2, 2, 8), 10.0)
    who = np.full((2, 2, 8), -1, np.int16)
    per_time[0, 1, 1], who[0, 1, 1] = 0.2, 7  # row 0: the release's pick (1) passes agent 7 at 1.0 s
    per_time[0, 1, 3], who[0, 1, 3] = 0.1, 9  # and agent 9 at 2.0 s, beyond a 1 s horizon
    a = {"setting": (1.0, 1.0, None), "per_time": per_time, "who": who}
    a["veto"] = veto.vetoed(per_time, 1.0, 1.0)
    data = {"pick": np.array([1, 0])}
    assert veto.vetoing_agent(a, data).tolist() == [7, -1]


def test_a_frozen_name_is_a_setting_of_the_grid():
    assert veto.setting_of("B-1-3", "m2-h1-c0.7") == (2.0, 1.0, 0.7)
    assert veto.setting_of("B-1-2", "m0.5-h2") == (0.5, 2.0, None)
    with pytest.raises(SystemExit, match="not a setting"):
        veto.setting_of("B-1-2", "m3-h1")


def test_b1_chooses_only_where_the_command_is_one_navtest_uses(tmp_path):
    ego = np.zeros((4, 4, 11), np.float32)
    for row, slot in enumerate((0, 1, 2, 3)):  # left, straight, right, unknown
        ego[row, -1, 7 + slot] = 1.0
    np.save(tmp_path / "ego_status.npy", ego)
    assert veto.known_command(tmp_path).tolist() == [True, True, True, False]
    ego[1, -1, 7:11] = 0.0  # no command at all
    np.save(tmp_path / "ego_status.npy", ego)
    with pytest.raises(SystemExit, match="not one-hot"):
        veto.known_command(tmp_path)
