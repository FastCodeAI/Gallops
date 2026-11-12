"""H15 Part B's data rules: the navtest and R-clean lists, cities, and the subsets (numpy only).

The lists are re-derived on every load and refused if edited (B-0, B-0-1). These tests hold the
rules to synthetic filters and logs, and the committed lists, once written, to their own counts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gallops.drivejepa_perception import navsim_data as nd


def _filters():
    """A navtrain filter naming two logs, another naming a third, and one naming only tokens."""
    return {
        "navtrain": {"log_names": ["a", "b"], "tokens": ["t1", "t2"], "sha256": "s-navtrain"},
        "navmini": {"log_names": ["c"], "tokens": ["t3"], "sha256": "s-navmini"},
        "private": {"log_names": None, "tokens": ["p1"], "sha256": "s-private"},
    }


def _fresh(windows=None, cities=None):
    windows = windows if windows is not None else {"d": ["w2", "w1"], "e": [], "f": ["w3"]}
    cities = cities if cities is not None else {"d": "us-nv-las-vegas-strip", "f": "us-ma-boston"}
    return nd.fresh_record(["a", "b", "c", "d", "e", "f"], _filters(), windows, cities)


def test_navtest_record_pins_the_vendor_filter(tmp_path):
    filt = {"log_names": ["y", "x"], "tokens": ["t2", "t1"], "file": "f/navtest.yaml", "sha256": "abc"}
    record = nd.navtest_record(filt)
    assert record["logs"] == ["x", "y"] and record["n_tokens"] == 2
    assert record["source"] == {"file": "f/navtest.yaml", "sha256": "abc"}
    assert record["data_split"] == "test"
    path = tmp_path / "navtest.json"
    path.write_text(json.dumps(record))
    assert nd.load_navtest(path, filt) == record


def test_an_edited_navtest_list_is_refused(tmp_path):
    filt = {"log_names": ["x", "y"], "tokens": ["t1", "t2"], "file": "f", "sha256": "abc"}
    record = nd.navtest_record(filt)
    record["logs"] = ["x"]
    path = tmp_path / "navtest.json"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="logs"):
        nd.load_navtest(path, filt)
    with pytest.raises(SystemExit, match="source"):
        nd.load_navtest(path, {**filt, "sha256": "changed"})


def test_fresh_is_the_unnamed_trainval_logs_with_a_routed_window():
    record = _fresh()
    assert record["n_candidates"] == 3  # d, e, f: a and b are navtrain's, c is navmini's
    assert record["logs"] == ["d", "f"]  # e has no routed window
    assert record["windows"] == {"d": 2, "f": 1} and record["n_tokens"] == 3
    assert record["windows_per_city"] == {"us-ma-boston": 1, "us-nv-las-vegas-strip": 2}
    assert record["cities"] == {"d": "us-nv-las-vegas-strip", "f": "us-ma-boston"}
    assert record["tokens_sha256"] == nd._sha256_lines(["w1", "w2", "w3"])
    assert record["sources"] == {"navmini": "s-navmini", "navtrain": "s-navtrain", "private": "s-private"}


def test_a_window_a_filter_names_is_refused():
    with pytest.raises(SystemExit, match="NAVSIM filter names"):
        _fresh(windows={"d": ["p1"]}, cities={"d": "us-nv-las-vegas-strip"})


def test_the_fresh_list_is_held_to_its_rule(tmp_path):
    record = _fresh()
    path = tmp_path / "fresh.json"
    path.write_text(json.dumps(record))
    logs = ["a", "b", "c", "d", "e", "f"]
    assert nd.load_fresh(path, logs, _filters()) == record
    with pytest.raises(SystemExit, match="trainval logs on disk changed"):
        nd.load_fresh(path, logs + ["g"], _filters())
    changed = _filters()
    changed["navmini"]["sha256"] = "s-other"
    with pytest.raises(SystemExit, match="scene filters changed"):
        nd.load_fresh(path, logs, changed)
    edited = dict(record, logs=["d", "f", "c"], windows={"d": 2, "f": 1, "c": 1})
    path.write_text(json.dumps(edited))
    with pytest.raises(SystemExit, match="not a candidate"):
        nd.load_fresh(path, logs, _filters())
    edited = dict(record, n_tokens=4)
    path.write_text(json.dumps(edited))
    with pytest.raises(SystemExit, match="do not add up"):
        nd.load_fresh(path, logs, _filters())


def test_a_log_has_one_city():
    def frames(city):
        return [{"map_location": city} for _ in range(nd.HISTORY + nd.FUTURE)]

    assert nd.city_of({"t1": frames("us-pa-pittsburgh-hazelwood")}) == "us-pa-pittsburgh-hazelwood"
    assert nd.city_of({}) is None
    with pytest.raises(SystemExit, match="name"):
        nd.city_of({"t1": frames("us-ma-boston"), "t2": frames("sg-one-north")})


def test_the_subsets_read_the_labels():
    # Row 0: a vehicle 20 m ahead in lane, the ego slowing. Row 1: a pedestrian 8 m to the left of the
    # proposals' first poses, the ego steady. Row 2: nothing near, the ego stopped.
    samples = []
    for cls, xy in ((0, (20.0, 0.5)), (1, (5.0, 8.0)), (0, (40.0, 20.0))):
        box = np.zeros((1, 7), np.float32)
        box[0, :2] = xy
        samples.append(
            {
                "cls": np.array([cls], np.int8),
                "name": np.zeros(1, np.int8),
                "box": box,
                "velocity": np.zeros((1, 2), np.float32),
                "path": np.zeros((1, 2, 2), np.float32),
                "path_ok": np.ones((1, 2), bool),
                "size_change": np.zeros((1, 2), np.float32),
                "track": np.zeros(1, np.int64),
                "ego_future": np.zeros((8, 3), np.float32),
                "ego_speed": np.zeros(5, np.float32),
            }
        )
    labels = nd.pack(samples)
    labels["token"] = np.array(["r0", "r1", "r2"])
    labels["ego_speed"] = np.array([[8, 7, 6, 5, 4], [8, 8, 8, 8, 8], [0.1] * 5], np.float32)
    proposals = np.zeros((3, 32, 8, 3), np.float32)
    proposals[..., 0] = np.arange(1, 9)  # straight ahead, 1 m per pose
    got = nd.subsets(labels, proposals)
    assert got["lead_30m"].tolist() == [True, False, False]
    assert got["agent_within_10m"].tolist() == [False, True, False]
    assert got["slowing_or_stopped"].tolist() == [True, False, True]


@pytest.mark.parametrize("which", ["navtest", "fresh"])
def test_the_committed_lists_add_up(which):
    path = nd.REPO / nd.LISTS[which]
    if not path.exists():
        pytest.skip(f"{path.name} is written by `navsim_data {which}-list` at B-0-1")
    record = json.loads(path.read_text())
    assert record["logs"] == sorted(record["logs"])
    if which == "navtest":
        assert (len(record["logs"]), record["n_tokens"]) == (136, 12146)
    else:
        assert sum(record["windows"].values()) == record["n_tokens"]
        assert sorted(record["windows"]) == record["logs"] == sorted(record["cities"])
        per_city: dict = {}
        for log, city in record["cities"].items():
            per_city[city] = per_city.get(city, 0) + record["windows"][log]
        assert per_city == record["windows_per_city"]


def _one_agent_labels(path_ok):
    """One row, one vehicle 10 m ahead moving 4 m/s forward, whose track lasts per `path_ok`."""
    box = np.zeros((1, 7), np.float32)
    box[0, :6] = [10.0, 0.0, 0.5, 4.0, 2.0, 1.5]
    sample = {
        "cls": np.array([0], np.int8),
        "name": np.zeros(1, np.int8),
        "box": box,
        "velocity": np.array([[4.0, 0.0]], np.float32),
        "path": np.array([[[12.1, 0.0], [14.3, 0.0]]], np.float32),
        "path_ok": np.array([path_ok]),
        "size_change": np.zeros((1, 2), np.float32),
        "track": np.zeros(1, np.int64),
        "ego_future": np.zeros((8, 3), np.float32),
        "ego_speed": np.zeros(5, np.float32),
    }
    return nd.pack([sample])


def _front_projection():
    """A camera at the ego origin looking along x: everything ahead is in view."""
    proj = np.zeros((1, 4, 4))
    proj[0, 0] = [256.0, -256.0, 0, 0]  # u = 256 (1 - y / x)
    proj[0, 1] = [128.0, 0, -128.0, 0]  # v = 128 (1 - z / x)
    proj[0, 2] = [1.0, 0, 0, 0]
    proj[0, 3] = [0, 0, 0, 1.0]
    return proj


def test_truth_agents_fill_a_lost_track_at_constant_velocity():
    from gallops.drivejepa_perception import targets as tg

    labels = _one_agent_labels([True, False])
    zero = tg.truth_agents(labels, _front_projection(), fill="zero")
    cv = tg.truth_agents(labels, _front_projection(), fill="constant_velocity")
    assert zero["mask"][0, 0] and zero["path_ok"][0, 0].tolist() == [True, False]
    assert np.allclose(zero["path"][0, 0], [[2.1, 0.0], [0.0, 0.0]])  # the first pass: stands still
    assert np.allclose(cv["path"][0, 0], [[2.1, 0.0], [4.1, 0.0]])  # 2.1 m, then 0.5 s at 4 m/s
    lost = tg.truth_agents(_one_agent_labels([False, False]), _front_projection(), fill="constant_velocity")
    assert np.allclose(lost["path"][0, 0], [[2.0, 0.0], [4.0, 0.0]])
    assert not cv["path"][0, 1:].any()  # padded slots stay zero
    with pytest.raises(ValueError):
        tg.truth_agents(labels, _front_projection(), fill="other")
