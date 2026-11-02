"""The moment, chain, graded and dynamics rules of `gallops.memory.ticks`, on synthetic logs."""

from __future__ import annotations

import numpy as np

from gallops.compat import zip_strict
from gallops.memory import ticks

CAMS = ticks.CURRENT_CAMERAS


def frame(i, t_us=None, vx=10.0, ax=0.0, token=None):
    return {
        "log_name": "log",
        "token": token or f"tok{i}",
        "timestamp": t_us if t_us is not None else i * 500_000,
        "cams": {c: {"data_path": f"log/{c}/img{i}_{c}.jpg"} for c in CAMS},
        "ego_dynamic_state": [vx, 0.0, ax, 0.0],
    }


def listing(frames, which, cams=CAMS):
    """One root's listing holding the images of the frames in `which`, for `cams`."""
    return {c: ({f"img{i}_{c}.jpg" for i in which} if c in cams else None) for c in CAMS}


def test_a_chain_s_first_frame_is_not_a_moment():
    frames = [frame(i) for i in range(5)]
    moments = ticks.log_moments(frames, [listing(frames, range(5))], graded=set())
    assert [m["frame_idx"] for m in moments] == [1, 2, 3, 4]
    assert [m["chain_pos"] for m in moments] == [1, 2, 3, 4]
    assert {m["chain"] for m in moments} == {0}


def test_a_missing_image_breaks_the_chain():
    frames = [frame(i) for i in range(7)]
    moments = ticks.log_moments(frames, [listing(frames, [0, 1, 2, 4, 5, 6])], graded=set())
    assert [(m["frame_idx"], m["chain"], m["chain_pos"]) for m in moments] == [
        (1, 0, 1),
        (2, 0, 2),
        (5, 1, 1),
        (6, 1, 2),
    ]


def test_a_gap_over_055_s_breaks_the_chain():
    times = [0, 500_000, 1_000_000, 1_600_000, 2_100_000]
    frames = [frame(i, t) for i, t in enumerate(times)]
    moments = ticks.log_moments(frames, [listing(frames, range(5))], graded=set())
    assert [(m["frame_idx"], m["chain"]) for m in moments] == [(1, 0), (2, 0), (4, 1)]


def test_second_root_fills_in_and_is_recorded():
    frames = [frame(i) for i in range(4)]
    ours, theirs = listing(frames, [0, 1]), listing(frames, [2, 3])
    moments = ticks.log_moments(frames, [ours, theirs], graded=set())
    assert [(m["frame_idx"], m["root_prev"], m["root_now"]) for m in moments] == [
        (1, 0, 0),
        (2, 0, 1),
        (3, 1, 1),
    ]


def test_ours_wins_when_both_roots_have_the_image():
    frames = [frame(i) for i in range(3)]
    moments = ticks.log_moments(frames, [listing(frames, range(3)), listing(frames, range(3))], graded=set())
    assert all(m["root_now"] == 0 and m["root_prev"] == 0 for m in moments)


def test_inputs_ok_needs_all_four_current_cameras():
    frames = [frame(i) for i in range(3)]
    front_only = listing(frames, range(3), cams=("CAM_F0",))
    moments = ticks.log_moments(frames, [front_only], graded=set())
    assert [m["inputs_ok"] for m in moments] == [False, False]
    full = ticks.log_moments(frames, [listing(frames, range(3))], graded=set())
    assert [m["inputs_ok"] for m in full] == [True, True]


def test_graded_is_membership_of_the_metric_cache():
    frames = [frame(i) for i in range(4)]
    moments = ticks.log_moments(frames, [listing(frames, range(4))], graded={"tok2"})
    assert [m["graded"] for m in moments] == [False, True, False]


def test_history_is_four_frames_clamped_at_the_log_start():
    assert ticks.history_indices(10) == [7, 8, 9, 10]
    assert ticks.history_indices(1) == [0, 0, 0, 1]
    assert ticks.history_indices(0) == [0, 0, 0, 0]


def test_dynamics_event_and_onset():
    # steady 10 m/s, then a 2 m/s drop with hard braking at frame 6
    vx = [10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8]
    ax = [0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0]
    frames = [frame(i, vx=v, ax=a) for i, (v, a) in enumerate(zip_strict(vx, ax))]
    d = ticks.dynamics(frames)
    steps = int(ticks.HORIZON_S / ticks.FRAME_DT_S)
    assert d["judged"].tolist() == [True] * (len(frames) - steps) + [False] * steps
    # frames 2..5 see the drop within 2 s; frame 6 is braking itself
    assert np.flatnonzero(d["event"]).tolist() == [2, 3, 4, 5]
    assert np.flatnonzero(d["onset"]).tolist() == [2, 3, 4, 5]
    assert d["braking"][6] and not d["onset"][6]


def test_chain_histogram_counts_long_chains():
    rows = []
    for chain, length in ((0, 3), (1, 9), (2, 17)):
        for pos in range(1, length):
            rows.append({"log": "l", "chain": chain, "chain_pos": pos, "graded": pos % 2 == 0})
    h = ticks.chain_histogram(rows)
    assert h["chains"] == 3
    assert h["length_counts"] == {"3": 1, "9": 1, "17": 1}
    assert h["moments_with_8_frames"] == 2 + 10  # positions 7-8 of chain 1, 7-16 of chain 2
    assert h["moments_with_16_frames"] == 2


def test_log_split_and_graded_tokens(tmp_path):
    split = tmp_path / "split.yaml"
    split.write_text("train_logs:\n  - a\n  - b\nval_logs:\n  - c\ntest_logs:\n  - d\nother:\n  - e\n")
    assert ticks.log_split(split) == {"a": "train", "b": "train", "c": "val", "d": "test"}
    csv = tmp_path / "meta.csv"
    csv.write_text("file_name\n/r/logA/unknown/t1/metric_cache.pkl\n/r/logA/unknown/t2/metric_cache.pkl\n")
    assert ticks.graded_tokens(csv) == {"logA": {"t1", "t2"}}


def _log(events, judged=100, onsets=5, city="a", vehicle="v"):
    return {
        "events": events,
        "judged": judged,
        "onsets": onsets,
        "not_braking": judged,
        "graded_tokens": judged,
        "speed_sum": 10.0 * judged,
        "city": city,
        "vehicle": vehicle,
    }


def test_compare_groups_flags_a_large_difference():
    per_log = {f"m{i}": _log(60) for i in range(20)} | {f"k{i}": _log(30) for i in range(40)}
    out = ticks.compare_groups(per_log, {f"m{i}" for i in range(20)}, reps=200)
    m = out["metrics"]["event_rate"]
    assert (m["missing"], m["kept"]) == (0.6, 0.3)
    assert m["relative"] == 1.0 and m["ci_excludes_zero"]
    assert out["flag"] == "DIFFERENT"


def test_compare_groups_similar_when_rates_match():
    per_log = {f"m{i}": _log(30 + i % 3) for i in range(20)} | {f"k{i}": _log(30 + i % 3) for i in range(40)}
    out = ticks.compare_groups(per_log, {f"m{i}" for i in range(20)}, reps=200)
    assert out["flag"] == "SIMILAR"
    assert out["city_share"]["missing"] == {"a": 1.0}


def test_ours_only_positions_restart_at_second_root_frames():
    table = {
        "log": np.asarray(["l"] * 5),
        "frame_idx": np.asarray([1, 2, 3, 4, 5]),
        "root_now": np.asarray([0, 0, 1, 0, 0]),
        "root_prev": np.asarray([0, 0, 0, 1, 0]),
    }
    assert ticks.ours_only_positions(table).tolist() == [1, 2, -1, -1, 1]
