"""teacher_scenes.py's turn/curve/straight classification and scene selection."""

from __future__ import annotations

from gallops.data.teacher_scenes import (
    HORIZON_SAMPLES,
    SceneMetrics,
    _overlaps_chosen,
    _turn_classification_flags,
    select_scenes,
)


def test_tail_samples_are_excluded_not_folded_into_straight() -> None:
    """Regression test for the horizon-truncation bug: a scene with a constant
    lateral drift throughout (a sustained, uninterrupted left turn from start
    to end) must not read any of its tail samples as "straight" just because
    they have less than a full HORIZON_SAMPLES lookahead remaining.

    Before the fix, `_score_scene` computed `j = min(i + HORIZON_SAMPLES, n - 1)`
    for every sample, silently shrinking the lookahead window near the end of
    the scene instead of excluding those samples. For this fixture (constant
    0.5 m/sample lateral drift, heading fixed at 0 so ego frame == world
    frame), the shrunk window at i=8 (1 sample of true lookahead instead of 6)
    measured a lateral offset of only 0.5 m -- read as "straight" even though
    the car never stopped turning. The fix excludes i>=n_valid entirely.
    """
    n = 10
    positions = [(float(k), 0.5 * k) for k in range(n)]
    headings = [0.0] * n

    left, right, curve, straight, n_valid = _turn_classification_flags(positions, headings)

    assert n_valid == n - HORIZON_SAMPLES == 4
    # Every sample with a full lookahead sees the same constant drift: a
    # sustained left turn (0.5 * 6 = 3.0 m >= TURN_THRESHOLD_M).
    assert left[:4] == [True, True, True, True]
    assert right[:4] == [False, False, False, False]
    assert curve[:4] == [False, False, False, False]
    assert straight[:4] == [False, False, False, False]
    # Every tail sample (no full lookahead) is excluded from every category --
    # none of them reads as "straight", "curve", or "turn".
    for i in range(4, n):
        assert not left[i] and not right[i] and not curve[i] and not straight[i], (
            f"sample {i} should be excluded entirely, not classified"
        )
    # So straight_fraction (sum(straight) / max(n_valid, 1)) is 0.0, not 0.2 as
    # the pre-fix code would have computed (2 of 10 raw samples read straight:
    # the shrunk-window sample above, plus the single explicitly-hardcoded
    # last-sample special case the old code carried).
    assert sum(straight) / max(n_valid, 1) == 0.0


def test_full_horizon_samples_detect_a_real_turn() -> None:
    n = 10
    positions = [(float(k), 0.5 * k) for k in range(n)]
    headings = [0.0] * n
    left, _, _, _, n_valid = _turn_classification_flags(positions, headings)
    assert all(left[:n_valid])


def test_straight_scene_has_no_turn_flags() -> None:
    n = 10
    positions = [(float(k), 0.0) for k in range(n)]
    headings = [0.0] * n
    left, right, curve, straight, n_valid = _turn_classification_flags(positions, headings)
    assert not any(left[:n_valid]) and not any(right[:n_valid]) and not any(curve[:n_valid])
    assert all(straight[:n_valid])


def test_overlaps_chosen_rejects_nearby_tracks() -> None:
    candidate = [(0.0, 0.0), (10.0, 0.0)]
    chosen = [[(5.0, 5.0), (15.0, 5.0)]]  # within SAME_LOCATION_DIST_M (25 m) of the candidate
    assert _overlaps_chosen(candidate, chosen)

    far = [[(500.0, 500.0), (510.0, 500.0)]]
    assert not _overlaps_chosen(candidate, far)


def _metric(name: str, log_token: str, category: str, good_actor_count: int = 1) -> SceneMetrics:
    return SceneMetrics(
        name=name,
        scene_token=name,
        log_token=log_token,
        location="boston-seaport",
        description="",
        skipped_reason="",
        n_samples=20,
        n_valid_samples=14,
        longest_left_run=5 if category == "turn_left" else 0,
        longest_right_run=5 if category == "turn_right" else 0,
        longest_curve_run=6 if category == "curved_road" else 0,
        straight_fraction=0.9 if category == "straight" else 0.0,
        total_yaw_deg=0.0,
        has_good_actor=True,
        good_actor_count=good_actor_count,
        category=category,
    )


def test_select_scenes_never_picks_two_scenes_from_the_same_log() -> None:
    metrics = [
        _metric("scene-0001", "log-a", "straight", good_actor_count=5),
        _metric("scene-0002", "log-a", "straight", good_actor_count=1),
    ]
    positions = {"scene-0001": [(0.0, 0.0), (1.0, 0.0)], "scene-0002": [(500.0, 500.0), (501.0, 500.0)]}

    train, _val = select_scenes(metrics, positions)

    names = {m.name for m in train}
    assert "scene-0001" in names
    assert "scene-0002" not in names, "same log_token must never contribute a second scene"
