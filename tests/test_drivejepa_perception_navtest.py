"""Part B's official-scorer plumbing (navtest.py), the parts that need neither navsim nor torch.

H04's arithmetic is ported with its tests (branch drivejepa-wajepa/run3a, e5ca907,
tests/test_drivejepa_wajepa.py): the numbers it gave are the ones the 87.78 anchor is held to.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gallops.drivejepa_perception import navtest

CSV_COLS = ["token", "valid", "score", *navtest.GATES, *(c for c, _ in navtest.SOFT)]


def write_csv(path: Path, rows: list) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def row(token: str, gates: tuple, soft: tuple, score: float, valid: str = "True") -> dict:
    r = {"token": token, "valid": valid, "score": score}
    assert len(gates) == len(navtest.GATES) and len(soft) == len(navtest.SOFT)
    r.update(dict(zip(navtest.GATES, gates)))  # noqa: B905 -- Python 3.9
    r.update(dict(zip((c for c, _ in navtest.SOFT), soft)))  # noqa: B905 -- Python 3.9
    return r


def test_navtest_score_recombines_the_human_filtered_columns(tmp_path):
    rows = [
        # all soft terms: (5*0.8 + 5*1 + 2*1 + 2*0.5 + 2*1) / 16 = 0.875
        row("a", (1, 1, 1, 1), (0.8, 1, 1, 0.5, 1), 0.9),
        # EC blank: left out of this scene's mean -> (5*1 + 5*1 + 2*1 + 2*1) / 14 = 1; one gate halves it
        row("b", (1, 0.5, 1, 1), (1, 1, 1, 1, ""), 0.7),
        row("c", (0, 0, 0, 0), (0, 0, 0, 0, 0), 0.0, valid="False"),  # invalid: dropped
        row("average_all_frames", (1, 1, 1, 1), (1, 1, 1, 1, 1), 1.0),  # summary row: dropped
    ]
    s = navtest.score(str(write_csv(tmp_path / "x.csv", rows)))
    assert s["scenes"] == 2
    assert s["epdms_human_filter"] == pytest.approx(100 * (0.875 + 0.5) / 2)
    assert s["csv_score"] == pytest.approx(80.0)
    assert s["sub_scores"]["EC"] == pytest.approx(1.0)  # the blank is not a zero
    tokens = navtest.per_token(str(tmp_path / "x.csv"))
    assert set(tokens) == {"a", "b"}
    assert tokens["a"]["epdms_hf"] == pytest.approx(0.875) and tokens["b"]["csv"] == pytest.approx(0.7)


def test_navtest_score_reads_the_latest_run_of_a_glob(tmp_path):
    for stamp in ("2026.10.17.13.45.48", "2026.10.18.09.00.00"):
        (tmp_path / stamp).mkdir()
        write_csv(tmp_path / stamp / "r.csv", [row("a", (1, 1, 1, 1), (1, 1, 1, 1, 1), 1.0)])
    assert navtest.resolve(str(tmp_path / "*" / "*.csv")).endswith("2026.10.18.09.00.00/r.csv")
    with pytest.raises(SystemExit, match="no CSV"):
        navtest.resolve(str(tmp_path / "none" / "*.csv"))


def test_a_plan_store_round_trips_and_refuses_bad_plans(tmp_path):
    poses = np.random.default_rng(0).normal(size=(3, 8, 3)).astype(np.float32)
    out = navtest.write_plans(
        tmp_path / "R", ["t1", "t2", "t3"], poses, {"arm": "R"}, pick=np.array([0, 5, 31])
    )
    plans = navtest.load_plans(out)
    assert set(plans) == {"t1", "t2", "t3"} and np.array_equal(plans["t2"], poses[1].astype(np.float64))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rows"] == 3 and manifest["arm"] == "R"
    with pytest.raises(SystemExit, match="exists"):
        navtest.write_plans(out, ["t1"], poses[:1], {})
    bad = poses.copy()
    bad[1, 4, 0] = np.nan
    with pytest.raises(SystemExit, match="not finite"):
        navtest.write_plans(tmp_path / "S", ["t1", "t2", "t3"], bad, {})


def test_coverage_names_every_way_a_replay_would_fail():
    poses = np.zeros((3, 8, 3))
    assert navtest.coverage_problems({"token": ["a", "b", "c"], "poses": poses}, {"a", "b", "c"}) == []
    problems = navtest.coverage_problems({"token": ["a", "b", "b"], "poses": poses}, {"a", "b", "c"})
    assert any("repeat" in p for p in problems) and any("no plan" in p for p in problems)
    problems = navtest.coverage_problems({"token": ["a"], "poses": np.zeros((1, 6, 3))}, {"a"})
    assert any("shape" in p for p in problems)


def _cache(root: Path, entries: dict, prefix=None) -> Path:
    """A metric cache's layout: one file per token, and the metadata CSV naming them."""
    (root / "metadata").mkdir(parents=True)
    lines = ["file_name"]
    for token, log in entries.items():
        path = navtest.expected_path(root, log, token)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        lines.append(str(path) if prefix is None else str(path).replace(str(root), prefix, 1))
    (root / "metadata" / f"{root.name}_metadata_node_0.csv").write_text("\n".join(lines) + "\n")
    return root


def test_cache_coverage_and_identity(tmp_path):
    root = _cache(tmp_path / "cache", {"t1": "log1", "t2": "log1", "t3": "log2"})
    assert navtest.coverage_of(root, {"t1": "log1", "t2": "log1", "t3": "log2"}) == []
    identity = navtest.cache_identity(root)
    assert (identity["tokens"], identity["logs"]) == (3, 2) and len(identity["csv_sha256"]) == 64
    problems = navtest.coverage_of(root, {"t1": "log1", "t2": "log2", "t4": "log2"})
    assert any("no entry" in p for p in problems) and any("unknown" in p for p in problems)
    navtest.expected_path(root, "log1", "t1").unlink()
    assert any(
        "do not exist" in p for p in navtest.coverage_of(root, {"t1": "log1", "t2": "log1", "t3": "log2"})
    )


def test_a_staged_cache_has_its_csv_moved_to_the_final_root(tmp_path):
    staging = _cache(tmp_path / "c.staging", {"t1": "log1"})
    final = str(tmp_path / "c")
    assert navtest.rewrite_csv_root(navtest.metadata_csv(staging), str(staging), final) == 1
    assert navtest.cache_paths(staging) == {"t1": str(navtest.expected_path(Path(final), "log1", "t1"))}
    with pytest.raises(SystemExit, match="not under"):
        navtest.rewrite_csv_root(navtest.metadata_csv(staging), str(staging), final)


def test_the_fixed_plans_exercise_the_cache():
    human = np.stack([np.arange(1, 9) * 4.0, np.zeros(8), np.zeros(8)], axis=1)
    plans = navtest.fixed_plans(human, speed=6.0)
    assert plans.shape == (8, 8, 3)
    assert np.array_equal(plans[0], human) and plans[1][-1, 0] == pytest.approx(24.0)
    assert not plans[2].any()  # the stop
    assert plans[3][0, 1] == -1.5 and plans[6][0, 1] == 1.5
    assert plans[7][-1, 0] == pytest.approx(0.8 * 32.0)


def test_the_fresh_scene_filter_is_the_list_and_navtrains_window():
    record = {"logs": ["b", "a"]}
    filt = navtest.fresh_scene_filter(record)
    assert filt["log_names"] == ["b", "a"] and filt["tokens"] is None and filt["has_route"] is True
    assert (filt["num_history_frames"], filt["num_future_frames"], filt["frame_interval"]) == (4, 10, 1)
