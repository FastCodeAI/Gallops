"""The training preflight catches what it was written for, and passes what is right.

Its grading check is the one that matters: it has to fail on the v1-v4 bug -- a
referee that caches the ego pose per episode -- and pass the fixed referee, on
the same labels. These build a small episode on disk with the files collection
and labelling leave behind, so they run where torch is absent.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from gallops.expert.grader import first_hit, grade
from gallops.train import preflight
from gallops.train.swerve import Referee
from gallops.train.swerve_data import episode_context

STAMPS = [1_000_000, 2_000_000, 3_000_000, 4_000_000]
EGO_X = [0.0, 6.0, 12.0, 15.0]


def plan(speed: float) -> np.ndarray:
    t = np.arange(1, 9) * 0.5
    return np.column_stack([speed * t, np.zeros_like(t), np.zeros_like(t)])


def build_episode(root):
    """Four ticks closing on a parked car at x = 22 m, plus the matching dump and labels."""
    episode = root / "collect" / "stationary-0099" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    poses = {}
    for i, stamp in enumerate(STAMPS):
        x = EGO_X[i]
        pose = np.eye(4)
        pose[0, 3] = x
        poses[str(stamp)] = pose.tolist()
    car = np.eye(4)
    car[0, 3] = 22.0
    actor = {
        "timestamps": [0, 20_000_000],
        "poses": [car.tolist(), car.tolist()],
        "dims": [1.9, 4.5, 1.6],
        "is_target_actor": True,
    }
    (episode / "ego_poses.json").write_text(json.dumps(poses))
    (episode / "actors.json").write_text(json.dumps([actor]))
    # the logged path runs straight down x, so nothing here leaves its lane
    (episode / "reference_trajectory.json").write_text(
        json.dumps({str(i): [float(i), 0.0, 0.0] for i in range(0, 40)})
    )
    (episode / "metrics.json").write_text("{}")

    speeds = np.linspace(0.5, 4.0, 32)
    driven = plan(2.0)
    rows = [
        {"timestamp": s, "trajectory": driven.tolist(), "proposals": [plan(v).tolist() for v in speeds]}
        for s in STAMPS
    ]
    # the engine's own record: six (x, y) waypoints per tick
    (episode / "trajectories.json").write_text(json.dumps({str(s): driven[:6, :2].tolist() for s in STAMPS}))
    dumps = root / "dumps"
    dumps.mkdir()
    (dumps / "ep_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    reference = np.asarray([[float(i), 0.0, 0.0] for i in range(0, 40)])
    ticks = []
    for i, stamp in enumerate(STAMPS):
        row = rows[i]
        tick = {
            "episode": "stationary-0099/run_0",
            "timestamp": stamp,
            "frame": str(episode / "CAM_FRONT" / f"{stamp}.jpg"),
            "speed": 2.0,
            "expert_kind": "brake",
            "expert": plan(0.5).tolist(),
            "driven_hits": None,
        }
        ego2world, targets, area = episode_context(tick)
        graded = [grade(np.asarray(p), ego2world, stamp, targets, reference, area) for p in row["proposals"]]
        tick["candidate_clear"] = [v.clear for v in graded]
        tick["candidate_hits"] = [v.hit_at for v in graded]
        tick["driven_hits"] = first_hit(driven, ego2world, stamp, targets)
        tick["expert_clear"] = grade(plan(0.5), ego2world, stamp, targets, reference, area).clear
        ticks.append(tick)
    return ticks, dumps


class CachedPerEpisode(Referee):
    """The v1-v4 referee: the first tick's context, pose included, for the whole episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = {}

    def context(self, tick):
        key = tick["frame"].rsplit("/", 2)[0]
        if key not in self._cached:
            self._cached[key] = episode_context(tick, self.drivable)
        return self._cached[key]


def test_the_episode_actually_separates_the_two_referees(tmp_path):
    """Guard on the fixture: the late ticks must crash where the early ones do not."""
    ticks, _ = build_episode(tmp_path)
    assert ticks[0]["driven_hits"] is None
    assert ticks[-1]["driven_hits"] is not None
    assert not all(ticks[-1]["candidate_clear"]) and any(ticks[-1]["candidate_clear"])


def test_grading_passes_the_fixed_referee(tmp_path):
    ticks, dumps = build_episode(tmp_path)
    report, problems = preflight.grading_checks(
        preflight.sample_ticks(ticks), Referee(anchors=np.zeros((1, 8, 3))), dumps
    )
    assert problems == [], problems
    assert report["candidates_reproduced"] == f"{len(ticks)}/{len(ticks)}"


def test_grading_fails_the_referee_that_caches_the_pose(tmp_path):
    ticks, dumps = build_episode(tmp_path)
    _, problems = preflight.grading_checks(
        preflight.sample_ticks(ticks), CachedPerEpisode(anchors=np.zeros((1, 8, 3))), dumps
    )
    assert any("disagrees with labelling" in p for p in problems), problems


def test_the_sample_visits_each_episode_late_tick_first():
    ticks = [{"episode": f"e{e}", "timestamp": t} for e in range(3) for t in range(20)]
    sample = preflight.sample_ticks(ticks, episodes=3, per_episode=4)
    assert len({t["episode"] for t in sample}) == 3
    for episode in {t["episode"] for t in sample}:
        stamps = [t["timestamp"] for t in sample if t["episode"] == episode]
        assert stamps == sorted(stamps, reverse=True) and len(stamps) == 4


def _two_rounds(tmp_path):
    """Round 0 and a DAgger round of the same scenario: same episode name, same stamps."""
    r0, d0 = build_episode(tmp_path / "round0")
    r1, d1 = build_episode(tmp_path / "round1")
    for tick in r1:
        tick["source"] = 1
    for tick in r0:
        tick["source"] = 0
    return r0 + r1, d0, d1


def test_two_rounds_of_one_scenario_stay_two_episodes(tmp_path):
    ticks, _, _ = _two_rounds(tmp_path)
    assert len({t["episode"] for t in ticks}) == 1  # the name alone would merge them
    sample = preflight.sample_ticks(ticks, episodes=2, per_episode=4)
    assert len({preflight.episode_key(t) for t in sample}) == 2
    assert preflight.describe(ticks)["episodes"] == 2


def test_grading_reads_each_sources_own_dumps(tmp_path):
    ticks, d0, d1 = _two_rounds(tmp_path)
    referee = Referee(anchors=np.zeros((1, 8, 3)))
    sources = [
        preflight.Source("r0", str(d0), preflight.START),
        preflight.Source("r1", str(d1), "later-checkpoint"),
    ]
    report, problems = preflight.grading_checks(ticks, referee, sources)
    assert problems == [] and report["graded"] == 8, problems
    empty = tmp_path / "empty"
    empty.mkdir()
    sources[1] = preflight.Source("r1", str(empty), "later-checkpoint")
    report, problems = preflight.grading_checks(ticks, referee, sources)
    assert report["no_dump"] == 4 and report["graded"] == 4


def test_no_source_from_the_starting_weights_is_refused(tmp_path, monkeypatch):
    """Fidelity is the only check on the rebuilt inputs; a run cannot skip it entirely."""
    ticks, _, d1 = _two_rounds(tmp_path)
    ticks = [t for t in ticks if t["source"] == 1]
    for tick in ticks:
        tick["source"] = 0
        tick["expert_kind"] = "brake"
    monkeypatch.setattr(preflight, "code_checks", lambda repo: ({}, []))
    sources = [preflight.Source("r1", str(d1), "later-checkpoint")]
    with pytest.raises(preflight.PreflightFailed) as failed:
        preflight.run(tmp_path, ticks, Referee(anchors=np.zeros((1, 8, 3))), sources, None, None, None, "cpu")
    assert "no source was driven by the starting weights" in str(failed.value)


def test_code_checks_refuse_uncommitted_source(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "thing.py").write_text("X = 1\n")
    (repo / "tests" / "test_thing.py").write_text("def test_ok():\n    assert True\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)

    report, problems = preflight.code_checks(repo)
    assert problems == [], problems
    assert report["git_clean"]

    (repo / "src" / "thing.py").write_text("X = 2\n")
    _, problems = preflight.code_checks(repo)
    assert any("uncommitted" in p for p in problems), problems


def test_a_failure_lists_every_problem():
    error = preflight.PreflightFailed(["one", "two"])
    assert "one" in str(error) and "two" in str(error)
    with pytest.raises(preflight.PreflightFailed):
        raise error


def test_a_tick_whose_past_was_a_priming_frame_is_not_rebuilt():
    """The model reads one past frame, 0.5 s back. Early in an episode that was a
    priming frame the logger never saved, and nearest-stamp selection would hand
    over the episode's first frame instead -- measured 0.4-0.5 m of pick error
    against 0.10 m once the real frame exists. Such a tick is refused, not guessed.
    """
    from gallops.train.swerve_data import history_stamps

    frames = [1_000_000 + 100_000 * i for i in range(30)]  # 10 Hz from t = 1.0 s
    assert history_stamps(frames, now=frames[2]) is None  # 0.5 s back is before the first frame
    assert history_stamps(frames, now=frames[4]) is None  # nearest is the first frame, 0.1 s off
    assert history_stamps(frames, now=frames[5]) is not None  # the first frame IS 0.5 s back
    chosen = history_stamps(frames, now=frames[10])
    assert chosen is not None
    assert chosen[-1] == frames[10] and chosen[-2] == frames[5]


def test_a_round_the_init_weights_drove_is_checked_against_them(tmp_path, monkeypatch):
    """INIT sources skip the start-of-run fidelity check and get their own once --init loads."""
    ticks, d0, d1 = _two_rounds(tmp_path)
    sources = [
        preflight.Source("r0", str(d0), preflight.START),
        preflight.Source("r1", str(d1), preflight.INIT),
    ]
    seen = []

    def fidelity(model, dataset_cls, sample, referee, dumps, torch, device):
        seen.append({t["source"] for t in sample})
        return {"compared": len(sample)}, []

    monkeypatch.setattr(preflight, "fidelity_check", fidelity)
    report = preflight.init_fidelity(ticks, None, sources, None, None, None, "cpu")
    assert seen == [{1}] and report["compared"] > 0
    assert (
        preflight.init_fidelity(
            [t for t in ticks if t["source"] == 0], None, sources, None, None, None, "cpu"
        )
        == {}
    )
    monkeypatch.setattr(preflight, "fidelity_check", lambda *a: ({}, ["fidelity: off"]))
    with pytest.raises(preflight.PreflightFailed):
        preflight.init_fidelity(ticks, None, sources, None, None, None, "cpu")
