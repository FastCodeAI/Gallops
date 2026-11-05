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
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert report["sha"] == head, "the sha every sidecar of the run records is taken here"

    (repo / "src" / "thing.py").write_text("X = 2\n")
    report, problems = preflight.code_checks(repo)
    assert any("uncommitted" in p for p in problems), problems
    assert report["sha"] == head + "-dirty"


def test_a_commit_during_the_run_does_not_change_the_recorded_sha(tmp_path):
    """v4fix's sidecar named 85e1f9c, committed mid-run; c18f488 had trained it."""
    from gallops.train.swerve import git_sha, head_moved

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "thing.py").write_text("X = 1\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "one"], check=True)
    started = git_sha(repo)
    assert head_moved(started, repo) == {}
    (repo / "src" / "thing.py").write_text("X = 2\n")
    subprocess.run([*git, "commit", "-qam", "two"], check=True)
    moved = head_moved(started, repo)
    assert moved["git_head_at_save"] == git_sha(repo) != started


def test_a_second_fit_into_an_existing_run_is_refused(tmp_path):
    from gallops.config import SiteConfig
    from gallops.train.swerve import MODEL, fresh_run

    cfg = SiteConfig(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    run = fresh_run(cfg, "smoke")
    assert run == cfg.checkpoint_dir(MODEL, "smoke", 1).parent and not run.exists()
    run.mkdir(parents=True)
    assert fresh_run(cfg, "smoke") == run  # an empty directory is not a fit
    (run / "metrics.jsonl").write_text("")
    with pytest.raises(SystemExit, match="already holds a fit"):
        fresh_run(cfg, "smoke")


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


class _Plan:
    """What `model(features)["trajectory"]` returns, minus torch."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float64)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _NoTorch:
    @staticmethod
    def no_grad():
        from contextlib import nullcontext

        return nullcontext()


class _Ticks:
    """A dataset that batches like the old TickDataset did: one row per tick asked for."""

    def __init__(self, ticks, device):
        self.ticks = ticks

    def batch(self, indices):
        return {"tick": np.asarray(indices)}, None, [self.ticks[i] for i in indices]


class _MixesTicks:
    """Drive-JEPA's batch behaviour in miniature: every element reads element 0's image.

    Alone, each tick drives exactly the recorded plan. Second or later in a batch, its
    plan moves 0.5 m sideways per position, as a neighbour's mask would move it.
    """

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def __call__(self, features):
        ticks = features["tick"]
        plans = np.stack([plan(2.0) for _ in ticks])
        plans[:, :, 1] += 0.5 * (ticks - ticks[0])[:, None]
        return {"trajectory": _Plan(plans)}


def test_fidelity_runs_one_tick_per_pass(tmp_path):
    """Batched, a correct loader fails fidelity on a model that mixes ticks (the
    blocker on #8: 0.00% alone, 1.65% batched, against a 1% limit)."""
    ticks, dumps = build_episode(tmp_path)
    referee = Referee(anchors=np.zeros((1, 8, 3)))
    model = _MixesTicks()
    report, problems = preflight.fidelity_check(model, _Ticks, ticks, referee, dumps, _NoTorch, "cpu")
    assert problems == [], problems
    assert report["compared"] == len(ticks) and report["median_pick_diff_m"] == 0.0
    assert model.training is True


def test_every_pass_is_one_tick():
    from gallops.train.swerve_data import TICKS_PER_PASS, TickDataset, one_tick_per_pass

    assert TICKS_PER_PASS == 1 and one_tick_per_pass("1") == 1
    with pytest.raises(SystemExit, match="mixes the ticks of a batch"):
        one_tick_per_pass("8")
    dataset = object.__new__(TickDataset)  # no torch here; the refusal comes first
    dataset.ticks = [{}, {}]
    with pytest.raises(ValueError, match="2 ticks in one pass"):
        dataset.batch([0, 1])


class _ByStamp:
    """One tick per pass, keyed by timestamp; `missing` stamps cannot be rebuilt."""

    missing: set = set()

    def __init__(self, ticks, device):
        self.ticks = ticks

    def batch(self, indices):
        used = [self.ticks[i] for i in indices if self.ticks[i]["timestamp"] not in self.missing]
        if not used:
            return None, None, []
        return {"stamp": np.asarray([t["timestamp"] for t in used])}, None, used


class _OffOn:
    """Drives the recorded plan, except 10 m sideways on the ticks in `off`."""

    def __init__(self, off):
        self.off, self.training = set(off), True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def __call__(self, features):
        plans = np.stack([plan(2.0) for _ in features["stamp"]])
        for k, stamp in enumerate(features["stamp"]):
            if int(stamp) in self.off:
                plans[k, :, 1] += 10.0
        return {"trajectory": _Plan(plans)}


def test_fidelity_sees_a_minority_of_ticks_rebuilt_wrong(tmp_path):
    """A median cannot: 1 of 4 ticks 10 m off leaves the median at 0."""
    ticks, dumps = build_episode(tmp_path)
    referee = Referee(anchors=np.zeros((1, 8, 3)))
    # the earliest tick: clear either way, so only the gap can flag it
    report, problems = preflight.fidelity_check(
        _OffOn({STAMPS[0]}), _ByStamp, ticks, referee, dumps, _NoTorch, "cpu"
    )
    assert report["median_pick_diff_of_travel"] == 0.0
    assert any("p90" in p for p in problems), problems


def test_fidelity_refuses_a_sample_it_mostly_could_not_rebuild(tmp_path, monkeypatch):
    ticks, dumps = build_episode(tmp_path)
    monkeypatch.setattr(_ByStamp, "missing", set(STAMPS[:2]))
    report, problems = preflight.fidelity_check(
        _OffOn(()), _ByStamp, ticks, Referee(anchors=np.zeros((1, 8, 3))), dumps, _NoTorch, "cpu"
    )
    assert report["skipped_unrebuildable"] == 2
    assert any("could not be rebuilt" in p for p in problems), problems


def _frames_episode(root, n=30):
    """An episode directory with 10 Hz front frames and a pose for every one but the last."""
    episode = root / "stationary-0099" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    stamps = [1_000_000 + 100_000 * i for i in range(n)]
    for stamp in stamps:
        (episode / "CAM_FRONT" / f"{stamp}.jpg").write_bytes(b"")
    (episode / "ego_poses.json").write_text(json.dumps({str(s): np.eye(4).tolist() for s in stamps[:-1]}))
    return episode, stamps


def test_rebuildable_makes_the_tests_the_dataset_makes(tmp_path):
    from gallops.train.swerve_data import rebuildable

    episode, stamps = _frames_episode(tmp_path)

    def tick(i):
        return {"frame": str(episode / "CAM_FRONT" / f"{stamps[i]}.jpg"), "timestamp": stamps[i]}

    assert not rebuildable(tick(4))  # its 0.5 s past would be the first frame, 0.1 s off
    assert rebuildable(tick(5)) and rebuildable(tick(20))
    assert not rebuildable(tick(len(stamps) - 1))  # no pose logged
    assert not rebuildable({"frame": None, "timestamp": stamps[10]})


def test_unrebuildable_ticks_are_counted_by_kind_and_bounded():
    ticks = [{"expert_kind": "brake", "ok": i > 0} for i in range(5)] + [
        {"expert_kind": "continue", "ok": True} for _ in range(15)
    ]
    kept, report = preflight.keep_rebuildable(ticks, lambda t: t["ok"])
    assert len(kept) == 19 and report["dropped"] == 1 and report["by_expert_kind"] == {"brake": "1/5"}
    with pytest.raises(SystemExit, match="cannot be rebuilt"):
        preflight.keep_rebuildable(ticks, lambda t: t["expert_kind"] == "continue")


def test_fidelity_samples_each_episodes_earliest_tick(tmp_path, monkeypatch):
    ticks = [
        {
            "episode": f"e{e}",
            "timestamp": t,
            "source": 0,
            "speed": 2.0,
            "expert_kind": "brake",
            "driven_hits": None,
        }
        for e in range(3)
        for t in range(5, 40)
    ]
    seen = {}

    def fidelity(model, dataset_cls, own, *args):
        seen["stamps"] = sorted(t["timestamp"] for t in own if t["episode"] == "e0")
        return {}, []

    monkeypatch.setattr(preflight, "code_checks", lambda repo: ({"sha": "x"}, []))
    monkeypatch.setattr(preflight, "grading_checks", lambda *a: ({}, []))
    monkeypatch.setattr(preflight, "drivable_checks", lambda *a: ({}, []))
    monkeypatch.setattr(preflight, "fidelity_check", fidelity)
    monkeypatch.setattr(preflight, "sight_check", lambda *a: ({}, []))
    sources = [preflight.Source("r0", "d0", preflight.START)]
    preflight.run(tmp_path, ticks, None, sources, object(), None, None, "cpu")
    assert seen["stamps"][0] == 5, seen
    assert preflight.earliest_per_episode(ticks, {"e1"}) == [ticks[35]]


def test_every_target_is_checked_drivable_not_a_sample():
    """#4 review, finding 11: one undrivable target in 200 escaped a 48-tick sample."""
    straight = plan(2.0).tolist()
    slide = [[0.0, 0.5 * (k + 1), 0.0] for k in range(8)]  # sideways, going nowhere
    ticks = [{"episode": "e", "timestamp": i, "expert": straight} for i in range(200)]
    ticks[137] = {**ticks[137], "expert": slide}
    report, problems = preflight.drivable_checks(ticks)
    assert report["undrivable"] == "1/200" and problems and "@ 137" in problems[0]
    assert preflight.drivable_checks(ticks[:137])[1] == []


def test_a_logged_human_path_is_not_held_to_the_limits_of_a_drawn_target():
    """#17 review, 🔴 1: a human's path is not drawn, and relabelling cannot change it.

    542 of labelled_nusc's 18,015 logged paths fail `undrivable`, so no run with a human
    source could start. A drawn target with the same path is still refused.
    """
    tight = [[0.0, 0.5 * (k + 1), 0.0] for k in range(8)]
    logged = {"episode": "real-0020/run_0", "timestamp": 1, "expert": tight, "expert_kind": "human"}
    report, problems = preflight.drivable_checks([logged])
    assert problems == [] and report == {"undrivable": "0/0", "logged": 1}
    drawn = {**logged, "expert_kind": "brake"}
    assert preflight.drivable_checks([logged, drawn])[0] == {"undrivable": "1/1", "logged": 1}


def test_a_source_called_human_holds_only_logged_driving():
    """#17 review, 🟠 6: `drove = human` skips grading and fidelity, so it has to be earned.

    A rendered round passed as human was never graded or fidelity-checked, and nothing said so.
    """
    sources = [preflight.Source("r0", "d0", preflight.START), preflight.Source("real", "", preflight.HUMAN)]
    good = [{"source": 0, "expert_kind": "brake"}, {"source": 1, "expert_kind": "human"}]
    assert preflight.human_checks(good, sources) == []

    rendered_as_human = good + [{"source": 1, "expert_kind": "brake"}]
    problems = preflight.human_checks(rendered_as_human, sources)
    assert len(problems) == 1 and "real" in problems[0] and "brake" in problems[0]

    # and the exemption cannot be claimed from the other side: a tick calling itself human in
    # a rendered source would skip `drivable_checks`
    claimed = good + [{"source": 0, "expert_kind": "human"}]
    problems = preflight.human_checks(claimed, sources)
    assert len(problems) == 1 and "r0" in problems[0]


def test_a_rendered_round_passed_as_human_stops_the_run(tmp_path, monkeypatch):
    ticks, dumps = build_episode(tmp_path)
    for tick in ticks:
        tick["source"] = 0
    rendered = [dict(t, source=1) for t in ticks]
    monkeypatch.setattr(preflight, "code_checks", lambda repo: ({"sha": "x"}, []))
    monkeypatch.setattr(preflight, "grading_checks", lambda *a: ({}, []))
    monkeypatch.setattr(preflight, "fidelity_check", lambda *a: ({}, []))
    monkeypatch.setattr(preflight, "sight_check", lambda *a: ({}, []))
    sources = [
        preflight.Source("r0", str(dumps), preflight.START),
        preflight.Source("r2", "", preflight.HUMAN),
    ]
    with pytest.raises(preflight.PreflightFailed, match="r2"):
        preflight.run(tmp_path, ticks + rendered, None, sources, object(), None, None, "cpu")
