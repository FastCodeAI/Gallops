"""`swerve.main` from argv to a written checkpoint, with torch and the model stubbed.

`make check` has no torch, so nothing used to reach this function, and it shows: from 14 Oct
every run stopped right after the preflight on a comparison of a git sha with a tick count, and
no test could have said so. What these pin down is the loop and what it writes, not what it learns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gallops import checkpoint as sidecar
from gallops.config import SiteConfig
from gallops.train import preflight, swerve, swerve_data

SHA = "sha-at-the-start"
ENV = ("NAVSIM_DEVKIT_ROOT", "NAVSIM_EXP_ROOT", "OPENSCENE_DATA_ROOT", "NUPLAN_MAPS_ROOT")
ENV += ("NUPLAN_MAP_VERSION", "DRIVE_JEPA_MSDA")


class Cfg(SiteConfig):
    def resolve(self, template: str, *args) -> str:
        return template


class Loss:
    """Enough of a tensor for the loop's arithmetic; `backward` is what the tests read."""

    def __init__(self, events: list[str]):
        self.events = events

    def __mul__(self, other):
        return self

    __rmul__ = __mul__
    __add__ = __mul__
    __truediv__ = __mul__

    def backward(self):
        self.events.append("backward")

    def detach(self):
        return self

    def __float__(self):
        return 1.0


class Model:
    training = True
    scorer = SimpleNamespace(train=lambda: None, eval=lambda: None)

    def __call__(self, features):
        return {}

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def modules(self):
        return []

    def parameters(self):
        return [SimpleNamespace(requires_grad=True, numel=lambda: 1_000_000)]

    def state_dict(self):
        return {"w": 1}


class Dataset:
    """Builds a tick unless it is marked `unbuilt`, as `TickDataset.batch` skips what it cannot."""

    def __init__(self, ticks, device):
        self.ticks = ticks

    def __len__(self):
        return len(self.ticks)

    def batch(self, indices):
        tick = self.ticks[indices[0]]
        return (None, None, None) if tick.get("unbuilt") else ({}, {}, [tick])


def tick(scene: str, i: int) -> dict:
    return {"episode": f"frontal-{scene}/run_0", "timestamp": i, "expert_kind": "brake", "frame": "f"}


def harness(monkeypatch, tmp_path, sources: dict[str, list[dict]], *argv: str) -> SimpleNamespace:
    """`swerve.main`, ready to call. Returns the config and the optimiser's event log."""
    cfg = Cfg(site="test", weights=tmp_path / "weights", runs=tmp_path / "runs")
    events: list[str] = []
    optimiser = SimpleNamespace(
        zero_grad=lambda set_to_none=True: events.append("zero"), step=lambda: events.append("step")
    )
    torch = SimpleNamespace(
        manual_seed=lambda seed: None,
        optim=SimpleNamespace(Adam=lambda parameters, lr: optimiser),
        nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda parameters, limit: None)),
        save=lambda state, path: Path(path).write_bytes(b"weights"),
    )
    for name in ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv("GALLOPS_FRAME_CACHE", raising=False)
    monkeypatch.setattr(SiteConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(swerve, "git_sha", lambda repo: SHA)
    monkeypatch.setattr(swerve, "rebuildable", lambda t: True)
    monkeypatch.setattr(swerve, "Referee", lambda *a, **k: None)
    monkeypatch.setattr(swerve, "evaluate", lambda *a, **k: {})
    monkeypatch.setattr(
        swerve, "build_model", lambda *a, **k: (SimpleNamespace(anchors=None), Model(), torch)
    )
    monkeypatch.setattr(
        swerve, "combined_loss", lambda *a, **k: (Loss(events), Loss(events), {"trajectory": 1.0})
    )
    monkeypatch.setattr(
        swerve,
        "load_ticks",
        lambda path, blind=False, source=0: [dict(t, source=source) for t in sources[path.name]],
    )
    monkeypatch.setattr(swerve_data, "TickDataset", Dataset)
    monkeypatch.setattr(preflight, "run", lambda *a, **k: {"code": {"sha": SHA}})
    spec = [a for name in sources for a in ("--source", str(tmp_path / name), "d", "start")]
    args = ["swerve", "--train-run", "r", "--device", "cpu", "--val-every", "0", *spec, *argv]
    monkeypatch.setattr(sys, "argv", args)
    return SimpleNamespace(cfg=cfg, events=events, run=cfg.checkpoint_dir(swerve.MODEL, "r", 0).parent)


def steps(run: Path) -> list[Path]:
    return sorted(p for p in run.iterdir() if p.name.startswith("step_"))


def groups_of(events: list[str]) -> list[int]:
    """The passes each optimiser step applied. Refuses gradients wiped, or a step with none new."""
    groups, pending = [], 0
    for event in events:
        if event == "backward":
            pending += 1
        elif event == "zero":
            assert pending == 0, f"{pending} passes wiped before any step: {events}"
        else:
            assert pending, f"a step with no new gradients: {events}"
            groups.append(pending)
            pending = 0
    assert pending == 0, f"{pending} passes never stepped: {events}"
    return groups


def test_a_run_gets_past_the_preflight_when_nothing_moved(tmp_path, monkeypatch):
    """#17 review: `loaded` held the sha, was overwritten with a tick count, then compared to the sha."""
    h = harness(
        monkeypatch,
        tmp_path,
        {"r0": [tick("0053", i) for i in range(4)]},
        "--epochs",
        "2",
        "--held-out",
        "0075",
    )
    swerve.main()
    assert [p.name for p in steps(h.run)] == ["step_000000004", "step_000000008"]
    assert sorted(p.name for p in steps(h.run)[0].iterdir()) == ["checkpoint.json", "model.pt"]
    assert sidecar.read(steps(h.run)[0]).train_git_sha == SHA


def test_a_tree_that_did_move_is_still_refused(tmp_path, monkeypatch):
    harness(monkeypatch, tmp_path, {"r0": [tick("0053", 0)]}, "--epochs", "1", "--held-out", "0075")
    monkeypatch.setattr(preflight, "run", lambda *a, **k: {"code": {"sha": "another"}})
    with pytest.raises(SystemExit, match="HEAD moved"):
        swerve.main()


def test_a_step_is_published_whole_or_not_at_all(tmp_path, monkeypatch):
    """#17 review: a job stopped between the two writes left `model.pt` with no sidecar."""
    h = harness(monkeypatch, tmp_path, {"r0": [tick("0053", 0)]}, "--epochs", "1", "--held-out", "0075")

    def killed(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(swerve.sidecar, "write", killed)
    with pytest.raises(KeyboardInterrupt):
        swerve.main()
    assert steps(h.run) == []
