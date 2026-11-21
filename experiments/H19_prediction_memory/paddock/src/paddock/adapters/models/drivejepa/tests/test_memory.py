"""H15's prediction memory (memory.py): state by timestamp, and parity with gallops' copy.

The layer needs torch, which paddock's own env lacks; those tests skip there and run under
Drive-JEPA's env (`<drive-jepa python> -m pytest .../test_memory.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.models.drivejepa import memory
from paddock.adapters.models.drivejepa.policy import POLICY

FIXTURE = Path(__file__).parent / "fixtures" / "memory_fixture.npz"


def test_memory_is_off_unless_asked_for(monkeypatch) -> None:
    monkeypatch.delenv(memory.ENV, raising=False)
    policy = POLICY(checkpoint="/x.ckpt")
    assert policy.memory_dir == "" and policy.memory is None
    assert "memory" not in policy.diagnostics()


def test_a_weights_directory_without_its_sidecar_is_refused(tmp_path) -> None:
    pytest.importorskip("torch")
    (tmp_path / memory.WEIGHTS).write_bytes(b"x")
    with pytest.raises(memory.MemoryCheckpointError, match="checkpoint.json"):
        memory.load(tmp_path)


def test_a_sidecar_that_disagrees_with_the_weights_is_refused(tmp_path) -> None:
    pytest.importorskip("torch")
    (tmp_path / memory.WEIGHTS).write_bytes(b"x")
    sidecar = {
        "model": memory.MODEL_NAME,
        "weights_sha256": "0" * 64,
        "notes": {"mode": "carry", "length": 4},
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(sidecar))
    with pytest.raises(memory.MemoryCheckpointError, match="sha256"):
        memory.load(tmp_path)


# --- the layer ---------------------------------------------------------------------------------------


def _fixture_model():
    torch = pytest.importorskip("torch")
    if not FIXTURE.is_file():
        # The array is not committed in gallops (experiments/H15_prediction_memory/POINTERS.md).
        pytest.skip(f"{FIXTURE.name} is absent: build it with gallops' scripts/make_memory_fixture.py")
    data = np.load(FIXTURE)
    state = {k[2:]: torch.as_tensor(data[k]) for k in data.files if k.startswith("w:")}
    model = memory.build_model(int(state["keep.weight"].shape[0]))
    model.load_state_dict(state)
    return model.eval(), data, torch


@pytest.mark.parametrize("mode", memory.MODES)
def test_the_layer_gives_gallops_scores_on_the_same_inputs(mode: str) -> None:
    model, data, torch = _fixture_model()
    args = [torch.as_tensor(data[k]) for k in ("embed", "logits", "ego", "match", "valid")]
    with torch.no_grad():
        scores, _ = model(*args, mode=mode)
    np.testing.assert_allclose(scores.numpy(), data[f"scores_{mode}"], atol=1e-6, rtol=0)


# --- state by timestamp ------------------------------------------------------------------------------


def _drive(n: int, hz: float, seed: int = 0):
    """A car driving straight at 10 m/s: moments at `hz`, each with its pose, proposals and inputs.

    The inputs of a moment depend only on its timestamp, so two feeds of the same frame agree.
    """
    torch = pytest.importorskip("torch")
    out = []
    for k in range(n):
        ts = int(round(k * 1e6 / hz))
        rng = np.random.default_rng([seed, ts])
        pose = np.eye(4)
        pose[0, 3] = 10.0 * ts / 1e6
        lateral = np.linspace(-3, 3, 32)[:, None] + rng.normal(scale=0.05, size=(32, 1))
        ahead = 10.0 * 0.5 * np.arange(1, 9)[None] * np.linspace(0.6, 1.2, 32)[:, None]
        proposals = np.stack([ahead, np.repeat(lateral, 8, 1), np.zeros((32, 8))], -1)
        embed = torch.as_tensor(rng.normal(size=(32, memory.EMBED)), dtype=torch.float32)
        logits = torch.as_tensor(rng.normal(size=(32, memory.LOGITS)), dtype=torch.float32)
        ego = torch.as_tensor(rng.normal(size=(memory.EGO,)), dtype=torch.float32)
        out.append((ts, pose, proposals, embed, logits, ego))
    return out


def _memory(mode: str, length: int = 4, zero: bool = False):
    model, _, torch = _fixture_model()
    if zero:
        model = memory.build_model(16)  # as trained from: last layer at zero
        model.eval()
    return memory.Memory(model, mode, length)


def test_a_zero_last_layer_leaves_every_score_as_released() -> None:
    mem = _memory("carry", zero=True)
    for moment in _drive(12, 10.0):
        assert float(mem.step(*moment).abs().max()) == 0.0


def test_carry_equals_reset_on_the_first_moment() -> None:
    first = _drive(1, 2.0)[0]
    a = _memory("carry").step(*first)
    b = _memory("reset").step(*first)
    assert a.numpy().tolist() == b.numpy().tolist()
    assert float(a.abs().max()) > 0


def test_carry_differs_from_reset_once_it_has_a_past() -> None:
    moments = _drive(3, 2.0)
    carry, reset = _memory("carry"), _memory("reset")
    for m in moments:
        a, b = carry.step(*m), reset.step(*m)
    assert not np.allclose(a.numpy(), b.numpy())
    assert carry.counts["chains"] == 1 and reset.counts["chains"] == 1


def test_reset_clears_state() -> None:
    moments = _drive(4, 2.0)
    mem = _memory("carry")
    for m in moments[:3]:
        mem.step(*m)
    mem.reset()
    assert mem.entries == [] and mem.counts["steps"] == 3 and mem.counts["resets"] == 1
    fresh = _memory("carry")
    assert mem.step(*moments[3]).numpy().tolist() == fresh.step(*moments[3]).numpy().tolist()


def test_the_ten_hertz_chain_equals_the_two_hertz_chain_on_the_same_frames() -> None:
    """At 10 Hz the node keeps five interleaved chains; each is the 2 Hz chain of its frames."""
    fast = _memory("carry")
    every = {m[0]: fast.step(*m) for m in _drive(41, 10.0)}  # 0 to 4.0 s
    assert fast.counts["chains"] == 5
    for offset in range(5):
        slow = _memory("carry")
        frames = [m for m in _drive(41, 10.0) if (m[0] // 100_000) % 5 == offset]
        for m in frames:
            np.testing.assert_array_equal(slow.step(*m).numpy(), every[m[0]].numpy())


def test_the_window_reaches_back_length_moments_and_no_further() -> None:
    moments = _drive(8, 2.0)
    base = _memory("carry", length=2)
    for m in moments:
        last = base.step(*m)
    # Change the moment L + 1 back (index 4 of 0..7): the score at 7 must not move.
    changed = list(moments)
    ts, pose, proposals, embed, logits, ego = changed[4]
    changed[4] = (ts, pose, proposals, embed + 1.0, logits, ego)
    other = _memory("carry", length=2)
    for m in changed:
        out = other.step(*m)
    np.testing.assert_array_equal(out.numpy(), last.numpy())
    # ... and changing the moment L back (index 5) must.
    changed[5] = (changed[5][0], changed[5][1], changed[5][2], changed[5][3] + 1.0, *changed[5][4:])
    third = _memory("carry", length=2)
    for m in changed:
        out = third.step(*m)
    assert not np.allclose(out.numpy(), last.numpy())


def test_a_gap_over_the_window_starts_a_new_chain_and_a_clock_jump_clears() -> None:
    moments = _drive(6, 2.0)
    mem = _memory("carry")
    for m in moments[:3]:
        mem.step(*m)
    mem.step(*moments[5])  # 1.5 s after the last: nothing 0.45-0.55 s back
    assert mem.counts["chains"] == 2
    mem.step(*moments[0])  # the clock went backwards: a new episode without a reset
    assert mem.counts["clock_jumps"] == 1 and mem.counts["chains"] == 3
    np.testing.assert_array_equal(mem.step(*moments[1]).numpy(), _two(moments)[1].numpy())


def _two(moments):
    mem = _memory("carry")
    return [mem.step(*m) for m in moments[:2]]


def test_paths_are_handed_to_the_closest_old_path_after_ego_motion() -> None:
    """A proposal set that moves with the car matches slot to slot; nothing beyond 2 m matches."""
    old = np.zeros((32, 8, 3))
    old[:, :, 0] = 5.0 * np.arange(1, 9)[None]
    old[:, :, 1] = np.arange(32)[:, None] * 3.0
    pose_old, pose_new = np.eye(4), np.eye(4)
    pose_new[0, 3] = 5.0  # 0.5 s at 10 m/s
    new = old.copy()
    m = memory.match(memory.distances(memory.align_previous(old, pose_old, pose_new), new))
    assert m.tolist() == list(range(32))
    far = new.copy()
    far[:, :, 1] += 2.5
    m = memory.match(memory.distances(memory.align_previous(old, pose_old, pose_new), far))
    assert (m[-1] == -1) and (m[:-1] == np.arange(1, 32)).all()


def test_a_plan_runs_the_memory_on_its_own_chain_only_and_never_twice() -> None:
    """KITScenes hands over 10 Hz history: only frames 0.5 s, 1.0 s, ... before the scored one run."""
    from types import SimpleNamespace

    pytest.importorskip("torch")  # `_catch_up` runs the passes under torch.no_grad

    stepped: list[int] = []

    class _Stub(POLICY):
        def __init__(self):
            self.history = []
            self.memory = SimpleNamespace(mode="carry", length=4, entries=[])
            self._memory_ctx = None

            def forward(features):
                ts = self._memory_ctx["timestamp"]
                assert self.history[-1]["timestamp"] == ts, "features must be computed as of that frame"
                stepped.append(ts)
                self.memory.entries.append({"ts": ts})

            self.agent = SimpleNamespace(forward=forward)

        def _features(self):
            return {"ego_status": np.zeros((1, 4, 11), np.float32)}

        @staticmethod
        def _memory_context(obs, features):
            return {"timestamp": obs.timestamp_us}

    stub = _Stub()
    stub.history = [{"timestamp": k * 100_000, "ego2world": np.eye(4)} for k in range(23)]  # 0 to 2.2 s
    stub._catch_up()
    # 2.2 s is scored; its chain is 1.7, 1.2, 0.7, 0.2 s, and 0.2 s has no frame 0.5 s before it.
    assert stepped == [700_000, 1_200_000, 1_700_000]
    assert len(stub.history) == 23 and stub._memory_ctx is None
    # The next plan, 0.5 s later, finds its chain already in memory once 2.2 s is in it.
    stub.memory.entries.append({"ts": 2_200_000})
    stub.history += [{"timestamp": k * 100_000, "ego2world": np.eye(4)} for k in range(23, 28)]
    stub._catch_up()
    assert stepped == [700_000, 1_200_000, 1_700_000]
