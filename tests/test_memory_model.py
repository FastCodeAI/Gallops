"""gallops.memory.model and train: windows, length limit, zero start, carry vs reset, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.memory import model as mm

torch = pytest.importorskip("torch")


def _chain(n=6, paths=32):
    """One chain of n moments (rows 0..n-1), each path matched to the same slot before it."""
    prev = np.array([-1, *range(n - 1)])
    match = np.tile(np.arange(paths), (n, 1))
    match[0] = -1
    return prev, match


def _inputs(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = int(max(rows)) + 1
    return (
        torch.randn(n, 32, mm.EMBED, generator=g),
        torch.randn(n, 32, mm.LOGITS, generator=g),
        torch.randn(n, mm.EGO, generator=g),
    )


def _score(net, embed, logits, ego, prev, match, target, earlier, mode):
    rows, m, valid = mm.batch_windows(prev, match, [target], earlier)
    r = torch.from_numpy(rows)
    return net(embed[r], logits[r], ego[r], torch.from_numpy(m), torch.from_numpy(valid), mode)[0]


def _net(seed=0, nonzero=True):
    torch.manual_seed(seed)
    net = mm.MemoryPicker()
    if nonzero:
        with torch.no_grad():
            net.out[-1].weight.normal_(0, 0.5)
            net.out[-1].bias.normal_(0, 0.5)
    return net.eval()


def test_window_rows_follow_the_chain_and_stop_at_its_start():
    prev, _ = _chain()
    assert mm.window_rows(prev, 5, 2) == [3, 4, 5]
    assert mm.window_rows(prev, 1, 4) == [0, 1]
    assert mm.window_rows(prev, 0, 4) == [0]


def test_batch_windows_left_pad_and_mask():
    prev, match = _chain()
    rows, m, valid = mm.batch_windows(prev, match, [5, 1], [2, 2])
    assert rows.tolist() == [[3, 4, 5], [0, 0, 1]]
    assert valid.tolist() == [[True, True, True], [False, True, True]]
    assert (m[0, 0] == -1).all() and (m[1, 1] == -1).all()  # nothing flows into a window's first step
    assert (m[0, 2] == np.arange(32)).all()


def test_an_untrained_model_scores_exactly_as_released():
    prev, match = _chain()
    embed, logits, ego = _inputs(range(6))
    net = _net(nonzero=False)
    for mode in mm.MODES:
        with torch.no_grad():
            s = _score(net, embed, logits, ego, prev, match, 5, 4, mode)
        assert torch.equal(s[0], logits[5, :, -1])


def test_length_limit_cuts_the_state_handed_in_from_more_than_l_moments_back():
    prev, match = _chain()
    embed, logits, ego = _inputs(range(6))
    net = _net()
    for length in (1, 2, 4):
        target = 5
        with torch.no_grad():
            base = _score(net, embed, logits, ego, prev, match, target, length, "carry")
            outside, inside = embed.clone(), embed.clone()
            outside[target - length - 1] += 5.0  # moment t - L - 1: must not matter
            inside[target - length] += 5.0  # moment t - L: must matter
            s_out = _score(net, outside, logits, ego, prev, match, target, length, "carry")
            s_in = _score(net, inside, logits, ego, prev, match, target, length, "carry")
        assert torch.equal(base, s_out), length
        assert not torch.equal(base, s_in), length


def test_reset_ignores_every_earlier_moment():
    prev, match = _chain()
    embed, logits, ego = _inputs(range(6))
    net = _net()
    with torch.no_grad():
        base = _score(net, embed, logits, ego, prev, match, 5, 4, "reset")
        bumped = embed.clone()
        bumped[1:5] += 5.0
        assert torch.equal(base, _score(net, bumped, logits, ego, prev, match, 5, 4, "reset"))
        assert torch.equal(base, _score(net, embed, logits, ego, prev, match, 5, 0, "carry"))


def test_carry_equals_reset_on_a_chain_start_and_differs_later():
    prev, match = _chain()
    embed, logits, ego = _inputs(range(6))
    net = _net()
    with torch.no_grad():
        assert torch.equal(
            _score(net, embed, logits, ego, prev, match, 0, 4, "carry"),
            _score(net, embed, logits, ego, prev, match, 0, 4, "reset"),
        )
        assert not torch.equal(
            _score(net, embed, logits, ego, prev, match, 3, 4, "carry"),
            _score(net, embed, logits, ego, prev, match, 3, 4, "reset"),
        )


def test_memory_follows_the_match_not_the_slot():
    prev, match = _chain(2)
    match[1] = np.roll(np.arange(32), 1)  # new path i inherits from old path i - 1
    match[1, 7] = -1  # and new path 7 inherits nothing
    embed, logits, ego = _inputs(range(2))
    net = _net()
    with torch.no_grad():
        base = _score(net, embed, logits, ego, prev, match, 1, 1, "carry")
        bumped = embed.clone()
        bumped[0, 4] += 5.0  # old path 4 feeds new path 5 only
        s = _score(net, bumped, logits, ego, prev, match, 1, 1, "carry")
        bumped7 = embed.clone()
        bumped7[0, 6] += 5.0  # old path 6 would feed new path 7, which inherits nothing
        s7 = _score(net, bumped7, logits, ego, prev, match, 1, 1, "carry")
    changed = (s - base).abs()[0] > 0
    assert changed.tolist() == [k == 5 for k in range(32)]
    assert torch.equal(s7, base)


def test_unknown_mode_is_refused():
    prev, match = _chain()
    embed, logits, ego = _inputs(range(6))
    with pytest.raises(ValueError, match="mode"):
        _score(_net(), embed, logits, ego, prev, match, 5, 1, "remember")


class _FakeData:
    """The parts of train.Data that train_one uses, on CPU, with a learnable target."""

    def __init__(self):
        rng = np.random.default_rng(0)
        n = 40
        self.prev = np.array([-1 if k % 8 == 0 else k - 1 for k in range(n)])
        self.match = np.tile(np.arange(32), (n, 1))
        self.match[self.prev < 0] = -1
        self.embed = torch.from_numpy(rng.normal(size=(n, 32, mm.EMBED)).astype(np.float32))
        self.logits = torch.from_numpy(rng.normal(size=(n, 32, mm.LOGITS)).astype(np.float32))
        self.ego = torch.from_numpy(rng.normal(size=(n, mm.EGO)).astype(np.float32))
        self.target_t = torch.sigmoid(self.embed[..., 0])
        self.device = "cpu"
        self.rows = {"train": np.arange(1, 32), "selection": np.arange(33, 40)}

    def batch(self, targets, earlier):
        rows, m, valid = mm.batch_windows(self.prev, self.match, targets, earlier)
        r = torch.from_numpy(rows)
        t = torch.from_numpy(np.asarray(targets, np.int64))
        return (
            self.embed[r],
            self.logits[r],
            self.ego[r],
            torch.from_numpy(m),
            torch.from_numpy(valid),
            self.target_t[t],
        )


def test_training_is_deterministic_and_arms_start_and_order_alike(monkeypatch):
    from gallops.memory import train

    monkeypatch.setattr(train, "BATCH", 8)
    data = _FakeData()
    runs = [train.train_one(data, "carry", 2, seed=3, epochs=2, log=lambda s: None) for _ in range(2)]
    for k in runs[0]["state"]:
        assert torch.equal(runs[0]["state"][k], runs[1]["state"][k]), k
    assert runs[0]["history"] == [
        {**h, "seconds": runs[0]["history"][i]["seconds"]} for i, h in enumerate(runs[1]["history"])
    ]
    # the order and warm-ups depend on seed and epoch only
    a, b = train.epoch_plan(31, 3, 1), train.epoch_plan(31, 3, 1)
    assert (
        (a[0] == b[0]).all() and (a[1] == b[1]).all() and a[1].min() >= 1 and a[1].max() <= train.WARMUP_MAX
    )
    # both arms start from the same weights
    train.set_determinism(3)
    w_a = mm.MemoryPicker().state_dict()
    train.set_determinism(3)
    w_b = mm.MemoryPicker().state_dict()
    assert all(torch.equal(w_a[k], w_b[k]) for k in w_a)
    # and training moved the selection loss
    assert runs[0]["history"][-1]["train_loss"] != runs[0]["history"][0]["train_loss"]
