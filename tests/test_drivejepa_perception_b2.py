"""B-2's inputs and adapter: the fold rule, the alignment arithmetic, and the side-car's hidden layer.

The pieces that need torch (the side-car, the committed fold record, which reads `train_sidecar`) skip
without it; the training interpreter's preflight runs them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gallops.drivejepa_perception import b2_inputs


def test_fold_of_logs_deals_every_log_round_robin():
    logs = [f"log{i:03d}" for i in range(23)]
    folds = b2_inputs.fold_of_logs(logs)
    assert set(folds) == set(logs)
    sizes = np.bincount(list(folds.values()), minlength=5)
    assert sizes.max() - sizes.min() <= 1
    assert b2_inputs.fold_of_logs(list(reversed(logs))) == folds  # the order given does not matter


def test_ridge_recovers_a_linear_map_and_r2_reads_it():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 6))
    w = rng.normal(size=(6, 6))
    y = x @ w + 0.5
    fit, bias = b2_inputs.ridge(x, y, lam=1e-6)
    assert np.allclose(fit, w, atol=1e-4) and np.allclose(bias, 0.5, atol=1e-4)
    assert b2_inputs.r_squared(x @ fit + bias, y) == pytest.approx(1.0, abs=1e-8)
    assert b2_inputs.r_squared(np.zeros_like(y) + y.mean(0), y) == pytest.approx(0.0, abs=1e-12)


def test_match_by_centre_pairs_the_nearest_slots():
    pytest.importorskip("scipy")
    a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    b = a[[2, 0, 1]] + 0.1
    assert b2_inputs.match_by_centre(a, b).tolist() == [1, 2, 0]


def test_the_committed_folds_rederive():
    pytest.importorskip("torch")
    record = b2_inputs.load_folds()
    from gallops.drivejepa_perception.train_sidecar import crossfit_folds

    splits = json.loads((b2_inputs.REPO / "recipes/drivejepa_perception/splits.json").read_text())
    for k, logs in enumerate(crossfit_folds(splits["train"])):
        assert all(record["train"][log] == k for log in logs)
    assert not set(record["select"]) & set(record["train"])


def test_the_sidecar_hands_over_its_hidden_layer_beside_its_outputs():
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception.sidecar import SideCar

    torch.manual_seed(0)
    car = SideCar(d_in=16, d=32, heads=4, layers=3, queries=8, ffn=64).eval()
    tokens = torch.randn(2, 512, 16)
    projection = torch.eye(4).expand(2, 4, 4).clone()
    projection[:, 0, 2] = 256.0
    projection[:, 1, 2] = 128.0
    with torch.no_grad():
        outs = car(tokens, projection)
    assert len(outs) == 3
    for out in outs:
        assert out["hidden"].shape == (2, 8, 32)
        # The heads read the hidden layer: the class logits are its linear readout.
    last = outs[-1]
    layer = car.layers[-1]
    assert torch.allclose(last["logits"], layer.cls_head(last["hidden"]).float())


def _context(torch, b: int, a: int, arm: str) -> dict:
    """A plausible context for every arm: agents ahead within 30 m, slots with confidences and folds."""
    g = torch.Generator().manual_seed(1)
    xyz = torch.rand(b, a, 3, generator=g) * torch.tensor([30.0, 10.0, 1.0]) + torch.tensor([5.0, -5.0, 0.0])
    ctx = {
        "cls": torch.zeros(b, a, dtype=torch.long),
        "conf": torch.rand(b, a, generator=g),
        "xyz": xyz,
        "lwh": torch.ones(b, a, 3) * torch.tensor([4.5, 1.9, 1.6]),
        "yaw": torch.zeros(b, a),
        "vel": torch.zeros(b, a, 2),
        "path": torch.zeros(b, a, 2, 2),
        "mask": torch.rand(b, a, generator=g) > 0.5,
        "hidden": torch.randn(b, a, 384, generator=g),
        "centre": xyz,
        "fold": torch.tensor([i % 5 for i in range(b)]),
    }
    return ctx


@pytest.mark.parametrize("arm", ["C", "O", "D", "HD", "H6", "H3"])
def test_every_arm_starts_as_the_release(arm):
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception.train_b2 import B2Adapter

    adapter = B2Adapter(arm)
    adapter.context = _context(torch, 2, 128, arm)
    proposals = torch.randn(2, 32, 8, 3)
    queries = torch.randn(2, 256, 256)
    assert torch.equal(adapter(proposals, queries), torch.zeros(2, 256, 256))


def test_a_blank_read_attends_only_its_null_token():
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception.train_b2 import SlotRead

    read = SlotRead(d_query=8, d=8, heads=2)
    out, null_share = read(
        torch.randn(1, 3, 8), torch.randn(1, 5, 8), torch.zeros(1, 3, 5), torch.zeros(1, 5, dtype=torch.bool)
    )
    assert float(null_share) == pytest.approx(1.0)
    assert out.shape == (1, 3, 8)


def test_the_pass_hook_sends_pass_four_alone_to_the_copy():
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception.train_b2 import PassHook

    calls = []

    class Spy(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, proposals, queries):
            calls.append(self.name)
            return torch.zeros_like(queries)

    hook = PassHook(Spy("stage 1"), Spy("pass-4 copy"))
    for _ in range(4):
        hook(None, (torch.zeros(1, 32, 8, 3), torch.zeros(1, 256, 256), None))
    assert calls == ["stage 1", "stage 1", "stage 1", "pass-4 copy"]


def test_each_row_reads_its_own_folds_input_map():
    torch = pytest.importorskip("torch")
    from gallops.drivejepa_perception.train_b2 import B2Adapter

    adapter = B2Adapter("H6")
    ctx = _context(torch, 5, 4, "H6")
    with torch.no_grad():
        tokens = adapter.hidden_tokens(ctx)
        for row in range(5):
            alone = adapter.norm(ctx["hidden"][row]) @ adapter.maps[row % 5] + adapter.map_bias[row % 5]
            assert torch.allclose(tokens[row], alone, atol=1e-5)


def test_stage_three_reads_each_rows_labels_by_row_id(tmp_path):
    pytest.importorskip("torch")
    from gallops.drivejepa_perception.train_b2 import stage3_labels

    dump = tmp_path / "dump-train"
    dump.mkdir()
    rows = np.array([3, 7, 9])
    candidates = np.random.default_rng(0).random((3, 32, 9)).astype(np.float32)
    np.savez(dump / "dump.npz", rows=rows)
    np.savez(dump / "labels.npz", candidates=candidates)
    labels = stage3_labels(tmp_path, rows, 12)
    assert np.array_equal(labels[7], candidates[1, :, -1])  # row 7 is the dump's second, not its eighth
    assert np.isnan(labels[[0, 1, 2, 4, 5, 6, 8, 10, 11]]).all()
    with pytest.raises(SystemExit):
        stage3_labels(tmp_path, np.array([3, 7]), 12)
