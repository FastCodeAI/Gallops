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
