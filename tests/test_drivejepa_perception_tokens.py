"""H15's token pass, the parts Part B added: tap rows off navtrain, and the release's own scores.

`tokens` imports torch at the top, so these skip where torch is absent (CI) and run in Drive-JEPA's
interpreter, where the preflight runs the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gallops.drivejepa_perception import tokens  # noqa: E402


def test_uniform_tap_rows_spread_over_the_pass():
    rows = tokens.tap_rows_uniform(12146)
    assert len(rows) == tokens.TAP_UNIFORM and rows[0] == 0 and rows[-1] == 12145
    assert np.all(np.diff(rows) > 0)
    assert tokens.tap_rows_uniform(10).tolist() == list(range(10))


def test_a_pick_must_be_the_first_argmax_of_its_scores():
    scores = np.array([[0.2, 0.9, 0.9], [0.5, 0.1, 0.4]], np.float32)
    assert tokens.pick_problems(scores, np.array([1, 0])) == []
    # A tie goes to the first index, as torch's argmax does; the second is a different pick.
    assert tokens.pick_problems(scores, np.array([2, 0]))


def test_release_scores_fall_back_to_the_fp32_sigmoid(tmp_path):
    logits = np.random.default_rng(0).normal(size=(4, 32, 6)).astype(np.float32)
    np.save(tmp_path / "logits.npy", logits)
    got = tokens.release_scores(tmp_path)
    assert got.dtype == np.float32 and got.shape == (4, 32)
    assert np.array_equal(got, torch.sigmoid(torch.from_numpy(logits))[..., -1].numpy())  # the vendor's order
    np.save(tmp_path / "scores.npy", got + 1)
    assert np.array_equal(tokens.release_scores(tmp_path), got + 1)  # a saved array wins
