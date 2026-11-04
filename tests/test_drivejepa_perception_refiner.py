"""H15 B2's teachers, from the gathered array (torch; it skips in gallops' own interpreter)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gallops.drivejepa_perception import train_refiner  # noqa: E402


def test_teachers_come_from_each_rows_own_good_anchors_and_none_without_a_file():
    vocabulary = np.arange(10 * 8 * 3, dtype=np.float32).reshape(10, 8, 3)
    anchors = np.full((3, train_refiner.MAX_ANCHORS), -1, np.int16)
    anchors[0, :2] = [3, 7]
    anchors[2, :5] = [1, 2, 4, 5, 9]
    count = np.array([2, 0, 5], np.int16)
    out = train_refiner.teachers([0, 1, 2], anchors, count, vocabulary, np.random.default_rng(0))
    assert out.shape == (3, train_refiner.TEACHERS, 8, 3)
    first = {int(t[0, 0] // 24) for t in out[0].numpy()}
    assert first <= {3, 7}  # drawn with replacement from the row's two
    assert torch.equal(out[1], torch.zeros_like(out[1]))  # no file, no teacher
    third = [int(t[0, 0] // 24) for t in out[2].numpy()]
    assert set(third) <= {1, 2, 4, 5, 9} and len(set(third)) == train_refiner.TEACHERS  # four distinct
