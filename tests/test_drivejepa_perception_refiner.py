"""H15 B2's teachers, from the gathered array (torch; it skips in gallops' own interpreter)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gallops.drivejepa_perception import train_refiner  # noqa: E402


def test_teachers_are_drawn_as_the_release_drew_them():
    vocabulary = np.arange(10 * 8 * 3, dtype=np.float32).reshape(10, 8, 3)
    anchors = np.full((3, train_refiner.MAX_ANCHORS), -1, np.int16)
    anchors[0, :2] = [3, 7]
    anchors[2, :5] = [1, 2, 4, 5, 9]
    count = np.array([2, 0, 5], np.int16)
    out, n = train_refiner.teachers([0, 1, 2], anchors, count, vocabulary, np.random.default_rng(0))
    assert out.shape == (3, train_refiner.TEACHERS, 8, 3)
    assert n.tolist() == [2, 0, 4]
    which = [[int(t[0, 0] // 24) for t in row.numpy()] for row in out]
    assert which[0][:2] == [3, 7]  # two good anchors: both, once each
    assert torch.equal(out[1], torch.zeros_like(out[1]))  # none: no teacher
    assert set(which[2]) <= {1, 2, 4, 5, 9} and len(set(which[2])) == 4  # five: four, without replacement


def test_a_row_without_teachers_adds_no_teacher_term():
    from gallops.drivejepa_b2d.train_generator import mtd_loss

    target = torch.zeros(1, 8, 3)
    props = [torch.ones(1, 32, 8, 3)]
    teach = torch.zeros(1, train_refiner.TEACHERS, 8, 3)
    alone = mtd_loss(props, target, teach, torch.tensor([0]), prev_weight=0.1)
    wta = (props[0] - target[:, None]).abs().sum(-1).mean(-1).amin(1).mean()
    assert torch.equal(alone, wta)
