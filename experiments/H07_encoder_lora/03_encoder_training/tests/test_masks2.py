"""Part B mask generator: shape of the output and the equal-count property."""

import numpy as np
import torch

import masks2 as M


def test_wide_cfgs_scale_only_aspect():
    for a, b in zip(M.P.MASK_CFGS, M.wide_cfgs()):
        assert a["num_blocks"] == b["num_blocks"] and a["spatial_scale"] == b["spatial_scale"]
        assert b["aspect_ratio"] == (a["aspect_ratio"][0] / 2, a["aspect_ratio"][1] / 2)


def test_equal_count_complement_sorted_and_seeded():
    for g in M.make():
        enc, pred, info = g(32, M.step_rng(1, 2, 3))
        assert enc.shape[0] == pred.shape[0] == 32
        assert enc.shape[1] + pred.shape[1] == M.N_TOKENS and pred.shape[1] == info["k"]
        for e, p in zip(enc, pred):
            assert torch.all(e[1:] > e[:-1]) and torch.all(p[1:] > p[:-1])
            assert not set(e.tolist()) & set(p.tolist())
        enc2, pred2, _ = g(32, M.step_rng(1, 2, 3))
        assert torch.equal(enc, enc2) and torch.equal(pred, pred2)


def test_samples_differ_within_batch():
    g = M.make()[0]
    _, pred, _ = g(8, M.step_rng(5))
    assert len({tuple(p.tolist()) for p in pred}) > 1


def test_hidden_tokens_are_union_of_blocks():
    g = M.make()[1]
    rng = M.step_rng(9)
    m, info = g(1, rng, equalise=False)
    grid = m[0].reshape(16, 32)
    rows = np.flatnonzero(grid.any(1)); cols = np.flatnonzero(grid.any(0))
    assert len(rows) >= info["h"] and len(cols) >= info["w"]
