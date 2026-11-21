"""Control (c)'s feature sets: which blocks of a window each variant keeps."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.memory import probe, probe_control

B, C, E = probe.PCA_DIM + probe.EGO_DIM, probe.PCA_DIM, probe.EGO_DIM


def kinds(cols):
    """(block, 'camera' | 'car') of each column."""
    return {(int(c) // B, "camera" if int(c) % B < C else "car") for c in cols}


@pytest.mark.parametrize("k", [2, 4, 8])
def test_feature_counts(k):
    counts = {v: len(probe_control.variant_columns(v, k)) for v in probe_control.VARIANTS}
    assert counts == {"V1": B * k, "V2": C + E * k, "V3": C * k + E, "V4": E * k}


@pytest.mark.parametrize("k", [2, 4, 8])
def test_now_is_the_newest_block_and_history_is_every_block(k):
    every = set(range(k))
    assert kinds(probe_control.variant_columns("V2", k)) == {(0, "camera")} | {(j, "car") for j in every}
    assert kinds(probe_control.variant_columns("V3", k)) == {(j, "camera") for j in every} | {(0, "car")}
    assert kinds(probe_control.variant_columns("V4", k)) == {(j, "car") for j in every}


def test_v1_and_v4_are_step2_s_columns_in_step2_s_order():
    for k in (2, 4, 8):
        assert np.array_equal(probe_control.variant_columns("V1", k), np.arange(B * k))
        step2_ego = np.concatenate([np.arange(C, B) + j * B for j in range(k)])
        assert np.array_equal(probe_control.variant_columns("V4", k), step2_ego)
