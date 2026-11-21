"""The decode pool and the read-ahead: results by position, never by finishing order."""

from __future__ import annotations

import random
import threading
import time

import numpy as np
import pytest

from gallops.compat import zip_strict
from gallops.memory import cache, inputs


def slow_load(path, k, blank):
    """A stand-in for `inputs.load_image`: finishes in a random order, returns what it was given."""
    time.sleep(random.random() * 0.02)
    return np.full((2, 2), path), k


def test_one_worker_is_no_pool():
    assert inputs.decode_pool(1) is None
    with pytest.raises(ValueError):
        inputs.decode_pool(0)


def test_load_all_keeps_the_jobs_order_whatever_finishes_first():
    jobs = [(p, np.eye(3) * p, False) for p in range(25)]
    serial = inputs.load_all(jobs, None, load=slow_load)
    pool = inputs.decode_pool(4)
    try:
        for _ in range(3):
            pooled = inputs.load_all(jobs, pool, load=slow_load)
            for (a, ka), (b, kb) in zip_strict(serial, pooled):
                assert np.array_equal(a, b) and np.array_equal(ka, kb)
            assert len(pooled) == len(serial) == 25
    finally:
        pool.shutdown()


def test_load_all_uses_several_threads():
    seen = set()

    def load(*job):
        seen.add(threading.get_ident())
        time.sleep(0.01)
        return job

    pool = inputs.decode_pool(4)
    try:
        inputs.load_all([(i,) for i in range(16)], pool, load=load)
    finally:
        pool.shutdown()
    assert len(seen) > 1


def test_load_all_raises_a_worker_s_error():
    def load(path):
        if path == 3:
            raise FileNotFoundError(path)
        return path

    pool = inputs.decode_pool(2)
    try:
        with pytest.raises(FileNotFoundError):
            inputs.load_all([(i,) for i in range(6)], pool, load=load)
    finally:
        pool.shutdown()


@pytest.mark.parametrize("workers", [1, 4])
def test_log_inputs_yields_in_order_with_and_without_read_ahead(monkeypatch, workers):
    def fake_agent_input(frames, frame_idx, roots, blank=False, pool=None):
        time.sleep(random.random() * 0.01)
        if frame_idx == 5:
            raise FileNotFoundError("img5")
        return ("input", frame_idx)

    monkeypatch.setattr(inputs, "agent_input", fake_agent_input)
    pool, ahead = cache.decoders(workers)
    assert (pool is None, ahead is None) == ((True, True) if workers == 1 else (False, False))
    try:
        out = list(cache.log_inputs([], [1, 2, 5, 7, 9], [], pool, ahead))
        assert list(cache.log_inputs([], [], [], pool, ahead)) == []
    finally:
        for executor in (ahead, pool):
            if executor is not None:
                executor.shutdown()
    assert out[:2] == [("input", 1), ("input", 2)]
    assert isinstance(out[2], FileNotFoundError)  # a missing image is a value, the log goes on
    assert out[3:] == [("input", 7), ("input", 9)]
