"""The on-disk frame cache returns exactly what decoding the frame would."""

from __future__ import annotations

import numpy as np
import pytest

from gallops.train import frame_cache
from gallops.train.frame_cache import FrameCache, build_episode, crop_resize


def test_a_cached_frame_is_the_decoded_frame(tmp_path):
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    episode = tmp_path / "collect" / "frontal-0075" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    rng = np.random.default_rng(0)
    paths = []
    for stamp in (200, 100):
        path = episode / "CAM_FRONT" / f"{stamp}.jpg"
        image_module.fromarray(rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8)).save(path, format="PNG")
        paths.append(path)
    root = tmp_path / "cache"
    root.mkdir()
    build_episode((str(episode), str(root)))
    cache = FrameCache(root)
    for path in paths:
        np.testing.assert_array_equal(cache.get(path), crop_resize(path))
    assert cache.get(episode / "CAM_FRONT" / "999.jpg") is None
    assert FrameCache(tmp_path / "empty").get(paths[0]) is None


def test_a_read_error_is_retried_and_a_lasting_one_raised(tmp_path, monkeypatch):
    path = tmp_path / "frames.npy"
    np.save(path, np.arange(3))
    real, calls = np.load, []

    def flaky(p, *args, **kwargs):
        calls.append(p)
        if len(calls) < 3:
            raise OSError(5, "Input/output error")
        return real(p, *args, **kwargs)

    monkeypatch.setattr(frame_cache.np, "load", flaky)
    np.testing.assert_array_equal(frame_cache._load(path, wait_s=0.0), np.arange(3))
    assert len(calls) == 3
    monkeypatch.setattr(
        frame_cache.np, "load", lambda p: (_ for _ in ()).throw(OSError(5, "Input/output error"))
    )
    with pytest.raises(OSError):
        frame_cache._load(path, attempts=2, wait_s=0.0)
