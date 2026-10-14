"""The on-disk frame cache returns exactly what decoding the frame would."""

from __future__ import annotations

import numpy as np
import pytest

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
