"""The on-disk frame cache returns exactly what decoding the frame would."""

from __future__ import annotations

from pathlib import Path

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


def _stub_decode(monkeypatch):
    """A decoder needing no cv2: every pixel is the frame file's first byte."""
    monkeypatch.setattr(
        frame_cache, "crop_resize", lambda p: np.full((256, 512, 3), Path(p).read_bytes()[0], np.uint8)
    )


def _one_frame(tmp_path, content: bytes) -> tuple[Path, Path, Path]:
    episode = tmp_path / "collect" / "frontal-0075" / "run_0"
    (episode / "CAM_FRONT").mkdir(parents=True)
    frame = episode / "CAM_FRONT" / "100.jpg"
    frame.write_bytes(content)
    root = tmp_path / "cache"
    root.mkdir()
    return episode, frame, root


def test_an_episode_collected_again_is_not_served_its_old_pixels(tmp_path, monkeypatch):
    """#17 review, 🟠 5: keyed by path alone, the old array came back and `build_episode` said cached."""
    _stub_decode(monkeypatch)
    episode, frame, root = _one_frame(tmp_path, b"\x01")
    build_episode((str(episode), str(root)))
    assert FrameCache(root).get(frame)[0, 0, 0] == 1

    frame.write_bytes(b"\x02 collected again")
    assert FrameCache(root).get(frame) is None  # so the trainer decodes it
    assert build_episode((str(episode), str(root))).startswith("wrote")
    assert FrameCache(root).get(frame)[0, 0, 0] == 2


def test_an_unchanged_episode_stays_cached(tmp_path, monkeypatch):
    _stub_decode(monkeypatch)
    episode, frame, root = _one_frame(tmp_path, b"\x01")
    build_episode((str(episode), str(root)))
    assert build_episode((str(episode), str(root))).startswith("cached")


def test_an_episode_is_listed_once_not_on_every_read(tmp_path, monkeypatch):
    """A scan of the NAS per frame would cost what the cache exists to save."""
    _stub_decode(monkeypatch)
    episode, frame, root = _one_frame(tmp_path, b"\x01")
    build_episode((str(episode), str(root)))
    calls, real = [], frame_cache.key
    monkeypatch.setattr(frame_cache, "key", lambda d: calls.append(d) or real(d))
    cache = FrameCache(root)
    for _ in range(5):
        cache.get(frame)
    assert len(calls) == 1


def test_the_trainer_decodes_with_the_caches_own_crop(tmp_path, monkeypatch):
    """#17 review, 🟠 5: the crop was written out twice, and the two could drift apart."""
    from types import SimpleNamespace

    from gallops.train import swerve_data

    monkeypatch.setattr(swerve_data, "crop_resize", lambda path: f"cropped {path}")
    dataset = SimpleNamespace(_frames={}, _frame_cache=lambda: None)
    assert swerve_data.TickDataset.frame(dataset, "a.jpg") == "cropped a.jpg"
    assert dataset._frames == {"a.jpg": "cropped a.jpg"}
