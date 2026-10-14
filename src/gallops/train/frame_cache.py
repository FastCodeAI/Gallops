"""Every episode's frames, cropped and resized as the model reads them, on disk once.

`TickDataset.frame` decodes the full image (a 1.6 MB lossless PNG for the camera-fixed rounds) and
crops and resizes it to 512x256 on first use. Round 1A spent 1.4-2.4 s of a 2.5-3 s training step
there: shuffled ticks rarely share an episode, so the in-memory cache barely helps in the first
epoch, and across 733 episodes it would grow to about 20 GB of RAM.

This does the same crop and resize ahead of time, in parallel, into one uint8 array per episode
(`<root>/<episode key>.npy`, frames in timestamp order, with `<key>.stamps.npy` beside it).
`TickDataset.frame` memory-maps them when GALLOPS_FRAME_CACHE is set, and falls back to decoding
otherwise. The crop and resize are `TickDataset.frame`'s own (the vendor's `_get_camera_feature`), and
the training preflight's fidelity check compares what the trainer rebuilds against what the model saw.

    python -m gallops.train.frame_cache --labelled <dir> [--labelled <dir> ...] --out <root> --jobs 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def key(episode_dir: Path) -> str:
    """A stable file name for an episode directory."""
    return hashlib.sha1(str(episode_dir).encode()).hexdigest()[:20]


def crop_resize(path: Path) -> np.ndarray:
    import cv2
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    return cv2.resize(image[28:-28], (512, 256))


def build_episode(args: tuple[str, str]) -> str:
    directory, root = Path(args[0]), Path(args[1])
    out = root / f"{key(directory)}.npy"
    if out.exists():
        return f"cached {directory}"
    frames = sorted((directory / "CAM_FRONT").glob("*.jpg"), key=lambda p: int(p.stem))
    stamps = np.asarray([int(p.stem) for p in frames], dtype=np.int64)
    stack = np.stack([crop_resize(p) for p in frames]) if frames else np.zeros((0, 256, 512, 3), np.uint8)
    partial = root / f"{key(directory)}.partial.npy"
    np.save(partial, stack)
    np.save(root / f"{key(directory)}.stamps.npy", stamps)
    partial.rename(out)  # written last: a present .npy is always complete
    return f"wrote {directory} ({len(frames)} frames)"


class FrameCache:
    """Read side: one memory-mapped array per episode, looked up by frame path."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._open: dict[str, tuple[dict[int, int], np.ndarray]] = {}

    def get(self, path) -> np.ndarray | None:
        path = Path(path)
        directory = path.parent.parent
        name = key(directory)
        if name not in self._open:
            frames = self.root / f"{name}.npy"
            if not frames.exists():
                return None
            stamps = np.load(self.root / f"{name}.stamps.npy")
            self._open[name] = ({int(s): i for i, s in enumerate(stamps)}, np.load(frames, mmap_mode="r"))
        index, stack = self._open[name]
        i = index.get(int(path.stem))
        return None if i is None else np.asarray(stack[i])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labelled", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = set()
    for labelled in args.labelled:
        for path in sorted(labelled.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                frame = json.loads(line).get("frame")
                if frame:
                    episodes.add(str(Path(frame).parent.parent))
                    break
    print(f"  {len(episodes)} episodes", flush=True)
    with Pool(args.jobs) as pool:
        for i, message in enumerate(
            pool.imap_unordered(build_episode, [(e, str(args.out)) for e in sorted(episodes)])
        ):
            if i % 50 == 0:
                print(f"  {i + 1}/{len(episodes)}: {message}", flush=True)
    print("  done", flush=True)


if __name__ == "__main__":
    main()
