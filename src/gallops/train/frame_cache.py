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
import time
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


#: Attempts at reading one cached file. Round 1B died at 93% of its last epoch on one
#: `OSError: [Errno 5] Input/output error` from the NAS; the same file read cleanly minutes later.
READ_ATTEMPTS = 5


def _load(path: Path, attempts: int = READ_ATTEMPTS, wait_s: float = 5.0) -> np.ndarray:
    """np.load, retried on OSError with a growing wait; the last failure is raised."""
    for attempt in range(attempts):
        try:
            return np.load(path)
        except OSError as error:
            if attempt == attempts - 1:
                raise
            print(f"  frame cache: {error}; retry {attempt + 1} of {attempts - 1}", flush=True)
            time.sleep(wait_s * (attempt + 1))
    raise AssertionError("unreachable")


class FrameCache:
    """Read side: each episode's array read whole on first use, the last `keep` episodes held in RAM.

    Memory-mapping the arrays off the NAS was slower than decoding (4.5 s a tick against 1.4 s on
    frontal-0070): every frame was a random network read. One sequential read per episode, about
    27 MB, is not.
    """

    def __init__(self, root: Path, keep: int = 64) -> None:
        from collections import OrderedDict

        self.root = Path(root)
        self.keep = keep
        self._open: OrderedDict[str, tuple[dict[int, int], np.ndarray] | None] = OrderedDict()

    def get(self, path) -> np.ndarray | None:
        path = Path(path)
        name = key(path.parent.parent)
        if name in self._open:
            self._open.move_to_end(name)
        else:
            frames = self.root / f"{name}.npy"
            if frames.exists():
                stamps = _load(self.root / f"{name}.stamps.npy")
                self._open[name] = ({int(s): i for i, s in enumerate(stamps)}, _load(frames))
            else:
                self._open[name] = None
            while len(self._open) > self.keep:
                self._open.popitem(last=False)
        entry = self._open[name]
        if entry is None:
            return None
        index, stack = entry
        i = index.get(int(path.stem))
        return None if i is None else stack[i]


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
