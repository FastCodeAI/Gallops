"""Our copy of Sid's frame cache reader and fixed sets (E1t/src/cache.py, evalsets.py). Changes: the source is
the Bench2Drive `native` cache from build_cache.py, pair gap and stride are two settings (Sid's FRAME_STEP
was both), and the ImageNet normalise is the model's own transform."""

import json

import numpy as np
import torch
from torchvision import transforms

from gallops.drivejepa_b2d_retrain import wiring as W

H, Wd = 256, 512
_to_tensor = transforms.ToTensor()
_norm = None


def normalise(t):
    global _norm
    if _norm is None:
        from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel

        _norm = DriveJEPAModel.make_transform(None)
    return _norm(t)


def pair_from_arrays(older, newer):
    return torch.stack([normalise(_to_tensor(older)), normalise(_to_tensor(newer))], dim=0)


class FrameCache:
    def __init__(self, root=W.CACHE):
        self.root = root
        self.frames = json.load(open(root / "index.json"))["frames"]
        self._maps = {}

    def clip(self, c):
        if c not in self._maps:
            self._maps[c] = np.memmap(
                self.root / f"{c}.bin", dtype=np.uint8, mode="r", shape=(len(self.frames[c]), H, Wd, 3)
            )
        return self._maps[c]

    def frame(self, c, i):
        return np.array(self.clip(c)[i])


def pair_items(clips, frames, gap, stride):
    return [(c, i) for c in clips for i in range(gap, len(frames[c]), stride)]


class CachedPairs(torch.utils.data.Dataset):
    def __init__(self, clips, gap=1, stride=1):
        self.gap = gap
        self.items = pair_items(clips, json.load(open(W.CACHE / "index.json"))["frames"], gap, stride)
        self._c = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        if self._c is None:
            self._c = FrameCache()
        c, i = self.items[k]
        return pair_from_arrays(self._c.frame(c, i - self.gap), self._c.frame(c, i)), k


class FixedSet:
    """Sid's fixed-set loss: fixed samples, fixed masks (step_rng(SEED, 7, b)), eval mode, bf16. n pairs
    evenly spaced over the listed clips."""

    def __init__(self, clips, P, M, n=2000, batch=32, gap=1, stride=1):
        self.P, self.M, self.batch, self.gap = P, M, batch, gap
        fc = FrameCache()
        items = pair_items(clips, fc.frames, gap, stride)
        idx = np.linspace(0, len(items) - 1, min(n, len(items))).astype(int)
        self.items = [items[i] for i in idx]
        self.fc = fc
        gens = M.make()
        self.masks = [
            M.sample(gens, min(batch, len(self.items) - b * batch), M.step_rng(W.SEED, 7, b))
            for b in range((len(self.items) + batch - 1) // batch)
        ]

    @torch.no_grad()
    def loss(self, learner, key, guesser):
        was = learner.training, guesser.training
        learner.eval()
        guesser.eval()
        tot = n = 0
        for b, masks in enumerate(self.masks):
            chunk = self.items[b * self.batch : (b + 1) * self.batch]
            x = torch.stack(
                [pair_from_arrays(self.fc.frame(c, i - self.gap), self.fc.frame(c, i)) for c, i in chunk]
            ).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = self.P.game_loss(learner, key, guesser, x, [(e.cuda(), p.cuda()) for e, p in masks])
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
        learner.train(was[0])
        guesser.train(was[1])
        return tot / n
