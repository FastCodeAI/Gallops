"""Write the fixture that pins paddock's copy of the memory layer to gallops' (same inputs, same scores).

    <drive-jepa python> experiments/H15_prediction_memory/scripts/make_memory_fixture.py <out.npz>

A small layer (hidden 16) with every weight random, the zero-started last layer included, so the nudge
is not zero; three windows of five moments, with -1 matches, a padded window and a chain start.
"""

import sys

import numpy as np
import torch

from gallops.memory.model import EGO, EMBED, LOGITS, MemoryPicker

torch.manual_seed(20261105)
model = MemoryPicker(hidden=16)
for p in model.parameters():
    torch.nn.init.normal_(p, std=0.2)
model.eval()
rng = np.random.default_rng(20261105)
b, w, paths = 3, 5, 32
embed = rng.normal(size=(b, w, paths, EMBED)).astype(np.float32)
logits = rng.normal(size=(b, w, paths, LOGITS)).astype(np.float32)
ego = rng.normal(size=(b, w, EGO)).astype(np.float32)
match = rng.integers(-1, paths, size=(b, w, paths)).astype(np.int64)
match[:, 0] = -1
valid = np.ones((b, w), bool)
valid[1, :2] = False  # a window two moments shorter, left-padded
match[1, :3] = -1
args = [torch.as_tensor(x) for x in (embed, logits, ego, match, valid)]
with torch.no_grad():
    carry, nudge_carry = model(*args, mode="carry")
    reset, nudge_reset = model(*args, mode="reset")
state = {f"w:{k}": v.numpy() for k, v in model.state_dict().items()}
np.savez(
    sys.argv[1], embed=embed, logits=logits, ego=ego, match=match, valid=valid,
    scores_carry=carry.numpy(), scores_reset=reset.numpy(), **state,
)
print("nudge carry max", float(nudge_carry.abs().max()), "reset max", float(nudge_reset.abs().max()))
