"""The prediction memory: a per-path state carried across moments that nudges the released score.

Per path i at moment t (plan :232-237), with e_i = the scorer's input for that path (256), l_i = the
released scorer's six logits and g = the ego status (11):

    x_i(t)   = MLP_in([e_i ; l_i ; g])                        -> 256
    keep_i   = sigmoid(W x_i(t) + b)                           per channel
    h_i(t)   = keep_i * h_{m(i)}(t - 1) + (1 - keep_i) * x_i(t)
    nudge_i  = MLP_out([e_i ; h_i(t)])                         last layer starts at zero
    score_i  = l_i[final] + nudge_i

m(i) is the previous moment's path this one inherits from (`align.match`); the inherited state is 0
when m(i) = -1 (no old path within 2 m), at a chain start, and always in **reset** mode, which is
the same layer with nothing carried (H19's arm (b)).

**Length L (carry mode).** h(t) is computed from the current moment and at most L earlier moments of
its chain: the state handed into moment t - L is zero, so anything from t - L - 1 or earlier cannot
reach h(t). (`window_rows` builds that window; `test_memory_model.py` checks that changing moment
t - L - 1 leaves the score unchanged and changing t - L does not.) With L = 1 the memory is the
previous moment's own state, with L = 4 the last two seconds.

Because the last layer starts at zero, an untrained model scores every path exactly as the released
model does, so its picks are the released picks; the trainer's preflight checks that on val moments.
"""

from __future__ import annotations

import numpy as np

try:  # torch only in the training interpreter; the window logic is importable without it
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

EMBED, LOGITS, EGO, HIDDEN = 256, 6, 11, 256
MODES = ("carry", "reset")


def window_rows(prev: np.ndarray, row: int, earlier: int) -> list:
    """The moments of `row`'s window, oldest first: `row` and up to `earlier` moments before it."""
    rows = [int(row)]
    while len(rows) <= earlier and prev[rows[0]] >= 0:
        rows.insert(0, int(prev[rows[0]]))
    return rows


def batch_windows(prev: np.ndarray, match: np.ndarray, targets, earlier) -> tuple[np.ndarray, ...]:
    """Left-padded windows for a batch: rows (B, W), match (B, W, 32), valid (B, W).

    `earlier` is an int or one per target. W = 1 + the largest `earlier`. match[b, s] maps each path at
    step s to its path at step s - 1 (-1: none); it is -1 at a window's first step, so nothing is
    inherited into it.
    """
    earlier = np.broadcast_to(np.asarray(earlier), (len(targets),))
    width = int(earlier.max()) + 1 if len(targets) else 1
    rows = np.zeros((len(targets), width), np.int64)
    valid = np.zeros((len(targets), width), bool)
    m = np.full((len(targets), width, match.shape[1]), -1, np.int64)
    for b, (t, e) in enumerate(zip(targets, earlier)):  # noqa: B905 -- broadcast above
        w = window_rows(prev, t, int(e))
        start = width - len(w)
        rows[b, start:] = w
        valid[b, start:] = True
        for s in range(1, len(w)):
            m[b, start + s] = match[w[s]]
        rows[b, :start] = w[0]  # padding repeats the first row; `valid` masks it
    return rows, m, valid


if nn is not None:

    class MemoryPicker(nn.Module):
        """See the module docstring. Inputs are per step of a window, oldest first."""

        def __init__(self, hidden: int = HIDDEN):
            super().__init__()
            self.inp = nn.Sequential(
                nn.Linear(EMBED + LOGITS + EGO, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            self.keep = nn.Linear(hidden, hidden)
            self.out = nn.Sequential(nn.Linear(EMBED + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

        def step_input(self, embed, logits, ego):
            """x_i: (..., 32, hidden) from (..., 32, 256), (..., 32, 6), (..., 11)."""
            ego = ego.unsqueeze(-2).expand(*embed.shape[:-1], ego.shape[-1])
            return self.inp(torch.cat([embed, logits, ego], -1))

        def forward(self, embed, logits, ego, match, valid, mode: str):
            """Scores (B, 32) at each window's last step, and the nudge.

            embed (B, W, 32, 256), logits (B, W, 32, 6), ego (B, W, 11), match (B, W, 32) long, valid
            (B, W) bool. In reset mode only the last step is used: nothing reaches it from before.
            """
            if mode not in MODES:
                raise ValueError(f"mode {mode!r}; expected one of {MODES}")
            steps = range(embed.shape[1]) if mode == "carry" else [embed.shape[1] - 1]
            b, _, paths, _ = embed.shape
            h = embed.new_zeros(b, paths, self.keep.out_features)
            for s in steps:
                x = self.step_input(embed[:, s], logits[:, s], ego[:, s])
                keep = torch.sigmoid(self.keep(x))
                if mode == "carry" and s > 0:
                    m = match[:, s]
                    inherited = torch.gather(h, 1, m.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
                    inherited = inherited * (m >= 0).unsqueeze(-1).to(h.dtype)
                else:
                    inherited = torch.zeros_like(x)
                h_new = keep * inherited + (1 - keep) * x
                h = torch.where(valid[:, s].view(-1, 1, 1), h_new, torch.zeros_like(h_new))
            nudge = self.out(torch.cat([embed[:, -1], h], -1)).squeeze(-1)
            return logits[:, -1, :, -1] + nudge, nudge
