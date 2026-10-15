"""Drive-JEPA's scorer head as a standalone module, and the AUC the head stages select on.

`scorer.pred_score` is an MLP(256, 1024, 6) over each of the 32 proposals'
pooled features. Column 5 is the only one the released model drives on
(`pdm_score = sigmoid(logit)[..., 5]`, pick = argmax); the split heads train
columns 0-3 as col / lane / ttc / red and leave 4 and 5 untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

#: The 10 head tensors, matched by suffix: the key prefix differs between the
#: Lightning checkpoint (`agent._pad_model.`) and paddock's loaded model.
HEAD_SUFFIXES = [f"scorer.pred_score.mlp.{i}.{p}" for i in (0, 1, 3, 4, 6) for p in ("weight", "bias")]


class Head(nn.Module):
    """Identical to Drive-JEPA's MLP(256, 1024, 6) (bevformer/transformer_decoder.py), including the `mlp.`
    prefix of its state-dict keys (mlp.{0,1,3,4,6}.{weight,bias}), so tensors move between here and the
    checkpoint unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 6),
        )

    def forward(self, x):
        return self.mlp(x)


def head_module() -> Head:
    return Head()


def head_tensors_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """The 10 head tensors of a checkpoint, keyed 'mlp.<i>.<p>'."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    out = {}
    for suffix in HEAD_SUFFIXES:
        hits = [k for k in sd if k.endswith(suffix)]
        if len(hits) != 1:
            raise SystemExit(f"{suffix}: expected exactly one key, found {hits}")
        out[suffix.split("scorer.pred_score.")[1]] = sd[hits[0]].clone().float()
    return out


def auc(s, y) -> float:
    """Rank AUC of scores `s` for binary labels `y` (ties broken by position, as every head stage was run)."""
    s, y = s.ravel(), y.ravel().astype(bool)
    if y.all() or not y.any():
        return float("nan")
    rk = np.argsort(np.argsort(s)).astype(np.float64) + 1
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
