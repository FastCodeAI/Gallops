"""H15's prediction memory for Drive-JEPA: a per-path state carried across moments that nudges the score.

Transcribed from gallops (`gallops.memory.model` and `gallops.memory.align`, branch
train/memory-picker-h15 at cdacc38), which trained the weights. Neither repo imports the other, so the
two copies are pinned to each other by a fixture: the same inputs give the same scores to 1e-6
(`tests/test_memory.py`). Active only when ``DRIVEJEPA_MEMORY`` names a checkpoint directory; empty,
the node is the released one bit for bit.

Per path i at a moment t, with e_i the scorer's input for that path (the last refiner pass's query
features, max over the path's 8 waypoints, 256 numbers), l_i the released scorer's six logits and g the
ego status the model was given (11 numbers):

    x_i(t)  = MLP_in([e_i ; l_i ; g])
    keep_i  = sigmoid(W x_i(t) + b)
    h_i(t)  = keep_i * h_m(i)(t - 0.5 s) + (1 - keep_i) * x_i(t)
    nudge_i = MLP_out([e_i ; h_i(t)])
    score_i = l_i[final] + nudge_i

m(i) is the previous moment's path closest to path i once the previous moment's proposals are moved
into the current ego frame (`align_previous`, `distances`, `match`), or none beyond 2 m. The momentum
stage, when it is on, then acts on the nudged score.

**By timestamp.** The previous moment is the one 0.5 s +/- 0.05 s earlier, the spacing the memory was
trained at, found the way the node finds its history frames and its momentum plan. A node called at
10 Hz therefore keeps five interleaved chains, each 0.5 s apart. A gap with nothing 0.45-0.55 s back
starts a new chain; a clock that goes backwards clears everything (a new episode without a reset).

**Length L** (from `checkpoint.json`): h(t) is computed from moment t and at most L earlier moments of
its chain, the state handed into t - L being zero, exactly as trained. The window is recomputed at
every moment from the stored inputs, which costs L + 1 small passes.

**Reset mode** (H15's arm (b)): the same layer with nothing carried.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ENV = "DRIVEJEPA_MEMORY"
WEIGHTS = "memory.pt"
MODEL_NAME = "drivejepa_memory"
EMBED, LOGITS, EGO, HIDDEN = 256, 6, 11, 256
MODES = ("carry", "reset")
LIMIT_M = 2.0  #: beyond it a new path starts with an empty memory (H15 PREDICTIONS.md)
TARGET_US, MIN_AGE_US, MAX_AGE_US = 500_000, 450_000, 550_000


class MemoryCheckpointError(RuntimeError):
    """The weights directory cannot be used; the message names the file."""


# --- geometry (gallops.memory.align) ---------------------------------------------------------------


def align_previous(previous: np.ndarray, pose_previous: np.ndarray, pose_now: np.ndarray) -> np.ndarray:
    """(32, 7, 2): the previous proposals' waypoints 2-8, in the current ego frame."""
    relative = np.linalg.inv(pose_previous) @ pose_now
    c, s = relative[0, 0], relative[1, 0]
    tx, ty = relative[0, 3], relative[1, 3]
    points = np.asarray(previous, float)[:, 1:, :2]
    dx, dy = points[..., 0] - tx, points[..., 1] - ty
    return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)


def distances(aligned_previous: np.ndarray, now: np.ndarray) -> np.ndarray:
    """(old 32, new 32) mean distance over the 3.5 s the two share (old 2-8 against new 1-7)."""
    current = np.asarray(now, float)[:, :7, :2]
    return np.linalg.norm(aligned_previous[:, None] - current[None], axis=-1).mean(-1)


def match(d: np.ndarray, limit: float = LIMIT_M) -> np.ndarray:
    """For each new path, the closest old path; -1 beyond `limit`."""
    best = d.argmin(0)
    best_m = d[best, np.arange(d.shape[1])]
    return np.where(best_m <= limit, best, -1).astype(np.int64)


# --- the layer (gallops.memory.model) --------------------------------------------------------------


def build_model(hidden: int = HIDDEN):
    """The memory layer as gallops trained it. torch is imported here: paddock's env has none."""
    import torch
    from torch import nn

    class MemoryPicker(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.inp = nn.Sequential(
                nn.Linear(EMBED + LOGITS + EGO, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            self.keep = nn.Linear(hidden, hidden)
            self.out = nn.Sequential(nn.Linear(EMBED + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

        def step_input(self, embed, logits, ego):
            ego = ego.unsqueeze(-2).expand(*embed.shape[:-1], ego.shape[-1])
            return self.inp(torch.cat([embed, logits, ego], -1))

        def forward(self, embed, logits, ego, match, valid, mode: str):
            """Scores (B, 32) at each window's last step, and the nudge (gallops' forward, verbatim)."""
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

    return MemoryPicker(hidden)


def load(directory: Path | str, device: str = "cpu") -> Memory:
    """Read a gallops memory checkpoint directory, refusing one without its `checkpoint.json`."""
    import torch

    directory = Path(directory)
    weights = directory / WEIGHTS
    if not weights.exists():
        raise MemoryCheckpointError(f"no memory weights at {weights}")
    sidecar_path = directory / "checkpoint.json"
    if not sidecar_path.exists():
        raise MemoryCheckpointError(
            f"no checkpoint.json beside {weights}; copy the whole checkpoint directory, since the "
            "sidecar records the memory's mode and length and what it was trained on"
        )
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("model") != MODEL_NAME:
        raise MemoryCheckpointError(
            f"{sidecar_path}: model {sidecar.get('model')!r}, expected {MODEL_NAME!r}"
        )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != sidecar.get("weights_sha256"):
        raise MemoryCheckpointError(
            f"{weights}: sha256 {digest[:12]} but the sidecar records {sidecar.get('weights_sha256')}"
        )
    notes = sidecar.get("notes", {})
    mode, length = notes.get("mode"), notes.get("length")
    if mode not in MODES or not isinstance(length, int):
        raise MemoryCheckpointError(
            f"{sidecar_path}: notes.mode {mode!r} / notes.length {length!r} unreadable"
        )
    state = torch.load(weights, map_location="cpu")
    model = build_model(int(state["keep.weight"].shape[0]))
    model.load_state_dict(state)
    model.eval().to(device)
    return Memory(model, mode, length, digest)


# --- state by timestamp ----------------------------------------------------------------------------


class Memory:
    """Per-moment inputs kept by timestamp; one `step` per moment the model plans or observes."""

    def __init__(self, model, mode: str, length: int, weights_sha256: str = "") -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}; expected one of {MODES}")
        self.model, self.mode, self.weights_sha256 = model, mode, weights_sha256
        # Reset mode carries nothing, so its window is the moment alone.
        self.length = int(length) if mode == "carry" else 0
        self.reset()

    def reset(self) -> None:
        """Forget every moment. The counts are kept (like momentum's), so a run's record covers it all."""
        self.entries: list[dict[str, Any]] = []  # recent moments, oldest first
        if not hasattr(self, "counts"):
            self.counts = {
                "resets": 0,
                "steps": 0,
                "chains": 0,
                "clock_jumps": 0,
                "paths": 0,
                "unmatched_paths": 0,
            }
        else:
            self.counts["resets"] += 1

    def _previous(self, ts: int) -> dict[str, Any] | None:
        candidates = [e for e in self.entries if MIN_AGE_US <= ts - e["ts"] <= MAX_AGE_US]
        return min(candidates, key=lambda e: abs(ts - e["ts"] - TARGET_US)) if candidates else None

    def step(self, ts: int, pose: np.ndarray, proposals: np.ndarray, embed, logits, ego):
        """The nudge (32,) for this moment; `embed` (32, 256), `logits` (32, 6), `ego` (11,) tensors."""
        import torch

        ts = int(ts)
        if self.entries and ts <= self.entries[-1]["ts"]:
            self.entries = []
            self.counts["clock_jumps"] += 1
        prev = self._previous(ts)
        if prev is None:
            m = np.full(proposals.shape[0], -1, np.int64)
            self.counts["chains"] += 1
        else:
            m = match(distances(align_previous(prev["proposals"], prev["pose"], pose), proposals))
            self.counts["paths"] += m.size
            self.counts["unmatched_paths"] += int((m < 0).sum())
        entry = {
            "ts": ts,
            "pose": np.asarray(pose, float),
            "proposals": np.asarray(proposals, float),
            "embed": embed,
            "logits": logits,
            "ego": ego,
            "prev": prev,
            "match": m,
        }
        window = [entry]
        while len(window) <= self.length and window[0]["prev"] is not None:
            window.insert(0, window[0]["prev"])
        # Nothing older than this window's first moment is ever read again: a later window starts later.
        window[0]["prev"] = None
        matches = np.stack([np.full_like(m, -1)] + [w["match"] for w in window[1:]])
        device = embed.device
        with torch.no_grad():
            _, nudge = self.model(
                torch.stack([w["embed"] for w in window])[None].float(),
                torch.stack([w["logits"] for w in window])[None].float(),
                torch.stack([w["ego"] for w in window])[None].float(),
                torch.as_tensor(matches, device=device)[None],
                torch.ones(1, len(window), dtype=torch.bool, device=device),
                self.mode,
            )
        self.entries = [e for e in self.entries if ts - e["ts"] <= MAX_AGE_US] + [entry]
        self.counts["steps"] += 1
        return nudge[0]

    def diagnostics(self) -> dict[str, Any]:
        c = self.counts
        return {
            "mode": self.mode,
            "length": self.length,
            "weights_sha256": self.weights_sha256,
            **c,
            "unmatched_fraction": round(c["unmatched_paths"] / c["paths"], 4) if c["paths"] else None,
        }
