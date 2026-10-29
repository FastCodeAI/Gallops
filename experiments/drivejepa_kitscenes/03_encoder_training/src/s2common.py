"""Stage 2 wiring: paths, the decided settings, and the shared-GPU wait.

Stage 1 code (02_encoder_build/src) is imported unchanged; run.sh puts it on PYTHONPATH.
Its common.OUT still points at 02_encoder_build/out, which we only read (splits.json,
the kit_root symlinks f_baseline.build_root re-asserts).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import common as C  # 02_encoder_build/src/common.py

HERE = Path(__file__).resolve().parent
S2 = HERE.parent
OUT = S2 / "out"
LOGS = S2 / "logs"

# Large files live on the local NVMe (/ has ~71 GB free; {p4} is root-owned,
# no directory there is writable by uid 1033).  out/DATA_POINTER.txt records it.
DATA = Path("{runs}/drivejepa_kitscenes/encoder_lora_kit/s2")
CACHE = Path(os.environ.get("S2_CACHE", DATA / "frame_cache"))  # override only for smoke tests
CKPT = DATA / "checkpoints"
SEL_FEATS = DATA / "selection_features.pt"

SEED = 20261016
FRAME_STEP = 5  # every 5th frame; a pair is (t-5, t), t % 5 == 0, t >= 5

SETTINGS = {
    "seed": SEED,
    "frames": "every 5th frame; pair = (t-5, t) with t a multiple of 5 and t >= 5",
    "lora": {"rank": 16, "alpha": 16, "targets": "blocks.N.{attn.qkv,attn.proj,mlp.fc1,mlp.fc2}, N=0..23",
             "trainable": 6291456, "dropout": 0.0},
    "learner": "agent encoder (sha256 740335c8...0d05) + LoRA",
    "answer_key": "copy of learner; only its LoRA tensors move, key <- m*key + (1-m)*learner each main step",
    "momentum": 0.99925,
    "guesser": "e50 predictor (sha256 46491827...668f) with the s1 grid fix (players.Guesser)",
    "optimizer": {
        "type": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
        "lora": {"lr": 2e-4, "weight_decay": 0.0},
        "guesser": {"lr": 1e-4, "weight_decay": 0.04,
                    "weight_decay_excludes": "biases and 1-D tensors (upstream init_opt, vjepa2/app/vjepa/utils.py:224-236)"},
        "schedule": "linear ramp from 0 over the first 200 steps of each phase, then constant; "
                    "each phase builds a fresh optimiser",
        "ramp_steps": 200,
        "grad_clip": None,
    },
    "precision": "bf16 autocast for training; fp32 for driving inference",
    "activation_checkpointing": "encoder from in1k.yaml (as s1), guesser use_activation_checkpointing=True",
    "batch": 32,
    "loss": "s1 players.game_loss: answer-key output, layer-normed, at hidden positions, L1, mean over the 2 mask configs",
    "masks": "see masks2.py (Part B)",
    "warmup_stop": "fixed-set loss improves < 1 % over the last 500 steps (checked every 250), or 3,000 steps",
    "main_stop": "selection ADE worse than best for 3 passes in a row, or 10 passes",
}


def gpu_free_mib() -> int:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return int(q.strip().splitlines()[0])


def wait_for_gpu(needed_mib: int, step: str, log=print, every_s: int = 600) -> float:
    """Block until the shared card has needed_mib free; retry every 10 min.
    Returns the seconds waited."""
    t0 = time.time()
    while True:
        free = gpu_free_mib()
        if free >= needed_mib:
            log(f"GPU check {step}: {free} MiB free >= {needed_mib}; waited {time.time() - t0:.0f} s")
            return time.time() - t0
        log(f"BLOCKED-GPU {step}: {free} MiB free < {needed_mib}; retry in {every_s} s")
        time.sleep(every_s)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(path)


__all__ = ["C", "OUT", "LOGS", "DATA", "CACHE", "CKPT", "SEL_FEATS", "SEED", "FRAME_STEP", "SETTINGS",
           "gpu_free_mib", "wait_for_gpu", "write_json"]
