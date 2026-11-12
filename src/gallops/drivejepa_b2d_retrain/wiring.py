"""Paths, settings, and Sid's modules imported unchanged, for recipes/drivejepa_b2d_retrain.toml's stages.

Locations come from the environment, which the recipe sets from gallops.toml; nothing here names a machine.

Sid's lora.py, players.py, common.py, masks2.py and f_final.py are read from a checkout of this repository at
e75aaac (branch train/drivejepa-kitscenes, PR #24), the commit Phase 1 made its byte copies from: the five
sha256 are in experiments/H16_drivejepa_b2d_retrain/POINTERS.md. `players` imports `common`, whose setup()
would point navsim at Naren's KIT paths; as in Sid's PAI s1_build/src/pai_model.py:38-50 we set DEVKIT and
mark setup as done, after doing the same steps ourselves. Devkit = the copy the Bench2Drive runner imports
(upstream e21f474).
"""

import functools
import importlib
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True


def _site(name):
    """A location the recipe sets; unset, it stays an impossible path, so a bare import needs no site."""
    return Path(os.environ.get(name, f"/nonexistent/{name}"))


SPLITS = _site("B2D_SPLITS")  # recipes/drivejepa_b2d_retrain/clips.json
DEVKIT = _site("NAVSIM_DEVKIT_ROOT")  # {code.drivejepa}/navsim_v2
EXP = _site("NAVSIM_EXP_ROOT")  # the runner's copy of the released files
SID = _site("SID_KITSCENES")  # a checkout of this repository at e75aaac
SID_BUILD = SID / "experiments/drivejepa_kitscenes/02_encoder_build/src"  # lora, players, common
SID_TRAIN = SID / "experiments/drivejepa_kitscenes/03_encoder_training/src"  # masks2, f_final
AGENT_CKPT = EXP / "Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
E50 = EXP / "Drive-JEPA-cache/vitl_merge_3dataset_e50.pt"
AGENT_SHA = "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05"
E50_SHA = "4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f"
BASE = _site("B2D_CLIPS")  # b2d_base/ and b2d_pilot/, each <clip>/camera/rgb_front/NNNNN.jpg
CACHE = _site("B2D_FRAME_CACHE")  # 256 x 512 x 3 uint8
RUNS = _site("B2D_OUT") / "phase1"  # runs/, merged/, ade_set.pt; Phase 2 writes beside it, in ../phase2
VOCAB = _site("B2D_VOCAB")  # b2d_<K>.npy, paths_cache.npz
DATA = _site("B2D_DATA")  # context/, inputs/, grade_v2/, teachers_2048_v2.npz
ENC_PATH = "_pad_model._backbone.img_backbone"
SEED = 20261016  # Sid's
SETTINGS = dict(
    seed=SEED,
    batch=32,
    precision="bf16 autocast in training",
    pair_gap_files=1,
    stride_files=1,
    warmup_steps=1000,
    main_passes=3,
    ramp_steps=200,
    lr_lora=2e-4,
    wd_lora=0.0,
    lr_guesser=1e-4,
    wd_guesser=0.04,
    betas=(0.9, 0.999),
    eps=1e-8,
    grad_clip=None,
    momentum=0.99925,
    lora_rank=16,
    lora_alpha=16,
    resize_800x450="cv2.INTER_AREA",
)


def stub_mmcv_ext():
    """common._stub_mmcv_ext, kept for the case mmcv's compiled ops don't load; the sm_120 build here does."""
    try:
        importlib.import_module("mmcv._ext")
    except ImportError:

        class _Stub(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError(f"mmcv._ext.{name} is stubbed")

        sys.modules["mmcv._ext"] = _Stub("mmcv._ext")


_done = False


def setup():
    """Same steps as Sid's common.setup(), with our devkit. Returns (L, P, M): lora, players, masks2."""
    global _done
    os.environ["NAVSIM_EXP_ROOT"] = str(EXP)
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(DEVKIT)
    os.environ["DRIVE_JEPA_MSDA"] = "pytorch"
    for p in (str(SID_TRAIN), str(SID_BUILD), str(DEVKIT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(DEVKIT)
    stub_mmcv_ext()
    import torch

    if not getattr(torch.load, "_shim", False):
        original = torch.load

        @functools.wraps(original)
        def load(*a, **k):
            k.setdefault("weights_only", False)
            return original(*a, **k)

        load._shim = True
        torch.load = load
    import common as KC

    KC.DEVKIT = DEVKIT
    KC._SETUP_DONE = True
    import lora as L
    import masks2 as M
    import players as P

    _done = True
    return L, P, M


def build_agent(device="cuda"):
    """As the runner's DriveJepaPolicy.load (paddock's drivejepa adapter): released agent, frozen."""
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=str(AGENT_CKPT))
    agent.initialize()
    agent.eval().to(device)
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent
