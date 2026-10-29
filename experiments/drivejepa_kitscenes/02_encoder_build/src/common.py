"""Shared wiring for the encoder-LoRA stage 1 build.

Everything Naren owns (the Drive-JEPA devkit that produced the 2.364 m KIT
baseline, his KIT data roots, the baseline pickle) is imported or read, never
written.  Python must not drop __pycache__ into his tree, so bytecode writing is
switched off before anything from it is imported.

Interpreter: the conda drive-jepa python with the torch 2.8 + cu128 add-on on
PYTHONPATH and DRIVE_JEPA_MSDA=pytorch (Gallops
experiments/drivejepa_retrain/06_first_training_run/run_s4.sh:2-4).  The run.sh
next to src/ sets that up.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
OUT = BUILD / "out"

# Naren's devkit: the code that actually produced kitscenes_e2e_final.pkl
# (Paddock run_kitscenes_inference.py:53-58 imports from $NAVSIM_DEVKIT_ROOT).
DEVKIT = Path("{drivejepa_repo}/navsim_v2")
KIT_TOOLKIT = Path("{kitscenes_toolkit}")
SPLIT_DIR = KIT_TOOLKIT / "kitscenes/split/generated_splits/default_geo_split_v1_0"
SPLIT_FILES = {
    "train": "train.txt",
    "val": "validation.txt",
    "test": "test.txt",
    "test-e2e": "test-e2e.txt",
    "overlap_train_val": "overlap_train_val.txt",
}
# Every root holding full-sensor KIT Multimodal scenes.  p4 is left out on
# purpose: its scenes are pruned to the ring cameras + lidar_top (s0 recon, Q3a),
# and the nasbackup copy mirrors the NAS one.
KIT_ROOTS = [
    Path("{data.kitscenes}"),
    Path("{data.kitscenes.second_copy}"),
]
BASELINE_PKL = Path("{naren_navsim_workspace}/exp/inference/kitscenes_e2e_final.pkl")
BASELINE_PKL_SHA256 = "b40a14e967a750c8f116659cb1312e726e660b0171f159db714d702ad40dc3f3"

# Our own copies of the released weights (same bytes as Naren's, checked by hash).
NAVSIM_EXP_ROOT = Path("{weights}")
AGENT_CKPT = NAVSIM_EXP_ROOT / "Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
AGENT_CKPT_SHA256 = "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05"
E50 = NAVSIM_EXP_ROOT / "Drive-JEPA-cache/vitl_merge_3dataset_e50.pt"
E50_SHA256 = "4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f"

NAS_OUT = Path("{nas}/siddesh-physicalai/encoder_lora_kit/02_encoder_build")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _stub_mmcv_ext() -> None:
    """mmcv-full 1.7.2's compiled ops are built for torch 2.1 and cannot load
    under the 2.8 add-on, yet bevformer/encoder.py imports them at module load.
    Same stub as Gallops 05_small_overfit_check/train_patch/s35_common.py:33-49:
    every symbol exists and raises if called, so only the pure-PyTorch
    deformable attention can run."""
    try:
        importlib.import_module("mmcv._ext")
        return
    except ImportError:
        pass

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)

            def _missing(*args, **kwargs):
                raise RuntimeError(f"mmcv._ext.{name} is stubbed; DRIVE_JEPA_MSDA=pytorch is required")

            return _missing

    sys.modules["mmcv._ext"] = _Stub("mmcv._ext")


_SETUP_DONE = False


def setup() -> None:
    """Point imports at the devkit exactly as the baseline runner did."""
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    os.environ["NAVSIM_EXP_ROOT"] = str(NAVSIM_EXP_ROOT)
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(DEVKIT)
    # On this sm_120 card the devkit's own msda_compat.py:26-28 already picks the
    # PyTorch path under "auto"; pinning it removes any doubt.
    os.environ["DRIVE_JEPA_MSDA"] = "pytorch"
    if str(DEVKIT) not in sys.path:
        sys.path.insert(0, str(DEVKIT))
    # The agent opens ./data/8192.npy and the encoder ./vjepa2/configs/... relative
    # to cwd (run_kitscenes_inference.py:54).
    os.chdir(DEVKIT)
    _stub_mmcv_ext()

    import torch

    # torch >= 2.6 defaults torch.load to weights_only=True, which refuses the
    # Lightning checkpoint and the V-JEPA training state.  Both are the released
    # files, verified by sha256 before use.
    if not getattr(torch.load, "_s1_shim", False):
        original = torch.load

        @functools.wraps(original)
        def load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original(*args, **kwargs)

        load._s1_shim = True
        torch.load = load
    _SETUP_DONE = True

    import navsim

    assert Path(navsim.__file__).resolve().is_relative_to(DEVKIT), navsim.__file__


def devkit_module(relpath: str, name: str):
    """Import a devkit file by path.  The devkit's scripts/ has no __init__.py,
    and this env also installs an unrelated regular package called ``scripts``
    (the AlpaSim gRPC tree), which wins over a namespace package; importing by
    path gets the exact file the baseline ran."""
    import importlib.util

    setup()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DEVKIT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def kit_adapter_module():
    return devkit_module("scripts/inference/kitscenes_adapter.py", "devkit_kitscenes_adapter")


def kit_runner_module():
    """The baseline runner (devkit scripts/inference/run_kitscenes_inference.py),
    loaded unchanged.  It does `from scripts.inference.kitscenes_adapter import ...`
    (:56-59); the names are pre-registered so that import resolves to the devkit
    file rather than the unrelated `scripts` package installed in this env."""
    adapter = kit_adapter_module()
    for name in ("scripts", "scripts.inference"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(DEVKIT / name.replace(".", "/"))]
        sys.modules[name] = pkg
    sys.modules["scripts.inference.kitscenes_adapter"] = adapter
    sys.modules["scripts.inference"].kitscenes_adapter = adapter
    return devkit_module("scripts/inference/run_kitscenes_inference.py", "devkit_run_kitscenes_inference")


def gpu_free_mib() -> int:
    import torch

    free, _ = torch.cuda.mem_get_info()
    return free // (1 << 20)


def require_gpu(needed_mib: int, step: str) -> bool:
    """True if the shared card has room for *step*; prints BLOCKED-GPU otherwise."""
    free = gpu_free_mib()
    if free < needed_mib:
        print(f"BLOCKED-GPU {step}: {free} MiB free < {needed_mib} MiB needed")
        return False
    print(f"GPU check {step}: {free} MiB free >= {needed_mib} MiB needed")
    return True
