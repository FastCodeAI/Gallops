"""Drive-JEPA's environment, entered the way paddock's model.yaml enters it for its node.

`navsim` is not installed in Drive-JEPA's interpreter at this site; it is imported from
`{code.drivejepa}/navsim_v2`, which goes first on sys.path. That tree must carry paddock's patch
0001: on GPUs above sm_86 mmcv's compiled deformable-attention kernel prints "no kernel image is
available", does not raise, and returns output as large as the signal it should have computed
(measured 0.989 off a reference whose largest value is 0.989, H15 S0.2). `DRIVE_JEPA_MSDA=auto`
then routes it to the pure-PyTorch path. `enter` records which `navsim` was imported, so a run's
sidecar names the code that ran rather than the code that was configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The released checkpoint, where paddock's model.yaml finds it.
RELEASE = "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"


def enter(cfg) -> dict:
    """Set Drive-JEPA's environment variables, sys.path and cwd; return what was entered.

    The cwd moves into navsim_v2/ because the agent opens ./data/8192.npy relative to it. Returns
    the resolved paths and the file `navsim` was imported from, for provenance.
    """
    ext = Path(cfg.resolve("{code.drivejepa}"))
    navsim_v2 = ext / "navsim_v2"
    navsim_root = cfg.resolve("{data.navsim}")
    os.environ.update(
        NAVSIM_DEVKIT_ROOT=str(navsim_v2),
        NAVSIM_EXP_ROOT=cfg.resolve("{data.navsim.exp}"),
        OPENSCENE_DATA_ROOT=navsim_root,
        NUPLAN_MAPS_ROOT=navsim_root + "/maps",
        NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
        DRIVE_JEPA_MSDA="auto",
    )
    if str(navsim_v2) not in sys.path:
        sys.path.insert(0, str(navsim_v2))
    os.chdir(navsim_v2)
    import navsim

    imported = str(Path(navsim.__file__).resolve())
    if not imported.startswith(str(navsim_v2.resolve())):
        raise SystemExit(f"navsim imported from {imported}, not from {navsim_v2}: fix PYTHONPATH")
    return {
        "code": str(ext),
        "code_sha": _git_head(ext),
        "navsim": imported,
        "navsim_root": navsim_root,
        "release": cfg.resolve(RELEASE),
        "msda": os.environ["DRIVE_JEPA_MSDA"],
    }


def _git_head(repo: Path) -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"
