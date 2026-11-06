"""Check 1 (b2d_drivejepa/vocab/CLAUDE.md): each file loads where Drive-JEPA loads data/8192.npy.

Runs the real DriveJEPAAgent.__init__ from Drive-JEPA-upstream with only the network stubbed out (no weights, no GPU), from a temporary
working directory whose data/8192.npy is a symlink to the file under test, because the path is hard-coded. Then, on the agent's own
self.anchors: navsim's Trajectory on every path, and the real trajectory_loss_anchors with random proposals and random indices.
Drive-JEPA's own file is run first as the reference. Nothing in Drive-JEPA is edited.

Writes (refuses to overwrite): check_load.json."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import torch

UPSTREAM = "{workspace_root}/wa-jepa/Drive-JEPA-upstream/navsim_v2"
sys.path.insert(0, UPSTREAM)
from navsim.agents.drive_jepa_perception_based import drive_jepa_agent as A        # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig   # noqa: E402
from navsim.common.dataclasses import Trajectory                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (256, 512, 1024, 2048, 4096, 8192)
FILES = [("navsim_8192", f"{UPSTREAM}/data/8192.npy")] + [(f"b2d_{k}", f"{HERE}/b2d_{k}.npy") for k in SIZES]
OUT = f"{HERE}/check_load.json"


def one(name, path):
    tmp = tempfile.mkdtemp(prefix="vocab_load_")
    cwd = os.getcwd()
    try:
        os.mkdir(f"{tmp}/data")
        os.symlink(path, f"{tmp}/data/8192.npy")
        os.chdir(tmp)
        cfg = DriveJEPAConfig()
        with mock.patch.object(A, "DriveJEPAModel", lambda c: torch.nn.Identity()):
            agent = A.DriveJEPAAgent(config=cfg, lr=1e-4, checkpoint_path="stub-so-init-skips-the-training-pool")
        raw = np.load("./data/8192.npy")
        anchors = agent.anchors
        for p in anchors:
            Trajectory(p)                                       # asserts 8 poses of (x, y, heading)
        rng = np.random.default_rng(0)
        props = [torch.randn(2, cfg.proposal_num, cfg.num_poses, 3)]
        target = torch.randn(2, cfg.num_poses, 3)
        idx = [rng.integers(0, len(anchors), 12), rng.integers(0, len(anchors), 3)]
        loss = agent.trajectory_loss_anchors(props, target, cfg, idx)[0]
        return dict(vocab=name, file_shape=list(raw.shape), file_dtype=str(raw.dtype), c_contiguous=bool(raw.flags["C_CONTIGUOUS"]),
                    all_finite=bool(np.isfinite(raw).all()), anchors_shape=list(anchors.shape), anchors_dtype=str(anchors.dtype),
                    trajectory_ok=len(anchors), loss_finite=bool(torch.isfinite(torch.as_tensor(loss)).all()))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp)


def main():
    if os.path.exists(OUT):
        sys.exit(f"STOP: {OUT} exists, not overwriting")
    rows = [one(n, p) for n, p in FILES]
    out = dict(rows=rows, module=A.__file__,
               pad_vjepa_module_present=importlib.util.find_spec("navsim.agents.pad_vjepa") is not None,
               get_sub_score_returns=5, calc_anchors_scores_unpacks=4)
    with open(OUT, "x") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
