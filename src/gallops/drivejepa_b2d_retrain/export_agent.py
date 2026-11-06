"""Phase 2 training state -> the checkpoint form DriveJEPAAgent.initialize() reads ({"state_dict":
{"agent._pad_model.*"}}). New file beside the save; the state is only read.
    python -m gallops.drivejepa_b2d_retrain.export_agent <epoch>"""

import hashlib
import os
import sys

import torch

from gallops.drivejepa_b2d_retrain import wiring as W

e = int(sys.argv[1])
d = str(W.RUNS.parent / f"phase2/runs/p2_05nov/ckpt_epoch{e:02d}")
out = f"{d}/agent.ckpt"
ck = torch.load(f"{d}/state.pt", map_location="cpu", weights_only=False)
sd = {f"agent._pad_model.{k}": v for k, v in ck["model"].items()}

assert not os.path.exists(out)
torch.save(dict(state_dict=sd, epoch=e, source="phase2 p2_05nov", state=ck["state"]), out)
h = hashlib.sha256()
f = open(out, "rb")
for b in iter(lambda: f.read(1 << 24), b""):
    h.update(b)
print(out, len(sd), "tensors", os.path.getsize(out), h.hexdigest())
open(f"{d}/agent.ckpt.sha256", "w").write(h.hexdigest() + "  agent.ckpt\n")
