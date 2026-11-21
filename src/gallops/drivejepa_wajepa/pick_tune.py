"""Stage run3a-pick-tune: Run 3A pick tuning (paddock-ops/run3a/CLAUDE.md, "Pick"): on the chosen checkpoint,
cache the 12 head outputs of the 32 proposals for every held-out sample and the training scorer's labels
(today's final label = EPDMS-style score), then grid w_g in {0.5, 1, 2} (5 gates) x route r in {0, 2, 5} x
alpha in {0, 0.5} (progress 5, TTC 5, lane keeping 2, comfort 2 fixed); report the default (all 1, r 2, alpha
0) and the best, which is then locked. Writes <out>/heads_labels.npz (refuses to overwrite) and
<out>/grid.json. 25 Oct, epoch 13: default 0.9174, best gates (0.5, 0.5, 1, 0.5, 0.5), route 2, alpha 0.5 =
0.9204; locked in the overlay's drive_jepa_config.py.

Ported from paddock-ops run3a/build/pick_tune.py; changed: this docstring, and the token cache comes from
WAJEPA_TOKEN_CACHE (as run, a constant naming that machine's path).

    python -m gallops.drivejepa_wajepa.pick_tune <ckpt> <out_dir> [workers]
(Drive-JEPA's interpreter, cwd = the Run 3A tree's navsim_v2, PYTHONPATH = the same, NAVSIM_EXP_ROOT set.)"""

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from navsim.agents.drive_jepa_run3a.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_run3a.drive_jepa_config import DriveJEPAConfig
from navsim.agents.drive_jepa_run3a.score_module.compute_navsim_score import get_scores
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.training.dataset import CacheOnlyDataset
from torch.utils.data import DataLoader

ckpt, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
os.makedirs(out, exist_ok=True)
npz = f"{out}/heads_labels.npz"
cfg = DriveJEPAConfig()
if not os.path.exists(npz):
    agent = DriveJEPAAgent(cfg, lr=1e-4, checkpoint_path=ckpt)
    agent.initialize()
    agent.cuda().eval()
    split = yaml.safe_load(open("navsim/planning/script/config/training/run3a_log_split.yaml"))
    ds = CacheOnlyDataset(
        os.environ["WAJEPA_TOKEN_CACHE"],
        agent.get_feature_builders(),
        agent.get_target_builders(),
        log_names=split["val_logs"],
    )
    paths = MetricCacheLoader(
        Path(os.environ["NAVSIM_EXP_ROOT"] + "/Drive-JEPA-cache/train_metric_cache_v2_pruned")
    ).metric_cache_paths
    logits, labels, tokens = [], [], []
    with ProcessPoolExecutor(workers) as pool:
        for features, targets in DataLoader(ds, batch_size=64, shuffle=False, num_workers=8):
            n = len(targets["token"])
            if n < 64:  # deformable-attention kernel: batch must divide im2col_step; pad, keep the first n
                features = {
                    k: (torch.cat([v, v[:1].expand(64 - n, *v.shape[1:])]) if torch.is_tensor(v) else v)
                    for k, v in features.items()
                }
            features = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in features.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pred = agent.forward(features)
            props = pred["proposals"][:n].float().cpu().numpy()
            pts = [{"token": paths[t], "poses": p, "test": True} for t, p in zip(targets["token"], props)]  # noqa: B905
            res = [
                r
                for f in [pool.submit(get_scores, pts[i : i + 2]) for i in range(0, len(pts), 2)]
                for r in f.result()
            ]
            logits.append(pred["pred_logit"][:n].float().cpu().numpy())
            labels.append(np.stack([r[0] for r in res]))
            tokens += list(targets["token"])
            print(len(tokens), "samples", flush=True)
    np.savez(
        npz, logits=np.concatenate(logits), labels=np.concatenate(labels), tokens=np.array(tokens), ckpt=ckpt
    )
d = np.load(npz)
z = torch.tensor(d["logits"], device="cuda")
lab = torch.tensor(d["labels"][..., -1], device="cuda")  # today's final label, unmasked
logsig, sig = torch.nn.functional.logsigmoid(z), torch.sigmoid(z)
rows = torch.arange(len(z), device="cuda")
res = []
for gw in itertools.product((0.5, 1.0, 2.0), repeat=5):
    lg = (logsig[..., :5] * torch.tensor(gw, device="cuda")).sum(-1)
    for r in (0.0, 2.0, 5.0):
        u = torch.tensor((5.0, 5.0, 2.0, 2.0, r), device="cuda")
        ls = lg + torch.log((sig[..., 5:10] * u).sum(-1) / u.sum() + 1e-6)
        for alpha in (0.0, 0.5):
            s = (1 - alpha) * ls + alpha * logsig[..., 11] if alpha else ls
            res.append(
                {"gates": gw, "route": r, "alpha": alpha, "score": float(lab[rows, s.argmax(1)].mean())}
            )
res.sort(key=lambda x: -x["score"])
default = next(x for x in res if x["gates"] == (1.0,) * 5 and x["route"] == 2.0 and x["alpha"] == 0.0)
oracle = float(lab.amax(1).mean())
print(
    f"{len(z)} held-out samples; default {default['score']:.4f}; best {res[0]}; "
    f"oracle (best of 32) {oracle:.4f}"
)
for x in res[:10]:
    print(" ", x)
json.dump(
    {
        "n": len(z),
        "default": default,
        "best": res[0],
        "top10": res[:10],
        "oracle": oracle,
        "ckpt": str(d["ckpt"]),
    },
    open(f"{out}/grid.json", "w"),
    indent=1,
)
