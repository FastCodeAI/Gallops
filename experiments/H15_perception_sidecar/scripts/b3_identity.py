"""Why B3.2's identity check refused: the frozen branch against the release, row by row. One-off.

    {python.drivejepa} experiments/H15_perception_sidecar/scripts/b3_identity.py \
        --block20 {runs}/drivejepa_perception/block20-v1 --tokens {runs}/drivejepa_perception/tokens-v1 \
        --labels {runs}/drivejepa_perception/labels-v1 --pdm {runs}/drivejepa_perception/pdm-v1 --out OUT.json

`train_vit_lora shared` runs the release's projector, refiner and scorer on the frozen branch's
output (the block-20 cache through blocks 21-24) and refuses unless the candidates match the
release's cached ones within preflight.REFINER_LIMITS. On the select logs they did (0.011 m, 99.98%
of picks); on the report logs the largest gap was 0.75 m with every pick equal. This measures the
distribution of the per-row largest gap on both splits, and where the outliers sit: in the pick, in
the release's best candidate, or elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    import torch

    from gallops.config import SiteConfig
    from gallops.drivejepa_b2d import batch_fix
    from gallops.drivejepa_b2d.head import Head, head_tensors_from_checkpoint
    from gallops.drivejepa_b2d.train_generator import refine_from_cache
    from gallops.drivejepa_perception import env, train_refiner, train_scorer, vit_lora

    ap = argparse.ArgumentParser()
    for flag in ("--block20", "--tokens", "--labels", "--pdm", "--out"):
        ap.add_argument(flag, required=True)
    args = ap.parse_args()
    entered = env.enter(SiteConfig.load())
    data = train_scorer.load(Path(args.tokens), Path(args.labels), Path(args.pdm), "C", None)
    cache = np.load(Path(args.block20) / "block20.npy", mmap_mode="r")
    where = {int(r): i for i, r in enumerate(np.load(Path(args.block20) / "rows.npy"))}
    lidar2img = np.load(Path(args.tokens) / "lidar2img.npy")
    img_shape = np.load(Path(args.tokens) / "img_shape.npy")
    spatial, level = torch.tensor([[16, 32]], dtype=torch.long), torch.tensor([0], dtype=torch.long)
    batch_fix.apply_all()
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=entered["release"])
    agent.initialize()
    model = agent._pad_model.eval()
    branch = vit_lora.LateBranch(model._backbone.img_backbone, lora=False).cuda().eval()
    projector = model._backbone.projector.cuda()
    model._backbone = model._backbone.cpu()
    model = model.cuda()
    head = Head()
    head.load_state_dict(head_tensors_from_checkpoint(Path(entered["release"])))
    head = head.cuda().eval()
    out = {}
    with torch.no_grad():
        for split in ("select", "report"):
            rows = data["rows"][split]
            gap = np.zeros((len(rows), 32), np.float32)  # largest |difference| over a candidate's poses
            pick = np.zeros(len(rows), np.int64)
            for i in range(0, len(rows), train_refiner.BATCH):
                chunk = rows[i : i + train_refiner.BATCH]
                x = torch.from_numpy(np.stack([np.asarray(cache[where[int(r)]]) for r in chunk])).cuda()
                props, bev = refine_from_cache(
                    model,
                    projector(branch(x.float())),
                    torch.from_numpy(lidar2img[chunk][:, 1:2]),
                    torch.from_numpy(img_shape[chunk]),
                    torch.from_numpy(data["ego"][chunk]),
                    spatial,
                    level,
                )
                mine = props[-1].float().cpu().numpy()
                gap[i : i + len(chunk)] = (
                    np.abs(mine - data["proposals"][chunk]).reshape(len(chunk), 32, -1).max(-1)
                )
                pick[i : i + len(chunk)] = (
                    head(bev.reshape(len(chunk), 32, 8, -1).amax(2))[..., -1].argmax(1).cpu().numpy()
                )
            release_pick = data["pick_release"][rows]
            best = data["pdm"][rows, :, -1].argmax(1)
            per_row = gap.max(1)
            worst = np.argsort(per_row)[::-1][:10]
            out[split] = {
                "rows": int(len(rows)),
                "picks_equal_share": float((pick == release_pick).mean()),
                "per_row_largest_gap_m": {
                    q: float(np.quantile(per_row, float(q))) for q in ("0.5", "0.9", "0.99", "0.999", "1.0")
                },
                "rows_over_0.05_m": int((per_row > 0.05).sum()),
                "candidates_over_0.05_m": int((gap > 0.05).sum()),
                "picked_candidate_gap_m": {
                    "max": float(gap[np.arange(len(rows)), release_pick].max()),
                    "over_0.05": int((gap[np.arange(len(rows)), release_pick] > 0.05).sum()),
                },
                "best_candidate_over_0.05": int((gap[np.arange(len(rows)), best] > 0.05).sum()),
                "worst_rows": [
                    {
                        "row": int(rows[w]),
                        "gap_m": float(per_row[w]),
                        "candidate": int(gap[w].argmax()),
                        "is_pick": bool(gap[w].argmax() == release_pick[w]),
                        "is_best": bool(gap[w].argmax() == best[w]),
                    }
                    for w in worst
                ],
            }
            print(json.dumps({split: {k: v for k, v in out[split].items() if k != "worst_rows"}}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
