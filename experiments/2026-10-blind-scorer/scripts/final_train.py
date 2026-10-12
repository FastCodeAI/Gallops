"""Final sub-score scorer for the closed-loop test: all five rendered scenes + real nuScenes (on-road, in-lane,
progress only), probes + geometry, the fold recipe unchanged (20 epochs, Adam 1e-4, batch 64, constants fixed).

    python final_train.py <out_dir> <seed> [<seed> ...]

Writes seed_<k>.pt (state_dict) and, once, scorer.json (everything the adapter must reproduce) beside a
gallops-style checkpoint.json sidecar.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "{gallops}/src")
from gallops.train import scorer_folds as sf

OUT = Path(sys.argv[1]); SEEDS = [int(s) for s in sys.argv[2:]]
OUT.mkdir(parents=True, exist_ok=True)
CK = "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
dev = "cuda"
parts = [dict(np.load(f"{dj_cache}/{c}/probe_input.npz")) for c in ("v2", "v3_nusc")]
d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
src = np.array([e.split("/")[0] for e in d["episode"]])
real = np.isin(src, ["nusc", "nusc_holdout"])
train = np.where(d["usable"] & (src != "nusc_holdout"))[0]
D = {"hit": d["hit5_at"], "off_road": d["off_road_at"], "ttc": d["ttc_at"], "dev": d["max_deviation"], "travel": d["travel"], "human": d["human_travel"]}
sub = torch.from_numpy(sf.subscore_targets(D)).to(dev)
chw = np.ones((len(src), 1, 6), np.float32); w = chw[real]; w[..., [0, 2, 5]] = 0.0; chw[real] = w
own = torch.from_numpy(d["own"]); probe = torch.from_numpy(d["probe"]); plans = d["plans"]
geo = torch.from_numpy(np.concatenate([plans[..., :2].reshape(*plans.shape[:2], 16) / 20.0, plans[..., 2],
                                       np.repeat(d["speed"][:, None, None] / 10.0, 64, 1)], -1).astype(np.float16))
state = sf.released_state(CK, torch); nn = torch.nn; GEO = 25
print(f"{len(train)} training ticks ({int((~real[train]).sum())} rendered, {int(real[train].sum())} real)", flush=True)


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256 + 256 + GEO, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 1024),
                                 nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 6))
        with torch.no_grad():
            self.mlp[0].weight.zero_(); self.mlp[0].weight[:, :256] = state["0.weight"]; self.mlp[0].bias.copy_(state["0.bias"])
            for i in (1, 3, 4, 6):
                self.mlp[i].weight.copy_(state[f"{i}.weight"]); self.mlp[i].bias.copy_(state[f"{i}.bias"])
        self.pos = nn.Parameter(torch.zeros(256, 256))
        self.q = nn.Linear(256 + GEO, 256)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)

    def forward(self, o, p, g):
        ctx, _ = self.attn(self.q(torch.cat([o, g], -1)), p + self.pos, p + self.pos)
        return self.mlp(torch.cat([o, ctx, g], -1))


for seed in SEEDS:
    torch.manual_seed(seed); m = Scorer().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-4); rng = np.random.default_rng(seed); t0 = time.time()
    for _ in range(20):
        order = rng.permutation(train)
        for s0 in range(0, len(order), 64):
            b = order[s0:s0 + 64]
            res = m(own[b].to(dev).float(), probe[b].to(dev).float(), geo[b].to(dev).float())
            wt = torch.from_numpy(chw[b]).to(dev)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(res, sub[b], reduction="none") * wt).sum() / (wt.sum() * res.shape[1]).clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(m.state_dict(), OUT / f"seed_{seed}.pt")
    print(f"seed {seed}: trained in {time.time() - t0:.0f} s", flush=True)

sha = subprocess.run(["git", "-C", "{gallops}", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
(OUT / "scorer.json").write_text(json.dumps({
    "architecture": "MLP(own 256 + ctx 256 + geo 25 -> 1024 -> LN -> ReLU -> 1024 -> LN -> ReLU -> 6), ctx = MultiheadAttention(256, 4 heads, batch_first) with query Linear(own+geo -> 256) over the 256 probe tokens + learned pos (256x256)",
    "channels": ["no_collision", "on_road", "no_ttc", "in_lane", "progress", "final"],
    "own": "per-candidate max over its 8 last-stage refiner tokens; candidates = 32 native (pass 1) + 32 swerves (pass 2)",
    "probe": "32 probe paths x 8 waypoints = 256 tokens, probe-major, from pass 2",
    "geo": "candidate (x, y) / 20 m (16), heading (8), ego speed / 10 m/s (1)",
    "vocab": "paddock-runs/ops/dj_cache/vocab_v1_probes.py (vocab_v1 swerves 0-31, probe_v1 32-63)",
    "pick": "released pick unless mean-of-seeds sigmoid(no_collision) of it < 0.5; then argmax of nc^4 * on_road * (5 progress + 5 no_ttc + 2 in_lane) / 12",
    "seeds": SEEDS}, indent=1))
(OUT / "checkpoint.json").write_text(json.dumps({
    "step": 20, "unit": "epochs", "train_git_sha": sha, "model": "drivejepa_subscore",
    "trained_on": [{"dataset": "gallops rendered neuroNCAP-style episodes", "scenes": ["0053", "0058", "0064", "0070", "0075"],
                    "sources": ["round0_v5", "gt", "round1_s2"]},
                   {"dataset": "nuscenes", "split": "train minus neuroNCAP scenes, scene-0061 and the 5 reconstructed scenes, minus 66 held-out",
                    "labels_used": ["on_road", "in_lane", "progress"]}],
    "params": int(sum(p.numel() for p in Scorer().parameters())), "precision": "fp32",
    "note": "closed-loop experiment only; its collision answer does not transfer to real camera images (real held-out override rate 55%)"}, indent=1))
print("wrote", OUT, flush=True)
