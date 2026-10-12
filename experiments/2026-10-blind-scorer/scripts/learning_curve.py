"""Learning curve: does held-out frontal improve as the number of TRAINING scenes grows?
Rendered data only (cache v2), probes + geometry scorer, fold recipe (20 epochs). For held-out scene H,
train on every subset of the other four scenes of size 1..4, 2 seeds each; report on H's frontal ticks:
 - veto AUC: does the no-collision answer on the released pick separate ticks where it will hit (5 s)?
 - frontal rescue >2 s before contact with the fixed 0.5 gate.
    python learning_curve.py <held-out scene>
"""
import itertools, json, sys, time
import numpy as np, torch
sys.path.insert(0, "{gallops}/src")
from gallops.train import scorer_folds as sf
H = sys.argv[1]; dev = "cuda"
d = dict(np.load("{dj_cache}/v2/probe_input.npz"))
D = {"hit": d["hit5_at"], "off_road": d["off_road_at"], "ttc": d["ttc_at"], "dev": d["max_deviation"], "travel": d["travel"], "human": d["human_travel"]}
sub = torch.from_numpy(sf.subscore_targets(D)).to(dev)
own = torch.from_numpy(d["own"]); probe = torch.from_numpy(d["probe"]); plans = d["plans"]
geo = torch.from_numpy(np.concatenate([plans[..., :2].reshape(*plans.shape[:2], 16) / 20.0, plans[..., 2], np.repeat(d["speed"][:, None, None] / 10.0, 64, 1)], -1).astype(np.float16))
state = sf.released_state("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt", torch); nn = torch.nn
class Scorer(nn.Module):
    def __init__(s):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(537, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 6))
        with torch.no_grad():
            s.mlp[0].weight.zero_(); s.mlp[0].weight[:, :256] = state["0.weight"]; s.mlp[0].bias.copy_(state["0.bias"])
            for i in (1, 3, 4, 6): s.mlp[i].weight.copy_(state[f"{i}.weight"]); s.mlp[i].bias.copy_(state[f"{i}.bias"])
        s.pos = nn.Parameter(torch.zeros(256, 256)); s.q = nn.Linear(281, 256); s.attn = nn.MultiheadAttention(256, 4, batch_first=True)
    def forward(s, o, p, g):
        ctx, _ = s.attn(s.q(torch.cat([o, g], -1)), p + s.pos, p + s.pos); return s.mlp(torch.cat([o, ctx, g], -1))
inp = lambda b: (own[b].to(dev).float(), probe[b].to(dev).float(), geo[b].to(dev).float())
te = np.where((d["scene"] == H) & (d["shape"] == "frontal"))[0]
rows = np.arange(len(te)); rel = d["pdm"][te].argmax(1); th = d["hit5_at"][te][rows, rel]
safe = np.isnan(d["hit5_at"][te]) & np.isnan(d["off_road_at"][te])
hitlab = ~np.isnan(th); fr2 = hitlab & d["in_frame"][te] & (th > 2)
def auc(score, label):
    pos, neg = score[label], score[~label]
    allv = np.concatenate([pos, neg]); r = np.empty(len(allv)); r[np.argsort(allv, kind="mergesort")] = np.arange(1, len(allv) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
others = [s for s in sf.SCENES if s != H]; out = []
for k in (1, 2, 3, 4):
    for subset in itertools.combinations(others, k):
        for seed in (0, 1):
            tr = np.where(np.isin(d["scene"], subset) & d["usable"])[0]
            torch.manual_seed(seed); m = Scorer().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-4); rng = np.random.default_rng(seed)
            for _ in range(20):
                order = rng.permutation(tr)
                for s0 in range(0, len(order), 64):
                    b = order[s0:s0 + 64]; loss = torch.nn.functional.binary_cross_entropy_with_logits(m(*inp(b)), sub[b]); opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                s = torch.sigmoid(torch.cat([m(*inp(te[i:i + 256])) for i in range(0, len(te), 256)])).cpu().numpy()
            f = s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12
            pick = np.where(s[rows, rel, 0] < 0.5, f.argmax(1), rel)
            r = {"held_out": H, "k": k, "train": list(subset), "seed": seed, "train_ticks": int(len(tr)),
                 "veto_auc": round(float(auc(1 - s[rows, rel, 0], hitlab)), 3), "rescue_gt2": round(float(safe[rows, pick][fr2].mean()), 3)}
            out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(f"{dj_cache}/v2/learning_curve_{H}.json", "w"), indent=1)
