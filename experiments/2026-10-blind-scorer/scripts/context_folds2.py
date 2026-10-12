"""Does a down-the-road context feature fix the collision answer's blindness for short plans?

Same folds, data, targets, epochs and pick rules as gallops scorer_folds (subscores only). The MLP input
becomes [own plan's pooled features, context]; the context columns start at zero so step 0 is the
released scorer. Context variants: 'none' (baseline rerun), 'straight' (max-pool of the straight-ahead
+-1 m swerves at current speed, vocab slots 46-47), 'all' (max-pool over all 64 plans).
"""
import sys, time, json
import numpy as np
import torch
sys.path.insert(0, "{gallops}/src")
from gallops.train import scorer_folds as sf

dev = sys.argv[1]; variant = sys.argv[2]; drop = "drop" in sys.argv[3:]
d = dict(np.load("{dj_cache}/v1/folds_input.npz"))
x = torch.from_numpy(d["features"])  # (T, 64, 256) fp16
if variant == "straight":
    ctx = x[:, 46:48].float().amax(1)
elif variant == "all":
    ctx = x.float().amax(1)
elif variant == "geo":
    ctx = x[:, 46:48].float().amax(1)
else:
    ctx = torch.zeros(x.shape[0], 256)
GEO = 0
if variant == "geo":
    P = np.load("{dj_cache}/v1/plans.npz")
    plans = P["plans"]  # (T, 64, 8, 3)
    seg = np.diff(np.concatenate([np.zeros_like(plans[:, :, :1, :2]), plans[..., :2]], 2), axis=2)
    assert np.allclose(np.linalg.norm(seg, axis=-1).sum(-1), d["travel"], atol=1e-3), "plans out of order"
    g = np.concatenate([plans[..., :2].reshape(*plans.shape[:2], 16) / 20.0, plans[..., 2].reshape(*plans.shape[:2], 8),
                        np.repeat(P["speed"][:, None, None] / 10.0, 64, 1)], -1).astype(np.float16)
    geo = torch.from_numpy(g); GEO = geo.shape[-1]
ctx = ctx.half()
state = sf.released_state("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt", torch)
nn = torch.nn

def make():
    m = nn.Sequential(nn.Linear(512 + GEO, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 6))
    with torch.no_grad():
        m[0].weight.zero_(); m[0].weight[:, :256] = state["0.weight"]; m[0].bias.copy_(state["0.bias"])
        for i in (1, 3, 4, 6):
            m[i].weight.copy_(state[f"{i}.weight"]); m[i].bias.copy_(state[f"{i}.bias"])
    return m.to(dev)

def inp(b):
    own = x[b].to(dev).float()
    c = ctx[b].to(dev).float()[:, None].expand(-1, own.shape[1], -1)
    parts = [own, c] + ([geo[b].to(dev).float()] if GEO else [])
    return torch.cat(parts, -1)

sub = torch.from_numpy(sf.subscore_targets(d)).to(dev)
sig = lambda z: 1 / (1 + np.exp(-z))
safe = np.isnan(d["hit"]) & np.isnan(d["off_road"])
res = {}
for scene in sf.SCENES:
    avoidable = safe.any(1)  # ticks where at least one plan is safe; the rest teach no ranking
    tr = np.where((d["scene"] != scene) & d["usable"] & (avoidable if drop else True))[0]; te = np.where(d["scene"] == scene)[0]
    torch.manual_seed(0); m = make(); opt = torch.optim.Adam(m.parameters(), lr=1e-4); rng = np.random.default_rng(0); t0 = time.time()
    for _ in range(20):
        order = rng.permutation(tr)
        for s0 in range(0, len(order), 64):
            b = order[s0:s0 + 64]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(inp(b)), sub[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lg = torch.cat([m(inp(te[s0:s0 + 256])) for s0 in range(0, len(te), 256)]).cpu().numpy()
    s = sig(lg); rows = np.arange(len(te)); rel = d["pdm"][te].argmax(1)
    formula = s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12
    veto = s[rows, rel, 0] < 0.5
    pick = np.where(veto, formula.argmax(1), rel)
    # confident gate: override only if the formula's choice itself looks safe (sigma(NC) >= 0.5)
    alt = formula.argmax(1)
    confident = veto & (s[rows, alt, 0] >= 0.5)
    pick_c = np.where(confident, alt, rel)
    sft = safe[te]; t_hit = d["hit"][te][rows, rel]
    unavoid = ~sft.any(1) & ~np.isnan(d["hit"][te][rows, rel])
    resc = ~np.isnan(t_hit) & d["in_frame"][te]; calm = sft[rows, rel]
    fr = resc & (d["shape"][te] == "frontal")
    hits = ~np.isnan(d["hit"][te]); trv = d["travel"][te]
    sel = fr[:, None] & (t_hit[:, None] > 2) & (trv >= 8) & (trv < 15)
    r = {"rescued": float(sft[rows, pick][resc].mean()), "kept": float(sft[rows, pick][calm].mean()),
         "net": int(sft[rows, pick][resc].sum() - (~sft[rows, pick][calm]).sum()),
         "frontal_gt2_rescued": float(sft[rows, pick][fr & (t_hit > 2)].mean()), "frontal_gt2_veto": float(veto[fr & (t_hit > 2)].mean()),
         "frontal_rescued": float(sft[rows, pick][fr].mean()),
         "short_hit_P": float(np.median(s[..., 0][sel & hits])), "short_safe_P": float(np.median(s[..., 0][sel & ~hits])),
         "c_rescued": float(sft[rows, pick_c][resc].mean()), "c_kept": float(sft[rows, pick_c][calm].mean()),
         "c_net": int(sft[rows, pick_c][resc].sum() - (~sft[rows, pick_c][calm]).sum()),
         "c_frontal_gt2_rescued": float(sft[rows, pick_c][fr & (t_hit > 2)].mean()),
         "unavoid_override": float((pick != rel)[unavoid].mean()), "c_unavoid_override": float((pick_c != rel)[unavoid].mean()),
         "unavoid_travel_ratio": float(np.median(d["travel"][te][rows, pick][unavoid] / np.maximum(d["travel"][te][rows, rel][unavoid], 0.1))),
         "auc_nc": sf.auc(lg[..., 0].ravel(), sub[te].cpu().numpy()[..., 0].ravel()), "secs": round(time.time() - t0)}
    res[scene] = r
    print(variant + ("+drop" if drop else ""), scene, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
json.dump(res, open(f"{dj_cache}/v1/context_{variant}{'_drop' if drop else ''}.json", "w"), indent=1)
