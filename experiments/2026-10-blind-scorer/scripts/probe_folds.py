"""Leave-one-scene-out on cache v2 (batch 1): do far-ahead probe features fix the frontal blindness?

Candidates = 32 native + 32 swerves (64). Context = the 32 probe paths' per-waypoint tokens (256 per tick),
which sample the road 4-128 m ahead. Variants:
  base   - released MLP on [own], retrained (sub-score targets), no context
  attn   - each candidate's feature attends over the 256 probe tokens (1 layer, 4 heads); MLP on [own, ctx]
  attngeo- attn + the candidate's waypoints and ego speed
The context and geometry columns start at zero, so step 0 is the released scorer. 20 epochs, constants fixed.
Pick rules evaluated: gated formula (override the released pick when sigma(NC) < 0.5) and confident gate
(also require the alternative's sigma(NC) >= 0.5).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "{gallops}/src")
from gallops.train import scorer_folds as sf

CACHE = Path("{dj_cache}/v2")
dev, variant = sys.argv[1], sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NO64 = 'no64' in sys.argv[4:]
REAL = any(a in sys.argv[4:] for a in ("real", "realmask", "realw"))
REALMASK = "realmask" in sys.argv[4:]
REALW = "realw" in sys.argv[4:]


def reduced(cache):
    red = cache / "probe_input.npz"
    if not red.exists():
        keys = ("hit5_at", "off_road_at", "ttc_at", "max_deviation", "travel", "human_travel")
        acc = {k: [] for k in ("own", "probe", "plans", "pdm", "usable", "in_frame", "speed", "scene", "shape", "episode") + keys}
        for src in sorted(p for p in cache.iterdir() if p.is_dir() and not p.name.startswith("__")):
            for f in sorted(src.glob("*.npz")):
                if f.name.endswith(".labels.npz") or ".tmp" in f.name:
                    continue
                lf = f.with_name(f.name.replace(".npz", ".labels.npz"))
                if not lf.exists():
                    continue
                z, lab = np.load(f), np.load(lf)
                own = np.concatenate([z["native_tokens"], z["vocab_tokens"][:, :32]], 1).max(axis=2)
                acc["own"].append(own); acc["probe"].append(z["vocab_tokens"][:, 32:].reshape(len(own), 256, -1))
                acc["plans"].append(np.concatenate([z["natives"], z["vocab"][:, :32]], 1).astype(np.float32))
                for k in ("pdm_score", "usable", "in_frame", "speed"):
                    acc["pdm" if k == "pdm_score" else k].append(z[k])
                for k in keys:
                    acc[k].append(lab[k] if k == "human_travel" else lab[k][:, :64])
                shape, scene = f.stem.split("__")[0].split("-")
                acc["scene"].append(np.full(len(own), scene)); acc["shape"].append(np.full(len(own), shape)); acc["episode"].append(np.full(len(own), f"{src.name}/{f.stem}"))
        np.savez(red, **{k: np.concatenate(v) for k, v in acc.items()})
    return dict(np.load(red))


d = reduced(CACHE)
N_RENDERED = len(d["scene"])
if REAL:
    r = reduced(Path("{dj_cache}/v3_nusc"))
    d = {k: np.concatenate([d[k], r[k]]) for k in d}
SOURCE = np.array([e.split("/")[0] for e in d["episode"]])
CHW = np.ones((len(SOURCE), 1, 6), np.float32)
if REALMASK:
    real_rows = np.isin(SOURCE, ["nusc", "nusc_holdout"])
    w = CHW[real_rows]; w[..., [0, 2, 5]] = 0.0; CHW[real_rows] = w  # real logs: on-road, in-lane, progress only
    assert CHW[real_rows][..., 0].max() == 0 and CHW[~real_rows].min() == 1
if REALW:
    CHW[np.isin(SOURCE, ["nusc", "nusc_holdout"])] = 0.2  # every answer, at a fifth of a rendered tick's weight
print(f"{variant}: {len(d['scene'])} ticks", flush=True)
D = {"hit": d["hit5_at"], "off_road": d["off_road_at"], "ttc": d["ttc_at"], "dev": d["max_deviation"], "travel": d["travel"], "human": d["human_travel"]}
sub = torch.from_numpy(sf.subscore_targets(D)).to(dev)
own = torch.from_numpy(d["own"]); probe = torch.from_numpy(d["probe"])
plans = d["plans"]
geo_np = np.concatenate([plans[..., :2].reshape(*plans.shape[:2], 16) / 20.0, plans[..., 2], np.repeat(d["speed"][:, None, None] / 10.0, 64, 1)], -1).astype(np.float16)
geo = torch.from_numpy(geo_np)
state = sf.released_state("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt", torch)
nn = torch.nn
USE_CTX = variant in ("attn", "attngeo", "attngeo_aux"); USE_GEO = variant in ("attngeo", "attngeo_aux"); GEO = geo.shape[-1] if USE_GEO else 0
USE_AUX = variant == "attngeo_aux"
HZ = torch.from_numpy(np.load(CACHE / "hazard_targets.npy")) if USE_AUX else None
assert not (USE_AUX and REAL), "hazard targets exist for rendered ticks only"  # (T, 4, 3) present, x, y


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        n_in = 256 + (256 if USE_CTX else 0) + GEO
        self.mlp = nn.Sequential(nn.Linear(n_in, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 6))
        with torch.no_grad():
            self.mlp[0].weight.zero_(); self.mlp[0].weight[:, :256] = state["0.weight"]; self.mlp[0].bias.copy_(state["0.bias"])
            for i in (1, 3, 4, 6):
                self.mlp[i].weight.copy_(state[f"{i}.weight"]); self.mlp[i].bias.copy_(state[f"{i}.bias"])
        if USE_CTX:
            self.pos = nn.Parameter(torch.zeros(256, 256))
            self.q = nn.Linear(256 + GEO, 256)
            self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        if USE_AUX:
            # one extra query that must say where the hazard car will be at +1..+4 s
            self.hq = nn.Parameter(torch.zeros(1, 1, 256))
            self.hhead = nn.Linear(256, 12)

    def forward(self, o, p, g):
        parts = [o]
        if USE_CTX:
            q = self.q(torch.cat([o, g], -1) if USE_GEO else o)
            ctx, _ = self.attn(q, p + self.pos, p + self.pos)
            parts.append(ctx)
        if USE_GEO:
            parts.append(g)
        out = self.mlp(torch.cat(parts, -1))
        if USE_AUX:
            h, _ = self.attn(self.hq.expand(len(o), -1, -1), p + self.pos, p + self.pos)
            return out, self.hhead(h[:, 0])
        return out


def batch(b):
    return own[b].to(dev).float(), probe[b].to(dev).float(), geo[b].to(dev).float()


sig = lambda z: 1 / (1 + np.exp(-z))
safe = np.isnan(D["hit"]) & np.isnan(D["off_road"])
res = {}
saved = {}
HOLD = np.where(SOURCE == "nusc_holdout")[0]
hold_sum = None
for scene in sf.SCENES:
    keep = ~((d["scene"] == "0064") & (d["shape"] == "frontal")) if NO64 else np.ones(len(d["scene"]), bool)
    te = np.where(d["scene"] == scene)[0]
    tr = np.where((d["scene"] != scene) & d["usable"] & keep & (SOURCE != "nusc_holdout"))[0]; te = np.where(d["scene"] == scene)[0]
    torch.manual_seed(SEED); m = Scorer().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-4); rng = np.random.default_rng(SEED); t0 = time.time()
    for _ in range(20):
        order = rng.permutation(tr)
        for s0 in range(0, len(order), 64):
            b = order[s0:s0 + 64]
            res_b = m(*batch(b))
            if USE_AUX:
                res_b, hz = res_b
                tgt = HZ[b].to(dev)
                pres = tgt[..., 0]
                aux = torch.nn.functional.binary_cross_entropy_with_logits(hz[:, :4], pres)
                xy = hz[:, 4:].reshape(-1, 4, 2)
                aux = aux + ((xy - tgt[..., 1:] / 20.0).abs().sum(-1) * pres).sum() / pres.sum().clamp_min(1)
                w = torch.from_numpy(CHW[b]).to(dev)
                per = torch.nn.functional.binary_cross_entropy_with_logits(res_b, sub[b], reduction="none") * w
                loss = per.sum() / (w.sum() * res_b.shape[1]).clamp_min(1) + aux
            else:
                w = torch.from_numpy(CHW[b]).to(dev)
                per = torch.nn.functional.binary_cross_entropy_with_logits(res_b, sub[b], reduction="none") * w
                loss = per.sum() / (w.sum() * res_b.shape[1]).clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        outs = [m(*batch(te[s0:s0 + 128])) for s0 in range(0, len(te), 128)]
        lg = torch.cat([o[0] if USE_AUX else o for o in outs]).cpu().numpy()
    saved[f"logits_{scene}"] = lg.astype(np.float32); saved[f"test_idx_{scene}"] = te
    if len(HOLD):
        with torch.no_grad():
            ho = [m(*batch(HOLD[s0:s0 + 128])) for s0 in range(0, len(HOLD), 128)]
            hl = torch.cat([o[0] if USE_AUX else o for o in ho]).cpu().numpy()
        hold_sum = hl if hold_sum is None else hold_sum + hl
    s = sig(lg); rows = np.arange(len(te)); rel = d["pdm"][te].argmax(1)
    formula = s[..., 0] ** 4 * s[..., 1] * (5 * s[..., 4] + 5 * s[..., 2] + 2 * s[..., 3]) / 12
    alt = formula.argmax(1); veto = s[rows, rel, 0] < 0.5
    pick = np.where(veto, alt, rel); pick_c = np.where(veto & (s[rows, alt, 0] >= 0.5), alt, rel)
    sft = safe[te]; t_hit = D["hit"][te][rows, rel]
    resc = ~np.isnan(t_hit) & d["in_frame"][te]; calm = sft[rows, rel]
    fr = resc & (d["shape"][te] == "frontal"); fr2 = fr & (t_hit > 2)
    unavoid = ~sft.any(1) & ~np.isnan(t_hit)
    hits = ~np.isnan(D["hit"][te]); trv = D["travel"][te]
    sel = fr2[:, None] & (trv >= 8) & (trv < 15)
    def net(p): return int(sft[rows, p][resc].sum() - (~sft[rows, p][calm]).sum())
    r = {"rescued": float(sft[rows, pick][resc].mean()), "kept": float(sft[rows, pick][calm].mean()), "net": net(pick),
         "fr_gt2_possible": float(sft.any(1)[fr2].mean()), "fr_gt2_rescued": float(sft[rows, pick][fr2].mean()), "fr_gt2_veto": float(veto[fr2].mean()),
         "c_net": net(pick_c), "c_kept": float(sft[rows, pick_c][calm].mean()), "c_fr_gt2_rescued": float(sft[rows, pick_c][fr2].mean()),
         "unavoid_override": float((pick != rel)[unavoid].mean()), "c_unavoid_override": float((pick_c != rel)[unavoid].mean()),
         "short_hit_P": float(np.median(s[..., 0][sel & hits])) if (sel & hits).any() else None,
         "short_safe_P": float(np.median(s[..., 0][sel & ~hits])) if (sel & ~hits).any() else None,
         "auc_nc": sf.auc(lg[..., 0].ravel(), sub[te].cpu().numpy()[..., 0].ravel()), "secs": round(time.time() - t0)}
    res[scene] = r
    print(variant, scene, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
tag = f"{variant}{'_no64' if NO64 else ''}{'_realmask' if REALMASK else ('_realw' if REALW else ('_real' if REAL else ''))}_s{SEED}"
if len(HOLD):
    saved["logits_nusc_holdout"] = (hold_sum / len(sf.SCENES)).astype(np.float32); saved["test_idx_nusc_holdout"] = HOLD
json.dump(res, open(CACHE / f"probe_{tag}.json", "w"), indent=1)
np.savez(CACHE / f"probe_{tag}.logits.npz", **saved)
