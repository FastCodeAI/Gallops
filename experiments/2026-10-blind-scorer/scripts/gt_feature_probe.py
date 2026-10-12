"""Rebuild two GT ticks (same scene, same timestamp, different inserted car) through scorer_cache's own path."""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "{gallops}/src")
from gallops.config import SiteConfig
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
cfg = SiteConfig.load()
for name, template in (("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"), ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
                       ("OPENSCENE_DATA_ROOT", "{data.navsim}"), ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps")):
    os.environ.setdefault(name, cfg.resolve(template))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"); os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")
import torch as _t
if DEV == 'cpu':
    _orig = _t.load
    _t.load = lambda *a, **k: _orig(*a, **{**k, 'map_location': 'cpu'})
from gallops.train.swerve import build_model, load_ticks
from gallops.train.swerve_data import TickDataset

L = Path("{runs}/labelled_gt")
ticks = [t for t in load_ticks(L, blind=True) if t["episode"] in ("frontal-0070/run_0", "side-0070/run_0")]
by = {}
for t in ticks: by.setdefault(t["timestamp"], {})[t["episode"]] = t
common = sorted(ts for ts, v in by.items() if len(v) == 2)
ts = common[-1]; pair = [by[ts]["frontal-0070/run_0"], by[ts]["side-0070/run_0"]]
print("timestamp", ts, [t["frame"] for t in pair])
ds = TickDataset(pair, DEV)
f = [ds.features(t) for t in pair]
for k in f[0]:
    a, b = f[0][k], f[1][k]
    if hasattr(a, "float"): print(f"  {k:18s} shape {tuple(a.shape)}  max diff {float((a.float() - b.float()).abs().max()):.3e}")
print(flush=True)
ckpt = cfg.resolve("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt")
agent, model, torch = build_model(ckpt, DEV, cfg.resolve("{data.drivejepa_repo}"), train_parts=("__nothing__",)); model.eval()
with torch.no_grad():
    outs = [model(x) for x in f]
print("  output keys:", sorted(outs[0]))
for k in ("trajectory", "pdm_score"):
    if k in outs[0]: print(f"  {k} max diff:", float((outs[0][k].float() - outs[1][k].float()).abs().max()))

def run(x):
    with torch.no_grad():
        o = model(x)
    return o["trajectory"].float(), o["pdm_score"].float(), o["proposals"].float()
fresh = lambda t, fast=True: ds.features(t, fast=fast)
base = run(fresh(pair[0]))
blank = fresh(pair[0]); blank["camera_feature_1"] = torch.zeros_like(blank["camera_feature_1"]); blank["camera_feature_2"] = torch.zeros_like(blank["camera_feature_2"])
out = run(blank)
print("  BLANK images vs real (fast path): trajectory diff", float((out[0] - base[0]).abs().max()), " proposals diff", float((out[2] - base[2]).abs().max()), flush=True)
noise = fresh(pair[0]); g = torch.Generator().manual_seed(0)
noise["camera_feature_1"] = torch.rand(noise["camera_feature_1"].shape, generator=g); noise["camera_feature_2"] = torch.rand(noise["camera_feature_2"].shape, generator=g)
out = run(noise)
print("  NOISE images vs real (fast path): trajectory diff", float((out[0] - base[0]).abs().max()), " proposals diff", float((out[2] - base[2]).abs().max()), flush=True)
s0, s1 = run(fresh(pair[0], False)), run(fresh(pair[1], False))
print("  SLOW path: frontal vs side trajectory diff", float((s0[0] - s1[0]).abs().max()), " proposals diff", float((s0[2] - s1[2]).abs().max()))
print("  SLOW vs FAST, frontal tick: proposals diff", float((s0[2] - base[2]).abs().max()))
