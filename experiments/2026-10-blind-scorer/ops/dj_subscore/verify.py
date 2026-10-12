"""Hold the drivejepa_subscore adapter to the training cache on cached rendered ticks.

Builds each tick's model input exactly as gallops' scorer_cache did (TickDataset), runs the adapter's own
pass-2 + scorer code on it, and compares with what training saw: plans, per-candidate features, probe
tokens (cache v2), and the scorer's answers computed offline from the cached features.
    drive-jepa python verify.py
"""
import os, sys, json, importlib
import numpy as np
W = "{paddock}/src"
G = "{gallops}/src"
sys.path[:0] = [W, G]
os.environ.setdefault("GALLOPS_CONFIG", "{gallops}/gallops.toml")
from gallops.config import SiteConfig
cfg = SiteConfig.load()
for name, tmpl in (("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"), ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"), ("OPENSCENE_DATA_ROOT", "{data.navsim}"), ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps")):
    os.environ.setdefault(name, cfg.resolve(tmpl))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"); os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")
os.environ["DRIVEJEPA_CHECKPOINT"] = cfg.resolve("{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt")
os.environ["DRIVEJEPA_SUBSCORE_DIR"] = "{weights}/drivejepa_subscore/realmask-attngeo-v1/step_000000020"
sys.path.insert(0, cfg.resolve("{data.drivejepa_repo}")); os.chdir(cfg.resolve("{data.drivejepa_repo}"))
import torch
from paddock.adapters.models.drivejepa_subscore.policy import DriveJepaSubscorePolicy, build_scorer
from gallops.train.swerve import load_ticks
from gallops.train.swerve_data import TickDataset

RULE = os.environ.get("DRIVEJEPA_SUBSCORE_RULE", "absolute"); MARGIN = float(os.environ.get("DRIVEJEPA_SUBSCORE_MARGIN", "0.2"))
pol = DriveJepaSubscorePolicy(); pol.load()
pol.history = [{"timestamp": 0}, {"timestamp": 500_000}]  # warm: as after 0.5 s of driving
R = "{runs}"
C = "{dj_cache}/v2"
P = np.load(f"{C}/probe_input.npz")
first = {}
for i, e in enumerate(P["episode"]):
    first.setdefault(str(e), i)
worst = {"plans": 0.0, "own": 0.0, "probe": 0.0, "sub": 0.0}
picks_agree = n = 0
for ep in ("frontal-0075__run_0", "stationary-0070__run_3", "side-0058__run_1"):
    z = np.load(f"{C}/round0_v5/{ep}.npz")
    ticks = [t for t in load_ticks(__import__("pathlib").Path(f"{R}/labelled_round0_v5"), blind=True) if t["episode"] == ep.replace("__", "/")]
    ticks = sorted(ticks, key=lambda t: t["timestamp"])
    ds = TickDataset(ticks, "cuda")
    for i in (5, 15, 25):
        by_ts = {int(t["timestamp"]): t for t in ticks}
        tick = by_ts[int(z["timestamp"][i])]
        feats = ds.features(tick)
        with torch.no_grad():
            out = pol.agent._pad_model(feats)
        res = pol.score(out, float(tick["speed"]), float(tick["accel"]))
        g = first[f"round0_v5/{ep}"] + i
        cached_plans = np.concatenate([z["natives"][i], z["vocab"][i][:32]])
        worst["plans"] = max(worst["plans"], float(np.abs(res["plans"] - cached_plans).max()))
        worst["own"] = max(worst["own"], float(np.abs(res["own"][0].cpu().numpy() - P["own"][g].astype(np.float32)).max()))
        worst["probe"] = max(worst["probe"], float(np.abs(res["probe"][0].cpu().numpy() - P["probe"][g].astype(np.float32)).max()))
        # offline: the same 5 scorers on the CACHED features
        own = torch.as_tensor(P["own"][g][None], device="cuda").float(); pr = torch.as_tensor(P["probe"][g][None], device="cuda").float()
        plans = P["plans"][g]; geo = np.concatenate([plans[..., :2].reshape(64, 16) / 20.0, plans[..., 2], np.full((64, 1), P["speed"][g] / 10.0)], -1)
        geo = torch.as_tensor(geo, device="cuda", dtype=torch.float16).float()[None]
        with torch.no_grad():
            off = torch.stack([torch.sigmoid(m(own, pr, geo)) for m in pol.scorers]).mean(0)[0].cpu().numpy()
        worst["sub"] = max(worst["sub"], float(np.abs(off - res["sub"]).max()))
        rel = int(P["pdm"][g].argmax()); f = off[..., 0] ** 4 * off[..., 1] * (5 * off[..., 4] + 5 * off[..., 2] + 2 * off[..., 3]) / 12
        alt = int(f.argmax())
        over = off[rel, 0] < 0.5 if RULE == "absolute" else off[alt, 0] - off[rel, 0] > MARGIN
        offpick = alt if over else rel
        picks_agree += offpick == res["pick"]; n += 1
print(json.dumps({"rule": RULE, "ticks": n, "max_abs_diff": worst, "pick_agrees": f"{picks_agree}/{n}"}, indent=1))
ok = worst["plans"] < 1e-4 and worst["own"] < 1e-2 and worst["probe"] < 1e-2 and worst["sub"] < 1e-2 and picks_agree == n
print("VERIFY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
