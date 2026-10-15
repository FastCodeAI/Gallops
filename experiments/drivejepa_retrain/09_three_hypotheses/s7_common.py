"""S7 shared helpers. Read-only on S6's dumps and manifests; definitions follow 08_scorer_diagnosis/s6_analyze.py so numbers line up."""
import glob, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent; S6 = RD / "08_scorer_diagnosis"
NAS6 = pathlib.Path("{runs}/drivejepa_retrain/08_scorer_diagnosis/dump"); NAS7 = pathlib.Path("{runs}/drivejepa_retrain/09_three_hypotheses")
SEED_BOOT = 20261007; SEED = 20261008; NB = 1000
TERMS = ["NC", "DAC", "TLC", "DDC", "EP", "TTC", "LK", "HC", "score"]
NAMES = dict(NC="collision (gate)", DAC="drivable area (gate)", TLC="traffic light (gate)", DDC="driving direction (gate)", EP="progress", TTC="time buffer", LK="lane keeping", HC="comfort")
SL = ["all", "highway", "close-pass", "ordinary"]

def load(name, keys):
    """Concatenate the named arrays over the 47 shards of one set and attach manifest columns (same join as s6_analyze.load)."""
    fs = sorted(f for f in glob.glob(str(NAS6 / name / "shard_*.npz")) if ".tmp." not in f); assert len(fs) == 47, (name, len(fs))
    D = {k: [] for k in ["token"] + list(keys)}
    for f in fs:
        z = np.load(f)
        for k in D: D[k].append(z[k])
    D = {k: np.concatenate(v) for k, v in D.items()}; D["_files"] = fs
    man = pd.read_parquet(S6 / f"set_{name}.parquet").set_index("sample_id"); assert len(D["token"]) == len(man) and set(D["token"]) == set(man.index)
    m = man.loc[D["token"]]; D["slice"] = m["slice"].to_numpy(); D["clip"] = m.clip_uuid.to_numpy(); D["anchor_t"] = m.anchor_t.to_numpy(); D["speed"] = m.speed.to_numpy(float); return D

def boot(clip, cols, stat, nb=NB):
    """clip-level bootstrap of stat(dict of per-example arrays) -> (value, lo, hi); same resampling as s6_analyze.boot"""
    v = stat(cols); u, inv = np.unique(clip, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(u))]; rng = np.random.default_rng(SEED_BOOT); out = []
    for _ in range(nb):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))]); out.append(stat({k: x[ii] for k, x in cols.items()}))
    lo, hi = np.nanpercentile(out, [2.5, 97.5]); return float(v), float(lo), float(hi)

def spearman_rows(s, g):
    a = rankdata(s, axis=1); b = rankdata(g, axis=1); a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True); den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)

def step_speeds(P):
    """(..., 8, >=2) poses 0.5 s apart in the car's frame -> (..., 8) step speeds, first step from the origin"""
    xy = P[..., :2].astype(np.float64); prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2); return np.linalg.norm(xy - prev, axis=-1) / 0.5
