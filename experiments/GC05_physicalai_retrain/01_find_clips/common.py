"""Shared config for 01_find_clips. Read-only on repo; all downloaded bytes go to NAS."""
import os, csv, time, pathlib
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
RELEASE = "sample_set/26.04_release"
HERE = pathlib.Path(__file__).resolve().parent
NAS = pathlib.Path("{runs}/drivejepa_retrain/01_find_clips")   # NAS = {site-nfs}
MANIFEST = HERE / "download_manifest.csv"                        # uuid,file,bytes,path,sha256,ts
os.environ.setdefault("HF_HOME", "{runs}/drivejepa_retrain/hf_home")
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

def manifest_add(uuid, fname, nbytes, path, sha=""):
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["uuid","file","bytes","path","sha256","ts"])
        w.writerow([uuid, fname, nbytes, path, sha, time.strftime("%Y-%m-%dT%H:%M:%S")])

def load_exclusions():
    p = HERE / "exam_exclusions.txt"
    s = {l.strip() for l in open(p) if l.strip() and not l.startswith("#")}
    assert len(s) >= 30, f"exam_exclusions.txt too small ({len(s)})"
    return s

# ---- Run 2 (2026-10-02): final exam views; every Run-2 script calls load_views() ----
import threading
_LOCK = threading.Lock()
_orig_manifest_add = manifest_add
def manifest_add(*a, **k):
    with _LOCK: _orig_manifest_add(*a, **k)

def _ids(name):
    return {l.strip() for l in open(HERE / name) if l.strip() and not l.startswith("#")}

def load_views():
    """Returns (nurec_2604_exclusions, raw_exclusions, pool) with the registered sizes asserted."""
    nurec, raw, pool = _ids("nurec_2604_exclusions.txt"), _ids("raw_exclusions.txt"), _ids("training_pool.txt")
    assert len(nurec) == 159, len(nurec)
    assert len(raw) == 913, len(raw)
    assert len(pool) == 1447, len(pool)
    assert nurec <= raw and not (pool & raw)
    return nurec, raw, pool

# ---- Run 3 (2026-10-03): split rule (Sid): train = raw split "train"; val = offline validation; test excluded everywhere; exam excluded on top ----
RAW_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
RAW_SNAP = pathlib.Path("{runs}/drivejepa_retrain/hf_home/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/33f9bf447ed3bcb7d545ce13f4226f824214fafb")
RAW_NAS = pathlib.Path("{runs}/drivejepa_retrain/01_find_clips/raw")
def load_split_views():
    """Returns dict of sets nurec_train, nurec_val, raw_train, raw_val; asserts no test or exam uuid in any of them."""
    import pandas as pd
    nurec, raw, pool = load_views()
    ci = pd.read_parquet(RAW_SNAP / "clip_index.parquet")
    test = set(ci.index[ci.split == "test"]); tr = set(ci.index[ci.split == "train"]); va = set(ci.index[ci.split == "val"])
    V = {k: _ids(k + ".txt") for k in ["nurec_train", "nurec_val", "raw_train", "raw_val"]}
    for k, s in V.items():
        assert not (s & test), f"{k} contains test clips"
        assert not (s & raw), f"{k} contains exam clips"
        assert s <= (tr if k.endswith("train") else va), f"{k} not inside its split"
    assert V["nurec_train"] <= pool and V["nurec_val"] <= pool and V["nurec_train"] <= V["raw_train"] and V["nurec_val"] <= V["raw_val"]
    return V
