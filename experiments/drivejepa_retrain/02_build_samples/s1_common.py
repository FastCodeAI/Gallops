"""S1 shared config. Imports S0's common (split views, exam lists) read-only; S1 keeps its own manifest and NAS root."""
import os, sys, csv, time, pathlib, threading
HERE = pathlib.Path(__file__).resolve().parent
S0 = HERE.parent / "01_find_clips"
sys.path.insert(0, str(S0))
import common as S0C  # noqa: E402
import hfrange        # noqa: E402  (S0 range reader)
NAS = pathlib.Path("{runs}/drivejepa_retrain/02_build_samples")
RAW0 = S0C.RAW_NAS                                   # S0 Run 3 raw cache (egomotion.offline zips, vehicle_dimensions, obstacle members)
MANIFEST = HERE / "download_manifest.csv"
RAW_REPO = S0C.RAW_REPO
FRONT = "camera_front_wide_120fov"
_L = threading.Lock()
def manifest_add(uuid, fname, nbytes, path, sha):
    with _L:
        new = not MANIFEST.exists()
        with open(MANIFEST, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["uuid", "file", "bytes", "path", "sha256", "ts"])
            w.writerow([uuid, fname, nbytes, path, sha, time.strftime("%Y-%m-%dT%H:%M:%S")])
def token():
    return os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
def assert_train(uuids):
    """Every clip must be raw_train; never exam / test / val."""
    import pandas as pd
    V = S0C.load_split_views(); _, exam, _ = S0C.load_views()
    ci = pd.read_parquet(S0C.RAW_SNAP / "clip_index.parquet")
    for u in uuids:
        assert u in V["raw_train"], f"{u} not raw_train"
        assert u not in exam and u not in V["raw_val"] and ci.loc[u, "split"] == "train", u
    return V
