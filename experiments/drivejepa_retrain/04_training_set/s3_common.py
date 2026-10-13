"""S3 shared config. Reuses S0 (split views, range reader, detector core), S1 (converter) and S2.1 (grader) by import; nothing is forked."""
import os, sys, csv, time, pathlib, threading, hashlib
HERE = pathlib.Path(__file__).resolve().parent
S0 = HERE.parent / "01_find_clips"; S1 = HERE.parent / "02_build_samples"; S2 = HERE.parent / "03_grade_candidates"
for p in (S0, S1, S2): sys.path.insert(0, str(p))
import common as S0C  # noqa: E402
import hfrange        # noqa: E402
NAS = pathlib.Path("{runs}/drivejepa_retrain/04_training_set")
S1ROOT = NAS / "s1"                 # S1 converter root for this stage: clips/, calibration/, cache/
CACHE = S1ROOT / "cache"
SCENES = NAS / "scenes"             # usdz map members per NuRec scene (S2 layout)
MANIFEST = HERE / "download_manifest.csv"
RAW_REPO = S0C.RAW_REPO; FRONT = "camera_front_wide_120fov"
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
_V = None
def views():
    global _V
    if _V is None:
        import pandas as pd
        V = S0C.load_split_views(); _, exam, _ = S0C.load_views(); ci = pd.read_parquet(S0C.RAW_SNAP / "clip_index.parquet")
        _V = dict(V=V, exam=exam, test=set(ci.index[ci.split == "test"]), val=set(ci.index[ci.split == "val"]), ci=ci)
    return _V
# ---- S3.1: split switch. S3_SPLIT=val repoints the working folder to 04_training_set/val/ and flips every split assertion to val.
# ---- S4 Part A: S3_SPLIT=val_full is the val mode (SPLIT == "val", same assertions) with the working folder 04_training_set/val_full/ and
# its own download manifest. S3_SPLIT=train / val behave exactly as before.
VAL_DIR = "val_full" if os.environ.get("S3_SPLIT") == "val_full" else "val"
SPLIT = "val" if os.environ.get("S3_SPLIT") == "val_full" else os.environ.get("S3_SPLIT", "train"); assert SPLIT in ("train", "val")
CODE = HERE
if SPLIT == "val":
    HERE = CODE / VAL_DIR; HERE.mkdir(exist_ok=True)
    if VAL_DIR == "val_full": MANIFEST = HERE / "download_manifest.csv"
    if not (HERE / "vocab_10240.npy").exists(): os.symlink(CODE / "vocab_10240.npy", HERE / "vocab_10240.npy")
def assert_val(uuids, nurec=False):
    """raw_val only (nurec_val when nurec=True); never exam / test / train."""
    v = views(); u = set(uuids); ci = v["ci"]
    assert not (u & v["exam"]), "exam uuid"; assert not (u & v["test"]), "test uuid"
    assert u <= v["V"]["raw_val"], "not raw_val"; assert not (u & v["V"]["raw_train"]), "train uuid in val"
    if nurec: assert u <= v["V"]["nurec_val"], "not nurec_val"
    return True
def assert_train(uuids, nurec=False):
    if SPLIT == "val": return assert_val(uuids, nurec)
    return _assert_train(uuids, nurec)
def _assert_train(uuids, nurec=False):
    """Z1 guard, called by every S3 script on everything it touches: raw_train only (nurec_train when nurec=True); never exam / test / val."""
    v = views(); u = set(uuids)
    assert not (u & v["exam"]), f"exam uuid: {sorted(u & v['exam'])[:3]}"
    assert not (u & v["test"]), "test uuid"; assert not (u & v["val"]), "val uuid"
    assert u <= v["V"]["raw_train"], "not raw_train"
    if nurec: assert u <= v["V"]["nurec_train"], "not nurec_train"
    return True
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
