"""S3.5 shared wiring: paths, flags, split guard, agent / dataset builders. The vendored Drive-JEPA / navsim_v2 tree is imported, never
edited; the two patched copies live next to this file (drive_jepa_agent_s35.py, dataset_s35.py) with their diffs.
Interpreter: the conda drive-jepa python (navsim + nuplan + mmcv). cwd is switched to navsim_v2 by setup_env() because upstream opens
./data/8192.npy and ./vjepa2/configs/... relative to cwd."""
import os, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
SMOKE = HERE.parent
RETRAIN = SMOKE.parent
S3 = RETRAIN / "04_training_set"
S2 = RETRAIN / "03_grade_candidates"
NAVSIM_V2 = RETRAIN.parents[1] / "adapters/drivejepa/src/navsim_v2"
BULK = pathlib.Path("{runs}/alpasim_bulk")
NAVSIM_EXP_ROOT = BULK / "navsim-exp"
CKPT = NAVSIM_EXP_ROOT / "Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
CKPT_SHA256 = "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05"   # adapters/drivejepa/real_model.py EXPECTED
VOCAB = {"orig": NAVSIM_V2 / "data/8192.npy", "ext": S3 / "vocab_10240.npy"}
NAS = pathlib.Path("{runs}/drivejepa_retrain")
MASKED_CACHE = NAS / "03_grade_candidates/s1_masked/cache"     # 640 raw-only samples (S2.1 bonnet-masked rerun of the S1 pilot)
S2_CACHE = NAS / "03_grade_candidates/s1/cache"                # 10 nurec_train pilot scenes with metric caches (S2)
S3_CACHE = NAS / "04_training_set/s1/cache"              # S3 layout: metric cache + anchor indices next to each sample


def setup_env():
    os.environ.setdefault("NAVSIM_EXP_ROOT", str(NAVSIM_EXP_ROOT))
    os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")
    for p in (HERE, S3, NAVSIM_V2):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(NAVSIM_V2)
    _stub_mmcv_ext()


def _stub_mmcv_ext():
    """[S35 Step 3] GPU route: torch 2.8+cu128 is layered over the conda env (PYTHONPATH overlay) because torch 2.1+cu121 has no
    sm_120 kernels. mmcv-full 1.7.2's compiled `_ext` was built against torch 2.1 and cannot load under 2.8, yet upstream imports it
    at module import time (bevformer/encoder.py:13 `ext_loader.load_ext`). When it cannot load, a stub takes its place: every symbol
    exists, and calling any of them raises. The model then has to use the pure-PyTorch deformable attention (DRIVE_JEPA_MSDA=pytorch,
    msda_compat.py), which the parity check compares against the compiled path."""
    import importlib, types
    try:
        importlib.import_module("mmcv._ext"); return False
    except ImportError:
        pass
    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"): raise AttributeError(name)
            def _missing(*a, **k): raise RuntimeError(f"mmcv._ext.{name} is stubbed (compiled ops unavailable under this torch); set DRIVE_JEPA_MSDA=pytorch")
            return _missing
    sys.modules["mmcv._ext"] = _Stub("mmcv._ext"); os.environ["DRIVE_JEPA_MSDA"] = "pytorch"; return True


def guard(uuids, nurec=False):
    """Exam / test / val assertion (S3 Z1 guard, s3_common.assert_train): raw_train only; nurec_train as well when nurec=True."""
    import s3_common as C3
    return C3.assert_train(list(uuids), nurec=nurec)


def build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", index_dir=None, pool_workers=16, lr=1e-4, score_chunk=2):
    """Flags: vocab orig|ext (1b), freeze_encoder (1d), init from the shipped checkpoint (1d). lr = upstream agent yaml (1e-4)."""
    setup_env()
    from drive_jepa_agent_s35 import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    agent = DriveJEPAAgent(config=DriveJEPAConfig(), lr=lr, checkpoint_path="", vocab_path=str(VOCAB[vocab]), index_kind=vocab,
                           init_checkpoint=str(CKPT) if init else None, freeze_encoder=freeze_encoder, scorer_impl=scorer_impl,
                           index_dir=index_dir, pool_workers=pool_workers, score_chunk=score_chunk)
    agent.initialize()
    return agent


def build_dataset(cache, agent, tokens=None, zero_accel=True, has_label2=None, nurec=False, log_names=None):
    from dataset_s35 import CacheOnlyDataset
    return CacheOnlyDataset(cache_path=str(cache), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(),
                            log_names=log_names, zero_accel=zero_accel, has_label2=has_label2, tokens=tokens,
                            split_guard=lambda u: guard(u, nurec=nurec))
