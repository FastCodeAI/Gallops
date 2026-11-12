"""Build 3 environment shim (2026-11-09). A root-disk relief move replaced Paddock/paddock-sid/Alpasim/analysis_3.0 with a symlink to
{runs}/paddock_sid_moved/Alpasim/analysis_3.0 (_move_log_20261109.txt). s35_common.py (read-only, in that tree) locates the vendored
navsim through its own resolved path (s35_common.py:6-11: RETRAIN.parents[1] / "adapters/drivejepa/src/navsim_v2"), which now points at
a NAS folder that does not exist (adapters/ stayed local). Nothing under the earlier lanes is edited: this module re-points the module
constant at the local tree before s35_common.setup_env() runs, in this process (patch) and in every spawned grading worker (init_worker
wraps pai_model_b._init_worker, which calls s35_common.setup_env() in the child). Import this module FIRST."""
import pathlib, sys
TRAIN_PATCH_S35 = pathlib.Path("{runs}/paddock_sid/Alpasim/analysis_3.0/Retrain-data/s3_5_smoke/train_patch")
LOCAL_NAVSIM = pathlib.Path("{runs}/paddock_sid/Alpasim/adapters/drivejepa/src/navsim_v2")
PATCHED = {"applied": False, "was": None}


def patch() -> dict:
    if str(TRAIN_PATCH_S35) not in sys.path:
        sys.path.insert(0, str(TRAIN_PATCH_S35))
    import s35_common
    PATCHED["was"] = str(s35_common.NAVSIM_V2)
    if not s35_common.NAVSIM_V2.exists():
        assert LOCAL_NAVSIM.is_dir(), LOCAL_NAVSIM
        s35_common.NAVSIM_V2 = LOCAL_NAVSIM
        s35_common.VOCAB["orig"] = LOCAL_NAVSIM / "data/8192.npy"
        PATCHED["applied"] = True
    return dict(PATCHED, now=str(s35_common.NAVSIM_V2))


def init_worker():
    """grading-pool initializer: the patch, then pai_model_b._init_worker (its original, kept at import time)."""
    patch()
    _ORIG()


_ORIG = None


def patch_grader(MB) -> None:
    global _ORIG
    if _ORIG is None:
        _ORIG = MB._init_worker
    MB._init_worker = init_worker


patch()
