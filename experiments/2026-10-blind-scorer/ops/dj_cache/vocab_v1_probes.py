"""Vocabulary v1 (32 drivable swerves, slots 0-31) followed by probe v1 (32 far-ahead probes, slots 32-63)."""
import importlib.util
import numpy as np

_DIR = "{this_folder}/ops/dj_cache/"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR + name + ".py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VOCAB, _PROBE = _load("vocab_v1"), _load("probe_v1")


def vocabulary(natives, pick_index, speed, accel=0.0):
    a, ma = _VOCAB.vocabulary(natives, pick_index, speed, accel)
    b, mb = _PROBE.vocabulary(natives, pick_index, speed, accel)
    return np.concatenate([a, b]), [dict(m, kind="swerve") for m in ma] + [dict(m, kind="probe") for m in mb]
