"""Part F.5 drift function: a dump against itself is all zero; a known shift is measured."""

import copy

import numpy as np

import drift as D


def _dump(seed=0, n=5):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        p = rng.normal(size=(32, 8, 3)).astype(np.float32)
        s = rng.uniform(size=32).astype(np.float32)
        out[(f"scene{i}", 40)] = {"proposals": p, "pdm_score": s, "chosen": int(s.argmax())}
    return out


def test_self_is_zero():
    d = _dump()
    s = D.drift(d, d)["summary"]
    assert s["mean_proposal_shift_m"] == 0 and s["max_proposal_shift_m"] == 0
    assert s["mean_grade_change"] == 0 and s["chosen_changed"] == 0


def test_known_shift_and_pick_change():
    a = _dump()
    b = copy.deepcopy(a)
    k = ("scene0", 40)
    b[k]["proposals"][..., 0] += 3.0  # every waypoint moves 3 m in x, then 4 m in y -> 5 m
    b[k]["proposals"][..., 1] += 4.0
    b[k]["pdm_score"] = b[k]["pdm_score"] + 0.1
    b[k]["chosen"] = (a[k]["chosen"] + 1) % 32
    r = D.drift(a, b)
    assert abs(r["per_sample"][k]["proposal_shift_m"] - 5.0) < 1e-5
    assert abs(r["per_sample"][k]["grade_change_abs"] - 0.1) < 1e-6
    assert r["summary"]["chosen_changed"] == 1
