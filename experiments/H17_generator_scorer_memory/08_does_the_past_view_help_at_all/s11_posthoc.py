"""Ceiling test, post hoc (information only, not in the pre-registration): true past views against shuffled past views, same reader, same
number of inputs. Uses only the stored out-of-fold predictions. -> out/ceiling_posthoc.json"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s11_ceiling as C
L2 = C.L2


def main():
    D = C.load(); n = len(D["rows"]); T = D["T"]; out = {"made": time.strftime("%F %T"), "pairs": {}}
    for name, pat in (("linear", "fold{}.npz"), ("mlp", "fold{}.npz"), ("mlp_long", "mlplong_fold{}.npz")):
        P = {}; pre = {"linear": "lin|", "mlp": "mlp|", "mlp_long": ""}[name]
        for fo in range(C.N_OUTER):
            z = np.load(C.PDIR / pat.format(fo))
            for cfg in ("PH+V24", "PH+V24shuf"):
                for t in C.ALL_T:
                    P.setdefault((cfg, t), np.full(n, np.nan))[z["te"]] = z[f"{pre}{cfg}|{t}"]
        def err(cfg, t):
            p = P[(cfg, t)]; assert np.isfinite(p).all()
            if t == "T4":
                p = np.clip(p, 1e-7, 1 - 1e-7); return -(T[t] * np.log(p) + (1 - T[t]) * np.log(1 - p))
            return (p - T[t]) ** 2
        out["pairs"][f"{name}: PH+V24 vs PH+V24shuf"] = {mk: {t: L2.gain_ci(err("PH+V24", t), err("PH+V24shuf", t), D["chunk"], m, np.random.RandomState(L2.BOOT_SEED)) for t in C.ALL_T}
                                                       for mk, m in (("all", np.ones(n, bool)), ("event", D["event"]))}
    (C.OUT / "ceiling_posthoc.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["pairs"].items():
        for mk in v:
            print(k, mk, " | ".join(f"{t} {100 * g['gain']:+.2f} [{100 * g['lo']:+.2f},{100 * g['hi']:+.2f}]" for t, g in v[mk].items()))


if __name__ == "__main__":
    main()
