"""neuroNCAP side pairs: score, braking start and impact speed, per run (H15 gate, D6).

    python ncap_braking.py <label>=<neuroncap run dir> ...

Braking start, as defined in GATE_EXPECTATIONS.md before any gate run: from ego_poses.json (10 Hz),
v_k is the distance between pose k and pose k+1 over their time step. The episode ends at the last
pose or at the first k with v_k < 1 m/s, whichever is first. Braking start is the time, from the first
pose, of the last k before the end at which v_k equals its maximum up to the end. If the speed at the
end is within 1 m/s of that peak, the car did not brake.
"""

import json
import math
import sys
from pathlib import Path

PAIRS = ("side-0103", "side-0110")


def braking(poses: dict) -> dict:
    keys = sorted(poses, key=int)
    p = [(int(k) / 1e6, poses[k][0][3], poses[k][1][3]) for k in keys]
    v = [math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) for a, b in zip(p, p[1:])]
    end = next((k for k, x in enumerate(v) if x < 1.0), len(v) - 1)
    upto = v[: end + 1]
    peak = max(upto)
    k_peak = max(k for k, x in enumerate(upto) if x == peak)
    braked = peak - upto[-1] >= 1.0
    return {
        "peak_mps": round(peak, 2),
        "brake_start_s": round(p[k_peak][0] - p[0][0], 2) if braked else None,
        "end_speed_mps": round(upto[-1], 2),
        "duration_s": round(p[-1][0] - p[0][0], 2),
    }


def main() -> None:
    out = {}
    for arg in sys.argv[1:]:
        label, run = arg.split("=", 1)
        run = Path(run)
        samples = {}
        for line in (run / "samples.jsonl").read_text().splitlines():
            s = json.loads(line)
            n = s["native_id"]
            samples[f"{n['scenario']}-{n['scene']}"] = s["metrics"]
        for pair in PAIRS:
            if pair not in samples:
                continue
            m = samples[pair]
            row = {"ncap_score": round(m["ncap_score"], 3), "impact_speed": round(m["impact_speed"], 2),
                   **braking(json.loads((run / pair / "ego_poses.json").read_text()))}
            out[f"{label} {pair}"] = row
            print(f"{label:9s} {pair}  {row}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
