"""Each episode's plans and poses from the Gate B report, one file per episode, for DRIVEJEPA_EXPERT_DIR.

Run inline on 14 Oct 2026 and never saved: recovered from that session's log. Re-run on
gateB_v3_T3.5.json it reproduces `expert_plans_v3` byte for byte (the digest in POINTERS.md).
"""

import json
from pathlib import Path

O = Path("{paddock_runs}/ops/dj_expert")
r = json.loads((O / "gateB_v3_T3.5.json").read_text())
out = Path("{runs}/expert_plans_v3")
n = 0
for ep, ticks in r["ticks"].items():
    path = out / f"{ep}.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plans": {str(t["timestamp"]): t["plan"] for t in ticks},
                                "poses": {str(t["timestamp"]): t["ego2world"] for t in ticks}}))
    n += 1
print(n, "episodes exported to", out)
