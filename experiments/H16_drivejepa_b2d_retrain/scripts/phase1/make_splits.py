"""944 clips = the 968 kept (phase0a-labels/results/b2d_overlap_drop.json) minus the 24 trigger twins (recon section_DE_evidence/twins.txt). Held-out about 5 %, by route:
routes in sha1 order until 47 clips. -> splits/clips.json, heldout_clips.txt, train_clips.txt"""
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

R = Path("{ops}/b2d_drivejepa/phase1/splits")
kept = json.load(open("{ops}/phase0a-labels/results/b2d_overlap_drop.json"))["kept"]
twins = set()
for line in open("{ops}/recon_b2d_drivejepa/section_DE_evidence/twins.txt"):
    m = re.search(r"(\[.*\])\s*$", line)
    if m and re.match(r"^\d+ Town", line):
        twins |= set(ast.literal_eval(m.group(1)))
assert len(kept) == 968 and len(twins) == 24 and twins <= set(kept), (len(kept), len(twins), len(twins - set(kept)))
clips = sorted(set(kept) - twins)
assert len(clips) == 944
route = lambda c: re.search(r"(Town\d+\w*?)_Route(\d+)", c).group(0)  # noqa: E731
by = defaultdict(list)
for c in clips:
    by[route(c)].append(c)
held = []
for r in sorted(by, key=lambda r: hashlib.sha1(r.encode()).hexdigest()):
    if len(held) >= 47:
        break
    held += by[r]
train = sorted(set(clips) - set(held))
assert not {route(c) for c in held} & {route(c) for c in train}
json.dump(dict(clips=clips, heldout=sorted(held), train=train, twins_removed=sorted(twins), routes=len(by)), open(R / "clips.json", "w"), indent=0)
(R / "heldout_clips.txt").write_text("\n".join(sorted(held)) + "\n")
(R / "train_clips.txt").write_text("\n".join(train) + "\n")
print(f"{len(clips)} clips, {len(by)} routes; held-out {len(held)} clips ({100 * len(held) / len(clips):.1f} %), train {len(train)}")
