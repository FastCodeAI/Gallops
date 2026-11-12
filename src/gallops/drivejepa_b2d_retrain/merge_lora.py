"""Phase 1, "At the end" (specs/phase1_lora_v3.md in H16's folder): merge one pass's LoRA into the encoder
as a new checkpoint, check tokens against the unmerged encoder on a batch, save with sha256. Sid's
merge() runs unchanged: it is lifted from f_final.py with ast, because that module's other imports
(f_baseline, s2common) are KIT-only. As in f_final.py:64-78 the LoRA merged is ck["lora"], the
learner's, not the slow copy.
    python -m gallops.drivejepa_b2d_retrain.merge_lora <save dir name> <tag>"""

import ast
import copy
import hashlib
import json
import sys
import time
import types

from gallops.drivejepa_b2d_retrain import wiring as W

L, P, M = W.setup()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from gallops.drivejepa_b2d_retrain import frames as D  # noqa: E402

save_name, tag = sys.argv[1], sys.argv[2]
RUN = W.RUNS / "runs/p1_05nov"
OUT = W.RUNS / "merged" / f"p1_05nov_{tag}"
OUT.mkdir(parents=True, exist_ok=False)  # never overwrite
PT = OUT / f"merged_encoder_{tag}.pt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


src = open(W.SID_TRAIN / "f_final.py").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "merge")
ns = dict(copy=copy, nn=nn, torch=torch, L=L, E=types.SimpleNamespace(ENC_PATH=W.ENC_PATH))
exec(ast.get_source_segment(src, fn), ns)
merge = ns["merge"]

assert sha256(W.AGENT_CKPT) == W.AGENT_SHA
ck = torch.load(RUN / save_name / "state.pt", map_location="cpu", weights_only=False)
S = W.SETTINGS
agent = W.build_agent("cuda")
plain = copy.deepcopy(agent.get_submodule(W.ENC_PATH)).eval()
enc = agent.get_submodule(W.ENC_PATH)
assert len(L.wrap(enc, S["lora_rank"], S["lora_alpha"])) == 96
params = dict(enc.named_parameters())
assert set(ck["lora"]) == {n for n in params if ".lora_" in n}
with torch.no_grad():
    for n, t in ck["lora"].items():
        params[n].copy_(t.cuda())
merged = merge(agent).eval()
agent.eval()
menc = merged.get_submodule(W.ENC_PATH)
keys_u = {k.replace(".base.", ".") for k in enc.state_dict() if ".lora_" not in k}
out = dict(
    tag=tag,
    source=str(RUN / save_name),
    phase_step=ck["state"]["phase_step"],
    lora_tensors=len(ck["lora"]),
    keys_match=keys_u == set(menc.state_dict()),
    keys_match_released=set(plain.state_dict()) == set(menc.state_dict()),
)

fc = D.FrameCache()
splits = json.load(open(W.SPLITS))
items = D.pair_items(splits["heldout"], fc.frames, 1, 1)
pick = np.linspace(0, len(items) - 1, 32).astype(int)
x = torch.stack(
    [D.pair_from_arrays(fc.frame(c, i - 1), fc.frame(c, i)) for c, i in (items[k] for k in pick)]
).cuda()


@torch.no_grad()
def tok(e, autocast):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        return torch.cat([P.encode(e, x[b : b + 8]).float() for b in range(0, len(x), 8)])


def diff(a, b):
    d = (a - b).abs()
    return dict(largest=float(d.max()), mean=float(d.mean()))


tu, tm, tb = tok(enc, False), tok(menc, False), tok(plain, False)
out.update(
    batch=int(len(x)),
    token_shape=list(tu.shape),
    token_rms=float(tu.pow(2).mean().sqrt()),
    fp32_merged_vs_unmerged=diff(tm, tu),
    fp32_merged_vs_released=diff(tm, tb),
    bf16_merged_vs_unmerged=diff(tok(menc, True), tok(enc, True)),
    finite=bool(torch.isfinite(tm).all()),
)
torch.save(menc.state_dict(), PT)
out["sha256"] = sha256(PT)
out["bytes"] = PT.stat().st_size
fresh = copy.deepcopy(plain)
fresh.load_state_dict(torch.load(PT, map_location="cuda"), strict=True)
out["reload_bit_equal"] = bool(torch.equal(tok(fresh.eval(), False), tm))
out["saved"] = time.strftime("%F %T")
json.dump(out, open(OUT / "merge_report.json", "w"), indent=1)
open(OUT / "SHA256SUMS", "w").write(f"{out['sha256']}  {PT.name}\n")
print(json.dumps(out, indent=1), flush=True)
print("MERGE_DONE", PT, flush=True)
