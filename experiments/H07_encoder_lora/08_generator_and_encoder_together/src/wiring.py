"""Pre-launch smoke test: phase-2 model build, 2 G2 steps, merged-copy evaluation on 5 moments, drift path."""
import json, torch, numpy as np
import phase2 as Q
from s2common import CKPT as ENC_CKPT
ctx = Q.Ctx()
p3 = torch.load(ENC_CKPT / "pass_03" / "state.pt", map_location="cpu", weights_only=False)
print("pass3 keys", sorted(p3))
m = Q.Model(True, p3)
print("trainable", m.n_trainable)
g2 = Q.gate_g2(m, ctx, n_steps=2)
print(json.dumps(g2, default=str)[:1500])
a = m.eval_agent()
d = Q.evaluate(m, a, ctx.main[:5], ctx.meta)
print("eval ok", {k: v["chosen"] for k, v in d.items()})
po = Q.projector_out(a, ctx.sel[:3]); print("proj", po[0].shape, Q.cos_drift(po, po))
print("p95 same first 5:", all(np.array_equal(d[k]["proposals"], ctx.p95[k]["proposals"]) for k in d))
