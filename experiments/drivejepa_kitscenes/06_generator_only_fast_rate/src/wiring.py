"""Wiring smoke before launch: model counts, one train step, one evaluation.  Nothing saved."""
import json, numpy as np, torch
import phase1 as P1
import kitdata as DT
meta = torch.load(DT.DATA / "scene_meta.pt", weights_only=False)
mom = torch.load(DT.DATA / "moments.pt", weights_only=False)
m = P1.Model()
print(json.dumps({"total": m.gen_total, "trainable": m.n_train, "tensors": len(m.train_params),
                  "fixed": {n: p.numel() for n, p in m.fixed}}))
ds = P1.TrainSet(mom["train"], meta)
b = torch.utils.data.default_collate([ds[i] for i in range(32)])
f, t = P1.to_gpu(b)
m.train_mode(); l = m.loss(f, t); l.backward()
print("loss", float(l), "nonzero", sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for _, p in m.train_params), "gm", m.gm)
d = P1.evaluate(m, mom["selection"][:3], meta); print("eval ok", len(d), "gm", m.gm)
