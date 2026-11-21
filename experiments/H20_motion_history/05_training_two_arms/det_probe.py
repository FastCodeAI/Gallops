"""TRAIN step 1 diagnosis: is one training step's gradient repeatable? Same state, same batch, same RNG, twice; list the tensors whose
gradients differ. Then the same with torch.use_deterministic_algorithms(True) (error mode) to name the first op without a deterministic
kernel. Output MH/out/det_probe.json."""
import json, sys, time, random
from pathlib import Path
import numpy as np
import s6_env  # noqa
import s6_env as _SE, pai_model_b as _MB
_SE.patch_grader(_MB)
import train_a4 as TA


def grads(model, opts, b, grader, rng):
    import torch, pai_train as T
    TA.set_rng_state(rng)
    f, t = T.to_dev(b)
    model.train_mode()
    r = TA.step_forward(model, f, t)
    g = grader.collect(grader.submit(t, r["proposals"]))
    L = model.scorer_lessons(r, t, g)
    loss = r["gen"] + L["grade"] + L["collision"] + L["drivable"] + L["human"]
    hs = model.hide_gen.get_state().clone()
    loss.backward()
    out = {n: p.grad.detach().clone() for n, p in model.m.named_parameters() if p.grad is not None}
    out.update({"new." + n: p.grad.detach().clone() for n, p in model.new_params if p.grad is not None})
    model.zero_grad_all()
    return float(loss), out, {k: float(L[k]) for k in ("grade", "human", "collision", "drivable")} | {"gen": float(r["gen"])}


def main():
    import torch, pai_data as PD, pai_model_b as MB, a4_data as D
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    if mode == "det":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    tr, pk, model, ds, info = TA.setup("A")
    grader = MB._GRADER = MB.Grader(TA.POOL)
    opts = model.optimisers()
    bl = PD.batches_017(PD.sampler_017(tr, TA.SEED, 1), TA.BATCH)
    b = ds.batch(bl[0])
    np.random.seed(1)
    rng = TA.rng_state(); hg = model.hide_gen.get_state().clone()
    res = {"mode": mode, "cudnn_benchmark": torch.backends.cudnn.benchmark}
    try:
        l1, g1, p1 = grads(model, opts, b, grader, rng); model.hide_gen.set_state(hg)
        l2, g2, p2 = grads(model, opts, b, grader, rng)
        diff = {n: float((g1[n] - g2[n]).abs().max()) for n in g1 if not torch.equal(g1[n], g2[n])}
        res.update({"loss": [l1, l2], "parts": [p1, p2], "tensors_with_grad": len(g1), "tensors_differ": len(diff),
                    "differ": dict(sorted(diff.items(), key=lambda kv: -kv[1])[:40])})
    except RuntimeError as e:
        res["error"] = str(e)[:2000]
    print(json.dumps({k: v for k, v in res.items() if k != "differ"}), flush=True)
    print(json.dumps(res.get("differ", {}), indent=0)[:4000], flush=True)
    (Path(__file__).resolve().parents[1] / "out" / f"det_probe_{mode}.json").write_text(json.dumps(res, indent=1))
    grader.close()


if __name__ == "__main__":
    main()
    import os; os._exit(0)
