#!/usr/bin/env python
"""0.1.7 build gate, training side: the S10 readout's model (s35_common.build_agent, torch 2.8 overlay, CPU, float32, eval) with the merged
checkpoint and the batch fix on gives its 32 plans for the 16 SEEN held-out examples of 12_second_training_run/gen_ckpt.json (same_plans_check), one
example per forward. The feature dicts are saved with the plans so the adapter side (gate_adapter_side.py) is fed the same inputs.
Run by gate_017.sh (environment of 12_second_training_run/run_s10_dump_gen.sh). Writes OUT/train_side.npz only."""
import os, sys, json, pathlib, hashlib
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parents[1]; S10 = RD / "12_second_training_run"; OUT = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(RD / "05_small_overfit_check/train_patch")); sys.path.insert(0, str(S10)); sys.path.insert(0, str(S10 / "train_patch"))
import s35_common as C
C.setup_env(); import numpy as np, pandas as pd, torch; torch.set_num_threads(4)
import s3_common as C3
from dataset_s35 import CacheOnlyDataset
from torch.utils.data.dataloader import default_collate
import s10_dump_gen_ckpt as LK, batch_fix as BF
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
info = json.load(open(S10 / "gen_ckpt.json")); CK = pathlib.Path(info["path"]); assert sha(CK) == info["sha256"]; toks = info["same_plans_check"]["tokens"]
man = pd.read_parquet(C.S3 / "manifest.parquet"); hc = set(pd.read_csv(S10 / "seen_heldout_clips.csv").clip_uuid); ho = man[man.clip_uuid.isin(hc)].reset_index(drop=True); assert len(ho) == 4400
agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval(); m = agent._pad_model
print(LK.load_ckpt(m, CK, torch), flush=True); print("batch_fix:", {k: len(v) for k, v in BF.install(m).items()}, flush=True); agent.eval(); BF.set_fix(1, 1, 1); assert not m.training
ds = CacheOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True, has_label2=dict(zip(ho.sample_id, ho.has_label2)),
                      token_clips=dict(zip(ho.sample_id, ho.clip_uuid)), slim_root=str(C.NAS / "06_first_training_run/slim"), split_guard=lambda u: C3._assert_train(list(u)))
idx = {t: i for i, t in enumerate(ds.tokens)}; Z = {}; P = []; L = []
with torch.no_grad():
    for n, t in enumerate(toks):
        f, tg = default_collate([ds[idx[t]]]); assert tg["token"][0] == t
        for k, v in f.items():
            if n == 0: print("feature", k, tuple(v.shape), v.dtype, flush=True)
            Z[f"f{n:02d}__{k}"] = v.numpy()
        o = agent.forward(f); P.append(o["proposals"].float().numpy()); L.append(o["pred_logit"].float().numpy()); print(n, t, P[-1].shape, flush=True)
P = np.concatenate(P); L = np.concatenate(L); assert P.shape == (16, 32, 8, 3) and np.isfinite(P).all()
np.savez(OUT / "train_side.npz", plans=P, logits=L, tokens=np.array(toks), slices=np.array(info["same_plans_check"]["slices"]), ckpt_sha256=np.array(info["sha256"]), torch=np.array(torch.__version__), **Z)
print("TRAIN SIDE DONE", P.shape, torch.__version__)
