"""Forward parity of the torch-2.8/cu128 overlay against the shipped env, before any training. Usage: s35_parity.py <tag> <device>"""
import sys, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "train_patch")); import s35_common as C; C.setup_env()
import torch, pandas as pd
from torch.utils.data.dataloader import default_collate
tag, dev = sys.argv[1], sys.argv[2]; sl = pd.read_parquet(HERE / "slice_train.parquet"); toks = list(sl.sample_id.iloc[::150][:8])
agent = C.build_agent(vocab="ext", pool_workers=0); ds = C.build_dataset(C.NAS / "04_training_set/s1/cache", agent, tokens=toks, log_names=sorted(sl[sl.sample_id.isin(toks)].clip_uuid.unique())); agent.to(dev).eval()
f, t = default_collate([ds[i] for i in range(len(ds))]); f = {k: v.to(dev) for k, v in f.items()}
with torch.no_grad(): p = agent.forward(f)
np.savez(HERE / f"parity_{tag}.npz", proposals=p["proposals"].float().cpu().numpy(), pdm=p["pdm_score"].float().cpu().numpy(), tokens=np.array(toks)); print(tag, torch.__version__, dev, "ok", p["proposals"].shape)
