#!/usr/bin/env python
"""S10 task 7: assemble the merged full checkpoint of the chosen generator pass the way 06_first_training_run/build_candidate.py built the pass-9
candidate (upstream layout, state_dict keys 'agent._pad_model.*'): encoder tensors bit for bit from the shipped file, every other tensor
from our pass file. Checks: scorer tensors bit-identical to pass 9, encoder tensors bit-identical to the shipped file, and on 16 SEEN
held-out examples (CPU, float32, batch 1) the merged file gives the same plans and scorer outputs as the pass file.
Usage (run_s10_dump_gen.sh environment): s10_build_gen.py PASS_FILE OUT_CKPT REPORT_JSON [--choice S10/final_choice.json]"""
import os, sys, json, time, hashlib, pathlib, argparse
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent; sys.path.insert(0, str(RD / "05_small_overfit_check/train_patch")); import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("pass_file"); ap.add_argument("out"); ap.add_argument("report"); ap.add_argument("--choice", default=""); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
C.setup_env(); import numpy as np, pandas as pd, torch; torch.set_num_threads(a.threads)
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
SC = "_pad_model.scorer."; P9 = C.NAS / "06_first_training_run/run/ckpt/pass_09.pt"; out = pathlib.Path(a.out); assert not out.exists(), f"{out} exists; not overwritten"
if a.choice: ch = json.load(open(a.choice)); assert pathlib.Path(ch["chosen_checkpoint"]) == pathlib.Path(a.pass_file), (ch["chosen_checkpoint"], a.pass_file)
ck = torch.load(a.pass_file, map_location="cpu"); tr = ck["trainable"]; ref9 = torch.load(P9, map_location="cpu"); assert ref9["passno"] == 9; ref9 = ref9["trainable"]
assert sha256(C.CKPT) == C.CKPT_SHA256; base = torch.load(C.CKPT, map_location="cpu")["state_dict"]; new = {}; n_enc = n_new = n_changed = 0
for k, v in base.items():                                                                    # build_candidate.py:8-13
    kk = k.replace("agent._pad_model", "_pad_model")
    if "_backbone" in kk: new[k] = v; n_enc += 1; assert kk not in tr
    else:
        w = tr[kk]; assert w.shape == v.shape and w.dtype == v.dtype, k; new[k] = w.clone(); n_new += 1; n_changed += int(not torch.equal(w, v))
assert set(tr) == {k.replace("agent._pad_model", "_pad_model") for k in base if "_backbone" not in k}
sck = [k for k in tr if k.startswith(SC)]; sc_same = sum(int(torch.equal(tr[k], ref9[k])) for k in sck); gen_diff9 = sum(int(not torch.equal(tr[k], ref9[k])) for k in tr if not k.startswith(SC))
meta = dict(pass_file=str(a.pass_file), pass_file_sha256=sha256(a.pass_file), passno=ck.get("passno"), lr=ck.get("lr"), lam=ck.get("lam"), init=ck.get("init"), base_sha256=C.CKPT_SHA256)
out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(".tmp"); torch.save({"state_dict": new, "s10": meta}, tmp); os.replace(tmp, out)
# re-read the written file and check it tensor by tensor
back = torch.load(out, map_location="cpu")["state_dict"]; assert list(back) == list(base)
enc_same = sum(int(torch.equal(back[k], base[k])) for k in base if "_backbone" in k); sc_back = sum(int(torch.equal(back["agent." + k], ref9[k])) for k in sck); oth_same = sum(int(torch.equal(back["agent." + k], tr[k])) for k in tr)
info = dict(path=str(out), sha256=sha256(out), bytes=out.stat().st_size, written=time.strftime("%Y-%m-%dT%H:%M:%S+05:30"), **meta, tensors_total=len(back), tensors_from_shipped_encoder=n_enc, encoder_tensors_bit_identical_to_shipped=enc_same,
            tensors_from_pass_file=n_new, of_those_bit_identical_to_pass_file=oth_same, pass_tensors_that_differ_from_shipped=n_changed, scorer_tensors=len(sck), scorer_tensors_bit_identical_to_pass9_in_pass_file=sc_same, scorer_tensors_bit_identical_to_pass9_in_merged_file=sc_back,
            generator_tensors=len(tr) - len(sck), generator_tensors_that_differ_from_pass9=gen_diff9, base_checkpoint=str(C.CKPT), pass9_candidate_for_comparison=json.load(open(RD / "06_first_training_run/candidate.json")))
assert enc_same == n_enc and sc_back == len(sck) == sc_same and oth_same == n_new, info
# ---------------------------------------------------------------- 16 held-out examples: merged file against the pass file
import s3_common as C3
from dataset_s35 import CacheOnlyDataset
from torch.utils.data.dataloader import default_collate
man = pd.read_parquet(C.S3 / "manifest.parquet"); hc = set(pd.read_csv(HERE / "seen_heldout_clips.csv").clip_uuid); ho = man[man.clip_uuid.isin(hc)].reset_index(drop=True); assert len(ho) == 4400 and len(hc) == 145
agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=0); agent.eval()
ds = CacheOnlyDataset(cache_path=str(C.NAS / "04_training_set/s1/cache"), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True, has_label2=dict(zip(ho.sample_id, ho.has_label2)),
                      token_clips=dict(zip(ho.sample_id, ho.clip_uuid)), slim_root=str(C.NAS / "06_first_training_run/slim"), split_guard=lambda u: C3._assert_train(list(u)))
ii = sorted(np.random.default_rng(20261008).permutation(len(ds))[:16].tolist()); toks = [ds.tokens[i] for i in ii]
@torch.no_grad()
def plans():
    P, L = [], []
    for i in ii:
        f, t = default_collate([ds[i]]); o = agent.forward(f); P.append(o["proposals"].float().numpy()); L.append(o["pred_logit"].float().numpy())
    return np.concatenate(P), np.concatenate(L)
mi, un = agent.load_state_dict(tr, strict=False); assert not un and all("_backbone" in k for k in mi); agent.eval(); Pa, La = plans()
mi, un = agent.load_state_dict({k[len("agent."):]: v for k, v in back.items()}, strict=True); agent.eval(); Pb, Lb = plans()
mi, un = agent.load_state_dict(ref9, strict=False); agent.eval(); P9p, _ = plans()
info["same_plans_check"] = dict(examples=16, tokens=toks, slices=ho.set_index("sample_id").loc[toks, "type"].tolist(), device="cpu float32 batch 1, the model's own forward", plan_max_abs_diff_merged_vs_pass_file=float(np.abs(Pa - Pb).max()), logit_max_abs_diff_merged_vs_pass_file=float(np.abs(La - Lb).max()),
                                plan_max_abs_diff_vs_pass9_m=float(np.abs(Pa - P9p).max()), plan_shape=list(Pa.shape), all_finite=bool(np.isfinite(Pa).all() and np.isfinite(Pb).all()))
json.dump(info, open(a.report, "w"), indent=1); print(json.dumps(info, indent=1))
