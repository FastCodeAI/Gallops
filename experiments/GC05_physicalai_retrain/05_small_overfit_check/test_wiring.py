#!/usr/bin/env python
"""S3.5 STEP 1 wiring tests (no training). CPU only. Writes wiring_checks.json next to this file.
Run: <conda drive-jepa python> test_wiring.py [unit|full640]
  unit    : W1a-W1d on small batches (640-sample dataloader sweep included)
  full640 : forward + backward on every one of the 640 masked raw-only samples (batches of 8)"""
import sys, json, time, hashlib, pathlib, tempfile
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "train_patch"))
import s35_common as C
C.setup_env()
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
torch.set_num_threads(8); MODE = sys.argv[1] if len(sys.argv) > 1 else "unit"
OUT = HERE / ("wiring_checks.json" if MODE == "unit" else "wiring_full640.json"); R = {}; T0 = time.time()
def rec(k, ok, **kw):
    R[k] = dict(ok=bool(ok), **kw); print(("PASS " if ok else "FAIL ") + k, json.dumps(kw, default=str)[:400], f"| {time.time()-T0:.0f} s", flush=True)
    json.dump(R, open(OUT, "w"), indent=1, default=str)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
gmax = lambda ps: max([float(p.grad.abs().max()) for p in ps if p.grad is not None] or [0.0])
def groups(agent):
    m = agent._pad_model
    return dict(scorer=list(m.scorer.parameters()), generator=list(m._trajectory_head.parameters()) + list(m.hist_encoding.parameters()) + list(m.init_feature.parameters()), encoder=list(m._backbone.parameters()))
def fwd_loss(agent, batch, seed=0):
    f, t = batch; f = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()}        # forward slices lidar2img in place
    np.random.seed(seed); torch.manual_seed(seed); pred = agent.forward(f); np.random.seed(seed)
    return pred, agent.compute_loss(f, t, pred)

# ---------------------------------------------------------------- data
samples = pd.read_csv(C.S2 / "s21_masked/samples.csv")
agent = C.build_agent(vocab="ext", index_dir=str(HERE / "test_fixtures/index_ext"), pool_workers=4)
agent.train(); G = groups(agent); calls = {"n": 0}; _wm = agent.worker_map
def counted(dp): calls["n"] += 1; return _wm(dp)
agent.worker_map = counted
raw = C.build_dataset(C.MASKED_CACHE, agent)                               # guard: raw_train only
assert len(raw) == 640 and set(raw.tokens) == set(samples.token), len(raw)

if MODE == "full640":
    dl = DataLoader(raw, batch_size=8, num_workers=6, shuffle=False); n = 0; worst = 0.0; losses = []; secs = []
    for b in dl:
        t1 = time.time(); agent.zero_grad(set_to_none=True); pred, ld = fwd_loss(agent, b, seed=n); ld["loss"].backward(); secs.append(time.time() - t1)
        assert torch.isfinite(ld["loss"]) and ld["n_label2"] == 0 and not b[1]["has_label2"].any() and all(p == "" for p in b[1]["metric_cache_path"])
        assert all(p.grad is None for p in G["scorer"]) and all(p.grad is None for p in G["encoder"]) and gmax(G["generator"]) > 0
        assert float(b[0]["ego_status"][..., 5:7].abs().max()) == 0.0
        n += len(b[1]["token"]); losses.append(float(ld["loss"])); print(n, f"{losses[-1]:.3f} {secs[-1]:.1f} s", flush=True)
    rec("W1a_full640", n == 640 and calls["n"] == 0, samples=n, batches=len(losses), loss_min=min(losses), loss_mean=float(np.mean(losses)), loss_max=max(losses),
        scorer_grad="None on every batch", metric_cache_lookups=calls["n"], cpu_s_per_batch8=float(np.median(secs)))
    sys.exit(0)

# S2 pilot scenes (metric cache + pad_label2 next to each sample; the anchor index is a DUMMY fixture: S2 wrote none)
meta = [l.strip() for l in open(next((C.S2_CACHE / "metadata").glob("*.csv"))).read().splitlines()[1:]]
s2tok = {p.split("/")[-2]: p for p in meta}
lab = C.build_dataset(C.S2_CACHE, agent, nurec=True, log_names=sorted({p.split("/")[-3] for p in meta}))
labelled = [t for t in lab.tokens if lab.label2_flag(t)]; assert len(labelled) >= 8, len(labelled)
rng = np.random.default_rng(20261006); FX = HERE / "test_fixtures"
for kind in ("orig", "ext"): (FX / f"index_{kind}").mkdir(parents=True, exist_ok=True)
for t in labelled:
    np.save(FX / f"index_orig/{t}.npy", np.sort(rng.choice(8192, 12, replace=False)))
    np.save(FX / f"index_ext/{t}.npy", np.sort(np.r_[rng.choice(8192, 8, replace=False), 8192 + rng.choice(2048, 8, replace=False)]))
pick_raw = [samples[samples.type == ty].token.iloc[k] for ty, k in (("A", 3), ("A", 40), (samples.type.unique()[1], 5), (samples.type.unique()[-1], 9))]
pick_lab = [labelled[0], labelled[len(labelled) // 3], labelled[2 * len(labelled) // 3], labelled[-1]]
item = lambda ds, tok: ds[ds.tokens.index(tok)]
B_raw = default_collate([item(raw, t) for t in pick_raw]); B_lab = default_collate([item(lab, t) for t in pick_lab])
B_mix = default_collate([item(raw, pick_raw[0]), item(lab, pick_lab[0]), item(raw, pick_raw[1]), item(lab, pick_lab[1])])
print("feature keys", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in B_raw[0].items()}); print("target keys", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in B_raw[1].items()})

# ---------------------------------------------------------------- W1d init + freeze (before any step)
st = torch.load(C.CKPT, map_location="cpu")["state_dict"]; own = agent.state_dict()
mapped = {k.replace("agent._pad_model", "_pad_model"): v for k, v in st.items()}
rec("W1d_init_strict", set(mapped) == set(own) and all(torch.equal(mapped[k], own[k]) for k in own), ckpt=str(C.CKPT), sha256_expected=C.CKPT_SHA256, tensors=len(own), missing=len(set(own) - set(mapped)), unexpected=len(set(mapped) - set(own)))
npar = {k: int(sum(p.numel() for p in v)) for k, v in G.items()}; ntrain = {k: int(sum(p.numel() for p in v if p.requires_grad)) for k, v in G.items()}
opt = agent.get_optimizers(); in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
import torch.nn as nn
drops = [(n, m.p) for n, m in agent._pad_model._backbone.named_modules() if isinstance(m, nn.Dropout) and m.p > 0]; dpath = [n for n, m in agent._pad_model._backbone.named_modules() if type(m).__name__ == "DropPath" and getattr(m, "drop_prob", 0)]
rec("W1d_encoder_frozen", ntrain["encoder"] == 0 and not any(id(p) in in_opt for p in G["encoder"]) and all(id(p) in in_opt for p in G["scorer"] + G["generator"]),
    params=npar, trainable=ntrain, optimiser_groups=[(len(g["params"]), g["lr"]) for g in opt.param_groups], encoder_dropout_p_gt0=len(drops), encoder_droppath=len(dpath), projector_in_encoder_group=True)
del st, mapped

# ---------------------------------------------------------------- W1b vocabulary
v0 = np.load(C.VOCAB["orig"]); v1 = np.load(C.VOCAB["ext"])
rec("W1b_vocab_file", v1.shape == (10240, 40, 3) and v1.dtype == np.float32 and v1[:8192].tobytes() == v0.tobytes() and agent.anchors.shape == (10240, 8, 3),
    orig=str(v0.shape), ext=str(v1.shape), first_8192_bit_identical=bool(v1[:8192].tobytes() == v0.tobytes()), sha256_orig=sha(C.VOCAB["orig"]), sha256_ext=sha(C.VOCAB["ext"]), agent_anchors=str(agent.anchors.shape))
dims = sorted({(k, tuple(v.shape)) for k, v in own.items() if any(d in (8192, 10240) for d in v.shape)})
rec("W1b_no_model_tensor_depends_on_vocab", not dims, state_dict_tensors_with_dim_8192_or_10240=dims, init_feature=str(tuple(own["_pad_model.init_feature.weight"].shape)), note="only self.anchors (agent :49-50, used :246) has the vocabulary size")

# ---------------------------------------------------------------- W1c acceleration slots
raw_keep = C.build_dataset(C.MASKED_CACHE, agent, zero_accel=False)
errs = []; other = True
for t in pick_raw + list(samples.token.iloc[::97]):
    a = item(raw_keep, t)[0]["ego_status"]; z = item(raw, t)[0]["ego_status"]; s = samples[samples.token == t].iloc[0]
    errs.append(float(np.abs(a[-1, 5:7].numpy() - np.array([s.ax, s.ay])).max())); other &= bool(a.shape == z.shape == (1, 11) and torch.equal(a[..., :5], z[..., :5]) and torch.equal(a[..., 7:], z[..., 7:]) and float(z[..., 5:7].abs().max()) == 0)
rec("W1c_accel_slots", max(errs) < 1e-4 and other, checked=len(errs), max_abs_diff_slot5_6_vs_samples_csv_ax_ay=max(errs), ego_status_shape=str(tuple(a.shape)), other_9_slots_untouched=other, flag_default="zero_accel=True")
n = 0; nz = 0.0; t1 = time.time()
for f, t in DataLoader(raw, batch_size=16, num_workers=6):
    n += len(t["token"]); nz = max(nz, float(f["ego_status"][..., 5:7].abs().max())); assert not t["has_label2"].any() and all(p == "" for p in t["metric_cache_path"]) and torch.isfinite(t["trajectory"]).all() and t["trajectory"].shape[1:] == (8, 3)
rec("W1c_all_640_loaded", n == 640 and nz == 0.0, samples=n, max_abs_accel=nz, has_label2_any=False, load_s=round(time.time() - t1, 1))

# ---------------------------------------------------------------- W1a edits
def backward(batch, seed=0):
    agent.zero_grad(set_to_none=True); pred, ld = fwd_loss(agent, batch, seed)
    for k in ("pred_logit", "pred_agents_states", "pred_area_logit"): pred[k].retain_grad()
    ld["loss"].backward(); return pred, ld
c0 = calls["n"]; t1 = time.time(); pred, ld = backward(B_raw); dt = time.time() - t1
sg = [p.grad for p in G["scorer"]]
rec("W1a_raw_only_batch", bool(torch.isfinite(ld["loss"])) and ld["n_label2"] == 0 and calls["n"] == c0 and all(g is None or float(g.abs().max()) == 0 for g in sg) and gmax(G["generator"]) > 0 and all(pred[k].grad is None for k in ("pred_logit", "pred_agents_states", "pred_area_logit")),
    tokens=pick_raw, loss=float(ld["loss"]), trajectory_loss=float(ld["trajectory_loss"]), scorer_terms=[float(ld[k]) for k in ("sub_score_loss", "final_score_loss", "pred_ce_loss", "pred_l1_loss", "pred_area_loss")],
    scorer_param_tensors=len(sg), scorer_grads_None=sum(g is None for g in sg), scorer_grad_max=gmax(G["scorer"]), generator_grad_max=gmax(G["generator"]), encoder_grad_max=gmax(G["encoder"]), metric_cache_lookups=calls["n"] - c0, cpu_s_fwd_bwd_b4=round(dt, 1))
c0 = calls["n"]; pred, ld_lab = backward(B_lab)
rec("W1a_labelled_batch", bool(torch.isfinite(ld_lab["loss"])) and ld_lab["n_label2"] == 4 and calls["n"] == c0 + 1 and gmax(G["scorer"]) > 0 and gmax(G["generator"]) > 0,
    tokens=pick_lab, **{k: float(v) for k, v in ld_lab.items()}, scorer_grad_max=gmax(G["scorer"]), generator_grad_max=gmax(G["generator"]), scoring_calls=calls["n"] - c0)
pred, ld_mix = backward(B_mix); m = B_mix[1]["has_label2"]
gz = {k: (float(pred[k].grad[~m].abs().max()), float(pred[k].grad[m].abs().max())) for k in ("pred_logit", "pred_agents_states", "pred_area_logit")}
rows = torch.nonzero(m).flatten(); sub_pred = {k: (v[rows] if torch.is_tensor(v) else v) for k, v in pred.items()}; sub_pred["proposal_list"] = [p[rows] for p in pred["proposal_list"]]
sub_t = {k: (v[rows] if torch.is_tensor(v) else [v[i] for i in rows.tolist()]) for k, v in B_mix[1].items()}
np.random.seed(0); ld_sub = agent.compute_loss(None, sub_t, sub_pred); terms = ("sub_score_loss", "final_score_loss", "pred_ce_loss", "pred_l1_loss", "pred_area_loss")
dmax = max(abs(float(ld_mix[k]) - float(ld_sub[k])) for k in terms)
rec("W1a_mixed_batch", bool(torch.isfinite(ld_mix["loss"])) and ld_mix["n_label2"] == 2 and all(a == 0.0 and b > 0 for a, b in gz.values()) and dmax < 1e-6,
    has_label2=m.tolist(), loss=float(ld_mix["loss"]), grad_on_scorer_outputs_raw_rows_vs_labelled_rows=gz, scorer_terms_mixed=[float(ld_mix[k]) for k in terms], scorer_terms_labelled_rows_alone=[float(ld_sub[k]) for k in terms], max_abs_diff=dmax)

# pseudo-target term (edit 2): the labelled rows' index changes the loss, a raw-only row's never does
agent._index_dir = str(FX / "index_orig"); agent.anchors = v0[:, 4::5]; agent._index_kind = "orig"
_, ld_o = fwd_loss(agent, B_lab); _, ld_ro = fwd_loss(agent, B_raw)
agent._index_dir = str(FX / "index_ext")                                    # ext index (rows >= 8192) against the ORIGINAL vocabulary must be refused
try: fwd_loss(agent, B_lab); refused = False
except AssertionError as e: refused = "anchor index beyond the vocabulary" in str(e)
agent.anchors = v1[:, 4::5]; agent._index_kind = "ext"; _, ld_e = fwd_loss(agent, B_lab); _, ld_re = fwd_loss(agent, B_raw)
rec("W1b_vocab_flag_in_loss", refused and abs(float(ld_o["min_loss"]) - float(ld_e["min_loss"])) > 1e-6 and float(ld_ro["loss"]) == float(ld_re["loss"]),
    labelled_min_loss_orig_index=float(ld_o["min_loss"]), labelled_min_loss_ext_index=float(ld_e["min_loss"]), raw_only_loss_orig=float(ld_ro["loss"]), raw_only_loss_ext=float(ld_re["loss"]), ext_index_on_orig_vocab_refused=refused,
    note="DUMMY index fixtures (12 orig rows; 8 orig + 8 added rows) - shape / wiring only")

# ---------------------------------------------------------------- upstream equivalence on an all-labelled batch
import navsim.agents.drive_jepa_perception_based.drive_jepa_agent as UA
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
import os
tmp = pathlib.Path(tempfile.mkdtemp(prefix="s35_upstream_", dir=os.environ.get("S35_SCRATCH"))); root = tmp / "Drive-JEPA-cache"; (root / "train_metric_cache_v2_pruned/metadata").mkdir(parents=True); lines = ["file_name"]
for t in pick_lab:
    lg = s2tok[t].split("/")[-3]; d = root / "train_metric_cache_v2_pruned" / lg / t; d.mkdir(parents=True); (d / "metric_cache.pkl").symlink_to(s2tok[t]); lines.append(str(d / "metric_cache.pkl"))
    di = root / "anchors_scores_index_v2" / lg / t; di.mkdir(parents=True); (di / "metric_cache.pkl.npy").symlink_to(FX / f"index_orig/{t}.npy")
(root / "train_metric_cache_v2_pruned/metadata/m.csv").write_text("\n".join(lines) + "\n")
real_root = os.environ["NAVSIM_EXP_ROOT"]; os.environ["NAVSIM_EXP_ROOT"] = str(tmp); _M = UA.DriveJEPAModel; UA.DriveJEPAModel = lambda cfg: agent._pad_model     # share the already-built model; no vendored file is edited
up = UA.DriveJEPAAgent(config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=""); UA.DriveJEPAModel = _M; os.environ["NAVSIM_EXP_ROOT"] = real_root; up.train()
agent._index_dir = str(FX / "index_orig"); agent.anchors = v0[:, 4::5]; agent._index_kind = "orig"; cmp = {}
for impl in ("upstream", "s3"):
    agent._scorer_impl = impl; pred, ld_p = fwd_loss(agent, B_lab, seed=7)
    tu = {k: v for k, v in B_lab[1].items() if k in ("trajectory", "token")}; np.random.seed(7); ld_u = up.compute_loss(None, tu, pred)
    cmp[impl] = max(abs(float(ld_p[k]) - float(ld_u[k])) for k in ld_u)
agent._scorer_impl = "s3"; agent._index_dir = str(FX / "index_ext"); agent.anchors = v1[:, 4::5]; agent._index_kind = "ext"
rec("W1a_equals_upstream_when_all_labelled", cmp["upstream"] == 0.0 and cmp["s3"] < 1e-6, max_abs_diff_any_loss_key=cmp, upstream_loss=float(ld_u["loss"]), keys=len(ld_u),
    note="vendored DriveJEPAAgent.pad_loss vs patched, same predictions, same seed, original vocabulary; S2 caches use the Pacifica box and carry no D3 flag, so the s3 scorer must agree too")
del up

# ---------------------------------------------------------------- one optimiser step: who moves
def snap(): return {k: [p.detach().clone() for p in v] for k, v in G.items()}
def moved(a): return {k: bool(any(not torch.equal(x, p) for x, p in zip(a[k], G[k]))) for k in G}
opt = agent.get_optimizers(); s0 = snap(); backward(B_raw); opt.step(); mv1 = moved(s0)
rec("W1a_raw_only_step_leaves_scorer", mv1 == dict(scorer=False, generator=True, encoder=False), weights_changed_after_one_raw_only_Adam_step=mv1)
s1 = snap(); backward(B_lab); opt.step(); mv2 = moved(s1); s2 = snap(); backward(B_raw); opt.step(); mv3 = moved(s2)
rec("W1d_frozen_encoder_never_moves", not mv2["encoder"] and not mv3["encoder"] and mv2["scorer"] and not mv3["scorer"], after_labelled_step=mv2, after_following_raw_only_step=mv3,
    note="torch Adam skips parameters whose grad is None, so with zero_grad(set_to_none=True) a raw-only step leaves the scorer bit-identical even after labelled steps have built momentum")

# ---------------------------------------------------------------- S3 layout, real extended index (read-only peek at already graded scenes)
try:
    gr = pd.concat([pd.read_parquet(p) for p in sorted((C.S3 / "grade_rows").glob("*.parquet"))[:60]]); gr = gr[(gr.status == "ok") & (gr.n_ok_added > 0)]
    pk = gr.sort_values("speed", ascending=False).drop_duplicates("uuid").head(2); pk = pd.concat([pk, gr[gr.speed < 5].head(2)])
    s3 = C.build_dataset(C.S3_CACHE, agent, nurec=True, log_names=sorted(pk.uuid.unique()), tokens=list(pk.sample_id)); agent._index_dir = None
    B3 = default_collate([s3[i] for i in range(len(s3))]); idx = [np.load(pathlib.Path(p).parent / "anchors_scores_index_ext.npy") for p in B3[1]["metric_cache_path"]]
    agent.zero_grad(set_to_none=True); pred, ld3 = fwd_loss(agent, B3); ld3["loss"].backward()
    rec("W1b_s3_layout_real_ext_index", bool(torch.isfinite(ld3["loss"])) and bool(B3[1]["has_label2"].all()) and gmax(G["scorer"]) > 0, tokens=list(pk.sample_id), speeds=[round(float(x), 1) for x in pk.speed],
        index_rows=[int(len(a)) for a in idx], index_rows_ge_8192=[int((a >= 8192).sum()) for a in idx], **{k: float(v) for k, v in ld3.items()}, note="S3 not final; read-only")
except Exception as e:
    rec("W1b_s3_layout_real_ext_index", False, error=repr(e)[:300])
print("ALL PASS" if all(v["ok"] for v in R.values()) else "FAILURES: " + ", ".join(k for k, v in R.items() if not v["ok"]))
