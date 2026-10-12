#!/usr/bin/env python
"""S10 phase 1: generator-only continuation from S4 pass 9. Extends 06_first_training_run/s4_train.py; the only model-side change is the ADDED
methods in train_patch/drive_jepa_agent_s10.py (diff next to it). Trained: _pad_model.hist_encoding, init_feature, _trajectory_head
(one shared refiner listed 4 times). Frozen: encoder (_backbone) and scorer (_pad_model.scorer: no grad, not in the optimiser, eval
mode). Loss = S4 generator loss (human term + 0.5 x catalogue term, stages combined with prev_weight) + w x spacing term; no scorer
loss, nothing graded (catalogue indices are read from anchors_scores_index_ext.npy directly). Kept from S4: batch 32, 16-bit mixed
precision with the loss in float32, no clipping, GridMask in train mode, zero_accel, sampler (type A or graded drawn twice), seed.
Data = S4 manifest minus the 145 SEEN held-out clips of the S7 split (s9_lib.seen_split); those clips are the readout set.
Modes: gate_a | probe (spacing_norm.json) | readout (pass-9 zero reference, = --readout-only) | train."""
import os, sys, json, time, pathlib, argparse, hashlib
HERE = pathlib.Path(__file__).resolve().parent; RD = HERE.parent
sys.path.insert(0, str(RD / "05_small_overfit_check" / "train_patch")); sys.path.insert(0, str(HERE / "train_patch"))
import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="train", choices=["gate_a", "gate_fix", "probe", "readout", "train", "diag", "diag2"]); ap.add_argument("--readout-only", action="store_true")
ap.add_argument("--run", default="dev"); ap.add_argument("--lam", type=float, default=None); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--passes", type=int, default=1); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=14); ap.add_argument("--cap-gb", type=float, default=12.0); ap.add_argument("--reserve-gb", type=float, default=8.0)
ap.add_argument("--seed", type=int, default=20261008); ap.add_argument("--init", default=str(C.NAS / "06_first_training_run/run/ckpt/pass_09.pt")); ap.add_argument("--csv-every", type=int, default=50); ap.add_argument("--probe-batches", type=int, default=300); ap.add_argument("--readout-batching", default="clip", choices=["clip", "shuffle", "one"])   # clip = manifest (clip) order, batch 32; shuffle = seeded shuffle, batch 32; one = batch of 1
ap.add_argument("--batch-fix", default="on", choices=["on", "off"])   # on = fix1+fix2+fix3 of train_patch/batch_fix.py in training, probe and readout (task 5; a no-op at batch 1)
ap.add_argument("--ref-range", default="std", choices=["std", "wide"])   # wide = forward reference axis -32 m .. x_hi of ref_range.json (train_patch/ref_range.py)
ap.add_argument("--tf32", default="default", choices=["default", "off"])   # gate_fix only: off = no TF32 in cuDNN / matmul (true float32 on the GPU)
a = ap.parse_args()
if a.readout_only: a.mode = "readout"
T0 = time.time(); NASOUT = C.NAS / "12_second_training_run" / "gen" / a.run; NASOUT.mkdir(parents=True, exist_ok=True); SLIM = C.NAS / "06_first_training_run" / "slim"
P9 = C.NAS / "06_first_training_run/run/ckpt/pass_09.pt"; P9_SHA = "b3130f1f2155f5a9a5c06191831dfba0e0c07a9841689b39d386c12061328a5a"   # 08_scorer_diagnosis/REPORT.md:58
C.setup_env()
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import s3_common as C3
from dataset_s35 import CacheOnlyDataset
from drive_jepa_agent_s10 import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
log_f = open(NASOUT / f"{a.mode}.log", "a")
def log(*x):
    s = f"[{time.strftime('%m-%d %H:%M:%S')} +{(time.time()-T0)/60:6.1f}m] " + " ".join(str(v) for v in x); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def fv(x): return float(x.detach().float().cpu()) if torch.is_tensor(x) else float(x)
dev = torch.device(a.device); cuda = dev.type == "cuda"
if cuda:
    tot = torch.cuda.get_device_properties(0).total_memory; torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 2**30 / tot))
log("launch args:", vars(a), "| torch", torch.__version__, "| script sha256", sha256(__file__)[:16], "| agent patch sha256", sha256(HERE / "train_patch/drive_jepa_agent_s10.py")[:16])

# ---------------------------------------------------------------- data: S4 manifest minus the 145 SEEN held-out clips
tr_all = pd.read_parquet(C.S3 / "manifest.parquet"); seen = pd.read_parquet(RD / "08_scorer_diagnosis/set_seen.parquet")
_nt = torch.get_num_threads(); sys.path.insert(0, str(RD / "09_three_hypotheses")); sys.path.insert(0, str(RD / "11_scorer_head_and_pullout_grades")); import s9_lib; torch.set_num_threads(_nt)   # s9_lib sets torch threads on import; put ours back
assert s9_lib.SEED == 20261008; _, va = s9_lib.seen_split(dict(clip=seen.clip_uuid.to_numpy())); HO_CLIPS = sorted(set(seen.clip_uuid.to_numpy()[va])); assert len(HO_CLIPS) == 145 and int(va.sum()) == 2366, (len(HO_CLIPS), int(va.sum()))
SLICE = {"A": "highway", "B_pass": "close-pass", "ordinary": "ordinary", "B_raw": "pull-out (ungraded)"}
ho = tr_all[tr_all.clip_uuid.isin(HO_CLIPS)].reset_index(drop=True); tr = tr_all[~tr_all.clip_uuid.isin(HO_CLIPS)].reset_index(drop=True); ho["slice"] = ho.type.map(SLICE)
assert len(ho) == 4400 and int(ho.has_label2.sum()) == 4360 and ho.clip_uuid.nunique() == 145 and not (set(tr.clip_uuid) & set(HO_CLIPS)), (len(ho), int(ho.has_label2.sum()))
C3._assert_train(set(tr_all.clip_uuid)); C3._assert_train(set(tr_all[tr_all.source == "nurec"].clip_uuid), nurec=True); assert set(tr_all.split) == {"train"}
log(f"data: S4 manifest {len(tr_all)} examples / {tr_all.clip_uuid.nunique()} clips; training {len(tr)} / {tr.clip_uuid.nunique()}; SEEN held-out {len(ho)} / {ho.clip_uuid.nunique()} ({int(ho.has_label2.sum())} graded) |", ho.slice.value_counts().to_dict())

# ---------------------------------------------------------------- model: S4 construction, pass-9 weights, scorer + encoder frozen
cfg = DriveJEPAConfig()
agent = DriveJEPAAgent(config=cfg, lr=a.lr, checkpoint_path="", vocab_path=str(C.VOCAB["ext"]), index_kind="ext", init_checkpoint=str(C.CKPT), freeze_encoder=True, scorer_impl="s3", pool_workers=0); agent.initialize()
assert agent.anchors.shape == (10240, 8, 3); assert sha256(P9) == P9_SHA, "pass_09.pt does not match the recorded sha256"
ref9 = torch.load(P9, map_location="cpu"); assert ref9["passno"] == 9; REF9 = ref9["trainable"]; del ref9
init_sd = REF9 if pathlib.Path(a.init) == P9 else torch.load(a.init, map_location="cpu")["trainable"]
missing, unexpected = agent.load_state_dict(init_sd, strict=False); assert not unexpected and all("_backbone" in k for k in missing)
m = agent._pad_model
for p_ in m.scorer.parameters(): p_.requires_grad_(False)
GEN_PREFIX = ("_pad_model.hist_encoding.", "_pad_model.init_feature.", "_pad_model._trajectory_head."); SC_PREFIX = "_pad_model.scorer."; ENC_PREFIX = "_pad_model._backbone."
assert all(k.startswith(GEN_PREFIX) or k.startswith(SC_PREFIX) or k.startswith(ENC_PREFIX) for k in agent.state_dict()), "a tensor outside generator / scorer / encoder"
assert not any(p_.requires_grad for n, p_ in m.named_parameters() if n.startswith(("scorer.", "_backbone.")))
agent.to(dev)
# [task 5] run-time corrections: the attention blocks are re-classed once; the switches decide per forward call
import batch_fix as BF, ref_range as RR
BF_FOUND = BF.install(m); RR_FOUND = RR.install(m._trajectory_head[0]); assert len(BF_FOUND["tsa"]) == len(RR_FOUND) > 0 and len(BF_FOUND["sca"]) > 0, (BF_FOUND, RR_FOUND)
X_HI = float(json.load(open(HERE / "ref_range.json"))["x_hi"]) if (a.ref_range == "wide" or a.mode == "gate_fix") else 32.0
def set_patches(fix=a.batch_fix == "on", wide=a.ref_range == "wide"): BF.set_fix(fix, fix, fix); RR.set_range(X_HI if wide else 32.0)
set_patches(); log(f"batch fix {a.batch_fix} (fix1+fix2+fix3; self-attention blocks {len(BF_FOUND['tsa'])}, cross-attention blocks {len(BF_FOUND['sca'])}) | reference range {a.ref_range}: forward -32 m .. {RR.STATE['x_hi']:.3f} m | batch_fix.py sha256 {sha256(HERE / 'train_patch/batch_fix.py')[:16]} | ref_range.py sha256 {sha256(HERE / 'train_patch/ref_range.py')[:16]}")
def set_train(): agent.train(); m.scorer.eval()            # jitter (GridMask) as S4; the frozen scorer stays in eval mode
def enc_hash():
    h = hashlib.sha256(); sd = m._backbone.state_dict()
    for k in sorted(sd): h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()
ENC0 = enc_hash()
def frozen_check():
    """scorer tensors bit-identical to the pass-9 file, encoder tensors (parameters and buffers) bit-identical to launch, generator vs pass 9"""
    sd = agent.state_dict(); sc = [k for k in sd if k.startswith(SC_PREFIX)]; gen = [k for k in sd if k.startswith(GEN_PREFIX)]
    sc_bad = [k for k in sc if not torch.equal(sd[k].cpu(), REF9[k])]; gen_changed = [k for k in gen if not torch.equal(sd[k].cpu(), REF9[k])]; eh = enc_hash()
    one = all(m._trajectory_head[i] is m._trajectory_head[0] for i in range(len(m._trajectory_head)))
    same4 = all(torch.equal(sd[k], sd[k.replace("_trajectory_head.%d." % i, "_trajectory_head.0.")]) for i in (1, 2, 3) for k in gen if k.startswith("_pad_model._trajectory_head.%d." % i))
    return dict(scorer_tensors=len(sc), scorer_changed=len(sc_bad), encoder_tensors=len(m._backbone.state_dict()), encoder_sha256_launch=ENC0, encoder_sha256_now=eh, encoder_identical=eh == ENC0,
                generator_keys=len(gen), generator_keys_changed_vs_pass9=len(gen_changed), refiner_entries_one_object=bool(one), refiner_entries_equal=bool(same4), scorer_training_flag=bool(m.scorer.training))
if cuda:
    try: _r = torch.empty(int(a.reserve_gb * 2**30), dtype=torch.uint8, device=dev); del _r; log(f"pre-reserved {torch.cuda.memory_reserved()/2**30:.1f} GiB of GPU memory (cap {a.cap_gb} GiB)")
    except torch.OutOfMemoryError: log("could not pre-reserve GPU memory (card is full right now); continuing without")
def guarded(fn, what):
    """as s4_train.guarded: an out-of-memory error while we hold < 90 % of our cap is the neighbours'; wait and retry (30 min)"""
    for k in range(31):
        try: return fn()
        except torch.OutOfMemoryError:
            ours = torch.cuda.memory_reserved() / 2**30
            if ours >= 0.9 * a.cap_gb or k == 30: raise
            log(f"  GPU full from other tenants during {what} (we hold {ours:.1f} GiB): waiting 60 s, retry {k+1}/30"); time.sleep(60)
cache = C.NAS / "04_training_set/s1/cache"
def make_ds(man):
    return CacheOnlyDataset(cache_path=str(cache), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), zero_accel=True,
                            has_label2=dict(zip(man.sample_id, man.has_label2)), token_clips=dict(zip(man.sample_id, man.clip_uuid)), slim_root=str(SLIM), split_guard=lambda u: C3._assert_train(list(u)))
def to_dev(b):
    f, t = b; f = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in f.items()}; t = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}; return f, t
def forward(ag, batch):
    f, t = to_dev(batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
        pred = ag.forward(f)
    pred = {k: (v.float() if torch.is_tensor(v) else ([x.float() for x in v] if isinstance(v, list) else v)) for k, v in pred.items()}      # loss in fp32
    return pred, t
def batches(n, bs):
    b = [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]
    if len(b) > 1 and len(b[-1]) < 4: tail = b.pop(); b[-1] = b[-1] + tail            # stages are zipped with batch rows: never fewer than 4 rows
    return b
def sampler(p):
    """S4 sampler (s4_train.py:134,163) on the reduced manifest: every example once, type A or graded twice, permuted with seed + pass"""
    TYPE = tr.type.to_numpy(); H2 = tr.has_label2.to_numpy(); twice = np.where((TYPE == "A") | H2)[0]; draws = np.concatenate([np.arange(len(tr)), twice])
    gen = torch.Generator().manual_seed(a.seed + p); return draws[torch.randperm(len(draws), generator=gen).numpy()]
def step_speeds(xy):
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2); return np.linalg.norm(xy - prev, axis=-1) / 0.5
SLICES = ["highway", "close-pass", "ordinary", "all", "highway, car >= 20 m/s"]
def slice_mask(df, sl):
    if sl == "all": return np.ones(len(df), bool)
    if sl == "highway, car >= 20 m/s": return ((df["slice"] == "highway") & (df.speed >= 20)).to_numpy()
    return (df["slice"] == sl).to_numpy()

@torch.no_grad()
def readout(tag, out_dir):
    """SEEN held-out, forward only (nothing graded), eval mode (GridMask off). Per example -> parquet; per slice -> json."""
    t0 = time.time(); ds = make_ds(ho); assert ds.tokens == list(ho.sample_id); agent.eval(); rows = []
    bl = batches(len(ds), a.bs)
    if a.readout_batching == "shuffle": o_ = np.random.default_rng(a.seed).permutation(len(ds)); bl = [o_[b].tolist() for b in bl]
    if a.readout_batching == "one": bl = [[i] for i in range(len(ds))]
    if a.readout_batching != "clip": tag = f"{tag}_batching_{a.readout_batching}"
    dl = DataLoader(ds, batch_sampler=bl, num_workers=a.workers, collate_fn=default_collate)
    for batch in dl:
        pred, t = guarded(lambda: forward(agent, batch), f"readout {tag}"); P = pred["proposals"]; tgt = t["trajectory"]
        hum = torch.linalg.norm(P - tgt[:, None], dim=-1, ord=1).mean(-1).amin(1).cpu().numpy()                      # drive_jepa_agent_s35.py:312-317 per row
        spc = agent.spacing_rows(P).cpu().numpy(); Pn = P.cpu().numpy().astype(np.float64); tn = tgt.cpu().numpy().astype(np.float64)
        d4 = np.linalg.norm(Pn[:, :, -1, :2] - tn[:, None, -1, :2], axis=-1).min(1); sp = step_speeds(Pn[..., :2]); sw = sp.max(-1) - sp.min(-1)
        for i, tok in enumerate(t["token"]): rows.append(dict(token=tok, human_term=float(hum[i]), best_dist4=float(d4[i]), spacing_sq=float(spc[i]), swing=sw[i].astype(np.float32).tolist()))
    df = pd.DataFrame(rows).set_index("token").loc[list(ho.sample_id)].reset_index(); assert list(df.token) == list(ho.sample_id); df = pd.concat([ho[["clip_uuid", "type", "slice", "speed", "has_label2"]], df], axis=1)
    df.to_parquet(out_dir / f"readout_{tag}_per_example.parquet", index=False); SW = np.stack(df.swing.to_numpy()); out = dict(tag=tag, set="SEEN held-out (145 clips of the S7 split)", graded=False, batching=a.readout_batching, batch_size=a.bs, batch_fix=a.batch_fix, ref_range=a.ref_range, ref_x_hi_m=RR.STATE["x_hi"], init=str(a.init), secs=time.time() - t0, slices={})
    for sl in SLICES:
        k = slice_mask(df, sl); d = df[k]
        out["slices"][sl] = dict(examples=int(k.sum()), clips=int(d.clip_uuid.nunique()), human_path_term=float(d.human_term.mean()), best_of_32_dist_4s_mean_m=float(d.best_dist4.mean()), best_of_32_dist_4s_median_m=float(d.best_dist4.median()),
                                 share_with_plan_within_1p5m_4s=float((d.best_dist4 <= 1.5).mean()), swing_median_all_32_plans_mps=float(np.median(SW[k])), share_plans_swing_above_5mps=float((SW[k] > 5).mean()),
                                 spacing_sq_mean_m2=float(d.spacing_sq.mean()), spacing_sq_median_m2=float(d.spacing_sq.median()))
    json.dump(out, open(out_dir / f"readout_{tag}.json", "w"), indent=1)
    log(f"readout {tag} ({out['secs']:.0f} s):\n" + pd.DataFrame(out["slices"]).T.round(3).to_string()); return out

# ================================================================ gate A
if a.mode == "gate_a":
    G = {}
    # spacing function on synthetic plans (float32, the function the loss uses)
    k = torch.arange(1, 9, dtype=torch.float32); z = torch.zeros(8)
    const_v = torch.stack([6.0 * k, z, z], -1)                                              # 12 m/s straight
    const_a = torch.stack([4.0 * k + 0.25 * k * (k + 1), z, z], -1)                         # d(k) = 4 + 0.5 k: constant acceleration
    th = 0.05 * k; arc_d = 5.0; arc = torch.stack([torch.cumsum(arc_d * torch.cos(th), 0), torch.cumsum(arc_d * torch.sin(th), 0), th], -1)   # equal chords on a curve
    jag = torch.tensor([[3.0, 0, 0], [3, 4, 0], [6, 4, 0], [6, 4, 0], [6, 9, 0], [8, 9, 0], [8, 10, 0], [8, 10, 0]])   # d = 3,4,3,0,5,2,1,0 -> s = -2,-2,8,-8,2,0 -> mean s^2 = 140/6
    f = lambda pl: float(DriveJEPAAgent.spacing_rows(pl[None, None])[0]); mix = torch.stack([const_v, jag])[None]
    G["spacing_synthetic"] = dict(constant_speed=f(const_v), constant_acceleration=f(const_a), equal_chords_on_curve=f(arc), jagged=f(jag), jagged_hand_value=140 / 6, two_plan_mean=float(DriveJEPAAgent.spacing_rows(mix)[0]), two_plan_hand_value=140 / 12)
    assert abs(G["spacing_synthetic"]["constant_speed"]) < 1e-8 and abs(G["spacing_synthetic"]["constant_acceleration"]) < 1e-8 and abs(G["spacing_synthetic"]["jagged"] - 140 / 6) < 1e-4 and abs(G["spacing_synthetic"]["two_plan_mean"] - 140 / 12) < 1e-4, G
    log("spacing synthetic:", G["spacing_synthetic"])
    # S4 path: s35 agent built by s35_common.build_agent, weights loaded as s4_train.load_weights (:75)
    a35 = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=4, score_chunk=2); sd = torch.load(P9, map_location="cpu"); mi, un = a35.load_state_dict(sd["trainable"], strict=False); assert not un and all("_backbone" in k for k in mi); a35.to(dev)
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(ho)); i16 = sorted(order[:16].tolist()); gr = np.where(ho.has_label2.to_numpy())[0]; i64 = sorted(rng.permutation(gr)[:64].tolist()); ds = make_ds(ho)
    def get(ii): return default_collate([ds[i] for i in ii])
    a35.eval(); agent.eval()
    with torch.no_grad():
        b = get(i16); p35, _ = forward(a35, b); p10, t = forward(agent, b)
    G["parity_16"] = dict(examples=16, slices=ho.slice.iloc[i16].value_counts().to_dict(), plans_max_abs_diff=float((p35["proposals"] - p10["proposals"]).abs().max()), stage_plans_max_abs_diff=max(float((x - y).abs().max()) for x, y in zip(p35["proposal_list"], p10["proposal_list"])),
                          scorer_logits_max_abs_diff=float((p35["pred_logit"] - p10["pred_logit"]).abs().max()), scorer_score_max_abs_diff=float((p35["pdm_score"] - p10["pdm_score"]).abs().max()), same_pick=bool((p35["pdm_score"].argmax(1) == p10["pdm_score"].argmax(1)).all()))
    log("parity 16:", G["parity_16"])
    # catalogue indices: direct read vs what compute_score (the S4 path, grading as a side effect) returns, 64 graded examples
    eq = 0; sizes = []
    for c in range(0, 64, 32):
        with torch.no_grad(): b = get(i64[c:c + 32]); p35, t = forward(a35, b)
        s4_idx = a35.compute_score(t, p35["proposals"].detach(), test=False)[-1]; mine = agent.read_scores_index(t); rows_ = torch.nonzero(t["has_label2"].bool().cpu()).flatten().tolist(); assert len(rows_) == 32
        for r, x in zip(rows_, s4_idx): eq += int(np.array_equal(x, mine[r]) and x.dtype == mine[r].dtype); sizes.append(int(x.size))
        if c == 0:   # loss parity on this batch: S4's trajectory_loss_anchors vs generator_loss_s10 at weight 0, same numpy seed
            np.random.seed(7); l35 = a35.trajectory_loss_anchors(p35["proposal_list"], t["trajectory"], cfg, s4_idx); np.random.seed(7); l10, parts = agent.generator_loss_s10(p35["proposal_list"], t["trajectory"], cfg, mine, 0.0)
            G["loss_parity_32"] = dict(s4_trajectory_loss=fv(l35[0]), s10_loss_at_weight_0=fv(l10), abs_diff=abs(fv(l35[0]) - fv(l10)), s4_min_loss=fv(l35[1]), s10_min_loss_last=fv(parts["min_loss_last"]), human_last=fv(parts["human_last"]), catalogue_last=fv(parts["catalogue_last"]),
                                       rows_sum_check=abs(fv(parts["human_rows"].mean() + parts["cat_rows"].sum() / 32) - fv(parts["min_loss_last"])))
    G["catalogue_index_64"] = dict(examples=64, equal=eq, index_sizes_min_median_max=[int(np.min(sizes)), float(np.median(sizes)), int(np.max(sizes))], empty=int(np.sum(np.array(sizes) == 0)))
    log("catalogue index 64:", G["catalogue_index_64"]); log("loss parity:", G["loss_parity_32"])
    G["generator_parameters_without_grad_as_built"] = {n: p_.numel() for n, p_ in m.named_parameters() if n.startswith(("hist_encoding.", "init_feature.", "_trajectory_head.")) and not p_.requires_grad}
    G["pass_09_sha256"] = P9_SHA; G["frozen_check_at_load"] = frozen_check(); G["optimizer_tensors"] = sum(len(g["params"]) for g in agent.get_generator_optimizer(a.lr).param_groups); G["optimizer_elements"] = sum(p_.numel() for g in agent.get_generator_optimizer(a.lr).param_groups for p_ in g["params"])
    ok = max(G["parity_16"]["plans_max_abs_diff"], G["parity_16"]["scorer_logits_max_abs_diff"]) <= 1e-5 and eq == 64 and G["loss_parity_32"]["abs_diff"] <= 1e-5; G["pass"] = bool(ok)
    json.dump(G, open(HERE / "gate_a.json", "w"), indent=1); log("GATE A", "PASS" if ok else "FAIL"); a35.persistent_pool.shutdown(); sys.exit(0 if ok else 2)

# ================================================================ gate_fix (task 5 Ga): batch 32 shuffled against batch 1 with the corrections on, GPU, float32 and autocast
if a.mode == "gate_fix":
    if a.tf32 == "off": torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    agent.eval(); ds = make_ds(ho); hw = np.where(((ho.slice == "highway") & (ho.speed >= 20)).to_numpy())[0]; od = np.where((ho.slice == "ordinary").to_numpy())[0]; assert len(hw) == 575 and len(od) == 2806
    rows = np.concatenate([hw[:32], hw[288:320], od[:32], od[1400:1432]]); grp = np.array(["highway >= 20 m/s"] * 64 + ["ordinary"] * 64); assert (ho.speed.to_numpy()[rows[:64]] >= 20).all()   # the 128 rows of task 3 Q1
    rng = np.random.default_rng(a.seed); rest = np.setdiff1d(np.arange(len(ho)), rows); order = rng.permutation(128); filler = rng.permutation(rest)[:16 * 24].reshape(16, 24); plan = []
    for b in range(16):
        ids = np.concatenate([rows[order[b * 8:(b + 1) * 8]], filler[b]]); pos = rng.permutation(32); plan.append((ids[pos], [(int(np.where(pos == k)[0][0]), int(order[b * 8 + k])) for k in range(8)]))   # (batch ids, [(position in batch, index into rows)])
    need = sorted(set(rows.tolist()) | set(filler.flatten().tolist())); t0 = time.time(); IT = {i: ds[i] for i in need}; log(f"gate_fix: {len(need)} held-out examples loaded in {time.time()-t0:.0f} s")
    def fwd(ids, amp):
        f, t = to_dev(default_collate([IT[int(i)] for i in ids]))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and cuda): pred = agent.forward(f)
        return torch.stack([x.float() for x in pred["proposal_list"]], 1).cpu(), pred["pred_logit"].float().cpu(), pred["pdm_score"].float().cpu()      # (B, 4 stages, 32 plans, 8, 3), logits, (B, 32) scores
    def b1(fix, wide, amp):
        set_patches(fix, wide); o = [fwd([i], amp) for i in rows]; return tuple(torch.cat([x[j] for x in o]) for j in range(3))
    def b32(fix, wide, amp):
        set_patches(fix, wide); P = [None] * 128; L = [None] * 128; S_ = [None] * 128
        for ids, where in plan:
            p_, l_, s_ = fwd(ids, amp)
            for pos, k in where: P[k] = p_[pos]; L[k] = l_[pos]; S_[k] = s_[pos]
        return torch.stack(P), torch.stack(L), torch.stack(S_)
    def cmp(x, y):
        dxy = (x[0][..., :2] - y[0][..., :2]).abs().amax(dim=(1, 2, 3, 4)).numpy(); dh = (x[0][..., 2] - y[0][..., 2]).abs().amax(dim=(1, 2, 3)).numpy(); dl = (x[1] - y[1]).abs().flatten(1).amax(1).numpy(); pk = (x[2].argmax(1) != y[2].argmax(1)).numpy(); st = (x[0][..., :2] - y[0][..., :2]).abs().amax(dim=(2, 3, 4)).numpy(); o = {}
        for g in ("highway >= 20 m/s", "ordinary", "all"):
            k = np.ones(128, bool) if g == "all" else grp == g; o[g] = dict(rows=int(k.sum()), plan_xy_max_abs_diff_m=float(dxy[k].max()), plan_xy_median_over_rows_m=float(np.median(dxy[k])), heading_max_abs_diff_rad=float(dh[k].max()), logit_max_abs_diff=float(dl[k].max()), picks_changed=int(pk[k].sum()), rows_above_2cm=int((dxy[k] > 0.02).sum()), rows_above_1mm=int((dxy[k] > 0.001).sum()), plan_xy_p90_over_rows_m=float(np.percentile(dxy[k], 90)), plan_xy_max_by_stage_m=[float(v) for v in st[k].max(0)])
        return o
    G = dict(what="pass 9, eval mode, GPU; 128 SEEN held-out rows (64 highway with car >= 20 m/s, 64 ordinary; the rows of task 3 Q1); batch 32 = 16 batches, each 8 of the rows + 24 other held-out rows, positions shuffled; differences over 4 stages x 32 plans x 8 points",
             device=str(dev), torch=torch.__version__, cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_benchmark=bool(torch.backends.cudnn.benchmark), ref_x_hi_m=X_HI, batch_fix_sha256=sha256(HERE / "train_patch/batch_fix.py"), ref_range_sha256=sha256(HERE / "train_patch/ref_range.py"), pass_09_sha256=P9_SHA); R = {}
    for wide in (False, True):
        for amp in (False, True):
            R[(wide, amp, "b1_on")] = b1(True, wide, amp); R[(wide, amp, "b1_off")] = b1(False, wide, amp); R[(wide, amp, "b32_on")] = b32(True, wide, amp)
        R[(wide, False, "b32_off")] = b32(False, wide, False)
    for wide in (False, True):
        g = {}
        for amp in (False, True):
            pr = "autocast_16bit" if amp else "float32"
            g[f"{pr}: batch 32 shuffled, fix on, against batch 1, fix on"] = cmp(R[(wide, amp, "b32_on")], R[(wide, amp, "b1_on")]); g[f"{pr}: batch 1, fix on against fix off"] = cmp(R[(wide, amp, "b1_on")], R[(wide, amp, "b1_off")])
        g["float32: batch 32 shuffled, fix OFF, against batch 1 (the uncorrected dependence, for scale)"] = cmp(R[(wide, False, "b32_off")], R[(wide, False, "b1_off")])
        g["batch 1, fix on: autocast against float32"] = cmp(R[(wide, True, "b1_on")], R[(wide, False, "b1_on")]); g["batch 32 shuffled under autocast against batch 1 in float32, fix on"] = cmp(R[(wide, True, "b32_on")], R[(wide, False, "b1_on")])
        G["range_wide" if wide else "range_std"] = g
    G["range wide against std, batch 1, float32, fix on"] = cmp(R[(True, False, "b1_on")], R[(False, False, "b1_on")]); G["ref_range_hook_calls"] = RR.STATE["calls"]
    f32 = max(G[r]["float32: batch 32 shuffled, fix on, against batch 1, fix on"]["all"]["plan_xy_max_abs_diff_m"] for r in ("range_std", "range_wide")); onoff = max(G[r][f"{pr}: batch 1, fix on against fix off"]["all"][q] for r in ("range_std", "range_wide") for pr in ("float32", "autocast_16bit") for q in ("plan_xy_max_abs_diff_m", "heading_max_abs_diff_rad", "logit_max_abs_diff"))
    G["float32_batch32_vs_batch1_max_m"] = f32; G["fix_on_vs_off_batch1_max"] = onoff; G["pass"] = bool(f32 < 0.02 and onoff == 0.0); G["frozen_check"] = frozen_check()
    json.dump(G, open(HERE / ("gate_fix.json" if a.tf32 == "default" else "gate_fix_tf32off.json"), "w"), indent=1); log(json.dumps(G, indent=1)); log("GATE Ga", "PASS" if G["pass"] else "FAIL"); sys.exit(0 if G["pass"] else 2)

# ================================================================ diag: does a row's menu depend on the rest of its batch / on precision? (eval mode, pass 9)
if a.mode == "diag":
    agent.eval(); dsh = make_ds(ho); dst = make_ds(tr); hw = ho[(ho.slice == "highway") & (ho.speed >= 20)]; rng = np.random.default_rng(a.seed)
    ii = sorted(np.concatenate([g.index.to_numpy()[:4] for _, g in [x for x in hw.groupby("clip_uuid") if len(x[1]) >= 4][:8]]).tolist()); assert len(ii) == 32; items = [dsh[i] for i in ii]; perm1 = sampler(1); others = [dst[int(j)] for j in perm1[:31]]
    S6 = {}
    import glob
    for f_ in sorted(glob.glob(str(C.NAS / "08_scorer_diagnosis/dump/seen/shard_*.npz"))):
        z = np.load(f_); tk = set(ho.sample_id.iloc[ii]) & set(z["token"].tolist())
        if tk: S6.update({t_: z["menu_9"][list(z["token"]).index(t_)] for t_ in tk})
    def run(its, amp=True):
        global cuda
        b = default_collate(its); keep = cuda; cuda = cuda and amp
        with torch.no_grad(): pred, _ = forward(agent, b)
        cuda = keep; return pred["proposals"].cpu()
    sq = lambda P: float(DriveJEPAAgent.spacing_rows(P).mean()); mx = lambda x, y: float((x - y).abs().max()); md = lambda x, y: float((x - y).abs().amax(dim=(1, 2, 3)).median())
    A = run(items); A32 = run(items, amp=False); rev = run(items[::-1]).flip(0); one = torch.cat([run([it]) for it in items]); one32 = torch.cat([run([it], amp=False) for it in items[:8]])
    mixed = torch.cat([run([it] + others)[:1] for it in items[:8]]); mixed_last = torch.cat([run(others + [it])[-1:] for it in items[:8]]); toks = list(ho.sample_id.iloc[ii]); have = [k for k, t_ in enumerate(toks) if t_ in S6]; s6 = torch.from_numpy(np.stack([S6[toks[k]] for k in have])).float() if have else None
    D = dict(examples=32, clips=8, what="pass 9, eval mode; A = the 32 rows as one batch in clip order (the readout's way), 16-bit",
             spacing_sq_mean=dict(A_clip_order_batch=sq(A), A_float32=sq(A32), A_reversed_order=sq(rev), each_row_alone_batch1=sq(one), first8_in_A=sq(A[:8]), first8_alone=sq(one[:8]), first8_alone_float32=sq(one32), first8_as_row0_with_31_training_rows=sq(mixed), first8_as_last_row_after_31_training_rows=sq(mixed_last)),
             plans_abs_diff_max_and_median_per_row=dict(A_vs_float32=[mx(A, A32), md(A, A32)], A_vs_reversed_order=[mx(A, rev), md(A, rev)], A_vs_alone=[mx(A, one), md(A, one)], alone_vs_alone_float32=[mx(one[:8], one32), md(one[:8], one32)],
                                                        alone_vs_row0_of_mixed_batch=[mx(one[:8], mixed), md(one[:8], mixed)], alone_vs_last_row_of_mixed_batch=[mx(one[:8], mixed_last), md(one[:8], mixed_last)]))
    if have: D["vs_s6_seen_dump_menu_9"] = dict(rows=len(have), spacing_sq_mean_s6=sq(s6), alone_vs_s6=[mx(one[have], s6), md(one[have], s6)], A_vs_s6=[mx(A[have], s6), md(A[have], s6)])
    json.dump(D, open(HERE / "diag_batch_dependence.json", "w"), indent=1); log(json.dumps(D, indent=1)); sys.exit(0)

# ================================================================ diag2: which module first gives row 0 a different output when other rows share its batch? (eval, float32)
if a.mode == "diag2":
    agent.eval(); dsh = make_ds(ho); dst = make_ds(tr); hw = ho[(ho.slice == "highway") & (ho.speed >= 20)]; perm1 = sampler(1); items = [dsh[int(hw.index[0])]] + [dst[int(j)] for j in perm1[:7]]; REC = {}; cuda_keep = cuda; cuda = False
    def first_tensor(o):
        if torch.is_tensor(o): return o
        if isinstance(o, (list, tuple)):
            for x in o:
                r_ = first_tensor(x)
                if r_ is not None: return r_
        return None
    def mk(name, store):
        def h(mod, inp, out):
            o = first_tensor(out); i_ = first_tensor(inp)
            store.setdefault(name, []).append((None if i_ is None else i_.detach().float().cpu(), None if o is None else o.detach().float().cpu()))
        return h
    def run(its):
        store = {}; hs = [mod.register_forward_hook(mk(n, store)) for n, mod in m.named_modules() if n and not any(True for _ in mod.children())]
        with torch.no_grad(): pred, _ = forward(agent, default_collate(its))
        for h in hs: h.remove()
        return store, pred["proposal_list"]
    def row0(x, y):
        """x from the batch of 8, y from the batch of 1 -> x's row 0 in y's shape, or None when the shapes do not differ in exactly one dim (8 vs 1)"""
        if x is None or y is None or x.dim() != y.dim(): return None
        dd = [k for k in range(x.dim()) if x.shape[k] != y.shape[k]]
        if len(dd) != 1 or y.shape[dd[0]] * 8 != x.shape[dd[0]]: return None
        n1 = y.shape[dd[0]]; return x.narrow(dd[0], 0, n1), dd[0]
    SB, PB = run(items); SA, PA = run(items[:1]); out = []; order = list(SA.keys())
    for n in order:
        for c, ((ia, oa), (ib, ob)) in enumerate(zip(SA[n], SB.get(n, []))):
            ri = row0(ib, ia); ro = row0(ob, oa)
            out.append(dict(module=n, call=c, type=type(dict(m.named_modules())[n]).__name__, in_shape_batch8=None if ib is None else list(ib.shape), out_shape_batch8=None if ob is None else list(ob.shape), batch_dim_out=None if ro is None else ro[1],
                            in_diff=None if ri is None else float((ri[0] - ia).abs().max()), out_diff=None if ro is None else float((ro[0] - oa).abs().max()), out_scale=None if oa is None else float(oa.abs().max())))
    df = pd.DataFrame(out); df.to_csv(HERE / "diag2_module_trace.csv", index=False); pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_rows", 500)
    bad = df[(df.out_diff > 1e-3)]; firsts = bad[(bad.in_diff.isna()) | (bad.in_diff <= 1e-3)]
    log("stage plans, row 0, batch of 8 vs alone, max abs diff per stage:", [float((x[:1].cpu() - y.cpu()).abs().max()) for x, y in zip(PB, PA)])
    log("modules compared:", len(df), "| with row-0 output differing > 1e-3:", len(bad), "| of those with matching (or uncomparable) input:", len(firsts)); log("first 12 differing calls in definition order:\n" + bad.head(12).to_string()); log("differing output with matching input:\n" + firsts.head(20).to_string()); sys.exit(0)

# ================================================================ pass-9 zero reference
if a.mode == "readout":
    tag = "pass_09" if pathlib.Path(a.init) == P9 else pathlib.Path(a.init).stem; readout(tag, NASOUT); log("frozen check:", frozen_check()); sys.exit(0)

NORM_FILE = "spacing_norm_run.json" if a.batch_fix == "on" else "spacing_norm.json"   # [task 5] the weight is measured with the corrections the run uses
# ================================================================ probe: sizes of the loss parts at pass 9 -> spacing_norm.json
if a.mode == "probe":
    perm = sampler(1); bl = batches(len(perm), a.bs)[:a.probe_batches]; ds = make_ds(tr); dl = DataLoader(Subset(ds, perm.tolist()), batch_sampler=bl, num_workers=a.workers, collate_fn=default_collate)
    agent.eval(); np.random.seed(1234); R = []; E = []; TY = dict(zip(tr.sample_id, tr.type)); SP = dict(zip(tr.sample_id, tr.speed))
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            pred, t = guarded(lambda: forward(agent, batch), "probe"); _, pr = agent.generator_loss_s10(pred["proposal_list"], t["trajectory"], cfg, agent.read_scores_index(t), 0.0); B = len(t["token"])
            R.append({k: fv(pr[k]) for k in ("human", "catalogue", "spacing", "human_last", "catalogue_last", "spacing_last")})
            for i, tok in enumerate(t["token"]): E.append(dict(batch=bi, token=tok, type=TY[tok], speed=SP[tok], graded=bool(t["has_label2"][i]), human=float(pr["human_rows"][i]), catalogue=float(pr["cat_rows"][i]), spacing=float(pr["spacing_rows"][i])))
    R = pd.DataFrame(R); E = pd.DataFrame(E); E["slice"] = E.type.map(SLICE); E.to_parquet(NASOUT / "probe_per_example.parquet", index=False); R.to_csv(NASOUT / "probe_per_batch.csv", index=False)
    C0 = float(R.catalogue_last.mean()); H0 = float(R.human_last.mean()); S0 = float(R.spacing_last.mean()); by = {}
    for sl in ["highway", "close-pass", "ordinary", "pull-out (ungraded)", "all", "highway, car >= 20 m/s"]:
        k = slice_mask(E, sl); d = E[k]
        if len(d): by[sl] = dict(draws=int(len(d)), share_of_draws=float(len(d) / len(E)), graded_share=float(d.graded.mean()), human_part=float(d.human.mean()), catalogue_part=float(d.catalogue.mean()), spacing_sq_mean_m2=float(d.spacing.mean()), spacing_sq_median_m2=float(d.spacing.median()), share_of_total_spacing=float(d.spacing.sum() / E.spacing.sum()))
    out = dict(definition="last refinement stage (the menu), per batch, mean over the probe batches; w(lambda) = lambda x C0 / S0, constant for the run", probe=f"first {len(R)} batches of the pass-1 sampler (seed {a.seed}+1) on the reduced training manifest, batch {a.bs}, eval mode (jitter off), no grad, pass 9; numpy seed 1234 for the draw of 4 catalogue curves",
               batches=int(len(R)), draws=int(len(E)), C0=C0, H0=H0, S0=S0, S0_median_per_batch=float(R.spacing_last.median()), S0_p99_per_batch=float(R.spacing_last.quantile(0.99)), S0_max_per_batch=float(R.spacing_last.max()), S0_min_per_batch=float(R.spacing_last.min()),
               stage_combined=dict(C=float(R.catalogue.mean()), H=float(R.human.mean()), S=float(R.spacing.mean()), C_over_S=float(R.catalogue.mean() / R.spacing.mean())), C0_over_S0=C0 / S0,
               w={str(l): l * C0 / S0 for l in (0.1, 1.0, 10.0)}, w_times_S0={str(l): l * C0 for l in (0.1, 1.0, 10.0)}, by_slice=by, pass_09_sha256=P9_SHA, batch_fix=a.batch_fix, ref_range=a.ref_range, precision="16-bit autocast forward, float32 loss" if cuda else "float32", device=str(dev))
    json.dump(out, open(HERE / NORM_FILE, "w"), indent=1); log("spacing_norm:", json.dumps({k: v for k, v in out.items() if k != "by_slice"})); log("\n" + pd.DataFrame(by).T.round(4).to_string()); sys.exit(0)

# ================================================================ train
assert a.lam is not None; NORM = json.load(open(HERE / NORM_FILE)); assert NORM["pass_09_sha256"] == P9_SHA and NORM.get("batch_fix", "off") == a.batch_fix and NORM.get("ref_range", "std") == "std" and (a.ref_range == "std" or a.lam == 0), "the spacing weight was not measured under these settings"; W = a.lam * NORM["C0"] / NORM["S0"]
opt = agent.get_generator_optimizer(a.lr); scaler = torch.amp.GradScaler("cuda", enabled=cuda); opt_ids = {id(p_) for g in opt.param_groups for p_ in g["params"]}
assert not (opt_ids & {id(p_) for p_ in m.scorer.parameters()}) and not (opt_ids & {id(p_) for p_ in m._backbone.parameters()}); assert sum(p_.requires_grad for p_ in agent.parameters()) == len(opt_ids), "a trainable tensor outside the optimiser"
log(f"lambda {a.lam} -> spacing weight w = {W:.6g} (C0 {NORM['C0']:.5f}, S0 {NORM['S0']:.4f}) | lr {a.lr} | optimiser: fresh Adam, {len(opt_ids)} tensors, {sum(p_.numel() for g in opt.param_groups for p_ in g['params'])} elements | init {a.init}")
log("frozen check at launch:", frozen_check()); ds_tr = make_ds(tr); assert ds_tr.tokens == list(tr.sample_id); STATE = NASOUT / "state.json"; state = dict(run=a.run, lam=a.lam, w=W, lr=a.lr, pass_done=0, stopped=None, pid=os.getpid(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), batch_fix=a.batch_fix, ref_range=a.ref_range, ref_x_hi_m=RR.STATE["x_hi"], norm_file=NORM_FILE, C0=NORM["C0"], S0=NORM["S0"])
def save_state(): json.dump(state, open(STATE, "w"), indent=1)
save_state(); GEN_NAMED = [(n_, p_) for n_, p_ in m.named_parameters() if id(p_) in opt_ids]; assert len(GEN_NAMED) == len(opt_ids) and sum(".attentions.0." in n_ for n_, _ in GEN_NAMED) > 0; PK = ("total", "human", "catalogue", "spacing", "human_last", "catalogue_last", "spacing_last")
for p in range(1, a.passes + 1):
    perm = sampler(p); bl = batches(len(perm), a.bs); np.random.seed(a.seed + p); torch.manual_seed(a.seed + p); set_train()
    if a.max_steps: bl = bl[:a.max_steps]
    dl = DataLoader(Subset(ds_tr, perm.tolist()), batch_sampler=bl, num_workers=a.workers, collate_fn=default_collate, prefetch_factor=4, persistent_workers=False)
    log(f"pass {p}: {len(bl)} steps of batch {a.bs} ({len(perm)} draws per full pass)"); CSV = open(NASOUT / f"steps_pass_{p:02d}.csv", "w"); CSV.write("step,total,human,catalogue,spacing_raw,w_spacing,share_w_spacing,human_last,catalogue_last,spacing_raw_last,sec_per_step,gpu_reserved_gib,grad_norm_generator,grad_norm_self_attention,grad_norm_self_attention_offsets_weights,updates_skipped_so_far,amp_scale\n")
    acc = {k: 0.0 for k in PK}; run = {k: 0.0 for k in PK}; nacc = steps = skipped = 0; t_ep = t_win = time.time(); bad = None
    for batch in dl:
        def one():
            opt.zero_grad(set_to_none=True); pred, t = forward(agent, batch); loss, parts = agent.generator_loss_s10(pred["proposal_list"], t["trajectory"], cfg, agent.read_scores_index(t), W)
            if torch.isfinite(loss): scaler.scale(loss).backward()
            return loss, parts
        def gnorm(keys):   # [task 5] gradient norm in loss units (the scaler's factor divided out), over the named generator tensors
            g = [p_.grad.float() for n_, p_ in GEN_NAMED if keys(n_) and p_.grad is not None]; return float(torch.sqrt(sum((x * x).sum() for x in g))) / scaler.get_scale() if g else float("nan")
        loss, parts = guarded(one, f"pass {p} step {steps}"); lv = fv(loss)
        if not np.isfinite(lv): bad = f"non-finite loss at pass {p} step {steps + 1}"; break
        GN = [gnorm(lambda n_: True), gnorm(lambda n_: ".attentions.0." in n_), gnorm(lambda n_: ".attentions.0.sampling_offsets." in n_ or ".attentions.0.attention_weights." in n_)] if (steps == 0 or (steps + 1) % a.csv_every == 0 or steps + 1 == len(bl)) else None
        sc0 = scaler.get_scale(); scaler.step(opt); scaler.update(); skipped += int(scaler.get_scale() < sc0); steps += 1;   # [task 5] a non-finite 16-bit gradient makes GradScaler skip the update and lower its factor: count those steps
        nacc += 1; cur = dict(total=lv, **{k: fv(parts[k]) for k in PK[1:]})
        for k in PK: acc[k] += cur[k]; run[k] += cur[k]
        if steps == 1 or steps % a.csv_every == 0 or steps == len(bl):
            sps = (time.time() - t_win) / nacc; g = torch.cuda.max_memory_reserved() / 2**30 if cuda else 0; ws = W * cur["spacing"]
            CSV.write(f"{steps},{cur['total']:.6g},{cur['human']:.6g},{cur['catalogue']:.6g},{cur['spacing']:.6g},{ws:.6g},{ws/cur['total']:.6g},{cur['human_last']:.6g},{cur['catalogue_last']:.6g},{cur['spacing_last']:.6g},{sps:.4f},{g:.2f},{GN[0]:.6g},{GN[1]:.6g},{GN[2]:.6g},{skipped},{scaler.get_scale():.6g}\n"); CSV.flush()
        if steps == 1 or steps % 50 == 0 or steps == len(bl):
            sps = (time.time() - t_win) / nacc; mw = {k: acc[k] / nacc for k in PK}
            log(f"  pass {p} step {steps}/{len(bl)}: total {mw['total']:.4f} | human {mw['human']:.4f} | catalogue {mw['catalogue']:.4f} | raw spacing {mw['spacing']:.4f} | w x spacing {W*mw['spacing']:.4f} ({100*W*mw['spacing']/mw['total']:.1f} % of total) | {sps:.2f} s/step | ETA {(len(bl)-steps)*sps/60:.0f} min | GPU {torch.cuda.max_memory_reserved()/2**30 if cuda else 0:.1f} GiB | grad norm generator {GN[0]:.4g}, self-attention {GN[1]:.4g}, its offset+weight layers {GN[2]:.4g} | updates skipped so far (non-finite 16-bit gradient) {skipped}, scaler factor {scaler.get_scale():.4g}   [means over the last {nacc} steps, stages combined]")
            acc = {k: 0.0 for k in PK}; nacc = 0; t_win = time.time()
    CSV.close()
    if bad: state["stopped"] = bad; save_state(); log("ABORT", bad); sys.exit(3)
    tr_secs = time.time() - t_ep; fc = frozen_check(); log(f"pass {p} trained: {steps} steps, {tr_secs/60:.1f} min, {tr_secs/steps:.3f} s/step | frozen check:", fc)
    if fc["scorer_changed"] or not fc["encoder_identical"] or not fc["refiner_entries_one_object"] or not fc["refiner_entries_equal"]: state["stopped"] = "frozen tensors changed"; state["frozen_check"] = fc; save_state(); log("ABORT frozen tensors changed"); sys.exit(4)
    ck = NASOUT / f"pass_{p:02d}.pt"; torch.save(dict(trainable={k: v.cpu() for k, v in agent.state_dict().items() if "_backbone" not in k}, opt=opt.state_dict(), scaler=scaler.state_dict(), passno=p, lam=a.lam, w=W, lr=a.lr, init=a.init, steps=steps), ck)
    ro = readout(f"pass_{p:02d}", NASOUT); set_train()
    state.update(pass_done=p, frozen_check=fc, updates_skipped=skipped, train_secs=tr_secs, sec_per_step=tr_secs / steps, steps=steps, gpu_max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0, gpu_max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0,
                 train_means={k: run[k] / steps for k in PK}, checkpoint=str(ck), readout=str(NASOUT / f"readout_pass_{p:02d}.json")); save_state(); log("saved", ck)
log("DONE", json.dumps({k: state[k] for k in ("run", "lam", "w", "pass_done", "stopped")}))
