#!/usr/bin/env python
"""S3.5 STEPS 3-4: train on the 1 % slice, evaluate every epoch on the slice and on the held-out val set, write the readouts.
Upstream recipe kept (scripts/training/train_drive_jepa_perception_based.sh: batch 32; agent yaml lr 1e-4; Adam via
agent.get_optimizers; default_training.yaml: 16-mixed precision, no gradient clipping; GridMask augmentation in train mode).
Changes per brief: 30 epochs, frozen encoder, extended vocabulary, acceleration zeroed, type-A samples counted twice per epoch.
Caps: 24 GB GPU memory (hard per-process limit), 2 h wall (checked before every epoch; stops at the cap)."""
import os, sys, json, time, hashlib, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "train_patch"))
import s35_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--cap-gb", type=float, default=24.0); ap.add_argument("--cap-h", type=float, default=2.0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="run"); ap.add_argument("--seed", type=int, default=20261007)
a = ap.parse_args(); T0 = time.time(); OUT = HERE / a.out; OUT.mkdir(exist_ok=True)
C.setup_env()
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import s3_common as C3
log_f = open(OUT / "train.log", "a")
def log(*x):
    s = f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:6.1f}m] " + " ".join(str(v) for v in x); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device(a.device); cuda = dev.type == "cuda"
if cuda:
    tot = torch.cuda.get_device_properties(0).total_memory; torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 2**30 / tot))
    log("GPU", torch.cuda.get_device_name(0), f"total {tot/2**30:.1f} GiB, process cap {a.cap_gb} GiB; torch {torch.__version__}")
torch.manual_seed(a.seed); np.random.seed(a.seed)
sl = pd.read_parquet(HERE / "slice_train.parquet"); va = pd.read_parquet(HERE / "heldout_val.parquet")
if a.limit: sl = sl.groupby("type", group_keys=False).head(a.limit); va = va.groupby("type", group_keys=False).head(a.limit)
C3._assert_train(set(sl.clip_uuid)); C3.assert_val(set(va.clip_uuid)); assert not (set(sl.group) & set(va.group))                       # S6
vocab = np.load(C.VOCAB["ext"]); log("vocabulary via the Step-1b path argument:", C.VOCAB["ext"], "shape", vocab.shape, vocab.dtype); assert vocab.shape == (10240, 40, 3)
agent = C.build_agent(vocab="ext", freeze_encoder=True, init=True, scorer_impl="s3", pool_workers=12); log("agent.anchors", agent.anchors.shape, "| checkpoint", C.CKPT.name, C.CKPT_SHA256[:12])
assert agent.anchors.shape == (10240, 8, 3)
cache = C.NAS / "04_training_set/s1/cache"
ds_tr = C.build_dataset(cache, agent, tokens=list(sl.sample_id), zero_accel=True, has_label2=dict(zip(sl.sample_id, sl.has_label2)), log_names=sorted(sl.clip_uuid.unique()))
from dataset_s35 import CacheOnlyDataset
ds_va = CacheOnlyDataset(cache_path=str(cache), feature_builders=agent.get_feature_builders(), target_builders=agent.get_target_builders(), log_names=sorted(va.clip_uuid.unique()), zero_accel=True,
                         has_label2=dict(zip(va.sample_id, va.has_label2)), tokens=list(va.sample_id), split_guard=lambda u: C3.assert_val(list(u)))
TYPE = {**dict(zip(sl.sample_id, sl.type)), **dict(zip(va.sample_id, va.type))}; SPEED = {**dict(zip(sl.sample_id, sl.speed)), **dict(zip(va.sample_id, va.speed))}
agent.to(dev); m = agent._pad_model
G = dict(scorer=list(m.scorer.parameters()), generator=list(m._trajectory_head.parameters()) + list(m.hist_encoding.parameters()) + list(m.init_feature.parameters()), encoder=list(m._backbone.parameters()))
def digest(ps):
    h = hashlib.sha256()
    for p in ps: h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
opt = agent.get_optimizers(); scaler = torch.cuda.amp.GradScaler(enabled=cuda)
log("trainable tensors", sum(len(g["params"]) for g in opt.param_groups), "| params: encoder", sum(p.numel() for p in G["encoder"]), "(frozen) generator", sum(p.numel() for p in G["generator"]), "scorer", sum(p.numel() for p in G["scorer"]))
def to_dev(b):
    f, t = b; f = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in f.items()}; t = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}; return f, t
LK = ["loss", "trajectory_loss", "final_score_loss", "sub_score_loss", "pred_ce_loss", "pred_l1_loss", "pred_area_loss", "min_loss", "inter_loss"]
def fv(x): return float(x.detach().float().cpu()) if torch.is_tensor(x) else float(x)
def step_loss(batch):
    f, t = to_dev(batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
        pred = agent.forward(f)
    pred = {k: (v.float() if torch.is_tensor(v) else ([x.float() for x in v] if isinstance(v, list) else v)) for k, v in pred.items()}      # loss in fp32
    return pred, agent.compute_loss(f, t, pred), t
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else np.nan
@torch.no_grad()
def evaluate(ds, name, epoch):
    """Eval mode (GridMask off). Per sample: loss terms are batch means, so loss is logged per batch weighted by size; readouts per sample."""
    agent.eval(); dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=default_collate); rows = []; L = {k: 0.0 for k in LK}; n = 0; nl = 0; Ls = {k: 0.0 for k in LK}
    for batch in dl:
        if len(batch[1]["token"]) < 4: continue
        pred, ld, t = step_loss(batch); B = len(t["token"]); n += B
        for k in LK: L[k] += fv(ld[k]) * B
        props = pred["proposals"].cpu().numpy(); pdm = pred["pdm_score"].cpu().numpy(); tgt = t["trajectory"].cpu().numpy(); h2 = t["has_label2"].cpu().numpy().astype(bool)
        ref = None
        if h2.any():
            fs, _, _, _, _, _, _ = agent.compute_score(t, pred["proposals"], test=False); ref = fs.cpu().numpy(); nl += int(h2.sum())
            for k in ("final_score_loss", "pred_ce_loss", "pred_l1_loss", "pred_area_loss", "sub_score_loss"): Ls[k] += fv(ld[k]) * int(h2.sum())
        j = 0
        for i, tok in enumerate(t["token"]):
            ch = int(pdm[i].argmax()); st = np.diff(np.concatenate([np.zeros((32, 1, 2)), props[i][:, :, :2]], 1), axis=1); ends = np.linalg.norm(st[:, -1], axis=1) / 0.5
            r = dict(epoch=epoch, set=name, token=tok, type=TYPE[tok], speed=SPEED[tok], has_label2=bool(h2[i]), dist4=float(np.linalg.norm(props[i, ch, -1, :2] - tgt[i, -1, :2])), ade=float(np.linalg.norm(props[i, ch, :, :2] - tgt[i, :, :2], axis=1).mean()),
                     best_dist4=float(np.linalg.norm(props[i, :, -1, :2] - tgt[i, -1, :2], axis=1).min()), fastest_end=float(ends.max()), chosen_end=float(ends[ch]), human_end=float(np.linalg.norm(tgt[i, -1, :2] - tgt[i, -2, :2]) / 0.5))
            if h2[i]:
                g = ref[j]; j += 1; r.update(top1=bool(g[ch] >= g.max() - 1e-9), top1_strict=bool(ch == int(g.argmax())), rank_corr=spearman(pdm[i], g), ref_chosen=float(g[ch]), ref_best=float(g.max()), ref_std=float(g.std()))
            rows.append(r)
    agent.train(); out = {k: L[k] / max(n, 1) for k in LK}; out.update({f"{k}_labelled": Ls[k] / max(nl, 1) for k in Ls}); out.update(n=n, n_label2=nl)
    return out, pd.DataFrame(rows)
A_idx = [i for i, t in enumerate(sl.sample_id) if TYPE[t] == "A"]; base_idx = list(range(len(ds_tr))); epoch_idx = base_idx + A_idx                      # Sid's Z5 remedy: every type-A sample twice per epoch
share = pd.Series([TYPE[sl.sample_id.iloc[i]] for i in epoch_idx]).value_counts(normalize=True).round(4).to_dict(); log("per-epoch sampler:", len(epoch_idx), "draws from", len(ds_tr), "samples; effective per-type share", share, "| raw share", sl.type.value_counts(normalize=True).round(4).to_dict())
curve = []; allrows = []; steps = 0; step_secs = []; raw_only = dict(batches=0, scorer_unchanged=0); stopped = None; gen = torch.Generator().manual_seed(a.seed)
def run_eval(ep):
    t = time.time(); etr, rtr = evaluate(ds_tr, "train", ep); eva, rva = evaluate(ds_va, "heldout", ep); allrows.extend([rtr, rva]); return etr, eva, time.time() - t
etr, eva, es = run_eval(0); curve.append(dict(epoch=0, **{f"eval_train_{k}": v for k, v in etr.items()}, **{f"heldout_{k}": v for k, v in eva.items()}, eval_secs=es)); log(f"epoch 0 (before training): slice eval loss {etr['loss']:.3f} held-out {eva['loss']:.3f} | eval {es:.0f} s")
agent.train()
for ep in range(1, a.epochs + 1):
    el = (time.time() - T0) / 3600; per = (time.time() - T0 - curve[0]["eval_secs"]) / max(ep - 1, 1) / 3600 if ep > 1 else 0
    if el + per > a.cap_h: stopped = f"wall cap {a.cap_h} h: {el:.2f} h used, next epoch needs ~{per:.2f} h"; log("STOP", stopped); break
    perm = [epoch_idx[i] for i in torch.randperm(len(epoch_idx), generator=gen).tolist()]; dl = DataLoader(Subset(ds_tr, perm), batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=default_collate, drop_last=(len(perm) % a.bs) in (1, 2, 3))
    L = {k: 0.0 for k in LK}; n = 0; nl = 0; seen = {}; t_ep = time.time()
    for batch in dl:
        ts = time.time(); has = bool(batch[1]["has_label2"].any()); before = digest(G["scorer"]) if not has else None
        opt.zero_grad(set_to_none=True); pred, ld, t = step_loss(batch); scaler.scale(ld["loss"]).backward(); scaler.step(opt); scaler.update()
        if cuda: torch.cuda.synchronize()
        if not has: raw_only["batches"] += 1; raw_only["scorer_unchanged"] += int(digest(G["scorer"]) == before)
        B = len(t["token"]); n += B; nl += int(batch[1]["has_label2"].sum()); steps += 1; step_secs.append(time.time() - ts)
        for k in LK: L[k] += fv(ld[k]) * B
        for tok in t["token"]: seen[TYPE[tok]] = seen.get(TYPE[tok], 0) + 1
        assert np.isfinite(fv(ld["loss"])), "non-finite loss"
    tr_secs = time.time() - t_ep; etr, eva, es = run_eval(ep)
    row = dict(epoch=ep, **{f"train_{k}": L[k] / n for k in LK}, **{f"eval_train_{k}": v for k, v in etr.items()}, **{f"heldout_{k}": v for k, v in eva.items()}, train_secs=tr_secs, eval_secs=es, steps=steps, samples_seen=n, label2_seen=nl,
               **{f"seen_share_{k}": v / n for k, v in seen.items()}, gpu_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0, gpu_reserved_gb=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0, sec_per_step=float(np.mean(step_secs[-len(dl):])))
    curve.append(row); pd.DataFrame(curve).to_csv(OUT / "curve.csv", index=False); pd.concat(allrows).to_parquet(OUT / "readouts.parquet", index=False)
    log(f"epoch {ep}: train loss {row['train_loss']:.3f} (traj {row['train_trajectory_loss']:.3f}, scorer {row['train_final_score_loss']:.3f}) | slice eval {etr['loss']:.3f} | held-out {eva['loss']:.3f} | {row['sec_per_step']:.2f} s/step, epoch {tr_secs:.0f} s + eval {es:.0f} s | seen {({k: round(v / n, 3) for k, v in seen.items()})} | GPU {row['gpu_reserved_gb']:.1f} GiB")
# ---- S4: raw-only optimiser steps after training must leave the scorer bit-identical
raw_tok = [i for i, t in enumerate(sl.sample_id) if not sl.has_label2.iloc[i]][: 3 * a.bs]; s4 = []
for k in range(0, len(raw_tok) - a.bs + 1, a.bs):
    batch = default_collate([ds_tr[i] for i in raw_tok[k:k + a.bs]]); d0 = digest(G["scorer"]); g0 = digest(G["generator"]); e0 = digest(G["encoder"])
    opt.zero_grad(set_to_none=True); pred, ld, t = step_loss(batch); scaler.scale(ld["loss"]).backward(); scaler.step(opt); scaler.update()
    s4.append(dict(scorer_unchanged=digest(G["scorer"]) == d0, generator_moved=digest(G["generator"]) != g0, encoder_unchanged=digest(G["encoder"]) == e0, scorer_grad_none=all(p.grad is None for p in G["scorer"])))
summary = dict(epochs_done=len(curve) - 1, stopped=stopped, steps=steps, sec_per_step_mean=float(np.mean(step_secs)) if step_secs else None, sec_per_step_median=float(np.median(step_secs)) if step_secs else None, wall_h=(time.time() - T0) / 3600,
               gpu_max_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0, gpu_max_reserved_gb=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0, sampler_share=share, raw_only_batches_in_training=raw_only, s4_post_training_raw_steps=s4,
               slice_samples=len(ds_tr), heldout_samples=len(ds_va), draws_per_epoch=len(epoch_idx), batch_size=a.bs, device=str(dev), torch=torch.__version__, vocab_shape=list(vocab.shape))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1); log("DONE", json.dumps({k: summary[k] for k in ("epochs_done", "stopped", "steps", "sec_per_step_mean", "wall_h", "gpu_max_reserved_gb")})); log("S4", s4)
