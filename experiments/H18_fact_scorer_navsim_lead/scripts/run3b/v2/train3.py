"""Run 3B v2 training (run3b/CLAUDE.md, revised 19 Nov ~02:15). One step = 64 samples (32 NAVSIM + 32 LEAD):
  1. images to the GPU, camera augmentation (angle / position / FOV jitter, camera dropout; lidar2img recomputed)
  2. encoder (WA-JEPA + LoRA) WITH gradient, in groups for memory, activation checkpointing per ViT block (block inputs
     kept; blocks recomputed in backward) -> tokens (64, 4, 512, 1024)
  3. heads ONCE on the 64: agent decoder -> run 3A's detection loss (AgentLoss) and its last-layer matching -> teacher
     forcing on the agents (per sample, schedule) -> planner (4 rounds) -> proposals (detached) to the GPU labeller ->
     path reader on the 129 pool paths + the 32 proposals
  4. label-free losses; then the labels -> fact losses, ranking, run 3A's aux losses on both datasets (key_mask)
  5. one backward, one Adam step (LoRA lr = 0.1 x heads')
Length: --hours of wall at the step time measured over steps 20-49 (or --steps); learning rate: linear warm-up then cosine to
0 over that length. Checkpoint every pass (trainable tensors only, new path). Held-out report (v2/eval/eval_v2.py) at
--smoke-step and after every pass. Stops only on: non-finite loss, total loss not falling (smoke / pass ends), loader wait
> 20 % of the step (from step 250, last 200), live-labelling errors.
    cd {runs}/run3b/v2 && $PY train3.py --name NAME"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (str(HERE / "model"), str(HERE / "data"), str(HERE / "eval"), "{runs}/run3b/r3b",
          "{runs}/run3b/gpulab", "{runs}/run3a-adapted",
          "{data.drivejepa_tree}/navsim_v2"):
    if p not in sys.path:
        sys.path.insert(0, p)
RUNS = Path("{data.paddock_data}/run3b/runs_v2")
UTIL = []


def _util_loop():
    import subprocess
    while True:
        try:
            UTIL.append(float(subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                                      timeout=10).decode().split()[0]))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hours", type=float, default=48.0, help="run length = what fits in this wall time at the measured step time")
    ap.add_argument("--steps", type=int, default=0, help="fixed length instead of --hours")
    ap.add_argument("--lr", type=float, default=1e-4, help="heads' lr (run 3A: Adam 1e-4)")
    ap.add_argument("--lora-mult", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--group", type=int, default=16, help="encoder samples per group (memory)")
    ap.add_argument("--reader-layers", type=int, default=2)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--P", type=float, default=0.108)
    ap.add_argument("--tf-end", type=float, default=0.2, help="teacher-forced share after the decay")
    ap.add_argument("--tf-decay-to", type=float, default=0.6, help="share of the run at which the decay ends")
    ap.add_argument("--tf-noise", default="0.3,3.0,0.5", help="position m, heading deg, speed m/s")
    ap.add_argument("--aug-angle", type=float, default=1.0)
    ap.add_argument("--aug-pos", type=float, default=0.05)
    ap.add_argument("--zoom-max", type=float, default=0.10, help="Navvrat: zoom factor in [1, 1 + zoom_max], one per sample")
    ap.add_argument("--zoom-p", type=float, default=0.5, help="share of samples zoomed")
    ap.add_argument("--aug-drop", type=float, default=0.1)
    ap.add_argument("--compile-heads", type=int, default=1)
    ap.add_argument("--heads-mode", default="reduce-overhead", help="torch.compile mode of the heads (CUDA graphs)")
    ap.add_argument("--compile-blocks", type=int, default=1, help="compile every ViT block (as Phase 1)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--lab-nav", type=int, default=4)
    ap.add_argument("--lab-carla", type=int, default=10)
    ap.add_argument("--smoke-step", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=1008)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps-probe", type=int, default=0, help="stop after this many steps (speed probe; no eval)")
    ap.add_argument("--profile-steps", default="")
    return ap.parse_args()


def tf_share(step, pass_steps, total, end=0.2, decay_to=0.6):
    """1.0 for the first pass, then linear to `end` at decay_to x the run, then constant."""
    if step < pass_steps:
        return 1.0
    stop = max(decay_to * total, pass_steps + 1)
    if step >= stop:
        return end
    return 1.0 + (end - 1.0) * (step - pass_steps) / (stop - pass_steps)


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(total - warmup, 1))))


def aux_losses(pl, got, metas, dev, est, info):
    """run 3A's auxiliary heads (DriveJEPAAgent.score_loss + losses.navsim_aux) on every sample with aux labels (both
    datasets), batch means; key_mask leaves keys with no valid slot out of the validity loss; aux_key_l1 divides by the
    running mean entries per sample x the batch's aux samples (its true count is known only after labelling)."""
    idx = [i for i in range(len(metas)) if i in got and "aux" in got[i]]
    n_all = len(metas)
    z = pl["proposals"].sum() * 0
    if not idx:
        return dict(aux_key_ce=z, aux_key_l1=z, aux_area=z, aux_map=z)
    ni = torch.as_tensor(idx, device=dev)
    A = [got[i]["aux"] for i in idx]
    gv_np = np.stack([a["key_valid"] for a in A])
    gt_s = torch.as_tensor(np.stack([a["key_corners"] for a in A]), device=dev)
    gt_v = torch.as_tensor(gv_np, device=dev)
    gt_a = torch.as_tensor(np.stack([a["ego_areas"] for a in A]), device=dev)
    gt_m = torch.as_tensor(np.stack([a["map_aux"] for a in A]), device=dev)
    km_np = np.stack([np.asarray(a.get("key_mask", np.ones((gv_np.shape[1], gv_np.shape[3]), bool))) for a in A])
    info["aux_masked_keys"] = int((~km_np).sum())
    km = torch.as_tensor(km_np, device=dev)[:, :, None, :].expand(gt_v.shape).float()
    ast = pl["pred_agents_states"][ni].float()
    pred_states = ast[..., :-1].reshape(gt_s.shape)
    pred_logits = ast[..., -1:].reshape(gt_v.shape)
    share = len(idx) / n_all
    out = {}
    out["aux_key_ce"] = (F.binary_cross_entropy_with_logits(pred_logits, gt_v.float(), reduction="none") * km).sum() / (
        n_all * int(np.prod(gt_v.shape[1:])))
    out["aux_area"] = F.binary_cross_entropy_with_logits(pl["pred_area_logit"][ni].float().reshape(gt_a.shape), gt_a.float()) * share
    out["aux_map"] = F.smooth_l1_loss(pl["pred_map_aux"][ni].float(), gt_m) * share
    cnt = int(gv_np.sum()) * int(np.prod(gt_s.shape[gv_np.ndim:]))
    hist = est.get("l1_hist", [])
    per = sum(x for x, _ in hist) / max(sum(m for _, m in hist), 1) if hist else cnt / len(idx)
    div = max(per * n_all, 1.0)
    mask = gt_v.reshape(*gt_v.shape, *([1] * (gt_s.dim() - gt_v.dim()))).float()
    out["aux_key_l1"] = (F.l1_loss(pred_states, gt_s, reduction="none") * mask).sum() / div
    est["l1_hist"] = (hist + [(float(cnt), len(idx))])[-200:]
    info.update(aux_l1_divisor=div, aux_l1_count=cnt, aux_samples=len(idx))
    return out


def main():
    args = parse()
    out = RUNS / args.name
    assert not out.exists(), "never overwrite a run"
    out.mkdir(parents=True)
    json.dump(vars(args), open(out / "args.json", "x"), indent=1)
    import service
    t0 = time.time()
    lab = service.GPULabeller(args.lab_nav, args.lab_carla)  # spawned before CUDA
    print(f"labellers ready {time.time() - t0:.0f} s", flush=True)
    dev = torch.device("cuda")
    import augment2 as AUG
    import run3b_data2 as D2
    import losses as L
    import model2 as M2
    import train2 as T2
    from navsim.agents.drive_jepa_perception_based.agent_decoder import AgentLoss
    torch.manual_seed(args.seed)
    import torch._dynamo.config as dyn
    dyn.recompile_limit = 64  # the fixed shapes need a few graphs per call site; never fall back to eager
    dyn.cache_size_limit = 64
    vit, msg = M2.build_encoder(dev, compile_blocks=bool(args.compile_blocks))
    net = M2.Run3Bv2(vit, args.reader_layers).to(dev)
    print("encoder load:", msg, flush=True)
    det_loss = AgentLoss()
    ds = D2.Run3BData2("train", seed=args.seed)
    ds_h = D2.Run3BData2("heldout", seed=12345)
    pass_steps = int(np.ceil((len(ds.nav) + len(ds.lead)) / 64))
    total = args.steps or 10 ** 9  # set from the measured step time at step 50 when --hours
    sampler = D2.StepSampler(ds, 10 ** 9, seed=args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, collate_fn=D2.collate2, prefetch_factor=2,
                                     persistent_workers=True, pin_memory=True)
    lora = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    heads = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.Adam([dict(params=heads, lr=args.lr, name="heads"), dict(params=lora, lr=args.lr * args.lora_mult, name="lora")])
    base_lr = [g["lr"] for g in opt.param_groups]
    if args.compile_heads:
        for name in ("agent_decoder", "planner", "reader"):
            setattr(net, name, torch.compile(getattr(net, name), mode=args.heads_mode, dynamic=False))
    W = dict(planner=1.0, det=1.0, aux_key_ce=1.0, aux_key_l1=0.1, aux_area=2.0, aux_map=1.0, metres_head=1.0,
             ranking=args.rank_weight, **{f"fact_{k}": 1.0 for k in L.YESNO + L.NUMBERS})
    json.dump(dict(weights=W, n_train_nav=len(ds.nav), n_train_lead=len(ds.lead), pass_steps=pass_steps, hours=args.hours,
                   steps=args.steps, n_heldout_nav=len(ds_h.nav), n_heldout_lead=len(ds_h.lead),
                   trainable=dict(heads=sum(p.numel() for p in heads), lora=sum(p.numel() for p in lora)),
                   tf=dict(first_pass=1.0, end=args.tf_end, decay_to=args.tf_decay_to, noise=args.tf_noise),
                   aug=dict(angle_deg=args.aug_angle, pos_m=args.aug_pos, zoom_max=args.zoom_max, zoom_p=args.zoom_p, drop=args.aug_drop),
                   lr=dict(heads=args.lr, lora=args.lr * args.lora_mult, warmup=args.warmup, schedule="cosine to 0 over the run"),
                   mps=os.path.exists("/tmp/nvidia-mps") or bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))),
              open(out / "setup.json", "x"), indent=1)
    noise = [float(x) for x in args.tf_noise.split(",")]
    noise = (noise[0], math.radians(noise[1]), noise[2])
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    gen_aug = torch.Generator(device=dev).manual_seed(args.seed + 1)
    gen_tf = torch.Generator(device=dev).manual_seed(args.seed + 2)
    est = {}
    logf = open(out / "train_log.jsonl", "a")
    net.train()
    import threading
    threading.Thread(target=_util_loop, daemon=True).start()
    it = iter(dl)
    lab.prep(0, [ds.label_meta(g) for g in sampler.batch(0)])
    hist_total, hist_wait, n_evals, t_run0 = [], [], 0, time.time()
    step = 0
    while step < total:
        T = {}
        t_step0 = time.time()
        torch.compiler.cudagraph_mark_step_begin()  # CUDA graph trees: a new step (earlier steps' graph outputs may be reused)
        t = time.time()
        batch = next(it)
        T["loader_wait"] = time.time() - t
        metas = batch["meta"]
        Bn = len(metas)
        lab.prep(step + 1, [ds.label_meta(g) for g in sampler.batch(step + 1)])
        is_nav = (batch["ds"] == 0).to(dev)
        frac = tf_share(step, pass_steps, total if total < 10 ** 9 else 10 * pass_steps, args.tf_end, args.tf_decay_to)
        tf = torch.rand(Bn, device=dev, generator=gen_tf) < frac
        for g, f in zip(opt.param_groups, base_lr):
            g["lr"] = f * lr_factor(step, total if total < 10 ** 9 else 10 ** 9, args.warmup)
        # 1-2 augmentation + encoder
        t = time.time()
        images = batch["images"].to(dev, non_blocking=True)
        images, l2i, _, aug_info = AUG.augment(images, batch["K"].to(dev), batch["ego2cam"].to(dev), gen_aug,
                                               dict(angle_deg=args.aug_angle, shift_m=args.aug_pos, fov=0.0, drop_p=args.aug_drop,
                                                    zoom_max=args.zoom_max, zoom_p=args.zoom_p), lidar2img=batch["lidar2img"].to(dev))
        tokens = net.encoder(D2.to_clips(images), group=args.group)
        T["encoder"] = time.time() - t
        # 3 heads once
        t = time.time()
        tgt_det = {k: v.to(dev) for k, v in batch["det"].items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs, agents = net.decode(tokens, l2i)
        l_det, det_parts = det_loss([{k: (v.float() if torch.is_tensor(v) and k != "feat" else v) for k, v in o.items()} for o in outs],
                                    tgt_det)
        agents = M2.agents_tf(agents, det_parts.pop("_match"), tgt_det, tf, noise, gen_tf)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl = net.plan(tokens, l2i, batch["ego_status"].to(dev), agents)
        props = pl["proposals"].detach().float()
        lab.label(step, metas, props.cpu().numpy())
        paths = torch.cat([batch["paths"].to(dev), props], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, numbers = net.read(paths, tokens, l2i, agents)
        Ls = {"det": l_det}
        Ls["planner"], plan_parts = L.planner(pl["proposal_list"], batch["expert"].to(dev), [x.to(dev) for x in batch["teachers"]])
        Ls["metres_head"] = (pl["expert_metres"].float() - batch["R"].to(dev)).abs().mean()
        T["heads"] = time.time() - t
        t = time.time()
        got, errs = lab.collect(step, Bn)
        T["label_wait"] = time.time() - t
        t = time.time()
        lt, ok, la, lb = T2.live_targets(got, list(range(Bn)), batch["R"].numpy(), batch["ntg"].numpy(), (batch["ds"] == 0).numpy(), args.P)
        tgt = {k: torch.cat([batch["facts"][k].to(dev), torch.as_tensor(lt[k], device=dev)], 1) for k in lt}
        valid = torch.cat([batch["valid"].to(dev), torch.as_tensor(ok, device=dev)[:, None].expand(-1, 32)], 1)
        Ls.update(L.facts(logits, numbers, tgt, valid, is_nav))
        true_sc = torch.cat([batch["a"] - args.P * batch["b"], torch.as_tensor(la - args.P * lb)], 1).to(dev)
        pred_sc = L.scores_from_pred(logits, numbers, batch["R"].to(dev), batch["ntg"].to(dev), is_nav, args.P)
        Ls["ranking"] = L.ranking(pred_sc, true_sc, valid, gen)
        info = {}
        Ls.update(aux_losses(pl, got, metas, dev, est, info))
        total_loss = sum(W[k] * v for k, v in Ls.items())
        total_loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        T["loss_backward"] = time.time() - t
        T["step"] = time.time() - t_step0
        vals = torch.stack([v.detach().float() for v in Ls.values()]).tolist()
        Lf = dict(zip(Ls.keys(), vals))
        tot = sum(W[k] * v for k, v in Lf.items())
        rec = dict(step=step, time=time.time(), losses=Lf, share={k: W[k] * v / tot for k, v in Lf.items()}, total=tot, timing=T,
                   det_parts={k: float(v) for k, v in det_parts.items()}, planner_rounds=plan_parts, tf_share=frac, tf_samples=int(tf.sum()),
                   lr=[g["lr"] for g in opt.param_groups], label_errors=len(errs), label_missing=int(Bn - len(got)), **info,
                   gpu_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, gpu_util=float(np.mean(UTIL)) if UTIL else None,
                   samples_per_s=Bn / T["step"], teachers=[int(x.shape[0]) for x in batch["teachers"]])
        if errs:
            rec["error_example"] = next(iter(errs.values()))[-600:]
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"step {step} total {tot:.3f} step_s {T['step']:.1f} (loader {T['loader_wait']:.1f} enc {T['encoder']:.1f} heads {T['heads']:.1f} "
              f"wait {T['label_wait']:.1f} loss+bwd {T['loss_backward']:.1f}) tf {frac:.2f} mem {rec['gpu_mem_gb']:.1f} GB lab_err {len(errs)}",
              flush=True)
        torch.cuda.reset_peak_memory_stats()
        UTIL.clear()
        hist_total.append(tot)
        hist_wait.append((T["loader_wait"], T["step"]))
        if args.max_steps_probe and step + 1 >= args.max_steps_probe:
            (out / "PROBE_DONE").write_text(json.dumps(dict(steps=step + 1, step_s_mean_last=float(np.mean([s for _, s in hist_wait[-10:]])))))
            break
        if not args.steps and total >= 10 ** 9 and step == 49:
            s_step = float(np.mean([s for _, s in hist_wait[20:50]]))
            total = int(args.hours * 3600 / s_step)
            json.dump(dict(measured_step_s=s_step, total_steps=total, passes=total / pass_steps), open(out / "length.json", "x"), indent=1)
            print(f"LENGTH: {s_step:.2f} s/step -> {total} steps ({total / pass_steps:.2f} passes) in {args.hours} h", flush=True)
        stop = None
        if not np.isfinite(tot):
            stop = f"non-finite total loss at step {step}"
        elif errs:
            stop = f"{len(errs)} live-labelling errors at step {step}"
        elif step >= 250 and np.mean([w for w, _ in hist_wait[-200:]]) > 0.2 * np.mean([s for _, s in hist_wait[-200:]]):
            stop = f"loader wait {np.mean([w for w, _ in hist_wait[-200:]]):.2f} s > 20 % of step"
        tags = []
        if step + 1 == args.smoke_step:
            tags.append("smoke")
        if (step + 1) % pass_steps == 0:
            tags.append(f"pass{(step + 1) // pass_steps}")
        if step + 1 == total and not tags:
            tags.append("final")
        if stop and not tags:
            tags.append("stopped")
        if tags:
            tag = "_".join(tags)
            sd = {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad}
            dst = out / f"trainable_step{step + 1:06d}_{tag}.pt"
            torch.save(dict(trainable=sd, step=step + 1, args=vars(args), tag=tag), str(dst) + ".tmp")
            os.replace(str(dst) + ".tmp", dst)
            if not (stop and stop.startswith("non-finite")):
              try:  # a failing held-out report must not stop training (spec: report without stopping); the error is logged
                import eval_v2 as EV
                te = time.time()
                res = EV.run_eval_v2(lambda b, teacher_forced: model_fn(net, b, teacher_forced, dev, det_loss, noise, gen_tf), lab, ds_h,
                                     args.eval_n, args.eval_n,
                                     32, True, args.P, collate=D2.collate2, step_base=-(10 ** 6) * (n_evals + 1),
                                     out=str(out / f"eval_step{step + 1:06d}_{tag}.json"),
                                     extra=dict(step=step + 1, tag=tag, ckpt=str(dst)))
                n_evals += 1
                print(f"EVAL {tag} step {step + 1} done in {(time.time() - te) / 60:.1f} min", flush=True)
              except Exception:  # noqa: BLE001
                import traceback
                (out / f"eval_step{step + 1:06d}_{tag}_FAILED.txt").write_text(traceback.format_exc())
                print(f"EVAL {tag} step {step + 1} FAILED (traceback in the run dir); training continues", flush=True)
              net.train()
              lab.prep(step + 1, [ds.label_meta(g) for g in sampler.batch(step + 1)])
            if ("smoke" in tags or any(t_.startswith("pass") for t_ in tags)) and len(hist_total) >= 450 and not stop:
                if np.mean(hist_total[-200:]) >= np.mean(hist_total[50:250]):
                    stop = f"total loss not falling at step {step + 1}"
        if stop:
            (out / "STOPPED").write_text(stop + "\n")
            print("STOPPED:", stop, flush=True)
            break
        step += 1
    else:
        (out / "TRAIN_DONE").write_text(str(total))
    lab.close()


@torch.no_grad()
def model_fn(net, batch, teacher_forced, dev, det_loss, noise, gen):
    """eval_v2's contract: the held-out batch through the model, with true agents (teacher_forced) or the decoder's."""
    import run3b_data2 as D2
    torch.compiler.cudagraph_mark_step_begin()  # each held-out forward is its own CUDA-graph step; outputs are cloned below
    net.eval()
    images = batch["images"].to(dev)
    l2i = batch["lidar2img"].to(dev).float()
    tokens = net.encoder(D2.to_clips(images), group=16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outs, agents = net.decode(tokens, l2i)
    if teacher_forced:
        tgt_det = {k: v.to(dev) for k, v in batch["det"].items()}
        _, parts = det_loss([{k: (v.float() if torch.is_tensor(v) and k != "feat" else v) for k, v in o.items()} for o in outs], tgt_det)
        agents = __import__("model2").agents_tf(agents, parts["_match"], tgt_det, torch.ones(len(images), dtype=torch.bool, device=dev),
                                               (0.0, 0.0, 0.0), gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pl = net.plan(tokens, l2i, batch["ego_status"].to(dev), agents)
        paths = torch.cat([batch["paths"].to(dev), pl["proposals"].float()], 1)
        logits, numbers = net.read(paths, tokens, l2i, agents)
    c = lambda x: x.float().clone()  # noqa: E731  (CUDA-graph outputs are overwritten by the next run)
    return dict(agent_outs=[{"logits": c(o["logits"]), "box": c(o["box"]), "vel": c(o["vel"])} for o in outs],
                proposals=c(pl["proposals"]), path_logits=c(logits), path_numbers=c(numbers), expert_metres=c(pl["expert_metres"]))


if __name__ == "__main__":
    main()
