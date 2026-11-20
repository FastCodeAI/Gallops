"""run 3B training, grouped step (18 Nov, Navvrat: "lets do this optimization"). Same model, data, losses and weights as
train.py; one step = batch 64 (32 NAVSIM + 32 LEAD) processed in groups of G (default 8) with ONE encoder pass:
  per group g: encoder (LoRA) forward WITH gradient -> agent decoder, planner, path reader on the group; the planner's 32 go
  to the labellers (detached); the label-free losses are formed at once; when the group's labels arrive the label losses
  are formed and the group's total is backpropagated (encoder included). Up to --inflight groups are held, so the labels of
  group g are computed while the GPU runs group g+1. Gradients of all groups add up; one Adam step per batch (as train.py).
Normalisers: every divisor is the BATCH's (64 samples), as in train.py, computed before the first group: detection's agent
count, planner / metres-head batch size, the fact masks' cell counts, NAVSIM aux sample count. Three divisors depend on
live labels and are fixed before labelling: fact_red's cell count counts every live cell as light-known (train.py: only
light-known cells; a live cell is unknown only when it crosses an unread light); ranking's sample count is the samples
whose stored rows already hold a pair (train.py: also samples whose pair needs the live rows; logged as rank_live_only);
aux_key_l1's valid-entry count is the running mean entries per NAVSIM sample (last 200 steps) x the batch's NAVSIM samples
(logged next to this step's true count; the count varies a lot from batch to batch). Each is one scale for the whole batch,
so every sample and entry keeps train.py's relative weight within the step. Label errors stop the run as before.
Resume: --resume FILE continues from a resume checkpoint (trainable tensors + Adam state + step), written every
--resume-every steps (resume_latest.pt, atomically).
    cd {runs}/run3b/r3b && $PY train2.py --name NAME [--group 8] [--inflight 2] [--resume FILE]"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), "{runs}/run3a-adapted", "{data.drivejepa_tree}/navsim_v2"]
RUNS = Path("{data.paddock_data}/run3b/runs")
UTIL = []


def _util_loop():
    """GPU utilisation (nvidia-smi, every 5 s; the whole GPU, other users' work included)."""
    import subprocess
    while True:
        try:
            UTIL.append(float(subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                                       "-i", "0"], timeout=10).decode().split()[0]))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
P1_CKPT = "{data.paddock_data}/run3b/phase1_local/ckpt_step0042780.pt"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = passes x pass length")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--smoke-step", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=1008, help="held-out samples per dataset in each evaluation")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-mult", type=float, default=0.1)
    ap.add_argument("--reader-layers", type=int, default=2)
    ap.add_argument("--scorer-grad-stop", action="store_true")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lab-nav", type=int, default=8)
    ap.add_argument("--lab-carla", type=int, default=12)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--P", type=float, default=0.108)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--group", type=int, default=8, help="samples per group (encoder + heads); x 3 cameras must be <= 32")
    ap.add_argument("--inflight", type=int, default=2, help="groups held while their labels are computed")
    ap.add_argument("--resume", default="", help="resume checkpoint to continue from")
    ap.add_argument("--resume-every", type=int, default=250)
    ap.add_argument("--lead-aux", action="store_true", help="run 3A's auxiliary heads on LEAD samples too (needs the GPU labeller's "
                    "LEAD aux labels)")
    ap.add_argument("--profile-steps", default="", help="comma list of steps to run under torch.profiler (trace + tables in "
                    "the run dir; training continues)")
    ap.add_argument("--gpu-labels", action="store_true", help="live labels from gpulab/service.GPULabeller (lab-nav / lab-carla = "
                    "its CPU prep workers)")
    return ap.parse_args()


def build(args, dev):
    from phase1.epoch_eval import CFG, load_model
    import model as M
    p1 = load_model(P1_CKPT, "lora", CFG)  # Phase 1 encoder (LoRA) + agent decoder, trainable tensors copied from the checkpoint
    enc, dec = p1.encoder, p1.agent_dec
    enc.vit.use_activation_checkpointing = False  # Phase 1 run: no gradient checkpointing, compiled blocks
    del p1
    torch.manual_seed(args.seed)
    net = M.Run3B(enc, dec, args.reader_layers, args.scorer_grad_stop).to(dev)
    for n, p in net.named_parameters():
        if n.startswith("encoder.") and ("A_q" not in n and "B_q" not in n and "A_v" not in n and "B_v" not in n):
            p.requires_grad_(False)
    if args.compile:
        torch._inductor.config.emulate_precision_casts = True  # as Phase 1's run (train.py --compile-vit)
        for blk in net.encoder.vit.blocks:
            blk.compile()
    return net, CFG


def param_groups(net, args):
    lora = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    rest = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    return [dict(params=rest, lr=args.lr, name="heads"), dict(params=lora, lr=args.lr * args.lora_mult, name="lora")]


def live_targets(got, slots, R, ntg, is_nav, P):
    """labeller output for the planner's 32 of each sample -> (B, 32, 8) targets + true score parts."""
    import score as S
    B = len(slots)
    t = {k: np.zeros((B, 32, 8), np.float32) for k in ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "comfort",
                                                         "metres_sofar", "ttc", "cutin_brake", "lane_off")}
    t["red_known"] = np.zeros((B, 32, 8), bool)
    ok = np.zeros(B, bool)
    a = np.zeros((B, 32), np.float32)
    b = np.zeros((B, 32), np.float32)
    for i, s in enumerate(slots):
        g = got.get(s)
        if g is None:
            continue
        ok[i] = True
        for k in ("hit", "offroad", "left_route", "stop", "wrong_way"):
            t[k][i] = g[k]
        t["red"][i] = g["red"] == 1
        t["red_known"][i] = g["red"] >= 0
        t["comfort"][i] = g["comfort_step"]
        t["metres_sofar"][i] = np.cumsum(g["metres"].astype(np.float64), -1)
        t["ttc"][i] = np.minimum(g["ttc"].astype(np.float32), 4.0)
        t["cutin_brake"][i] = np.minimum(g["cutin_brake"].astype(np.float32), 8.0)
        t["lane_off"][i] = g["lane_off"]
        a[i], b[i] = S.parts_np(dict(g, comfort_step=g["comfort_step"]), R[i], bool(ntg[i]), bool(is_nav[i]))
    return t, ok, a, b


def make_labeller(n_nav, n_carla, gpu=False):
    """labeller.Labeller plus slot-level label / prep / collect; results of other slots are kept until asked for.
    gpu: gpulab/service.GPULabeller instead (same interface; CPU prep workers + one GPU label process)."""
    if gpu:
        sys.path.insert(0, "{runs}/run3b/gpulab")
        import service
        return service.GPULabeller(n_nav, n_carla)
    from labeller import Labeller

    class GroupLabeller(Labeller):
        def __init__(self, a, b):
            super().__init__(a, b)
            self.buf = {}

        def label_slots(self, step, slots, metas, poses):
            for s, m, p in zip(slots, metas, poses):
                self._worker(self.kind_of(m), s)[2].put(("label", (step, s), (np.asarray(p, np.float32), m)))

        def prep_slots(self, step, slots, metas):
            for s, m in zip(slots, metas):
                self._worker(self.kind_of(m), s)[1].put(("prep", (step, s), m))

        def collect_keys(self, keys, timeout=600):
            keys = list(keys)
            t_end = time.time() + timeout
            while any(k not in self.buf for k in keys):
                if time.time() > t_end:
                    raise TimeoutError(f"labels missing after {timeout} s: {[k for k in keys if k not in self.buf][:4]}")
                try:
                    msg = self.out_q.get(timeout=1.0)
                except Exception:  # noqa: BLE001  (queue.Empty)
                    continue
                if msg[0] in ("label", "error"):
                    self.buf[msg[1]] = msg
            got, errs = {}, {}
            for k in keys:
                msg = self.buf.pop(k)
                (got if msg[0] == "label" else errs)[k[1]] = msg[2]
            return got, errs

        def collect(self, step, n, timeout=600):
            return self.collect_keys([(step, s) for s in range(n)], timeout)

    return GroupLabeller(n_nav, n_carla)


def det_losses(crit, agent_outs, pack, n_total, dev):
    """losses.detection on one group, divided by the batch's agent count."""
    from phase1.loss.criterion import _to_dev
    sums = {"agent_cls": 0.0, "agent_box": 0.0, "agent_vel": 0.0}
    for o in agent_outs:
        m = crit.match_agents_all(o, pack)
        idx = _to_dev(list(crit._flat_matches(m, pack.a_off)), dev)
        for k, v in crit.agent_sums(o, pack, idx).items():
            sums[k] = sums[k] + v / len(agent_outs)
    n = float(n_total)
    return {"det_cls": sums["agent_cls"] / max(n, 1.0), "det_box": sums["agent_box"] / max(8 * n, 1.0),
            "det_vel": sums["agent_vel"] / max(2 * n, 1.0)}


def planner_g(proposal_list, expert, teachers, B_total, prev_weight=0.1):
    """losses.planner on one group, divided by the batch size (its means over samples are sums / B)."""
    traj, parts = 0.0, {}
    for r, props in enumerate(proposal_list):
        props = props.float()
        min_loss = torch.linalg.norm(props - expert[:, None].float(), dim=-1, ord=1).mean(-1).amin(1).sum() / B_total
        pseudo = props.new_zeros(())
        for b in range(props.shape[0]):
            t = teachers[b]
            if t.shape[0] == 0:
                continue
            d = props[b, None] - t[:, None].to(props)
            pseudo = pseudo + torch.linalg.norm(d, dim=-1, ord=1).mean(-1).amin(-1).mean()
        min_loss = min_loss + 0.5 * pseudo / B_total
        traj = prev_weight * traj + min_loss
        parts[f"round{r}"] = min_loss.detach()
    return traj, parts


def facts_g(logits, numbers, tgt, valid, is_nav, Dv):
    """losses.facts on one group, each mask's sum divided by the batch's cell count Dv[k]."""
    import torch.nn.functional as F
    import losses as L
    out = {}
    v = valid[..., None].expand(-1, -1, 8).float()
    for i, k in enumerate(L.YESNO):
        m = v.clone()
        if k == "red":
            m = m * tgt["red_known"].float()
        if k == "comfort":
            m = m * is_nav[:, None, None].float()
        bce = F.binary_cross_entropy_with_logits(logits[..., i], tgt[k].float(), reduction="none")
        out[f"fact_{k}"] = (bce * m).sum() / Dv[k]
    for i, k in enumerate(L.NUMBERS):
        sl = F.smooth_l1_loss(numbers[..., i], tgt[k].float(), reduction="none")
        out[f"fact_{k}"] = (sl * v).sum() / Dv["_cells"]
    return out


def ranking_sum(pred_score, true_score, valid, gen):
    """losses.ranking's sum over samples (same draws, same order) and the samples used."""
    import torch.nn.functional as F
    import losses as L
    B = pred_score.shape[0]
    tot, used = pred_score.new_zeros(()), []
    for b in range(B):
        idx = torch.where(valid[b])[0]
        if len(idx) < 2:
            continue
        s = true_score[b, idx]
        diff = s[:, None] - s[None, :]
        ok = diff > L.RANK_MARGIN
        if not ok.any():
            continue
        pct = (torch.argsort(torch.argsort(s)).float() + 0.5) / len(s)
        w = (pct[:, None].expand_as(diff) * ok.float()).reshape(-1)
        k = min(L.RANK_PAIRS, int(ok.sum()))
        pick = torch.multinomial(w, k, replacement=False, generator=gen)
        hi, lo = pick // len(idx), pick % len(idx)
        ps = pred_score[b, idx]
        tot = tot + F.softplus(-(ps[hi] - ps[lo])).mean()
        used.append(b)
    return tot, used


def RF(name):
    from torch.profiler import record_function
    return record_function(name)


def _up(x, dev):
    """host -> device without a stream sync (pinned staging, non_blocking)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    return x.pin_memory().to(dev, non_blocking=True)


_NORM = {}


def _normalise_dev(images, dev):
    """phase1.model.encoder.normalise with mean / std cached on the device (normalise builds them from a list per call, a
    blocking host-to-device copy = a stream sync). Same arithmetic."""
    from phase1.model.encoder import MEAN, STD
    if dev not in _NORM:
        _NORM[dev] = (torch.tensor(MEAN, device=dev), torch.tensor(STD, device=dev))
    mean, std = _NORM[dev]
    x = images.float() / 255.0
    shape = [1] * (x.dim() - 3) + [3, 1, 1]
    return (x - mean.view(shape)) / std.view(shape)


def ranking_sum_np(pred_score, true_np, valid_np, gen, dev):
    """ranking_sum with its data-dependent sizes (valid paths, pair mask, pair count) taken on the host, where the true scores
    and validity already are: no stream syncs. The percentile weights and the draw are computed on the device exactly as
    ranking_sum does (same values, same generator, same draws)."""
    import torch.nn.functional as F
    import losses as L
    tot, used = pred_score.new_zeros(()), []
    true_d = _up(np.ascontiguousarray(true_np, np.float32), dev)
    for b in range(true_np.shape[0]):
        idx = np.nonzero(valid_np[b])[0]
        if len(idx) < 2:
            continue
        s_np = true_np[b, idx].astype(np.float32)
        ok_np = (s_np[:, None] - s_np[None, :]) > np.float32(L.RANK_MARGIN)
        if not ok_np.any():
            continue
        idx_d = _up(idx.astype(np.int64), dev)
        s = true_d[b].index_select(0, idx_d)
        pct = (torch.argsort(torch.argsort(s)).float() + 0.5) / len(s)
        w = (pct[:, None].expand(len(s), len(s)) * _up(ok_np, dev).float()).reshape(-1)
        k = min(L.RANK_PAIRS, int(ok_np.sum()))
        pick = torch.multinomial(w, k, replacement=False, generator=gen)
        hi, lo = pick // len(idx), pick % len(idx)
        ps = pred_score[b].index_select(0, idx_d)
        tot = tot + F.softplus(-(ps[hi] - ps[lo])).mean()
        used.append(b)
    return tot, used


def grouped_step(net, crit, lab, batch, step, args, W, gen, dev, metas_next, T, est):
    """one batch, grouped (module docstring): forward + labels + backward of every group; gradients accumulate in the
    parameters (no optimizer step here). Returns losses (batch-normalised sums over groups), planner parts, labels got,
    label errors, info. est: carried across steps (aux_key_l1's divisor: running entries per NAVSIM sample).
    Order (inflight 2): queue group g's encoder -> prepare + queue group g-1's backward (host work while the GPU runs g's
    encoder) -> g's heads -> g's finish (proposals to the labellers, path reader queued, label-free losses). The batch goes
    to the device once (pinned, non-blocking); no host read of a loss until the step's end, so the GPU queue stays full."""
    import torch.nn.functional as F
    import losses as L
    t_setup = time.time()
    metas = batch["meta"]
    Bn = batch["images"].shape[0]
    bounds = [(s, min(s + args.group, Bn)) for s in range(0, Bn, args.group)]
    # host-side divisors (train.py's batch values)
    packs = [L.AgentPack(batch["targets"][a:b], dev, crit.mc) for a, b in bounds]
    n_agents = float(sum(sum(p.na) for p in packs))
    vb = torch.cat([batch["valid"], torch.ones(Bn, 32, dtype=torch.bool)], 1)
    cells = max(float(vb.sum()) * 8, 1.0)
    Dv = {k: cells for k in L.YESNO}
    Dv["_cells"] = cells
    Dv["red"] = max(float((batch["valid"][..., None] & batch["facts"]["red_known"].bool()).sum()) + Bn * 32 * 8, 1.0)
    Dv["comfort"] = max(float(vb[batch["ds"] == 0].sum()) * 8, 1.0)
    true_st_np = (batch["a"] - args.P * batch["b"]).float().numpy()
    valid_np = batch["valid"].numpy()
    stored_pair = np.zeros(Bn, bool)
    for i in range(Bn):
        v_ = true_st_np[i][valid_np[i]]
        stored_pair[i] = len(v_) >= 2 and float(v_.max() - v_.min()) > L.RANK_MARGIN
    D_rank = max(int(stored_pair.sum()), 1)
    n_nav_total = int((batch["ds"] == 0).sum())
    n_aux_total = n_nav_total + (int((batch["ds"] != 0).sum()) if getattr(args, "lead_aux", False) else 0)
    # the batch on the device, once
    G = {k: _up(batch[k], dev) for k in ("images", "lidar2img", "ego_status", "expert", "R", "ntg", "paths", "valid")}
    G["is_nav"] = _up(batch["ds"] == 0, dev)
    G["facts"] = {k: _up(v, dev) for k, v in batch["facts"].items()}
    G["teachers"] = [_up(x, dev) for x in batch["teachers"]]
    Ls, parts_tot, got_all, errs_all = {}, {}, {}, {}
    rank_live_only = 0
    l1_true = 0
    n_masked_keys = 0
    amp = bool(getattr(args, "amp", True))
    T["setup"] = T.get("setup", 0.0) + time.time() - t_setup
    for k in ("enc", "heads", "finish", "label_wait", "loss_bwd", "tail"):
        T.setdefault(k, 0.0)

    def enc(gi):
        a, b = bounds[gi]
        t = time.time()
        with RF("enc"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = net.encoder(_normalise_dev(G["images"][a:b], dev))
        T["enc"] += time.time() - t
        return dict(gi=gi, a=a, b=b, y=y)

    def heads(c):
        a, b = c["a"], c["b"]
        t = time.time()
        with RF("heads"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            c["agent_outs"], c["pl"], c["agents"], c["tl"] = net.heads(c.pop("y"), G["lidar2img"][a:b], G["ego_status"][a:b])
        T["heads"] += time.time() - t

    def finish(c):
        a, b = c["a"], c["b"]
        t = time.time()
        pl = c["pl"]
        props = pl["proposals"].detach().float()
        with RF("finish.props_to_labeller"):
            lab.label_slots(step, list(range(a, b)), metas[a:b], props.cpu().numpy())
        paths = torch.cat([G["paths"][a:b], props], 1)  # (G, 129 + 32, 8, 3)
        with RF("finish.path_reader"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            c["logits"], c["numbers"] = net.read(paths, c.pop("tl"), G["lidar2img"][a:b], c.pop("agents"))
        with RF("finish.detection_loss"):
            Lg = det_losses(crit, c.pop("agent_outs"), packs[c["gi"]], n_agents, dev)
        with RF("finish.planner_loss"):
            Lg["planner"], c["parts"] = planner_g(pl["proposal_list"], G["expert"][a:b], G["teachers"][a:b], Bn)
            Lg["metres_head"] = (pl["expert_metres"].float() - G["R"][a:b]).abs().sum() / (Bn * 8)
        c["Lg"] = Lg
        T["finish"] += time.time() - t

    def backward(c):
        nonlocal rank_live_only, l1_true, n_masked_keys
        a, b = c["a"], c["b"]
        n = b - a
        t = time.time()
        with RF("bwd.label_wait"):
            got, errs = lab.collect_keys([(step, i) for i in range(a, b)])
        T["label_wait"] += time.time() - t
        if metas_next is not None:  # the next batch's scenes for these slots load now
            lab.prep_slots(step + 1, list(range(a, b)), metas_next[a:b])
        t = time.time()
        loc = {i - a: v for i, v in got.items()}
        lt, ok, la, lb = live_targets(loc, list(range(n)), batch["R"][a:b].numpy(), batch["ntg"][a:b].numpy(),
                                      (batch["ds"][a:b] == 0).numpy(), args.P)
        tgt = {k: torch.cat([G["facts"][k][a:b], _up(lt[k], dev)], 1) for k in lt}
        valid = torch.cat([G["valid"][a:b], _up(ok, dev)[:, None].expand(-1, 32)], 1)
        is_nav = G["is_nav"][a:b]
        Lg, pl = c["Lg"], c["pl"]
        with RF("bwd.fact_losses"):
            Lg.update(facts_g(c["logits"], c["numbers"], tgt, valid, is_nav, Dv))
        true_np = np.concatenate([true_st_np[a:b], (la - args.P * lb).astype(np.float32)], 1)
        valid_np_g = np.concatenate([valid_np[a:b], np.repeat(ok[:, None], 32, 1)], 1)
        with RF("bwd.ranking_loss"):
            pred_sc = L.scores_from_pred(c["logits"], c["numbers"], G["R"][a:b], G["ntg"][a:b], is_nav, args.P)
            tot, used = ranking_sum_np(pred_sc, true_np, valid_np_g, gen, dev)
        Lg["ranking"] = tot / D_rank
        rank_live_only += int(sum(1 for i in used if not stored_pair[a + i]))
        # run 3A's auxiliary heads (DriveJEPAAgent.score_loss + losses.navsim_aux, written without host reads) on every sample
        # whose labels carry "aux": NAVSIM always; LEAD with --lead-aux (Navvrat 19 Nov: "train the auxiliary heads on LEAD
        # samples too, same weights as NAVSIM"). Divisors are the batch's aux samples (n_aux_total, known before labelling).
        aux_loc = [i for i in range(n) if i in loc and "aux" in loc[i] and (metas[a + i]["ds"] == 0 or getattr(args, "lead_aux", False))]
        if aux_loc:
            share = len(aux_loc) / max(n_aux_total, 1)
            ni = _up(np.asarray(aux_loc, np.int64), dev)
            gv_np = np.stack([loc[i]["aux"]["key_valid"] for i in aux_loc])
            gt_s = _up(np.stack([loc[i]["aux"]["key_corners"] for i in aux_loc]), dev)
            gt_v = _up(gv_np, dev)
            gt_a = _up(np.stack([loc[i]["aux"]["ego_areas"] for i in aux_loc]), dev)
            gt_m = _up(np.stack([loc[i]["aux"]["map_aux"] for i in aux_loc]), dev)
            # key_mask (P, 2): False = a key agent exists but no slot is valid (its validity target would say "none"); those
            # (path, slot) entries are left out of the validity loss. Labels without key_mask (NAVSIM CPU labeller) mask nothing.
            km_np = np.stack([np.asarray(loc[i]["aux"].get("key_mask", np.ones(gv_np.shape[1:2] + gv_np.shape[3:], bool)))
                              for i in aux_loc])  # (n, P, 2)
            n_masked_keys += int((~km_np).sum())
            km = _up(km_np, dev)[:, :, None, :].expand(gt_v.shape).float()
            ast = pl["pred_agents_states"].index_select(0, ni).float()
            pred_states = ast[..., :-1].reshape(gt_s.shape)
            pred_logits = ast[..., -1:].reshape(gt_v.shape)
            per_sample = int(np.prod(gt_v.shape[1:]))
            Lg["aux_key_ce"] = (F.binary_cross_entropy_with_logits(pred_logits, gt_v.float(), reduction="none") * km).sum() / (
                max(n_aux_total, 1) * per_sample)
            Lg["aux_area"] = F.binary_cross_entropy_with_logits(pl["pred_area_logit"].index_select(0, ni).float().reshape(gt_a.shape),
                                                                gt_a.float(), reduction="mean") * share
            Lg["aux_map"] = F.smooth_l1_loss(pl["pred_map_aux"].index_select(0, ni).float(), gt_m) * share
            # aux_key_l1: run 3A's mean over the batch's valid key-agent entries = sum here / the entry count (running mean)
            cnt = int(gv_np.sum()) * int(np.prod(gt_s.shape[gv_np.ndim:]))
            l1_true += cnt
            mask = gt_v.reshape(*gt_v.shape, *([1] * (gt_s.dim() - gt_v.dim()))).float()
            if est.get("l1_div") is None:
                hist = est.get("l1_hist", [])
                per = sum(x for x, _ in hist) / max(sum(m for _, m in hist), 1) if hist else cnt / len(aux_loc)
                est["l1_div"] = max(per * n_aux_total, 1.0)
            Lg["aux_key_l1"] = (F.l1_loss(pred_states, gt_s, reduction="none") * mask).sum() / est["l1_div"]
        else:
            z = pl["proposals"].sum() * 0
            for k in ("aux_key_ce", "aux_key_l1", "aux_area", "aux_map"):
                Lg[k] = z
        total_g = sum(W[k] * v for k, v in Lg.items())
        with RF("bwd.autograd_backward"):
            total_g.backward()
        for k, v in Lg.items():
            Ls[k] = Ls.get(k, 0.0) + v.detach()
        for k, v in c["parts"].items():
            parts_tot[k] = parts_tot.get(k, 0.0) + v
        got_all.update({a + i: v for i, v in loc.items()})
        errs_all.update({k: v for k, v in errs.items()})
        c.clear()
        T["loss_bwd"] += time.time() - t

    nG = len(bounds)
    if args.inflight >= 2:
        c = enc(0)
        heads(c)
        finish(c)
        prev = c
        for gi in range(1, nG):
            c = enc(gi)
            backward(prev)
            heads(c)
            finish(c)
            prev = c
        backward(prev)
    else:
        for gi in range(nG):
            c = enc(gi)
            heads(c)
            finish(c)
            backward(c)
    t = time.time()
    keys = list(Ls)
    vals = torch.stack([Ls[k].float() for k in keys] + [torch.as_tensor(parts_tot[k], device=dev).float() for k in parts_tot]).tolist()
    Ls = dict(zip(keys, vals[:len(keys)]))
    parts_tot = dict(zip(list(parts_tot), vals[len(keys):]))
    T["tail"] += time.time() - t
    T["fwd"] = T["enc"] + T["heads"] + T["finish"]
    info = dict(rank_divisor=D_rank, rank_live_only=rank_live_only, aux_l1_divisor=est.pop("l1_div", None), aux_l1_count=l1_true,
                aux_samples=n_aux_total, aux_masked_keys=n_masked_keys)
    if n_aux_total:
        est["l1_hist"] = (est.get("l1_hist", []) + [(float(l1_true), n_aux_total)])[-200:]
    return Ls, parts_tot, got_all, errs_all, info


def main():
    args = parse()
    out = RUNS / args.name
    assert not out.exists(), "never overwrite a run"
    out.mkdir(parents=True)
    json.dump(vars(args), open(out / "args.json", "x"), indent=1)
    t0 = time.time()
    lab = make_labeller(args.lab_nav, args.lab_carla, args.gpu_labels)  # spawned before CUDA
    print(f"labellers ready {time.time() - t0:.0f} s", flush=True)
    dev = torch.device("cuda")
    import data as D
    import losses as L
    from navsim.agents.drive_jepa_run3a.drive_jepa_agent import DriveJEPAAgent
    from phase1.loss.criterion import Phase1Criterion
    from phase1.model.encoder import normalise
    net, CFG = build(args, dev)
    crit = Phase1Criterion(CFG)
    ds = D.Run3BData("train", CFG, seed=args.seed)
    ds_h = D.Run3BData("heldout", CFG, seed=12345)
    pass_steps = int(np.ceil((len(ds.nav) + len(ds.lead)) / 64))  # one pass = the training samples / batch (expected count)
    if args.steps == 0:
        args.steps = args.passes * pass_steps
    start = 0
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        start = int(ck["step"])
    sampler = D.StepSampler(ds, args.steps, seed=args.seed, start=start)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, collate_fn=D.collate, prefetch_factor=1,
                                     persistent_workers=True, pin_memory=False)
    groups = param_groups(net, args)
    opt = torch.optim.Adam(groups)
    W = dict(planner=1.0, det_cls=2.0, det_box=1.0, det_vel=0.25, aux_key_ce=1.0, aux_key_l1=0.1, aux_area=2.0, aux_map=1.0,
             metres_head=1.0, ranking=args.rank_weight, **{f"fact_{k}": 1.0 for k in L.YESNO + L.NUMBERS})
    json.dump(dict(weights=W, n_train_nav=len(ds.nav), n_train_lead=len(ds.lead), nav_missing=ds.nav_missing, pass_steps=pass_steps,
                   steps=args.steps, max_hours=args.max_hours, n_heldout_nav=len(ds_h.nav), n_heldout_lead=len(ds_h.lead),
                   stop_rules=dict(nan="non-finite total loss", label_errors="any live-labelling error", loader="mean loader wait over the "
                                   "last 200 steps > 20 % of the mean step time (from step 250)", not_falling="at the smoke step and each pass "
                                   "end: mean total of the last 200 steps >= mean of steps 50-249"),
                   trainable={g["name"]: sum(p.numel() for p in g["params"]) for g in groups},
                   draw=dict(per=D.DRAW_PER, cap=D.DRAW_CAP), rank=dict(margin=L.RANK_MARGIN, pairs=L.RANK_PAIRS),
                   grouped=dict(group=args.group, inflight=args.inflight), resumed_from=args.resume or None, start_step=start),
              open(out / "setup.json", "x"), indent=1)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    est = {}
    hist_total, hist_wait, n_evals, t_run0 = [], [], 0, time.time()
    if ck is not None:
        named = dict(net.named_parameters())
        for n, v in ck["trainable"].items():
            named[n].data.copy_(v.to(dev))
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        hist_total, hist_wait, n_evals = list(ck["hist_total"]), [tuple(x) for x in ck["hist_wait"]], int(ck["n_evals"])
        est = dict(ck.get("est", {}))
        tl_old = Path(args.resume).parent / "train_log.jsonl"
        if "l1_hist" not in est and tl_old.exists():  # runs saved before the running-mean divisor: seed it from their log
            recs = [json.loads(x) for x in open(tl_old)][-200:]
            est = dict(l1_hist=[(float(r["aux_l1_count"]), 32) for r in recs if "aux_l1_count" in r])  # 32 NAVSIM per batch
        t_run0 = time.time() - 3600 * float(ck["hours"])
        print(f"resumed from {args.resume} at step {start} ({ck['hours']:.2f} h)", flush=True)
        del ck
    lab.prep(start, [ds.label_meta(g) for g in sampler.batch(start)])
    logf = open(out / "train_log.jsonl", "a")
    net.train()
    import threading
    from collections import deque
    threading.Thread(target=_util_loop, daemon=True).start()
    it = iter(dl)
    prof_steps = sorted(int(x) for x in args.profile_steps.split(",") if x.strip())
    prof = None
    for step in range(start, args.steps):
        if prof_steps and step == prof_steps[0] and prof is None:
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
            prof.__enter__()
            t_prof0 = time.time()
        T = dict(loader_wait=0.0, fwd=0.0, label_wait=0.0, loss_bwd=0.0)
        t_step0 = time.time()
        t = time.time()
        batch = next(it)
        T["loader_wait"] = time.time() - t
        metas = batch["meta"]
        Bn = batch["images"].shape[0]
        t = time.time()
        metas_next = [ds.label_meta(g) for g in sampler.batch(step + 1)] if step + 1 < args.steps else None
        T["meta_next"] = time.time() - t
        Ls, parts_tot, got_all, errs_all, info = grouped_step(net, crit, lab, batch, step, args, W, gen, dev, metas_next, T, est)
        t = time.time()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        T["opt"] = time.time() - t
        T["step"] = time.time() - t_step0
        errs = errs_all
        got = got_all
        total = sum(W[k] * v for k, v in Ls.items())
        rec = dict(step=step, time=time.time(), losses=Ls, share={k: W[k] * v / total for k, v in Ls.items()}, total=total,
                   timing=T, label_cpu_s={f"{nm}_{k}": float(sum(got[i]["seconds"][k] for i in got if (metas[i]["ds"] == 0) == (nm == "nav")))
                                          for nm in ("nav", "lead") for k in ("context", "label")},
                   label_errors=len(errs), label_missing=int(Bn - len(got)), planner_rounds=parts_tot,
                   **info,
                   gpu_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, gpu_util=float(np.mean(UTIL)) if UTIL else None,
                   samples_per_s=Bn / T["step"], n_drawn_mean=float(batch["n_drawn"].float().mean()),
                   teachers=[int(x.shape[0]) for x in batch["teachers"]],
                   views_from_archive=int(sum(m.get("views_from_archive", 0) for m in metas)))
        if errs:
            rec["error_example"] = next(iter(errs.values()))[-600:]
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if step % args.log_every == 0:
            print(f"step {step} total {rec['total']:.3f} step_s {T['step']:.1f} (setup {T.get('setup', 0):.1f} enc {T.get('enc', 0):.1f} "
                  f"heads {T.get('heads', 0):.1f} finish {T.get('finish', 0):.1f} wait {T['label_wait']:.1f} loss+bwd {T['loss_bwd']:.1f} "
                  f"tail {T.get('tail', 0):.1f} opt {T.get('opt', 0):.1f} loader {T['loader_wait']:.1f}) mem {rec['gpu_mem_gb']:.1f} GB "
                  f"lab_err {len(errs)}", flush=True)
        if prof is not None and step == prof_steps[-1]:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            wall = time.time() - t_prof0
            ka = prof.key_averages()
            (out / "profile_table_device.txt").write_text(ka.table(sort_by="device_time_total", row_limit=80))
            (out / "profile_table_cpu.txt").write_text(ka.table(sort_by="cpu_time_total", row_limit=80))
            rows = [dict(name=e.key, calls=e.count, cpu_total_ms=e.cpu_time_total / 1e3, cpu_self_ms=e.self_cpu_time_total / 1e3,
                         device_total_ms=getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e3,
                         device_self_ms=getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) / 1e3) for e in ka]
            json.dump(dict(steps=prof_steps, wall_s=wall, rows=rows), open(out / "profile_summary.json", "w"), indent=1)
            try:
                prof.export_chrome_trace(str(out / "profile_trace.json"))
            except Exception as e:  # noqa: BLE001
                print("trace export failed", e, flush=True)
            print(f"PROFILE steps {prof_steps} done ({wall:.1f} s): profile_*.txt / profile_summary.json in {out}", flush=True)
            prof = None
        torch.cuda.reset_peak_memory_stats()
        UTIL.clear()
        hist_total.append(rec["total"])
        hist_wait.append((T["loader_wait"], T["step"]))
        stop = None
        if not np.isfinite(rec["total"]):
            stop = f"non-finite total loss at step {step}"
        elif errs:
            stop = f"{len(errs)} live-labelling errors at step {step}"
        elif step >= 250 and np.mean([w for w, _ in hist_wait[-200:]]) > 0.2 * np.mean([t_ for _, t_ in hist_wait[-200:]]):
            stop = f"loader wait {np.mean([w for w, _ in hist_wait[-200:]]):.2f} s > 20 % of step {np.mean([t_ for _, t_ in hist_wait[-200:]]):.2f} s"
        hours = (time.time() - t_run0) / 3600
        tags = []
        if step + 1 == args.smoke_step:
            tags.append("smoke")
        if (step + 1) % pass_steps == 0:
            tags.append(f"pass{(step + 1) // pass_steps}")
        if hours >= args.max_hours and not stop:
            stop = f"time limit {args.max_hours} h at step {step + 1}"
        if step + 1 == args.steps and not tags:
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
                import evaluate as EV
                te = time.time()
                res = EV.run_eval(net, lab, ds_h, args.P, 32, args.eval_n, args.eval_n, step_base=-(10 ** 6) * (n_evals + 1))
                n_evals += 1
                res.update(step=step + 1, tag=tag, ckpt=str(dst), train_hours=round(hours, 2), eval_minutes=round((time.time() - te) / 60, 1))
                json.dump(res, open(out / f"eval_step{step + 1:06d}_{tag}.json", "x"), indent=1)
                print(f"EVAL {tag} step {step + 1} done in {(time.time() - te) / 60:.1f} min", flush=True)
                lab.prep(step + 1, [ds.label_meta(g) for g in sampler.batch(step + 1)]) if step + 1 < args.steps else None
            if ("smoke" in tags or any(t_.startswith("pass") for t_ in tags)) and len(hist_total) >= 450 and not stop:
                if np.mean(hist_total[-200:]) >= np.mean(hist_total[50:250]):
                    stop = f"total loss not falling at step {step + 1}: last 200 {np.mean(hist_total[-200:]):.3f} >= steps 50-249 {np.mean(hist_total[50:250]):.3f}"
        if (step + 1) % args.resume_every == 0 or stop:
            rs = dict(trainable={n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad}, opt=opt.state_dict(),
                      step=step + 1, gen=gen.get_state(), hist_total=hist_total, hist_wait=hist_wait, n_evals=n_evals, est=est,
                      hours=(time.time() - t_run0) / 3600, args=vars(args))
            torch.save(rs, str(out / "resume_latest.pt") + ".tmp")
            os.replace(str(out / "resume_latest.pt") + ".tmp", out / "resume_latest.pt")
        if stop:
            (out / "STOPPED").write_text(stop + "\n")
            print("STOPPED:", stop, flush=True)
            break
    else:
        (out / "TRAIN_DONE").write_text(str(args.steps))
    lab.close()


if __name__ == "__main__":
    main()
