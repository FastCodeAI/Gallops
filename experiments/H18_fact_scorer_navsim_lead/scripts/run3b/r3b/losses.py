"""run 3B losses (run3b/CLAUDE.md section 3 + Navvrat's 18 Nov answers). Every term is returned separately with its weight.

planner      run 3A's trajectory_loss_anchors (drive_jepa_agent.py), both datasets: per refinement round, min over the 32 of the
             mean-over-steps L1 (x, y, heading) to the expert path, + 0.5 x (per sample: mean over its <= 4 teachers of the
             min over the 32; summed over samples with teachers, / batch); rounds folded as 0.1 x previous + this round.
             Teachers: NAVSIM run 3A's lists (anchors_scores_index_v2_rearok_lane, 4 drawn); LEAD the pool's top-4 clean rows.
navsim aux   run 3A's score_loss (key agents: BCE validity x1, corner L1 x0.1; ego areas BCE x2) + map aux smooth L1 x1, on
             the NAVSIM samples, labels from run 3A's get_sub_score on the planner's 32 (live).
detection    Phase 1's agent loss (phase1/loss/criterion.py: Hungarian on class + centre, set focal on the class, L1 box, L1
             velocity; the 3 layers averaged; / agents in the batch), weights 2 / 1 / 0.25, both datasets, 5 classes, 32 m.
facts        per yes/no head: BCE per waypoint; per number head: smooth L1 per waypoint; on the expert + the balanced draw +
             the planner's 32; masks: padding, red of unknown colour, comfort on LEAD. Each head its own term, weight 1.
metres head  L1 to the expert's metres so far (8 numbers).
ranking      on the score computed from the predicted facts (score.py, true expert metres): pairs of paths of one sample whose
             true scores differ by > 0.02; logistic loss on the predicted difference; RANK_PAIRS pairs per sample drawn with
             probability proportional to the higher path's true-score percentile in its sample (KNOB); weight 0.1."""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "{runs}/run3a-adapted")
from phase1 import constants as K  # noqa: E402
from phase1.loss.criterion import _to_dev, norm_box8, pred_box_from_target  # noqa: E402

from score import score_torch  # noqa: E402

RANK_MARGIN, RANK_PAIRS = 0.02, 512
YESNO = ("hit", "offroad", "left_route", "red", "stop", "wrong_way", "comfort")
NUMBERS = ("metres_sofar", "lane_off", "ttc", "cutin_brake")


class AgentPack:
    """the agent part of Phase 1's criterion.Pack (phase1/loss/criterion.py Pack.__init__, agent lines, copied exactly);
    Pack itself also packs the map labels, which run 3B has none of."""

    def __init__(self, targets, dev, mc):
        cpu = lambda x: x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)  # noqa: E731
        self.B = len(targets)
        self.datasets = [int(t["dataset"]) for t in targets]
        na = [len(t["agent_cls"]) for t in targets]
        self.na = na
        self.a_off = np.r_[0, np.cumsum(na)].astype(np.int64)
        a_cls = torch.cat([cpu(t["agent_cls"]).long().reshape(-1) for t in targets])
        a_box = torch.cat([cpu(t["agent_box"]).float().reshape(-1, 7) for t in targets])
        a_vel = torch.cat([cpu(t["agent_vel"]).float().reshape(-1, 2) for t in targets])
        a_ds = np.repeat(np.asarray(self.datasets, np.int64), na)
        C = K.AGENT_NOTHING + 1
        aidx = np.full((len(a_cls), 2), C, np.int64)
        aidx[:, 0] = a_cls.numpy()
        for i, (c, d) in enumerate(zip(a_cls.tolist(), a_ds.tolist())):
            extra = K.AGENT_MERGE[d].get(c)
            if extra:
                aidx[i, :len(extra)] = extra
        self.a_aidx = torch.as_tensor(aidx).to(dev)
        self.a_ctr = a_box[:, :3].to(dev)
        self.a_box8n = norm_box8(pred_box_from_target(a_box), mc).to(dev)
        self.a_vel_n = (a_vel / mc["agent_vel_scale"]).to(dev)


def detection(crit, agent_outs, targets, dev):
    P = AgentPack(targets, dev, crit.mc)
    n = float(sum(P.na))
    sums = {"agent_cls": 0.0, "agent_box": 0.0, "agent_vel": 0.0}
    for o in agent_outs:
        m = crit.match_agents_all(o, P)
        idx = _to_dev(list(crit._flat_matches(m, P.a_off)), dev)
        for k, v in crit.agent_sums(o, P, idx).items():
            sums[k] = sums[k] + v / len(agent_outs)
    return {"det_cls": sums["agent_cls"] / max(n, 1.0), "det_box": sums["agent_box"] / max(8 * n, 1.0),
            "det_vel": sums["agent_vel"] / max(2 * n, 1.0)}


def planner(proposal_list, expert, teachers, prev_weight=0.1):
    """run 3A trajectory_loss_anchors with the teachers given per sample (list of (K_b, 8, 3) tensors)."""
    B = expert.shape[0]
    traj, parts = 0.0, {}
    for r, props in enumerate(proposal_list):
        props = props.float()
        min_loss = torch.linalg.norm(props - expert[:, None].float(), dim=-1, ord=1).mean(-1).amin(1).mean()
        pseudo = props.new_zeros(())
        for b in range(B):
            t = teachers[b]
            if t.shape[0] == 0:
                continue
            d = props[b, None] - t[:, None].to(props)                      # (K, P, T, 3)
            pseudo = pseudo + torch.linalg.norm(d, dim=-1, ord=1).mean(-1).amin(-1).mean()
        min_loss = min_loss + 0.5 * pseudo / B
        traj = prev_weight * traj + min_loss
        parts[f"round{r}"] = float(min_loss.detach())
    return traj, parts


def navsim_aux(pl, nav_idx, aux, run3a_score_loss):
    """run 3A's auxiliary heads on the NAVSIM samples. aux: per NAVSIM sample (in nav_idx order) the labeller's dict."""
    if len(nav_idx) == 0:
        z = pl["proposals"].sum() * 0
        return {"aux_key_ce": z, "aux_key_l1": z, "aux_area": z, "aux_map": z}
    dev = pl["proposals"].device
    gt_states = torch.as_tensor(np.stack([a["key_corners"] for a in aux]), device=dev)
    gt_valid = torch.as_tensor(np.stack([a["key_valid"] for a in aux]), device=dev)
    gt_areas = torch.as_tensor(np.stack([a["ego_areas"] for a in aux]), device=dev)
    gt_map = torch.as_tensor(np.stack([a["map_aux"] for a in aux]), device=dev)
    ni = torch.as_tensor(nav_idx, device=dev)
    ce, l1, area = run3a_score_loss(None, None, pl["pred_agents_states"][ni].float(), pl["pred_area_logit"][ni].float(), None,
                                    gt_states, gt_valid, gt_areas)
    return {"aux_key_ce": ce, "aux_key_l1": l1, "aux_area": area, "aux_map": F.smooth_l1_loss(pl["pred_map_aux"][ni].float(), gt_map)}


def facts(logits, numbers, tgt, valid, is_nav):
    """logits (B, P, 8, 7), numbers (B, P, 8, 4); tgt: dict (B, P, 8) tensors; valid (B, P); is_nav (B,)."""
    out = {}
    v = valid[..., None].expand(-1, -1, 8).float()
    for i, k in enumerate(YESNO):
        m = v.clone()
        if k == "red":
            m = m * tgt["red_known"].float()
        if k == "comfort":
            m = m * is_nav[:, None, None].float()
        bce = F.binary_cross_entropy_with_logits(logits[..., i], tgt[k].float(), reduction="none")
        out[f"fact_{k}"] = (bce * m).sum() / m.sum().clamp(min=1.0)
    for i, k in enumerate(NUMBERS):
        sl = F.smooth_l1_loss(numbers[..., i], tgt[k].float(), reduction="none")
        out[f"fact_{k}"] = (sl * v).sum() / v.sum().clamp(min=1.0)
    return out


def predicted_facts(logits, numbers):
    p = torch.sigmoid(logits)
    d = {k: p[..., i] for i, k in enumerate(YESNO)}
    for i, k in enumerate(NUMBERS):
        d[k] = numbers[..., i]
    d["ttc"] = d["ttc"].clamp(0.0, 4.0)
    d["cutin_brake"] = d["cutin_brake"].clamp(0.0, 8.0)
    return d


def ranking(pred_score, true_score, valid, gen):
    """pred_score, true_score (B, P); valid (B, P) bool."""
    B, P = pred_score.shape
    dev = pred_score.device
    tot, n_used = pred_score.new_zeros(()), 0
    for b in range(B):
        idx = torch.where(valid[b])[0]
        if len(idx) < 2:
            continue
        s = true_score[b, idx]
        diff = s[:, None] - s[None, :]                      # hi - lo
        ok = diff > RANK_MARGIN
        if not ok.any():
            continue
        pct = (torch.argsort(torch.argsort(s)).float() + 0.5) / len(s)
        w = (pct[:, None].expand_as(diff) * ok.float()).reshape(-1)
        k = min(RANK_PAIRS, int(ok.sum()))
        pick = torch.multinomial(w, k, replacement=False, generator=gen)
        hi, lo = pick // len(idx), pick % len(idx)
        ps = pred_score[b, idx]
        tot = tot + F.softplus(-(ps[hi] - ps[lo])).mean()
        n_used += 1
    return tot / max(n_used, 1)


def scores_from_pred(logits, numbers, R, ntg, is_nav, P):
    sc, a, b = score_torch(predicted_facts(logits.float(), numbers.float()), R.float(), ntg, is_nav, P=P)
    return sc
