"""run 3B held-out evaluation (smoke-report items 3-7). Fixed subsets: N_NAV NAVSIM held-out samples (the 51 logs) and N_LEAD
LEAD held-out frames (the 416 routes), evenly spaced, the same draw every time (deterministic). The planner's 32 are labelled
live (labeller.py) for items 4, 6 and 7.
  3  road users: run 3A's evaluation rule (lead_inputs_v1/part_c_eval.py, agent-decoder/common.py), Phase 1's five classes;
     NAVSIM truth = Phase 1's targets (emergency detections count as vehicle, Phase 1's merge); LEAD truth 'as Bench2Drive' and
     'camera-visible' (also in a view with LEAD visible_pixels > 0).
  4  facts: within-scene pairwise accuracy per yes/no head (pairs of paths of one scene whose path-level labels differ: the
     share the head orders correctly; path label = the fact at some step, path prediction = the max over steps of the
     probability; averaged over scenes with contrast; red: paths of unknown colour left out); on all scored paths and on the
     planner's 32 alone. Number heads: mean absolute error per waypoint (closeness capped 4 s, cut-in braking capped 8).
  5  expert's-metres head: absolute error at 2 s (step 4) and 4 s (step 8); LEAD split by 'expert stops' (the expert row
     gains < 0.25 m in its last 0.5 s step) against the rest.
  6  the pick = the planner's proposal with the highest score from predicted facts (with the true expert metres, and with
     the head's): its true score (live facts, true expert metres) against the best true score of the 32; share of samples
     whose pick is a true top (within 1e-6); share of picks that stop (< 0.5 m over 4 s) while the expert moves (>= 0.5 m).
  7  planner: distance of the nearest proposal to the expert path (mean over the 8 waypoints of the xy distance); spread of
     the 32 (mean pairwise distance, same measure).
    cd {runs}/run3b/r3b && $PY evaluate.py --ckpt init|PATH --out NAME.json"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), "{runs}/run3a-adapted", "{data.drivejepa_tree}/navsim_v2",
                "{runs}/run3a/lead_inputs_v1"]
OUT = Path("{data.paddock_data}/run3b/eval")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-nav", type=int, default=1008)
    ap.add_argument("--n-lead", type=int, default=1008)
    ap.add_argument("--batch", type=int, default=32)  # batch x 3 cameras must divide by mmcv im2col_step 32
    ap.add_argument("--lab-nav", type=int, default=8)
    ap.add_argument("--lab-carla", type=int, default=12)
    ap.add_argument("--reader-layers", type=int, default=2)
    ap.add_argument("--P", type=float, default=0.108)
    ap.add_argument("--compile", type=int, default=0)
    return ap.parse_args()


def pairwise(lab, pred, groups):
    """lab bool (n,), pred (n,), groups (n,) scene ids -> (mean over scenes with contrast, scenes, pairs)."""
    accs, npairs = [], 0
    for g in np.unique(groups):
        m = groups == g
        l, p = lab[m], pred[m]
        pos, neg = p[l], p[~l]
        if len(pos) == 0 or len(neg) == 0:
            continue
        cmp = (pos[:, None] > neg[None]).astype(float) + 0.5 * (pos[:, None] == neg[None])
        accs.append(cmp.mean())
        npairs += cmp.size
    return (float(np.mean(accs)) if accs else None), len(accs), npairs


def main():
    args = parse()
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / args.out
    assert not dst.exists()
    from labeller import Labeller
    lab = Labeller(args.lab_nav, args.lab_carla)
    dev = torch.device("cuda")
    import data as D
    import train as TR
    targs = argparse.Namespace(seed=0, reader_layers=args.reader_layers, scorer_grad_stop=False, compile=args.compile)
    net, CFG = TR.build(targs, dev)
    if args.ckpt != "init":
        st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        named = dict(net.named_parameters())
        with torch.no_grad():
            for k, v in st["trainable"].items():
                named[k].copy_(v)
    ds = D.Run3BData("heldout", CFG, seed=12345)
    res = run_eval(net, lab, ds, args.P, args.batch, args.n_nav, args.n_lead)
    lab.close()
    res["ckpt"] = args.ckpt
    json.dump(res, open(dst, "x"), indent=1)
    print("EVAL_DONE", dst, flush=True)


def run_eval(net, lab, ds, P_pen, batch=32, n_nav=1008, n_lead=1008, step_base=-10 ** 6):
    """the evaluation on a built model and a running labeller (also called from train.py between steps)."""
    import data as D  # noqa: F401
    import losses as L
    import part_c_eval as PE
    import train as TR
    from phase1.model.encoder import normalise
    args = argparse.Namespace(P=P_pen, batch=batch, n_nav=n_nav, n_lead=n_lead)
    dev = torch.device("cuda")
    was_training = net.training
    net.eval()
    ds.deterministic = True
    nav_i = np.linspace(0, len(ds.nav) - 1, min(args.n_nav, len(ds.nav))).round().astype(int)
    lead_i = np.linspace(0, len(ds.lead) - 1, min(args.n_lead, len(ds.lead))).round().astype(int) + len(ds.nav)
    order = np.r_[nav_i, lead_i]
    names = PE.CLS5
    tal = {"NAVSIM": PE.Tally(names), "LEAD as Bench2Drive": PE.Tally(names), "LEAD camera-visible": PE.Tally(names)}
    R = defaultdict(list)
    t0 = time.time()
    step = 0
    chunks = [[int(g) for g in order[s:s + args.batch]] for s in range(0, len(order), args.batch)]
    lab.prep(step_base, [ds.label_meta(g) for g in chunks[0]])
    for step, idx in enumerate(chunks):
        s = step * args.batch
        if step + 1 < len(chunks):
            lab.prep(step_base + step + 1, [ds.label_meta(g) for g in chunks[step + 1]])
        samples = [ds[g] for g in idx]
        batch = D.collate(samples)
        metas = [ds.label_meta(g) for g in idx]
        Bn = len(idx)
        l2i = batch["lidar2img"].to(dev)
        is_nav = (batch["ds"] == 0).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tok = net.encoder(normalise(batch["images"].to(dev)))
            agent_outs, pl, agents, tl = net.heads(tok, l2i, batch["ego_status"].to(dev))
            props = pl["proposals"].float()
            lab.label(step_base + step, metas, props.cpu().numpy())
            paths = torch.cat([batch["paths"].to(dev), props], 1)
            logits, numbers = net.read(paths, tl, l2i, agents)
        got, errs = lab.collect(step_base + step, Bn)
        R["label_errors"].append(len(errs))
        lt, ok, la, lb = TR.live_targets(got, list(range(Bn)), batch["R"].numpy(), batch["ntg"].numpy(), (batch["ds"] == 0).numpy(), args.P)
        logits, numbers = logits.float(), numbers.float()
        # 3 road users
        last = agent_outs[-1]
        p = last["cls"].float().softmax(-1)
        conf, cls, xy = (1 - p[..., 5]).cpu().numpy(), p[..., :5].argmax(-1).cpu().numpy(), last["box"][..., :2].float().cpu().numpy()
        for i, g in enumerate(idx):
            t = batch["targets"][i]
            gxy, gc = t["agent_box"][:, :2].numpy(), t["agent_cls"].numpy()
            if batch["ds"][i] == 0:
                PE.score(conf[i], cls[i], xy[i], gxy, gc, names, [""], tal["NAVSIM"], merge=True)
            else:
                m = ds.lead_pack.meta[ds.lead[g - len(ds.nav)]]
                rec = ds._route(m["v2_file"])[m["v2_i"]]
                Bx = rec["boxes"]
                base = Bx["in_range"] & ~Bx["underground"] & (Bx["cls5"] >= 0)
                vis = base & Bx["in_views"] & (Bx["vis_px"] > 0)
                PE.score(conf[i], cls[i], xy[i], Bx["box"][base, :2], Bx["cls5"][base].astype(np.int64), names, [""], tal["LEAD as Bench2Drive"])
                PE.score(conf[i], cls[i], xy[i], Bx["box"][vis, :2], Bx["cls5"][vis].astype(np.int64), names, [""], tal["LEAD camera-visible"])
        # 4 facts (labels: loader rows + live rows)
        tgt = {k: np.concatenate([batch["facts"][k].numpy(), lt[k]], 1) for k in lt}
        valid = np.concatenate([batch["valid"].numpy(), np.repeat(ok[:, None], 32, 1)], 1)
        prob = torch.sigmoid(logits).cpu().numpy()
        num = numbers.cpu().numpy()
        planner_col = np.zeros(valid.shape, bool)
        planner_col[:, -32:] = True
        for i, g in enumerate(idx):
            dsn = "NAVSIM" if batch["ds"][i] == 0 else "LEAD"
            v = valid[i]
            for j, k in enumerate(L.YESNO):
                if k == "comfort" and dsn == "LEAD":
                    continue
                labp = tgt[k][i].astype(bool).any(-1)
                keep = v.copy()
                if k == "red":
                    keep &= tgt["red_known"][i].all(-1)
                R[f"{dsn}|{k}|lab"].append(labp[keep])
                R[f"{dsn}|{k}|pred"].append(prob[i, :, :, j].max(-1)[keep])
                R[f"{dsn}|{k}|grp"].append(np.full(int(keep.sum()), g))
                R[f"{dsn}|{k}|planner"].append(planner_col[i][keep])
            for j, k in enumerate(L.NUMBERS):
                e = np.abs(num[i, :, :, j] - tgt[k][i])[v]
                if k == "ttc":
                    e = np.abs(np.clip(num[i, :, :, j], 0, 4) - tgt[k][i])[v]
                if k == "cutin_brake":
                    e = np.abs(np.clip(num[i, :, :, j], 0, 8) - tgt[k][i])[v]
                R[f"{dsn}|{k}|abs_err"].append(e.reshape(-1))
        # 5 expert metres head
        em = pl["expert_metres"].float().cpu().numpy()
        Rt = batch["R"].numpy()
        for i in range(Bn):
            dsn = "NAVSIM" if batch["ds"][i] == 0 else "LEAD"
            stops = bool(Rt[i, 7] - Rt[i, 6] < 0.25)
            R[f"{dsn}|metres|err2"].append(abs(em[i, 3] - Rt[i, 3]))
            R[f"{dsn}|metres|err4"].append(abs(em[i, 7] - Rt[i, 7]))
            R[f"{dsn}|metres|stops"].append(stops)
        # 6 pick
        for which, Ruse in (("true", batch["R"].to(dev)), ("head", pl["expert_metres"].float())):
            ntg = batch["ntg"].to(dev) if which == "true" else (Ruse[:, -1] < 0.5)
            sc = L.scores_from_pred(logits[:, -32:], numbers[:, -32:], Ruse, ntg, is_nav, args.P).cpu().numpy()
            pick = sc.argmax(1)
            true = la - args.P * lb
            for i in range(Bn):
                if not ok[i]:
                    continue
                dsn = "NAVSIM" if batch["ds"][i] == 0 else "LEAD"
                best = true[i].max()
                R[f"{dsn}|pick_{which}|gap"].append(best - true[i, pick[i]])
                R[f"{dsn}|pick_{which}|top"].append(true[i, pick[i]] >= best - 1e-6)
                pm = lt["metres_sofar"][i, pick[i], -1]
                R[f"{dsn}|pick_{which}|stops_expert_moves"].append(bool(pm < 0.5 and Rt[i, -1] >= 0.5))
                R[f"{dsn}|pick_{which}|expert_moves"].append(bool(Rt[i, -1] >= 0.5))
        # 7 planner
        pr = props.cpu().numpy()
        ex = batch["expert"].numpy()
        for i in range(Bn):
            dsn = "NAVSIM" if batch["ds"][i] == 0 else "LEAD"
            d = np.linalg.norm(pr[i, :, :, :2] - ex[i, None, :, :2], axis=-1).mean(-1)
            R[f"{dsn}|planner|min_dist"].append(d.min())
            pd = np.linalg.norm(pr[i, :, None, :, :2] - pr[i, None, :, :, :2], axis=-1).mean(-1)
            R[f"{dsn}|planner|spread"].append(pd[np.triu_indices(32, 1)].mean())
        if (step + 1) % 10 == 0:
            print(f"{s + Bn}/{len(order)} samples, {(time.time() - t0) / 60:.1f} min", flush=True)
    res = dict(n_nav=int(len(nav_i)), n_lead=int(len(lead_i)), P=args.P, label_errors=int(sum(R["label_errors"])),
               minutes=round((time.time() - t0) / 60, 1), road_users={k: v.out() for k, v in tal.items()})
    facts = {}
    for dsn in ("NAVSIM", "LEAD"):
        for k in L.YESNO:
            if f"{dsn}|{k}|lab" not in R:
                continue
            lab_, pred, grp, plc = (np.concatenate(R[f"{dsn}|{k}|{x}"]) for x in ("lab", "pred", "grp", "planner"))
            a_all = pairwise(lab_, pred, grp)
            a_32 = pairwise(lab_[plc], pred[plc], grp[plc])
            facts[f"{dsn} | {k}"] = dict(pairwise_all=a_all[0], scenes_all=a_all[1], pairs_all=a_all[2], pairwise_planner32=a_32[0],
                                         scenes_planner32=a_32[1], pairs_planner32=a_32[2], base_rate=float(lab_.mean()), n_paths=int(len(lab_)))
        for k in L.NUMBERS:
            e = np.concatenate(R[f"{dsn}|{k}|abs_err"])
            facts[f"{dsn} | {k} (abs err)"] = dict(mean=float(e.mean()), p50=float(np.median(e)), p90=float(np.percentile(e, 90)), n=int(len(e)))
    res["facts"] = facts
    met = {}
    for dsn in ("NAVSIM", "LEAD"):
        e2, e4, st = (np.asarray(R[f"{dsn}|metres|{x}"]) for x in ("err2", "err4", "stops"))
        met[dsn] = dict(err_2s_mean=float(e2.mean()), err_4s_mean=float(e4.mean()), err_4s_p90=float(np.percentile(e4, 90)), n=int(len(e2)))
        if dsn == "LEAD":
            for nm, m in (("expert stops", st), ("rest", ~st)):
                met[f"LEAD {nm}"] = dict(err_2s_mean=float(e2[m].mean()) if m.any() else None, err_4s_mean=float(e4[m].mean()) if m.any() else None, n=int(m.sum()))
    res["expert_metres_head"] = met
    pick = {}
    for dsn in ("NAVSIM", "LEAD"):
        for which in ("true", "head"):
            gap, top, se, em_ = (np.asarray(R[f"{dsn}|pick_{which}|{x}"]) for x in ("gap", "top", "stops_expert_moves", "expert_moves"))
            pick[f"{dsn} | expert metres {which}"] = dict(mean_gap_to_best=float(gap.mean()), p90_gap=float(np.percentile(gap, 90)),
                                                         share_true_top=float(top.mean()), share_stops_while_expert_moves=float(se.sum() / max(em_.sum(), 1)),
                                                         n=int(len(gap)), n_expert_moves=int(em_.sum()))
    res["pick"] = pick
    res["planner"] = {dsn: dict(min_dist_mean=float(np.mean(R[f"{dsn}|planner|min_dist"])), min_dist_p50=float(np.median(R[f"{dsn}|planner|min_dist"])),
                                min_dist_p90=float(np.percentile(R[f"{dsn}|planner|min_dist"], 90)), spread_mean=float(np.mean(R[f"{dsn}|planner|spread"])),
                                n=len(R[f"{dsn}|planner|min_dist"])) for dsn in ("NAVSIM", "LEAD")}
    if was_training:
        net.train()
    return res


if __name__ == "__main__":
    main()
