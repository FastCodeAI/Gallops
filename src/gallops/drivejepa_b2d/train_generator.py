"""Stage final-generator: fine-tune Drive-JEPA's proposal generator on Bench2Drive with MTD-H pseudo-teachers.

The loss is Drive-JEPA's own multimodal one -- expert winner-takes-all plus 0.5 x the pseudo-teacher term --
with the pseudo-teachers taken from the frozen variant-H teacher set (up to 4 per frame, built by the MTD
stages 1-3) instead of the scorer's top anchors. Trainable: init_feature (the 32 x 8 learned queries),
hist_encoding (ego 11-vector -> 256) and the ONE shared refiner behind _trajectory_head[0..3]. Frozen: the
backbone (never run here -- its output comes from the feature cache) and the scorer. Init: the released
NAVSIM-v2 checkpoint, loaded as paddock's policy loads it. Adam 2e-5, batch 32, 10 epochs, seed 0; held-out =
the 150 held-out clips (the cache splits them into meta_held_*). The batch fix (batch_fix.py) is applied
unless --no-batch-fix.

Loss per refinement pass (DriveJEPAAgent.trajectory_loss_anchors, inter_weight 0), L1 = sum over (x, y,
heading) averaged over the 8 poses:
  pass_loss = mean_b min_P L1(prop_b, expert_b) + 0.5 / B * sum_b mean_{k < n_b} min_P L1(prop_b, teacher_bk)
  loss = discounted sum over the 4 passes, loss <- prev_weight (0.1) * loss + pass_loss.
Parity: on the first batch the same loss is recomputed through the agent's own trajectory_loss_anchors with
agent.anchors set to the batch's teachers; the run stops if they differ.

Kept epoch = lowest held-out value of the same objective; every epoch's generator is saved too. Held-out also
reports teacher coverage: the share of teachers with a proposal within 1 m / 2 m ADE over 3 s, overall and by
frame tag (M2_t blocked pull-out, M3 turn, M1 merge, N normal).

Runs inside Drive-JEPA's own interpreter (Python 3.9) with its cwd and environment -- the agent loads
./data/8192.npy relative to the cwd -- so this module stays valid Python 3.9. Ported from paddock-ops
mtd/stage4/train_mtd.py, with refine_from_cache and load_split from generator-ft (which imported paddock's
policy at module level; only these two functions were used). Arithmetic unchanged.

**RAM: about 26 GB.** load_split holds every feature chunk, train and held-out, in memory (22 GB of feat_*.npy
for the Final's cache; 26 GB RSS measured on 5 Oct). The stage refuses to start with less available than the
cache plus 4 GB (host.require_free_ram), and two of it at once do not fit beside other work on a 123 GB box.

    python -m gallops.drivejepa_b2d.train_generator --cache <feature cache> --teachers <dir> --checkpoint
        <ckpt>
        --out <dir> [--lr 2e-5 --batch 32 --epochs 10 --teacher-weight 0.5]
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from gallops.drivejepa_b2d.host import OVERHEAD_BYTES, cache_bytes, require_free_ram
from gallops.drivejepa_b2d.provenance import git_sha

TAGS = ("M2_t", "M3", "M1merge", "N")


def load_split(cache: Path, split: str):
    """Every chunk of one split of the feature cache, features in RAM (~0.5 MB/frame): NAS reads per step
    were the bottleneck."""
    parts = []
    for m in sorted(glob.glob(str(cache / f"meta_{split}_*.npz"))):
        z = np.load(m)
        feat = np.load(m.replace("/meta_", "/feat_").replace(".npz", ".npy"))
        assert len(feat) == len(z["sample_id"]), m
        parts.append(dict(feat=feat, **{k: z[k] for k in z.files}))
    return parts


def refine_from_cache(model, feat, l2i, img_shape, ego, spatial, level):
    """The generator part of DriveJEPAModel.forward, fed from cached tensors. feat (B, 512, 256)."""
    dev = model.init_feature.weight.device  # the backbone may sit on the CPU (training never runs it)
    feat_flatten = feat.to(dev).float().unsqueeze(0).permute(0, 2, 1, 3)  # (1, 512, B, 256)
    kwargs = {"img_metas": {"lidar2img": l2i.to(dev).float(), "img_shape": img_shape.to(dev).float()}}
    image_feature = (feat_flatten, spatial.to(dev), level.to(dev), kwargs)
    bev = model.hist_encoding(ego.to(dev).float())[:, None] + model.init_feature.weight[None]
    props = []
    for refine in model._trajectory_head:
        bev, p = refine(bev, image_feature)
        props.append(p)
    return props, bev


def load_teachers(teach: Path):
    """sample_id 'clip/kkkkk' (10 Hz frame index) -> (teachers (4, 8, 3), n, tags, expert)."""
    out = {}
    for f in sorted(teach.glob("*.npz")):
        z = np.load(f)
        rows = zip(z["frame_id"], z["n"], z["teachers"], z["expert"], z["tags"])  # noqa: B905 -- Python 3.9
        for fid, n, t, e, tg in rows:
            _, clip, ms = str(fid).split("|")
            assert int(ms) % 100 == 0, fid
            out[f"{clip}/{int(ms) // 100:05d}"] = (
                t.astype(np.float32),
                int(n),
                str(tg),
                e.astype(np.float32),
            )
    return out


def attach(parts, tz):
    """Add teach (N, 4, 8, 3), tn (N,), tags (N,) to every cache chunk; check the teacher file's expert is the
    cache's target."""
    hit, worst = 0, 0.0
    for p in parts:
        N = len(p["sample_id"])
        p["teach"] = np.zeros((N, 4, 8, 3), np.float32)
        p["tn"] = np.zeros(N, np.int64)
        p["tags"] = np.array([""] * N, dtype=object)
        for i, s in enumerate(p["sample_id"]):
            v = tz.get(str(s))
            if v is None:
                continue
            p["teach"][i], p["tn"][i], p["tags"][i] = v[0], v[1], v[2]
            worst = max(worst, float(np.abs(v[3] - p["target"][i]).max()))
            hit += 1
    assert worst < 1e-2, f"teacher expert != cache target (max {worst})"
    return hit


def mtd_loss(props, target, teach, tn, prev_weight, w=0.5):
    B = target.shape[0]
    kmask = (torch.arange(teach.shape[1], device=teach.device)[None] < tn[:, None]).float()  # (B, 4)
    loss = 0.0
    for p in props:  # (B, P, T, 3)
        wta = (p - target[:, None]).abs().sum(-1).mean(-1).amin(1).mean()
        d = (
            (p[:, None] - teach[:, :, None]).abs().sum(-1).mean(-1).amin(-1)
        )  # (B, 4): best proposal per teacher
        pseudo = ((d * kmask).sum(1) / tn.clamp(min=1)).sum() / B
        loss = prev_weight * loss + wta + w * pseudo
    return loss


def batches_t(parts, batch, rng, group=6):
    """Shuffle chunk order, load `group` chunks at a time, shuffle frames within them (drop_last)."""
    order = rng.permutation(len(parts))
    for g in range(0, len(order), group):
        sel = [parts[i] for i in order[g : g + group]]
        feat = np.concatenate([np.asarray(p["feat"]) for p in sel])
        rest = {
            k: np.concatenate([p[k] for p in sel])
            for k in ("l2i", "img_shape", "ego", "target", "teach", "tn")
        }
        perm = rng.permutation(len(feat))
        for i in range(0, len(perm) - len(perm) % batch, batch):
            ix = perm[i : i + batch]
            yield feat[ix], {k: v[ix] for k, v in rest.items()}


def evaluate(model, parts, spatial, level, prev_weight, batch=64):
    """Held-out objective, expert coverage and teacher coverage of the last pass's 32 proposals."""
    P, T, C, TE, TN, TG = [], [], [], [], [], []
    obj, nobj = 0.0, 0
    with torch.no_grad():
        for p in parts:
            feat = np.asarray(p["feat"])
            for i in range(0, len(feat), batch):
                sl = slice(i, i + batch)
                props, _ = refine_from_cache(
                    model,
                    torch.from_numpy(feat[sl]),
                    torch.from_numpy(p["l2i"][sl]),
                    torch.from_numpy(p["img_shape"][sl]),
                    torch.from_numpy(p["ego"][sl]),
                    spatial,
                    level,
                )
                tgt, te, tn = (torch.from_numpy(p[k][sl]).cuda() for k in ("target", "teach", "tn"))
                obj += float(mtd_loss(props, tgt, te, tn, prev_weight)) * len(tgt)
                nobj += len(tgt)
                P.append(props[-1].float().cpu().numpy())
            for k, L in (("target", T), ("cmd_raw", C), ("teach", TE), ("tn", TN), ("tags", TG)):
                L.append(p[k])
    P, T, C, TE, TN, TG = (np.concatenate(x) for x in (P, T, C, TE, TN, TG))
    d = np.linalg.norm(P[..., :2] - T[:, None, :, :2], axis=-1)
    ade3 = d[..., :6].mean(-1).min(1)
    turn = np.isin(C, [1, 2])
    dt = np.linalg.norm(P[:, None, :, :6, :2] - TE[:, :, None, :6, :2], axis=-1).mean(-1).min(-1)  # (N, 4)
    valid = np.arange(4)[None] < TN[:, None]
    cov = {}
    for name in ("all",) + TAGS:
        if name == "all":
            rows = np.ones(len(TN), bool)
        else:
            rows = np.array([any(t.startswith(name) for t in str(s).split(",")) for s in TG])
        v = valid & rows[:, None]
        if v.any():
            cov[name] = dict(
                n_teachers=int(v.sum()),
                n_frames=int((rows & (TN > 0)).sum()),
                within1m=float((dt[v] <= 1).mean()),
                within2m=float((dt[v] <= 2).mean()),
                frames_all_within2m=float(((dt <= 2) | ~valid)[rows & (TN > 0)].all(1).mean()),
            )
    return dict(
        n=int(len(P)),
        objective=obj / nobj,
        minade3=float(ade3.mean()),
        miss2m_ade3=float((ade3 > 2).mean()),
        miss2m_ade3_turncmd=float((ade3[turn] > 2).mean()) if turn.any() else None,
        lateral_spread3s=float(P[:, :, 5, 1].std(1).mean()),
        teacher_cov=cov,
    )


def line(m):
    c = m["teacher_cov"]
    tc = " ".join(f"{k} {100 * c[k]['within2m']:.1f}%" for k in ("all",) + TAGS if k in c)
    return (
        f"held-out objective {m['objective']:.4f} | expert minADE3 {m['minade3']:.3f} miss@2m "
        f"{100 * m['miss2m_ade3']:.1f}% (turn cmd {100 * (m['miss2m_ade3_turncmd'] or 0):.1f}%) spread3s "
        f"{m['lateral_spread3s']:.2f} | teachers covered <=2m: {tc}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="feature cache (meta_/feat_ <split> chunks)")
    ap.add_argument(
        "--teachers", type=Path, required=True, help="dir of per-clip teacher npz (MTD-H, Bench2Drive)"
    )
    ap.add_argument("--checkpoint", required=True, help="released Drive-JEPA checkpoint to start from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--teacher-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-batch-fix", action="store_true", help="upstream attention (batch > 1 mixes elements)"
    )
    a = ap.parse_args()
    assert a.batch >= 4, (
        "B must be >= ref_num (4): the agent's loss zips the 4 passes with the B entries of scores_index"
    )
    assert not a.out.exists() or not any(a.out.glob("*.pt")), (
        f"{a.out} already has checkpoints -- never overwrite"
    )
    require_free_ram(cache_bytes(a.cache) + OVERHEAD_BYTES, "the generator stage (its whole cache in RAM)")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

    config = DriveJEPAConfig()
    agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=a.checkpoint)  # as paddock's policy.load
    agent.initialize()
    if not a.no_batch_fix:
        from gallops.drivejepa_b2d import batch_fix

        batch_fix.apply_all()
        print(
            "batch fix applied (SpatialCrossAttention per-element masks, TSA value[0::2], encoder ref "
            "points)",
            flush=True,
        )
    model = agent._pad_model
    model._backbone.to("cpu")  # never run here: its output is cached
    for name in ("hist_encoding", "init_feature", "_trajectory_head", "scorer"):
        getattr(model, name).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [model.init_feature, model.hist_encoding, model._trajectory_head[0]]
    for m in trainable:
        for p in m.parameters():
            p.requires_grad_(True)
    params = [p for m in trainable for p in m.parameters()]
    assert all(m is model._trajectory_head[0] for m in model._trajectory_head)
    print(
        f"trainable params {sum(p.numel() for p in params):,}; inter_weight {config.inter_weight}, "
        f"prev_weight "
        f"{config.prev_weight}, teacher weight {a.teacher_weight}",
        flush=True,
    )

    tz = load_teachers(a.teachers)
    train, held = load_split(a.cache, "train"), load_split(a.cache, "held")
    h_tr, h_he = attach(train, tz), attach(held, tz)
    spatial, level = torch.from_numpy(train[0]["spatial"]), torch.from_numpy(train[0]["level"])
    n_train, n_held = sum(len(p["sample_id"]) for p in train), sum(len(p["sample_id"]) for p in held)
    t_tr = int(sum(p["tn"].sum() for p in train))
    print(
        f"teacher frames {len(tz):,}: train {h_tr:,} of {n_train:,} cache frames ({t_tr:,} teachers), "
        f"held-out {h_he:,} of {n_held:,}",
        flush=True,
    )
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=0.0)

    def gen_state():
        keep = ("init_feature", "hist_encoding", "_trajectory_head.0.")
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(keep)}

    curve = []
    t0 = time.perf_counter()
    m0 = evaluate(model, held, spatial, level, config.prev_weight)
    m0.update(epoch=0, train_loss=None)
    curve.append(m0)
    best = (m0["objective"], 0, gen_state())
    print(f"epoch 0 (checkpoint): {line(m0)}", flush=True)
    first = True
    for epoch in range(1, a.epochs + 1):
        tot, n = 0.0, 0
        for feat, b in batches_t(train, a.batch, rng):
            B = len(feat)
            props, _ = refine_from_cache(
                model,
                torch.from_numpy(feat),
                torch.from_numpy(b["l2i"]),
                torch.from_numpy(b["img_shape"]),
                torch.from_numpy(b["ego"]),
                spatial,
                level,
            )
            target, teach, tn = (torch.from_numpy(b[k]).cuda() for k in ("target", "teach", "tn"))
            loss = mtd_loss(props, target, teach, tn, config.prev_weight, a.teacher_weight)
            if (
                first
            ):  # the same number through the agent's own loss, the batch's teachers passed as its anchors
                with torch.no_grad():
                    flat, idx, k0 = [], [], 0
                    for i in range(B):
                        k = int(b["tn"][i])
                        flat.append(b["teach"][i, :k])
                        idx.append(np.arange(k0, k0 + k))
                        k0 += k
                    saved = agent.anchors
                    agent.anchors = np.concatenate(flat)
                    scores_index = idx if a.teacher_weight > 0 else [np.empty(0, dtype=int)] * B
                    ref = agent.trajectory_loss_anchors(props, target, config, scores_index)[0]
                    agent.anchors = saved
                print(
                    f"first batch: loss {float(loss):.5f}, agent's trajectory_loss_anchors {float(ref):.5f} "
                    f"({int(b['tn'].sum())} teachers on {int((b['tn'] > 0).sum())}/{B} frames)",
                    flush=True,
                )
                if a.teacher_weight in (0.0, 0.5):  # the agent's own loss has the teacher weight 0.5 built in
                    assert abs(float(loss) - float(ref)) < 1e-4 * max(1.0, float(ref))
                first = False
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * B
            n += B
        m = evaluate(model, held, spatial, level, config.prev_weight)
        m.update(epoch=epoch, train_loss=tot / n)
        curve.append(m)
        torch.save(gen_state(), a.out / f"gen_epoch{epoch:02d}.pt")
        if m["objective"] < best[0]:
            best = (m["objective"], epoch, gen_state())
        print(
            f"epoch {epoch}: train loss {tot / n:.4f} | {line(m)}  ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
        json.dump(curve, open(a.out / "curve.json", "w"), indent=1)
    torch.save(best[2], a.out / "gen_best.pt")
    metrics = dict(
        best_epoch=best[1],
        best_objective=best[0],
        settings={k: str(v) for k, v in vars(a).items()},
        n_train=n_train,
        n_held=n_held,
        teacher_frames_train=h_tr,
        teacher_frames_held=h_he,
        teachers=str(a.teachers),
        trainable=["init_feature", "hist_encoding", "_trajectory_head (shared refiner)"],
        curve=curve,
        git_sha=git_sha(),
    )
    json.dump(metrics, open(a.out / "metrics.json", "w"), indent=1)
    print(f"kept epoch {best[1]} (held-out objective {best[0]:.4f}) -> {a.out}/gen_best.pt", flush=True)


if __name__ == "__main__":
    main()
