"""Stage ipad-base-start: retrain Drive-JEPA's scorer head so column 5 follows iPad's rule score.

Column 5 is what the released model drives on (`pdm_score = sigmoid(logit)[..., 5]`, pick = argmax). Here it
is fitted, per candidate, to the iPad-style rule score `final` of each of the 32 proposals (BCE, mean over the
32). Inputs are the (32, 256) features the head itself sees, captured once from the released model, so the 32
proposals are fixed and only the pick changes. Base rows plus route-start rows; the held-out clips are
excluded from training and choose the kept epoch (highest mean rule score of the pick).

Init = the released checkpoint's head; the run refuses to start if that head does not reproduce the model's
own pick from the stored features (the features or the head would be wrong). Adam 1e-4, batch 64, 40 epochs,
seed 0.

Ported from paddock-ops scorer-retrain/train_scorer_head.py (run of 29 Sep, target ipad-final). The trainer
had other targets (softmax / WTA over ADE, ipad-dist, ds) and a stopped-frame reweighting; only the path that
produced the model is kept, and its arithmetic is unchanged -- including the all-ones weights in the loss, so
a rerun is bitwise comparable with the recorded head. The held-out clips used to come from a re-ranker
checkpoint directory (--holdout-from), which no longer exists; --holdout-list takes the same 150 clips from a
JSON file.

    python -m gallops.drivejepa_b2d.train_ipad_head --feats '<feats glob>' --dataset <rows.npz> --labels
        <labels.npz>
        --extra-dataset <start rows> --extra-feats '<glob>' --extra-labels <npz> --holdout-list <json>
        --init-ckpt <released ckpt> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from gallops.drivejepa_b2d.head import HEAD_SUFFIXES, head_module, head_tensors_from_checkpoint
from gallops.drivejepa_b2d.provenance import git_sha


def load_feats(pattern: str, order: list[str]):
    """(N, 32, 256) features and the (N, 32, 6) logits the model returned in the same forward, in dataset
    order."""
    by, lg = {}, {}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        for sid, feat, logit in zip(z["sample_id"], z["feats"], z["pred_logit"]):  # noqa: B905 -- one file's arrays
            by[str(sid)] = feat
            lg[str(sid)] = logit
    missing = [s for s in order if s not in by]
    if missing:
        raise SystemExit(f"{len(missing)} dataset rows have no features (e.g. {missing[:3]})")
    return np.stack([by[s] for s in order]).astype(np.float32), np.stack([lg[s] for s in order]).astype(
        np.float32
    )


def pick_ade(head, feats, ade, bs=1024, w=None):
    """Mean ADE (weighted by w when given) of the candidate argmax(column 5) would drive, over a set."""
    picks = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            picks.append(head(feats[i : i + bs])[..., 5].argmax(-1))
    picks = torch.cat(picks)
    v = ade[torch.arange(len(ade)), picks]
    if w is None:
        return float(v.mean()), picks
    return float((w * v).sum() / w.sum()), picks


def load_set(ds_path, feats_pat, labels_path):
    z = np.load(ds_path, allow_pickle=True)
    sids = [str(s) for s in z["sample_id"]]
    scenes = [str(s) for s in z["scene"]]
    ade_np = z["ade"].astype(np.float32)
    model_pick = z["model_pick"] if "model_pick" in z.files else None
    feats_np, logit_np = load_feats(feats_pat, sids)
    L = np.load(labels_path, allow_pickle=True)
    pos = {str(s): i for i, s in enumerate(L["sample_id"])}
    miss = [s for s in sids if s not in pos]
    if miss:
        raise SystemExit(f"{len(miss)} rows have no rule label (e.g. {miss[:3]})")
    o = np.array([pos[s] for s in sids])
    lab = {k: L[k][o][:, :32].astype(np.float32) for k in ("final", "s_ds")}
    return sids, scenes, ade_np, model_pick, feats_np, logit_np, lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--labels", type=Path, required=True, help="rule-label npz (label_candidates.py) with `final`"
    )
    ap.add_argument("--extra-dataset", type=Path, help="second row set to append (route-start rows)")
    ap.add_argument("--extra-feats")
    ap.add_argument("--extra-labels", type=Path)
    ap.add_argument("--holdout-list", type=Path, required=True, help='JSON {"val": [held-out clip names]}')
    ap.add_argument("--init-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=False)

    sids, scenes, ade_np, model_pick, feats_np, logit_np, lab = load_set(a.dataset, a.feats, a.labels)
    n_base = len(sids)
    if a.extra_dataset is not None:
        s2, sc2, ade2, mp2, f2, l2, lab2 = load_set(a.extra_dataset, a.extra_feats, a.extra_labels)
        sids += s2
        scenes += sc2
        ade_np = np.concatenate([ade_np, ade2])
        feats_np = np.concatenate([feats_np, f2])
        logit_np = np.concatenate([logit_np, l2])
        model_pick = np.concatenate([model_pick, mp2]) if model_pick is not None and mp2 is not None else None
        lab = {k: np.concatenate([lab[k], lab2[k]]) for k in lab}
        print(f"appended {len(s2):,} extra rows (route-start) to {n_base:,} base rows", flush=True)
    held_scenes = {str(v).split("/")[-1] for v in json.load(open(a.holdout_list))["val"]}
    held = np.array([s in held_scenes for s in scenes])
    train_idx = np.where(~held)[0]
    held_idx = np.where(held)[0]
    print(
        f"{len(sids):,} frames: train {len(train_idx):,}, held-out {len(held_idx):,} "
        f"({len(held_scenes)} clips); "
        f"feats {feats_np.shape}; target ipad-final",
        flush=True,
    )

    # All-ones weights, kept as in the recorded run: (w * ce).sum() / w.sum() is not bitwise ce.mean().
    w = np.ones(len(sids), dtype=np.float64)
    dev = torch.device(a.device)
    feats = torch.from_numpy(feats_np).to(dev)
    ade = torch.from_numpy(ade_np).to(dev)
    wt_all = torch.from_numpy(w).to(dev)
    target = torch.from_numpy(lab["final"]).to(dev)
    rule = torch.from_numpy(lab["final"]).to(dev)
    rule_ds = torch.from_numpy(lab["s_ds"]).to(dev)

    def pick_rule(hd, ix):
        _, pk = pick_ade(hd, feats[ix], ade[ix])
        r = torch.arange(len(ix), device=dev)
        return float(rule[ix][r, pk].mean()), float(rule_ds[ix][r, pk].mean())

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    init = head_tensors_from_checkpoint(a.init_ckpt)
    head = head_module()
    head.load_state_dict(init, strict=True)
    head.to(dev)
    with (
        torch.no_grad()
    ):  # the initial head must reproduce the model's own logits and pick from these features
        l0 = head(feats[held_idx][:2048]).cpu().numpy()
        diff = np.abs(l0 - logit_np[held_idx][:2048]).max()
        print(
            f"initial head vs the logits the model returned (2,048 held-out frames): max |diff| {diff:.2e}",
            flush=True,
        )
        p0 = l0[..., 5].argmax(-1)
        if model_pick is not None:
            agree = float(np.mean(p0 == model_pick[held_idx][:2048]))
            print(
                f"initial head reproduces Drive-JEPA's pick on {agree * 100:.1f}% of 2,048 held-out frames",
                flush=True,
            )
            if agree < 0.99:
                raise SystemExit(
                    "the features or the initial head do not reproduce the deployed pick -- stop"
                )
    base_h, _ = pick_ade(head, feats[held_idx], ade[held_idx])
    oracle_h = float(ade[held_idx].min(1).values.mean())
    base_rule, base_rule_ds = pick_rule(head, held_idx)
    print(
        f"held-out before training: pick ADE {base_h:.3f} m | oracle {oracle_h:.3f} m | pick rule final "
        f"{base_rule:.3f} | pick s_ds {base_rule_ds:.3f} | best-of-32 final "
        f"{float(rule[held_idx].max(1).values.mean()):.3f}",
        flush=True,
    )

    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    best = (base_rule, 0, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    curve = []
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        head.train()
        perm = torch.from_numpy(np.random.permutation(train_idx)).to(dev)
        tot = 0.0
        n = 0
        for i in range(0, len(perm), a.batch):
            b = perm[i : i + a.batch]
            zc = head(feats[b])[..., 5]
            ce = bce(zc, target[b]).mean(-1)  # BCE per candidate, mean over the 32 (and the batch below)
            loss = (wt_all[b] * ce).sum() / wt_all[b].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            n += len(b)
        head.eval()
        h_ade, _ = pick_ade(head, feats[held_idx], ade[held_idx])
        t_ade, _ = pick_ade(head, feats[train_idx], ade[train_idx])
        h_rule, h_rule_ds = pick_rule(head, held_idx)
        t_rule, _ = pick_rule(head, train_idx)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": tot / n,
                "train_pick_ade": t_ade,
                "holdout_pick_ade": h_ade,
                "holdout_pick_final": h_rule,
                "holdout_pick_s_ds": h_rule_ds,
                "train_pick_final": t_rule,
            }
        )
        print(
            f"epoch {epoch:3d}  loss {tot / n:.4f}  train pick ADE {t_ade:.3f} final {t_rule:.3f}  "
            f"held-out pick ADE "
            f"{h_ade:.3f} final {h_rule:.3f} s_ds {h_rule_ds:.3f}  ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
        if h_rule > best[0] + 1e-5:
            best = (h_rule, epoch, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[2])
    head.to(dev).eval()
    h_ade, _ = pick_ade(head, feats[held_idx], ade[held_idx])
    probe_idx = held_idx[:4]
    with torch.no_grad():
        probe_scores = head(feats[probe_idx])[..., 5].cpu().numpy()
    torch.save(best[2], a.out / "head_best.pt")
    np.savez(
        a.out / "probe.npz",
        sample_id=np.array([sids[i] for i in probe_idx]),
        feats=feats_np[probe_idx],
        col5=probe_scores,
    )
    fin_rule, fin_rule_ds = pick_rule(head, held_idx)
    metrics = {
        "best_epoch": best[1],
        "holdout_pick_ade": h_ade,
        "holdout_pick_ade_before": base_h,
        "holdout_oracle_ade": oracle_h,
        "holdout_pick_final": fin_rule,
        "holdout_pick_final_before": base_rule,
        "holdout_pick_s_ds": fin_rule_ds,
        "holdout_pick_s_ds_before": base_rule_ds,
        "holdout_best_of_32_final": float(rule[held_idx].max(1).values.mean()),
        "selection": "final",
        "n_base_rows": n_base,
        "n_extra_rows": len(sids) - n_base,
        "holdout_model_pick_ade": float(
            ade_np[held_idx][np.arange(len(held_idx)), model_pick[held_idx]].mean()
        )
        if model_pick is not None
        else None,
        "holdout_frames": int(len(held_idx)),
        "train_frames": int(len(train_idx)),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "target": "ipad-final",
        "init_ckpt": str(a.init_ckpt),
        "head_suffixes": HEAD_SUFFIXES,
        "git_sha": git_sha(),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    json.dump(metrics, open(a.out / "metrics.json", "w"), indent=1)
    json.dump(curve, open(a.out / "curve.json", "w"), indent=1)
    print(
        json.dumps({k: v for k, v in metrics.items() if k not in ("settings", "head_suffixes")}, indent=1),
        flush=True,
    )


if __name__ == "__main__":
    main()
