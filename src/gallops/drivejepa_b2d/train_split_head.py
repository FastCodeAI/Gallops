"""Stages split-full, split-v4aug and final-split-head: scorer-head columns 0-3 as four yes/no heads.

Per candidate (first 6 poses, 3 s): col (no collision), lane (Bench2Drive lane check passes at every pose),
ttc (the time-to-collision term passes), red (no red-light infraction). Loss: BCE per column, mean over the 32
candidates, summed over the four; Adam 1e-4, batch 64, 40 epochs, seed 0, every layer trained but output rows
4 and 5 frozen (their gradients are zeroed, and the run asserts they are unchanged). Kept epoch = best mean
held-out pooled AUC over the four columns. Held-out = the base + route-start rows of the 150 held-out clips;
augmented rows are training-only.

One recipe, three runs, differing only in what they read:

  split-full         v3 labels, no augmented rows, init = ipad-base-start          (29 Sep)
  split-v4aug        v4 labels (red = Bench2Drive's stop-line rule) + 8,316         (30 Sep; Split-v4's head)
                     speed-randomised copies of training light frames, init = split-full
  final-split-head   the same v4 recipe on the MTD-H generator's candidates and     (6 Oct; the Final's head)
                     their m05b labels (0.5 m moving-actor margin), init = split-v4aug

Ported from paddock-ops scorer-retrain/train_split_heads.py, train_split_heads_v4.py and
mtd/stage4/train_split_m05b.py, whose training arithmetic is identical; only the inputs were hard-coded.

    python -m gallops.drivejepa_b2d.train_split_head --dataset-base <npz> --dataset-start <npz>
        --feats-base '<glob>' --feats-start '<glob>' --labels-base <npz> --labels-start <npz>
        [--feats-aug '<glob>' --labels-aug <npz>] --holdout-list <json> --init <head.pt> --labels-name v4
            --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from gallops.drivejepa_b2d.head import auc, head_module
from gallops.drivejepa_b2d.provenance import git_sha

COLS = ("col", "lane", "ttc", "red")
#: One optimizer step per batch, the last one short: metrics.json records ceil(n_train / BATCH) steps per
#: epoch, which is what a checkpoint's `step` counts (write_checkpoint).
BATCH = 64


def labels(lp, sids):
    L = np.load(lp, allow_pickle=True)
    pl = {str(s): i for i, s in enumerate(L["sample_id"])}
    o = np.array([pl[s] for s in sids])
    cols = [
        L["collision"][o][:, :32] > 0.5,
        L["lane"][o][:, :32] > 0.5,
        L["ttc"][o][:, :32] > 0.5,
        L["red"][o][:, :32] > 0.5,
    ]
    return np.stack(cols, -1).astype(np.float32)


def feats(fp, sids):
    want = set(sids)
    fe = {}
    for f in sorted(glob.glob(fp)):
        q = np.load(f)
        qs = q["sample_id"].astype(str)
        m = np.isin(qs, list(want))
        if m.any():
            fe.update(zip(qs[m], q["feats"][m]))  # noqa: B905 -- same length by construction
    return np.stack([fe[s] for s in sids]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in ("base", "start"):
        ap.add_argument(f"--dataset-{s}", type=Path, required=True, help="rows npz (sample_id, scene)")
        ap.add_argument(f"--feats-{s}", required=True, help="glob of head-input feature files")
        ap.add_argument(f"--labels-{s}", type=Path, required=True)
    ap.add_argument("--feats-aug", help="augmented light-frame features; training-only rows")
    ap.add_argument("--labels-aug", type=Path)
    ap.add_argument("--holdout-list", type=Path, required=True, help='JSON {"val": [held-out clip names]}')
    ap.add_argument(
        "--init", type=Path, required=True, help="head to start from (head_best.pt of the stage before)"
    )
    ap.add_argument("--labels-name", required=True, help="label version, recorded in metrics.json (v3, v4)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.feats_aug is None) != (a.labels_aug is None):
        ap.error("--feats-aug and --labels-aug go together")
    torch.manual_seed(0)
    np.random.seed(0)
    a.out.mkdir(parents=True, exist_ok=False)
    held_clips = set(json.load(open(a.holdout_list))["val"])

    F, Y, H, S, AUG = [], [], [], [], []
    sets = ((a.dataset_base, a.feats_base, a.labels_base), (a.dataset_start, a.feats_start, a.labels_start))
    for ds, fp, lp in sets:
        z = np.load(ds, allow_pickle=True)
        sids = [str(s) for s in z["sample_id"]]
        F.append(feats(fp, sids))
        Y.append(labels(lp, sids))
        H.append(np.array([str(c) in held_clips for c in z["scene"]]))
        S += sids
        AUG.append(np.zeros(len(sids), bool))
    if a.labels_aug is not None:
        La = np.load(a.labels_aug, allow_pickle=True)
        sids = [str(s) for s in La["sample_id"]]
        assert not any(s.split("/")[0] in held_clips for s in sids), "augmented rows from held-out clips"
        F.append(feats(a.feats_aug, sids))
        Y.append(labels(a.labels_aug, sids))
        H.append(np.zeros(len(sids), bool))
        S += sids
        AUG.append(np.ones(len(sids), bool))
    X = torch.from_numpy(np.concatenate(F)).cuda()
    Yt = torch.from_numpy(np.concatenate(Y)).cuda()
    held = np.concatenate(H)
    aug = np.concatenate(AUG)
    tr, he = np.where(~held)[0], np.where(held)[0]
    rates = ", ".join(f"{c} {Yt[tr][..., i].mean().item() * 100:.1f}%" for i, c in enumerate(COLS))
    print(
        f"{len(X):,} frames: train {len(tr):,} (of which augmented {aug[tr].sum():,}), held-out {len(he):,}; "
        f"label-1 rates (train): {rates}",
        flush=True,
    )

    head = head_module()
    init = torch.load(a.init, map_location="cpu")
    head.load_state_dict(init)
    head.cuda()
    w5, b5 = init["mlp.6.weight"][5].clone(), init["mlp.6.bias"][5].clone()
    w4, b4 = init["mlp.6.weight"][4].clone(), init["mlp.6.bias"][4].clone()
    opt = torch.optim.Adam(head.parameters(), lr=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def evaluate(idx):
        head.eval()
        with torch.no_grad():
            chunks = [head(X[idx][i : i + 4096])[..., :4] for i in range(0, len(idx), 4096)]
            p = torch.sigmoid(torch.cat(chunks)).cpu().numpy()
        y = Yt[idx].cpu().numpy()
        out = {}
        for i, c in enumerate(COLS):
            yy = y[..., i] > 0.5
            pp = p[..., i]
            wf = [auc(pp[f], yy[f]) for f in range(len(pp)) if yy[f].any() and not yy[f].all()]
            out[c] = dict(
                auc=auc(pp, yy),
                auc_within=float(np.nanmean(wf)) if wf else float("nan"),
                n_mixed=len(wf),
                p_label0=float(pp[~yy].mean()),
                p_label1=float(pp[yy].mean()),
                n0=int((~yy).sum()),
            )
        return out

    ev0 = evaluate(he)
    print(
        "init on held-out: " + " ".join(f"{c} {ev0[c]['auc']:.3f}/{ev0[c]['auc_within']:.3f}" for c in COLS),
        flush=True,
    )
    best = (-1, 0, None)
    curve = []
    t0 = time.perf_counter()
    for ep in range(1, 41):
        head.train()
        perm = torch.from_numpy(np.random.permutation(tr)).cuda()
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i : i + BATCH]
            z = head(X[b])[..., :4]
            loss = bce(z, Yt[b]).mean((0, 1)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            head.mlp[6].weight.grad[4:] = 0  # rows 4 and 5 never move
            head.mlp[6].bias.grad[4:] = 0
            opt.step()
            tot += float(loss) * len(b)
        ev = evaluate(he)
        m = float(np.mean([ev[c]["auc"] for c in COLS]))
        curve.append(
            dict(
                epoch=ep,
                loss=tot / len(tr),
                mean_auc=m,
                **{f"{c}_auc": ev[c]["auc"] for c in COLS},
                red_within=ev["red"]["auc_within"],
            )
        )
        if m > best[0]:
            best = (m, ep, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        if ep % 5 == 0 or ep == 1:
            held_auc = " ".join(f"{c} {ev[c]['auc']:.3f}" for c in COLS)
            print(
                f"ep {ep:2d} loss {tot / len(tr):.4f} | held AUC {held_auc} | red within "
                f"{ev['red']['auc_within']:.3f} | mean {m:.3f} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    head.load_state_dict(best[2])
    for r, w, bb in ((5, w5, b5), (4, w4, b4)):
        same = torch.equal(head.mlp[6].weight[r].cpu(), w) and torch.equal(head.mlp[6].bias[r].cpu(), bb)
        assert same, f"column {r} output weights changed"
    ev = evaluate(he)
    torch.save(best[2], a.out / "head_best.pt")
    with torch.no_grad():
        allc = head(X[he[:4]]).cpu().numpy()
    # col5: what write_checkpoint probes the written checkpoint against
    np.savez(
        a.out / "probe.npz",
        sample_id=np.array([S[i] for i in he[:4]]),
        feats=X[he[:4]].cpu().numpy(),
        col5=allc[..., 5],
        logits=allc,
    )
    metrics = dict(
        best_epoch=best[1],
        mean_auc=best[0],
        held=ev,
        held_init=ev0,
        curve=curve,
        init=str(a.init),
        n_train=int(len(tr)),
        batch=BATCH,
        steps_per_epoch=-(-len(tr) // BATCH),
        n_aug=int(aug[tr].sum()),
        labels=a.labels_name,
        settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        git_sha=git_sha(),
    )
    json.dump(metrics, open(a.out / "metrics.json", "w"), indent=1)
    print(f"\nkept epoch {best[1]} (mean held-out AUC {best[0]:.3f})")
    for c in COLS:
        e = ev[c]
        print(
            f"{c}: AUC {e['auc']:.3f} within {e['auc_within']:.3f} ({e['n_mixed']} mixed) "
            f"p0 {e['p_label0']:.3f} "
            f"p1 {e['p_label1']:.3f} n0 {e['n0']:,}"
        )


if __name__ == "__main__":
    main()
