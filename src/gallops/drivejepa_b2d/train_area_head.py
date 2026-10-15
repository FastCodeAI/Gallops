"""Stage final-area-head: fine-tune Drive-JEPA's own area head (`scorer.pred_area`) on Bench2Drive.

`pred_area` is an MLP(256, 1024, 10) on each per-pose query of the 32 candidates (32 x 8 x 256, stored fp16).
Its output per candidate is (40 steps, 2): [off-road, oncoming], NAVSIM's [NON_DRIVABLE_AREA,
ONCOMING_TRAFFIC] order. Loss = 2 x BCE over the steps whose annotation frame exists (step_mask), against the
v6 step labels (oncoming is connector-aware; own-lane-blocked only for objects that never move). Encoder,
generator and refiner are frozen and never loaded: training reads the stored head inputs of the MTD-H
generator's candidates (the Final's generator). Starts from the released NAVSIM weights. Adam 1e-4, batch 64
frames, 40 epochs, seed 0. Same split as the split head: 150 held-out clips, base + route-start rows held out,
the augmented light copies training-only. Per candidate, off-road / oncoming = max over steps. Kept epoch =
best mean held-out within-frame AUC of the two outputs.

The deployed policy loads this head from its own file (DRIVEJEPA_SPLIT_AREA), not from the weights file:
final-checkpoint copies head_best.pt into the step directory as area_head.pt and names it in the sidecar.

Ported from paddock-ops mtd/stage4/train_area_mtd.py (run of 5 Oct, `--head area`); the trainer's `--head col`
branch (pred_col_agent) produced nothing deployed and is left out. Arithmetic unchanged.

    python -m gallops.drivejepa_b2d.train_area_head --rows-base <npz> --rows-start <npz> --dump '<dir>'
        --labels <dir> --holdout-list <json> --init-ckpt <released ckpt> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gallops.drivejepa_b2d.head import auc
from gallops.drivejepa_b2d.provenance import git_sha

PREFIX = "agent._pad_model.scorer.pred_area.mlp."
NAMES = ("offroad", "oncoming")


def mlp(out):
    return nn.Sequential(
        nn.Linear(256, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Linear(1024, out),
    )


def load_feats(globs, sids, key):
    want = set(sids)
    fe = {}
    for g in globs:
        for f in sorted(glob.glob(g)):
            q = np.load(f)
            qs = q["sample_id"].astype(str)
            m = np.isin(qs, list(want))
            if m.any():
                fe.update(zip(qs[m], q[key][m]))  # noqa: B905 -- same length by construction
    miss = [s for s in sids if s not in fe]
    if miss:
        raise SystemExit(f"{len(miss)} rows without {key} (e.g. {miss[:3]})")
    return np.stack([fe[s] for s in sids])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-base", type=Path, required=True, help="base rows npz (sample_id, scene)")
    ap.add_argument("--rows-start", type=Path, required=True, help="route-start rows npz (sample_id, scene)")
    ap.add_argument(
        "--dump", type=Path, required=True, help="candidate dump dir with head_<set>_*.npz (feats8)"
    )
    ap.add_argument("--labels", type=Path, required=True, help="dir with b2d_<set>_v6_steps.npz")
    ap.add_argument("--holdout-list", type=Path, required=True, help='JSON {"val": [held-out clip names]}')
    ap.add_argument("--init-ckpt", type=Path, required=True, help="released Drive-JEPA checkpoint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    a.out.mkdir(parents=True, exist_ok=False)
    dev = "cuda"
    held_clips = set(json.load(open(a.holdout_list))["val"])
    sets = [("base", a.rows_base), ("start", a.rows_start), ("aug", None)]

    S, H, F = [], [], []
    LAB = {k: [] for k in ("offroad", "oncoming", "step_mask")}
    for name, ids in sets:
        V = np.load(a.labels / f"b2d_{name}_v6_steps.npz", allow_pickle=True)
        vs = [str(s) for s in V["sample_id"]]
        if ids is None:
            held = np.zeros(len(vs), bool)
        else:
            z = np.load(ids, allow_pickle=True)
            sc = {str(s): str(c) for s, c in zip(z["sample_id"], z["scene"])}  # noqa: B905 -- one file's arrays
            held = np.array([sc[s] in held_clips for s in vs])
        for k in LAB:
            LAB[k].append(V[k])
        S += vs
        H.append(held)
        F.append(load_feats([str(a.dump / f"head_{name}_*.npz")], vs, "feats8"))
        print(f"{name}: {len(vs):,} rows, held-out {held.sum():,}", flush=True)
    LAB = {k: np.concatenate(v) for k, v in LAB.items()}
    held = np.concatenate(H)
    X = torch.from_numpy(np.concatenate(F)).to(dev)
    tr, he = np.where(~held)[0], np.where(held)[0]
    M = torch.from_numpy(LAB["step_mask"]).to(dev)  # (N, 40)
    T = torch.from_numpy(np.stack([LAB["offroad"], LAB["oncoming"]], -1).astype(np.float32)).to(dev)
    head = mlp(10)
    sd = torch.load(a.init_ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    head.load_state_dict({k[len(PREFIX) :]: v.float().clone() for k, v in sd.items() if k.startswith(PREFIX)})
    del sd
    head.to(dev)
    pos = ", ".join(
        f"{k} {((LAB[k] & LAB['step_mask'][:, None, :]).any(2)).mean() * 100:.1f}%" for k in NAMES
    )
    print(
        f"{len(S):,} rows: train {len(tr):,}, held-out {len(he):,}; positive candidate rate (any step): "
        f"{pos}",
        flush=True,
    )

    def forward(idx):
        x = X[idx].float()
        return head(x.reshape(len(idx), 256, 256)).reshape(len(idx), 32, 40, 2)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(head.parameters(), lr=1e-4)

    def loss_of(idx):
        logit = forward(idx)
        m = M[idx][:, None, :, None].expand_as(logit).float()
        loss = (bce(logit, T[idx]) * m).sum() / m.sum().clamp(min=1)
        return 2.0 * loss

    def evaluate(idx):
        head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                p = torch.sigmoid(forward(idx[i : i + 256])) * M[idx[i : i + 256]][:, None, :, None]
                out.append(p.amax(2).cpu().numpy())
        P = np.concatenate(out)
        Y = (T[idx].cpu().numpy() > 0.5) & (M[idx].cpu().numpy()[:, None, :, None])
        Y = Y.any(2)  # (n, 32, 2) per candidate
        res = {}
        for c in range(2):
            s = -P[..., c]
            y = ~Y[..., c]  # high = safe, label 1 = passes (as the split head's columns)
            w = [auc(s[i], y[i]) for i in range(len(s)) if y[i].any() and not y[i].all()]
            res[c] = dict(
                auc=auc(s, y),
                within=float(np.nanmean(w)) if w else float("nan"),
                n_mixed=len(w),
                pos_rate=float((~y).mean()),
            )
        return res, P

    def line(ev):
        return " | ".join(
            f"{NAMES[c]} within {ev[c]['within']:.3f} pooled {ev[c]['auc']:.3f}" for c in range(2)
        )

    ev0, _ = evaluate(he)
    print(f"NAVSIM weights (epoch 0): {line(ev0)}", flush=True)
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best = (np.nanmean([ev0[c]["within"] for c in range(2)]), 0, state)
    curve = []
    t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = torch.from_numpy(np.random.permutation(tr)).to(dev)
        tot = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            loss = loss_of(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        ev, _ = evaluate(he)
        m = float(np.nanmean([ev[c]["within"] for c in range(2)]))
        curve.append(
            dict(
                epoch=ep,
                loss=tot / len(tr),
                **{f"{NAMES[c]}_within": ev[c]["within"] for c in range(2)},
                **{f"{NAMES[c]}_auc": ev[c]["auc"] for c in range(2)},
            )
        )
        if m > best[0]:
            best = (m, ep, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        if ep % 5 == 0 or ep == 1:
            print(
                f"ep {ep:2d} loss {tot / len(tr):.4f} | {line(ev)} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    head.load_state_dict(best[2])
    ev, P = evaluate(he)
    torch.save(best[2], a.out / "head_best.pt")
    np.savez(a.out / "heldout_scores.npz", sample_id=np.array([S[i] for i in he]), p=P)
    metrics = dict(
        head="area",
        best_epoch=best[1],
        best_mean_within=best[0],
        held=ev,
        held_navsim=ev0,
        curve=curve,
        settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        git_sha=git_sha(),
    )
    json.dump(metrics, open(a.out / "metrics.json", "w"), indent=1, default=float)
    print(f"kept epoch {best[1]}: {line(ev)}", flush=True)


if __name__ == "__main__":
    main()
