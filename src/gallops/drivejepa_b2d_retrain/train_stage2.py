"""Step B: Drive-JEPA's Stage 2 on Bench2Drive (specs/phase2_v2.md in H16's folder). Drive-JEPA's own model,
losses and
optimiser (DriveJEPAAgent.pad_loss / get_optimizers, imported unchanged). What is new:
  compute_score   the Step A grade on the 32 live proposals, in place of NAVSIM's simulator + PDM scorer
  teachers        the Step A index, K = 4 drawn per round by Drive-JEPA's own trajectory_loss_anchors;
                  anchors = b2d_2048.npy[:, 4::5]
  encoder         Phase 1's merged checkpoint, sha256 checked, backbone group at lr x 0.1. Planner, scorer
                  and the projector from scratch
  batch fix       gallops.drivejepa_b2d.batch_fix.apply_all(): upstream's refiner mixes batch elements
                  when batch > 1 (the run applied paddock-ops mtd/stage4/sca_fix.py, the same three patches)
Scorer columns, NAVSIM order [NC, DAC, TL, DDC, EP, TTC, LK, EC, final]: NC = collision factors, DAC = 1 -
off-lane share, TL = 0.7^n_red, DDC = 1 - oncoming share, EP = P, TTC and EC have no term (1), LK = 1 -
outside, final = S.
    python -m gallops.drivejepa_b2d_retrain.train_stage2 --name NAME [--micro 32] [--resume SAVE]
        [--max-steps N] [--epochs 20] [--init-released]"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gallops.drivejepa_b2d_retrain import wiring as W

L, Pl, M = W.setup()
from navsim.agents.abstract_agent import AbstractAgent  # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent  # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig  # noqa: E402
from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel  # noqa: E402

from gallops.drivejepa_b2d import batch_fix  # noqa: E402
from gallops.drivejepa_b2d_retrain import grade as G2  # noqa: E402
from gallops.drivejepa_b2d_retrain import samples as D2  # noqa: E402

batch_fix.apply_all()
ENC = W.RUNS / "merged/p1_05nov_pass_03/merged_encoder_pass_03.pt"
ENC_SHA = "339d74428a15de9d20976568cae9decbe71f3ac81286b7085a68bd6fe53d5e94"
VOCAB = W.VOCAB / "b2d_2048.npy"
RUNS = W.RUNS.parent / "phase2/runs"
SEED = 20261105
OFF = G2.OFF_REAR


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class B2DAgent(DriveJEPAAgent):
    def __init__(self, config, lr):
        AbstractAgent.__init__(self, config.trajectory_sampling)
        self._config, self._lr, self._checkpoint_path = config, lr, None
        self._pad_model = DriveJEPAModel(config)
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
        self.anchors = np.load(VOCAB)[:, 4::5]

    def compute_score(self, targets, proposals, test=False):
        g = targets["g"]
        t = G2.terms(g, G2.to_centre_t(proposals.detach().float()), steps=True)
        dref = targets["dref"]
        S, P = G2.grade(t, dref)
        one = torch.ones_like(S)
        col = 0.5 ** t["n_ped"].float() * 0.6 ** t["n_veh"].float() * 0.65 ** t["n_static"].float()
        ts = torch.stack(
            [
                col,
                1 - t["off_share"],
                0.7 ** t["n_red"].float(),
                1 - t["onc_share"],
                P,
                one,
                1 - t["outside"],
                one,
                S,
            ],
            -1,
        )
        B, C = S.shape
        corners = torch.zeros(B, C, G2.K, 2, 4, 2, device=S.device)
        labels = torch.zeros(B, C, G2.K, 2, dtype=torch.bool, device=S.device)
        hit = (t["first_agent"] >= 0).nonzero()
        if len(hit):
            f, c = hit[:, 0], hit[:, 1]
            a = t["first_agent"][f, c]
            ks = torch.arange(1, G2.K + 1, device=S.device)
            ax, ay, ac, as_ = (
                g["ax"][f, a][:, 1:],
                g["ay"][f, a][:, 1:],
                g["ac"][f, a][:, 1:],
                g["as_"][f, a][:, 1:],
            )
            hl, hw = g["alen"][f, a][:, None] / 2, g["awid"][f, a][:, None] / 2
            sx = torch.tensor([1.0, 1.0, -1.0, -1.0], device=S.device)
            sy = torch.tensor([1.0, -1.0, -1.0, 1.0], device=S.device)
            lx, ly = hl[..., None] * sx, hw[..., None] * sy
            X = ax[..., None] + lx * ac[..., None] - ly * as_[..., None] + OFF
            Y = ay[..., None] + lx * as_[..., None] + ly * ac[..., None]
            corners[f, c, :, 0] = torch.stack([X, Y], -1)
            labels[f, c, :, 0] = (ks[None] <= (t["first_step"][f, c] + 1)[:, None]) & g["valid"][f, a][:, 1:]
        areas = torch.stack([t["off_step"], t["onc_step"]], -1)
        targets["last_terms"] = t
        return S, S.amax(-1), ts, corners, labels, areas, targets["teachers"]


def features(b, T, dev):
    B = len(b["k"])
    return dict(
        camera_feature_1=b["new"].to(dev, non_blocking=True),
        camera_feature_2=b["old"].to(dev, non_blocking=True),
        ego_status=b["ego"].to(dev),
        lidar2img=T.lidar2img[None].expand(B, *T.lidar2img.shape).to(dev),
        img_shape=torch.as_tensor(np.asarray(T.img_shape))[None]
        .expand(B, *np.asarray(T.img_shape).shape)
        .to(dev),
        past_ego_simulated_states=None,
        metric_cache=None,
        observation_interval=None,
    )


def targets_of(b, dev):
    g = G2.prepare2(b["ctx"], dev)
    g["route_len"] = b["route"].to(dev)
    return dict(trajectory=b["traj"].to(dev), dref=b["dref"].to(dev), g=g, teachers=b["teachers"])


def build(a, dev):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    cfg = DriveJEPAConfig()
    ag = B2DAgent(cfg, 1e-4).to(dev)
    if (
        a is not None and getattr(a, "init_released", False)
        # 06 Nov: fine-tune the released planner instead of training it from scratch; the encoder is still
        # Phase
        # 1's merged one
    ):
        assert sha256(W.AGENT_CKPT) == W.AGENT_SHA
        sd = torch.load(W.AGENT_CKPT, map_location=dev)["state_dict"]
        ag._pad_model.load_state_dict(
            {k.replace("agent._pad_model.", ""): v for k, v in sd.items()}, strict=True
        )
        del sd
    assert sha256(ENC) == ENC_SHA
    ag._pad_model._backbone.img_backbone.load_state_dict(torch.load(ENC, map_location=dev), strict=True)
    return ag, cfg


@torch.no_grad()
def validate(ag, cfg, Tv, loader, dev):
    m = ag._pad_model
    m.eval()
    m.scorer.train()
    acc = {}
    rows = []
    iu = np.triu_indices(32, 1)
    for b in loader:
        tg = targets_of(b, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = ag.forward(features(b, Tv, dev))
            ld = ag.pad_loss(tg, pred, cfg)
        for k, v in ld.items():
            acc[k] = acc.get(k, 0.0) + float(v) * len(b["k"])
        p = pred["proposals"].float()
        e = tg["trajectory"][:, None, :6, :2]
        ade = (p[:, :, :6, :2] - e).norm(dim=-1).mean(-1)
        pick = pred["pdm_score"].argmax(1)
        r = torch.arange(len(pick), device=dev)
        S, P = G2.grade(tg.pop("last_terms"), tg["dref"])
        te = G2.terms(tg["g"], tg["g"]["expert"][:, None])
        Se, _ = G2.grade(te, tg["dref"])
        d = (p[:, :, None, :6, :2] - p[:, None, :, :6, :2]).norm(dim=-1).mean(-1)[:, iu[0], iu[1]].mean(-1)
        rows.append(torch.stack([ade.amin(1), ade[r, pick], S[r, pick], Se[:, 0], S.amax(1), d], 1).cpu())
    x = torch.cat(rows).numpy()
    n = len(x)
    m.train()
    return dict(
        losses={k: v / n for k, v in acc.items()},
        frames=n,
        closest_ade_3s=float(x[:, 0].mean()),
        pick_ade_3s=float(x[:, 1].mean()),
        pick_grade=float(x[:, 2].mean()),
        expert_grade=float(x[:, 3].mean()),
        best_of_32_grade=float(x[:, 4].mean()),
        variety_m=float(x[:, 5].mean()),
    )


def save(path, ag, opt, sc, state):
    tmp = Path(str(path) + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    torch.save(
        dict(
            model=ag._pad_model.state_dict(),
            opt=opt.state_dict(),
            scaler=sc.state_dict(),
            state=state,
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
            ),
        ),
        tmp / "state.pt",
    )
    json.dump(state, open(tmp / "state.json", "w"))
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--resume")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--save-at", type=int)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--limit-clips", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--init-released", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    if os.environ.get("P2_MEM_FRAC"):
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ["P2_MEM_FRAC"])
        )  # our own limit, so an out-of-memory stops this run and not run 3A-M
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    log = lambda *x: print(time.strftime("%T"), *x, flush=True)  # noqa: E731
    log("tables")
    Tt = D2.Tables("train", a.limit_clips)
    Tv = D2.Tables("heldout", a.limit_clips)
    log("frames", len(Tt), len(Tv))
    ag, cfg = build(a, dev)
    ag.train()
    opt = ag.get_optimizers()
    sc = torch.amp.GradScaler("cuda")
    state = dict(epoch=0, batch=0, step=0, name=a.name, micro=a.micro, batch_size=a.batch)
    if a.resume:
        ck = torch.load(Path(a.resume) / "state.pt", map_location="cpu", weights_only=False)
        ag._pad_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sc.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log("resumed", a.resume, state)
    vload = torch.utils.data.DataLoader(
        D2.Samples(Tv), batch_size=32, shuffle=False, num_workers=2, collate_fn=D2.collate
    )
    ds = D2.Samples(Tt)
    slog = open(run / "steps.jsonl", "a")
    acc = a.batch // a.micro
    for ep in range(state["epoch"], a.epochs):
        g = torch.Generator()
        g.manual_seed(SEED * 1000 + ep)
        perm = torch.randperm(len(ds), generator=g).tolist()
        nb = len(perm) // a.batch
        micro = [perm[i : i + a.micro] for i in range(state["batch"] * a.batch, nb * a.batch, a.micro)]
        loader = iter(
            torch.utils.data.DataLoader(
                ds,
                batch_sampler=micro,
                num_workers=a.workers,
                collate_fn=D2.collate,
                prefetch_factor=3,
                persistent_workers=False,
            )
        )
        for bi in range(state["batch"], nb):
            t0 = time.time()
            wait = 0.0
            ls = {}
            for _ in range(acc):
                w0 = time.time()
                b = next(loader)
                wait += time.time() - w0
                tg = targets_of(b, dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = ag.forward(features(b, Tt, dev))
                    ld = ag.pad_loss(tg, pred, cfg)
                sc.scale(ld["loss"] / acc).backward()
                for k, v in ld.items():
                    ls[k] = ls.get(k, 0.0) + float(v) / acc
            if not all(math.isfinite(v) for v in ls.values()):
                log("NONFINITE", ls)
            sc.step(opt)
            sc.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            state.update(batch=bi + 1, step=state["step"] + 1)
            slog.write(
                json.dumps(
                    dict(
                        step=state["step"],
                        epoch=ep,
                        step_s=round(time.time() - t0, 3),
                        wait_s=round(wait, 3),
                        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        scale=sc.get_scale(),
                        **{k: round(v, 5) for k, v in ls.items()},
                    )
                )
                + "\n"
            )
            if state["step"] % 20 == 0:
                slog.flush()
                log(
                    "epoch",
                    ep + 1,
                    "step",
                    state["step"],
                    "loss",
                    round(ls["loss"], 4),
                    "traj",
                    round(ls["trajectory_loss"], 3),
                    "final",
                    round(ls["final_score_loss"], 4),
                    "score",
                    round(ls["score"], 3),
                    "best",
                    round(ls["best_score"], 3),
                    round(time.time() - t0, 2),
                    "s wait",
                    round(wait, 2),
                )
            if a.save_at == state["step"]:
                save(run / f"save_step{state['step']:06d}", ag, opt, sc, dict(state))
                log("saved at", state["step"])
            if a.max_steps and state["step"] >= a.max_steps:
                slog.flush()
                log("MAX_STEPS", state["step"], "peak", round(torch.cuda.max_memory_allocated() / 2**30, 2))
                return
        del loader
        state.update(epoch=ep + 1, batch=0)
        save(run / f"ckpt_epoch{ep + 1:02d}", ag, opt, sc, dict(state))
        if not a.no_val:
            r = validate(ag, cfg, Tv, vload, dev)
            json.dump(r, open(run / f"ckpt_epoch{ep + 1:02d}/heldout.json", "w"), indent=1)
            log(
                "EPOCH_DONE",
                ep + 1,
                json.dumps(
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "losses"}
                ),
                "loss",
                round(r["losses"]["loss"], 4),
            )
    log("TRAIN_DONE")


if __name__ == "__main__":
    main()
