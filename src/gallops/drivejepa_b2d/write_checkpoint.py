"""Stages split-v4-checkpoint and final-checkpoint: write a deployable Drive-JEPA checkpoint and its sidecar.

Copies --src, replaces the trained tensors, saves to
{weights}/<model>/<train_run>/step_<n>/drive_jepa_vitl_v2.ckpt with checkpoint.json beside it
(docs/checkpoints.md). The step directory is staged and renamed into place only once the probe passed and the
sidecar is written (sidecar.publish), so {weights} never holds the weights without it. Refuses a train run
that already holds a checkpoint. Tensors are matched under the Lightning prefix `agent._pad_model.`:

  --head  scorer.pred_score.mlp.{0,1,3,4,6}.{weight,bias} (10 tensors)
  --gen   init_feature.weight, hist_encoding.{weight,bias}, and the shared refiner `_trajectory_head.0.*`
          written into all four _trajectory_head.{0,1,2,3} copies (171 tensors from 45)

  --area  not a tensor in the weights file: the policy loads the area head from its own file, so its
          head_best.pt is copied beside the weights as area_head.pt and the policy env names it
          (DRIVEJEPA_SPLIT_AREA)

Split-v4 = the released checkpoint + the split-v4aug head. The Final = the released checkpoint + the MTD-H
generator + its own split head, with its area head beside them. The step is the head stage's optimizer steps
up to its kept epoch.

Probe: the written file is reloaded and its head must give exactly the column-5 scores the trained head gave
on the 4 held-out probe rows saved at training time. The full-forward probe through paddock's policy is
paddock's job.

Ported from paddock-ops scorer-retrain/write_scorer_checkpoint.py (Split-v4) and
generator-ft/step4_write_ckpt.py (the Final), which wrote the same tensors with a hand-written sidecar beside
the file.

The sidecar also records how the checkpoint must be driven (--policy-env, paddock's adapter switches) and
what pins its inputs (--data-manifest, --holdout-list): sidecar.parse_env and sidecar.input_notes.

    python -m gallops.drivejepa_b2d.write_checkpoint --src <ckpt> --head <dir>
        [--gen <gen_best.pt> --area <dir>] --model drivejepa_scorer_b2d --train-run <name>
        --policy-env KEY=VALUE ... --data-manifest <data_*.json> ... --holdout-list <json> [--note "..."]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from gallops.checkpoint import Checkpoint, TrainedOn
from gallops.config import SiteConfig
from gallops.drivejepa_b2d.head import HEAD_SUFFIXES, head_module
from gallops.drivejepa_b2d.provenance import git_sha
from gallops.drivejepa_b2d.sidecar import (
    AREA_HEAD_FILE,
    input_notes,
    optimizer_step,
    parse_env,
    publish,
    refuse_existing_run,
    sha256,
    with_area_head,
)

PREFIX = "agent._pad_model."
WEIGHTS_NAME = "drive_jepa_vitl_v2.ckpt"

#: What every checkpoint of this recipe saw. Drive-JEPA's release was trained on OpenScene and then NAVSIM
#: navtrain, so a NAVSIM score of any of these checkpoints is not zero-shot. The Bench2Drive base set is split
#: by clip: recipes/drivejepa_b2d/heldout_150.json holds out 150 of the 1,000 clips, which chose every kept
#: epoch.
TRAINED_ON = (
    TrainedOn("navsim", "navtrain"),
    TrainedOn("bench2drive", "base-train"),
    TrainedOn("bench2drive", "base-heldout-150"),
)


def replacements(head_sd: dict, gen_sd: dict) -> dict:
    repl = {}
    for k, v in gen_sd.items():
        # One shared refiner: the same tensors go into each of the 4 copies.
        if k.startswith("_trajectory_head.0."):
            for i in range(4):
                repl[PREFIX + k.replace("_trajectory_head.0.", f"_trajectory_head.{i}.")] = v
        else:
            repl[PREFIX + k] = v
    for k, v in head_sd.items():  # keys 'mlp.<i>.<p>'
        repl[PREFIX + "scorer.pred_score." + k] = v
    return repl


def probe(dst: Path, head_sd: dict, probe_npz: Path) -> float:
    """Reload the written file; its head must reproduce the trained head's column 5 on the probe rows
    exactly."""
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    reloaded = head_module()
    reloaded.load_state_dict(
        {s.split("scorer.pred_score.")[1]: sd[PREFIX + s].float() for s in HEAD_SUFFIXES}
    )
    trained = head_module()
    trained.load_state_dict({k: v.float() for k, v in head_sd.items()})
    pr = np.load(probe_npz)
    x = torch.from_numpy(pr["feats"])
    with torch.no_grad():
        c_reloaded = reloaded.eval()(x)[..., 5].numpy()
        c_trained = trained.eval()(x)[..., 5].numpy()
    fp32 = all(sd[PREFIX + s].dtype == torch.float32 for s in HEAD_SUFFIXES)
    gap = float(np.abs(c_reloaded - c_trained).max())
    if (fp32 and gap != 0.0) or gap > 1e-3:
        raise SystemExit(f"PROBE FAILED: reloaded head differs from the trained head by up to {gap:.2e}")
    saved = float(np.abs(c_trained - pr["col5"]).max())
    if saved > 1e-4:
        raise SystemExit(
            f"PROBE FAILED: trained head differs from the scores saved at training time by {saved:.2e}"
        )
    return gap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="checkpoint to copy")
    ap.add_argument(
        "--head", type=Path, required=True, help="split-head run dir (head_best.pt, probe.npz, metrics.json)"
    )
    ap.add_argument("--gen", type=Path, help="generator gen_best.pt; omit to keep --src's generator")
    ap.add_argument(
        "--area", type=Path, help="area-head run dir (head_best.pt, metrics.json), shipped beside"
    )
    ap.add_argument("--model", required=True, help="paddock adapter name, e.g. drivejepa_scorer_b2d")
    ap.add_argument("--train-run", required=True)
    ap.add_argument(
        "--policy-env",
        action="append",
        required=True,
        help="KEY=VALUE the adapter is driven with; repeatable",
    )
    ap.add_argument(
        "--data-manifest", type=Path, action="append", required=True, help="tracked data_*.json; repeatable"
    )
    ap.add_argument(
        "--holdout-list", type=Path, required=True, help="the held-out clips every epoch was kept on"
    )
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    policy_env = parse_env(a.policy_env)
    if a.area:
        policy_env = with_area_head(policy_env)
        area_epoch = int(json.load(open(a.area / "metrics.json"))["best_epoch"])

    head_metrics = json.load(open(a.head / "metrics.json"))
    step = optimizer_step(head_metrics)
    out_dir = SiteConfig.load().checkpoint_dir(a.model, a.train_run, step)
    refuse_existing_run(out_dir.parent)  # before reading 3 GB, not only at publish time

    head_sd = torch.load(a.head / "head_best.pt", map_location="cpu")
    gen_sd = torch.load(a.gen, map_location="cpu") if a.gen else {}
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    repl = replacements(head_sd, gen_sd)
    missing = [k for k in repl if k not in sd]
    if missing:
        raise SystemExit(f"keys not in the checkpoint: {missing[:5]}")
    for k, v in repl.items():
        if tuple(sd[k].shape) != tuple(v.shape):
            raise SystemExit(f"{k}: shape {tuple(sd[k].shape)} vs trained {tuple(v.shape)}")
        sd[k] = v.to(sd[k].dtype).clone()  # clone: the four refiner copies must not share one storage
    expected = 10 + (171 if gen_sd else 0)
    if len(repl) != expected:
        raise SystemExit(f"replaced {len(repl)} tensors, expected {expected}")

    dtypes = {v.dtype for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()}
    record = Checkpoint(
        model=a.model,
        step=step,
        train_git_sha=git_sha(),
        train_run=a.train_run,
        trained_on=TRAINED_ON,
        selection_split="bench2drive/base-heldout-150",
        precision="fp32" if dtypes == {torch.float32} else "mixed",
        notes={
            "weights_file": WEIGHTS_NAME,
            "source_checkpoint": str(a.src),
            "source_sha256": sha256(a.src),
            "head_run": str(a.head),
            "head_git_sha": head_metrics.get("git_sha", "not recorded (trained before gallops)"),
            "generator": str(a.gen) if a.gen else "unchanged from source",
            "replaced_tensors": len(repl),
            "step_is": "optimizer steps of the --head stage up to its kept epoch; earlier stages not counted",
            "kept_epoch": int(head_metrics["best_epoch"]),
            "variant_selection": "the variant (heads, selection rule, policy switches) was chosen by "
            "closed-loop driving score on the same 50 Bench2Drive routes its headline is reported on; only "
            "the kept epochs were chosen on held-out data",
            "policy_env": policy_env,
            **({"policy_env_files": ["DRIVEJEPA_SPLIT_AREA"]} if a.area else {}),
            **input_notes(a.data_manifest, a.holdout_list),
            "note": a.note,
        },
    )

    def save(staging: Path) -> dict:
        torch.save(ck, staging / WEIGHTS_NAME)
        gap = probe(staging / WEIGHTS_NAME, head_sd, a.head / "probe.npz")
        notes: dict = {"probe": f"reloaded head == trained head on 4 held-out rows (max |diff| {gap:.1e})"}
        if a.area:
            shutil.copyfile(a.area / "head_best.pt", staging / AREA_HEAD_FILE)
            copied = sha256(staging / AREA_HEAD_FILE)
            if copied != sha256(a.area / "head_best.pt"):
                raise SystemExit(f"{AREA_HEAD_FILE}: the copy differs from {a.area / 'head_best.pt'}")
            notes["area_head"] = {
                "file": AREA_HEAD_FILE,
                "sha256": copied,
                "run": str(a.area),
                "kept_epoch": area_epoch,
            }
        return notes

    record = publish(out_dir, record, WEIGHTS_NAME, save)
    dst = out_dir / WEIGHTS_NAME
    print(
        f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB), replaced {len(repl)} tensors, probe OK, sha256 "
        f"{record.weights_sha256[:16]}...",
        flush=True,
    )


if __name__ == "__main__":
    main()
