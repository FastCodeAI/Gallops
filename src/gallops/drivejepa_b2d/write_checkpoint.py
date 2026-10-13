"""Stage split-v4-checkpoint: write a deployable Drive-JEPA checkpoint and its sidecar.

Copies --src, replaces the 10 scorer-head tensors (scorer.pred_score.mlp.{0,1,3,4,6}.{weight,bias}, matched
under the Lightning prefix `agent._pad_model.`), saves to
{weights}/<model>/<train_run>/step_<n>/drive_jepa_vitl_v2.ckpt and writes checkpoint.json beside it in the
same function (docs/checkpoints.md). Never overwrites: refuses if the step directory already holds a
checkpoint. Split-v4 = the released checkpoint + the split-v4aug head; the step is the head's kept epoch.

Probe: the written file is reloaded and its head must give exactly the column-5 scores the trained head gave
on the 4 held-out probe rows saved at training time. The full-forward probe through paddock's policy is
paddock's job.

Ported from paddock-ops scorer-retrain/write_scorer_checkpoint.py, which wrote the same tensors with a
hand-written sidecar beside the file.

    python -m gallops.drivejepa_b2d.write_checkpoint --src <ckpt> --head <dir> --model drivejepa_scorer_b2d
        --train-run <name> [--note "..."]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gallops.checkpoint import Checkpoint, TrainedOn, write
from gallops.config import SiteConfig
from gallops.drivejepa_b2d.head import HEAD_SUFFIXES, head_module
from gallops.drivejepa_b2d.provenance import git_sha

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def replacements(head_sd: dict) -> dict:
    repl = {}
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
    ap.add_argument("--model", required=True, help="paddock adapter name, e.g. drivejepa_scorer_b2d")
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    head_metrics = json.load(open(a.head / "metrics.json"))
    step = int(head_metrics["best_epoch"])
    out_dir = SiteConfig.load().checkpoint_dir(a.model, a.train_run, step)
    dst = out_dir / WEIGHTS_NAME
    if dst.exists() or dst.resolve() == a.src.resolve():
        raise SystemExit(f"refusing: {dst} exists or is the source")

    head_sd = torch.load(a.head / "head_best.pt", map_location="cpu")
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    repl = replacements(head_sd)
    missing = [k for k in repl if k not in sd]
    if missing:
        raise SystemExit(f"keys not in the checkpoint: {missing[:5]}")
    for k, v in repl.items():
        if tuple(sd[k].shape) != tuple(v.shape):
            raise SystemExit(f"{k}: shape {tuple(sd[k].shape)} vs trained {tuple(v.shape)}")
        sd[k] = v.to(sd[k].dtype).clone()
    if len(repl) != 10:
        raise SystemExit(f"replaced {len(repl)} tensors, expected 10")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    gap = probe(dst, head_sd, a.head / "probe.npz")
    dtypes = {v.dtype for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()}
    record = Checkpoint(
        model=a.model,
        step=step,
        train_git_sha=git_sha(),
        train_run=a.train_run,
        trained_on=TRAINED_ON,
        selection_split="bench2drive/base-heldout-150",
        precision="fp32" if dtypes == {torch.float32} else "mixed",
        weights_sha256=sha256(dst),
        notes={
            "weights_file": WEIGHTS_NAME,
            "source_checkpoint": str(a.src),
            "source_sha256": sha256(a.src),
            "head_run": str(a.head),
            "head_git_sha": head_metrics.get("git_sha", "not recorded (trained before gallops)"),
            "replaced_tensors": len(repl),
            "probe": f"reloaded head == trained head on 4 held-out rows (max |diff| {gap:.1e})",
            "step_is": "the split head's kept epoch",
            "variant_selection": "the variant (heads, selection rule, policy switches) was chosen by "
            "closed-loop driving score on the same 50 Bench2Drive routes its headline is reported on; only "
            "the kept epochs were chosen on held-out data",
            "note": a.note,
        },
    )
    write(out_dir, record)
    print(
        f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB), replaced {len(repl)} tensors, probe OK, sha256 "
        f"{record.weights_sha256[:16]}...",
        flush=True,
    )


if __name__ == "__main__":
    main()
