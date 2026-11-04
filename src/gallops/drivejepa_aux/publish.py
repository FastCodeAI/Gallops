"""The served checkpoint of an aux run: the release with its refiner path replaced, the heads beside it.

    {weights}/drivejepa_aux/<train_run>/step_<n>/
        drive_jepa_vitl_v2.ckpt   the release's file, its refiner-path tensors replaced
        aux_heads.pt              the heads, which inference never runs (T arms only)
        checkpoint.json

**Nothing changes at inference**, so nothing new serves it: paddock's `drivejepa` adapter loads the
file through DRIVEJEPA_CHECKPOINT, as it loaded the Physical AI and Bench2Drive retrains. In eval
mode upstream's heads do not run and ours exist only in the trainer, so the heads stay out of the
served file and go beside it.

**The one shared refiner is stored four times** in the release's file, once per refinement pass
(`_trajectory_head.{0,1,2,3}`); a run trains it once, under `_trajectory_head.0.`, and every copy
gets it -- 168 tensors from 42, plus the query embedding and the ego encoder: 171 in all, the count
`drivejepa_b2d.write_checkpoint` replaces for its generator. Torch-free apart from `write`, which
`sidecar.publish` makes atomic: the step directory appears with its sidecar, or not at all.
"""

from __future__ import annotations

from pathlib import Path

#: Prefixes of upstream's switched-on heads inside the model, the only keys the release lacks.
UPSTREAM_HEAD_KEYS = ("scorer.map_head.", "scorer._agent_head.")
MODEL = "drivejepa_aux"
WEIGHTS_FILE = "drive_jepa_vitl_v2.ckpt"
AUX_HEADS_FILE = "aux_heads.pt"
#: Keys in the release's Lightning file sit under this prefix.
PREFIX = "agent._pad_model."
REFINER_PASSES = 4
SHARED_REFINER = "_trajectory_head.0."
#: What the served file takes from a run: the refiner path. Nothing else trains.
SERVED = ("init_feature.", "hist_encoding.", SHARED_REFINER)


def served_names(key: str) -> list[str]:
    """Where one trained tensor goes in the release's file: every pass's copy of the refiner."""
    if key.startswith(SHARED_REFINER):
        rest = key[len(SHARED_REFINER) :]
        return [f"{PREFIX}_trajectory_head.{i}.{rest}" for i in range(REFINER_PASSES)]
    return [PREFIX + key]


def replacements(trained: dict) -> dict:
    """{name in the file: tensor} for every trained refiner-path tensor; the heads are not served."""
    out = {}
    for key, value in trained.items():
        if key.startswith(SERVED):
            for name in served_names(key):
                out[name] = value
        elif not key.startswith(UPSTREAM_HEAD_KEYS):
            raise SystemExit(f"{key}: neither the refiner path nor a head; nothing else may train")
    return out


def replace(state_dict: dict, trained: dict) -> int:
    """Put the trained refiner path into the release's state dict, in place. Refuses a name the file
    lacks or a shape it disagrees with; clones, so the four copies do not share one storage."""
    repl = replacements(trained)
    missing = [k for k in repl if k not in state_dict]
    if missing:
        raise SystemExit(f"not in the released file: {missing[:5]}")
    for name, value in repl.items():
        if tuple(state_dict[name].shape) != tuple(value.shape):
            raise SystemExit(
                f"{name}: released shape {tuple(state_dict[name].shape)}, trained {tuple(value.shape)}"
            )
        state_dict[name] = value.to(state_dict[name].dtype).clone()
    return len(repl)


def write(out_dir: Path, record, release: Path, trained: dict, probe) -> object:
    """Write the step directory whole: weights, heads and sidecar. `probe(weights_path)` reloads the
    written file the way paddock does and returns what it found; it raises if the file is not the
    model that trained."""
    import torch

    from gallops.drivejepa_b2d.sidecar import publish, sha256

    def save(staging: Path) -> dict:
        checkpoint = torch.load(release, map_location="cpu")
        replaced = replace(checkpoint["state_dict"], trained["model"])
        # The state dict alone. The release's file also carries its own Lightning training state --
        # optimizer_states, loops, callbacks: 2.5 of its 3.7 GB -- which describes the release's
        # training, not this run's, and which nothing that serves it reads (the agent's `initialize`
        # takes ["state_dict"]).
        torch.save({"state_dict": checkpoint["state_dict"]}, staging / WEIGHTS_FILE)
        dropped = sorted(k for k in checkpoint if k != "state_dict")
        notes = {"replaced_tensors": replaced, "weights_file": WEIGHTS_FILE, "release_keys_dropped": dropped}
        heads = {k: v for k, v in trained["model"].items() if k.startswith(UPSTREAM_HEAD_KEYS)}
        if heads or trained.get("scene_heads"):
            torch.save(
                {"upstream": heads, "scene_heads": trained.get("scene_heads")}, staging / AUX_HEADS_FILE
            )
            notes["aux_heads"] = {"file": AUX_HEADS_FILE, "sha256": sha256(staging / AUX_HEADS_FILE)}
        notes["probe"] = probe(staging / WEIGHTS_FILE)
        return notes

    return publish(out_dir, record, WEIGHTS_FILE, save)
