"""Cache what Drive-JEPA's scorer reads, for its 32 plans and for a vocabulary of extra ones.

Every scorer run so far reran the 300 M-parameter encoder on every step, so a run
saw about 2,000 steps at batch 8 and the held-out ranking never beat chance. With
the encoder, the queries and the refiner frozen, everything upstream of the scorer
is a fixed function of the tick, so it is computed once here and training becomes
minutes over arrays.

Per tick, with the released weights:

* **pass 1** runs the unchanged model: the 32 native plans, their scores, and the
  per-waypoint tokens the scorer reads (32 x 8 x 256);
* the **vocabulary** (``--vocab``, a file defining
  ``vocabulary(natives, pick_index, speed, accel) -> (paths, meta)``) is built from
  what the model has at drive time;
* **pass 2** reruns only the refiner's last stage -- its query grid is fixed at
  32 x 8, so extra plans cannot be appended -- with each vocabulary path in the slot
  of the native plan nearest it (one-to-one), so the refiner samples the image
  along the new path and its memory comes from a similar plan. 32 paths per pass.

Two guards run on every batch and are written into each file: putting the native
plans back in their own slots must reproduce pass 1's tokens, and the scorer on the
cached tokens must reproduce the model's own logits.

    python -m gallops.train.scorer_cache --source <labelled dir> <name> [...] \\
        --vocab vocab.py --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gallops.config import SiteConfig
from gallops.train.swerve_data import one_tick_per_pass

SLOTS = 32


def load_vocab(path: Path):
    spec = importlib.util.spec_from_file_location("vocab_under_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vocabulary


def assign_slots(paths: np.ndarray, natives: np.ndarray) -> np.ndarray:
    """For up to 32 paths, the native slot each takes: one-to-one, nearest by mean waypoint distance."""
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(paths[:, None, :, :2] - natives[None, :, :, :2], axis=-1).mean(-1)
    rows, cols = linear_sum_assignment(cost)
    slots = np.empty(len(paths), dtype=int)
    slots[rows] = cols
    return slots


class LastStage:
    """Captures the inputs and outputs of the refiner's last call during a normal forward."""

    def __init__(self, model):
        self.refiner = model._trajectory_head[0]
        self.calls_per_forward = len(model._trajectory_head)
        assert all(r is self.refiner for r in model._trajectory_head), "refiner is no longer shared"
        inner = self.refiner.forward
        self.count = 0
        self.captured = {}

        def forward(bev_feature, image_feature):
            self.count += 1
            out = inner(bev_feature, image_feature)
            if self.count % self.calls_per_forward == 0:
                self.captured = {
                    "bev_in": bev_feature,
                    "image": image_feature,
                    "bev_out": out[0],
                    "proposals": out[1],
                }
            return out

        self.refiner.forward = forward

    def rerun(self, proposals):
        c = self.captured
        return self.refiner.Bev_refiner(proposals, c["bev_in"], c["image"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("LABELLED", "NAME"))
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # One tick per pass (TICKS_PER_PASS in swerve_data): the identity and scorer guards
    # below rerun inside the same pass, so they cannot see ticks mixed across a batch.
    parser.add_argument("--batch", type=one_tick_per_pass, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-episodes", type=int, default=0, help="smoke: first N episodes per source")
    args = parser.parse_args()

    cfg = SiteConfig.load()
    for name, template in (
        ("NAVSIM_DEVKIT_ROOT", "{data.drivejepa_repo}"),
        ("NAVSIM_EXP_ROOT", "{data.navsim.exp}"),
        ("OPENSCENE_DATA_ROOT", "{data.navsim}"),
        ("NUPLAN_MAPS_ROOT", "{data.navsim}/maps"),
    ):
        os.environ.setdefault(name, cfg.resolve(template))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")

    from gallops.train.swerve import build_model, load_ticks
    from gallops.train.swerve_data import TickDataset

    vocabulary = load_vocab(args.vocab)
    out_root = args.out.resolve()
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for labelled, name in args.source:
        ticks = load_ticks(Path(labelled), blind=True)  # keep all; usability is a label, decided at training
        for tick in ticks:
            groups[(name, tick["episode"])].append(tick)
    if args.limit_episodes:
        kept, per = {}, defaultdict(int)
        for key in sorted(groups):
            if per[key[0]] < args.limit_episodes:
                kept[key] = groups[key]
                per[key[0]] += 1
        groups = kept

    checkpoint = cfg.resolve(
        "{data.navsim.exp}/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
    )
    agent, model, torch = build_model(
        checkpoint, args.device, cfg.resolve("{data.drivejepa_repo}"), train_parts=("__nothing__",)
    )
    model.eval()
    stage = LastStage(model)
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {"sources": args.source, "vocab": str(args.vocab), "checkpoint": checkpoint, "slots": SLOTS},
            indent=1,
        )
    )

    # One dataset for every tick: it caches each scene's map features and decoded
    # frames, and rebuilding it per episode recomputed them every time.
    order = sorted(groups)
    all_ticks = [t for key in order for t in sorted(groups[key], key=lambda t: t["timestamp"])]
    dataset = TickDataset(all_ticks, args.device)
    offsets, at = {}, 0
    for key in order:
        offsets[key] = (at, at + len(groups[key]))
        at += len(groups[key])

    started, done = time.time(), 0
    for name, episode in order:
        target = out_root / name / (episode.replace("/", "__") + ".npz")
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        first, last = offsets[(name, episode)]
        ticks = all_ticks[first:last]
        rec = defaultdict(list)
        identity_drift = scorer_drift = 0.0
        for start in range(first, last, args.batch):
            chunk = list(range(start, min(start + args.batch, last)))
            features, _, used = dataset.batch(chunk)
            if features is None:
                continue
            with torch.no_grad():
                output = model(features)
                cap = stage.captured
                proposals = cap["proposals"]  # (B, 32, 8, 3)
                b = proposals.shape[0]
                tokens = cap["bev_out"].reshape(b, SLOTS, -1, cap["bev_out"].shape[-1])  # (B, 32, 8, 256)
                again = stage.rerun(proposals).reshape(tokens.shape)
                identity_drift = max(identity_drift, float((again - tokens).abs().max()))
                logit = model.scorer(proposals, cap["bev_out"])[0]
                scorer_drift = max(scorer_drift, float((logit - output["pred_logit"]).abs().max()))

                natives = proposals.float().cpu().numpy()
                score = output["pdm_score"].float().cpu().numpy()
                built = []
                for i, tick in enumerate(used):
                    paths, meta = vocabulary(
                        natives[i], int(score[i].argmax()), float(tick["speed"]), float(tick["accel"])
                    )
                    built.append((np.asarray(paths, dtype=np.float32), meta))
                n_vocab = len(built[0][0])
                assert n_vocab % SLOTS == 0, f"vocabulary size {n_vocab} is not a multiple of {SLOTS}"
                vocab_tokens = np.zeros((b, n_vocab, tokens.shape[2], tokens.shape[3]), dtype=np.float16)
                vocab_slots = np.zeros((b, n_vocab), dtype=np.int16)
                for k in range(0, n_vocab, SLOTS):
                    swapped = proposals.clone()
                    slot_rows = []
                    for i, (paths, _) in enumerate(built):
                        slots = assign_slots(paths[k : k + SLOTS], natives[i])
                        swapped[i, torch.as_tensor(slots, device=swapped.device)] = torch.as_tensor(
                            paths[k : k + SLOTS], device=swapped.device, dtype=swapped.dtype
                        )
                        slot_rows.append(slots)
                    redone = stage.rerun(swapped).reshape(tokens.shape)
                    for i, slots in enumerate(slot_rows):
                        vocab_tokens[i, k : k + SLOTS] = (
                            redone[i, torch.as_tensor(slots, device=redone.device)].half().cpu().numpy()
                        )
                        vocab_slots[i, k : k + SLOTS] = slots

            rec["timestamp"].append(np.array([t["timestamp"] for t in used], dtype=np.int64))
            rec["speed"].append(np.array([t["speed"] for t in used], dtype=np.float32))
            rec["accel"].append(np.array([t["accel"] for t in used], dtype=np.float32))
            rec["command"].append(np.array([t.get("command", 0) for t in used], dtype=np.int8))
            rec["usable"].append(np.array([bool(t.get("usable_for_imitation", True)) for t in used]))
            rec["in_frame"].append(np.array([bool(t.get("hazard_in_frame")) for t in used]))
            rec["natives"].append(natives.astype(np.float32))
            rec["pdm_score"].append(score.astype(np.float32))
            rec["pred_logit"].append(output["pred_logit"].float().cpu().numpy())
            rec["native_tokens"].append(tokens.half().cpu().numpy())
            rec["vocab"].append(np.stack([p for p, _ in built]))
            rec["vocab_tokens"].append(vocab_tokens)
            rec["vocab_slots"].append(vocab_slots)
            meta_names = [json.dumps(m, sort_keys=True) for m in built[0][1]]
        if not rec:
            print(f"  {name}/{episode}: no rebuildable ticks", flush=True)
            continue
        arrays = {k: np.concatenate(v) for k, v in rec.items()}
        tmp = target.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            **arrays,
            vocab_meta=np.array(meta_names),
            frame_dir=np.array(str(Path(ticks[0]["frame"]).parent.parent)),
            identity_drift=np.float32(identity_drift),
            scorer_drift=np.float32(scorer_drift),
        )
        tmp.rename(target)
        done += 1
        dataset._frames.clear()  # decoded frames are per episode; keep memory flat
        print(
            f"  {name}/{episode}: {len(arrays['timestamp'])} ticks, identity drift {identity_drift:.2e}, "
            f"scorer drift {scorer_drift:.2e}, {(time.time() - started) / done:.0f} s/episode",
            flush=True,
        )


if __name__ == "__main__":
    main()
