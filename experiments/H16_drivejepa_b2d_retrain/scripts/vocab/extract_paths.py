"""Step 1 of the Bench2Drive path vocabularies (b2d_drivejepa/vocab/CLAUDE.md): read Phase 0a's labels once, cache the expert paths.

One path per 2 Hz frame with a full 4 s of future, for the clips in phase1/splits/clips.json only (train and heldout; never any
other clip). Source is Phase 0a's labels, not phase1-targets-v2: Phase 1's packed targets drop the first frames of each clip.

Frame: Phase 0a's `ego.future` is the actor origin in the ego frame at t, already y left / yaw counter-clockwise (labels_b2d.ego_pose
flips CARLA once). Nothing here flips anything. The only change applied is the rear-axle shift, once, on all 40 poses, by importing
Phase 1's rear_axle_poses (geometry.B2D_REAR_AXLE_OFFSET).

Timing: future[j] is frame n + 1 + j at 10 Hz, so index 0 is t + 0.1 s and index 39 is t + 4.0 s; [4::5] is 0.5 .. 4.0 s.

Writes (refuses to overwrite): paths_cache.npz, frames_used.csv, extract_stats.json."""
import argparse
import collections
import csv
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "{ops}/phase0a-labels")
sys.path.insert(0, "{ops}/run3a-adapted")
import labels_b2d as lb                                   # noqa: E402  Phase 0a's loader
from phase1.data import geometry as G                     # noqa: E402  Phase 1's rear-axle constant
from phase1.data.targets import rear_axle_poses           # noqa: E402  Phase 1's rear-axle shift

SPLITS = "{ops}/b2d_drivejepa/phase1/splits/clips.json"
LABELS = "{data.b2d_clips}/labels_b2d_v4/b2d"
HERE = os.path.dirname(os.path.abspath(__file__))
COMMANDS = ["VOID", "LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW", "CHANGELANELEFT", "CHANGELANERIGHT", "NONE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--limit", type=int, default=0, help="first N clips per split, for a test run")
    a = ap.parse_args()
    targets = [f"{a.out}/{n}" for n in ("paths_cache.npz", "frames_used.csv", "extract_stats.json")]
    for t in targets:
        if os.path.exists(t):
            sys.exit(f"STOP: {t} exists, not overwriting")

    sp = json.load(open(SPLITS))
    train, held = sorted(sp["train"]), sorted(sp["heldout"])
    assert not set(train) & set(held) and set(train) | set(held) == set(sp["clips"])
    if a.limit:
        train, held = train[:a.limit], held[:a.limit]
    clips = [(c, 0) for c in train] + [(c, 1) for c in held]

    raw, rear, speed, cmd, cmd_far, clip_id, frame, split = [], [], [], [], [], [], [], []
    rows = []
    n_valid_hist = {0: collections.Counter(), 1: collections.Counter()}
    frames_total = {0: 0, 1: 0}
    interior_gap, nan_in_valid, non5_steps, schemas, clips_no_path = [], 0, 0, collections.Counter(), []
    t0 = time.time()
    for ci, (clip, sid) in enumerate(clips):
        d = lb.load_clip(f"{LABELS}/{clip}.pkl.gz")
        assert d["clip"] == clip
        schemas[d["schema"][:20]] += 1
        idx = [s["frame_idx"] for s in d["samples"]]
        non5_steps += int(sum(b - a_ != 5 for a_, b in zip(idx, idx[1:])))
        kept = 0
        for s in d["samples"]:
            eg = s["ego"]
            frames_total[sid] += 1
            fv = np.asarray(eg["future_valid"], bool)
            fut = np.asarray(eg["future"], np.float64)
            n_valid_hist[sid][int(fv.sum())] += 1
            assert fut.shape == (40, 3) and fv.shape == (40,)
            if not fv.all():
                if fv.sum() and not fv[:int(fv.sum())].all():      # a hole inside the 4 s, not just the clip's end
                    interior_gap.append((clip, int(s["frame_idx"]), int(fv.sum())))
                continue
            if not np.isfinite(fut).all():
                nan_in_valid += 1
                continue
            raw.append(fut.astype(np.float32))
            rear.append(rear_axle_poses(fut))                      # the one change applied, on all 40 poses
            speed.append(float(eg["velocity"][0]))
            c = eg["command_near"]
            cmd.append(COMMANDS.index(c if c in COMMANDS else "NONE"))
            c = eg["command_far"]
            cmd_far.append(COMMANDS.index(c if c in COMMANDS else "NONE"))
            clip_id.append(ci)
            frame.append(int(s["frame_idx"]))
            split.append(sid)
            rows.append((clip, int(s["frame_idx"]), ("train", "heldout")[sid]))
            kept += 1
        if not kept:
            clips_no_path.append(clip)
        del d
        if ci % 50 == 0:
            print(f"{ci + 1}/{len(clips)} clips, {len(rows)} paths, {time.time() - t0:.0f} s", flush=True)

    with open(targets[0], "xb") as fh:
        np.savez(fh, rear=np.stack(rear), raw=np.stack(raw), speed=np.asarray(speed, np.float32), cmd=np.asarray(cmd, np.int8),
                 cmd_far=np.asarray(cmd_far, np.int8), clip_id=np.asarray(clip_id, np.int32), frame=np.asarray(frame, np.int32),
                 split=np.asarray(split, np.int8), clip_names=np.asarray([c for c, _ in clips]), commands=np.asarray(COMMANDS))
    with open(targets[1], "x", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "frame", "split"])
        w.writerows(rows)
    used = {sid: sum(1 for s in split if s == sid) for sid in (0, 1)}
    stats = dict(
        clips=dict(train=len(train), heldout=len(held)), frames_total=dict(train=frames_total[0], heldout=frames_total[1]),
        frames_used=dict(train=used[0], heldout=used[1]),
        frames_skipped=dict(train=frames_total[0] - used[0], heldout=frames_total[1] - used[1]),
        skipped_by_valid_future_steps={("train", "heldout")[k]: {str(n): c for n, c in sorted(v.items()) if n != 40}
                                       for k, v in n_valid_hist.items()},
        holes_inside_future=interior_gap, nan_where_valid=nan_in_valid, sample_steps_not_5_frames=non5_steps,
        clips_with_no_path=clips_no_path, schemas=dict(schemas), rear_axle_offset_m=G.B2D_REAR_AXLE_OFFSET,
        seconds=round(time.time() - t0, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
    with open(targets[2], "x") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "holes_inside_future"}, indent=1))


if __name__ == "__main__":
    main()
