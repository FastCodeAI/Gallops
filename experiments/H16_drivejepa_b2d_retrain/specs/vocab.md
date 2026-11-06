# CLAUDE.md: Bench2Drive Drive-JEPA — vocabulary from Bench2Drive expert paths

(Given by Navvrat on 05 Nov 2026, verbatim.)

**What:** k-means on the Bench2Drive expert's 4 s paths, at six sizes: 256, 512, 1,024, 2,048, 4,096 and 8,192. Navvrat picks one for Phase 2, where the vocabulary supplies the teacher paths. Drive-JEPA built its 8,192 the same way from NAVSIM paths.
- CPU only; the GPU stays with run 3A-M.
- Report numbers only, no thresholds or verdicts: Navvrat decides.

## Fixed (ask before changing)

| | |
|---|---|
| **Clips** | The LoRA run's training clips: the 944, minus its held-out list (`b2d_drivejepa/phase1/`). So never the 220 eval routes, the v0.0.4 validation routes or the 24 trigger twins |
| **Paths** | One per 2 Hz frame with a full 4 s of future: the expert's future from Phase 0a's labels. Report how many frames are skipped |
| **Format** | Drive-JEPA's `data/8192.npy`: the same dtype, pose layout and timing (NAVSIM's: 8 poses of x, y, heading, 0.5 s apart). Report what you find |
| **Frame** | Drive-JEPA's: ego at t, rear axle, x forward, y left, yaw counter-clockwise. Phase 0a's labels already flip CARLA's axes; the rear-axle shift is in run 3A Phase 1's `phase1/data/geometry.py`. Import it, apply each once, and report what you applied |
| **Clustering** | k-means on the 8 (x, y) points, with the same paths and settings at every size. Each centre's heading is the circular mean of its members'. k-means++ start, fixed seed |
| **Output** | `b2d_<size>.npy` for each size, in a new `b2d_drivejepa/vocab/` folder, each with its sha256; the list of frames used; the script |

## Checks

1. **Loads:** each file loads where Drive-JEPA loads `8192.npy`. Report any place in Drive-JEPA's code that assumes 8,192 paths.
2. **Handedness:** a known left turn curves toward +y, in the expert path and in its nearest vocabulary path.

## Report, one row per size

- **Groups:** paths per group (min, median, max), and how many groups hold a single path.
- **Coverage:** for each expert path, the distance to its nearest vocabulary path (ADE over the 8 points, in m): mean, median, p95, max. Separately for the training paths and for the LoRA run's held-out clips. Add Drive-JEPA's NAVSIM 8,192 as one more row. Per scenario type: a CSV.
- **Speed:** the vocabulary's top mean and peak speed, next to the expert's p99 and max.

## Rules

- Never use Bench2Drive's 220 eval routes, the v0.0.4 validation routes, or HUGSIM scenes.
- Import Phase 0a's and run 3A Phase 1's code; don't edit it. New files in `b2d_drivejepa/vocab/`.
- Never overwrite a checkpoint, cache or earlier result; write to new paths.
- Log in `agent_memory/<date>.md`. Ask before downloading or installing anything, and before changing anything in this file.
