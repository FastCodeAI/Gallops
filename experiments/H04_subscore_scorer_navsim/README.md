# H04 — Sub-score scorer: the sub-score agent-look model and its agent decoder

**Owner:** Navvrat, with Siddesh · **Priority:** 4 of 11 · **Started:** 22 Oct 2026 · **Status:** running — gated by hand,
overall HOLD; no scorer-only arm yet, and no paddock ref carries its adapter (hypothesis)

The facts are in [experiment.toml](experiment.toml), and what is off the repository is in
[POINTERS.md](POINTERS.md). This page is the account.

## Why

The 4-camera base trails released Drive-JEPA on navtest, and 72 % of its failing scenes are a wrong pick with a good
plan available. The released recipe trains the scorer on the final score alone; this asks whether training it on the
sub-scores closes the gap.

## What was done

The sub-score agent-look model (`H04.subscore-agent-look`, called Run 3A while it ran) is the 4-camera planner with an agent decoder, an agent look in place of the global look, and 12 scorer heads
trained on the sub-scores. How the code is kept, the stages of the 4-camera planner and the data are in the
[4-camera stack's account](../H01_four_camera_base/README.md); this page is the rest.

| Run | navtest EPDMS, human filter | navtest, CSV score | Bench2Drive, 50 routes, mean DS (2 Hz) | HUGSIM HD-Score, 419 scenarios |
|---|---:|---:|---:|---:|
| Released Drive-JEPA (ViT-L, the authors' checkpoint, same evaluation) | 87.78 | 84.49 | 57.60 | 33.8 |
| **The 4-camera planner** | **85.27** | **81.98** | 50.07 (18 Oct); 51.70 and 51.95 with the command fix (two runs) | not run |
| **The sub-score agent-look model**, default pick | **87.00** | **83.89** | | |
| **The sub-score agent-look model**, locked pick | **87.30** | **84.09** | 63.29 with the command fix (one run, 26 Oct) | 40.4 |

Recipe: [`recipes/drivejepa_wajepa.toml`](../../recipes/drivejepa_wajepa.toml).

## The agent decoder

The sub-score agent-look model's agent look reads detected agents, so a decoder is trained first, on its
own, from the same token cache (`agent_decoder.py`, 4.25 M parameters for D):
learned slots, each with a reference point in metres, attend to all 2,048
tokens of the four cameras with the token rays as position codes; each layer
moves its reference point, which is the box centre, and predicts class, box and
velocity. Hungarian-matched loss per layer.
Targets: every vehicle, pedestrian and cyclist within 32 m in x and y at time t,
nearest 64 kept. Train = navtrain minus the 51 held-out logs
(`recipes/drivejepa_wajepa/heldout_logs.txt`, sha1(log) % 20 == 0); navtest is
never read.

| Stage | What | Output |
|---|---|---|
| `code-agent-decoder` | upstream + the 4-camera planner's overlay + `agent_decoder.py` and the agent target builder | `{runs}/.../agent_decoder/code` |
| `extract-agents` | the agents of every cached sample, read once from the navtrain logs | `navtrain_agents.pkl` |
| `agent-targets` | `agent_targets.gz` beside each sample's tokens (never overwrites) | in `{data.wajepa_tokens}` |
| `agent-decoder` | AdamW 2e-4, weight decay 0.01, batch 64, fp16, cosine over 12 epochs, gradient clip 35; `QUERIES=300 LAYERS=3` | a checkpoint per epoch |
| `agent-decoder-eval` | held-out metrics of epoch 11 | JSON |

The slot count and depth were chosen by a sweep on the held-out logs (22–23 Oct:
64 × 3, 128 × 3, 64 × 6, 128 × 6, 300 × 3). The pick, **D = 300 slots × 3
layers, epoch 11** (the last of 12): recall 0.497 and precision 0.748 at 2 m,
recall of moving vehicles within 20 m 0.849, on 4,457 held-out samples.

## The sub-score agent-look model

Called Run 3A while it ran. Code and checkpoint paths keep that name (`run3a`, `drive_jepa_run3a/`,
`wajepa_dj_run3a`), so the record points at the files that trained.

Spec: the user's of 23 Oct 2026 (revision 2). One new package,
`drive_jepa_run3a/`, a copy of the 4-camera planner's with these changes (the 4-camera planner's package is
untouched):

| What | Where |
|---|---|
| The agent decoder D loaded whole (queries and layers asserted) and trained on with its detection loss (weight 1) | `drive_jepa_model.py`, `agent_decoder.py` |
| Agent look after each round's refiner, one block shared by the 4 rounds: per waypoint, the k = 16 nearest detections (P(none) < 0.5) moved to the waypoint's time, a relative code in the waypoint's frame, attention with a learned "nothing" slot, `W_o` zero-initialised. The global look is removed | `agent_look.py`, `traj_refiner.py` |
| 12 scorer heads read from each path's 8 waypoint tokens (max over waypoints): gates at-fault collision, drivable area, direction, traffic light, rear safety; goals progress, TTC, lane keeping, comfort, route; edge distance; final score. BCE, smooth L1 for progress and edge distance | `score_module/scorer.py`, `drive_jepa_agent.py` |
| New labels per candidate: rear safety (a lane change that makes a car behind brake harder than 4 m/s²; IDM), edge distance, route share, progress over the 32 paths and PDM-Closed together; map auxiliary targets per waypoint | `score_module/run3a_labels.py`, `score_module/compute_navsim_score.py` |
| The pick: Σ gate weight × log σ(gate) + log of the weighted mean of σ(goals), blended with log σ(final score) | `drive_jepa_model.py` (`pick_score`), `drive_jepa_config.py` |
| Validation on the held-out logs, epoch means; `val/score_epoch` chooses the checkpoint | `planning/training/agent_lightning_module.py` |
| Train = navtrain minus the held-out logs (1,141 logs, 98,831 samples), val = the 51 held-out logs | `config/training/run3a_log_split.yaml`, `run3a_training.yaml` |

| Stage | What | Output |
|---|---|---|
| `code-run3a` | upstream + the sub-score agent-look model's overlay | `{runs}/.../run3a/code` |
| `run3a-checks` | checks 1–3 of the spec on synthetic inputs: the relative transform, the k-nearest selection, the empty list | exits non-zero on a failure |
| `lane-centrelines` | every lane centreline and the lane graph of the four nuPlan maps, for the labels | one pickle per map |
| `teacher-rearok` | the teacher list: Drive-JEPA's v2 anchors minus those that fail rear safety (with the lane gate). 24 Oct: 17,257,415 → 17,201,172 anchors | beside the metric cache |
| `run3a-train` | 20 epochs, batch 64, Adam 1e-4, 16-mixed, seed 0; planner from scratch; refuses to start without the decoder checkpoint and the centrelines | every epoch's checkpoint |
| `run3a-pick-tune` | on epoch 13: the 12 head outputs for every held-out sample, then a grid over the gate weights, the route weight and the blend | `grid.json` |
| `run3a-navtest-default`, `run3a-navtest` | navtest under the defaults the checkpoint was chosen with, and under the locked pick | navsim's CSVs |
| `run3a-navtest-score` | the two navtest scores and every sub-score | `navtest.json` |
| `run3a-checkpoint` | epoch 13 into `{weights}/wajepa_dj_run3a/run3a/step_<n>/` with `checkpoint.json` | |

Selection and the pick, all on the 51 held-out logs: `val/score_epoch` was best
at epoch 13 (0.9173; epoch 19: 0.9102) under the default pick (gates 1, route 2,
no blend). The grid on epoch 13 then gave gates (0.5, 0.5, 1, 0.5, 0.5), route
2, blend 0.5: 0.9204 against the default's 0.9174 (best of 32: 0.9611). Those
weights are locked in the overlay's `drive_jepa_config.py`.

Training: 1,544 steps an epoch at about 2.0 s a step, 8.3 GiB of VRAM, 13.22 M
trained parameters. Loss shares over the first 50 steps: trajectory 52 %,
detection 26 %, the 12 heads 16 %, map auxiliary 1 %.

navtest sub-scores, the sub-score agent-look model locked / the 4-camera planner: NC .986 / .980, DAC .984 / .972,
DDC .993 / .990, TLC .998 / .998, EP .900 / .879, TTC .981 / .973, LK .977 / .967,
HC .977 / .977, EC .406 / .401. Paired per scene: +2.03 [+1.65, +2.44] over the 4-camera planner,
−0.49 [−0.84, −0.11] against the release.

Bench2Drive (26 Oct, 50 routes, 2 Hz, the command fix, one run per route): DS
63.29, RC 79.02, 30 routes completed; the 4-camera planner in the same setup 51.70 / 51.95 (two
runs). HUGSIM (419 scenarios, command only): HD-Score 40.4 against 33.8 for the
release, paired +6.6 [+3.6, +9.7]; KITTI-360 and Waymo have no rear camera, so
The sub-score agent-look model drove those with a black rear image.

## The regression gate

Run by hand from local paddock worktrees, because no paddock ref carries the adapter. Each block is copied from the
gate's own table into [experiment.toml](experiment.toml). The sub-score agent-look model with the locked pick:

| Date | Benchmark | Metric | Released | The sub-score agent-look model | t | Units | Verdict | Against the 4-camera planner |
|---|---|---|---:|---:|---:|---:|---|---|
| 25 Oct | NAVSIM, 1,000 tokens | EPDMS | 0.875 | 0.879 | 0.55 | 136 | NEUTRAL | 0.863 → 0.879, t 1.89, NEUTRAL |
| 25 Oct | KITScenes | ADE | 3.222 | 2.159 | 5.54 | 140 | GAIN | 1.969 → 2.159, t −1.62, NEUTRAL |
| 25 Oct | nuScenes | L2 | 2.497 | 1.914 | 2.98 | 30 | GAIN | 1.956 → 1.914, t 0.28, NEUTRAL |
| 25 Oct | Bench2Drive, 7 routes, 2 Hz | DS | 58.801 | 60.391 | 0.13 | 7 | NEUTRAL | 62.757 → 60.391, t −0.23, NEUTRAL |
| 27 Oct | neuroNCAP, 15 pairs | NCAP | 3.679 | 3.691 | 0.05 | 15 | NEUTRAL | 3.280 → 3.691, t 0.99, NEUTRAL |
| 28 Oct | KITScenes, queue subset | ADE | 2.537 | 2.121 | 2.10 | 117 | GAIN | 1.969 → 2.159, t −1.62, NEUTRAL |
| 28 Oct | nuScenes, queue subset | L2 | 2.431 | 2.151 | 2.19 | 143 | GAIN | 2.057 → 2.151, t −1.06, NEUTRAL |

AlpaSim, 10 scenes: scene score 0.416 → 0.439, t 0.11; the gate says CONFIG DIFFERS, because our driver profile is not
the baseline's. Against the 4-camera planner: 0.450 → 0.439, t −0.08, NEUTRAL. Overall **HOLD** each time: nothing regressed, and no
closed-loop benchmark gained.

## Decision

Not decided. The gate says HOLD, and no scorer-only arm is recorded, so the record stays `running` until the team
takes it.

## Verified, and how

| Claim | Strength |
|---|---|
| The sub-score agent-look model's overlay is the code that trained: no file of the package, its configs or the Lightning module changed between the launch (24 Oct 17:34) and the overlay, except `drive_jepa_config.py` (25 Oct: the locked pick); the tree `code-run3a` builds equals the training checkout but for the machine-path defaults | verified, file by file (27 Oct) |
| The agent decoder's two files are the ones D trained with (last changed 22 Oct 23:11, D trained 23 Oct 15:56–17:11) | verified by timestamps and the code-agent-decoder tree's diff |
| `navtest-score` gives the sub-score agent-look model's 87.00 / 83.89 and 87.30 / 84.09 | verified (27 Oct) |
| The ported scripts of the sub-score agent-look model and the agent decoder compute what the originals did | verified: their syntax trees equal the originals' but for the edits each docstring names (paths to arguments or the environment, the data module's import, an exit code, one renamed variable) |
| Checks 4–7 of the sub-score agent-look model's spec (W_o = 0 gives identical outputs, overfit 64 scenes, the first 50 steps, the lane-change labels) | ran on 24 Oct (numbers in this doc); their scripts are not in this repo |

## Before quoting a number from these checkpoints

- **The sub-score agent-look model's pick was tuned on the same 51 logs that chose its epoch**, which
  are navtrain logs, not navtest; its navtest numbers were not used to choose
  anything.
