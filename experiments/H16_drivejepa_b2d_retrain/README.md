# H16 — Drive-JEPA's training rebuilt on Bench2Drive

**Owner:** Navvrat · **Started:** 5 Nov 2026 · **Status:** running: the fine-tune is not finished, the 50 routes stopped at
13, and NAVSIM and AlpaSim have no gate result

The facts (runs, data, checkpoints, results, gate verdicts) live in [experiment.toml](experiment.toml) and are rendered
into the repository README. This page is the account: why, what was done, what it showed, and what was decided. Numbers
only; Navvrat decides. What ran and what it was asked for is in [specs/](specs/), what each step reported in
[reports/](reports/), the numbers in [tables/](tables/), the scripts as they ran in [scripts/](scripts/), and what is off
the repository in [POINTERS.md](POINTERS.md).

## Why

Released Drive-JEPA is trained on NAVSIM. On paddock's 50 Bench2Drive routes it reaches a mean DS of 57.60 at 2 Hz; the
paper's 64.52 comes from a Bench2Drive-trained variant that was not released, with no checkpoint or converter
([recon](reports/recon.md)). [GC04](../GC04_bench2drive/CARD.md) reached 73.6 by retraining the scorer and then the
generator, with the encoder frozen, on the 1,000 clips that include same-spot twins of eval routes.

This experiment asks what happens when everything Drive-JEPA's training rests on comes from Bench2Drive: the encoder
adapted by Sid's hide-and-guess ([H07](../H07_encoder_lora/CARD.md)), a vocabulary from the expert's paths, a grade from
Bench2Drive's own penalties, and Stage 2 on those. The eval routes, the v0.0.4 validation routes and the 24 trigger twins
stay out of every step.

It is its own hypothesis by the rule in [docs/experiments.md](../../docs/experiments.md#runs). H07's verdict turns on
AlpaSim and on WA-JEPA's encoder, and nothing here touches either, while these runs also change the planner. GC04 is
phase 1, which is closed.

**Not pre-registered.** The specs of 5 Nov fix the recipe and what to report, not an expected effect, and nothing was
registered here before the runs. The hypothesis in the record is read from those specs, not taken from them.

## What was done

Seven stages, 5 and 6 Nov. Each ran from the training site's own folders; the code is here now, as
[below](#where-the-code-is). The stage names are those of [the recipe](../../recipes/drivejepa_b2d_retrain.toml).

| # | Stage | Inputs | Command as it ran | Output |
|---|---|---|---|---|
| 1 | LoRA, `H16.r1` | released agent `740335c800e2`, e50 `4649182770ef`, 897 clips' front camera | `run_train.sh p1_05nov`, then `merge_pass.py save_0014_pass_03 pass_03` | merged encoder `339d74428a15` |
| 2 | Vocabulary | the expert's 4 s paths on the 897 clips, from Phase 0a's labels | `extract_paths.py`, `build_vocab.py`, `evaluate.py`, `elbow/run_all.sh`, `b2d_grade/grade.py` | `b2d_<K>.npy` at six sizes; 2,048 is `9a39f95a8508` |
| 3 | Step A v2 | the 2,048, all 944 clips' scorer context | `build_context.py`, `blocked_all.py`, `build_inputs.py`, `grade_all.py --shard i 2`, `teachers.py` | `teachers_2048_v2.npz`, `8f1ed0972e6b` |
| 4 | Stage 2, `H16.r2` | stages 1 to 3 | `train_b2d.py --name p2_05nov --micro 64`, then `export_ckpt.py 18` | 20 epochs; `agent.ckpt`, `ed07e8a10f69` |
| 5 | Fine-tune, `H16.r3` | stage 4's inputs and the released planner | `train_b2d.py --name p2ft_06nov --micro 64 --epochs 20 --workers 4 --init-released` | 8 of 20 epochs so far |
| 6 | Closed loop | `agent.ckpt`, paddock's `drivejepa_p2` | `run_lane.sh <lane>` on 3 lanes: `paddock run --model drivejepa_p2 --bench bench2drive --scene <route>`, `PADDOCK_INFER_HZ=2` | **partial**, 13 of 50 routes |
| 7 | Gate | `agent.ckpt`, the queue's baselines | `run_openloop.sh`, `run_closed.sh`, `run_alpasim.sh`, each `python -m tools.gate smoke` | FAIL; NAVSIM and AlpaSim no result |

### 1. LoRA on Bench2Drive (`H16.r1`)

Sid's Experiment 1 recipe: rank 16, alpha 16, on qkv, proj, fc1 and fc2 of all 24 blocks, B starting at zero; his masks,
guesser, slow copy and loss. 1,000 guesser-only steps, then 3 passes of 1,350 steps at batch 32 in bf16, with no stop
rule ([spec](specs/phase1_lora_v3.md)). The input is the front camera, two frames 0.5 s apart, crop `native`: 1600 x 900
resized to 800 x 450, then Drive-JEPA's own `_get_camera_feature`.

| Point | Hide-and-guess loss, training | Held-out |
|---|---:|---:|
| End of warm-up | .49514 | .49556 |
| Pass 1 | .47575 | .47969 |
| Pass 2 | .47005 | .47918 |
| Pass 3 | .46683 | .47869 |

Held-out is a fixed set of 2,000 pairs from the 47 clips. Pass 3 was merged: sha256 `339d7442…5e94`, 1,215,651,437 bytes.
Merged against unmerged on 32 held-out pairs: fp32 largest .0158, mean 1.9e-5. In bf16 the largest is 23.3, which is
rounding (bf16 against fp32 on the released encoder is 24.6). Cosine with the released tokens is .796.

The running process still computed the released planner's ADE each pass, which spec v3 dropped while it was on pass 3. On
2,204 held-out moments the pick ADE went 2.635 (released encoder), 2.659, 2.780, 2.843 m, with closest ADE .710, .706,
.700, .706 m ([table](tables/p1_05nov_reports.csv)). The pass was not chosen on these.

### 2. Vocabulary at six sizes

k-means on the 36,897 training paths at 256, 512, 1,024, 2,048, 4,096 and 8,192 ([spec](specs/vocab.md),
[report](reports/vocab.md)). Files are `(K, 40, 3)` float32 as Drive-JEPA's `data/8192.npy` really is, not the 8 poses the
spec says. Handedness and load checks pass.

The [elbow check](reports/vocab_elbow.md) ran 33 fits (11 sizes x 3 seeds) and found no sharp elbow: held-out error falls
about as K^-0.18. Held-out and training ADE part after 1,024, and entries from one clip go 2.1% at 1,024, 7.3% at 2,048,
50.2% at 8,192. An ADE pick of 1,024 was withdrawn (Navvrat: "ade is not a metric, b2d scores are"), and sizes were
[graded](reports/vocab_b2d_grade.md) instead, on 298 held-out and 299 training frames with an early stand-in grade.

| Size | Held-out with a teacher | Held-out turns with one | Training turns with four or more |
|---|---:|---:|---:|
| 512 | .862 | .451 | .340 |
| 1,024 | .930 | .686 | .574 |
| 2,048 | .956 | .784 | .745 |
| 4,096 | .966 | .843 | .809 |
| 8,192 | .980 | .882 | .915 |
| NAVSIM's 8,192 | .775 | .412 | — |

The 2,048 went forward, sha256 `9a39f95a…7f71`. Navvrat's words were "looks good can we start it", after that report:
neither the size nor pass 3 was named outright, so both are to confirm.

### 3. Step A: the grade on all 39,148 frames

One path over 4 s, with Bench2Drive's coefficients ([spec v2](specs/phase2_v2.md), [report](reports/step_a_v2.md)):

```
S = P x 0.5^n_ped x 0.6^n_veh x 0.65^n_static x 0.7^n_red x 0.8^n_stop x (1 - outside)
```

P is `min(1, d / d_expert)`, and 1 when the expert moves under 0.5 m. Outside is the path's metres off lanes or in
oncoming lanes over the route's length. Teachers are feasible paths with S at or above 0.95, capped at 256 a frame. All
2,048 paths ran on 36,897 training and 2,251 held-out frames: 21,171,222 graded paths.

| | Training | Held-out |
|---|---:|---:|
| Expert reaches 0.95 | 94.4% | 95.1% |
| Expert mean grade | .981 | .983 |
| Expert moves under 0.5 m | 24.5% | 32.1% |
| Frames with a teacher: all / turns | .964 / .864 | .958 / .761 |
| Frames with four or more: all / turns | .916 / .670 | .882 / .518 |

Where the expert falls below 0.95 (2,072 training frames, 110 held-out), the term that fires: vehicle 1,100 / 63, outside
642 / 35, static 453 / 8, pedestrian 41 / 8, stop 11 / 0, red 0 / 8, progress 0 / 0.

**The expert's collisions against its own log** ([check](tables/check_expert_log.txt)). 39,148 frames; frames with a hit:

| | Path, the grade's view | 10 Hz log, same margins | 10 Hz log, real boxes | Path and real | Path, not real |
|---|---:|---:|---:|---:|---:|
| Pedestrian | 49 | 17 | 17 | 17 | 32 |
| Vehicle | 1,131 | 1,121 | 763 | 657 | 474 |
| Static | 461 | 406 | 44 | 44 | 417 |

42 frames carry a vehicle infraction with no contact on the path: the follower rule. The ego box after the rear-axle
shift sits a mean -0.008 m along and -0.000 m across from the anno box centre, largest 0.045 and 0.038 m.

**Three frames where the expert passes through the oncoming lane** ([numbers](tables/three_frames.json)):

| Frame | | Expert S | Best stopping path | Best path | Teachers (stop / go) |
|---|---|---:|---:|---:|---|
| HazardAtSideLaneTwoWays Route1147, 18000, 3.1 m/s | v1 | .054 | .537 | 1.000 | 1 (0 / 1) |
| | v2 | .781 | .588 | .966 | 1 (0 / 1) |
| AccidentTwoWays Route1103, 30500, 3.5 m/s | v1 | .226 | .623 | .820 | 0 |
| | v2 | .822 | .568 | .938 | 0 |
| VehicleOpensDoorTwoWays Route1203, 21500, 3.3 m/s | v1 | .248 | 1.000 | 1.000 | 13 (1 / 12) |
| | v2 | .930 | 1.000 | 1.000 | 22 (3 / 19) |

**v1, superseded.** [v1](specs/phase2_v1.md) measured progress against the best clean path and outside lanes as a share of
the 4 s path. Passing through the oncoming lane then cost the expert more than stopping, as the table shows, so spec v2
changed both, moved the stop-sign threshold to 1.0 m/s, and added the log check. v1's [report](reports/step_a_v1.md) and
its teacher index `c065c90c5ae4` stand as a record; nothing trained on them.

### 4. Stage 2 from scratch (`H16.r2`, `p2_05nov`)

Drive-JEPA's Stage 2 with its own model, losses and optimiser, changed only where Bench2Drive needs it: Adam 1e-4
(backbone 1e-5), no scheduler, 20 epochs, batch 64, 16-mixed. Generator, scorer and projector start from scratch and
the encoder from `H16.r1`. The scorer's labels come from its 32 live proposals, graded every step by the Step A grade. Its
columns map as NC = collision factors, DAC = 1 - off-lane share, TL = 0.7^n_red, DDC = 1 - oncoming share, EP = P,
LK = 1 - outside, final = S; TTC and EC have no term and stay 1. Ego status is as the runner sends it: speed (v, 0),
acceleration (dv/dt, 0), command with FOLLOW sent as straight. Training mix: straight .945, left .037, right .018.

Before the long run, a path graded in the loop matched Step A on 6,144 paths (largest S difference 1.8e-7), the crop was
bit-identical to Phase 1's, 50 steps ran at 1.44 s a step and 11.28 GiB, and a resume repeated the first loss exactly
(9.94492). 5 Nov 22:39 to 6 Nov 03:40, exit 0. On the 47 held-out clips, 2,251 frames (expert grade .9827 throughout):

| Epoch | Loss | Closest ADE (m) | Pick ADE (m) | Pick grade | Best of 32 | Variety (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.119 | 0.4414 | 3.0270 | 0.7950 | 0.9244 | 2.996 |
| 2 | 2.622 | 0.4140 | 3.3203 | 0.8237 | 0.9480 | 2.812 |
| 3 | 2.325 | 0.3079 | 2.5699 | 0.8495 | 0.9667 | 2.963 |
| 4 | 2.207 | 0.2952 | 2.6744 | 0.8559 | 0.9690 | 2.831 |
| 5 | 2.155 | 0.2861 | 2.5671 | 0.8770 | 0.9700 | 2.931 |
| 6 | 2.128 | 0.2819 | 2.3132 | 0.8634 | 0.9724 | 3.001 |
| 7 | 2.080 | 0.2803 | 2.4480 | 0.8748 | 0.9709 | 2.761 |
| 8 | 2.040 | 0.2752 | 2.6723 | 0.8767 | 0.9722 | 3.109 |
| 9 | 2.134 | 0.3061 | 2.6863 | 0.8660 | 0.9688 | 3.015 |
| 10 | 2.006 | 0.2678 | 2.4350 | 0.8875 | 0.9726 | 3.002 |
| 11 | 2.002 | 0.2866 | 2.8287 | 0.8856 | 0.9720 | 2.874 |
| 12 | 2.040 | 0.2720 | 2.2916 | 0.8730 | 0.9711 | 2.601 |
| 13 | 2.024 | 0.2735 | 2.5066 | 0.8862 | 0.9719 | 2.993 |
| 14 | 1.976 | 0.2583 | 2.5515 | 0.8820 | 0.9772 | 3.022 |
| 15 | 1.961 | 0.2487 | 2.3583 | 0.8924 | 0.9733 | 2.952 |
| 16 | 1.940 | 0.2534 | 2.4491 | 0.8921 | 0.9712 | 2.700 |
| 17 | 1.923 | 0.2453 | 2.2785 | 0.8779 | 0.9785 | 2.745 |
| **18** | 1.916 | 0.2398 | 2.2316 | 0.8937 | 0.9805 | 2.613 |
| 19 | 1.948 | 0.2506 | 2.0518 | 0.8909 | 0.9721 | 2.739 |
| 20 | 1.915 | 0.2462 | 2.5292 | 0.8873 | 0.9735 | 2.785 |

All losses: [p2_05nov_epochs.csv](tables/p2_05nov_epochs.csv). Highest pick grade and best of 32, and lowest closest ADE, at
epoch 18; lowest pick ADE at epoch 19; lowest loss at epoch 20. No epoch was named for the evaluations, so epoch 18 was
exported: sha256 `ed07e8a1…a27a`, 1,245,093,039 bytes. It was picked on these same clips, so its row carries that
selection.

### 5. Fine-tune from the released planner (`H16.r3`, `p2ft_06nov`): not finished

Navvrat: "instead of training the refiner and planner from scratch we fine tune it over b2d score using the same pseudo
trajectory". As stage 4, with generator, refiner, scorer and projector from the released checkpoint and `H16.r1`'s encoder
laid over them. Everything else is the same, data order included.

It was **stopped by hand** with the box's other processes during epoch 9, at about 15:07 on 6 Nov, and 8 of 20 epochs had
saved. It was **resumed from `ckpt_epoch08` at 15:33** in a 14 GB memory scope, so the run is not closed: read these as
its first 8 epochs, not as its result. Held-out as above:

| Epoch | Loss | Closest ADE (m) | Pick ADE (m) | Pick grade | Best of 32 | Variety (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.358 | 0.3270 | 2.4483 | 0.8270 | 0.9771 | 3.107 |
| 2 | 2.181 | 0.2878 | 2.6881 | 0.8504 | 0.9837 | 3.087 |
| 3 | 2.116 | 0.2704 | 2.5810 | 0.8675 | 0.9851 | 3.013 |
| 4 | 2.096 | 0.2880 | 2.6433 | 0.8726 | 0.9725 | 3.071 |
| 5 | 2.034 | 0.2716 | 2.6327 | 0.8803 | 0.9799 | 2.966 |
| 6 | 2.055 | 0.2742 | 2.6686 | 0.8850 | 0.9827 | 2.839 |
| 7 | 1.970 | 0.2497 | 2.7588 | 0.8903 | 0.9888 | 3.093 |
| 8 | 1.971 | 0.2618 | 2.6017 | 0.8917 | 0.9737 | 2.774 |

All losses: [p2ft_06nov_epochs.csv](tables/p2ft_06nov_epochs.csv). No checkpoint was exported, and nothing here has driven.

### 6. Closed loop for epoch 18: partial

**This is not a score.** The run on the 50 routes was stopped by hand with 13 finished, one run each, at 2 Hz. It is set
beside Run 3A and the released model on those same 13 routes only.

| Queue | Route | DS | RC | Penalty | Ended | Run 3A DS | Run 3A RC | Released DS |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 27582 | 32.5 | 100.0 | 0.325 | Completed | 18.5 | 53.0 | 45.5 |
| 2 | 27515 | 35.1 | 50.2 | 0.700 | Failed - Agent deviated from the route | 31.7 | 48.2 | 64.9 |
| 3 | 27529 | 40.0 | 50.0 | 0.800 | Failed - Agent deviated from the route | 97.7 | 100.0 | 40.0 |
| 4 | 26956 | 12.2 | 79.7 | 0.153 | Failed - TickRuntime | 21.3 | 22.6 | 60.0 |
| 5 | 2201 | 42.0 | 100.0 | 0.420 | Completed | 15.1 | 100.0 | 32.1 |
| 6 | 2050 | 24.7 | 65.7 | 0.377 | Failed - TickRuntime | 60.0 | 100.0 | 19.9 |
| 7 | 3936 | 21.5 | 56.6 | 0.380 | Failed - Agent deviated from the route | 55.1 | 55.1 | 40.9 |
| 8 | 26990 | 22.4 | 53.3 | 0.420 | Failed - Agent deviated from the route | 36.3 | 51.9 | 39.6 |
| 9 | 2084 | 8.5 | 50.8 | 0.167 | Failed - TickRuntime | 50.8 | 50.8 | 50.8 |
| 10 | 4183 | 22.5 | 53.5 | 0.420 | Failed - Agent deviated from the route | 53.5 | 53.5 | 37.0 |
| 11 | 2086 | 41.9 | 52.4 | 0.800 | Failed - Agent deviated from the route | 41.9 | 52.4 | 41.9 |
| 12 | 4669 | 54.5 | 54.5 | 1.000 | Failed - Agent deviated from the route | 53.0 | 53.0 | 20.8 |
| 13 | 4468 | 54.7 | 54.7 | 1.000 | Failed - Agent deviated from the route | 54.7 | 54.7 | 54.7 |

| Mean over these 13 | DS | RC |
|---|---:|---:|
| `H16.r2`, epoch 18 | 31.73 | 63.17 |
| Run 3A (`H04.run3a`) | 45.36 | 61.17 |
| Released, 2 Hz | 42.16 | |

Two of the 13 completed. These 13 are positions 1 to 13 of [the queue](scripts/closed_loop/queue.txt), and the lane
script's header says that queue opens with the 14 junction-turn routes, carried over from Run 2's lanes and not
re-measured here, so they are not a sample of the 50. On all 50, released is 57.60 and Run 3A 63.29. Bench2Drive runs are
not repeatable here. A smoke run on route 2802, a separate run, gave RC 100, DS 21.6, 3 vehicle collisions. The next 4
routes in the queue (27018, 1852, 2509, 3410) took 9 attempts and left no scored record: 7 ended "no episodes completed"
and the manual stop cut the last 2. The day's log puts the failed starts down to the preflight's import check timing out
under disk load.

### 7. Regression gate for epoch 18

Run by hand with `tools.gate smoke` from local worktrees, because the queue only runs adapters on a pushed branch.
Baselines are the queue's released Drive-JEPA runs. The verdicts are the gate's own, copied from
[verdict_ncap.json](tables/verdict_ncap.json) and [verdict_openloop.json](tables/verdict_openloop.json): overall **FAIL** in
both, the second with NAVSIM unfinished.

| Benchmark | Metric | Released | Epoch 18 | t | MDE | Units | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| KITScenes | ade | 2.537 | 3.159 | -2.91 | 0.604 | 117 | REGRESSION |
| nuScenes | l2_avg | 2.431 | 3.796 | -8.13 | 0.474 | 143 | REGRESSION |
| neuroNCAP, closed tier | ncap_score | 3.679 | 1.820 | -3.24 | 1.731 | 15 | REGRESSION |
| neuroNCAP, quick tier | ncap_score | 3.179 | 0.098 | -3.21 | 5.146 | 3 | NEUTRAL (cannot ship) |
| NAVSIM | | | | | | | no result |
| AlpaSim | | | | | | | no result |
| Bench2Drive, gate's 7 routes | | | | | | | not compared |

PDM terms never decide the gate. Beyond t crit on the closed neuroNCAP pair: pdm_nc .637 -> .380, pdm_dac .998 -> .831,
pdm_ttc .582 -> .304, pdm_style_score .525 -> .231. KITScenes pdm_ep .820 -> .905 and nuScenes .864 -> .966 are GAINs;
pdm_comfort falls .435 -> .027 and .412 -> .063.

**NAVSIM** failed its node start on all 3 attempts: it did not answer `/alive` within 15 min. **AlpaSim**'s quick scene timed
out at 60 min and its closed subset's node failed `/alive` too. The day's log puts both down to disk load, which it says
was not measured directly. Both are to rerun. **Bench2Drive's** 7 gate routes were left to come from the 50-route run, as
for Run 2 and Run 3A.

The adapter `drivejepa_p2` is a stock `drivejepa` node with this checkpoint, crop `native`, and FOLLOW sent as straight. It
lives in paddock worktrees `rerank-native` and `gate-oct27`, uncommitted, and [paddock/](paddock/) holds a copy.

## Caveats

- **The Bench2Drive scoring here is an offline 4 s stand-in, not a driving score.** The grade cannot see yield to emergency
  vehicles, blocked, timeouts or the tick cap, and DS is a route-level number. Pick grades say nothing direct about DS.
- **The stop-sign threshold is an adaptation.** Bench2Drive's 0.1 m/s is instantaneous; a path has only 0.5 s average
  speeds. A stop counts when a 0.5 s segment is under 1.0 m/s, which gives 11 expert failures against 12 in its 10 Hz log,
  where 0.1 m/s gave 289.
- **The camera mount differs between training and driving.** Training frames were captured at x 0.8, z 1.6; the runner
  mounts it at x 1.5, z 1.5. No crop removes that, and the input checks cannot see it.
- **The 0.3 m static margin overstates contact.** On 417 of the 461 frames where the expert's path hits something static
  there is no real contact, and each hit counted costs a factor of 0.65.
- **The batch patch was applied at runtime.** `mtd/stage4/sca_fix.apply_all()` patched Drive-JEPA's refiner in the training
  process; the tree on disk does not carry it. [`gallops.drivejepa_b2d.batch_fix`](../../src/gallops/drivejepa_b2d/batch_fix.py)
  is the same three patches, which is what the recipe applies.
- **The gate's pairs compare more than the training.** The candidate ran crop `native` with momentum off; the baselines
  were `openscene`, with momentum on for neuroNCAP's. The runs also used different paddock code. The gate prints all of this
  in its notes.
- **Epoch 18 was picked on the 47 held-out clips,** so its offline rows are `selected-on-report`. Only epoch 18 has been
  driven or gated.
- **Twins.** The 24 trigger twins are out of training, and the 968 kept clips exclude route-start twins of the 220 eval
  routes and of v0.0.4 validation. The trigger-twin check against v0.0.4 validation has not been run.

## Verified, and how

Measured on 6 Nov, when this record was written, with no GPU:

- **Sid's code.** The five byte copies Phase 1 imported are identical to the blobs at e75aaac, by sha256, so this PR points
  at them and does not add them.
- **The port.** `ast.unparse` of each of the 13 ported modules against the file as it ran, docstrings dropped
  ([script](scripts/port_astdiff.py), [every changed line](tables/port_astdiff_output.txt)): 141 changed lines. All are import statements and path expressions,
  apart from five unused locals and one unused assignment target removed for lint, `--cache` and `--out` made required in
  `build_vocab`, and an `exists()` guard around the speed figure `teachers` reads from a log. `scorer.py`: no change.
  `wiring.py` was rewritten by hand, to read locations from the environment.
- **Against what ran.** [scripts/port_check.py](scripts/port_check.py), on CPU under the training interpreter (Python 3.9.25),
  against the real Drive-JEPA tree: `scorer`, `grade`, `frames`, `ade`, `samples`, `build_cache`, `build_vocab`, `train_lora`,
  `train_stage2` and `grade_all` import, which applies `batch_fix` to that tree and passes its upstream checks. `grade_all`'s path then regraded 9 frames of
  3 clips, one of them held out, and reproduced Step A's files: the same feasible paths, every count identical, largest S
  difference 6e-8 ([output](tables/port_check_output.txt)).
- **The sha256** of the vocabulary and teacher index, read from the files today, match the record. The two checkpoints'
  come from their `SHA256SUMS` and `agent.ckpt.sha256`, written when they were saved; they were not hashed again.
- **The record against its tables.** `tests/test_drivejepa_b2d_retrain.py` holds the offline results, the closed-loop means
  and every gate pair to the tables and verdict files in this folder.

Reported by the runs themselves, and not repeated here: everything under "What was done".

Argued, not measured:

- That the port trains as the scripts did. No stage has been run from this repository. `merge_lora`, `teachers` and
  `export_agent` are top-level scripts, so importing one runs it: they have only been compiled and diffed.
- That `batch_fix` trains as `sca_fix` did. Same three patches by reading; not compared by running.
- That the 13 routes are the harder ones. It rests on the lane script's header and on released and Run 3A both scoring lower
  on them than on all 50.

## Where the code is

| Kind | Where |
|---|---|
| The pipeline | [src/gallops/drivejepa_b2d_retrain/](../../src/gallops/drivejepa_b2d_retrain/), through [the recipe](../../recipes/drivejepa_b2d_retrain.toml) |
| Scripts as they ran | [scripts/](scripts/), with machine roots as placeholders (`{ops}`, `{runs}`, `{data.NAME}`): a record, not runnable as it stands |
| sha256 of every file as it ran, and where it went | [AS_RUN_SHA256.txt](AS_RUN_SHA256.txt) |
| Sid's five files | not added: the repository has them at e75aaac, and the recipe reads them from a checkout of it |
| The paddock adapter | [paddock/](paddock/), for a throwaway checkout: paddock keeps released checkpoints only |

In `scripts/`, `paddock/`, `reports/` and `specs/`, a placeholder stands for a folder at the training site: `{ops}` paddock-ops,
`{runs}` the run root, `{data.b2d_retrain}` Phase 2's built data (`{built}` in the recipe), `{data.b2d_phase1}` the frame
cache's parent, `{data.b2d_clips}` Phase 0a's clips, `{code.drivejepa}` the Drive-JEPA tree, `{drivejepa_env}` its
interpreter's folder. The others (`{workspace}`, `{repos}`, `{conda}`, `{carla}`, `{nas}`, `{runs_p3}`, `{djexp_p3}`,
`{sid_checkout}`, `{paddock_site}`) are that site's folders by the same pattern. POINTERS.md uses the same names.

Two files in `src/` are not quite what ran. `train_lora` is the spec v3 file; the process that trained was already running
[train_b2d_with_ade_as_run.py](scripts/phase1/train_b2d_with_ade_as_run.py), with the ADE report on. `train_stage2` gained
`--init-released` after `H16.r2`, which ran [train_b2d_p2_05nov_copy.py](scripts/phase2/train_b2d_p2_05nov_copy.py).

The recipe starts from built inputs. Phase 0a's labels, the context builder (`mtd/build_context_b2d.py`), the blocked flags
(`b2d-labels/v6/blocked_v6.py`) and run 3A Phase 1's geometry are site tooling in no repository of ours, so the scripts
that use them are here as a record and [POINTERS.md](POINTERS.md) pins them by sha256. `route_len.json` has no surviving
builder; the file itself is pinned.

## Rules this did not meet

- **Registered after the runs.** No `PREDICTIONS.md`, as above.
- **No preflight.** `gallops.train.preflight` did not run, and the code was not committed when it trained. The specs' own
  checks did run, the grade-in-the-loop one being of the kind this repository asks for. `train_git_sha` is the Drive-JEPA
  tree's upstream commit, as for `H01.run2` and `H04.run3a`.
- **No sidecar.** Both checkpoints sit at the training site in their run folders; nothing has moved to `{weights}`.

## Decision

The gate says FAIL, with NAVSIM and AlpaSim unfinished. Nothing has been decided, so there is no decision-log row; the
status stays `running` and the verdict `none` until Navvrat decides. [docs/experiments.md](../../docs/experiments.md#deciding)
says what follows from a FAIL: read the failing pairs first, then either run a fix as the next run or close the hypothesis.
