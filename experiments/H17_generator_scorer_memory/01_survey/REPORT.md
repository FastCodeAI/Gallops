# Drive-JEPA memory, approach 3: read-only recon, runs 1 and 2 (parts A–D)

shakaal, 2026-11-07 (run 1: parts A–B; run 2 the same day: parts C–D, OPEN list). Read-only everywhere except this folder. No training, no GPU, no docker, no pushes. All computation CPU-only, `taskset -c 0-1 nice -n 19`, each check under 15 min (longest: 9.6 min, the 4,015-clip gap scan).

**How to read this.** Every fact carries a source: `path:line`, or a script in `src/` whose output is in `out/` and `logs/`. Verdict words: **yes / no / unclear**, **NOT CHECKED**, **NOT CHECKABLE**. "Measured" means I ran it today; "recorded" means I read it from a stage report or run record.

Path abbreviations:
- `G` = `Gallops-sid/LORA Refiner/Generator-Joint-PAI-alpasim/` (the earlier Physical AI AV lane)
- `RD` = `Paddock/paddock-sid/Alpasim/analysis_3.0/Retrain-data/`
- `NAS` = `{runs}/generator_joint_pai/`
- `PAI` = `{runs}/physicalai/`
- `MEM1` = `Gallops-sid/memory/1_camera_memory_H05/reports/` (camera memory, H05); `MEM2` = `Gallops-sid/memory/2_scorer_memory_H15/reports/` (scorer memory, H15)
- `src/`, `out/`, `logs/` = the sibling folders of `reports/` in `Gallops-sid/memory/3_generator_scorer_memory_PAI/`

Plain names: **encoder** (ViT-L image backbone, trained through a rank-16 LoRA), **generator** (proposes the 32-plan menu), **scorer** (ranks the menu: rating head + collision head + drivable-area head, 30 tensors), **grade** (0.1.7's offline simulator-style score of a plan), **pick grade** (mean grade of the scorer's pick on the 1,131 map moments of the 2,000-moment picking set), **ADE (3 s)** (mean distance to the human path over 0.5–3 s).

---

## Part A: the starting model

### A1. Which checkpoint is "0.1.8b"

**There is no single "0.1.8b" file. The lane exported three snapshots of one training run (attempt 2), and the one chosen by the pre-registered rule is pass 0.6, "0.1.8b-p06", marked "below 0.1.7 offline".** The other two (p02, p10) were driven as diagnostic arms.

| item | value | source |
|---|---|---|
| chosen export | `NAS/018b/release/drive_jepa_018b_p06.ckpt`, 1,273,003,357 bytes, 495 tensors | `NAS/018b/release/drive_jepa_018b_p06.json` keys `path`, `bytes`, `tensors` |
| sha256, **recomputed today** | `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66` | `out/sha256_018b_releases.txt`; equals the pin in `drive_jepa_018b_p06.json: sha256` and `G/CLOSING_REPORT.md:261` |
| source checkpoint | `NAS/018b/runs/018b_attempt2/ckpt/p00.6.pt`, sha `86b799b5…737d` (model state only) | `drive_jepa_018b_p06.json: source_sha256`, `meta.source_checkpoint`; `G/s9_018b_train/REPORT.md:11` |
| the other two exports (recomputed today) | p02 `7469b45d…2657`; p10 `ff50947e…07fe` | `out/sha256_018b_releases.txt`; match `G/CLOSING_REPORT.md:262-263` |
| chosen by | "max mean pick grade (map moments) among trained checks passing both gates (hard picks <= step 0 AND go rate where the human pulled away >= 0.66); chosen even if below step 0 (marked)" | `NAS/018b/runs/018b_attempt2/run.json: choice`; `chosen: 0.6`, `chosen_mark: "below 0.1.7 offline"`, `chosen_beats_step0_pick_grade: false` |
| chosen check's numbers | pick grade 0.7685 (step 0 = 0.1.7: 0.7694); hard-brake picks 0/745 (step 0: 43); go where the human pulled away 84.0 % (gate ≥ 66 %, n = 106) | `G/s9_018b_train/REPORT.md:3-5, 76-96` |
| on which slice | the picking set: 2,000 moments / 84 clips / 46 whole val_full chunks, 1,131 of them map (graded) moments; `NAS/picking/picking_set.parquet` | `G/s1_build/REPORT.md:28, 80-100`; read today (`src/probe_manifests.py`) |

**Which parts were trained** (recorded; `G/s8_018b_build/REPORT.md:4-8, 48-58, 132-151`):

| part | trained? | detail |
|---|---|---|
| encoder | **LoRA only**, fresh (B = 0 at start), rank 16 / alpha 16 on qkv, proj, fc1, fc2 of all 24 blocks = 96 layers / 192 tensors; merged into the encoder at export (96 encoder tensors changed vs 0.1.7) | `s8:50`; `run.json: start_info.lora`; `drive_jepa_018b_p06.json: tensors_changed_vs_017.encoder_merged = 96` |
| encoder base weights, projector, grids, answer key | frozen (no gradient) | `s8:136-143` (G2 routing table, last column 0) |
| generator | trained, starting from 0.1.7's (155 generator tensors changed vs 0.1.7) | `s8:51`; `…p06.json: tensors_changed_vs_017.generator = 155` |
| scorer | all 30 tensors trained (rating head 10, collision head 10, drivable-area head 10), lr 1e-4, starting from 0.1.7's | `s8:8, 57`; `…p06.json: scorer_30 = 30` |
| hide-and-guess guesser + answer-key LoRA | trained during the run, not part of the export | `s8:56, 59`; export writes encoder/generator/scorer only (`…p06.json: meta`) |

**Data slice, losses, steps** (recorded):

| item | value | source |
|---|---|---|
| training data | the s3_assembly manifest, unchanged: 126,981 moments / 4,015 clips / 1,398 chunks, all NVIDIA split `train`; manifest sha `ce834ee7…` asserted by the trainer | `G/s1_build/REPORT.md:51`; counts re-read today (`logs/`, `src/probe_manifests.py`) |
| sampler | 0.1.7's: every sample once plus type-A or graded samples a second time → 148,796 draws / 4,650 steps per pass, batch 32, bf16, seed 0 | `s8:30, 59`; `s1_build:51` |
| run length | attempt 2: 5,580 steps = 1.2 passes, 2026-10-30 22:37 → 2026-10-31 03:21 IST; stopped on the "6 checks without a better choosable pick" rule; checks at steps 0, 930, 1,860, **2,790 (chosen, pass 0.6)**, 3,720, 4,650, 5,580 | `run.json: checks[*].step, stopped, ended`; `s9:5, 23` |
| loss = | generator (0.1.7's `generator_loss_s10`, 0.5 × vocabulary term) + grade (BCE on the last logit, map moments, weight 1) + collision head (1 × BCE + 0.1 × L1) + drivable-area head (2 × BCE) + **0.5 × human closeness** (softmax(−ADE/0.5 m) targets, all moments) + 1.0 × hide-and-guess | `s8:51-55, 86` (step 6); attempt 2 weight `run.json: settings.w_human = 0.5` |
| extras | **pull-away × 2** on the grade, collision, drivable-area and human lessons (`settings.w_pull_away = 2.0`); **hard-brake rule**: grade target 0 for a plan whose first step brakes harder than 6 m/s² at > 10 m/s (map moments) | `s9:23`; `s8:53` |
| learning rates | generator Adam 3e-5; scorer Adam 1e-4; LoRA and guesser AdamW 1e-4 | `s8:57, 59`; `s1_build:50` |
| starting weights | 0.1.7's file `PAI/s10_retrain/ckpt/drive_jepa_physicalai_generator.ckpt`, sha `29d5f951…2916`; step 0 reproduced 0.1.7 bit for bit on all 2,000 picking moments | `run.json: start_info.start`; `s8:33` |
| attempt 1 | human weight 2, no pull-away weight; lost to a host crash at pass 1 step 3,550; its recipe was abandoned | `s9:21-22` |

**How it differs from 0.1.7, 0.1.8 and 0.1.8c** (recorded; `G/CLOSING_REPORT.md:41-49` unless noted):

| | 0.1.7 | 0.1.8 | **0.1.8b-p06** | 0.1.8c-p04 |
|---|---|---|---|---|
| start | released | released | 0.1.7 | 0.1.7 |
| encoder | released, frozen | + fresh LoRA | + fresh LoRA | + fresh LoRA |
| generator | retrained on Physical AI AV, generator only (pass 9 + 1 pass lr 1e-4 + 2 passes lr 2.5e-5; encoder and scorer frozen) `RD/s10_retrain/PREDICTIONS_phase1.md:16-18`, `final_choice.json: chosen "+2"` | trained jointly from released | 0.1.7's, trained further | 0.1.7's, trained further |
| scorer | the earlier retrain's pass-9 scorer, 30 tensors untouched (`RD/s10_retrain/gen_ckpt.json: scorer_tensors_bit_identical_to_pass9 = 30`) | rating head only; lesson = human closeness | all 30 tensors; grade + collision + drivable + human × 0.5 + pull-away × 2 + hard-brake rule | all 30 tensors; 0.1.7's three lessons only (grade, collision, drivable) |
| hide-and-guess | no | yes | yes | off |
| passes trained / chosen | 3 (see above) | stopped by rule; chosen pass 0.4 = step 1,860, by lowest slow (< 5 m/s) pick ADE (`G/s2_train/REPORT.md:12`, `s1_build:135`) | 1.2 run; chosen pass 0.6 by pick grade + two gates | stopped at pass 0.4 (hard-brake gate failed twice); nothing choosable; p04 driven as a diagnostic (`CLOSING_REPORT.md:102-103`) |
| export sha256 | `29d5f951…2916` | `b6686d8c…98d3` | `3d566eb2…8e66` | `cd013d53…6bdd` |

### A2. How 0.1.8b stands on the 30 AlpaSim scenes

**0.1.8b was never run on the 30 scenes.** The lane's bar for a 30-scene run was "12-scene mean ≥ 0.6742 (0.1.7's) and no new at-fault crash vs 0.1.7"; all three snapshots failed it, so no 30-scene run was started (`G/s10_018b_closed_loop/REPORT.md:8-14, 25`; `G/CLOSING_REPORT.md:72`).

**Closed loop** (recorded; scores 0–1, one rollout per scene):

| model | 30 scenes (mean) | 12 diagnosis scenes (mean) | new at-fault crashes vs 0.1.7 | source |
|---|---|---|---|---|
| released (via adapter 0.1.4b) | **0.5433** | 0.4480 (taken from its 30-scene runs; never run as a 12-scene arm) | — | `G/s4_closed_loop/REPORT.md:193-197`; `CLOSING_REPORT.md:55, 67` |
| 0.1.7 | **0.7697** | **0.6742** | — | same; `s10:94` |
| 0.1.8 | 0.7350 | 0.5875 | 8023dce5, e3ed554f | `s4:193-201`; `s10:94` |
| **0.1.8b-p06** | **not run** | **0.5919** | 8023dce5, 04749bb9 | `s10:12, 94, 100-105`; `s10_018b_closed_loop/tables/stepD.csv` |
| 0.1.8b-p02 | not run | 0.5432 | 8023dce5 | `s10:13` |
| 0.1.8b-p10 | not run | 0.5880 | 8023dce5, 04749bb9 | `s10:14` |

Per-scene for p06 vs 0.1.7 (`s10:80-94`): p06 gains d2715023 (0.041 → 1.000) and bae463d3 (0.311 → 0.380); loses 8023dce5 (1.000 → 0.000, at-fault, scorer picks unsafe plans with 11–17 safe plans on the menu, `s10:112-113`), 04749bb9 (0.956 → 0.000, at-fault) and b1b170d0 (0.782 → 0.723).

**Offline** (recorded):

| metric | released | 0.1.7 | 0.1.8 | 0.1.8b-p06 | source |
|---|---|---|---|---|---|
| pick grade (1,131 map picking moments) | 0.692 | 0.7694 | 0.714 | 0.7685 | `CLOSING_REPORT.md:53-60` |
| hard-brake picks above 10 m/s, of 745 | 528 | 43 | 1 | 0 | same |
| held-out pick ADE (3 s), all / fast (300 clips, 9,581 moments, read once) | 6.222 / 20.179 | 1.623 / 2.108 | 1.268 / 2.723 | **not read** | `G/s3_offline/REPORT.md` § 2 table; `CLOSING_REPORT.md:69` |
| picking-set pick ADE (3 s), all / fast (not held out) | — | 1.637 / 1.798 | — | 1.108 / 1.271 | `s9:105, 108` |
| unsafe-pick rate while a safe plan exists | — | 9.84 % | — | 9.74 % | `s9:78, 81` |

**Regression gate (`public_2601_gate10`): not run on 0.1.8b.** The gate's 10 scenes are all inside the 30 (`Paddock-gate/src/paddock/adapters/benchmarks/alpasim/scenes/public_2601_gate10.yaml:15-24`; subset check measured today, `logs/b1_b3.log` line "gate10 subset of our30: True"). No run record under `paddock-memory/results/`, `Paddock/results/` or `Paddock-gate/results/` mentions 0.1.8b (`grep -rl "018b\|0\.1\.8b"` → no hits). The gate runs that exist on the Drive-JEPA row are the memory-picker (H15) runs of 2026-11-05/06.

**Seeds and spread:**
- Training: one seed (0) and one run per recipe (`s8:59`; `s1_build:247`); training is not bit-reproducible run to run (step-25 loss 3.6298 / 3.6434 / 3.6421 across launches, `CLOSING_REPORT.md:230`).
- Closed loop: one rollout per scene, no repeat of any 0.1.8b scene. The lane records "No closed-loop rerun has given a different result" and bit-identical per-tick metrics when 0.1.7 was re-run on 4 scenes after two simulator-image rebuilds (`CLOSING_REPORT.md:207, 213, 233`). **Disagreement:** the Paddock gate documentation says that through the Paddock bridge "a single scene can move by up to about 0.55 between two runs of the same code, from near-ties between two proposals after the handover (71d7d03b, bae463d3 and 5f064b02), while the 30-scene mean moved by 0.013" (`Paddock-gate/docs/gate.md:490-496`). That is a different harness (bridge) from the native runs the lane used, so both may be true; no spread figure exists for 0.1.8b itself.

**The lane's closing conclusion, quoted** (`G/CLOSING_REPORT.md:30-33`):
> "**Answer: no, not in this setup.** None of the eight encoder-LoRA variants beat 0.1.7 in closed loop: on the 30 scenes: 0.1.7 0.7697 against 0.1.8 0.7350; on the 12 scenes: 0.1.7 0.6742 against 0.5256–0.5919. The clean test (0.1.8c: 0.1.7 + an encoder LoRA, 0.1.7's own lessons only) scores 0.5379 on the 12 scenes. It adds a new at-fault crash, so the rule says 'LoRA adds no value in this setup' (§3)."

The brief's statement "0.1.8b did not beat 0.1.7 in closed loop" is **confirmed**, with one correction: the comparison is on the 12 diagnosis scenes only; 0.1.8b was never driven on the 30.

**Plain answers:**
1. **Is 0.1.8b better than released? Yes on what was measured, but not proven on the 30 scenes.** On the 12 scenes p06 0.5919 vs released 0.4480 (released's 12-scene value derived from its 30-scene runs); offline pick grade 0.7685 vs 0.692 and hard-brake picks 0 vs 528. No 30-scene run and no held-out ADE read exist for 0.1.8b.
2. **Is 0.1.8b better than 0.1.7? No.** 12-scene mean 0.5919 vs 0.6742 with two new at-fault crashes (8023dce5, 04749bb9); offline pick grade 0.7685 vs 0.7694 ("below 0.1.7 offline" by the lane's own rule). Its picking-set pick ADE (3 s) is better (1.108 vs 1.637 m), but the lane found offline ADE does not predict closed loop (`CLOSING_REPORT.md:117-122`).

---

## Part B: the data

### B1. Physical AI AV data on the NAS

**The dataset** (`nvidia/PhysicalAI-Autonomous-Vehicles`, snapshot `33f9bf44…`, `PAI/hf_home/hub/…/snapshots/33f9bf447ed3bcb7d545ce13f4226f824214fafb/`), measured today (`src/probe_manifests.py`, `src/b1_b3.py`, `logs/b1_b3.log`):

| item | value | source |
|---|---|---|
| clips | 306,152, all `clip_is_valid`; split train 153,625 / val 90,928 / test 61,599; 3,146 chunks of ≤ 100 clips, every chunk single-split | `clip_index.parquet` read today; `RD/s0_census/REPORT.md:415-416` |
| clip length | 20 s nominal: over the 149,773 raw_train clips scanned by the census, mean 19.995 s, min 19.0, max 20.9, median 20.0 | `RD/s0_census/r3_raw_clips.csv` column `dur`, summarised today |
| total hours | ≈ 1,701 h for all 306,152 clips at 20 s (**extrapolated**); measured 831.9 h over the 149,773 scanned raw_train clips | same |
| poses | `egomotion.offline`: 10 Hz, 202 rows per 20 s clip, dt median 0.1000 s; columns timestamp (clip-relative, starts at 0), x, y, z, qx…qw | `logs/probe_timing.log`; `src/probe_timing.py` |
| camera frame rate | 30 Hz: 604–605 frames over 20.13 s, dt median 0.03333 s | `logs/probe_timing.log` (frame-timestamp table) |
| cameras in the dataset | 7: `camera_front_wide_120fov`, `camera_front_tele_30fov`, `camera_cross_left_120fov`, `camera_cross_right_120fov`, `camera_rear_left_70fov`, `camera_rear_right_70fov`, `camera_rear_tele_30fov`; plus one lidar and 20 radar feeds | `…/snapshots/33f9bf44…/features.csv` |
| cameras on the NAS | **front-wide only**: mp4 + frame-timestamps parquet per clip under `PAI/s3_assembly/s1/clips/` (4,834 clip folders) and `NAS/heldout/clips/` (300) | `ls` today (`logs/b1_b3.log`) |
| labels on the NAS | ego poses: 2,083 `egomotion.offline` chunk zips (`PAI/s0_census/raw/labels/egomotion.offline/`) + 426 for the held-out pool (`NAS/heldout/raw/labels/`); agent tracks: `obstacle.offline` members for the 4,935 raw_train clips with a lateral move-off (1,427 chunk folders under `…/obstacle.offline.members/`) + 400 for held-out; NuRec `clipgt` egomotion + obstacle + clip tables for the 1,447 pool scenes (`PAI/s0_census/pool/<uuid>/`) | `ls` today; `RD/s0_census/REPORT.md:531-533` |
| per-sample model cache | `PAI/s3_assembly/s1/cache/` ≈ 0.40 TB (the lane's `du` timed out; **NOT CHECKED** today) | `G/s0_recon/REPORT.md:254` |

**Slices used by the earlier lane** (measured today from the manifests; durations from `r3_raw_clips.csv` for train and from `NAS/heldout/mining/{clips_scan,valfull_clips_scan}.csv` for the val-split slices):

| slice | moments | clips | hours | where the list lives | role | clean? |
|---|---|---|---|---|---|---|
| **train** | 126,981 | 4,015 (1,398 chunks, 25 countries) | **22.30** (all 4,015 durations known) | `RD/s3_assembly/manifest.parquet` | training for 0.1.7, 0.1.8, 0.1.8b, 0.1.8c | NVIDIA train split; 0 overlap with exam (B3) |
| val (smoke) | 608 | 20 | 0.11 (20 s assumed) | `RD/s3_assembly/val/manifest_val.parquet` | 0.1.6 smoke eval | NVIDIA val |
| val_full | 24,657 | 801 (499 chunks) | 4.45 | `RD/s3_assembly/val_full/manifest_val_full.parquet` | 0.1.6/0.1.7 validation; **pass 9 was chosen on it** | not clean for choosing (`G/s0_recon/REPORT.md:284-287`) |
| picking | 2,000 (1,131 graded) | 84 (46 whole val_full chunks) | 0.47 | `NAS/picking/picking_set.parquet` | checkpoint choice for 0.1.8/0.1.8b/0.1.8c | ⊂ val_full; 0 chunks shared with train or held-out (`G/s1_build/REPORT.md:98`) |
| **held-out** | 9,581 | 300 (221 chunks) | **1.67** | `NAS/heldout/manifest_heldout.parquet`, `heldout_clips.csv` | read once for 0.1.8 (`s3_offline`); never for 0.1.8b | NVIDIA val; raw only, ungraded; no chunk shared with train/val/val_full/exam (`s1_build:107-109`) |
| raw_val, untouched | — | 90,927 − 300 − (val/val_full raw clips) | ≈ 500 h at 20 s | `RD/s0_census/raw_val.txt` | unused | not downloaded |

Train composition (`logs/probe_manifests.log`): raw 104,940 moments (ordinary 95,887 in 3,000 clips; pull-out `B_raw` 9,053 in 283 clips) + NuRec 22,041 moments (high-speed `A` 6,415 / 205 clips; close-pass `B_pass` 2,527 / 89; ordinary 13,099 / 438); 21,806 graded (label 2).

### B2. Unbroken history per training moment

**The step is 0.5 s, confirmed from the converter:** `RD/s1_converter/s1_convert.py:25` `STEP_US, HORIZON_US, HIST = 500_000, 4_000_000, 4`; anchors are `range(STEP_US, ego.ts[-1] + 1, STEP_US)` kept only if `t + HORIZON_US <= ego.ts[-1]` and `t − STEP_US >= ego.ts[0]` (`:104-105`). So a moment sits at t = 0.5, 1.0, … s from the first pose, and the last one at 16.0 s (4 s of future needed inside a 20 s clip). The manifest confirms: `anchor_t` 0.5 … 16.0 at 0.5 s steps, median 8.0 (`logs/probe_manifests.log`). The model itself sees two frames (t and t − 0.5 s, `:128`) and up to 4 ego-state rows at 0.5 s spacing (`:121-126`), so its own window is 0.5 s of images and 1.5 s of car state.

**Definition used.** History of a moment = anchor_t minus the later of (first pose time, first frame time, end of the last gap before the anchor). A gap is a pose step > 0.15 s (1.5 × the nominal 0.1 s) or a camera-frame step > 0.05 s (1.5 × the nominal 1/30 s). Script `src/b2_gaps.py`; outputs `out/b2_train.json`, `out/b2_heldout.json`, per-clip CSVs.

**Training slice, all 4,015 clips scanned (measured, 575 s):** 1 clip has a pose gap (0.200 s), 0 clips have a frame gap; frames start 0.03–0.31 s before the first pose.

| history bin (s) | [0, 0.5) | [0.5, 1) | [1, 2) | [2, 5) | [5, 10) | [10, 20) | 20+ |
|---|---|---|---|---|---|---|---|
| train moments (126,981) | 180 | 3,973 | 7,946 | 23,841 | 39,693 | 51,348 | 0 |
| share | 0.1 % | 3.1 % | 6.3 % | 18.8 % | 31.3 % | 40.4 % | 0 |

Median **8.00 s**; mean 8.24 s; share with **≥ 5 s: 71.7 %**; ≥ 2 s 90.5 %; ≥ 4 s 78.1 %; ≥ 8 s 53.1 %; ≥ 10 s 40.6 %; maximum 16.0 s (`logs/b2_train_full.log`). The 20+ bin is empty **by construction**: clips are 20 s and the last anchor is at 16 s.

**Held-out slice, all 300 clips (measured, 69 s):** 0 pose gaps, 1 clip with one frame gap (0.074 s).

| history bin (s) | [0, 0.5) | [0.5, 1) | [1, 2) | [2, 5) | [5, 10) | [10, 20) | 20+ |
|---|---|---|---|---|---|---|---|
| held-out moments (9,581) | 19 | 301 | 602 | 1,803 | 3,000 | 3,856 | 0 |
| share | 0.2 % | 3.1 % | 6.3 % | 18.8 % | 31.3 % | 40.2 % | 0 |

Median **8.00 s**; share ≥ 5 s **71.6 %**; ≥ 4 s 78.1 %; ≥ 8 s 53.0 % (`logs/b2_heldout.log`).

**Compared with NAVSIM.** The "~3.5 s" figure is the NAVSIM chain median: "Chains are short: a median of 7 moments, p90 17–18, at most 64" at one moment per 0.5 s (`MEM2/MEMORY_PLAN_V2_REVIEW.md:102`), i.e. 3.5 s. The chain-length histogram in frames is at `MEM1/MEMORY_STEP1.md:86-88` (2: 163, 3: 1,682, …, 41–65: 82; longest 65 frames = 32 s). NAVSIM val graded moments with ≥ 4 / ≥ 8 / ≥ 16 earlier moments (≥ 2 / 4 / 8 s): 77 % / 45 % / 16 % (`MEM2/MEMORY_PLAN_V2_REVIEW.md:102`). Physical AI AV: ≥ 2 / 4 / 8 s = 90.6 % / 78.1 % / 53.1 %. So the median history is **8 s against 3.5 s**, and 8 s of history is available on half the moments against one sixth; but NAVSIM's tail reaches 32 s while Physical AI AV never exceeds 16 s, because every clip is 20 s.

### B3. Leakage against the 30 AlpaSim scenes and the gate10 scenes

**Scene lists used** (measured, `src/b1_b3.py`, `logs/b1_b3.log`): our 30 = `RD/s0_census/b2_our30.csv` (30 uuids); `public_2601` = 913 scene ids from `Paddock/paddock-sid/Alpasim/alpasim/data/scenes/sim_suites.csv`; gate10 = the 10 scenes in `public_2601_gate10.yaml:15-24`. Checked: gate10 ⊂ our 30 ⊂ public_2601 (both True).

**By clip id: none.**

| slice | clips | overlap with public_2601 ∪ our 30 ∪ gate10 |
|---|---|---|
| train | 4,015 | **0** |
| val | 20 | 0 |
| val_full | 801 | 0 |
| picking | 84 | 0 |
| held-out | 300 | 0 |

This re-confirms `G/s0_recon/REPORT.md:262-269` (E2) and `G/s1_build/REPORT.md:62` (the trainer's own split guard refuses any clip in public_2601 or our 30).

**By chunk (the only grouping key the dataset has): none.** 198 of the 913 exam clips exist in the raw `clip_index` (196 test, 1 train, 1 val), spread over 31 chunks; 9 of our 30 are in the raw dataset (all split `test`, chunks 2657, 3103, 3111, 3114, 3115, 3135); the other 21 are not in the raw release at all. None of those 31 chunks appears among the train (1,398), val (20), val_full (499), held-out (221) or picking chunks. The one train-split exam clip (chunk 3080) contributes 0 training clips (`s0_recon:271`).

**By time range: NOT CHECKABLE with the metadata on hand.** Command and output: `src/b3_timerange.py` → `logs/b3_timerange.log`, `out/test30_timeranges.csv`, `out/nurec_timeranges.csv`, `out/timerange_hits.csv`.
- Raw clips carry only clip-relative timestamps (first pose = 0; `logs/probe_timing.log`, `t0 = 0` on 20/20 clips) and `metadata/data_collection.parquet` has only `country, month, hour_of_day, platform_class, radar_config`: no drive, session or date key (`RD/s3_assembly/REPORT.md:59-61`; columns re-read today).
- NuRec scenes do carry absolute-looking `timestamp_micros` (usdz `metadata.yaml: time_range`; pool `clipgt` tables), but they are **per-vehicle device clocks**: the 30 test scenes' starts span 3,943 s to 5,322,243 s, and windows "overlap" between clips from different countries (e.g. test 791d68ce, Germany, and train clip a45776a3, United States, gap 0 s). 347 NuRec-sourced train/val_full clips fall within ± 10 min of some test scene, 8 overlap exactly, 44 within 60 s, and the cross-country cases show these are coincidences of unrelated clocks, not shared drives.
- Weak proxy, reported only: raw training clips sharing (country, month, hour_of_day) with a test scene, for the 9 test scenes that exist in the raw dataset: 01d503d4 6, 5f064b02 6, d1a25a99 4, 8023dce5 3, b44a4a44 3, 02e075b9 1, 048b974e 1, 791d68ce 1, acd22430 0. The other 21 test scenes have no raw metadata.

**Verdict: no leakage found by clip id or by chunk; same-drive adjacency by time cannot be tested** because the dataset publishes no shared clock or session key.

### B4. How often moments where memory should matter occur

**Metadata that exists** (measured; `logs/probe_timing.log`):
- Ego poses: 10 Hz for every training clip (`egomotion.offline`, clip-relative timestamp, position, quaternion).
- Agent tracks **with ids**: (i) NuRec `clipgt/obstacle.parquet` per pool scene: `trackline_id`, centre, size, orientation, `category`, per-track ~10 Hz, in the clip world frame; (ii) raw `obstacle.offline` per clip: `track_id`, box in the rig frame at `reference_frame_timestamp_us`, `label_class`. Categories seen: automobile, person, heavy_truck, trailer, rider, bus, protruding_object, other_vehicle, animal, stroller, train_or_tram_car.
- Also in NuRec `clipgt`: lanes, lane lines, traffic lights, wait lines, crosswalks, road boundaries (`RD/s0_census/REPORT.md:95-96`); not used here.
- **Coverage gap:** tracks are on the NAS for 732 of 735 NuRec training clips (pool) and for the 283 pull-out (`B_raw`) clips (members), but **not for the 3,000 ordinary raw clips** (75 % of training clips, 16.7 of 22.3 h): their obstacle labels were never fetched (`RD/s3_assembly/REPORT.md:174-175`, F14; existence check today in `logs/probe_timing.log`). Fetching them would be ≈ 1.7 GB (561 KB per clip, `RD/s0_census/REPORT.md:527`), outside this read-only run.

**Detection rules** (exact text in `out/b4_nurec.json: rules`; code `src/b4_events.py`, reusing the census's `r3_core.build_ego` / `build_tracks` for the 10 Hz grids):
- **hard braking:** acc = d/dt of the 0.5 s-smoothed speed at 10 Hz; an event is ≥ 3 consecutive samples (0.3 s) with acc ≤ −THR m/s² and speed ≥ 2 m/s at the run start; runs < 1 s apart merged; THR ∈ {3, 4, 6}.
- **cut-in:** a vehicle track (not person / stroller / protruding_object / animal / rider / bicycle / motorcycle) at sample j with ego speed ≥ 2 m/s, track 0 < lon ≤ 40 m ahead in the ego frame, lateral distance to the ego's next-5 s path ≤ 1.25 m, track speed ≥ 1 m/s, motion direction within 45° of the ego heading; and within the previous 2 s the same track had lateral distance ≥ 2.5 m while ahead; one event per track per 5 s.
- **out-of-view-and-return:** same track id with a gap ≥ 1.0 s between consecutive raw observations, the track within 60 m of the ego at both ends; one event per gap.
- **crossing traffic:** track speed ≥ 1 m/s, motion direction 60–120° from the ego heading, 0 < lon ≤ 50 m ahead, lateral distance to the ego's next-5 s path ≤ 1.75 m; one event per track.
- **memory-relevant moment:** a training moment within (t_event, t_event + 5 s].

**Results on the two track-bearing subsets of the training slice** (measured; `out/b4_nurec.json`, `out/b4_braw.json`, per-clip CSVs; 0 clip failures):

| | NuRec clips (A / B_pass / ordinary) | pull-out clips (B_raw) |
|---|---|---|
| clips / clip-hours / moments / tracks | 732 / 4.07 h / 22,041 / 58,107 | 283 / 1.57 h / 9,053 / 20,139 |
| hard braking ≥ 3 m/s²: events (per clip-hour) | 24 (5.9 /h) in 24 clips | 4 (2.5 /h) |
| hard braking ≥ 4 m/s² | 4 (1.0 /h) | 0 |
| hard braking ≥ 6 m/s² | 1 (0.25 /h) | 0 |
| cut-ins | 81 (19.9 /h) in 71 clips | 15 (9.5 /h) |
| out-of-view-and-return | **0** (see flag) | **0** |
| crossing traffic | 57 (14.0 /h) in 50 clips | 3 (1.9 /h) |
| moments within 5 s after a hard brake (≥ 3) | 149 (0.7 % of moments) | 31 (0.3 %) |
| moments within 5 s after a cut-in | 621 (2.8 %) | 138 (1.5 %) |
| moments within 5 s after a crossing | 436 (2.0 %) | 24 (0.3 %) |

**Hard braking on all 4,015 training clips from the manifest speeds** (ego only, coarser: speed drop between consecutive 0.5 s anchors with the earlier speed ≥ 2 m/s; `out/b4_manifest_brake.json`): decel ≥ 3 m/s²: 280 anchor pairs in 127 clips = 12.6 per clip-hour over 22.3 h; ≥ 4: 33 pairs (1.5 /h); ≥ 6: 4 pairs in 3 clips (0.18 /h). The 0.5 s measure counts more than the 10 Hz 0.3 s-run rule; both say a brake at 6 m/s² is roughly a once-per-five-hours event, consistent with the lane's "1 in 45,340 moments" for human hard braking (`CLOSING_REPORT.md:188`).

**Why out-of-view-and-return is zero:** on 20 NuRec clips (2,760 tracks) the largest gap between consecutive observations of any track id is 0.204 s (p99 0.202 s); no track has a gap ≥ 0.5 s (`logs/b4_extra.log`). The autolabels do not keep an id across an occlusion; a track that disappears comes back as a new id (or is interpolated). So this event type is **NOT MEASURABLE** from track ids. A proxy (a new id appearing near where an old id vanished) was not built.

**Caveats.** Both measured subsets are selected sets: NuRec clips were chosen for reconstruction availability and by type (high-speed, close-pass, ordinary), pull-out clips start from standstill behind an obstacle; the unmeasured ordinary raw clips were screened on ego motion only. Rates on the full 22.3 h are therefore unknown for cut-ins and crossings; the ego-only hard-brake rate above is the only full-slice figure.

### B4b: all training clips

*(added 2026-11-07 after run 2; the B4 text above is unchanged. Folder note: the task named `approach3_pai/`, which is the symlink to this folder, so `approach3_pai/out/…` and `out/…` are the same files.)*

**What was fetched.** The `obstacle.offline` members (agent tracks, nothing else) for the 3,000 ordinary raw training clips, with the census's range-reader (`RD/s0_census/hfrange.py` `open_zip` / `fetch_member`, `r3_rawio.url`; the same mechanism as `r3_raw_actors.py:25-38`), into a new folder on the NAS, with its own manifest (sha256 per file). Script `src/b4b_fetch.py`; plan and result in `logs/b4b_fetch_plan.log`, `logs/b4b_fetch.log`.

| item | value | source |
|---|---|---|
| clips to fetch | 3,000 ordinary raw clips in 1,254 chunks (0 already on the NAS; none in the exam set, asserted) | `logs/b4b_fetch_plan.log:2` |
| planned size | 3,000 × 561 KB = 1.68 GB (census figure, `RD/s0_census/REPORT.md:527`); target had 1.7 TB free before and after | `logs/b4b_fetch_plan.log:3-4` |
| fetched / failed | **3,000 / 0**, 1.659 GB in 1,857 s (8 threads, `taskset -c 0-1 nice 19`) | `logs/b4b_fetch.log` last lines |
| on disk | `{runs}/physicalai/approach3_tracks/obstacle.offline.members/chunk_NNNN/<uuid>.obstacle.offline.parquet` (1.6 GB by `du`), `download_manifest.csv` 3,000 rows | `du -sh`, `wc -l` |
| verification | every clip has its file; 3,000 / 3,000 sha256 match the manifest; 3,000 read as parquet with the expected columns; all rows `reference_frame = rig` | `logs/b4b_verify.log`, `out/b4b_fetch_verify.csv` |
| rows per clip | min 0, p5 258, median 5,208, p95 18,523, max 59,953 (census pull-out members for reference: median 6,742, min 425) | `logs/b4b_verify.log` |
| tracks per clip (raw ids) | median 90, max 885; label span median 20.0 s | `logs/b4b_verify.log` |

**106 sparse clips (not fetch failures).** 103 clips have a label span < 15 s and 40 have < 50 rows (37 both; 1 clip, `9692de32`, has 0 rows), listed in `logs/b4b_verify.log`. Their files are small (median 16 KB vs 436 KB) and hash-correct, so this is what the dataset holds (empty road, or labels that stop early), not a transfer problem. After the 1 s / 5-sample track filter of `r3_core.build_tracks`, 17 of them have no usable track; these 106 clips contribute 0 cut-ins, 0 crossings and 11 hard brakes (ego-only). They are kept in the denominators below.

**Re-run.** `src/b4b_events.py` is `src/b4_events.py` with the loaders generalised (modes `nurec` / `braw` / `ordinary` / `heldout`) and per-event times written out; the rule text, `NONVEH`, and the functions `runs` / `lat_to_path` / `events` are byte-identical to run 1 (checked with `diff` on the two blocks; see the session command). Run 1's `nurec` and `braw` outputs were reproduced exactly (events and memory-relevant moment counts equal, `out/b4_*.json` vs `out/b4b_*.json`). 0 clip failures in all four modes (`logs/b4b_*.log`). Hard-braking rates are per clip-hour of all clips in the row; cut-in and crossing rates are per clip-hour of clips that have tracks (on training these are the same, since every training clip now has tracks). Moment shares are of the moments in the same denominator.

**Training slice, all 4,015 clips (measured; `out/b4b_summary_train.csv`):**

| | ordinary | pull-out | reconstructed | **all training** |
|---|---|---|---|---|
| clips / clip-hours / moments | 3,000 / 16.67 h / 95,887 | 283 / 1.57 h / 9,053 | 732 / 4.07 h / 22,041 | **4,015 / 22.30 h / 126,981** |
| hard braking ≥ 3 m/s²: events (per h) in clips | 114 (6.84 /h) in 106 | 4 (2.54 /h) in 4 | 24 (5.90 /h) in 24 | **142 (6.37 /h) in 134** |
| hard braking ≥ 4 m/s² | 20 (1.20 /h) in 18 | 0 | 4 (0.98 /h) in 4 | **24 (1.08 /h) in 22** |
| hard braking ≥ 6 m/s² | 4 (0.24 /h) in 2 | 0 | 1 (0.25 /h) in 1 | **5 (0.22 /h) in 3** |
| cut-ins | 282 (16.92 /h) in 235 | 15 (9.54 /h) in 15 | 81 (19.92 /h) in 71 | **378 (16.95 /h) in 321** |
| crossing traffic | 265 (15.90 /h) in 210 | 3 (1.91 /h) in 3 | 57 (14.02 /h) in 50 | **325 (14.57 /h) in 263** |
| out-of-view-and-return | 0 | 0 | 0 | **0** (NOT MEASURABLE, flag 12) |
| moments ≤ 5 s after a brake ≥ 3 (share) | 859 (0.90 %) | 31 (0.34 %) | 149 (0.68 %) | **1,039 (0.82 %)** |
| … after a brake ≥ 4 / ≥ 6 | 136 (0.14 %) / 23 (0.02 %) | 0 / 0 | 15 (0.07 %) / 1 (0.00 %) | **151 (0.12 %) / 24 (0.02 %)** |
| … after a cut-in | 2,023 (2.11 %) | 138 (1.52 %) | 621 (2.82 %) | **2,782 (2.19 %)** |
| … after a crossing | 2,055 (2.14 %) | 24 (0.27 %) | 436 (1.98 %) | **2,515 (1.98 %)** |
| **moments after any event (share)** | 4,809 (5.02 %) | 193 (2.13 %) | 1,186 (5.38 %) | **6,188 (4.87 %)** |

The ordinary clips' cut-in and crossing rates (16.9 and 15.9 per hour) sit between the two subsets measured in run 1 (reconstructed 19.9 / 14.0, pull-out 9.5 / 1.9), so the run 1 caveat that the full-slice rates were unknown is closed: on the full 22.3 h, a cut-in happens about every 3.5 minutes and a crossing about every 4 minutes, a brake ≥ 3 m/s² every 9 minutes, a brake ≥ 6 m/s² every 4.5 hours. The 10 Hz 0.3 s-run brake rule gives 6.4 /h at ≥ 3 m/s² against 12.6 /h from the coarser 0.5 s manifest measure in B4 (same ordering as in run 1).

**Held-out, 300 clips (measured; `out/b4b_summary_heldout.csv`).** Only 84 of the 300 clips have agent tracks on the NAS (82 pull-out + 2 ordinary, 0.47 of 1.67 h; `NAS/heldout/raw/labels/obstacle.offline.members/`, checked per clip); this run fetched nothing for the held-out set, per the task. Hard braking (ego only) covers all 300; cut-ins and crossings cover the 84.

| | ordinary (143; 2 with tracks) | pull-out (82; all) | raw A (75; 0 with tracks) | **all 300** |
|---|---|---|---|---|
| clip-hours / moments | 0.79 h / 4,566 | 0.46 h / 2,622 | 0.42 h / 2,393 | **1.67 h / 9,581** |
| hard braking ≥ 3 / ≥ 4 / ≥ 6: events | 10 / 2 / 0 | 0 / 0 / 0 | 2 / 1 / 1 | **12 (7.2 /h) / 3 (1.8 /h) / 1 (0.6 /h)** |
| cut-ins (per tracked hour) | 0 on 0.01 h | 4 (8.8 /h) | no tracks | **4 (8.6 /h on 0.47 h)** |
| crossings (per tracked hour) | 0 on 0.01 h | 7 (15.4 /h) | no tracks | **7 (15.0 /h on 0.47 h)** |
| moments after any event | 84 | 57 | 17 | **158 (1.65 % of 9,581; 2.12 % of the 2,686 moments in clips with tracks)** |

**Memory-moment lists (the candidate test subset).** `out/memory_moments_train.parquet` (6,511 rows = 6,188 distinct moments of 126,981) and `out/memory_moments_heldout.parquet` (191 rows = 158 distinct moments of 9,581); columns `clip_id`, `anchor_t`, `event_type` (`hb3` / `hb4` / `hb6` nested thresholds, `cut_in`, `crossing`), `s_since_event` (0.1–5.0 s, the nearest preceding event of that type). One row per (moment, event type), so a moment after both a brake and a cut-in has two rows; de-duplicate on (`clip_id`, `anchor_t`) for the subset. Built by `src/b4b_moments.py` from `out/b4b_<mode>_events.csv`.

**Caveats.** (i) The held-out memory subset is thin (158 moments, 84 clips with tracks); if OPEN 7 picks a fresh held-out, re-run `b4b_events.py heldout` on it after fetching its tracks (≈ 300 × 561 KB). (ii) The 106 sparse-label ordinary clips are in the denominators; excluding them (0.59 h) would raise the ordinary-clip rates by ≈ 4 %. (iii) The raw `obstacle.offline` labels and the NuRec `clipgt` tracks come from different pipelines (rig-frame boxes vs world-frame tracklines); the per-type rows should be compared with that in mind.


---

## Flags

1. **"0.1.8b" is three exports (p06 / p02 / p10) from one run; the chosen one is p06, marked "below 0.1.7 offline".** p10 had the best pick grade of any model (0.7746) but failed the go gate (`s9:14`).
2. **The go gate was changed mid-run** (from ≥ 70 % of all standstills to ≥ 66 % of the 106 pull-away moments) and checks 0–0.6 were re-gated from stored dumps; pass 0.2 flipped from fail to pass (`s9:41-49, 187-191`).
3. **The adapter-equivalence bar was raised** from 3e-4 m to 1 mm so p06 and p02 could run (`s10:45-50`).
4. **No 30-scene run and no gate10 run exist for 0.1.8b**; all closed-loop evidence is 12 scenes, one rollout each.
5. **Sources disagree on closed-loop run-to-run spread:** the lane says reruns are bit-identical (native harness); the Paddock gate doc says up to 0.55 per scene through the bridge (`gate.md:490-496`).
6. **Training is single-seed and not bit-reproducible** (`CLOSING_REPORT.md:230`).
7. **The 0.1.8b-p06 held-out ADE (3 s) was never read**; its ADE figures are on the picking set, which also chose it (`CLOSING_REPORT.md:69`).
8. **H05 nuance.** The record says one extra camera frame (1 s ago) did help the probe (camera part −0.0134 m/s, CI excludes 0) and only beyond 1 s did camera history add nothing (`MEM1/MEMORY_STEP2_PROBE.md:21`; `gallops-memory/experiments/H05_ssm_memory/CARD.md:26`). The brief's "beyond the model's own two frames, past camera features added nothing" is a little stronger than that. Status: parked, inconclusive (`H05 README.md:3`).
9. **H15 numbers confirmed:** +0.0011 [+0.0002, +0.0021] grade offline (`MEM2/MEMORY_H15_STEP3.md:6`); gate: (a) carried vs (b) wiped NEUTRAL on all six benchmarks, both HOLD (`MEM2/MEMORY_H15_GATE.md:16-20, 102`).
10. **val_full and the picking set are not clean** (pass 9 chosen on val_full; `s0_recon:284-287`); the only clean set is the 300-clip raw held-out, which is ungraded (human-distance metrics only).
11. **Time-range leakage is NOT CHECKABLE** (no shared clock, no session key); only id and chunk checks were possible (B3).
12. **Out-of-view-and-return is NOT MEASURABLE** from track ids (B4).
13. **RESOLVED 2026-11-07 (B4b):** agent tracks for the 3,000 ordinary training clips were fetched (3,000 / 0 failed, 1.66 GB, `{runs}/physicalai/approach3_tracks/`); the B4 rates on 1,015 clips are superseded by the all-4,015-clip rates in B4b (cut-in 16.95 /h, crossing 14.57 /h, brake ≥ 3 m/s² 6.37 /h; 4.87 % of training moments within 5 s of an event). Still open on the held-out: 216 of 300 clips have no tracks, so its cut-in/crossing figures cover 84 clips.
14. **Clip-count wording mismatch:** `RD/s3_assembly/REPORT.md:14` says 4,018 clips; the manifest has 4,015 (3 NuRec clips lost every moment; `s0_recon:248`).
15. **History is capped at 16 s by construction** (20 s clips, 4 s future), so "20 s+" can never occur in this data; any memory longer than 16 s is untestable here, against 32 s in NAVSIM's longest chain.
16. **NOT CHECKED:** total dataset hours (extrapolated from the 149,773-clip scan), the per-sample cache size (du timed out in the lane), and whether the 20 val / 801 val_full clips' durations differ from 20 s (assumed for val; measured for val_full via the held-out mining scan).
17. **0.1.7's adapter pin named a renamed file** (`drive_jepa_s10_gen.ckpt`), fixed by a symlink (`s0_recon:34, 88-93`; `s1_build:219-221`).
18. The 30-scene runs for 0.1.8 used two versions of the scene runner (runs 1–4 frozen, 5–30 edited); the lane says the wizard config was unchanged (`CLOSING_REPORT.md:235-239`).

---

## Part C: the model and training code

Run 2, 2026-11-07. Read-only; the only writes are in this folder. Abbreviations as in run 1, plus:
- `UP` = `Paddock/paddock-sid/Alpasim/adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based/` (the vendored upstream model; every 0.1.x adapter's `src` symlinks here)
- `AD` = `Paddock/paddock-sid/Alpasim/adapters/drivejepa018bp06/` (the native 0.1.8b-p06 adapter)
- `PMEM` = `paddock-memory/src/paddock/adapters/models/drivejepa/` (the Paddock bridge policy with the H15 memory layer; identical to `gallops-memory/experiments/H15_prediction_memory/paddock/src/…`, `diff -q` empty)
- `GM` = `gallops-memory/src/gallops/memory/`
- Starting model, re-verified: `NAS/018b/release/drive_jepa_018b_p06.ckpt`, sha256 `3d566eb2…8e66` recomputed in run 1 (`out/sha256_018b_releases.txt`), pinned in `AD/constants.py:66`.

**Folder note.** This report now lives in `Gallops-sid/memory/3_generator_scorer_memory_PAI/reports/` after the memory folder was reorganised on 2026-11-07; `Gallops-sid/memory/approach3_pai` is a symlink to that folder so the agent-track files the other session writes to `approach3_pai/out/memory_moments_{train,heldout}.parquet` land in `out/`. Neither file existed at 02:10 IST on 2026-11-07 (`ls out/ | grep moment` empty).

### C1. How the generator and the scorer work today

One moment, batch size B (B = 1 in the car and at the checks, 32 in training). Shapes are from the code; the recorded shapes in `G/s1_build/REPORT.md:68` (inputs) and `G/s8_018b_build/REPORT.md:151` (head outputs) agree with them.

**Encoder** (`UP/bevformer/simple_image_encoder.py`):

| step | shape | source |
|---|---|---|
| input: two front frames, t and t − 0.5 s, ImageNet-normalised and stacked | (B, 2, 3, 256, 512) | `UP/drive_jepa_model.py:53-57` |
| GridMask (training only, p = 0.7) | same | `simple_image_encoder.py:21, 44-45` |
| V-JEPA ViT-L, the two frames fused as one 2-frame clip (`img_as_video_nframes = 2`) | (B, 512, 1024) | `:33, 47` |
| projector Linear(1024 → 256), re-laid as a 16 × 32 token grid | (B, 256, 16, 32) | `:38, 48-49` |
| flattened for deformable attention: (num_cam = 1, 512, B, 256) + spatial shape (16, 32) | | `:56-66` |

The 0.1.8b encoder LoRA (rank 16, alpha 16, on qkv / proj / fc1 / fc2 of all 24 blocks, 192 tensors) is merged into these weights at export (`G/s1_build/src/pai_model.py:99, 282-297`; `drive_jepa_018b_p06.json: tensors_changed_vs_017.encoder_merged = 96`).

**Generator** (`UP/drive_jepa_model.py`, `UP/traj_refiner.py`, `UP/bevformer/bev_refiner.py`):

| step | shape | source |
|---|---|---|
| ego input: the newest `ego_status` row [x, y, heading, vx, vy, ax, ay, command one-hot(4)], acceleration zeroed | (B, 11) | `drive_jepa_model.py:51`; `RD/s3_5_smoke/train_patch/dataset_s35.py:253-259` |
| `hist_encoding` Linear(11 → 256) | (B, 1, 256) | `drive_jepa_model.py:32, 61` |
| queries: `init_feature` Embedding(32 proposals × 8 poses = 256, 256) added to the ego feature | bev_feature (B, 256, 256) | `:33, 62` |
| refiner pass (× 4, one shared `Traj_refiner`): `traj_decoder` MLP(256 → 1024 → 1024 → 3) on every query → proposals | (B, 32, 8, 3) [x, y, heading] at 0.5 … 4 s | `:35-36, 64-67`; `traj_refiner.py:21-24` |
| then `Bev_refiner`: one BEVFormer layer = temporal self-attention over the 256 queries, spatial deformable cross-attention into the 512 image tokens, FFN; the 3-D reference points are the box corners of each proposal pose at 4 heights (16 points per query) projected through `lidar2img`; a learned positional encoding on the 32 × 8 grid is added | bev_feature (B, 256, 256) | `bev_refiner.py:19, 31-32, 45-92, 107-142`; `UP/bevformer/encoder.py:233-253, 272-299` |
| output: the 4th pass's proposals (the **menu**) and bev_feature | (B, 32, 8, 3), (B, 256, 256) | `drive_jepa_model.py:69-73` |

Two facts matter for memory. (i) The "temporal" self-attention has a two-slot queue (`num_bev_queue = 2`, offsets Linear(512 → 128), `value_proj` 256 → 256) that BEVFormer meant for the previous frame's BEV, but in the proposal-query path it is fed the current queries twice (`temporal_self_attention.py:174-178, 195`; `encoder.py:233-237`); the previous-BEV branch exists only for the grid path (`encoder.py:258-270`). (ii) Query slot i·8 + k is a fixed identity (row of `init_feature`) that exists at every moment, while the trajectory it decodes is redrawn every moment.

**Scorer** (`UP/score_module/scorer.py`):

| step | shape | source |
|---|---|---|
| proposal feature: bev_feature reshaped (B, 32, 8, 256), max over the 8 poses | (B, 32, 256) | `scorer.py:41` |
| rating head `pred_score` MLP(256 → 1024 → 1024 → 6); the **last** logit is the rating | (B, 32, 6) | `:15, 42`; `drive_jepa_model.py:85` |
| collision head `pred_col_agent` MLP(256 → 1024 → 1024 → 720), training mode only: per proposal, per pose, 5 key agents × (corner boxes 8 + valid 1) × 2 | (B, 32, 8, 5, 2, 9) | `scorer.py:20, 50-51` |
| drivable-area head `pred_area` MLP(256 → 1024 → 1024 → 10) on every query, training mode only | (B, 256, 10) | `:25, 47-48` |
| pick: argmax of sigmoid(last logit); in the car the comfort re-score first blends it with the previous plan, S ← (14 S + 2 EC)/16 | (B,) | `drive_jepa_model.py:85-91, 98-150` |

**Parameter count, measured from the 0.1.8b-p06 file** (`src/c1_params.py`, `logs/c1_params.log`):

| part | parameters | tensors |
|---|---|---|
| encoder ViT-L (LoRA merged) | 303,885,312 | 292 |
| encoder projector | 262,400 | 2 |
| generator: ego encoding + queries | 68,608 | 3 |
| generator: refiner `traj_decoder` MLP | 1,319,939 | 10 |
| generator: refiner `Bev_refiner` (attention, FFN, positional encoding, fixed reference buffers) | 1,001,280 | 32 |
| scorer: rating head | 1,323,014 | 10 |
| scorer: collision head | 2,054,864 | 10 |
| scorer: drivable-area head | 1,327,114 | 10 |
| **total, unique** | **311,242,531** | 369 (495 keys; the shared refiner is stored under 4 names) |

The trainer's own count of trainable generator values is 2,381,123 over 41 tensors (`G/s1_build/src/pai_model.py:30`): the 2.39 M above minus the fixed reference buffers.

**The same forward in the 0.1.8b training code** (`G/s8_018b_build/src/pai_model_b.py`, `G/s1_build/src/pai_model.py`):
- `Model018b.forward` (`pai_model_b.py:105-129`): `self.m(feats)` is the upstream forward above in bf16 inside the GridMask stream; hooks capture the image tokens and the scorer's bev input (`pai_model.py:146-161`).
- **Replay** (`pai_model.py:194-201`): the scorer's lessons do not read the trained forward's bev_feature. They read a second pass of the refiner with its weights detached, started from the same ego/query init and the same four proposal lists, with the image tokens attached. So the scorer lessons reach the encoder LoRA and the three heads, not the generator (`G/s8_018b_build/REPORT.md:136-143`, G2 routing table).
- The replay's scorer call runs in training mode so the collision and drivable-area heads produce outputs (`pai_model_b.py:116-120`); the rating logit is bit-identical in both modes (`s8:150`).
- `scorer_lessons` (`pai_model_b.py:131-155`; attempt 2's weighted version `G/s9_018b_train/src/pai_model_c.py:44-87`) turns the online grades into the four losses (C4).

### C2. Where memory could enter

All four candidates share two constraints from C1 and C5: the model is called once per 0.5 s (the divider, `AD/divider.py:122, 189-206`), and the only per-path identity that survives a moment is the query slot index. Costs are per moment and per batch row; the encoder forward (≈ 2 × 304 M × 512 tokens ≈ 0.3 TFLOP) is the yardstick.

| option | what it reads | how state passes between moments (paths are redrawn) | extra parameters | extra compute / memory per moment |
|---|---|---|---|---|
| **A. Generator queries carry a state** | the previous moment's final bev_feature (256 × 256), the ego motion between the two moments | **by slot index**: slot i·8+k at t is initialised from slot i·8+k at t − 0.5 s through a gate, e.g. q₀ = hist(ego) + init_feature + GRU(bev_prev ∘ ego-motion embedding); no path matching. Cheapest variant: feed bev_prev into the unused second queue of the temporal self-attention (`temporal_self_attention.py:174-178`) with the previous poses shifted by the ego motion, as the grid path does (`encoder.py:258-268`) | GRU per slot 2 × 3 × 256² ≈ 0.4 M (+ 0.2 M ego-motion embedding); queue variant: 0 new weights (the offsets layer already takes 512 inputs) | < 1 % of the encoder; state 256 × 256 fp32 = 262 KB |
| **B. Generator cross-attention to cached past encoder tokens** | the last K token maps (512 × 256 each) with a learned (Δpose, Δt) embedding on the keys | **by cache**: nothing recurrent; the K maps are kept in a ring buffer; the relative-pose embedding does the alignment (deformable projection of past maps would need past `lidar2img`, so use plain attention) | one cross-attention layer + FFN ≈ 0.26 M + 0.52 M, embedding 0.07 M ≈ 0.85 M | 256 queries × 512 K keys × 256: K = 16 (8 s) ≈ 1 GFLOP, < 1 % of the encoder; buffer K × 512 KB fp32 = 8 MB (4 MB bf16) |
| **C. A recurrent scene state read by both parts** | M scene tokens (e.g. 16 × 256) updated each moment from the pooled current tokens, the ego row and a summary of the last menu + pick; read by the generator (cross-attention from the 256 queries into the M tokens before the first refiner pass) and by the scorer (attention-pooled scene vector concatenated to each proposal feature, 256 + 256 → rating head) | **by one state vector**: no per-path identity needed; ego-motion embedding added at the update | update block ≈ 0.8 M, generator read ≈ 0.3 M, scorer read 0.33 M (rating head input 512 → 1024) ≈ 1.5 M | negligible (256 × 16 keys); state 16 × 256 = 16 KB |
| **D. Scorer state per path (H15) or per scene** | per path: previous proposal feature + logits + ego of the matched path; per scene: a pooled scorer feature | per path: H15's alignment — the previous menu moved into the current ego frame, nearest old path within 2 m on the shared 0.5–3.5 s window, else reset (`PMEM/memory.py:49-50, 60-80, 116-123`); **or by slot index** (unverified that slot i is the "same" path at t and t + 0.5, see OPEN 11); per scene: a GRU on the max-pooled proposal feature | H15's `MemoryPicker` 256 + 6 + 11 → 256 → gate → 1 ≈ 0.2 M | negligible; per-path state 32 × 256 = 32 KB; per-path matching re-runs up to L earlier moments through the layer (`memory.py:226-241`) |

Reading: A and C need no matching and work through the divider as is; B has no warm-up problem (the window is explicit) but is the only one whose cost grows with history; D per-path carries the H15 matching machinery and its unmatched-path resets (H15 gate: `unmatched_fraction` is a diagnostic for a reason).

### C3. How 0.1.8b sampled data, and what sequences would take

**Today: independent single moments.**
- Batch 32 (`G/s2_train/src/pai_train.py:50`). A pass is a global permutation of 148,796 draws (every moment once, type-A and graded moments twice; `G/s1_build/src/pai_data.py:82-89`, the 0.1.7 rule `RD/s10_retrain/s10_gen_train.py:117-120`), cut into consecutive slices of 32 (`pai_data.py:92-99`), 4,650 steps per pass (`G/s8_018b_build/REPORT.md:30`). Nothing groups rows by clip; two moments of one clip can share a batch (about 0.12 same-clip pairs per batch, argued).
- One sample = one slim npz: two PNG frames, `lidar2img` (2, 4, 4), `img_shape` (2, 3), `ego_status` cut to its newest row (1, 11) with acceleration zeroed, `trajectory` (8, 3), `token`, plus `has_label2` and the metric-cache path from the manifest (`dataset_s35.py:190-214, 248-262`). Every key is per moment; the clip id appears only inside the sample id and the cache path.
- The model never sees a clip id, a previous plan or a recurrent state; `past_ego_simulated_states` is never fed in training or at the checks (`pai_model.py:305`, `drive_jepa_model.py:87-88`), so the car's comfort re-score path is never trained.
- Step = forward (planner + replay + index read) → grade submit (10 processes) → hide-and-guess forward → grade collect → lessons → one backward, three optimisers (`G/s9_018b_train/src/train_018b.py:218-248`). Timings: 3.09 s/step overlapped at the gate (`G/s9_018b_train/REPORT.md:36`); 3.49 s/step from the log timestamps of steps 25–900 (measured by the trace); the whole run, 5,580 steps and 7 checks (46 min of checks, `run.json: checks[*].time_s`), took 4 h 43 min 58 s (`run.json: start.started, ended`), i.e. 3.05 s/step including checks. **Disagreement** between the two per-step figures: the log-derived 3.49 s/step would give 5.4 h of pure training, more than the run's wall time; the run included one resume (01:49–01:51) and no other gap. The 3.05 s/step (wall, checks included) is the number used below.
- Cache reads per row: one slim npz ≈ 240 KB (32 per clip, 7.4 MB per clip directory); graded rows add `metric_cache.pkl` ≈ 25 KB and the index file < 1 KB; the hide-and-guess pair stream reads a second npz of another moment (trace of `{runs}/physicalai/s4_train/slim/<clip>/` and `…/s3_assembly/s1/cache/<clip>/<sid>/`).

**Sequences of moments within a clip, with a warm-up.**
- **Batch construction.** A batch becomes W windows × L consecutive moments of one clip, in time order; the loader reads the L slim npz files of one clip directory (L × 240 KB, one directory, sequential) and the metric caches of the graded moments among them. Keeping 32 moments per step (W × L = 32) keeps the encoder cost per step unchanged; W = 2, L = 16 or W = 4, L = 8.
- **Truncated backprop.** State flows across the L moments; the loss is applied on moments w + 1 … L (w = warm-up, e.g. 4 = 2 s) and the state is detached at the window start (no carry between steps). With a recurrent design (A, C, D) the effective history is still unbounded at test time; the training window bounds what is learned.
- **Windows available** (measured from the manifest, `src/` note in `logs/`; 115 of 4,015 training clips have a missing anchor inside, 0 in held-out):

| window L (seconds) | sliding windows, train | per moment | non-overlapping windows | moments covered |
|---|---|---|---|---|
| 8 (4 s) | 98,880 | 0.78 | 15,675 | 98.8 % |
| 12 (6 s) | 82,841 | 0.65 | 7,956 | 75.2 % |
| 16 (8 s) | 66,846 | 0.53 | 7,712 | 97.2 % |
| 24 (12 s) | 35,036 | 0.28 | 3,951 | 74.7 % |
| 32 (16 s) | 3,721 | 0.03 | 3,721 | 93.8 % |

So with L = 16 and a random window start per clip and pass, every moment is still visited (97 % coverage with non-overlapping windows; sliding starts give 0.53 windows per moment), and a pass of 7,712 windows × 16 = 123,392 moments ≈ 3,856 steps of 32 moments, close to today's 4,650. With L = 32 only one window per clip exists, so history beyond 8 s is trained on 3,721 windows per pass.
- **Sampler change.** The 0.1.7 "graded or type A twice" rule has to move from moments to windows (weight a window by its graded share) or be dropped; the pull-away ×2 and the hard-brake rule are per moment and survive unchanged.
- **If the encoder is frozen** (D1), the 512 × 256 token map of every moment can be cached once (bf16: 126,981 × 262 KB = 33.3 GB; held-out 2.5 GB; a selection slice 0.5 GB) and the sequence step runs only the generator, scorer and memory: the per-step cost drops from ≈ 3 s to well under 1 s (the grading wait alone is 0.18 s/step, `run.json: grade_wait_s` mean). GridMask and hide-and-guess disappear with the encoder gradient, which is the trade.
- **Checks.** The check forward is batch 1, fp32, merged agent, 2,000 moments (`train_018b.py:192-210`; 492 s at step 0); for a memory model each check must replay each picking clip's window in order (the picking set holds whole chunks, so consecutive moments exist, `G/s1_build/REPORT.md:85-87`), which the check code does not do today.

### C4. Training targets on Physical AI AV

| target | definition | moments that have it | source |
|---|---|---|---|
| generator | 0.1.7's `generator_loss_s10`: winner-takes-all L1 between the 32 plans and the human (8, 3) future, stages combined with `prev_weight` 0.1, plus 0.5 × a vocabulary pseudo-target term drawn from the anchor index on graded rows | all 126,981 (vocabulary term on the 21,806 graded rows that have a non-empty index) | `pai_model.py:8-11, 216-217`; `UP/drive_jepa_agent.py:221-269` (upstream form); `G/s1_build/REPORT.md:47` |
| scorer: grade | the generator's own 32 plans are graded online by 0.1.7's grader (`S35.compute_score` → `get_scores_s35` → `s3_scorer.score` on the moment's metric cache: no collision, drivable area, traffic light, driving direction, progress, time to collision, lane keeping, comfort → one final score per plan); BCE between the rating head's last logit and the final score (`final_score_weight` 1, `sub_score_weight` 0) | **map moments only**: `has_label2`, 21,806 of 126,981 (17.2 %), all NuRec clips | `pai_model_b.py:12-16, 201-252`; `RD/s3_5_smoke/train_patch/drive_jepa_agent_s35.py:179-216, 218-250`; `G/s8_018b_build/REPORT.md:52` |
| scorer: collision head | S35 `score_loss` key-agent part: BCE on the valid flag + 0.1 × L1 on the key-agent corners from the metric cache | map moments only | `pai_model_b.py:17, 149-154`; `drive_jepa_agent_s35.py:219-230` |
| scorer: drivable area | 2 × BCE between the drivable-area head and the ego-area targets from the metric cache | map moments only | `pai_model_b.py:18`; `drive_jepa_agent_s35.py:236-241` |
| scorer: closeness to the human | targets softmax(−d/0.5 m) over the 32 plans, d = mean distance to the human path over 0.5–3 s; cross-entropy with the last logit; weight **0.5** in attempt 2 | all moments | `pai_model.py:252-259`; `pai_model_c.py:24, 49-52` |
| pull-away weight | a moment with ego speed < 1 m/s whose human future is ≥ 1 m from the start at 4 s counts ×2 in the grade, collision, drivable-area and human lessons (weighted means) | all moments (6,132 pull-away moments in the training slice, `G/s7_018b_recon/REPORT.md:185`) | `pai_model_c.py:25-34, 44-87` |
| hard-brake rule | grade target set to 0 for a plan whose first 0.5 s step decelerates harder than 6 m/s² when the ego is above 10 m/s | map moments only (it edits the grade target) | `pai_model_b.py:54, 143-148, 168-173` |

**How the grade exists on this data.** The metric cache of a NuRec moment was built in the training-set assembly from the usdz map tables with two declared departures from standard EPDMS: the real vehicle box and a driving-direction exemption; the reference plan is the human path, not PDM-Closed (`RD/s3_assembly/REPORT.md:18-24, 166-167`). Raw clips have no map, so no metric cache and no grade (`has_label2 = False` on all 104,940 raw moments).

**On the held-out slice every scorer target except human closeness is missing**: all 300 clips are raw, `has_label2` = 0 on 9,581 of 9,581 rows (`logs/probe_manifests.log`), and the lane already noted "raw only and ungraded, so it supports human-distance metrics only" (`G/s1_build/REPORT.md:122`). Pull-away flags and the hard-brake rule need only ego speed and the human future, so they can be computed there. Consequence for D2: a **graded** held-out or selection slice can only be carved from NuRec clips, and all 404 NuRec validation scenes are inside val_full (`G/s0_recon/REPORT.md:284-287`).

### C5. Closed-loop path

**Native harness (0.1.8b-p06)**, `AD/` and `Paddock/paddock-sid/Alpasim/drafts/drivejepa-0.1.8b-p06/`:
- The simulator asks every 50 ms; the `ReplanDivider` calls the model every 0.5 s of simulated time and re-anchors the cached plan on the other ticks (`AD/constants.py:207-221`; `AD/divider.py:122, 189-206, 236-265`). The driver scaffold keeps a 16-frame image cache per session; `_select_frames` picks the newest frame and the one nearest 0.5 s back, within ±10 % (`AD/constants.py:241, 254`; `AD/planner.py:631-665`). `build_features` builds the same inputs as training (`AD/planner.py:674-730, 815-857`): two frames through Pipeline A, `ego_status` rows with acceleration zeroed, the command from the route (first waypoint ≥ 20 m ahead, ±3 m lateral; `AD/planner.py:420`).
- Process layout: a Python 3.12 driver (planner + divider) and a Python 3.9 model worker child, one worker and one lock per process (`AD/bridge.py:9-14, 211-265`). **The worker holds no per-scene state** (`bridge.py:360-378`). The only state across calls is the previous chosen plan for the comfort re-score, kept in the planner keyed by `session_uuid` (`planner.py:537, 948-978`) and sent with each request; in the worker it becomes `past_ego_simulated_states` if younger than 0.55 s (`AD/momentum.py:122-133`). It is dropped on `on_session_start` / `on_session_close` (`planner.py:1024-1034`). The run script starts a fresh driver per scene (`run_018bp06_scene.sh:204, 241-242, 259`).
- Provenance: checkpoint path, sha and bytes pinned (`AD/constants.py:65-67`); the worker verifies size and sha before loading and logs "verified … source pinned" (`bridge.py:121-133, 333-338`); the launcher refuses a placeholder pin or `DRIVEJEPA_BATCH_FIX=off`; the run script refuses `DRIVEJEPA_CHECKPOINT_OVERRIDE`, checks the constants tuple `"0.1.8b-p06 37 True True True on <sha>"` and greps both logs for the pinned source (`run_018bp06_scene.sh:62, 166-169, 272-273`).
- Comfort re-score switch `DRIVEJEPA_MOMENTUM` ∈ {on, off}, default on; "off" leaves the key out, which is exactly the 0.1.4 call (`AD/constants.py:279-289`; `momentum.py:234-237`).

**Paddock bridge** (`paddock-memory/src/paddock/adapters/benchmarks/alpasim/harness.py:192-220`, `drivejepa_rhythm.py`, `PMEM/policy.py`):
- One bridge process and one wizard per scene against a long-lived policy node; every scene starts with `/reset`, which clears the node's frame history, momentum and memory (`harness.py:725-800`; `bridge.py:87-89, 482`; `PMEM/policy.py:218-224`).
- The bridge **emulates the native 0.1.4b driver** ("--planner drivejepa-0.1.4b"): the 2 Hz grid, the frame pair, speed and command are ported (`drivejepa_rhythm.py:1-50, 365-421`); it sends (vx, 0) and (ax, 0), has **no bonnet mask, no zeroed acceleration and no batch fix**, i.e. none of the 0.1.8b preprocessing.
- The checkpoint comes from the env `DRIVEJEPA_CHECKPOINT` and the node does no sha check; any site override of a `*_CHECKPOINT` key makes the run `valid: false` by design (`paddock-memory/src/paddock/runner.py:175-178, 204-240, 294-297`). All five H15 AlpaSim gate runs are `valid: false` for that reason. **A 0.1.8b run through the bridge cannot be a ranked Paddock run.**
- **Note on the folder named in the brief:** `Paddock/src/paddock/adapters/models/drivejepa/` is on branch `alpasim-20hz-results`; its `model.yaml` does not list alpasim and its `policy.py` has no momentum. The current code is in `Paddock-gate` (HEAD 9585f81) and `paddock-memory` (HEAD 0fae534).

**What a memory switch needs, and what the H15 adapter already gives** (`PMEM/memory.py`, `PMEM/policy.py`):
- The switch: `DRIVEJEPA_MEMORY` holds a checkpoint **directory**; empty means the released node bit for bit (`memory.py:5-7, 44`). The loader refuses a missing `memory.pt`, a missing `checkpoint.json`, a wrong model name or a sha mismatch (`:136-153`). **Carry vs wiped is a property of the sidecar** (`notes.mode` ∈ {carry, reset}; `:48, 154-159`): in reset mode the window is forced to 0 and only the current step runs with nothing inherited (`:110, 116-121, 178`). "Wiped" was H15's name for that arm.
- State across calls: per-moment entries keyed by timestamp (`:181-183, 216-225`); the previous moment must be 0.45–0.55 s back, else a new chain starts; a clock jump clears everything; entries older than 0.55 s are pruned (`:50, 196-211, 242`). A node called at 10 Hz keeps five interleaved chains (`:23-26`). `_catch_up` runs observe-only frames on the chain. Diagnostics: mode, length, sha, resets, steps, chains, clock jumps, matched and unmatched paths (`:246-254`).
- Reset per scene: node `/reset` (`policy.py:223-224`).
- Injection: a forward hook on the scorer adds a nudge to the final logit, so the comfort re-score and the argmax both see it; a pre-hook captures (proposals, bev query) and pools the proposal embedding exactly as `scorer.py:41` does.
- **Reusable as is:** the env switch, the sidecar loader with sha refusal, the chain rule and clock-jump reset, the diagnostics counters, the hook pattern, the fixture test that pins the Paddock copy to gallops. **H15-specific:** the `MemoryPicker` layer, the input [embed 256; logits 6; ego 11], nudging only the final logit, per-path matching within 2 m, re-running the window from stored inputs.
- **For the native adapter** the same switch has to be added to the worker: the state lives in the worker keyed by `session_uuid`, the planner's `on_session_start` / `on_session_close` send a reset message (today they only pop the momentum entry), the request carries the moment's timestamp so the 0.45–0.55 s chain rule can run, and the launcher's provenance line prints the memory mode, sha and checkpoint beside the model pin. Wiped mode = the same weights with the carried state zeroed on every call, selected by the sidecar, never by hand.

**Which harness.** Run 1's flag 5 rested on `Paddock-gate/docs/gate.md:490-496` ("a single scene can move by up to about 0.55 between two runs of the same code"). The underlying data is **not** a same-code repeat: it is the bridge (released checkpoint, Paddock node) against the native 0.1.4b driver, and the doc itself names the causes: a different runtime, the first-plan momentum difference and the missing sideways values (`Paddock-gate/docs/models/drivejepa.md:301-346`; per-scene table `Paddock/paddock-sid/DRIVEJEPA_PADDOCK_ROUTE_CLOSE.md:28-62`: 71d7d03b −0.546, bae463d3 −0.460, 5f064b02 +0.346, 04394343 +0.244, sixteen scenes identical, mean −0.013). No same-code repeat table exists in either harness; the lane's same-code reruns of 0.1.7 on 4 scenes were bit-identical (`G/CLOSING_REPORT.md:207, 213, 233`). Reading: the native harness is deterministic for fixed weights and inputs; the 0.55 swings are near-tie scenes flipping under a *different* driver, which is a warning about single-scene evidence, not about repeat noise. **Recommendation: native harness for all arms** (deterministic; it is the harness 0.1.7, 0.1.8 and 0.1.8b were measured in; it applies the 0.1.8b preprocessing; it pins the sha). The gate10 scenes are a subset of the 30, so the gate numbers can be reported from the same runs; a Paddock run would be `valid: false` anyway.

---

## Part D: design proposal (no code)

### D1. Designs

Common to all three: start from 0.1.8b-p06 (merged encoder, 0.1.8b generator and scorer); memory in **both** the generator and the scorer, trained together with the generator and scorer weights; the training window is L moments of one clip with a warm-up w; the 0.1.8b lessons (C4) unchanged on the moments after the warm-up; the memory layers start at zero output (zero-initialised last layer, as H15) so step 0 reproduces 0.1.8b bit for bit, which gives the same G1-style gate the lane used.

**History length: train on 16-moment windows (8 s), warm-up 4 moments (2 s), recurrent state unbounded at test time.** From B2: 78 % of training moments have ≥ 4 s of past and 53 % have ≥ 8 s; 16-moment windows cover 97 % of moments with 7,712 windows per pass, while 32-moment windows (16 s) exist only once per clip (3,721) and would starve the loss on history beyond 8 s. The memory-relevant events (B4) sit 0–5 s before the moments they should help, well inside 8 s. NAVSIM's median chain was 3.5 s; 8 s is the longest window Physical AI AV can supply at full coverage.

**Encoder LoRA: frozen** (the merged 0.1.8b weights, no new LoRA). The lane's clean test found "LoRA adds no value in this setup" and every LoRA variant crashed into the truck on 8023dce5 (`G/CLOSING_REPORT.md:33-34, 113`); freezing it also lets the token maps be cached (C3), which is what makes sequence training affordable, and removes GridMask and hide-and-guess from the loop. The alternative (train a fresh LoRA with the memory) is costed below.

| design | what is remembered | reads / writes | trained | new parameters | cost per run (3 seeds? no: one seed, 3 passes) |
|---|---|---|---|---|---|
| **1. Slot memory** (C2 A + D by slot) | per query slot: a 256-d state carried from the previous moment's final bev_feature, aligned by an ego-motion embedding; the scorer gets the slot state of each proposal concatenated to its proposal feature | generator: GRU on the 256 slots at query init; scorer: rating head input 512 | generator 2.38 M + scorer 4.70 M + memory ≈ 0.9 M | ≈ 0.9 M (GRU 0.4 M, ego-motion embedding 0.2 M, rating head widening 0.26 M) | cached features: ≈ 3 h; with a trained LoRA: ≈ 14 h |
| **2. Scene tokens** (C2 C) — *recommended* | 16 scene tokens × 256, updated each moment from the pooled current tokens, the ego row and the previous pick's summary; read by the generator (cross-attention of the 256 queries into the 16 tokens before the first refiner pass) and by the scorer (attention-pooled scene vector concatenated to every proposal feature) | one update block (transformer layer over 16 tokens + 32 pooled image tokens), one read layer per part | same | ≈ 1.5 M | same |
| **3. Cached-token attention** (C2 B) | the last 16 token maps (8 s) with (Δpose, Δt) key embeddings; the generator's queries cross-attend to them; the scorer attends from each proposal feature to the same keys | ring buffer of 16 × 512 × 256 | same | ≈ 1.7 M (two cross-attention layers + embedding) | ≈ 3 h; buffer 4 MB bf16 per scene |

**Cost basis.** 0.1.8b-p06: 5,580 steps and 7 checks in 4 h 44 min on the shared GPU (3.05 s/step wall, C3), encoder gradient included. With the encoder frozen and its tokens cached, the step runs 4 refiner passes + replay + scorer + memory for 32 moments and waits ≈ 0.18 s for the grading pool; argued ≈ 0.6 s/step → a pass of ≈ 3,900 window-steps ≈ 40 min, 3 passes ≈ 2 h, plus 5 checks per pass at ≈ 3 min with cached features (2,000 moments in clip order + grading 25 s) ≈ 45 min: **≈ 3 GPU-hours per run**. One-off feature cache for the training, selection and held-out slices: ≈ 138,000 moments × 0.18 s (the H05 forward floor at batch 1, `MEM1/MEMORY_STEP1.md:144`) ≈ **7 GPU-hours**, less with batching. With a trained LoRA instead: ≈ 3.05 s/step × 4,650 × 3 passes + checks ≈ **14 GPU-hours per run**. These are wall hours on the shared RTX PRO 6000 at the load the lane saw; they are argued from the 0.1.8b run and the H05 cache run, not measured.

**Recommendation: design 2.** It needs no per-path identity (so no H15-style matching and no unverified slot-persistence assumption), its state is tiny and survives the 2 Hz divider and the per-scene reset trivially, both parts read one state so the "memory in generator and scorer, trained together" requirement is met with one mechanism, and its cost is the same as the others. Design 1 is the fallback if slot persistence turns out to hold (OPEN 11); design 3 is the fallback if recurrence proves hard to train on 8 s windows (no warm-up, explicit window).

### D2. The comparison

**Arms.** (a) design 2, memory carried; (b) the identical model and training with the state zeroed at every moment (sidecar mode `reset`, as H15's wiped arm); (c) 0.1.8b-p06 unchanged (`3d566eb2…8e66`); 0.1.7 (`29d5f951…2916`) reported beside every number, never re-trained. Three seeds for (a) and (b) (seed 0, 1, 2 for the window sampler, the memory initialisation and the GridMask-free stream); (c) and 0.1.7 are fixed weights. **GPU cost:** 6 runs × 3 h + 7 h cache ≈ **25 GPU-hours** offline (vs ≈ 90 with a trained LoRA); closed loop in D3.

**Slices.**
- *Training:* the s3_assembly manifest (126,981 moments / 4,015 clips / 22.3 h) **minus the carve-out below**; windows of 16 moments.
- *Selection slice (replaces the picking set):* carve **60 NuRec training clips** (≈ 1,900 graded moments, ≈ 0.33 h; whole chunks, stratified by type A / B_pass / ordinary) out of the training manifest before training. They are graded (grade, collision, drivable area), consecutive (whole clips, so memory can run), and nobody has chosen anything on them. Cost: no download; 1.5 % of training moments and 8 % of graded moments. The alternative, a new slice from untouched raw validation clips (90,627 untouched raw_val clips), costs ≈ 7 min of download per 300 clips (4.09 GB, `G/s1_build/REPORT.md:110`) plus conversion (≈ 1 h, 22 GB cache) but is ungraded (C4), so it could only select on human closeness. The picking set stays for the two safety gates only (hard-brake picks ≤ 0, pull-away go rate ≥ 66 %), never for the choice.
- *Held-out, read once:* a **fresh 300-clip raw_val slice** mined with the same rule as the current one (82 pull-outs + 75 fast + 143 ordinary, seed changed; `G/s1_build/REPORT.md:107-110`), because the existing held-out was already read for 0.1.8 (`G/s3_offline`). Cost ≈ 7 min download + ≈ 1 h conversion + the feature cache (9,581 × 0.18 s ≈ 30 min). It is ungraded, so it carries ADE (3 s), unsafe-pick proxies from ego speed and the human path, and the memory-moment subset. Plus a **graded held-out of 60 more NuRec training clips**, carved the same way as the selection slice and read once (the only way to get a grade on clean, consecutive moments; cost another 1.5 % of training moments).
- *Memory-moment subset:* the moments within 5 s after a hard brake (≥ 3 m/s²), a cut-in or a crossing by the B4 rules (`src/b4_events.py`), as the other session will write them to `out/memory_moments_{train,heldout}.parquet`. On the training slice B4 measured ≈ 5 % of moments on the clips that have tracks (NuRec: 149 + 621 + 436 of 22,041); on the raw held-out it needs the fetched tracks. If a fresh held-out is mined, the subset must be recomputed on it with the same script (the file the other session writes refers to the current 300 clips).
- **Never trained on or chosen on:** the 30 AlpaSim scenes and gate10 (0 overlap by id and chunk, B3), both held-out slices, the fresh raw held-out's clips and chunks (excluded from training by chunk, as the current one is).

### D3. Pre-registered predictions (draft)

- **Primary measure:** mean pick grade (the scorer's own pick, no comfort re-score) on the **graded held-out** (60 NuRec clips), arm (a) − arm (b), paired by moment, 3-seed mean with a clip-clustered 95 % bootstrap interval. Reported on all its moments **and** on its memory-moment subset. Predicted size: +0.005 on all moments, +0.02 on the memory subset. **Useful** = the all-moment difference ≥ +0.005 with the interval excluding 0 (the H15 bar, `MEM2/MEMORY_H15_STEP3.md:145`) **and** the memory-subset difference ≥ +0.01. Reported beside: (a) − (c) and (b) − (c) (does training with the extra layer, even wiped, move the start?), and 0.1.7.
- **Secondary, offline:** scorer-pick ADE (3 s) on the fresh raw held-out, all moments and the memory subset (prediction: (a) − (b) ≤ −0.05 m on the subset, about 0 elsewhere); hard-brake picks at > 10 m/s (must stay 0); go rate where the human pulled away (≥ 66 %); unsafe-pick rate on the graded held-out; menu variety and closest-of-32 ADE (3 s) (the generator must not collapse: closest-of-32 within +0.02 m of (c)).
- **Closed loop, native harness, one rollout per scene (deterministic for fixed weights):** stage 1, the 12 diagnosis scenes (`G/s10_018b_closed_loop/tables/scenes_12.txt`) for every seed of (a) and (b): bar = mean ≥ arm (c)'s 0.5919 and no at-fault crash that (c) does not have, per seed. Stage 2, the 30 scenes for the arms that pass, all 3 seeds, i.e. up to 180 scene-runs ≈ 30 h wall at ≈ 10 min per scene. **Smallest detectable difference:** no seed-to-seed spread exists yet for any model; the three 0.1.8b snapshots (different checkpoints, not seeds) spread 0.5432–0.5919 on 12 scenes. Assuming a seed-to-seed SD of 0.04 in the 30-scene mean, the standard error of a 3-seed difference is 0.04 × √(2/3) ≈ 0.033, so a difference below **≈ 0.07** cannot be told from seed noise; a single scene cannot be evidence (near-tie scenes flip by up to 0.55 under a different driver, C5). Prediction: (a) − (b) ≥ +0.03 on the 30-scene mean, which is below the detectable bar, so the closed loop is a **no-harm** check, not the primary evidence.
- **Establishing arm (c):** 0.1.8b-p06 has no 30-scene and no gate result. Run the 30 scenes natively once (≈ 5 h wall; 12 of them exist, `NAS/018b/closed_loop/runs/`, and can be reused if the simulator image check passes again) and report gate10 as the subset of those 30. No Paddock run (it would be `valid: false` and would use the 0.1.4b preprocessing).
- **Decision rule.** Go to a 30-scene run only if the primary holds. Call the memory useful only if the primary holds **and** stage 1 passes for ≥ 2 of 3 seeds **and** the 30-scene mean of (a) is not below (b) by more than 0.07 **and** no new at-fault crash against (c) appears in ≥ 2 seeds. Anything else: stop, write it up, and keep (c).

### D4. Optional history probe (costed only)

Question: do 5–16 s of Physical AI AV history carry information about the human's future beyond the model's two frames? Reuse `GM/probe.py` (ridge / logistic / MLP read-outs over windows of k moments, log-clustered bootstrap, `probe.py:55, 215-315, 371-520`) and `GM/probe_control.py` (car-state-past control, `:47-58`) with 0.1.8b's token maps.

Steps (nothing run):
1. Feature cache: hook the merged 0.1.8b encoder output (the 16 × 32 × 256 map, the same grid as H05's `pool_map`, `probe.py:51, 66-72`) and the (11,) ego row for every training moment, pooled in the hook to 8,448 floats (4.3 GB) instead of the full map (66 GB on NAVSIM). Cost ≈ 126,981 × 0.18 s ≈ **6.3 GPU-hours** at batch 1 (less batched); **free if D1's cache is built anyway** (store the pooled copy beside it).
2. Moments table in `ticks.py`'s layout (`GM/ticks.py:65-67, 282-315`): log = clip, chain = 0, chain_pos = moment index, split = train/held-out; chain_pos ≥ k for window k.
3. Targets from the (8, 3) human future at 0.5 s steps (speed at +1/+2/+3 s by centred differences; braking onset by the `ticks.py:175-181` rule on second differences) — a new `targets_for_log` (`probe.py:141-168`); current speed and acceleration from the ego row.
4. Windows k = 1, 2, 4, 8, **16, 32** (0.5 … 16 s), compared against k = 1 (the model's own view, `MEM1/MEMORY_STEP2C_CONTROL.md:7`), with the car-state control at every k; save per-row predictions and bootstrap draws so k = 16 and 32 get paired intervals without a refit (the H05 gap, `MEM1/MEMORY_STEP2C_CONTROL.md:19, 157`).
5. Reads: `log_weights`, `fit_ridge`, `fit_logit`, `fit_mlp`, `go_rule` unchanged (`probe.py:199-325, 556-589`); `evaluate` / `main` paths re-pointed (`:376-405, 645`).
Cost: ≈ 95,000 rows at k = 8 (≈ 2.2 × NAVSIM's 43,172), probe ≈ 1.5–2 h on cores 0–1 (H05: 54 min for 43k rows, `MEM1/MEMORY_STEP2_PROBE.md:216-225`) plus the control ≈ 1 h; with k = 16 and 32 added, ≈ 3 h CPU. Decision value: if no window beyond k = 2 beats k = 1 on the events slice with the car-state control subtracted, design 2's expected gain is small and the campaign should start with the cheaper arm (b)-vs-(c) question (does the extra layer alone move 0.1.8b?).

### D5. Risks and mitigations

| risk | why it is real here | mitigation |
|---|---|---|
| many changes at once (sequence sampler, cached features, memory layers in two parts, new slices) | the lane lost two months to scorer-style side effects (`G/CLOSING_REPORT.md` §4) | zero-output memory init with a bit-identity gate at step 0 (G1 as before); arm (b) isolates "sequence training + new layer" from "memory carried"; a G2-style routing table per lesson; the same five offline checks per 930 steps |
| rare memory-relevant moments | B4: ≈ 5 % of moments on the clips with tracks; 75 % of clips have no tracks on the NAS | the memory-moment subset is a reported slice with its own prediction; over-sample windows that contain an event ×2 in the sampler (OPEN 15) and report with and without it; fetch the ordinary raw clips' tracks (≈ 1.7 GB) so the subset is defined on all 4,015 clips |
| starting from a model below 0.1.7 | 0.1.8b-p06 12-scene 0.5919 vs 0.6742, truck crash on 8023dce5 (A2) | every table carries 0.1.7; arm (c) is established on the 30 scenes first; the 12-scene bar is relative to (c) so a memory gain is measurable even if 0.1.7 is not reached; the truck scene is reported on its own |
| single-seed noise | training is not bit-reproducible (`G/CLOSING_REPORT.md:230`); no seed spread exists for any model | 3 seeds for (a) and (b) (affordable only because the encoder is frozen: 3 h per run); the closed-loop rule needs ≥ 2 of 3 seeds |
| the ungraded held-out slice | raw clips have no map, so no grade (C4) | a 60-clip graded held-out carved from NuRec training clips, read once; the raw held-out keeps the human-distance measures |
| the frozen encoder hides a gain that needs new features | the lane's LoRA variants changed the scorer's inputs and broke it (`G/CLOSING_REPORT.md:138-144`) | state it as a limit of this campaign; the trained-LoRA variant is costed (14 h per run) as a second campaign if (a) − (b) is positive but small |
| the checks and the adapter do not run clips in order | the check code is batch-1 random order; the native worker has no state | both are listed in OPEN; the check must replay whole picking/selection clips in time order, and the worker gets the session-keyed state and reset message (C5) |

### Flags from run 2

19. **Flag 5 corrected:** the 0.55 per-scene swing in `gate.md:490-496` is bridge-vs-native, not a same-code repeat (C5). No same-code repeat table exists in either harness.
20. **Two per-step figures for 0.1.8b disagree** (3.05 s/step wall with checks vs 3.49 s/step from log stamps, C3); the cost estimates use 3.05.
21. The brief's Paddock folder is a stale branch (C5); the live bridge is in `Paddock-gate` / `paddock-memory`.
22. The Paddock bridge emulates 0.1.4b (no bonnet mask, no zeroed acceleration, no batch fix) and marks any non-released checkpoint `valid: false`.
23. The temporal self-attention's previous-BEV slot is unused in the proposal-query path (C1): a zero-parameter memory entry exists in the vendor code.
24. The held-out slice has no scorer targets except human closeness (C4); graded clean moments exist only inside NuRec training clips.
25. The memory folder was renamed on 2026-11-07; `approach3_pai` is a symlink. The other session's `memory_moments_*` files had not arrived at 02:10 IST.
26. **NOT CHECKED:** whether query slot i stays the "same path" across consecutive moments (needed for design 1 / D-by-slot). Estimate: 20 min CPU on the stored picking-set menus of consecutive moments (`NAS/018b/runs/018b_attempt2/dumps/p00.6.npz`), aligning slot i at t to slot i at t + 0.5 s by the ego motion and measuring ADE (3 s).
27. **NOT CHECKED:** the per-step cost with cached features (argued ≈ 0.6 s/step); a 30-step timing on cached tokens would take ≈ 15 min once a cache of ~1,000 moments exists.

### OPEN decisions for Sid

1. **Harness for all closed-loop arms.** Recommend: native adapter (deterministic, 0.1.8b preprocessing, pinned sha), gate10 reported as the subset of the 30. Alternative: Paddock bridge for comparability with the gate site, accepting `valid: false` and the 0.1.4b preprocessing.
2. **Encoder LoRA.** Recommend: frozen (merged 0.1.8b), tokens cached, 3 h per run. Alternative: a fresh LoRA trained with the memory, 14 h per run, no cache, GridMask and hide-and-guess kept.
3. **Design.** Recommend: design 2, scene tokens read by both parts. Alternatives: design 1 (slot memory; needs OPEN 11 first) or design 3 (cached-token attention; explicit 8 s window, no recurrence).
4. **History length.** Recommend: 16-moment (8 s) training windows, warm-up 4, recurrent state unbounded in the car. Alternative: 32-moment windows (16 s) on the 3,721 full clips only, or an explicit 8 s buffer (design 3).
5. **Selection slice.** Recommend: carve 60 NuRec training clips (graded, consecutive) out of training; the picking set keeps only the two safety gates. Alternative: a new raw_val slice (ungraded; selects on human closeness only) or keep the picking set (not clean).
6. **Graded held-out.** Recommend: carve 60 more NuRec training clips, read once. Alternative: none (grade reported only on the selection slice, which is then not clean for reporting).
7. **Raw held-out.** Recommend: mine a fresh 300-clip raw_val slice (≈ 1 h + 22 GB) and read it once. Alternative: re-read the existing 300 (already opened for 0.1.8; 0.1.7 and 0.1.8 numbers exist for continuity).
8. **Seeds.** Recommend: 3 per trained arm (a, b), ≈ 25 GPU-hours offline in total. Alternative: 1 seed (≈ 13 GPU-hours), with the H15 caveat that P1 was seed-sensitive.
9. **Primary measure.** Recommend: pick grade on the graded held-out, (a) − (b), useful at ≥ +0.005 overall and ≥ +0.01 on the memory subset. Alternative: ADE (3 s) on the raw held-out as primary (larger slice, but the lane found ADE does not predict closed loop).
10. **Closed-loop stages.** Recommend: 12 scenes per seed with a bar relative to arm (c), then 30 scenes for passing arms. Alternative: 30 scenes for seed 0 only.
11. **Slot-persistence check** (flag 26): recommend running it (20 min CPU) before any design-1 work. Alternative: skip, since design 2 does not need it.
12. **Establish arm (c)**: recommend one native 30-scene run of 0.1.8b-p06 now (≈ 5 h), reusing the 12 existing scene runs if the image check passes. Alternative: wait until the offline primary holds.
13. **Run the D4 probe first?** Recommend: yes if the feature cache is built anyway (adds ≈ 3 h CPU, 0 GPU); otherwise no. Alternative: skip and let arm (b) vs (c) answer the cheaper question.
14. **"Wiped" definition.** Recommend: sidecar mode `reset` (state zeroed every moment, same weights and training), as H15. Alternative: train arm (b) with no memory layer at all (then it is not "identical model").
15. **Event over-sampling** in the window sampler (×2 for windows containing a memory-relevant event). Recommend: yes, reported both ways. Alternative: uniform windows.
16. **Memory-moment files.** Recommend: the other session writes to `approach3_pai/out/` (the symlink) and, if OPEN 7 picks a fresh held-out, re-runs `src/b4_events.py` on it. Alternative: keep the subset on the existing held-out and re-read that slice (conflicts with OPEN 7).


---

## Files in this folder

- `src/probe_manifests.py`, `src/probe_timing.py`: schemas, counts, read timings.
- `src/b1_b3.py`: slice hours, id and chunk overlap. `src/b3_timerange.py`: the time-range attempt.
- `src/b2_gaps.py`: history per moment (train: all 4,015 clips; held-out: all 300).
- `src/b4_events.py` (`nurec` / `braw`), `src/b4_extra.py`: event census, manifest-speed braking, track-gap check.
- `src/b4b_fetch.py`, `src/b4b_verify.py`, `src/b4b_events.py` (`nurec` / `braw` / `ordinary` / `heldout`), `src/b4b_moments.py`: B4b track fetch, verification, event census on all clips, memory-moment lists (`out/memory_moments_{train,heldout}.parquet`, `out/b4b_summary_{train,heldout}.csv`, `out/b4b_*_events.csv`).
- `out/`: JSON summaries, per-clip CSVs, `sha256_018b_releases.txt`, `slice_clips.json` (clip lists of train / held-out / picking).
- `logs/`: stdout of every run, with timings.
- `src/c1_params.py`, `logs/c1_params.log`: parameter count per part from the 0.1.8b-p06 file (run 2).
- Run 2 traces (training loop, adapters, probe) were read-only; their citations are inline in parts C–D.
