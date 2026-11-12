# Recon: Drive-JEPA (front camera) on Bench2Drive

05 Nov 2026. Read-only. Short report; every cite in full, with tables and scripts, is in `section_A.md`, `section_B.md`, `section_C.md`, `section_DE.md` next to this file. Numbers only.

**Cite key.** `v2/` = `wa-jepa/Drive-JEPA-upstream/navsim_v2` @ 548bb82; `A/` = `v2/navsim/agents/drive_jepa_perception_based`. `E1t` / `E1b` = Sid's `experiments/drivejepa_kitscenes/03_encoder_training` / `02_encoder_build` in `{sid_checkout}/Gallops-kitscenes` (PR #24 head e75aaac, open). `RT` = same repo, `experiments/drivejepa_retrain`. `PAI` = `Gallops-sid/LORA Refiner/Generator-Joint-PAI-alpasim`. `SM` / `AC` = `Evaluations/Bench2Drive/.../statistics_manager.py` / `atomic_criteria.py`. `RN` = `paddock-ops/worktrees/rerank-native`. `R3` = `run3a-adapted/CLAUDE.md`. `P§` = paper, arXiv 2601.22032v2.

**Checked by hand on review:** vocabulary speed limits, 968 / 32, the penalty table and DS formula, Sid's lr / EMA / batch / precision, and that `train.py` deletes checkpoints. Two errors found and fixed: section C said the 01-05 Nov logs don't exist (they do; no Phase 0b entry in them), and sections B and D described different runner crops (see D).

## A. Drive-JEPA's own two stages

**Stage 1 pretraining.** Code is generic only: Meta's V-JEPA 2 trainer (`v2/vjepa2/app/vjepa/train.py`); the only configs are K710 / SSv2 / HowTo at 256 px (`vjepa2/configs/train/vitl16/pretrain-256px-16f.yaml:9-12`). No driving config, dataset list or clip builder; crops are square only (`app/vjepa/transforms.py:102-103`).
- Objective: L1 from predicted to layer-normed EMA target, on hidden tokens (`train.py:414-435`, EMA `:457-465`).
- Clip: 8 frames, 2 Hz, 512×256, front camera; CoVLA + DrivingDojo + OpenScene, 330 h (P§3.2).
- From the saved state `vitl_merge_3dataset_e50.pt` (sha256 `46491827…668f`): 15,300 steps, batch 64 × 8 GPUs, last lr 5.25e-4, wd 0.04, loss 0.4846; predictor 12 blocks, width 384, 22.1 M.
- Driving masks, EMA value, warm-up: **not found**. Generic config: 8 blocks × 0.15 + 2 × 0.70, EMA 0.99925 (`pretrain-256px-16f.yaml:41-67,87-89`).

**Stage 2.** Everything trains: Adam 1e-4, backbone × 0.1 = 1e-5, no scheduler, 20 epochs, batch 64, 16-mixed (`A/drive_jepa_agent.py:356-363`, `A/drive_jepa_config.py:51`; confirmed in the released checkpoint's param groups).
- Losses (`A/drive_jepa_agent.py:313-323`): human path L1 × 1; teachers × 0.5; earlier rounds × 0.1 (`prev_weight`); score BCE × 1; area BCE × 2; collision BCE × 1, L1 × 0.1.
- **Scorer labels come from the 32 live proposals, simulated every step, not from vocabulary grades** (`A/drive_jepa_agent.py:276-278`, `A/score_module/compute_navsim_score.py:52-76`). Plan item 3 differs from this.
- Teachers: all 8,192 scored offline, keep final ≥ 0.95, random cap 256 (`v2/scripts/misc/calc_anchors_scores.py:12-19`, `convert_to_index.py:18-22`); K = 4 drawn per round (`A/drive_jepa_agent.py:237-243`). The scripts import a module not in the repo (`calc_anchors_scores.py:8`).
- Vocabulary: `data/8192.npy`, sha256 `cc44a31e…ad35`, k-means on 100k+ paths (P§3.4), script not found. Feeds teachers only (`A/drive_jepa_agent.py:246`).
- Pick: sigmoid of last logit, argmax of 32 (`A/drive_jepa_model.py:91,96-97`). Momentum `(14S + 2S_c)/16` only through `run_pdm_score_sequential.py:121-138`.

**Input contract.** `cam_f0`, frames t − 0.5 s then t, rows `[28:-28]`, `cv2.resize` to 512×256, ImageNet norm (`A/drive_jepa_features.py:47-59`, `A/drive_jepa_model.py:43-48,63`). 512 tokens. Ego: `x, y, heading, vx, vy, ax, ay, cmd×4`, current frame only (`A/drive_jepa_features.py:85`, `A/drive_jepa_model.py:33,51`). Slot meaning of the command is not in this repo.

**Bench2Drive in the repo.** Configs target iPad's `PadAgent`, not Drive-JEPA (`navsim_v1/.../agent/b2d_agent.yaml:1,5`). An offline grade exists (`navsim_v1/navsim/agents/pad/score_module/compute_b2d_score.py:434`). Missing: `b2d_8192.npy`, cache, converter, closed-loop agent, checkpoint.

## B. Sid's code (also Step 0 of the LoRA spec)

**Experiment 1, hide-and-guess.** `E1t/src/train.py warmup` then `main`; settings in `E1t/src/s2common.py:33-59`, no config file.

| Item | Value | Source |
|---|---|---|
| LoRA | rank 16, alpha 16, qkv / proj / fc1 / fc2 × 24 blocks = 96 layers, 6,291,456 params; A Kaiming, B zero | `E1b/src/lora.py:27-29,42-44` |
| Masks | 8 × 0.15 and 2 × 0.70, spatial only (same patches in both frames), equal count per batch; hides 61 % / 81 % | `E1t/src/masks2.py:1-26`, `E1t/REPORT.md:93-94` |
| Guesser | V-JEPA predictor from e50, 12 blocks, width 384, 22.1 M, grid fix | `E1b/src/players.py:130-139,176-188` |
| Slow copy | deep copy of the encoder with its own LoRA; EMA 0.99925 on LoRA tensors, main phase only | `E1t/src/train.py:110-113,145-146` |
| Loss | L1 to layer-normed target at hidden tokens, mean of 2 mask configs | `E1b/src/players.py:224-241` |
| Optimiser | AdamW; LoRA 2e-4 wd 0; **guesser 1e-4** wd 0.04; 200-step ramp per phase | `E1t/src/s2common.py:43-49` |
| Batch, precision | 32, bf16 | `E1t/src/s2common.py:52,54` |
| Steps | warm-up 1,000 (guesser only); main 3,522; chose pass 3 = 1,761 | `E1t/REPORT.md:111-115,130-141` |
| Step time | 0.44 s at build; 0.54–1.34 s in the run, shared card | `E1b/REPORT.md:181`; `main_steps.jsonl` |
| Starts from | released **agent** encoder, sha256 `740335c8…0d05`; our runner uses Naren's copy, same size, same recorded hash, not re-hashed | `E1t/src/train.py:99`, `PD/models/drivejepa/model.yaml:20` |

Differs from V-JEPA 2: 2 frames vs 16, one time slice vs 8, LoRA + guesser vs whole encoder, starts from the agent encoder, guesser from e50 with a warm-up (section_B table).

**Against the spec's Fixed table.** Rank, alpha, layers, B zero, EMA, batch, precision all match. Differences:
1. lr 2e-4 is the LoRA's only; guesser 1e-4, with a ramp.
2. A guesser-only warm-up phase exists that the table doesn't list.
3. **Sid's stop and pass choice use planner ADE vs human paths** (`E1t/src/train.py:330,343-349`); the spec says no labels or planner, and gives no replacement. His fixed-set loss was also lowest at pass 3 (`E1t/REPORT.md:132-138`), though he warns it isn't strictly comparable across passes (`:206`).
4. Stops and saves are per pass, not per step; no fixed step count exists.
5. `train.py` deletes and overwrites checkpoints (`:159-160,329-336`), and saves no RNG or sampler. Full version: `PAI/s9_018b_train/src/train_018b.py:1088-1095`.
6. "Same next 10 losses" can't be bit-exact: 3.4642 vs 3.4621 at +10 steps (`PAI/s9_018b_train/REPORT.md:63`).
7. Reboot watcher exists only in `PAI/s9_018b_train/src/`, hard-wired to Sid's paths, restart path "untested end to end" (`:66-67`).
8. `s5_all_three` is Experiment 3, not 1 (`Generator-Joint-KIT/s5_all_three/src/allthree.py:67,223`). On PAI the LoRA was fresh and hide-and-guess ran only jointly with the planner (`PAI/s1_build/src/pai_model.py:6`). Experiment 1 is the only standalone run.
9. `FRAME_STEP` is both pair gap and sampling stride (`E1t/src/s2common.py:31`, `cache.py:54-60`); the spec's N needs them split.

**Bench2Drive changes.** As is: `lora.py`, `players.py`, `masks2.py`, the loss, the merge. Must change: loader, frame cache source, splits, label-free stop rule, new-path checkpoints, RNG + sampler saves, watcher paths. KIT's selection check doesn't carry over.

**PAI retrain 0.1.7.** Its code is in `RT`, not the PAI folder, which is the 0.1.8 family (`PAI/CLOSING_REPORT.md:44`).
- Training: `RT/06_first_training_run/s4_train.py` (generator + scorer), `RT/12_second_training_run/s10_gen_train.py` (generator); encoder frozen.
- Scorer grades: online on the 32 plans, `NC·DAC·TLC·DDC × (5EP + 5TTC + 2LK + 2HC)/14`, human path as EP reference (`RT/03_grade_candidates/s2_grade.py:85-86`).
- Teachers: 10,240 shapes offline, grade ≥ 0.95, cap 256 (`RT/04_training_set/s3_grade.py:57-68`). Vocabulary widened by 2,048 fast shapes, top 40.7 m/s (`RT/04_training_set/s3_vocab.py:9,18-21`).
- Stop rules: tripwires on unseen error, 4 s distance, non-finite, caps (`s4_train.py:142-181`).
- Export check: same saved features through the car's worker; 1.49e-4 m on 16 examples (`RT/07_closed_loop_test/runs_017/tables/gate_017.json`).
- The grader is NAVSIM's formula on PAI scene files: plan item 3 needs a new one.

**Results worth having in view.** KIT Experiment 1: pick ADE 2.372 → 1.879 m, closest 0.374 → 0.378 (`docs/drivejepa-kitscenes.md:168-169`). PAI: LoRA run 0.5379 vs 0.6742 over 12 scenes, and offline ADE did not predict closed loop (`PAI/CLOSING_REPORT.md:33,107,117-122`). KIT Experiment 3 on Bench2Drive: DS 59.04 → 0.27 on 7 routes, cause not verified (`docs/drivejepa-kitscenes.md:246,258-261,275`).

## C. Bench2Drive scoring

**Score.** Route: `max(RC × P, 0)` (`SM:413`). Benchmark: plain mean (`SM:455`). P multiplies per event (`SM:23-29,35`): pedestrian 0.5, vehicle 0.6, static 0.65, red 0.7, stop 0.8, scenario timeout 0.7, yield to EV 0.7, outside lanes (1 − % of completed route). Min speed unused (`SM:36-37`).

**What ends a route.** Completed; deviation at 30 m, or 15 m for over 30 % of the route (`AC:1400-1413`); blocked 60 s under 0.1 m/s (`AC:447-459`); 4,000-tick cap = 200 s (`MGR:182-183`); route timeout; agent or simulator crash. Collisions never end a route. The 200 s cap is below the 240 s scenario timeout: in 96 records, TickRuntime fired 38 times, scenario and route timeouts 0.

**Offline from Phase 0a labels** (section_C has the full table).

| Yes | Partly | No |
|---|---|---|
| Red light on the log's timing; lane distance | RC (no ordered route or length); collisions (log-replay agents; 7 of 8 layout hits were objects not in the labels); stop sign; wrong lane; deviation; blocked | Tick cap, route timeout, scenario timeout, yield to EV, min speed |

DS is route-level, so a 4 s per-frame grade can only stand in for it. No controller rollout exists in Phase 0a.

**Phase 0b is not built:** spec only (`R3:69-91,148-194`), no code, no scores. Nearest is MTD (`paddock-ops/mtd/scorer.py`): 4,096 paths on 80,598 frames, older tables, no stop-sign, yield, blocked or route term, Driving lanes only while the benchmark also allows Parking. Term-by-term table in section_C.

**Sid's warning (go vs stay at a pull-away).** Measured on the 968 clips, 64-path menu, MTD vocabulary:

| Grade | Expert moves ≤ 5 m | Expert moves > 5 m |
|---|---|---|
| Ties, MTD total | 4 of 582 | 0 of 293 |
| Ties, Drive-JEPA's 5 m progress rule, expert as reference (approximate) | 524 of 582 | 0 of 293 |

Cause: `A/score_module/train_pdm_scorer.py:78,243-254`. Bench2Drive's own rules see go vs stay only through RC, unless go runs a red or leaves a stop zone unstopped. MTD avoids ties, but 53.8 % of safe go plans grade below stationary when the expert moves ≤ 5 m. Phase 0b's formula has no tie, but needs a speed limit, missing on 90 % of lanes (`step3_checks_b2d_v4.json`).

## D. Data and inputs

**Frames.** 968 clips: 895 in `paddock-data/phase0a-labels/b2d_base`, 73 in `b2d_pilot`; `<clip>/camera/rgb_front/NNNNN.jpg`; 47,831 frames, 1600×900 (`phase0a-labels/results/b2d_full_run_01nov.json:9`). Local copies keep `frame % 5 == 0` of a 10 Hz source (`b2d_full_fetch.py:1-2`), so neighbouring files are 0.5 s apart: 46,863 pairs before N. At batch 32 that's 1,464 steps per pass (arithmetic).

**Crop: "the runner's function" has several versions.** The function itself is Drive-JEPA's `_get_camera_feature` (`A/drive_jepa_features.py:38-65`); the 28 is a fixed row count, so the result depends on input size.

| Path | Where | Input → crop | Scale fx, fy at 512×256 |
|---|---|---|---|
| NAVSIM training | | 1920×1080, 28 rows | 412.0, 386.3 |
| `native` | `RN` default (`policy.py:99`); **all 24- and 50-route Drive-JEPA runs** | 800×450, 28 rows | 365.6, 371.2 |
| `openscene` | `origin/main` since #106 | window → 1920×1080, 28 rows | 412.0, 386.3 |
| Upscale to 1920×1080 | current `Paddock/` checkout (`nuscenes-benchmark` a75d304, `policy.py:65-82`); Run 2 / 3A (`wajepa_dj/policy.py:30-44`) | 800×450 → 1920×1080, 28 rows | 365.6, 342.8 |
| Phase 1 256×512 copies | `phase1/data/geometry.py:34-35` | 1600×900, 23 rows | 365.6, 342.5 |

The function run directly on 1600×900 frames matches none of these. The spec's check 3 can't pass on the Phase 1 copies.

**Camera.** Base frames: 70°, x 0.8, z 1.6. Runner: 70°, x 1.5, z 1.5 (`bench2drive/agent.py:66,83`). NAVSIM: 63.7°, x 1.62–1.69, z about 1.52, pitch −1 to −2°, rear-axle origin. Bench2Drive's ego origin is 1.3886 m ahead of the rear axle (`phase1/data/geometry.py:10-13`). So camera ahead of rear axle: NAVSIM 1.66 m, base frames 2.19 m, runner 2.89 m. No crop removes this; check 3 can't see it.

**Runner inputs.**
- FOLLOW: sent as `unknown` to `drivejepa` and all head runs (`RN/.../drivejepa/policy.py:69`). The fix is only in `WajepaDjPolicy` (`wajepa_dj/policy.py:94-104`). `unknown` appears in none of 103,288 NAVSIM samples (`phase1-targets-v2/stats.json`).
- Speed `[speed, 0]`; acceleration `[dv/dt, 0]` over one tick (`agent.py:444-450`). Anno `acceleration` reads as an accelerometer, z about 9.8.
- Command: nearest plan point, LEFT / RIGHT only 0.1–1.2 s before the turn (`agent.py:152-153`). Bench2Drive mix is 92.7 % straight vs NAVSIM 63.3 %.

**Vocabulary speeds.** Mean-speed max 14.73 m/s, peak 14.79, 0 anchors at or above 15 (recomputed on review). Base-set expert: p99 10.7–11.0, max 14.98 m/s over 79,022 frames (`mtd/frames.parquet`, not a random sample). Traffic: 1.24 % of moving vehicles at or above 14.73. Speeds the 220 routes need: not measured. Sid's limit: `RT/03_grade_candidates/REPORT.md:408-409`.

**Overlap.** 968 / 32 confirmed: 23 near the 220, 9 near v0.0.4 val (`phase0a-labels/results/b2d_overlap_drop.json`, sha256 `3afd60ab…`). The rule checks route start, not scenario trigger, so **24 same-scenario clips within 10 m of the trigger of 13 eval routes remain** (`section_DE_evidence/twins.txt`), 4 of those routes in our 50: 2086, 2286, 18311, 24224.

## E. Baseline, 50 routes unless stated

| Checkpoint | Hz | Command | DS | RC | Source |
|---|---|---|---|---|---|
| Released | 20 | unknown | 31.71 | 39.10 | `RN/results/bench2drive/drivejepa/pooled50-…/metrics.json` |
| Released, 24 routes | 20 | unknown | 31.11 | 40.44 | `paddock-ops/tables/bench2drive_official_24.json` |
| **Released** | 2 | unknown | **57.60** | 70.97 | `RN/results/bench2drive/drivejepa/native50-2hz-…/metrics.json` |
| Split-v4 | 2 | unknown | 68.09 | 85.55 | `paddock-ops/splitv4/all50.json` |
| **Final** | 2 | unknown, look-ahead | **73.63** | 87.51 | `paddock-ops/tags/final-73.6/results50.md` |
| Run 2 | 2 | straight, 2 reps | 51.70, 51.95 | 76.07, 74.18 | `{runs_p3}/run2_cmdfix` |
| Run 3A | 2 | straight | 63.29 | 79.02 | `{runs_p3}/run3a_cmdfix/repa` |

About 30 smaller runs are in section_DE. No run on all 220 routes exists. Split-v4 and Final trained on the 1,000 clips before the 32 were dropped, so they include the twins. Paper's 64.52 is an unreleased variant. Runs aren't repeatable, and raw `results.json` was checked on 2 routes only.

## Missing or unclear

**Decisions for Navvrat**
1. **GPU order:** run 3A-M is running, M2 queued. LoRA training before, after or beside it?
2. **Which crop is "the runner's":** `native` (all published runs), `openscene` (`origin/main`), or upscale (current checkout, Run 2 / 3A)?
3. **LoRA stop rule:** Sid's uses planner ADE. Fixed step count, held-out loss, or his rule?
4. **Camera mount:** train on x 0.8, z 1.6 frames and run at x 1.5, z 1.5, or change one?
5. **Trigger twins:** keep the 24 clips?
6. **Spec wording:** add the guesser's lr 1e-4 and the warm-up phase to the Fixed table? The spec file needs your OK to change.
7. **Plan item 3:** Drive-JEPA's scorer learns from live proposals, not vocabulary grades. Which do you want? Either way a new grader is needed; DS can't be graded per frame as it stands.
8. **FOLLOW:** Stage 2 trains on straight; the released baseline ran on unknown. Change the runner for Drive-JEPA?

**Not found**
- Sid's HANDOVER and CONTEXT notes: not in `{sid_checkout}/`. Please paste them.
- Driving pretrain masks, EMA, warm-up, start lr, clip builder. Which K the released model used (config 4, run name `topk_anchors_1`).
- Phase 0b code. An ordered route for Base clips; per-frame scenario state.
- Command slot order in Drive-JEPA's repo; `command_near` lead time as one number.
- Trigger-twin check against v0.0.4 val: not run. Speeds the 220 routes need; whether the generator exceeds the vocabulary on Bench2Drive.
- Paddock commit for the Sep campaigns; Drive-JEPA commit beyond `pinned_sha` e21f474; `RN` has moved since the Final tag.
- `{weights}` in `gallops.toml` has no `Drive-JEPA-cache/`, so POINTERS.md doesn't resolve as written.
- Checkpoint sha256: E50 and the released checkpoint are recorded values, not re-hashed.

**For a look:** the off-lane check during ParkingExit is in docstrings only, not code (`parking_exit.py:184-213`), though `labels_b2d.py:334` relies on it. One Stopsign clip has no stop line (Town05 Route258).
