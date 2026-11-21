# Drive-JEPA memory, approach 3: build step 1 — the new slices and a token-cache sample

shakaal, 2026-11-07 (05:00–11:50 IST). CPU work pinned (`taskset -c 0-1 nice -n 19`); one GPU job (step 4, shared GPU). No training, no full cache, no closed loop, no docker, no pushes. Writes only under `{experiment_store}/` (new) and this memory folder (`src/`, `out/`, `logs/`). The earlier lane folders and `NAS/heldout/` were read, never written.

**How to read this.** Every number is from a file in `out/` or `logs/` of this folder, or from a path cited `path:line`. Scripts in `src/`; their stdout in `logs/`. Plain names: encoder (frozen V-JEPA ViT-L + projector), generator (32-plan menu), scorer (rating head). Distances carry their window, e.g. ADE (3 s).

Path abbreviations: `G` = `Gallops-sid/LORA Refiner/Generator-Joint-PAI-alpasim/`; `RD` = `Paddock/paddock-sid/Alpasim/analysis_3.0/Retrain-data/`; `NAS` = `{retrain_store}/`; `PAI` = `{dataset_store}/`; `A3` = `{experiment_store}/`; `UP` = the vendored upstream model (`Paddock/paddock-sid/Alpasim/adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based/`); `GM` = `gallops-memory/src/gallops/memory/`.

**Starting model, verified before any use:** `NAS/018b/release/drive_jepa_018b_p06.ckpt`, 1,273,003,357 bytes, sha256 recomputed at 05:08 = `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66` (equals the pin), and again inside the GPU job before loading (`out/s4_cache_sample.json: p06_sha_verified_s`).

## Summary

| step | result |
|---|---|
| 1. graded held-out | **355 clips / 232 chunks / 10,516 moments / 10,468 graded / 1.97 h**, reconstructed (NuRec) val_full clips outside the picking set's 46 chunks; 0 chunks shared with training, picking, the current held-out or the 31 exam chunks; every slim sample, metric cache and pool track file present. `A3/slices/manifest_graded_heldout.parquet` |
| 2. fresh raw held-out | **300 clips / 177 chunks / 9,566 moments / 1.67 h**, downloaded 4.13 GB (+ 0.16 GB tracks); **0 pull-outs** (flag 1), 75 fast, 225 ordinary; history median 8.00 s, 71.5 % with ≥ 5 s. `A3/slices/manifest_raw_heldout.parquet` |
| 3. memory moments | graded held-out **373 / 10,516 (3.55 %)**; raw held-out **540 / 9,566 (5.64 %)**. `out/memory_moments_{graded_heldout,raw_heldout}.parquet` |
| 4. cache sample | 1,022 moments (32 whole picking clips). **fp32 cache: exact** (picks 100 %, plans and logits 0 difference, grades identical). **bf16: 99.9 %** (1 pick of 1,022; max plan difference 1.53 m on that moment, mean 0.4 mm; pick grade 0.78409 → 0.78409). Encoder 0.024 s/moment at batch 1 fp32 (0.0077 batched 8); training step from the cache **0.057 s** per 32 moments (no grading). Full cache: ≈ 1 GPU-hour of compute, **78 GB fp32 / 39 GB bf16** for all four slices; the wall time is NAS-read bound (≈ 2 h at the lane's 25 samples/s, 74 h at the rate measured today under the download — flag 4). |

**Recommended precision: fp32** (exact; 67 GB for training on a NAS with 1.7 TB free). bf16 is near-exact and halves the disk; keep it as the fallback only if disk becomes the constraint.

---

## Step 1: graded held-out (val_full, CPU)

Script `src/s1_graded_heldout.py`; stdout `logs/s1_graded_heldout.log`; record `out/s1_graded_heldout.json` (363 s).

**Definition (Sid's decision):** the reconstructed clips of val_full (`RD/s3_assembly/val_full/manifest_val_full.parquet`, column `source == "nurec"`) whose chunk (`group`) is not one of the picking set's 46 chunks (`NAS/picking/picking_set.parquet`, column `chunk`, 46 unique).

| item | value | source |
|---|---|---|
| val_full | 24,657 moments / 801 clips / 499 chunks; 11,891 reconstructed moments in 401 clips / 265 chunks, 11,842 of them graded | `logs/s1_graded_heldout.log:1-2` |
| removed: reconstructed moments in picking chunks | 1,375 moments / 46 clips / 33 chunks | `:3` |
| **graded held-out** | **355 clips / 232 chunks / 10,516 moments / 10,468 graded (`has_label2`) / 1.97 h** (durations from the pool `clipgt__egomotion_estimate` timestamps: 20.0 s each) | `out/s1_graded_heldout.json: summary` |
| by type | high-speed A: 99 clips / 3,013 moments; close-pass B_pass: 43 / 1,173; ordinary: 213 / 6,330 | same, `by_type` |
| moments per clip | min 1, median 32, max 32 (anchors 0.5 … 16.0 s; one clip has a single moment) | same |
| 48 ungraded moments | reconstructed moments whose metric cache was not built (`has_label2 = False`, no `metric_cache.pkl`); kept in the slice, excluded from grade measures | `files.ungraded_with_metric_cache = 0`, `graded_without_metric_cache = 0` |

**Overlap checks (all empty):** chunks shared with training (1,398 chunks, `manifest.parquet: group`): **0**; with the picking set: **0** (by construction); with the current held-out (221 chunks): **0**; with the 31 exam chunks (public_2601 ∪ our 30 → `clip_index` chunks, as `NAS/heldout/src/b3_pool.py:18`): **0**; clip ids in the exam set or our 30: **0**; clips or sample ids shared with training: **0**; clips shared with the picking set: **0** (`out/s1_graded_heldout.json: checks`).

**Files exist and load:** slim samples (`PAI/s4_train/slim/<clip>/<sid>.npz`) 10,516 / 10,516; `pad_feature.gz` 10,516 / 10,516; `metric_cache.pkl` 10,468 / 10,468 graded; `pad_label2.gz` 10,468; pool tracks (`PAI/s0_census/pool/<uuid>/clipgt__obstacle.parquet`) 355 / 355 clips. Sample of 20 graded moments loaded: slim keys `ego_status, img_shape, lidar2img, png1, png2, token, trajectory`, shapes lidar2img (2, 4, 4), img_shape (2, 3), ego_status (4, 11), trajectory (8, 3), token equals the sample id on 20 / 20; metric cache unpickles (lzma) to `MetricCache` on 20 / 20 (`loaded_sample`).

**History per moment** (`src/b2_gaps_slice.py` = `b2_gaps.py` with its path block taken from the environment, every other line identical, `logs/b2_gaps_slice.diff` 11 lines; `logs/b2_graded_heldout.log`, 980 s): 0 clips with a pose gap, 0 with a frame gap; **median 8.49 s, 71.8 % ≥ 5 s, 90.6 % ≥ 2 s**; bins [0, 0.5) 13, [0.5, 1) 326, [1, 2) 647, [2, 5) 1,983, [5, 10) 3,267, [10, 20) 4,280.

**Caveat (recorded, not new):** val_full chose 0.1.6's pass 9 (`G/s0_recon/REPORT.md:284-287`); nothing was trained on it and nothing has been chosen on these 355 clips since. Read once, later, as Sid decided.

**Outputs:** `A3/slices/manifest_graded_heldout.parquet` (train schema, 15 columns, sha256 `006ec7c8…36d3`), `A3/slices/graded_heldout_clips.csv` (355 rows: clip_uuid, chunk, type, country, dur_s, tracks; sha256 `62aa2bed…08e8`).

---

## Step 2: fresh raw held-out (untouched raw validation)

**Pool and rule.** The current held-out was mined from the strict validation pool: NVIDIA split `val`, not in val / val_full / exam, no chunk shared with train, val, val_full or exam, converter inputs present → 39,971 clips in 426 chunks (`NAS/heldout/pool_counts.json`; `b3_pool.py:19-32`). Its mining tables cover that whole pool (`NAS/heldout/mining/clips_scan.csv` 39,971 rows; `moveoffs_actors.csv`), and the scan is deterministic, so no re-mining was needed: the same tables, restricted to chunks outside the current held-out's 221, are the fresh pool. Selection rule = `b3_select.py:13-30` unchanged (pull-outs up to 100, 75 fast clips with a 4.5 s window held ≥ 16 m/s, the rest ordinary to 300, chunk cap 2, seeded draws), **seed 20261107**. Script `src/s2_select.py`; record `A3/raw_heldout/heldout_select_summary.json`, `logs/select.log` (under `A3/raw_heldout/logs/`).

| item | value | source |
|---|---|---|
| excluded chunks | train 1,398 + val 20 + val_full 499 + exam 31 + current held-out 221 + graded held-out 232 (⊂ val_full) = 2,157 distinct | `heldout_select_summary.json: excluded_chunks` |
| remaining pool | **19,056 clips / 205 chunks** | `remaining_pool` |
| available | **pull-out clips 0** (3 move-offs with a bumper actor, all with a heading change > 45°), fast 5,375, ordinary 13,681 | `available`; `logs/select.log`; the pre-check in the session: all 82 pull-out clips of the strict pool are in the current held-out |
| selected | **300 = 0 pull-out + 75 fast + 225 ordinary**, 177 chunks (max 2 per chunk), 24 countries (US 129, DE 28, FR 14 …), 12 still clips (max speed < 3 m/s) | `selected`, `chunks`, `countries`, `still_clips` |
| clip list | `A3/raw_heldout/heldout_clips.csv` (the name `b3_download/convert/manifest.py` expect), sha256 `e846a029…f097` | `heldout_clips.csv.sha256` |

**Download** (`NAS/heldout/src/b3_download.py` unchanged, run through `src/s2_run_b3.py`, which only re-points `b3_common.OUT` / `MANIFEST` to `A3/raw_heldout/` before the script runs; `RAW` stays the old held-out's raw folder, read-only, because it already holds every strict-pool chunk's `egomotion.offline` zip from the earlier mining (`b3_mine.py` stage A) and the converter reads ego poses from there, `RD/s1_converter/s1_convert.py:86`):

| item | value | source |
|---|---|---|
| planned (zip central directories only, no bodies) | 300 clips × (front-wide mp4 + frame timestamps) = 4,103,830,456 B + calibration 24,211,618 B = **4.128 GB ≤ 10 GB → GO** | `A3/raw_heldout/download_estimate.json`; `logs/download_plan.log` |
| fetched | **300 / 300 clips, 4.10 GB, 61.9 min** (6 threads on the 2 pinned cores; the earlier lane took 6.8 min unpinned); 600 clip files + 354 calibration files = 4,128,042,074 B, equal to the plan; 20 files re-hashed against the manifest: 0 mismatches | `logs/download.log`; `download_manifest.csv` (954 rows); session check |
| agent tracks | `obstacle.offline` members for the 300 clips with the census range-reader (`src/s2_fetch_tracks.py` = `b4b_fetch.py`'s mechanism): **300 / 300 ok, 156.7 MB, 609 s** → `A3/raw_heldout/raw/labels/obstacle.offline.members/` | `logs/fetch_tracks.log`; `download_manifest_tracks.csv` |
| not downloaded | nothing else: ego poses were already on the NAS; no other camera, lidar or radar feed | — |

**Conversion** (`NAS/heldout/src/b3_convert.py` unchanged = the S1 converter's own functions, bonnet mask 37 rows, 8 × 0.5 s human future; 2 then 8 processes on the two pinned cores; `b3_manifest.py` unchanged): **300 / 300 clips converted, 0 errors, 9,566 moments** (`manifest_summary.json: clips_converted 300, conversion_errors 0, samples 9566`); every moment passes the bonnet mask (`masked_ok 9566`); 16 moments have a frame more than 20 ms off the anchor (`frame_off_20ms 16`), 600 have fewer than 4 ego-status rows (`lt4_status_rows 600`, the first two moments of each clip, as in the lane's held-out). Pass 1 started with 2 processes (16 clips in 13 min, cores idle, iowait 20–56 %), was restarted with 8 processes at 06:39 and finished 284 clips in 302 min (`A3/raw_heldout/logs/convert.log`); pass 2 found 0 clips left. `b3_manifest.py` needs the strict-pool mining table `mining/moveoffs_actors.csv` under its output root for the pull-out tags (`b3_manifest.py:31`); it was copied from `NAS/heldout/mining/` (sha256 `3b3de806…88e3`, identical, `A3/raw_heldout/mining/COPIED_FROM.txt`) and the script re-run unchanged (`logs/manifest.log: manifest rc=0`).

| item | value | source |
|---|---|---|
| **manifest** | **300 clips / 177 chunks / 9,566 moments / 1.67 h** (durations from `clips_scan.csv: dur`, 19.1–20.0 s); `has_label2 = False` on every row (raw clips, no map, no grade) | `A3/slices/manifest_raw_heldout.parquet` (sha256 `2a916fc0…197d9f`), `A3/raw_heldout/manifest_summary.json` |
| by block | fast (type A) 75 clips / 2,385 moments; ordinary 225 clips / 7,181 moments; pull-out (B_raw) 0 | same |
| tags | slow (< 5 m/s) 2,795 moments / 158 clips; slow turn 737 / 80; fast (≥ 16) 2,084 / 89; very fast (≥ 23) 946 / 39; turn 850 / 92; pull-out clip / moment 0 / 0; commands S 8,428, R 670, L 468; speed bands [0,5) 2,795, [5,10) 2,514, [10,16) 2,173, [16,23) 1,138, [23,30) 553, ≥ 30 393 | `manifest_heldout_tags.parquet` |
| schema | the 15 train columns and dtypes (asserted by `b3_manifest.py:23`); 0 chunks or sample ids shared with training or val_full (`:26`) | `logs/manifest.log` |

**History per moment** (`src/b2_gaps_slice.py`, computation unchanged; `logs/b2_raw_heldout.log`): 0 clips with a pose gap, 0 with a frame gap; **median 8.00 s, 71.5 % ≥ 5 s, 90.3 % ≥ 2 s** (`out/b2_raw_heldout.json`); bins [0, 0.5) 24, [0.5, 1) 300, [1, 2) 600, [2, 5) 1,800, [5, 10) 3,000, [10, 20) 3,842. The clean 0/300/600/1,800/3,000 counts are the 20 s clips sampled at 0.5 s with the anchor at the moment's own time (the converter's fixed grid), so history = anchor time; the current held-out gave the same shape (`out/b2_heldout.json`).

---

## Step 3: memory-moment subsets on both new slices

Event rules **unchanged** from run 1 / B4b: `src/b4c_events.py` is `src/b4b_events.py` with two loader modes added (`graded_heldout`: the reconstructed clips' pool `clipgt` tracks through the existing `load_nurec`; `raw_heldout2`: the fresh clips' fetched `obstacle.offline` members through the existing `load_raw`); the rule text, `NONVEH`, `runs`, `lat_to_path` and `events` are untouched (`logs/b4c_events.diff`: 23 lines, all in the mode/loader block). Memory-relevant moment = a moment within (t_event, t_event + 5 s]; one row per (moment, event type), de-duplicate on (`clip_id`, `anchor_t`) for the subset (`src/s3_moments.py` = `b4b_moments.py:8-15`). Outputs: `out/b4b_{graded_heldout,raw_heldout2}_{clips,events}.csv`, `out/memory_moments_{graded_heldout,raw_heldout}.parquet`, `out/s3_moments_*.json`.

| | graded held-out (reconstructed tracks) | fresh raw held-out (fetched `obstacle.offline`) |
|---|---|---|
| clips / with tracks / clip-hours / moments | 355 / 355 / 1.97 h / 10,516 | 300 / 300 / 1.67 h / 9,566 |
| tracks (after the 1 s / 5-sample filter) | 29,514 (automobile 23,785, person 3,604, heavy truck 1,027, …) | 22,277 (automobile 17,946, person 2,730, heavy truck 586, trailer 325, rider 276, …) |
| hard braking ≥ 3 / ≥ 4 / ≥ 6 m/s²: events (per h) | 8 (4.06 /h) / 2 (1.01 /h) / 0 | 14 (8.40 /h) / 4 (2.40 /h) / 0 |
| cut-ins (per h) | 33 (16.7 /h) in 31 clips | 37 (22.2 /h) in 33 clips |
| crossing traffic (per h) | 20 (10.1 /h) in 14 clips | 28 (16.8 /h) in 24 clips |
| out-of-view-and-return | 0 (NOT MEASURABLE from track ids, recon flag 12) | 0 |
| moments ≤ 5 s after a brake ≥ 3 / ≥ 4 / ≥ 6 | 44 (0.42 %) / 4 (0.04 %) / 0 | 92 (0.96 %) / 24 (0.25 %) / 0 |
| … after a cut-in | 219 (2.08 %) | 248 (2.59 %) |
| … after a crossing | 118 (1.12 %) | 224 (2.34 %) |
| **memory moments (any event)** | **373 / 10,516 = 3.55 %** (A 111 rows, B_pass 29, ordinary 245) | **540 / 9,566 = 5.64 %** (A 149 rows, ordinary 439) |

Context: the training slice has 4.87 % memory moments (cut-in 16.95 /h, crossing 14.57 /h, brake ≥ 3 m/s² 6.37 /h; recon B4b). The graded held-out's cut-in rate matches (16.7 /h); its crossing and braking rates are lower (10.1 /h, 4.1 /h).

---

## Step 4: token-cache sample (the one GPU job)

Script `src/s4_cache_sample.py`, launcher `src/run_model.sh` (the lane's interpreter: conda `drive-jepa` python + torch 2.8.0+cu128 overlay, with `G/s1_build/src`, `s2_train/src`, `s8_018b_build/src` on the path, imported never edited); stdout `logs/s4_cache_sample.log`; record `out/s4_cache_sample.json`. GPU before: 48.5 GB of 97.9 GB used by other users' processes, 4 % busy at launch (`gpu_before`); the job peaked at 4.0 GB (`timing_step_from_cache.gpu_peak_alloc_gb`). Wall 05:28 → 06:35.

**What is cached, from the code.** The model stacks the two frames (t − 0.5 s, t) after ImageNet normalisation and calls the backbone once (`UP/drive_jepa_model.py:53-59`). The backbone (`UP/bevformer/simple_image_encoder.py`) runs V-JEPA ViT-L with `img_as_video_nframes = 2` (`:33`), so the two frames are fused into **one** token map; the projector Linear(1024 → 256) (`:38, 48`) gives 512 tokens laid out 16 × 32 (`:49`); the return value is `(feat_flatten, spatial_shape, level_start_index, kwargs)` with `feat_flatten` of shape (num_cam = 1, 512, B, 256) (`:56-66`). That tuple is what every refiner pass reads (`UP/traj_refiner.py:23-27`; `UP/bevformer/bev_refiner.py:107-108`, `img = image_feature[0]`), and `feat_flatten` is exactly what the lane's training hook captures (`G/s1_build/src/pai_model.py:152-154, 160`, `img_hook` on `_backbone`). **Cached tensor = `feat_flatten[0, :, b, :]`, (512, 256) per moment.** GridMask is off: the agent is in eval mode (the check forward asserts GridMask inactive, `G/s9_018b_train/src/train_018b.py:133`), and the job forwarded the last moment twice and asserted bit-identical plans and logits (`s4_cache_sample.py`, "pass (a) not deterministic" guard).

**Pooled copy beside each moment:** `GM/probe.py:66-72` `pool_map` (grid 16 × 32 × 256, `:51-52`): global mean + 4 × 8 block means = 33 × 256 = 8,448 floats, stored fp32 (33,792 B per moment).

**Model loading.** `DriveJEPAAgent(…, init_checkpoint = p06, freeze_encoder = True, scorer_impl = "s3")` → strict load with the `agent._pad_model` → `_pad_model` key mapping (`RD/s10_retrain/train_patch/drive_jepa_agent_s10.py:135-137`), the same way 0.1.7's file is loaded; 0.1.7's batch fix installed and on (`G/s1_build/src/pai_model.py:90-93`); every one of the 495 tensors re-compared to the file after loading: 495 / 495 equal (`agent.loaded_equal`).

**Sample.** The picking set holds ~24 of each clip's 32 moments, so whole clips were taken from val_full (the picking set's parent) in picking-set order: **32 clips, 1,022 moments, 775 of them picking-set moments, 414 graded (335 in the picking set)**, 99.9 % of consecutive anchor pairs 0.5 s apart (`out/s4_sample_moments.json`; `src/s4_sample_list.py`). Loaded through the lane's read-only slim dataset (`pai_slim.ReadOnlySlim`, zero acceleration, the check forward's inputs).

**Pass (a), the normal forward** = `pai_model.forward_dump` (batch 1, fp32, no previous plan, the scorer's own pick; `pai_model.py:300-310`): 1,022 moments; GPU forward 0.0323 s per moment (encoder + generator + scorer); wall 1,820 s = 1.78 s per moment because the slim PNGs were read from the NAS while the 4 GB download ran (flag 4).

**Cache written:** `A3/cache_sample/<clip>.pt`, 32 files, 839 MB: per moment fp32 (524,288 B), bf16 (262,144 B), pooled fp32 (33,792 B), `anchor_t`.

**Pass (b), the forward from the cache:** the backbone module replaced by one that returns the cached map (and the captured `spatial_shape`, `level_start_index`, kwargs) for the same `lidar2img`, `img_shape`, `ego_status`; the images are zeros; everything after the encoder is the unmodified model.

| | **fp32 cache** | **bf16 cache (up-cast to fp32 at read)** |
|---|---|---|
| pick agreement | **100.0 %** (0 of 1,022 changed) | **99.90 %** (1 of 1,022 changed) |
| plan difference, 32 × 8 poses, xy (m) | max **0.000**, mean 0.000 | max **1.528** (that one moment), mean **0.00042** |
| rating logit difference (last logit) | max **0.000** | max 0.0315, mean of per-moment max 0.0015 |
| pick grade on the 414 graded moments (S35 grader, `pai_model_b.Grader.grade_menus`, 484 s on 2 workers) | (a) 0.784086 → (b) **0.784086**, 0 picked plans' grades changed, menu grades identical | (b) 0.784086 (+ 0.0000003): 128 picked plans' grades move by ≤ 3.6 × 10⁻⁵; one menu plan's grade flips by 1.0 (a threshold crossing inside the grader) |

(`out/s4_cache_sample.json: exactness, grades`.)

**Reading.** The fp32 cache reproduces the full forward bit for bit, so training and checks from it are the same computation. bf16 storage perturbs the tokens by ~0.4 % relative; on 1,021 of 1,022 moments the plans move by < 1 mm and the logits by < 0.03, and the pick grade is unchanged to 6 decimals; on one near-tie moment the pick flips to a plan 1.5 m away. **Recommendation: fp32** — the brief asks for exact or near-exact picks, fp32 is exact, and 67 GB for the training cache fits the NAS (1.7 TB free, `df` at 06:35). bf16 (33 GB) is the fallback if disk becomes the constraint; it is near-exact (99.9 %), not exact.

**Timing** (`timing_encoder_s_per_moment`, `timing_step_from_cache`; 30 timed after 3 warm-up):

| measure | value |
|---|---|
| encoder forward, batch 1, fp32 (zeros input) | **0.0243 s / moment** |
| encoder forward, batch 8, fp32 | 0.0077 s / moment |
| encoder forward, batch 8, bf16 autocast | 0.0026 s / moment |
| full check forward at batch 1 fp32 (encoder + generator + scorer, real inputs) | 0.0323 s / moment |
| **training-style step from the cache, 32 moments** (generator forward in train mode under bf16 autocast + frozen-generator replay + scorer in training mode + backward into generator and scorer; tokens in GPU memory; lr 0) | **0.0569 s ± 0.012** |
| not in that step | the online grading pool (the 0.1.8b run waited ≈ 0.18 s/step for it, `NAS/018b/runs/018b_attempt2/run.json: grade_wait_s`), the slim-npz read of the 32 moments' ego rows and `lidar2img` (tiny), the memory layers (not built yet), optimiser updates of real size |

So a cached step is ≈ 0.06 s of GPU work; with the grading pool overlapped the step is grading-bound at ≈ 0.2–0.3 s, against 3.05 s/step with the encoder in the loop (recon C3). A 3,900-step pass would take ≈ 15–20 min instead of 3.3 h.

**Projection for the full cache** (`projection`; moments: training 126,981, picking 2,000, graded held-out 10,516, raw held-out 9,566):

| slice | moments | GPU-h at batch-1 fp32 encoder (0.0243 s) | GPU-h at batch-8 fp32 (0.0077 s) | disk fp32 | disk bf16 | pooled copy |
|---|---|---|---|---|---|---|
| training | 126,981 | 0.86 | 0.27 | 66.6 GB | 33.3 GB | 4.3 GB |
| picking set | 2,000 (whole clips: ~2,700) | 0.01 | 0.00 | 1.0 GB | 0.5 GB | 0.07 GB |
| graded held-out | 10,516 | 0.07 | 0.02 | 5.5 GB | 2.8 GB | 0.36 GB |
| raw held-out | 9,566 | 0.06 | 0.02 | 5.0 GB | 2.5 GB | 0.32 GB |
| **total** | **149,063** | **≈ 1.0** | **≈ 0.3** | **≈ 78 GB** | **≈ 39 GB** | **≈ 5 GB** |

**The wall time is not the GPU.** Reading one slim sample from the NAS cost 1.78 s today with the download running (62 h for the training slice at that rate), and the lane measured ≈ 25 samples/s from the NAS for its loader (`RD/s3_5_smoke/train_patch/dataset_s35.py:186`) → ≈ 1.4 h for training, ≈ 1.7 h for all four slices with 8 loader workers and no competing transfer. Plan the full cache as **≈ 2 h wall, ≈ 1 GPU-hour, 78 GB fp32**, run when nothing else is pulling from the NAS, with spawned loader workers (the job today loaded in-process on 2 pinned cores). The recon's "≈ 7 GPU-hours at 0.18 s/moment" (H05's floor on another GPU) is superseded.

**Not started:** the full cache (as the brief says).

---

## Flags

1. **The fresh raw held-out has no pull-out clips.** The mining rule (bumper actor and |heading change| ≤ 45°, `b3_select.py:17`) found 82 pull-out clips in the whole 39,971-clip strict pool, and the current held-out took all 82 (`NAS/heldout/heldout_select_summary.json: available.pullout_clips 82, selected.pullout 82`); their chunks are among the current held-out's 221 chunks, which Sid's rule excludes. The 205 remaining chunks hold 3 bumper-actor move-offs, all turning > 45°. So "82 pull-outs + 75 fast + 143 ordinary" is impossible for a fresh slice under the chunk exclusions; the rule was applied as written and gave 0 + 75 + 225. OPEN 1.
2. **One bf16 near-tie flip** (1 of 1,022 picks, plan 1.5 m away) and one menu grade flipping by 1.0: bf16 is near-exact, not exact. fp32 is exact. OPEN 2.
3. **The picking set is not whole clips** (≈ 24 of 32 moments per clip); the sample used whole clips from val_full (the parent slice) so consecutive moments exist; 775 of the 1,022 moments are picking-set moments. A memory check on the picking set will need the same completion (OPEN 4).
4. **Timing contamination:** the GPU job ran while the 4 GB download and the track fetch shared the NAS and the two pinned cores; its data-loading wall (1.78 s/moment) is not representative; the GPU-only timings (synchronised, zeros input or in-memory tokens) are. The lane's 25 samples/s figure is used for the wall projection.
5. **48 reconstructed moments in the graded held-out are ungraded** (no metric cache built); they stay in the slice for ADE-type measures only.
6. **Conversion on two pinned cores is slow:** 2 processes gave 1.2 clips/min with the cores 30–40 % idle (iowait 20–56 %), 8 processes on the same two cores 0.94 clips/min (284 clips in 302 min, `A3/raw_heldout/logs/convert.log`), so the conversion is NAS-read-bound, not CPU-bound; the whole 300 took ≈ 5.3 h; the lane converted 299 clips in 8.6 min with 8 processes (`NAS/heldout/logs/convert.log`).
7. The graded held-out comes from val_full, on which the earlier lane chose 0.1.6's pass 9 (`G/s0_recon/REPORT.md:284-287`); it was never trained on and nothing in this project has been chosen on these 355 clips.
8. **Event-rate differences between slices:** the graded held-out's crossing (10.1 /h) and braking (4.1 /h) rates are below training's (14.6 /h, 6.4 /h); its cut-in rate matches (16.7 vs 17.0 /h). The reconstructed tracks (`clipgt`, world-frame tracklines) and the raw `obstacle.offline` labels (rig-frame boxes) are different pipelines (recon B4b caveat iii).
9. **`b3_manifest.py` is not self-contained:** its pull-out tags read `mining/moveoffs_actors.csv` under the output root (`:31`), a strict-pool table that only exists in `NAS/heldout/mining/`. The first manifest run stopped there after writing the manifest itself (`:27`); the table was copied, verbatim, into `A3/raw_heldout/mining/` and the script re-run unchanged. The manifest content is unaffected (the copy is identical; the slice has 0 pull-outs either way).
10. **The fresh raw held-out is busier than the graded held-out and the training slice:** cut-ins 22.2 /h, crossings 16.8 /h, brakes ≥ 3 m/s² 8.4 /h, 5.64 % memory moments, against training's 17.0 / 14.6 / 6.4 /h and 4.87 % (recon B4b) and the graded held-out's 16.7 / 10.1 / 4.1 /h and 3.55 %. Part of that is the 75-clip fast block (25 % of the slice against 3,000 ordinary clips in B4b), part the different track pipelines (flag 8); on 1.67 h the counts are small (14 brakes, 37 cut-ins, 28 crossings), so a difference of a few events per hour is within sampling noise.

## OPEN for Sid

1. **Pull-outs in the fresh raw held-out (flag 1).** Recommend: keep the fresh 300 (75 fast + 225 ordinary, no pull-outs) as the clean raw held-out and report pull-out measures on the current held-out's 82 pull-out clips, labelled as already read for 0.1.8 (0.1.7 and 0.1.8 numbers exist there). Alternative: relax the chunk rule for the pull-out block only (allow chunks shared with val/val_full but no clip in any slice: the loose pool has 90,020 clips / 933 chunks, `pool_counts.json`), re-mine ≈ 50 k clips (≈ 1.5 GB of ego poses, ≈ 20 min), and swap 82 pull-outs in for 82 ordinary clips.
2. **Cache precision.** Recommend: fp32 (exact, 78 GB for all four slices, 1.7 TB free). Alternative: bf16 (39 GB), accepting 99.9 % pick agreement and rare near-tie flips.
3. **What to cache for the picking set.** Recommend: the whole clips of the 84 picking clips (≈ 2,700 moments, all in val_full) so memory checks replay clips in order; the 2,000 picking moments stay the ones graded. Alternative: the 2,000 moments only (no consecutive history; memory checks impossible on it).
4. **When to run the full cache.** Recommend: one job, 8 loader workers, when no other NAS transfer runs (≈ 2 h wall, ≈ 1 GPU-hour), with the pooled copy written beside it (5 GB) so the D4 history probe costs no GPU. Alternative: cache training only now (67 GB) and the three small slices at read time.
5. **Graded held-out size.** 10,516 moments / 355 clips is 5 × the recon's planned 60-clip carve-out and costs no training data. Recommend: keep all 355 clips (read once). Alternative: carve a 60-clip selection slice out of it for checkpoint choice and keep the rest held out (Sid chose the picking set for choosing, so this is only if the picking set proves unusable for a memory check).
6. **Conversion speed (flag 6).** Recommend: nothing now (done). For future conversions allow 8 processes on the shared cores at nice 19 (the lane's setting), since the 2-core pin turns 9 min into hours.

## Files

- `src/s1_graded_heldout.py` → `out/s1_graded_heldout.json`, `logs/s1_graded_heldout.log`, `A3/slices/manifest_graded_heldout.parquet`, `A3/slices/graded_heldout_clips.csv`
- `src/s2_select.py`, `src/s2_run_b3.py` (runs `NAS/heldout/src/b3_download.py`, `b3_convert.py`, `b3_manifest.py` unchanged with their output root re-pointed), `src/s2_fetch_tracks.py` → `A3/raw_heldout/{heldout_clips.csv, heldout_select_summary.json, download_estimate.json, download_manifest.csv, download_manifest_tracks.csv, tracks_fetch_status.csv, clips/, calibration/, raw/labels/obstacle.offline.members/, cache/, convert_rows/, manifest_heldout.parquet, manifest_heldout_tags.parquet, manifest_summary.json, logs/}`; `A3/slices/manifest_raw_heldout.parquet` (copy of `manifest_heldout.parquet`)
- `src/b2_gaps_slice.py` (+ `logs/b2_gaps_slice.diff`) → `out/b2_{graded_heldout,raw_heldout}.json`, `out/b2_*_clips.csv`, `logs/b2_*.log`
- `src/b4c_events.py` (+ `logs/b4c_events.diff`), `src/s3_moments.py` → `out/b4b_{graded_heldout,raw_heldout2}_{clips,events}.csv`, `out/b4b_*.json`, `out/memory_moments_{graded_heldout,raw_heldout}.parquet`, `out/s3_moments_*.json`
- `src/s4_sample_list.py` → `out/s4_sample_moments.parquet`, `.json`; `src/run_model.sh`, `src/s4_cache_sample.py` → `out/s4_cache_sample.json`, `logs/s4_cache_sample.log`, `A3/cache_sample/<clip>.pt` (32 files, 839 MB)
