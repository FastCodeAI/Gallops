# S1 — label-1 converter (raw clip → Drive-JEPA training sample), pilot on 20 raw_train clips — 2026-10-03

**VERDICT: not CONVERTER-OK by the pre-registered rule. C1 = 0 and C4 hold; C3 fails (84.4 % vs ≥ 90 %).** The C3 misses are all but one
on anchors above 30 m/s, a speed range MODE1 never measured (its fastest handover was 23.7 m/s). Nothing was tuned.

Env. S0's interpreter (conda `drive-jepa`, python 3.9) cannot import the adapter (`ModuleNotFoundError: utils_rs`) or
`alpasim_runtime`, so the converter and checks run the way the mode1 offline harness runs: `alpasim/.venv/bin/python` (3.12) with the
shipped model in the conda `drive-jepa` worker through `BridgedDriveJEPA(device="cpu")`, `CUDA_VISIBLE_DEVICES=""`. Clip picking,
downloads and the cache-loader test use the conda `drive-jepa` python. Scripts: `s1_common.py`, `s1_pick.py`, `s1_download.py`,
`s1_convert.py`, `s1_checks.py`. Outputs here: `pilot_clips.csv`, `download_stats.csv`, `samples.csv` (640 rows), `clips_converted.csv`,
`checks_tables.md`, `c2_png/`. NAS `{runs}/drivejepa_retrain/02_build_samples/` (2.1 GB: `clips/`, `calibration/`, `cache/`, `menus/`,
`rectified/`); `download_manifest.csv` 80 rows, 271 MB, all with sha256. All 20 clips pass `common.load_split_views()` as raw_train
(`s1_common.assert_train`: not exam, not val, split == train), asserted in every script. Nothing under adapters/, runs/, drafts/,
alpasim/ or navsim_v2 was modified.

## Step 1 — contract table

(a) = adapter 0.1.4 at inference (`adapters/drivejepa014/planner.py`); (b) = upstream training code at commit e21f474
(`adapters/drivejepa/src/navsim_v2/navsim/…`); (c) = converter (`s1_convert.py`).

| field | (a) adapter feeds | (b) training builders expect | (c) converter source | match a↔b |
|---|---|---|---|---|
| `camera_feature_1` | **newest** frame; RGB; `[28:-28]` crop, `cv2.resize (512,256)`, /255, CHW float32, no ImageNet norm (`planner.py:247-263`, `:638`) | `cameras[-1].cam_f0.image[28:-28]`, `cv2.resize(…,(512,256))`, `ToTensor` (`drive_jepa_features.py:48-58`); PIL RGB (`dataclasses.py:81`); norm inside model (`drive_jepa_model.py:42-56`) | front-wide frame nearest t, rectified, then the adapter's `_pipeline_a` | yes |
| `camera_feature_2` | **older** frame, nearest newest − 500 ms, ± 10 % else error (`planner.py:555-618`, `:639`) | `cameras[-2].cam_f0` (`drive_jepa_features.py:54-59`), 2 Hz history (`SensorConfig cam_f0=[2,3]`, `drive_jepa_agent.py:68-80`); model stacks `[_2, _1]` = older first (`drive_jepa_model.py:53-57`) | frame nearest t − 0.5 s, same path | yes |
| camera view | declared `opencv_pinhole` f 1545, pp **(960, 560)**, radial (−0.356, 0.173, −0.052), tangential (−0.0021, 0.0005), pose t = (1.659, −0.014, 1.515) m (`adapters/alpasim_configs/driver/drivejepa.yaml:49-63`) | NAVSIM `cam_f0` image as logged, 1920×1080 | rotation-only reprojection of the raw f-theta image into that declared camera, distortion included | by design (not verifiable from vendored code) |
| `ego_status[-1]` pose (3) | zeros: pose relative to current (`planner.py:749-787`, `_relative_pose :224`) | relative to last history pose → zeros (`dataclasses.py:198-211`) | `P._relative_pose(T(t), T(t))` | yes |
| velocity (2) | `DynamicState.linear_velocity` x, y, rig frame, m/s; zero sentinel replaced by pose difference (`planner.py:680-747`) | `ego_dynamic_state[:2]`, ego frame, m/s (`dataclasses.py:212`) | backward difference of egomotion.offline in the rig frame at t | yes (source differs) |
| acceleration (2) | `DynamicState.linear_acceleration` x, y, m/s² (`planner.py:781`) | `ego_dynamic_state[2:]` (`dataclasses.py:213`) | backward difference of that velocity; **kept** | yes (source differs) |
| command (4) | one-hot `[L, S, R, unknown]`; first route waypoint ≥ 20 m away, |y| > 3 m ⇒ turn (`planner.py:401-424`, `constants.py:105-106`) | `driving_command` read from the log (`dataclasses.py:214`); the rule that produced it is not in the vendored code | adapter rule on the human's own future path | **derivation differs (settled by the brief)** |
| `ego_status` rows | one row per buffered pose | 4 rows, 2 Hz (`drive_jepa_features.py:76-87`); only `[:, -1]` is read (`drive_jepa_model.py:51`) | up to 4 rows at 0.5 s (2 or 3 at t = 0.5 / 1.0 s) | differs, unread |
| `camera_feature` (BEV) | 2 slots: dummy + front (`planner.py:641-674`) | 4 slots b0, f0, l0, r0 (`bev_feature_build.py:27`); pixels never read by `forward` | adapter-identical, 2 slots | differs, unread |
| `lidar2img`, `img_shape` | per slot from the declared K and pose; model keeps slot 1 = front (`drive_jepa_model.py:50`) | same construction (`bev_feature_build.py:35-57`) | same adapter functions, declared K, R, t | yes for the slot read |
| label 1 | n/a (model output: 8 × (x, y, heading), rear-axle frame at t, 0.5 s) | `scene.get_future_trajectory(8)`: poses t+0.5…t+4.0 s relative to the current pose, x fwd, y left, heading CCW (`drive_jepa_features.py:143-150`, `dataclasses.py:341-372`, `drive_jepa_config.py:36,62`) | `P._relative_pose(T(t + k·0.5 s), T(t))`, k = 1…8, float64 | n/a |
| on-disk format | n/a | `<cache>/<log_name>/<token>/pad_feature.gz` + `pad_target.gz`, gzip pickles of dicts of tensors (`training/dataset.py:17-28`, `:99-129`; names `drive_jepa_features.py:36`, `:141`) | exactly that; `log_name` = clip uuid, token = `<uuid8>_<t ms>` | loads with upstream `CacheOnlyDataset` (640 samples, torch 2.1) |

**Gate reading.** (a) and (b) agree on every field `forward` reads. They differ on three things that are either unread or already
decided: the number of `ego_status` rows, the number of BEV slots, and how the command is derived. I treated that as "no contract
mismatch" and continued. If you read the gate strictly, this is where S1 should have stopped; the samples then need no rework, only your call.

Two corrections to the brief's settled inputs, taken from disk: the declared principal point is **(960, 560)**, not (960, 540), and
the declared camera **has** lens distortion. AlpaSim renders at the **declared** pose (the yaml's `extra_cameras` entry, a nuPlan-like
mount), not at the real camera's mount. The real mounts in the pilot sit 0.14–0.48 m from it (median 0.37 m), 1.0–2.7° rotated.
The converter reproduces the declared rotation and intrinsics exactly but cannot move the viewpoint (flag F2).

Also needed by upstream training but outside label 1: `compute_loss` scores proposals against a per-token PDM metric cache
(`drive_jepa_agent.py:45-47`, `:103-131`). Raw clips have none (flag F6).

## Step 2 — pilot and camera budget

Seed 20261004 (`pilot_clips.csv`). Eligible: A 141 nurec_train clips holding ≥ 20 m/s for 4.5 s; B 285 raw pull-outs (S0 Run 3);
ordinary 358 nurec_train clips with no A or B event and an anchor at 5–15 m/s. Clips in the one train chunk that contains an exam
clip were excluded. Per clip, only `<clip>.camera_front_wide_120fov.mp4` and `.timestamps.parquet` were range-read from the camera
chunk zip (1.2–1.5 GB each, never downloaded whole), plus the chunk's `camera_intrinsics.offline` and `sensor_extrinsics.offline` parquets.

| measured | value |
|---|---|
| mp4 per clip | mean 13.4 MB (min 3.0, max 25.1); timestamps 9 KB; calibration 70–190 KB per chunk |
| time per clip | 8.8 s serial (7 requests) |

| budget | size | serial | 6 workers |
|---|---|---|---|
| 285 pull-outs | 3.8 GB | 0.7 h | 0.12 h |
| 208 A scenes | 2.8 GB | 0.5 h | 0.08 h |
| 5,000 ordinary | 67 GB | 12.2 h | 2.0 h |

Conversion cost: about 23 s per clip (decode + rectify + 32 samples + 32 CPU forwards at 0.55 s); cache 0.8 MB per sample.

## Step 3 — converter

Camera model: `ftheta` (`camera_intrinsics.offline.model_parameters`): `principal_point`, `angle_to_pixeldist_poly` (pixel radius as a
polynomial of the ray's angle from the optical axis), `linear_cde` (affine term), `max_angle` (valid field, 1.348 rad);
extrinsics `sensor_extrinsics.offline` (`x, y, z, qx…qw` = camera pose in the rig). For each declared-camera pixel the converter
undistorts with the declared opencv coefficients, rotates the ray into the real camera, and samples the f-theta image
(same math as `Re-ranker/s2b_parity/parity.py:74-96`). No pixel fell outside the real camera's field in any clip.

### Sample statistics

| type | clips | anchors | speed_med | speed_max |
|---|---|---|---|---|
| A | 5 | 160 | 27.9 | 40.6 |
| B | 5 | 160 | 1.61 | 14.2 |
| O | 10 | 320 | 6.9 | 14.3 |

Frame timing: |frame − target| max 22.6 ms (older 22.6, newer 16.7); anchors > 20 ms: 1; pair gap − 500 ms: max |33.33| ms; egomotion lag (backward difference) max 100 ms.
ego_status rows per sample: {2: 20, 3: 20, 4: 600} (4 = upstream count; fewer at t = 0.5 / 1.0 s).
Command (human future path as route): {'S': 517, 'R': 72, 'L': 51}; by type: A {'S': 160}; B {'R': 63, 'S': 61, 'L': 36}; O {'S': 296, 'L': 15, 'R': 9}.
AlpaSim RouteGeneratorRecorded comparison: available on 576/640 anchors (fails on 2 clips); same command on 572/576.
Acceleration slot (backward difference of 10 Hz velocity): all anchors ax p1/p50/p99 -3.44/-0.09/2.80 m/s², |ax| > 4 on 8 anchors; ay p1/p99 -1.90/1.39.

## Step 4 — checks

### C1 — bit identity

| token | frame_newer | frame_older | max_abs_diff_cf1 | max_abs_diff_cf2 | shape | dtype |
|---|---|---|---|---|---|---|
| f1fd39ee_09500 | 286 | 271 | 0 | 0 | (3, 256, 512) | torch.float32 |
| 12ed6f9c_12000 | 362 | 347 | 0 | 0 | (3, 256, 512) | torch.float32 |
| 47fead0a_11500 | 347 | 332 | 0 | 0 | (3, 256, 512) | torch.float32 |
| 76bc2969_09500 | 286 | 271 | 0 | 0 | (3, 256, 512) | torch.float32 |
| db469eed_13000 | 390 | 375 | 0 | 0 | (3, 256, 512) | torch.float32 |

C1 max abs difference over 5 samples (stored sample vs fresh decode → rectify → adapter `_pipeline_a`): **0.0**.

### C2 — geometry PNGs

| type | uuid | t_s | speed | command | label_x8 | label_y8 | pts_outside_1080p | pts_outside_input | png |
|---|---|---|---|---|---|---|---|---|---|
| A | 49bcab6f | 8 | 19.4 | S | 71.5 | 10.6 | 0 | 0 | c2_png/A_49bcab6f_t080.png |
| A | 12ed6f9c | 8 | 23.8 | S | 89.1 | -4.9 | 0 | 0 | c2_png/A_12ed6f9c_t080.png |
| A | 8457182d | 8 | 33.4 | S | 133 | 1 | 0 | 0 | c2_png/A_8457182d_t080.png |
| A | aa5f29c2 | 8 | 36.6 | S | 146 | 3.9 | 0 | 0 | c2_png/A_aa5f29c2_t080.png |
| A | 7cba46df | 8 | 28.7 | S | 114 | -0.2 | 0 | 0 | c2_png/A_7cba46df_t080.png |
| B | f1fd39ee | 13 | 2.1 | L | 16.1 | 2.7 | 3 | 3 | c2_png/B_f1fd39ee_t130.png |
| B | 653ce0cd | 9 | 2.6 | S | 36.9 | -4.5 | 2 | 2 | c2_png/B_653ce0cd_t090.png |
| B | 3c1fa662 | 10.5 | 2 | R | 13 | -2.9 | 3 | 3 | c2_png/B_3c1fa662_t105.png |
| B | db469eed | 5 | 1.9 | L | 17.3 | 4.2 | 3 | 3 | c2_png/B_db469eed_t050.png |
| B | 47fead0a | 9.5 | 2.4 | R | 16.8 | -5.9 | 3 | 3 | c2_png/B_47fead0a_t095.png |
| O | 88272099 | 8 | 11.5 | S | 26.3 | -0.2 | 1 | 1 | c2_png/O_88272099_t080.png |
| O | d5f732f6 | 8 | 4.9 | S | 16.8 | -0.2 | 2 | 2 | c2_png/O_d5f732f6_t080.png |
| O | 76bc2969 | 8 | 9.7 | S | 38.8 | 0.2 | 1 | 1 | c2_png/O_76bc2969_t080.png |
| O | 35d707e1 | 8 | 7.1 | S | 33.4 | 0.5 | 1 | 1 | c2_png/O_35d707e1_t080.png |
| O | 37d17fd8 | 8 | 9.9 | S | 42.1 | 0.3 | 1 | 1 | c2_png/O_37d17fd8_t080.png |
| O | 14850e11 | 8 | 6.6 | S | 27.7 | -0.1 | 1 | 1 | c2_png/O_14850e11_t080.png |
| O | 464f923c | 8 | 6.4 | S | 23.3 | -0.6 | 2 | 2 | c2_png/O_464f923c_t080.png |
| O | 2f3ed8fe | 8 | 5.2 | S | 16.4 | -1.3 | 2 | 2 | c2_png/O_2f3ed8fe_t080.png |
| O | 4e4a2a51 | 8 | 5.7 | S | 31 | 0.6 | 1 | 2 | c2_png/O_4e4a2a51_t080.png |
| O | 8a812020 | 8 | 10.6 | S | 33.9 | -0.1 | 1 | 1 | c2_png/O_8a812020_t080.png |

All 640 anchors: 1526/5120 projected label points fall outside the 1920×1080 rectified image, 1573/5120 outside the cropped model input (projection with the real camera position, ground at rig z = 0).

### C3 — menu ceiling (A clips, human speed ≥ 20 m/s)

| uuid | anchors | human_speed_med | fastest_med | fastest_max | chosen_step1_med | chosen_end_med | n_over_18_8 |
|---|---|---|---|---|---|---|---|
| 12ed6f9c | 25 | 24.3 | 15.7 | 18.5 | 15.2 | 11.3 | 0 |
| 49bcab6f | 20 | 21 | 13.1 | 14.3 | 14.2 | 9.65 | 0 |
| 7cba46df | 32 | 27.9 | 17.6 | 19 | 15.9 | 12 | 1 |
| 8457182d | 32 | 33.4 | 18.1 | 23.7 | 12.3 | 14.3 | 11 |
| aa5f29c2 | 32 | 37.2 | 17.1 | 25.8 | 12.7 | 11.7 | 10 |

C3: fastest-of-32 end speed ≤ 18.8 m/s on 119/141 anchors = **84.4 %** (prediction ≥ 90 %). Pool: fastest median 17.34, max 25.76 m/s; human median 28.5 m/s.

### C4 — city speed match (ordinary clips, human speed 5–15 m/s)

| uuid | anchors | human_speed_med | ratio_med | ratio_min | ratio_max |
|---|---|---|---|---|---|
| 14850e11 | 23 | 6.38 | 1.02 | 0.89 | 1.32 |
| 2f3ed8fe | 17 | 5.51 | 0.94 | 0.686 | 1.07 |
| 35d707e1 | 32 | 7.25 | 0.994 | 0.95 | 1.15 |
| 37d17fd8 | 32 | 9.49 | 0.938 | 0.847 | 1.25 |
| 464f923c | 26 | 7.12 | 0.915 | 0.63 | 1.04 |
| 4e4a2a51 | 18 | 7.55 | 0.901 | 0.0146 | 1.09 |
| 76bc2969 | 32 | 9.71 | 0.925 | 0.418 | 0.982 |
| 88272099 | 21 | 13.7 | 0.828 | 0.698 | 0.917 |
| 8a812020 | 32 | 10.6 | 0.931 | 0.806 | 1.17 |
| d5f732f6 | 15 | 8.66 | 0.926 | 0.837 | 1.01 |

C4: median(chosen-plan first-step speed / human speed) = **0.936** over 248 anchors in 10 clips (prediction in [0.8, 1.2]); IQR 0.89–0.99; share below 0.8: 10 %.

### C5 — pull-outs at standstill (descriptive)

| uuid | t | t_move | t_depart | command | n_lat1 | n_lat1_moving | max_prop_abs_y | max_prop_len | chosen_len | chosen_max_abs_y | label_len | label_y8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1fd39ee | 0.5 | 11.6 | 12.9 | R | 2 | 2 | 1.63 | 10.7 | 2.92 | 0.03 | 0.05 | -0 |
| f1fd39ee | 1 | 11.6 | 12.9 | R | 2 | 2 | 1.7 | 11.3 | 3.38 | 0.08 | 0.05 | 0 |
| f1fd39ee | 1.5 | 11.6 | 12.9 | R | 2 | 2 | 1.61 | 11.1 | 3.23 | 0.04 | 0.06 | -0 |
| f1fd39ee | 2 | 11.6 | 12.9 | R | 0 | 0 | 0.79 | 12 | 6.19 | 0.04 | 0.05 | -0.01 |
| f1fd39ee | 2.5 | 11.6 | 12.9 | R | 0 | 0 | 0.97 | 10.8 | 2.25 | 0.03 | 0.05 | 0 |
| f1fd39ee | 3 | 11.6 | 12.9 | R | 2 | 2 | 1.54 | 11.9 | 3.8 | 0.03 | 0.05 | 0.01 |
| f1fd39ee | 3.5 | 11.6 | 12.9 | R | 4 | 4 | 2.35 | 12 | 3.49 | 0.07 | 0.04 | -0.01 |
| f1fd39ee | 4 | 11.6 | 12.9 | R | 0 | 0 | 0.99 | 12.4 | 3.84 | 0.04 | 0.05 | -0 |
| f1fd39ee | 4.5 | 11.6 | 12.9 | R | 0 | 0 | 0.22 | 12.6 | 6.46 | 0.03 | 0.05 | 0.01 |
| f1fd39ee | 5 | 11.6 | 12.9 | R | 1 | 1 | 1.16 | 11.2 | 1.28 | 0.02 | 0.05 | 0 |
| f1fd39ee | 5.5 | 11.6 | 12.9 | R | 0 | 0 | 0.72 | 11.6 | 3 | 0.04 | 0.04 | 0 |
| f1fd39ee | 6 | 11.6 | 12.9 | R | 0 | 0 | 0.99 | 11.7 | 3.12 | 0.03 | 0.04 | 0 |
| f1fd39ee | 6.5 | 11.6 | 12.9 | R | 5 | 5 | 2.41 | 12.9 | 4.68 | 0.15 | 0.04 | -0 |
| f1fd39ee | 7 | 11.6 | 12.9 | R | 2 | 2 | 2.19 | 11.7 | 3.14 | 0.03 | 0.05 | -0.01 |
| f1fd39ee | 7.5 | 11.6 | 12.9 | R | 2 | 2 | 2.23 | 12.4 | 3.81 | 0.03 | 0.07 | 0 |
| f1fd39ee | 8 | 11.6 | 12.9 | R | 2 | 2 | 1.68 | 11.3 | 2.67 | 0.03 | 0.29 | -0 |
| f1fd39ee | 8.5 | 11.6 | 12.9 | R | 2 | 2 | 1.87 | 11 | 1.61 | 0.03 | 0.82 | -0.07 |
| f1fd39ee | 9 | 11.6 | 12.9 | R | 1 | 1 | 1.22 | 10.8 | 0.63 | 0.03 | 1.73 | -0.3 |
| f1fd39ee | 9.5 | 11.6 | 12.9 | R | 4 | 4 | 1.28 | 12 | 2.6 | 0.02 | 2.99 | -0.79 |
| f1fd39ee | 10 | 11.6 | 12.9 | R | 2 | 2 | 1.5 | 10.8 | 4.4 | 0.04 | 4.55 | -1.5 |
| f1fd39ee | 10.5 | 11.6 | 12.9 | R | 0 | 0 | 0.77 | 10.6 | 1.05 | 0.05 | 6.35 | -2.25 |
| f1fd39ee | 11 | 11.6 | 12.9 | R | 0 | 0 | 0.94 | 10.5 | 1.59 | 0.07 | 8.35 | -2.86 |
| f1fd39ee | 11.5 | 11.6 | 12.9 | R | 9 | 9 | 2.31 | 9.53 | 0.66 | 0.03 | 10.5 | -3.16 |
| 653ce0cd | 6 | 7.8 | 8.8 | S | 10 | 10 | 2.04 | 12.8 | 0.19 | 0.02 | 5.93 | 0.82 |
| 653ce0cd | 6.5 | 7.8 | 8.8 | S | 8 | 8 | 2.7 | 14.6 | 7.49 | 1 | 9.35 | 1.02 |
| 653ce0cd | 7 | 7.8 | 8.8 | S | 10 | 10 | 3.02 | 14.4 | 8.69 | 1.09 | 13.7 | 1.17 |
| 653ce0cd | 7.5 | 7.8 | 8.8 | S | 10 | 10 | 3.81 | 14.2 | 6.53 | 1.4 | 18.9 | 1.58 |
| 3c1fa662 | 0.5 | 9.2 | 10.5 | R | 15 | 15 | 6.51 | 11 | 0.42 | 0.04 | 1.4 | -0 |
| 3c1fa662 | 1 | 9.2 | 10.5 | R | 11 | 11 | 5.23 | 10.1 | 0.15 | 0.03 | 1.61 | 0.01 |
| 3c1fa662 | 1.5 | 9.2 | 10.5 | R | 4 | 4 | 2.92 | 8.49 | 0.19 | 0.03 | 1.6 | 0.01 |
| 3c1fa662 | 2 | 9.2 | 10.5 | R | 16 | 16 | 6.14 | 10.9 | 0.2 | 0.03 | 1.59 | 0.01 |
| 3c1fa662 | 2.5 | 9.2 | 10.5 | R | 6 | 6 | 4.8 | 8.85 | 0.23 | 0.03 | 1.56 | 0 |
| 3c1fa662 | 5 | 9.2 | 10.5 | R | 23 | 23 | 6.85 | 12.2 | 0.11 | 0.03 | 0.05 | -0 |
| 3c1fa662 | 5.5 | 9.2 | 10.5 | R | 11 | 11 | 6.99 | 11.2 | 0.17 | 0.03 | 0.19 | -0.01 |
| 3c1fa662 | 6 | 9.2 | 10.5 | R | 16 | 16 | 7.9 | 12.5 | 0.7 | 0.08 | 0.81 | -0.1 |
| 3c1fa662 | 6.5 | 9.2 | 10.5 | R | 16 | 16 | 6.83 | 12.7 | 0.93 | 0.07 | 1.74 | -0.43 |
| 3c1fa662 | 7 | 9.2 | 10.5 | R | 25 | 25 | 8.5 | 13.7 | 2.51 | 0.27 | 2.79 | -0.95 |
| 3c1fa662 | 7.5 | 9.2 | 10.5 | R | 20 | 20 | 7.58 | 13 | 0.47 | 0.05 | 4.16 | -1.73 |
| 3c1fa662 | 8 | 9.2 | 10.5 | R | 23 | 23 | 10.1 | 16.2 | 0.12 | 0.03 | 5.79 | -2.71 |
| 3c1fa662 | 8.5 | 9.2 | 10.5 | R | 19 | 19 | 9.49 | 13.9 | 0.28 | 0.07 | 7.53 | -3.72 |
| 3c1fa662 | 9 | 9.2 | 10.5 | R | 20 | 20 | 10.5 | 15.9 | 0.1 | 0.03 | 9.33 | -4.68 |
| db469eed | 0.5 | 2.7 | 5 | L | 8 | 8 | 5.02 | 10.3 | 0.14 | 0.03 | 1.4 | 0.04 |
| db469eed | 1 | 2.7 | 5 | L | 9 | 9 | 5.13 | 10.7 | 0.13 | 0.03 | 2.3 | 0.14 |
| db469eed | 1.5 | 2.7 | 5 | L | 14 | 14 | 5.52 | 11.6 | 3.54 | 0.45 | 3.51 | 0.36 |
| db469eed | 2 | 2.7 | 5 | L | 17 | 17 | 5.8 | 12.1 | 4.31 | 0.63 | 5.06 | 0.79 |
| db469eed | 2.5 | 2.7 | 5 | L | 21 | 21 | 5.67 | 13.2 | 8.98 | 1.77 | 6.94 | 1.47 |
| 47fead0a | 0.5 | 6.9 | 9.1 | S | 5 | 5 | 5.1 | 10.1 | 0.19 | 0.03 | 0.29 | 0 |
| 47fead0a | 1 | 6.9 | 9.1 | S | 6 | 6 | 4.86 | 9.3 | 0.17 | 0.04 | 0.22 | -0 |
| 47fead0a | 1.5 | 6.9 | 9.1 | S | 6 | 6 | 4.47 | 11.1 | 0.35 | 0.05 | 0.11 | -0 |
| 47fead0a | 2 | 6.9 | 9.1 | L | 8 | 8 | 5.64 | 10.4 | 0.16 | 0.05 | 0.04 | -0 |
| 47fead0a | 2.5 | 6.9 | 9.1 | L | 9 | 9 | 5.16 | 10.1 | 0.34 | 0.04 | 0.05 | 0 |
| 47fead0a | 3 | 6.9 | 9.1 | L | 6 | 6 | 4.21 | 10.3 | 0.17 | 0.03 | 0.15 | 0.01 |
| 47fead0a | 3.5 | 6.9 | 9.1 | L | 6 | 6 | 4.34 | 9.97 | 0.22 | 0.03 | 0.4 | 0.02 |
| 47fead0a | 4 | 6.9 | 9.1 | L | 7 | 7 | 4.54 | 10.1 | 0.25 | 0.03 | 0.88 | 0.08 |
| 47fead0a | 4.5 | 6.9 | 9.1 | L | 8 | 8 | 4.11 | 10.5 | 0.25 | 0.03 | 1.6 | 0.27 |
| 47fead0a | 5 | 6.9 | 9.1 | L | 5 | 5 | 4.55 | 9.91 | 0.17 | 0.03 | 2.51 | 0.6 |
| 47fead0a | 5.5 | 6.9 | 9.1 | L | 8 | 8 | 4.23 | 11.2 | 0.5 | 0.03 | 3.61 | 1.08 |
| 47fead0a | 6 | 6.9 | 9.1 | L | 9 | 9 | 4.3 | 11.3 | 0.69 | 0.03 | 5.02 | 1.69 |
| 47fead0a | 6.5 | 6.9 | 9.1 | L | 6 | 6 | 4.85 | 8.84 | 0.15 | 0.03 | 6.75 | 2.3 |
| 47fead0a | 7 | 6.9 | 9.1 | S | 8 | 8 | 3.95 | 10.9 | 0.44 | 0.02 | 8.59 | 2.62 |

C5: standstill anchors before departure 60 in 5 clips; with ≥ 1 proposal reaching ≥ 1 m sideways: 52; with ≥ 1 such proposal that also travels ≥ 2 m: 52; chosen plan moves ≥ 2 m on 24; human label moves ≥ 2 m within 4 s on 23.

**VERDICT: NOT CONVERTER-OK — C3 failed (C1 = 0, C4 = 0.936 hold).** Most likely cause from Step 1: none of the contract rows; the failing anchors are at 30–40 m/s, outside the range MODE1 measured (F1).

## Flags (not fixed)

- **F1 — C3 fails on speed range, not obviously on the converter.** Fastest-of-32 above 18.8 m/s: 0 of 30 anchors at 20–24 m/s, 1 of 47 at
  24–30 m/s, 21 of 64 above 30 m/s (max 25.8 m/s). MODE1's ceiling came from handovers up to 23.7 m/s. Within that range the
  prediction holds (76 of 77). The ceiling still exists at 33–37 m/s; it sits near 18–26 m/s, far below the human speed.
- **F2 — viewpoint not matched.** Rotation-only rectification keeps the real camera position, 0.14–0.48 m from the declared one. The C2
  PNGs draw the label both ways (green real position, magenta declared).
- **F3 — the ego bonnet and a privacy-blur patch are visible** at the bottom of the rectified frames (see the PNGs). Whether AlpaSim renders
  contain a bonnet was not checked; the calibration ships an `ego_mask_image_png` that was not applied.
- **F4 — command stand-in.** Using the human's future path as the route gives S 517, R 72, L 51. It agrees with AlpaSim's own
  `RouteGeneratorRecorded` on 572 of 576 anchors; that generator refuses 2 pull-out clips ("route folds back"), which is why it is not the primary source.
- **F5 — acceleration slot is noisy.** It is a double backward difference of 10 Hz positions: |ax| > 4 m/s² on 8 anchors at steady cruise. Kept as instructed.
- **F6 — label 1 alone does not make a trainable sample upstream.** The loss needs a PDM metric cache per token.
- **F7 — unread-field differences**: 40 samples have 2 or 3 `ego_status` rows (anchors at 0.5 and 1.0 s); BEV has 2 slots. Batching these with
  NAVSIM cache samples needs a decision. No JPEG round trip is applied (S2c applied one at q95; inference frames are JPEG-decoded).
- **F8 — frame timing.** One anchor is 22.6 ms from its target frame (> 20 ms), and the pair gap is off by up to 33 ms (one frame). All others ≤ 16.7 ms.
- **F9 — 30 % of projected label points are outside the image.** These are the near poses (first 0.5–1.5 s at low speed), which lie below the
  camera's field. At highway speed none are outside.
- **F10 — ground height assumed** at rig z = 0 for the C2 overlay; the real rig origin height differs by vehicle (native camera z 1.25–1.58 m).
- **F11 — C5**: at 52 of 60 standstill anchors at least one proposal goes ≥ 1 m sideways and moves ≥ 2 m, but the chosen plan stays within
  0.1 m laterally on most of them and often does not move at all (24 of 60 move ≥ 2 m). The menu contains a pull-out; the scorer rarely picks it.
