# 01_find_clips — NuRec 26.04 census: listing, exam carve-out, range-read pilot (2026-10-01)

Branch `alpasim-20hz-results` (`git branch --show-current`). Work folder `analysis_3.0/Retrain-data/01_find_clips/` (scripts + report only,
1.2 MB). All downloaded bytes on the NAS: `{runs}/drivejepa_retrain/01_find_clips/` (NFS {site-nfs}, 13 MB, 27 files,
every copy in `download_manifest.csv` with sha256). Interpreter: `conda-envs/drive-jepa/bin/python` (huggingface_hub 1.8.0);
HF account {hf-account}, token `~/.cache/huggingface/token`, gate accepted (dataset_info `gated: auto`, whoami OK). No GPU, no mp4/usdz
downloads, no exam scene opened.

**Status: STOPPED at the STEP B gate.** AlpaSim ships no eval/val/test-labelled scene list. Steps A, B, C are complete; the D fetch/detect
scripts are written and validated on the 3 Step-C pilot scenes only. Decision needed from Sid (see §B GATE) before `d_fetch.py` runs.

## STEP A — listing and sidecars

| item | value | evidence |
|---|---|---|
| HF repo / revision | `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` @ `be464484…` (2026-07-30) | `a1_listing.py` (dataset_info) |
| `sample_set/` contents | only `26.04_release/` (no 26.01 folder on `main`; 26.01 lives on HF revision `26.01`, per AlpaSim `data/scenes/README.md`) | tree listing |
| layout | **flat**: 1,607 UUID folders directly under `26.04_release/` (no Batch dirs) | `a1_tree_26.04.json`: depth hist {2: 1607, 3: 3214} |
| per-folder files | exactly 2 everywhere: `<uuid>.usdz` + `camera_front_wide_120fov.mp4` | 3,214 files = 1,607 × 2 |
| `labels.json` | **0 of 1,607** — not shipped in 26.04 (README row 25.05: "only available for files under Batch0002 and onwards"; 26.04 has no batches) | `a2_sidecars.py` output |
| usdz size | min 1,450 / median 1,718 / max 2,386 MB; total 2.80 TB | tree sizes |
| `clip_ratings_26.04.csv` | repo root, 1,607 rows, columns `clip_id, quality_score, time_of_day, ego_speed`; covers all 1,607 UUIDs, no extras | NAS `root_files/clip_ratings_26.04.csv` |
| `ego_speed` | medium 548 · slow 538 · high 488 · **extremely slow 33** (README documents only High/medium/slow) | same |
| `time_of_day` | night 871 · day 684 · evening 46 · dusk 4 · dawn 2 | same |
| `quality_score` | mean 69.4, sd 10.0, min 9.8, max 90.9 | same |

labels.json category values: **not available** (no file exists in this release). Stratification can only use `ego_speed` / `time_of_day`.

## STEP B — exam exclusion

### B1 — AlpaSim's official scene lists (read-only search of `paddock-sid/Alpasim/alpasim/`)

All scene lists live in `alpasim/data/scenes/` (`sim_suites.csv`, `sim_scenes.csv`, legacy `*_2505.csv`). Counts re-verified from disk
(`awk` over `sim_suites.csv`; `test_scene_csvs.py:62-88` pins the same numbers).

| path:line | list name | n scenes | release | how it is labelled | consumer |
|---|---|---|---|---|---|
| `data/scenes/sim_suites.csv:2-914` | `public_2601` | 913 | HF rev 26.01, NRE 26.1.112 | README.md:30 "Use `public_2601` for new runs"; :35 "Recommended public NRE suite"; `src/wizard/configs/exp/presets/alpamayo1_5_comparison_4cam.yaml:4` "# public_2601 evaluation: …" | `scenes.test_suite_id` (sceneset.py:295-316), TUTORIAL.md:422-429, CHANGELOG.md:180-183 |
| `data/scenes/sim_suites.csv:915-1643` | `public_2601_video_model` | 729 (⊂ public_2601) | 26.01 | README.md:36 "UUID-pinned subset … video-model path" | VIDEO_MODEL.md:139-149 |
| `data/scenes/sim_suites.csv:1644-3249` | `public_2604` | 1,606 | HF rev 26.04, NRE 26.4.96 | README.md:37 "1,606 of the 1,607 artifacts in the 26.04 release … only 159 scenes overlap with public_2601"; CHANGELOG.md:56-57 "after removing known-invalid artifacts" | VIDEO_MODEL.md; test_scene_csvs.py |
| `data/scenes/sim_scenes.csv:2-2524` | catalog (no suite) | 916 (26.01) + 1,607 (26.04) | both | README.md:7 "Public scene artifact metadata for all current releases" | base_config.yaml:116-129 |
| `data/scenes/sim_suites_2505.csv:2-911` | `public_2507` | 910 | HF rev 25.05 | README.md:10 "Legacy … not loaded by default" | tests only |
| `src/wizard/configs/base_config.yaml:117-118` | `scenes.scene_ids` | 1 (clipgt-01d503d4) | 26.01 | "default OSS scene from huggingface" | wizard default |
| `src/wizard/tests/test_sceneset.py:57-62` | `dev.alpasim.unit_tests.v0` | 3 | internal SwiftStack 25.02.19 | unit-test fixture | pytest only |

Not found anywhere in the checkout: a list labelled eval / val / validation / test-set / held-out / benchmark / leaderboard for NuRec
scenes; any code that lists the HF dataset (only per-UUID `hf_hub_download`, `sceneset.py:459-465`); any reference to a split file or
to `clip_ratings`. The dataset itself ships no split file either (Step A: root = README, LICENSE, `.gitattributes`, `clip_ratings_26.04.csv`).
Searched: `data/`, `docs/`, `src/`, `plugins/`, `tools/`, `.agents/`, `.github/`, `.cursor/`, root md/yaml (not `.git`, `.venv`, `outputs/`).

Terminology caveat: AlpaSim calls every suite a *test suite* (`test_suite_id` column; `data/scenes/README.md:1` "# Test Suites"). Under a
literal "union of every test list" reading, `public_2601 ∪ public_2604` = all 1,606 valid 26.04 scenes → **training pool = 0**.

### B2 — our 30 diagnosed scenes (`b2_our30.csv`)

Sources: 10 pilot10 run dirs `runs/drivejepa/drivejepa_014_<scene>/wizard-config.yaml`, 20 ext20 run dirs via
`analysis_2.0/drivejepa_014_ext20/run_*` → `{bulk_disk}/…/drivejepa_014_ext20_<scene>/wizard-config.yaml`, cross-checked against the
30 `scene` ids in `analysis_2.0/Drive JEPA analysis/mode1_high_speed_stop/census.csv` (match asserted in `b_exam.py`).

| membership | count / 30 |
|---|---|
| in `public_2601` | **30** (all were drawn from it) |
| in `public_2601_video_model` | 24 |
| in `public_2604` / present as a 26.04 folder | **8** (01d503d4, 02e075b9, 048b974e, 5f064b02, 8023dce5, acd22430, b44a4a44, d1a25a99) |
| missing from 26.04 (still excluded from every other pool) | 22 |

The 26.04 folder name equals the clip uuid in `clipgt-<uuid>` for all 1,607 catalog rows (asserted); the 26.04 *artifact* uuid differs
(e.g. 01d503d4 → artifact `23dd34ea…`, the "26.04 duplicate" seen in earlier reports) — exclusions are keyed on the clip uuid.
`public_2601` ids present as 26.04 folders: 159 (matches README.md:37); video-model subset: 123. Known-invalid 26.04 artifact
(in tree, not in `public_2604`): `e0fcd3bb-0d6f-496c-8b79-7d59d17a31c2` (`known_invalid_2604.txt`), excluded from every pool.

### B3 — exclusion candidates (both files written; `common.load_exclusions()` asserts ≥ 30 ids, every D script asserts pool ∩ exam = ∅)

| candidate | definition | ids | of which 26.04 folders | training pool (26.04 − exam − invalid) |
|---|---|---|---|---|
| **A** (`exam_exclusions.txt`, used by C/D scripts) | `public_2601` (913) ∪ our 30 | 913 | 159 | **1,447** |
| strict (`exam_exclusions_strict_our30.txt`) | our 30 only (no eval-labelled list exists) | 30 | 8 | 1,598 |
| literal "every test suite" | `public_2601 ∪ public_2604` ∪ our 30 | 2,360 | 1,606 | **0** |

**GATE (STOP):** no list labelled eval/val/test exists, so per the brief I did not pick one. Candidate A is written as the *working* file
because it is the only reading that (i) uses AlpaSim's own recommended suite, which is also the suite every Paddock benchmark number was
computed on (all 30 diagnosed scenes ∈ `public_2601`), and (ii) leaves a non-empty pool; it is header-marked "pending Sid's
confirmation". Step D was **not** run on the pool. Options: (a) confirm candidate A; (b) revert to the earlier E2 random-draw design
(the 1,447-scene pool is still untouched except the 3 pilot scenes below, which must then be excluded from any E2 draw); (c) something else.

## STEP C — range-read pilot (3 scenes, seed 20261001, drawn from the candidate-A pool; `c_pilot.py`, `hfrange.py`)

Pilot UUIDs (`c_pilot_uuids.txt`, now *touched*, must stay out of any future exam): `4bdcda96-d062-…`, `6eb099c6-e0d0-…`, `7f34da5c-9ef6-…`.

**C1** — `HEAD` on `…/resolve/main/<path>.usdz` redirects to `us.aws.cdn.hf.co`, `Accept-Ranges: bytes`, `Content-Length` = usdz size.
`Range: bytes=…` returns 206. The usdz is a **STORED zip (compress_type 0 for all 39 members)**; the central directory was read with 3
requests / 2,815 B. Member table for 4bdcda96 (1,758.6 MB): `default.usda` 1 KB, `mesh_ground.usd` 6.7 MB, `volume.usda`, `pose_record.json`
83 KB, `volume.nurec` 580 MB, `sequence_tracks.json` 1.6 MB, `mesh_ground.ply` 12 MB, `rig_trajectories.json` 2.2 MB, `checkpoint.ckpt`
1,147 MB, `map.xodr` 682 KB, `sequence_tracks.usda`, `parsed_config.yaml`, `datasource_summary.json` 4.2 MB, `rig_trajectories.usda`,
`metadata.yaml` 598 B, `data_info.json` 2 KB, 19 × `clipgt/*.parquet` (traffic_sign, road_boundary, road_island, calibration_estimate,
gore_area, buffer_zone, intersection_area, egomotion_estimate 17 KB, clip, lane_line, wait_line, traffic_light, road_marking, mads_trace,
pole, **lane** 151 KB, **obstacle** 1.06 MB, crosswalk, association), 4 × `frames/<camera>/<ts>.jpeg` (one per camera, ~250 KB). Full
offsets/sizes in `c_pilot_results.json`. **No `clipgt/drivable_space.parquet` in 26.04** (nearest: `road_boundary`, `intersection_area`).

**C2** — bytes fetched per scene (all 9 requested files minus the missing drivable_space):

| scene | usdz MB | egomotion | obstacle | lane | seq_tracks | rig_traj | map.xodr | meta+data_info | total fetched | % of usdz | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4bdcda96 | 1,758.6 | 17.3 KB | 1,062 KB | 151 KB | 1,622 KB | 2,157 KB | 682 KB | 2.6 KB | 5.70 MB (18 req) | 0.32 % | 15.0 |
| 6eb099c6 | 1,895 | 17.2 KB | 226 KB | 39 KB | 355 KB | 2,169 KB | 238 KB | 2.6 KB | 3.06 MB (18 req) | 0.16 % | 11.9 |
| 7f34da5c | 1,782 | 17.2 KB | 562 KB | 35 KB | 871 KB | 2,133 KB | 108 KB | 2.6 KB | 3.74 MB (18 req) | 0.21 % | 11.5 |

**C3** — schemas (all frames share `egomotion_label_class_id = gtaas:egomotion_deepmap:v0`: a metric world frame whose origin is the
first ego pose, x forward at t0; timestamps are absolute `timestamp_micros`, clip start = `data_info.json pose-range.start-timestamp_us`
= `metadata.yaml time_range.start`):
- `clipgt/egomotion_estimate.parquet` — 202 rows, **10 Hz** (dt 99.99 ms, span 20.0 s), `location{x,y,z}` m + `orientation{x,y,z,w}`
  quaternion. → **used for ego speed/pose** (finite difference; matches the 202-pose `rig_trajectories.json T_rig_worlds`, which is the
  same trajectory in 4×4 form and 130× larger).
- `clipgt/obstacle.parquet` — one row per (track, timestamp): `trackline_id, center{x,y,z}, size{x,y,z}, orientation, category`
  (automobile / person / other_vehicle / protruding_object / heavy_truck / bus …), per-track ~10 Hz with per-object lidar timestamps
  (10,186 rows, 179 tracks in 4bdcda96). → **used for actor boxes**: per-track finite-difference speed gives static vs moving; categories
  are needed for the mode-2/3 obstacle-type breakdown. `sequence_tracks.json` (178 tracks, poses + cuboid dims, what the sim replays)
  is the same track set without categories; kept only in the pilot for cross-checking.
- `clipgt/lane.parquet` — 470 lanes: `left_rail` / `right_rail` polylines, `lane_direction`, `speed_limit` (0 = unknown), styles.
  **Sparse: 2–31 points per rail, median 4–5** → centreline = mean of resampled rails; see flag F3.
- `map.xodr` — OpenDRIVE 1.4 (DeepMap), tmerc georeference; not needed by the detectors.

**C4** — throughput 12.8 s/scene serial, 4.2 MB/scene for the full C2 set; the D set (egomotion+obstacle+lane+meta) is ≈ 0.1–1.3 MB and
≈ 6–7 s/scene. Projection for the 1,447-scene candidate-A pool: **≤ 6.1 GB / 5.2 h serial** (full C2 set) or ≈ 1.5 GB / 2.7 h serial
(D set); with 4 workers ≈ 0.7 h. Gate C: **PASS** (range reads work, projection ≪ 48 h). The mp4s (1,607 × ~12 MB) were never touched.

## STEP D — NOT RUN on the pool (B gate). Scripts ready: `d_fetch.py --workers 4` then `d_detect.py --stage pool --out d_pool`.

Detector validation on the 3 pilot scenes (`d_pilot_results.json`):

| scene | rating | v_max m/s | tracks | lanes | mode 1 anchors | mode 2 events | mode 3 anchors (fallback) |
|---|---|---|---|---|---|---|---|
| 4bdcda96 | slow | 13.3 | 179 | 470 | 0 | 0 | 0 (1) — 22 anchors had an actor in range; all failed the 0.5 m lane test (dev 0.5–2.9 m) |
| 6eb099c6 | high | **32.9** | 73 | 153 | **32/32** (≥ 17 m/s for the whole clip) | 0 | 0 |
| 7f34da5c | medium | 12.3 | 126 | 84 | 0 | 0 | 1 (slow-moving actor, max dev 0.41 m) |

## Predictions

| id | statement | result |
|---|---|---|
| P1 | range reads work, fetched < 5 % of usdz | **HELD** (0.16–0.32 %) |
| P2 | mode 1 ≥ 20 pool scenes, some > 23 m/s | not evaluated (D not run); 1 of 3 pilot scenes is a 32.9 m/s highway clip |
| P3 | mode 2 ≥ 30 events in ≥ 20 scenes | not evaluated |
| P4 | mode 3 ≥ 100 scenes | not evaluated; F3 suggests the fixed 0.5 m lane criterion will under-count |
| P5 | ego_speed=High holds ≥ 80 % of mode-1 scenes | not evaluated |

Per-mode verdict: **PENDING** (D blocked by the B gate). Projected download budget if D runs on candidate A: ≈ 1.5 GB (D set) / ≤ 6.1 GB
(full C2 set), ≈ 0.7 h with 4 workers, all to `{runs}/drivejepa_retrain/01_find_clips/pool/`.

Exclusion sizes: candidate A 913 ids (159 in 26.04) → pool 1,447; strict 30 ids (8 in 26.04) → pool 1,598; literal every-suite → pool 0.

## Flags (not fixed)

- **F1** No eval/val/test-labelled list in AlpaSim; both public suites are "test suites"; taking the literal union leaves no training data.
- **F2** `labels.json` does not exist in 26.04 (contradicts the brief's assumption); stratification limited to `clip_ratings` columns.
  `ego_speed` has a 4th value "extremely slow" (33) undocumented in the README.
- **F3** `lane.parquet` rails are sparse (median 4–5 points, curves chorded); the fixed 0.5 m centreline criterion for mode 3 failed on all 22
  candidate anchors of a 179-track urban clip. Expect mode 3 to be under-counted or to fall back to the heading test; the threshold was not changed.
- **F4** `clipgt/drivable_space.parquet` is absent from the 26.04 usdz (19 clipgt tables; nearest `road_boundary`).
- **F5** Mode-2 gap is measured from the rig origin (egomotion frame), not the front bumper; the ego overhang is unknown from these files.
- **F6** 22 of our 30 diagnosed scenes are not in 26.04 at all; the 8 that are carry a different *artifact* uuid in 26.04 (in-place
  re-reconstruction, NRE 26.4.96 vs 26.1.112) — the 26.01 usdz cached on {site-host} is not the 26.04 artifact.
- **F7** The 3 pilot scenes are now touched (metadata + clipgt tables read) and must be excluded from any future random exam draw.


---

# Run 2 — revised types (A high speed, B move-off, ordinary) — 2026-10-02/03

Work folder moved (not by this run) to `paddock-sid/Alpasim/analysis_3.0/Retrain-data/01_find_clips/`; all scripts resolve paths from their own
location. Scripts: `b3_finalise.py`, `d_fetch.py`, `d2_detect.py`, `d2_aggregate.py`, `dc_raw_match.py`. Outputs: `d2_scenes.csv` (per scene),
`d2_anchors.csv` (46,252 anchors with tags), `d2_moveoffs.csv`, `dc_raw_match.csv`, `d2_tables.md`. Interpreter: conda `drive-jepa` python
(the Paddock `.venv` has no pandas/pyarrow/requests/huggingface_hub — flag R2-F1).

## Step B — final

Decision (Sid, 2026-10-02): **exam = AlpaSim `public_2601` (913 scene ids, `alpasim/data/scenes/sim_suites.csv:2-914`) ∪ our 30**. Our 30 are
all inside `public_2601`, so the union is 913. The "pending" header is removed; `exam_exclusions.txt` now carries the date and decision.
Caveat: `public_2604` is a catalogue of the 26.04 release (`data/scenes/README.md:30-37`), not an eval split, and is not excluded.

| view | file | ids | reconciliation |
|---|---|---|---|
| raw_exclusions | `raw_exclusions.txt` | 913 | 913 ∪ 30 = 913; for any raw-dataset step |
| nurec_2604_exclusions | `nurec_2604_exclusions.txt` | 159 | raw ∩ 26.04 folders |
| training pool | `training_pool.txt` | 1,447 | 1,607 − 159 − 1 known-invalid (`e0fcd3bb`) |

Both views and the pool size are asserted in `common.load_views()` (159 / 913 / 1,447, nurec ⊆ raw, pool ∩ raw = ∅), called by every Run-2
script; `d2_detect.py` additionally asserts that no exam uuid folder exists under the NAS pool directory.

## Fetch

1,447 / 1,447 scenes, 0 failures, 34.0 min with 6 workers. 1.10 GB fetched = **0.044 %** of the 2.53 TB of pool usdz. Per scene:
`clipgt/egomotion_estimate.parquet`, `clipgt/obstacle.parquet`, `clipgt/clip.parquet`, `metadata.yaml`, `data_info.json`. Destination
`{runs}/drivejepa_retrain/01_find_clips/pool/<uuid>/`; 7,262 rows with sha256 in `download_manifest.csv`. No mp4, no full usdz, no exam file.

## Step D

Method notes (implementation choices, thresholds unchanged): ego pose resampled to a 10 Hz grid, speed = finite difference of position;
acceleration = derivative of a 0.5 s moving-average speed; actor speed = displacement over a 1 s baseline (cuboid centres jitter); "moving"
= ≥ 0.5 m/s at the anchor; anchor bin = minimum speed in its 4.5 s window; gaps measured from the rig origin to the near face of the box.

Scenes analysed: 1447 of 1447 pool scenes (load errors 0); anchors 46252; clip duration min/median/max 19.1/20.0/20.0 s; scenes with < 32 anchors: 44.

### D-0 — speed envelope

| window-min speed m/s | anchors | scenes |
|---|---|---|
| 0–2 | 10783 | 655 |
| 2–4 | 4141 | 719 |
| 4–6 | 4839 | 762 |
| 6–8 | 4790 | 742 |
| 8–10 | 3960 | 625 |
| 10–12 | 3680 | 483 |
| 12–14 | 2544 | 347 |
| 14–16 | 1729 | 248 |
| 16–18 | 1492 | 182 |
| 18–20 | 1110 | 136 |
| 20–22 | 1369 | 121 |
| 22–24 | 1048 | 101 |
| 24–26 | 1124 | 98 |
| 26–28 | 896 | 82 |
| 28–30 | 969 | 75 |
| 30–32 | 644 | 55 |
| 32–34 | 408 | 32 |
| 34–36 | 460 | 25 |
| 36–38 | 248 | 15 |
| 38–40 | 18 | 2 |

Top bin reached: 38–40 m/s. Samples > 50 m/s: scenes with vmax > 50 = 0; anchors with window-min ≥ 50 = 0.

Per scene: max speed median 12.8, max 40.8 m/s; max speed held 4.5 s median/p90/p99 11.1/26.3/36.0, max 39.0 m/s.

| uuid | max held 4.5 s m/s | max m/s | clip_ratings ego_speed |
|---|---|---|---|
| 3a107de6-a55c-4f0a-81aa-e9af966f81d2 | 39.05 | 40.8 | high |
| 4cb05d65-b2b9-48fc-8f14-d0507d6cda9e | 38.53 | 38.65 | high |
| 170a3280-64dc-488a-bd73-0a5f6fc5617a | 37.86 | 38.29 | high |
| aa5f29c2-e81e-4d45-88d8-88db3d6ca701 | 37.63 | 40.63 | high |
| 43b053e4-ae10-4b88-8338-6f108cfa805d | 37.59 | 38.11 | high |
| 32be03e4-a072-4da5-b597-114cffa52af8 | 37.12 | 38.31 | high |
| 82f67f40-acba-4083-991b-f63355300952 | 36.7 | 37.52 | high |
| 06b3e399-a820-49ef-9e69-134cfdb7652a | 36.69 | 36.89 | high |
| d0d8a2b6-ac68-4e21-ae93-b30284b3fe2f | 36.51 | 36.68 | high |
| 1f0db45a-1da6-4d5a-aa58-7905dfc2051d | 36.51 | 37.76 | high |

### D-A — high speed (whole 4.5 s window ≥ 16 m/s, no cap)

Distinct scenes 404; anchors 9786; anchors per qualifying scene median 32 (min 1, max 32); scenes qualifying on every anchor 253; per-scene window min/max speed range 16.0–40.8 m/s (per-scene values in d2_scenes.csv).

| bin m/s | scenes | anchors | brake a | follow a | curve a | cruise a | brake sc | follow sc | curve sc | cruise sc |
|---|---|---|---|---|---|---|---|---|---|---|
| 16–18 | 182 | 1492 | 225 | 336 | 197 | 859 | 71 | 44 | 38 | 107 |
| 18–20 | 136 | 1110 | 152 | 180 | 150 | 724 | 47 | 29 | 30 | 90 |
| 20–22 | 121 | 1369 | 147 | 383 | 231 | 752 | 40 | 33 | 23 | 84 |
| 22–24 | 101 | 1048 | 111 | 365 | 178 | 512 | 29 | 33 | 22 | 61 |
| 24–26 | 98 | 1124 | 79 | 399 | 178 | 551 | 19 | 34 | 21 | 68 |
| 26–28 | 82 | 896 | 55 | 190 | 61 | 632 | 16 | 21 | 9 | 62 |
| 28–30 | 75 | 969 | 71 | 113 | 16 | 800 | 13 | 15 | 2 | 63 |
| 30–32 | 55 | 644 | 34 | 111 | 13 | 486 | 9 | 12 | 2 | 45 |
| 32–34 | 32 | 408 | 22 | 54 | 0 | 332 | 6 | 5 | 0 | 28 |
| 34–36 | 25 | 460 | 44 | 17 | 0 | 399 | 4 | 3 | 0 | 25 |
| 36–38 | 15 | 248 | 6 | 2 | 0 | 240 | 1 | 1 | 0 | 15 |
| 38–40 | 2 | 18 | 3 | 0 | 0 | 15 | 1 | 0 | 0 | 2 |

Tags, pool total (anchors / scenes): brake 949 / 126; follow 2150 / 120; curve 1024 / 86; cruise 6302 / 322. Multi-tag anchors 605.

Bins with ≥ 1 scene: 12. Highest bin with ≥ 5 distinct scenes: 36–38 m/s.

| rating | no D-A anchor | has D-A anchor | share with D-A |
|---|---|---|---|
| extremely slow | 30 | 0 | 0 |
| high | 70 | 368 | 0.84 |
| medium | 470 | 36 | 0.07 |
| slow | 473 | 0 | 0 |

Share of D-A scenes rated 'high': 0.91.

### D-B — move-off from standstill

| set | events | distinct scenes |
|---|---|---|
| all move-offs | 258 | 254 |
| lateral ≥ 1.0 m (4 s from move onset), no actor condition | 59 | 59 |
| stationary actor 0–8 m ahead, ±2 m | 6 | 6 |
| **lateral subset** (lateral ≥ 1.0 m AND actor) | 3 | 3 |
| alt: lateral ≥ 1.0 m measured 4 s from t_depart, no actor condition | 107 | 105 |
| alt: lateral from t_depart AND actor | 4 | 4 |

Standstill observed from clip start (stop onset censored): 181 of 258. Stop-to-depart delay median 1.3 s. Lateral (from move onset) median 0.16 m, p90 2.13 m. 4 s window truncated by clip end: 8. Actor classes: {'automobile': 5, 'person': 1}. Rating of move-off scenes: {'slow': 198, 'medium': 49, 'extremely slow': 6, 'high': 1}.

Lateral-subset events:

| uuid | t_stop_start | t_move | t_depart | lat4_signed | actor_cat | actor_gap | actor_lat |
|---|---|---|---|---|---|---|---|
| fab24f4f-5d90-4460-bd85-7850faf12739 | 6 | 8 | 10.2 | 2.47 | automobile | 7.63 | -0.21 |
| 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 0 | 5.7 | 8.8 | 1.64 | automobile | 6.07 | -0.5 |
| febd36c4-abb1-498c-9e79-7f4d90d0285f | 5.7 | 8.6 | 10 | 1.1 | person | 5.53 | -1.24 |

### D-D — overlap

| set | scenes |
|---|---|
| A (D-A anchors) | 404 |
| B (any move-off) | 254 |
| B lateral subset | 3 |
| A ∩ B | 6 |
| neither A nor B (ordinary candidates) | 795 |

### D-E — spot-check list (front mp4 at `sample_set/26.04_release/<uuid>/camera_front_wide_120fov.mp4`, t = seconds from clip start)

| D-A bin | uuid | t s | window-min m/s | tags |
|---|---|---|---|---|
| 16–18 | 96cc7239-9216-4579-aa86-40f9f36280ea | 5.5 | 16.6 | cruise |
| 16–18 | 28f20041-1346-4c7c-b76f-02b000c81324 | 13 | 16.2 | follow |
| 16–18 | 1182b9c0-56b4-4c28-b7c3-5c9de0256699 | 3 | 17.1 | follow |
| 18–20 | 132ae6cb-1e58-42aa-8bf3-4381eb958e43 | 16 | 19.9 | cruise |
| 18–20 | 1a89339b-48d2-42c9-9b6b-277dd098112a | 5.5 | 19.1 | brake |
| 18–20 | 1a394766-ea99-4cf5-98de-d1821222612b | 3.5 | 18.8 | brake |
| 20–22 | 87a3b6d4-fd0f-487f-b409-f8ef2fe53681 | 9 | 21.9 | cruise |
| 20–22 | a9162b2d-b0f6-4c56-9642-c96a11e69fdb | 11 | 21.4 | curve |
| 20–22 | 743997a7-f90e-44cc-873a-2b4f38496ea7 | 8 | 20.2 | follow |
| 22–24 | 1a89339b-48d2-42c9-9b6b-277dd098112a | 1 | 22.7 | brake+curve |
| 22–24 | dd9fd9d1-f3cd-4bae-8093-b1273eddba08 | 13 | 22.1 | follow |
| 22–24 | 5a23ec1c-1c6f-4da6-ae62-c31677896960 | 15 | 23.3 | cruise |
| 24–26 | 7ee3f88d-0da8-4e7f-a0a2-e2798392cb34 | 11 | 25.1 | cruise |
| 24–26 | 504ee5fd-7588-4ec5-b8f3-2bde1b06c05a | 12.5 | 24.7 | follow |
| 24–26 | 63ac00e5-2b62-4b85-acd5-b1594313ffc9 | 1 | 25.7 | follow |
| 26–28 | 6fc87bb8-794b-4884-a1f8-50193c9b6357 | 6.5 | 26.4 | cruise |
| 26–28 | 3dbcdf90-5dc0-4797-94ff-e6bdf47adc14 | 16 | 26.2 | brake |
| 26–28 | bd6387a0-ca7d-4c02-a684-90c141acbc19 | 9.5 | 27.2 | cruise |
| 28–30 | 4d4352f8-8565-42a8-99ac-8b2c790d2591 | 8 | 28.8 | cruise |
| 28–30 | 3d8253d9-9382-40e7-964c-5cafa6f24efc | 2 | 29 | cruise |
| 28–30 | ca97c826-dfb7-4fec-8087-b9fb69987d7b | 0.5 | 29.7 | cruise |
| 30–32 | c131bc24-ae60-4d16-a60c-2f4501763269 | 13 | 30.9 | cruise |
| 30–32 | 4d496419-40e7-4822-ae8a-1af44a22bd83 | 0.5 | 30 | cruise |
| 30–32 | d8ba63a8-1ac8-453f-9f33-a8511914621a | 9 | 30.6 | cruise |
| 32–34 | 4fdce89c-8dc9-4c3a-95e5-1a6c4da202e2 | 4 | 32.9 | cruise |
| 32–34 | 92e0d11d-184c-4614-881c-2325ba7bae5a | 10.5 | 32.6 | cruise |
| 32–34 | 7e27eb97-a5db-45f1-bac2-3a62089c1183 | 7.5 | 32.7 | follow |
| 34–36 | 3d31d122-c1fe-46b9-bcc7-9e8c6c93bd7f | 2 | 34.6 | cruise |
| 34–36 | d2a7ae02-b6ed-4ffc-b245-be1b4b676c8f | 8.5 | 34.9 | brake |
| 34–36 | b45734f4-01d2-48f7-9296-2ae79a7a89b9 | 3 | 34.3 | cruise |
| 36–38 | 1f0db45a-1da6-4d5a-aa58-7905dfc2051d | 14 | 36.2 | cruise |
| 36–38 | 06b3e399-a820-49ef-9e69-134cfdb7652a | 8 | 36.5 | cruise |
| 36–38 | 47650c67-906f-4a73-a012-be29fad601ff | 15 | 36.3 | cruise |
| 38–40 | 3a107de6-a55c-4f0a-81aa-e9af966f81d2 | 7 | 38.6 | brake |
| 38–40 | 4cb05d65-b2b9-48fc-8f14-d0507d6cda9e | 4 | 38.1 | cruise |

D-B (lateral subset first, then filled with the largest-lateral move-offs without a qualifying actor):

| uuid | t_stop_start | t_move | t_depart | lat4_signed | actor | actor_cat | actor_gap |
|---|---|---|---|---|---|---|---|
| fab24f4f-5d90-4460-bd85-7850faf12739 | 6 | 8 | 10.2 | 2.47 | True | automobile | 7.63 |
| 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 0 | 5.7 | 8.8 | 1.64 | True | automobile | 6.07 |
| febd36c4-abb1-498c-9e79-7f4d90d0285f | 5.7 | 8.6 | 10 | 1.1 | True | person | 5.53 |
| d54d0ed1-a494-42c6-bfb0-ffde27c7ef57 | 0 | 8.2 | 9.3 | 12.32 | False |  |  |
| fd01e9ab-05d6-445a-b974-02a33793ab49 | 0 | 4.8 | 5.7 | 10.44 | False |  |  |

### Pre-registered

| id | statement | result | verdict |
|---|---|---|---|
| P-A | ≥ 30 distinct scenes with D-A anchors, spanning ≥ 4 bins | 404 scenes, 12 bins | ENOUGH |
| P-A2 | highest 2 m/s bin with ≥ 5 scenes (descriptive) | 36–38 m/s | — |
| P-B | ≥ 30 move-off events | 258 events in 254 scenes | ENOUGH |
| P-BL | ≥ 15 lateral-subset events | 3 events in 3 scenes | THIN |
| P-C | ≥ 90 % of pool UUIDs have a raw clip with front_wide camera | 1,447 / 1,447 = 100 % | ENOUGH |

### D-C — raw-photo availability (metadata only)

Match key: NuRec folder uuid (= the uuid in scene id `clipgt-<uuid>`, verified equal for all 1,607 catalogue rows in Run 1) against the
`clip_id` index of `nvidia/PhysicalAI-Autonomous-Vehicles` `clip_index.parquet`, and column `camera_front_wide_120fov` of
`metadata/feature_presence.parquet`. Files read from the NAS snapshot `33f9bf44…`, which equals the current remote sha (checked over the network).

| check | result |
|---|---|
| pool uuids in `clip_index.parquet` | 1,447 / 1,447, all `clip_is_valid` |
| `camera_front_wide_120fov` present | 1,447 / 1,447 |
| ID mismatches / near-misses | 0 |
| raw-dataset split of pool clips | train 741 · val 404 · test 302 |
| platform / chunks / countries | hyperion_8.1 ×1,447 · 946 distinct chunks · US 774, Germany 407, 9 other countries 266 |

## Flags (Run 2, not fixed)

- **R2-F1** Paddock `.venv` lacks the data libraries; the conda `drive-jepa` python was used instead.
- **R2-F2** P-BL is THIN because of the actor condition, not the lateral one: 59 move-offs have lateral ≥ 1.0 m, only 3 also have a
  stationary actor 0–8 m ahead within ±2 m. Sensitivity on those 59 (descriptive only): stationary actor within 0–15 m and ±3.5 m → 8;
  any actor (moving or not) in that wider box → 24. The gap is measured from the rig origin, so a car directly ahead of the bumper sits at
  roughly 4 m or more; the "throughout" rule also drops queue leads that drive off.
- **R2-F3** 181 of 258 move-offs start stationary at clip start, so the stop onset is censored (the ≥ 2 s standstill is still observed).
- **R2-F4** The lateral measure depends on the window origin: 59 events from move onset, 107 from `t_depart`. The largest values (10–12 m)
  are turns from standstill, not pull-outs around an obstacle.
- **R2-F5** `follow` has no heading filter; all leads in the 60-scene sample were automobiles at 17–34 m/s, but oncoming actors on the ego
  path are not excluded by the definition.
- **R2-F6** 706 of the 1,447 pool clips are `val`/`test` in the raw dataset's own split (the raw dataset does ship a split; NuRec does not).
  Whether that matters for retraining is Sid's call.
- **R2-F7** 44 clips are shorter than 20 s (min 19.1 s) and yield 30–31 anchors instead of 32.
- **R2-F8** `clip_ratings ego_speed = high` holds 91 % of D-A scenes; 36 D-A scenes are rated `medium`, and 70 `high` clips have no D-A anchor.


---

# Run 3 — split rule, two detector fixes, B for each head — 2026-10-03

Env: same as Run 2, conda `drive-jepa` python (`{runs}/alpasim_bulk/conda-envs/drive-jepa/bin/python`); the Paddock
`.venv` was sourced but lacks pandas/pyarrow/requests. Scripts: `r3_splits.py`, `r3_rigbbox.py`, `r3_core.py` (shared detectors),
`d3_nurec.py`, `d3_aggregate.py`, `r3_rawio.py`, `r3_raw_probe.py`, `r3_raw_stage1.py`, `r3_raw_scan.py`, `r3_raw_actors.py`,
`r3_raw_aggregate.py`. Outputs: `d3_{scenes,anchors,moveoffs,around}.csv`, `r3_raw_{clips,moveoffs_ego,moveoffs}.csv`.
No GPU, no camera/mp4, no full usdz, no exam member read. All bytes under `{runs}/drivejepa_retrain/01_find_clips/`
(`pool/` NuRec, `raw/` raw dataset, 5.5 GB); `download_manifest.csv` now has 16,586 rows, every one with sha256.

## Split views

Rule (Sid): train = raw split `train`; val = offline validation; `test` excluded from everything; exam (913) excluded on top. Split source:
`clip_index.parquet` column `split` of `nvidia/PhysicalAI-Autonomous-Vehicles` (snapshot `33f9bf44…` = current remote sha). Every chunk
is single-split (0 of 3,146 chunks mix splits).

| view | file | clips | chunks | note |
|---|---|---|---|---|
| nurec_train | `nurec_train.txt` | 741 | 487 | pool 1,447 = train 741 + val 404 + test 302 (dropped) |
| nurec_val | `nurec_val.txt` | 404 | 267 | |
| raw_train | `raw_train.txt` | 153,624 | 1,576 | raw train 153,625 − 1 exam clip |
| raw_val | `raw_val.txt` | 90,927 | 934 | raw val 90,928 − 1 exam clip |

Of the 913 exam UUIDs, 198 exist in the raw dataset: 196 test, 1 train, 1 val. `common.load_split_views()` asserts for all four views: no
test clip, no exam clip, inside the right split, nurec ⊆ raw; every Run-3 script calls it and re-asserts on its outputs.

## Part 1 — NuRec recount with fixes (Run 2 cache; new bytes: 4 KB per scene for the ego box)

Consistency: Run 3 code reproduces Run 2 on the same 1145 scenes — D-A flag equal on 36605/36605 anchors; Run 2 follow == follow_old on 7668/7668 D-A anchors.

Ego box from usdz `rig_trajectories.json` → `rig_trajectories[0].rig_bbox` (`centroid`, `dim`, rig frame): present in 1145/1145 scenes, `rot` = 0 and centroid y = 0 in all. Front-bumper offset F = centroid_x + dim_x/2: min 3.656, median 4.071, max 4.093 m (4 distinct values); width 1.999–2.157 m.

### 1a + 1c — D-A bins (window ≥ 16 m/s), scenes / anchors; follow before → after the ±30° heading filter (anchors)

| bin m/s | train scenes | train anchors | train follow | train brake | train curve | train cruise | val scenes | val anchors | val follow | val brake | val curve | val cruise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16–18 | 93 | 716 | 202 → 202 | 134 | 103 | 372 → 372 | 56 | 459 | 66 → 66 | 63 | 64 | 282 → 282 |
| 18–20 | 77 | 603 | 136 → 136 | 114 | 79 | 351 → 351 | 41 | 365 | 33 → 33 | 33 | 53 | 265 → 265 |
| 20–22 | 68 | 787 | 208 → 208 | 90 | 140 | 439 → 439 | 33 | 291 | 74 → 74 | 42 | 31 | 166 → 166 |
| 22–24 | 60 | 588 | 235 → 233 | 54 | 102 | 253 → 255 | 19 | 172 | 18 → 18 | 33 | 27 | 110 → 110 |
| 24–26 | 50 | 526 | 120 → 120 | 33 | 117 | 280 → 280 | 23 | 304 | 169 → 169 | 29 | 27 | 115 → 115 |
| 26–28 | 45 | 475 | 82 → 82 | 31 | 32 | 338 → 338 | 22 | 241 | 66 → 66 | 13 | 8 | 163 → 163 |
| 28–30 | 36 | 373 | 14 → 14 | 18 | 0 | 342 → 342 | 23 | 313 | 92 → 92 | 39 | 7 | 204 → 204 |
| 30–32 | 34 | 434 | 70 → 70 | 27 | 0 | 337 → 337 | 9 | 115 | 20 → 20 | 4 | 8 | 83 → 83 |
| 32–34 | 21 | 231 | 47 → 47 | 14 | 0 | 170 → 170 | 6 | 85 | 0 → 0 | 8 | 0 | 77 → 77 |
| 34–36 | 13 | 229 | 0 → 0 | 26 | 0 | 203 → 203 | 6 | 129 | 5 → 5 | 0 | 0 | 124 → 124 |
| 36–38 | 8 | 122 | 0 → 0 | 6 | 0 | 116 → 116 | 4 | 92 | 0 → 0 | 0 | 0 | 92 → 92 |
| 38–40 | 1 | 4 | 0 → 0 | 3 | 0 | 1 → 1 | 1 | 14 | 0 → 0 | 0 | 0 | 14 → 14 |
| **all** | 208 | 5088 | 1114 → 1112 | 550 | 573 | 3202 → 3204 | 110 | 2580 | 543 → 543 | 264 | 225 | 1695 → 1695 |

nurec_train: 208 scenes, 12 bins, highest bin with ≥ 5 scenes 36–38 m/s. Follow scenes 66 → 65; anchors dropped by the heading filter 2; motion-direction variant gives 1112 anchors.
nurec_val: 110 scenes, 12 bins, highest bin with ≥ 5 scenes 34–36 m/s. Follow scenes 27 → 27; anchors dropped by the heading filter 0; motion-direction variant gives 543 anchors.

### 1a + 1b — D-B move-offs; actor gap from rig origin (before) → front bumper (after)

| set | train events | train scenes | val events | val scenes |
|---|---|---|---|---|
| all move-offs | 144 | 142 | 73 | 71 |
| lateral ≥ 1.0 m | 32 | 32 | 18 | 18 |
| stationary actor 0–8 m, ±2 m — rig origin (Run 2) | 2 | 2 | 4 | 4 |
| stationary actor 0–8 m, ±2 m — front bumper | 3 | 3 | 4 | 3 |
| lateral subset — rig origin (Run 2) | 2 | 2 | 1 | 1 |
| **lateral subset — front bumper** | 2 | 2 | 2 | 2 |
| lateral subset, bumper, heading change ≤ 45° | 2 | 2 | 2 | 2 |

Lateral-subset events (bumper gap):

| split | uuid | t_stop_start | t_move | t_depart | lat4_signed | dyaw4_deg | acat_bum | gap_bum | alat_bum |
|---|---|---|---|---|---|---|---|---|---|
| val | 7fb3b33e-85d4-4d3e-b999-5eee301c2308 | 3.8 | 6.6 | 8.3 | -2.76 | -11.44 | automobile | 3.95 | -0.12 |
| train | 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 0 | 5.7 | 8.8 | 1.64 | 18.71 | automobile | 2 | -0.5 |
| val | fab24f4f-5d90-4460-bd85-7850faf12739 | 6 | 8 | 10.2 | 2.47 | 21.43 | automobile | 3.56 | -0.21 |
| train | febd36c4-abb1-498c-9e79-7f4d90d0285f | 5.7 | 8.6 | 10 | 1.1 | -8.51 | person | 1.88 | -1.24 |

## Part 2 — NuRec

### Part 2 — around a still obstacle (one event per actor per scene, earliest qualifying anchor)

| nurec_train: ego speed m/s \ clearance m | ≤0.5 | 0.5–1.0 | 1.0–2.0 | >2.0 | all |
|---|---|---|---|---|---|
| 0–2 | 0 | 2 | 3 | 1 | 6 |
| 2–8 | 13 | 41 | 62 | 69 | 185 |
| 8–16 | 7 | 7 | 18 | 19 | 51 |
| ≥16 | 0 | 0 | 0 | 0 | 0 |
| all | 20 | 50 | 83 | 89 | 242 |

nurec_train: 242 events in 93 scenes; clearance ≤ 1.0 m: 70 events in 33 scenes; classes {'automobile': 206, 'person': 18, 'protruding_object': 5, 'trailer': 5, 'other_vehicle': 4, 'heavy_truck': 3, 'bus': 1}; |path lateral at pass| median 4.05 m, > 1.0 m in 203; |heading change at pass| > 20° in 90; candidates dropped for box overlap 0.

| nurec_val: ego speed m/s \ clearance m | ≤0.5 | 0.5–1.0 | 1.0–2.0 | >2.0 | all |
|---|---|---|---|---|---|
| 0–2 | 2 | 4 | 2 | 1 | 9 |
| 2–8 | 10 | 13 | 31 | 49 | 103 |
| 8–16 | 1 | 8 | 9 | 4 | 22 |
| ≥16 | 0 | 0 | 0 | 0 | 0 |
| all | 13 | 25 | 42 | 54 | 134 |

nurec_val: 134 events in 57 scenes; clearance ≤ 1.0 m: 38 events in 22 scenes; classes {'automobile': 116, 'person': 14, 'heavy_truck': 1, 'trailer': 1, 'protruding_object': 1, 'bus': 1}; |path lateral at pass| median 3.78 m, > 1.0 m in 118; |heading change at pass| > 20° in 39; candidates dropped for box overlap 0.

Descriptive (nurec_train, heading change at pass ≤ 20°, i.e. not a turn/curve artefact): 152 events, 64 with clearance ≤ 1.0 m; of those, human path offset ≥ 0.5 m at the pass: 45.

Spot-check — 5 smallest clearances (train first):

| split | uuid | t | t_pass | ego_speed | clearance | cat | actor_lat | path_lat_at_pass |
|---|---|---|---|---|---|---|---|---|
| train | 8a812020-cb7b-4189-8664-021a23b4d190 | 9.5 | 13.5 | 9.77 | 0.19 | trailer | -2.2 | 0.34 |
| train | 14850e11-6a08-4cd7-9875-be353f1a764b | 8.5 | 11.7 | 6.78 | 0.2 | automobile | -2.47 | 0.04 |
| train | b706e6a9-0c43-4faa-940e-91d80529804f | 1 | 4.4 | 2.66 | 0.27 | automobile | 2.4 | 0.02 |
| train | f335b031-bbaf-42a1-bd5d-465081ec5f4b | 10.5 | 13.9 | 7.46 | 0.28 | automobile | 2.3 | -0.06 |
| train | 14850e11-6a08-4cd7-9875-be353f1a764b | 3.5 | 6.4 | 5.17 | 0.33 | automobile | 2.35 | -0.12 |
| val | 9c8e6289-e412-4457-849f-0890a6e1f64f | 9 | 12.7 | 1.35 | 0.19 | automobile | 2.6 | 0.15 |
| val | b661da67-eb54-47d7-8d7b-1d1f8ca1f1e0 | 5 | 9 | 4.85 | 0.21 | person | 1.19 | -0.6 |
| val | 7fb3b33e-85d4-4d3e-b999-5eee301c2308 | 7 | 10.5 | 0.54 | 0.25 | automobile | 0.23 | -2.61 |
| val | 22ab49c5-2c92-463b-b890-0b2208c436b1 | 4.5 | 5.6 | 6.79 | 0.3 | automobile | -2.44 | 0.01 |
| val | a2bd8a78-0011-4790-9981-cb48bceb3896 | 10 | 14 | 3.55 | 0.34 | automobile | -1.65 | 1.02 |

## Part 3 — raw-pool pull-out census

**3a cost probe.** Packaging is per chunk (≤ 100 clips), declared in `features.csv`: `labels/egomotion.offline/egomotion.offline.chunk_NNNN.zip`
and `labels/obstacle.offline/obstacle.offline.chunk_NNNN.zip` (STORED zips, one `<clip_id>.<feature>.parquet` per clip),
`calibration/vehicle_dimensions/vehicle_dimensions.chunk_NNNN.parquet` (one row per clip). The online `labels/egomotion` zips are 41 MB per
chunk; I used `egomotion.offline` (10 Hz, 16 KB per clip), which is the pose the offline boxes are referenced to.

| probe (seed 20261003, chunks 10, 57, 2409; `r3_raw_probe.csv`) | value |
|---|---|
| 3 packages (ego + obstacle + vehicle dims) | 167.5 MB in 20.7 s = 8.1 MB/s (obstacle zips alone 10–15 MB/s) |
| bytes per clip | 578 KB (obstacle 561 KB, egomotion 17 KB) |
| raw_train remote size (tree listing, `r3_raw_tree_sizes.json`) | egomotion.offline 2.48 GB + obstacle.offline 79.33 GB + vehicle dims 0.01 GB = 81.8 GB over 1,576 chunks |
| projection, full packages, serial | **2.8 h** → P-COST ENOUGH; gate 3b passes, all raw_train scanned, no sampling |

**3b what was actually fetched.** Leaner than the projection, same coverage: all 1,576 egomotion.offline packages + 1,575 vehicle_dimensions files
(2.50 GB, 11.2 min, 8 workers), then range-reads of the obstacle member only for the 4,935 clips with a lateral ≥ 1.0 m move-off
(3.13 GB, 24 min). Total 5.6 GB fetched (5.5 GB on the NAS by `du`) instead of 82 GB. Chunk 3080 holds the one exam clip in the train split: its egomotion
zip was read member by member skipping the exam member, and its vehicle_dimensions file was not downloaded (1 event used the 4.07 m fallback).

**3c detector.** Same code as Part 1 (`r3_core.moveoffs`, `attach_actor`). Boxes in `obstacle.offline` are in the rig frame
(`reference_frame = rig`) and are moved to the clip frame with the `egomotion.offline` pose at `reference_frame_timestamp_us` (planar).
Bumper offset F = `rear_axle_to_bbox_center + length/2` from `vehicle_dimensions`. Turn filter: |heading change over 4 s from move onset| > 45°.

Clips scanned: 149773 of 153624 raw_train clips (97.5 %; the other 3851 have no `egomotion.offline`/`obstacle.offline` per `feature_presence.parquet`), 1576 chunks; scan errors 0; clip duration median 20.0 s.

Actor attachment: 4941 lateral events, status {'ok': 4941}; bumper offset source {'vehicle_dimensions': 4940, 'fallback_4.07': 1}; F min/median/max 3.66/4.06/4.09 m.

| set | events | distinct clips | per 1,000 clips scanned |
|---|---|---|---|
| all move-offs | 19398 | 19134 | 129.52 |
| lateral ≥ 1.0 m | 4941 | 4935 | 32.99 |
| lateral ≥ 1.0 m, heading change ≤ 45° | 3257 | 3255 | 21.75 |
| lateral + stationary box 0–8 m ahead of bumper, ±2 m (before turn filter) | 373 | 373 | 2.49 |
|   of which dropped as turns (> 45°) | 88 | 88 | 0.59 |
| **pull-out events (after removing turns)** | 285 | 285 | 1.9 |
| for reference: same with the rig-origin gap | 186 | 186 | 1.24 |

Rate 1.90 pull-out events per 1,000 scanned clips. Whole raw_train was scanned (no sampling): actual total 285; extrapolated to all 153624 raw_train clips incl. those without offline labels: 292.

Pull-out events: actor classes {'automobile': 277, 'person': 2, 'trailer': 2, 'protruding_object': 1, 'bus': 1, 'heavy_truck': 1, 'stroller': 1}; bumper gap median 3.21 m; lateral median 2.01 m; |heading change| median 16.8°; stop observed from clip start 141; lateral toward left 216 / right 69.

Countries of pull-out clips: {'United States': 163, 'Germany': 52, 'France': 16, 'Sweden': 15, 'Finland': 7, 'Portugal': 5}; in nurec_train: 2 clips.

Cross-check on the 737 nurec_train clips also scanned from raw: move-offs from clipgt (usdz) 144, from raw egomotion.offline 142, same (uuid, t_move ± 0.5 s) 137.

Spot-check — 10 pull-out events (seed 20261003):

| uuid | chunk | t_stop_start | t_move | t_depart | lat4_signed | dyaw4_deg | acat_bum | gap_bum | alat_bum |
|---|---|---|---|---|---|---|---|---|---|
| 35c86436-c06d-4c2f-8189-36546f201e86 | 2402 | 0 | 3.7 | 5.1 | 2.66 | 4.86 | automobile | 3.95 | -0.2 |
| 695863ef-7a28-4e3a-8faf-a677efc60a2b | 2706 | 10.5 | 13.8 | 19 | 1.73 | 40.74 | automobile | 4.52 | -1.3 |
| 7ce8500b-7b9c-4f73-96bd-1f5f2990a5da | 718 | 0 | 6.5 | 10.9 | 1.07 | 26.56 | automobile | 2.23 | -0.26 |
| 81ce115e-0c2a-4c27-9e1a-6d6716cf0ee1 | 386 | 0 | 5.7 | 8.8 | 1.64 | 18.71 | automobile | 2.03 | -0.47 |
| 8a4e8a71-24f9-48e0-b564-448de181c549 | 400 | 5.4 | 8.1 | 9.8 | -2.22 | -24.41 | automobile | 4.51 | 1.63 |
| 8f35f7db-f9ec-4d03-acc6-d91709b4fc8b | 776 | 0 | 2.6 | 4.2 | 2.65 | 6.89 | automobile | 3.34 | 0.04 |
| a3f56c5e-0c28-4bc9-a407-203b3198139b | 2831 | 0 | 2.9 | 4.1 | -2.64 | -34.72 | automobile | 3.3 | 1.04 |
| a75c5079-8a11-4cad-bdb1-fccd91678385 | 744 | 3.3 | 8.9 | 10 | 1.78 | 2.95 | automobile | 0.87 | -1.63 |
| b63218e7-9a27-4d0f-9cca-4f9565c74e8f | 403 | 0 | 7 | 10.4 | 1.84 | 40.89 | automobile | 3.71 | 0.16 |
| be89c713-08e2-49cf-9497-15e4adc9e376 | 478 | 4.9 | 7.5 | 9.3 | 3.74 | 40.33 | automobile | 2.22 | -1.91 |

## Run 3 verdicts

| id | statement | result | verdict |
|---|---|---|---|
| P-A3 | nurec_train ≥ 30 high-speed scenes, ≥ 4 bins | 208 scenes, 12 bins | ENOUGH |
| P-NA | nurec_train ≥ 100 around-obstacle events | 242 events in 93 scenes | ENOUGH |
| P-NAt | nurec_train ≥ 30 with clearance ≤ 1.0 m | 70 events in 33 scenes | ENOUGH |
| P-RBL | raw_train ≥ 100 pull-out events after removing turns | 285 events in 285 clips (actual, full scan) | ENOUGH |
| P-COST | raw egomotion + obstacle projection ≤ 24 h | 2.8 h projected; 35 min used | ENOUGH |

## Flags (Run 3, not fixed)

- **R3-F1** NuRec pull-outs stay negligible after the bumper fix: 2 events in nurec_train, 2 in nurec_val. The generator's pull-out data has to
  come from the raw pool (285 events), of which only 2 clips have a NuRec reconstruction.
- **R3-F2** The bumper fix changes which events qualify, not only how many: Run 2's box started at the rig origin, so its first ~4 m lay under
  the ego. On raw_train the rig-origin rule gives 186 events, the bumper rule 285.
- **R3-F3** The ±30° heading filter is almost a no-op at ≥ 16 m/s: 2 of 1,657 follow anchors dropped (train + val). Oncoming actors were not
  contaminating the tag.
- **R3-F4** Part 2's straight-ahead corridor also catches parked cars on the outside of a bend or at a turn: 90 of 242 train events have a
  heading change above 20° at the pass, and the path offset at the pass has a 4 m median. Restricting to ≤ 20° leaves 152 events, 64 with
  clearance ≤ 1.0 m, 45 of them with the human path offset ≥ 0.5 m. Both prediction thresholds still hold on that stricter set.
- **R3-F5** Part 2 has no events at ≥ 16 m/s and only 6 at 0–2 m/s in train; the tight passes are almost all at 2–8 m/s beside parked cars.
- **R3-F6** Ego box width from the usdz is 2.00–2.16 m (it appears to include mirrors), so clearances are conservative by roughly 0.1 m per side.
- **R3-F7** 3,851 raw_train clips (2.5 %) have no offline egomotion/obstacle labels and were not scanned; the extrapolated total is 292.
- **R3-F8** 141 of the 285 raw pull-outs are already stopped at clip start (stop onset censored). 216 go left, 69 right. 277 of 285 obstacles are automobiles.
- **R3-F9** Raw vs usdz agreement on move-offs for the same clips is 137 of 144 (± 0.5 s); the two sources are not bit-identical trajectories.
- **R3-F10** raw_val was written as a view but not scanned (not requested).
- **R3-F11** The three probe chunks' full obstacle zips (163 MB) remain on the NAS under `raw/labels/obstacle.offline/`; they are in the manifest.
