# S3 — training-set assembly (convert, grade, extend the vocabulary, split, manifest) — 2026-10-05/06

**VERDICT: not SET-READY. Z3 and Z5 fail; Z1, Z2, Z4 and Z6 hold.** No training was run. Nothing was rebalanced or re-thresholded.

| id | statement | result | verdict |
|---|---|---|---|
| Z1 | zero exam / test UUIDs in any output | 0 in the manifest and in every NAS folder (also 0 val) | holds |
| Z2 | extended-vocabulary index non-empty on ≥ 80 % of highway anchors | 94.7 % (original vocabulary: 12.1 %) | holds |
| Z3 | D2 drops ≤ 5 % of nurec_train anchors | 6.2 % (1,459 of 23,500) | **fails** |
| Z4 | D3 exemptions ≤ 3 %; tight-pass human ≥ 0.8 on ≥ 80 % | 0.7 %; 93.3 % | holds |
| Z5 | mix within ±20 % of each target share | A 5.1 % (target 10), B 9.1 % (target 5), ordinary 85.8 % (target 70) | **fails** |
| Z6 | original 8,192 vocabulary rows bit-identical | yes | holds |

**What exists.** 126,981 training samples from 4,018 raw_train clips (1,398 chunks, 25 countries), all with the bonnet mask (N = 37).
21,806 of them (NuRec scenes) carry label 2: a metric cache, grades, and an anchor index for both vocabularies. `vocab_10240.npy` adds
2,048 fast shapes to the 8,192.

**Declared departures from standard EPDMS (decided by Sid, applied in `s3_scorer.py`; vendored scorer untouched):**
- **D2** the ego box is the real vehicle from `calibration/vehicle_dimensions` (`length`, `width`, `rear_axle_to_bbox_center`, `wheelbase`;
  4 variants, 4.69–5.39 m × 2.00–2.16 m) instead of the Pacifica, and anchors whose human path fails the drivable-area gate are dropped:
  **1,459 anchors (6.2 %)** in 164 scenes.
- **D3** driving-direction gate exemption (speed ≤ 12 m/s for 4 s, stationary actor within 15 m ahead and ±3.5 m at the anchor,
  off-route time ≤ 4.5 s): applied to the human path on **160 anchors (0.7 %)** in 30 scenes, to stay-still on 88, and 5.5 million times to
  vocabulary shapes across 2,804 anchors. Every exemption is logged in `grade_rows/*.parquet`.

Env (same as S1/S2): `alpasim/.venv/bin/python` for the S1 converter (`s3_labels.py`, `s3_convert.py`, `s3_spotcheck.py`); conda
`drive-jepa` python with cwd = vendored navsim_v2 for picking, vocabulary, grading and the manifest (`run_s3_grade.sh`). CPU only, niced.
Scripts: `s3_common.py`, `s3_pick.py`, `s3_download.py`, `s3_fetch_maps.py`, `s3_labels.py`, `s3_vocab.py`, `s3_convert.py`,
`s3_scorer.py`, `s3_grade.py`, `run_grade_loop.sh`, `s3_manifest.py`, `s3_spotcheck.py`. Reused by import: `s1_convert.py` (masked,
without the model), `s2_scene.py` and `s2_grade.py` (fixed map; `build_cache` gained a `vp=` argument). `C.assert_train` (raw_train,
not exam / test / val) runs in every script on every clip; `s3_manifest.py` repeats it at the end on the manifest and on the NAS folder
names. NAS: `{runs}/drivejepa_retrain/04_training_set/` (`s1/clips`, `s1/calibration`, `s1/cache`, `scenes`); `download_manifest.csv`
21,952 rows, 57.4 GB, all with sha256. Nothing under adapters/, runs/, drafts/, alpasim/ or navsim_v2 was edited; the vocabulary is
written here, not into `navsim_v2/data/`.

## Part C — extended vocabulary

- **Source.** 14,376 label-1 futures with human speed ≥ 14 m/s, from 919 train clips (split asserted).
- **Format.** `8192.npy` is (8192, 40, 3) float32: pose k at (k+1)·0.1 s, columns x, y, heading, rear-axle frame; training takes
  `poses[:, 4::5]` (`drive_jepa_agent.py:49-50`). Each 8-pose future is turned into 40 poses by a cubic spline through the origin and
  the 8 poses (heading unwrapped), so poses 4, 9, …, 39 are the label poses exactly.
- **Clustering.** k-means in the 120-dimensional pose space, seed 20261006, 3 restarts → **2,048 shapes** (members per shape 1–90,
  median 4), appended in order of mean speed.

| mean speed of added shape, m/s | < 14 | 14–16 | 16–20 | 20–24 | 24–28 | 28–32 | 32–36 | 36–40 | ≥ 40 |
|---|---|---|---|---|---|---|---|---|---|
| shapes | 369 | 479 | 388 | 327 | 224 | 134 | 85 | 41 | 1 |

Mean speed 5.9–40.1 m/s (the slow ones start at ≥ 14 m/s and brake); mean curvature −0.054 to +0.044 1/m; heading change −118° to
+101°. Histogram: `vocab_hist.png`; per-shape table: `vocab_added.csv`.
- **File.** `vocab_10240.npy`, (10240, 40, 3) float32, sha256 `4b0bc3c9…9e785`; rows 0–8191 byte-identical to the original
  (sha256 `cc44a31e…8ad35`), checked after writing.
- **Can upstream load it without edits? No.** The size is not hard-coded (it comes from the array), but the **path is**:
  `np.load("./data/8192.npy")` at `drive_jepa_agent.py:49` and `scripts/misc/calc_anchors_scores.py:15`. The index file location is also
  derived by string replacement of upstream directory names (`compute_navsim_score.py:54-55`). Flagged, not edited.

## Part D — split, manifest

- **Group key.** `metadata/data_collection.parquet` has only `country`, `month`, `hour_of_day`, `platform_class`, `radar_config`: no
  drive, session, date or city. Fallback used: the dataset **chunk** (≤ 100 clips, single country, the closest thing to a drive group),
  from `clip_index.parquet`. All 3,146 chunks are single-split in NVIDIA's own split, so every group is in exactly one split (asserted).
- **Split.** Everything built here is NVIDIA `train`. No val samples were built (flag F3).
- **Manifest.** `manifest.parquet`, 126,981 rows: `sample_id`, `clip_uuid`, `source` (raw | nurec), `type` (A | B_raw | B_pass |
  ordinary), `split`, `has_label2`, `direction_exempt`, `anchor_t`, `converter_commit`, `grader_commit`, `vocab_sha256`, plus `group`,
  `country`, `speed`, `command`. The two commit fields hold the repo HEAD (`4254433`) plus a sha256 prefix of the script files, because
  `paddock-sid/` is outside git (flag F6).
- **On disk per sample** (`s1/cache/<clip_uuid>/<sample_id>/`): `pad_feature.gz`, `pad_target.gz`; with label 2 also `metric_cache.pkl`,
  `pad_label2.gz`, `anchors_scores_index_orig.npy`, `anchors_scores_index_ext.npy`, `anchors_scores_final.npy`.

### Download

| item | value |
|---|---|
| clips with front-wide video | 4018 of 4018 |
| front-wide video bytes | 57.0 GB (14.2 MB per clip) |
| wall time, 6 workers | 1.16 h |
| S1 projection for this many clips (13.4 MB, 8.8 s serial / 6) | 53.8 GB, 1.64 h |
| calibration parquets (chunks) | 2796 |
| NuRec map members (scenes) | 695 fetched + 40 copied from S2 |
| manifest rows / total bytes / rows without sha256 | 21952 / 57.4 GB / 0 |

### Samples

| source | type | clips | anchors_converted | dropped_d2 | samples_in_manifest | has_label2 | direction_exempt |
|---|---|---|---|---|---|---|---|
| nurec | A | 205 | 6550 | 135 | 6415 | 6406 | 1 |
| nurec | B_pass | 89 | 2848 | 321 | 2527 | 2471 | 134 |
| nurec | ordinary | 441 | 14102 | 1003 | 13099 | 12929 | 25 |
| raw | B_raw | 283 | 9053 | 0 | 9053 | 0 | 0 |
| raw | ordinary | 3000 | 95887 | 0 | 95887 | 0 | 0 |

Clips converted 4018 of 4018 (failed: 0); anchors 128440; in manifest 126981. Bonnet mask present on 128440 of 128440 samples (N = 37). Frame timing: > 20 ms from target on 87 anchors. Fewer than 4 ego_status rows: 8036. Command L / S / R: {'S': 105874, 'R': 11419, 'L': 11147}.

### Grading (nurec_train)

| status | anchors | share |
|---|---|---|
| ok | 21806 | 92.8 % |
| dropped_d2 | 1459 | 6.2 % |
| ego not on any lane | 200 | 0.9 % |
| error: GEOSException('TopologyException: side location conflict at -158.06373039412668 26.188535821052888. This can occur if the input geometry is invalid.') | 32 | 0.1 % |
| no 4 s future | 3 | 0.0 % |

**D2** real vehicle box (length 4.69–5.39 m, width 2.00–2.16 m, 4 variants). Anchors dropped because the human path fails the drivable-area gate: **1459 of 23500 = 6.2 %**, in 164 scenes (30 scenes lose half or more of their anchors). By type: {'ordinary': 1003, 'B_pass': 321, 'A': 135}.

**D3** (declared departure from standard EPDMS): direction-gate exemption applied to the human path on **160 anchors = 0.7 %** of nurec_train anchors, in 30 scenes; to the stay-still plan on 88; to vocabulary shapes 5522398 times across 2804 anchors. Anchors with a stationary actor ahead (condition b): 2844. Human anchors whose gate stays below 1 after D3: 615. Log: `grade_rows/*.parquet` (`human_exempt`, `human_ddc_standard`, `human_score_standard`, `human_opp_s`, `vocab_exemptions`).

Human path on graded anchors: mean score 0.928, ≥ 0.8 on 94.8 %; terms below 1: NC 0.5 %, DAC 0.0 %, TLC 0.0 %, DDC 2.8 %, EP 24.9 %, TTC 1.4 %, LK 26.1 %, HC 4.3 %. Clock offset raw→usdz -0.27–0.28 s; position residual median 0.061 m, scenes above 0.2 m: 118.

Tight passes (the 70 of S0 Run 3, same anchors as S2): graded 60, dropped by D2 9, not graded/unmatched 1. After D2 + D3: human ≥ 0.8 on **56 of 60 = 93.3 %**; human > stay on 96.7 %; exempted 5. Still below 0.8: NC 1, DAC 0, DDC 2, TTC 3, LK 2, HC 0.

Anchor index (shapes graded ≥ 0.95, `convert_to_index.py:17-22`), original 8,192 vs extended 10,240:

| human speed m/s | anchors | empty_orig | empty_ext | median_n_orig | median_n_ext | median_added | best_orig | best_ext |
|---|---|---|---|---|---|---|---|---|
| 0–2 | 3792 | 12.8 % | 12.8 % | 1.65e+03 | 1.65e+03 | 0 | 1 | 1 |
| 2–8 | 6281 | 21.6 % | 21.6 % | 300 | 300 | 0 | 1 | 1 |
| 8–16 | 6354 | 24.8 % | 17.3 % | 118 | 168 | 11 | 1 | 1 |
| 16–24 | 2859 | 77.3 % | 5.0 % | 0 | 70 | 67 | 0.792 | 1 |
| 24–32 | 1808 | 100.0 % | 5.8 % | 0 | 56 | 56 | 0.749 | 1 |
| ≥32 | 712 | 100.0 % | 4.9 % | 0 | 45 | 45 |  | 1 |

Z2: extended index non-empty on 5096 of 5379 highway anchors (≥ 16 m/s) = **94.7 %**; original index non-empty on 12.1 %. Shapes simulated per anchor: median 6572 of 10,240 (exact prefilter, validated on 117 anchors against full scoring: identical index sizes).

### Split and mix

| split | type | samples | clips | share_of_samples | target | relative_gap | within_20pct |
|---|---|---|---|---|---|---|---|
| train | A | 6415 | 205 | 5.1 % | ≥ 10.0 % | -49 % | False |
| train | B | 11580 | 372 | 9.1 % | ≥ 5.0 % | +82 % | False |
| train | ordinary | 108986 | 3438 | 85.8 % | ≥ 70.0 % | +23 % | False |

| split | source | type | samples | clips | has_label2 |
|---|---|---|---|---|---|
| train | nurec | A | 6415 | 205 | 6406 |
| train | nurec | B_pass | 2527 | 89 | 2471 |
| train | nurec | ordinary | 13099 | 438 | 12929 |
| train | raw | B_raw | 9053 | 283 | 0 |
| train | raw | ordinary | 95887 | 3000 | 0 |

Groups (chunks): 1398, every one in exactly one split (asserted; all 3146 raw chunks are single-split). Countries: {'United States': 1557, 'Germany': 581, 'France': 124, 'Sweden': 115, 'Italy': 101} … (25 in total). Dedupe: 0 raw clips share a UUID with a graded NuRec scene (the 2 raw pull-outs that are NuRec scenes are carried once, as NuRec).

## Spot-check images (B3)

`spot_png/` (20 PNGs + `_contact_sheet.png`), index `spot_index.csv`. White = label-1 path; green = 3 best-graded shapes; red = 3 worst;
drawn on the masked 512×256 input. Raw-only samples show the label alone.

## Flags (not fixed)

- **F1 — Z3: the kerb rule costs 6.2 %, concentrated.** 164 scenes lose anchors and 30 lose half or more; 1,003 of the 1,459 are in
  ordinary scenes, 321 in pass scenes. Even with the real box the human's own path leaves the derived drivable area there.
- **F2 — Z5: the mix is what the pools give.** Highway is 5.1 % because only 205 clips qualify; pull-outs and passes together are 9.1 %.
  Reaching 10 % highway needs either about 200 more highway clips or sample weights (S4).
- **F3 — no validation set was built.** Only train clips were converted and graded; NVIDIA's val split (404 NuRec scenes, 90,927 raw
  clips) is untouched. The brief's Part A and B name train sets only.
- **F4 — label 2 is missing on 235 NuRec anchors that stay in the manifest** (`has_label2 = false`): 200 where the ego is on no mapped
  lane, 32 from a geometry error in one scene's map, 3 with no 4 s future.
- **F5 — the extended index is still empty on 5 % of highway anchors**, and on 13–22 % of anchors below 8 m/s for both vocabularies
  (nothing reaches 0.95 there; e.g. `232d240e_07500`, where every shape scores 0).
- **F6 — no commit ids exist for our scripts** (`paddock-sid/` is git-excluded); file hashes stand in.
- **F7 — upstream cannot load `vocab_10240.npy` or our index files without an edit or a path shim** (hard-coded `./data/8192.npy`
  and directory-name string replacement). S4 scope.
- **F8 — the vocabulary prefilter is validated, not proven.** Shapes shorter than 0.6 × the reference progress − 3 m are not simulated.
  Full scoring on 117 anchors gave identical index sizes. Unscored shapes are NaN in `anchors_scores_final.npy`.
- **F9 — the index pick is seeded per token** (upstream draws 256 at random without a seed).
- **F10 — reference plan is still the human path** (no PDM-Closed), as in S2. The human's own label, built from the raw egomotion,
  differs slightly from that reference, so its progress term reads just under 1 on 25 % of anchors.
- **F11 — clock alignment.** Raw→usdz clock offset spans −0.27 to +0.28 s; 118 scenes have an untransformed position residual above
  0.2 m (median 0.06 m). Only the clock offset is applied; S2's rigid fit was not repeated per scene.
- **F12 — 8,036 samples have fewer than 4 `ego_status` rows** (anchors at 0.5 and 1.0 s) and 87 anchors are more than 20 ms from their
  target frame; both as in S1.
- **F13 — 6 of 741 nurec_train scenes and 2 of 285 pull-out clips are absent**: missing offline calibration or sitting in the one train
  chunk that contains an exam clip. The 2 raw pull-outs that are NuRec scenes are carried once, as NuRec.
- **F14 — ordinary raw clips were only screened on egomotion** (no move-off, never ≥ 16 m/s for 4.5 s, moves at least 3 m/s). Their
  obstacle labels were not read, so some contain passes.
- **F15 — shared machine.** Grading took 8.2 h of wall time on a host other users had loaded to 30–80; my jobs ran niced.

---

# S3.1 — validation (smoke scale) — 2026-10-06

**VERDICT: VAL-SMOKE-READY.** All V3 asserts pass and all 8 NuRec validation scenes are graded.

**Decisions from Sid (S3 review), recorded:**
- **Z3 accepted at 6.2 %.** The dropped anchors reflect our derived map's kerb lines, not the driving.
- **Z5 accepted.** Remedy is sample weights in training (highway counted twice); no rebalancing of the set.
- **The 235 ungraded NuRec moments stay** in the training set as generator-only samples (`has_label2 = false`).
- **The full validation set** (404 nurec_val scenes plus a raw_val sample) **is deferred to S4.** What follows is a 1 % scale set for the smoke test.

**V1 pick** (seed 20261007, `val/clips.csv`): 8 nurec_val scenes by S0 Run 3 flags (2 high-speed, 3 close-pass with clearance ≤ 1.0 m,
3 ordinary; eligible pools 110 / 22 / 183) and 12 raw_val clips, one per chunk, 20 distinct chunks, 7 countries. **All 12 raw clips are
ordinary**: raw_val was never scanned for pull-outs (S0 Run 3 scanned raw_train only), so none are known. They were screened on
egomotion like the train ordinary clips (no move-off, never ≥ 16 m/s for 4.5 s, moves at least 3 m/s).

**V2 convert and grade.** Same scripts run with `S3_SPLIT=val`, which repoints the working folder to `04_training_set/val/` and flips every
split assertion to val (`s3_common.assert_val`: raw_val / nurec_val, never exam, test or train). Same converter (bonnet mask N = 37 on
639 of 639 samples), same wrapper `s3_scorer.py` (D2 + D3), same `vocab_10240.npy`. New downloads: 0.27 GB of front-wide video, 8 map
sets, and the egomotion package and vehicle-dimension file for the 20 val chunks; all in `download_manifest.csv` with sha256.

| source | type | clips | samples in manifest | has_label2 |
|---|---|---|---|---|
| nurec | A (high-speed) | 2 | 64 | 64 |
| nurec | B_pass (close-pass) | 3 | 65 | 65 |
| nurec | ordinary | 3 | 96 | 96 |
| raw | ordinary | 12 | 383 | 0 |
| **total** | | **20** | **608** | **225** |

Grading: 8 of 8 scenes graded, no scene errors; 256 anchors → 225 graded, 31 dropped by D2 (all in two close-pass scenes, 13 and 18
anchors); direction-gate exemption on 9 human paths; human path ≥ 0.8 on 92.9 % of graded anchors. Highway anchors (≥ 16 m/s): 64, extended
anchor index non-empty on 100 %, original on 10.9 %.

**V3 manifest.** `val/manifest_val.parquet`, 608 rows, same 15 columns as the train manifest, `split = val`. Asserted in
`s31_val_manifest.py`: column list equals train's; every clip is raw_val (NuRec ones nurec_val) per `clip_index.parquet`; 0 exam UUIDs;
0 test UUIDs; 0 clip UUIDs or sample ids shared with train; **0 of the 20 val chunks appear among the 1,398 train chunks**.

**V4 spot-check.** `val/spot_png/` (4 PNGs + contact sheet), index `val/spot_index.csv`, same format as train.

Flags:
- **No pull-outs in validation.** Finding any needs an obstacle scan of raw_val (934 chunks), which belongs with the full set in S4.
- **D2 drops 12 % of these val anchors** (31 of 256), above the train rate, because two of the three close-pass scenes are narrow streets.
- **One raw "ordinary" spot-check sample is at 16.6 m/s**: ordinary clips may touch highway speed briefly; the screen only excludes
  holding ≥ 16 m/s for 4.5 s.
- **Scripts changed for the split switch:** `s3_common.py` (`S3_SPLIT`, `assert_val`), `s3_convert.py` / `s3_labels.py` / `s3_spotcheck.py`
  (data paths through `C.HERE`). Train outputs were not regenerated or touched.
