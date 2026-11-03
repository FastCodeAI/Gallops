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

### Pre-registered

| id | statement | result | verdict |
|---|---|---|---|
| Z1 | zero exam / test UUIDs in any output | manifest and NAS folders: {'manifest_exam': 0, 'manifest_test': 0, 'manifest_val': 0, 'disk_exam': 0, 'disk_test': 0, 'disk_val': 0, 'not_raw_train': 0} | holds |
| Z2 | extended index non-empty on ≥ 80 % of highway anchors | 94.7 % | holds |
| Z3 | D2 drops ≤ 5 % of nurec_train anchors | 6.2 % | fails |
| Z4 | D3 exemptions ≤ 3 % AND tight-pass human ≥ 0.8 on ≥ 80 % | 0.7 %; 93.3 % | holds |
| Z5 | mix within ±20 % of each target share | A 5.1 %; B 9.1 %; ordinary 85.8 % | fails |
| Z6 | original 8,192 rows bit-identical | verified at write time (s3_vocab.py) | holds |
