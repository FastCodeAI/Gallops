### Part A — map fixes (40 scenes)

| quantity | S2 | S2_1 |
|---|---|---|
| drivable area, m² per scene (median) | 1.7e+04 | 1.91e+04 |
| drivable area, m² (sum over 40 scenes) | 6.5e+05 | 728706 |
| road boundaries read (sum) | 0 | 7455 |
| boundary–lane pairs from `ROAD_BOUNDARY_TO_LANE` (sum) | 0 | 10585 |
| strips added (sum) | 0 | 5744 |
| pairs skipped: more than 8 m apart / degenerate | 0 | 52 / 438 |
| on-route gaps bridged (scenes) | 0 | 36 (20) |

Area added by the boundary strips: +12.2 % overall (per scene 172–5182 m², median 1712).
Bridged gaps, mean width (m): ≤0.05 m: 8; 0.05–0.1 m: 3; 0.1–0.25 m: 20; 0.25–0.5 m: 4; 0.5–1.0 m: 1. Median width 0.14 m, max 0.73 m; 23 of 36 lie under the ego-centre path. Most are long hairline seams between neighbouring lanes (median length 35 m), not end-to-end gaps.

Before/after PNGs: `s21_png/parking_strip_3259c4fe.png`, `s21_png/highspeed_gap_ad134e5b.png`, `s21_png/tight_pass_0771fd01.png`.

### Part B — re-grade, same anchors and plans

| quantity | S2 | S2.1 |
|---|---|---|
| pilot anchors graded | 319 | 319 |
| Q1 stay-still fails drivable area: pilot anchors | 14.1 % | 7.8 % |
| Q1 … tight-pass anchors | 18.8 % | 14.5 % |
| Q1 … all graded anchors | 14.9 % | 9.0 % of 388 |
| Q2 / P2 human ≥ 0.8, high-speed + ordinary | 83.8 % | 100.0 % |
|   high-speed | 75.6 % | 100.0 % |
|   ordinary | 100.0 % | 100.0 % |
|   tight-pass scenes (all anchors) | 68.0 % | 70.3 % |
| P3 (informational) std of 32 grades > 0.05, high-speed | 22.0 % (median std 0.010) | 18.9 % (median std 0.010) |
| Q3 / P4 tight passes graded | 69 | 69 |
| Q3 human ≥ 0.8 | 69.6 % | 69.6 % |
| Q3 human > stay | 73.9 % | 73.9 % |
| tight-pass human path, mean per term (NC / DAC / DDC / TTC / LK / HC) | 0.97 / 0.83 / 0.90 / 0.91 / 0.54 / 0.97 | 0.97 / 0.84 / 0.90 / 0.91 / 0.54 / 0.97 |
| pilot human path, mean per term (NC / DAC / DDC / TTC / LK / HC) | 0.99 / 0.84 / 0.94 / 0.97 / 0.66 / 0.96 | 0.99 / 0.89 / 1.00 / 0.97 / 0.66 / 0.96 |
| model's chosen plan, mean grade / best of its 32 | 0.650 / 0.741 | 0.674 / 0.761 |

Why stay-still still fails the drivable-area gate (36 anchors, `s21_dac_diag.csv`): the box centre is inside the drivable area on all 36; on 17 a corner of the scorer's 2.30 m-wide box is already over the mapped road edge at the anchor (median overhang 0.33 m, max 1.46 m; 6 would fit with a 2.0 m box), on the other 19 a corner crosses it while the plan brakes to a stop. They sit in 11 scenes, all narrow streets.

Tight passes below 0.8 in S2.1 (21): terms below 1 — NC 2, DAC 11, TLC 0, DDC 8, EP 2, TTC 6, LK 15, HC 2.

### B2 — driving-direction gate, measured (diagnostic only; not written to any cache)

| formula | events | human_mean | stay_mean | human_ge_0_8 | human_gt_stay |
|---|---|---|---|---|---|
| standard formula | 69 | 0.687 | 0.476 | 69.6 % | 73.9 % |
| direction gate disabled | 69 | 0.759 | 0.498 | 78.3 % | 82.6 % |

Events where the gate bites the human pass: 8 of 69 (gate 0.5 on 2, 0 on 6). In these the human spends 0.6–4.1 s of the 4 s (median 3.4 s) in an opposite-direction lane at 2.5–10.6 m/s; the obstacle is stationary in 4 of 8. Disabling the gate lifts 6 of them to ≥ 0.8; the others still fail on NC 0, DAC 1, TTC 1, LK 2.

Any time in an opposite-direction lane: 8 of 69 events; gate untouched (DDC = 1) on 0 of them (less than 2 m of travel there within 1 s).

| uuid | t_anchor | speed | clearance | cat | obstacle_stationary | opp_lane_s | opp_lane_speed | human_DDC | human_score | human_score_no_ddc | stay_score | stay_score_no_ddc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8a812020 | 7 | 10.9 | 0.33 | trailer | False | 2.1 | 10.6 | 0 | 0 | 0.86 | 0 | 0.47 |
| 454a378b | 3.5 | 4.41 | 0.42 | automobile | False | 2.9 | 8.67 | 0 | 0 | 0.86 | 0.86 | 0.56 |
| e5718dc1 | 7.5 | 2.79 | 0.63 | automobile | True | 2.7 | 2.49 | 0.5 | 0.43 | 0.86 | 0.43 | 0.63 |
| 46d6b6fc | 10 | 7.3 | 0.73 | automobile | False | 3.8 | 7.01 | 0 | 0 | 0.86 | 0.36 | 0.46 |
| a032b66e | 4.5 | 10.3 | 0.74 | automobile | True | 0.6 | 10.3 | 0 | 0 | 0 | 0 | 0.86 |
| 46d6b6fc | 9 | 7.52 | 0.75 | automobile | True | 4.1 | 7.16 | 0 | 0 | 0.86 | 0.36 | 0.46 |
| 46d6b6fc | 8 | 7.49 | 0.82 | automobile | False | 4.1 | 7.33 | 0 | 0 | 0.86 | 0.36 | 0.46 |
| e5718dc1 | 6 | 3.18 | 0.9 | automobile | True | 4.1 | 2.66 | 0.5 | 0.25 | 0.5 | 0 | 0 |

NuRec pull-outs:

| uuid | t_anchor | speed | opp_lane_s | opp_lane_speed | human_DDC | human_DAC | human_NC | human_score | human_score_no_ddc | stay_score | stay_score_no_ddc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 81ce115e | 5.5 | 0.25 | 0 |  | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| febd36c4 | 8.5 | 0.23 | 3.8 | 2.43 | 0.5 | 1 | 1 | 0.43 | 0.85 | 0.5 | 0.16 |

Per-event table for all 72: `btest.csv` (columns `opp_lane_s`, `opp_lane_speed`, `obstacle_stationary`, `*_score`, `*_score_no_ddc`, per-term).

### Part C — vocabulary

| speed above, m/s | anchors by max speed | anchors by final speed | anchors by mean speed | % of 8,192 |
|---|---|---|---|---|
| 16 | 0 | 0 | 0 | 0 |
| 20 | 0 | 0 | 0 | 0 |
| 25 | 0 | 0 | 0 | 0 |
| 30 | 0 | 0 | 0 | 0 |
| 35 | 0 | 0 | 0 | 0 |

Max speed per anchor: median 6.2, p99 13.9, **maximum 14.79 m/s**; final speed maximum 14.61 m/s; farthest end point 58.9 m ahead in 4 s. Histogram: `s21_png/vocab_speed_hist.png`. Q4: 0.00 % of anchors exceed 20 m/s → **VOCAB-CAPPED**.

Anchor-index generation on our own metric caches (upstream recipe, 8,192 anchors per token):

| token | type | human_speed | secs | anchors_ge_0_95 | best | max_speed_among_ok | median_speed_among_ok | human_grade |
|---|---|---|---|---|---|---|---|---|
| fcd8cc77_04500 | A | 31.4 | 14.8 | 0 | 0.724 |  |  | 1 |
| ad134e5b_03000 | A | 22.8 | 17.3 | 0 | 0.748 |  |  | 1 |
| 2f3ed8fe_04500 | T | 5.5 | 18.3 | 383 | 1 | 10.1 | 2.8 | 0 |
| 25020844_10500 | O | 4.1 | 24.9 | 556 | 1 | 7 | 1.8 | 1 |

### Part D — bonnet mask

Bonnet top edge found on 29 of 30 clips: bonnet height 8–35 input rows (median 23); N = 35 + 2 = **37 rows** of 256. Failed detections: ['f1fd39ee'] (edge placed at the top of the search window; see `s21_png/bonnet_edges.png`). The dataset's own ego mask is not usable for this: it is a template reaching 48 rows (median) and covers road above the bonnet (`s21_png/bonnet_mask_check.png`).

| check | n | unmasked | masked | change |
|---|---|---|---|---|
| C3 fastest of 32 ≤ 18.8 m/s (high-speed anchors ≥ 20 m/s) | 141 | 84.4 % | 82.3 % | -2.1 points |
| C4 median chosen first-step speed ÷ human speed (ordinary, 5–15 m/s) | 248 | 0.936 | 0.930 | -0.005 |
| fastest-of-32 end speed, median over all anchors (m/s) | 640 | 10.50 | 10.17 | -0.33 |
| chosen plan end speed, mean |masked − unmasked| (m/s) | 640 |  | 0.93 | same proposal index on 48.4 % |

Q5: C4 in [0.8, 1.2] yes; C3 within 5 points yes → **holds**.
