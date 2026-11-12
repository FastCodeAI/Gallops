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

| id | statement | result | verdict |
|---|---|---|---|
| P-A3 | nurec_train ≥ 30 high-speed scenes, ≥ 4 bins | 208 scenes, 12 bins | ENOUGH |
| P-NA | nurec_train ≥ 100 around-obstacle events | 242 events in 93 scenes | ENOUGH |
| P-NAt | nurec_train ≥ 30 with clearance ≤ 1.0 m | 70 | ENOUGH |
