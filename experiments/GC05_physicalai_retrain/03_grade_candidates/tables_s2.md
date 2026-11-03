### Step 1–3 — map, agents, alignment (pilot)

| type | uuid | lanes | route_lanes | intersections | stop_lines_explicit | traffic_lights | tracks | drivable_m2 | dt_s | pos_med | pos_max | head_med | head_max | png |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 7cba46df | 50 | 19 | 0 | 0 | 0 | 1 | 6772 | 0.01 | 0.154 | 0.336 | 0.0238 | 0.0991 | scene_png/A_7cba46df.png |
| A | fcd8cc77 | 188 | 69 | 0 | 0 | 0 | 46 | 22193 | 0.01 | 0.147 | 0.43 | 0.0255 | 0.0855 | scene_png/A_fcd8cc77.png |
| A | 82471d1e | 273 | 52 | 4 | 0 | 0 | 81 | 22773 | 0.01 | 0.0983 | 0.247 | 0.0399 | 0.208 | scene_png/A_82471d1e.png |
| A | ad134e5b | 232 | 108 | 0 | 0 | 0 | 20 | 26987 | 0.03 | 0.113 | 0.862 | 0.025 | 0.0915 | scene_png/A_ad134e5b.png |
| T | 3259c4fe | 120 | 20 | 7 | 0 | 0 | 34 | 7477 | 0 | 0.043 | 0.0781 | 0.0291 | 0.0528 | scene_png/T_3259c4fe.png |
| T | 2f3ed8fe | 372 | 55 | 5 | 61 | 0 | 224 | 19280 | 0.01 | 0.053 | 0.0844 | 0.0981 | 0.164 | scene_png/T_2f3ed8fe.png |
| T | 0771fd01 | 86 | 22 | 0 | 0 | 0 | 19 | 2684 | 0.01 | 0.034 | 0.0914 | 0.0323 | 0.0806 | scene_png/T_0771fd01.png |
| T | 1e883544 | 47 | 25 | 1 | 0 | 0 | 346 | 3447 | 0 | 0.0128 | 0.0323 | 0.0402 | 0.0503 | scene_png/T_1e883544.png |
| O | 25020844 | 296 | 24 | 2 | 42 | 73 | 86 | 19102 | 0 | 0.0135 | 0.0489 | 0.00686 | 0.0254 | scene_png/O_25020844.png |
| O | 7d257301 | 179 | 64 | 5 | 4 | 10 | 100 | 9747 | 0.01 | 0.0503 | 0.0778 | 0.0106 | 0.0333 | scene_png/O_7d257301.png |

P1: median residual ≤ 0.2 m and ≤ 0.5° on **10 of 10** pilot scenes (position median max 0.154 m, heading median max 0.098°; worst single-pose residual 0.86 m). Clock offset 0.00–0.03 s. `rig_trajectories.json` vs clipgt egomotion after a rigid fit: max residual 1.2e-13 m (same trajectory).

### Step 4 — pilot grades

Anchors graded 319 of 319; not graded: {}.

| type | plan | n | NC | DAC | TLC | DDC | EP | TTC | LK | HC | score | share_ge_0_8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | human | 127 | 1 | 0.913 | 1 | 0.858 | 1 | 1 | 0.921 | 0.953 | 0.764 | 0.756 |
| A | stay still | 127 | 0.984 | 0.976 | 1 | 0.988 | 0.427 | 0.598 | 0.858 | 0 | 0.466 | 0.0315 |
| A | model chosen | 127 | 0.992 | 0.961 | 1 | 0.988 | 0.676 | 0.772 | 0.906 | 0 | 0.614 | 0.0945 |
| A | 32 proposals (all) | 4064 | 0.998 | 0.946 | 1 | 0.988 | 0.674 | 0.801 | 0.897 | 0.00295 | 0.616 | 0.101 |
| O | human | 64 | 1 | 1 | 1 | 1 | 1 | 1 | 0.5 | 1 | 0.929 | 1 |
| O | stay still | 64 | 1 | 1 | 1 | 1 | 0.514 | 1 | 0.531 | 0.438 | 0.679 | 0.297 |
| O | model chosen | 64 | 1 | 0.922 | 1 | 0.984 | 0.944 | 0.953 | 0.5 | 1 | 0.824 | 0.844 |
| O | 32 proposals (all) | 2048 | 0.892 | 0.907 | 1 | 0.965 | 0.846 | 0.886 | 0.497 | 0.966 | 0.733 | 0.735 |
| T | human | 128 | 0.977 | 0.695 | 1 | 1 | 0.992 | 0.914 | 0.477 | 0.945 | 0.653 | 0.68 |
| T | stay still | 128 | 0.961 | 0.672 | 1 | 1 | 0.553 | 0.953 | 0.508 | 0.531 | 0.465 | 0.156 |
| T | model chosen | 128 | 0.922 | 0.68 | 1 | 0.977 | 0.855 | 0.898 | 0.516 | 0.969 | 0.599 | 0.586 |
| T | 32 proposals (all) | 4096 | 0.81 | 0.66 | 1 | 0.969 | 0.797 | 0.79 | 0.513 | 0.948 | 0.503 | 0.487 |

P2: human EPDMS ≥ 0.8 on 160 of 191 A and ordinary anchors = **83.8 %** (A 76 %, ordinary 100 %, tight-pass scenes 68 %). Terms below 1 on those human paths: NC 0 %, DAC 6 %, TLC 0 %, DDC 10 %, EP 8 %, TTC 0 %, LK 22 %, HC 3 %.

P3: std of the 32 grades > 0.05 on 28 of 127 A anchors = **22.0 %** (median std A 0.010, T 0.211, O 0.299). On A anchors the terms that vary across the 32: NC 2 %, DAC 6 %, TLC 0 %, DDC 0 %, EP 79 %, TTC 13 %, LK 10 %, HC 7 %.

Model's chosen plan vs the best of its 32 by this grader: mean chosen 0.650, mean best 0.741; chosen is the top-graded plan on 46 % of anchors. Correlation (Spearman, per anchor, model's own score vs grade): median 0.16.

### Complete sample through the upstream loaders

`CacheOnlyDataset` (`training/dataset.py:33`) over `{runs}/drivejepa_retrain/03_grade_candidates/s1/cache` with builders `pad_feature`, `pad_target`, `pad_label2`: 319 samples. One sample (`2f3ed8fe_00500`):

| part | content |
|---|---|
| images + ego_status (pad_feature.gz) | camera_feature_1 (3, 256, 512); camera_feature_2 (3, 256, 512); camera_feature (2, 3, 448, 768); lidar2img (2, 4, 4); img_shape (2, 3); ego_status (2, 11) |
| label 1 (pad_target.gz) | trajectory (8, 3) torch.float64 |
| label 2 grades (pad_label2.gz) | label2_scores (34, 9) = 34 plans × ['NC', 'DAC', 'TLC', 'DDC', 'EP', 'TTC', 'LK', 'HC', 'score']; has_label2 True |
| label 2 source (metric_cache.pkl, lzma) | `MetricCacheLoader` finds 319 tokens; `load_metric_cache` → MetricCache: observation 41 steps, centerline 215 m, 55 route lanes, drivable map 150 polygons, ego v 5.2 m/s |

### Step 5 — B formula test (human path from the usdz vs stay still)

Tight passes graded: 69 of 70 ({'ego not on any lane': 1}). Anchor = the last 0.5 s anchor at least 0.5 s before the pass.

| clearance_m | events | human_score | stay_score | human_ge_0_8 | human_gt_stay | h_NC | h_DAC | h_TLC | h_DDC | h_EP | h_TTC | h_LK | h_HC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ≤0.3 | 4 | 0.911 | 0.632 | 0.75 | 1 | 1 | 1 | 1 | 1 | 0.999 | 0.75 | 1 | 1 |
| 0.3–0.5 | 15 | 0.562 | 0.365 | 0.6 | 0.6 | 0.933 | 0.733 | 1 | 0.867 | 0.999 | 0.933 | 0.333 | 1 |
| 0.5–0.75 | 24 | 0.75 | 0.519 | 0.75 | 0.75 | 1 | 0.875 | 1 | 0.854 | 1 | 1 | 0.625 | 1 |
| 0.75–1.0 | 26 | 0.666 | 0.425 | 0.692 | 0.769 | 0.962 | 0.808 | 1 | 0.942 | 1 | 0.846 | 0.5 | 0.923 |
| all | 69 | 0.687 | 0.457 | 0.696 | 0.739 | 0.971 | 0.826 | 1 | 0.899 | 1 | 0.913 | 0.536 | 0.971 |

P4: human EPDMS ≥ 0.8 on 48 of 69 = **69.6 %**; human > stay on 51 of 69 = **73.9 %** (both need ≥ 80 %).

Why the 21 human passes below 0.8 fall short (a term counts when it is below 1): NC 2, DAC 12, TLC 0, DDC 8, EP 2, TTC 6, LK 15, HC 2. Gate terms (score forced to 0 or halved): NC 2, DAC 12, DDC 8. Lane keeping alone caps the score at 0.857; LK is the only failing term on 0 of them.

Human ≤ stay on 18 events: both 0 on 12, tie above 0 on 1, stay strictly higher on 5. Stay still loses mainly on comfort (HC mean 0.41, a hard stop from speed) and progress (EP mean 0.45).

Lane-keeping failure rate of the human pass by clearance: ≤0.3 0 %; 0.3–0.5 67 %; 0.5–0.75 38 %; 0.75–1.0 50 %. Share of the 4 s the scorer's ego box straddles two lanes: mean 0.55.

Per-event table: `btest.csv` (70 tight + 2 pull-outs; per-term columns for human and stay). The 10 lowest human grades:

| uuid | t_anchor | speed | clearance | cat | human_score | stay_score | human_NC | human_DAC | human_DDC | human_TTC | human_LK | human_EP | human_HC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8a812020 | 7 | 10.9 | 0.33 | trailer | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 |
| 03f14707 | 7.5 | 7.02 | 0.39 | heavy_truck | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| 03f14707 | 8 | 6.9 | 0.4 | automobile | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| 454a378b | 3.5 | 4.41 | 0.42 | automobile | 0 | 0.86 | 1 | 1 | 0 | 1 | 0 | 1 | 1 |
| 0771fd01 | 4 | 10.6 | 0.46 | automobile | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| dc1966f4 | 16 | 6.39 | 0.49 | protruding_object | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| 03f14707 | 9 | 6.9 | 0.53 | automobile | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| 2ff0eef3 | 4 | 7.19 | 0.7 | person | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| 46d6b6fc | 10 | 7.3 | 0.73 | automobile | 0 | 0.36 | 1 | 1 | 0 | 1 | 0 | 1 | 1 |
| a032b66e | 4.5 | 10.3 | 0.74 | automobile | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |

NuRec pull-outs (anchor at standstill just before move-off):

| uuid | t_anchor | speed | human_score | stay_score | human_NC | human_DAC | human_TLC | human_DDC | human_EP | human_TTC | human_LK | human_HC | stay_NC | stay_DAC | stay_TLC | stay_DDC | stay_EP | stay_TTC | stay_LK | stay_HC | human_nondrivable | stay_nondrivable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81ce115e | 5.5 | 0.25 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0.73 | 1 |
| febd36c4 | 8.5 | 0.23 | 0.43 | 0.5 | 1 | 1 | 1 | 0.5 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
