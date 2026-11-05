# S2 — label-2 converter (NuRec map + raw photos → per-plan grades), pilot + B formula test — 2026-10-03

**VERDICT: not LABEL2-OK. Alignment holds (P1). FORMULA-BIAS (P4 fails) and REFEREE-FLAT (P3 fails) both apply; P2 also fails.**
Most of the human-path failures trace to the map I derived from the usdz (lane-polygon drivable area, lane-level route), not to the
scorer's formula; one term is a real formula effect (driving direction, when the human passes using an opposite-direction lane). Nothing
in the formula or thresholds was changed.

| id | statement | result | verdict |
|---|---|---|---|
| P1 | alignment median ≤ 0.2 m and ≤ 0.5° on ≥ 9 of 10 | 10 of 10 (worst median 0.154 m, 0.098°) | holds |
| P2 | human grade ≥ 0.8 on ≥ 90 % of A + ordinary anchors | 160 of 191 = 83.8 % (A 76 %, ordinary 100 %) | fails |
| P3 | std of the 32 grades > 0.05 on ≥ 80 % of A anchors | 28 of 127 = 22.0 % | fails → REFEREE-FLAT |
| P4 | human ≥ 0.8 on ≥ 80 % of 70 tight passes AND human > stay on ≥ 80 % | 69.6 % and 73.9 % (69 graded) | fails → FORMULA-BIAS |
| P5 | loss split | yes, one sum at `drive_jepa_agent.py:313-323`; but both parts need the metric cache (0a) | yes, with a caveat |

**Which term drags the human pass down (P4).** Of the 21 tight passes graded below 0.8: drivable-area gate 12, driving-direction gate 8,
lane keeping 15, TTC 6, collision 2. There is **no clearance threshold**: gate failures occur from 0.33 m to 0.97 m and the four tightest
passes (≤ 0.3 m) pass both gates.
- **Drivable area (12):** on 10 of the 12 the stay-still plan fails the same gate, i.e. the ego is already outside the derived drivable
  area at the anchor. That is the derived map being narrower than the road the human uses (parked-car strips are not lanes), not the pass.
- **Driving direction (8):** the human's centre spends 75 % of the 4 s in a lane mapped in the opposite direction (245 of 328 steps;
  `ddc_attribution.csv`). This is the formula: more than 6 m of progress in an off-route lane within 1 s sets the gate to 0.
- **Lane keeping:** fails on 46 % of tight passes (more than 0.5 m from the centreline for 2 s) but costs at most 0.14, so alone it never
  takes a pass below 0.8.
- With the 19 gate-failing events set aside, the remaining 50 give human ≥ 0.8 on 96 % and human > stay on 100 %.

**Why the highway is flat (P3).** On A anchors every one of the 32 proposals slows down hard from the human's speed, so history comfort is 0 for
99.7 % of them and only progress differs: grades cluster at 0.69–0.72 (median std 0.010). The grader does separate the human (mean 0.76
with the map artefacts, 1.00 on the clean scene) from the menu (0.62). The flatness is in the menu, not in the referee's sensitivity.

Env (same as S1): `alpasim/.venv/bin/python` + conda `drive-jepa` worker for photos, ego_status, label 1 and the 32 proposals (S1's
`s1_download.py` and `s1_convert.py`, imported unmodified through `s2_s1_run.py`, which only repoints their output roots); the conda
`drive-jepa` python (nuplan + vendored navsim_v2, cwd = navsim_v2, CPU) for map, agents and grading (`run_s2.sh`). Scripts: `s2_common.py`,
`s2_pick.py`, `s2_fetch.py`, `s2_s1_run.py`, `s2_scene.py`, `s2_grade.py`, `s2_run.py`, `s2_tables.py`, `s2_attrib.py`. All 40 scenes pass
`common.load_split_views()` as nurec_train (`s2_common.assert_nurec_train`: not exam, not val, split == train), asserted in every script.
NAS `{runs}/drivejepa_retrain/03_grade_candidates/` 1.2 GB; `download_manifest.csv` 740 rows, 170 MB, all with sha256. No GPU, no full usdz,
nothing under adapters/, runs/, drafts/, alpasim/, navsim_v2 or the S0/S1 folders was edited.

## Step 0 — read-only answers

**0a Loss split (P5).** One sum, `pad_loss` in `navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py:313-323`:
`trajectory_weight·trajectory_loss + sub_score_weight·sub_score_loss + final_score_weight·final_score_loss + pred_ce/l1/area + agent + bev`
(weights `drive_jepa_config.py:49-56`: trajectory 1, final score 1, sub score 0, ce 1, l1 0.1, area 2). Generator loss =
`trajectory_loss_anchors` (`:221-269`); scorer loss = `score_loss` (`:133-165`). They are separate terms, but **not independent of label 2**:
- `compute_score` (`:276`, body `:103-131`) runs first for every sample and indexes `metric_cache_paths[token]` (`:113`); a token without a
  metric cache raises there, before any loss term.
- The generator loss itself adds pseudo-targets taken from scorer-labelled vocabulary anchors (`scores_index`, `:234-261`,
  loaded from `anchors_scores_index_v2/<token>.npy`, `compute_navsim_score.py:54-55`).
A per-sample `has_label2` mask therefore has three places to go: skip the sample in `compute_score` (`:112-115`), drop the pseudo-target
part of the generator loss (`:234-261`), and mask `score_loss` inputs (`:290-299`). Raw-only samples can train head 1 with the human-target
term (`:227-232`) alone.

**0b Metric cache.** `MetricCache` (`navsim/planning/metric_caching/metric_cache.py:33-55`), written by
`MetricCacheProcessor.compute_metric_cache` (`metric_cache_processor.py:49`, `:360`):

| field | what it is | computed from |
|---|---|---|
| `ego_state` | nuPlan `EgoState` at the anchor | `scenario.initial_ego_state` |
| `trajectory` | PDM-Closed reference plan | `PDMClosedPlanner` on the nuPlan map API, route roadblock ids, mission goal, traffic-light status at iteration 0 |
| `observation` | `PDMObservation`: one occupancy map per 0.1 s for 4 s | agent boxes sampled at 2 Hz and linearly interpolated to 10 Hz; red lights as polygons of on-route lane connectors whose status is red |
| `centerline` | `PDMPath` along the route lanes | lane-graph search inside PDM-Closed |
| `route_lane_ids` | lanes of the route roadblocks | `scenario.get_route_roadblock_ids()` → `map_api.get_map_object(ROADBLOCK / ROADBLOCK_CONNECTOR).interior_edges` |
| `drivable_area_map` | `PDMDrivableMap` polygons with layer types | `PDMDrivableMap.from_simulation(map_api, ego_state, radius)`: lanes, lane connectors, roadblocks, intersections, drivable area, car parks |
| `past_human_trajectory` | 1.5 s of human ego states | `scenario.get_ego_past_trajectory(0, 1.5)` |
| `human_trajectory`, tracked-object lists, `map_parameters` | bookkeeping | scenario |

The scorer Drive-JEPA trains against is its own copy, `score_module/train_pdm_scorer.py` (called from `compute_navsim_score.py:52-76`):

| term | type | inputs | rule |
|---|---|---|---|
| NC no at-fault collision (`:404`) | gate (0, or 0.5 for static objects) | ego box per step, agent occupancy maps, ego area | collision with a stopped track, or front collision, or lateral collision while in several lanes / off road |
| DAC drivable area (`:484`) | gate | 4 ego corners vs drivable polygons | 0 if any corner is outside at any step |
| TLC traffic light (`:686`) | gate | red-light polygons in the observation | 0 if the ego box touches one |
| DDC driving direction (`:497`) | gate (1 / 0.5 / 0) | ego centre vs on-route lane polygons | progress made while outside every on-route lane, over 1 s: < 2 m, < 6 m, else |
| EP progress (`:548`, `:234`) | weight 5 | centreline projection of start and end | progress ÷ max(own, PDM-Closed reference) when that max > 5 m |
| TTC (`:566`) | weight 5 | ego box projected 0–0.9 s ahead, agents | 0 on any projected overlap that is ahead or (in several lanes / intersection) not behind |
| LK lane keeping (`:723`) | weight 2 | ego centre vs centreline, intersection layer | 0 if > 0.5 m off for ≥ 2 s outside intersections |
| HC history comfort (`:769`) | weight 2 | past human states + plan | comfort bounds on the padded sequence |

Score = product of gates × weighted mean of the four weighted terms (`_aggregate_pdm_scores`, `:234-270`); the two-frame extended comfort
term of full EPDMS is excluded in this training scorer. Ego box = Pacifica (5.18 × 2.30 m, `:107`).

**0c Bonnet / viewpoint.** `step0c_bonnet.png`: an S1 rectified frame and a frame of the run `drivejepa_014_01d503d4` (full-res replay
built from that run's `rollout.asl`), both through the adapter crop to 512×256. Raw frame: the bonnet fills the bottom ~24 of 256 input rows
(static-pixel share 0.44 at rows 224–231, 0.94–1.00 at rows 232–255). Render: no bonnet, road texture to the last row (static share ≤ 0.15
in every bottom band). So the bottom ~9 % of the training image differs systematically from what the model sees in AlpaSim.

## Steps 1–2 — what was built, and from where

Everything is in the usdz's own metric frame (`gtaas:egomotion_deepmap:v0`, origin at the first ego pose): map, agents and ego.
- **Source.** AlpaSim itself loads the map from the `clipgt/` parquet tables first and uses `map.xodr` only as a fallback
  (`alpasim/src/utils/alpasim_utils/artifact.py:187-231`), so I used the clipgt tables (the referee's own source). `map.xodr` was fetched for
  all 40 scenes but not parsed (flag F8).
- **Drivable area.** Union of every `clipgt/lane.parquet` polygon (left rail + reversed right rail) and every `intersection_area` polygon,
  morphologically closed by 0.3 m to fill hairline gaps; passed to the scorer as `DRIVABLE_AREA` polygons. Lane polygons are passed as
  `LANE`, or `LANE_CONNECTOR` when listed under `INTERSECTION_AREA_TO_LANE`; intersections as `INTERSECTION`.
- **Centrelines with direction.** Mean of the two rails, resampled; direction = rail order. Per anchor: the ego's current lane from its
  start, then `NEXT_LANE` successors preferring lanes the human took.
- **Route lanes.** Lanes the human occupies over the clip, their `ROAD_SEGMENT_SIBLING_LANE` neighbours (same-direction siblings, the
  roadblock analogue), and successors past the clip end.
- **Stop lines.** `wait_line.parquet` (categories STOP / YIELD / UNKNOWN, explicit or implicit); drawn on the PNGs, not used by any term.
- **Traffic lights.** `traffic_light.parquet` has positions only (`center`, `dimensions`, `orientation`, `category`); 15 of 40 scenes
  have any, 513 rows in total. **No light states over time exist anywhere in the usdz.** TLC therefore cannot be computed: with no red
  polygons in the observation the scorer returns 1 for every plan (`train_pdm_scorer.py:686-721`).
- **Road users.** `clipgt/obstacle.parquet` tracks (S0 cache) on the 10 Hz grid, 41 steps per anchor, as nuPlan `Agent` (vehicles,
  persons, riders) or `StaticObject`; speed below 0.5 m/s over a 1 s baseline is set to 0. `rig_trajectories.json` equals the clipgt
  egomotion to 1e-13 m after a rigid fit, so the usdz rig trajectory and the map share this frame.
- **Reference plan.** PDM-Closed needs a nuPlan lane graph and cannot run here. `MetricCache.trajectory` is filled with the **human
  path** as a stand-in (flag F1).

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


## Flags (not fixed)

- **F1 — reference plan is the human path, not PDM-Closed.** Progress is normalised by max(own, reference), so the human scores EP = 1 by
  construction and every other plan's EP is measured against the human's distance.
- **F2 — derived drivable area is lanes + intersections only.** It is narrower than the paved road (no shoulders, parking strips, car
  parks). 10 of 12 drivable-area failures in the B test, and scene `82471d1e` in the pilot, come from this.
- **F3 — lane-polygon gaps trip the direction gate at speed.** All 20 pilot A anchors with human DDC < 1 are 1–2 steps where the ego centre
  falls between lane polygons (8 steps in total); at 30 m/s that already exceeds the 6 m limit.
- **F4 — the scorer's ego box is a Pacifica**, 2.30 m wide against 2.00–2.16 m for the real cars; clearances are judged about 0.1 m tighter per side.
- **F5 — no traffic-light states**, so TLC is constant 1; red-light stops cannot be graded from NuRec data.
- **F6 — route is lane-level and same-direction only.** Passing an obstacle through the opposite lane zeroes the direction gate (8 tight
  passes). Whether upstream nuPlan roadblocks would treat such streets differently could not be checked.
- **F7 — one tight pass could not be graded** (ego not on any lane at the anchor); NuRec pull-outs: `81ce115e` scores 0 for both human and
  stay (ego stands outside the derived drivable area, and the scorer's box hits the parked car on the way out), `febd36c4` human 0.43 vs stay 0.50.
- **F8 — `map.xodr` not used** although the brief named it; the clipgt tables are what AlpaSim loads and carry lanes, intersections and stop lines.
- **F9 — agents are taken at 10 Hz directly**; upstream samples them at 2 Hz and interpolates. Velocities below 0.5 m/s are zeroed so parked
  cars count as stopped tracks.
- **F10 — the model's own ranking barely agrees with this grader**: its chosen plan is the top-graded of the 32 on 46 % of anchors (median
  Spearman 0.16).
- **F11 — S0's 7 of 144 move-off disagreements:** none of the 10 pilot scenes contains a move-off, so none falls in the pilot.
- **F12 — training needs more than this cache**: upstream also loads `anchors_scores_index_v2/<token>.npy` per token; not produced here.
- **F13 — bonnet**: the bottom ~24 input rows of every raw sample show the bonnet; AlpaSim renders do not (0c).


---

# S2.1 — map fixes, re-grade, direction gate, vocabulary, bonnet mask — 2026-10-05

| part | pre-registered | result | verdict |
|---|---|---|---|
| A / B | Q1 stay-still fails the drivable-area gate on ≤ 2 % of graded anchors | 9.0 % of 388 (was 14.9 %) | **fails** |
| B | Q2 human ≥ 0.8 on ≥ 90 % of high-speed + ordinary anchors | 100 % (was 83.8 %) | **holds** |
| B | Q3 standard formula: human ≥ 0.8 on ≥ 80 % AND human > stay on ≥ 80 % of tight passes | 69.6 % and 73.9 % (unchanged) | **fails** |
| B | Q3 diagnostic, direction gate disabled | 78.3 % and 82.6 % | no verdict |
| B | P3 (informational) highway grades not flat | 18.9 % | still flat |
| C | Q4 VOCAB-CAPPED if < 1 % of anchors exceed 20 m/s | 0 of 8,192; fastest anchor 14.79 m/s | **VOCAB-CAPPED** |
| D | Q5 masked samples keep C4 in [0.8, 1.2] and C3 within 5 points | C4 0.930, C3 −2.1 points | **holds** |

**What changed and what did not.** The high-speed failures of S2 were mine, not the map's seams: the per-anchor map was cut at 120 m, which
is less than 4 s of travel above 29 m/s, so on-route lanes ran out under fast plans (S2 flag F3 was wrong). With the radius scaled to speed,
Q2 is 100 %. The road-boundary strips add 12 % drivable area but **do not rescue the tight passes**: the remaining drivable-area failures
are corners of the scorer's 2.30 m box over the mapped curb line by a median 0.33 m on narrow streets, with the box centre always inside.
The direction gate accounts for 6 of the 21 tight passes below 0.8; even with it disabled Q3 misses on the first leg (78.3 %).

No formula term or threshold was changed. Metric caches were rewritten from the standard formula only (319 pilot tokens, fixed map);
the gate-disabled numbers exist only in `btest.csv`. Env as in S2. New/changed scripts: `s2_scene.py` (`boundary_strips`,
`bridge_route_gaps`, `load_scene(fix=)`), `s2_grade.py` (bridges as on-route lanes, speed-scaled radius, `grade(diag_no_ddc=)`),
`s2_run.py` (`S2_MAPFIX`, `png`, B2 columns), `s21_maptest.py`, `s21_diag.py`, `s21_dacdiag.py`, `s21_vocab.py`, `s21_bonnet.py`,
`s21_masked_run.py`, `s21_tables.py`; `02_build_samples/s1_convert.py` gained `BONNET_MASK_ROWS` / `--bonnet-rows`. S2 outputs are kept as
`*_s2.csv`, `scene_png_s2/`. `S2_MAPFIX=0 ./run_s2.sh …` reproduces S2. Split assertion (`assert_nurec_train` / `assert_train`) in every script.
No new downloads (manifest unchanged at 740 rows); NAS now 2.9 GB with the masked S1 rerun.

## How the map was fixed

- **A1 drivable area.** Source: `clipgt/road_boundary.parquet` (curbs and road edges; the table AlpaSim loads as `RoadEdge`,
  trajdata `dataset_specific/mads/mads_utils.py:302`, `:367`). Boundaries are open polylines. Each is closed against the nearer rail of every
  lane the map ties it to (`association.parquet`, kind `ROAD_BOUNDARY_TO_LANE`): polygon = rail section + boundary section reversed. Pairs
  more than 8 m apart are skipped. The strips are unioned with the S2 lane and intersection polygons.
- **A2 lane gaps.** On-route lane polygons are unioned and morphologically closed by 0.5 m (fills gaps up to 1.0 m); what the closing adds
  within 1.5 m of the ego-centre path becomes an on-route lane polygon.

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


## Part C answers

- **C1 The vocabulary.** `navsim_v2/data/8192.npy`, shape (8192, 40, 3) float32: 8,192 anchors, 40 poses (x, y, heading) at 0.1 s over 4 s,
  rear-axle frame. Training uses every fifth pose, `poses[:, 4::5]` → 8 poses at 0.5 s (`drive_jepa_agent.py:49-50`). How it was built is
  **not in the checkout**: no script generates it, and the README (`adapters/drivejepa/src/README.md:38-47`) only names "Multimodal
  Trajectory Distillation". 1,181 anchors are near-stationary (path under 0.5 m).
- **C3 The per-token anchor index.** Two upstream scripts: `scripts/misc/calc_anchors_scores.py:11-19` scores all 8,192 anchors against
  the token's metric cache with the same `get_sub_score`; `scripts/misc/convert_to_index.py:11-25` keeps the indices whose final score is
  ≥ 0.95, sampling 256 at random if there are more. At loss time up to 4 of them per sample are drawn as extra targets
  (`drive_jepa_agent.py:234-261`). **We can generate it**: `s2_grade.grade(mc, anchors)` on our cache takes 15–25 s per token on one core.
  At highway speed it comes back **empty** (best anchor 0.72–0.75, because nothing in the vocabulary goes fast enough to keep comfort and
  progress), so those samples would train on the human target alone.
- **The 32 proposals are independent of the vocabulary at inference.** They are regressed by an MLP from 32 × 8 learned query embeddings
  plus the ego-status feature, refined over 4 stages (`drive_jepa_model.py:33`, `:62-69`; `traj_refiner.py:22-29`). The vocabulary enters
  only in training, as those pseudo-targets.

## Part D — proposed adapter change

`proposed_adapter_bonnet_mask.diff` (48 lines, **not applied**; `patch --dry-run` passes against `adapters/drivejepa014/`): adds
`BONNET_MASK_ROWS = 37` to `constants.py`, zeroes the bottom 37 rows of each frame at the end of `_pipeline_a` in `planner.py`, and sets
`ADAPTER_VERSION = "0.1.5"`. It is only valid with a checkpoint trained on masked samples.

## Flags (S2.1, not fixed)

- **G1 — Q1 fails for a geometric reason, not a missing area.** 36 stay-still anchors in 11 narrow-street scenes: centre inside, a box corner
  over the curb line. Only 6 of the 17 that start outside would fit with a 2.0 m box. The rest need either a tolerance on the curb line or
  acceptance that the mapped curb and the label disagree by a few decimetres.
- **G2 — S2's high-speed result was a bug of mine** (120 m map radius). S2 flag F3 and the S2 P2 number (83.8 %) should be read as superseded.
- **G3 — "stationary obstacle" is not stable.** 4 of the 8 gate-hit passes have an obstacle that moves within the graded 4 s, although S0
  selected it as stationary over its own window.
- **G4 — direction gate, for Sid's decision.** It zeroes passes that use the opposite lane for 0.6–4.1 s at 2.5–10.6 m/s. One of the two NuRec
  pull-outs (`febd36c4`) goes from 0.43 to 0.85 without it and then beats stay-still; the other fails on collision and drivable area regardless.
- **G5 — lane keeping** still fails on 46 % of tight passes (15 of the 21 below 0.8 carry it) but is never the only failing term.
- **G6 — vocabulary cap matches MODE1.** The fastest anchor (14.8 m/s) is the ceiling MODE1 measured in the menu (~15 m/s). Pseudo-targets
  cannot teach highway speeds; the high-speed anchor index is empty.
- **G7 — bonnet N rests on 29 clips.** One detection failed (`f1fd39ee`); the mask covers 14.5 % of the image height. Masking moves the
  shipped model's choice (same proposal on only 48 % of anchors, 0.9 m/s mean change in end speed) although C3 and C4 hold.
- **G8 — version number.** `adapters/drivejepa015/` already declares `ADAPTER_VERSION = "0.1.5"` (the S4 reranker adapter). The patch uses
  0.1.5 as the brief says; the number needs to change or the two need to be reconciled before anything is applied.
- **G9 — S1 converter default changed.** `s1_convert.py` now masks by default; the S1 samples on the NAS (and S1's C1 check) are unmasked.
  The S2 pilot `pad_feature.gz` files are also still unmasked; only the 640 masked S1 samples under `03_grade_candidates/s1_masked/` carry the mask.
- **G10 — bridged "gaps" are mostly seams between neighbouring lanes** (median 0.14 m wide, 35 m long), not end-to-end gaps; they did not
  drive any result once the radius was fixed.
