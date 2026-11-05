# S5 — closed-loop trial of the retrained model (adapter 0.1.6) on the 30 census scenes

**VERDICT: MIXED — C1 and C5 fail; C2 and C3 hold.** Pooled score over the 30 scenes rises from 0.543 (0.1.4b, recorded) to
0.703 (0.1.6). Two of 0.1.4b's four at-fault collisions are gone and none is new. Neither highway scene is fixed and neither pull-out
moves.

**Review page (Tailscale only): http://{site-host}:8778/** — tmux `s5_review`. Left = recorded 0.1.4b, right = 0.1.6.

Runs 2026-10-07 23:45 → 2026-10-08 04:49, one run per scene, 30 of 30 complete with every gate green.

## Pre-registered

| id | statement | result | verdict |
|---|---|---|---|
| C1 | highway: 04749bb9 and acd22430 no longer leave the road in the first second; chosen speed ≥ 0.8 × human for the first 3 s | 04749bb9 off the road at +2.7 s (was +0.4 s), acd22430 at +0.25 s (was +0.15 s); chosen plan ≥ 0.8 × human on 2 of 7 replans in each | **fails** |
| C2 | no new at-fault collision among the 13 full-score scenes | 0 new at-fault collisions, on these 13 and on all 30 | holds |
| C3 | pooled score over 30 ≥ 0.1.4b's | 0.703 against 0.543 | holds |
| C4 | pull-outs d2715023, b44a4a44: report only | neither moves (1.3 m and 0.2 m in 16.8 s) | reported |
| C5 | peak steer ≤ 0.3 rad on scenes where 0.1.4b stayed below it | 19 of 20 hold (highest 0.21 rad); d2715023 0.20 → 0.32 rad | **fails** |

- **C5 fails by the rule, on a scene that is nearly stationary.** d2715023's mean speed is 0.07 m/s; the 0.32 rad is the wheels
  turning while the car creeps 1.3 m. It is not a swerve from a too-fast pick, which is what C5 was written to catch. Reported as a fail.
- **"The 13 full-score scenes" is taken as the 13 highest-scoring under 0.1.4b** (0.86 to 1.00; the 14th is 0.65). C2 holds on any
  reading, since no scene gains an at-fault collision. Four of the 13 lose score through progress (see Worse).

## Totals

| | 0.1.4b (recorded) | 0.1.6 |
|---|---|---|
| Mean score, 30 scenes | 0.543 | 0.703 |
| Improved / worse / unchanged (±0.02) | — | 10 / 5 / 15 |
| Scenes at full score | 10 | 15 |
| At-fault collisions | 4 (04394343, 8023dce5, b1b170d0, b44a4a44) | 2 (8023dce5, b44a4a44) |
| Any collision | 10 | 7 |
| Off-road exits | 4 | 3 (04749bb9, 791d68ce, acd22430) |
| Mean progress | 0.537 | 0.698 |
| Mean distance travelled | 75.8 m | 115.4 m |
| Peak demanded deceleration of the driven plan, median | 7.5 m/s² | 1.4 m/s² |
| Scenes with peak steer above 0.3 rad | 10 | 7 |
| Stalls (S5 rule) | 3 | 4 (791d68ce, 8023dce5, bae463d3, bd07fb91) |
| Replans where §3.5 changed the pick | 366 of 1,170 (31 %) | 479 of 1,170 (41 %) |

## Step 1 — adapter 0.1.6 and the setup proof

- **Package** `adapters/drivejepa016/` = copy of `adapters/drivejepa014b/` + `06_first_training_run/adapter_0.1.6.diff` + the momentum zeroing.
  `momentum.py` and `divider.py` are byte-identical to 0.1.4b's. As-built diff: `adapter_0.1.6_as_built.diff` (188 lines). Launcher draft
  `drafts/drivejepa-0.1.6/serve_drivejepa_0_1_6.py` (0.1.4b's with names changed, nothing else).
- **Weights** `{runs}/drivejepa_retrain/06_first_training_run/candidate/drive_jepa_s4_pass09.ckpt`, sha256 `8a71c3c0…acdd2` verified before the build, by
  the sweep at start, and by the bridge worker at every driver start (`bridge.py:121-133`, called at `:331`).
- **Bonnet mask** 37 rows: `planner.py:276-280`. **Model-input zeroing**: `planner.py:733-744` (`_model_ego_status`, the switch test at `:740`), applied at `:727`.
  **Momentum re-score zeroing**: `planner.py:939-956` fills `ego_acceleration` with zeros; `momentum.py:111` builds the re-score's ego
  state from it. Switch values: `constants.py:53,57,63`.
- **Momentum code path**, unchanged from 0.1.4b: `planner.py` `_momentum_request` → feature key `momentum` → `momentum.py:97-153`
  (`ego_state`, `previous`, `simulate_chosen`) → vendored `calibrate_score` (`drive_jepa_model.py:87-88,98-151`).
- **Proof on recorded 0.1.4b runs** (`tables/step1_proof.json`): the 0.1.6 planner with shipped weights and the three switches off
  reproduces the recorded pick, pick-without-momentum and momentum code on 117 of 117 replans (02e075b9, 16b83346, 04749bb9; plans within
  0.016 mm), including the three named ticks. With the switches on, both input images equal the S4 training loader's output bit for bit
  on 3 of 3, the acceleration fed to the model and to the re-score is zero on 3 of 3, and every other input is unchanged.
- **Menu check on the new runs**: regenerating all 1,020 post-warm-up menus offline reproduces the logged 0.1.6 pick on 30 of 30 scenes, 0 mismatches.
- The older adapter packages (`drivejepa`, `drivejepa014`, `drivejepa014b`, `drivejepa015`) hash the same before and after.

## Step 2 — runs

- **Config** as the 0.1.4b campaign: `driver=drivejepa014` (`drafts/drivejepa-0.1.4/alpasim_configs/driver/drivejepa014.yaml`:
  `control_timestep_us: 50_000`, `force_gt_duration_us: 3_000_000`, `skip_driver_during_force_gt: false`, `n_sim_steps: 400`), driver on
  port 9147, CPU. Scene files and pins: pilot10 csv `94f50aa3…`, ext20 csv `1b778280…`; NRE `26.1.112-59673f88` on 30 of 30.
  Runner `run_016_scene.sh` (copy of 0.1.4b's; `tables/runner_diff.txt`, 71 lines); GPU floor 8,000 MiB as 0.1.4b's runs 11–30.
- **Outputs** per scene under `runs/drivejepa/drivejepa_016_<scene>/` (4.9 GB in all): rollout metrics parquet, annotated H.264 replay and
  HTML replay (`tools/make_replays.py`; collected in `07_closed_loop_test/replays/`), and the proposal-overlay video (`07_closed_loop_test/videos/d016_<scene>.mp4`).
- **Provenance** correct on 30 of 30 (`adapter_version=0.1.6`, planner under `adapters/drivejepa016/`, the 0.1.6 line with `switch=on`).

## Step 3 — per scene, 0.1.4b → 0.1.6

| scene | group | score | progress | mean speed, m/s | collision | first off-road (time off) | peak steer, rad | peak demanded decel, m/s² | first-plan speed ratio | §3.5 changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 04394343 | improved | 0.00 → 1.00 | 0.19 → 0.89 | 7.3 → 13.1 | at fault +2.1 s → none | 10.8 s (1.5 s) → — | 0.79 → 0.02 | 18.2 → 1.0 | 0.62 → 0.97 | 8 → 15 |
| b1b170d0 | improved | 0.00 → 1.00 | 0.18 → 0.86 | 16.4 → 13.8 | at fault +0.4 s → none | 1.0 s (7.4 s) → — | 0.78 → 0.02 | 20.7 → 1.0 | 0.82 → 0.97 | 14 → 11 |
| bd07fb91 | improved | 0.00 → 1.00 | 0.36 → 0.85 | 14.7 → 15.2 | none → not at fault +7.2 s | 0.7 s (16.2 s) → — | 0.77 → 0.78 | 13.7 → 9.1 | 0.82 → 0.99 | 2 → 13 |
| 032b6f21 | improved | 0.41 → 1.00 | 0.33 → 0.94 | 8.2 → 9.9 | not at fault +4.5 s → none | — → — | 0.76 → 0.01 | 13.2 → 1.3 | 0.99 → 0.97 | 17 → 21 |
| 89f2eeb8 | improved | 0.37 → 0.90 | 0.29 → 0.72 | 5.0 → 11.5 | not at fault +8.5 s → not at fault +14.8 s | — → — | 0.12 → 0.01 | 9.5 → 1.6 | 0.88 → 0.94 | 11 → 8 |
| 71d7d03b | improved | 0.55 → 1.00 | 0.44 → 0.98 | 8.0 → 14.3 | none → none | — → — | 0.19 → 0.01 | 5.4 → 1.0 | 0.92 → 0.98 | 9 → 14 |
| 2e1ce314 | improved | 0.59 → 1.00 | 0.48 → 0.99 | 10.4 → 14.3 | not at fault +8.9 s → none | — → — | 0.02 → 0.01 | 4.8 → 1.0 | 0.87 → 0.99 | 18 → 19 |
| d1a25a99 | improved | 0.33 → 0.67 | 0.26 → 0.54 | 1.6 → 4.5 | none → not at fault +10.4 s | — → — | 0.47 → 0.31 | 10.0 → 1.3 | 0.01 → 0.95 | 9 → 19 |
| 98dd56b2 | improved | 0.16 → 0.34 | 0.13 → 0.27 | 3.7 → 5.8 | not at fault +5.5 s → not at fault +7.8 s | — → — | 0.06 → 0.00 | 9.6 → 1.5 | 1.21 → 1.08 | 12 → 22 |
| 5f064b02 | improved | 0.65 → 0.69 | 0.52 → 0.55 | 4.1 → 4.3 | not at fault +14.2 s → not at fault +14.6 s | — → — | 0.11 → 0.01 | 7.7 → 1.4 | — → — | 15 → 31 |
| e121e37d | worse | 0.91 → 0.68 | 0.73 → 0.54 | 9.4 → 11.1 | none → none | — → — | 0.06 → 0.03 | 4.9 → 1.2 | 0.87 → 0.95 | 17 → 19 |
| 0245ff75 | worse | 0.86 → 0.70 | 0.69 → 0.56 | 8.0 → 7.5 | not at fault +12.9 s → not at fault +11.0 s | — → — | 0.01 → 0.00 | 4.6 → 1.5 | 0.88 → 1.01 | 22 → 20 |
| bae463d3 | worse | 0.46 → 0.30 | 0.37 → 0.24 | 2.2 → 1.4 | none → none | — → — | 0.60 → 0.62 | 3.6 → 0.9 | 0.71 → 1.10 | 16 → 17 |
| 1acf2f5d | worse | 0.99 → 0.86 | 0.79 → 0.69 | 4.7 → 4.0 | none → none | — → — | 0.10 → 0.11 | 7.2 → 2.8 | 0.96 → 0.93 | 10 → 22 |
| cdd9448c | worse | 1.00 → 0.93 | 0.80 → 0.74 | 4.2 → 4.9 | none → none | — → — | 0.36 → 0.27 | 2.5 → 1.4 | 0.97 → 1.07 | 16 → 21 |
| 01d503d4 | full score under both | 1.00 → 1.00 | 1.00 → 1.00 | 5.9 → 4.8 | none → none | — → — | 0.07 → 0.00 | 4.6 → 3.8 | 0.83 → 0.93 | 10 → 14 |
| 023b7fcc | full score under both | 1.00 → 1.00 | 0.87 → 1.00 | 4.6 → 5.7 | none → none | — → — | 0.13 → 0.01 | 3.8 → 1.1 | 0.96 → 1.02 | 12 → 6 |
| 026d6a39 | full score under both | 1.00 → 1.00 | 0.84 → 1.00 | 5.0 → 9.5 | none → none | — → — | 0.05 → 0.03 | 4.2 → 1.2 | 1.04 → 0.99 | 14 → 11 |
| 02e075b9 | full score under both | 1.00 → 1.00 | 1.00 → 1.00 | 8.9 → 8.5 | none → none | — → — | 0.00 → 0.00 | 2.0 → 5.4 | 0.96 → 0.67 | 19 → 25 |
| 02eadd92 | full score under both | 1.00 → 1.00 | 0.82 → 1.00 | 3.0 → 5.8 | none → none | — → — | 0.04 → 0.00 | 10.6 → 2.7 | 0.99 → 0.99 | 23 → 20 |
| 048b974e | full score under both | 1.00 → 1.00 | 0.84 → 1.00 | 4.6 → 7.6 | none → none | — → — | 0.07 → 0.01 | 7.7 → 1.4 | 0.95 → 0.97 | 15 → 23 |
| 16b83346 | full score under both | 1.00 → 1.00 | 1.00 → 1.00 | 4.8 → 6.0 | none → none | — → — | 0.09 → 0.09 | 4.0 → 6.9 | 1.01 → 0.96 | 19 → 13 |
| 6725df2d | full score under both | 1.00 → 1.00 | 0.89 → 1.00 | 4.5 → 6.5 | none → none | — → — | 0.11 → 0.00 | 2.9 → 1.5 | 0.96 → 0.94 | 18 → 30 |
| e3ed554f | full score under both | 1.00 → 1.00 | 1.00 → 1.00 | 1.0 → 0.8 | none → none | — → — | 0.02 → 0.02 | 8.8 → 4.4 | — → — | 6 → 15 |
| 04749bb9 | leaves the road early under both | 0.00 → 0.00 | 0.18 → 0.28 | 22.7 → 20.5 | none → none | 0.4 s (0.5 s) → 2.7 s (0.3 s) | 0.78 → 0.78 | 30.9 → 14.1 | 0.70 → 0.86 | 7 → 9 |
| acd22430 | leaves the road early under both | 0.00 → 0.00 | 0.17 → 0.17 | 23.3 → 23.4 | none → none | 0.1 s (0.3 s) → 0.2 s (0.4 s) | 0.78 → 0.78 | 28.4 → 12.0 | 0.63 → 0.88 | 0 → 10 |
| 791d68ce | leaves the road early under both | 0.00 → 0.00 | 0.15 → 0.15 | 2.0 → 2.0 | not at fault +10.0 s → not at fault +12.5 s | 0.0 s (2.9 s) → 0.0 s (3.6 s) | 0.33 → 0.45 | 8.0 → 4.1 | 0.95 → 1.02 | 15 → 19 |
| 8023dce5 | same at-fault collision | 0.00 → 0.00 | 0.75 → 0.94 | 2.7 → 2.1 | at fault +4.1 s → at fault +7.2 s | — → — | 0.16 → 0.21 | 9.0 → 2.3 | 0.94 → 0.98 | 11 → 9 |
| b44a4a44 | same at-fault collision | 0.00 → 0.00 | 0.00 → 0.00 | 0.0 → 0.0 | at fault +6.6 s → at fault +6.6 s | — → — | 0.00 → 0.00 | -0.1 → -0.0 | — → — | 1 → 3 |
| d2715023 | does not pull out | 0.02 → 0.03 | 0.02 → 0.03 | 0.1 → 0.1 | none → none | — → — | 0.20 → 0.32 | -0.1 → 0.3 | — → — | 0 → 0 |

Off-road column: time of the first off-road tick after handover, and total time off-road in brackets. Peak demanded deceleration is the
largest first-half-second braking asked by a driven plan. First-plan speed ratio is the handover plan's first-step speed over the car's speed.

### Improved (10)

Ten scenes. In six of them the event that ended or zeroed the 0.1.4b run does not happen under 0.1.6: two at-fault collisions, one off-road exit, two early collisions that were not the car's fault, and one departure from the human path. A seventh, 89f2eeb8, loses its stop. Six of the ten now reach full score.

- Two of 0.1.4b's four at-fault collisions are gone: 04394343 (was at +2.2 s) and b1b170d0 (was at +0.4 s). Both now run to the end of the clip at full score. No scene gains an at-fault collision.
- The hard brake after handover is gone where it mattered. Peak demanded deceleration of the driven plan falls from 18.2 to 1.0 m/s² on 04394343, from 20.7 to 1.0 on b1b170d0 and from 13.2 to 1.3 on 032b6f21, and peak steering falls from 0.76-0.79 rad to 0.01-0.02 rad on all three.
- 89f2eeb8, the no-cue stop: the stop is gone. 0.1.4b braked to 0.0 m/s by +2.5 s; 0.1.6 never drops below 9.7 m/s and scoring runs to +14.85 s instead of +8.5 s.
- bd07fb91 reaches full score without leaving the road, but it still shows a full steering swing (0.78 rad) at +6.5 s, the moment scoring ends; a collision follows at +7.15 s. The score does not see either.
- Three gains are partial: 5f064b02 (0.65 to 0.69) and 98dd56b2 (0.16 to 0.34) still end on a collision that is not the car's fault, later than before; d1a25a99 (0.33 to 0.67) still ends early, at +10.45 s.

### Worse (5)

Five scenes, none with an at-fault collision or an off-road exit under either version. All five lose progress, for two different reasons.

- Two leave the human path and the scoring window closes early: e121e37d is 4 m off the path at +9.65 s (it ran the full 16.8 s under 0.1.4b) and cdd9448c at +13.4 s. On e121e37d the car is faster than before (11.1 against 9.4 m/s) when it drifts.
- Three drive slower: 1acf2f5d (mean 4.0 against 4.7 m/s), bae463d3 (1.4 against 2.2 m/s, stalled under both) and 0245ff75 (7.5 against 8.0 m/s), where the collision that ends scoring, not the car's fault in either run, comes at +11.0 s instead of +12.9 s.
- Four of the five were among 0.1.4b's best scenes (0.86 to 1.00). None has a new at-fault collision.

### Unchanged (15)

Fifteen scenes. Nine are at full score under both versions. Six fail in the same way under both: three leave the road early, two have the same at-fault collision, one never pulls out.

- **Full score under both (9):** 01d503d4, 023b7fcc, 026d6a39, 02e075b9, 02eadd92, 048b974e, 16b83346, 6725df2d, e3ed554f
- **Leaves the road early under both (3):** 04749bb9, acd22430, 791d68ce
- **Same at-fault collision (2):** 8023dce5, b44a4a44
- **Does not pull out (1):** d2715023

## Scene reads

### Highway — 04749bb9 and acd22430

Neither holds 23.5 m/s and both still leave the road with a full steering swing (0.78 rad).

- **04749bb9:** car speed at +1 / +2 / +3 s is [18.4, 20.0, 21.9] m/s (0.1.4b: [15.9, 6.8, 1.6]). Off the road at +2.7 s.
- **acd22430:** [19.4, 16.3, 9.1] m/s (0.1.4b: [15.9, 6.8, 1.4]). Off the road at +0.25 s.

04749bb9 under 0.1.6, per replan:

| t after handover | car, m/s | human, m/s | chosen plan, mean over 4 s | chosen ≥ 0.8 × human | fastest of 32, mean over 4 s | demanded accel, first 0.5 s |
|---|---|---|---|---|---|---|
| +0.0 s | 23.7 | 23.6 | 18.9 | no | 67.8 | -6.7 |
| +0.5 s | 22.1 | 23.7 | 12.8 | no | 25.9 | -12.3 |
| +1.0 s | 18.2 | 23.7 | 22.1 | yes | 28.2 | -1.3 |
| +1.5 s | 18.8 | 23.8 | 18.1 | no | 27.4 | -2.1 |
| +2.0 s | 20.0 | 23.7 | 21.1 | yes | 28.0 | -0.9 |
| +2.5 s | 21.4 | 23.8 | 19.0 | no | 61.4 | -5.3 |
| +3.0 s | 21.9 | 23.9 | 8.8 | no | 23.2 | -14.1 |

acd22430 under 0.1.6, per replan:

| t after handover | car, m/s | human, m/s | chosen plan, mean over 4 s | chosen ≥ 0.8 × human | fastest of 32, mean over 4 s | demanded accel, first 0.5 s |
|---|---|---|---|---|---|---|
| +0.0 s | 23.4 | 23.3 | 20.4 | yes | 67.1 | -5.4 |
| +0.5 s | 22.4 | 23.3 | 15.1 | no | 62.8 | -12.0 |
| +1.0 s | 19.2 | 23.2 | 20.6 | yes | 26.8 | -1.6 |
| +1.5 s | 18.1 | 23.2 | 12.2 | no | 27.4 | -8.2 |
| +2.0 s | 16.1 | 23.4 | 9.8 | no | 27.0 | -6.8 |
| +2.5 s | 12.3 | 23.4 | 6.6 | no | 15.5 | -3.8 |
| +3.0 s | 8.9 | 23.4 | 4.8 | no | 14.8 | -2.2 |

- The handover plan now asks for 5–7 m/s² of braking (0.1.4b: 14–17), but the plan half a second later asks for 12 m/s² on both scenes.
  That is above the roughly 6 m/s² at which the controller was earlier found to swerve under a hard brake demand.
- The scorer keeps choosing plans slower than the human while much faster ones are on the menu.
- **The menu contains absurd plans:** the fastest of the 32 averages 61–68 m/s on some replans (human 23.5). None was chosen here.

### Pull-outs — d2715023 and b44a4a44

Neither moves. Departing plans (end ≥ 2 m away) are on the menu at every replan, and the scorer never picks one.

| scene | version | departing plans of 32, median | best rank of a departing plan, median (best) | score of the pick | best score of a departing plan | departing plan picked |
|---|---|---|---|---|---|---|
| d2715023 | 0.1.4b | 19 | 11.5 (10) | 0.92 | 0.36 | 0 of 34 |
| d2715023 | 0.1.6 | 21 | 12.0 (10) | 0.95 | 0.38 | 0 of 34 |
| b44a4a44 | 0.1.4b | 25 | 8.0 (4) | 0.72 | 0.49 | 0 of 34 |
| b44a4a44 | 0.1.6 | 24 | 9.0 (7) | 0.93 | 0.85 | 0 of 34 |

On b44a4a44 the best departing plan's score rose from 0.49 to 0.85, but the stay-put pick rose to 0.93. b44a4a44 is struck at +6.6 s as before.

### No-cue stop — 89f2eeb8

The stop is gone: lowest speed 9.7 m/s against 0.0. Under 0.1.4b the chosen plan's mean speed fell to 0.5 m/s by
+1.5 s with a 9.5 m/s² brake demand; under 0.1.6 it stays at 11–14 m/s and never asks for more than 1.6 m/s². The run cannot say whether
the weights or the zeroed acceleration removed it, since both changed together.

### The 13 best scenes under 0.1.4b

Nine are unchanged at full score. Four lose score, all through progress, none with an at-fault collision or an off-road exit:
cdd9448c 1.00 → 0.93, 1acf2f5d 0.99 → 0.86, e121e37d 0.91 → 0.68, 0245ff75 0.86 → 0.70. e121e37d and cdd9448c leave the human path
(4 m off at +9.65 s and +13.4 s); 1acf2f5d and 0245ff75 drive slower.

## GPU and time

- Sweep wall time 5 h 04 min for 30 runs; simulation per run 4.3–60 min, median 5.4 min (run 1 includes the 23-minute image rebuild).
- The driver ran on CPU as in the 0.1.4b campaign (load time 8 s to 236 s per run). The GPU served rendering only.
- Whole-card memory during our runs: 35–97 GiB used, shared with other tenants; the per-run rise was 5–17 GiB typically
  (`tables/gpu_time.csv`). No run failed for memory.

## Flags

1. **Base image rebuilt.** `alpasim-base:0.134.0` was no longer on the machine. The first launch (23:15) failed rebuilding it on a full
   root disk (0 GiB free at 23:32) and the 29 scenes behind it were refused by the disk check; nothing was simulated. The relaunch rebuilt it
   (ID `1e0d5fea6ae1`, 35.8 GB, same Dockerfile, today's system packages). The 0.1.4b runs used the earlier build. Failed launch kept in
   `runs/drivejepa/_attic_016_launch1_20261007T2332/`.
2. **Copied from `drivejepa014b`, not `drivejepa014`** as the brief's rule says, because 0.1.6 is defined as 0.1.4b plus changes.
3. **Launcher written under `drafts/drivejepa-0.1.6/`**, following the 0.1.4b pattern; the brief names only `adapters/` and `runs/`.
4. **Runner changes beyond names:** driver-load timeout 600 s → 3,600 s (the machine was at a load of ~100); the build check compares
   against hashes frozen after Step 1 instead of a NAS archive (no 0.1.6 archive exists); the queue stops on a hard failure.
5. **One run per scene, no repeats:** differences of a few hundredths are within what a rerun could move; no seed variance was measured.
6. **0.1.4b is the recorded campaign**, on the older image build and a quieter machine. Scene start times and config are the same.
7. **§3.5 changes the pick more often under 0.1.6** (41 % of replans against 31 %); on 5f064b02 and 6725df2d it does so on 30–31 of 39 replans.
8. **bd07fb91** scores 1.00 with a full steering swing at +6.5 s and a collision (not at fault) at +7.15 s, after scoring has ended.
9. **Not tested:** which of the three changes (weights, mask, zeroed acceleration) produced which effect.
10. No adapter test file was written for 0.1.6; the Step 1 proof and the 30-scene menu check stand in for it.
