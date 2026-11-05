# T01 Part A — inventory (paths relative to `paddock-sid/Alpasim/` unless absolute)

Per-run rows for all 90 Drive-JEPA runs: `inventory.csv` (folder, run_time, completion, logged adapter version / planner / weights sha, simulator settings). Config comparison: `config_diff.txt`.

## A1 — post-retrain set (0.1.6)
- 30 folders `runs/drivejepa/drivejepa_016_<scene8>/`, one per scene of `analysis_3.0/Retrain-data/07_closed_loop_test/tables/scenes_to_run.txt` (10 pilot10 + 20 ext20), no duplicates. `run_metadata.yaml` run_time 2026-10-08 00:03:10 → 04:44:07.
- Each: one rollout dir, `_complete` present, one `=== START` and one `RUN END rc=0` in `run_ext20_scene.log`, `ext20_gates.json` ok.
- Recorded by each run: `driver.log:1` `adapter_version=0.1.6 version_id=drivejepa-v2-driver-0.1.6 planner_module=.../adapters/drivejepa016/planner.py`; `driver.log:2` "S4 pass-9 weights, bonnet mask 37 rows, acceleration zeroed in the model input and in the §3.5 re-score | switch=on"; `run_ext20_scene.log:7` `prov OK: ... build 0.1.6 37 True True 8a71c3c0…acdd2` (30 of 30); `rollout.asl` egodriver `drivejepa-v2-driver-0.1.6` (30 of 30).
- The weights PATH is not printed in any run log. It is fixed in `adapters/drivejepa016/constants.py:45-47`, and `bridge.py` `_verify_checkpoint` refuses to load a file whose size or sha256 differs. The live package files hash equal to `07_closed_loop_test/tables/build_016.sha256`.
- sha256 recomputed now for `{runs}/drivejepa_retrain/06_first_training_run/candidate/drive_jepa_s4_pass09.ckpt` (1,273,002,795 bytes): `8a71c3c0d84a9758de46c7b46eaf18556a62ac4837feea7271040e7a0caacdd2` — equal to the logged value.
- Other folders with 0.1.6 names: `runs/drivejepa/_attic_016_launch1_20261007T2332/` (30 folders, 53 files, 0 `rollout.asl`: nothing simulated) and `runs/drivejepa/_preflight_dryrun/drivejepa_016_*` (0 `rollout.asl`). No rerun found under `runs/`, the HDD run roots, `analysis_3.0/` or `drafts/`.

## A2 — what 0.1.6 served (adapter code + as-built diff)
- Retrained weights: yes (`constants.py:45-47`, `bridge.py` passes `checkpoint_path=K016.CHECKPOINT_PATH`).
- Bonnet mask: yes, bottom 37 image rows set to 0 (`constants.py:53`, `planner.py:276-280`).
- Acceleration zeroed in the model input: yes, ego_status slots 5:7 (`constants.py:57-58`, `planner.py:727,733-744`).
- Acceleration zeroed in the momentum (§3.5) re-score: yes (`constants.py:63`, `planner.py:937-947`; `momentum.py:111` reads it).
- Nothing else differs from 0.1.4b: `momentum.py`, `divider.py`, `real_model.py` are byte-identical; other changes are the version string, the provenance line and import names.
- Planned (`06_first_training_run/adapter_0.1.6.diff`) vs as built (`07_closed_loop_test/adapter_0.1.6_as_built.diff`): the planned diff had no `ZERO_MOMENTUM_ACCELERATION`; with `ZERO_ACCELERATION` on it passed the simulator-reported `linear_acceleration` to the re-score. As built adds `ZERO_MOMENTUM_ACCELERATION = True` and passes zeros. All other hunks are the same.

## A3 — 0.1.4 vs 0.1.4b, and the "before" set
- 0.1.4b = 0.1.4 plus one behaviour: the plan chosen at the previous replan is handed to the model so the vendored `calibrate_score` (paper §3.5 momentum-aware selection) runs before the argmax; previous plan dropped when older than 0.55 s; switch `DRIVEJEPA_MOMENTUM` (on in all 30 runs). Files: new `momentum.py`; `planner.py`, `constants.py`, `bridge.py` changed; `divider.py` identical. Same shipped checkpoint (`real_model.py` DEFAULT_CHECKPOINT `drive_jepa_perception_based_agent_vitl_v2.ckpt`; neither set logs a weights hash).
- 0.1.4 set: `runs/drivejepa/drivejepa_014_<scene8>/` (10, 2026-09-09 17:43 → 22:40) and `runs/drivejepa/drivejepa_014_ext20/drivejepa_014_ext20_<scene8>/` (20, 2026-09-22 21:33 → 2026-09-23 01:09). One (`…ext20_e3ed554f`) has wizard rc=1: simulation complete, aggregate recovered by CLI per its `ext20_gates.json`.
- 0.1.4b set: `{runs}/alpasim_bulk/baseline_014b/runs/b014b_<scene8>_on/` (30, 2026-10-01 22:21 → 2026-10-02 05:37; reached via `analysis_3.0/Re-ranker/baseline_014b/hdd`; `baseline_014b/runs/` in the repo is empty). 22 logs show 2–3 `START` lines: the earlier ones are preflight refusals ("root free 14 GiB < 15"); each folder has one rollout and one `RUN END rc=0`.
- Before set used for the like-for-like comparison: **0.1.4b** — 0.1.6 is built from it and the same runner, launcher pattern and switch are used. 0.1.4 is carried as a separate column.
- Same in before (0.1.4b) and after: `wizard-config.yaml` identical on 30 of 30 scenes after normalising run-dir paths (0 differing keys; also 0 against 0.1.4): `control_timestep_us 50000`, `force_gt_duration_us 3000000`, `skip_driver_during_force_gt false`, `n_sim_steps 400` (396/397/395 executed, same per scene), `base_image alpasim-base:0.134.0`, NRE `26.1.112-59673f88` in every `rollout.asl`, scene csv sha `94f50aa3` / `1b778280`, driver `--device cpu` (`run_014b_scene.sh:232`, `run_016_scene.sh:231`), driver port 9147.
- Differences that remain: (1) `alpasim-base:0.134.0` was rebuilt on 2026-10-07 (S5 flag 1; the image id is not recorded in the run folders, so I could not verify it); (2) machine load (S5 flag 4/6; not verified here); (3) driver-load timeout 600 s → 3600 s (`tables/runner_diff.txt`); (4) GPU-free wait floor 14,394 MiB for 0.1.4b runs 1–10, 8,000 MiB afterwards and for all 0.1.6 runs; (5) run dirs on the HDD (0.1.4b) vs under `runs/drivejepa` (0.1.6); (6) build check against a NAS manifest (0.1.4b) vs frozen hashes (0.1.6). One run per scene on both sides; no repeat exists to size run-to-run variation.

## A4 — no 0.1.5 run in either set
`rollout.asl` egodriver is `drivejepa-v2-driver-0.1.4` / `-0.1.4b` / `-0.1.6` on 30 of 30 in each set, and each `driver.log` names its own package's `planner.py`. The 0.1.5 runs live under `analysis_3.0/Re-ranker/s5_closed_loop/` and are not read.

## Constant-velocity floor
`runs/baselines/cv_023_20hz_pilot10/` (one run, 10 scenes, 2026-09-11) and `runs/baselines/cv_023_20hz_ext20/cv_023_20hz_ext20_<scene8>/` (20, 2026-09-23). Same 20 Hz step, force_gt 3 s, 400 steps, but `skip_driver_during_force_gt: true` (Drive-JEPA: false) — reference only.

## How the metrics are stored
- `rollouts/clipgt-*/<uuid>/metrics.parquet`: long table, one row per metric per tick (`name, timestamps_us, values, time_aggregation`), ~398 ticks at 20 Hz covering the whole clip including the 3 s replay. `dist_traveled_m` and `progress_rel_to_total` are cumulative from the clip start.
- `aggregate/metrics_results.parquet`: one row per run. The aggregator drops ticks before the first `eval_relevant` tick, then keeps ticks up to and including the first with `dist_to_gt_trajectory ≥ 4.0` (`eval/aggregation/main.py:135-140`, `eval-config.yaml` `max_dist_to_gt_trajectory`) or `collision_any | offroad` (`processing.py:39-66`), then takes max (events) or last (progress, distance). `collision_at_fault` = front or lateral. `duration_frac_20s × 20` = window end in seconds after handover.
- Handover tick = first `eval_relevant` tick = `render_start_us + force_gt_duration_us` (3.0 s) on 120 of 120 runs, and = first driver query + 3.0 s on the 90 Drive-JEPA runs.
- Steering: `controller/alpasim_controller_*.csv` `front_steering_angle`. Driven plan: regenerated menus (`baseline_014b/menus`, `07_closed_loop_test/menus`; chosen plan within 0.02 mm of the logged one); `step1_decel[chosen]` is the demanded acceleration over the plan's first 0.5 s; lateral offsets are the chosen plan's y at 1 s and 2 s in the ego frame at plan time (+ left). Menus do not exist for 0.1.4, so its decel column is empty.
