# Arm (c) check — has Drive-JEPA 0.1.8b been driven on all 30 AlpaSim scenes?

2026-11-07, shakaal. Read-only; CPU only (`taskset -c 0-1 nice -n 19`). No runs, no GPU, no docker. The only write is this file.

Path shorthands: `G` = `Gallops/Gallops-sid/LORA Refiner/Generator-Joint-PAI-alpasim`, `AS` = `Paddock/paddock-sid/Alpasim`, `NAS018b` = `{retrain_store}/018b`.

## Answer

**No. 0.1.8b-p06 has been driven in closed loop on 12 of the 30 scenes, once each, natively, on 2026-10-31. No later run exists for any 0.1.8b snapshot.** The same 12 scenes, and only those, exist for p02 and p10. The 30-scene mean and the gate10 mean for 0.1.8b are therefore **not available**.

What Sid is most likely remembering is the **0.1.8** 30-scene run of 2026-10-27 (checkpoint `drive_jepa_pai_joint_p04.ckpt`, sha `b6686d8c…`), mean 0.7350 vs 0.1.7's 0.7697. That is a different model from 0.1.8b (sha `3d566eb2…`), which was exported four days later.

## 1. Starting facts, verified

| fact | evidence |
|---|---|
| p06 checkpoint path and sha `3d566eb211b5…8e66`, 1,273,003,357 bytes | every p06 run's worker log: `NAS018b/closed_loop/runs/drivejepa_018bp06_8023dce5/driver_worker.log` line "drivejepa-0.1.8b-p06 checkpoint: verified {retrain_store}/018b/release/drive_jepa_018b_p06.ckpt sha256 3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66 bytes 1273003357 source pinned"; also `G/s10_018b_closed_loop/logs/run_tags.log:7` |
| sister snapshots p02 `7469b45d…`, p10 `ff50947e…` | `NAS018b/release/` holds `drive_jepa_018b_{p02,p06,p10,step0}.ckpt`; p02/p10 worker logs cite those shas (table in §3) |
| the 30 scenes | `AS/analysis_3.0/Retrain-data/s0_census/b2_our30.csv`: 30 rows + header (`wc -l` = 31); lanes: 10 `pilot10`, 20 `ext20` (`cut -d, -f2 \| sort \| uniq -c`) |
| the 12 diagnosis scenes | `G/s10_018b_closed_loop/tables/scenes_12.txt` (12 lines; the brief's path lacks the leading `Gallops/`) — e3ed554f 8023dce5 5f064b02 e121e37d 048b974e cdd9448c d2715023 bd07fb91 bae463d3 d1a25a99 b1b170d0 04749bb9; two of them (048b974e, 04749bb9) are pilot10 scenes, the other 10 are ext20 (`grep` of each id in `b2_our30.csv`) |
| native adapter | `AS/adapters/drivejepa018bp06/` (bridge.py, constants.py, planner.py dated 2026-10-31 03:53) |
| run script | `AS/drafts/drivejepa-0.1.8b-p06/run_018bp06_scene.sh` (the brief put it under the adapter folder; it is in `drafts/`). Its output root is hard-wired: `OUTROOT={retrain_store}/018b/closed_loop` and `S5A=$OUTROOT/runs` (`run_018bp06_scene.sh:50-51`), so every run this script made is in the known run folder |

## 2. Where I searched

| search | command (pinned, nice 19) | result |
|---|---|---|
| known run folder | `ls NAS018b/closed_loop/runs` | 36 run dirs + 36 `.preflight_only_*` dirs: 12 scenes × {p02, p06, p10}; the scene set is identical for the three snapshots |
| local trees by name | `find {checkouts} {home}/data {alpasim_bulk} -maxdepth 8 -iname '*018b*' -o '*0.1.8b*' -o '*drivejepa018b*' -o '*3d566eb2*' -o '*7469b45d*' -o '*ff50947e*' -o 'drive_jepa_018b*'` (prunes .git, venvs) | only the build/adapter/draft folders, `G/s7…s11` work folders, and `Gallops-sid/memory/...` — no run folder outside the known one |
| local trees by sha prefix | `grep -rl '3d566eb2\|7469b45d\|ff50947e'` over `Paddock Paddock-gate paddock-memory Paddock-retrained-adapter Gallops {home}/data {alpasim_bulk}` (log/yaml/json/md/txt/csv/sh/toml) | 35 files, all in `AS/drafts/drivejepa-0.1.8b-*`, `G/s10_018b_closed_loop/`, `G/CLOSING_REPORT.md`, and the approach-3 memory folder. No run log outside s10 names a 0.1.8b sha |
| Paddock run stores by date | `ls <worktree>/results/alpasim/<model>/` for all 17 `Paddock*`/`paddock-memory` worktrees, names ≥ `20261031` | only `paddock-memory/results/alpasim/drivejepa/`: 5 runs dated 2026-11-05/06 (see §4) |
| NAS, my trees | `ls` of `{nas_store}/{alpasim/paddock-runs, gallops-runs, tmp, workspace-siddesh, checkpoints, memory_cache, paddock-h15-runs}`, `{retrain_store}/*`, `{dataset_store}/` | the only closed-loop roots are `generator_joint_pai/{closed_loop (0.1.8, 30 scenes, 2026-10-27), diagnosis (0.1.8-rs), graft (0.1.8-p9), 018b/closed_loop (0.1.8b), 018c/closed_loop (0.1.8c-p04, 12 scenes, 2026-11-03)}` and `paddock-h15-runs/alpasim/drivejepa` (copies of the 5 November bridge runs + a symlink to one 2026-10-27 run) |
| NAS, shared Paddock store | `ls {paddock_site_nas}/runs/alpasim/drivejepa` | 3 runs, all 2026-10-27/28, `scene_set public_2601_gate10`, checkpoints `<site path:a90f1310>` and `<site path:52e100ac>` (`run_config.json`); dated before p06 existed (exported 2026-10-31 03:53) |
| NAS, full name search | `find {nas_store} {dataset_store} -maxdepth 6 -iname …` | **NOT CHECKED to completion**: killed at 840 s with no hits printed (estimate: > 30 min on this NAS). The shallow listings above cover every root that holds closed-loop runs |

## 3. Every 0.1.8b run found

All 36 are native (the s10 run script + `drivejepa018bp*` adapter, AlpaSim wizard, 20 Hz referee, 26.01 artifacts: `G/s10_018b_closed_loop/REPORT.md:64-75`, `tables/harness_p06_real.json` 12/12 PASS). Date and sha are from each run's `run_metadata.yaml` (`run_time`) and `driver_worker.log` ("checkpoint: verified … sha256 …"). Scores are the lane's scene score from `G/s10_018b_closed_loop/tables/stepD.csv` (columns `s018bp06`, `exit018bp06`, etc.); at-fault is cross-checked against `aggregate/results-summary.json` → `collision_at_fault` in each run folder (p06: 1.0 on 04749bb9, 8023dce5, bd07fb91; 0.0 elsewhere; `offroad` 1.0 on d1a25a99).

| scene | lane | snapshot | harness | date (IST) | score | at-fault | path (`NAS018b/closed_loop/runs/…`) |
|---|---|---|---|---|---|---|---|
| e3ed554f | ext20 | p06 | native | 2026-10-31 05:44 | 1.000 | no | `drivejepa_018bp06_e3ed554f` |
| 8023dce5 | ext20 | p06 | native | 2026-10-31 05:49 | 0.000 | **yes** (front, +6.90 s) | `drivejepa_018bp06_8023dce5` |
| 5f064b02 | ext20 | p06 | native | 2026-10-31 05:54 | 1.000 | no | `drivejepa_018bp06_5f064b02` |
| e121e37d | ext20 | p06 | native | 2026-10-31 06:00 | 1.000 | no | `drivejepa_018bp06_e121e37d` |
| 048b974e | pilot10 | p06 | native | 2026-10-31 06:05 | 1.000 | no | `drivejepa_018bp06_048b974e` |
| cdd9448c | ext20 | p06 | native | 2026-10-31 06:10 | 1.000 | no | `drivejepa_018bp06_cdd9448c` |
| d2715023 | ext20 | p06 | native | 2026-10-31 06:16 | 1.000 | no | `drivejepa_018bp06_d2715023` |
| bd07fb91 | ext20 | p06 | native | 2026-10-31 06:21 | 0.000 | **yes** (front, +6.95 s; 0.1.7 also crashes here) | `drivejepa_018bp06_bd07fb91` |
| bae463d3 | ext20 | p06 | native | 2026-10-31 06:26 | 0.380 | no | `drivejepa_018bp06_bae463d3` |
| d1a25a99 | ext20 | p06 | native | 2026-10-31 06:32 | 0.000 | no (off-road +6.95 s) | `drivejepa_018bp06_d1a25a99` |
| b1b170d0 | ext20 | p06 | native | 2026-10-31 06:37 | 0.723 | no | `drivejepa_018bp06_b1b170d0` |
| 04749bb9 | pilot10 | p06 | native | 2026-10-31 06:42 | 0.000 | **yes** (front, +12.40 s) | `drivejepa_018bp06_04749bb9` |
| same 12 | | p02 (`7469b45d`) | native | 2026-10-31 07:09–08:03 | mean 0.5432; at-fault 8023dce5, bd07fb91 | | `drivejepa_018bp02_<scene>` |
| same 12 | | p10 (`ff50947e`) | native | 2026-10-31 04:18–05:16 | mean 0.5880; at-fault 8023dce5, bd07fb91, 04749bb9 | | `drivejepa_018bp10_<scene>` |

p06 12-scene mean **0.5919** (`tables/stepD.json` → `means.018bp06`), vs 0.1.7 0.6742 on the same 12. New at-fault vs 0.1.7 on the 12: 8023dce5 and 04749bb9 (`stepD.json` → `rule.018bp06.new_atfault_scenes`).

Sweep logs confirm 12 launches per snapshot and no more: `G/s10_018b_closed_loop/logs/sweep_018bp06_all.log` ends "=== RUN 12 rc=0 === … SWEEP DONE" (2026-10-31 06:46); same for p02 (08:06) and p10 (05:19); `logs/run_all.log` ends "ALL DONE" at 05:28 and lists no further sweep.

## 4. Cross against the 30 scenes

**p06 runs exist for 12 / 30.** Missing (18): 01d503d4, 023b7fcc, 0245ff75, 026d6a39, 02e075b9, 02eadd92, 032b6f21, 04394343 (pilot10) and 16b83346, 1acf2f5d, 2e1ce314, 6725df2d, 71d7d03b, 791d68ce, 89f2eeb8, 98dd56b2, acd22430, b44a4a44 (ext20). Of the gate10 (= pilot10) scenes, only 2 / 10 have a p06 run.

**Nothing superseded the 12.** Every closed-loop run found that started after 2026-10-31 is one of:

| run | date | model actually loaded | scenes | harness | why it is not a 0.1.8b run |
|---|---|---|---|---|---|
| `NAS…/generator_joint_pai/018c/closed_loop/runs/drivejepa_018cp04_<12 scenes>` | 2026-11-03 | 0.1.8c-p04, sha `cd013d53…` (`…_8023dce5/driver_worker.log`) | the same 12 | native | different checkpoint (clean-LoRA test, `G/CLOSING_REPORT.md:33`) |
| `paddock-memory/results/alpasim/drivejepa/20261105T221005-55cefc`, `…T233339-494d1d`, `…T234110-c47908` | 2026-11-05 | released v2 checkpoint | 1 scene (1acf2f5d) each | Paddock bridge, `valid: false` | `run.json` → `config.model_env.DRIVEJEPA_CHECKPOINT = <site path:a90f1310>`; `sha256("{paddock_site}/models/drivejepa/drive_jepa_perception_based_agent_vitl_v2.ckpt")[:8] = a90f1310` (the value in `paddock-memory/paddock.toml:54`); that file hashes to `740335c8…d05` (3,735,352,913 bytes), not a 0.1.8b sha. These are the H15 scorer-memory gate runs (`provenance.policy.memory.mode` carry/reset) |
| `…/20261106T013721-fda566` (carry) and `…/20261106T040355-9f4cd1` (reset) | 2026-11-06 | released v2 checkpoint (same digest) | gate10 (n = 10), means 0.394 / 0.508 | Paddock bridge, `valid: false` | same as above; copies under `{nas_store}/paddock-h15-runs/alpasim/drivejepa/` |

Runs dated before 2026-10-31 cannot contain 0.1.8b (p06 exported 2026-10-31 03:53, `NAS018b/release/`; `NAS018b/DONE` = "2026-10-31T03:22:01+05:30 rc 0"). The 0.1.8 30-scene run (`NAS…/generator_joint_pai/closed_loop/runs/drivejepa_018_<30 scenes>`, 2026-10-27 07:45–23:10) loaded `drive_jepa_pai_joint_p04.ckpt` sha `b6686d8c…` (`…/drivejepa_018_01d503d4/driver_worker.log`); its review page records means 0.1.8 0.7350 / 0.1.7 0.7697 / released 0.5433 (`NAS…/closed_loop/review/key_numbers.json`).

## 5. Means (what can and cannot be given)

| quantity | 0.1.8b-p06 | 0.1.7 (for reference) | source |
|---|---|---|---|
| 30-scene mean | **not available** (12 / 30 run) | 0.7697 | 0.1.7: `G/s4_closed_loop/tables/per_scene.csv` (`017_scene_score`, 30 rows) |
| gate10 mean | **not available** (2 / 10 run: 048b974e 1.000, 04749bb9 0.000) | 0.9956 | same table, pilot10 rows |
| 12-scene mean | 0.5919 | 0.6742 | `G/s10_018b_closed_loop/tables/stepD.json` |
| at-fault crashes | 3 on the 12 (8023dce5, bd07fb91, 04749bb9) | 3 on the 30 (acd22430, b44a4a44, bd07fb91) | p06: `results-summary.json` per run; 0.1.7: `per_scene.csv` → `017_collision_at_fault` |
| new at-fault vs 0.1.7 | 8023dce5, 04749bb9 (on the 12; the other 18 scenes are unknown) | — | `stepD.json` |

No other snapshot (p02, p10) has 30 scenes either: 12 each.

## 6. Disagreements and corrections

1. **Sid's recollection vs files:** "all 30 were run" is **not supported** for 0.1.8b. All 30 were run for **0.1.8** (2026-10-27) and for 0.1.7, 0.1.6, 0.1.4b; 0.1.8b, 0.1.8-rs, 0.1.8-p9 and 0.1.8c were run on the 12 diagnosis scenes only. Run 1 of the approach-3 recon (`APPROACH3_RECON.md:72, 293`) and the lane's own reports (`G/s10_018b_closed_loop/REPORT.md:8, 25`; `G/CLOSING_REPORT.md:60, 72, 113`) agree with the files.
2. **Brief's paths:** `scenes_12.txt` is under `Gallops/Gallops-sid/…` (the brief omitted `Gallops/`); the run script is in `AS/drafts/drivejepa-0.1.8b-p06/`, not in the adapter folder.
3. **Memory-worktree runs dated 2026-11-05/06 look like Drive-JEPA runs after 0.1.8b but are not:** they load the released checkpoint through the bridge and carry `valid: false` ("site env override may change what was measured", `run.json` → `notes`).
4. **Not fully checked:** the deep `find` over the whole NAS was killed at 14 min; the shallow listings of every root that holds closed-loop runs are the evidence instead.

## 7. What this means for arm (c)

Arm (c) of the approach-3 design (0.1.8b-p06 unchanged) still has no 30-scene and no gate10 result; OPEN item 12 of `APPROACH3_RECON.md` (one native 30-scene run of p06, ≈ 5 h, reusing the 12 existing runs if the simulator-image check passes) stands.
