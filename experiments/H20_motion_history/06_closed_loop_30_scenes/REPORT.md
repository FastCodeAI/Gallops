# Closed loop, approach 4: arm A and arm B on the 30 scenes

Started 2026-11-20 under Sid's terms of that day. Pre-registration `PREREG_APPROACH4.md`, sha256 `f8f4b975…a450`, checked by the queue at start. Score = the windowed `rollouts[0].score`. There is one run per scene and no selective re-runs.

The earlier version of this file was a STOP: 22 GB free on `/` before the images existed. That is superseded. Approach 3's session (tmux `cl20ic`) built both images on 2026-11-20 between 00:05 and 00:30.

## 1. Images (no rebuild, per Sid)

| image | id | saved copy (`{approach4_store}/images/`) |
|---|---|---|
| `alpasim-base:0.134.0` | `sha256:ab16bc78a4686dcd63c2b77a77fe585df3846528a61b552db6b71f9553983a44`, built 2026-11-20 00:30:50 by approach 3's queue from the same Dockerfile | `alpasim-base_0.134.0_ab16bc78.tar`, 12,088,414,208 B, plus a `.sha256` file |
| `nvcr.io/nvidia/nre/nre-ga:26.04` | `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284`, the digest named in the brief | `nre-ga_26.04_97f43e71.tar`, 14,287,742,464 B (the same size as the NAS tar), plus a `.sha256` file |

The queue refuses to start unless both ids are still these.

## 2. Image check (PREREG precondition): p06 on the 8 named scenes

**Reused from approach 3's image check**, under Sid's three conditions, all verified 2026-11-20 01:35: e3ed554f, 8023dce5, d2715023, 04749bb9.
- **(a) p06 on the native driver: verified.** Every `driver.log` shows `checkpoint=…drive_jepa_018b_p06.ckpt sha256=3d566eb2…8e66 … source=PINNED`, `planner_module=…/adapters/drivejepa018bp06/planner.py` and `adapter_version=0.1.8b-p06`.
- **(b) image `ab16bc78`: inferred, not logged.** The run logs carry no image id. The evidence:
  - the cl20ic queue started with no alpasim-base image at all (its log: `image before:` empty);
  - the only build, `ab16bc78`, was created at 00:30:50 and is still the image under that tag;
  - every one of the 4 drives launched its simulation containers after that time. The first, e3ed554f, did so at 00:58:55 (`wizard_console.log`: "Built 3 simulation containers").
- **(c) per-tick rows: verified bit-identical to the recorded runs.** All 4 have equal scores (1, 0, 1, 0) and 9,004 / 9,027 / 9,004 / 9,004 rows (`src/cl_imagecheck_cmp.py`, approach 3's rule unchanged; `out/cl_imagecheck_e3ed554f_8023dce5_d2715023_04749bb9.json`).

**Driven here:** 791d68ce, 026d6a39, 02e075b9 and 89f2eeb8. The runner is p06's own `run_018bp06_scene.sh` with only its output root changed (`adapters/drivejepa_a4/launch/run_018bp06_imagecheck_a4.sh`).

Results are appended below by the queue.

## 3. Driver wiring: `adapters/drivejepa_a4/` (a copy; `drivejepa018bp06` is not edited)

- **Planner:** one new feature key, `ego_history` (1, 8, 6), built beside `ego_status` in `build_features`, as `BUILD1_HISTORY_DATA.md` 4c specifies:
  - `build_history` from `MH/src/ego_history.py` (sha256 checked against the pre-registered `01a96fdb…`, loaded by path);
  - over every pose of the session, with `t_us` = the newest pose's timestamp and `T_ref` = that pose (the same frame as `_build_ego_status`).
- **Worker** (`a4.py`, installed after the batch fix):
  - It builds `MH/src/model_a4.py`'s `A4Model` around the live p06 agent. `load_trainable` copies the checkpoint's 41 generator and 30 scorer tensors into the agent and loads the new blocks. Eval mode.
  - The checkpoint path comes from a per-lane config, `cl/model/<lane>/a4_config.json` (checkpoint, sha256, arm), named by the environment variable `DRIVEJEPA_A4_CONFIG`. Launcher and worker both verify and log it.
  - The history enters through hooks at the exact points where model_a4 adds it:
    - `z = summary(history, training=False)`: arm A never hides; arm B always uses the "no history" vector;
    - link 1 is added to the first refiner pass's input, i.e. after `hist_encoding(ego) + init_feature`;
    - link 2 is added to the output of the rating head's first layer.
  - Each call asserts 4 refiner passes and 1 first-layer call.
- **Unchanged:** the momentum and comfort re-score, the encoder, the batch fix, the divider and the wizard config.
- **Offline worker check** (`src/a4_worker_check.py`, `out/a4_worker_check.json`; worker env, CPU; run on cores 2-3 while cl20ic held 0-1, no simulator):

  | check | result |
  |---|---|
  | step-0 snapshot vs p06 | bit-identical, max diff 0.0 |
  | arm A (p00.2) vs p06 | moves |
  | arm A, a different history | moves (9.44) |
  | arm B (p00.6), a different history | does not move (0.0) |

- **The worker's torch is 2.1.0; training ran on torch 2.8.0** (the cu128 overlay). The new blocks are plain Linear and GELU layers, and the frozen network is the one p06's runs use.
- **Build freeze:** `launch/tables/build_a4.sha256` (14 files) and `launch/tables/model_a4.sha256` (model_a4.py, ego_history.py, pai_model.py, pai_common.py). The runner refuses to start if any of them changes.
- **Runner:** `launch/run_a4_scene.sh`, a copy of p06's runner. It takes a lane tag `step0A | armA | armB`, writes to `{approach4_store}/cl/runs/a4_<tag>_<scene>`, and has provenance gates for the approach-4 checkpoint and arm. It requires the wiring proof to have passed for the arm lanes.

**Snapshots:**

| lane | file | sha256 |
|---|---|---|
| wiring proof (`step0A`) | `runs/arm_A/ckpt/p00.0.pt` | `d2ff692cfbad8bd94d2f7ff4f5725fc4768daca1a4c665294910682ab73dcdc5` |
| arm A (`armA`) | `runs/arm_A/ckpt/p00.2.pt` | `d1a40752613e0dad4e678bc44cba904dc314fc7b57ebfd5dcd79c592450b4f4e` |
| arm B (`armB`) | `runs/arm_B/ckpt/p00.6.pt` | `35c83bf67e89b4579cb3d8f77fc45862da2f0b2070caecc1f5193caed940dd9a` |

## 4. Queue

- `src/cl_queue.sh` runs in tmux `a4cl` under `taskset -c 0-1 nice -n 19`.
- It waits for cl20ic to finish, then:
  1. drives the 4 image-check scenes and compares all 8;
  2. runs the wiring proof (e3ed554f, which must be per-tick bit-identical to p06's recorded run and to today's p06 re-run);
  3. drives arm A on the 12 diagnosis scenes, writes the preview table, then drives the other 18;
  4. drives arm B on all 30, in the same order;
  5. writes the final tables and the gate.
- **Before each scene:** at least 12 GB free on the GPU (checked twice, 60 s apart; no time limit on the wait) and the memory gate.
- **When it stops:** any failed check, or a drive that ends without a scored rollout. It never re-runs a scene.
- **Status and logs:** `cl/STATUS.txt` and `cl/queue.log`.

**Queue order (Sid, 2026-11-20).** Approach 3's session started its 40-drive queue (tmux `cl20`) at 01:52, right after its image check. The two queues cannot share the machine: they use the same driver port and allow one wizard at a time. Sid chose: approach 3's `cl20` runs first. At 03:44 `cl20_queue.sh` was restarted to wait until neither of approach 3's queue processes is alive and no wizard of ours is running, on two checks 5 min apart; then it runs as above.
- **08:32, Sid "run now":** cl20 had already finished at 08:07, but our wait check was matching the still-open `cl20ic` tmux session, whose process title keeps its launch command line. I tightened the check to real queue and wizard processes and restarted the queue (it waits 5 min, then starts).

## Image check: PASS on 8

```
e3ed554f {'orig': 1.0, 'rerun': 1.0, 'equal_3dp': True, 'metric_rows': (9004, 9004), 'per_tick_bit_identical': True, 'columns_differing': None}
8023dce5 {'orig': 0.0, 'rerun': 0.0, 'equal_3dp': True, 'metric_rows': (9027, 9027), 'per_tick_bit_identical': True, 'columns_differing': None}
d2715023 {'orig': 1.0, 'rerun': 1.0, 'equal_3dp': True, 'metric_rows': (9004, 9004), 'per_tick_bit_identical': True, 'columns_differing': None}
04749bb9 {'orig': 0.0, 'rerun': 0.0, 'equal_3dp': True, 'metric_rows': (9004, 9004), 'per_tick_bit_identical': True, 'columns_differing': None}
791d68ce {'orig': 0.0, 'rerun': 0.0, 'equal_3dp': True, 'metric_rows': (8981, 8981), 'per_tick_bit_identical': True, 'columns_differing': None}
026d6a39 {'orig': 1.0, 'rerun': 1.0, 'equal_3dp': True, 'metric_rows': (9004, 9004), 'per_tick_bit_identical': True, 'columns_differing': None}
02e075b9 {'orig': 1.0, 'rerun': 1.0, 'equal_3dp': True, 'metric_rows': (8981, 8981), 'per_tick_bit_identical': True, 'columns_differing': None}
89f2eeb8 {'orig': 1.0, 'rerun': 1.0, 'equal_3dp': True, 'metric_rows': (9027, 9027), 'per_tick_bit_identical': True, 'columns_differing': None}
IMAGE CHECK PASS sha256:ab16bc78a4686dcd63c2b77a77fe585df3846528a61b552db6b71f9553983a44 2026-11-20T00:30:50.652230704+05:30
```

## Wiring proof: PASS

```
{"new": "{approach4_store}/cl/runs/a4_step0A_e3ed554f", "recorded_p06": {"ref": "{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_e3ed554f", "score_ref": 1.0, "score_new": 1.0, "rows": [9004, 9004], "per_tick_bit_identical": true}, "p06_rerun_this_image": {"ref": "{h17_bulk}/runs/new40/imagecheck/018bp06/runs/drivejepa_018bp06_e3ed554f", "score_ref": 1.0, "score_new": 1.0, "rows": [9004, 9004], "per_tick_bit_identical": true}, "worker_logged_step0_snapshot": true, "pass": true}
WIRING PROOF PASS
```

## Preview: arm A on the 12 diagnosis scenes (recorded, never judged)

| # | scene | set | p06 | arm A | vs p06 |
|---|---|---|---|---|---|
| 1 | e3ed554f | 12 | 1.000000 | 0.000000 | FLIP down |
| 2 | 8023dce5 | 12 | 0.000000 | 0.000000 | same |
| 3 | 5f064b02 | 12 | 1.000000 | 1.000000 | same |
| 4 | e121e37d | 12 | 1.000000 | 0.724176 | partial -0.276 |
| 5 | 048b974e | 12 | 1.000000 | 1.000000 | same |
| 6 | cdd9448c | 12 | 1.000000 | 1.000000 | same |
| 7 | d2715023 | 12 | 1.000000 | 0.030296 | partial -0.970 |
| 8 | bd07fb91 | 12 | 0.000000 | 0.000000 | same |
| 9 | bae463d3 | 12 | 0.379891 | 1.000000 | partial +0.620 |
| 10 | d1a25a99 | 12 | 0.000000 | 0.873375 | partial +0.873 |
| 11 | b1b170d0 | 12 | 0.723395 | 0.684016 | partial -0.039 |
| 12 | 04749bb9 | 12 | 0.000000 | 1.000000 | FLIP up |

Arm A mean over the 12: **0.6093** vs p06 0.5919. Flips up 1, flips down 1, partial 5, same 5.

## Results: arms A and B on all 30

| # | scene | set | p06 | arm A | vs p06 | arm B | vs p06 |
|---|---|---|---|---|---|---|---|
| 1 | e3ed554f | 12 | 1.000000 | 0.000000 | FLIP down | 1.000000 | same |
| 2 | 8023dce5 | 12 | 0.000000 | 0.000000 | same | 1.000000 | FLIP up |
| 3 | 5f064b02 | 12 | 1.000000 | 1.000000 | same | 0.923611 | partial -0.076 |
| 4 | e121e37d | 12 | 1.000000 | 0.724176 | partial -0.276 | 1.000000 | same |
| 5 | 048b974e | 12 | 1.000000 | 1.000000 | same | 0.880659 | partial -0.119 |
| 6 | cdd9448c | 12 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 7 | d2715023 | 12 | 1.000000 | 0.030296 | partial -0.970 | 1.000000 | same |
| 8 | bd07fb91 | 12 | 0.000000 | 0.000000 | same | 0.000000 | same |
| 9 | bae463d3 | 12 | 0.379891 | 1.000000 | partial +0.620 | 0.388687 | partial +0.009 |
| 10 | d1a25a99 | 12 | 0.000000 | 0.873375 | partial +0.873 | 0.844746 | partial +0.845 |
| 11 | b1b170d0 | 12 | 0.723395 | 0.684016 | partial -0.039 | 0.674153 | partial -0.049 |
| 12 | 04749bb9 | 12 | 0.000000 | 1.000000 | FLIP up | 0.000000 | same |
| 13 | 01d503d4 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 14 | 023b7fcc | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 15 | 0245ff75 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 16 | 026d6a39 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 17 | 02e075b9 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 18 | 02eadd92 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 19 | 032b6f21 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 20 | 04394343 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 21 | 16b83346 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 22 | 1acf2f5d | 18 | 0.834999 | 1.000000 | partial +0.165 | 0.900378 | partial +0.065 |
| 23 | 2e1ce314 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 24 | 6725df2d | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 25 | 71d7d03b | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 26 | 791d68ce | 18 | 0.000000 | 0.000000 | same | 0.000000 | same |
| 27 | 89f2eeb8 | 18 | 1.000000 | 1.000000 | same | 1.000000 | same |
| 28 | 98dd56b2 | 18 | 0.791024 | 1.000000 | partial +0.209 | 1.000000 | partial +0.209 |
| 29 | acd22430 | 18 | 0.000000 | 0.000000 | same | 1.000000 | FLIP up |
| 30 | b44a4a44 | 18 | 0.851624 | 1.000000 | partial +0.148 | 0.680326 | partial -0.171 |

| | mean 30 | mean 12 | mean 18 | flips up | flips down | partial | same |
|---|---|---|---|---|---|---|---|
| p06 (baseline) | 0.7527 | 0.5919 | 0.8599 | | | | |
| arm A | **0.7771** | 0.6093 | 0.8889 | 1 | 1 | 8 | 20 |
| arm B | **0.8431** | 0.7260 | 0.9212 | 2 | 0 | 8 | 20 |

**Gate (PREREG section 2): PASS iff mean(A) ≥ 0.8194 AND mean(A) > mean(B).** mean(A) = 0.7771 (< 0.8194: no); mean(A) ≤ mean(B) = 0.8431: no. **Gate: FAIL.**
Difference vs p06: arm A +0.0244 (+0.7 scene-equivalents), arm B +0.0904.
