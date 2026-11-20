# Closed loop on the 18 remaining scenes: memory carried vs 0.1.8b pass 0.6 — pre-registration

Written 2026-11-13 21:41:49 IST, before any of these scenes was run. Continues `reports/CL12_MEMORY.md`.

## Scenes

The 30-scene set is the list of the lane that ran 0.1.8 on 30 scenes: `LORA Refiner/Generator-Joint-PAI-alpasim/s4_closed_loop/tables/scenes_30.txt` (sha256 `84d2a439…`; its `REPORT.md` line 97 says it equals the released and 0.1.7 30-scene lists; checked here: the 30 `drivejepa_018_<scene>` run folders of that lane name exactly these 30). Removing the 12 diagnosis scenes (`s10_018b_closed_loop/tables/scenes_12.txt`; all 12 are in the 30) leaves **18**:

`01d503d4 023b7fcc 0245ff75 026d6a39 02e075b9 02eadd92 032b6f21 04394343 16b83346 1acf2f5d 2e1ce314 6725df2d 71d7d03b 791d68ce 89f2eeb8 98dd56b2 acd22430 b44a4a44`

Saved with their clip ids and scene files as `out/cl18_scenes_18.txt` (sha256 `8a76391386cce356…`). All 18 scene artifacts are cached (checked 2026-11-13).

## Arms (one rollout per scene and arm, 36 runs)

| arm | what | runner |
|---|---|---|
| carry | frozen memory adapter `adapters/drivejepamemory`, version 0.1.8b-p06-memory5, mode carry from `{experiment_store}/runs/cl12/model/carry/memory_mode.json`, build freeze `launch/tables/build_memory.sha256` (checked by the runner and the queue) | `src/s18_run_memory_scene.sh` = the frozen `launch/run_memory_scene.sh` with two lines changed: output root `{experiment_store}/runs/cl18` and the model folder pointing at the 12-scene one |
| 0.1.8b | 0.1.8b pass 0.6 through its own `adapters/drivejepa018bp06` adapter, launcher and build freeze | `src/s18_run_018bp06_scene.sh` = `drafts/drivejepa-0.1.8b-p06/run_018bp06_scene.sh` with one line changed: output root `{experiment_store}/runs/cl18/018bp06` |

- Comfort re-score on in both, as before. No reset arm.
- **Shadow stays on in the carry arm.** The frozen run script refuses `DRIVEJEPA_MEMORY_SHADOW` for real runs (`run_memory_scene.sh`, provenance block), and changing that would change the frozen build. The shadow does not touch the driven plan (12-scene proof A) and is logged per call; its "would have picked differently" share is reported as information.
- Simulator image: `alpasim-base:0.134.0` = `sha256:9c5909da…f8881`, the 12-scene image. The queue refuses to start, and checks again before every run, if the image differs.
- Same wizard command, scene files, 20 Hz settings and run gates as the 12-scene runs; each run makes its annotated video and replay.
- **CPU pinning.** All CPU-side work runs under `taskset -c 0-1` and `nice 19` (the queue and everything it starts: wizard, driver, model worker). The 12-scene runs were not pinned. Checked before this pre-registration (`out/cl18_thread_check.json`): the 0.1.8b-p06 worker on its 64 recorded inputs and the memory worker in carry mode on clip 0238dfe4 give **bit-identical** plans and scores pinned to 2 cores and unpinned (32 cores); about 2.1 s per call pinned against 1.5–1.8 s unpinned.

## Measures (same code as the 12-scene check)

- Score: `s5_diagnosis/src/stepb.py` `score`. Exit: `exit_of` (first of front collision, side collision, off-road, hit from behind, with time after handover). **At-fault** = front or side collision. Off-road and hit-from-behind are reported, not counted.

## Primary rule (18 new scenes only, fixed now)

d = (18-scene mean of carry) − (18-scene mean of 0.1.8b).

- **HELPS** if d ≥ +0.08 AND carry has no at-fault crash on a scene where 0.1.8b has none.
- **HURTS** if d ≤ −0.08, OR carry has an at-fault crash on a scene where 0.1.8b has none (this clause wins).
- **NO DETECTABLE EFFECT** otherwise.

One scene flipping moves an 18-scene mean by about 0.056 (1/18), so the bar is about one and a half scenes.

## Robustness (reported, not deciding)

- d with the two scenes most in carry's favour (largest carry − 0.1.8b) removed, over the remaining 16.
- Number of scenes where carry wins (carry − 0.1.8b > 0.1), loses (< −0.1), ties (otherwise).
- Off-road exits listed separately for both arms.
- The carry run's shadow: share of calls where the start-state memory would have picked differently.

## Secondary (reported, not deciding)

30-scene means for carry and 0.1.8b: the 12 from the 12-scene check (carry: `{experiment_store}/runs/cl12/runs/memory_carry_*`; 0.1.8b: its existing runs `generator_joint_pai/018b/closed_loop/runs/drivejepa_018bp06_*`, which reproduced bit for bit on this image) plus the 18 new. Beside them, 0.1.7's 30 existing runs (`{dataset_store}/s5_trial_017/runs/drivejepa_017_*`, all 30 present), flagged **"earlier image"**. The 12 carry runs were unpinned; the 18 are pinned (see above).

## Execution

Queue `src/s18_queue.sh`, started detached in tmux, the same machine gate as before inside it and inside each run script (5-minute load < 32, available RAM > 40 GB, GPU free ≥ 8,000 MiB, no other wizard of ours, NAS ≥ 50 GB). Order per scene: carry, then 0.1.8b. A completed rollout is never rerun; a run without a rollout stops the queue. Everything under `{experiment_store}/runs/cl18/`. No training, no held-out data, no pushes.
