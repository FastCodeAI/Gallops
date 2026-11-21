# Closed-loop check of the memory on 12 scenes: pre-registration

Written 2026-11-12 21:27:28 IST, before any scene was run and before the simulator image exists on the machine (it has to be rebuilt first, see "Simulator image"). The adapter proofs were still running when this was written; they decide whether anything is run at all, not how the result is read.

## Question

Does build 5's model drive differently, in closed loop, when its memory is carried from call to call than when the same model has its memory reset to the start state at every call?

## Model (both arms, identical weights)

- Encoder: 0.1.8b pass 0.6, merged, run live on the camera frames. File `{retrain_store}/018b/release/drive_jepa_018b_p06.ckpt`, sha256 `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66` (re-hashed today).
- Generator, scorer, memory blocks: build 5, step 930, model version 3. File `{experiment_store}/runs/build5_short/ckpt/step0930.pt`, sha256 `84359783ef502e1ccc3a2dcff013bf45d215f9fb11e046cc8110c64581434e21` (hashed today). The training-only read-out is not loaded.
- **Offline, this checkpoint is below its starting point:** pick grade 0.7313 with the memory carried and 0.7355 with the memory reset every moment, against 0.7685 for 0.1.8b pass 0.6 (`reports/BUILD5_RECALL_LESSON.md` sections 3.1 and 3.3; 0.7685 from `LORA Refiner/Generator-Joint-PAI-alpasim/s9_018b_train/REPORT.md` line 4). This is stated beside every result.
- Adapter: `Paddock/paddock-sid/Alpasim/adapters/drivejepamemory/`, a copy of the 0.1.8b-p06 native adapter with a memory switch. Frame selection, ego row, zeroed acceleration, batch fix, bonnet mask and the comfort re-score (on) are the 0.1.8b-p06 adapter's.

## Arms

| arm | what | source |
|---|---|---|
| carry | memory carried per session; reset to the start state at session start, session close, and when the previous call is not 0.45–0.55 s back in simulated time | new runs |
| reset | same weights, memory read from the start state at every call | new runs |
| 0.1.8b (reference) | pass 0.6, 12-scene mean 0.5919 | existing runs `{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_*` if the image check passes; otherwise rerun |
| 0.1.7 (reference) | 12-scene mean 0.6742 | existing runs, not rerun |

- Scenes: the 12 diagnosis scenes of `LORA Refiner/Generator-Joint-PAI-alpasim/s10_018b_closed_loop/tables/scenes_12.txt`, unchanged.
- One rollout per scene and arm, native harness, the same wizard command, scene files, 20 Hz referee settings and run-script gates as 0.1.8b's 12 runs (the run script is a copy of 0.1.8b's with names, output folder, provenance checks and a machine gate changed).
- The mode comes from a sidecar file beside the checkpoint (`{experiment_store}/runs/cl12/model/{carry,reset}/memory_mode.json`); the two folders differ in that one field.
- A completed rollout is never rerun. A scene whose run produces no rollout (machine, docker or wizard failure) is started again and the first attempt is listed in the report.

## Measures

- **Score** per scene: the lane's scoring code, unchanged (`s5_diagnosis/src/stepb.py`, `score`), as used for 0.5919 and 0.6742.
- **At-fault crash**: the scene ends in a front or side collision by the same code's exit classification (`exit_of`). Being hit from behind and leaving the road are reported, not counted as at-fault, as in the lane.
- **Off-road**: reported per scene.
- **How often carry and reset pick differently**: at every model call each run also evaluates the other mode's memory on the same input (same encoder output, same weights). Reported per scene and pooled, for each arm's run: the share of calls where the scorer's pick (before the comfort re-score) differs, and the share where the final pick (after it) differs. Calls during the first 3 s (the simulator drives) are included and also shown separately.
- **Diagnostics** per scene: number of model calls, chain resets, mean memory norm, mean gate.

## Rule (fixed now)

Let d = (12-scene mean of carry) − (12-scene mean of reset).

- **"Memory helps in closed loop"** if d ≥ +0.08 AND carry has no at-fault crash on a scene where reset has none.
- **"Memory hurts"** if d ≤ −0.08, OR carry has an at-fault crash on a scene where reset has none (this clause wins even if d ≥ +0.08).
- **"No detectable effect"** otherwise.

The references (0.1.8b 0.5919, 0.1.7 0.6742) are shown beside the result and take no part in the rule.

## What 12 scenes can and cannot show

Twelve scenes with one rollout each detect only large effects. Most scenes score near 0 or near 1, so one scene flipping moves the 12-scene mean by about 0.08 (1/12 = 0.083); the threshold is one scene's worth. A verdict of "helps" or "hurts" can therefore rest on a single scene, and "no detectable effect" does not show the memory is inert. No repeat of the same arm is run, so run-to-run spread is not measured here; the lane's reruns of 0.1.7 with fixed weights were bit-identical (`CLOSING_REPORT.md` lines 207–213), which is the basis for one rollout per arm.

## Simulator image

On 2026-11-12 the machine holds no simulator image (`docker images` lists none), as on 2026-11-03. The run script rebuilds it on the first run. Check, same rule as 2026-10-27 and 2026-11-03, fixed before looking: 0.1.7 is re-run on 04749bb9, acd22430, d2715023 and b44a4a44 on the rebuilt image and must equal its original runs, scores to 3 decimals and per-tick metrics bit-identical.

- Pass: the existing 0.1.8b and 0.1.7 runs are used as references.
- Fail: 0.1.8b pass 0.6 is rerun on the 12 scenes on the new image (its own adapter, a copy of its run script writing under `{experiment_store}/runs/cl12/`), the 0.1.7 figure is flagged "different simulator build", and carry vs reset is still read by the rule above since both arms run on the same image.

## Before any scene

Adapter proofs, all three must pass or nothing is run:
- A. With the memory reads disabled in a test copy, the adapter reproduces 0.1.8b-p06's plans and picks bit for bit on the recorded inputs of the 0.1.8b-p06 equivalence test.
- B. One picking clip's recorded frames through the adapter worker match the offline replay of the same checkpoint within 1 mm on plans and exactly on picks, in carry mode and in reset mode.
- C. Two sessions back to back: the second starts from the start state.

Machine gates inside the run script: GPU free memory ≥ 8,000 MiB (as 0.1.8b's runs), 5-minute load < 32, available RAM > 40 GB, NAS free ≥ 50 GB.

## Not done

No training. No held-out slice opened (proof B uses a picking clip). No pushes.

## Amendment 1, 2026-11-12 22:28:51 IST (before any scene has run; no simulator image exists yet)

The image check as written above cannot be run. The 0.1.7 runner refused to start at 22:27 on its own provenance check, before launching anything (`logs/cl12_queue.log`): its build folder was moved to the NAS on 2026-11-09 (`analysis_3.0` is now a link), so its path check fails, and its launcher then cannot find the shared adapter folder (`--check-only` tried at 22:30: assertion on `{paddock_sid_moved}/Alpasim/adapters`). Fixing either means editing or adding files in earlier lane folders, which this task may not do.

Replacement, fixed now, before any run:
- **Image check:** 0.1.8b pass 0.6 itself (its own adapter, launcher, build freeze and checkpoint; a copy of its run script whose only change is the output root, under `{experiment_store}/runs/cl12/imagecheck_018bp06/`) is re-run on four of the 12 scenes: e3ed554f and 8023dce5 (the first two of the list) and d2715023 and 04749bb9 (the two scenes of the earlier image checks that are among the 12). Same equality rule: each re-run must equal the original 0.1.8b run, score to 3 decimals and per-tick metrics bit-identical.
- Pass: the existing 0.1.8b runs are the reference (0.5919). Fail: 0.1.8b is re-run on the other eight scenes too and the new 12 are the reference.
- **0.1.7 (0.6742) is not re-checked on this image.** If the 0.1.8b check passes, the simulator reproduces 0.1.8b bit for bit, and 0.1.7 was reproduced bit for bit across the two earlier rebuilds; the figure is then shown as a reference with that stated. If the check fails, it is flagged "different simulator build".
- Nothing else changes: arms, scenes, measures and the rule are as above.
