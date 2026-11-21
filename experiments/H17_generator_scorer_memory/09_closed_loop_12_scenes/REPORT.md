# Closed-loop check of the memory on 12 scenes: carried vs reset

2026-11-13, shakaal. Runs 2026-11-12 23:36 to 2026-11-13 14:52 IST. Pre-registration: `reports/CL12_MEMORY_PREREG.md` (written 2026-11-12 21:27 IST, before any scene; amendment 1 at 22:29, before any scene). Numbers are **measured** from the run outputs unless marked *argued*.

## Answer

**By the pre-registered rule: "memory helps in closed loop".** Carried 0.7213 against reset 0.5177 on the 12-scene mean, a difference of +0.204, above the +0.08 bar. Carry has no at-fault crash that reset does not also have. Reset has one more: 04749bb9.

**Read it narrowly.** The whole difference comes from three scenes, and two of them carry most of it:

- bae463d3: carry 1.000, reset 0.040
- 5f064b02: carry 1.000, reset 0.172
- 04749bb9: carry 0.707, reset 0.000

Leave out bae463d3 and the difference is +0.135 over 11 scenes; leave out 5f064b02 and it is +0.147. Leave out both and it falls to +0.066, below the bar. With one rollout per arm and no repeat, this is two to three scene flips, not a measured rate. The pre-registration says as much: "12 scenes detect only large effects; one scene flipping moves the mean by about 0.08".

**The offline picture points the other way.** Offline, the same weights picked slightly better with the memory reset than carried: pick grade 0.7355 reset, 0.7313 carried. Both are below 0.1.8b pass 0.6 at 0.7685 (`reports/BUILD5_RECALL_LESSON.md` §3.1 and §3.3; `LORA Refiner/Generator-Joint-PAI-alpasim/s9_018b_train/REPORT.md` line 4).

| arm | 12-scene mean | at-fault crashes | offline pick grade |
|---|---|---|---|
| **carry** (build 5, memory carried) | **0.7213** | 8023dce5, bd07fb91 | 0.7313 |
| **reset** (same weights, memory reset at every call) | **0.5177** | 8023dce5, bd07fb91, 04749bb9 | 0.7355 |
| 0.1.8b pass 0.6 (existing runs, image check passed) | 0.5919 | 8023dce5, bd07fb91, 04749bb9 | 0.7685 |
| 0.1.7 (existing runs) | 0.6742 | bd07fb91 | — |

Sources: `out/cl12_table.json`, `out/cl12_table.csv` (built by `src/s12_table.py`, log `logs/cl12_table.log`).

## 1. The adapter

**Folder.** `Paddock/paddock-sid/Alpasim/adapters/drivejepamemory/`, version string `0.1.8b-p06-memory5`. It is a new folder; the 0.1.8b-p06 adapter was not edited.

- **Byte-identical to 0.1.8b-p06:** `momentum.py`, `divider.py` and `batch_fix.py` (checked with `cmp`). The links `real_model.py`, `fake_model.py`, `__init__.py` and `src` are the same as p06's.
- **Changed:** `constants.py`, `planner.py` and `bridge.py`. The diff against p06 is `launch/tables/adapter_018bp06_to_memory.diff` (264 lines).
- **New:** `memory.py`.
- **Launch files:** the launcher, run script and extra gates are in `launch/`.
- **Build freeze:** `launch/tables/build_memory.sha256`. Every run checked it before starting.

**Model.**
- **Encoder:** 0.1.8b pass 0.6 (merged), run live on the camera frames. File `drive_jepa_018b_p06.ckpt`, sha256 `3d566eb2…8e66`.
- **Generator, scorer and memory blocks:** build 5, step 930, model version 3. File `runs/build5_short/ckpt/step0930.pt`, sha256 `84359783…4e21`, 36,748,734 bytes.
- **What the worker loads from the build-5 file:**
  - 41 generator tensors and 30 scorer tensors, copied over the 0.1.8b ones;
  - 46 memory tensors;
  - the 18 read-out tensors are **skipped**: the read-out is training-only. This is logged on every run.

**The memory switch.**
- **Where the state lives:** in the model worker, keyed by `session_uuid`.
- **Reset to the start state:** on session start and on session close. The planner sends a control message from `on_session_start` / `on_session_close`.
- **Chain rule:** the memory is carried only if the session's previous call is 450–550 ms back in simulated time. Otherwise it restarts from the start state, and the restart is counted as a chain reset.
- **Mode:** set by `memory_mode.json` beside the checkpoint, in `{experiment_store}/runs/cl12/model/{carry,reset}/`. There is no command-line flag.
  - The run script chooses only the folder.
  - It refuses to start if the sidecar's mode is not the arm, or if the file is a test copy.
- **Provenance:** both the launcher's and the worker's provenance lines carry the mode, the memory sha256, the version and "read-out NOT loaded". The run script checks all of them.

**What else is unchanged from 0.1.8b-p06:**
- frame selection, the ego row, acceleration zeroed, the bonnet mask;
- the batch fix;
- the comfort re-score: on, the same code.

**Which pick is written into the memory.** The scorer's own pick, taken before the comfort re-score. That is what the update block saw in training (`src/s6_model.py:314-315`) and what the offline replay writes. The plan that is driven is the pick after the comfort re-score, as in 0.1.8b.

**Shadow (to measure how often the two modes pick differently).** At every call the worker also runs the generator and scorer once more on the same encoder output, with the other mode's memory:
- in the carry run, the start state;
- in the reset run, a memory carried along the reset run's own drive.

The shadow never changes the driven plan. Proof A shows that, and `planner.py` reads only the driving branch.

**Diagnostics.** One `memory: call {...}` line per model call in each run's `driver.log`. It records: call number, gap to the previous call, chain reset, memory norm, gate mean/min/max, the scorer and final picks of both modes, and the gap between the two picked plans.

## 2. Proofs before any scene (all pass)

All three were run on the frozen build, and all three files are copied into `launch/tables/`.

| proof | what | result | file |
|---|---|---|---|
| A, identity | Test copy: 0.1.8b-p06's generator and scorer, with the generator read's output projection and the rating head's new columns set to zero. Fed the 64 recorded inputs of the 0.1.8b-p06 equivalence test (`release/gate_018/train_side.npz`) against the 0.1.8b-p06 worker. Pass 1: single calls, no comfort re-score (the earlier test's call), with the memory live (63/64 calls read a carried memory, norm 60.9–64.0, gate 0.16–0.40). Pass 2: one session, 0.5 s apart, comfort re-score on (63 rescored calls, 14 where it changed the pick). | **PASS.** Proposals, logits, scores, the chosen plan, and the picks before and after the re-score are bit for bit equal on 64/64 in both passes. | `out/cl12_proof_A.json` |
| B, offline match | Picking clip 0238dfe4 (32 moments, one unbroken chain), decoded from the same lossless frame copies the offline cache was built from, against the offline replay of the same checkpoint (`runs/build5_short/dumps/step0930_{carried,reset}.npz`). | **PASS.** Carry: largest plan difference 0.56 mm (median 0.04 mm), picks 32/32, 1 call read the start state, 0 chain resets. Reset: 0.15 mm, picks 32/32, 32/32 calls read the start state. | `out/cl12_proof_B.json` |
| C, reset works | Carry worker, two sessions of 6 calls back to back. | **PASS.** Session 2's first call reads the start state, and all 6 of its calls equal session 1's bit for bit. Calls 2–6 of session 1 read a carried memory. An unseen session starts at the start state. A 1.0 s gap resets the chain (counted 1), and the next 0.5 s call carries. A repeated start resets. | `out/cl12_proof_C.json` |

**Both launchers start on the real checkpoint.** `--check-only` was run for carry and for reset: READY, with the provenance lines as above (`logs/cl12_check_only_{carry,reset}.log`).

## 3. Simulator image

shakaal held no simulator image, so the first run rebuilt it:
- `alpasim-base:0.134.0`, id `sha256:9c5909da…f8881`, built 2026-11-12 22:56;
- the renderer image `nre-ga:26.04`, digest `97f43e71…`, is the same as on 2026-11-03.

**Check (amendment 1):** 0.1.8b pass 0.6 was re-run on e3ed554f, 8023dce5, d2715023 and 04749bb9, through its own adapter and a copy of its run script whose only change is the output folder.
- **Result: PASS.** All four scores are equal (1.0, 0.0, 1.0, 0.0), and all 9,004 per-tick metric rows per scene are bit-identical (`out/cl12_imagecheck.json`; runs in `{experiment_store}/runs/cl12/imagecheck_018bp06/`).
- So the existing 0.1.8b runs are the reference. 0.1.7 was not re-checked on this image (flag 2).

## 4. Per scene

Scores are the lane's scoring code, unchanged (`s5_diagnosis/src/stepb.py` `score`). The exit is the first of front collision, side collision, off-road or hit from behind, by the same code's `exit_of`. "At fault" means a front or side collision; off-road and hit-from-behind are reported but not counted, as in the lane.

| scene | carry | reset | 0.1.8b | 0.1.7 | carry exit | reset exit | final picks differ, carry run | final picks differ, reset run |
|---|---|---|---|---|---|---|---|---|
| e3ed554f | 1.000 | 1.000 | 1.000 | 1.000 | pass | pass | 20.5 % | — |
| 8023dce5 | 0.000 | 0.000 | 0.000 | 1.000 | **at-fault front +6.30 s** | **at-fault front +6.35 s** | 23.1 % | |
| 5f064b02 | **1.000** | **0.172** | 1.000 | 1.000 | pass | hit from behind +9.20 s | 20.5 % | |
| e121e37d | 1.000 | 1.000 | 1.000 | 1.000 | pass | pass | 7.7 % | |
| 048b974e | 0.948 | 1.000 | 1.000 | 1.000 | pass | pass | 7.7 % | |
| cdd9448c | 1.000 | 1.000 | 1.000 | 1.000 | pass | pass | 17.9 % | |
| d2715023 | 1.000 | 1.000 | 1.000 | 0.041 | pass | pass | 30.8 % | |
| bd07fb91 | 0.000 | 0.000 | 0.000 | 0.000 | **at-fault front +5.85 s** | **at-fault front +5.90 s** | 12.8 % | |
| bae463d3 | **1.000** | **0.040** | 0.380 | 0.311 | pass | pass | 30.8 % | |
| d1a25a99 | 0.000 | 0.000 | 0.000 | 0.000 | off-road +5.80 s | off-road +5.80 s | 10.3 % | |
| b1b170d0 | 1.000 | 1.000 | 0.723 | 0.782 | pass | pass | 2.6 % | |
| 04749bb9 | **0.707** | **0.000** | 0.000 | 0.956 | off-road +6.90 s | **at-fault front +6.30 s** | 7.7 % | |
| **mean** | **0.7213** | **0.5177** | 0.5919 | 0.6742 | | | | |

The per-scene values of the last column and every other diagnostic are in `out/cl12_table.csv`, columns `pct_differ_*`, `calls_*`, `chain_resets_*`, `memory_norm_*`, `gate_mean_*`. Exits for 0.1.8b and 0.1.7 are in the same file.

**Scenes where carry and reset differ by more than 0.1:** 5f064b02 (+0.828), bae463d3 (+0.960) and 04749bb9 (+0.707), all in carry's favour. 048b974e is the only scene where reset is ahead, by 0.052.

**How often carry and reset picked differently** (the shadow, on identical input, so this measures the memory at fixed state):

| | calls | scorer pick differs | final pick differs | final pick differs, after handover |
|---|---|---|---|---|
| carry run (shadow = start state) | 468 | 17.3 % | 16.0 % | 17.7 % (396 calls) |
| reset run (shadow = memory carried along the reset drive) | 468 | 19.4 % | 19.0 % | 20.7 % (396 calls) |
| both runs | 936 | 18.4 % | 17.5 % | 19.2 % |

Per scene, the final picks differ from 2.6 % of calls (b1b170d0) to 30.8 % (d2715023, bae463d3).

**Diagnostics:**
- Every run has 39 model calls (33 after handover) and **0 chain resets**.
- Mean memory norm: 60.8–62.3 in the carry runs, against the start state's 63.99.
- Gate mean: 0.20–0.28 when carried, against 0.40 when every call starts from the start state. These match the offline figures, 0.216 and 0.40 (`BUILD5_RECALL_LESSON.md` §3.1 and flag 8).

**Gates and media.** All 24 runs are green on every run-script gate:
- wizard exit;
- 26.01 artifacts only;
- push cadence;
- the 0.1.4 smoke gates;
- the extra gates;
- the replay;
- every logged call was in the run's own mode.

Each run has an annotated video, a replay page and a trace in `runs/memory_<arm>_<scene>/replay/`, with links in `{experiment_store}/runs/cl12/lane/annotated_videos/` (24).

## 5. What it shows and what it does not

- **Verified.** Carrying the memory changed the driven pick on roughly one call in six. On three scenes the drive diverged far enough to change the outcome, and all three went carry's way.
- **Verified.** On 04749bb9, reset makes an at-fault front collision at +6.30 s, which 0.1.8b also makes (at +12.40 s). Carry instead leaves the road at +6.90 s and still scores 0.707.
  - By the lane's rule, off-road is not counted as at-fault.
  - So the "no extra at-fault crash" part of the rule holds partly because carry crashed in a way the rule does not count. Without 04749bb9 the difference is still +0.158 over 11 scenes.
- **Verified.** The reset arm (build 5's weights, memory always at the start state) scores 0.5177. That is below 0.1.8b's 0.5919 on the same scenes, and in line with its lower offline pick grade.
  - Carry, 0.7213, is above both 0.1.8b and 0.1.7 (0.6742), but on 12 scenes, one rollout each.
  - Carry still crashes into the truck on 8023dce5, as 0.1.8b and reset do.
- **Not established:** that the memory makes the driving better in general.
  - The offline measure on 1,131 graded moments says carried is 0.004 below reset (`BUILD5_RECALL_LESSON.md` flag 5).
  - The closed-loop gain rests on two to three scenes.
  - *Argued:* these are near-tie scenes. bae463d3 has scored anywhere from 0.000 to 1.000 across the seven earlier models run on it (`s10_018b_closed_loop/REPORT.md` §3). 5f064b02 is the stall scene, where small pick changes at a standstill decide the outcome. A gain on such scenes from any change to the picks is expected about as often as a loss.
  - What would settle it: the same comparison on the 30 scenes, or repeated rollouts with perturbed inputs.

## 6. Flags

1. **The result rests on two to three scenes.** Removing bae463d3 and 5f064b02 together drops the difference to +0.066, below the bar. There is one rollout per arm and no run-to-run spread is measured.
2. **The image check was changed before any run (amendment 1).** 0.1.7's runner cannot start since its folder moved to the NAS on 2026-11-09: its provenance gate refused at 22:27, and its launcher's `--check-only` fails on the moved path. 0.1.8b was checked instead (PASS). 0.1.7's 0.6742 comes from runs on an earlier image and was not re-checked on this one. 0.1.7 did reproduce bit for bit across the two earlier rebuilds, and 0.1.8b reproduces bit for bit on this one.
3. **Proof A, extra check.** The vehicle simulator's states of the chosen plan, which feed the next call's comfort re-score, differed between the adapter and the 0.1.8b-p06 worker on 1 call of 64:
   - by 1.7e-18 on the first run;
   - by 2.8e-16 on the second, at call 12, 3 values;
   - the same difference again with the shadow switched off (`out/cl12_proof_A_shadow_off.json`).

   Two 0.1.8b-p06 workers against each other: 0 differences (`out/cl12_proof_A_control.json`). On the frozen build: 64/64 equal (`out/cl12_proof_A.json`). The cause is not established; plans, scores and picks were bit-identical in every run. After the first run I made the pass rule the brief's criterion (plans and picks); the first run's file is kept as `out/cl12_proof_A_first_run.json`.
4. **The comfort re-score's input could in principle differ by 1e-16** between the adapter and 0.1.8b (flag 3). Over 39 calls per scene that cannot move a 0/1 comfort flag unless a plan sits exactly on a threshold (*argued*).
5. **Carry's "no extra at-fault crash" on 04749bb9 is an off-road exit**, which the lane's rule does not count (§5).
6. **Long waits at the machine gate.** Most runs took 5–8 minutes; several waited up to 5 h for the load or for another session's run. Run 24 started 5 h after run 23. No run hit the 3-hour wait limit. No run was repeated.
7. **The shadow runs inside the driving call** and adds its time to every call: one extra generator and scorer pass. Measured over all 936 calls: median 90 ms, 90th percentile 1.53 s on a loaded machine (`ms_shadow` in the call lines), against about 2.5–3 s of wall clock between replans. The simulator waits for the driver, so this costs wall time, not simulated time; the divider's cadence and push-cadence gates stayed green on every run.
8. **The offline pick grades** (0.7313 / 0.7355 / 0.7685) are quoted from earlier reports, not re-measured here.
9. **Three launcher fixes during the build**, before the freeze and before any scene:
   - the launcher's adapters path (one level too deep);
   - the shadow switch (offline proofs only; the run script refuses it);
   - the proof script's pass rule (flag 3).

   The proofs were re-run on the frozen build after all three.
10. **The image-check runner refused before launching.** The first queue start ran the 0.1.7 runner, which refused at its provenance gate (exit 6) before launching anything. Nothing was written to earlier lane folders.
11. **Earlier lanes were not edited.** Only new files were written in `adapters/drivejepamemory/`, the memory folder and `{experiment_store}/runs/cl12/`. Earlier adapters and lanes were read only.

## 7. OPEN for Sid

1. **Whether to believe the closed-loop gain.**
   - *Recommend:* run the same two arms on the 30 scenes (the 12 are a subset; about 36 more runs, ~4–6 h on a quiet machine). By the same rule, scaled to 30 scenes (0.033 per scene), it would show whether the gain survives beyond two near-tie scenes.
   - *Alternative:* repeat the three diverging scenes with small input perturbations, to see whether carry's outcomes there are stable.
2. **The offline and closed-loop signs disagree** (offline carried 0.004 below reset; closed loop +0.204).
   - *Recommend:* treat closed loop on 12 scenes as weaker evidence than offline on 1,131 moments until item 1 is done.
   - *Alternative:* look at the three scenes' replays first; the videos are in `runs/memory_*_{5f064b02,bae463d3,04749bb9}/replay/`.
3. **The reset arm is 0.07 below 0.1.8b.** Build 5's weights drive worse than 0.1.8b's without the memory, as offline.
   - *Recommend:* any further closed-loop arm should also carry 0.1.8b itself on the same image (done here by the image check, which passed).
4. **The image check now uses 0.1.8b**, because 0.1.7's runner no longer starts from its NAS folder.
   - *Recommend:* keep 0.1.8b as the image-check model from now on.
   - *Alternative:* a fresh 0.1.7 runner copy outside the lane folder, with your permission.
5. **The truck scene 8023dce5** fails in every arm, carry included. Memory does not change it.

## Files

- **Reports:** `reports/CL12_MEMORY_PREREG.md` (with amendment 1), `reports/CL12_MEMORY.md` (this file).
- **Scripts:**
  - `src/s12_adapter_proofs.py`, `src/s12_proofs_chain.sh`;
  - `src/s12_queue.sh`, `src/s12_run_018bp06_imagecheck.sh`;
  - `src/s12_imagecheck_cmp.py`, `src/s12_table.py`.
- **Outputs:** `out/cl12_proof_{A,B,C}.json`, `out/cl12_proof_A_{control,shadow_off,first_run,second_run}.json`, `out/cl12_imagecheck.json`, `out/cl12_table.{csv,json}`.
- **Logs:** `logs/cl12_queue.log`, `logs/cl12_proofs_chain.log`, `logs/cl12_proof_*.log`, `logs/cl12_check_only_*.log`, `logs/cl12_table.log`.
- **Adapter:** `Paddock/paddock-sid/Alpasim/adapters/drivejepamemory/`, with `launch/` and `launch/tables/`.
- **Runs:** `{experiment_store}/runs/cl12/`, containing:
  - `model/{carry,reset}/`;
  - `proofs/identity_copy/`;
  - `imagecheck_018bp06/`;
  - `runs/memory_{carry,reset}_<scene>/`;
  - `lane/annotated_videos/`.
