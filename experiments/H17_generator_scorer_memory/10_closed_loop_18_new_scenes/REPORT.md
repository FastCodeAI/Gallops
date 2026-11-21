# Closed loop on the 18 remaining scenes: memory carried vs 0.1.8b pass 0.6

2026-11-14, shakaal. Runs 2026-11-14 01:20 to 06:11 IST. Pre-registration: `reports/CL18_MEMORY_PREREG.md` (written 2026-11-13 21:41 IST, before any of these scenes; sha256 `53076902…` logged by the queue). Continues `reports/CL12_MEMORY.md`. Numbers are **measured** from the run outputs unless marked *argued*.

## Answer by the pre-registered rule: **HURTS**

| | 18-scene mean | at-fault crashes |
|---|---|---|
| **carry** (build 5, memory carried) | **0.8865** | 0245ff75 (front, +15.75 s), 032b6f21 (front, +15.50 s) |
| **0.1.8b pass 0.6** | **0.8599** | 026d6a39 (side, +14.55 s) |
| 0.1.7 (earlier image, reference only) | 0.8333 | 026d6a39, acd22430, b44a4a44 |

- **The mean says no detectable effect.** d = carry − 0.1.8b = **+0.027**, inside ±0.08.
- **The verdict is HURTS on the crash clause.** Carry has two at-fault crashes on scenes where 0.1.8b has none: 0245ff75 and 032b6f21. By the rule, that clause wins.
- **Both crashes come at the very end of the rollout**, at +15.50 and +15.75 s after handover. The last metric tick is at +16.75 / +16.85 s. Both runs are still scored 1.000, status "pass", because the score's window has ended by then.
- **0.1.8b's own crash** (026d6a39, +14.55 s) is just as late and also scores 1.000.
- **The rule counts these crashes as written:** the exit code the 12-scene check used, with no time limit. See flag 1.

Sources: `out/cl18_table.json`, `out/cl18_table.csv` (`src/s18_table.py`, log `logs/cl18_table.log`).

## Scenes

**Where the 30 come from:** the 0.1.8 lane's `s4_closed_loop/tables/scenes_30.txt` (sha256 `84d2a439…`). That lane's 30 run folders name exactly these clips.

**The 18 are the 30 minus the 12 diagnosis scenes:**

`01d503d4 023b7fcc 0245ff75 026d6a39 02e075b9 02eadd92 032b6f21 04394343 16b83346 1acf2f5d 2e1ce314 6725df2d 71d7d03b 791d68ce 89f2eeb8 98dd56b2 acd22430 b44a4a44`

The list with clip ids is saved as `out/cl18_scenes_18.txt` (sha256 `8a763913…`).

## Per scene (18 new)

Score and exit use `s5_diagnosis/src/stepb.py` `score` / `exit_of`, unchanged. The exit is the first of front collision, side collision, off-road and hit-from-behind, with its time after handover. At-fault means a front or side collision.

| scene | carry | 0.1.8b | 0.1.7 * | carry exit | 0.1.8b exit | 0.1.7 exit * |
|---|---|---|---|---|---|---|
| 01d503d4 | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 023b7fcc | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 0245ff75 | 1.000 | 1.000 | 1.000 | **at-fault front +15.75 s** | pass | pass |
| 026d6a39 | 1.000 | 1.000 | 1.000 | off-road +11.20 s | **at-fault side +14.55 s** | at-fault front +10.75 s |
| 02e075b9 | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 02eadd92 | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 032b6f21 | 1.000 | 1.000 | 1.000 | **at-fault front +15.50 s** | pass | pass |
| 04394343 | 1.000 | 1.000 | 1.000 | off-road +11.65 s | pass | pass |
| 16b83346 | 1.000 | 1.000 | 1.000 | off-road +11.65 s | off-road +14.15 s | pass |
| 1acf2f5d | **1.000** | **0.835** | 1.000 | pass | pass | pass |
| 2e1ce314 | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 6725df2d | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 71d7d03b | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 791d68ce | 0.000 | 0.000 | 0.000 | off-road −3.21 s | off-road −3.21 s | off-road −3.21 s |
| 89f2eeb8 | 1.000 | 1.000 | 1.000 | pass | pass | pass |
| 98dd56b2 | **1.000** | **0.791** | 1.000 | pass | hit from behind +12.60 s | pass |
| acd22430 | 0.000 | 0.000 | 0.000 | off-road +4.70 s | off-road +7.00 s | at-fault front +15.40 s |
| b44a4a44 | **0.957** | **0.852** | 0.000 | pass | pass | at-fault side +6.60 s |
| **mean** | **0.8865** | **0.8599** | 0.8333 | | | |

\* 0.1.7: its existing runs (`{dataset_store}/s5_trial_017/runs/drivejepa_017_*`), **earlier image**, not re-run.

## Robustness (reported, not deciding)

- **Without the two scenes most in carry's favour** (98dd56b2 +0.209 and 1acf2f5d +0.165), d over the other 16 is **+0.007**.
- **Wins, losses, ties** (counting a difference above 0.1): carry wins 3 (1acf2f5d, 98dd56b2, b44a4a44), loses 0, ties 15.
- **Off-road exits:** carry 5 (026d6a39, 04394343, 16b83346, 791d68ce, acd22430); 0.1.8b 3 (16b83346, 791d68ce, acd22430).
  - 791d68ce leaves the road at −3.21 s, before handover, in every arm and in 0.1.7, so it is not the driver's doing (flag 4).
  - Without it, carry has 4 off-road exits and 0.1.8b has 2. Carry's three extra exits (026d6a39, 04394343, 16b83346) all come at +11.2 s or later, and all three runs still score 1.000.
- **Shadow (carry runs)**, the share of calls where the start-state memory would have picked differently on the same input: final pick 12.7 %, scorer pick 13.2 %, over 702 calls. That compares with 16.0 % / 17.3 % on the 12 diagnosis scenes. There were 0 chain resets.

## 30 scenes (secondary: the 12 from the 12-scene check plus the 18 new)

| | 30-scene mean | at-fault | off-road |
|---|---|---|---|
| carry | **0.8204** | 4 | 7 |
| 0.1.8b pass 0.6 | **0.7527** | 4 | 4 |
| 0.1.7 (earlier image) | 0.7697 | 4 | 2 |

- **Where the runs come from:**
  - carry's 12 diagnosis scenes are the 12-scene check's carry runs;
  - 0.1.8b's 12 are its 2026-10-31 runs, which reproduced bit for bit on this image (four re-runs in `reports/CL12_MEMORY.md` §3);
  - 0.1.7's 30 are its existing runs, earlier image (0.7697 equals its published 30-scene figure).
- This is **the first 30-scene number for 0.1.8b pass 0.6**: 0.7527, below 0.1.7.
- **Carry's 30-scene lead over 0.1.8b (+0.068) comes almost entirely from the 12 diagnosis scenes:**
  - 18 new scenes: +0.027;
  - 12 diagnosis scenes: +0.129 (bae463d3, b1b170d0 and 04749bb9).
- **At-fault crashes, 30 scenes:**
  - carry 4: 8023dce5, bd07fb91, 0245ff75, 032b6f21;
  - 0.1.8b 4: 8023dce5, bd07fb91, 04749bb9, 026d6a39.

## Setup (as pre-registered)

- **Carry arm:**
  - the frozen `adapters/drivejepamemory` (0.1.8b-p06-memory5), mode carry from the 12-scene sidecar;
  - the build freeze `launch/tables/build_memory.sha256` was checked by the queue and by every run;
  - runner `src/s18_run_memory_scene.sh`, which is the frozen run script with two lines changed: the output root and the model folder.
- **0.1.8b arm:**
  - its own `adapters/drivejepa018bp06`;
  - runner `src/s18_run_018bp06_scene.sh`, which is its run script with one line changed: the output root.
  - The queue refused to start if either runner differed by more.
- **Shadow on in the carry arm.** The frozen run script refuses to switch it off for real runs (flag 3).
- **Simulator image:** `alpasim-base:0.134.0` = `sha256:9c5909da…f8881`. It was checked before every run and after the queue, and was unchanged throughout.
- **CPU pinning:** all CPU-side work ran under `taskset -c 0-1` and `nice 19` (the queue, wizard, driver and model worker).
  - Checked before the pre-registration: both workers give bit-identical plans and scores pinned and unpinned (`out/cl18_thread_check.json`). It cost about 2.1 s per call, against 1.5–1.8 s unpinned.
- **Run checks:** all 36 runs are green on every run-script gate.
- **Media:** 36 annotated videos with replay pages, under:
  - `{experiment_store}/runs/cl18/runs/memory_carry_*/replay/`;
  - `{experiment_store}/runs/cl18/018bp06/runs/drivejepa_018bp06_*/replay/`;
  - links in `cl18/lane/annotated_videos/` and `cl18/018bp06/lane/annotated_videos/`.
- **Queue:** `src/s18_queue.sh`, log `logs/cl18_queue.log`. One rollout per scene and arm; no run was repeated.

## Flags

1. **The HURTS verdict rests on two crashes the score does not see.**
   - Both of carry's extra crashes, and 0.1.8b's own 026d6a39 crash, fall between +14.5 s and +15.8 s, in the last ~1.3–2.3 s of a rollout whose last metric tick is at about +16.8 s.
   - All of these runs score 1.000 and report "pass".
   - The rule takes at-fault from the exit code with no time limit, as in the 12-scene check, where every at-fault crash was before +7 s.
   - I did not change the rule.
   - *Argued:* under a rule limited to the scored window, the verdict would be NO DETECTABLE EFFECT. That is information, not the result.
2. **Without the at-fault clause, the 18 scenes show no detectable effect.** d is +0.027, falling to +0.007 without the two best scenes, with 3 wins, 0 losses and 15 ties. The 12-scene gain (+0.204 against reset, +0.129 against 0.1.8b) does not carry over to these 18 scenes.
3. **Shadow on.** Carry runs evaluate the other mode once per call (median about 0.1 s). It does not touch the driven plan (12-scene proof A).
4. **791d68ce is off-road before handover** (−3.21 s), in carry, 0.1.8b and 0.1.7 alike: a property of the scene start, not of any driver. It scores 0 for everyone.
5. **The exit column shows only the first event.**
   - Carry on 026d6a39 leaves the road at +11.20 s. Any later collision in that run is not shown, and the same holds for every run.
   - *Argued:* this cuts both ways between arms.
6. **0.1.7 is on an earlier image.** Its 18-scene and 30-scene figures are beside the others for reference only.
7. **Pinned and unpinned runs are mixed in the 30-scene table.** The 18 new runs were pinned to 2 cores; the 12 diagnosis-scene runs were not. The thread check shows identical outputs on fixed inputs; that is not a closed-loop proof.
8. **The 0.1.8b 30-scene figure mixes runs from 2026-10-31 and 2026-11-14.** The older 12 reproduced bit for bit on this image on four of the 12 scenes; the other eight were not re-run.
9. **Start delay.** The queue waited at the machine gate for about 3.5 h while another of our runs (the Alpamayo Edge 30-scene run) held the simulator. It timed out once at 00:42 and was restarted by a watcher (`src/s18_relaunch.sh`, log `logs/cl18_relaunch.log`) that restarts only after that time-out. Nothing ran before 01:20.

## OPEN for Sid

1. **How to read late crashes in the rule.**
   - *Recommend:* keep this verdict as recorded (HURTS, by the pre-registered rule), and for future pre-registrations count at-fault only inside the scored window, or report both.
   - *Alternative:* accept the narrower reading now and record it as a post-hoc reading.
2. **What the two closed-loop checks together say.**
   - *Recommend:* treat the memory's closed-loop effect as not shown.
     - On the 12 scenes it came from two or three near-tie scenes.
     - On 18 new scenes the mean gap is +0.027.
     - Offline it was slightly negative.
     - It now adds two late crashes.
   - Go back to the open design question from build 5: nothing teaches the generator or scorer to use the memory.
3. **Watch the three new carry crashes / exits.** 0245ff75 and 032b6f21 (late front collisions) and the three late off-road exits are carry-only. *Recommend:* look at those replays before any further memory run.

## Files

- **Reports:** `reports/CL18_MEMORY_PREREG.md`, `reports/CL18_MEMORY.md`.
- **Scripts:** `src/s18_thread_check.py`, `src/s18_run_memory_scene.sh`, `src/s18_run_018bp06_scene.sh`, `src/s18_queue.sh`, `src/s18_relaunch.sh`, `src/s18_table.py`.
- **Outputs:** `out/cl18_scenes_18.txt`, `out/cl18_thread_check.json`, `out/cl18_thread_{all,two}.npz`, `out/cl18_table.{csv,json}`.
- **Logs:** `logs/cl18_queue.log`, `logs/cl18_relaunch.log`, `logs/cl18_thread_*.log`, `logs/cl18_table.log`.
- **Runs:** `{experiment_store}/runs/cl18/`.
