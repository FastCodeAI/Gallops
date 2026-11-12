# CLAUDE.md: Bench2Drive Drive-JEPA, Phase 2 — Bench2Drive grade, then Drive-JEPA training (v2)

(Given by Navvrat on 05 Nov 2026, verbatim. v1 is CLAUDE_v1_05nov.md.)

**v2 changes:** progress is measured against the expert's distance, not the best clean path. Outside lanes is a share of the route, not of the 4 s path. The stop-sign threshold is the agent's 1.0 m/s. The expert's collisions are checked against its log. Regrade Step A, report, then go on to Step B (checks 1–3 are done).

**What:** two steps, in order.
- **Step A:** grade the 2,048 vocabulary with Bench2Drive's own scoring terms, on every training and held-out frame, and write the teacher index.
- **Step B:** Drive-JEPA's Stage 2 training on Bench2Drive (generator + scorer), on the Phase 1 encoder, with the Step A grade.

Report numbers only, no thresholds or verdicts: Navvrat decides.

## Step A: the Bench2Drive grade

Done in v1: the 17.06 m/s path is real motion (`b2d_2048.npy` unchanged); all 944 clips have scorer context.

**The grade of one path over 4 s,** with Bench2Drive's coefficients (`statistics_manager.py`):

```
S = P × 0.5^n_ped × 0.6^n_veh × 0.65^n_static × 0.7^n_red × 0.8^n_stop × (1 − outside)

n_x = how many times the path commits that infraction in 4 s
```

| Term | |
|---|---|
| **P, progress** | min(1, d / d_expert). d: metres gained along the expert's 4 s path, which stands in for the route. d_expert: the expert's own distance. If the expert moves under 0.5 m, P = 1 for every path |
| **Collisions** | Counted by type: pedestrian 0.5, vehicle 0.6, static 0.65. Keep the reactive follower model. Check the expert's hits against its log, as for stop signs: report how many match, and where the ego box sits after the rear-axle shift |
| **Red light** | 0.7, the current port of Bench2Drive's test |
| **Stop sign** | 0.8, ported from Bench2Drive's stop-sign test. A stop counts when a 0.5 s segment is under 1.0 m/s (the agent's choice: it matches the expert's log, 11 vs 12) |
| **Outside lanes** | As Bench2Drive's outside-lanes test, a share of the route: 1 − (the path's distance off lanes or in oncoming lanes ÷ the route's length). Route length: the expert's distance over the whole clip. Lanes as Bench2Drive counts them, Parking included |
| **Left out** | Yield to emergency vehicle, blocked, timeouts and the tick cap: a 4 s window can't judge them. Bench2Drive doesn't use minimum speed |

**Teachers:** feasible paths (the current filter, MTD's) with S ≥ 0.95, capped at 256 per frame, as Drive-JEPA.

**Run:** the 2,048 on all 36,897 training frames and all 2,251 held-out frames. Save, per frame, every graded path's index and each term. Write the teacher index file to a new path.

**Report:**
- **The expert's own grade:** the share of frames where it reaches 0.95, and for the rest, which term holds it back.
- **Teacher supply:** frames with at least one and at least four teachers, for all frames, straight, turns and stopped, on training and held-out.
- **Terms:** how often each one fires among feasible paths.
- **Three frames where the expert crosses into the oncoming lane to pass an obstacle:** the grade of the expert's path, of the best path that stops, and the teachers, under v1 and v2.

## Step B: Drive-JEPA training

As Drive-JEPA's Stage 2, changed only where Bench2Drive needs it.

| | |
|---|---|
| **Encoder** | The Phase 1 merged encoder (pass 3, sha256 `339d7442…`), trained at Drive-JEPA's backbone rate, × 0.1 |
| **Planner** | Drive-JEPA's generator and scorer, from scratch |
| **Settings** | Drive-JEPA's: Adam 1e-4 (backbone 1e-5), no scheduler, 20 epochs, batch 64, 16-mixed |
| **Losses** | Drive-JEPA's, with its weights |
| **Teachers** | From Step A. K = 4 drawn per round, as Drive-JEPA |
| **Scorer labels** | As Drive-JEPA: its 32 live proposals graded every step, with the Step A grade in place of NAVSIM's (progress against the expert's distance at that frame). Each scorer head takes the matching term; report the mapping and any head without one |
| **Vocabulary** | The 2,048 from Step A. Drive-JEPA's code takes 8 of its 40 poses |
| **Input** | As Phase 1: front camera, 2 frames 0.5 s apart, crop `native`. Ego status as the runner sends it: speed (v, 0) and acceleration (dv/dt, 0), not the labels' accelerometer reading. Command from the runner's own function, with FOLLOW sent as straight, as `wajepa_dj/policy.py:94-104` does |
| **Data** | The LoRA run's 897 training clips. Its 47 held-out clips for validation only |
| **GPU** | Alongside run 3A-M if both fit. Run 3A-M has priority: if they don't fit, ask Navvrat |

**Checks before the long run:**
1. **Encoder:** the loaded encoder's sha256 matches Phase 1's merged checkpoint. Done.
2. **Grade in the loop:** a vocabulary path graded during training gets the same S and terms as in Step A. Rerun it on the v2 grade and report the largest difference.
3. **Inputs:** the crop is bit-identical to Phase 1's, and the command mix is reported (done). Also report the 1st and 99th percentiles of speed and acceleration, and where the extremes come from (v1 found −2.8 m/s, and −27 to +19 m/s²).
4. **First 50 steps, beside run 3A-M:** step time with the grade in the loop, peak memory, loader wait, every loss finite. Estimate the total time.
5. **Resume:** as Phase 1: optimiser, sampler and RNG state saved to new paths; report the next 10 losses after a resume.

**Every epoch, on the held-out clips:** each loss; closest-candidate and scorer-pick ADE (3 s); the mean grade of the scorer's pick next to the expert's; menu variety. Save a checkpoint every epoch to a new path. Navvrat picks the checkpoint.

Closed-loop evaluation comes in its own md after training.

## Rules

- Never train or tune on Bench2Drive's 220 eval routes, the v0.0.4 validation routes, or HUGSIM scenes.
- New files in `b2d_drivejepa/phase2/`. Import Drive-JEPA's, Phase 1's and the current grader's code; don't edit it.
- Never overwrite a checkpoint, cache or earlier result; write to new paths.
- Log in `agent_memory/<date>.md`. Ask before downloading or installing anything, and before changing anything in this file.
