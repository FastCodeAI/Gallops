# Step 0 — oracle-pick check on d2715023 and b44a4a44 (2026-10-12)

Citations: `F:n` = `tables/facts.txt:n`; `C:n` = `tables/b44a4a44_collision.txt:n`; `G:n` = `tables/gate_offline.json:n`.
Step logs: `{runs}/drivejepa_retrain/13_best_possible_pick/steplog/<scene>_oracle_steps.jsonl` (line = step). Videos: `…/13_best_possible_pick/videos/`.
Run folders: `…/13_best_possible_pick/runs/drivejepa_017oracle_<scene>/`. Pre-registration: `PREREG.md` (hash in `prereg.sha256`; original text hash in `tables/prereg_original.sha256`).

## Verdict

**"oracle doesn't move either"** by the pre-registered rule: P0a, P0b and P0c all fail (F:138, F:139, F:140). No training started.
The two scenes are not the same case:

- **d2715023 does not pull out.** 2.28 m in the scored window against 1.64 m for 0.1.7 (F:11, F:34). Of the brief's three causes:
  the picked plans are too short, because no collision-free plan on the menu leaves. The controller follows them and the menu does not collapse.
- **b44a4a44 does move** (above 1 m/s at +0.95 s, 168 m by the end; F:88, F:82) but P0b fails as written: it touches an oncoming car at
  +2.15 s, which ends the scored window with 3.0 m driven and 0.26 m sideways (F:73, F:79, F:80). Score 0, as 0.1.7 (F:70, F:93).

## Per scene

| | d2715023 oracle | d2715023 0.1.7 | b44a4a44 oracle | b44a4a44 0.1.7 |
|---|---|---|---|---|
| score | 0.057 (F:2) | 0.041 (F:25) | 0 (F:70) | 0 (F:93) |
| scored window | 16.8 s (F:5) | 16.8 s (F:28) | 2.15 s, collision at fault (F:73, F:72) | 6.6 s, collision at fault (F:96, F:95) |
| farthest from start, in the window | 2.28 m (F:11) | 1.64 m (F:34) | 3.01 m (F:79) | 0.03 m (F:102) |
| largest sideways offset, in the window | 0.25 m (F:12) | 0.09 m (F:35) | 0.26 m (F:80) | 0.00 m (F:103) |
| farthest from start, whole rollout | 2.28 m (F:14) | 1.64 m (F:37) | 167.6 m (F:82) | 0.10 m (F:105) |
| top speed after handover | 0.72 m/s (F:21) | 0.63 m/s (F:44) | 27.9 m/s (F:89) | 0.02 m/s (F:112) |
| best-graded plan wider than 1.5 m at waypoint 8, steps from handover | 0 of 34 (F:50, F:48) | | 3 of 34 (F:118, F:116) | |
| steps with no collision-free moving plan (scorer's pick sent) | 0 (F:55) | | 20 of 34 (F:123) | |

P0c pooled: 3 of 68 steps, 4.4 % (F:137).

### d2715023
- The car creeps 2.3 m toward the parked car and stops. Collision-free plans per menu: median 4.5 of 32; the best of them makes 0.29 m of
  progress in 4 s (median) and ends 0.28 m away (F:57, F:58, F:60).
- A collision-free plan wider than 1.5 m exists on 0 of 34 menus (F:61). So the grade never had a pull-out to pick.
- Controller: the picked plans ask 0.16 m/s over their first half second and the car does 0.14 m/s (F:66).
- Menu while standing (under 0.1 m/s, 26 steps): widest sideways offset 1.12 / 3.97 / 7.65 m (min / median / max, F:65). Steps 1, 20, 39 (the
  rollout has 39 steps): 7.65 m, 1.12 m, 4.05 m (F:62, F:63, F:64). It changes from step to step; it does not shrink to nothing.
- Excerpt, steps 6–15: `tables/excerpt_d2715023.txt:1-10`.

### b44a4a44
- At handover the grade picks slot 7 (12.2 m ahead, 1.28 m left at waypoint 8) where the scorer picks a stay plan (`tables/excerpt_b44a4a44.txt:1`).
- Contact: simulator reports a front collision at +2.15 s (C:1, C:3). The boxes first overlap at +2.00 s with actor 25, a car coming the
  other way (heading 161–180° from ours, C:22-26) whose centre passes 2.0–2.2 m to our left (C:23-26); the two half-widths add up to 2.06 m
  (C:7, C:29). The grader called all 32 plans collision-free at +1.5 s and none at +2.0 s (C:18, C:19): it did not see this contact coming.
- After the window the car runs to 27.9 m/s, leaves the road at +8.4 s (C:6) and is steered by the scorer's own pick on 20 of 34 steps (F:123).
  "Highest progress" has no speed limit; nothing after +2.15 s is evidence about pulling out.
- Menu while standing: only 6 steps, widest 1.44 / 1.54 / 1.70 m (F:133). Steps 1, 20, 39: 1.52 m (standing), 2.26 m (at 20.7 m/s), 8.87 m (F:130-132).
- Excerpt, steps 6–15: `tables/excerpt_b44a4a44.txt:1-10`.

## What was built and checked

- One change only: offline replay of the two recorded 0.1.7 runs through the oracle planner gives the 0.1.7 menus to 0.05 mm (the step
  log's rounding) and the 0.1.7 scorer pick at 39 of 39 replans, both scenes (G:4, G:5, G:18, G:19); car pose to the grader within 0.04 mm (G:7, G:21).
- Both runs: all gates green, driver id `drivejepa-v2-driver-0.1.7-oracle`, actors replayed equal to the file the grader read (F:67, F:135, F:1, F:69; `logs/sweep.log`).
- Grade as pre-registered: collision-free (calculator, strict) x progress in metres. S9's grade had no progress term; I added one. Amendment 1
  (before any closed-loop run): actors are held in place past the end of the log.

## Flags

1. The "0.1.7: no movement" premise is off for d2715023: 0.1.7 itself moved 1.64 m (F:34).
2. The calculator's collision check and the simulator disagree on b44a4a44 at +1.5 s (above). Against an independent box-overlap check the
   grader's flag agrees on 81 % (d2715023) and 72 % (b44a4a44) of plans (G:8, G:22). The oracle is only as good as that check.
3. P0b's 2 m sideways test does not fit this scene's exit: the oracle left mostly straight ahead and reached 2 m sideways only at +8.55 s (F:87).
4. Simulator base image differs from the 0.1.7 trial's (`eca97837` today, `1e0d5fea` then; `logs/sweep.log:1`).
5. One run per scene.
