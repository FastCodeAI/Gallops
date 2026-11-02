# Step 0 — oracle-pick check on the two stuck scenes: pre-registration

Written 2026-10-12, before the oracle grader was run on any menu of these two scenes and before any closed-loop run.
This file is hashed (`prereg.sha256`) before the first run. Changes after that go in a dated "Amendment" section at the
end, with a new hash, and never by editing the text above it.

## Question

Does the car pull out when the scorer's pick is bypassed? Nothing else.

## What is run

- Driver `drivejepa017_oracle` = adapter 0.1.7 exactly as frozen (`07_closed_loop_test/runs_017/tables/build_017.sha256`; S10
  generator checkpoint sha256 `29d5f951…`, batch fix on, bonnet mask 37 rows, acceleration input zeroed) with ONE change:
  at each replan the plan sent to the controller is the one of the 32 with the highest calculator grade (below), not the
  scorer's top-scored one. Momentum nudge off (`DRIVEJEPA_MOMENTUM=off`). The 0.1.7 package and the vendored tree are
  not edited; the change is a planner subclass in `13_best_possible_pick/adapters/drivejepa017_oracle/`.
- Scenes: d2715023 and b44a4a44 only. Closed loop, same simulator settings as the 0.1.7 trial (runner copied from
  `07_closed_loop_test/runs_017/run_017_scene.sh`, same scene csv, wizard overrides, GPU floor 8000 MiB). One run per scene, no retry.
- Known difference from the 0.1.7 trial that I cannot remove: the simulator base image. The trial ran on
  `alpasim-base:0.134.0` = `1e0d5fea` (no longer on the machine); today's tag is `eca97837` (built 2026-10-12).

## The grade (fixed here)

Per replan, each of the 32 plans is graded live with the S9 part C map-free calculator calls (`11_scorer_head_and_pullout_grades/s9_c_dump.py`
`grade()`: the vendored `train_pdm_scorer.PDMScorer` methods `_reset`, `_calculate_no_at_fault_collision`,
`_calculate_ttc`, `_calculate_history_comfort`, with `_calculate_ego_area` not called), on:

- the car's state as the driver receives it (anchor pose in the simulator's local frame, speed from the model's own
  ego input, acceleration zero, as 0.1.7's own re-score);
- the other road users' boxes replayed by the simulator, read beforehand from the 0.1.7 run's `rollout.asl` of the same
  scene (the simulator replays them from the log; checked equal between earlier runs) and looked up by simulator time,
  including their next 4 s. This is information the driver never has; that is what makes it an oracle;
- the simulator's own ego box (EGO aabb and rig-to-box offset from the log) with the simulator's wheelbase (2.85 m).

S9's grade had no progress term (collision gate x (5 x time buffer + 2 x comfort) / 7), under which standing still
scores 1. The brief names the grade as "collision-free + progress", so the pick uses:

- **collision-free** = the calculator's no-at-fault-collision value equals 1 in S9's *strict* setting ("in more than one
  lane" true at every step, so side contacts count against the car);
- **progress** = straight-line distance, in metres, from the first to the last (4.0 s) simulated position of the car's
  box centre under the calculator's own simulation of the plan; 0 when the end point is behind the start (negative
  forward component in the car's frame at the replan). This replaces the calculator's centre-line projection
  (`train_pdm_scorer.py:548-564`), which needs a map;
- **grade = collision-free x progress** (metres). Pick = highest grade.
- If no plan has grade > 0 (every plan collides, or no collision-free plan moves forward), the scorer's own top plan is
  sent and the replan is logged as `fallback`. Such replans count as "not > 1.5 m" in P0c.

Time buffer, comfort, the lenient collision value and the "full" map-free grade
`collision gate x (5 x time buffer + 2 x comfort + 5 x progress / best progress) / 12` are computed and logged for every
plan at every replan but do NOT enter the pick.

## Definitions for the predictions

- Handover = render start + 3.0 s (force-GT duration in the log). Start position = the car's position at handover.
- Scored window = from handover for `duration_frac_20s x 20` s, from the run's `aggregate/results-summary.json`
  (as `baseline_014b/b014b_readout.py:45-49`).
- "Moves" = largest straight-line distance of the rig origin from the start position inside the scored window.
- "Lateral" = largest |sideways offset| of the rig origin from the start pose, in the start pose's own frame, inside the
  scored window.
- "Step" = one replan (every 0.5 s of simulator time), numbered from 1 at the first replan of the rollout, replans
  before handover included (the oracle picks there too; the simulator ignores the driver until handover).
- "Lateral offset at waypoint 8" = |y| of the picked plan's last waypoint (4.0 s) in the car's frame at that replan.

## Predictions (Sid's, from the brief)

- **P0a** d2715023: the car moves > 4 m from the start within the scored window (0.1.7: score 0.04, no movement).
- **P0b** b44a4a44: the car leaves the bay: lateral > 2 m within the scored window.
- **P0c** at each step the best-graded plan has lateral offset > 1.5 m at waypoint 8, in > 50 % of steps. Counted over
  all replans from handover to the end of the rollout, per scene and pooled; P0c holds if the pooled share is > 50 %.
  The share inside the scored window is reported next to it.

## Reported per scene

Did it move, how far, the score (`rollouts[].score`), a 10-step excerpt of the log, and whether the menu changes across
steps while the car is stationary: widest lateral offset (largest |y| at waypoint 8 over the 32 plans) at steps 1, 20
and 40 (the last step, if the rollout has fewer than 40).

## Verdict rule

- "oracle moves, scorer doesn't": P0a and P0b both hold.
- "oracle doesn't move either": P0a and P0b both fail. Then report which of: picked plans too short, controller ignores
  them, menu collapses in closed loop. No training is started.
- One holds and one fails: reported as split, per scene, with the same diagnosis for the scene that did not move.

No training in any case. Read-only on the models.

## Amendment 1 — 2026-10-12, before any closed-loop run

Found in the offline build gate (`logs/gate_offline.log`, the oracle planner replayed over the recorded, stationary 0.1.7 drive
of d2715023; no simulator run had been made): the simulator's actor log ends with the rollout (+16.8 s after handover), and
the first grader dropped actors beyond it. From about +13 s the calculator's 4 s look-ahead was therefore cut short, plans
that reach the parked car after the end of the log graded collision-free (32 of 32 at +15.5 s), and the pick turned wide
and fast for that reason alone. That would have counted toward P0a and P0c as an artefact.

Change: every actor present at the last logged step is held at its last pose, speed zero, for 4.5 s past the end of the
log, so every replan is graded over a full 4 s. Exact for parked vehicles; for moving ones it freezes them where the log
ends. Nothing else in the grade, the pick rule, the definitions or the predictions changes. The step log's
`future_frames` still counts logged frames, and the readout also gives P0c for the steps whose 4 s lie inside the log.
