# Drive-JEPA after retraining: why it still swerves at highway speed and still does not pull out

Written 2026-10-09 by the loop "brain" from six worker tasks. Offline and CPU only; no new simulator runs.
Evidence folders (all under this folder): `part1_inventory/`, `part2_why_it_swerves/`, `part3_swerve_fixes/`, `part4_why_it_stays_stopped/`, `part5_stuck_fixes/`.
Words used: the *generator* is the part of the model that proposes 32 candidate paths (the "menu") twice a second; the *scorer* is the part that gives each a score and picks the top one; the *controller* is the simulator's own steering-and-braking controller, shared by every model we test; *handover* is the moment, 3 s into each clip, when the model takes over from the replayed human drive.

## The short answer

1. **Highway swerve (scenes 04749bb9, acd22430).** It is a braking problem that the controller turns into steering. The retrained model's chosen plan starts about 3 m/s slower than the car is travelling. The controller cannot shed that speed instantly, and above roughly 15 m/s it answers any plan that needs more than about 5 m/s² of braking with a steering swing. The fault starts in the model (the scorer picks a too-slow plan; the generator's menu is ragged at this speed), and the controller is what makes it violent. It is not our harness or adapter.
2. **Red-light scene (bd07fb91).** Scored perfect, but only because scoring stopped early. The model holds 14 m/s toward a red light with queued cars, then asks for 9 m/s² of braking, swerves, and hits a car 0.7 s after scoring ends. Late braking by the model, then the same controller behaviour.
3. **Car that never pulls out (b44a4a44, d2715023).** Not the harness: the car moves when given a pull-out plan. On b44a4a44 the generator offers a good pull-out most of the time and the scorer prefers "stay" — scorer fault. On d2715023 the generator's pull-outs are too shallow and run into the parked car, and the scorer rejects every pull-out regardless of how much room it has — both at fault.
4. **Your belief that both generator and scorer are wrong** holds, with this split: scorer first on the highway and on b44a4a44; generator and scorer together on d2715023; plus a controller share on every swerve.

## 1. What was compared (verified from disk)

- After: 30 runs `runs/drivejepa/drivejepa_016_<scene>/`, adapter 0.1.6, retrained weights `drive_jepa_s4_pass09.ckpt` (checksum recomputed, starts `8a71c3c0`), bonnet mask on, acceleration input zeroed (also in the adapter's second "momentum" scoring step, which the plan did not include but the build did).
- Before: adapter 0.1.4b, 30 runs on the HDD under `baseline_014b/runs/b014b_<scene>_on`. Simulator settings identical on all 30 scenes. The original 0.1.4 is a different, worse baseline (7 at-fault collisions) and is kept as a separate column. No reranker (0.1.5) run is in either set.
- One run per scene on each side, so differences of a few hundredths are not meaningful.
- Details: `part1_inventory/INVENTORY.md`, `part1_inventory/honest_table.csv`.

## 2. Before and after, on the honest measures (n = 30)

| measure | before (0.1.4b) | after (0.1.6) |
|---|---|---|
| route progress at the end of the scored part | 0.537 (scored 61 % of the clip) | 0.698 (scored 72 %) |
| at-fault collisions inside the scored part | 4 | 2 (8023dce5, b44a4a44) |
| at-fault collisions over the whole clip | 4 | 3 (adds bd07fb91) |
| scenes leaving the road, whole clip | 6 | 3 (04749bb9, acd22430, 791d68ce) |
| distance driven after handover | 72 m | 110 m (human 144 m) |

- Improved: 17 scenes gain progress; the hard brake after handover is gone on the 15–17 m/s scenes (04394343, b1b170d0).
- Not improved: 04749bb9 (off the road at +2.7 s), acd22430 (+0.25 s), d2715023 (moves 1.23 m; human 44 m), b44a4a44 (0.07 m; human 100 m).
- Worse: e121e37d, 0245ff75, bae463d3, 1acf2f5d, cdd9448c lose progress; none gains a collision.
- Cautions about the measures themselves: 0.15 of the progress number is earned by the replayed human before the model drives; scene 791d68ce is off the road during the human replay and scores 0 for any driver; on b44a4a44 the "at-fault collision" is replayed pedestrians walking into the stationary car; the constant-speed dummy driver (reference only, it ran with a different setting) scores 1.00 on 04749bb9 where Drive-JEPA scores 0.28.

## 3. Highway swerve

### What was tested, with the prediction written first (`part2_why_it_swerves/README.md`)

| hypothesis | test | predicted | measured |
|---|---|---|---|
| our harness causes it | replay the recorded plans through an offline copy of the controller | reproduces the run | reproduced to 15 decimal places; steering passes 0.3 rad at +0.10 s as recorded. Harness ruled out |
| braking causes it | same path, re-timed to hold speed | steering falls > 90 % | 0.525 → 0.003 rad (04749bb9), 0.401 → 0.002 (acd22430) |
| the plan's heading causes it | replace heading only | little change | no change at all (0.525 → 0.525) |
| the sudden first step causes it, not the overall slowdown | same path and same 4 s end point, smooth deceleration | steering under 0.1 rad | 0.005 and 0.001 rad |
| a good plan is missing from the menu | count proposals that hold speed and stay under the controller's limit | rare | wrong at handover: 15 and 12 of 32 qualify, best scorer rank 5 and 8, and they steer 0.02 rad. Right half a second later: 1 of 32 |
| bd07fb91 is the same thing | look at the menu at +6.0 s | a straight plan exists and loses | wrong: all 32 plans slow and go right toward the human's lane; red lights and queued cars are in view; the model simply braked late |

- The controller's response at 23.5 m/s (peak steering against steady braking asked): 0.01 rad at 4 m/s², 0.06–0.12 at 5, 0.25 at 6, 0.45 at 7, 0.7 and above from 8. At 15.7 m/s it jumps between 6 and 7. At 5 m/s there is none up to 12. (`part2_why_it_swerves/braking_sweep.csv`)
- An earlier report understated the braking asked for: a plan whose first half-second is 3 m/s slow needs the equivalent of about 13 m/s² to follow, not 6.5.
- Once the first swing has started, even the human's own path or a straight constant-speed plan cannot stop it for the next second.

### Verdict
Starts in the **scorer** at handover (it ranks first a plan with a 3 m/s speed drop while speed-holding plans sit at rank 5–8), with a **generator** share (at 20 m/s and above its plans have ragged spacing, up to 17 m/s between steps of one plan, and on later ticks almost no acceptable plan exists), and is turned into a swerve by the **controller**. Acceleration input: not involved (it is zeroed and the behaviour is there). Harness and adapter: ruled out by exact replay; the plan the controller receives equals the model's raw waypoints to 0.03 mm.

### Proposed fix and what the offline test showed (`part3_swerve_fixes/README.md`)
- **Fix A, recommended: smooth the speed profile of the chosen plan in our adapter, above 10 m/s only.** Same path, same end point after 4 s, but a steady deceleration from the car's actual speed instead of a sudden step. Offline: 0.525 → 0.005 rad and 0.401 → 0.001 rad at the two highway handovers; 0.78 → 0.08 at the bd07fb91 red light. Of 262 undisturbed plans at 10 m/s or more, 3 swung past 0.3 rad before and 0 after. Below 10 m/s it should stay off (it made 4 of 579 crawling-speed plans swing instead of 1).
- Fix B, not recommended: filtering the menu fixes 04749bb9 (0.019 rad) but not acd22430 (0.289), and at the red light it picks a plan that stops 15 m later.
- Fix C, for the simulator's owners: adding a penalty on fast steering changes in the controller gives 0.003 / 0.002 / 0.151 rad on the same three plans, unmodified. It costs up to 0.11 m of sideways accuracy in slow turns; not tested on fast turns because no such scene exists in our 30.
- **What this does not show.** Only 3 undisturbed test moments exist. After the first changed plan the car would be somewhere else and the model would see different pictures, so the rest of the drive cannot be predicted offline. Fix A removes the swing, not the slowness: the model still wants about 19 m/s where the human does 23.5. That part needs retraining (the planned spacing term for the generator, `analysis_3.0/Retrain-data/11_scorer_head_and_pullout_grades/A_spec.md`) and a scorer that stops preferring slow plans on the highway.

## 4. The car that never pulls out

### What was tested (`part4_why_it_stays_stopped/README.md`, `part5_stuck_fixes/README.md`)

| hypothesis | test | predicted | measured |
|---|---|---|---|
| our harness holds the car | feed a pull-out plan from the menu to the controller from standstill | car moves > 1 m in 2 s | 1.6–3.1 m on b44a4a44, 2.2 m on d2715023. Harness ruled out; the model's inputs are also correct (speed right, frames differ) |
| staying is the right thing to do | when does the human leave | within 3 s | +1.0 s (d2715023), +3.0 s (b44a4a44) |
| the menu has no usable pull-out on d2715023 | check every proposal against the recorded positions of other road users | usable on < 25 % of moments | 56 % of the 34 moments with zero margin; with a 0.3 m margin 9 % of all 34, and 0 of the 26 moments where the full 4 s could be checked. 164 of 203 prompt pull-outs run into the parked car; they end 1.7 m to the left where the human goes 4 m |
| the menu has a usable pull-out on b44a4a44 | same | ≥ 75 % | 26 of 34 moments (85 % before the pedestrians arrive), close to the human's own path |
| the scorer ranks the pull-out below staying | scores | rank worse than 5 | rank 16 of 32, score 0.67 against 0.93 for staying (b44a4a44); 0.06 against 0.95 (d2715023). Retraining narrowed the gap on b44a4a44 (0.51 → 0.22) and widened it on d2715023 |
| the scorer would accept a proper wide pull-out | put the human's path into the model | cannot be asked directly | confirmed from the code: the scorer never reads the path, only a learned feature tied to each of the 32 menu positions. A human pull-out placed in a "leave" position scores 0.15, in a "stay" position 0.79. Scores do not rise with room from obstacles (0.50 touching, 0.55 with over 1 m, b44a4a44) |
| the same failure occurs elsewhere | scan all 30 scenes | at least one more | none; the pattern is these two scenes only |

### Verdict
- **b44a4a44: scorer.** A prompt, contact-free, human-like pull-out is on the menu and loses on score at every moment.
- **d2715023: generator and scorer.** The generator's pull-outs are too shallow to clear the parked car, and the scorer rejects every pull-out there whatever its clearance.
- Acceleration input, controller, harness: ruled out. n = 2 scenes.

### Proposed fix and what the offline test showed
- **Anti-stall rule** (standing for 3 s, model chose "stay", a prompt pull-out scores within 0.3 of it → take it): on b44a4a44 it fires at +3.0 s, when the human also leaves, and the car moves 3.14 m in 2 s with no contact. **But it also fires on all 3 moments in other scenes where the human correctly waits** (e3ed554f in a queue, 8023dce5), and no threshold separates b44a4a44 from those. On d2715023 it never fires when standing time is counted from handover, or when a clearance check is added; if standing time before handover is also counted and there is no clearance check, it fires at handover on a plan that runs into the parked car (and on b44a4a44 it then fires 3 s before the human leaves). So the rule is not validated: the scorer's numbers cannot tell a stall from a proper wait. Only two correct waits exist in the 30 scenes, so this is weak evidence either way.
- **The real fix is retraining, which I did not run:** (a) the scorer needs a progress term that punishes standing still when the way is clear, and inputs that reflect the path rather than the menu position; (b) the generator needs wide pull-outs in its repertoire (the 283 pull-out clips already in training did not produce them). Earlier CPU experiments on a replacement scoring head (`analysis_3.0/Retrain-data/11_scorer_head_and_pullout_grades/REPORT.md`) did not move either car, which fits finding (a): the features themselves are tied to menu position.

## 5. What I would ask you to approve next (none of it was run)

1. **Closed-loop check of Fix A**: adapter 0.1.6 plus speed-profile smoothing above 10 m/s, on 04749bb9, acd22430, bd07fb91 and three scenes that already work (02e075b9, b1b170d0, 04394343). Six runs, about an hour of shared GPU for rendering. Success: no road departure in the first 3 s and steering under 0.3 rad on the three failing scenes, nothing lost on the three good ones. Expect the car to remain slower than the human.
2. **Tell the simulator's owners** about the controller: braking above about 5 m/s² at speed becomes steering, and a steering-rate penalty removes it (`part3_swerve_fixes/fixC_sweep.csv`). Also: bd07fb91 scores 1.00 while crashing just after the scored part; 791d68ce scores 0 for every driver; pedestrians walking into a stationary car count as its fault.
3. **Retraining decisions** (yours): generator spacing term for highway speeds; a scorer change for standstill. Neither can be validated offline.

## 6. Still open

- Whether Fix A holds over a whole drive (needs item 1).
- Why the scorer prefers slow plans on the highway; earlier work measured that it does (it picks the best plan less often than chance there) but not why.
- The anti-stall rule needs more correct-wait examples than our 30 scenes contain before it can be tuned.
- How much any per-scene number moves between two identical runs; no repeat exists.
- Verified versus argued: sections 1–2 and every "measured" cell are computed from run files; the verdicts are my reading of them; everything in section 5 is a proposal.
