# H15 gate: what we expect, written before any gate run

**Written 5 Nov 2026 (Pacific), before the first gate run of either arm.** Committed to gallops PR #31 as
`experiments/H15_prediction_memory/GATE_EXPECTATIONS.md`; the commit sha is in `MEMORY_H15_GATE.md`.
Nothing below may be changed after the first run; anything added later goes under a dated heading.

## What runs

| | |
|---|---|
| Arms | **(a)** memory carried, length 4, seed 0: `h15-r3-a-L4-s0`, weights sha256 `0679ea453b97…`. **(b)** the same layer wiped every moment, seed 0: `h15-r3-b-L0-s0`, `dfd944c1c3f1…`. One seed per arm (Sid D28: seed 0 is the median seed of offline P1 (a) − (b), +0.00078 against +0.00020 / +0.00237, chosen before any gate run). |
| Released | not re-run. The baselines are `{paddock_site}/queue/baselines/drivejepa.toml` (all 7 subsets; closed tier from job 4b7a, 26–27 Oct). |
| Momentum | ON for both arms, as released ships (Sid D29); it acts on the nudged score. |
| How | a throwaway paddock worktree off origin/main with the adapter added to `drivejepa/` behind `DRIVEJEPA_MEMORY` (empty = released, bit for bit). `python -m tools.gate smoke --model drivejepa --env DRIVEJEPA_MEMORY=<checkpoint dir>`, quick tier for (a) then (b), then closed tier for (a) then (b) (Sid D30). `--keep-going` so every subset of a tier runs and is reported even after a REGRESSION. Then `tools.gate compare` (a) against (b) per benchmark. |
| Quick tier | KITScenes (223 keyframes, decides on 186), nuScenes (1,000 keyframes), NAVSIM (1,018 navtest tokens), neuroNCAP quick (frontal-0103, side-0103, stationary-0783), and the two crash checks: Bench2Drive route 3717 and AlpaSim scene 1acf2f5d (one sample each: no verdict). |
| Closed tier | neuroNCAP 15 pairs, Bench2Drive 7 routes, AlpaSim 10 gate scenes. |
| AlpaSim image | `alpasim-base:0.134.0` is present (`9ee8dfb8`, rebuilt 3 Nov 02:45 after all AlpaSim images were removed; checked then against 0.1.7, PASS). It is not the image the 27 Oct AlpaSim baseline ran on. |

## P3: no REGRESSION against released

- **Pass condition (as pre-registered in PREDICTIONS.md):** for arm (a), no pair the gate prints, quick tier or
  closed tier, is a **REGRESSION** against the released baseline. GAIN, NEUTRAL, TOO FEW UNITS (the two crash
  checks, if they ran cleanly) count as no regression. **INCOMPLETE, NO RECORDS, SUBSET DIFFERS or CONFIG
  DIFFERS on any pair means P3 is not decided** on that benchmark, and the report says so; it is not a pass.
- (b) is run through the same tiers and its verdicts are reported, but P3 is a claim about (a).
- **Expected risk: NAVSIM.** Its EPDMS includes two-frame extended comfort. Offline, (a) is 3.0 points worse than
  released on two-frame comfort (−0.0299 [−0.0415, −0.0196]) and (b) 5.0 points worse. NAVSIM resets the model before
  every scenario and gives it 3 moments, so the memory carries at most 2 moments there; the trained layer acts
  at every moment whether or not it carries. A NAVSIM REGRESSION is possible for both arms; if one comes, (b) is
  expected to be at least as bad as (a).
- On grade, offline (a) is level with released (−0.0005 [−0.0030, +0.0021]), so no regression is expected on the
  open-loop benchmarks from grade alone. KITScenes and nuScenes measure distance to the human path, which the
  training target does not see: no expectation either way beyond "not a regression".

**What each outcome would mean**

| outcome | reading |
|---|---|
| no REGRESSION anywhere | P3 holds. With one seed and the gate's units, this says (a) is not worse by more than each pair's MDE, not that it is equal. |
| REGRESSION on NAVSIM only | the comfort cost seen offline reached the benchmark score. It is a cost of the training target (no comfort term), not evidence against carrying memory, if (b) regresses as much or more. P3 fails as written. The next version adds the two-frame comfort term (the plan's own follow-up). |
| REGRESSION on NAVSIM in (a) but not in (b) | against the offline evidence; read the EC sub-score and the pairs before any conclusion. P3 fails. |
| REGRESSION elsewhere (KITScenes, nuScenes, neuroNCAP, Bench2Drive, AlpaSim) | not predicted by anything offline; the memory or the nudge changes closed-loop or human-distance behaviour in a way the navtrain logs do not show. P3 fails. A failure on Bench2Drive (7 routes, cannot ship) or neuroNCAP quick (3 pairs, cannot ship) still fails P3, as the gate rules say, but each is a reason to look at that pair, not proof. |
| a crash check INCOMPLETE | the adapter cannot drive that simulator; stop and report (Sid D30). |

## P4: neuroNCAP side pairs 0103 and 0110

- **Pass condition (as pre-registered):** on the side pairs 0103 and 0110 of the closed tier, **(a) scores above
  (b) and brakes no later.** Read exactly as:
  1. the mean `ncap_score` of (a) over the two pairs is **strictly above** that of (b) (a tie, e.g. both at 5.0 on
     both pairs, is not "above": P4 is then not supported on score);
  2. on **each** of the two pairs, (a)'s braking start is **at or before** (b)'s.
  Both must hold. One run (seed 0 of the simulator) per arm per pair, so no interval exists; the numbers are
  shown as they are.
- **Braking start, defined now:** from the run's `ego_poses.json` (10 Hz), ego speed v_k is the distance between
  pose k and pose k+1 over their time step. The episode ends at the last pose (impact or the scenario's end) or at
  the first k with v_k < 1 m/s, whichever is first. **Braking start is the time, from the first pose, of the last
  k before the end at which v_k equals its maximum over the episode up to the end** (the speed peak before the
  final slow-down). If the speed at the end is within 1 m/s of that peak, the car **did not brake** (recorded as
  such; it counts as later than any braking start).
  - On the released baselines (read before writing this, from job 4b7a's run `20261026T232704-7bcf0e`): side-0103
    peaks 7.7 m/s at about 1.5 s, collides at 3.17 m/s, score 2.44; side-0110 peaks 10.3 m/s at about 0.7 s, no
    collision, score 5.0. The exact released values are recomputed by the same script that reads the arms.
- Speed at impact is `impact_speed` from the run's samples (0 = no impact).
- **Released is shown beside them, but P4 compares (a) with (b) only.**
- Offline evidence (R1, not a prediction): on the 546 val moments where a best path brakes and the released pick
  does not, (a) picks a braking path 12.6% of the time, (b) 14.7%, released 0%. That points the other way from P4:
  if anything, (b) brakes more often offline. Released already scores 5.0 on side-0110, so on that pair neither
  arm can score above it, and any difference in score between (a) and (b) on P4 comes from side-0103.

**What each outcome would mean**

| outcome | reading |
|---|---|
| P4 holds | consistent with the claim that carried memory brakes earlier in side crashes; one seed and two pairs, so it is a direction, not a measurement. |
| (a) above (b) on score, but brakes later on a pair | score gained some other way (e.g. a swerve); P4 fails as written. |
| scores tied (both 5.0, or equal) | P4 not supported: the side pairs cannot separate the arms at one seed. |
| (b) above (a) | P4 fails; matches the offline R1 direction (wiped layer brakes more often). |

## What one seed per arm limits

- Each arm is one trained seed. Offline the per-seed P1 (a) − (b) ranged +0.0002 to +0.0024; a gate difference
  between (a) and (b) mixes the arm with the seed, and nothing here can separate them.
- The closed tier is one simulator run per pair or route or scene. With 15 neuroNCAP pairs the gate's MDE for a
  small finetune is about 0.55 points; Bench2Drive's 7 routes detect only a 17–38 point drop; AlpaSim has 10
  scenes. A NEUTRAL there is "not worse by more than the MDE", not "the same".
- `tools.gate compare` (a) against (b) uses the same clustering; a NEUTRAL there says the gate cannot tell the
  arms apart, which is the expected result at this effect size (offline +0.0011 in grade).
- Several pairs at once: an unchanged model is called a REGRESSION somewhere on the 7 pairs about 16% of the
  time. A single REGRESSION is read with that in mind, but it still fails P3 as written.

## Changed after the first run (12 Nov 2026)

- The Released row named the queue's baseline file by its real location on the training machine. It is
  now written `{paddock_site}/queue/baselines/drivejepa.toml`, the form docs/experiments.md uses, because
  a tracked file may not hold a machine path. Nothing else in this file changed; the line as it was is in
  the commit before (`d978349`).
