# Pre-registration: approach-3 pilot, memory carried vs memory wiped (seed 0)

Written 2026-11-09 23:09 IST, amended 23:22 IST for Sid's amended brief (2 passes instead of 3; the CLOSE rule; what GO leads to), before any training step was taken (launch 1 at 23:09 crashed on an import; launch 2 at 23:10 was stopped by hand inside its step-0 check when the amended brief arrived; launch 3 follows this file) (launch times are in `logs/pilot_chain.log`; the two arms' `run.json: started`).

Arms: **carry** and **wiped**, the build-3 model and trainer as built and tested (`reports/BUILD3_MODEL_TRAINER.md`), seed 0, the same batch order (one seeded pass plan per pass, `s6_data.plan_pass`), trained in parallel on their own two cores each (`src/run_model_pilot.sh`, CORES 0-1 / 2-3, nice 19). Up to **2 passes** (lean plan; this pilot is also the main seed-0 training run); a check every 930 steps and at each pass end; stop after 6 checks without a better choosable pick; chosen = best pick grade among checks passing the safety gates (hard-brake picks above 10 m/s = 0, pull-away go rate ≥ 0.66). Nothing of either held-out slice is opened.

## Measures, at each arm's CHOSEN checkpoint, on the picking set replayed in order from each clip's first moment (fp32, batch 1)

1. **Overall pick grade**: the mean grade of the scorer's pick on all **1,131** graded in_picking moments (`out/s6_t1.json: step0_018b.map_moments`), 0.1.7's grader on the moment's own menu (`train_018b.run_check`).
2. **Event pick grade**: the same on the graded in_picking **event** moments of step 1: **34 moments** (`out/pilot_event_subset.json: in_picking_graded_event_moments`; 25 cut-in, 12 crossing, 3 hard-brake rows over 34 distinct moments in 5 clips; `out/memory_moments_picking.parquet`, `out/pilot_event_moments_in_picking.csv`). **This subset is too small to read (< 50 moments)**: a difference on 34 moments from 5 clips is dominated by those clips; it is reported as the rule requires, with that caveat written beside it.
3. **Safety gates**, per arm: hard-brake picks above 10 m/s (must be 0; 0.1.8b at step 0: 0 of 745); pull-away go rate (≥ 0.66; 0.1.8b: 84.0 % of 106); closest-of-32 ADE (3 s) against 0.1.8b's **0.238 m** with a +0.02 m tolerance (`out/s6_t1.json: step0_018b`).

## Pilot rule (fixed now)

- **GO** to 3 seeds per arm if ALL of these hold at the chosen checkpoints:
  - carry overall pick grade ≥ wiped overall pick grade;
  - carry − wiped on the graded event moments ≥ +0.01;
  - both arms pass the safety gates (hard-brake picks = 0, pull-away go rate ≥ 0.66, closest-of-32 ADE (3 s) ≤ 0.238 + 0.02 m).
  GO means the next steps are: read both held-out slices once (carry, wiped, 0.1.8b), then the memory switch in the AlpaSim adapter, then closed loop on the 12 diagnosis scenes.
- Otherwise **STOP** and report for review.
- The result is flagged **CLOSE** if any of these is true (Sid then decides whether to add seeds 1 and 2):
  - the overall gap (carry − wiped) is under 0.005 either way;
  - the event gap is between 0 and +0.02;
  - the carry model's pick grade with its memory zeroed at check time is within 0.002 of its own pick grade.
- This is a one-seed filter. Clip-clustered paired bootstrap intervals (clips resampled with replacement, 2,000 draws, 95 %) of carry − wiped on the overall and the event pick grade are reported as **information, not as the rule**.
- Event-subset size: **34 graded event moments** — too small to read; stated again in the result.

## Also recorded at every check (diagnostics, not part of the rule)

Slot diversity (mean pairwise cosine between the 16 slots of the memory read at in_picking moments after the warm-up, and of the memory written), gate mean per slot, memory-use norms (generator read output projection; rating head's new 256 columns), pick-follows-previous rate, scorer-pick ADE (3 s) on all in_picking moments and on the 34 event moments. After both arms finish: the carry arm's chosen checkpoint re-evaluated with its memory zeroed at check time (relies on memory if the pick grade drops).

## Files

`src/s7_pilot.py` (the two arms), `src/run_model_pilot.sh`, `src/b4c_events.py` (mode `picking`, rules unchanged), `src/s3_moments.py` (mode `picking`); outputs `{runs}/generator_scorer_memory/runs/pilot_seed0/{carry,wiped}/`.
