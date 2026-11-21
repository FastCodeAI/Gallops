# Build step 4: pre-registration of the short check (memory fix)

Written 2026-11-11 (timestamp below) BEFORE the short check was launched. Nothing in the rule is changed after this point.

Written: 2026-11-11 02:43 IST

## What is being checked

The pilot (`reports/PILOT_SEED0.md`, flag 2) found the memory collapsed: at the carry arm's chosen checkpoint the 16 slots were 99.96 % alike
(mean pairwise cosine 0.9996) and the gate mean per slot was 0.006, i.e. the memory barely changed after the first moments. At step 930 of
that arm (`{experiment_store}/runs/pilot_seed0/carry/run.json`, check at step 930, pass 0.217): cosine 0.99998, gate mean 0.0167
on every slot, memory norm 18.5.

Build 4 changes two things (Sid's decisions 1 and 2, `src/s6_model.py` version 2; the pilot's checkpoints stay version 1 and load unchanged):

1. A learned start state M0 (16 × 256) replaces the zeros at the start of every window, clip or scene. Initialised as 16 orthogonal rows
   (pairwise cosine 0) each of norm 16 = sqrt(256): the written candidate rows are LayerNorm outputs (unit variance per element, row norm
   about 16), so M0 and written rows have the same magnitude. "Wiped" now means reset to M0 before every moment.
2. A per-slot gate bias, initialised so that the gate starts at 0.9 (slot 1) down to 0.1 (slot 16), evenly spaced in probability; the
   gate's weight is initialised small (N(0, 0.001²)) so the gate starts at its bias. Weight and biases stay learnable.

Everything else is the pilot's trainer as built (reads, losses, windows of 16 moments, batches of 2 windows, learning rates generator
3e-5 / scorer 1e-4 / memory blocks and M0 1e-4, event over-sampling, replay routing with M detached in the replay's generator read).

## The short check

Carry mode only, version 2, seed 0, the first 930 steps of the pilot's seeded pass-1 plan (same batch order as the pilot's carry arm).
Outputs under `{experiment_store}/runs/build4_short/`. No held-out slice, no closed loop, no further training.

Measures at step 930 on the 84 picking clips replayed in order from each clip's first moment, on the in_picking moments after the 4
warm-up moments of a chain (the pilot's population for slot diversity, 1,705 moments):

- (a) slot diversity: mean pairwise cosine similarity between the 16 slots of the memory read at the moment
- (b) gate mean per slot (16 values)
- (c) memory change per moment: mean over those moments of ‖M_t − M_(t−1)‖ / ‖M_(t−1)‖, M_(t−1) the memory read, M_t the memory written
- (d) no NaN or inf in the memory written, the plans or the logits at any of the 2,591 moments
- (e) pick grade on the 1,131 graded in_picking moments (the standard check's pick grade, map moments)

Also recorded at steps 0, 310 and 620 on a fixed 20-clip subset (the first 20 clips of the picking index in its order; listed in
`run.json: subset_clips`): (a), (b), (c), (d) and the read-projection norms (generator read output projection, rating head new columns,
M0, gate biases). At step 930 also: the full standard check (pick grade, hard-brake picks above 10 m/s, pull-away go rate, closest-of-32
ADE (3 s), pick-follows-previous), the same checkpoint in wiped mode (memory reset to M0 before every moment) with its pick grade, and the
mean cosine between clips of the memory written at each clip's last in_picking moment (information, not part of the rule).

## The rule (Sid's, verbatim thresholds)

PASS if ALL of these hold at step 930:

- (a) ≤ 0.90
- (b) at least 8 of 16 slots have gate mean ≥ 0.05, AND the largest gate mean ≥ 3 × the smallest
- (c) ≥ 0.05
- (d) none
- (e) ≥ 0.74 (sanity only; the pilot's carry arm was 0.7526 at step 930)

Otherwise FAIL: the report names the failed condition(s) and the work stops; no further tuning without Sid.

The "before" column: (a), (b), (c) computed with the SAME code (`src/s8_short.py before`) on the pilot's carry checkpoint at step 930
(`runs/pilot_seed0/carry/ckpt/p00.2.pt`, pass 0.217, version 1) on the same 84 clips, written to `out/build4_before.json`; the report
states whether they agree with the pilot's own check record.

Checkpoint choice rule for any later run (decision 3, not exercised here: one checkpoint): best pick grade among checks with hard-brake
picks above 10 m/s = 0, pull-away go rate ≥ 0.66 and closest-of-32 ADE (3 s) ≤ 0.238 + 0.02 m. The step-930 check reports all three gates.

## Order of jobs

1. T1–T4 on the version-2 model (`MEMORY_VERSION=2 src/s6_tests.py all`); the short check is launched only if T1 (step-0 identity) and
   T3 (wiped) pass and T4 is finite.
2. The "before" column.
3. The short check (`src/s8_short.py run`), detached, pinned to cores 0–1, nice 19, after the machine gate (5-min load < 32, available RAM
   > 40 GB; NAS read speed recorded).
