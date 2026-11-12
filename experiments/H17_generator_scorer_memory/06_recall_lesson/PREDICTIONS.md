# Build 5 short check: pre-registration

Written **2026-11-11 20:46:10 IST**, before the short check is launched (`src/build5_chain2.sh` refuses to start without this file; the launch time is in `logs/build5_chain.log`).

## The run

Carry mode, model version 3 (learned start state, spread gates, recall lesson at fixed weight 2.0), seed 0, the first 930 steps of the same seeded pass plan as the pilot and build 4. Starting model 0.1.8b pass 0.6, sha256 `3d566eb2…8b66`, verified inside the job before loading. Learning rates unchanged. Outputs in `{runs}/generator_scorer_memory/runs/build5_short/`. No held-out slice is opened; no closed loop.

## The health rule (step 930, carry mode, all 84 picking clips, picking moments after warm-up)

PASS only if **all** hold:

| | condition | threshold |
|---|---|---|
| (a) | mean pairwise cosine between the 16 slots of the memory read | ≤ 0.90 |
| (b) | recall lesson at 2 s: carried memory lower than the same weights reset to the start state every moment | ≥ 5 % lower |
| (c) | recall lesson at 4 s: same comparison | ≥ 6 % lower |
| (d) | absolute memory change per moment, ‖M_t − M_(t−1)‖ / ‖M0‖ | ≥ 0.025 |
| (e) | memory norm, ‖M‖ / ‖M0‖ | ≥ 0.5 |
| (f) | gate: mean over the 16 slots ≥ 0.10, and ≥ 8 slots at ≥ 0.05, and max / min ≥ 3 | all three |
| (g) | no NaN / inf in any replayed moment | 0 |
| (h) | pick grade on the 1,131 graded moments | ≥ 0.74 |

(b) and (c) on the build-5 model use its own trained read-out (smooth-L1 on standardised targets, the lesson's own measure). Code: `src/s9_short.py` (`RULE`, `rule_eval`), replay in `src/s8_short.py` (`replay_memory`).

## Reported for information only (not part of the rule)

- linear-probe recall gap beside (b) and (c) (kernel ridge on [current block means; memory], 5 folds by clip);
- gate mean over training windows, first 100 and last 100 steps;
- between-clip cosine of the memory after removing the mean over clips;
- carried vs reset pick grade (same weights);
- the three choice gates against build 4's (hard 0; go 83.0 %; closest-of-32 ADE (3 s) 0.246 m);
- the recall lesson's training value over time (with the all-zero-answer reference per step);
- gradient-norm ratio, recall vs the three driving lessons, on the update block at each probe;
- read-projection norms at each probe (steps 0, 310, 620 on the first 20 picking clips; 930 on all 84).

## Resume test (pre-stated criterion)

The run stops on purpose after the step-310 probe. Before stopping it runs steps 311–320 without saving them (the "uninterrupted record"). The resumed job restores weights, optimisers and random-number states, runs 311–320 again and compares total loss per step. **Matched** = first resumed step |Δ loss| ≤ 0.005 **and** all ten steps |Δ loss| ≤ 0.05.

## The "before" columns

Same code (`src/s9_short.py before`, finished 2026-11-11 20:45 IST, `out/build5_before.json`), all 84 picking clips, 1,705 moments after warm-up. These two checkpoints have **no trained read-out**, so (b) and (c) are the **linear-probe** recall gap, not the lesson's own measure. Pick grades are from each checkpoint's own step-930 check (`out/build4_short.json`, pilot `run.json`).

| | measure | pilot step 930 (version 1) | build 4 step 930 (version 2) | threshold |
|---|---|---|---|---|
| (a) | slots alike (cosine) | 1.000 ✗ | 0.442 ✓ | ≤ 0.90 |
| (b) | recall 2 s, carried vs reset — *linear probe* | 4.9 % ✗ | 7.2 % ✓ | ≥ 5 % |
| (c) | recall 4 s, carried vs reset — *linear probe* | 11.1 % ✓ | 15.4 % ✓ | ≥ 6 % |
| (d) | absolute change per moment / ‖M0‖ | 0.012 ✗ | 0.039 ✓ | ≥ 0.025 |
| (e) | memory norm / ‖M0‖ | 0.29 ✗ | 0.93 ✓ | ≥ 0.5 |
| (f) | gate mean; slots ≥ 0.05; max/min (min / median / max) | 0.017; 0; 1.00 (0.017 / 0.017 / 0.017) ✗ | 0.167; 13; 25.7 (0.022 / 0.126 / 0.553) ✓ | ≥ 0.10; ≥ 8; ≥ 3 |
| (g) | non-finite moments | 0 ✓ | 0 ✓ | 0 |
| (h) | pick grade (1,131 graded) | 0.7526 ✓ | 0.7496 ✓ | ≥ 0.74 |
| | by this rule | FAIL (a, b, d, e, f) | all eight hold | |

The pilot's ‖M0‖ reference is 64.0 (it has no learned start state; 64.0 is the version-2 start-state norm, used so the columns are comparable).

## Stated before the run, so they are not read afterwards as results

1. **Build 4 already satisfies every condition when (b) and (c) are read with the linear probe** (7.2 % and 15.4 %). The probe and the trained read-out are different instruments: the probe is fitted on the 84 picking clips themselves (5 folds by clip) and sees 12,288 inputs (8,192 current block means + 4,096 memory values); the read-out is trained on the training slice only. A build-5 result on (b), (c) is therefore not directly comparable with the build-4 column; the probe gap will be printed beside it for the like-for-like comparison.
2. **These probe gaps disagree with the design note** (`reports/BUILD5_DESIGN_NOTE.md`, `out/build5_measure.json`: build 4 0.4 % / 2.4 %, pilot −0.4 % / 4.4 %). That measurement used 20 clips and a plain ridge probe; this one uses 84 clips and a kernel ridge with the regularisation chosen per fit. The disagreement is unresolved and is carried to the report as a flag.
3. At weight 2.0 the recall lesson's gradient on the update block was 1.5 % of the three driving lessons' after 5 steps and 5.9 % after 50 (`out/s6_t2_map_v3.json`, `out/s6_t4_v3.json`). The weight is not changed.
4. Thresholds were given by Sid; none was adjusted after seeing the "before" columns.

*Wording correction 2026-11-11 20:46:26 IST, after launch, no rule content changed: the probe's input count in note 1 was corrected to 12,288.*
