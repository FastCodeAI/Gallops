# H15 — Prediction memory

**Owner:** Naren · **Contributor:** Siddesh · **Started:** 5 Nov 2026 · **Status:** running (hypothesis): trained, evaluated and gated; the owner's decision is open

The facts (runs, data, checkpoints, results, gate verdicts) live in
[experiment.toml](experiment.toml) and are rendered into the repository README and [CARD.md](CARD.md). The
predictions, committed before anything trained, are in [PREDICTIONS.md](PREDICTIONS.md); what the gate
should show, committed before any gate run, is in [GATE_EXPECTATIONS.md](GATE_EXPECTATIONS.md). What is
not in the repository is listed in [POINTERS.md](POINTERS.md). This page is the account.

## Why

Released Drive-JEPA decides every moment from scratch. On the saved outputs of H05.r1 (40 val logs,
momentum off), back-to-back picks are more than 1 m apart on 45% of moments, although a path within
0.34 m of the old pick is still among the 32, and the top two *predicted* scores differ by only 0.003.
Many picks look like near-ties that flip. These numbers are re-measured with true grades and
momentum on once the proposals are graded.

H05's step 0 found that a longer camera history adds little to one moment's view. This experiment
carries something else forward: the planner's and picker's own conclusions, never the camera.
That is a different claim, so it is a new record ([H05](../H05_ssm_memory/README.md) is its parent).

## The plan

| Step | What | Decides |
|---|---|---|
| 1 | File H05's results, register H15, commit PREDICTIONS.md | — |
| 2 | Grade the 32 released proposals at every graded moment, once (CPU, 4 cores): report logs first, then selection logs, then the rest. Timed on 200 moments first | — |
| 3 | Align proposals across moments; train (a) carried, (b) wiped, against (c) released, 3 seeds each, momentum off | — |
| 4 | **Go / no-go** on the 214 val logs: P1, (a) − (b) in true grade of the pick, CI excluding 0; flips and comfort secondary | whether to gate |
| 5 | Gate by hand: the adapter (kept here) copied into a throwaway paddock checkout, quick then closed tier, (a) and (b), momentum on | — |
| 6 | Keep it only if (a) beats (b), nothing regresses against released, braking starts no later | keep or stop |

## What was done

Everything released stays frozen throughout: the encoder, the 32 proposed paths and their scores are the
released model's. Only a small memory, 0.333 M parameters, is trained. It nudges the released final score.

- **Grades (H15.r1).** Each of the 32 released paths at every graded navtrain moment (91,977) was given
  its true grade by the vendor's training-time grader. A check against a paddock NAVSIM run on 200 picks
  agreed exactly on seven terms; every remaining difference is explained by the benchmark's human filter or
  its finer lane check.
- **Hand-over (H15.r2).** Each path takes the memory of the closest path of the previous moment, moved
  into the current ego frame, or starts empty when none is within 2 m. On the 214 held-out logs the
  closest old path is 0.344 m away at the median, and 0.60 % of paths start empty.
- **Training (H15.r3).** Two arms of the same size, three seeds each: **(a)** the memory carried along
  each path, and **(b)** the same layer wiped at every moment, which separates memory from extra capacity.
  **(c)** is the released model. The 100 selection logs chose a memory length of 4 moments and 6 epochs;
  the three lengths' selection losses differ by 5e-6, so the length is a coin toss.
- **Gate (6 Nov).** Seed 0 of (a) and of (b) went through paddock's regression gate against released
  Drive-JEPA, quick tier then closed tier. The memory never went into paddock: the adapter ran from a
  throwaway paddock checkout and is recorded in [paddock/](paddock/README.md) as it ran.

## What it showed

**Offline, on the 214 held-out logs (15,320 graded moments, momentum off, mean of 3 seeds):**

| Measure | (a) carried | (b) wiped | (a) − (b), 95 % interval | (c) released |
|---|---:|---:|---|---:|
| mean true grade of the pick | 0.86827 | 0.86715 | +0.00112 [+0.00020, +0.00212] | 0.86880 |
| flips: back-to-back picks more than 1 m apart | 0.4590 | 0.4788 | −0.0198 [−0.0234, −0.0162] | 0.4571 |
| two-frame comfort pass rate | 0.3962 | 0.3759 | +0.0204 [+0.0162, +0.0250] | 0.4261 |

- The carried memory picks better-graded paths than the wiped layer, and the interval excludes zero. But
  the gain is a fifth of the +0.005 set beforehand as useful, and a tenth of the +0.01 predicted:
  **supported, not useful.**
- **The interval covers logs, not training seeds.** The sign held in all three seeds, but the three
  per-seed differences (+0.00078, +0.00020, +0.00237) give t = 1.72 on 2 degrees of freedom, and the
  seeds of one arm differ by up to 0.0045, four times the effect. A fourth seed could remove it, so the
  record rates it moderate, not strong.
- Against the released model the carried memory is level on grade (−0.00052 [−0.00298, +0.00206]) and
  flips, and **3.0 points worse on two-frame comfort**. Both trained arms are less comfortable than released.
- Where a best path brakes and released does not (546 moments), the carried memory picks a braking path
  less often than the wiped layer: 12.6 % against 14.7 %.

**At the gate, against released Drive-JEPA (6 Nov, seed 0, momentum on):**

| Benchmark | Released | (a) carried | Verdict | (b) wiped | Verdict |
|---|---:|---:|---|---:|---|
| KITScenes ADE (m), 117 units | 2.537 | 2.292 | GAIN | 2.261 | GAIN |
| nuScenes L2 (m), 143 units | 2.431 | 2.336 | NEUTRAL | 2.288 | GAIN |
| NAVSIM EPDMS, 126 units | 0.857 | 0.858 | NEUTRAL | 0.856 | NEUTRAL |
| neuroNCAP score, 15 pairs | 3.679 | 3.404 | NEUTRAL | 3.531 | NEUTRAL |
| Bench2Drive driving score, 7 routes | 59.04 | 68.51 | NEUTRAL | 45.96 | NEUTRAL |
| AlpaSim scene score, 10 scenes | 0.434 | 0.394 | NEUTRAL | 0.508 | NEUTRAL |

- **Both arms: HOLD on the quick tier and HOLD on the closed tier.** Nothing regressed on a benchmark's
  own score and no closed-loop pair gained. The open-loop gains are shared by the wiped layer, so they are
  not the memory.
- **Carried against wiped, paired by the gate: NEUTRAL on all six benchmarks**
  (`paddock/results/gate/compare_a_vs_b.txt`).
- **The side crashes did not improve.** On neuroNCAP's two side scenarios the carried arm's mean score is
  1.52 and the wiped arm's 2.50. On one of them the carried arm never braked. The prediction that the
  carried memory scores above the wiped layer there and brakes no later failed.
- **One term to know about.** Beside a NEUTRAL NAVSIM score, the carried arm's no-at-fault-collisions
  term moved 0.993 → 0.988 (t −2.19). The gate does not decide on per-term rows, and with many terms
  tested some cross by chance. It was not followed up.

What is measured and what is argued: every number above is copied from `evaluate.json` or from the gate's
own files, as `experiment.toml` records. That the offline gain is too small to change driving is argued
from those numbers, not measured separately.

**Not checked: that the gate fed the memory what it was trained on.** The adapter's memory layer is
pinned to gallops' by a fixture (same inputs, same scores), and a zero nudge reproduces the released
picks. But nothing compared the three inputs the adapter builds at each call (each path's scorer
input, its six logits, the ego status) with the cache rows the memory trained on. The adapter's hook is
a transcription of `gallops.memory.cache`, so a mistake there would be shared. If paddock's inputs
differ on some benchmark, the gate tested a mis-fed layer, and the failed side-crash prediction would
say nothing about memory. The check is one navtrain val log through the adapter, its inputs and nudges
compared with the cache and the trainer's scores; it needs the GPU and has not been run.

**What the six sidecars get wrong** (`known_wrong` in the record; the files are not rewritten): the three
(b) sidecars say `length: 4` where the weights wipe their memory every moment; all six say they trained
on all 878 train logs, where only the 763 with a graded moment gave a target.

## Where it stands (12 Nov 2026)

Not decided. The record stays `running` with no decision row. By `docs/experiments.md`, a HOLD on the
closed tier leads either to a stronger version as the next run, or to closing the hypothesis as
inconclusive with a decision row; that choice is the owner's. One more thing is open: the ID `H15` is also
claimed by an unmerged branch (`drivejepa/auxilliary-head-02`), and the owner rules on which keeps it.

The follow-on, a memory that the generator and the scorer are trained to use, is a different claim and is
recorded as H17.
