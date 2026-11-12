# Ceiling test — pre-registration

Written 2026-11-12, 04:20 IST (file time stamp 04:20:41), **before any probe was run on real data**. At this time: the pooled
views are extracted from the cache, the sample is counted (`out/ceiling_stats.json`), the probe code has run only on synthetic data
(`out/ceiling_selftest.json`). No probe output exists (`{experiment_store}/runs/ceiling/probe/` does not exist).

Code fixed at this point (sha256 prefix): `src/s11_ceiling.py` bf11edb16439, `src/s11_extract.py` 3b4739cf241f,
`src/s10_probe.py` 80bea0621571 (the link-2 probe, imported).

## Question

The link-2 test showed the trained memory adds nothing for predicting the driver. Either (A) the past camera view holds nothing
driving needs beyond the present view and the ego history, or (B) it does, and the memory was taught to keep the wrong things.
Here the probe is given the true past view directly. That is the ceiling for any memory built on these views.

No model, no memory, no training of the driving model, no GPU, no held-out slice, no closed loop.

## Sample

- All training clips in the cache: 4,015 clips, 126,981 moments (`runs/ceiling/extract.json`).
- Moments used: at least 8 earlier consecutive moments in the clip (4 s of history): **94,192 moments in 4,006 clips, 1,396 chunks**
  (`out/ceiling_stats.json`; `src/s11_ceiling.py:51-66`).
- Event moments (within 5 s after an event, `out/memory_moments_train.parquet`): **5,393 moments in 610 clips, 477 chunks**.
- The brief names 3,958 training clips; the cache index has 4,015, of which 4,006 have a moment with 4 s of history. Flagged.
- Folds: 5 outer folds by chunk, seed 20261114 (`:221`). Bootstrap clustered by chunk.
- **Time rule (fixed now, depends on time only):** the run uses all 4,006 clips. If the first fold takes more than 2.5 h, the run is
  stopped before any result is read and restarted on a seeded subsample of 2,000 clips: every clip with an event moment plus random
  others, seed 20261116 (`:56-59`).

## Targets

T1–T6 exactly as in `reports/LINK2_PREREG.md` ("Targets"), by import of `s10_probe.targets` (`src/s10_probe.py:40-49`):
speed at +1, +2, +3 s; braking onset within 3 s; lateral offset at +3 s; heading change at +3 s.

## Feature sets

Low-dimensional part, by import, as in link 2: current ego row (accelerations zeroed) and the 4 s ego history
(`s10_probe.history`, `src/s10_probe.py:52-64`), unpenalised.

| set | content |
|---|---|
| P | ego row + current view, 4 × 8 block means (8,192 values) |
| PH | P + ego history (the link-2 baseline) |
| PH+V2 | PH + true view 2 s ago (row t − 4) |
| PH+V24 | PH + true views 2 s and 4 s ago (rows t − 4, t − 8) |
| PH+V124 | PH + true views 1, 2 and 4 s ago (rows t − 2, t − 4, t − 8) |
| PHf | PH with the current view pooled 8 × 16 (32,768 values) |
| PHf+Vf24 | PHf + true views 2 s and 4 s ago, pooled 8 × 16 |
| PH+V24shuf | PH + the 2 s and 4 s views of a moment of another clip (negative control) |
| P+V24 | P + true views 2 s and 4 s ago, no ego history (positive control) |

The 4 × 8 views equal the block means used in link 2 (max difference 1.9e-6 over 2,000 link-2 moments, checked from
`runs/link2/feats_primary.npz`). Extraction: `src/s11_extract.py`.

**Difference from link 2, forced by size.** Link 2 used all 8,192 standardised values per view in an exact kernel ridge
(8,191 moments). With 94,192 moments that is not possible on two cores. Here, inside every outer training fold
(`src/s11_ceiling.py:82-101`):

- each value is standardised with the fold's mean and sd of the current view;
- the top **1,024 principal directions** of the standardised current view are found (4 × 8: exact; 8 × 16: randomised with one power
  step); the same map is applied to the past views; scores are divided by √(number of raw values), as in link 2;
- the share of variance kept is reported per fold.

So the linear reader is the link-2 ridge restricted to the top 1,024 directions of each view.

## Readers

- **Linear** (`:128-173`). Ridge with the low-dimensional part unpenalised; ridge strength per target by 5-fold cross-validation over
  chunks inside the training fold, grid 10^−3 … 10^3 in half-decades (the link-2 grid). T4: L2-penalised logistic regression on the
  low-dimensional part + the top 600 principal directions of the penalised part; strength chosen on one inner hold-out fold
  (grid 10^−2 … 10^3). Link 2 used 5 inner folds for the logistic; one hold-out fold is used here for time.
- **MLP** (`:177-209`), for PH, PH+V24, PHf, PHf+Vf24 and PH+V24shuf. One hidden layer of 256 units (ReLU) plus a linear skip. Inputs:
  standardised low-dimensional part + the first **256** principal scores per view, whitened. Six outputs (five standardised
  regression targets, one braking-onset logit) trained together; Adam, learning rate 1e-3, weight decay 1e-4, batch 512, at most
  60 epochs; early stopping per target on one inner hold-out fold (patience 6); mean of 2 seeds.
- Error: mean squared error (T1–T3, T5, T6, control); log-loss (T4). **Gain of A over B = 1 − error(A) / error(B)** on out-of-fold
  predictions (`s10_probe.gain_ci`). Each "+past" set is compared with its own baseline in the same reader (PH or PHf).
- 95 % interval: chunk-clustered bootstrap, 2,000 draws, seed 20261112. Six-target (Bonferroni) intervals as information.
- All moments and event moments reported separately.
- Self-test on synthetic data (`out/ceiling_selftest.json`): cross-validation from Gram matrices equals refits (1.3e-7 relative);
  this ridge equals the link-2 kernel form (4.5e-7); randomised directions equal exact ones on a low-rank test; the MLP finds a
  planted non-linear signal (gain 0.83 [0.82, 0.83]) and shows none when it is shuffled (−0.03 [−0.04, −0.02]).

## Controls (both must pass, or the verdict is UNCLEAR)

- **Positive.** Linear reader, target = the car's own logged speed 3 s ago (‖ego[t − 6, 3:5]‖, `:63`). **Passes if** P+V24 beats P by
  ≥ 5 % with the 95 % interval above 0, all moments.
- **Negative.** Past views shuffled across clips (seed 20261115; `s10_probe.shuffle_across_clips`), each moment keeps its own PH.
  **Passes if**, for both readers and every target T1–T6, the gain of PH+V24shuf over PH is below 1 % or its 95 % interval includes
  0, all moments.

## Rule

- **CAUSE B (the past view helps)** if, for PH+V24 or PHf+Vf24, in the linear reader OR the MLP, either holds, and both controls pass:
  - all moments: ≥ 2 % gain on at least one of T1–T6, 95 % interval excluding 0; or
  - event moments: ≥ 5 % gain on at least one target, interval excluding 0.
- **CAUSE A (nothing there)** if no "+past" set (PH+V2, PH+V24, PH+V124, PHf+Vf24 linear; PH+V24, PHf+Vf24 MLP) shows a gain with its
  interval excluding 0 on any target, on all moments or event moments, and both controls pass.
- **UNCLEAR** otherwise.

Information, not rule: the best gain found anywhere and how it compares with the ego history's own gain (PH vs P; link 2 found
+62.4 % on speed at +1 s); PHf vs PH; MLP vs linear on PH; Bonferroni intervals.

## Known limits, stated before the result

1. Views are reduced to 1,024 directions (linear) and 256 (MLP). Information in weaker directions is not seen.
2. Many comparisons: 6 "+past" comparisons × 6 targets × 2 moment sets with unadjusted 95 % intervals. A few intervals excluding 0
   by chance are expected; CAUSE A is therefore hard to reach and small isolated gains should be read with the Bonferroni interval.
3. The MLP's seed-to-seed variation is not in the bootstrap interval (2 seeds averaged).
4. The MLP shares one hidden layer across the six targets.
5. All kept moments have full 4 s history, unlike link 2, where moments 4–7 of a chain were included.
6. T4 logistic: 600 directions, one inner hold-out fold.
7. The views come from the frozen encoder. A different encoder could put other things in them.

## Amendment 1 — written 2026-11-12 04:39 IST, during the run, before any result was read

Only the training log of fold 0 had been looked at (`logs/ceiling_probe.log`: timings, chosen ridge strengths, MLP stopping epochs).
No prediction, error or gain had been computed or read (`out/ceiling_results.json` did not exist).

Observation: the MLP ran to its 60-epoch cap in every run; the best hold-out epoch for speed at +1 s (and often +2 s) was 46–59,
so those two targets were still improving when training stopped. The other targets peaked at epochs 1–4.

Decision: the rule is unchanged and uses the MLP as registered (60 epochs). In addition, **as information only**, the same MLP is
re-run with at most 400 epochs and patience 30 (`src/s11_mlp_long.py`, sha256 prefix 592beda1b0e9); results in
`out/ceiling_results_mlp_long.json`. If the two disagree, the report says so.
