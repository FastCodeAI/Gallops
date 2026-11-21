# Ceiling test — does the true past camera view help predict the driver?

2026-11-12. Pre-registration: `reports/CEILING_PREREG.md` (written 04:20 IST, before any probe; Amendment 1 at 04:39 IST, before any
result was read). Probes on stored data only: no model, no memory, no training of the driving model, no GPU, no held-out slice,
no closed loop, no pushes. CPU cores 0–1, nice 19.

## Verdict by the pre-registered rule: UNCLEAR

- Both controls pass.
- **CAUSE B is not met.** The best gain of "views 2 s and 4 s ago" over its own baseline is +1.2 % (speed at +3 s, coarse views) and
  +1.3 % (fine views) on all moments, below the 2 % bar; on event moments the best is +1.7 %, below the 5 % bar. The MLP shows no
  gain at all.
- **CAUSE A is not met.** The linear reader shows gains on speed at +1, +2 and +3 s whose 95 % intervals exclude 0, for every
  past-view set, also with the six-target (Bonferroni) interval.

Plain reading: the true past view holds a **small, real amount of extra information about future speed (about 1–2 % of the
remaining error; up to about 4 % on event moments), and nothing measurable about braking onset, lateral offset or heading**. For
scale, the ego history gives +57 % on speed at +1 s. So the ceiling for any memory built on these views, read this way, is low.
This is much closer to cause A than to cause B, but the rule does not allow calling it A.

## Sample

Source: `out/ceiling_stats.json`, `out/ceiling_results.json`, `{experiment_store}/runs/ceiling/extract.json`.

| | count |
|---|---|
| training clips in the cache | 4,015 (126,981 moments) |
| clips with a moment that has 4 s of history | **4,006** (ordinary 3,433; pull-out 283; other event types 204 + 86) |
| moments used | **94,192** in 1,396 chunks |
| event moments (within 5 s after an event) | **5,393** in 610 clips, 477 chunks |
| braking onset rate | 4.17 % all, 6.58 % event |
| subsampling | none (about 15 min per fold) |

Folds: 5 outer folds by chunk, seed 20261114. Intervals: chunk-clustered bootstrap, 2,000 draws, on out-of-fold predictions.

## Targets and feature sets

Targets exactly as in `reports/LINK2_PREREG.md` ("Targets"), by import (`src/s10_probe.py:40-49`): speed at +1 s, +2 s, +3 s;
braking onset within 3 s; lateral offset at +3 s; heading change at +3 s.

| set | content |
|---|---|
| present | ego row + current view (4 × 8 block means) |
| baseline | present + 4 s ego history (the link-2 baseline, `src/s10_probe.py:52-64`) |
| + view 2 s | baseline + true view 2 s ago |
| + views 2, 4 s | baseline + true views 2 s and 4 s ago |
| + views 1, 2, 4 s | baseline + true views 1, 2 and 4 s ago |
| fine baseline | baseline with the current view pooled 8 × 16 |
| fine + views 2, 4 s | fine baseline + true views 2 s and 4 s ago, pooled 8 × 16 |

Views are reduced inside each training fold to the top 1,024 principal directions of the current view (variance kept: 92.6 %
coarse, 75.2 % fine; `out/ceiling_results.json` "folds"). The MLP uses the first 256 (84.5 % coarse, 67.9 % fine).

## Controls

| control | result | passes? |
|---|---|---|
| Positive: predict the car's speed 3 s ago from present + views 2, 4 s, against present alone (linear, all moments) | **+79.9 % [+79.0, +80.8]** (error 0.628 vs 3.133 m²/s²); events +86.0 % [+83.8, +87.8] | yes (bar: ≥ 5 %, interval above 0) |
| Negative: past views shuffled across clips, against baseline, all moments | linear −1.5 % to −4.3 %; MLP −6.9 % to −30.0 %; no target shows a gain | yes (bar: gain below 1 % or interval includes 0, every target, both readers) |

The shuffled gain is clearly below 0, not about 0: extra inputs that carry nothing cost the reader a little (linear) or a lot
(MLP). See flag 3.

## Results — linear reader

Gain = 1 − error(with past) / error(own baseline), in %, [95 % interval]. Source `out/ceiling_results.json` "pairs".
**Bold** = interval excludes 0 on the positive side.

All moments (94,192):

| compared with baseline | speed +1 s | speed +2 s | speed +3 s | braking onset (3 s) | lateral offset (3 s) | heading change (3 s) |
|---|---|---|---|---|---|---|
| + view 2 s | **+1.5 [1.0, 2.0]** | **+1.7 [1.3, 2.1]** | **+1.6 [1.2, 2.1]** | −0.1 [−0.7, 0.4] | −0.3 [−0.7, 0.2] | +0.2 [−0.2, 0.7] |
| + views 2, 4 s | **+1.0 [0.6, 1.4]** | **+1.1 [0.7, 1.6]** | **+1.2 [0.8, 1.7]** | −0.4 [−1.1, 0.2] | −2.0 [−2.7, −1.4] | −0.3 [−0.9, 0.2] |
| + views 1, 2, 4 s | **+1.4 [1.0, 1.8]** | **+1.7 [1.3, 2.1]** | **+1.7 [1.3, 2.2]** | −0.4 [−1.1, 0.3] | −1.5 [−2.2, −0.9] | +0.2 [−0.5, 0.8] |
| fine + views 2, 4 s (vs fine baseline) | **+1.1 [0.7, 1.4]** | **+1.2 [0.8, 1.6]** | **+1.3 [0.8, 1.7]** | −0.3 [−0.9, 0.3] | −1.0 [−1.6, −0.4] | −0.2 [−0.7, 0.4] |
| shuffled views 2, 4 s (negative control) | −1.6 [−1.9, −1.3] | −1.9 [−2.2, −1.5] | −2.1 [−2.5, −1.7] | −1.6 [−2.3, −0.8] | −4.3 [−4.9, −3.8] | −2.6 [−3.1, −2.1] |

Event moments (5,393):

| compared with baseline | speed +1 s | speed +2 s | speed +3 s | braking onset (3 s) | lateral offset (3 s) | heading change (3 s) |
|---|---|---|---|---|---|---|
| + view 2 s | **+2.8 [0.9, 3.9]** | **+3.5 [1.7, 4.9]** | **+3.9 [2.1, 5.5]** | +0.1 [−2.0, 2.2] | +1.0 [−0.4, 2.4] | −0.9 [−2.6, 0.8] |
| + views 2, 4 s | **+1.5 [0.7, 2.8]** | **+1.7 [0.5, 3.0]** | **+1.7 [0.0, 3.1]** | −0.1 [−2.4, 2.3] | +0.0 [−1.9, 1.8] | −1.3 [−3.6, 0.9] |
| + views 1, 2, 4 s | **+1.8 [1.0, 3.2]** | **+2.5 [1.2, 3.9]** | **+3.1 [1.5, 4.7]** | +0.4 [−2.0, 3.0] | +1.3 [−0.6, 3.1] | −0.7 [−3.1, 1.5] |
| fine + views 2, 4 s (vs fine baseline) | **+1.4 [0.6, 2.6]** | +1.0 [−0.4, 2.1] | +1.3 [−0.3, 2.7] | +0.4 [−1.6, 2.6] | +0.5 [−1.2, 2.1] | −0.9 [−2.9, 1.1] |
| shuffled views 2, 4 s | −2.3 | −2.6 | −2.9 | −2.2 | −2.4 | −3.0 |

Six-target intervals, + views 2, 4 s, all moments: speed +1 s [0.4, 1.5], +2 s [0.6, 1.7], +3 s [0.7, 1.8]; lateral offset
[−2.9, −1.2]. The speed gains survive; so does the small loss on lateral offset.

Ego history for scale (baseline vs present, linear, all moments): speed +1 s **+57.1 % [52.6, 59.6]**, +2 s +39.0 %, +3 s +25.7 %,
braking onset +11.0 %, lateral offset +9.5 %, heading +5.1 %. Link 2 found +62.4 % on speed at +1 s (`reports/LINK2_TEST.md`);
see flag 9.

Best gain found anywhere: + view 2 s, event moments, speed at +3 s, **+3.9 % [2.1, 5.5]**. Best on all moments: + views 1, 2, 4 s,
speed at +3 s, +1.7 %. The ego history's own gain is 15 to 40 times larger on the same speed targets.

Fine against coarse baseline (linear, all moments): −0.7, −0.7, −0.7, −0.2, −2.0, −1.0 %, all intervals below 0. Finer pooling does
not help this reader.

## Results — MLP reader

As registered (60 epochs), gain in % [95 %], `out/ceiling_results.json`:

| compared with own baseline | moments | speed +1 s | speed +2 s | speed +3 s | braking onset | lateral offset | heading change |
|---|---|---|---|---|---|---|---|
| + views 2, 4 s | all | −13.4 [−15.3, −11.5] | −15.2 [−17.1, −13.3] | −15.8 [−17.4, −14.3] | −2.6 [−4.0, −1.4] | −16.2 [−18.2, −14.2] | −14.7 [−16.9, −12.7] |
| + views 2, 4 s | event | −17.2 | −17.5 | −16.8 | −3.4 [−9.1, 1.9] | −13.2 | −13.6 |
| fine + views 2, 4 s | all | −13.1 [−15.5, −10.9] | −13.7 [−15.5, −11.7] | −17.0 [−18.5, −15.3] | −3.6 [−4.8, −2.3] | −19.6 [−21.5, −17.6] | −15.6 [−17.6, −13.6] |
| fine + views 2, 4 s | event | −14.3 | −9.7 | −14.5 | −3.8 [−8.6, 0.9] | −14.9 | −13.8 |
| shuffled views 2, 4 s | all | −22.7 | −24.6 | −30.0 | −6.9 | −25.8 | −25.3 |

Longer training (at most 400 epochs, patience 30; information only, Amendment 1; `out/ceiling_results_mlp_long.json`):

| compared with own baseline | moments | speed +1 s | speed +2 s | speed +3 s | braking onset | lateral offset | heading change |
|---|---|---|---|---|---|---|---|
| + views 2, 4 s | all | −3.6 [−5.2, −1.9] | −2.8 [−4.6, −1.2] | −15.0 [−16.6, −13.5] | −2.6 [−4.0, −1.4] | −13.0 [−15.1, −11.0] | −10.3 [−12.5, −8.2] |
| + views 2, 4 s | event | −7.9 | −8.2 | −16.2 | −3.4 [−9.1, 1.9] | −12.2 | −11.9 |
| fine + views 2, 4 s | all | −4.2 [−5.8, −2.6] | −2.6 [−4.2, −0.9] | −17.0 [−18.5, −15.3] | −3.6 [−4.8, −2.3] | −17.0 [−19.0, −14.9] | −13.1 [−15.2, −11.2] |
| shuffled views 2, 4 s | all | −10.2 | −9.8 | −24.5 | −6.9 | −25.8 | −25.3 |

The two MLP runs agree: adding past views makes the MLP worse on every target. Longer training shrinks the loss on speed at +1 s
and +2 s only.

The MLP is a weak reader here (flag 2). Its baseline against the linear baseline: speed +1 s −80 % (60 epochs; −54 % with long
training), +2 s −55 %, +3 s −30 %, braking onset −12 %; better on lateral offset +12.3 % [10.7, 14.0] and heading +12.2 %
[10.6, 13.9] (`out/ceiling_results.json` "mlp_vs_linear_PH").

## Post hoc, information only: true past views against shuffled past views

Not in the pre-registration. Same reader, same number of inputs; only whether the past views belong to the moment
(`src/s11_posthoc.py`, `out/ceiling_posthoc.json`). All moments, gain in % [95 %]:

| reader | speed +1 s | speed +2 s | speed +3 s | braking onset | lateral offset | heading change |
|---|---|---|---|---|---|---|
| linear | +2.5 [1.9, 3.0] | +2.9 [2.4, 3.5] | +3.2 [2.7, 3.8] | +1.2 [0.4, 1.9] | +2.2 [1.4, 3.0] | +2.2 [1.5, 2.9] |
| MLP (60 epochs) | +7.6 [6.0, 9.3] | +7.6 [5.7, 9.4] | +10.9 [9.4, 12.4] | +4.0 [2.3, 5.6] | +7.7 [5.9, 9.4] | +8.4 [6.8, 10.1] |
| MLP (long) | +6.0 [4.1, 7.9] | +6.4 [4.4, 8.2] | +7.6 [5.9, 9.4] | +4.0 [2.3, 5.6] | +10.2 [8.5, 12.0] | +12.0 [9.8, 14.1] |

True views beat shuffled views on every target. This is **not** clean evidence of driving information: a true past view resembles
the present view, so it disturbs the reader less than a view from another clip. It does show that the cost of the extra inputs
(1.5–4 % linear, far more in the MLP) is of the same size as, or larger than, what the past view adds — so the net gains above are
a lower bound for what a reader with no such cost would find. On my reading the bound is not far off for the linear reader
(at most about +3 % on speed), but this was not tested.

## Flags

1. **Views are reduced before reading.** 1,024 principal directions (92.6 % of variance coarse, 75.2 % fine); the MLP sees 256.
   Link 2 used all 8,192 values in an exact kernel ridge; that is not possible at 94,192 moments on two cores. Information in weak
   directions, or in single tokens that pooling averages away, is not seen. "Ceiling" means ceiling for these pooled views.
2. **The MLP is a poor reader.** Worse than the linear reader on speed by 30–80 %, and worse with more inputs. Its negative result
   says the MLP overfits, not that the views are empty. The non-linear half of the test is therefore weak.
3. **Negative control is below 0, not about 0.** It passes the written bar. The reader pays for useless inputs, which pushes all
   "+past" gains down by a similar amount.
4. **Many comparisons** (6 comparisons × 6 targets × 2 moment sets, unadjusted intervals). The speed gains hold with the six-target
   interval and repeat across four feature sets, so they are not chance; isolated event-moment values should be read with care.
5. **Lateral offset gets slightly worse** with past views (−1.0 to −2.0 %, interval below 0). Consistent with the input cost in
   flag 3 and no information to pay for it.
6. **One view 2 s ago does better than two views (2 s and 4 s).** More views cost more than the 4 s view adds. The rule names the
   two-view set; the one-view set would not have met the 2 % bar either (best +1.7 %).
7. **Finer pooling does not help** the linear reader (fine baseline 0.2–2.0 % worse than coarse), and keeps only 75 % of variance.
8. **Clip count.** The brief says 3,958 training clips; the cache has 4,015, of which 4,006 have a moment with 4 s of history.
9. **Disagreement with link 2 on the ego-history gain:** +57.1 % here against +62.4 % there, speed at +1 s. Different sample
   (94,192 moments from all training clips, all with full 4 s history, against 8,191 moments from 300 clips), 1,024 directions
   against all 8,192, different folds. Same direction and size class; not investigated further.
10. **Event-moment intervals are wide** for the ego-history gain (speed +1 s: +42 % [5, 62]); a few chunks dominate the error there.
11. **MLP seed-to-seed variation is not in its intervals** (mean of 2 seeds); the MLP shares one hidden layer over six targets.
12. **Braking onset:** logistic on 600 directions, strength chosen on one inner hold-out fold (link 2 used five).
13. **Frozen encoder.** The views come from the frozen encoder. A different or fine-tuned encoder could put other things in them.
14. **Amendment 1** added a longer MLP run after the fold-0 training log showed speed still improving at the epoch cap. Written
    before any result was read; the rule uses the registered MLP only.
15. **The post-hoc table is not pre-registered** and has the confound stated under it.
16. **Pre-registration time.** The first draft said "about 04:25"; corrected to the file time 04:20:41. Probes started 04:20:44.

## OPEN for Sid

1. **How to read UNCLEAR.**
   Recommend: treat it as "low ceiling": the pooled past view adds about 1–2 % on future speed (about 4 % on event moments) and
   nothing on braking or steering. A memory of these views cannot be expected to change driving much.
   Alternative: keep the question open until a stronger reader is tried (item 2).
2. **A stronger reader before closing the question.**
   Recommend: one more CPU probe with a small attention reader over the past view's 32 block tokens (not principal directions),
   with a regularised baseline that beats the linear reader first; 2–3 h. Only this can tell "nothing there" from "linear reader
   cannot see it".
   Alternative: accept the linear result and skip it.
3. **Where to look for cause B instead.**
   Recommend: test whether *actor tracks* from the past (where the other cars were 2–4 s ago), not camera views, help predict the
   driver on event moments. The tracks exist for 3,000 ordinary clips. If they help, the memory should be taught tracks-like
   content, not view recall.
   Alternative: drop the memory line for approach 3.
4. **The recall lesson (build 5).**
   Recommend: stop tuning it. It teaches the memory to recall the past view, and the past view itself is worth at most 1–2 %.
   Alternative: keep it as the sanity signal only, at low weight.
5. **Event moments.**
   Recommend: break the +3.9 % event gain down by event type (pull-out, cut-in, others) from the stored predictions; 10 min CPU,
   no new probes. It would show whether any one event type carries the gain.
   Alternative: leave it; the sample per type is small.
6. **Rule thresholds.** The 2 % and 5 % bars were fixed in the brief.
   Recommend: keep them; the observed gains are too small to matter in driving either way.
   Alternative: restate the verdict as "A in practice" in the lane summary.

## Files

- Pre-registration: `reports/CEILING_PREREG.md`
- Code: `src/s11_extract.py`, `src/s11_ceiling.py`, `src/s11_mlp_long.py`, `src/s11_posthoc.py`, `src/ceiling_chain.sh`,
  `src/ceiling_chain2.sh`; link-2 functions imported from `src/s10_probe.py`
- Results: `out/ceiling_stats.json`, `out/ceiling_selftest.json`, `out/ceiling_results.json`,
  `out/ceiling_results_mlp_long.json`, `out/ceiling_posthoc.json`
- Run folder: `{experiment_store}/runs/ceiling/` (pooled views, per-fold predictions under `probe/`)
- Logs: `logs/ceiling_extract.log`, `logs/ceiling_chain.log`, `logs/ceiling_stats.log`, `logs/ceiling_probe.log`,
  `logs/ceiling_summary.log`, `logs/ceiling_mlp_long.log`
