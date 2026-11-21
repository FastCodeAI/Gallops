# Link-2 test — does the trained memory hold information that matters for driving?

2026-11-12. No training, no held-out slice, no closed loop. Forward replays (GPU, batch 1) and linear probes (CPU, 2 cores).
Pre-registration: `reports/LINK2_PREREG.md` (file time 00:49 IST; first probe on real data started 01:20 IST, `logs/link2_chain.log`).

## Result

**LINK 2 BROKEN by the pre-registered rule.** On the primary sample, all moments, adding the carried memory predicts the human
driver's next 3 s no better than adding the wiped memory on any of the six targets, and both controls pass.

What the numbers say beyond the rule:

- The carried memory adds nothing measurable on top of the current view and the ego history (PH+C vs PH: −2.8 % to +0.6 %, no
  interval clear of 0 on the positive side).
- The carried memory does hold the past: the same probe recalls the view of 2 s ago 17.2 % better from the carried than from the
  wiped memory. So the memory remembers, but what it remembers does not help predict driving.
- The wiped memory is not an empty baseline. It predicts the lateral offset at +3 s 40 % better than PH, and the heading 9 % better.
  The carried memory has lost this. That is why the carried-vs-wiped gains on lateral offset and heading are strongly negative.
- Event moments give a weak hint in the other direction on speed (PH+C vs PH: +6.0 % on speed at +3 s, interval −1.1 to +12.1;
  27 clips). Not significant.

All numbers below are from `out/link2_results.json` unless another file is named.

## Samples

| | clips | moments replayed | after warm-up, used | event moments | event clips | braking-onset rate |
|---|---|---|---|---|---|---|
| Primary (training clips unused in the first 930 steps) | 300 (221 ordinary, 59 reconstructed, 20 pull-out) | 9,454 | 8,191 | 252 | 27 | 4.4 % |
| Picking clips | 84 | 2,591 | 1,705 (picking moments only) | 29 | 5 | 3.4 % |

Sources: `out/link2_sample.json` (clip list, selection counts: 1,611 clips used by the 930 steps, 2,403 unused),
`out/link2_stats.json`, `{experiment_store}/runs/link2/replay.json`. Starting model sha256 3d566eb2…2b8e66 verified,
495/495 tensors equal (`replay.json`). In the replays the pick differs between carried and wiped on 12.8 % of primary moments
(build 5), 10.3 % (build 4), 3.9 % (pilot).

## How well each feature set predicts (primary, all moments)

Share of variance explained (T4: AUC). Targets and feature sets are defined with file:line in the pre-registration.

| feature set | speed +1 s | speed +2 s | speed +3 s | braking onset (AUC) | lateral +3 s | heading +3 s |
|---|---|---|---|---|---|---|
| P (current view + ego row) | 0.9869 | 0.9599 | 0.9247 | 0.809 | 0.837 | 0.721 |
| PH (+ ego history) | 0.9951 | 0.9775 | 0.9467 | 0.850 | 0.854 | 0.747 |
| PH+W (+ wiped memory) | 0.9951 | 0.9777 | 0.9470 | 0.857 | **0.913** | **0.769** |
| PH+C (+ carried memory) | 0.9951 | 0.9776 | 0.9466 | 0.848 | 0.850 | 0.744 |
| PH+Cs (+ scorer's memory summary) | 0.9951 | 0.9776 | 0.9469 | 0.848 | 0.848 | 0.746 |
| PH+C shuffled across clips | 0.9950 | 0.9771 | 0.9454 | 0.836 | 0.852 | 0.743 |

## Gains, % relative error reduction [95 % interval] (primary)

Error = mean squared error; log-loss for braking onset. Positive = the first set is better.

**All moments (8,191 moments, 300 clips)**

| comparison | speed +1 s | speed +2 s | speed +3 s | braking onset | lateral +3 s | heading +3 s |
|---|---|---|---|---|---|---|
| PH vs P | +62.4 [+59.5, +64.9] | +44.0 [+40.2, +47.3] | +29.2 [+25.1, +32.6] | +10.7 [+2.6, +18.0] | +10.7 [+8.1, +13.4] | +9.0 [+4.1, +13.1] |
| PH+W vs PH | +1.1 [+0.4, +1.8] | +1.0 [+0.2, +1.7] | +0.6 [−0.2, +1.4] | +2.7 [+0.6, +4.9] | +40.3 [+34.0, +45.9] | +9.0 [+4.9, +13.6] |
| **PH+C vs PH+W (the rule)** | −0.7 [−1.9, +0.4] | −0.8 [−2.3, +0.7] | −0.7 [−2.4, +0.9] | −2.2 [−3.5, −0.8] | −72.3 [−90.0, −56.8] | −11.1 [−16.6, −6.4] |
| PH+C vs PH | +0.4 [−0.6, +1.3] | +0.2 [−1.1, +1.5] | −0.1 [−1.7, +1.4] | +0.6 [−1.9, +3.3] | −2.8 [−5.6, −0.0] | −1.1 [−3.7, +1.5] |
| PH+Cs vs PH+W | −0.7 [−1.8, +0.3] | −0.5 [−1.8, +0.7] | −0.1 [−1.7, +1.3] | −2.5 [−4.1, −0.9] | −74.8 [−93.1, −58.6] | −10.3 [−15.8, −5.7] |
| PH+Cs vs PH | +0.4 [−0.4, +1.1] | +0.5 [−0.7, +1.5] | +0.5 [−0.9, +1.8] | +0.3 [−2.4, +3.2] | −4.3 [−7.2, −1.7] | −0.4 [−2.6, +1.9] |

**Event moments (252 moments, 27 clips)**

| comparison | speed +1 s | speed +2 s | speed +3 s | braking onset | lateral +3 s | heading +3 s |
|---|---|---|---|---|---|---|
| PH vs P | +72.2 [+60.2, +79.3] | +58.1 [+40.0, +69.0] | +44.5 [+23.1, +57.2] | +4.4 [−35.0, +71.9] | +2.9 [−9.4, +14.3] | +4.1 [−29.8, +11.5] |
| PH+W vs PH | +2.6 [+0.1, +5.0] | +4.0 [+1.6, +6.2] | +4.9 [+1.8, +7.7] | +3.6 [−8.9, +5.6] | +50.2 [+7.1, +71.0] | +18.7 [−0.9, +28.4] |
| PH+C vs PH+W | −0.9 [−4.2, +2.7] | −0.0 [−5.2, +5.2] | +1.2 [−7.2, +8.3] | −2.8 [−4.6, +9.7] | −109.7 [−263.7, −14.1] | −17.4 [−33.8, +1.9] |
| PH+C vs PH | +1.8 [−0.7, +4.9] | +3.9 [−0.6, +8.8] | +6.0 [−1.1, +12.1] | +0.9 [−8.7, +9.5] | −4.4 [−11.4, −0.4] | +4.6 [−5.6, +9.0] |
| PH+Cs vs PH+W | −0.9 [−4.5, +2.7] | +0.1 [−4.2, +4.9] | +1.2 [−5.2, +7.3] | −1.8 [−3.8, +14.7] | −112.4 [−278.4, −14.3] | −18.0 [−35.3, −2.8] |
| PH+Cs vs PH | +1.7 [−0.6, +4.7] | +4.1 [+0.3, +8.3] | +6.1 [+0.5, +11.3] | +1.9 [−3.7, +14.0] | −5.7 [−13.2, +1.4] | +4.1 [−7.9, +8.5] |

Braking-onset AUC, all moments: PH+C 0.848 vs PH+W 0.857, difference −0.008 [−0.015, −0.003].

## Controls

| control | result | pass rule | verdict |
|---|---|---|---|
| Positive: recall of the view 2 s ago, PH+C vs PH+W | **+17.2 % [+16.4, +18.0]** (error 0.393 vs 0.474); event moments +18.7 % [+15.8, +21.4] | ≥ 3 %, interval above 0 | pass |
| Negative: carried memory shuffled across clips, vs PH+W | speed −2.8 / −3.0 / −3.0 %, braking −6.4 %, lateral −70.5 %, heading −11.3 % (all intervals below 0) | below 1 % or interval includes 0, on every target | pass |

- The positive control is larger than the read-out's 6.0 % (`reports/BUILD5_RECALL_LESSON.md`). It is a different measurement:
  a linear probe fitted here, with the current view and ego history also given. The wiped memory itself adds nothing to recall
  (PH+W vs PH: +0.1 %).
- The negative control shows no false gain. It is not "about 0" but clearly negative, for the same reason as the main result: the
  wiped memory holds lateral and heading information that any carried memory, own or another clip's, lacks. On lateral offset and
  heading the shuffled memory does as badly as the clip's own carried memory (−70.5 % vs −72.3 %; −11.3 % vs −11.1 %).

## Picking clips (information)

1,705 moments, 84 clips. PH+C vs PH+W: speed −2.1 [−5.6, +1.4], −3.7 [−7.1, −0.1], −4.0 [−7.4, −0.8]; braking −5.5 [−12.8, +0.2];
lateral −55.4 [−83.6, −29.8]; heading −6.4 [−14.3, +0.3]. PH+C vs PH: no gain on any target (−1.4 to −5.3 %). Recall control
+12.7 % [+11.4, +13.9]. Same picture as the primary sample. Event moments here are 29 moments in 5 clips with 3 braking onsets;
those figures are not usable.

## Older checkpoints (information; primary, all moments, carried vs wiped memory)

| checkpoint | speed +1 s | speed +2 s | speed +3 s | braking onset | lateral +3 s | heading +3 s | recall control |
|---|---|---|---|---|---|---|---|
| Build 4 (version 2) | −1.4 [−2.7, −0.3] | −1.3 [−2.7, +0.1] | −1.2 [−2.7, +0.3] | −2.8 [−5.3, −0.5] | −51.9 [−64.2, −40.6] | −18.7 [−26.5, −12.7] | +10.6 % [+9.9, +11.4] |
| Pilot (version 1) | −0.1 [−1.1, +0.9] | +0.1 [−1.3, +1.4] | −0.2 [−2.1, +1.7] | −1.7 [−3.9, +0.3] | −17.8 [−22.8, −12.7] | −3.4 [−6.7, −0.5] | +5.0 % [+4.5, +5.5] |

No checkpoint shows a gain. The recall lesson raised what the memory remembers (5.0 % → 10.6 % → 17.2 %) without making it more
useful for predicting driving. The wiped memory's lateral information is present in all three (PH+W lateral variance explained:
pilot 0.872, build 4 0.907, build 5 0.913, against PH 0.854).

## Post hoc (not pre-registered): own carried memory vs another clip's

`out/link2_posthoc.json`, from the saved predictions, no new probe. PH+C vs PH+C-shuffled, primary, all moments: speed +2.0 %
[+0.8, +3.2], +2.2 % [+0.4, +4.0], +2.2 % [+0.0, +4.4]; braking +3.9 % [−0.5, +8.3]; lateral −1.0 %; heading +0.2 %. Event moments:
speed +3 s +13.0 % [+1.7, +21.8]. So the carried memory does hold a little clip-specific speed information, but it is information
the ego history already gives (PH+C vs PH is flat).

## Flags

1. **The wiped memory is not an empty baseline.** It adds +40.3 % on lateral offset and +9.0 % on heading over PH. The rule compares
   carried against wiped, so on those two targets the result mostly measures what carrying loses, not what it adds. PH+C vs PH is
   the comparison free of this, and it is flat.
2. **Why the carried memory lacks the lateral information is not established.** Candidates: the low update gate (mean 0.216) dilutes
   the current write; the slots' content drifts when carried. Not tested.
3. Negative control passes by the written criterion but is strongly negative, not near 0 (see Controls).
4. Positive control is 17.2 %, not the read-out's 6.0 %. Different measurement; both say the carried memory holds the past.
5. Build 4's recall here (10.6 % primary, 6.7 % picking) agrees with the 7.2 % in `BUILD5_RECALL_LESSON.md` flag 3 and not with
   the 0.4 % in `BUILD5_DESIGN_NOTE.md`. This supports the view that the design note's 20-clip probe was unreliable.
6. "Not used in the first 930 steps" rests on the deterministic plan function and seed; the runs did not log clip names per step.
7. The starting generator and scorer were trained on the whole training slice; only the memory blocks had not seen these clips.
8. Event moments are few: 252 moments in 27 clips (primary), 29 in 5 (picking). Event intervals are wide; the event hints on speed
   are unadjusted for six targets and several comparisons.
9. 1,235 of 8,191 primary moments have a partial ego history (moments 4–7 of a chain; `out/link2_stats.json`).
10. One linear probe. BROKEN means "not linearly readable beyond ego history and the current view". A non-linear reader was not tried.
11. Adding 4,096 uninformative features costs a little: PH+C is slightly worse than PH on lateral offset (−2.8 % [−5.6, −0.0]).
    Small gains could be hidden by the same cost.
12. Braking onset is rare (4.4 %) and counts moments already braking. The logistic probe uses 600 directions.
13. The pre-registration first carried the time 00:55; corrected to the file time 00:49 before the probes started. Target
    statistics were computed after it was written, as stated in it.
14. The post-hoc comparison (own vs another clip's carried memory) was added after seeing the results.

## OPEN for Sid

1. **Accept the reading?** Recommend: accept "the carried memory remembers the past view but adds nothing linear for predicting
   driving beyond ego history" as the working conclusion. Alternative: first run a small non-linear probe (flag 10) before accepting.
2. **Cleaner follow-up: does carried add on top of wiped (PH+W+C vs PH+W)?** Recommend: run it, about 1 h on two cores, no GPU. It
   removes flag 1 from the main comparison. Alternative: skip, since PH+C vs PH is already flat.
3. **Carrying loses lateral information the wiped write has.** Recommend: a short probe by slot and by gate level to find where it
   is lost (replays already saved; CPU only). Alternative: leave it until the next build decision.
4. **Event moments.** Recommend: a second sample of every unused training clip with event moments, same probes, to settle whether
   the speed hint on events (+6 %, not significant) is real. Alternative: treat events as untested.
5. **What the memory is taught.** The recall lesson made the memory remember more of the past view (5 % → 17 %) without making it
   more useful. Recommend: before any further training, decide whether the next lesson should target something driving needs that
   the current view and ego history do not give (for example, what other road users did). Alternative: the recall-weight-0 control
   proposed in the build 5 report.
6. **Rule wording for future tests.** Recommend: use "+ carried on top of wiped" or "+ carried vs ego-history baseline" as the rule
   comparison. Alternative: keep carried vs wiped.

## Files

- Code: `src/s10_link2.py` (sample, replays), `src/s10_probe.py` (probes), `src/link2_chain1.sh`, `src/link2_chain2.sh`.
- Out: `out/link2_sample.{json,csv}`, `out/link2_stats.json`, `out/link2_selftest.json`, `out/link2_results.json`, `out/link2_posthoc.json`.
- Run: `{experiment_store}/runs/link2/` (features, memories, per-set out-of-fold predictions in `probe/`).
- Logs: `logs/link2_chain.log`, `logs/link2_replay.log`, `logs/link2_stats.log`, `logs/link2_probe.log`, `logs/link2_summary.log`.
