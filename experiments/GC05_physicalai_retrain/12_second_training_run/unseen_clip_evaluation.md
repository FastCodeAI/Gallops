# The second run's checkpoint on validation clips it had not trained on

This file was written when the pull request was reviewed, not when the work was
done. It holds two tables copied without change from the evaluation the
training site made after the checkpoint was chosen and merged. The full
evaluation (33 KB of tables and the nine scripts that made them) is still on
the training site's store only; `POINTERS.md` has its location, and its sha256
is `6780de352273c9759576165a1845a661f88777eebab730d1b648e39c918aa266`.

How to read the column headings, which are in the working shorthand:

- "new generator" is the shipped checkpoint: the second run's generator with
  the first run's scorer.
- "pass 9" is the first run's checkpoint.
- "batch 1" means one example at a time, which is how the simulator's driver
  runs the model, with the batch correction on.
- "UNSEEN" is NVIDIA's `val` split: 11,842 graded examples from clips neither
  run trained on. It is the split that chose which pass of the first run to
  keep, so it is not untouched for the first run's column.
- Intervals are 95 %, bootstrapped over clips.

## The targets the second run set itself beforehand

`PREDICTIONS_phase1.md` fixed these before the pilots were read. One target
missed at its point estimate: highway instants with a good plan on the menu
came in at 78.9 % against "at least 80 %" (the table's last column prints the
range that was predicted, 68 to 80, and not the target). The other targets
hold.

| item | new generator | pass 9, batch 1 | paired difference | threshold |
|---|---|---|---|---|
| T-G1 highway instants with a plan graded >= 0.95, % | 78.9 (74.1 to 82.9) | 54.5 (47.0 to 61.2) | +24.4 (+19.0 to +30.2) | predicted 68 to 80 |
| T-G2 median swing, all 32 plans, car >= 20 m/s, m/s | 3.18 (3.04 to 3.32) | 25.01 (20.38 to 34.27) | -21.84 (-31.09 to -17.29) | predicted under 5 |
| T-G3 slots passing comfort on > 25 % of instants, car >= 20 m/s (of 32) | 26 (24 to 26) | 0 (0 to 2) | +26 (+23 to +26) | predicted at least 24 |
| W1 good-plan share, ordinary, % | 73.97 (69.80 to 77.96) | 73.62 (69.44 to 77.62) | +0.34 (-0.06 to +0.71) | at least 71.62 |
| W1 good-plan share, close-pass, % | 51.31 (42.39 to 59.85) | 49.58 (40.91 to 58.32) | +1.73 (+0.52 to +3.19) | at least 47.58 |
| W2 human-path term, highway | 0.6471 (0.5945 to 0.7116) | 1.6650 (1.4001 to 1.9947) | -1.0179 (-1.3006 to -0.7897) | at most 1.7482 |
| W2 human-path term, close-pass | 0.4091 (0.3744 to 0.4437) | 0.4739 (0.4359 to 0.5117) | -0.0648 (-0.0795 to -0.0513) | at most 0.4976 |
| W2 human-path term, ordinary | 0.3338 (0.3111 to 0.3588) | 0.3815 (0.3561 to 0.4083) | -0.0478 (-0.0547 to -0.0413) | at most 0.4006 |

## What the scorer picks from the shipped checkpoint's candidates

The scorer's weights are the first run's, unchanged. Its picks are not the
first run's, because the scorer reads features that the generator produces and
the generator was retrained. "pick = true best" is the measure the first run's
check used (target: at least 50 %).

| measure | slice | instants | new generator | pass 9, batch 1 | paired difference (new - pass 9) |
|---|---|---|---|---|---|
| picks graded 0, % | highway | 3315 | 10.05 (6.88 to 13.80) | 7.51 (5.35 to 10.17) | +2.53 (+0.89 to +4.29) |
| picks graded 0, % | close-pass | 1563 | 13.31 (9.92 to 17.48) | 12.99 (9.35 to 17.22) | +0.32 (-1.75 to +2.06) |
| picks graded 0, % | ordinary | 6964 | 7.75 (6.19 to 9.41) | 7.08 (5.53 to 8.76) | +0.67 (+0.10 to +1.21) |
| picks graded 0, % | all | 11842 | 9.13 (7.78 to 10.67) | 7.98 (6.77 to 9.29) | +1.15 (+0.53 to +1.78) |
| regret, mean | highway | 3315 | 0.1061 (0.0821 to 0.1336) | 0.1490 (0.1294 to 0.1707) | -0.0430 (-0.0613 to -0.0252) |
| regret, mean | close-pass | 1563 | 0.1443 (0.1195 to 0.1760) | 0.1472 (0.1183 to 0.1808) | -0.0028 (-0.0203 to +0.0128) |
| regret, mean | ordinary | 6964 | 0.0885 (0.0761 to 0.1013) | 0.0889 (0.0771 to 0.1009) | -0.0004 (-0.0060 to +0.0050) |
| regret, mean | all | 11842 | 0.1008 (0.0907 to 0.1120) | 0.1134 (0.1037 to 0.1239) | -0.0127 (-0.0193 to -0.0065) |
| random-pick regret, mean | highway | 3315 | 0.1741 (0.1493 to 0.2017) | 0.2130 (0.1893 to 0.2377) | -0.0389 (-0.0513 to -0.0269) |
| random-pick regret, mean | close-pass | 1563 | 0.2408 (0.2098 to 0.2760) | 0.2618 (0.2259 to 0.3024) | -0.0211 (-0.0304 to -0.0121) |
| random-pick regret, mean | ordinary | 6964 | 0.1697 (0.1533 to 0.1872) | 0.1969 (0.1783 to 0.2160) | -0.0272 (-0.0329 to -0.0220) |
| random-pick regret, mean | all | 11842 | 0.1803 (0.1684 to 0.1932) | 0.2100 (0.1969 to 0.2242) | -0.0297 (-0.0348 to -0.0249) |
| rank agreement, median | highway | 3315 | 0.574 (0.527 to 0.618) | 0.023 (-0.077 to 0.111) | +0.551 (+0.461 to +0.663) |
| rank agreement, median | close-pass | 1563 | 0.475 (0.426 to 0.532) | 0.443 (0.379 to 0.486) | +0.032 (-0.008 to +0.079) |
| rank agreement, median | ordinary | 6964 | 0.567 (0.534 to 0.598) | 0.512 (0.475 to 0.553) | +0.056 (+0.026 to +0.079) |
| rank agreement, median | all | 11842 | 0.558 (0.533 to 0.576) | 0.378 (0.338 to 0.417) | +0.180 (+0.146 to +0.217) |
| pick = true best, % | highway | 3315 | 44.1 (39.2 to 49.1) | 11.0 (8.1 to 14.1) | +33.2 (+28.2 to +38.1) |
| pick = true best, % | close-pass | 1563 | 47.4 (42.6 to 52.2) | 44.0 (39.1 to 49.2) | +3.5 (+0.6 to +6.1) |
| pick = true best, % | ordinary | 6964 | 61.1 (58.1 to 64.4) | 50.1 (46.5 to 53.9) | +11.0 (+9.0 to +13.1) |
| pick = true best, % | all | 11842 | 54.6 (52.1 to 56.9) | 38.3 (35.6 to 41.5) | +16.2 (+14.1 to +18.4) |
