# H15 — Prediction memory

**Owner:** Naren · **Contributor:** Siddesh · **Started:** 5 Nov 2026 · **Status:** planned (hypothesis)

The facts (runs, data, checkpoints, results, gate verdicts) live in
[experiment.toml](experiment.toml) and are rendered into the repository README. The predictions,
committed before anything trains, are in [PREDICTIONS.md](PREDICTIONS.md). This page is the account.

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

Nothing yet: registered 5 Nov 2026.
