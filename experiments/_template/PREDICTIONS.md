# {{ID}} — predictions, written before the first run

**Commit this file before anything trains.** Its commit date is the evidence that the predictions
came first; a prediction written after the result is not one.

- **Written:** {{DATE}}, by {{OWNER}}
- **Hypothesis:** <one sentence, as in experiment.toml>

## What will be measured

| # | Benchmark / subset | Metric | Baseline | Predicted | Counts as support if | Counts as refuted if |
|---|---|---|---|---|---|---|
| P1 | <e.g. NAVSIM, the gate's 1,000 scenarios> | <EPDMS> | <released, 0.875> | <+0.01> | <GAIN at the gate's bar> | <REGRESSION, or NEUTRAL with MDE below the predicted effect> |

## Controls

<What rules out the boring explanation: e.g. the same layer reset every tick, a seed sweep, the
same data without the change.>

## Decision rules

- If supported: <what ships or what runs next>.
- If refuted: <what is dropped, and what that tells the team>.
- If inconclusive: <what would make it decidable, and whether it is worth the cost>.

## Selection

<Which split chooses the checkpoint (epoch, arm, threshold), and which split reports it. They must
differ: docs/conventions.md, "Select on one split, report on another".>
