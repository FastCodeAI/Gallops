<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H15 · A perception side-car on frozen Drive-JEPA: 3D boxes and each agent's 1 s path, tested as an input to the scorer, to the refiner, and against LoRA on the ViT

**Shreedhar** · NAVSIM, nuScenes · ⏳ running · 3 Nov 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Builds on:** [H14](../H14_aux_heads/CARD.md), [H04](../H04_subscore_scorer_navsim/CARD.md)

## Question

Drive-JEPA's planner reads image tokens but no explicit account of the other agents. Can a small network trained on the frozen ViT's tokens detect them in 3D and predict each one's next second, and when its decoded agents are handed to the scorer or the refiner, does the plan improve, how much of the scorer or refiner must change to use them, and does LoRA on the ViT do better than the frozen tap? The scorer's use of named agents is also the route to explanations that can be checked.

## Hypothesis

H-A: a side-car of about 4 M parameters on the frozen tokens beats no-camera and kinematic baselines at detecting agents (recall and precision at 2 m) and at their 1 s paths. H-B1: given its decoded agents, a scorer picks better plans (EPDMS of the pick on held-out navtrain logs) than the identical scorer adaptation without them (arm C), by more than C's seed range and twice the paired, log-clustered SE on every seed; tested with the scorer frozen behind an adapter, with LoRA, fully fine-tuned, and replaced. H-B2: the same for the refiner's proposals (best-of-32 EPDMS). H-B3: LoRA on the ViT beats the frozen tap. H-info: shuffled agents (S) do not recover the gain and ground-truth agents (O) are at least as good. Refuted per question if F does not beat C under any mode; the side-car claim is refuted if it does not beat its baselines.

## Predictions

[PREDICTIONS.md](PREDICTIONS.md), written before the run.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H15.s0**<br>S0: the data passes on NAVSIM navtrain: splits, agent labels with each agent's 1 s path, the token cache and the release's outputs | released | —; frozen: encoder, projector, refiner, scorer | navsim `navtrain, train logs (1,071)` (train)<br>navsim `navtrain, select logs (H04's 51 held-out)` (selection)<br>navsim `navtrain, report logs (70, seeded draw)` (held-out) | [drivejepa_perception.toml#labels](../../recipes/drivejepa_perception.toml)<br>Drive-JEPA (linhanwang/Drive-JEPA) @ `e21f474` + patches | — |

- **H15.s0:** Nothing trains here. Labels: every annotated agent within 60 m at the current frame, in its ego frame, with its positions at t + 0.5 s and t + 1.0 s joined on track_token and moved through global (docs/drivejepa-perception-sidecar.md, S0.4). Held to NAVSIM's and nuPlan's own code by `navsim_data check`.
