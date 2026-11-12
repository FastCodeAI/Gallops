<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H11 · HUGSIM trial beside neuroNCAP, checked by reproducing WA-JEPA's published 0.4462

**Navvrat** · HUGSIM · ⏳ running · 21 Oct 2026 · enabler

[← all experiments](../README.md#enablers) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 11

## Question

Should HUGSIM be adopted beside neuroNCAP? It has 436 scenarios from 4 datasets, with attacking cars, and its score is already PDM-style. Against it: no scene data on the machine, a CUDA rebuild for the Blackwell GPU, runs that do not repeat (its controller is timed by the wall clock), about 19 min a scenario. Decided after a trial on a nuScenes + PandaSet slice, checked by reproducing WA-JEPA's published 0.4462.

## What we learned

So far: HUGSIM is a paddock benchmark (#118, merged 27 Oct PDT, 28 Oct IST) on 420 of the release's 436 scenarios (16 fail for every model), with upstream PR #57's controller fix. A 22 Oct campaign scored the released Drive-JEPA at HD-Score 0.338 on 419 of 420 (stored partial: HUGSIM crashed on one scenario) and AutoE2E at 0.057 on 420, one run per scenario, on adapters older than main's. WA-JEPA's 0.4462 has not been reproduced and no re-run noise is measured; no episode has yet been driven through the merged harness.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H11.drivejepa**<br>released Drive-JEPA (NAVSIM v2 ViT-L checkpoint) on HUGSIM, 2 Hz, command only; 22 Oct campaign ingested by paddock #118 | — | — | — | — | — |
| **H11.autoe2e**<br>AutoE2E v1.0 (KITScenes epoch 5) on HUGSIM, every step (4 Hz), no command; 22 Oct campaign ingested by paddock #118 | — | — | — | — | — |

- **H11.drivejepa:** Served by paddock's drivejepa adapter as of ac7788f plus a per-call record of the 32 candidates: cameras uncropped, plain-score pick (main now crops, #106, and selects with momentum, #95). Front camera rendered only. Two lanes sharing one scenario queue. The weights are named in run.json's notes, not hashed there.
- **H11.autoe2e:** From paddock b507811 unmodified; its adapter has changed since (#92, #97, #98, #107, #108). All six cameras rendered; upstream's no-route input.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H11.drivejepa ★ | HUGSIM HD-Score **0.338133** (n=419 scenarios)<br>×100 in paddock's docs: 33.8; PDMS 71.7, route completion 39.8, NC 80.8, DAC 95.2, TTC 75.5, comfort 93.6. Stored partial / valid: false: HUGSIM crashed on nuscenes scene-0166-medium-01. One run per scenario. WA-JEPA's published 0.4462 is a different model over 436 scenarios. | — | ●●○○ moderate | paddock results/hugsim/drivejepa/campaign-87d4f5097a85db3c-ce3603/metrics.json |
| H11.autoe2e | HUGSIM HD-Score 0.056697 (n=420 scenarios)<br>×100 in paddock's docs: 5.7; PDMS 35.7, route completion 14.1. Valid, complete. Half its episodes end within 18 steps (4.5 s), mostly against scenery. | — | ●●○○ moderate | paddock results/hugsim/autoe2e/campaign-f7d8f9d4b9736ddd-d03155/metrics.json |

## Checks that decided something

- **HUGSIM or neuroNCAP** (21 Oct): for: 436 scenarios from 4 datasets with attacking cars, a PDM-style score, WA-JEPA's 0.4462 to check against; against: no scene data, a CUDA rebuild, runs that do not repeat, an upstream steering bug, half the scenes without a back camera, about 19 min a scenario against about 2.5 min for a neuroNCAP episode, and crash-avoidance work defined on neuroNCAP → neuroNCAP stays, and HUGSIM is trialled beside it (who is not stated)
- **The scenario set** (27 Oct): 16 of the 436 released scenarios fail before their first step for every model (15 PandaSet *-medium-02 pass an argument the released ConstantPlanner does not take; one Waymo scenario has start_velo '1R'); the benchmark is 420 → listed in harness.EXCLUDED and in every run.json
- **What #118 verified** (27 Oct): verified: the stored means reproduce the campaign's from each eval.json; bridge.py's outputs equal the campaign's bridge to 1e-12; patch 0001 (upstream PR #57) gives bit-identical controls on 200 plans; both patches apply at 62c690d. Argued: patch 0002 renders what the campaign rendered; the node drove as main's adapter did on 22 Oct (5 of 7 scenarios identical, the two longest diverged). Unverified: launcher.py running a scenario end to end

## Evidence outside this repo

- paddock PR #118 (9c4e293); results/hugsim/drivejepa/campaign-87d4f5097a85db3c-ce3603/ and results/hugsim/autoe2e/campaign-f7d8f9d4b9736ddd-d03155/ (metrics.json, run.json); docs/benchmarks/hugsim.md.
