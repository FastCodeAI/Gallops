<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H08 · Real sideways velocity and a matched camera field of view are free gains

**Naren** · KITScenes, NAVSIM, nuScenes, neuroNCAP, Bench2Drive, AlpaSim · ⏳ running · 21 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 8 · **Evidence for:** D16 ([decision log](../README.md#decision-log))

## Question

Drive-JEPA trained with inputs paddock does not give it: the car's sideways velocity, and the camera field of view it was trained on. The switches are already built. Is each a free gain, about 1 h per arm through the gate? It decides whether to keep the inputs.

## Hypothesis

Claim: giving Drive-JEPA the real sideways velocity, and cameras matched to the field of view it trained on, improves it at no cost. Expected: each arm, through the regression gate, gains or holds on every benchmark (the page gives no size). Refuted for an input whose arm regresses on any benchmark: it is not free, and it is not kept.

## Predictions

⚠️ Not pre-registered: the camera arm was adopted in paddock PR #106 on a KITScenes comparison, and no prediction was written before it

## What we learned

So far, the camera half only: cropping every camera onto the OpenScene camera the checkpoint was trained on took KITScenes ADE from 3.222 m to 2.521 m (t −4.0, 223 keyframes), and paddock made the crop Drive-JEPA's default on every benchmark (#106, 23 Oct). No gate verdict for it is recorded. The sideways-velocity arm has not run: paddock still sends Drive-JEPA a sideways speed of 0.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H08.crop**<br>matched field of view: every camera cropped onto the OpenScene camera (64 degrees) Drive-JEPA trained on; paddock #106, merged 23 Oct | — | — | — | — | — |

- **H08.crop:** Paddock-side input change on the released weights; no training. Now the model's own setting, so every benchmark and every gate run gets it; a site that sets 'native' is the override and its runs are not valid.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H08.crop ★ | KITScenes ADE (m), cropped 3.222 → **2.521** (n=223 keyframes)<br>t -4.0; uncropped, KITScenes' 87-degree camera reached the model squeezed: objects 0.65x as wide and 0.55x as tall as in training | released, uncropped (native geometry) | ●●○○ moderate · open-loop | paddock PR #106 (f737ae5); src/paddock/adapters/models/drivejepa/model.yaml (DRIVEJEPA_IMAGE_GEOMETRY) |

## Evidence outside this repo

- Camera arm: paddock PR #106 (f737ae5) and the DRIVEJEPA_IMAGE_GEOMETRY comment in src/paddock/adapters/models/drivejepa/model.yaml on main.
- A gate smoke of #106's branch (crop and 2 Hz replanning together) left a state dir {runs}/gate/smoke/drivejepa-pr106-crop-rate-20261022/ under paddock's runs root with no verdict written down; its Bench2Drive lane exited 1.
