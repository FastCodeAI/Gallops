<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H12 · nuScenes teacher-scene selection for the privileged teacher's data collection

**Ruhan** · neuroNCAP · ⏸ parked · 14 Oct – 5 Nov 2026 · enabler

[← all experiments](../README.md#enablers) · [the account](https://github.com/FastCodeAI/Gallops/blob/ca5817d2cf6a48bf910b12b0d72a1df487009205/docs/data.md) (on commit ⎇) · [experiment.toml](experiment.toml)

> The record is on commit `ca5817d`, not merged yet; its links go to that commit.

**Evidence for:** D19 ([decision log](../README.md#decision-log))

## Question

Which nuScenes train scenes should be reconstructed with NeuRAD for the privileged closed-loop teacher's training data, without touching paddock's 14-scene neuroNCAP eval set or the scenes already reconstructed for the swerve finetune?

## What we learned

A selector that scores every eligible nuScenes train scene and picks 12 training scenes by turn / curve / straight quota plus 3 disjoint validation scenes, with paddock's own 3 s, 2.0 m command rule; an annotator draws the 'good actor' boxes on each pick's front video for human review. Ported from paddock PR #75 with its horizon-truncation bug fixed; no selection is recorded on the branch. Parked: PR #19 closed unmerged on 5 Nov 2026, as nuScenes curation is not needed for now. Its review lists what to fix before reopening it.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H12.r1**<br>the selector and annotator as ported into gallops (19 Oct); tests only, no selection recorded | paddock ruhan/alpamayo-teacher-scenes 9a3a2d4 (select_scenes.py, annotate_actors.py) | — | nuscenes `v1.0-trainval, train scenes not in neuroNCAP's set or already reconstructed` (train) — the pool the selector scores; what it picked is not recorded on the branch | — | — |

- **H12.r1:** Not a training run: it records the code the branch adds. Exclusions come from gallops.data.nuscenes_real (neuroncap_scenes(), which imports paddock's OFFICIAL_SET, and RECONSTRUCTED), not a restated list. Quotas: train turn_left 2, turn_right 2, curved_road 4, straight 4; val turn_left 1, curved_road 1, straight 1. Runs in an env with nuscenes-devkit, outside make setup.

## Checks that decided something

- **Horizon-truncation fix and its regression test** (19 Oct): the original shrank the 3 s lookahead for every scene's last ~6 samples, biasing them toward 'straight'; tail samples are now excluded from every category, as paddock's nuScenes harness drops them. tests/test_teacher_scenes.py guards it; make check passed (232 passed, 4 skipped), per the commit message

## Evidence outside this repo

- The original selector and annotator: paddock branch ruhan/alpamayo-teacher-scenes (9a3a2d4, 14 Oct; paddock PR #75), not on paddock main.
