# H08 — Sideways velocity and camera field of view

**Owner:** Naren · **Priority:** 8 of 11 · **Started:** 21 Oct 2026 · **Status:** running — the camera
half is in paddock, the sideways-velocity half has not run (hypothesis)

The facts are in [experiment.toml](experiment.toml). This page transcribes the sources named at each
claim and adds no analysis.

## Why

Item 8 of the 21 Oct priority list: "Real sideways velocity and a matched camera field of view are free
gains" · Naren · why now: "Drive-JEPA trained with inputs paddock does not give it; the switches are
already built" · cost "~1 h per arm through the gate" · decides "Keep the inputs or not" (Renesas
Progress and Plan, 22 Oct 2026). The 19 Oct status names it as the gate's next job: "The two input fixes
(sideways velocity, camera field of view) through the gate."

What paddock gives Drive-JEPA instead, from its own docs:

- **Sideways velocity.** On NAVSIM, Drive-JEPA's paddock policy rebuilds `ego_velocity` as `[speed, 0]`
  from the CAN bus, dropping lateral velocity and acceleration; on that data |vy| ≈ 0.09 m/s against
  |vx| ≈ 4.4 m/s (max 0.40). It "comes from the NeuroNCAP wire contract" (`docs/benchmarks/navsim.md`).
  On AlpaSim the node's input has forward speed only, so the model gets 0 for sideways speed and
  acceleration; on scene 04394343 the native driver was sent 0.78 m/s and 4.0 m/s² at the handover and
  the plans differ from the next replan. "Passing them is a separate change" (`docs/models/drivejepa.md`).
- **Field of view.** Drive-JEPA was trained on the OpenScene camera (64 degrees). Uncropped, a wider
  camera reaches the model squeezed: on KITScenes (87 degrees) objects arrive 0.65x as wide and 0.55x as
  tall as in training (Drive-JEPA's `model.yaml` on paddock `main`).

## What was done

**The camera arm (run `H08.crop`).** paddock #106, by Naren, merged 23 Oct (`f737ae5`), set
`DRIVEJEPA_IMAGE_GEOMETRY: "openscene"` in Drive-JEPA's `model.yaml`: every camera is cropped and
rescaled onto the OpenScene camera, "the way the devkit's own KITScenes loader does", for every
benchmark and every gate run; a site that sets `native` is the override and its runs are not valid. The
evidence it cites: KITScenes ADE 3.222 m uncropped against 2.521 m cropped (t −4.0, 223 keyframes),
open loop. The same PR made Drive-JEPA replan at 2 Hz by default and drove neuroNCAP plans through
NAVSIM's simulator before scoring. Strength moderate: one benchmark, open loop.

No gate verdict for the crop is written down anywhere found. On disk there is a gate smoke state for the
PR's branch, `{runs}/gate/smoke/drivejepa-pr106-crop-rate-20261022/` (paddock's runs root), which ran
the crop and the 2 Hz rate together against the released model's runs on the redrawn subsets (state dir
`drivejepa-released-sampling-20261022`); its manifest pairs six subsets and its state file records the
Bench2Drive lane exiting 1. No verdict was read from it for this record.

**The sideways-velocity arm** has not run. No commit in paddock or gallops since 21 Oct passes Drive-JEPA
a sideways speed.

## What is next

The sideways-velocity arm through the gate (about 1 h per arm, by the page's estimate), and a gate
verdict for the crop on its own if one is wanted: the gate queue's Drive-JEPA baselines are now the
cropped model, because baselines are remade when a model's own settings change on `main`, "as PR #106
changed Drive-JEPA's" (paddock `docs/queue.md`).

## How it will be decided

Each input through paddock's regression gate: keep it if its arm does not regress anywhere (item 8's
"Keep the inputs or not").
