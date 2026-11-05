# H11 — HUGSIM trial

**Owner:** Navvrat † · **Priority:** 11 of 11 · **Started:** 21 Oct 2026 · **Status:** running — the
benchmark is in paddock; the check against WA-JEPA's 0.4462 and the adoption decision are open (enabler)

† The 21 Oct list leaves this item unassigned. It is recorded under Navvrat because he authored the
work since: paddock #118 (merged 27 Oct PDT, 28 Oct IST).

The facts are in [experiment.toml](experiment.toml). This page transcribes the sources named at each
claim and adds no analysis.

## Why

Item 11 of the 21 Oct priority list (Renesas Progress and Plan, 22 Oct 2026): "HUGSIM beside neuroNCAP,
checked by reproducing WA-JEPA's published 0.4462" · owner "—" · why: "436 scenarios from 4 datasets,
with attacking cars; its score is already PDM-style" · cost: "Large: no scene data here, a CUDA rebuild
for the Blackwell GPU, runs that do not repeat, ~19 min a scenario" · decides: "Whether to adopt it,
after a trial on a nuScenes + PandaSet slice".

The page's HUGSIM section adds, against it: an upstream steering bug (WA-JEPA patches it), half the
scenes with no back camera, runs that do not repeat because the controller is timed by the wall clock,
about 19 min a scenario against about 2.5 min for a neuroNCAP episode, and that the team's
crash-avoidance work is defined on neuroNCAP. Its plan: a trial on a nuScenes + PandaSet slice;
reproduce WA-JEPA's per-split numbers; measure re-run noise; then decide. WA-JEPA reports HUGSIM
zero-shot 0.4462 over 436 scenarios (the page's WA-JEPA card).

## What was done

**HUGSIM is a paddock benchmark** — #118, by Navvrat, merged 27 Oct PDT (28 Oct IST, `9c4e293`). A launcher runs
upstream's simulation loop in HUGSIM's interpreter; a bridge drives a paddock model node with the route
command only, at the model's own planning rate. HUGSIM is pinned at `62c690d` with two patches: upstream
PR #57's controller fix (without it a straight plan steered 4.3 m right in 14 m) and rendering only the
model's cameras. 16 of the 436 released scenarios fail before their first step for every model, so the
benchmark is 420. The headline is HUGSIM's HD-Score (`docs/benchmarks/hugsim.md`).

**Two campaigns, driven 22 Oct and ingested by #118** (`results/hugsim/`; scores ×100 in paddock's docs):

| Run | Model | Scenarios | HD-Score | PDMS | Route completion | Stored |
|---|---|---:|---:|---:|---:|---|
| `H11.drivejepa` | released Drive-JEPA (NAVSIM v2 checkpoint), 2 Hz | 419 of 420 | 33.8 | 71.7 | 39.8 | partial, `valid: false` |
| `H11.autoe2e` | AutoE2E v1.0, 4 Hz | 420 of 420 | 5.7 | 35.7 | 14.1 | valid |

Drive-JEPA's run is partial because HUGSIM itself crashed on `nuscenes/scene-0166-medium-01` at step 60,
on both attempts; AutoE2E finished it. Drive-JEPA ended 22.7% of episodes completed, 38.9% hitting a
vehicle, 6.9% hitting scenery, 11.5% off route and 20.0% at the step cap. Neither run is today's adapter:
Drive-JEPA was served by the adapter as of `ac7788f`, uncropped and picking on the plain score (main now
crops, #106, and uses momentum selection, #95); AutoE2E by `b507811`. One run per scenario: no spread is
known. The same campaign also drove the Final (Bench2Drive-trained Drive-JEPA) and Run 3A (WA-JEPA
tokens with Drive-JEPA's planner), which are not paddock adapters on `main` and are not stored
(`docs/benchmarks/hugsim.md`).

Verified in #118: the stored means reproduce the campaign's from each scenario's `eval.json`; the
tracked bridge matches the campaign's to 1e-12; patch 0001 gives bit-identical controls on 200 random
plans. Unverified: a scenario driven end to end through the merged launcher (`docs/benchmarks/hugsim.md`,
"Verified, and argued"; `docs/ROADMAP.md`).

## What is not done

- **The check the item named**: reproducing WA-JEPA's published 0.4462 (or its per-split numbers). No
  WA-JEPA HUGSIM run is in paddock; paddock's `wajepa` adapter (#83) has a neuroNCAP smoke only.
- **Re-run noise**: no repeat of any scenario.
- **A run through the harness itself**, and a re-run of either model on `main`'s adapters.
- **The decision**: HUGSIM has no column in paddock's support matrix yet, and it is not a gate benchmark.

## How it will be decided

The 21 Oct call stands: neuroNCAP stays, and HUGSIM is trialled beside it. Whether to adopt it is decided
after the trial reproduces WA-JEPA's numbers and measures re-run noise (Renesas Progress and Plan, 22 Oct
2026, "HUGSIM").
