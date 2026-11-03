<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H06 · The global read is what makes a model brake for the crossing car

**Navvrat** · neuroNCAP, NAVSIM, Bench2Drive · ⏳ running · 19 Oct 2026 · hypothesis

[← all experiments](../../README.md#hypothesis-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 6

## Question

neuroNCAP's side crashes on 0103 and 0110 are runs where Drive-JEPA never brakes, though the front camera already sees the crossing car. Navvrat's 4-camera stack adds a global read (scene-level tokens over all four cameras that every plan slot reads, beside its path reads). Is that read what makes a model brake for the crossing car? Cost about 5 h GPU: on vs off, 10 seeds, 2 pairs. It decides whether to keep the global read or move side crashes to the scorer work.

## Hypothesis

Drive-JEPA's plan slots read the image only along their own paths, and the crossing car starts about 10 m off every candidate path; WA-JEPA, which reads more of the scene, brakes on side 0103 where Drive-JEPA does not (argued: the two also differ in encoder, planner and data). Claim: in Navvrat's 4-camera stack, the global read is what makes it brake for the crossing car. Expected: with the global read on it brakes on neuroNCAP side 0103 and 0110, and with it off it does not (no size is given). Refuted if on and off brake alike over 10 seeds on the two pairs: then side crashes move to the scorer work.

## Predictions

⚠️ Not pre-registered: the two ablations below predate the item (Navvrat's dashboard and 19 Oct failure report) and no prediction was written for the neuroNCAP test, which has not run

## What we learned

So far, open loop and one closed-loop benchmark: turning the global read off costs NAVSIM 0.018 on 500 scenes (0.870 → 0.852, 95% CI −0.038 to −0.000), mostly lane keeping, and has no detectable effect on Bench2Drive's 25 routes (51.1 → 50.4, CI −12.6 to +11.0). The side-crash question itself is untested.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H06.navsim_off**<br>global read off, NAVSIM 500 scenes, Navvrat's 4-camera stack | — | — | — | — | — |
| **H06.b2d_off**<br>global read off, Bench2Drive all 25 routes, Navvrat's 4-camera stack | — | — | — | — | — |

- **H06.navsim_off:** Run date not given on the page. Separately, on passing navtest scenes turning the global read off moves the plan most of any input (−0.038).
- **H06.b2d_off:** Run date not given on the page.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H06.navsim_off ★ | NAVSIM score, global read off 0.870 → **0.852** (n=500 scenes)<br>−0.018, 95% CI −0.038 to −0.000; mostly lane keeping (0.978 → 0.946). The page does not name the metric for this pair. | global read on | ●○○○ argued · no-record, open-loop | Renesas Progress and Plan, 22 Oct 2026 (design decision 5), from Navvrat's dashboard |
| H06.b2d_off | Bench2Drive driving score, global read off 51.1 → 50.4 (n=25 routes)<br>−0.7, 95% CI −12.6 to +11.0: no detectable effect; the same model re-run moves a route 15.8 points on average | global read on | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (design decision 5), from Navvrat's dashboard |

## Checks that decided something

- **Side crashes are a braking failure, not a camera one** (19 Oct): only 0103 (7 of 10 runs hit) and 0110 (3 of 10) of the five side pairs fail; in all 26 side episodes checked the crossing car is in the front camera from the first tick, 21–35 m away; 9 of Drive-JEPA's 10 side crashes never brake. WA-JEPA (4 cameras, 3 runs) hits 3 of 3 on side 0103 but all braked → more cameras dropped as the side fix; next: global read on vs off on these two pairs (this item)

## Evidence outside this repo

- Both ablations: Navvrat's dashboard and his 19 Oct failure report, as copied onto Renesas Progress and Plan (22 Oct 2026); no run record in gallops or paddock.
