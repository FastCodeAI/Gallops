# H06 — Global read and side braking

**Owner:** Navvrat · **Priority:** 6 of 11 · **Started:** by 19 Oct 2026 · **Status:** running — partly
tested; the deciding neuroNCAP test has not run (hypothesis)

The facts are in [experiment.toml](experiment.toml). Everything below is transcribed from Renesas
Progress and Plan, 22 Oct 2026, whose 4-camera numbers are "Navvrat's (dashboard and 19 Oct failure
report); not re-run by us". No analysis is added.

## Why

On neuroNCAP only two of the five side pairs fail: 0103 (7 of 10 runs hit) and 0110 (3 of 10). In
every one of the 26 side episodes checked, the crossing car is in the front camera's view from the first
tick, 21–35 m away; 9 of Drive-JEPA's 10 side crashes are runs where it never brakes. WA-JEPA, on four
cameras, hit 3 of 3 on side 0103 but braked in all three. The leading explanation is Drive-JEPA's
path-only reads: the car starts about 10 m off every candidate path. That is argued, because WA-JEPA also
differs in encoder, planner and training data ("Side crashes are a braking failure, not a camera one",
19 Oct, from recorded runs). The next step named there: on Navvrat's stack, global read on versus off on
these two pairs.

Item 6 of the priority list: "The global read is what makes a model brake for the crossing car" ·
Navvrat · "~5 h GPU (on vs off, 10 seeds, 2 pairs)" · decides "Keep the global read, or move side
crashes to the scorer work".

## What was done

The global read is part of the agreed merge base, Navvrat's 4-camera stack: global tokens over all four
cameras that every plan slot reads, beside its path reads (design decision 5, "built · partly tested").
Two ablations, both global read **off** against on, are on the page:

| Run | Benchmark | On → off | Difference | Reading |
|---|---|---|---|---|
| `H06.navsim_off` | NAVSIM, 500 scenes | 0.870 → 0.852 | −0.018, 95% CI −0.038 to −0.000 | mostly lane keeping (0.978 → 0.946) |
| `H06.b2d_off` | Bench2Drive, all 25 routes | 51.1 → 50.4 | −0.7, 95% CI −12.6 to +11.0 | no detectable effect; the same model re-run moves a route 15.8 points on average |

Also from the page: on passing NAVSIM scenes the global read moves the plan most of any input (−0.038).
The record marks both results `no-record`, strength argued: the runs exist only on Navvrat's dashboard.
Neither run's date is given.

## What is not done

The test this item exists for: global read on versus off on neuroNCAP side 0103 and 0110, 10 seeds each.
The page says it needs Navvrat's adapter rebased onto the camera fix (paddock #62) first, because the
stack's adapter predates it and would be blind on neuroNCAP. As of 3 Nov 2026 no commit in gallops or
paddock records that test or the rebase; paddock has no adapter for the 4-camera stack on `main`
(its HUGSIM page says the 22 Oct campaign also drove "Run 3A (WA-JEPA tokens with Drive-JEPA's
planner)", which is not a paddock adapter on `main`).

## How it will be decided

On versus off on the two failing side pairs, 10 seeds each: if the global read makes the stack brake
for the crossing car, keep it; if not, side crashes move to the scorer work (item 6's "Decides").
