# H10 — PDM-style terms on every benchmark

**Owner:** Naren · **Priority:** 10 of 11 · **When:** 21–22 Oct 2026 · **Status:** done (enabler)

The facts are in [experiment.toml](experiment.toml). This page transcribes the sources named at each
claim and adds no analysis.

## Why

Item 10 of the 21 Oct priority list (Renesas Progress and Plan, 22 Oct 2026): "One set of PDM-style
terms on every benchmark (collision, drivable area, time to collision, progress, comfort), combined by
NAVSIM's formula". Why: "One vocabulary for which term moved. The same formula is not the same scale:
0.85 on nuScenes and 0.85 on NAVSIM are different claims". Decides: "Reported beside each benchmark's
own score; may decide later, one benchmark at a time". On 22 Oct the page listed it as "in progress ·
PR #102 (open)", its neuroNCAP scorer done, with "Left: merge main into #102, then its review".

paddock `docs/gate.md` gives the reason a headline is not enough: a candidate that collides more and
makes more progress can keep the same EPDMS or driving score.

## What was done

paddock #102 merged on 22 Oct, 18:08 PDT (`7d4c41c`). Under every gate pair the gate prints NAVSIM v1's
terms, NC × DAC × (5·TTC + 5·EP + 2·C) / 12, as far as the benchmark can measure them, with the same
pairing and unit as the headline; a term a benchmark cannot measure is named as "not measured", never
filled with a pass (`docs/gate.md`):

| Benchmark | NC | DAC | TTC | EP | C | Composite |
|---|---|---|---|---|---|---|
| NAVSIM v2 | ✓ | ✓ | ✓ | ✓ | history comfort | built by the gate |
| neuroNCAP | ✓ NAVSIM's at-fault rules | ✓ nuScenes' drivable area | ✓ | ✓ | ✓ | mean of every decision's composite |
| AlpaSim | at-fault collision | off-road | — | progress | — | — |
| nuScenes | box overlap | — (no map loaded) | — | ✓ | ✓ | — |
| Bench2Drive | collision rate | — | — | route completion | — | — |
| KITScenes, nuPlan | — | — | — | ✓ | ✓ | — |

The neuroNCAP scorer follows NAVSIM's rules term by term and scores each plan against the actors'
real motion. Its checks (2026-10-21, shakaal, `docs/benchmarks/neuroncap.md`): the engine's own
collision check, restated, agrees on 600 of 600 runs (three Drive-JEPA sweeps of 200; 223 collided); a
plan's 1 s point lies 0.28 m from where the car was 1 s later (1.47 m with y flipped); the human's logged
footprint is on the drivable raster at 100% of keyframes in all 14 scenes. Since #106 (23 Oct) every
neuroNCAP plan is first driven by NAVSIM's own vehicle simulator, ported to numpy, before it is scored.

What the terms showed first: re-scored, the camera fix (Drive-JEPA blind against camera-fixed, 20 pairs ×
10) is NEUTRAL on the NCAP score (+0.56, MDE 1.30) and on the composite (0.621 → 0.661, MDE 0.21), and
worse beyond t crit on comfort (0.998 → 0.935, t −6.7) (`docs/benchmarks/neuroncap.md`, `docs/gate.md`).

## Decision

**21 Oct, Naren: each benchmark's own score keeps deciding the gate.** The PDM-style terms are reported
beside it, and may decide later, one benchmark at a time, only after they agree with the official score
and catch changes at least as small (Renesas Progress and Plan, 22 Oct 2026). paddock `docs/gate.md`
adds why a row cannot fail a checkpoint: each is its own 5% test, and over the quick tier's 17 rows an
unchanged model would show a wrong-way row about 35% of the time if they were independent (51% over
`--tier all`'s 28).

## What is open

The empty cells. What they would take (`docs/gate.md`): for AlpaSim, the ego and actor poses each run
already writes; for Bench2Drive, `outside_route_lanes` and the 20 Hz `metric_info.json`, both written per
route and not yet read, or PR #79's recorder; "those need fresh baselines". For nuScenes and nuPlan,
loading their maps and agent tracks.
No benchmark has yet been moved to deciding on the terms.
