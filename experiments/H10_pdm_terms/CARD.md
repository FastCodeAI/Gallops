<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H10 · One set of PDM-style terms on every benchmark, combined by NAVSIM's formula (paddock #102)

**Naren** · NAVSIM, KITScenes, nuScenes, neuroNCAP, Bench2Drive, AlpaSim, nuPlan · ☑️ done · 21 Oct – 22 Oct 2026 · enabler

[← all experiments](../README.md#enablers) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 10

## Question

One vocabulary for which term moved: collision (NC), drivable area (DAC), time to collision (TTC), progress (EP) and comfort (C) on every benchmark, combined by NAVSIM's formula NC × DAC × (5 EP + 5 TTC + 2 C) / 12, reported beside each benchmark's own score. The same formula is not the same scale: 0.85 on nuScenes and 0.85 on NAVSIM are different claims.

## What we learned

paddock #102, merged 22 Oct: NAVSIM's five terms print under every gate pair, as far as each benchmark can measure them, and never change a verdict. neuroNCAP has all five by NAVSIM's rules, its collision check agreeing with neuroNCAP's own on 600 of 600 stored runs; NAVSIM has all five; AlpaSim NC, DAC and EP; nuScenes NC, EP and C; Bench2Drive NC and EP; KITScenes and nuPlan EP and C. Each benchmark's own score keeps deciding (21 Oct). Re-scored, the camera fix is NEUTRAL on neuroNCAP's score and composite but worse on comfort beyond t crit.

## Checks that decided something

- **neuroNCAP's collision check against the engine's own** (21 Oct): 600 of 600 runs agree (three Drive-JEPA sweeps of 200; 223 collided); the plan's frame and the drivable raster were also checked against logged motion → the NAVSIM-rule scorer fills all five terms on neuroNCAP
- **Which score decides the gate** (21 Oct): the same formula is not the same scale across benchmarks; the terms are shown, never deciding → each benchmark's own score keeps deciding the gate; the PDM-style terms are reported beside it and may decide later, one benchmark at a time, only after they agree with the official score and catch changes at least as small (Naren)
- **PDM rows are not part of the verdict**: each row is its own 5% test: if independent, an unchanged model would show a wrong-way row beyond t crit about 35% of the time over the quick tier's 17 rows and about 51% over --tier all's 28; a row worse beyond t crit is named under the overall line as a pointer
- **The camera fix, re-scored with the terms** (22 Oct): Drive-JEPA blind against camera-fixed, 20 pairs × 10, every plan driven by NAVSIM's simulator: NCAP score NEUTRAL (+0.56, MDE 1.30), composite NEUTRAL (0.621 → 0.661, MDE 0.21), comfort worse beyond t crit (0.998 → 0.935, t −6.7)

## Evidence outside this repo

- paddock PR #102 (7d4c41c); docs/gate.md 'PDM terms beside the score'; docs/benchmarks/neuroncap.md; tools/benchmarks/neuroncap/check_pdm_geometry.py and pdm_rescore.py.
