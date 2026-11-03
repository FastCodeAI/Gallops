<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# H02 · Smallest valid gate: subset size per benchmark from the spread of past paired runs

**Naren** · NAVSIM, KITScenes, nuScenes, neuroNCAP, Bench2Drive, AlpaSim · ☑️ done · 20 Oct – 21 Oct 2026 · enabler

[← all experiments](../../README.md#enablers) · [the account](README.md) · [experiment.toml](experiment.toml)

**Priority:** 2 · **Evidence for:** D13, D15 ([decision log](../../README.md#decision-log))

## Question

How small can each benchmark's gate subset be and still catch a change worth catching? Measured as the smallest drop each subset detects at 80% power, from the spread of past paired runs (CPU and existing results only), so that every checkpoint can go through the gate.

## What we learned

Smallest drop each subset detects (80% power): neuroNCAP closed tier 0.73 for a small change and 2.85 for a big one, out of 5; its quick tier 2.5 / 6.7, a pipeline check only; NAVSIM about 0.0035; KITScenes 0.51 m for a big change; Bench2Drive only 17–38 points; AlpaSim about 0.23, argued. Decided 21 Oct: the subset sizes stay, every verdict prints its MDE, and each benchmark's own score decides. The first timed run (20 Oct, released against itself) took 2 h 40 min, the quick tier 19 min, and matched its baselines to the last digit: HOLD.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **H02.mde**<br>smallest detectable drop (MDE, 80% power, two-sided 5%) per gate subset, from the spread of existing paired runs | — | — | — | — | — |
| **H02.timing**<br>first timed gate run: released Drive-JEPA against itself, --tier all, three lanes for the quick tier | — | — | — | — | `{data.navsim.exp}/Drive-JEPA-cache/`<br>sha256 `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` |

- **H02.mde:** Cost: CPU, existing results. A small change is a finetune of the same model; a big change is one that changes what the model sees (page). Bench2Drive's 17–38 points and the not-adopted neuroNCAP alternative are under checks: a range is not one number.
- **H02.timing:** 18:49–21:30 PDT on a quiet evening (box CPU 14%, NAS 65 MB/s, GPU shared with one other job); the 16 Oct first smoke (one lane, before tiers) took 7.7 h. Each run.json records paddock 340b7fa, clean, and the adapter as it then was: image_geometry native (uncropped), momentum off -- before #106 and #95 changed both defaults.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| H02.mde ★ | neuroNCAP MDE, small change (closed tier, 6 pairs × 3 seeds) **0.73** (n=6 scenario pairs)<br>out of 5; measured from past paired runs per the page, not in either repo | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |
| H02.mde | neuroNCAP MDE, big change (closed tier, 6 pairs × 3 seeds) 2.85 (n=6 scenario pairs)<br>out of 5 | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |
| H02.mde | neuroNCAP MDE, small change (quick tier, 0103 × 3 shapes, 1 seed) 2.5 (n=3 scenario pairs)<br>out of 5; a pipeline check only | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |
| H02.mde | neuroNCAP MDE, big change (quick tier, 0103 × 3 shapes, 1 seed) 6.7 (n=3 scenario pairs)<br>out of 5; a pipeline check only | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |
| H02.mde | NAVSIM MDE in EPDMS, small change (1,000 of 12,146 navtest scenarios) 0.0035 (n=136 logs)<br>'about 0.0035' | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |
| H02.mde | KITScenes MDE in ADE (m), big change (all 223 keyframes) 0.51 (n=140 scenes) | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |
| H02.mde | AlpaSim MDE (10 clips, public_2601_first10) 0.23 (n=10 clips)<br>'about 0.23, argued from PR #93's swings, not measured' | — | ●○○○ argued · no-record | Renesas Progress and Plan, 22 Oct 2026 (Next steps item 2; Regression gate subset table) |

## Regression gate

**H02.timing** — 20 Oct 2026, all tier, against the released model's own earlier runs (15–16 Oct smokes; 12 Oct full nuScenes and neuroNCAP runs): ⏸ **HOLD**

| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |
|---|---|---|---|---|---|---|---|
| KITScenes | ADE (m) | 3.222 | 3.222 |  |  | 140 | ⚪ NEUTRAL |
| nuScenes | L2 | 2.497 | 2.497 |  |  | 30 | ⚪ NEUTRAL |
| NAVSIM | EPDMS | 0.875 | 0.875 |  |  | 136 | ⚪ NEUTRAL |
| neuroNCAP | NCAP score (quick: 0103 × 3 shapes, 1 seed) | 4.000 | 4.000 |  |  | 3 | ⚪ NEUTRAL |
| neuroNCAP | NCAP score (closed: 6 pairs × 3 seeds) | 3.175 | 3.175 |  |  | 6 | ⚪ NEUTRAL |
| Bench2Drive | driving score (+7.0; cannot ship) |  |  | 0.82 | 28.7 | 7 | ⚪ NEUTRAL |
| AlpaSim |  |  |  |  |  |  | — NOT RUN |

Total 2 h 40 min: quick tier 19 min, closed tier 2 h 21 min. HOLD overall, as expected for the model against itself. The page and paddock docs/gate.md record the open-loop and neuroNCAP pairs as identical to the last digit; NEUTRAL is the gate's word for a zero difference (tools/gate.py), not a word copied from the run. The first neuroNCAP row is the quick tier (0103 × 3 shapes, 1 seed), the second the closed tier (6 pairs × 3 seeds). AlpaSim was not run: Drive-JEPA had no AlpaSim baseline yet, and its model.yaml did not declare AlpaSim.

## Checks that decided something

- **Smallest drop each subset detects (80% power)** (21 Oct): neuroNCAP closed 0.73 small / 2.85 big (of 5); quick 2.5 / 6.7, a pipeline check only; NAVSIM about 0.0035; KITScenes 0.51 m big; Bench2Drive only 17–38 points (paired route differences of 13.6–30.4 sd); AlpaSim about 0.23, argued → the subset sizes stay, and every verdict prints its MDE (Naren); shipped in paddock #100, merged 22 Oct
- **neuroNCAP: the same ~20 episodes as 20 pairs × 1 seed** (21 Oct): would detect 0.55 small / 1.49 big, because scene-to-scene noise is larger than seed noise → not adopted: Naren kept the subset sizes
- **Bench2Drive's 7 routes and the quick tier's 3 neuroNCAP episodes** (22 Oct): Bench2Drive's 7 routes detect only a 17–38 point drop; 3 episodes are too few to ship on → both can fail a checkpoint but never ship one (can_ship = false), paddock #100, merged 22 Oct
- **Every subset redrawn within groups (paddock #105)** (22 Oct): groups by maneuver, speed, city, crash type, scenario family; every group the same share, random within it (seed 20261022). NAVSIM 1,018 tokens from 126 logs; nuScenes 1,000 keyframes from 143 scenes (was the first 30 scenes); neuroNCAP closed 15 pairs × 1 run (was 6 × 3), quick 1 pair per crash type (was always 0103); Bench2Drive 7 of all 220 routes; KITScenes decides on its 186 val keyframes → agreed 22 Oct (commit message; who is not stated), merged 23 Oct
- **AlpaSim's 10 gate scenes** (27 Oct): the 22 Oct draw of 10 of all 913 never ran (none on disk); #110 drew 10 from the 50 downloaded; 2 were not on the drivers' scene list (#116) and on 2 more every model left the road at once (#120) → swap only those 4 for seeded picks from the same group, keep the other 8 so the baseline keeps its finished scenes; the quick tier's crash-check scene redrawn as 1acf2f5d

## Evidence outside this repo

- The MDE table exists only on the team page (Renesas Progress and Plan, 22 Oct 2026); paddock records Bench2Drive's 17-38 points in tools/gate_smoke/bench2drive.toml and docs/gate.md.
- The 20 Oct run: paddock docs/gate.md ('The tiered smoke, timed'), state dir {runs}/gate/smoke/drivejepa-released-timing-20261021/ under paddock's runs root.
- Later gate changes: paddock PRs #100, #105, #110, #111, #113, #114, #115, #116, #120 (docs/gate.md on main).
