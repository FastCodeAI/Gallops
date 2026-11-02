# H02 — Smallest valid gate

**Owner:** Naren · **Priority:** 2 of 11 · **When:** 20–21 Oct 2026 · **Status:** done (enabler)

The facts (runs, results, the gate run and its pairs) are in [experiment.toml](experiment.toml). This
page is the account, transcribed from the sources named at each claim. It adds no analysis.

## Why

Every later item on the 21 Oct priority list is decided by paddock's regression gate, so the gate has
to be cheap enough to run on every checkpoint and sensitive enough to mean something. Item 2 asked
for the subset size per benchmark, from the spread of past paired runs, at the cost of CPU time and
existing results only (Renesas Progress and Plan, 22 Oct 2026, "Next steps", item 2).

## What was done

**The smallest drop each subset detects** (run `H02.mde`). Each subset's MDE at 80% power, two-sided
5%, measured from the spread of existing paired runs; AlpaSim's is argued from PR #93's swings, not
measured. A small change is a finetune of the same model; a big change is one that changes what the
model sees (Renesas Progress and Plan, 22 Oct 2026, "Regression gate" subset table):

| Subset | Tier | Units | Smallest drop detected |
|---|---|---|---|
| neuroNCAP, 6 pairs × 3 seeds | closed | 6 scenario pairs | 0.73 small / 2.85 big, out of 5 |
| neuroNCAP, 0103 × 3 shapes, 1 seed | quick | 3 scenario pairs | 2.5 small / 6.7 big: a pipeline check only |
| NAVSIM, 1,000 of 12,146 navtest scenarios | quick | 136 logs | about 0.0035 EPDMS, small change |
| KITScenes, all 223 keyframes | quick | 140 scenes | 0.51 m ADE, big change |
| nuScenes, first 30 of 150 val scenes | quick | 30 scenes | — (none given) |
| Bench2Drive, 7 of our 50 routes | closed | 7 routes | only a drop of 17–38 points |
| AlpaSim, 10 clips | closed | 10 clips | about 0.23, argued |

These numbers exist only on the team page, except Bench2Drive's 17–38 points, which paddock records in
`tools/gate_smoke/bench2drive.toml` ("from paired route differences of 13.6–30.4 sd") and `docs/gate.md`.
The record marks them `no-record`, strength argued.

**The first timed gate run** (run `H02.timing`, 20 Oct 2026). The released Drive-JEPA against itself,
`--tier all`, three lanes for the quick tier, 18:49–21:30 PDT (paddock `docs/gate.md`, "The tiered
smoke, timed"; Renesas Progress and Plan, 22 Oct 2026):

| Subset | Tier | Wall clock | Against its baseline |
|---|---|---|---|
| KITScenes, 223 keyframes | quick | 5.7 min | identical, ADE 3.222 |
| nuScenes, 30 scenes | quick | 7.8 min | identical, L2 2.497 |
| NAVSIM, 1,000 scenarios | quick | 13.4 min | identical, EPDMS 0.875 |
| neuroNCAP 0103 × 3 shapes, 1 seed | quick | 12.8 min | identical, 4.000 |
| neuroNCAP, 6 pairs × 3 seeds | closed | 45.0 min | identical, 3.175 |
| Bench2Drive, 7 routes | closed | 95.7 min (7.4–23.0 min a route) | NEUTRAL +7.0 driving score, t 0.82, MDE 28.7; cannot ship |
| **Total** | | **2 h 40 min** | quick tier 19 min, closed tier 2 h 21 min; **HOLD** overall |

AlpaSim was not run: Drive-JEPA had no AlpaSim baseline yet (page). The evening was quiet (box CPU 14%,
NAS 65 MB/s, GPU shared with one other job); the 16 Oct first smoke, one lane and before the tiers,
took 7.7 h, and one run cannot separate the evening from the code (paddock `docs/gate.md`).

Read from the run's own files, not from a document: each candidate `run.json` records paddock commit
`340b7fa` (clean) and the Drive-JEPA adapter as it then was, `image_geometry: native` (uncropped) and
`momentum: off`. The manifest pairs KITScenes, NAVSIM and Bench2Drive with the 15–16 Oct smoke runs and
nuScenes and neuroNCAP with the 12 Oct full runs. The weights it evaluated are the model's default
checkpoint, `drive_jepa_perception_based_agent_vitl_v2.ckpt`; its sha256 was re-hashed on 3 Nov and
matches `740335c800e2…`. The identical pairs are recorded NEUTRAL because that is what the gate's code
calls a zero difference (`tools/gate.py`); the sources say "identical".

## Decision

- **21 Oct, Naren: the subset sizes stay, and every verdict prints its MDE.** The alternative, the same
  ~20 neuroNCAP episodes spread as 20 pairs × 1 seed, would detect 0.55 / 1.49, because scene-to-scene
  noise is larger than seed noise; it was not adopted (Renesas Progress and Plan, 22 Oct 2026,
  "Regression gate"). The MDE column shipped in paddock #100, merged 22 Oct (`ee62a6b`), which also
  made Bench2Drive's 7 routes and the quick tier's 3 neuroNCAP episodes able to fail a checkpoint but
  never ship one.
- **21 Oct, Naren: each benchmark's own score keeps deciding the gate**; the PDM-style terms are shown
  beside it (see [H10](../H10_pdm_terms/README.md)).

## What changed in paddock since

From paddock's git history on `main` (each PR's merge commit and `docs/gate.md`):

- **Every subset redrawn within groups** — #105, agreed 22 Oct (the commit message says "Agreed
  2026-10-22", not by whom), merged 23 Oct (`bbcbc42`). Each benchmark is split into groups by maneuver,
  speed band, city or town, crash type or scenario family; every group gets the same share, drawn at
  random with seed 20261022. NAVSIM became 1,018 tokens from 126 logs; nuScenes 1,000 keyframes from 143
  of 150 scenes (was the first 30 scenes, two city areas); neuroNCAP's closed tier 15 pairs × 1 run (was
  6 × 3) and its quick tier one pair per crash type (was always 0103); Bench2Drive 7 of all 220 routes,
  one per group; KITScenes still runs all 223 and decides on its 186 `val` keyframes. The gate's mean
  now weighs rare driving more than the full benchmark does (`docs/gate.md`).
- **AlpaSim's subset** — #110 (26 Oct, `54306af`): the 22 Oct draw of 10 of all 913 scenes never ran,
  none being on disk, so the 10 were drawn from the 50 already downloaded. #116 (27 Oct, `9819e36`):
  two were not on the scene list the drivers hand the wizard and were swapped. #120 (27 Oct, `9585f81`):
  on two more, constant velocity and Drive-JEPA both left the road at once, so they were swapped too; the
  other 8 were kept so the baseline kept its finished scenes, and the quick tier's scene became `1acf2f5d`.
- **Quick-tier crash checks** — #113 (26 Oct, `74c5756`): the quick tier also drives one Bench2Drive
  route (3717) and one AlpaSim scene; one sample each, never a verdict, but a run that cannot drive is
  INCOMPLETE and stops the gate before the closed tier.
- **Lanes** — #111 (26 Oct, `31e3714`): neuroNCAP, Bench2Drive and AlpaSim run side by side, NAVSIM on
  two model nodes; #115 (26 Oct, `058a88d`) pins AlpaSim's containers to the lane's cores so it can join them.
- **Speed** — #114 (26 Oct, `fd20eb5`): neuroNCAP reads a 41 MB nuScenes cut-down to its 14 scenes
  instead of 2.5 GB of tables at every start, skips the two drawn debug views (about 1.05 s of each
  2.66 s step), and with several lanes the longest subsets launch first. The quick tier is now "about
  40 minutes" with the crash checks (`docs/queue.md`); no new end-to-end timing of `--tier all` is
  recorded in paddock.
- **Model defaults** — #106 (23 Oct) cropped Drive-JEPA's cameras by default and #95 (26 Oct) turned
  momentum selection on, so the 20 Oct baselines no longer describe the default model; the gate queue's
  baselines are remade when a model's own settings change (`docs/queue.md`). See
  [H08](../H08_sideways_velocity_fov/README.md) and [H09](../H09_shared_runner/README.md).

## What is open

Not recorded anywhere found: the MDE of each subset as redrawn by #105 (the 21 Oct table is for the
old subsets), and nuScenes' MDE. The Bench2Drive smoke baseline was mostly "Agent got blocked" and has
not been checked against the 50-route runs (paddock `docs/gate.md`, Limits).
