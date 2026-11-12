# H15 · The perception side-car: overview, tracker and discussion log

**Updated 11 Nov 2026.** This is the common record of H15. Every document of this work links here.
- Every change to one of them adds a dated line to the [log](#log) and updates the [tracker](#tracker).
- A decision also gets its D-number in [experiments/decisions.toml](../../experiments/decisions.toml).

## The crux

Drive-JEPA plans from camera tokens, and has no explicit account of the other agents. H15 asks two questions.
- **Part A.** Can a small side-car, reading the same frozen ViT, find the agents ahead in 3D and say where each
  will be in a second?
- **Part B.** Given that account, does Drive-JEPA's planner plan better on scenes it never trained on?

## Where it stands (11 Nov)

- **Part A, the side-car.**
  - Steps 1–5 are done: v3, 18.4 M parameters, 24 epochs, select F1 0.717 over three seeds.
  - Step 7 is done: on nuScenes it finds 53% of the vehicles within 30 m, so Q9 holds. Warping the images to
    NAVSIM's camera did not help.
  - Step 6, reading the report logs once, waits for the owner.
- **Part B, the side-car in the planner.**
  - Four experiments are chosen and planned, one document each, in this order: B-1, B-2, B-3, B-4.
  - They share one plan of foundations, B-0, and a LoRA primer.
  - **Execution opened on 11 Nov (D24): B-0 in full, then B-1.** B-2 to B-4 wait for the review after B-1.
    What has run is in the [log](#log).

![H15: Part A, and Part B's four experiments in order](../../experiments/H15_perception_sidecar/figures/b-overview.png)

<a id="naming"></a>

## Naming

**Part B's experiments are named `B-<n>-<v>`.**
- `n` is the experiment:
  - 0 is the shared foundations;
  - 1 to 4 are the four chosen first, in the owner's order;
  - 5 to 16 are the other ideas, not chosen yet.
- `v` is a sub-version within it: 1, 2, and so on. For example, B-1-3 is the veto with the side-car's agents.
- **Runs** in `experiment.toml` keep the registry's IDs, `H15.r6` onward. Each run's label names its `B-<n>-<v>`.
- **The arms** of a trained experiment:

  | Arm | What it is |
  |---|---|
  | R | the released model |
  | R̃ | the release run through the same cached path, every adapter at zero |
  | C | the identical adaptation, without agents |
  | F | with the side-car's agents |
  | O | with the labelled agents |

**Old labels, and what they are now:**

| Was | Now | Note |
|---|---|---|
| C4, the analytic veto | **B-1** | chosen first |
| C2, agents into the scorer | **B-2** | chosen second |
| C1, agents into the refiner | **B-3** | chosen third |
| C3, "B-LoRA": both, with LoRA | **B-4** | chosen fourth |
| C5 to C16 | B-5 to B-16 | the same number |
| the 10 Nov plan's steps B1 (data) and B2 (headroom) | B-0-1, B-0-2 and B-0-3 | the foundations |
| that plan's B3 (the veto) | B-1 | |
| that plan's B4–B8 | each experiment's own pre-registration, runs, navtest read and close | |
| the first pass's B1, B2, B3, B3.2 (3–4 Nov) | **unchanged** | recorded runs `H15.b1`, `H15.b2`, `H15.b3`, `H15.b32`, always written with "the first pass" or the run ID. Their lineage: B-2 continues the first pass's B1; B-3 its B2 |

<a id="tracker"></a>

## The tracker

**Part B:**

| ID | The question | What changes | What trains | Status | Plan | Needs |
|---|---|---|---|---|---|---|
| B-0 | Can a planning gain be measured honestly here? | nothing: data, scoring, checks | nothing, unless cross-fitting is needed | **running** (D24) | [B-0, the foundations](part-b/B-0-foundations.md) | — |
| B-1 | Does a rule on the agents pick better than the release? | the pick, by a rule | nothing | built and pre-registered; waits for B-0 | [B-1, the veto](part-b/B-1-veto.md) | B-0 |
| B-2 | Does a scorer that reads the agents pick better? | the scorer | an adapter + LoRA: 0.28 M (C) to 0.42 M (F, O) | planned | [B-2, the scorer](part-b/B-2-scorer.md) | B-0; B-1 is its bar |
| B-3 | Do the proposals themselves get better? | the refiner | an adapter + LoRA: 0.38 M (C) to 0.51 M (F, O) | planned | [B-3, the refiner](part-b/B-3-refiner.md) | B-0 |
| B-4 | Both together: does it meet or beat the release? | refiner and scorer | both, in two stages | planned | [B-4, both with LoRA](part-b/B-4-joint-lora.md) | B-2, B-3 |
| B-5 to B-16 | the other ideas | — | — | not chosen | [the ideas](part-b/ideas.md) | by what B-1 to B-4 show |

- **What B-2, B-3 and B-4 adapt, and how:** [the LoRA primer](part-b/lora-primer.md).
- **Order of work:** B-0, then B-1, a review and the go-ahead to train, then B-2, B-3 and B-4, each closed by a
  review.

![Part B's order of work, with the owner's reviews](../../experiments/H15_perception_sidecar/figures/b-steps.png)

**Part A:**

| Step | Status | Run | Where |
|---|---|---|---|
| 1–3 · data, model and loss, pipeline check | done, reviewed (D21) | `H15.r1` | plan A §3–6, notebooks 01–03 |
| 4 · criteria committed | done | — | PREDICTIONS.md |
| 5 · three sizes, then v3 at three seeds | done | `H15.r3`, `H15.r4` | plan A §9.2, notebook 06 |
| 6 · report logs, read once | **waits for the owner** | — | plan A §8.1 |
| 7 · nuScenes, as recorded and warped | done | `H15.r2` (data), `H15.r5` | plan A §9.3, notebook 07 |
| 8 · close Part A | open | — | — |

<a id="log"></a>

## The discussion log

Newest last. Each entry is what was discussed or asked, what was decided, and what was done.

- **3 Nov · H15 registered; the first pass.**
  - Planned and run in a day and a half: the side-car (A), then its agents into the scorer (the first pass's B1),
    into the refiner (B2), and LoRA on the ViT (B3), plus nuScenes.
  - B1 and B2 came out neutral on navtrain's logs, where the release had trained: its pick scores NC and TTC
    0.996 there.
- **4 Nov · inconclusive; redone one step at a time.**
  - H14 is cancelled (D19).
  - The owner judges H15's first pass inconclusive: more ran than could be reviewed (D20). Part A is redone in
    eight reviewed steps, and phases B and C wait.
  - Steps 1–3 reviewed: "use recommended settings" (D21).
- **4–5 Nov · step 5.** v3 for 24 epochs, chosen by the committed rules: select F1 0.717.
- **9 Nov · the executive update** of Part A
  ([part-a/executive-update.md](part-a/executive-update.md)).
- **10 Nov · the next steps (D22).**
  - **The owner asked for:**
    - nuScenes validation in parallel;
    - a brainstorm of where the side-car could connect to Drive-JEPA;
    - as the first idea, the side-car as an extra input to the downstream planner with a LoRA fine-tune, meeting
      or beating NAVSIM, with an analysis of how the proposals change;
    - a plan for the other ideas.
  - **Asked and answered:**
    - Step 6 waits, and only step 7 runs.
    - Phase B is recorded inside H15.
    - A fixed recipe reads navtest once, and an R-clean selection split is built for later choices.
    - The pending Part A edits are committed on `-02` first.
    - Then: "documents first, like Part A; I shortlist; no training until the go-ahead". Step 7 runs now, since it
      is evaluation only.
  - **Done:**
    - the branch;
    - step 7 (`H15.r5`): Q9 holds; the warp does not help;
    - [the ideas](part-b/ideas.md) (then C1–C16) and the first B plan.
- **11 Nov · reprioritised and restructured (D23).**
  - **The owner chose four ideas, in order:** the veto, the scorer, the refiner, both with LoRA. They are now
    B-1 to B-4.
  - **The owner asked for:**
    - one plan document each, opening with the crux and what to expect, then the concepts as questions and
      answers that build to the plan;
    - a LoRA primer for these experiments;
    - diagrams;
    - one naming for B and C: `B-<n>-<v>`;
    - this overview as the common record;
    - the branch renamed `drivejepa/auxilliary-head-part-b`, committed and pushed.
  - **Not implemented.** The owner reviews the plans, and execution starts in another session.
  - **Counted for the primer:** the refiner's 13 linear layers and the scorer's 3. Rank-16 LoRA on them adds
    177,200 and 69,728 parameters.
- **11 Nov · one picture of the four experiments.**
  - At the owner's request, [B-0](part-b/B-0-foundations.md) opens with a single diagram of B-1 to B-4
    inside Drive-JEPA.
  - It shows each experiment beside the part of the planner it changes, with what it trains, the one question it
    tests, and what is read.
  - The figure is `b0-four-experiments`.

- **11 Nov · every side-car document in one folder.** At the owner's request, H15's documents move under
  `docs/drivejepa-perception-sidecar/`:
  - this overview is the folder's `README.md`, beside `first-pass-plan.md`, which spans both parts;
  - `part-a/` holds Part A's plan and its executive update;
  - `part-b/` holds B-0 to B-4, the ideas and the LoRA primer.
  - Every link and path that named the old files was rewritten, including the paths quoted in D20–D23.
- **11 Nov · execution opens: B-0 in full, then B-1 (D24).**
  - **Asked and answered** (the owner):
    - Scope: all of B-0, cross-fitting included, then B-1. B-2 to B-4 wait for the review after B-1.
    - B-1's four decisions and B-0's (a), (b) and (d)–(g): the recommendations.
    - B-0 (c): reuse the navtest metric cache Paddock built on 10 Nov, once verified here, rather than build one.
    - Cross-fitting, if the train rows read more than 0.03 better than select, trains inside B-0, before B-1.
    - A smoke run of B-1 on the select logs: allowed, flagged, choosing nothing.
    - "Keep documenting all the progress with good commit history": one commit per step, each with a line here.
  - **Found while planning, and corrected in [B-0](part-b/B-0-foundations.md):**
    - The vendored scorer's TTC loop is the official one, line for line (*read*). It is lane keeping that it checks
      every 0.5 s. B-0, `pdm_labels` and [the metric-cache page](../navsim-metric-cache.md) said TTC.
    - A navtest metric cache exists at this site after all: Paddock's, 12,146 tokens over 136 logs (*counted*).
      Paddock built it from another tree and interpreter, so H15 re-caches 32 tokens here before trusting it.
    - The release's navtest score per token is not on this site, so the 87.78 anchor is measured here for the
      first time.
    - The side-car names only vehicles, pedestrians and cyclists. A collision with a static object is room for
      B-1-1 but not for B-1-2 or B-1-3, and B-0-3 counts it apart.
  - **Pre-registered:** B-0's gates and its cross-fit rule, in PREDICTIONS.md, before any B-0 pass. B-1's section
    is written before B-1 reads R-clean.
- **11 Nov · B-0 runs; the anchor reproduces here.** Details, gate by gate, in
  [B-0's "What has run"](part-b/B-0-foundations.md#what-has-run-11-nov).
  - **Done:** the two lists (B-0's counts, scripted for the first time), labels, token and ViT passes, the
    side-car's agents for both splits; Paddock's navtest cache verified; navtest's PDM labels.
  - **The anchor:** the release's own picks, replayed through the official scorer here, give EPDMS (human filter)
    87.783 and CSV 84.495, against H04's 87.78 and 84.49. R̃, the cached path the trained arms start from,
    differs from R by +0.0014 ± 0.0011 points.
  - **Headroom (B-0-3):** of 275 navtest scenes whose pick fails NC or TTC, 271 have a safe proposal among the
    32; 156 of those name an agent in CAM_F0's view within 50 m, and the side-car found 127–135 of them.
  - **Cross-fitting runs:** the side-car reads 0.085 better in F1 on rows it trained on (limit 0.03), so five
    cross-fit side-cars train now (D24), before B-1.
  - **B-1 is built and pre-registered:** `veto.py`, a smoke run on select (chooses nothing), and its PREDICTIONS.md
    section, committed before any R-clean read.
  - **Waiting:** R-clean's metric cache, PDM labels and preflight; the live release (B0-G9); the cross-fit
    side-cars; B-0-4.
- **12 Nov · R-clean is unlike navtest; B-1 chooses on its known-command windows (D25).**
  - **Found** (B-0-1's check of R-clean before use): 77% of its windows carry the "unknown" driving command, which
    navtrain and navtest never have, and 55% have the ego stopped (navtest 7%). On those windows the release's
    input is out of distribution: its pick fails NC four to six times as often.
  - **Asked and answered:** which windows B-1's grid chooses on. The owner: only those whose command navtest uses,
    2,624 windows over 20 logs, still city-reweighted. PREDICTIONS.md's B-1 section is amended, dated, before any
    B-1 rule read R-clean.
  - **Done since:** R-clean's cache (B0-G4), labels and live preflight (B0-G5, B0-G6); the five cross-fit
    side-cars (select F1 0.700–0.708, within 0.02 of seed 0's: B0-G11); B-0-4 on R against R̃.
  - **The live release (B0-G9):** picks equal on every token and EPDMS within 6e-8 points of the replay, but the
    gate's "every column equal" holds on 52% of tokens, not 99.9%. Every discrete column is identical everywhere;
    only EP differs, by at most 1.5e-6, because the plans differ by at most 5e-5 m. Brought to the owner before
    B-1 reads navtest.

<a id="open"></a>

## Open decisions for the owner

| From | Decision |
|---|---|
| Part A | step 6, the report logs, read once; retrain the no-camera control in fp32; weighting what matters (11.9); head changes (11.11); test the warp's three changes apart (plan A §9.3) |
| B-0 | decided on 11 Nov (D24) |
| B-1 | decided on 11 Nov (D24) |
| B-2 | which sub-versions run (B-2-1 always; B-2-2 and B-2-3 or not); whether the scorer's sub-score heads train |
| B-3 | which sub-versions run; LoRA on the sampling offsets, or not |
| B-4 | two stages or joint; whether B-4-3 waits for B-2 and B-3 to differ |

Each experiment's own document gives these decisions in full, with a recommendation.

## The documents

| Document | What it is |
|---|---|
| [first-pass-plan.md](first-pass-plan.md) | the first pass's plan (3 Nov), kept as it was |
| [part-a/plan.md](part-a/plan.md) | Part A's plan, in reviewed steps, with what has run (§9) |
| [part-a/executive-update.md](part-a/executive-update.md) | Part A's executive update (9 Nov) |
| [part-b/ideas.md](part-b/ideas.md) | every idea for Part B, B-1 to B-16 |
| [part-b/B-0-foundations.md](part-b/B-0-foundations.md) | **B-0**: the foundations and the protocol every experiment shares |
| [part-b/B-1-veto.md](part-b/B-1-veto.md) | **B-1**, the analytic veto |
| [part-b/B-2-scorer.md](part-b/B-2-scorer.md) | **B-2**, the agents into the scorer |
| [part-b/B-3-refiner.md](part-b/B-3-refiner.md) | **B-3**, the agents into the refiner |
| [part-b/B-4-joint-lora.md](part-b/B-4-joint-lora.md) | **B-4**, both with LoRA |
| [part-b/lora-primer.md](part-b/lora-primer.md) | LoRA, for these experiments |
| [experiments/H15_perception_sidecar/](../../experiments/H15_perception_sidecar/) | the record: README, PREDICTIONS.md, experiment.toml, the notebooks, the figures |
| [experiments/decisions.toml](../../experiments/decisions.toml) | the decisions, D19–D25 for this work |

## How this set is kept

- **Every side-car document lives in this folder:** cross-part ones at its top, the rest in `part-a/` or `part-b/`.
  Figures stay with the experiment's record, in `experiments/H15_perception_sidecar/figures/`.
- **Each document opens with its status and date.** Experiment plans open with the crux and what to expect.
- **Any change to a document adds a dated line to the log here,** and updates the tracker.
- **A decision gets a D-number** in `experiments/decisions.toml`, and is mirrored in the log.
- **A run is recorded in `experiment.toml`,** and the README and CARD are rendered, never typed.
- **Numbers are copied from records, never re-derived,** and each carries its label: *counted*, *measured*, *read*
  or *argued*.
