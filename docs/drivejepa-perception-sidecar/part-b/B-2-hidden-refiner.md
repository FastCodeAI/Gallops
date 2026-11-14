# H15 Part B · B-2: the side-car's hidden layer into the refiner, with the planner fine-tuned downstream

**Status (14 Nov 2026): run, and stopped at its phase-1 gate.** Re-planned on 13 Nov (D28) and approved the same
day with every recommendation (D29). B-2-0 (`H15.r11`) and phase 1 (`H15.r12`) ran. Labelled agents added nothing to
the fine-tuned planner on navtest (K0), so by the pre-registered rule phase 2 did not run.
[What has run](#results) has the numbers and figures.
- [The overview](../README.md) holds the tracker, the naming and the log.
- This plan stands on two documents:
  - [B-0, the foundations](B-0-foundations.md);
  - B-1's result, in [B-1, the veto](B-1-veto.md) and [B-1, explained](B-1-explained.md).
- It replaces three of the 11 Nov plans and one idea:
  - B-2, the agents into the scorer alone, which becomes [B-3](B-3-scorer.md), not scheduled;
  - B-3, the agents into the refiner;
  - B-4, both with LoRA;
  - B-5, agent-aware initial proposals.

  What still held in them is here; the rest is in git history. See [What changed on 13 Nov](#changed).
- How LoRA adapts the refiner and the scorer: [the LoRA primer](lora-primer.md).
- **Numbers follow the 13 Nov renumbering (D28).** B-3 is the scorer alone, and B-4 to B-14 are the other ideas.
  The overview's [naming table](../README.md#naming) translates records written before it.
- **The figures:**
  - Every number in a chart is drawn by `experiments/H15_perception_sidecar/scripts/part_b2_plan_figures.py` from
    the passes' own outputs, and written to
    [part-b2-plan-numbers.json](../../../experiments/H15_perception_sidecar/figures/part-b2-plan-numbers.json).
  - The diagrams are Graphviz sources beside their pictures.
- **Labels on numbers:** *counted* on disk · *measured* by a run · *read* in code or a record · *argued*. The
  counts marked "13 Nov" are new, made for this plan; every other number is copied from a record.

## The crux

**B-1 gave the planner the side-car's agents as a decoded list, through a rule, and lost.**
- On navtest it lost 0.21 points with labelled agents, and 0.15 to 0.25 with the side-car's.
- The ceiling, which reads the grader's labels, gained +1.58.

**The owner's question for B-2:** is the decoded list the bottleneck? That list is class, confidence, box,
velocity and 1 s path, cut at confidence 0.5.
- The module that makes the proposals, the refiner, is given the side-car's **last hidden layer** instead.
- The planner downstream is **fine-tuned**.
- Does it then plan better on navtest:
  - than the same fine-tune without the side-car (C);
  - than the same fine-tune with the decoded list (D);
  - than the release (R)?
- And how do the proposals and the pick change?

## What to expect

- **Small, if anything** (*argued*). The draft predictions:
  - HD − D, the primary contrast: −0.1 to +0.2 points;
  - HD − C: −0.1 to +0.3;
  - HD against R: −0.4 to +0.3, so "meets" is more likely than "beats".

  Here HD is the decoded list with each agent's hidden slot appended.
- **The strongest reasons against are measured or read, not argued:**
  - B-1's pre-registered rule fired. Explicit agents near the ego added nothing, even perfect ones.
  - The release already learned what NC and TTC need, more directly than the side-car did. Each proposal's feature
    was trained to predict that proposal's collision and TTC agents, every 0.1 s over 4 s, from all logged agents.
  - The room is in the pick, not in the proposals: 271 of 275 failing navtest scenes already have a safe proposal.
- **The strongest reasons for are counted:**
  - Decoding keeps 4.75 of the side-car's 128 slots a row. Another 31.6 hold evidence at confidence 0.1 to 0.5,
    which a learned reader could weigh and a list cannot.
  - The side-car reads the ViT's 1024-d tokens, before the planner's 256-d projector.
- **What it is worth either way:**
  - If HD − D holds, the decoded list was a bottleneck. The scorer alone (B-3) and the closed-loop gate (B-12) then
    follow with slots.
  - If it does not, explicit agents add little on NAVSIM, decoded or hidden, to a planner already supervised on
    per-proposal collision agents. The remaining room (the ceiling's +1.58) belongs to the pick's horizon: B-6
    (4 s forecasts) or closed loop (B-12).

<a id="pictures"></a>

## B-2 in pictures

![B-2 in one picture: where the side-car's hidden layer enters Drive-JEPA, what each arm hands over, what each recipe trains](../../../experiments/H15_perception_sidecar/figures/b2-overview.png)

**How to read it.**
- **Down the left is Drive-JEPA as it runs.**
  - Grey is frozen.
  - The refiner and the scorer are green: B-2 adapts them by LoRA or trains their weights, by recipe.
- **Down the right is the side-car**, frozen.
  - Its hidden layer: 128 slots of 384 numbers, each with a centre and a confidence.
  - Its decoded list, which the heads make from those slots.
- **Between them sit the arms**, each handing the adapter different tokens, and the adapter itself.
  - The adapter adds δ to the refiner's queries before every BEVFormer layer, on all four passes.
  - Its output layer starts at zero, so at step 0 every arm is the release's cached path, R̃, exactly.

![B-2's order of work: passes, a gate, the grid, one navtest read per phase, and the owner's reviews](../../../experiments/H15_perception_sidecar/figures/b2-phases.png)

**How to read it.**
- **B-2-0** builds and checks the inputs. Nothing trains there.
- **Phase 1** trains only C and O.
  - Their seed spread gives the noise floor.
  - Their navtest read says whether perfect agents leave any room for a learned planner.
- **The owner decides whether to go on.**
- **Phase 2** trains the other arms and recipes, fixed in advance, and reads navtest once.

## The concepts, question by question

**Q1. What did B-1 leave, and why change the input rather than stop?**
- **What B-1 showed** (*measured*, `H15.r9`, `H15.r10`):
  - A rule on explicit agents at the pick lost on navtest:
    - labelled agents, −0.21 ± 0.05;
    - the side-car's agents, −0.25, −0.19 and −0.25 by seed;
    - softened into a penalty, −0.16, −0.15 and −0.16.
  - The ceiling gained +1.58 ± 0.19.
  - B-1-2's 1 s clearance caught 1,949 of 40,595 failing proposals. The failures come later than 1 s, or from
    agents out of view: 94 of the 271 scenes with room (35%; B-0-3).
- **The pre-registered rule** (PREDICTIONS.md, Part B · B-1) says that with V2 not supported, a learned use of the
  agents is unlikely to show a NAVSIM gain, and the closed-loop gate becomes the place.
  - Going on is the owner's decision (D28), made knowing that rule.
  - What B-1 did not test is a *learned* use of a *richer* input. B-2 tests exactly that, and nothing else.
- **The room on navtest is real, and mostly in the pick** (*counted*, 13 Nov, and B-0-3's record):

![Where the room is: the release rarely fails where it trained; on navtest the better plan is usually already among its 32](../../../experiments/H15_perception_sidecar/figures/b2x-room.png)

- **How to read it:**
  - On navtrain's train rows the release's pick fails NC or TTC 0.81% of the time; on navtest, 2.31% (vendored
    labels, the same scorer for both).
  - Best of 32 beats the pick by 3.75 points in-sample and 7.41 on navtest.
  - On the right is B-0-3's navtest funnel:
    - 275 failing scenes;
    - 271 of them with a safe proposal among the 32;
    - 156 with the agent behind the failure in CAM_F0's view within 50 m;
    - 127, 133 and 135 of those found by the side-car's seeds.

**Q2. What exactly is the side-car's last hidden layer?**
- **The model.** The side-car is v3 (*read*, `train_sidecar.VARIANTS` and its runs' `run.json`):
  - six decoder layers, width 384, 12 heads, 128 slots, 18.44 M parameters;
  - it reads the ViT's output, 512 tokens of 1024, before the projector the planner uses.
- **The hidden layer.** After each layer ℓ every slot holds a vector aℓ of 384 numbers. The class, box, velocity
  and path heads read it. B-2's "last hidden layer" is a₆: 128 × 384 per frame.
- **A slot's centre is not in aℓ** (*read*, `sidecar.py` 136–158; the picture below):
  - The centre is a reference point carried beside aℓ: refℓ = refℓ₋₁ + ref_mlp(aℓ), detached before the next layer.
  - The position code reaches only the attention's queries and keys. The values are aℓ itself and the image tokens.
  - The ViT's own position code is RoPE (*read*, `vjepa2/configs/eval/vitl/in1k.yaml:161`), applied inside
    attention, so the tokens carry none added.
- **So aℓ holds what an agent is, and how to move its reference point, but not where it is:**
  - its class, size, heading and motion, and its 1 s path;
  - its slot's identity.
- **Every hidden arm therefore carries the slot's centre**, refℓ, beside aℓ. Without it the refiner could not place
  an agent at all.
- **Why layer 3 too:**
  - Part A found refinement done by layer 3: v3 at 12 epochs reads the same F1 at layer 3 as at layer 6
    (*measured*, notebook 06).
  - But every layer has its own heads and the same loss (*read*, `SideCarLoss`), so layer 3 is no less specialised
    to detection. H3 ≈ H6 is the likely result (*argued*).

![What the side-car's hidden layer holds: one decoder layer, the content stream against the reference point](../../../experiments/H15_perception_sidecar/figures/b2-slot.png)

**Q3. What does decoding throw away?**
- **The cut.** The decoded list keeps a slot only at confidence ≥ 0.5, and then only its argmax class, a point box,
  a velocity and a 1 s path. No uncertainty is kept.
- **How much it drops** (*counted*, 13 Nov):
  - On the train rows (the cross-fit side-cars) the list keeps 4.75 slots a row.
  - 12.7 slots clear 0.3, and 36.4 clear 0.1.
  - On navtest (seed 0) the same counts are 3.33, 9.70 and 33.4.
- **How good that evidence is:**
  - Part A measured the side-car over-confident below 0.7: a slot at 0.55 is right about 40% of the time.
  - So most of what the list drops is noise, and some of it is the agent that mattered.

![What decoding discards: of the side-car's 128 slots a row, the decoded list keeps under five](../../../experiments/H15_perception_sidecar/figures/b2x-slots.png)

- **Two navtest scenes, each the extreme of a rule** (the rules and their numbers are in the JSON; side-car seed 0,
  as B-1 read it). The rules are written in the script and printed with their numbers.
  - **What decoding drops, where it mattered** (navtest `baf0c6e99bec57db`):
    - The release's pick fails on a parked car in view, which seed 0 has no slot on at 0.5. Arm D cannot see it.
    - A vehicle slot at 0.18 lies 0.26 m from its centre. Only the hidden arms receive it.
  - **What a hidden arm must learn to discount** (`abe4fa26de85552d`):
    - The pick is safe.
    - Ten slots at 0.11–0.25 lie on its plan with no labelled centre within 2 m. Most sit around the car ahead at
      other depths: Part A's commonest false alarm.

![The same scene, two inputs: the decoded list (arm D) and the slots beneath it](../../../experiments/H15_perception_sidecar/figures/b2x-scene.jpg)

<a id="why"></a>

**Q4. Why might it work, and why might it not?**

![Why B-2 might, or might not, work: each argument, its evidence, and what tests it](../../../experiments/H15_perception_sidecar/figures/b2-arguments.png)

**For:**
1. **Decoding discards evidence** (Q3). A learned reader can weigh 36 slots a row; a list holds under five.
2. **The side-car reads what the projector may drop.** It reads 1024-d tokens, the planner a 256-d projection of
   them (*read*).
3. **A learned reader is not bound to a 1 s clearance.** B-1's caught 1,949 of 40,595 failing proposals
   (*measured*).
4. **Planners elsewhere read agent queries, not boxes:** UniAD, VAD and SparseDrive take content plus position
   (*argued*). They train perception and planning together, on far more data.
5. **Retraining the refiner helped out of domain.** In the first pass every retrained refiner brought the release's
   pick on nuScenes 0.28–0.36 m closer to the human at 3 s, with or without agents (*measured*, in-sample; now
   B-14). Arm C re-measures it.

**Against:**
1. **B-1's pre-registered rule already fired** (Q1). Going on is D28.
2. **The release already learned this, more directly** (*read*, `drive_jepa_config.py`, `compute_navsim_score.py`
   95–124).
   - It trained with `agent_pred`: weight 1 on the presence and 0.1 on the corners.
   - Each proposal's pooled feature predicts that proposal's at-fault collision agent and its TTC agent, every
     0.1 s over 4 s, from PDM's simulation of all logged agents, out of view included.
   - The side-car's supervision is per agent, 1 s, front camera. The H15 documents had mentioned these heads only in
     passing.
3. **The room is in the pick, not the proposals.**
   - 271 of 275 failing scenes have a safe proposal (*measured*).
   - The refiner's own loss has no NC or TTC term (Q6).
   - The first pass's agent-fed refiner moved best of 32 by +0.004, −0.013 and +0.001 points, even with labelled
     agents (*measured*, in-sample).
4. **The planner is in-sample on navtrain.** Its pick fails NC or TTC on 0.81% of train rows. The adapter is taught
   where the release needs no help, and may learn to ignore its input. Cross-fitting cures the side-car's optimism,
   not the planner's (*argued*).
5. **Slots hold corrections, not positions** (Q2). The forecast that arm D gets analytically, H6 must learn.
6. **B-1's failures are about horizon and view** (*measured*): after 1 s, and 35% out of view. The slots see the
   same camera and were trained to 1 s.
7. **Seed noise.** The first pass's F − C ran from −0.21 to +0.24 across seeds (*measured*), so three seeds
   against three resolve about 0.3–0.4 points (*argued*), above the plausible effect.
   - There is also more to overfit: a row's hidden layer is 49,152 numbers, against about 2,300 in D's list.
   - And EC: B-1 lost half its points to picks that flicker between frames.

**Q5. Where does the hidden layer enter the planner, and what flows where?**
- **Through the 11 Nov B-3-1 adapter** (*read*, `train_refiner.RefinerAdapter`). It is a pre-hook on the shared
  `Bev_refiner` that adds δ to the 256 queries before each pass's BEVFormer layer.
- **Its own half** reads each query with its pose. It runs in every arm, C included.
- **Its side-car half** gives each query attention over the arm's tokens, with a learned "no agent" slot to attend
  to.
- **Its output layer starts at zero**, so the untrained model is R̃ bit for bit. The preflight checks it.
- **What each arm hands the adapter:**

| Arm | Tokens | Attention bias | Slots |
|---|---|---|---|
| C | none: the own half only | — | — |
| O | labelled agents in CAM_F0's view within 50 m, in D's format (confidence 1) | clearance, as D | the labelled ones |
| D | the decoded list: 18 numbers each (class, confidence, centre, size, yaw, velocity, path) → MLP | −clearance / 5 from each pose to each agent at its time, + 13 clearance features per proposal (`geometry.terms`) | confidence ≥ 0.5 (4.75 a row) |
| HD | D's tokens + LayerNorm(a₆) → 128, per kept agent | D's | D's |
| H6 | LayerNorm(a₆) → 128, + a code of ref₆ | log(confidence); no clearance | all 128 |
| H3 | as H6, from a₃ and ref₃ | as H6 | all 128 |

![B-2's inputs, outputs and every tensor between them, for one sample](../../../experiments/H15_perception_sidecar/figures/b2-tensors.png)

- **How to read it.** It traces one sample from CAM_F0 to the plan.
  - Left, the planner: ViT-out 512 × 1024 → map 16 × 32 × 256 → queries 256 × 256 → per pass, proposals
    32 × 8 × 3 and δ 256 × 256 → pooled 32 × 256 → logits 32 × 6 → the plan, 8 × 3.
  - Right, the side-car: 128 slots × 384 per layer, its heads, its decoded list.
  - The table between them is what each arm hands over.

**Q6. Which loss trains which route? This is why the recipes differ.**
- **The order inside a pass** (*read*, `traj_refiner.py`): each pass first decodes its proposals from the incoming
  queries, then runs its BEVFormer layer on the queries plus δ. So:
  - p₁ is decoded before any δ, so pass 1 is agent-blind;
  - δ₁, δ₂ and δ₃ shape p₂, p₃ and p₄;
  - δ₄ changes only q₄, which only the scorer reads.
- **The planning loss** (*read*, `drivejepa_b2d.train_generator.mtd_loss`):
  - winner-takes-all L1 over x, y and heading: only the proposal nearest the human is pulled to it;
  - plus 0.5 × the same toward up to 4 good anchors: proposals from the release's vocabulary of 8,192 that NAVSIM
    scored ≥ 0.95 on that scene;
  - each earlier pass is discounted × 0.1 per pass (p₃ 0.1, p₂ 0.01, p₁ 0.001). The 11 Nov plans said "each
    earlier pass at 0.1"; the code discounts geometrically.
  - It has no NC or TTC term, and it never reaches δ₄.
- **The score loss** is BCE of each proposal's final score against its PDM label.
  - The release trained it jointly, with its refiner.
  - It reaches q₄, and so δ₄. Through the shared weights it reaches every pass.
- **So, by recipe** (Q9): in B-2-1 the score loss trains the read's route into the pick. In the two-stage recipes,
  stage 3 must train a pass-4 copy of the read. Otherwise a two-stage recipe tests imitation, not choosing.

![Where the side-car enters the refiner, pass by pass, and which loss trains each write](../../../experiments/H15_perception_sidecar/figures/b2-passes.png)

**Q7. What does each arm isolate?**

| Contrast | It isolates | Kind |
|---|---|---|
| **O − C** | room for any agent input to a learned planner: perfect agents, in D's form | the gate (phase 1) |
| **HD − D** | the hidden content alone: everything else is D's | **the primary** |
| HD − C | the side-car, as HD hands it over | claim, if the primary holds |
| HD against R | meets or beats the release | claim, if the primary holds |
| H6 − HD | a learned geometry in place of the analytic one, and all 128 slots in place of the list | claim, if the above hold |
| H3 − H6 | an earlier layer | claim only at three seeds |
| D − C | the decoded list, learned rather than ruled | reported |
| C − R | the fine-tune itself, out of sample | reported |

- **No trained S arm** (another log's slots).
  - A trained S mostly re-measures C. Swapping the input at test costs nothing and says more.
  - **At test, without retraining, HD and H6 are also read with:**
    - no slots;
    - another log's slots;
    - another fold's side-car;
    - labelled agents in place of D's list.
- **There is no hidden-layer O.** Labelled agents have no slots, so perception's ceiling exists only for the
  decoded form.

**Q8. Why must the side-cars be fold-matched, at test too?**
- **Decoded agents are side-car-agnostic; hidden states are not.**
  - Decoded agents are in metres and classes, the same whichever side-car made them. The 11 Nov plans trained on the
    cross-fit side-cars' agents and read the full seed-k side-cars at test.
  - Hidden states are each model's own coordinates. A reader trained on the cross-fit side-cars' slots cannot read
    another model's.
- **The rule:**
  - Each train row reads the cross-fit side-car that never saw its log, as `agents-train-crossfit-v3` does
    (94,195 rows, five folds of the 1,071 train logs; *counted*, `H15.r8`).
  - Select, R-clean and navtest logs are each assigned to one fold's side-car by a fixed seeded rule, written once.
  - Every arm, D included, reads that assignment.
- **B-2-0 checks alignment first.** On select rows, every side-car's slots are fitted to side-car 0's:
  - matched by centre (Hungarian), and by slot index, since all five started from the same seed-0 slots;
  - with a ridge or orthogonal Procrustes map, its R² read on R-clean.
  - If the five align, one shared input map is learned on all rows. If not, one per fold.
- **Departures from B-0's Q8, recorded in D28:**
  - The side-car's seed is no longer sampled. All five folds are seed 0, select F1 0.7005–0.7077, against the full
    seeds' 0.7132–0.7195.
  - Arm D is therefore not B-1-3's input.

![Fold-matched cross-fitting: every row's slots come from a side-car that never saw its log](../../../experiments/H15_perception_sidecar/figures/b2-crossfit.png)

**Q9. Which fine-tuning recipes, and why compare all three?** The owner asked for three. They differ in how the
score loss reaches the read, and in how much of the planner moves.

![B-2's three fine-tuning recipes, side by side](../../../experiments/H15_perception_sidecar/figures/b2-recipes.png)

- **B-2-1 · joint fine-tuning (primary).** The release's own objective, resumed (*read*, `drive_jepa_config.py`):
  - **Losses and weights:**
    - planning loss × 1;
    - final-score BCE × 1, against PDM labels scored online by the vendored scorer on CPU workers;
    - the collision- and TTC-agent heads × 1 (presence) and × 0.1 (corners);
    - drivable area × 2;
    - sub-scores at 0, as released.
  - **What trains:**
    - the adapter;
    - rank-16 LoRA on the refiner (177,200) and on the scorer (69,728; *counted*, the LoRA primer).
  - The auxiliary heads stay frozen; their losses only shape the features (decision (f)).
  - 3 epochs, about 4,400 steps of 64.
  - About 1–1.5 h a run (*estimate*). The vendored labels took 596 s for 103,288 rows (*measured*).
- **B-2-2 · two stages with LoRA** (was B-4-1):
  1. The refiner: the adapter + refiner LoRA, the planning loss only.
  2. Dump every row's new proposals and label them once with the vendored scorer: train, select, R-clean, navtest.
  3. The pick: a pass-4-only copy of the read, with the scorer's LoRA and own half, on BCE.
     - Passes 1–3 are frozen, so p₄ and its labels stay as dumped.
     - The logits are re-based on stage 2's, so stage 3 starts as "the release's scorer on these proposals", exactly.
- **B-2-3 · two stages, full.**
  - As B-2-2, but the refiner (2.32 M, with the initial queries and ego encoder) and the scorer MLP (1.32 M) train
    their weights at 2e-5.
  - It is the first pass's "full" mode, which brought the largest nuScenes gain and has the most room to drift.
- **Comparisons across recipes are descriptive.** The recipes differ in steps, parameters, label source and
  auxiliaries, so a difference between them has many causes. The claims are made within a recipe.

**Q10. How is it read?**

![B-2's metrics: what is measured, by which scorer, and how a result is read](../../../experiments/H15_perception_sidecar/figures/b2-metrics.png)

- **The headline** is navtest's EPDMS with the human filter, by NAVSIM's official scorer, through the replay agent.
  - It is B-0-2's pipeline, which reproduces the release's 87.783.
  - Paired per scene, with SE clustered over 136 logs, per seed.
  - Momentum off, as the anchor; on, as a second read.
- **The explanation** comes from the vendored labels of every proposal: best of 32, the share passing NC and TTC,
  and the release scorer's pick on the new proposals.
- **The proposal analysis (B-0-4)** reads clearance, spread, moved proposals, pick switches and steadiness, and the
  subsets.
- **nuScenes** reads the pick's L2 at 1, 2 and 3 s, collisions within 3 s and minADE, with the fold side-cars run
  on nuScenes' tokens.
- **The decision chain is read in order.** A step is a claim only if every step above it holds:
  1. the gate;
  2. the primary;
  3. HD − C;
  4. HD against R;
  5. H6 − HD;
  6. H3 − H6.

  Everything else is reported, never claimed.

**Q11. What can go wrong?**
- **The release needs no help where it trains** (Q4, against 4). The adapter may learn to ignore its input. The
  test-time swaps (Q7) show whether it uses it.
- **Seed noise hides a real effect** (Q4, against 7). Phase 1 measures C's spread before any H arm trains, and
  decision (e) adds seeds if it is too wide.
- **The fine-tune costs out of sample.**
  - H07's exp3 fine-tuned a generator and scorer and fell at the gate from 0.857 to 0.753 (D18).
  - The first pass's refitted scorer cost 0.23–0.44 m on nuScenes.
  - C − R, nuScenes and R̃ make it visible.
- **The 32 lose their spread** (11 Nov B-3's R5). This is measured.
- **EC.** New proposals and a refit scorer change picks between frames. It is measured, and flagged if it falls.
- **The cross-fit ties the planner to five side-cars.** Shipping it means shipping those, or one fold's.
- **Out of domain the slots shift more than boxes.** On nuScenes the side-car's F1 is 0.50 (`H15.r5`).
- **The shared GPU.** The grid is days of wall time, not hours (the cost, below).

## The plan

### Sub-versions: the recipes

| ID | Recipe | Arms × seeds | Recommended |
|---|---|---|---|
| **B-2-1** | joint fine-tuning, the release's own objective, online PDM labels | C, O (phase 1); D, HD, H6, H3 (phase 2a) × 0–2 | yes: primary |
| **B-2-2** | two stages, LoRA; stage 3 trains a pass-4 copy of the read with the scorer | C, D, HD × 0–2 (phase 2b) | yes |
| **B-2-3** | two stages, full fine-tune | C, D, HD × 0–2 (phase 2b) | yes |

![The run grid: which arm runs under which recipe, in which phase](../../../experiments/H15_perception_sidecar/figures/b2x-grid.png)

The staged grid is 34 runs, against 54 for the full factorial (decision (g)):

| Phase | Runs | Estimated hours |
|---|---|---|
| 1 | 6 | 6–9 |
| 2a | 10, with H3 as a seed-0 pilot | 10–15 |
| 2b | 18 | 14–22 |

All estimates are wall time on one shared GPU (*estimate*).

### Steps

1. **After the owner's shortlist and go-ahead, B-2-0, no training:**
   - The hidden-state pass. Each fold's side-car on its own fold's train rows and on its assigned select, R-clean and
     navtest logs. It writes layers 3 and 6, the slots' centres and confidences, in fp16: about 98 KB a row a layer,
     ~25 GB in all (*estimate*).
     - **Its gate:** the stored slots, decoded by the same model's heads, equal its stored agents, and give each
       fold's recorded select F1 (B0-G7, B0-G11).
   - The fold-assigned decoded passes, for D on select, R-clean and navtest.
   - The fold assignment, committed: `recipes/drivejepa_perception/`, re-derived and refused if edited.
   - The alignment check (Q8).
   - No report-log row anywhere.
2. **Pre-register** in PREDICTIONS.md (Part B · B-2), as approved, before the first training step:
   - the chain;
   - every prediction;
   - the gate's rule.
3. **Preflight** (extended):
   - the untrained composed model equals R̃ bit for bit on 64 rows;
   - the slots recomputed live equal the stored ones;
   - the fold rule;
   - the teachers equal their files;
   - the PDM labels and the auxiliaries' targets re-score exactly;
   - no report-log row.
4. **Phase 1:** C and O under B-2-1, seeds 0–2.
   - C's seed spread on R-clean's known-command windows (D25) and on select.
   - Navtest once, for C and O. Then the owner's review.
5. **Phase 2:**
   - D, HD and H6 under B-2-1 (H3 per decision (c));
   - then B-2-2 and B-2-3 on C, D and HD;
   - the last epoch kept: nothing chosen.
6. **Dump and label** every run's proposals (vendored): train, select, R-clean, navtest.
7. **Health read on select**, which chooses nothing. The owner sees the curves: losses, gradient norms, the read's
   attention to its "no agent" slot.
8. **Navtest, once,** after every phase-2 run: R̃, C, D, HD, H6 (H3), every seed and recipe.
9. **Analysis:** B-0-4 on every arm; the test-time swaps (Q7); BEV drawings of the scenes whose pick changed.
10. **nuScenes** with the fold side-cars.
11. **Record** `H15.r<n>`:
    - labels "B-2-1", "B-2-2", "B-2-3";
    - every checkpoint pinned by its weights' sha256;
    - a line in the overview's log.

<a id="criteria"></a>

### Success criteria (a draft, for PREDICTIONS.md once approved)

On navtest, by the official scorer, paired per scene, SE clustered by log, every seed. Under B-2-1 unless named. The
chain is read in order (Q10).

| # | What | Predicted (*argued*) | Supports if | Refutes if |
|---|---|---|---|---|
| K0 | O − C, the gate | 0.0 to +0.4 points | above 0 beyond 2 SE, every seed | at or below 0, every seed: no room (decision (b)) |
| **K1** | **HD − D, the primary** | −0.1 to +0.2 | beyond D's seed range and 2 SE, every seed | below 0 beyond 2 SE, any seed |
| K2 | HD − C | −0.1 to +0.3 | beyond C's seed range and 2 SE, every seed | below 0 beyond 2 SE, any seed |
| K3 | HD against R | −0.4 to +0.3 | meets: the interval's low end ≥ −0.4, every seed; beats: above 0, every seed | high end below 0, any seed |
| K4 | H6 − HD | −0.3 to +0.1 | beyond HD's seed range and 2 SE, every seed | — |
| K5 | H3 − H6 | −0.1 to +0.1 | as K4, only at three seeds | — |

Reported, never claimed:

| What | Predicted (*argued*) | Flag if |
|---|---|---|
| D − C | −0.1 to +0.3 | — |
| C − R, the fine-tune alone | −0.5 to +0.3 | — |
| best of 32 (vendored), HD − C | 0 to +0.3 points | — |
| the share of the 32 passing NC and TTC, HD − C | 0 to +1.5 percentage points | — |
| EC, HD − R | within ± 0.02 | lower by more than 0.02 |
| the 32's lateral spread at 4 s, HD against R̃ | within ± 10% | below by more than 20%: collapse |
| HD under B-2-2 and B-2-3, against B-2-1 | −0.3 to +0.2 and −0.4 to +0.2 | — |
| nuScenes L2 at 3 s, HD − R | −0.3 to +0.3 m | — |
| another log's slots change HD's pick | on ≥ 5% of navtest scenes | under 1%: the input is not used |

![B-2's draft predictions, in the decision chain's order, beside what B-1 measured](../../../experiments/H15_perception_sidecar/figures/b2x-predictions.png)

### Later variants, each by a stated trigger

| Variant | Was | Trigger |
|---|---|---|
| the scorer reads the slots directly (the cheap route into the pick) | B-4's "both" | best of 32 rises but the pick does not |
| pass 1 sees the slots: the read on the initial queries too | B-3-3 and B-5 | HD's proposals move little in pass 2 |
| LoRA kept off the deformable sampling offsets and attention weights | B-3-4 | where the refiner looks drifts |
| adapter only, no LoRA | B-3's `frozen` mode | if HD wins: how little suffices |
| sub-score heads trained beside the final score | B-2's decision (b) | if HD wins: the reasons made explicit |
| a KL anchor to the release's scores | new | EC falls by more than 0.02 |
| a planner-side slot layer (self-attention among the 128, ego-conditioned) | new | H6 trails HD: learned geometry needs more room |
| a training-only clearance term on the 32 (labelled futures; never an input) | new | the refiner ignores its input (the swaps) |
| D over all 128 decoded slots (the threshold factor alone) | new | H6 differs from HD |
| LoRA into the side-car's last layer, its detection loss kept | new | delicate under cross-fit: a fold trained on its held-out rows' planning labels becomes in-sample there |

### Decisions for the owner

| | Decision | Recommended |
|---|---|---|
| (a) | the primary contrast | HD − D: it changes one factor, the owner's question |
| (b) | the phase-1 gate | the owner decides after the read; stop if O − C is at or below 0 on every seed |
| (c) | H3 | a seed-0 pilot: likely ≈ H6 (Q2) |
| (d) | navtest's side-car | one fold per log, as trained; or each scene's metrics averaged over five readings, at 5× the scoring CPU |
| (e) | more seeds | five for C, D and HD under B-2-1 if C's seed range on R-clean exceeds 0.3 points |
| (f) | B-2-1's auxiliary heads | frozen, their losses kept: the release's way of making its features collision-aware |
| (g) | the grid | staged (34 runs) over the full factorial (54) |
| (h) | the later variants | none in the first grid; each by its trigger |
| (i) | nuScenes | yes: where the first pass found both a cost and a gain |
| (j) | momentum | off, as the anchor; on, as a second read (B-0's (d)) |

### Risks

1. **No room for a learned planner** (K0). The gate says so before the H arms are spent.
2. **Power.** A real +0.2 may stay unresolved at three seeds. Phase 1 measures the floor, and decision (e) answers it.
3. **The cost of adapting**, out of sample (C − R, nuScenes). It is reported beside every gain. Nothing ships that
   loses to R.
4. **The report logs.** They are never read: the preflight's guard. The existing `train_refiner` and `train_scorer`
   read them and keep a chosen epoch, so they are not reused as they are.

### Cost (*estimate*)

- **B-2-0:** about one GPU-hour and ~25 GB.
- **The grid:** 34 runs, about 30–46 h of wall time on the shared GPU.
- **Official reads:** ~12 CPU-hours.
- Days, not hours. The figures are per run, from the first pass's 0.20 s a batch (*measured*) and the vendored
  labels' 596 s per pass (*measured*).

### Code, when approved

None of this exists yet. Written only after the go-ahead, each piece with its check:

| Piece | What it does | Its check |
|---|---|---|
| `sidecar.SideCar.forward` | returns each layer's aℓ beside its outputs | outputs unchanged, tested |
| `train_sidecar hidden` | the hidden-state pass, by fold, on any source | its gate (step 1); report rows refused |
| the fold assignment | B-2's rule, written once | re-derived and refused if edited |
| the alignment check | the five folds against fold 0, R² on R-clean | — |
| a slot reader beside `scorers.AgentRead` | for HD, H6 and H3: LayerNorm + projection, a centre code, a log-confidence bias, the null slot | its zero-init identity |
| B-2's trainer, on Part B's splits only | joint (online vendored labels and the auxiliaries' targets, on CPU workers) and two stages (the pass-4 copy) | p₄ unchanged by stage 3; the last epoch kept |
| the preflight | step 3 | — |

Recipe stages are `part-b-2-*`.

<a id="results"></a>

## What has run (13–14 Nov)

Records: `H15.r11` (B-2-0) and `H15.r12` (phase 1). The gates, predictions and rules are PREDICTIONS.md's (Part B ·
B-2), committed before any run (032d0c0). The charts are drawn by
`experiments/H15_perception_sidecar/scripts/part_b2_results_figures.py` from the runs' records, and every number in
them is in [part-b2-results-numbers.json](../../../experiments/H15_perception_sidecar/figures/part-b2-results-numbers.json).

**In one line.** Labelled agents, given to a refiner fine-tuned with the release's own objective, did no better than
the same fine-tune without them: O − C was −0.06 to −0.08 points on every seed. The gate stopped B-2 there, so the
side-car's hidden layer was never tried. The fine-tune itself gained about 0.4 points on navtest, but lost
elsewhere (R-clean, nuScenes).

### B-2-0, the inputs (`H15.r11`; *measured*)

- **B2-G1 holds.**
  - Each of the five fold side-cars, decoded by the pass's own code, gives exactly the select F1 it recorded.
  - On all 94,195 train rows the decoded list equals B-0's cross-fit agents in every value.
- **B2-G4: the five side-cars' hidden spaces do not align.**
  - Matched slot for slot, R² is −0.09 to −0.19.
  - A fitted linear map reaches only 0.47 to 0.55.
  - By the rule, each fold gets its own input map.
- **Added on 14 Nov:**
  - the nuScenes keyframes, decoded by the fold side-cars (each scene dealt a fold by the same rule; the gate holds);
  - a navtest pass read by the "wrong" fold, for the test-time swaps.

### The code, and what its checks caught (*measured*)

- **The trainer:** `train_b2.py`, the joint and the two-stage recipes, every arm. The preflight (B2-G3) runs before
  each run's first step.
- **Smoke-tested before phase 1:** every recipe and arm, including arms HD and H6, B-2-2's whole chain and B-2-3's
  stage 1. Each passed its preflight, so phase 2 was ready to run.
- **Two bugs were caught before they cost a run:**
  - stage 3 would have read each train row's targets by row id from a dump stored by position, which is every target
    another row's. The preflight now checks the targets on 16 spread rows against the vendored scorer run live.
  - the read would have left C's extra seeds out had the noise floor fired. Now every seed range spans every seed
    read.
- **Disclosed:** to test the scoring path, the 60-step joint HD smoke run was scored on navtest on 13 Nov (+0.39 ±
  0.14 against R). It chose nothing, and every rule had been committed before it.

### Phase 1: C and O under B-2-1 (`H15.r12`)

**Training (*measured*).**
- The release's own objective, resumed:
  - `mtd_loss`;
  - the final-score BCE against labels its own scorer computes online;
  - its collision-agent and area auxiliaries.
- 3 epochs and 4,413 steps on the 94,195 train rows, the last epoch kept.
- The six runs trained together on one GPU, 1.9 h each.
- Every preflight held:
  - bit-for-bit identity before the first step;
  - the online labeller equal to `pdm-v1` on every checked row.
- The planning loss fell from about 0.65 to 0.596, and the final-score BCE from 0.31 to 0.300.
- O's read gives its agents 37% of its attention; the rest goes to its "no agent" slot.

![B-2-1's training telemetry: the planning loss, the final-score BCE and O's attention to its null token, every run](../../../experiments/H15_perception_sidecar/figures/b2r-training.png)

**How to read it.** Each line is a run. C (blue) and O (orange) are nearly identical seed by seed, because the same
seed gives the same batches. The labelled agents barely change the losses.

**Health, which chooses nothing (*measured*).**
- **Select (4,457 rows):** every run gains a little on the pick's vendored final. C gains +0.11 to +0.14 against R's
  92.01, and O +0.11 to +0.25.
- **R-clean (B-1's value, city-weighted):** every run loses.
  - C loses 3.23, 3.28 and 0.80 points against the release's 75.66; O loses 3.16, 3.44 and 0.64.
  - Most changed windows improve (754 against 203 for C seed 0). The worse ones fail a gate the score multiplies by:
    DDC on 59, 67 and 24 windows, NC on 32, 11 and 4.
  - All 53 windows that fail DDC on both C seeds 0 and 1 are in Las Vegas. Boston's 133 windows lose the most each,
    and carry 36.5% of the city weight.
  - Checked before believing it:
    - the untrained cached path picks as R on all 2,624 windows;
    - the dump's labeller reproduces `pdm-fresh-v1` exactly.
- **The noise floor fired:** C's seed range is 2.48 points against the limit of 0.3. Phase 2 would have run C, D and
  HD on seeds 3 and 4 too. It became moot when K0 stopped B-2.

![Health reads: select's vendored pick final and R-clean's value, each run minus the release](../../../experiments/H15_perception_sidecar/figures/b2r-health.png)

**Navtest, read once (*measured*).** The official scorer through the replay agent, momentum off, every plan equal to
its store. Paired with R's 87.783, SE clustered over 136 logs.

| Run | EPDMS − R | NC | DAC | DDC | TTC | EP | EC | picks switched |
|---|---|---|---|---|---|---|---|---|
| C, seed 0 | **+0.38 ± 0.20** | −0.0032 | +0.0024 | −0.0005 | −0.0012 | +0.0093 | +0.0123 | 58.9% |
| C, seed 1 | **+0.51 ± 0.18** | −0.0019 | +0.0026 | −0.0002 | −0.0007 | +0.0099 | +0.0137 | 57.7% |
| C, seed 2 | **+0.36 ± 0.19** | −0.0028 | +0.0012 | −0.0007 | −0.0025 | +0.0081 | +0.0197 | 56.1% |
| O, seed 0 | **+0.31 ± 0.20** | −0.0026 | +0.0017 | −0.0003 | −0.0012 | +0.0093 | +0.0088 | 57.7% |
| O, seed 1 | **+0.45 ± 0.19** | −0.0021 | +0.0023 | −0.0004 | −0.0012 | +0.0102 | +0.0100 | 58.7% |
| O, seed 2 | **+0.29 ± 0.19** | −0.0026 | +0.0007 | −0.0009 | −0.0020 | +0.0084 | +0.0159 | 58.4% |

![Every B-2 run against the release on navtest, one seed ± 2 SE](../../../experiments/H15_perception_sidecar/figures/b2r-navtest.png)

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| K0, O − C | 0.0 to +0.4 | −0.075 ± 0.084, −0.062 ± 0.075, −0.069 ± 0.073 | **refuted** by its own rule (at or below 0, every seed): **B-2 stops** |
| K1 to K5 | see [the criteria](#criteria) | not run | — |
| C − R (reported) | −0.5 to +0.3 | +0.38, +0.51, +0.36 | above the range on every seed |
| D − C, HD's EC, best of 32 and safe share, the B-2-2 and B-2-3 contrasts (reported) | — | not run | — |

- No seed of K0 is beyond 2 SE either way. An effect above about +0.1 would have shown.
- The rule reads the point estimates, as pre-registered.

![B-2's pre-registered contrasts per seed, over their predicted ranges](../../../experiments/H15_perception_sidecar/figures/b2r-chain.png)

**How to read it.** Each row is a pre-registered contrast. The blue bar is its predicted range, and the dots are seeds
0, 1 and 2 from the top, each with ± 2 SE. K0 sits just left of zero on every seed. The rows below were never run,
because K0 stopped B-2.

![Where each run's difference from R comes from, sub-score by sub-score](../../../experiments/H15_perception_sidecar/figures/b2r-subscores.png)

**How to read it.** One panel per sub-score, in percentage points against R. EP and EC carry the gain, DAC adds a
little, and NC and TTC lose a little. C and O move the same terms by the same amounts.

**Does O use its agents? (*measured*; the test-time swaps, Q7).** Yes.
- Removing O's agents at test changes its pick on 9.0–11.1% of scenes, and by more than 1 m at 4 s on 6.0–7.4%.
- It costs 0.13 to 0.32 points of the vendored final: −0.198 ± 0.083, −0.323 ± 0.087, −0.131 ± 0.076.
- Another log's agents cost less: −0.04 to −0.22.

So O leans on its agents, yet it is no better than C, which never had them.

**How the proposals and the pick change (B-0-4; vendored; *measured*).**

| | R | C, seeds 0–2 | O, seeds 0–2 |
|---|---|---|---|
| minADE to the human (3 s) | 0.163 m | 0.130 m | 0.130 m |
| share of the 32 passing NC and TTC | 89.6% | 92.1–92.2% | 92.1–92.2% |
| best of 32 (vendored final) | 0.9590 | 0.9599–0.9600 | 0.9599–0.9602 |
| the pick's vendored final | 0.8849 | 0.8871–0.8890 | 0.8865–0.8885 |
| lateral spread of the 32 at 4 s (std) | 1.09 m | 1.16–1.17 m | 1.16–1.17 m |
| the pick's move between frames (an EC proxy) | 1.22 m | 1.26–1.29 m | 1.28–1.30 m |
| the 32's matched move from R | — | 0.50–0.55 m | 0.54–0.56 m |

- The release's scorer is nearly tied: its top two scores are within 0.01 on 88.5% of navtest. So any training
  changes over half the picks, mostly between near-equal plans.
- The 32 move about half a metre. R's lower-ranked proposals move the most, and none collapses: the lateral spread
  widens by 7%.

![The 32 on navtest by the vendored labels, against the release](../../../experiments/H15_perception_sidecar/figures/b2r-readouts.png)

**nuScenes, out of domain (*measured*; reported).**
- The best of the 32 improves by 0.016–0.018 m of ADE on every run.
- The pick moves away from the human: at 3 s its L2 rises by 0.18, 0.26 and 0.15 m (C) and by 0.27, 0.30 and 0.14 m
  (O), against the release's 3.735 m, SE about 0.06 clustered by scene.
- Collisions within 3 s move by −0.005, within 2 SE.

The fine-tuned scorer picks worse there even though its proposals got better.

![nuScenes: each run against the release, per seed](../../../experiments/H15_perception_sidecar/figures/b2r-nusc.png)

**Three scenes, each the extreme of a rule (*measured*; the rules are in the JSON).** All three are low-speed
decisions, and the marks along each pick are its poses, 0.5 s apart.
- **Where the fine-tune gains:** R creeps 3.5 m into the intersection on a red light (TLC 0, so EPDMS 0). Every C
  seed holds at 0.3 m and scores 1.
- **Where it loses:** R goes 9.4 m at an even pace. C starts slowly, then accelerates into the car merging from the
  left (NC 0, every seed), while the human slows to 5.4 m.
- **On R-clean:** at a crosswalk where the human and R stay put, C and O creep 2.7 m and fail DDC.

![Three scenes: what the fine-tune changes](../../../experiments/H15_perception_sidecar/figures/b2r-scenes.jpg)

### What it says (PREDICTIONS.md's decision rules; *argued*)

- **K0 stops B-2.** On navtest, perfect front-camera agents within 50 m add nothing to a planner fine-tuned with the
  release's own objective. The read uses them, and loses when they are removed, but no more than C learned without
  them.
  - O was the ceiling for any agent input through D's reader. So HD, H6 and H3, which carry the side-car's noisier
    view of the same agents, were unlikely to beat C.
  - The pre-registered reading: the room on NAVSIM belongs to the pick's horizon, not to agent inputs. That points
    to B-6 (4 s forecasts) or B-12 (closed loop).
  - This agrees with B-1, and with the release's own `agent_pred` training: it already learned per-proposal collision
    and TTC agents over 4 s, from every logged agent.
- **What K0 does not rule out:**
  - another reader;
  - inputs at pass 1, or to the scorer alone (B-3);
  - agents beyond 1 s or out of view;
  - closed loop, where agents react.

  B-2 tested one learned use, by one recipe, open loop.
- **The fine-tune itself is the surprise, and it is not a claim** (C − R was "reported, never claimed").
  - Resuming the release's own objective for three epochs with LoRA gains about 0.4 points on navtest, through
    progress and comfort. It costs a little in collisions.
  - The same weights lose 0.8 to 3.3 points on R-clean, varying by seed, and pick farther from the human on nuScenes.
    Its changes concentrate on low-speed decisions: whether to hold or creep.
  - Whether it is worth anything needs the closed-loop gate, and a look at R-clean's direction failures. Navtest is
    one held-out set, and the other two disagree with it.

### What did not run

Phase 2 did not run, as pre-registered:
- D, HD, H6 and H3 under B-2-1;
- B-2-2 and B-2-3 on C, D and HD;
- the noise floor's seeds 3 and 4.

Their code is in place and smoke-tested (`b2_phase.py --phase 2`), should the owner choose to run them despite the
gate. The decision chain K1 to K5 was not read.

## Questions B-2 can answer

| Question | Read from | Answer (14 Nov) |
|---|---|---|
| Is there room for any agent input to a learned planner on NAVSIM? | K0 | **no, not through this reader:** O − C is −0.06 to −0.08 on every seed |
| Does the side-car's hidden layer add to its decoded list? | K1 | not run (K0) |
| Does the side-car help the planner at all, in that form? | K2 | not run (K0) |
| Does it meet or beat the release? | K3 | not run (K0); the fine-tune alone (C) beats R by about 0.4 on navtest |
| Can a learned reader replace the analytic clearance? | K4 | not run (K0) |
| Does an earlier layer carry more? | K5 | not run (K0) |
| Does the planner use its input, or ignore it? | the test-time swaps | O uses its agents (about 10% of picks change without them) |
| How do the proposals and the pick change? | B-0-4, every arm | the 32 move about 0.5 m and grow safer; more than half the picks change |
| Which recipe moves the planner, and at what cost out of domain? | the recipes, descriptively; nuScenes | B-2-1 only: the pick's L2 at 3 s rises 0.15–0.30 m on nuScenes |

## What B-2 does not answer

- **The scorer alone, with the proposals fixed:** B-3.
- **Agents beyond 1 s, or out of view:** B-6 (4 s forecasts); no camera here sees behind.
- **Closed loop, where agents react:** B-12, the gate.

<a id="changed"></a>

## What changed on 13 Nov (D28)

**B-2 was re-planned at the owner's request after B-1's review.** The owner was asked nine questions, in three
rounds:
- the hidden layer goes into the refiner, and the planner downstream is fine-tuned;
- try the three tensor options;
- keep the decoded list as arm D;
- fold-matched cross-fitting;
- compare all three recipes;
- this is the new B-2, the other ideas renumbered;
- the absorbed 11 Nov plans deleted;
- documents only.

**Approving the plan,** the owner asked for a new branch, pushed, and for these visualizations.

**An independent review was run, and its claims checked against the code.** It added:
- the joint recipe as primary, and the pass-4 copy (Q6);
- one primary contrast with gatekeeping;
- O first, as a gate;
- test-time swaps instead of a trained S;
- the alignment check;
- the release's `agent_pred` heads as evidence.

| 11 Nov | 13 Nov |
|---|---|
| B-2, the agents into the scorer alone | B-3, its plan kept: [B-3](B-3-scorer.md), not scheduled |
| B-3-1, `RefinerAdapter` + LoRA | arm D's adapter, in every recipe |
| B-3-2, H04's agent look | not carried over |
| B-3-3 (agents into the initial queries) and idea B-5 | a later variant: pass 1 sees the slots |
| B-3-4, LoRA off the sampling offsets | a later variant |
| B-4-1, two stages | B-2-2, with a pass-4 copy of the read in stage 3 |
| B-4-2, joint, online PDM scoring | B-2-1, now primary |
| B-4-3 | not carried over |
| arms C, F, O; S in the first pass | C, O, D (was F's input), HD, H6, H3; S replaced by test-time swaps |
| F read the full seed-k side-cars at test | every arm reads the five fold side-cars, at test too |
| "earlier passes at 0.1" (B-3's Q3) | geometric: 0.1, 0.01, 0.001 (`mtd_loss`) |
| B-16, the refiner-retrain finding | B-14; B-2's C − R measures it again |
