# S9 — predictions, written before anything is run

Written 2026-10-09 after S8's REPORT.md. Before this file was hashed I read code and listed stored files (the calculator's
collision, time-buffer and comfort code; S0's raw box loader; S6's dump script; the S3 manifest) and counted which pull-out
clips have boxes on the NAS. Nothing was computed from features, grades, menus or scores.
The two predictions are Sid's, copied from the brief. The definitions are mine and are fixed here.

## P-B (two-stage head on frozen features)

**Prediction.** Picks graded 0 < 4 % (S7 head 9.5 %) and regret < 0.08 (S7 head 0.110).

Definitions.
- Data as S7 H3: `slot_9` (256 numbers per plan) and true menu grades from S6's dumps; SEEN for training, UNSEEN for the
  final evaluation only. Split inside SEEN: S7's split (80 % / 20 % of clips, seed 20261008), so S7's head is comparable.
  All seeds 20261008; bootstrap seed 20261007.
- Gates. "Kerb" = drivable area, "oncoming" = driving direction (after the D3 exemption, as stored), collision. A plan
  fails a gate when the stored gate value is below 1 (collision 0.5 counts as failing). Traffic light never fails in this
  data (S8) and is left out.
- **Gate head**: 256 → 64 → 3 with one ReLU, one output per gate, BCE on "fails" per gate (unknown labels masked; none are
  unknown in B). P of a plan = 1 − Π(1 − p_gate) over the three outputs = P(any gate fails). Adam, learning rate 1e-3,
  256 instants per batch, at most 60 passes, stop when validation loss has not improved for 8 passes; weight decay from
  {0, 1e-4, 1e-3, 1e-2} by validation loss. Three separate outputs instead of one "any" output so that part C can add
  examples that have a collision label only.
- B.2: AUROC on UNSEEN, pooled over plans, per gate (that gate's own output against that gate failing) and jointly (P
  against any gate failing), 95 % clip-bootstrap interval with 200 resamples (as S6's probe table). Compared with S6 table
  4.7: collision 0.88, drivable area 0.88, driving direction 0.89.
- **Ranker**: S7's head (256 → 64 → 1, pairwise hinge, margin 0.1, pairs with grades ≥ 0.05 apart, weight decay 0.01,
  same optimiser and stopping rule on validation mean regret of the whole two-stage selection). Two changes: each pair's
  loss is multiplied by |grade gap| and the batch loss is the weighted sum divided by the sum of weights; and only pairs
  where both plans have gate P ≤ t are used. One ranker per t.
- **Selection**: plans with P > t are vetoed; the pick is the ranker's top plan among the rest; if all 32 are vetoed, the
  plan with the lowest P.
- **t** from {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (no veto)}, by mean regret on the SEEN validation
  clips; ties to the larger t. The whole curve is reported. Written to `b_selection.json` before UNSEEN is opened.
  Known limit, stated now: the ranker's pair filter uses the gate head's P on its own training clips, which is optimistic.
- UNSEEN evaluation at that one t, by slice: regret mean and 90th percentile, picks graded exactly 0 and which gate zeroed
  them (S8's rule), true-best rate, share of picks ≥ 0.9, rank agreement (per-instant Spearman of the ranker's output with
  the grade over all 32 plans, median). With the momentum nudge: S6's offline stand-in, (14 × sigmoid(ranker output) + 2 ×
  comfort) / 16 over the plans that are not vetoed.
- Rows: the two-stage head; baselines scorer, S7 ranking head (loaded from `09_three_hypotheses/h3_models.pt`), random.
  Also reported, to separate the two changes: veto + S7's ranker; weighted ranker with no veto (t = 1.0); gate head alone
  (pick the lowest P).
- **Verdict**: UNSEEN, all instants, raw pick, point estimates: consistent if picks graded 0 < 4 % AND mean regret < 0.08.

Not blind: S8 showed that 65 % of the S7 head's regret comes from the 9.5 % of picks graded 0, that its regret without
them is 0.043, and that drivable area zeroes 60 % of them; S6 gave probe AUROCs of 0.88–0.89 for the gates. No gate-head
or veto result has been computed.

## P-C (partial grades on the raw pull-out clips)

**Prediction.** A leaving plan passes collision on > 60 % of pull-out menus; after C.4, the gate P of the best leave plan on
d2715023 drops below 0.5 at a majority of replans.

Definitions.
- Clips: the pull-out clips in the training manifest (`type == B_raw`; 283 clips, 9,053 examples; the brief says 285). They
  are raw training clips: the pass-9 generator trained on them (human path only, no grades). All have boxes
  (`obstacle.offline`) and egomotion on the NAS.
- Menu: 32 plans from the pass-9 generator, CPU, same split forward as S6 (parity with the model's own forward checked).
- Map-free calculator run: the vendored calculator's own methods, called in its own order, on a scene with the logged
  boxes, the car's state from egomotion and the real vehicle box (D2), and no map. Its collision and time-buffer code read
  three map facts about the car at each step: "in more than one lane", "off the drivable area", "in an intersection"
  (C.1). With no map these must be set by hand. Two settings, both stored:
  - **strict (primary)**: the car is treated as in more than one lane at every step, so a side contact is its fault and
    the time buffer counts every actor that is not behind. Chosen as primary because a car pulling out from the kerb is
    outside a lane.
  - **lenient**: none of the three, so side contacts with moving actors are never the car's fault and the time buffer
    counts only actors ahead.
- Per example and plan: collision gate (1, 0.5, 0), time buffer (1 / 0), comfort (1 / 0), and
  **map-free grade = collision gate × (5 × time buffer + 2 × comfort) / 7** (the calculator's own weights, the other terms
  left out). "Passes collision" = gate equal to 1.
- Leave rule as S7 H2: the plan ends ≥ 2 m to the side or ≥ 5 m forward of the car.
- **First half, verdict**: share of all 9,053 pull-out menus on which at least one plan leaves and passes collision
  (strict) > 60 %. Reported next to it, because once the car is moving nearly every plan "leaves": the same share over
  menus with the car under 0.5 m/s, and over menus before the clip's first move-off time (`01_find_clips/r3_raw_moveoffs.csv`),
  by clip, strict and lenient.
- C.4: the gate head is retrained with the pull-out examples added (80 % / 20 % of pull-out clips to train / validation,
  seed 20261008), collision output only (label = strict map-free collision gate below 1), the other two outputs masked.
  The ranker is retrained with the same recipe on the new gate head's P; t is not re-chosen. B.5 is re-evaluated; "new
  gate head + B's ranker" is reported too.
- Trial scenes d2715023 and b44a4a44 (**exam clips, inspection only, not an evaluation**): the 256 features at each replan
  are not stored in the trial folder; they are recomputed on CPU from the trial's logged camera frames with the same
  pass-9 weights, nothing written under `07_closed_loop_test/`, and accepted only if the recomputed 32 plans match the stored ones.
  Per replan: gate P (B's head and the C.4 head) of every plan and the new pick. Best leave plan = the leave plan with the
  highest raw scorer score (S7 H2).
- **Second half, verdict**: on d2715023, C.4-head gate P of the best leave plan < 0.5 at more than half of the replans.
  The B-head value at the same replans is reported so that "drops" can be read.
- P-C as a whole is consistent only if both halves hold.

Not blind: S7/S8 gave the scorer's raw scores on the two trial scenes (best leave 0.09 on d2715023, 0.80 on b44a4a44).
No gate P and no map-free check value has been computed.

## A

Spec only (`A_spec.md`). Nothing is run.
