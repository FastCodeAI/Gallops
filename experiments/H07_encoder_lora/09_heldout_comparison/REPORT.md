# Final comparison of four Drive-JEPA models on the 223 held-out KITScenes moments

Date 2026-10-20, shakaal.
- **Data read:** this is the single read of the 223 held-out moments (val: 117 scenes, 186 moments; overlap_train_val: 23 scenes, 37 moments). Test was not used.
- **Nothing chosen or tuned:** every checkpoint was fixed before this run, and none was retrained.
- **GPU time:** about 28 min, including the gates. The held-out pass took 279 s for all four models.

**The four models.** All four use the released scorer and projector. Each is evaluated with batch 1, in eval mode, with no image masking (GridMask).

1. **Released model:** the release checkpoint (sha256 checked), unchanged.
2. **Encoder only, tuned on KIT:** the release plus the encoder LoRA (a small low-rank weight change to the camera encoder, `encoder_lora_kit/s2/checkpoints/pass_03/state.pt`), merged into the weights.
3. **Generator only, trained on KIT:** model 2 plus the trained generator, the part that proposes the 32 candidate paths (`generator_joint_kit/s2/w0/ckpt/p09.5.pt`).
4. **Generator and encoder trained together:** the release plus the LoRA and the generator, both from `generator_joint_kit/s3/phase2/ckpt/p01.0.pt`.
   - That one file holds both: `generator` (41 tensors) and `lora` (192). It also holds the guessing game's `key_lora` and `guesser`, which are not used here.
   - The LoRA is merged exactly as in phase-2 evaluation.

**Terms.**
- **Moment:** one keyframe of one scene.
- **Menu:** the 32 candidate paths.
- **Scorer's pick:** the candidate the scorer chooses, which is the model's output.
- **ADE (3 s):** the average distance to the human path over the waypoints at 0.5, 1.0, …, 3.0 s. This is exactly the wrapper's `displacement_errors` with a 3 s horizon: its first round(3 / 0.5) = 6 waypoints.
- **FDE (1 s / 2 s / 3 s):** the distance to the human path at that waypoint.
- **Menu metrics:**
  - **a. Closest candidate ADE (3 s):** the lowest ADE (3 s) among the 32 candidates.
  - **b. Closest candidate FDE (3 s):** that same candidate's FDE (3 s).
  - **c. Candidates within 2 m:** the number of the 32 with ADE (3 s) under 2 m.
  - **d. Variety:** the mean over the 496 candidate pairs of the mean waypoint distance over 0.5–3 s.
- **Groups:**
  - **SLOW:** speed below 15 m/s.
  - **FAST:** 15 m/s and above.
  - **SLOW TURNS:** SLOW moments where the human path's heading at 3 s differs from now by more than 0.3 rad.
  - Speed is the current ego velocity from the model's own ego-status input, the same definition as on the picking moments.
- **Tails** (SLOW TURNS, on b, against Encoder only, tuned on KIT):
  - the number of turns worse by more than 0.5 m;
  - the largest single increase, with its moment;
  - the mean of b over the worst 10 % (4 of 37 moments).
- **Picking moments:** the 1,763 moments of the 50 training-split scenes held out for choosing checkpoints. Every checkpoint here was chosen on them.

## Summary

- **Training the generator gives a clearly better menu on held-out moments.** Generator only, against Encoder only: closest candidate ADE (3 s) **0.378 → 0.194 m** (difference −0.185, 95 % interval [−0.231, −0.146]); 138 moments are better by more than 10 cm and 8 are worse.
- **The gain is in every group that can be read:** SLOW −0.165 m [−0.199, −0.136]; SLOW TURNS −0.207 m [−0.389, −0.080]. FAST has only 2 moments (too few to read); both improve, by about 2.4 m.
- **The menu is tighter:** 17.7 → 25.9 candidates within 2 m, and variety 2.82 → 1.63 m.
- **The scorer's pick improves too:** ADE (3 s) 1.879 → **1.146 m** (−0.733 [−0.930, −0.545]); FDE (3 s) 3.442 → 2.644 m. Against the Released model (2.372 m), that is about half the error.
- **Training the encoder together with the generator does not differ beyond noise:**
  - menu −0.004 m [−0.013, +0.004];
  - pick −0.045 m [−0.114, +0.026];
  - SLOW TURNS pick +0.219 m (worse), [−0.055, +0.535].
- **Encoder only, against Released** reproduces the earlier finding: the menu is unchanged (+0.004 [−0.009, +0.018]) and the pick is better (−0.493 [−0.687, −0.313]).
- **Held-out compared with picking moments:**
  - The SLOW gain from training the generator is at least as large on held-out moments: −0.165 against −0.142 m on picking moments.
  - Absolute SLOW values are 0.01–0.04 m worse on held-out moments for every model, Encoder only included. So the held-out moments are slightly harder; it is not a sign of overfitting.
  - The ALL values differ mostly because the picking moments include 82 FAST moments and the held-out set has only 2.
- **Worst turns:** Generator only has 1 slow turn worse by more than 0.5 m, and the mean of b over the worst 4 turns falls from 5.75 to 2.80 m.

## Gates

| Gate | Result |
|---|---|
| **G2:** models 3 and 4 load as they were evaluated (checked before any held-out read) | **PASS, exact.** On the 103 baseline picking moments, through the path they were checked with: menus, grades and picks are bit-identical to the saved dumps (103 / 103 each). Closest candidate FDE (3 s) on the 103 is 0.402655873168921 (model 3) and 0.3690581848373963 (model 4), equal to the recorded values. Through the KIT wrapper path used on the held-out moments, the same 103 menus and picks are also bit-identical (max difference 0.0 m). |
| **G1:** models 1 and 2 reproduce the earlier 223-moment results | **PASS, exact** (see deviation 1). Scorer's pick ADE (3 s) is **2.371879947736172** and **1.8793429804954402**, equal to the recorded values to full precision. All 223 per-moment rows (ADE, FDE at 1 / 2 / 3 s, predicted path) and all 223 dumps (menu, grades, pick, output) are identical for both models. |

## Main table (223 held-out moments; mean in m, median in brackets for ALL)

Groups: ALL (n = 223), SLOW (n = 221), FAST (n = 2, too few to read), SLOW TURNS (n = 37).

| Model | ALL: closest candidate ADE (3 s) | ALL: closest candidate FDE (3 s) | ALL: candidates within 2 m | ALL: variety | SLOW: closest candidate ADE (3 s) / FDE (3 s) | FAST: closest candidate ADE (3 s) / FDE (3 s) | SLOW TURNS: closest candidate ADE (3 s) / FDE (3 s) | scorer's pick ADE (3 s) | scorer's pick FDE (1 s) | scorer's pick FDE (2 s) | scorer's pick FDE (3 s) | SLOW TURNS tails vs Encoder only, tuned on KIT: worse > 0.5 m on FDE · largest increase (moment) · mean FDE of worst 10 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Released model | 0.374 (0.301) | 0.664 (0.421) | 17.5 | 2.83 | 0.357 / 0.628 | 2.337 / 4.696 ‡ | 0.558 / 1.129 | 2.372 (1.392) | 1.314 | 2.679 | 4.206 | 3 · +0.78 m (`873eebf4` t = 130) · 5.16 |
| Encoder only, tuned on KIT | 0.378 (0.307) | 0.662 (0.426) | 17.7 | 2.82 | 0.359 / 0.624 | 2.523 / 4.949 ‡ | 0.557 / 1.192 | 1.879 (1.246) | 1.002 | 2.085 | 3.442 | — · 5.75 |
| Generator only, trained on KIT | 0.194 (0.146) | 0.379 (0.265) | 25.9 | 1.63 | 0.194 / 0.380 | 0.156 / 0.320 ‡ | 0.350 / 0.701 | 1.146 (0.925) | 0.319 | 1.223 | 2.644 | 1 · +0.60 m (`8dcfc2ba` t = 40) · 2.80 |
| Generator and encoder trained together | 0.190 (0.138) | 0.377 (0.243) | 26.0 | 1.60 | 0.189 / 0.377 | 0.267 / 0.329 ‡ | 0.346 / 0.750 | 1.101 (0.833) | 0.313 | 1.169 | 2.533 | 3 · +0.78 m (`6f47eb66` t = 490) · 2.85 |

‡ too few to read (fewer than 20 moments).

## Paired differences (second model minus first; negative = the newer model is better)

95 % interval: paired bootstrap, 10,000 resamples of whole scenes, seed 0.

| Pair | Metric | Group | n | mean (m) | median (m) | better / worse by > 10 cm | 95 % interval (m) |
|---|---|---|---|---|---|---|---|
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | ALL | 223 | +0.004 | +0.002 | 19 / 19 | [-0.009, +0.018] |
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | SLOW | 221 | +0.002 | +0.002 | 19 / 18 | [-0.011, +0.017] |
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | FAST ‡ | 2 | +0.186 | +0.186 | 0 / 1 | [-0.006, +0.379] |
| Encoder only, tuned on KIT vs Released model | closest candidate ADE (3 s) | SLOW TURNS | 37 | -0.000 | -0.014 | 8 / 3 | [-0.058, +0.071] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | ALL | 223 | -0.493 | -0.053 | 96 / 55 | [-0.687, -0.313] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | SLOW | 221 | -0.497 | -0.053 | 95 / 54 | [-0.696, -0.318] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | FAST ‡ | 2 | +0.050 | +0.050 | 1 / 1 | [-0.176, +0.276] |
| Encoder only, tuned on KIT vs Released model | scorer's pick ADE (3 s) | SLOW TURNS | 37 | -0.455 | -0.050 | 14 / 7 | [-0.864, -0.088] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | ALL | 223 | -0.185 | -0.143 | 138 / 8 | [-0.231, -0.146] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | SLOW | 221 | -0.165 | -0.142 | 136 / 8 | [-0.199, -0.136] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | FAST ‡ | 2 | -2.368 | -2.368 | 2 / 0 | [-3.331, -1.404] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | closest candidate ADE (3 s) | SLOW TURNS | 37 | -0.207 | -0.115 | 21 / 2 | [-0.389, -0.080] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | ALL | 223 | -0.733 | -0.398 | 144 / 58 | [-0.930, -0.545] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | SLOW | 221 | -0.702 | -0.383 | 142 / 58 | [-0.888, -0.518] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | FAST ‡ | 2 | -4.250 | -4.250 | 2 / 0 | [-5.749, -2.750] |
| Generator only, trained on KIT vs Encoder only, tuned on KIT | scorer's pick ADE (3 s) | SLOW TURNS | 37 | -1.026 | -0.594 | 29 / 8 | [-1.601, -0.539] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | ALL | 223 | -0.004 | -0.007 | 7 / 9 | [-0.013, +0.004] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | SLOW | 221 | -0.005 | -0.008 | 7 / 8 | [-0.013, +0.003] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | FAST ‡ | 2 | +0.112 | +0.112 | 0 / 1 | [+0.095, +0.128] |
| Generator and encoder trained together vs Generator only, trained on KIT | closest candidate ADE (3 s) | SLOW TURNS | 37 | -0.004 | -0.010 | 4 / 6 | [-0.044, +0.032] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | ALL | 223 | -0.045 | -0.070 | 98 / 65 | [-0.114, +0.026] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | SLOW | 221 | -0.047 | -0.070 | 97 / 64 | [-0.115, +0.024] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | FAST ‡ | 2 | +0.236 | +0.236 | 1 / 1 | [-0.297, +0.769] |
| Generator and encoder trained together vs Generator only, trained on KIT | scorer's pick ADE (3 s) | SLOW TURNS | 37 | +0.219 | +0.086 | 7 / 17 | [-0.055, +0.535] |

## Picking moments vs held-out moments (mean, m)

Picking moments: the 1,763 moments of the 50 picking scenes the checkpoints were chosen on. Held-out: the 223 moments above.

| Model | Group | closest candidate ADE (3 s): picking → held-out | closest candidate FDE (3 s): picking → held-out |
|---|---|---|---|
| Encoder only, tuned on KIT (reference) | ALL | 0.617 → 0.378 | 1.093 → 0.662 |
| Encoder only, tuned on KIT (reference) | SLOW | 0.322 → 0.359 | 0.572 → 0.624 |
| Generator only, trained on KIT | ALL | 0.202 → 0.194 | 0.391 → 0.379 |
| Generator only, trained on KIT | SLOW | 0.180 → 0.194 | 0.346 → 0.380 |
| Generator and encoder trained together | ALL | 0.201 → 0.190 | 0.405 → 0.377 |
| Generator and encoder trained together | SLOW | 0.177 → 0.189 | 0.355 → 0.377 |
| gain of Generator only, trained on KIT over Encoder only, tuned on KIT | ALL | -0.415 → -0.185 | -0.702 → -0.283 |
| gain of Generator only, trained on KIT over Encoder only, tuned on KIT | SLOW | -0.142 → -0.165 | -0.226 → -0.244 |
| gain of Generator and encoder trained together over Encoder only, tuned on KIT | ALL | -0.415 → -0.189 | -0.688 → -0.286 |
| gain of Generator and encoder trained together over Encoder only, tuned on KIT | SLOW | -0.144 → -0.170 | -0.217 → -0.246 |


## Pictures

Each picture is top-down with forward up and one column per model. Grey lines are the 32 candidates. Red is the closest candidate, blue dotted is the scorer's pick, and the dashed line is the human path.

- **`out/menus_most_improved_generator_vs_encoder.png`:** the 3 moments where closest candidate ADE (3 s) improves most, Generator only against Encoder only.
  - `54cd7acd` t = 40, FAST: 3.50 → 0.16 m. The released menu stops short of the human path at 15.9 m/s; the trained menu reaches it.
  - `f0b1b61b` t = 130: 5.08 → 2.35 m. The human path goes right while every released candidate goes left. The trained menus add a few rightward candidates but still miss.
  - `ba826f49` t = 310, FAST: 1.55 → 0.15 m.
- **`out/menus_most_worsened_generator_vs_encoder.png`:** the 3 moments where it worsens most.
  - `6f47eb66` t = 490: 0.53 → 1.17 m. The trained menus fan out less to the right.
  - `ba826f49` t = 220: 0.55 → 0.86 m. The trained menus lean left, while the human path bears right.
  - `6f47eb66` t = 400: 0.18 → 0.39 m. Still a right turn that the menu covers, only slightly less closely.
  - All three are small losses compared with the gains.
- **`out/menus_fast.png`:** both FAST moments (there are only 2). They are the same two moments as above (`54cd7acd` t = 40 and `ba826f49` t = 310).

Values for every pictured moment are in `out/final_tables.json` (`pictures`).

## Deviations and choices, with reasons

1. **G1 was evaluated twice. The model outputs were never recomputed and the held-out moments were not read again.**
   - The first G1 compared the mean ADE in loop order. For model 1 that gave 2.3718799477361716, against the recorded 2.371879947736172, a difference in the last binary digit.
   - In that same first check, all 223 per-moment rows and all 223 dumps were already identical.
   - The earlier evaluation averaged over sorted moment keys. Averaged the same way, the mean equals the recorded value exactly, for both models.
   - I fixed the gate's averaging order and re-ran only G1 from the saved results (`final.py g1`, `logs/g1_regate.log`). The first result is kept in `out/state.json` as `G1_first_attempt_loop_order_mean`.
   - This deserves a second look from you, because the brief says to stop when a gate fails. I judged it a gate-code error, not a reproduction failure.
2. **All four models run on each sample inside one loop.** This is the earlier evaluation loop unchanged (adapter `build_sample`, `compute_trajectory`, `displacement_errors` at 3 s, `forward_dump`), extended to four models. It is how the moments are read only once. G1 shows the results are bit-identical to the earlier separate runs.
3. **The wrapper's `build_root` was not called.** It can create links inside experiments/drivejepa_kitscenes. Instead, a read-only check confirms that all 140 scene links already exist, and that no held-out scene is in the train or picking scenes.
4. **Where the held-out scenes come from.** `splits.json` holds only the train and picking scenes. The 117 + 23 held-out scenes come from the same links the earlier 223-moment evaluation used. Their moments are checked against the baseline pickle (sha256 checked): 223 moments, identical human paths.
5. **Worst 10 % of 37 slow turns** is rounded up to 4 moments.
6. **FAST has 2 moments.** They are reported throughout and marked "too few to read" (‡). Their bootstrap interval is shown but means little.
7. **Additions to the brief:**
   - the scorer's pick is drawn in the pictures;
   - tails are also given for the Released model;
   - the side table adds Encoder only on the picking moments, from its saved picking dump, as the reference;
   - the paired table is broken down by group.
8. **A wiring test ran first,** on 2 picking scenes (`final.py smoke`, `logs/smoke.log`), so that a code error could not cost a second held-out read.

## Files

- **Code:**
  - `src/final.py`: the gates and the single held-out pass. Subcommands: `all`, `smoke`, `g1`.
  - `src/final_report.py`: the tables, bootstrap and pictures; CPU only.
  - Run both with `./run.sh`.
- **Logs:** `logs/final.log`, `logs/g1_regate.log`, `logs/report.log`, `logs/smoke.log`.
- **Results:**
  - `out/state.json`: preflight, model checks, G2, G1.
  - `out/final_tables.json`: every number in this report.
  - `out/tables.md`: the generated tables.
- **Large outputs:** `{runs}/drivejepa_kitscenes/generator_joint_kit/s4/heldout223.pkl` holds the per-moment rows and dumps for all four models.
