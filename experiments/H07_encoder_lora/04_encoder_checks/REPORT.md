# Encoder LoRA on KIT — two closing checks (no training)

Date 2026-10-17, shakaal.
- **Folder:** `experiments/drivejepa_kitscenes/04_encoder_checks/`.
- **Code:** one script, `src/close.py`. It imports `02_encoder_build/src` and `03_encoder_training/src` unchanged and runs via `./run.sh`, the same interpreter as s1/s2.
- **GPU:** about 5 min in total (Check 1 160 s, Check 2 147 s), with no wait.
- **Data rules:**
  - Test and test-e2e were never read.
  - Naren's files were only read; the pickle sha256 was re-checked inside the script.
  - Val and overlap were read once, by the zero-LoRA run in Check 2. You authorised that run in this task.

## Verdicts

1. **Check 1 — TRAINED-MATTERS.**
   - Pass 3's selection ADE is 1.873. The best of the 5 random LoRAs, with the same per-layer change size, is 2.413, so pass 3 wins by 0.540 m (the rule needs > 0.1 m).
   - The random LoRAs are on average **0.593 m worse** than pass 3, and every one is slightly worse than zero LoRA (2.392).
2. **Check 2 — same-software "before" on the 223 moments.**
   - **Zero LoRA in our software: ADE 2.3719, against August's 2.3636.** All of that difference comes from **one moment whose pick flips** (val `6f47eb66…`, keyframe 310: ΔADE +1.835 m). The other 222 moments agree within 4.3 cm per waypoint; the median gap is 1.9 mm.
   - **Against this exact before, pass 3 gives −0.493 m ADE** (2.3719 → 1.8793). Best-of-32 is unchanged (0.374 → 0.378), and pick regret falls from 2.00 to 1.50 m.
   - This confirms on val/overlap what selection showed: the scorer picks better from an essentially unchanged menu.

## Check 1 — random-change control (103 selection moments)

**How the random LoRAs were built** (`close.py` `random_lora`, `layer_norms`):
- For each of the 96 layers, the pass-3 change is `ΔW = (α/r)·B·A`, with α/r = 1. This follows s1 `lora.py` `LoRALinear.forward` / `merged_weight`.
- Seeds 1–5 draw A and B from N(0, 1) using `torch.Generator().manual_seed(seed)`. B is then rescaled so ‖(α/r)·B·A‖_F equals the pass-3 layer's. Norms match to a relative error of 1.5e-9 in the worst layer.
- Pass-3 layer norms range from 0.95 to 6.36 and sum to 264.9.

**How they were evaluated:**
- Exactly as the s2 per-pass check: the agent with the encoder wrapped by LoRA, fp32, unmerged, run through `evalsets.selection_eval` on the saved selection features. Drift is measured against the s2 zero-LoRA dump with s1 `drift.py`. Best-of-32 and plan speed are as in s2 `g_extra.py`.
- Pass 3 was re-run in the same process and reproduces its recorded ADE **exactly** (1.8732948781716308).

| | ADE mean | ADE median | FDE@3s | Best-of-32 ADE | Mean / max proposal shift (m) | Mean \|Δgrade\| | Picks changed | Plan speed (m/s) |
|---|---|---|---|---|---|---|---|---|
| Zero LoRA | 2.392 | 1.414 | 4.346 | 0.573 | 0 / 0 | 0 | 0 | 5.87 |
| **Pass 3** | **1.873** | **1.001** | **3.477** | 0.563 | 0.267 / 5.99 | 0.090 | 74 | 6.17 |
| Seed 1 | 2.465 | 1.419 | 4.451 | 0.573 | 0.048 / 3.60 | 0.011 | 15 | 5.82 |
| Seed 2 | 2.441 | 1.469 | 4.447 | 0.573 | 0.049 / 5.71 | 0.009 | 16 | 5.86 |
| Seed 3 | 2.534 | 1.507 | 4.512 | 0.568 | 0.042 / 3.41 | 0.011 | 20 | 5.77 |
| Seed 4 | 2.479 | 1.525 | 4.467 | 0.574 | 0.050 / 5.58 | 0.012 | 19 | 5.79 |
| Seed 5 | 2.413 | 1.458 | 4.375 | 0.571 | 0.045 / 5.49 | 0.010 | 15 | 5.89 |
| **Random: mean [range]** | **2.467** [2.413, 2.534] | 1.476 [1.419, 1.525] | 4.450 [4.375, 4.512] | 0.572 [0.568, 0.574] | 0.047 [0.042, 0.050] / 4.76 | 0.011 [0.009, 0.012] | 17 [15, 20] | 5.83 [5.77, 5.89] |

**Verdict rule** (fixed in the task):
- TRAINED-MATTERS if pass 3 beats every seed by more than 0.1 m.
- DRIFT if the random mean is within 0.1 m of pass 3.
- Otherwise MIXED.

The margin to the best seed is 0.540 m, so the result is **TRAINED-MATTERS** (`out/check1_random.json`).

**One caveat on the control.** Matching the size of the weight change does not match the size of the output change. Random directions move the planner far less than the trained LoRA does: mean proposal shift 0.047 against 0.267 m, |Δgrade| 0.011 against 0.090, and 17 picks changed against 74.
- So the control rules out "any perturbation of this weight size helps": random ones hurt slightly.
- It does not test a random change scaled to the same output drift. That would need a different rescaling rule, and I did not run one.

## Check 2 — same-software "before" on val + overlap_train_val (223 moments)

**Run.**
- This is the s2 final-evaluation loop (s2 `f_final.py`, which is s1 `f_baseline.py`'s copy of `RUN.main` with `--splits val overlap_train_val --protocol e2e --command gt --eval-horizon 3`). It used the plain agent: zero LoRA, the original encoder.
- Every moment's dumped trajectory equals `compute_trajectory`'s output bit-for-bit, and every recorded human future equals the pickle's (both asserted).
- Saved to `out/zero223_rows.pkl` and `out/zero223_dump.pkl` (32 proposals, grades, pick, trajectory).

**Against the August pickle** (`out/check2_zero223.json` `vs_august`):

| Measure | Value |
|---|---|
| Bit-identical moments | 0 / 223 |
| Max per-waypoint difference: > 1 mm / > 1 cm / > 10 cm | 157 / 8 / **1** moments |
| Median of the per-moment max difference | 1.9 mm |
| Per-moment ADE difference: mean / mean \|·\| / max \|·\| | +0.0083 / 0.0088 / **1.835** m |
| Moments with \|ΔADE\| > 1 cm | **1** (the same moment) |

- **The one outlier** is `6f47eb66` (val), keyframe 310: max pose difference 2.32 m, ADE 1.90 → 3.73 m. A 2.3 m jump means a different candidate was picked, so the tiny software difference tipped a close scorer decision.
- The next-largest difference is 4.3 cm (`b3dd3c95`, keyframe 40).
- **This corrects s2's noise statement.** The earlier "≤ 1.2 mm" came from checking one moment. Over all 223, the noise is millimetres everywhere except one pick flip worth +1.8 m on that moment, or +0.008 m on the 223-moment mean.

**Against pass 3** (pass 3 numbers are from the s2 Part F dump):

| Split | n | ADE mean, zero → pass 3 | ADE median, zero → pass 3 | FDE@3s mean | Best-of-32 ADE mean (median) | Pick regret mean (median) | Plan speed (m/s) | Pass 3 vs zero: picks changed / better > 10 cm / worse > 10 cm |
|---|---|---|---|---|---|---|---|---|
| val | 186 | 2.379 → **1.903** | 1.467 → 1.249 | 4.214 → 3.490 | 0.388 (0.303) → 0.392 (0.308) | 1.99 (1.04) → 1.51 (0.89) | 6.13 → 6.49 | 104 / 84 / 45 |
| overlap_train_val | 37 | 2.335 → **1.759** | 0.996 → **1.122** | 4.169 → 3.202 | 0.307 (0.277) → 0.311 (0.295) | 2.03 (0.78) → 1.45 (0.80) | 6.97 → 7.50 | 15 / 12 / 10 |
| all | 223 | 2.372 → **1.879** | 1.392 → 1.246 | 4.206 → 3.442 | 0.374 (0.301) → 0.378 (0.307) | 2.00 (0.99) → 1.50 (0.87) | 6.27 → 6.66 | 119 / 96 / 55 |

- Mean proposal shift, zero → pass 3, is 0.30 m (max 15.7 m), and mean |Δgrade| is 0.068.
- **What this shows:**
  - The menu's best candidate is no better after LoRA. Best-of-32 rises slightly, by 4–6 mm.
  - The whole gain is in the pick: mean regret drops by about 0.5 m on both splits.
  - The overlap median gets worse by 0.13 m under the same-software before too, so this is not an artefact of comparing against August.

## Open questions for Sid (not decided)

1. **Which before to cite.** The headline can now say 2.372 (zero LoRA in our software) → 1.879. Or keep August's 2.364 and note the one pick flip. Which one?
2. **Output-matched control.** Do you want a random control rescaled to match pass 3's output drift (e.g. mean proposal shift 0.27 m) rather than its weight norm? That is a stronger test of "is it the direction or just the amount of change".
3. **Fragile picks.** `6f47eb66` keyframe 310 flips on millimetre-level noise, so its candidates' grades must be nearly tied. Are near-tied picks worth listing? On the 223, the gap between the top two grades can be read from `zero223_dump.pkl`.

## Files

- **`src/close.py`:** the `random` and `zero223` subcommands.
- **`out/`:**
  - `check1_random.json` and `random_seed{1..5}_dump.pkl`
  - `check2_zero223.json`, `zero223_rows.pkl` and `zero223_dump.pkl`
- **`logs/check2.log`.**
