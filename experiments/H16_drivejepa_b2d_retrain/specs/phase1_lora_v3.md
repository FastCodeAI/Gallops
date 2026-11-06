# CLAUDE.md: Bench2Drive Drive-JEPA, Phase 1 — LoRA pretraining of the encoder

(Given by Navvrat on 05 Nov 2026, verbatim. Third version, sent while the long run was on pass 3: no planner report. The second is CLAUDE_v2_05nov.md, the first recon_b2d_drivejepa/SPEC_phase1_lora.md.)

**What:** LoRA on Drive-JEPA's encoder, trained with its own self-supervised objective on Bench2Drive front-camera clips.
- **The objective (Sid's "hide-and-guess"):** hide parts of the two frames; a guesser predicts what a slow-moving copy of the encoder sees there.
- **Not in this phase:** labels, the planner, scores. The planner is Phase 2.
- **Output:** a LoRA, and a merged encoder for Phase 2 (Drive-JEPA-style planner training).

Report numbers only, no thresholds or verdicts: Navvrat decides.

## Sid's code (found in the recon; read-only)

Sid's two notes come with this file: "HANDOVER: Drive-JEPA encoder-LoRA retraining on Physical AI AV" and "CONTEXT: Drive-JEPA re-training with LoRA".

- **The recipe is his KIT Experiment 1,** the only standalone hide-and-guess run: `experiments/drivejepa_kitscenes/03_encoder_training` (E1t) and `02_encoder_build` (E1b) in `{sid_checkout}/Gallops-kitscenes`, PR #24 head e75aaac. Entry: `E1t/src/train.py`, `warmup` then `main`. Settings: `E1t/src/s2common.py:33-59`.
- **Import as is:** `E1b/src/lora.py`, `E1b/src/players.py` (guesser and loss), `E1t/src/masks2.py`, and his merge.
- **Change in your copy:** the loader and frame source, the splits, the stop rule, checkpoint saving, and the watcher's paths. Each is set below.
- **Not this recipe:** `Generator-Joint-KIT/s5_all_three` is Experiment 3, and `Generator-Joint-PAI-alpasim` is the 0.1.8 family; both train the LoRA together with the planner. From PAI, take only the full checkpoint save (`s9_018b_train/src/train_018b.py:1088-1095`) and the reboot watcher.

## Fixed (ask before changing)

| | |
|---|---|
| **Encoder** | Drive-JEPA's released agent encoder, frozen: the copy our Bench2Drive runner loads, the one Sid started from. Only the LoRA and the guesser train |
| **LoRA** | Sid's: rank 16, alpha 16, on qkv / proj / fc1 / fc2 of all 24 blocks. B starts at zero, so step 0 equals the released encoder exactly |
| **Objective** | Sid's hide-and-guess, as in his code (below) |
| **Optimiser** | Sid's AdamW: LoRA lr 2e-4, wd 0; guesser lr 1e-4, wd 0.04; a 200-step ramp in each phase |
| **Schedule** | Sid's guesser-only warm-up, 1,000 steps. Then 3 passes of the main phase. Batch 32, bf16 |
| **Input** | Front camera, 2 frames 0.5 s apart, crop `native` (below) |
| **Data** | 944 clips: the base set's 968 (895 in `paddock-data/phase0a-labels/b2d_base`, 73 in `b2d_pilot`) minus the 24 trigger twins in the recon's `section_DE_evidence/twins.txt`. Never the 220 eval routes or the v0.0.4 validation routes |
| **Held-out** | About 5% of the 944 clips, by route, for the held-out loss only. List them in a file |
| **GPU** | Alongside run 3A-M, on the same GPU. Run 3A-M has priority: if memory runs short, stop Phase 1 (it resumes from its last save) and ask Navvrat |

Report every value used.

**Hide-and-guess, from Sid's code:**
- **Masks:** 8 × 0.15 and 2 × 0.70, the same patches in both frames.
- **Guesser:** the predictor from Drive-JEPA's pretraining checkpoint `vitl_merge_3dataset_e50.pt`, with his grid fix.
- **Slow copy:** EMA 0.99925 of the LoRA, main phase only.
- **Loss:** L1 to the layer-normed target at the hidden tokens.

**Input:**
- **Frames:** the 1600 × 900 originals (`<clip>/camera/rgb_front/`), not run 3A Phase 1's 256 × 512 copies: their crop differs.
- **Pairs:** (t − 0.5 s, t) are neighbouring files, since the local copies keep every 5th frame of 10 Hz. Take every pair. Keep the pair gap and the stride as two settings; Sid's `FRAME_STEP` is both (`E1t/src/s2common.py:31`).
- **Crop `native`,** the one every published Drive-JEPA Bench2Drive run used (`paddock-ops/worktrees/rerank-native`, `policy.py:99`): resize to 800 × 450, then Drive-JEPA's `_get_camera_feature` (`drive_jepa_perception_based/drive_jepa_features.py:38-65`: rows [28:-28], 512 × 256, ImageNet norm). Import it; don't rewrite it.

## Checks before the long run

1. **Step 0:** encoder outputs with the LoRA attached equal the released encoder's, bit for bit, on one batch. Report the sha256 of the encoder and of the e50 checkpoint you load (recorded: `740335c8…0d05`, `46491827…668f`).
2. **What trains:** every base ViT weight stays unchanged. After 10 warm-up steps only the guesser has changed; after 10 main-phase steps the LoRA and the guesser have.
3. **Input contract:** a 1600 × 900 frame resized to 800 × 450 gives identical tensors through the training loader and through the runner's `native` path. Report the largest difference, and fx, fy at 512 × 256 (the runner's: 365.6, 371.2).
4. **First 50 steps, beside run 3A-M:** Phase 1's peak memory, step time and loader wait; the GPU's total memory in use; run 3A-M's step time before and after Phase 1 started; loss finite. Estimate the time per pass and in total.
5. **Resume:** every save holds the LoRA, guesser, slow copy, optimiser, sampler and RNG state, at a new path. Sid's `train.py` deletes and overwrites checkpoints and saves no RNG or sampler; fix both in your copy. Resume from a save, run 10 steps, and report the losses beside the uninterrupted run's, with the largest difference. They can't match bit for bit: Sid's PAI resume gave 3.4642 vs 3.4621 after 10 steps. Keep Sid's reboot watcher, with our paths.

## During training

- **At the end of the warm-up and of every pass:** save (check 5), and report the hide-and-guess loss on the training clips and on the held-out clips.
- **No stop rule:** all 3 passes run. Navvrat picks the pass from the losses.

## At the end

1. Merge the pass Navvrat picks into the encoder as a new checkpoint. Check that its tokens match the unmerged encoder's on a batch, and report the largest difference.
2. Save the merged encoder to a new path, with its sha256. Phase 2 starts from it.

## Rules

- Never train or tune on Bench2Drive's 220 eval routes, the v0.0.4 validation routes, or HUGSIM scenes.
- Read Sid's Gallops code; don't edit it. Copy or import what you need into a new `b2d_drivejepa/phase1/` folder.
- Never overwrite a checkpoint, cache or earlier result; write to new paths.
- Log in `agent_memory/<date>.md`. Ask before downloading or installing anything, and before changing anything in this file.
