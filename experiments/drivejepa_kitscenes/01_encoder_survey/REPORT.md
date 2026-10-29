# Encoder LoRA on KIT — step 0 recon (read-only)

Date: 2026-10-16. Machine: shakaal. Nothing was trained, installed or downloaded (only README/config text and
Hugging Face metadata). No tracked file was edited. Abbreviations used below:
`DEVKIT` = `{drivejepa_repo}/navsim_v2` (Naren's Drive-JEPA checkout that produced the KIT baseline),
`VEND` = `{paddock}/paddock-sid/Alpasim/adapters/drivejepa/src/navsim_v2` (our vendored copy of upstream Drive-JEPA at `e21f474`, same code),
`ADP` = `Paddock/src/paddock/adapters/models/drivejepa/kitscenes_adapter.py`,
`RUN` = `Paddock/src/paddock/adapters/benchmarks/navsim/run_kitscenes_inference.py`,
`GB` = Gallops branch `origin/train/drivejepa-physicalai-retrain` @ `99ab1b0`, folder `experiments/drivejepa_retrain/`.

## 1. Verdict

**BLOCKED** — the plan assumes the KIT benchmark is KITScenes **LongTail**; the KIT benchmark in Paddock is KITScenes **Multimodal**.

Blockers, one line each:
1. **Wrong dataset.** The Paddock KIT baseline runs on KITScenes Multimodal (10 Hz sensor folders, toolkit `KIT-MRT/KITScenes-Multimodal`), scored by ADE/FDE to 3 s. No LongTail data, code, multi-maneuver score or text instruction exists anywhere on shakaal (§Q1, §Q2, §Q3).
2. **Data rules cannot hold on Multimodal as-is.** The baseline was scored on Multimodal *val* (117 scenes) + *overlap_train_val* (23). Choosing the checkpoint on val would choose it on the report scenes (§Q2b).
3. **LongTail is not downloadable today.** The gated repo returns 403 for account `Siddesh239` (terms not accepted). Train + val is about 177 GB, and only the NAS has room (§Q3c).
4. **Hide-and-guess on the driving input has no time axis.** The encoder sees 2 frames → tubelet of 2 → **one** time slice of 16×32 tokens. Masking can only hide space, not the future. This is a design question, not a code fix (§Q4b, §Q5c).
5. **Upstream pretraining input ≠ driving input** (16 frames at 0.25 s, 256×256, random crops + flips vs 2 frames at 0.5 s, 256×512, fixed warp). A custom data path is required. Otherwise this is a BLOCKER per the rule in Q5c (§3).

Gaps (not blockers): `peft` is not installed in any env; the cost of an encoder backward pass has never been measured; the baseline runner cannot dump the 32 candidates (hook location found); the baseline has no videos, only PNG plots.

Good news: the JEPA **predictor (22.1 M) and momentum target encoder are on disk**, matching our encoder shapes exactly (§Q5a).

## Q1 — KIT Drive-JEPA adapter in Paddock

**a) Location.**
- `ADP` blob `c26d23a1f5eb` is identical on all 53 refs that contain it. Last commit `1486799` (2026-09-01, Narendiran), "M2: adapt the Drive-JEPA node…". First added in `9a80414` the same day. Runner `RUN`: last commit `29a41dd` (2026-09-01, Narendiran).
- **The Paddock copy is not what ran.** `RUN:53-58` does `chdir($NAVSIM_DEVKIT_ROOT)` and imports `scripts.inference.kitscenes_adapter` from there. That is Naren's untracked `DEVKIT/scripts/inference/{kitscenes_adapter,run_kitscenes_inference}.py` (mtime 2026-08-11). Diff against Paddock: typing/formatting only, except the `--kitscenes-repo` default, which is hard-coded in Naren's copy.
- `DEVKIT` is upstream `e21f474` ("release code for navsim_v2", Linhan Wang). This matches the pin at `Paddock/repos.yaml:39-46`. Its local changes are the MSDA attention patch (Paddock `patches/Drive-JEPA`).
- A second, Paddock-native KIT path exists **only on `origin/main`** (`f0b96d7`, PR #54, 2026-10-13, Narendiran): `adapters/benchmarks/kitscenes/{harness,loader}.py`, `adapters/models/drivejepa/policy.py`, `tools/models/drivejepa/paddock_candidates.py`. It is used by the `drivejepa_rerank` row. It is not on the checked-out branch `alpasim-20hz-results`.
- Paddock's `model.yaml` wiring does not resolve on shakaal as-is (`{repo}` → `paddock-sid/Alpasim/Drive-JEPA`, which is missing; `navsim_v2/scripts/inference/` is missing from `VEND`). The baseline was produced from `DEVKIT`.
- LongTail: `git grep -iE 'longtail|long_tail|mms_score|multi.maneuver'` over every ref plus all text under `{workspace}`, the toolkit and NAS returned **no hits**.

**b) Weights — the upstream release, unmodified.**
- `RUN:61-63` loads `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt`.
- Naren's copy: `{naren_navsim_workspace}/exp/Drive-JEPA-cache/…vitl_v2.ckpt`, 3,735,352,913 B, sha256 `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05`.
- Its `.cache/huggingface/download/…ckpt.metadata` records HF commit `9d435e12…` with the same hash as etag (`huggingface-cli download LinhanWang/Drive-JEPA --repo-type dataset`). Checked by me.
- Same hash at `{nas}/naren/weights/drivejepa_released/agent.ckpt` (its `sha256.txt`) and at `{weights}/Drive-JEPA-cache/`.
- `model.yaml:34` says `size_gb: 1.23`; the file is 3.74 GB (doc error only).
- **The pretraining file is also loaded at inference.** `VEND/navsim/agents/drive_jepa_perception_based/bevformer/simple_image_encoder.py:25,36` builds the ViT from `vitl_merge_3dataset_e50.pt` (key `target_encoder`) before the agent ckpt overwrites it.

**c) Input building (baseline path, `ADP` line numbers; Naren's copy is within a few lines).**
1. **Camera.** `CAMERA_MAP` `:58-63` maps `cam_f0` → `camera_ring_front`. `l0/r0/b0` are also loaded (`:324-329`), but the model reads only `camera_feature_1/2`, both built from `cam_f0` (`VEND/…/drive_jepa_features.py:48-57`; `drive_jepa_model.py:53-57`). Front camera only.
2. **Frames.** KIT is 10 Hz. `HISTORY_STRIDE=5`, `NUM_HISTORY_FRAMES=4` (`:47-49`). History = keyframe −15/−10/−5/0 (`:303-305`). Images come from history slots 2 and 3 = **keyframe−5 and keyframe, 0.5 s apart** (`:295, :325-329`), matching the agent `SensorConfig` (`drive_jepa_agent.py:70-71`).
3. **Keyframes** (e2e protocol): `valid_keyframes(num_future_frames=10, stride=90, first_frame=40)` (`RUN:233-236`; `ADP:257-269`), i.e. frames 40, 130, 220, …
4. **Geometry.** Crop about the principal point with `scale = 1545/f` (`:163-180`), bilinear resize to 1920×1080 with OpenScene intrinsics f=1545, c=(960,560) (`:51-54, :183-188, :242-243`); distortion zeroed (`:251`). Then the agent crops rows `[28:-28]` → 1920×1024 and `cv2.resize` → **512×256** (`drive_jepa_features.py:48-57`). Example: KIT ring front 3504×2272, f≈1841, box (596,472,2884,1759) on val scene `00fc1b9b`.
5. **Normalisation.** `ToTensor` → [0,1] (`drive_jepa_features.py:58-59`), then ImageNet mean (0.485,0.456,0.406) / std (0.229,0.224,0.225) (`drive_jepa_model.py:42-47, 55-56`).
6. **Ego status.**
   - One per history frame (`ADP:313-322`). Velocity = central difference of 10 Hz poses; acceleration = difference of velocities; both rotated into the body frame (`:125-155`).
   - The model uses **only the last one**: `ego_status[:, -1]`, 11-dim = pose 3 + vel 2 + acc 2 + command 4 (`drive_jepa_model.py:32, 51, 61`).
   - There is no 4 s / 0.2 s history. The runner docstring admits 1.5 s (`RUN:27-28`).
7. **Driving command — not from text.** `--command gt` is the default (`RUN:172`). `_driving_command` (`ADP:271-289`, checked) takes the **recorded future** lateral offset 4 s ahead: > +5 m → left, < −5 m → right, else straight. This is a future-derived route hint. Baseline mix: 184 straight / 21 left / 18 right.
8. The Paddock-native path differs: native geometry by default, always STRAIGHT, speed from CAN, no acceleration (`origin/main` `policy.py:112-121, 239-297`; `harness.py:329,341`).

**d) Output bridging — none needed, and none exists.**
- The model outputs 8 poses at 0.5 s (`RUN:209-211`). GT is built on the same 0.5 s grid (`ADP:345-353`).
- `displacement_errors` (`RUN:115-132`, checked) truncates to the first 6 poses (3 s): ADE over 0–3 s, FDE at 1/2/3 s.
- The 5 s future that keyframe selection requires is never scored. No 0.2 s / 5 s bridge exists. A LongTail bridge would need resampling to 0.2 s **and** extrapolation past 4 s (UNVERIFIED design).

**e) Dumping 32 candidates + scorer grades.**
- **Baseline path: no.** `RUN:254` → `agent.compute_trajectory` → `abstract_agent.py:91-95` keeps only `predictions["trajectory"]`. The model already returns `proposals` (32×8×3), `pred_logit` (32×6) and `pdm_score` (`drive_jepa_model.py:76-94`; `proposal_num=32` at `drive_jepa_config.py:27`; pick = argmax of sigmoid(logit)[…,−1] at `:85-91`).
- **Hook:** at `RUN:254`, build features with `agent.get_feature_builders()`, call `agent.forward(features)` under `no_grad`, and save the three tensors in the per-sample dict (`:260-273`). The comfort re-score is inactive here (no `past_ego_simulated_states`).
- **Paddock-native path: yes.** `DriveJepaPolicy.forward()` returns all 32 + logits (`origin/main` `policy.py:320-350`). `paddock_candidates.py --bench kitscenes --output … --rows …` writes JSONL with `logits` and `proposals` (`:138-157`). Its inputs differ from the baseline's (above), so its candidates are **not** the 2.364 baseline's candidates.
- **AlpaSim reranker extractor** (`paddock-sid/Alpasim/analysis_3.0/Re-ranker/s1_features`): **not reusable as-is.** `extract_s1.py:38-39,121-140` replays AlpaSim `rollout.asl` logs. `fill_dumps_v2.py:26-31` needs AlpaSim replay helpers. Portable pieces: `proposal_geometry()` (`extract_s1.py:307`) and the 6-logit column layout. The KIT-native equivalent is PR #54's `ranker.candidate_features`.

## Q2 — KIT baseline

**a) Where.**
- Per-sample results: `{naren_navsim_workspace}/exp/inference/kitscenes_e2e_final.pkl`, sha256 `b40a14e967a750c8f116659cb1312e726e660b0171f159db714d702ad40dc3f3` (2026-08-11). 223 rows: scene_id, split, keyframe_index, predicted/human poses (8×3), command, speed, ade, fde@1/2/3s. It is `kitscenes_e2e_full.pkl` (187) + `…_overlap.pkl` (36) concatenated.
- Aggregate: `Paddock/results/kitscenes/drivejepa/legacy-kitscenes_e2e_final/{run.json,metrics.json}` (PR #22, `8a17991`). **ADE 2.3636 mean / 1.392 median, n=223** (checked); FDE@3s 4.199; val 186 samples (ADE 2.369), overlap 37 (2.335).
- This pkl is also the sample list for every Paddock KIT run (`[benchmarks.kitscenes] sample_list`, `origin/main` `harness.py:112-132`).
- Reranker row: `results/kitscenes/drivejepa_rerank/20260921T210621-d7ed08` (ADE 0.973).
- **Videos: none found.** Only PNGs in `…/exp/inference/plots/`. Some searches timed out, so UNVERIFIED that none exist.

**b) Split.**
- **140 scenes / 223 samples** = all 117 of `validation.txt` + all 23 of `overlap_train_val.txt` (toolkit `kitscenes/split/generated_splits/default_geo_split_v1_0/`). The ids are the `scene_id` column of the pkl.
- **0 baseline scenes in `train.txt`** (534 lines), 0 in test/test-e2e.
- ⚠ **FLAG 1 — the baseline split IS Multimodal val.** On Multimodal, "choose on val, report on the baseline split" would use the same 117 scenes for both.
- ⚠ **FLAG 2 — `overlap_train_val`** (23 scenes / 37 samples) is undefined in the toolkit (only listed, `dataset.py:27`). The name suggests geographic overlap with train under a geo split. UNVERIFIED (KIT authors or the paper would settle it).

**c) Metric.**
- ADE 0–3 s and FDE@1/2/3 s on a 0.5 s grid against the single recorded future. **Reimplemented, not vendored** (`RUN:115-132`; Paddock `metrics/trajectory.py:151-171`).
- **No multi-maneuver score, no dataset-card code, no bit-equality test.** The map-based metrics are not implemented (`RUN:30-33`).

**d) Re-run command** (reconstructed from code — no launch log found, so the exact original invocation is UNVERIFIED):
```
source {drivejepa_repo}/exports.sh   # sets NAVSIM_DEVKIT_ROOT / NAVSIM_EXP_ROOT
cd $NAVSIM_DEVKIT_ROOT        # conda env drive-jepa (py3.9); needs a GPU (RUN:180)
python scripts/inference/run_kitscenes_inference.py --kitscenes-root <merged root> \
  --splits val overlap_train_val --protocol e2e --command gt --eval-horizon 3 \
  --output $NAVSIM_EXP_ROOT/inference/kitscenes_e2e_final.pkl
```
- The 140 scenes are split across two roots (NAS 109+18, `{bulk_disk_root}/…/kitscenes_data` 8+5), so this needs a merged root or two passes.
- On sm_120 it needs the torch 2.8 add-on (§Q7a). The baseline itself ran under torch 2.1/cu121 (UNVERIFIED on which card).
- **Determinism:** there is no seed or cudnn flag. Eval-mode forward with no sampling. Four samples re-run in separate earlier runs are **bit-identical** (`c34c778f` kf 40/130 in `kitscenes_gt.pkl` vs `kitscenes_e2e.pkl`; `00fc1b9b`, `02be7d74` kf 40 in `kitscenes_streamtest.pkl`). There has been no full 223-sample repeat.

**e) References on val.**
- Val scene `00fc1b9b` has `poses.txt` (TUM, 10 Hz), calib, 6 ring cameras, lidar, maps. That is enough for ADE/FDE checkpoint selection.
- It has **no multi-maneuver references and no instruction text**, so LongTail-style scoring is impossible from this data.

## Q3 — KIT data on shakaal

**a) Locations — all KITScenes Multimodal, all owned by naren.**

| Location | Splits (scenes) | Size |
|---|---|---|
| `{data.kitscenes}/` | train 532, val 109, overlap 18 | ≈3.16 TB (manifest sum; NFS `du` too slow — UNVERIFIED on disk) |
| `{nas_backup}/FastCode/naren/kitscenes/data/` | same counts (backup) | not measured |
| `{p4}/naren/kitscenes/data/` | val 109, overlap 18 (ring cams + lidar_top + calib + poses only) | 161 G |
| `{data.kitscenes.second_copy}/` | val 8, overlap 5 (fills the NAS gaps exactly) | 43 G |

- Combined: val 117/117, overlap 23/23, train 532/533 (missing `ad32c036-…`), test 0/206, test-e2e 0/127.
- Version: `sequence_archives.csv.metadata` records commit `213352cf…`, the current head of `KIT-MRT/KITScenes-Multimodal` (2026-06-03). Toolkit default repo id: `{kitscenes_toolkit}/kitscenes/download/downloader.py:45`.
- **LongTail (1.0 or 2.0): not present anywhere.** No HF cache entry (`~/.cache/huggingface` has no `hub/`; `{bulk_disk}/hf_cache/hub` and `{nas}/siddesh-physicalai/hf_home/hub` hold other repos).

**b) Format.** Unpacked per-scene folders: JPEG per camera (6 ring + 3 base-front), lidar per-frame parquet, radar, GNSS, `maps/`, `calibration/calib.json`, TUM `poses.txt`, `timestamp.reference.txt`. **10 Hz**: timestamps step 0.100 s; 130 frames ≈ 13 s in train `008fba36`, 100 in val `02be7d74`.

**c) LongTail download** (HF tree API, no download):
- `KIT-MRT/KITScenes-LongTail` @ `18653cf4…` (2026-10-06); the README changelog confirms **v2.0 (Jul 21 2026, adds val + full train)**.
- Sizes: train 50.62 + train_raw 43.07 + train_future 58.11 GB; val 8.48 + val_raw 7.00 + val_future 9.78 GB; test 33.98 + test_raw 28.13 GB; few-shot 0.79 GB.
- **Train+val (all six files) ≈ 177 GB; processed train+val only ≈ 59 GB; whole repo ≈ 240 GB.**
- **Gated** (CC-BY-NC-4.0). The token at `~/.cache/huggingface/token` authenticates as `Siddesh239` but gets **403** on LongTail files. Fix: accept the terms on the dataset page.
- `~/.bashrc:135` sets `HF_HOME={bulk_disk}/siddesh/hf_home`, which does not exist and sits on a 100 %-full disk. The token is not found unless `HF_TOKEN_PATH` or `HF_HOME` is changed.
- Target: `{nas}` (3.9 T free), e.g. `HF_HOME={nas}/siddesh-physicalai/hf_home`. No local disk fits 177 GB (§Q7b).
- LongTail frame rate is UNVERIFIED. The ungated Voxel51 mirror's README says 5 Hz with 21 past frames (~4 s) and 25 future waypoints. Settle by reading `train_few_shot-00001-of-00002.parquet` (72 MB) after accepting the gate.

## Q4 — The encoder inside the checkpoint

**a) Architecture**
- Shape: ViT-L/16, embed 1024, depth 24, 16 heads, MLP ratio 4 (GELU), qkv bias, LayerNorm 1e-6, **3D RoPE (no pos_embed)**. Built by `vit_large` (`VEND/vjepa2/src/models/vision_transformer.py:275-286`) with the kwargs from `vjepa2/configs/eval/vitl/in1k.yaml:153-164`, overridden at `simple_image_encoder.py:30-35` (resolution (256,512), `img_as_video_nframes=2`; checked).
- Patch embed: Conv3d with kernel and stride (2,16,16), weight (1024,3,2,16,16) (`utils/patch_embed.py:26-48`).
- **LoRA targets** (prefix `agent._pad_model._backbone.img_backbone.`):
  - `blocks.N.attn.qkv` (3072×1024)
  - `blocks.N.attn.proj` (1024×1024)
  - `blocks.N.mlp.fc1` (4096×1024)
  - `blocks.N.mlp.fc2` (1024×4096)
  - N = 0..23 (`utils/modules.py:67-81, 261-282, 500-562`).

**b) Tokens and the encoder boundary**
- Frames are stacked as `[older, newer]` → B×2×3×256×512 (`drive_jepa_model.py:57`).
- GridMask is applied in training only (`simple_image_encoder.py:44-45`, prob 0.7). Then a rearrange to B×3×2×256×512 (`:46`).
- Tubelet 2 fuses both frames into **one time slice**: 1×16×32 = **512 tokens × 1024**.
- **Encoder ends at `simple_image_encoder.py:47`**: `img_feat = self.img_backbone(img)` → (B,512,1024) (checked).
- `projector Linear(1024→256)` (`:48`) sits inside `_backbone` and so shares the encoder's learning-rate group.
- The image feature then goes to the generator (`drive_jepa_model.py:60,66`, `Traj_refiner`, run 4×). The scorer takes proposals + BEV feature (`:71`), never the image tokens directly.

**c) Parameter counts** (agent ckpt, 495 keys)

| Part | Parameters |
|---|---|
| Encoder ViT | 303,885,312 |
| Projector | 262,400 |
| Generator | 2,389,827 (the trajectory head is stored 4× but is one shared module) |
| Scorer | 4,704,992 (inference head `pred_score` 1.32 M + two training-only heads) |
| Unique total | 311.2 M |

- The October retrain logs the same encoder+projector count (304,147,712).
- The checkpoint's optimizer state shows the encoder **was fine-tuned** at lr 1e-5: it differs from the pretraining `target_encoder` by 5.2 % relative norm.

## Q5 — Pieces for hide-and-guess

**a) Predictor + momentum target: present locally.**
- File: `{weights}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt`, 5,127,748,765 B, sha256 `4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f` (matches HF LFS of `LinhanWang/Drive-JEPA`).
- It is a full V-JEPA 2 training state:
  - `encoder` (303.9 M)
  - `target_encoder` (EMA copy, ≠ encoder)
  - **`predictor` 22,086,016 params** (12 blocks × 384 dim, 10 mask tokens, RoPE)
  - AdamW state, scaler
  - epoch 51, loss 0.4846, batch 64 × 8 GPUs
- Shapes match our encoder exactly.
- **Caveat:** the predictor was trained against the *pre-fine-tune* target. The driving encoder has since moved 5.2 %, so the predictor likely needs a warm-up (UNVERIFIED; a short predictor-only run would settle it).
- Public fallback with identical shapes: `facebook/vjepa2-vitl-fpc64-256` (`original/model.pth`, sha256 `5346856e…7012`, not downloaded).
- Our retrained checkpoints carry no predictor.

**b) Upstream Drive-JEPA pretraining recipe: NOT RELEASED (UNVERIFIED).**
- No driving pretraining config is in `VEND` or upstream HEAD `548bb82`. The vendored `vjepa2` is an unmodified older Meta revision.
- From the checkpoint: global batch 512, AdamW wd 0.04, lr 5.25e-4 at save, 51 epochs.
- Nearest recipe, generic V-JEPA 2 ViT-L (`VEND/vjepa2/configs/train/vitl16/pretrain-256px-16f.yaml`):
  - Clips: 16 frames at 4 fps (`:20-25`).
  - Crop: 256² (`:18`), random-resized-crop scale 0.3–1.0, aspect 0.75–1.35 (`:32-37`), h-flip (`app/vjepa/train.py:226-227`).
  - Masking: multiblock 3D, 8 blocks at 0.15 spatial scale + 2 at 0.7, full-time tubes (`:41-67`).
  - Loss: L1 (`:39-40`; `train.py:425-435`).
  - EMA: 0.99925 constant (`:87-89`; `train.py:285-288, 456-465`).
  - Optimiser: AdamW, lr 1e-4 → 5.25e-4 warm-up, wd 0.04, bf16 (`:69, 86-98`).
  - Batch: 24/GPU (`:17`).
  - Predictor: 12 × 384 (`:77-85`).

## Q6 — Was the encoder frozen before?

**a) Upstream planning training: trained, not frozen.**
- `VEND/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py:356-364` (checked): Adam with `_backbone` at `0.1 * lr`, everything else at `lr`.
- `lr: 1e-4` (`…/config/common/agent/drive_jepa_perception_based_agent.yaml:17`), so the encoder ran at 1e-5.
- No `requires_grad` changes in `navsim/planning/training/`. Upstream master now exposes `backbone_lr_mult: 0.1` (`drive_jepa_config.py:51`).

**b) October retrain (Gallops): frozen in both runs.**
- First run: `GB/06_first_training_run/s4_train.py:39` (`freeze_encoder=True`). The freeze is in `GB/05_small_overfit_check/train_patch/drive_jepa_agent.diff:108-110` (`requires_grad_(False)`) and `:261-265` (encoder group dropped). `settings_manifest.json:1` records "encoder frozen (304,147,712)".
- Second run: `GB/12_second_training_run/s10_gen_train.py:59,65,68,328,362` (encoder and scorer frozen, encoder hash checked).
- Logs: `paddock-sid/Alpasim/analysis_3.0/Retrain-data/s3_5_smoke/wiring_unit.log:44` (encoder trainable 0, grad max 0.0); `s4_train/run_stdout.log:32,113,159` ("encoder frozen").
- **No gradient has ever flowed through this ViT on shakaal.**

## Q7 — Where training can run

**a) Environment.**
- The retrain used `{envs}/drive-jepa/bin/python` (Python 3.9.23) with `PYTHONPATH=…/conda-envs/dj-torch-cu128-overlay` (6.4 GB), launched by `GB/06_first_training_run/run_s4.sh:2-4`.
- With the add-on: **torch 2.8.0+cu128, arch list includes sm_120, CUDA available**. Without it: torch 2.1/cu121 with no sm_120.
- The mmcv CUDA ops are stubbed, and attention runs on the pure-PyTorch path `DRIVE_JEPA_MSDA=pytorch` (`GB/05_small_overfit_check/train_patch/s35_common.py:32-49`).
- Also installed: timm 1.0.29, pytorch-lightning 2.2.1.
- **`peft`: not installed** in drive-jepa (with or without the add-on), nor in the Paddock venv (which has no torch).

**b) Snapshot** (2026-10-16 00:25 IST):
- GPU: RTX PRO 6000 Blackwell, **63.4 / 97.9 GiB used** by 8 other users' processes, 35 % utilisation.
- RAM: available 78 GB, swap 49/63 GB used, 5-min load 22.7 / 32 CPUs. The memory gate passes.
- Free disk:

| Mount | Free |
|---|---|
| `/` | 75 G (96 %) |
| `{bulk_disk_root}` | 18 G (100 %) |
| `{nas}` | 3.9 T (82 %) |
| `{nas_backup}` | 2.4 T |
| `{p4}` | 54 G |
| `{ssd}` | 7.5 G |

**c) Cost of an encoder forward+backward: UNVERIFIED.** No run ever back-propagated through the encoder.
- Nearest measurements (frozen-encoder forward + head training, batch 32, fp16): 0.55–1.29 s/step, peak ~4.5–9.2 GiB (`GB/06_first_training_run/REPORT.md:211-222,238`; `GB/05_small_overfit_check/REPORT.md:235-236`).
- To settle: a one-off timing in the env above with `freeze_encoder=False`, input (B,2,3,256,512), recording `max_memory_allocated`.

## 3. MATCH / MISMATCH — upstream pretraining input vs driving input (Q5c)

The "upstream" column is the generic V-JEPA 2 recipe, because Drive-JEPA's own driving pretraining config is not released.

| Item | Upstream pretraining (as-is) | KIT driving input | Verdict |
|---|---|---|---|
| Frames | 16 per clip → 8 time slices | 2 → 1 time slice (`simple_image_encoder.py:33`) | **BLOCKER as-is**; with 1 slice, masking is spatial only |
| Frame spacing | 0.25 s (4 fps) | 0.5 s (`ADP:48`) | **BLOCKER**: the tubelet would fuse 0.25 s pairs in training vs 0.5 s while driving |
| Resolution | 256×256 | 256×512 (16×32 grid) | **BLOCKER as-is** (RoPE copes; the transform must produce 256×512) |
| Aspect | 1:1, jitter 0.75–1.35 | 2:1 after squashing 1920×1024 | MISMATCH (fixed with the above) |
| Crop | Random-resized-crop 0.3–1.0 + h-flip | Fixed principal-point warp → `[28:-28]` → resize (`ADP:163-180,241-243`; `drive_jepa_features.py:48-57`); never flipped | **BLOCKER as-is**; flips mirror traffic side |
| Normalisation | ImageNet mean/std | ImageNet mean/std (`drive_jepa_model.py:42-47`) | MATCH |
| Camera | Unknown "3-dataset" driving mix | NAVSIM cam_f0 in fine-tune; KIT ring front warped to OpenScene | UNVERIFIED for pretraining; MATCH by construction if trained through `ADP` |
| Tubelet order | Consecutive frames, chronological | `[older, newer]` (`drive_jepa_model.py:57`) | MATCH on order |

All the blockers go away if training reuses the adapter's own frame selection, warp, crop and normalisation, except the one-time-slice issue (open question 3).

## 4. Open questions for Sid (not decided here)

1. **Which KIT?**
   - (A) Get KITScenes LongTail 2.0: accept the gate, ≈177 GB to NAS, then build a new adapter + 0.2 s / 5 s bridge + multi-maneuver scoring. That means a new baseline too.
   - (B) Use KITScenes Multimodal, which is already on disk and has the 2.364 baseline. This needs open question 2 resolved first.
   - Evidence: §Q1a, §Q2, §Q3.
2. **Data split on Multimodal.** The baseline scenes are val + overlap_train_val. Options:
   - (a) Carve a selection set out of train (532 scenes) and keep val+overlap as the report set.
   - (b) Report on test-e2e (127 scenes, not on disk; a new baseline would be needed).
   - (c) Drop `overlap_train_val` from everything until its meaning is known.
   - Evidence: §Q2b.
3. **Hide-and-guess with one time slice.**
   - (a) Spatial-only masking on 2-frame clips: the same input as driving, but no future prediction.
   - (b) Train on longer clips (e.g. 4–8 frames at 0.5 s → 2–4 slices), which differs from the driving input but restores time masking.
   - (c) A mix of (a) and (b).
   - Evidence: §Q4b, §Q5c.
4. **Starting state.**
   - (a) Online = target = the fine-tuned agent encoder, with the e50 predictor plus a warm-up.
   - (b) Use the e50 encoder/target as-is. This is *not* the encoder that drives, so it is the wrong base for LoRA.
   - Evidence: §Q4c, §Q5a.
5. **Driving command.** The baseline uses a command derived from the recorded future (`--command gt`). Keep it for comparability, or switch both runs to `straight`? Evidence: §Q1c.7.
6. **Which inference path for the drift check?**
   - (a) Add the dump hook to the baseline runner (`RUN:254`): same inputs as 2.364.
   - (b) Use the Paddock-native `paddock_candidates.py`, which already dumps but uses different inputs.
   - Evidence: §Q1e.
7. **Environment.** Install `peft` into the drive-jepa env (or a copy of it), or write LoRA by hand (4 linear layers × 24 blocks)? And get approval for a one-off encoder backward timing run, given the GPU is ~65 % occupied by other users. Evidence: §Q7.
