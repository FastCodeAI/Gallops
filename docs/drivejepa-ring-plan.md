# Plan: Drive-JEPA on KITScenes with surround views, foveated crops and pseudo-label aux heads

**Status: a plan. Nothing in it has been built or run (2026-10-20).** Where a claim below is a
measurement, it says so. Where it is an argument or an estimate, it says that instead.

## Why

This plan tests three ideas from the "Closing the Planner Gaps" notes, which are not tracked:

1. Train Drive-JEPA with all six ring cameras on KITScenes.
2. Add auxiliary outputs: object detection, semantic segmentation and BEV prediction.
3. Crop images so the model sees more context.

All three are a first, measured slice of the **Drive-JEPA-Next design draft** (2026-10-19), which
is also untracked. Each idea maps to one item in that draft:

| Idea | Drive-JEPA-Next item | What this plan runs |
|---|---|---|
| 1. Six cameras on KITScenes | Item 2. Tier-1 views get dense tokens (front, front-left, front-right, rear). Tier-2 views get pooled tokens (the two rear corners). Ray embeddings. View dropout. | A2.1 (front → Tier-1 → Tier-1+2), with A2.3 optional |
| 2. Detection, segmentation and BEV heads | Item 6, pseudo-label group, plus the A6.3 probe readout | A6, trained on camera pseudo-labels only |
| 3. Crops for context | Item 8, lever 2: a foveated dual stream. Lever 3 / A8.3: high-res → low-res consistency. | Full-FOV context views plus a front focus view, then a consistency loss |

**Decisions already taken**
- **Views:** a full-FOV context view plus a focus view, and a crop-consistency loss.
- **Aux labels:** camera pseudo-labels from off-the-shelf models only. No Lanelet2 labels and no
  lidar.
- **Model code:** a git copy of Drive-JEPA with its history, at `external_code/Drive-JEPA-ext`.
  It will be pushed to its own remote.
- **Compute:** assume one shared GPU. That means a fixed VRAM budget and an optional feature
  cache.
- **Evaluation:** Paddock's regression gate.

**The regression gate** is `tools/gate.py` on Paddock's unmerged branch `tools/regression-gate`;
Paddock's `docs/gate.md` describes it.
- It pairs runs per benchmark, with standard errors clustered by scene, log or scenario pair.
  Each benchmark gets a verdict of GAIN, NEUTRAL or REGRESSION, and the whole run gets SHIP, HOLD
  or FAIL.
- Its smoke tier covers:
  - KITScenes: all 223 keyframes, over 140 val and overlap scenes;
  - nuScenes: 30 scenes;
  - NAVSIM: 1,000 tokens;
  - neuroNCAP: 6 scenario pairs × 3 runs.
- Both sides are named as `model:run_id`, so a run from a new adapter can be paired with the
  released `drivejepa` runs.

**What is in scope.** This is Stage 1 of the Drive-JEPA-Next design:
- The encoder stays frozen, and there is no domain-adaptive pretraining.
- Training uses KITScenes only, so arm A1 below plays the role of the design's baseline B1.
- The output stays at Drive-JEPA's 4 s at 2 Hz, not the design's unified 5 s at 10 Hz. Every
  benchmark in the gate scores waypoints 0.5 s apart.

**What a reader gets at the end:** a chain of ablation checkpoints. Each is attributable to one
idea, each carries an honest sidecar, and each has a gate verdict against both its parent arm and
the released model.

## What the code and data say

Everything here was read in the code or on disk, unless it is marked *argued*.

**The model reads one camera.**
- It reads only `cam_f0`, as 2 frames: `[28:-28]`, then 512×256 (`drive_jepa_features.py:38-65`).
- `forward()` slices `lidar2img[:, 1:2]` (`drive_jepa_model.py:50`).
- `num_cams = 1` is hard-coded in `simple_image_encoder.py:16` and `bev_refiner.py:36`, and so is
  the token grid `H=16, W=32`.
- The projector `Linear(1024→256)` is a separate layer from the ViT.

**SCA works for any number of views**, and so does the copy in Gallops' `batch_fix`. Two limits:
- All views in one SCA call must share one token grid.
- The `img_shape` divisor is taken from one image for all of them (`encoder.py:141`).

**Upstream has a small vertical projection error.** It divides v by the padded height (1120),
while the image the model is fed corresponds to `(v−28)/1024`. The error is about ±1 token row.
Paddock's adapter and Gallops' Physical AI converter both copy this convention. This plan keeps
it for the legacy view, so that view stays exact, and builds exact matrices for every new view.

***Argued:* the planner cannot see the rear cameras.**
- Its queries are 32 proposals × 8 poses, with reference points on the forward path footprint.
  Those points do not land in the rear or rear-corner views. Diagnostic D-vis below measures
  this.
- So the design's statement that "Tier-1 is read directly by the planner" does not hold for the
  rear view in today's planner.
- Until the planner is reworked (design item E1), a scene-fusion stage on a coarse BEV grid is
  what carries the rear and Tier-2 views into the plan.

**The BEV heads exist but are switched off.**
- `bev_map` and `bev_agent` are switches in `drive_jepa_config.py:23-24`.
- Neither has its targets built, and neither has weights in the release.
- `MapHead` covers only the front half (7×128×256 at 0.25 m), decoded from trajectory features.

**Gallops' `freeze_encoder` patch freezes the projector as well as the ViT.**

**KITScenes as held on this site:**
- **All 533** train scenes are present; `data.md` says 373, which is out of date.
- Six ring cameras at 3504×2272, 10 Hz, pinhole, f ≈ 1841. Horizontal FOV is about 87°. Yaws are
  0, ±60, ±120 and 180.
- No 3D boxes. No lidar: it was deleted at download.
- The ego body covers the image below a per-camera row, `ROOF_ROWS`. That table was measured in
  the qwendrive_inference KITScenes study.
- A downscaled ring cache (long side 876) holds every 10 Hz frame. This was checked on one scene.
  The code that built it is not in Gallops.

**Contamination.** The gate's KITScenes rows are all 117 `val` scenes and the 23
`overlap_train_val` scenes. Train only on `train`.

**Inputs the gate uses on KITScenes:**
- The driving command is always STRAIGHT.
- Speed comes from the poses.
- The `drivejepa` baseline runs with geometry `native`.

## Approach

### 1. `Drive-JEPA-ext` (model code)

Start from `git clone` of the Drive-JEPA checkout, then branch. Every addition is a switch that
is off by default (design principle P4). The commits, in order:

1. **Paddock's patches 0001–0003**, applied with `git am`. The result must be byte-identical to
   Paddock's patched checkout.
2. **A new package, `navsim/agents/drive_jepa_ext/`.** The upstream agent stays untouched, so the
   bit-exact tests can load both side by side.
3. **Batch-safe attention.** Port `batch_fix` fixes 1–3 into the source. Use a divisor per view,
   and sample points per view group.
4. **Grouped multi-view encoder.**
   - Fold views into the batch, and take the token grid from the input size.
   - The legacy (focus) view keeps its own ViT call.
   - A trainable `projector_ctx` handles the new views.
   - Add `encode_tokens()` for the cache.
   - GridMask is off.
5. **Rig-agnostic view identity** (design principle P1).
   - Each patch gets a Plücker ray embedding from K and the extrinsics, plus an azimuth-sector
     slot embedding. Both go through a projection initialised to zero.
   - A learned per-camera embedding is the alternative tested in optional arm A2.3.
6. **Gated, grouped SCA.**
   - `out = SCA_legacy + g_pass · SCA_new`. Each group has its own count and spatial shape, so
     Tier-1 (16×32) and pooled Tier-2 (8×16) can be used together.
   - `g` starts at 0. At batch size 1 this reproduces the released model exactly.
   - Taps at refinement pass 0 feed the consistency loss.
7. **Scene fusion.**
   - A coarse BEV grid over ±32 m, built with `Bev_refiner(proposal_query=False)`. That mode is
     already supported (`encoder.py:255-271`). The grid reads every view.
   - Trajectory queries read the grid through a cross-attention whose output projection starts
     at zero.
   - The 360° BEV aux heads attach to this grid.
8. **View dropout** (design A2.5). Drop 0–2 views per sample, Tier-2 views more often.
9. **`views.py`.**
   - **Focus view:** Paddock's `openscene` crop. This is the released model's input (64°), with
     the legacy `lidar2img`.
   - **Context views:** the full ~87° 2:1 window above the ego body, with an exact `lidar2img`.
   - **Optional A8.2:** the design's centre-third horizon band at 3× density.
   - `ROOF_ROWS` is ported with a citation and re-checked by overlay.
10. **Aux heads, for training only** (they are not in the inference graph).
    - Per-view 2D detection (CenterNet-style) and 2D segmentation, both on `projector_ctx`
      tokens.
    - BEV semantic and BEV agent heads on the fusion grid. The agent loss uses x and y only.
    - Optional: the design's default label-free heads — ego odometry, and predicting the frozen
      encoder's features at t + 1 s.
11. **A `drive_jepa_ext` config dict saved inside the weights.** It records:
    - the views, tiers and geometry;
    - the `ROOF_ROWS` digest;
    - the sha of the extension;
    - a 2-sample probe that the adapter replays at load.
12. **Tests.**
    - The legacy path is bit-exact against upstream at batch size 1.
    - Batch size 16 gives the same result as batch size 1.
    - Projections round-trip for every view.
    - Checkpoints round-trip.

### 2. Gallops: `gallops.drivejepa_ring` and `recipes/drivejepa_ring.toml`

**Branches.** The view and crop work goes on `train/drivejepa-6cam`. The aux work goes on
`train/drivejepa-auxilliary`, stacked on top of it.

**Modules.**
- Python 3.9 modules (add them to `PY39` in `tests/test_recipe.py`): `samples`, `ipm`,
  `teachers`, `train`, `preflight`, `cache`, `probes`, plus `train/losses.py`.
- Modules that run in the train environment: `pseudo_labels`, `write_checkpoint`.

**Reused from existing code.**
- `drivejepa_b2d/sidecar.publish`, `input_notes` and `optimizer_step`
- `host.require_free_ram`
- `provenance.git_sha`
- `train/preflight.code_checks` and `PreflightFailed`
- `checkpoint.staged` and `write`
- the loss-parity check in `drivejepa_b2d/train_generator.py:317-339`
- `data/rerank.keyframes()`, given a `stride` parameter whose default stays 10

**Samples.**
- All 533 train scenes at 2 Hz: about 15.7k keyframes, each with 1.5 s of history and 4 s of
  future.
- Hold out 15% of the scenes, chosen with seed 0 and committed as `heldout_scenes.json`.
  Checkpoints and arms are selected on this split and on nothing else.
- The training scenes are asserted to share none with `val`, `overlap_train_val` or the gate.
- Ego status is built exactly as the gate builds it, with acceleration zeroed.

**Planner recipe, identical in every arm.**
- **Generator:** `trajectory_loss_anchors`, with the expert trained winner-takes-all. Its teachers
  are:
  - the 2 vocabulary entries nearest the expert by ADE;
  - the released model's top 2.
- **Scorer:** only the final-layer row of `pred_score` column 5 trains. Its target is a soft ADE
  score, the same label the KITScenes re-ranker used.
- **Frozen:** the other score columns, the collision and area heads, and `hist_encoding`.
- **Regularisation:** L2-SP toward the release, which also serves as the design's feature anchor.

**Compute.**
- The ViT-L is frozen and runs online in bf16 under no_grad, reading the 876-px cache.
- *Argued estimate:* about 2.3 TFLOP per sample with 7 views, and 5–7 GB of VRAM at batch 16.
- The trainer refuses to start without 16 GB free. Do not run it next to a gate smoke on the
  same card.
- An optional token cache (fp16, 1024-d, about 117 GiB, on the NAS) comes with an online-parity
  check.

**Pseudo-labels.**
- **`pseudo-label` stage**, in its own `[python] label` environment. It runs a downloaded 2D
  detector and a downloaded semantic segmenter on all six ring cameras at every keyframe.
  - *Open item:* the teachers' licences must allow a commercial checkpoint. No AGPL, and weights
    trained on Cityscapes or Vistas carry non-commercial terms.
  - Each teacher's name, version and sha are recorded in the sidecar.
- **`lift-bev` stage.** Flat-ground IPM turns road and lane pixels into BEV semantic labels, and
  box bottom-centres into agent x and y.
  - Labels are capped at about 25 m, because IPM range error grows as d²·δθ/h.
  - Ignore masks cover whatever the cameras or the teachers cannot supply.
- **`probes`** (design A6.3). Linear probes of the pseudo-labels on the frozen features, run for
  every arm.

**Preflight.** It runs before each arm's first step and has no skip flag. Each check has the form
"what the trainer computes equals what was recorded".
1. **code:** `code_checks`.
2. **inputs:**
   - The focus view's `lidar2img` and `img_shape`, the ego vector and the ground-truth future all
     match a Paddock adapter dump exactly.
   - Cached context pixels match a full-resolution decode, within a measured tolerance.
3. **fidelity:**
   - At initialisation (`g = 0`), the plans match `drivejepa` in openscene mode, in fp32, within
     the thresholds of `train/preflight.py`.
   - Batch size 16 gives the same plans as batch size 1.
4. **sight at initialisation:**
   - Blanking every context view leaves the plans identical.
   - Blanking the focus view changes them.
5. **D-vis** (recorded, not a pass/fail check): the SCA hit rate for each view, on both the
   trajectory path and the fusion path.
6. **labels:**
   - Warping a box or mask into a view gives the same result as warping the image.
   - The expert's 4 s future lies on the IPM road for at least a share of waypoints fixed before
     the run.
   - The pseudo-labels agree with Seamseg on the front and rear cameras. Seamseg's frame-index
     offset has to be resolved first.
7. **parity:** loss parity on the first batch, and cache parity.

**Ablation arms.**
- Each arm is one `--train-run` and is gated against its parent arm.
- Each arm's parent and the decision rule are pre-registered in a hashed `PREDICTIONS.md`, as in
  `experiments/drivejepa_retrain`.
- **Seeds** (design §12.3):
  - screening runs use 1 seed;
  - A1 runs 3 seeds once, to measure the seed-to-seed spread;
  - any arm proposed for adoption is rerun with 3 seeds.

| arm | design item | adds over its parent |
|---|---|---|
| A0 | — | Nothing, and no training. The release through the new code and adapter, to test the plumbing. |
| A1 | B1, KITScenes only | The focus view, fine-tuned with the planner recipe. This is the control. |
| A2 | A2.1 Tier-1 + item 8, lever 2 | Full-FOV context views for the front, front-left, front-right and rear cameras, plus ray embeddings and the fusion grid. |
| A3 | A2.1 Tier-1+2, A2.5 | The rear-left and rear-right cameras as pooled Tier-2 views, plus view dropout. |
| A4 | A8.3-style | Crop consistency, as a stop-grad cosine loss over two pairs: the focus view against the front context view, token-aligned at pass 0 with no extra ViT pass; and random re-crops of the context views, which cost one extra no-grad ViT pass. |
| A5 | A6, pseudo-label group | Per-view 2D detection and segmentation heads, and BEV semantic and agent heads on the fusion grid. |
| optional | A2.3 · A8.2 · A6 label-free · A7.1 | Ray embedding against a per-camera embedding · a centre-third focus view · odometry and future-latent heads · training the last k ViT blocks, or LoRA. |

The pseudo-labelling runs in parallel with arms A1–A4.

**Checkpoint.** `write-checkpoint` writes the weights and the sidecar together.
- `trained_on`: `kitscenes/train` with its fraction, and `navsim/navtrain` inherited from the
  release.
- `selection_split`: the held-out train scenes.
- `notes`:
  - the views, tiers and geometry, and the `Drive-JEPA-ext` sha;
  - the teacher shas;
  - the preflight report, D-vis and the probe scores;
  - the parent arm and the seed;
  - the measured latency at batch size 1 and the token count per step (design §12.3,
    [STRATEGY.md](STRATEGY.md) §5).

**Docs.**
- A results page, `drivejepa-ring.md`, with a verified / argued / not-verified table.
- Update [data.md](data.md) to say 533 scenes.
- Add `[code.drivejepa_ext]` to `gallops.toml.example`.

### 3. Paddock (separate PRs; Gallops never imports Paddock)

- Build on `tools/regression-gate`.
- Add `Drive-JEPA-ext` to `repos.yaml`, pinned by sha.
- **A new adapter, `adapters/models/drivejepa_ext/`:**
  - Six cameras. Tier-1 reuses the existing f0/l0/r0/b0 mapping.
  - Tier-2 maps to NAVSIM's l2/r2, with the yaws checked.
  - The focus view goes through the `openscene` path.
  - The adapter builds the model from the config stored in the weights, refuses an unknown sha,
    and replays the probe at load.
- **`tools/models/drivejepa_ext/dump_inputs.py`**, which produces the recorded side of the
  preflight.
- **Baselines.** Smoke the released `drivejepa` on every benchmark. Also smoke it on KITScenes in
  openscene mode, as the reference A0 must match.
- Confirm that the neuroNCAP renderer serves all six cameras.

### 4. Evaluation with the gate

For each arm:

```bash
python -m tools.gate smoke --model drivejepa_ext --env DRIVEJEPA_CHECKPOINT=<ckpt> \
    --baseline baselines.toml --execute
```

Then run `check` with two pairs per benchmark:
- against the **parent arm**, for attribution;
- against the **released `drivejepa`**, for SHIP, HOLD or FAIL.

The design's acceptance rule applies on top: no REGRESSION verdict, and the arm is no worse than
seed noise on every benchmark.

**How to read the verdicts:**
- KITScenes is open-loop (ADE, plus lateral and longitudinal error), so treat it as a tripwire.
- SHIP needs a closed-loop GAIN, so expect HOLD until neuroNCAP moves.
- nuScenes and NAVSIM are rigs the model never trained on, which is the nearest available stand-in
  for the design's leave-one-dataset-out test.
- Report the 23 overlap scenes separately.
- Run the full tier only for the one or two arms that would ship.

## Verification

- **`make check`**, including the portability test. Check `$?`. New tests cover:
  - view geometry, crop windows, and ray embeddings on synthetic cameras;
  - the keyframe stride, with its default unchanged;
  - split disjointness, from the committed lists;
  - IPM and the label warps;
  - sidecar and publish, on synthetic files;
  - `test_recipe.py` run over the new recipe, plus the `PY39` list.
- **Drive-JEPA-ext tests**, in the Drive-JEPA environment:
  - the legacy path is bit-exact;
  - results do not depend on batch size;
  - projections and checkpoints round-trip.
- **A0 end to end.** `tools.gate compare` of `drivejepa_ext:A0` against `drivejepa` in openscene
  mode on KITScenes must show a difference of 0, and the per-sample plan diff must be 0.
- **Every arm passes preflight**, and its sidecar is checked against its weights:
  - a parameter diff proves which parts were frozen;
  - the dtypes prove the precision.
- **Sight per view after training.** Blanking each ring view must move the plan on turning
  keyframes chosen before the run. If the rear and Tier-2 views never move it, A2 and A3 have not
  shown what they claim.

## Risks and open items

1. **NAVSIM forgetting.** Training on KITScenes only can erode NAVSIM performance. L2-SP and
   freezing are the mitigation. The gate's NAVSIM pair detects it.
2. **A column 5 trained on ADE ignores safety.** neuroNCAP may regress, and the gate would then
   FAIL.
3. **Weak pseudo-labels with a frozen ViT may not move the planner at all.** The probes show
   whether they do. A7.1 is the fallback.
4. **Train/serve geometry drift.** The sources are:
   - the legacy projection quirk;
   - the 1.461 m offset between the pod origin and the rear axle;
   - the constant 2.2 m pod height;
   - the 876-px cache against full resolution.

   Preflight check 2 and the probe cover them.
5. **About 15.7k samples, and they are correlated.** The scene hold-out and L2-SP guard against
   overfitting.
6. **Licensing.** KITScenes is CC BY-NC, and the teacher licences are still open. Both matter for
   a commercial checkpoint (design open question 4).
7. **The encoder costs about 6× more** (the design's estimate). Each arm's sidecar records its
   latency, which answers design open question 1.
