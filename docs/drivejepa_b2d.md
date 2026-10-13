# Drive-JEPA on Bench2Drive

The released Drive-JEPA checkpoint (NAVSIM-v2, ViT-L), with parts retrained on
the Bench2Drive base set and driven closed loop through paddock's
`drivejepa_scorer_b2d` adapter. The backbone is never trained.

| Model | Bench2Drive, 50 routes, mean DS | What is retrained |
|---|---:|---|
| Split-v4 | 68.1 (30 Sep 2026, one run per route) | the scorer head |
| Final | 73.6 (6 Oct 2026, one run per route) | the proposal generator, the split head, the area head |

Recipe: [`recipes/drivejepa_b2d.toml`](../recipes/drivejepa_b2d.toml).
`gallops recipe drivejepa_b2d` prints every command for this machine;
`gallops recipe drivejepa_b2d <stage> --run` runs one.

## Split-v4

Drive-JEPA proposes 32 trajectories and drives the one its scorer head
(`scorer.pred_score`, an MLP 256-1024-1024-6, 1.32 M parameters) rates highest
in column 5. Split-v4 keeps the 32 proposals and retrains only that head, so
only the pick changes. Columns 0-3 become four yes/no heads -- collision, lane,
time-to-collision, red light -- and the policy combines them with route
progress to pick (paddock side: `DRIVEJEPA_PICK=split`, exponent selector, k = 4).

Three stages, each initialised from the one before:

| Stage | Trainer | Init | Data | Kept |
|---|---|---|---|---|
| `ipad-base-start` | `train_ipad_head` | released head | base 20,449 + route-start 3,000 rows; column 5 fitted to the iPad rule score | epoch 17 |
| `split-full` | `train_split_head` | ipad-base-start | v3 labels for col / lane / ttc / red | epoch 32, mean AUC 0.940 |
| `split-v4aug` | `train_split_head` | split-full | v4 labels (red = Bench2Drive's stop-line rule) + 8,316 speed-randomised copies of training light frames | epoch 20, mean AUC 0.940 |

Then `split-v4-checkpoint` writes the split-v4aug head into a copy of the
released checkpoint, with `checkpoint.json` beside it.

**Split.** 150 of the 1,000 base clips are held out
([`recipes/drivejepa_b2d/heldout_150.json`](../recipes/drivejepa_b2d/heldout_150.json)):
3,499 held-out frames against 19,950 training frames. Every kept epoch was
chosen on them. The list was recovered on 29 Sep from the feature cache after
the re-ranker directory that first defined it was deleted; the bitwise rerun
below confirms it is the same 150 clips.

## The Final

Three trained parts and one policy switch:

1. **Generator** (`final-generator`). Drive-JEPA's proposal generator --
   `init_feature`, `hist_encoding` and the one shared refiner -- fine-tuned with
   its own multimodal loss (expert winner-takes-all + 0.5 x pseudo-teachers),
   the pseudo-teachers being the MTD-H teacher set (up to 4 per frame, computed
   on Bench2Drive frames only) instead of the scorer's top anchors. From the
   release, Adam 2e-5, batch 32, 10 epochs, seed 0, kept epoch 10 by held-out
   objective. Runs from a cache of backbone features in Drive-JEPA's own Python
   3.9 env, with the batch fix below.
2. **Split head** (`final-split-head`). Split-v4's recipe, initialised from
   split-v4aug, on the new generator's candidates and their m05b labels (0.5 m
   margin on moving actors, expert-masked). Kept epoch 25.
3. **Area head** (`final-area-head`). Drive-JEPA's own `scorer.pred_area`
   (off-road, oncoming per pose) fine-tuned from the release on v6 step labels.
   Kept epoch 40. The policy loads it separately (`DRIVEJEPA_SPLIT_AREA`); it is
   not in the checkpoint.
4. **Progress guard**, policy only: if no candidate that moves passes every
   gate, drop lane, then ttc, then red -- never collision
   (`DRIVEJEPA_SPLIT_GUARD=1`).

`final-checkpoint` writes the generator and split head into a copy of the
release. The Final was originally written over Split-v4's checkpoint; the split
head replaces the same 10 tensors, so the result is the same tensors.

**The batch fix.** Upstream, the refiner mixes batch elements in three places
(`batch_fix.py`): a dump at batch 64 was a median 1.49 m off the same frames at
batch 1. Deployment runs batch 1, so closed-loop numbers are unaffected, but
anything that trains the refiner batched is. The Final's generator was trained
with the fix.

## What the stages read

Named in `gallops.toml`, never in a tracked file:

| Template | What |
|---|---|
| `{python.train}` | Python >= 3.11 with torch |
| `{data.navsim.exp}` | holds the released checkpoint, `Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt` |
| `{data.bench2drive.rerank}` | the candidate tree: `_dataset/` (rows), `_feats/` (the (32, 256) head inputs), `_labels/`; for the Final also `_genft_cache/` (backbone features) and `_mtdh/` (the new generator's candidates, head inputs and labels) |
| `{python.drivejepa}`, `{code.drivejepa}` | Drive-JEPA's interpreter and checkout; the generator stage runs with `navsim_v2/` as its cwd, as paddock's node does |
| `{data.bench2drive.teachers}` | the MTD-H teacher set, one npz per clip |

**Not produced by this repo yet.** The rows, the head-input features and the
labels were made by paddock-side tooling: the candidate and feature dump
(`tools/models/drivejepa/b2d_candidates.py`, which runs the released model
through paddock's own adapter so the features are exactly what the head sees in
closed loop), the row builder (`rerank.py dataset`), the light-frame
augmentation, and the label chain (v3 col / lane / ttc, v4 red). For the Final
also the backbone-feature cache, the MTD stages 1-3 that build the teachers, the
re-dump of the candidates with the new generator, and their m05b / v6 labels.
The stages here read their recorded output. Moving that tooling is a separate
change.

## Verified, and how

| Claim | Strength | How |
|---|---|---|
| The three head stages reproduce the recorded heads | **verified** | 13 Oct: `ipad-base-start`, `split-full` and `split-v4aug` rerun from this code on the recorded data, each initialised from the previous rerun's output; every `head_best.pt` **bitwise identical** to the recorded one (same GPU and torch build) |
| The held-out list is the original one | **verified** | it follows from the above: different clips would give different training data and different weights |
| The deployed Split-v4 checkpoint is the release + this head | **verified** | its 10 head tensors equal `split-v4aug/head_best.pt` and its other 495 equal the release, tensor by tensor |
| `write_checkpoint` writes that checkpoint | **argued** | ported from the writer that produced it, same tensor replacement; not run from this repo |
| The deployed Final checkpoint = release + the generator + the m05b split head | **verified** | 13 Oct: its 10 head tensors equal the m05b head, the generator's 45 tensors equal its 171 (the refiner copied into four heads), and nothing else differs from the release |
| The Final's generator, split head and area head stages reproduce the recorded ones | **argued** | ported with the arithmetic unchanged; none of the three was rerun from this repo |

## Before quoting a number from these checkpoints

- **NAVSIM is not zero-shot.** The release was trained on OpenScene and then
  NAVSIM `navtrain`; `checkpoint.json` declares it.
- **The variant was selected on the reporting routes.** The kept epochs were
  chosen on the held-out clips, but which head and which selection rule to keep
  was decided by closed-loop driving score on the same 50 routes the 68.1 is
  reported on. That is selection on the test set; the sidecar's notes say so.
- **Same-spot twins.** No Bench2Drive eval route id is in the base set, but it
  holds same-scenario, same-location twins of 26 eval routes.
- **One run per route.** Bench2Drive runs are not repeatable here (rain and
  foliage render differently run to run), so a single-run difference of a few
  points is not a result.
