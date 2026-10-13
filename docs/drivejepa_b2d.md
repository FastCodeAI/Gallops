# Drive-JEPA on Bench2Drive

The released Drive-JEPA checkpoint (NAVSIM-v2, ViT-L), with parts retrained on
the Bench2Drive base set and driven closed loop through paddock's
`drivejepa_scorer_b2d` adapter. The backbone is never trained.

**That adapter is not in paddock yet.** On 13 Oct 2026 it existed only as
uncommitted files in one local paddock worktree (`rerank-native`), on no paddock
branch; paddock `main` has `drivejepa` and `drivejepa_free`. Until it lands
there, paddock cannot score a checkpoint this recipe writes.

| Model | Bench2Drive, 50 routes, mean DS | What is retrained |
|---|---:|---|
| Split-v4 | 68.1 (30 Sep 2026, one run per route) | the scorer head |

Recipe: [`recipes/drivejepa_b2d.toml`](../recipes/drivejepa_b2d.toml).
`gallops recipe drivejepa_b2d` prints every command for this machine;
`gallops recipe drivejepa_b2d <stage> --run` runs one.

## Split-v4

Drive-JEPA proposes 32 trajectories and drives the one its scorer head
(`scorer.pred_score`, an MLP 256-1024-1024-6, 1.32 M parameters) rates highest
in column 5. Split-v4 keeps the 32 proposals and retrains only that head, so
only the pick changes. Columns 0-3 become four yes/no heads -- collision, lane,
time-to-collision, red light -- and the policy combines them with route
progress to pick (paddock side: `DRIVEJEPA_PICK=split`, `DRIVEJEPA_SPLIT_MODE=exponent`,
`DRIVEJEPA_SPLIT_K=4`). Those switches are part of the model: under the
adapter's defaults it picks by column 5, whose values moved when the hidden
layers were retrained. `checkpoint.json` records them as `notes.policy_env`.

Three stages, each initialised from the one before:

| Stage | Trainer | Init | Data | Kept |
|---|---|---|---|---|
| `ipad-base-start` | `train_ipad_head` | released head | base 20,449 + route-start 3,000 rows; column 5 fitted to the iPad rule score | epoch 17 |
| `split-full` | `train_split_head` | ipad-base-start | v3 labels for col / lane / ttc / red | epoch 32, mean AUC 0.940 |
| `split-v4aug` | `train_split_head` | split-full | v4 labels (red = Bench2Drive's stop-line rule) + 8,316 speed-randomised copies of training light frames | epoch 20, mean AUC 0.940 |

Then `split-v4-checkpoint` writes the split-v4aug head into a copy of the
released checkpoint, with `checkpoint.json` beside it. The step directory is
built under a temporary name and renamed into place only after the probe passes
and the sidecar is written, so `{weights}` never holds the weights without it;
a train run that already holds a checkpoint is refused. The sidecar's `step` is
the optimizer step of the head's stage up to its kept epoch (Split-v4: 20 × 442
= 8,840), and its notes pin the inputs: the sha256 of
`data_split_v4.json` and the 150 held-out clips with their list's sha256.

**Split.** 150 of the 1,000 base clips are held out
([`recipes/drivejepa_b2d/heldout_150.json`](../recipes/drivejepa_b2d/heldout_150.json)):
3,499 held-out frames against 19,950 training frames. Every kept epoch was
chosen on them. The list was recovered on 29 Sep from the feature cache after
the re-ranker directory that first defined it was deleted; the bitwise rerun
below confirms it is the same 150 clips.

## The data

The stages read one frozen dump, `[datasets.drivejepa_b2d]` in `gallops.toml`:

| In the dump | What |
|---|---|
| `release/` | the released Drive-JEPA checkpoint every stage starts from |
| `rerank/_dataset/` | the rows: base 20,449 and route-start 3,000 frames, with each candidate's ADE and the clip |
| `rerank/_feats/` | the (32, 256) features the scorer head sees, per row (base, start, augmented) |
| `rerank/_labels/` | rule labels per candidate: iPad `final` (v2), col / lane / ttc (v3), red (v4) |

Its file list, sizes and sha256 are tracked in
[`recipes/drivejepa_b2d/data_split_v4.json`](../recipes/drivejepa_b2d/data_split_v4.json);
the `check-data` stage compares a site's copy against it (sizes in seconds,
`--hash` for contents). It runs first when the whole recipe runs;
`gallops recipe drivejepa_b2d <stage> --run` runs only the stages it names, so
name `check-data` too. This machine's copy is `drivejepa_b2d-v1`, made 13 Oct
2026 by copying exactly these files out of the working directories they were
trained from, each hashed as it was copied.

**How the dump was made is not in this repo yet.** The rows, features and labels
come from paddock-side tooling: the candidate and feature dump
(`tools/models/drivejepa/b2d_candidates.py`, which runs the released model
through paddock's own adapter so the features are exactly what the head sees in
closed loop), the row builder (`rerank.py dataset`), the light-frame
augmentation, and the label chain (v3 col / lane / ttc, v4 red). The dump is
their recorded output; moving that tooling here is a separate change.

## Verified, and how

| Claim | Strength | How |
|---|---|---|
| The three head stages reproduce the recorded heads | **verified** | 13 Oct: `ipad-base-start`, `split-full` and `split-v4aug` rerun from this code on the recorded data, each initialised from the previous rerun's output; every `head_best.pt` **bitwise identical** to the recorded one (same GPU and torch build) |
| The dump holds the files they were trained on | **strong** for Split-v4 | the bitwise rerun read the working directories, not the dump; the dump copied the same paths about 30 minutes later (sha256 recorded during the copy), and each file is older than the head trained from it. No stage was rerun from the dump |
| The held-out list is the original one | **verified** | it follows from the above: different clips would give different training data and different weights |
| The deployed Split-v4 checkpoint is the release + this head | **verified** | its 10 head tensors equal `split-v4aug/head_best.pt` and its other 495 equal the release, tensor by tensor |
| `write_checkpoint` writes that checkpoint | **argued** | ported from the writer that produced it, same tensor replacement; never run on the release. Run on a synthetic head-only checkpoint (CPU): it publishes whole, and leaves nothing on a failed probe |

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
