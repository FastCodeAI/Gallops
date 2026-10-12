# Drive-JEPA, retrained on Physical AI AV

The first model trained under this repository that paddock's simulator can
drive. It is not gallops-v0 and shares no code with it: it is the released
[Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) checkpoint, trained
further on NVIDIA's Physical AI AV dataset. The scripts, step-by-step reports
and result tables are under
[experiments/drivejepa_retrain/](../experiments/drivejepa_retrain/).

Drive-JEPA has two parts: a generator that proposes 32 candidate paths, and a
scorer that rates them and picks one. Both were trained together in the first
training run. The scorer did not pass that run's own quality check, so it was
frozen for the second training run, which retrained only the generator. The
shipped checkpoint therefore contains the first run's scorer unchanged.

**Where it got to.** The measurement is a closed-loop test: the model's plan
steers a simulated car, and the next camera frame it sees depends on where that
plan took it. On 30 scenes in the AlpaSim simulator, one drive each, the mean
`scene_score` goes from 0.5433 with the released checkpoint to 0.7697 with the
retrained one. At-fault collisions — collisions the scoring attributes to the
driven car, because it ran into something — go from 4 to 3, and two of those 3
are in scenes the released checkpoint drove without one. Read the caveats at
the end before quoting either number.

<!-- table:results -->
| Adapter | Weights | Mean `scene_score` | At-fault collisions | Any collision | Off-road | Stalls | Mean progress | Mean distance (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1.4b | released checkpoint, momentum-aware selection | 0.5433 | 4 / 30 | 10 / 30 | 4 / 30 | 3 / 30 | 0.5366 | 75.8 |
| 0.1.6 | intermediate checkpoint, not shipped (first training run) | 0.7032 | 2 / 30 | 7 / 30 | 3 / 30 | 4 / 30 | 0.6977 | 115.4 |
| 0.1.7 | retrained checkpoint (second training run, generator only) | 0.7697 | 3 / 30 | 3 / 30 | 2 / 30 | 2 / 30 | 0.8284 | 151.6 |
<!-- /table:results -->

The version numbers name the driver that feeds the model inside the simulator.
0.1.4b is the released checkpoint behind a driver that adds momentum: a step
from the Drive-JEPA paper that favours candidates consistent with the plan
already being driven. 0.1.7 is the same driver with the retrained checkpoint
and the input changes that checkpoint needs. 0.1.6 drove the first training
run's checkpoint and was never shipped; it is shown once so the two runs can be
told apart.

For scale, a constant-velocity baseline — a car that keeps its current speed
and heading, with no model — scores 0.4876 on the same 30 scenes.

Three things this page will not let a reader assume:

- **The scorer was trained once, and it did not pass.** Its check asked that on
  held-out clips it pick the truly best of the 32 candidates at least 50 % of
  the time. It does so 37.9 % of the time (the released scorer: 34.2 %). It was
  not trained again; the shipped file holds it bit for bit
  (`06_first_training_run/REPORT.md`, line 43;
  `12_second_training_run/s10_gen_train.py`, line 65;
  `12_second_training_run/gen_ckpt.json`).
- **The spacing term was switched off.** The second run's code has a loss term
  that penalises unevenly spaced waypoints. It exists in the code and was tried
  at weights 0.1, 1 and 10 in a pilot of one pass (a pass is one sweep through
  the training data), but it was switched off (weight 0) for the shipped
  checkpoint.
- **The committed code has not been re-run.** Machine paths were replaced by
  placeholders before committing and nothing has been executed since. See
  [the experiment README](../experiments/drivejepa_retrain/README.md).

---

## What was trained

| Part | Parameters | First training run | Second training run |
|---|---:|---|---|
| Encoder, V-JEPA 2 ViT-L — turns camera frames into features | 304.1 M | frozen | frozen |
| Generator — proposes 32 candidate paths | 2.38 M | trained, 10 passes over the data | trained, 3 more passes |
| Scorer — rates them and picks one | 4.70 M | trained, 10 passes | **frozen** |

Three changes to what the model is given, all applied in training and all
required again when the model drives:

- **Bonnet mask.** The bottom 37 rows of every 256-row model frame are set to
  black. Raw clips show the recording car's own bonnet there (its highest
  measured edge is 35 rows up; the mask is that plus 2), and the simulator's
  frames show road to the last row. Without the mask, training and test frames
  differ in those rows.
- **Acceleration zeroed.** The two acceleration slots of the model's ego-state
  input are 0.0 on every sample. Zeroed, not removed: the slots are still
  there.
- **A larger vocabulary.** The generator builds its candidates from a fixed
  list of path shapes. That list grows from the released 8,192 to 10,240: 2,048
  shapes built from human driving at 14 m/s and above, because the fastest
  released shape is 14.8 m/s.

The second run also corrects two lines in the upstream model code that mix up
the examples of a batch (`12_second_training_run/train_patch/batch_fix.py`).
With one example at a time the correction changes nothing; training runs 32 at
a time, so the generator was trained with it on and must be run with it on.

## Data and splits

All training data is NVIDIA Physical AI AV. `docs/data.md` says what of it is
held.

| Use | Split | Size | Where it is recorded |
|---|---|---|---|
| Training | NVIDIA `train` only | 126,981 samples from 4,018 clips, 1,398 chunks, 25 countries | `04_training_set/clips.csv` (the clip list); manifest in POINTERS.md |
| Choosing which pass of the first run to keep | NVIDIA `val` | 24,657 examples from 801 clips | `06_first_training_run/REPORT.md` |
| Choosing which pass of the second run to keep | 145 clips of `train`, held out of that run | 4,400 examples | `12_second_training_run/final_choice.json` |
| Excluded from everything | NVIDIA `test` | — | asserted in `04_training_set/s3_common.py` |
| Excluded from everything | the simulator's 913 public scenes, which include the 30 test scenes | — | `01_find_clips/exam_exclusions.txt`; asserted in every step's scripts |
| Reporting | the 30 test scenes | 30 | `07_closed_loop_test/runs_017/tables/` |

Every chunk of the dataset lies in exactly one of NVIDIA's splits (asserted over
all 3,146), so a clip cannot leak across them by sharing a recording session.
`tests/test_drivejepa_retrain.py` re-checks the two exclusions from the
committed lists: no training clip is on the exclusion list, and each of the 30
test scenes is.

**The 30 scenes are not untouched.** No weights were chosen on them. But the
first run's failures on these same scenes are what the diagnosis steps
examined, and that diagnosis shaped the second run. The 0.1.7 number is a
report on scenes that informed the design, not on scenes nobody had looked at.

## The steps, and what each one's own check said

Each step wrote down its pass conditions before running. Most did not pass,
and the work went on by a recorded decision each time. That is the honest
shape of this experiment, so it is the shape of this table.

| Folder | What it did | What its own check said |
|---|---|---|
| `01_find_clips/` | listed the dataset, set aside the evaluation scenes, counted the clips available for training | no pass/fail rule; fixed the exclusion list at 913 scenes |
| `02_build_samples/` | raw clip → model input | **did not pass**: one of three checks fails (84.4 % against at least 90 %) |
| `03_grade_candidates/` | grades candidate paths against a map derived from the clip | **did not pass**: three of four checks fail |
| `04_training_set/` | built the training set and the 10,240 vocabulary | **did not pass**: two of six checks fail (6.2 % of samples dropped against at most 5 %; the mix of clip types off target). Both accepted, in writing |
| `05_small_overfit_check/` | 30 epochs on 1 % of the set | **did not pass**: two of six checks fail; judged under-trained rather than broken, and continued |
| `06_first_training_run/` | generator and scorer, 10 passes | **partly passed**: the generator improves on unseen clips, the scorer does not (two of five checks fail) |
| `07_closed_loop_test/` | 30 simulated scenes, first run's checkpoint | **mixed**: two conditions fail, two hold; neither highway scene and neither scene that starts from a standstill is fixed |
| `08_scorer_diagnosis/`, `09_three_hypotheses/`, `10_three_quick_looks/`, `11_scorer_head_and_pullout_grades/` | why the scorer did not improve; a small replacement scorer head on frozen features | diagnosis; the replacement head fails on unseen clips |
| `12_second_training_run/` | generator only, scorer frozen | no written report; tables below |
| `13_best_possible_pick/` | would the two scenes where the car never moves be driven if the best candidate were always picked | **no**: three of three conditions fail |
| `14_after_retraining_diagnosis/` | why the first run's checkpoint still swerves and stalls | written about the first run's checkpoint, not the shipped one |

## The 30-scene closed-loop test of the retrained checkpoint

**A table, not a report.** No report was written for this test; the one in
`07_closed_loop_test/REPORT.md` covers the first run's checkpoint. These values
are rendered from `07_closed_loop_test/runs_017/tables/totals.json` and
`gate_017.json`.

<!-- table:trial_017 -->
| 0.1.7, 30 scenes | Recorded value |
|---|---|
| Runs complete / checks passed / provenance verified | 30 / 30 / 30 of 30 |
| Renderer | 26.1.112-59673f88 |
| Checkpoint sha256 | `29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916` |
| As-built driver diff sha256 | `dc8bc7693ca649963c25d03210523e4d5ad715bb05e4c4523983656135f24813` |
| Mean `scene_score` | 0.7697 |
| Scenes at full score | 21 of 30 |
| At-fault collisions | 3: acd22430, b44a4a44, bd07fb91 |
| Any collision | 3 |
| Off-road | 2: 791d68ce, d1a25a99 |
| Stalls | 2: 8023dce5, bae463d3 |
| Mean progress | 0.8284 |
| Mean distance travelled (m) | 151.6 |
| Driver vs trainer on the same 16 examples: plans agree within 2 cm | yes |
| Batch fix changes a plan at batch 1 | no |
| Batch fix changes plans at batch 2 | yes |
<!-- /table:trial_017 -->

## The second training run

**Tables, not a report.** `12_second_training_run/` has no written report.
These are rendered from `pilot_choice.json`, `final_choice.json` and
`gen_ckpt.json`.

Five pilot arms, one pass each, all starting from the first run's checkpoint:

<!-- table:second_run_pilot -->
| Pilot arm | Run | Spacing weight (`lam`) | Refiner reference range | Trained | Continued |
|---|---|---:|---|---|---|
| A | `pilot_fix_l0` | 0.0 | standard | yes | **yes** |
| B | `pilot_fix_l0p1` | 0.1 | standard | yes | no |
| C | `pilot_fix_l1` | 1.0 | standard | yes | no |
| D | `pilot_fix_l10` | 10.0 | standard | yes | no |
| E | `pilot_fixwide_l0` | 0.0 | wide | yes | no |

Compared on 4,400 examples from 145 training clips held out of the second run.
<!-- /table:second_run_pilot -->

The arm that was continued was taken two passes further. The rule that chose
among the three resulting checkpoints was written down before the pilot was
read (`12_second_training_run/PREDICTIONS_phase1.md`): among the candidates
whose error against the human's path stays within 5 % of the first run's on
every kind of clip, take the one whose highway plans change speed least from
one waypoint to the next.

<!-- table:second_run_choice -->
| Checkpoint | What it is | Human-path term: highway | close-pass | ordinary | Highway speed swing, car ≥ 20 m/s (m/s) | 95 % interval | Qualifies | Chosen |
|---|---|---:|---:|---:|---:|---|---|---|
| reference | the first run's checkpoint | 1.3111 | 0.4421 | 0.3667 | 16.7260 | — | — | — |
| limit | reference × 1.05 | 1.3767 | 0.4642 | 0.3850 | — | — | — | — |
| pilot | the first run's generator + 1 pilot pass at learning rate 1e-4 | 0.7586 | 0.4370 | 0.3415 | 3.9878 | 3.5815 to 4.3990 | yes | no |
| pilot + 1 pass | the pilot + 1 pass at learning rate 2.5e-5 | 0.5616 | 0.4098 | 0.3246 | 3.1047 | 2.8060 to 3.4560 | yes | no |
| pilot + 2 passes | the pilot + 2 passes at learning rate 2.5e-5 | 0.5453 | 0.4131 | 0.3201 | 2.9433 | 2.7340 to 3.1757 | yes | **yes** |

Compared on 4,400 examples from 145 training clips held out of the second run, in float32.

The two further passes, run `final_P0`: learning rate 2.5e-05, spacing weight (`lam`) 0.0, 2 passes, 4377 steps.
<!-- /table:second_run_choice -->

What the shipped file is made of:

<!-- table:second_run_checkpoint -->
| Merged checkpoint | Recorded value |
|---|---|
| sha256 | `29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916` |
| Bytes | 1273002536 |
| Tensors | 495 |
| Encoder tensors, bit-identical to the released checkpoint | 294 of 294 |
| Scorer tensors, bit-identical to the first run's scorer | 30 of 30 |
| Generator tensors that differ from the first run's generator | 155 of 171 |
| Pass merged | 2 of the two further passes, learning rate 2.5e-05 |
| Spacing weight (`lam`) of that run | 0.0 |
| Released base checkpoint sha256 | `740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05` |
| Merged file vs the trainer's own file, 16 examples: max plan difference | 0.0 |
<!-- /table:second_run_checkpoint -->

After the checkpoint was chosen and merged, it was also evaluated on clips it
had not seen. Those tables and the nine scripts that made them read the
finished checkpoint and played no part in producing or choosing it; they were
written straight to the training site's store and are not in this repository
(POINTERS.md, last row).

## Running the weights

Gallops holds no driver. The retrained checkpoint is run in the simulator by
paddock's Drive-JEPA driver, version 0.1.7:

> paddock, `tools/models/drivejepa/alpasim_driver/` — on the branch
> `drivejepa/retrained-adapter` at the time of writing, with its inputs, how it
> picks a plan and how to launch it in paddock's `docs/models/drivejepa.md`.

That driver applies the bonnet mask, zeroes acceleration and installs the batch
correction unconditionally, and refuses any checkpoint whose size and sha256
are not the ones below. `07_closed_loop_test/adapter_0.1.7_as_built.diff`
records what the test ran; it is not a second copy of the driver.

## Weights, and the sidecar

| | |
|---|---|
| Location | `{weights}/drivejepa_physicalai/second-run-generator/step_000000002/drive_jepa_physicalai_generator.ckpt` |
| Bytes | 1,273,002,536 |
| sha256 | `29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916` |
| Vocabulary | `vocab_10240.npy` beside it, 4,915,328 bytes, sha256 `4b0bc3c9a55c1787cd54d4a27c8bfd7119bbe6ae2cbe1bf1cfe3343da639e785` |

Every other artefact, with size and hash, is in
[POINTERS.md](../experiments/drivejepa_retrain/POINTERS.md).

The `checkpoint.json` written beside the weights, with `gallops.checkpoint`:

```json
{
  "created_utc": "2026-10-12T19:27:17Z",
  "model": "drivejepa_physicalai",
  "notes": {
    "base": "released Drive-JEPA checkpoint, sha256 740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05; trained on navsim navtrain upstream",
    "excluded_from_training": "physicalai_av test; the AlpaSim simulator's 913 public scenes, which include the 30 closed-loop test scenes",
    "inputs_required_at_inference": "bottom 37 rows of each 256-row frame masked; ego_status acceleration slots zeroed; batch fix on",
    "parts": "encoder 294 tensors bit-identical to the base; scorer 30 tensors from the first training run, frozen in the second; generator 171 tensors from the second training run",
    "physicalai_av_is_not_a_paddock_dataset_key": "paddock datasets.yaml has no entry of this name, so the declaration cannot be cross-checked there yet",
    "served_by": "paddock tools/models/drivejepa/alpasim_driver, adapter 0.1.7",
    "sidecar_written": "2026-10-13, after the weights (2026-10-09), not by the function that saved them",
    "spacing_weight": 0.0,
    "step_meaning": "pass 2 of the second training run's two further passes; that run starts from the first run's kept checkpoint plus one pilot pass",
    "train_git_sha_set_after_the_fact": "The training code was not in git when the runs were made (2026-10-07 and 2026-10-09). The sha is the commit on gallops branch train/drivejepa-physicalai-retrain that first holds all of it; experiments/drivejepa_retrain/AS_RUN.sha256 has the hashes of the files as they ran. It stays reachable only if that branch is merged with a merge commit; after a squash merge this field is updated to the merged main sha.",
    "vocabulary": "vocab_10240.npy, sha256 4b0bc3c9a55c1787cd54d4a27c8bfd7119bbe6ae2cbe1bf1cfe3343da639e785",
    "weights_bytes": 1273002536,
    "weights_file": "drive_jepa_physicalai_generator.ckpt (upstream Drive-JEPA state dict, not model.safetensors)"
  },
  "params_m": 311.2,
  "precision": "fp32",
  "selection_split": "physicalai_av val for the first training run; 145 physicalai_av train clips held out of the second training run",
  "step": 2,
  "train_git_sha": "3e8901a42b616bd3c5c171c6fdd003cc0ff42b92",
  "train_run": "second-run-generator",
  "trained_on": [
    {
      "dataset": "physicalai_av",
      "split": "train"
    },
    {
      "dataset": "navsim",
      "split": "navtrain"
    }
  ],
  "version": 1,
  "weights_sha256": "29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916"
}
```

Four things about it that a reader should not have to infer:

- **`train_git_sha` was set after the fact.** The training code was not in any
  git repository when either run was made. The sha recorded is the commit on
  this branch that first holds all of that code; `notes` says so, and
  `experiments/drivejepa_retrain/AS_RUN.sha256` holds the hashes of the files
  as they ran. That commit stays reachable only if this branch is merged with
  a merge commit. If it is squash-merged, the sidecar is updated to the merged
  `main` sha as a follow-up.
- **The sidecar was written days after the weights**, not in the function that
  saved them, which is what `docs/checkpoints.md` asks for.
- **`trained_on` names `physicalai_av`**, and paddock has no `datasets.yaml`
  key of that name yet. Until it does, paddock cannot cross-check this
  declaration against a benchmark's scenes. `navsim` / `navtrain` is listed
  too, because the released checkpoint this started from was trained on it.
- **The weights file is not `model.safetensors`.** It is upstream's `.ckpt`
  layout, because upstream's loader is what reads it.

## The upstream patches

Three Drive-JEPA training files were changed. They are committed as diffs, and
the diffs were checked before committing: applied to upstream at `e21f474`,
they reproduce the files that trained, byte for byte.

| Diff | Applies to | Result's sha256 |
|---|---|---|
| `05_small_overfit_check/train_patch/drive_jepa_agent.diff` | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py` | `14561d5584bd6ce11922cef6fc07dfabf72ef93fac73f5dbaca63195263f0813` |
| `05_small_overfit_check/train_patch/dataset.diff` | `navsim_v2/navsim/planning/training/dataset.py` | `c123e5514d1328dddd0f2b683ad7744a3d3a03d120fbf0655c12e6c74baf6c79` |
| `12_second_training_run/train_patch/drive_jepa_agent_s10.diff` | the result of the first diff | `63bcc78b7c2a8716d3ff2f9092fe9cb9302ccf57f6a7337b8a7f71f2def3b377` |

The spacing term is in the third (`spacing_rows`, `generator_loss_s10`).

## Verified, argued, not verified

| Claim | Evidence | Strength |
|---|---|---|
| The training set is `train` only and holds no evaluation scene | asserted in every step's scripts; re-checked from the committed clip lists by `tests/test_drivejepa_retrain.py` | **strong** |
| The shipped file is released encoder + first-run scorer + second-run generator | `gen_ckpt.json`: 294 of 294, 30 of 30, 155 of 171 tensors, compared bit for bit | **proof** |
| The driver and the trainer compute the same plans from that file | `gate_017.json`: 16 examples, agreement within 2 cm | **strong** |
| The three diffs rebuild the trained files | applied at `e21f474`, compared with `cmp`, at commit time | **proof** |
| The committed files differ from the as-run ones only in paths and folder names | undoing the replacements reproduces all 206 hashes in `AS_RUN.sha256` | **proof** |
| 0.1.7 scores 0.7697 on the 30 scenes, 0.1.4b 0.5433 | `totals.json`; 30 of 30 drives complete, each with its checks passed and its driver version and checkpoint hash verified | **moderate** — one drive per scene |
| The gain from the first run to the second is the generator's | the scorer is identical between the first run's checkpoint and the shipped one, and the score still rose | **argued** — the batch correction changed between them too, and it is one drive per scene |
| The retrained model is safer | at-fault collisions 4 → 3 | **not supported**: two of the three are new, and one drive per scene cannot separate 4 from 3 |

**Not verified:** any repeat or seed sweep; any scene outside the 30; any
benchmark through `paddock run` (no row in paddock's `results/` comes from
these weights); the committed scripts since their paths were replaced; the
effect of the spacing term at a non-zero weight beyond the one-pass pilot.

## What is not in the repository

Weights, the training set, the run folders, tables over 100 KB, and the
after-the-fact evaluation of the second run — all listed, with sizes and hashes
where a single file exists, in
[POINTERS.md](../experiments/drivejepa_retrain/POINTERS.md). An earlier line of
work that tried to re-rank the released model's candidates without retraining
is not here at all.

## Where this came from

This is the only place the working names appear. The experiment was run from a
personal working tree, in folders named by step code, and the reports and
scripts inside each folder still use those codes. `AS_RUN.sha256` lists files
under these original paths.

| Folder here | Original working folder |
|---|---|
| `01_find_clips/` | `Retrain-data/s0_census/` |
| `02_build_samples/` | `Retrain-data/s1_converter/` |
| `03_grade_candidates/` | `Retrain-data/s2_label2/` |
| `04_training_set/` | `Retrain-data/s3_assembly/` |
| `05_small_overfit_check/` | `Retrain-data/s3_5_smoke/` |
| `06_first_training_run/` | `Retrain-data/s4_train/` |
| `07_closed_loop_test/` | `Retrain-data/s5_trial/` |
| `08_scorer_diagnosis/` | `Retrain-data/s6_scorer_diag/` |
| `09_three_hypotheses/` | `Retrain-data/s7_hypotheses/` |
| `10_three_quick_looks/` | `Retrain-data/s8_looks/` |
| `11_scorer_head_and_pullout_grades/` | `Retrain-data/s9_build/` |
| `12_second_training_run/` | `Retrain-data/s10_retrain/` |
| `13_best_possible_pick/` | `Retrain-data/s11_oracle_step0/` |
| `14_after_retraining_diagnosis/` | `Post-retrain diagnosis/` |
| `14_after_retraining_diagnosis/part1_inventory/` | `Post-retrain diagnosis/t01_inventory/` |
| `14_after_retraining_diagnosis/part2_why_it_swerves/` | `Post-retrain diagnosis/t02_swerve_mpc/` |
| `14_after_retraining_diagnosis/part3_swerve_fixes/` | `Post-retrain diagnosis/t03_swerve_fix/` |
| `14_after_retraining_diagnosis/part4_why_it_stays_stopped/` | `Post-retrain diagnosis/t04_stuck/` |
| `14_after_retraining_diagnosis/part5_stuck_fixes/` | `Post-retrain diagnosis/t05_stuck_fix/` |
| `{runs}/drivejepa_retrain/07_closed_loop_test_017/` (run folders, on the store only) | `s5_trial_017/` |

The same replacement was applied inside every committed file, so a script that
names a sibling folder names it as it is here. One file name was replaced the
same way, the retrained checkpoint's, which was renamed on the store; the first
line of `AS_RUN.sha256` records what it was. The three upstream diffs are the
exception: they are byte for byte as they were, so that they still rebuild the
files that trained, and their comments keep the working names. Three working names occur only
inside files: the first run is S4, the second run is S10, and the first run's
kept checkpoint is "pass 9". File names inside the folders were not changed.
