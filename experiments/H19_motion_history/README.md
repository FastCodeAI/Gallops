# H19 — The car's own motion history

**Owner:** Siddesh · **Started:** 2026-11-17 · **Status:** parked, verdict negative

The facts — runs, data, checkpoints, results — live in [experiment.toml](experiment.toml) and are rendered
into the repository README. This page is the account: why, what was done, what it showed, and what was
decided.

## Why

Drive-JEPA plans each moment from two camera frames 0.5 s apart and the car's current state. Three memory
experiments ([H05](../H05_ssm_memory/README.md), H15 and [H17](../H17_generator_scorer_memory/README.md))
found that the camera's past adds little. In H05 and H17 the stronger signal was the car's own motion over
the last few seconds, and H17 named it as a separate, later experiment. This is that experiment: it gives
that motion to the model and asks whether it drives better.

## What was done

The starting model is Drive-JEPA 0.1.8b (pass 0.6), an earlier retrain on NVIDIA's Physical AI AV data,
identified by its sha256 ([POINTERS.md](POINTERS.md)).

- The car's last 4 s are summarised as 8 rows, one every 0.5 s: position, heading, speed, turn rate, and a
  flag for rows that are missing.
- The rows are built from poses only. The simulator reports zero speed for the first 3 s of every drive
  while the car is in fact moving, so its reported speed cannot be used
  ([01_what_the_car_state_holds](01_what_the_car_state_holds/REPORT.md)).
- A small encoder turns these rows into 256 numbers. They enter the model through two links that start at
  zero, so the untrained model is exactly the starting model (checked on 1,131 moments: every pick the
  same, the scores differ by 0.0). One link goes into the generator's plan starting points, the other into
  the first layer of the scorer's rating head.
- Two copies were trained the same way for 5,580 steps from the same start, on 126,981 moments:
  - *history shown* (`H19.r1`): hidden at random half the time in training, always shown in the car;
  - *history hidden* (`H19.r2`): always hidden, which measures what the retraining itself does.
- Each copy's checkpoint was chosen offline by the starting model's own rule, on the picking set.
- Both were driven once on each of 30 AlpaSim scenes. Before that, the rebuilt simulator was shown to
  reproduce the starting model's earlier drives bit for bit on 8 scenes, and the new driver to reproduce
  it at step 0 on one scene.

## What it showed

The rule written before training
([02_…/PREDICTIONS.md](02_baselines_and_the_rule_written_beforehand/PREDICTIONS.md), section 2): history
passes if its mean score over the 30 scenes is at least 0.8194 (the starting model's 0.7527 plus two
scenes) and above the hidden copy's.

| | mean, 30 scenes | 12 diagnosis scenes | other 18 |
|---|---:|---:|---:|
| starting model | 0.7527 | 0.5919 | 0.8599 |
| history shown | **0.7771** | 0.6093 | 0.8889 |
| history hidden | 0.8431 | 0.7260 | 0.9212 |

- **Both conditions fail, so the rule says FAIL** (moderate: one seed, one checkpoint per copy, and one
  scene moves the mean by 0.033).
- History won one scene, lost one to a collision, and stalled on a third that the starting model
  completes (d2715023: 1.2 m travelled against the human driver's 44.4 m). Offline, its pull-away rate
  had passed the floor set beforehand (0.755 against 0.66), so that check did not predict the stall.
- The hidden copy fixed two scenes the starting model crashed on and lost none outright (8023dce5, a
  collision, and acd22430, where the starting model left the road). It has no rule of its own, so that
  gain is an observation, not a result this experiment set out to test.
- Training was not repeatable step for step until PyTorch's `grid_sample` backward, which uses atomic
  adds, was replaced by a deterministic one (`05_training_two_arms/det_grid_sample.py`). The starting
  model's own training ran on the same non-repeatable path; that is argued from the code, not re-measured.

## Decision

Parked, verdict negative. The checkpoints never went to paddock's regression gate, because neither was a
candidate to ship: the rule written beforehand had already said FAIL. No decision row: nothing here
changes what the team does next. The hidden copy's gain (retraining on cached tokens without GridMask, the
rule file's amendment 1) is a possible new hypothesis for the team to decide on. `det_grid_sample.py` is a
candidate for a separate enabler PR under `src/gallops/`; it needs torch tests that CI cannot run.

## Limits

- One seed, one checkpoint per copy, 30 scenes.
- The two copies' checkpoints were chosen on the picking set, and the offline numbers are read on that
  same set (`selected-on-report`). The closed-loop scenes chose nothing.
- The rule was written and hashed before training (`AS_RUN.sha256`: `f8f4b975…a450`; first training step
  18 Nov 03:09), but in a folder outside git, so no commit date shows it and the record says
  `predictions = "none: ..."`.
- For 4 of the 8 image-check drives, reused from H17's check, the image they ran on is inferred from
  timing, not logged.
- On one drive (history shown, d1a25a99) the launcher's check of the live simulator configuration did not
  run: the simulator wrote its configuration after the launcher's 600 s wait. The check before launch
  passed, and that configuration differs from the same copy's bae463d3 drive only in the scene set's hash
  (compared on 21 Nov). Its `config_diff_launch.json` is therefore missing from `drives/`.
- The history-hidden copy was stopped twice (a machine reboot, then a GPU out-of-memory error caused by
  other jobs) and resumed from its saves; the resume test shows a resumed run equals a straight one.
- The torch code was not re-run for this PR; the GPU checks are the ones recorded on 18 Nov.

## These scripts do not run from this repository

They are a record of what ran. They import the earlier Physical AI AV training code, which is not here:
`pai_common`, `pai_model`, `pai_model_b`, `pai_model_c`, `pai_data`, `pai_train`, `train_018b`, H17's
`s6_env` and `s6_model`, and the patched agent `drive_jepa_agent_s10`. When they ran, all scripts sat in
one folder, `src/`, and imported each other by name.

The simulator driver in `paddock/drivejepa_a4/` is a copy of the starting model's driver with the
history added. It is not a paddock adapter: it loads `ego_history.py` and `model_a4.py` by absolute path
(after a sha256 check), and its symlinked files (`__init__.py`, `real_model.py`, `fake_model.py` and the
vendored Drive-JEPA source) are the starting model's driver's and are not copied. The build-freeze hashes
in `paddock/drivejepa_a4/launch/tables/` pin every file it loaded. `paddock/ADAPTER_DIFF.diff` is its
change against the starting model's driver.

## Reading a path in these files

Every machine path was replaced by a placeholder before committing. Nothing else in a file was changed.
[AS_RUN.sha256](AS_RUN.sha256) holds the sha256 of each changed file as it was, against its original path;
it is a record of the originals, so `sha256sum -c` does not pass against the committed files.

| Placeholder | Stands for |
|---|---|
| `{approach4_store}` | the training site's bulk data disk folder for this experiment: token copy, run folders, drives, image tars |
| `{staging}` | a folder on the local disk holding the history side files (the network store was read-only when they were built) |
| `{retrain_store}` | the store of the earlier retrain that produced the starting model, and its closed-loop drives |
| `{experiment_store_h17}`, `{h17_bulk}` | H17's store (its token cache and drives) and its bulk disk folder |
| `{dataset_store}` | the store holding the dataset download and the earlier retrain's training set |
| `{alpasim_store}`, `{alpasim_bulk}` | simulator results on the network store, and the bulk disk holding the environments |
| `{paddock_sid}` | a personal working tree inside a paddock checkout, where the driver lives |
| `{working_folder}`, `{working_notes}` | the folder these files were written in (`reports/`, `src/`, `out/`, `logs/`), and the personal notes folder above it |
| `{nas}`, `{nas_store}` | the network store's mount, and a personal folder on it |
| `{other_user_folder_1}`, `{other_user_folder_2}` | two other users' folders, named only to say they were not touched |

Only `{weights}`, in the checkpoints' template locations, is a variable `gallops.config` expands. The rest
are labels: they name where things sit at the training site, and none of those folders is under that
site's `[roots.runs]`.

## Where this came from

The folders and reports were renamed so that a reader does not need the working notes. The reports and
scripts are committed as written and still cite each other by their working names: `MH/` or `MH/reports/`
is this folder, `src/<script>` a script in its step folder, `out/<table>` a table in its step folder's
`tables/`, and `logs/<log>` a log listed in POINTERS.md.

| Here | Working name |
|---|---|
| `01_what_the_car_state_holds/REPORT.md` | `reports/READONLY_CARSTATE_HISTORY.md` |
| `02_baselines_and_the_rule_written_beforehand/PREDICTIONS.md` | `PREREG_APPROACH4.md` |
| `02_baselines_and_the_rule_written_beforehand/REPORT_baselines.md` | `reports/BASELINES_FOR_PREREG.md` |
| `03_history_data/REPORT.md` | `reports/BUILD1_HISTORY_DATA.md` |
| `04_model_and_trainer/REPORT.md` | `reports/BUILD2_MODEL.md` |
| `05_training_two_arms/REPORT.md` | `reports/TRAIN_ARMS.md` |
| `06_closed_loop_30_scenes/REPORT.md` | `reports/CL_ARM_A.md` |
| `06_closed_loop_30_scenes/drives/history_shown/`, `history_hidden/` | drive folders `a4_armA_<scene>`, `a4_armB_<scene>` |
| `06_closed_loop_30_scenes/drives/starting_model/` | drive folders `drivejepa_018bp06_<scene>` |
| `06_closed_loop_30_scenes/drives/wiring_proof_step0/` | drive folder `a4_step0A_e3ed554f` |
| `06_closed_loop_30_scenes/tables/scenes_30.txt` | the 12-scene and 18-scene lists the queue read, one after the other |
| `paddock/drivejepa_a4/` | `{paddock_sid}/Alpasim/adapters/drivejepa_a4/` |

Words the reports use, and what they mean here:

| In the reports | Means |
|---|---|
| approach 4, MH | this experiment; approach 3 is H17 |
| p06, 0.1.8b-p06, NASG | the starting model (Drive-JEPA 0.1.8b, pass 0.6), and its retrain's store |
| arm A, arm B | history shown (`H19.r1`), history hidden (`H19.r2`) |
| PREREG, gate | the rule written before training, and its pass or fail. Here "gate" never means paddock's regression gate, which did not run |
| build 1, build 2 | the history data, then the model and trainer |
| link 1, link 2 | the link into the generator's plan starting points, the link into the scorer's rating head |
| kill switch | the pull-away floor (0.66) below which a copy would not have been driven |
| image check, wiring proof | the simulator reproducing the starting model's drives; the new driver reproducing it at step 0 |
| cl20ic, cl20, cl18 | H17's image check and closed-loop queues, which ran on the same machine first |
| "Sid's terms", "Sid's go" | the owner's conditions and approvals, given on the dates stated |
| OPEN, flags | questions left for the owner, and caveats, at the end of each report |

## What each script does

| Script | Step folder | What it does |
|---|---|---|
| `ego_history.py` | `03_history_data` | the one history builder, used by the training side file and by the driver in the car |
| `test_ego_history.py` | `03_history_data` | its four tests: no future leak, relative pose matches both planners, interpolation matches the converter, shape and valid flag |
| `build_side_file.py` | `03_history_data` | builds the history for every moment of the training slice |
| `build_picking_history.py` | `03_history_data` | builds the history for the picking-set moments, which are not in the training slice |
| `checks.py` | `03_history_data` | checks the side file: completeness, coverage, frame offsets |
| `drive_dry_run.py` | `03_history_data` | feeds a recorded drive's pose stream to the builder, without the simulator |
| `a4_config.py` | `04_model_and_trainer` | every location the scripts read, in one place |
| `a4_data.py` | `04_model_and_trainer` | the loader: the starting model's sample list and sampler with cached tokens and the history |
| `model_a4.py` | `04_model_and_trainer` | the model: the starting model wrapped, not edited, plus the motion encoder and the two links |
| `copy_tokens.py` | `04_model_and_trainer` | copies H17's token cache into two local arrays in one pass |
| `data_check.py` | `04_model_and_trainer` | checks the loader's values against the starting model's own dataset code |
| `reader_bench.py` | `04_model_and_trainer` | times three ways of reading the tokens of 32 random moments |
| `local_reader_bench.py` | `04_model_and_trainer` | the same timing on the local token copy |
| `step0_check.py` | `04_model_and_trainer` | the step-0 check on the GPU: the new model equals the starting model on 1,131 moments |
| `run.sh` | `04_model_and_trainer` | launcher: the training interpreter with the earlier code on its path, on cores 0-1 at lowest priority |
| `train_a4.py` | `05_training_two_arms` | the trainer: the starting model's recipe, checks every 930 steps, full saves and resume |
| `det_grid_sample.py` | `05_training_two_arms` | a repeatable backward for `grid_sample`, installed for training only |
| `det_probe.py` | `05_training_two_arms` | runs one training step twice on the same state and lists the tensors whose gradients differ |
| `resume_compare.py` | `05_training_two_arms` | compares the resume test's straight and resumed runs |
| `report_arm.py` | `05_training_two_arms` | writes each copy's check table and chosen checkpoint into the report |
| `queue.sh` | `05_training_two_arms` | trains history shown, then history hidden, each after the memory gate |
| `queue_resume_B.sh` | `05_training_two_arms` | resumes the history-hidden copy after the machine reboot |
| `queue_resume_B2.sh` | `05_training_two_arms` | resumes it again after the GPU out-of-memory stop |
| `cl_queue.sh` | `06_closed_loop_30_scenes` | the closed-loop queue: image check, wiring proof, history shown on 30, history hidden on 30 |
| `cl_imagecheck_cmp.py` | `06_closed_loop_30_scenes` | compares the starting model's re-runs on the 8 scenes with its recorded drives, tick by tick |
| `cl_wiring_cmp.py` | `06_closed_loop_30_scenes` | compares the step-0 drive through the new driver with the starting model's drives |
| `cl_scores.py` | `06_closed_loop_30_scenes` | the score tables and the verdict by the rule written beforehand |
| `a4_worker_check.py` | `06_closed_loop_30_scenes` | checks the new driver's model code offline in the driver's own environment |
