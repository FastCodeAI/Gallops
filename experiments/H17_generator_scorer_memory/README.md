# H17 — Memory in the generator and the scorer

**Owner:** Siddesh · **Started:** 7 Nov 2026 · **Status:** parked, negative (hypothesis)

The facts — runs, data, checkpoints, results — live in [experiment.toml](experiment.toml) and are rendered
into the repository README and [CARD.md](CARD.md). This page is the account: why, what was done, what it
showed, and what was decided. The step folders hold the reports, the rule written before each run, the
scripts as they ran and the small result tables. [POINTERS.md](POINTERS.md) lists what is not in the
repository.

## Why

Drive-JEPA decides every moment from scratch: it sees two frames 0.5 s apart and keeps nothing. Two earlier
experiments asked whether a memory helps while the released model stays frozen:

- [H05](../H05_ssm_memory/README.md): the camera's past, read through a small probe, added little.
- [H15](../H15_prediction_memory/README.md): a memory of the planner's own earlier outputs, per path, could
  not be told from the same layer wiped.

Neither let the model learn to use a memory. This experiment does: the generator, which proposes 32
candidate paths, and the scorer, which picks one, are trained together with a memory they both read.

## What was done

The starting model is Drive-JEPA 0.1.8b: an earlier retrain of the released model on NVIDIA's Physical AI
AV dataset. Its own record is not in this repository yet, so it is pinned here by sha256 only.

1. **Survey** (`01_survey`). What data, code and held-out clips the earlier retrain left, how often
   cut-ins, crossings and hard braking occur, and whether the starting model had been driven on all 30
   closed-loop scenes (it had not: 12).
2. **Data** (`02_data_slices_and_token_cache`). The encoder stays frozen, so its output was saved once
   for every moment: 126,981 training moments, 2,591 moments of a *picking set* (84 clips, used to choose
   checkpoints; 1,131 of them are graded), and two held-out slices that were cached and never opened.
3. **Model and trainer** (`03_memory_model_and_trainer`). A memory of 16 slots of 256 numbers. At every
   moment the generator and the scorer read it, then it is updated from the current view. 1.67 M new
   parameters. The trainer walks each clip in time order. Five checks passed before any training; the
   first is that the untrained model reproduces the starting model's plans, scores and picks bit for bit.
4. **First training run** (`04_carried_vs_wiped_training_run`). Two copies, two passes each, one seed:
   the memory *carried* from moment to moment, and the same layer *wiped* at every moment. The wiped copy
   separates a memory from extra capacity.
5. **Memory-collapse fix** (`05_memory_collapse_fix`). The first run's memory had collapsed, so it was
   given a learned start state and one gate per slot, and trained for 930 steps.
6. **Recall lesson** (`06_recall_lesson`). The memory was also taught to recall the view 2 s and 4 s ago,
   so that it has a reason to keep the past. 930 steps.
7. **What the memory holds** (`07_does_the_memory_hold_driving_information`). No training: simple readers
   predict the driver's next 3 s with and without the trained memory.
8. **What the past view holds** (`08_does_the_past_view_help_at_all`). The same readers are given the
   true past camera view directly. That is the most any memory of these views could offer.
9. **Closed loop** (`09_closed_loop_12_scenes`). The same weights driven on 12 simulator scenes with the
   memory carried and with it reset at every call. Not finished when this was recorded: the folder holds
   the rule written beforehand and the scripts, and no result.

## What it showed

All numbers are open loop, on the picking set's 1,131 graded moments, with one seed. The *pick grade* is
the mean grade of the path the scorer picks, from the grader the earlier retrain used. The starting model
scores 0.7685.

- **With a healthy memory, carrying it does not help.** After the recall lesson the memory passes every
  health condition written beforehand: its slots are distinct, its gates are open, and it recalls the
  view 2 s and 4 s back better than a memory reset each moment (6.0 % and 10.5 % lower error). On the same
  weights the pick grade is 0.7313 with the memory carried and 0.7355 with it reset every moment:
  carried is 0.0042 lower. This is the headline. Both are below the starting model.
- **The first run's gap came from a collapsed memory.** Carried scored 0.7752 and wiped 0.7639. But the
  carried memory's 16 slots were 99.96 % alike and its gates were almost shut, so the gap is not the
  memory working. Its 95 % interval, −0.008 to +0.032, crosses zero, and on the 34 moments after an event
  the carried copy was lower. The rule written beforehand said stop.
- **The collapse fix alone gave the same picture:** 0.7496 carried, 0.7520 reset.
- **The trained memory adds nothing for predicting the driver** beyond the current view and the car's own
  motion history (−2.8 % to +0.6 % across six targets). By the rule written beforehand: no.
- **The true past view adds little.** Views from 2 s and 4 s ago improve a linear reader's speed
  prediction by 1 to 2 %, and add nothing on braking, sideways offset or heading. By the rule written
  beforehand: unclear, because the small gain on speed is real but below the bar.

What is measured and what is argued:

- Measured: every number above, each copied from the json file named in `experiment.toml`.
- Argued: that the low ceiling is why training did not find a use for the memory. The tests show the
  memory holds little that helps and that the past view offers little; they do not show one causes the other.
- Not established: that the recall lesson caused the pick grade to fall below the starting model. The
  same weights score 0.7355 with the memory reset, so the drop sits in the weights, not in the reading.

## Decision

Parked, verdict negative. The gate has not run: no checkpoint here is a candidate to ship, and every one
scores below its starting model offline. No decision row is added, because nothing changes what the team
does next beyond this record. The car's own motion history was the stronger signal in both reader tests;
that is a different claim and will be its own experiment.

## Limits

- One seed, one data set, 930 steps for the two runs after the first.
- The picking set chose the first run's checkpoints and is the set reported on; that result carries the
  `selected-on-report` flag. The headline does not: its step was fixed beforehand.
- The two held-out slices were never opened, so nothing here is a held-out number.
- The rule for each run was written before the run started, in a folder outside git. The file times and
  the run logs agree, but no commit date shows it, so the record says `predictions = "none: ..."`.
- Numbers quoted in the reports from the starting model's own training (for example its closed-loop
  scores) have no record in this repository.

## These scripts do not run from this repository

They are a record of what ran. They import the earlier Physical AI AV training code, which is not here:
`pai_common`, `pai_model`, `pai_model_b`, `pai_model_c`, `pai_data`, `pai_slim`, `pai_train`, `train_018b`,
`s35_common`, `batch_fix`, `b3_common` and the patched agent `drive_jepa_agent_s10`. Two scripts
(`s4_cache_sample.py`, `s5_cache_full.py`) also import `gallops.memory.probe.pool_map`, which is in this
repository. When they ran, all scripts sat in one folder, `src/`, and imported each other by name.

## Reading a path in these files

Every machine path was replaced by a placeholder before committing. One other character was changed: a
space was added inside a code fragment in `03_memory_model_and_trainer/REPORT.md` (`mlp[0] (x)`), because
the repository's link check read it as a link. Nothing else in a file was changed.
[AS_RUN.sha256](AS_RUN.sha256) holds the sha256 of each changed file as it was, against its original path;
it is a record of the originals, so `sha256sum -c` does not pass against the committed files.

| Placeholder | Stands for |
|---|---|
| `{runs}/generator_scorer_memory` | the training site's network store for this experiment: cache, slices, run folders |
| `{runs}/generator_joint_pai` | the store of the earlier retrain that produced the starting model |
| `{runs}/physicalai` | the store holding the dataset download and the earlier retrain's training set |
| `{runs}/paddock_sid`, `{runs}/paddock_sid_moved` | a personal working tree inside a paddock checkout, and the part of it that was moved to the network store |
| `{runs}/alpasim`, `{runs}/alpasim_bulk` | simulator results on the network store, and the bulk disk holding the environments |
| `{working_folder}` | the folder these files were written in, with `reports/`, `src/`, `out/` and `logs/` |
| `{working_notes}`, `{gallops}`, `{gallops_main}`, `{checkouts}`, `{paddock}` | the personal notes folder, two gallops checkouts, their parent folder, a paddock checkout |
| `{data.navtrain.cache}` | H05's saved outputs of the released model |
| `{home}`, `{nas}`, `{nas_store}`, `{bulk_disk}`, `{paddock_site}`, `{paddock_site_nas}`, `/tmp/{session}` | a home directory, a mount, a personal folder on it, a disk, paddock's site folders, one session's temporary directory |

Only `{runs}` and `{weights}` are variables `gallops.config` expands. The rest are labels.

## Where this came from

The folders and reports were renamed so that a reader does not need the working notes. The reports and
scripts are committed as written and still cite each other by their working names:
`reports/<working name>`, `src/<script>`, `out/<table>` and `logs/<log>`. A table cited as `out/<name>` is
in its step folder's `tables/`, or in POINTERS.md if it was left out. Logs are in POINTERS.md.

| Here | Working name |
|---|---|
| `01_survey/REPORT.md` | `reports/APPROACH3_RECON.md` |
| `01_survey/has_the_start_model_driven_30_scenes.md` | `reports/ARM_C_CHECK.md` |
| `02_data_slices_and_token_cache/REPORT_slices.md` | `reports/BUILD1_SLICES_CACHE.md` |
| `02_data_slices_and_token_cache/REPORT_full_cache.md` | `reports/BUILD2_CACHE.md` |
| `03_memory_model_and_trainer/REPORT.md` | `reports/BUILD3_MODEL_TRAINER.md` |
| `04_carried_vs_wiped_training_run/PREDICTIONS.md`, `REPORT.md` | `reports/PILOT_PREREG.md`, `PILOT_SEED0.md` |
| `05_memory_collapse_fix/PREDICTIONS.md`, `REPORT.md` | `reports/BUILD4_PREREG.md`, `BUILD4_MEMORY_FIX.md` |
| `06_recall_lesson/DESIGN.md`, `PREDICTIONS.md`, `REPORT.md` | `reports/BUILD5_DESIGN_NOTE.md`, `BUILD5_PREREG.md`, `BUILD5_RECALL_LESSON.md` |
| `07_does_the_memory_hold_driving_information/PREDICTIONS.md`, `REPORT.md` | `reports/LINK2_PREREG.md`, `LINK2_TEST.md` |
| `08_does_the_past_view_help_at_all/PREDICTIONS.md`, `REPORT.md` | `reports/CEILING_PREREG.md`, `CEILING_TEST.md` |
| `09_closed_loop_12_scenes/PREDICTIONS.md` | `reports/CL12_MEMORY_PREREG.md` |

Words the reports use, and what they mean here:

| In the reports | Means |
|---|---|
| approach 3 | this experiment; approaches 1 and 2 are H05 and H15 |
| recon; build 1 to build 5 | the survey; then the slices, the full cache, the model and trainer, the memory-collapse fix, the recall lesson |
| pilot | the first training run |
| carry, wiped, reset | memory kept from moment to moment; the same layer trained with its memory zeroed every moment; a trained memory put back to its start state every moment at test time |
| T1 to T5 | the five checks of the model and trainer: identity at step 0, gradient routing, wiped mode, a 50-step smoke run, timing |
| link 2 | the test of whether the memory holds anything driving needs (link 1 is whether the model reads the memory) |
| ceiling test | the test with the true past view |
| CL12 | the 12-scene closed-loop check |
| 0.1.8b, pass 0.6, p06 | the starting model and the point in its training it was taken from |
| 0.1.7 | the retrained checkpoint of [GC05](../GC05_physicalai_retrain/README.md), whose grader is used here |
| arm (c) | the starting model driven as a third arm beside carried and wiped |
| pre-registration, PREREG | the rule for reading a result, written before the run |
| OPEN, flags | questions left for the owner, and caveats, at the end of each report |

## What each script does

| Script | Step folder | What it does |
|---|---|---|
| `b1_b3.py` | `01_survey` | counts the clips, moments and gaps in the existing training and held-out manifests |
| `b2_gaps.py` | `01_survey` | measures gaps between consecutive moments inside each clip of a manifest |
| `b2_gaps_slice.py` | `01_survey` | the same gap scan with the manifest paths taken from the environment |
| `b3_timerange.py` | `01_survey` | reads each clip's recorded time range from the dataset's index |
| `b4_events.py` | `01_survey` | counts cut-ins, crossings and hard braking on the training clips that have agent tracks |
| `b4_extra.py` | `01_survey` | counts hard braking from the manifest speeds on all 4,015 training clips |
| `b4b_events.py` | `01_survey` | the same event count on all training clips and the 300 held-out clips |
| `b4b_fetch.py` | `01_survey` | downloads the agent tracks of 3,000 ordinary training clips |
| `b4b_moments.py` | `01_survey` | lists the moments within 5 s after an event, and the per-type summary |
| `b4b_verify.py` | `01_survey` | checks every downloaded track file against its manifest hash |
| `b4c_events.py` | `01_survey` | the same event count on the two new held-out slices and the picking set |
| `c1_params.py` | `01_survey` | counts the starting model's parameters by part |
| `probe_manifests.py` | `01_survey` | prints the columns and sizes of the existing manifests |
| `probe_timing.py` | `01_survey` | times one clip read from the dataset |
| `run_model_cache.sh` | `02_data_slices_and_token_cache` | the same launcher without the core pin, for the one full-cache job |
| `s1_graded_heldout.py` | `02_data_slices_and_token_cache` | builds the graded held-out slice from reconstructed validation clips |
| `s2_fetch_tracks.py` | `02_data_slices_and_token_cache` | downloads agent tracks for the fresh raw held-out clips |
| `s2_run_b3.py` | `02_data_slices_and_token_cache` | runs the earlier download and convert steps on the fresh slice, unchanged |
| `s2_select.py` | `02_data_slices_and_token_cache` | selects 300 fresh raw held-out clips |
| `s3_moments.py` | `02_data_slices_and_token_cache` | lists the moments within 5 s after an event for the new slices |
| `s4_cache_sample.py` | `02_data_slices_and_token_cache` | caches the encoder's output for the sample and checks it is exact |
| `s4_sample_list.py` | `02_data_slices_and_token_cache` | picks about 1,000 moments as whole clips for the cache sample |
| `s5_cache_check.py` | `02_data_slices_and_token_cache` | checks the cache against a full forward pass on 200 moments per slice |
| `s5_cache_full.py` | `02_data_slices_and_token_cache` | caches the encoder's output for one whole slice |
| `run_model.sh` | `03_memory_model_and_trainer` | launcher: the training interpreter on cores 0-1 at lowest priority |
| `s6_data.py` | `03_memory_model_and_trainer` | reads clips from the token cache for the trainer |
| `s6_env.py` | `03_memory_model_and_trainer` | points an earlier module at a folder that moved; import shim |
| `s6_fill_report.py` | `03_memory_model_and_trainer` | fills the model report's tables from the test outputs |
| `s6_model.py` | `03_memory_model_and_trainer` | the memory model: 16 slots read by the generator and the scorer, updated every moment |
| `s6_tests.py` | `03_memory_model_and_trainer` | five checks of the model and trainer on the GPU (identity at step 0, gradient routing, wiped mode, 50-step smoke, timing) |
| `s6_train.py` | `03_memory_model_and_trainer` | the trainer that walks each clip in time order, and the check on the picking set |
| `pilot_status.sh` | `04_carried_vs_wiped_training_run` | prints one status line per arm while training |
| `run_model_pilot.sh` | `04_carried_vs_wiped_training_run` | the same launcher with the cores taken from the environment, for two arms side by side |
| `s7_fill_report.py` | `04_carried_vs_wiped_training_run` | fills the training-run report from the run files |
| `s7_pilot.py` | `04_carried_vs_wiped_training_run` | trains one arm, memory carried or memory wiped, for two passes |
| `s7_pilot_final.py` | `04_carried_vs_wiped_training_run` | compares the two arms' chosen checkpoints by the rule written beforehand |
| `s7_report_extras.py` | `04_carried_vs_wiped_training_run` | builds the flags and open-question text of that report from the run files |
| `build4_chain.sh` | `05_memory_collapse_fix` | runs the memory fix's checks and its short run in order |
| `s8_fill_report.py` | `05_memory_collapse_fix` | fills the memory-fix report from its outputs |
| `s8_short.py` | `05_memory_collapse_fix` | the 930-step health check of the memory fix |
| `build5_chain1.sh` | `06_recall_lesson` | runs target statistics, the five checks and the before columns in order |
| `build5_chain1b.sh` | `06_recall_lesson` | runs the before columns again after the first attempt was refused |
| `build5_chain2.sh` | `06_recall_lesson` | runs the 930-step short run, with one planned stop and resume |
| `build5_measure.py` | `06_recall_lesson` | forward-only measurement for the design note: how much an untrained memory recalls |
| `build5_measure_launch.sh` | `06_recall_lesson` | launcher for that measurement |
| `s9_short.py` | `06_recall_lesson` | the 930-step health check of the recall lesson |
| `s9_target_stats.py` | `06_recall_lesson` | mean and spread of the recall targets over the training slice |
| `link2_chain1.sh` | `07_does_the_memory_hold_driving_information` | runs the replays |
| `link2_chain2.sh` | `07_does_the_memory_hold_driving_information` | runs the target statistics, the readers and the summary |
| `s10_link2.py` | `07_does_the_memory_hold_driving_information` | picks the sample and replays the trained models to save their memory at each moment |
| `s10_probe.py` | `07_does_the_memory_hold_driving_information` | fits linear readers that predict the driver's next 3 s with and without the memory |
| `ceiling_chain.sh` | `08_does_the_past_view_help_at_all` | runs the statistics, the readers and the summary |
| `ceiling_chain2.sh` | `08_does_the_past_view_help_at_all` | runs the longer-budget reader after the first chain |
| `s11_ceiling.py` | `08_does_the_past_view_help_at_all` | fits readers that predict the driver with and without the true past view |
| `s11_extract.py` | `08_does_the_past_view_help_at_all` | pools the cached camera views of every training moment |
| `s11_mlp_long.py` | `08_does_the_past_view_help_at_all` | the same small network reader with a longer training budget |
| `s11_posthoc.py` | `08_does_the_past_view_help_at_all` | true past views against shuffled past views, from stored predictions |
| `s12_adapter_proofs.py` | `09_closed_loop_12_scenes` | proves the simulator adapter with a memory switch before any scene is driven |
| `s12_imagecheck_cmp.py` | `09_closed_loop_12_scenes` | checks a rebuilt simulator image against an earlier run on 4 scenes |
| `s12_proofs_chain.sh` | `09_closed_loop_12_scenes` | runs the three adapter proofs in order |
| `s12_queue.sh` | `09_closed_loop_12_scenes` | the queue that drives the 12 scenes, carried then reset |
| `s12_table.py` | `09_closed_loop_12_scenes` | builds the per-scene table and the verdict by the rule written beforehand |
