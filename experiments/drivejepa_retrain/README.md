# Drive-JEPA, retrained on Physical AI AV — the working record

The scripts, reports and result tables of one experiment, step by step, as they
were written. The account of what was done and what it measured is
[docs/drivejepa-retrain.md](../../docs/drivejepa-retrain.md); start there.

**This is a record, not a package.** It sits outside `src/gallops/` and outside
`make lint`. The scripts ran as scripts, from a working tree at the training
site, and are committed in that form.

## What is here

The folders are numbered in the order the work was done.

| Folder | What it did | Report |
|---|---|---|
| `01_find_clips/` | listed what the dataset holds, set aside the evaluation scenes, counted the clips available for training | `REPORT.md` |
| `02_build_samples/` | turned a raw clip into a model input (crop, resize, bonnet mask) | `REPORT.md` |
| `03_grade_candidates/` | graded each candidate path against a map derived from the clip | `REPORT.md` |
| `04_training_set/` | built the training set: convert, grade, extend the trajectory vocabulary, split | `REPORT.md` |
| `05_small_overfit_check/` | trained 30 epochs on 1 % of it, to see whether the loss moves | `REPORT.md` |
| `06_first_training_run/` | the first training run: generator and scorer together, 10 passes (a pass is one sweep through the training data) | `REPORT.md` |
| `07_closed_loop_test/` | 30 simulated scenes driven with the first run's checkpoint; `runs_017/` is the same test with the second run's | `REPORT.md` (first run only) |
| `08_scorer_diagnosis/` | why the scorer did not improve | `REPORT.md` |
| `09_three_hypotheses/` | three explanations, each checked on stored outputs | `REPORT.md` |
| `10_three_quick_looks/` | three further checks before any more GPU time | `REPORT.md` |
| `11_scorer_head_and_pullout_grades/` | a small replacement scorer head on frozen features, and grades for pulling out from a standstill | `REPORT.md` |
| `12_second_training_run/` | the second training run: generator only, scorer frozen | **none** — tables only, see below |
| `13_best_possible_pick/` | would the two scenes where the car never moves be driven if the best candidate were always picked | `REPORT.md` |
| `14_after_retraining_diagnosis/` | why the first run's checkpoint still swerves and still does not pull out, in five parts | `NOTE_FOR_SID.md`, a `README.md` per part |

Two steps have tables and no written report: `12_second_training_run/` and the
0.1.7 test under `07_closed_loop_test/runs_017/`. `doc_tables.py` renders their
tables from the JSON; the docs quote its output and a test keeps the two in
step.

**The reports and scripts are in their authors' working shorthand.** They were
written day by day and are committed as written, so they use short labels for
steps, checkpoints and checks that the docs do not. The folder names above were
given at commit time; the names the work was done under are in the "Where this
came from" table in `docs/drivejepa-retrain.md`, and file names inside each
folder are unchanged.

## What is not here

- **No driver source.** The code that runs these weights in the simulator lives
  in paddock, at `tools/models/drivejepa/alpasim_driver/`. The two
  `07_closed_loop_test/adapter_*_as_built.diff` files are records of what the
  tests ran, not something to apply: 0.1.7's is written against 0.1.6's, which
  is why both are kept.
- **No full copies of upstream files.** The three patched Drive-JEPA training
  files are committed as diffs against upstream at `e21f474`
  (`05_small_overfit_check/train_patch/dataset.diff`, `drive_jepa_agent.diff`,
  and `12_second_training_run/train_patch/drive_jepa_agent_s10.diff`, which
  applies on top of the second). Applying them reproduces the files that
  trained, byte for byte; their sha256 are in `docs/drivejepa-retrain.md`.
- **No weights, arrays, frames, videos, caches or logs**, and no table over
  100 KB outside the closed-loop test. [POINTERS.md](POINTERS.md) lists them
  with sizes and hashes.

## Reading a path in these files

Every machine path was replaced by a placeholder before committing, and every
mention of a working folder name by the folder's name here. Nothing else in a
file was changed. The three upstream diffs are the exception: they are byte for
byte as they were, working folder names in their comments included, so that
they still rebuild the files that trained.

| Placeholder | Stands for |
|---|---|
| `experiments/drivejepa_retrain` | this folder, when it was the working tree at the training site |
| `{runs}/drivejepa_retrain` | that site's network store for this experiment (checkpoints, manifests, downloads) |
| `{runs}/alpasim_bulk` | that site's bulk disk for the simulator: environments, run folders, the released weights |
| `{runs}/paddock_sid` | a personal working tree inside a paddock checkout, where the drivers and earlier analyses live |
| `{paddock}` | that paddock checkout |
| `{runs}/vm_paddock_sid`, `{runs}/vm_alpasim_bulk`, `{vm_home}` | the same things on an earlier machine |
| `{home}`, `{nas}`, `{bulk_disk}`, `{site-host}`, `{site-nfs}`, `{hf-account}`, `/tmp/{session}` | a home directory, a mount, a disk, a host address, an NFS export, an account name, a temporary directory |

Only `{runs}` and `{weights}` are variables `gallops.config` expands. The
rest are labels: **a script here will not run until its placeholders are put
back to real locations, and none has been run since they were put in.** The
folder names apply under `{runs}/drivejepa_retrain` too: on the training site's
store each folder still carries its working name.

[AS_RUN.sha256](AS_RUN.sha256) holds the sha256 of each of the 206 files as it
was before these replacements, against its original path. Its first line
records the one file name that was also replaced: the retrained checkpoint's. It is a record of
the originals, so `sha256sum -c` does not pass against the committed files;
undoing the replacements reproduces every hash. Files not listed are
byte-identical to what ran.
