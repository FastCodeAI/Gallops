# Drive-JEPA, retrained on KITScenes — the working record

The scripts, reports and result tables of three experiments, as they were written. The account of
what was done and what it measured is [README.md](README.md) in this folder;
start there.

**This is a record, not a package.** It sits outside `src/gallops/` and outside `make lint`, like
`experiments/drivejepa_retrain/`. The scripts ran as scripts, from a working tree at the training
site, under Drive-JEPA's own Python 3.9 environment, and are committed in that form.

**The checkpoints are a research record, not for deployment.** None has a `checkpoint.json`, none
is in `{weights}`, and the one that went through Paddock's regression gate failed it.

## What is here

The folders are numbered in the order the work was done.

| Folder | Experiment | What it did | Report |
|---|---|---|---|
| `01_encoder_survey/` | 1 | read the encoder, the data and the earlier KITScenes evaluation before building anything | `REPORT.md` |
| `02_encoder_build/` | 1 | the scene split (482 training, 50 picking), the LoRA, the hide-and-guess players, the baseline re-run | `REPORT.md` |
| `03_encoder_training/` | 1 | frame cache, masks, warm-up and 6 passes of hide-and-guess; pass 3 chosen; one read of the held-out moments | `REPORT.md` |
| `04_encoder_checks/` | 1 | random LoRAs of the same size (does training matter?) and a same-software re-run with the LoRA at zero | `REPORT.md` |
| `05_generator_survey/` | 2 | what the generator's loss is, which half of it KITScenes can supply, memory and speed | `REPORT.md` |
| `06_generator_only_fast_rate/` | 2 | generator only, learning rate 1e-4; stopped after 1 pass when the menu lost too much variety | `REPORT.md` |
| `07_generator_only_gentle_rate/` | 2 | generator only, learning rate 3e-5, with a stay-near-original term at weight 0, 0.3 and 1.0; weight 0 at 9.5 passes chosen | `REPORT.md` |
| `08_generator_and_encoder_together/` | 2 | generator and encoder LoRA trained together, beside a generator-only control trained longer | `REPORT.md` |
| `09_heldout_comparison/` | 1, 2 | the four models of experiments 1 and 2 on the 223 held-out moments; `heldout_moments.csv` lists them | `REPORT.md` |
| `10_encoder_generator_scorer/` | 3 | encoder LoRA, generator and scorer trained together; 4.5 passes chosen; one read of the held-out moments | `REPORT.md` |
| `11_regression_gate/` | 3 | the chosen checkpoint's export for Paddock (`PREP.md`, `prep_export.py`) and its regression-gate result (`RESULTS.md`, `results.json`, `build_results.py`); `GATE_EXPLAINED.md` describes the gate from Paddock's code | `RESULTS.md` |

`doc_tables.py` renders the doc's tables from the committed JSON; the doc quotes its output and
`tests/test_drivejepa_kitscenes.py` keeps the two in step. `09_heldout_comparison/heldout_moments.csv`
was written at commit time from the held-out results file (`POINTERS.md`); everything else here is
as it ran.

**The reports and scripts are in their authors' working shorthand** (stage labels, "pass 3", "p9.5",
"phase 2", "G1"…). They were written day by day and are committed as written. The folder names
above were given at commit time; the "Where this came from" table at the end of the doc maps each
to the name it was worked under. In the regression-gate folder, only the result, the explanation,
the export notes and their two scripts are included; the two survey notes written before the job
was submitted are not.

## What is not here

- **No weights, arrays, frames, caches or logs**, and no table over 100 KB. [POINTERS.md](POINTERS.md)
  lists them with sizes and sha256, including each checkpoint and the Drive-JEPA files the code
  imports.
- **No KITScenes images, poses or per-moment arrays.** The scene-id lists (`02_encoder_build/out/splits.json`,
  `09_heldout_comparison/heldout_moments.csv`) and per-moment tables of ids, speeds and errors are
  committed. The per-scene location table is not. Whether KITScenes' terms allow publishing the
  committed tables is for whoever accepted those terms to confirm before this repository is opened
  beyond the people who have access.
- **No copy of Drive-JEPA.** The code imports it from the vendored tree Paddock's KITScenes runner
  uses, and never edits it.

## Reading a path in these files

Every machine path was replaced by a placeholder before committing, and every mention of a working
folder name by the folder's name here. Apart from the edits listed at the end of this section,
nothing else in a file was changed.

| Placeholder | Stands for |
|---|---|
| `experiments/drivejepa_kitscenes` | this folder, when it was the working tree at the training site |
| `{runs}/drivejepa_kitscenes` | that site's local store for these experiments (checkpoints, caches, per-moment results); folders under its `encoder_lora_kit/` and `generator_joint_kit/` keep short working names (`s1` … `s5`) |
| `{weights}` | the folder holding the released Drive-JEPA checkpoint and the guesser's starting weights (`Drive-JEPA-cache/`) |
| `{data.kitscenes}`, `{data.kitscenes.second_copy}` | two copies of the KITScenes Multimodal scenes |
| `{kitscenes_toolkit}` | the KITScenes toolkit checkout, which holds the official split lists |
| `{drivejepa_repo}` | the vendored Drive-JEPA checkout (upstream `e21f474`) the code imports from; its `navsim_v2/` is called the devkit in the reports |
| `{envs}` | the conda environments (`drive-jepa`, Python 3.9, and a torch add-on) |
| `{paddock}`, `{paddock_gate_checkout}` | a Paddock checkout, and the one the gate job was submitted from |
| `{paddock_site}`, `{paddock_runs}`, `{paddock_checkpoints}` | the gate queue's site folder, Paddock's runs root and its checkpoint store |
| `{naren_workspace}`, `{naren_navsim_workspace}` | a colleague's workspace, where the earlier KITScenes evaluation of the released model lives |
| `{home}`, `{workspace}`, `{gallops}`, `{nas}`, `{nas_backup}`, `{bulk_disk}`, `{bulk_disk_root}`, `{p3}`, `{p4}`, `{ssd}` | a home directory, the working folder holding the checkouts, a Gallops checkout, network and local disks |

Only `{runs}` and `{weights}` are variables `gallops.config` expands. The rest are labels: **a script
here will not run until its placeholders are put back to real locations, and none has been run since
they were put in.** The `run.sh` launchers put earlier folders' `src/` on `PYTHONPATH`, so the
folders only work together.

[AS_RUN.sha256](AS_RUN.sha256) holds the sha256 of each of the 47 files as it was before these
changes, against its original path. It is a record of the originals, so `sha256sum -c` does not
pass against the committed files. The originals are kept at the training site. The replacements cannot
be undone from the committed text alone, because they are many-to-one: `{workspace}` and `{paddock}`
each stand for two spellings of a path (absolute and `~/`), and several working folder names became
the same name here. The script that made the copy is not committed. Files not listed are
byte-identical to what ran.

Edited after review, beyond the placeholders:

- `05_generator_survey/REPORT.md`, the **Folder** paragraph: reworded, because the renaming had turned
  four working folder names into one and the sentence no longer read.
- `05_generator_survey/out/ce_parts.json`, `ckpt_meta.callbacks`: replaced by a one-line note. It held the
  released checkpoint's training-callback state, which names the upstream author's machine paths.
  Its original hash is in `AS_RUN.sha256`.
