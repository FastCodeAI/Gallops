# 40 drives, step 0: output paths moved off the read-only NAS

Written 2026-11-17 04:50 IST on shakaal. No drive has been run; the only runs were the two dry runs below, which stop before anything is launched.

## Result

| check | result |
|---|---|
| Every write location found and moved | **done.** One output root per runner, `{local_store}/runs/new40/`. Nothing that affects driving was changed. |
| Run-script gates accept the new root | **yes.** No gate refused. None was changed. |
| Disk estimate for 40 drives | **about 6.5 GiB, at most 9 GiB**, well under the 100 GB stop line. data8t has 2.0 TB free. |
| Dry run, one per arm | **both pass** (exit 0, "gates passed, nothing launched"). |
| Nothing written under {nas} | **confirmed.** 1,119,105 entries are identical before and after (path, size, modification time), and none is newer than the marker. |
| New comparison page | **live on port 8786** (tmux `cmppage2`). Old videos are read from the NAS and new ones would come from data8t. The old server on 8785 is untouched. |

**Blockers for the real runs, outside step 0:**

1. **The simulator image is gone.** `docker images` lists no `alpasim-base` image at all. The queue gate requires `alpasim-base:0.134.0` = `sha256:9c5909da…` (`src/s18_queue.sh:13`, `image_ok` at :16). It will refuse, correctly. A rebuild plus an image re-check is needed first, and that is your call. This is the third time this has happened; earlier notes: alpasim-images-gone-2026-11.
2. **The rest of the brief is not on disk.** No 40-drive pre-registration, scene list or step 1 exists in this folder. Because of that:
   - the queue copy and its scene-list pin are not made yet;
   - the deviations below are recorded here, ready to paste into the pre-registration.

## Every place a drive writes (file:line), and where it goes now

The runners are `src/s20_run_memory_scene.sh` (carry) and `src/s20_run_018bp06_scene.sh` (0.1.8b). They are copies of the CL18 runners, and the line numbers below refer to them. All run outputs hang off `OUTROOT`, so changing that one line per runner moves everything.

| what is written | carry | 0.1.8b | before | now |
|---|---|---|---|---|
| **output root** | :68 | :57 | `{experiment_store}/runs/cl18[/018bp06]` | `{local_store}/runs/new40[/018bp06]` |
| run folders `<root>/runs/<run>/` | :69, :95 | :58, :81 | under the root | under the root |
| run folder created, run log tee'd into it | :118–119 | :104–105 | 〃 | 〃 |
| temp folder and matplotlib cache (`TMPDIR`, `MPLCONFIGDIR`; also takes uv's lock file) | :77–78 | :63–64 | 〃 | 〃 |
| wizard output = `wizard.log_dir` (rollouts, metrics, aggregate, configs, compose file, prometheus, Hydra `outputs/`, txt-logs); the wizard is started from inside that folder (`cd "$1"`) | :216 | :201 | 〃 | 〃 |
| config-check dry-run folder `runs/_preflight_dryrun/` | :233 | :218 | 〃 | 〃 |
| container logs | :253 | :238 | 〃 | 〃 |
| GPU sample log | :268 | :253 | 〃 | 〃 |
| driver log and worker log (`DRIVEJEPA_WORKER_LOG`) | :274 | :259 | 〃 | 〃 |
| gate results (`ext20_gates.json`, `dj_extra_gates.*`, `smoke_gates.txt`) | :335–355 | :316–335 | 〃 | 〃 |
| replay and video output `<run>/replay/` (`tools/make_replays.py --out-dir`) | :365–368 | :345–348 | 〃 | 〃 |
| lane folder of symlinks `<root>/lane/` | :84, :369, :377 | :70, :349, :357 | 〃 | 〃 |
| scene cache and renderer artifacts (`scenes.scene_cache`), scene-set lock files | :67, :217 | :56, :202 | local HDD `{alpasim_bulk}/nre-artifacts` | unchanged (not NAS) |
| HF cache (`HF_HOME`) | `Alpasim/alpasim_env.sh:5`, sourced at :133 | sourced at :119 | local HDD `$BULK/hf-cache` | unchanged (not NAS) |
| renderer and simulator temp inside containers | `alpasim/src/wizard/configs/base_config.yaml:148-149,162` | 〃 | container `/tmp` | unchanged (not NAS) |
| uv package cache | wizard `uv run` | 〃 | `~/.cache/uv` (local) | unchanged |
| queue log, STOP file | `src/s18_queue.sh:14` (log → `logs/`), :17 (STOP under the root) | | NAS root | the new queue copy will use the new root (not made yet, blocker 2) |
| comparison page | `src/s13_compare_page.py:32` | | `{experiment_store}/compare_page` | new copy `src/s20_compare_page.py` writes only to `{local_store}/compare_page` (asserted) |

**Read from the NAS, read-only, unchanged:**
- the memory model folder (`{experiment_store}/runs/cl12/model/<arm>`, carry :72);
- the 0.1.8b checkpoint (`adapters/drivejepa018bp06/constants.py:65`, same in `drivejepamemory/constants.py:65`).

I grepped both adapters and the 0.1.8b draft for writes. The only writes are the logs and gate files above, all inside the run folder. Nothing writes to the model folder.

## Deviations, for the pre-registration

1. **Output root, carry.** `src/s20_run_memory_scene.sh:68` reads `OUTROOT={local_store}/runs/new40`.
   - Against the frozen `adapters/drivejepamemory/launch/run_memory_scene.sh`, the diff is still exactly 2 lines (4 `<>` lines): this one, plus `MEMDIR` pointing at the 12-scene model folder.
   - That matches the CL18 copy.
2. **Output root, 0.1.8b.** `src/s20_run_018bp06_scene.sh:57` reads `OUTROOT={local_store}/runs/new40/018bp06`.
   - Against the frozen `drafts/drivejepa-0.1.8b-p06/run_018bp06_scene.sh`, the diff is still exactly 1 line (2 `<>` lines).
3. **Comparison page.** It is written to data8t and served on port 8786 instead of 8785. Its content code is unchanged (`s20_compare_page.py` imports `s13_compare_page.py` and changes only the output folder and where finished videos are looked up).

Nothing else changed: wizard command, configs, adapters, checkpoints, gates and thresholds are all the same.

## Gates and the new root

These gates were checked in code and passed in the dry runs:
- **Run-folder check** (`case "$LOGDIR"`, carry :114, 0.1.8b :100). It is relative to `$S5A`, so it follows the root.
- **Config-diff gate** (`drafts/ext20/config_diff_gate.py:49`). It waives `wizard.log_dir` by design. In the dry runs it reported PASS, with 3 waived, 0 refused and 334 keys compared.
- **Disk floor** (carry :146, 0.1.8b :132). It is `df` of `$S5A`, so it now measures data8t (2014 GiB ≥ 50). Its log line still says "NAS free", which is only the label.
- **Queue gates** (`s18_queue.sh:24-25`). The diff-count checks of 4 and 2 hold for the s20 runners. `image_ok` will refuse, as described in blocker 1.

## Disk size

All 36 CL18 drive folders on the NAS (`du`): 5,979 MiB in total, 166 MiB per drive on average, 227 MiB at most.
- Example: carry and 0.1.8b on `0245ff75` are 193 MiB and 188 MiB.
- 40 drives: about 6.5 GiB, or 8.9 GiB if every drive were the largest.
- Page videos: about 4.5 MB per drive (325 MB for 72 on the NAS), so about 0.2 GB more.

## Dry runs

`--preflight-only` was run on `0245ff75` (pilot10 csv, pinned sha 94f50aa3), under `taskset 0-1` and `nice 19`. It stops after the config-diff gate and launches no driver and no simulator.
- **Carry** (`logs/new40_dryrun_carry.log`):
  - scene csv pin OK;
  - disk floor OK;
  - machine gate 2/2;
  - provenance OK (memory checkpoint 84359783…, base checkpoint 3d566eb2…, mode carry);
  - config-diff PASS;
  - 26.01 artifact ×1;
  - "gates passed, nothing launched".
- **0.1.8b** (`logs/new40_dryrun_018bp06.log`): the same, with PASS. Neither log has any "read-only", "permission denied" or errno line.
- **NAS check:** `find {nas_store} {dataset_store}` listed every entry with path, size and modification time before the dry runs and again after them (and after the page build). The two lists are identical: 1,119,105 entries each, 0 diff lines, 0 errors, and no entry newer than the marker taken before the first dry run.

## Comparison page

- **Folder:** `{local_store}/compare_page/`, built by `src/s20_compare_page.py`.
- **Old videos:** all 72 are taken read-only from the NAS through the symlink `media_nas -> {experiment_store}/compare_page/media`. They are used only when their stamp matches the run exactly; 0 were re-rendered.
- **New videos:** new drives would be rendered into `media/` on data8t.
- **Server:** tmux `cmppage2` runs `serve.py` (with range support) on `{page_host}:8786`. The 8785 server is untouched.
- **Headless check:**
  - the metrics blocks show the same numbers as 8785;
  - the replay filter shows 18 / 12 / 30;
  - both arms' videos load and stay in sync;
  - no console errors, and no sideways scroll at phone width.
- **To do:** the new 40 drives are not discovered by the page yet. Their scenes and blocks come with the rest of the brief.

## New files under data8t (to copy to the NAS once it is writable)

The full list, with sizes and times, is in `out/new40_data8t_files.txt` (110 entries, 60 files, 1.2 MB):
- `approach3/compare_page/`: `index.html`, `data.json`, `serve.py`, `media/playhead.png`, symlink `media_nas`;
- `approach3/runs/new40/`:
  - the two dry-run folders `*.preflight_only_20261117T04*`;
  - `runs/_preflight_dryrun/` and `018bp06/runs/_preflight_dryrun/`;
  - `tmp/`.

Only the page is worth keeping. The dry-run folders can be deleted instead of copied.

Local files changed or added outside data8t:
- `src/s20_run_memory_scene.sh`
- `src/s20_run_018bp06_scene.sh`
- `src/s20_compare_page.py`
- this report
- `out/new40_data8t_files.txt`
- `logs/new40_*.log`
- one scene-set lock file in the local scene cache (`nre-artifacts/scenesets/448068b3….lock`)
