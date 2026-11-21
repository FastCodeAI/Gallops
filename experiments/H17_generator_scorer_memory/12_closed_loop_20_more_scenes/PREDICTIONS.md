# CL20 (40 drives): pre-registration

Written 2026-11-17 05:20 IST, shakaal. **This file holds only the parts fixed so far: the simulator-image check and the output-path deviations.** The rest of the brief (step 1 onward: scenes, arms, decision rule) is not here yet. It will be added as a dated section before any of it is run. No drive has been run, and no image has been built.

## Simulator-image check (step 0b), fixed before any check drive

**Why it is needed:** on 2026-11-17 neither `alpasim-base:0.134.0` (old id `sha256:9c5909dafcd5574dca1c783e6abc29dfbdfcc9fc2860725dd80030475b8f8881`, built 2026-11-12, used by the 12-scene and 18-scene checks) nor the renderer `nvcr.io/nvidia/nre/nre-ga:26.04` (digest 97f43e71…) is on shakaal.

**Rebuild:** done the same way as the 12-scene check (`reports/CL12_MEMORY.md` §3, `logs/cl12_queue.log:4824-4828`). The first check drive's `docker compose`:
- pulls `nre-ga:26.04` from nvcr.io;
- builds `alpasim-base:0.134.0` from the AlpaSim checkout's Dockerfile.

Docker storage is `/var/lib/docker` on the local root disk, not the NAS.

**Check drives:** 6 in total, under `{local_store}/runs/new40/imagecheck/`, through each model's own runner with only the output root changed.

| model | scenes | reference runs |
|---|---|---|
| 0.1.8b pass 0.6 | e3ed554f, 8023dce5, d2715023, 04749bb9 | the existing runs `{retrain_store}/018b/closed_loop/runs/drivejepa_018bp06_<s>`. These were already reproduced bit for bit on 9c5909da (`out/cl12_imagecheck.json`). |
| memory model, carry | bae463d3 | `{experiment_store}/runs/cl12/runs/memory_carry_bae463d3` |
| memory model, carry | 0245ff75 | `{experiment_store}/runs/cl18/runs/memory_carry_0245ff75` |

**Rule (the same comparison as `src/s12_imagecheck_cmp.py`, extended to the two carry drives):**
- For each drive, the score must be equal to 3 decimals.
- Every per-tick metric row of `metrics.parquet` must also be bit-identical. Run identifiers (`rollout_id`, `run_uuid`, `run_name`) are excluded, rows are sorted by name and timestamp, and timestamps are cast to float64.
- **PASS only if all 6 drives match.** If any differs: STOP, report the differences, and drive nothing.

**Known risk, stated before running:** the CL12 carry run (bae463d3) was not CPU-pinned, while the check drive will be (`taskset 0-1`). Pinned and unpinned memory-worker plans were bit-identical on recorded inputs (`out/cl18_thread_check.json`). That covers the model, but it is not a closed-loop proof. A mismatch on bae463d3 alone still counts as FAIL under the rule.

**Only after PASS:**
- the queue's pinned image id (`IMAGE=` in the queue copy, from `src/s18_queue.sh:13`) is changed to the new id;
- that change is recorded here as a deviation, with the old id, the new id and the check result;
- nothing else in the gate changes.

## Deviations so far

1. Output root moved from the NAS to `{local_store}/runs/new40/`:
   - carry: `src/s20_run_memory_scene.sh:68`;
   - 0.1.8b: `src/s20_run_018bp06_scene.sh:57`;
   - details and dry runs: `reports/NEW40_STEP0_PATHS.md`.
2. The comparison page is on data8t, port 8786 (`src/s20_compare_page.py`).

## State at writing (05:20 IST)

**Step 0b is stopped before the rebuild.** The local disk cannot hold the two images; see `reports/NEW40_STEP0B_IMAGE.md`.

---

## Steps 1–2: scenes and decision rule

Written 2026-11-17 23:45 IST, shakaal.
- **No drive on these scenes has been run in this lane, and no model's result on any of them has been read.**
- The image check above is still on hold: 1.7 GB free on the root disk against about 80 GB needed.

### Scenes (step 1)

**Rule, from the brief:**
- Use every AlpaSim scene on shakaal beyond the 30 that has its scene file on disk and that the wizard accepts.
- Fewer than 20: stop. Exactly 20: take all of them. More than 20: draw 20 at random with seed 0 from the sorted list.

**What is on shakaal:**
- **Scene cache:** `{alpasim_bulk}/nre-artifacts/all-usdzs/` (the runners' `scenes.scene_cache`). It holds 53 scene files, 95 GB in all.
- **What they are,** matched by uuid against the wizard's own catalog `alpasim/data/scenes/sim_scenes.csv` and every scene CSV under `paddock-sid/Alpasim` (`{local_store}/new40_scenes/cached_usdz.json`):

  | scene files | how many |
  |---|---|
  | 26.01 build, among the 30 | 30 |
  | 26.04 copies of scenes among the 30 (not the pinned build, unusable) | 3 |
  | **26.01 build (`26.1.112-59673f88`, the runners' pin), not among the 30** | **20** |
  | a leftover staging folder `.s10_fresh_staging` | 0 scenes |

- **I found no other scene cache.**
  - I searched `{bulk_disk}` and `{local_disk}` to depth 4 for any `all-usdzs` or `nre-artifacts` folder. It found only ours, but the search stopped at its 200 s limit.
  - A full search for scene files over 500 MB was stopped after 30 minutes stuck waiting on the disk, with nothing found.
  - The runners read only this cache, so another cache would not be accepted without changing `scenes.scene_cache`.
- **No earlier drive on these 20 found:** I checked 7,099 folders, three levels under each of our 33 AlpaSim run roots on the NAS, data8t and the HDD (`new40_scenes/run_roots.txt`, `run_tree.txt`), and none is named after any of the 20. I opened no run's contents.

**Where the 20 come from:**
- They are exactly the "fresh20" set of the Drive-JEPA retraining (`Gallops/experiments/drivejepa_retrain/12_second_training_run/fresh20.csv`, sha256 `06670a41…`).
- It was drawn on 2026-10-09 with `np.random.default_rng(20261009)` from the 852 `public_2601` scenes never opened in any diagnosis (`loop/runs/retraining1.0_20261009_0546/tasks/04_…/report/launch.md:74-77`).
- They were downloaded from the gated NuRec dataset on 2026-10-09/10, with every size and sha256 equal to the repository's record (the same `launch.md:117-136`, and `tasks/08_…/report/phase2.md:9`).
- **All 20 are on the training exclusion list:** `drivejepa_retrain/01_find_clips/exam_exclusions.txt` excludes all of `public_2601` plus the 30. So no model in this lane was trained on them.
- That lane's `GOAL.md:28,92` reserved them as fresh closed-loop scenes, with "never tune anything on … the 20 fresh scenes". This test only evaluates on them; nothing is tuned.

**The wizard accepts all 20.** I ran the wizard's own checks read-only:
- each scene id has exactly one row in the CSV, at the pinned build (`sceneset.py:273-293`);
- the file is in `all-usdzs/`;
- the uuid and build in the file's `metadata.yaml` match the row (`sceneset.py:469-472`). This held for 20 of 20 (`new40_scenes/wizard_acceptance_check.json`).

The per-scene `--preflight-only` run was **not** done. It refuses below 4 GiB free on the root disk, and only 1.7 GB is free. It will run as the first step of the step-3 queue.

**Count: exactly 20, so all 20 are taken; no draw.**

```
29815599 320c11f3 34226720 49e0ea24 4a495c2d 5c1b7555 68e53160 6ae687bb 7747986c 8578f2c7
8fd8f4f0 9d8fca3b 9e6753c2 a32549eb c438a779 dd0f84df e883822a f005ac5f f3a87e7a f890fe10
```

**Files:**
- `out/cl20_scenes_20.txt` (sha256 `8f88473e…`): short id, clip id and scene file.
- The runners' scene CSV, `out/sim_scenes_new20_26p01.csv` (sha256 `b95455b551d5c384…`). Its master copy is `{local_store}/new40_scenes/sim_scenes_new20_26p01.csv`.
  - Same columns and rows as the wizard catalog (taken from `scene_sets/ext20_pool_26p01.csv`, which holds all 20).
  - It is passed as `--scenes-csv` and pinned with `EXT20_SCENES_SHA`, as in CL18.

### Arms and execution (step 3, on hold)

- **Carry:** `src/s20_run_memory_scene.sh`, the frozen memory adapter with its build freeze checked.
- **0.1.8b pass 0.6:** `src/s20_run_018bp06_scene.sh`, its own adapter.
- **Same as CL18 otherwise** (`reports/CL18_MEMORY_PREREG.md`):
  - comfort re-score on, shadow on in carry, `taskset -c 0-1` and `nice 19`;
  - order per scene: carry, then 0.1.8b;
  - a completed rollout is never rerun, and a run without a rollout stops the queue;
  - annotated video and replay for each drive.
- **Image:** the one that passes the image check above, and only after PASS.

### Measures

- **Score:** the standard score. It is `s5_diagnosis/src/stepb.py` `score` = `aggregate/results-summary.json` `rollouts[0].score`.
  - The window ends at the first crash or off-road, or at 4 m from the human path. The renderer is not valid beyond 4 m.
  - Scored with the code used by `src/s18_table.py`.
- **At-fault (changed from CL18, as the brief defines it):** a front or side collision **inside that scoring window**.
  - It is the window's `collision_at_fault` from the evaluation's own aggregation with both window-end cuts kept: `src/s19_fulldrive_rescore.py`, field `old_collision_at_fault`. That path reproduced every old score to 1e-9 on 102 drives (`reports/FULLDRIVE_RESCORE.md`).
  - CL18 instead counted `stepb.exit_of`, the first event over the whole drive. The report gives both and says where they differ.
- **Win / loss / tie per scene:** carry − 0.1.8b > 0.1 is a win, < −0.1 a loss, otherwise a tie.

### Primary rule (the 20 new scenes only)

d = (20-scene mean of carry) − (20-scene mean of 0.1.8b).

- **SUPPORTS memory** if d ≥ +0.05 **and** wins > losses **and** carry has no at-fault crash on a scene where 0.1.8b has none.
- **AGAINST** if d ≤ −0.05, **or** losses > wins, **or** carry has an at-fault crash on a scene where 0.1.8b has none.
- **NO DETECTABLE EFFECT** otherwise.

The first two cannot both hold. One scene flipping between 0 and 1 moves the 20-scene mean by 0.05.

If a drive has no rollout, the queue stops (as above). The rule is applied only when all 40 drives have a rollout. Otherwise the report gives the missing drives and no verdict.

### Secondary (reported, not deciding)

All 50 scenes (the 30 plus the 20), giving each arm's mean, d, and wins, losses and ties.
- **Carry's 30:** the 12 from `{experiment_store}/runs/cl12/runs/memory_carry_*` (not pinned to 2 cores) and the 18 from `cl18/runs/memory_carry_*`.
- **0.1.8b's 30:**
  - the 12 from `generator_joint_pai/018b/closed_loop/runs/drivejepa_018bp06_*`;
  - the 18 from `cl18/018bp06/runs/drivejepa_018bp06_*`.
  - These are the folders `src/s18_table.py` reads, read-only from the NAS.
- **The 30 old drives used image `9c5909da…`, the 20 new ones the rebuilt image.** They are pooled only because the image check requires bit-identical drives on 6 of the old scenes.

### Not deciding

- How many of the 20 scored 1.0 in both arms (easy scenes).
- d over the 20 without the two scenes most in carry's favour (the largest carry − 0.1.8b).

### What this does and does not show

This compares two models: carry and 0.1.8b pass 0.6. Carry differs from 0.1.8b in two ways:
- the memory;
- the extra 930 training steps that came with it.

This test does not separate the two. A difference shows the combined effect, not memory alone.

### Deviations (output root only, from step 0)

1. Carry output root: `src/s20_run_memory_scene.sh:68` is `{local_store}/runs/new40` (was the NAS `cl18` root).
2. 0.1.8b output root: `src/s20_run_018bp06_scene.sh:57` is `{local_store}/runs/new40/018bp06`.
3. The queue's STOP file and run folders follow that root; the queue log stays in `logs/`.
4. The comparison page is on data8t, port 8786 (`src/s20_compare_page.py`).

Nothing else differs from CL18 apart from:
- the scene CSV;
- the windowed at-fault definition above, which the brief sets;
- the image id, if the image check passes.

Details: `reports/NEW40_STEP0_PATHS.md`.

### State at writing (23:45 IST)

- Steps 1–2 are done.
- Steps 0b, 3 and 4 are on hold until there is about 80 GB free on the root disk (1.7 GB now).

---

## Image check: launch record (2026-11-20 00:03 IST, before the first check drive)

- **Root disk:** 130G free.
- **Real cost of the rebuild:** about 104 GiB, not the ~78 GB estimated above. On 2026-11-12 root free went from 196 GiB to 92 GiB across the first drive (`logs/cl12_queue.log`).
  - Docker uses the containerd image store, which keeps the compressed download beside the unpacked layers.
  - `alpasim-base` builds from `nvidia/cuda:12.4.1-cudnn-devel`, so it shares no layers with the renderer.
- **Renderer on the registry:** `nre-ga:26.04` is still digest `97f43e7130c5…` (`docker buildx imagetools inspect`). The queue checks it after the pull.
- **Queue:** `src/s20_imagecheck_queue.sh`, a copy of `s12_queue.sh` part 1, run in tmux `cl20ic` under `taskset -c 0-1` and `nice 19`. It writes `logs/cl20_imagecheck_queue.log`.
  - Drive order: 0.1.8b on e3ed554f (the one that rebuilds), 8023dce5, d2715023 and 04749bb9, then carry on bae463d3 and 0245ff75.
- **Runners:** `src/s20_run_018bp06_imagecheck.sh` and `src/s20_run_memory_imagecheck.sh`.
  - Each differs from its s20 runner only in OUTROOT (`new40/imagecheck[/018bp06]`).
  - Against the frozen runners the diffs are still 2 and 4 lines; the queue checks this.
- **Added safety (not a gate change):**
  - the queue refuses to start below 110 GiB free on root;
  - a watchdog writes STOP and sends TERM to the running drive if root free drops below 10 GiB.
- **Compare:** `src/s20_imagecheck_cmp.py`, the s12 rule unchanged plus the two carry drives. It writes `out/cl20_imagecheck.json`.
  - Self-test: each of the 6 reference runs compared with itself gives identical rows and no differences.
- **After the queue:** nothing starts on its own. The 40 drives start only after PASS, once the image pin is updated and recorded here.

## Image check: first attempt stopped by the root-disk watchdog (2026-11-20 00:42 IST)

- **What happened:** the first drive (0.1.8b on e3ed554f) pulled and built both images. Root free space then fell from 130 GiB to 9 GiB, the watchdog sent TERM, and the runner tore everything down cleanly (0 networks left). No rollout was written, so nothing was compared.
- **Images after:**
  - renderer `nre-ga:26.04` is id `97f43e7130c5…`, the pinned digest; disk 42.5 GB, content 14.3 GB;
  - `alpasim-base:0.134.0` is the new id `sha256:ab16bc78a468…`, built 00:30:50. Its disk usage equals its content size (12.1 GB), so it is probably not unpacked yet, and the first container would unpack it.
- **Where the space went:** the shared build cache grew from 19.4 GB to 67.3 GB. 43 entries, 47.9 GB, were created by this build.
- **Folders:** the drive's folder (empty `rollouts/`) was renamed `drivejepa_018bp06_e3ed554f.killed_watchdog_20261120T0042`, kept, so the queue does not skip the scene.
- **The STOP file** written by the watchdog is still in place. The queue stays stopped until the space question is settled.

## Image check: resumed (2026-11-20 00:52 IST, Sid's go)

- **Build cache:** our own entries from this build were deleted by id, and nothing else was touched.
  - 22 of the 43 are gone (27.4 GB).
  - 21 (20.5 GB) could not be removed: buildkit keeps them in use for the new image's layers.
  - All 107 older entries (19.4 GB) are untouched. The ids are in `out/cl20_buildcache_ids_deleted.txt`.
- **Root free:** 67 GiB.
- **Queue start check:** lowered from 110 GiB to 50 GiB, since both images are built. The 10 GiB watchdog stop is unchanged.
- **Unchanged:** the 6 drives, their order and the rule.

## Image check result: PASS (2026-11-20 01:52 IST)

All 6 drives match: score equal to 3 decimals and every per-tick metric row bit-identical (`out/cl20_imagecheck.json`, `logs/cl20_imagecheck_cmp.log`).

| drive | original score | re-run score | metric rows |
|---|---|---|---|
| 0.1.8b e3ed554f | 1.0 | 1.0 | 9,004 = 9,004 |
| 0.1.8b 8023dce5 | 0.0 | 0.0 | 9,027 = 9,027 |
| 0.1.8b d2715023 | 1.0 | 1.0 | 9,004 = 9,004 |
| 0.1.8b 04749bb9 | 0.0 | 0.0 | 9,004 = 9,004 |
| carry bae463d3 | 1.0 | 1.0 | 9,027 = 9,027 |
| carry 0245ff75 | 1.0 | 1.0 | 9,027 = 9,027 |

The known risk on the unpinned carry run (bae463d3) did not show up.

- **Renderer:** digest is `97f43e7130c5…`, as pinned.
- **Keep-placeholders:** `siddesh-keep-alpasim-base` and `siddesh-keep-nre-ga` are running with `sleep infinity`, restart unless-stopped, no network, 16 MB memory and 0.01 CPU. Each is labelled "KEEP: AlpaSim images, Siddesh/Naren, do not prune".

### Deviation 5: queue image pin

`src/s20_queue.sh` `IMAGE=` changed:
- **old:** `sha256:9c5909dafcd5574dca1c783e6abc29dfbdfcc9fc2860725dd80030475b8f8881` (`s18_queue.sh:13`);
- **new:** `sha256:ab16bc78a4686dcd63c2b77a77fe585df3846528a61b552db6b71f9553983a44` (`alpasim-base:0.134.0`, built 2026-11-20 00:30:50);
- **reason:** image check PASS, 6 of 6.

Nothing else in the gate changed.

### Step 3 started

The queue is `src/s20_queue.sh`, run in tmux `cl20` under `taskset -c 0-1` and `nice 19`, logging to `logs/cl20_queue.log`.

## Amendment: scene 9e6753c2 replaced (2026-11-20 05:52 IST, Sid's instruction, before the replacement is drawn)

**What happened:** drive 25 (carry on 9e6753c2) ended with no metrics, and the queue stopped as the rule requires.
- The simulator's route generator raised `ValueError: Waypoint 201 fails sanity check: route folds back on itself with angle 180.00 deg` (`alpasim_runtime/route_generator.py:258`). This happened about 12 s into the drive, when it extended the route from the scene's map.
- The simulator's own automatic resume tried 3 times, and all 3 failed at the same waypoint.
- This is a map fault in the scene, not a driving event. The folder is kept as `memory_carry_9e6753c2.simfail_route_foldback`.

**Sid's instruction:** replace 9e6753c2 with another scene, so there are still 20.
- 9e6753c2 leaves the set in both arms. 0.1.8b is not run on it.

**Replacement rule, fixed now:**
- **Pool:** the same pool the 20 were drawn from: `12_second_training_run/s10_fresh20.py`, the 852 never-used `public_2601` scenes with no mention beyond the suite listing and no local copy on 2026-10-09.
  - Minus the 20 fresh scenes.
  - Minus any scene whose scene file is now in the local cache.
- **Draw:** sort the scene ids and take `numpy.random.default_rng(0).permutation` of them. The first one is the replacement.
- **If it also ends without metrics** for a simulator reason (not a driving event), take the next one in the same permutation, and record it here.
- **The new scene:**
  - joins the queue's scene list and CSV in place of 9e6753c2, as a new pin recorded here;
  - is downloaded from the NuRec dataset, with its size and sha256 checked against the repository record;
  - is stored on `{local_store}/new40_scenes/usdz/` and linked into the scene cache (`all-usdzs/<uuid>.usdz`).
- **Drives:** both arms are run on it. The rule is unchanged and is applied to the 20 scenes as replaced.

### Replacement drawn and fetched

- **Pool:** 852 scenes; after removing the 20 fresh scenes and the locally cached scenes, 832 remain.
- **Draw:** `default_rng(0).permutation`, with numpy 2.5.2. The first five are `44409f00 36509650 81b0e06d dd427ac5 c42044c0`, in order (`new40_scenes/replacement_order_seed0.csv`).
- **The replacement is `clipgt-44409f00-9b06-4103-86d7-d48e41a539e0`** (uuid `6ffea280…`, build 26.1.112).
  - Downloaded from NuRec, 1,871,395,562 bytes. Its sha256 `8f92e1c9…` equals the repository record (`new40_scenes/replacement_download.json`).
  - Stored on data8t, with a symlink in `all-usdzs/`.
  - The `metadata.yaml` uuid and build both match.
- **New pins:**
  - scene list `out/cl20_scenes_20_v2.txt` (sha256 `655a3149…`);
  - CSV `sim_scenes_new20_v2_26p01.csv` (sha256 `350fe886…`).
  - The queue and `src/s20_table.py` now read these. Nothing else changed.
- **The 12 scenes already driven are unchanged.** Their runs used the v1 CSV, which has the same rows for those scenes.
- **The failed run** is kept as `runs/memory_carry_9e6753c2.simfail_route_foldback`.

### Replacement, first attempt: setup error, retried on the same scene (06:01 IST)

- **What failed:** the first carry drive on 44409f00 stopped at startup with `FileNotFoundError: {container_scene_cache}/scenesets/…/6ffea280….usdz`.
- **Cause, my setup:** the scene file had been put in `all-usdzs/` as a symlink to data8t. The simulator containers mount only the scene cache, so the link could not be followed. The `--preflight-only` check does not open the file inside a container, so it passed.
- **Not a fault of the scene,** so the replacement rule's "next scene" clause does not apply. The same scene is retried.
- **Fix:**
  - the symlink was replaced by a real copy, sha256 `8f92e1c9…` (equal to the repository record);
  - the copy is on the local HDD scene cache, beside the other 53 scene files (71 GiB free there);
  - the data8t copy is kept.
- **The failed folder** is kept as `runs/memory_carry_44409f00.harnessfail_symlink_not_mounted`.
