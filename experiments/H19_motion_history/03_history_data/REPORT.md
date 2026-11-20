# Approach 4, step 1: car-state history side file and shared builder

shakaal, 2026-11-17 04:00–04:40 IST. Data and checks only: no model change, no training, no simulation, and no edits to the converter, adapters or Paddock. Everything ran on CPU, pinned with `taskset -c 0-1 nice -n 19`, in the alpasim venv (`{alpasim_bulk}/alpasim-venv`, Python 3.12.14, scipy 1.18.1, numpy 2.5.2). That is the same interpreter the AlpaSim driver runs (`alpasim/.venv` resolves to it). The memory gate passed at the start: 5-minute load 7.6, 93 GB free. `{other_user_folder_1}` and `{other_user_folder_2}` were not touched.

Path abbreviations as in `READONLY_CARSTATE_HISTORY.md`, plus:
- `MH` = `Gallops/Gallops-sid/memory/4_motion_history/`
- `STAGE` = `{staging}/` (see the warning below)

Confidence words: **certain** (measured or read in code that ran), **likely**, **unknown**.

> **The side file is NOT on the NAS yet.** Every write to `{nas}` failed from this host with `OSError: [Errno 30] Read-only file system`. That happened even with the session sandbox off, and in every folder tried (`{nas}`, `{nas_store}`, `{alpasim_store}`, `{dataset_store}`). The mount itself says `rw` (nfs4), the folders are mode 777, reads work, and the NAS is 95 % full (`df`: 20 T of 21 T). A server-side read-only state is likely, but I did not confirm it. I did not try to fix the NAS. The outputs are small (47 MB side file + 32 MB diagnostics), so they were written to `STAGE` instead, through the `MH_OUT` override (`MH/src/build_side_file.py:34-37`). To move them once the NAS accepts writes:
> `rsync -a STAGE/ {nas_store}/memory/4_motion_history/` (both folders, `ego_history/` and `diag/`).

---

## 1. The shared builder: `MH/src/ego_history.py`

**What it does** (`ego_history.py:89-104`, `build_history(ts_us, T_list, t_us, T_ref=None) -> (8, 6) float32`):
- Rows are at t−4.0 … t−0.5 s, oldest first. Each row is `[x, y, heading, speed, turn_rate, valid]`.
- **x, y, heading:** `_relative_pose(pose(tj), T_ref)` (`:102`).
- **speed:** the planar distance between `pose(tj)` and `pose(tj − 0.5)`, both in the rig frame at t, divided by 0.5.
- **turn_rate:** the heading change over the same pair, wrapped to (−π, π] (`:83-86`), divided by 0.5 (`:103`).
- **valid:** if `tj − 0.5 < first pose`, the row stays all zero with valid 0 (`:100-101`).
- **Poses only.** No velocity or acceleration input exists in the signature.
- **Interpolation** is the converter's: scipy `Slerp` on rotations, `np.interp` per axis, time clipped to the pose range (`_Poses`, `:52-66`, = `s1_convert.py:73-75`).
- **Leak guard:** poses with timestamp > t are dropped before anything is computed (`_kept`, `:74-80`).

**Importable from both sides without changing either: certain.** It imports only numpy and scipy. Importing `drivejepa.planner` or `s1_convert` was not an option, because they pull grpc, the scaffold and the model harness. So `_yaw_of` and `_relative_pose` are **copied verbatim** (`:40-48`). Test T2 proves the copies are identical to the real functions (below). This is the one deviation from "via the same `_relative_pose` the converter imports": the code is the same, checked, rather than an import.

**Tests** (`MH/src/test_ego_history.py`; output `MH/logs/test_ego_history.log`, run on clip `00040136…`, 40 anchors). All pass:

| test | what it checks | result |
|---|---|---|
| T1 leak | passing all poses vs only poses ≤ t vs all poses with every pose after t set to NaN (training-style and drive-style calls) | max abs diff **0.0** over 40 anchors |
| T2 relative pose | source text of `_yaw_of` / `_relative_pose` in `adapters/drivejepa014/planner.py` and `adapters/drivejepa018bp06/planner.py` (pulled by `ast`) vs the copies; 200 random pose pairs | text identical in both; max abs diff **0.0** |
| T3 interpolation | `_Poses.at` vs the converter's own `Ego` class (its source pulled from `s1_convert.py` by `ast` and run, not retyped), 997 times across the clip | max abs diff of the 4×4 matrices **3.3e-16** |
| T4 shape / validity | (8, 6) float32, rows before the first pose are all zero with valid 0, all values finite | 0 failures |

### Flag 1: the reference frame at t needs a pose up to 0.1 s after t (does not fit the design as written)

Physical AI AV egomotion timestamps are **not on the 0.5 s grid**. In clip `00040136…` they run 0, 91,089, 191,083, …; only 2 of 202 are multiples of 0.5 s (measured). The converter's frame for a moment, `Tn = ego.T(t)` (`s1_convert.py:109`), therefore interpolates between the last pose before t and the **first pose after t**, which is up to 0.1 s later. The current ego row and the labels are expressed in that frame (`:125, 135`).

"Rows use only poses at or before t" cannot also mean "x, y, heading in that same frame". What I built:
- **`T_ref` is an optional argument** (this is the deviation in the signature).
- **Training** passes the converter's frame, `converter_reference()` (`ego_history.py:69-71`; `build_side_file.py:62-63`), so history sits in the labels' frame. Every history row still reads only poses ≤ t: T1 holds with `T_ref` fixed. The frame itself adds nothing the existing path doesn't already use.
- **Driving** passes nothing, and the frame is the newest pose ≤ t, the pose the AlpaSim driver anchors on (`AD/planner.py:822`).

How far apart the two frames are, measured on all 126,981 moments (`out/checks.json: ref_frame_offset`): **median 0.27 m, p95 1.16 m, max 3.81 m; yaw median 0.0004, p95 0.015, max 0.082 rad.** That is the car's travel in under 0.1 s. It is **not a training/driving mismatch** as long as the driver calls with `t_us` = the newest pose's timestamp: then "frame at t" is exactly that pose, as in training. See 4c. **Confidence:** certain (measured).

---

## 2. Training side file

**Inputs, all read only:**
- `RD/s3_assembly/manifest.parquet`: 126,981 moments, 4,015 clips.
- `RD/s3_assembly/clips.csv`: maps each clip to its chunk; 1,398 chunks.
- The egomotion zips the converter read: `{dataset_store}/s0_census/raw/labels/egomotion.offline/egomotion.offline.chunk_XXXX.zip` (= `s1_common.RAW0`, `s0_census/common.py:47`; `s1_convert.py:86`; `s3_convert.py` does not repoint it).

**Each moment** (`build_side_file.py:57-77`): t = `anchor_t × 1e6` µs, asserted to reproduce the sample id exactly as the converter forms it (`:61`; `s1_convert.py:162`). History is `build_history(ts, T, t, T_ref=converter_reference(...))`.

**Output** (in `STAGE` for now, see the warning above):
- `ego_history/<clip_uuid>.npz`: `sample_id` (N,), `anchor_t` (N,) float64, `history` (N, 8, 6) float32. Each file is verified on a round trip, then renamed into place (`:80-89`).
- `diag/<clip_uuid>.parquet`: per-moment diagnostics, not part of the side file.

**2a. Timing (written before the full pass):** chunk 931, the largest at 16 clips / 512 moments, took **0.9 s end to end = 0.054 s/clip → estimate for 4,015 clips 216 s = 0.06 h** (`MH/logs/side_file_time.log`). Far below 4 h, so I continued.

**2b. Full pass:** 3,999 clips in **262 s (0.07 h)** plus the 16 from the timing run. 4,015 files on disk. Progress was logged every 100 clips at 0.065–0.067 s/clip (`MH/logs/side_file_full.log`).
- **Completeness** (`out/checks.json: completeness`): 4,015 files; 126,981 moments; the sample-id set is identical to the manifest's; shape (126,981, 8, 6), float32, all finite.

**Confidence:** certain.

---

## 3. Checks (`MH/src/checks.py`, output `MH/out/checks.json`, log `MH/logs/checks.log`)

### 3a. Agreement with stored rows: **exact (0.0)**

These are the same 41 four-row slim samples as the earlier report (same draw: `default_rng(0)`, 40 folders, first and middle file). **Correction to that report:** 11 of the 41 are **validation** clips, not training clips. The slim folder holds 4,828 clip folders, 813 more than the training set (`val`/`val_full`). Their history was built on the fly with the same builder and the same zips, using the chunk from `s3_assembly/{val,val_full}/clips.csv`. The other 30 come from the side file.

Stored rows 0/1/2 (t−1.5/−1.0/−0.5) were compared with history rows 5/6/7, over 123 rows:
- **x, y:** max abs diff **0.0**, median 0.0 (246 values).
- **heading:** max abs diff **0.0**, median 0.0.

The values are bit-identical after the cast to float32, as expected: the interpolation, the frame and `_relative_pose` are all the converter's. **Confidence:** certain.

### 3b. Speed sanity: median 0.051 m/s, p95 0.246 m/s

History speed at t−0.5 vs the stored `vx` of the t−0.5 row, 41 samples:
- **Abs diff:** median **0.0508**, p95 **0.2456**, max 0.2913 m/s.
- **Against the stored planar speed** |(vx, vy)|: median 0.0509, p95 0.209.

**Why they differ:**
- The stored `vx` is a **0.1 s backward difference** of positions, taken at the last pose at or before t−0.5 (`s1_convert.py:70-71, 77`). It covers roughly [t−0.6, t−0.5].
- History speed is the **mean over [t−1.0, t−0.5]**.
- The two windows are centred about 0.2 s apart. So a car changing speed at a m/s² differs by about 0.2·a, plus 10 Hz position noise in the stored value. The largest differences are on the samples with large stored acceleration:
  - `7a4ffd49_08500`: vx 4.18 vs 3.89, ax +2.10;
  - `29897585_08500`: 4.27 vs 4.48, ax −2.32;
  - `8bfc7da8_08500`: stored vx −0.18 (reversing or noise at near standstill) vs 0.09. History speed is a distance, so it is never negative.

**Confidence:** certain for the numbers, likely for the explanation.

### 3c. Coverage: 95,203 moments with 8 valid rows; 31,778 (25.03 %) with fewer

| valid rows | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|---|---|---|---|
| moments | 95,203 | 3,977 | 3,973 | 3,969 | 3,973 | 3,970 | 3,972 | 3,973 | 3,971 |

**Not the expected ~18.8 % (flag 2).** The oldest row, t−4.0, needs the pose at t−4.5 s for its speed and turn rate. So a moment has all 8 rows only when `anchor_t ≥ 4.5 s`:
- moments with `anchor_t < 4.5`: **31,778**, exactly the fewer-than-8 count;
- moments with `anchor_t < 4.0`: 27,801.

Every clip's first pose is at 0 µs (`first_pose_us` min = max = 0), so coverage depends only on `anchor_t`. The 3,971 moments with 0 valid rows are the anchors at 0.5 s (one per clip). The per-moment diagnostics agree with the history file (`diag_agrees: true`). **Confidence:** certain.

### 3d. Mid-clip gaps: 16 moments, 40 rows, all in one clip

- **Moments** with a step > 0.15 s between consecutive poses anywhere in [t−4.5, t]: **16**, all in clip `1a8897cc…` (anchors 1.0–8.5 s). The largest step is 200,072 µs: a single missing pose.
- **Rows interpolated across such a step:** **40**, in those 16 moments. Kept, not dropped.
- **Valid rows whose time fell after the newest pose ≤ t** (that would mean holding a stale pose): **0**.

**Confidence:** certain.

### 3e. Value ranges over valid rows (872,865 rows)

| column | min | median | max |
|---|---|---|---|
| x (m) | −160.56 | −12.97 | 15.13 |
| y (m) | −34.61 | 0.0016 | 33.31 |
| heading (rad) | −3.006 | 0.00004 | 2.921 |
| speed (m/s) | 0.0001 | 7.83 | **40.63** |
| turn_rate (rad/s) | −0.847 | −0.00003 | 0.999 |

- **Speed > 40 m/s: 21 moments in 2 clips.**
  - `3a107de6_07500` … `_13000` (12 moments);
  - `aa5f29c2_01000` … `_05000` (9 moments).
  
  Both are type-A clips from Germany, and the manifest's own speed for them reaches 40.7 / 40.6 m/s (the manifest's maximum is 40.7, with 8 moments above 40). So these are real highway speeds, about 146 km/h, not artefacts.
- **|turn_rate| > 2 rad/s: 0 moments.** The largest is 0.999 rad/s, in `653ce0cd_04000` … `_06500`, a tight turn at 3.8 m/s.
- **x > 0 (up to 15.1 m):** past positions ahead of the car. This is expected while reversing or after a U-turn, where heading reaches ±3.0. Not investigated per clip.

**Confidence:** certain for the numbers.

---

## 4. Drive-side dry run (no simulation): `MH/src/drive_dry_run.py`, output `MH/out/drive_dry_run.json`

### 4a. Pose cadence: median 0.050 s, max 0.100 s

**Files:** the 0.1.8b-p06 runs of approach-3 CL18, read from the NAS:
- `{experiment_store_h17}/runs/cl18/018bp06/runs/drivejepa_018bp06_01d503d4/rollouts/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554/b15aa566-df9c-11f1-9d16-b5da731c4277/rollout.asl`
- the matching `…_89f2eeb8/…/b11fe596-dfbc-11f1-b10a-1df1ec79903c/rollout.asl`

**Method:** `driver_ego_trajectory` entries read with `alpasim_utils.logs.async_read_pb_log` and accumulated as the scaffold does (extend, then sort; `adapters/scaffold.py:558-578`). Quaternions are converted w-first (`AD/planner.py:192-202`).

| | 01d503d4 | 89f2eeb8 |
|---|---|---|
| poses (all unique timestamps) | 398 | 400 |
| egomotion messages | 395: 394 × 1 pose, **1 × 4 poses** (the first) | 397: 396 × 1, 1 × 4 |
| spacing median / p99 / max | **0.050 / 0.050 / 0.100 s** (min 0.014) | 0.050 / 0.050 / 0.100 s (min 0.011) |
| span | 19.9 s | 19.9 s |
| driver requests | 395, median 0.05 s apart | 397, 0.05 s |
| reported speed exactly 0 within the first 3.05 s | **56 of 60 poses** (last zero at 3.21 s); the first pose carries 2.12 m/s | **58 of 62** (last zero at 3.15 s); the first pose 11.08 m/s |

So the runtime pushes one pose per 50 ms tick after an initial batch of 4. That is finer than the 0.1 s training egomotion, so 0.5 s rows are interpolated on both sides. The force-GT zero velocity is confirmed in this run: only a pose-derived history avoids it. **Confidence:** certain.

### 4b. `build_history` on the pose stream: all checks pass on both rollouts

14 query times per rollout (relative to the first pose): 0, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0, **3.9**, 4.0, 4.5, 6.0, 10.0, mid-rollout and end − 0.05 s. Poses only, `T_ref=None`. Every call:
- returned shape (8, 6), float32, all finite;
- set the valid flags exactly as expected, i.e. 1 iff t − (k + 1)·0.5 ≥ first pose;
- left invalid rows all zero;
- gave bit-identical output with every pose after t set to NaN.

Valid flags, oldest first:
- t = 0 to 0.75 s: all 0;
- t = 1.0: `00000001`;
- t = 2.0: `00000111`;
- t = 3.0: `00011111`;
- t = 3.9: `00111111`;
- t = 4.0: `01111111`;
- t ≥ 4.5: `11111111`.

Example, 01d503d4 at t = 9.953 s, rows oldest first `[x, y, heading, speed, turn_rate, valid]`:
```
[-18.4488, 0.1388, -0.0102, 3.9602, -0.0010, 1]
[-16.3980, 0.1177, -0.0104, 4.1019, -0.0004, 1]
[-14.2932, 0.0956, -0.0108, 4.2097, -0.0008, 1]
[-12.1361, 0.0727, -0.0104, 4.3145,  0.0007, 1]
[ -9.8378, 0.0501, -0.0093, 4.5968,  0.0021, 1]
[ -7.4207, 0.0295, -0.0079, 4.8345,  0.0029, 1]
[ -4.9441, 0.0124, -0.0057, 4.9533,  0.0042, 1]
[ -2.3727, 0.0018, -0.0021, 5.1428,  0.0072, 1]
```
The car is accelerating from 4.0 to 5.1 m/s on a nearly straight road, and the x spacing is consistent with the speeds (e.g. 2.57 m over 0.5 s ≈ 5.14 m/s). **Confidence:** certain.

### 4c. Where the call would go in `AD/planner.py` (not edited)

- **Place:** in `build_features` (`AD/planner.py:674-730`), beside `ego_status` (`:727-729`), as a new key:
  ```python
  "ego_history": build_history(
      [int(p.timestamp_us) for p in observations.poses],
      [_pose_to_matrix(p.pose) for p in observations.poses],
      t_us=int(observations.poses[-1].timestamp_us),
      T_ref=current,            # = _pose_to_matrix(observations.poses[-1].pose), as _build_ego_status uses (:822)
  )[None]                       # (1, 8, 6)
  ```
  It goes beside `_build_ego_status` (`:815-852`), not inside it. That keeps the current 11-value row and its path untouched (the design's "current moment keeps its existing path").
- **Why `t_us` = the newest pose's timestamp:** then `T_ref` is exactly the frame of the current row (`:822, 834`) and of the plan's anchor. Using the newer camera frame's time (`newer[0]`, which `plan` uses as `model_t0_us`, `:1055, 1063`) instead would put history in a frame up to one tick (≤ 50 ms) away from the current row's.
- **Input:** `observations.poses` holds every pose of the session, untrimmed (`adapters/scaffold.py:571-574`). So nothing new has to be stored.
- **Cost:** about 2 ms per call (the 0.054 s/clip timing covers 32 moments).
- **Import:** `ego_history.py` must be on the driver's `sys.path`. It needs only numpy and scipy, which the driver venv has.
- **Worker:** nothing is needed for transport. `real_model.forward` turns every numpy feature into a tensor and passes it to `agent.forward` (`adapters/drivejepa/real_model.py:147-153`). The approach-4 model will have to read `features["ego_history"]`. The bridge sends the dict as is (`AD/bridge.py:249`). A float32 (1, 8, 6) array is safe across the driver/worker numpy versions, the same as `ego_status` today.
- **Momentum / memory:** untouched. `_momentum_request` reads only `ego_status[0, -1]` (`AD/planner.py:935`).
- **Approach-3 driver:** `adapters/drivejepamemory/planner.py` has the same `build_features`, so the same one-line addition applies there.

**Confidence:** certain for the locations and the transport, likely for the cost at drive time (not timed inside the driver).

---

## Blocking issues for training

1. **The NAS is read-only from shakaal.** The side file lives only in `STAGE` on the HDD, which is 99 % full (39 GB free). It has to be copied to `{nas_store}/memory/4_motion_history/` before any training job reads it, or the trainer must be pointed at `STAGE`. Someone with NAS access has to check why writes fail (likely full or read-only volume).
2. **Coverage is 25.0 %, not 18.8 %** (31,778 moments with < 8 valid rows; 3,971 with 0). The model build has to read the `valid` column, mask or ignore zero rows, and not treat a zero row as "standing still at the current pose". A decision is needed on whether moments with partial history train normally, are down-weighted, or are dropped (dropping removes all of the first 4 s of every clip).
3. **The drive call must use `t_us` = the newest pose time with `T_ref` = that pose** (4c). Otherwise history and the current row sit in frames up to 50 ms apart at drive time. This is a rule for the adapter change, which has not been written yet.

## Not blocking

- **The training frame uses a pose up to 0.1 s after t** (flag 1). This is the existing labels' frame, history rows themselves read only poses ≤ t (T1), and the frame offset is 0.27 m median. Documented in the builder.
- **History speed vs stored vx differ by 0.05 m/s median** (3b). This is the expected window difference, and stored vx is not used.
- **Gaps:** 40 rows in 16 moments of one clip interpolated across a 0.2 s step; no stale holds.
- **21 moments above 40 m/s:** real German highway clips, max 40.6.
- **11 of the earlier report's "41 training samples" are validation clips** (correction noted in 3a). The check used all 41.
- **The relative-pose functions are copied rather than imported** (T2 proves they are identical; re-run `test_ego_history.py` if either planner's copy ever changes).
- **Drive-time cost** is not measured inside the driver (≈ 2 ms per call offline).

## Files
- Code: `MH/src/ego_history.py` (sha256 `01a96fdb77a033e9…`), `test_ego_history.py`, `build_side_file.py`, `checks.py`, `drive_dry_run.py`
- Logs: `MH/logs/test_ego_history.log`, `side_file_time.log`, `side_file_full.log`, `checks.log`
- Outputs: `MH/out/checks.json`, `MH/out/drive_dry_run.json`
- Data: `STAGE/ego_history/` (4,015 npz, 47 MB), `STAGE/diag/` (4,015 parquet, 32 MB). Target `{nas_store}/memory/4_motion_history/` once the NAS accepts writes.
