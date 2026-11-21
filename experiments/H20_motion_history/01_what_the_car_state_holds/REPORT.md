# Car-state history for Drive-JEPA: read-only check before approach 4

shakaal, 2026-11-17. Read-only: no code, config, checkpoint or data was changed, and nothing was trained or simulated. Two small CPU reads (`taskset -c 0-1 nice -n 19`, alpasim venv, `python -I`): the training manifest (`anchor_t`) and 80 training-sample files (their `ego_status`). Scripts are in the session scratchpad, not kept.

**Confidence words.** *certain*: read in the code that runs, or measured today. *likely*: read in a report or inferred from code without running it. *unknown*: not found; what I looked at is listed.

**Path abbreviations**
- `RD` = `Paddock/paddock-sid/Alpasim/analysis_3.0/Retrain-data/`
- `G` = `Gallops/Gallops-sid/LORA Refiner/Generator-Joint-PAI-alpasim/` (the 0.1.8b training lane)
- `AD` = `Paddock/paddock-sid/Alpasim/adapters/drivejepa018bp06/` (the native 0.1.8b-p06 AlpaSim driver). `adapters/drivejepamemory/` (the approach-3 driver) has the same `planner.py` apart from the memory hooks (`diff` checked), so everything said about `AD/planner.py` also applies to it.
- `UP` = `Paddock/paddock-sid/Alpasim/adapters/drivejepa/src/navsim_v2/navsim/agents/drive_jepa_perception_based/` (the vendored upstream model)
- `PM` = Paddock `origin/main` at `73e9ac6a` (2026-11-16), read with `git show`. The checked-out Paddock tree is on an old branch (`alpasim-20hz-results`), so I did not use it.
- `A3` = `Gallops/Gallops-sid/memory/3_generator_scorer_memory_PAI/`

**Car-state row layout** (11 values, the same in training and driving): `[x, y, heading, vx, vy, ax, ay, command one-hot ×4]`. Built by `UP/drive_jepa_features.py:79-87`, with the slot meaning written down at `RD/s3_5_smoke/train_patch/dataset_s35.py:253`.

---

## 1. Car state today

### 1a. Which car-state fields reach the 0.1.8b model

**Answer.** One row of 11 values reaches the model. The model's only car-state input is `hist_encoding = Linear(11 → 256)`, applied to the newest row (`UP/drive_jepa_model.py:32, 51, 61`). The row holds:

| slot | field | units | frame | what reaches 0.1.8b |
|---|---|---|---|---|
| 0–2 | x, y, heading | m, m, rad | the current rig frame (rear axle, x forward, y left). In the newest row these are always 0, because it is the current pose relative to itself | 0, 0, 0 |
| 3–4 | vx, vy | m/s | the rig frame at that moment | the real values |
| 5–6 | ax, ay | m/s² | the rig frame | **always 0.0** (see 1c) |
| 7–10 | driving command (left, straight, right, unknown) | one-hot | – | the real value |

So in practice the model learns from velocity (2 values) and the command (4 values). The pose slots are always 0 in the newest row, and the acceleration slots are forced to 0.

**Training path:**
1. `RD/s1_converter/s1_convert.py:120-126` (`sample()`) builds 2–4 rows from the Physical AI AV egomotion. `RD/s3_assembly/s3_convert.py:9, 24` reuses that function for the full training set.
2. The rows are written to `pad_feature.gz` (`s1_convert.py:163`) and copied unchanged into the slim npz files (`dataset_s35.py:181, 190, 228`).
3. `dataset_s35.CacheOnlyDataset._finish` cuts to the newest row and zeroes acceleration (`:257-259`).
4. `G/s1_build/src/pai_data.py:75-79` (`planner_dataset(..., zero_accel=True)`) is the loader the trainer uses (`G/s9_018b_train/src/train_018b.py:55` via `s2_train/pai_train`).
5. `default_collate` stacks the samples into batches (`train_018b.py:355` and other call sites).
6. `G/s8_018b_build/src/pai_model_b.py:107` and `G/s1_build/src/pai_model.py:198, 205` take `feats["ego_status"][:, -1]` and feed it to `hist_encoding`.

**Driving path (native):** `AD/planner.py:727-729` (`build_features` → `_build_ego_status` → `_model_ego_status`) → worker → `UP/drive_jepa_model.py:51`.

**Evidence:** paths above. **Confidence:** certain.

### 1b. Where the history is cut to the newest row

**Answer.** It is cut in two places. Neither is a config option.
1. **The loader cuts it, always.** `RD/s3_5_smoke/train_patch/dataset_s35.py:257`: `features["ego_status"] = features["ego_status"][-1:].clone()`. This is hardcoded with no flag. The S3.5 wiring table lists it as "extra | one ego_status row | … | keep only the newest row | **always**" (`RD/s3_5_smoke/REPORT.md:32`). The reason recorded at `dataset_s35.py:255-256` and `REPORT.md:58-59` is: "our converter writes 2-4 ego_status rows per sample … which the default collate cannot stack. The model reads only the newest row … so keep that one."
2. **The model cuts it again, always.** `UP/drive_jepa_model.py:51`: `ego_status = features["ego_status"][:, -1]`. Upstream does this too. The training wrapper repeats it (`pai_model_b.py:107`, `pai_model.py:205`), and so does the approach-3 worker (`adapters/drivejepamemory/memory.py:245`).

So the collate step is not where the cut happens: after the loader there is only one row left to stack. At drive time the native driver builds the full stack of rows (`AD/planner.py:815-852`, "the full stack is built because that is what the interface specifies and what a future variant would consume"), and the model then uses only `[:, -1]`.

**Confidence:** certain.

### 1c. Where acceleration is set to zero, and why

**Lines that zero it:**
- Training: `RD/s3_5_smoke/train_patch/dataset_s35.py:258-259`
  ```python
  if self._zero_accel:
      features["ego_status"][..., 5:7] = 0.0
  ```
  The flag defaults to `zero_accel: bool = True` (`:40`) and is passed explicitly as `zero_accel=True` by `G/s1_build/src/pai_data.py:78`. It is a flag of the loader. The 0.1.8b recipe does not expose it as a config setting: `G/s8_018b_build/src/build_018b.py:13` says "acceleration zeroed … inherited unchanged from 0.1.8 (asserted)".
- Driving, model input: `AD/planner.py:740-744` (`out[..., K.EGO_ACCEL_SLOTS[0] : K.EGO_ACCEL_SLOTS[1]] = 0.0`), switched by `AD/constants.py:113` `ZERO_ACCELERATION = True`.
- Driving, comfort re-score: `AD/planner.py:942-946`, switched by `AD/constants.py:119` `ZERO_MOMENTUM_ACCELERATION = True`.
- Approach-3 trainer: `A3/reports/BUILD3_MODEL_TRAINER.md:78` says slots 5:7 of the cached row are zeroed "as the model's loader does".

**What the records say about why** (exact wording):
- No comment in the code gives a reason. The docstring only says what the line does: `dataset_s35.py:54` ":param zero_accel: [S35-1c] zero the two acceleration slots of ego_status at load time".
- `paddock-sid/` is excluded from git (`Paddock/.git/info/exclude:10`), so `git blame` is not possible on the working copy. The Gallops commit that first tracked the patch, `4cfdb44` (2026-10-13, "drivejepa-retrain: the training set, a small overfit check and the first training run"), gives no reason in its message. `git log -S zero_accel` finds only that commit and later ones that copy it (`e393509`, `fedfd16`, H17 `0256252`/`c7f08e6`).
- The S3.5 report says only that it was ordered: "**Per brief**: 30 epochs, encoder frozen, extended vocabulary, acceleration slots zeroed, init from the shipped checkpoint." (`RD/s3_5_smoke/REPORT.md:142`; the same line is at `Gallops/experiments/GC05_physicalai_retrain/05_small_overfit_check/REPORT.md:142`). I did not find the brief itself.
- The nearest thing to a reason is in the Gallops PR text (`paddock-sid/all md files/PR_BODY_GALLOPS.md:9`): "…and the acceleration input had been implicated by earlier analysis of the simulator runs. This is the experiment that trained those out". The earlier analysis is `DRIVEJEPA_MODE4_JUNCTION.md:203-205`: "Zeroing acceleration alone (b) removes 30–45 % of the recorded slide on the three swerve scenes … the −8.1 m/s² acceleration slot is what turns the +1.0 s plan into a full stop".
- The converter report flags noise in this slot: "**F5 — acceleration slot is noisy.** It is a double backward difference of 10 Hz positions: |ax| > 4 m/s² on 8 anchors at steady cruise. Kept as instructed." (`RD/s1_converter/REPORT.md:240`). It does not say this was the reason for zeroing.
- The driver side gives a reason only for matching training: "The S4 weights were trained with these two slots at zero on every sample; feeding the reported acceleration would be an input they have never seen." (`AD/planner.py:736-738`). For the re-score: "Decided by Sid for the S5 trial" (`AD/constants.py:116`).

**Verdict:** **no reason recorded at the line, in the commit or in the config.** The only recorded basis is "Per brief", plus the PR's statement that earlier simulator analysis had "implicated" the acceleration input.

**Confidence:** certain for the lines and the wording. *Likely* that MODE4_JUNCTION is the analysis the PR means: the PR does not cite it by name.

### 1d. Was a multi-row history or a non-zero acceleration ever used?

**Multi-row history fed to the model: never.** Every Drive-JEPA forward I found reads only `[:, -1]`: upstream, the 0.1.6–0.1.8c training wrappers, and the approach-3 worker (1b). Several drivers build more rows, but the model discards them:
- the native driver builds the full stack (`AD/planner.py:815-852`);
- the Paddock node builds 4 rows at 0.5 s (`PM src/paddock/adapters/models/drivejepa/policy.py:116-117, 326-384`);
- the converter writes 2–4 rows (`s1_convert.py:120-126`).

`git log -S` for `ego_status"][:, -2` and similar patterns found nothing (Gallops, Paddock).

**Non-zero acceleration fed to the model: yes, but only to the released (upstream NAVSIM) checkpoint. Never to a checkpoint retrained on Physical AI AV (0.1.6 and later).**
- 0.1.4 / 0.1.4b drove the released checkpoint with the reported acceleration, which is zero during the 3 s force-GT warm-up and real after it (`all md files/DRIVEJEPA_MODE4_ACCEL.md:20`). Their 30-scene results: 0.1.4 0.4209, 0.1.4b 0.5433.
- The Paddock node feeds forward acceleration from the CAN bus, never zeroed, to whatever checkpoint it serves (`PM …/policy.py:61-68, 386-402`; bridge `PM …/alpasim/drivejepa_rhythm.py:381, 410`). The approach-3 recon notes that the bridge has "no zeroed acceleration" (`A3/reports/APPROACH3_RECON.md:449`).

**What was tried, and what happened:**

| test | what it fed | result | evidence |
|---|---|---|---|
| Acceleration counterfactual, released checkpoint | the real acceleration in place of zeros during warm-up | "VERDICT: NO. The zero-acceleration input is exonerated. The slow-plan behaviour is the model's." | `all md files/DJ014_ACCEL_COUNTERFACTUAL.md:7-8` |
| MODE4_JUNCTION, released checkpoint | the reported acceleration vs zeroed, at slow-down onsets | zeroing removes 30–45 % of the slide on the swerve scenes; −8.1 m/s² turns a plan into a full stop | `DRIVEJEPA_MODE4_JUNCTION.md:203-205` |
| MODE4_ACCEL, released checkpoint | −0.5 … −5 m/s² written into the slot on cruising ticks | the menu moves gradually (median −1.2…−2.3 m/s for +1 → −5 m/s²); 1 of 6 runs away | `DRIVEJEPA_MODE4_ACCEL.md` (A1) |
| KITScenes input checks, released checkpoint | acceleration zeroed or smoothed | ≤ 0.07 m change in the pick | `Gallops/experiments/GC01_rerank_kitscenes/CARD.md:41` |
| 0.1.6 closed loop | weights + mask + zeroed acceleration, all changed at once | "Not tested: which of the three changes … produced which effect" | `GC05…/07_closed_loop_test/REPORT.md:192, 221` |

**Car-state history tried offline, never inside the model:**

| study | data | result | evidence |
|---|---|---|---|
| NAVSIM, image-memory step 2 control | ridge probe | the gain comes from the car state's past: k4 −0.0072, CI excludes 0 | `Gallops-sid/memory/1_camera_memory_H05/reports/MEMORY_STEP2C_CONTROL.md` (via memory note) |
| H15 refit | same | k4−k1 −0.0278, k8−k1 −0.0241 (speed) | memory note |
| Approach 3, link-2 | Physical AI AV; 4 s ego history (speed + yaw rate per 0.5 s, logged speed, flag) | "PH (+ ego history)" beats "P" on every target, for example 0.9951 vs 0.9869 | `A3/reports/LINK2_TEST.md:42-43`; `A3/src/s10_probe.py:52-64` |
| Approach 3, ceiling test | same | ego history: speed at +1 s **+57.1 %**, +2 s +39.0 %, +3 s +25.7 %; "15 to 40 times larger" than the true past view | `A3/reports/CEILING_TEST.md:92-97` |

**Closed-loop runs that fed history: none found.** I searched the A3 CL12/CL18 reports, the `G/s4`/`s10` closed-loop reports, the 0.1.4b, S5 and retrain S5/S10 notes, and `git log -S` in Gallops and Paddock.

**Confidence:** certain for "the model never read more than one row". *Likely* for "no closed-loop run fed history": the search was thorough, but absence is hard to prove.

---

## 2. What the data can supply (training side)

### 2a. Past car-state rows available per moment in the raw Physical AI AV files

**Answer.**
- **Source:** `labels/egomotion.offline/egomotion.offline.chunk_XXXX.zip` → `<uuid>.egomotion.offline.parquet`, with columns `timestamp, x, y, z, qx, qy, qz, qw` (`s1_convert.py:68-69, 86-87`).
- **Rate and frame:** nominally 10 Hz (0.1 s), in the clip's own local frame (`APPROACH3_RECON.md:149`: "pose step > 0.15 s (1.5 × the nominal 0.1 s)" counts as a gap; `s1_convert.py:66`: "egomotion.offline -> rig pose T(t) (clip frame <- rig)").
- **No velocity or acceleration channel is stored.** The converter derives velocity as a backward difference of positions, rotated into the rig frame at each pose (`:70-71`). It derives acceleration as a backward difference of that velocity (`:72`).
- **Coverage:** every pose of the 20 s clip is in the file. So a moment at `anchor_t` has every pose from clip start up to `t`: `anchor_t / 0.1 s` past poses, up to 160 at t = 16 s.
- **Stored today, per training sample:** only 2–4 rows at 0.5 s (`HIST = 4`, `s1_convert.py:25, 121-126`). Measured on 80 slim samples: 2 rows on 39, 4 rows on 41. The 2-row samples were the clips' first moments.

Example, `d760c449_08500` (4 rows, oldest first). Positions are in the current rig frame, and the newest row is at 0:
```
[-5.541 -0.174  0.019  3.521 -0.042  0.427 -0.224  0 0 1 0]
[-3.731 -0.117  0.046  3.683  0.028  0.555  0.214  0 0 1 0]
[-1.875 -0.041  0.037  3.731 -0.029  0.209 -0.414  0 0 1 0]
[ 0.    -0.    -0.     3.715  0.023 -0.594  0.071  0 0 1 0]
```
The slim files still hold the converter's non-zero acceleration. It is zeroed only at load.

**Confidence:** certain (code + measured).

### 2b. Can we get 8 rows at 0.5 s covering the last 4 s?

**Answer: yes, from the raw egomotion, but not from anything already stored.**
- The egomotion is at about 0.1 s, so 0.5 s rows are every 5th pose. The converter already places rows at `t − k·0.5 s` by interpolation: slerp for rotation, linear interpolation for position (`s1_convert.py:73-75, 122, 125`).
- **The arithmetic:** 8 rows at 0.5 s including the current one span t−3.5 … t, which is 3.5 s. Covering the full last 4 s takes 9 rows (t−4.0 … t), or 8 past rows plus the current row.
- **To build it:** re-run `sample()` with `HIST = 8` or 9 over the egomotion parquet, or write a separate small builder that writes only the ego rows (no images) per `sample_id`. Either way it means reading the egomotion zips again. The stored files (pad_feature, slim, token cache) hold at most 1.5 s.
- **What would change:**
  - the velocity in past rows is the 10 Hz backward difference at that pose (`Ego.dyn`, `:76-78`), not an average over the 0.5 s step;
  - the acceleration slots are noisy (`REPORT.md:240`).

**Confidence:** certain that the data exists; *likely* for the cost (not run).

### 2c. Moments with less than 4 s of history

**Answer: there is no flag and no padding. Missing rows are simply dropped.**
- `s1_convert.py:123`: `if tj < ego.ts[0]: continue  # rows before the clip start do not exist; forward reads only [-1]`.
- A moment needs only `t − 0.5 s ≥ ts[0]` (`:105`), so it has 2, 3 or 4 rows. The row count is not stored in the manifest: its columns are `sample_id, clip_uuid, source, type, split, has_label2, direction_exempt, anchor_t, converter_commit, grader_commit, vocab_sha256, group, country, speed, command`, measured. The loader then throws the extra rows away (1b).
- **How many moments are short (measured on the manifest, 126,981 moments):** `anchor_t` runs from 0.5 to 16.0 s, median 8.0. 3,971 of 4,018 clips start at 0.5 s.
  - **23,828 moments (18.8 %) have anchor_t < 3.5 s**, so they cannot have 8 rows at 0.5 s.
  - 7,944 (6.3 %) have anchor_t < 1.5 s, so they cannot have even today's 4 rows.
- Mid-clip gaps are not detected in the ego rows. The 0.15 s gap rule exists only in the recon scan (`A3/src/b2_gaps.py`). Interpolation would bridge a gap silently (`s1_convert.py:73-75`).
- The approach-3 link-2 probe did add a "this lag exists" flag (`A3/src/s10_probe.py:56-63`), which is a pattern that could be reused.

**Confidence:** certain.

### 2d. The window ends at the current moment; no future rows enter it

**Answer: yes for the poses, velocity and acceleration in the car-state rows. The command slots, though, are taken from the human's future path in training.**
- **Row times:** `tj = t − (HIST − 1 − j)·STEP_US` with j ≤ HIST − 1, so `tj ≤ t` (`s1_convert.py:122`).
- **Positions:** `ego.T(tj)` interpolates between the two poses around `tj` (`:74-75`). For `tj ≤ t − 0.5 s` both poses are at or before `tj + 0.1 s < t`.
  - For the current row (`tj = t`) the interpolation can use the first pose after `t` (≤ 0.1 s ahead). But that is the anchor pose `Tn` itself (`:109`), and the row stores it relative to itself as (0, 0, 0). So nothing from the future reaches the row values.
  - The past-to-current transform is `P._relative_pose(ego.T(tj), Tn)` (`:125`): the past pose expressed in the frame at t. It uses only `T(tj)` and `T(t)`.
- **Velocity and acceleration:** `Ego.dyn(tj)` uses `i = searchsorted(ts, tj, side="right") − 1`, the last pose at or before `tj` (`:77`). `v_rig[i]` is `(p[i] − p[i−1]) / dt` and `a_rig[i]` is `(v[i] − v[i−1]) / dt` (`:70-72`): backward differences only.
- **The exception, the command one-hot:** it is computed from `route = every egomotion sample after t` ("Route stand-in = the human's own future path", `:110-113`). The same command is copied into every row (`:125`). So past rows would carry the command decided from the future at t, not the command at their own time.
  - At drive time the command comes from AlpaSim's route (`AD/planner.py:823-827`).
  - This difference already exists for the newest row and is recorded at `:110-111` ("KNOWN DIFFERENCE vs inference").
- **Training targets:** the labels are `T(t + k·0.5 s)`, k = 1…8 (`:135`). They are separate from the rows.

**Confidence:** certain (read in the code; I did not re-run it).

### 2e. Does the 88.6 GB frozen-encoder token cache need to change?

**Answer: the tokens do not need to change. The car state stored beside them is only the newest row, so it cannot supply history.**
- Per clip, the cache file holds `{sample_id, anchor_t, tokens (N, 512, 256), pooled (N, 33, 256), ego_status (N, 11), trajectory (N, 8, 3), has_label2}` (`A3/reports/BUILD2_CACHE.md:34`).
- The `ego_status` stored there is "the newest ego-status row (11 values, the converter's accelerations kept since the encoder never reads `ego_status`…)" (`:32`). The encoder input is two images only (`UP/drive_jepa_model.py:53-60`). `ego_status` enters only after the encoder, at `hist_encoding` (`:61`). So adding history does not invalidate any token.
- **Caveat:** the newest row's x, y and heading are always 0, so the cached rows of consecutive moments **cannot be chained into past positions**. A 4 s history needs either a separate side file of past rows per `sample_id`, built from the egomotion as in 2b, or derived quantities like the link-2 probe's (speeds and yaw rates from each earlier moment's logged waypoint 0, `s10_probe.py:52-64`).

**Confidence:** certain.

---

## 3. What the car can supply (drive side)

There are two drive routes.
- **(N) the native driver** `AD` / `drivejepamemory`. Every approach-3 and 0.1.8b closed-loop number came from this route (`A3/src/s18_run_memory_scene.sh:4, 70`).
- **(P) the Paddock bridge + policy node** at `PM`. The approach-3 recon says a 0.1.8b run through it "cannot be a ranked Paddock run" (`APPROACH3_RECON.md:451`).

### 3a. Car-state fields passed at drive time

**(N) native:** `AD/planner.py:815-852` builds one row per pose the scaffold has received:
- **x, y, heading:** each past pose in the current rig frame (`_relative_pose`, `:834`).
- **vx, vy:** `DynamicState.linear_velocity` x, y, in m/s, rig frame (`:835-839`). At handover, if the reported value is exactly zero, the newest row gets the speed derived from the last two poses instead (`:746-813`).
- **ax, ay:** `DynamicState.linear_acceleration` x, y, in m/s² (`:847`), then zeroed for the model (`:727-729, 740-744`).
- **command:** a one-hot from the route (`:823-827`).

**(P) Paddock:** the bridge sends one observation per 0.5 s replan:
- `canbus[13]` = vx (forward speed, signed) and `canbus[7]` = ax, which is the reported value and is not zeroed (`PM …/alpasim/drivejepa_rhythm.py:349-357, 409-411`);
- `ego2world` = the pose nearest the frame;
- `command`.

The node builds 4 rows at about 0.5 s from its own history: `ego_pose = relative_pose(frame.ego2world, current)`, `ego_velocity = [speed, 0]`, `ego_acceleration = [accel, 0]` (`PM …/models/drivejepa/policy.py:366-384`). Sideways velocity and acceleration are always 0 ("the model gets (vx, 0) and (ax, 0)", `drivejepa_rhythm.py:37-41`).

**Confidence:** certain.

### 3b. Does the drive side keep past states?

**(N) native: yes, the whole rollout.** The scaffold's `submit_egomotion` appends every pose and dynamic state the runtime sends ("The runtime sends *every* pose since the previous tick … so we extend rather than replace", `adapters/scaffold.py:553-578`). Nothing ever trims `poses` or `dynamic_states`: only images are capped, at `image_history` (`scaffold.py:539-540`). Every call gets them oldest first in `ObservationView.poses` / `.dynamic_states` (`:119-122, 324-325`).

So a 4 s buffer is already there. The natural place to cut it is `_build_ego_status` (`AD/planner.py:815-852`), which already loops over every pose and builds relative rows. It would need to resample to the 0.5 s grid as the converter does (by time, not by index), because the pose cadence at drive time is not 0.5 s.

**Other per-session state kept today:**
- the previous chosen plan for the comfort re-score (`AD/planner.py:947-978`, `_momentum_prev[session_uuid]`, kept ≤ 0.55 s);
- the approach-3 memory state, in the worker per `session_uuid` (`drivejepamemory/planner.py`, `_memory_control`; `memory.py:329`).

Neither keeps a car-state history: the re-score keeps only the last plan's simulated states.

**(P) Paddock: yes, about 1.5 s is used.** The node keeps up to 128 observations (`policy.py:215-240`) and selects 4 at 0.5 s by timestamp (`:326-338`). Through the AlpaSim bridge it receives one observation per 0.5 s, so 128 entries is about 64 s. Raising `HISTORY_FRAMES` (`:116`) would give a 4 s window there too.
- **Caveat:** each selected frame also brings its images (`_decode`, `:258`), so the car-state window and the camera window are tied together in this node.

**Not verified:** the exact pose cadence the AlpaSim runtime pushes. The scaffold ticks at 50 ms and the frames are 50 ms apart (`G/s8_018b_build/src/g7_adapter_check.py:84`), but I did not measure the pose spacing from a recorded rollout.

**Confidence:** certain for the buffer; *unknown* for the exact pose cadence.

### 3c. Training vs drive-time car state, side by side

| field | training (Physical AI AV converter) | native driver (N) | Paddock node (P) |
|---|---|---|---|
| x, y | past pose in the rig frame at t, m, from slerp/interp of 10 Hz egomotion (`s1_convert.py:125`) | past pose in the current rig frame, m (`AD/planner.py:834`), same `_relative_pose` function (the converter imports `drivejepa.planner as P`, `s1_convert.py:21`) | `relative_pose(ego2world, current)` (`policy.py:377`), a different function in a different repo; not checked for numeric equality |
| heading | relative yaw, rad, same function | same | Paddock's `relative_pose`/`yaw_of` (`policy.py:128-137`) |
| vx | backward difference of 10 Hz positions, rotated to the rig frame at that pose, m/s (`:70-71`) | reported `DynamicState.linear_velocity.x`, m/s; **zero during the force-GT warm-up** except the newest row's sentinel fix | `canbus[13]` = vx sent by the bridge (signed forward speed, m/s) |
| vy | same derivation, m/s | reported vy | **always 0** |
| ax, ay | backward difference of velocity, m/s², noisy; **zeroed at load** | reported, **zeroed** for the model and the re-score | reported ax, **not zeroed**; ay 0 |
| turn rate | not an input | not an input | not an input (`canbus[12]` holds yaw rate but the node does not read it into `ego_status`) |
| command | from the human's **future** path (route stand-in) | from the AlpaSim route, ≥ 20 m ahead, ±3 m | from the route via the bridge |
| row spacing | exactly 0.5 s by interpolation | **one row per received pose**, not 0.5 s | about 0.5 s, nearest received frame |

Units agree everywhere (m, rad, m/s, m/s²; no km/h anywhere). The frame convention agrees: rear axle, x forward, y left (`AD/planner.py:845, 874-876`; Paddock `CLAUDE.md` Trajectory convention).

Differences that matter for history:
1. **Velocity source.** Training uses differenced positions; driving uses the simulator's reported velocity, which is zero for ticks 1–60 (`DRIVEJEPA_MODE4_ACCEL.md:20`).
2. **Row spacing in the native driver.**
3. **The Paddock node's vy = 0 and un-zeroed ax.**
4. **The command's source.**

**Confidence:** certain for each field's code. The numeric equality of Paddock's `relative_pose` with the adapter's was not checked.

### 3d. How acceleration is obtained at drive time

**Answer:**
- **(N) native:** measured. It is the simulator's `DynamicState.linear_acceleration` (`AD/planner.py:847, 941`), which is exactly 0.0 through the warm-up up to and including the handover tick, and real after it (`DRIVEJEPA_MODE4_ACCEL.md:20, 37`). For 0.1.6 and later it is then overwritten with 0 for both the model and the re-score.
- **(P) Paddock:** the CAN-bus acceleration channel when the source has filled it at least once in the episode. Otherwise it is the least-squares slope of the recorded speed over the last 0.5 s (`PM …/policy.py:61-68, 107-109, 386-402`). The AlpaSim bridge always fills `canbus[7]` with the reported ax (`drivejepa_rhythm.py:353`), so through AlpaSim it is the measured value.
- **Training:** differenced from position twice; never measured.

**Confidence:** certain.

---

## 4. The existing "momentum" bonus (comfort re-score)

### 4a. Where it is applied, its formula and its weight

- **Applied in:** `UP/drive_jepa_model.py:87-88, 98-151` (`calibrate_score`), called inside `forward` when `past_ego_simulated_states` is given. It is fed by `AD/momentum.py` (worker) and `AD/planner.py:915-978` (keeps the previous chosen plan per session).
- **Formula** (`drive_jepa_model.py:149`): `pdm_score[:, idx] = (14.0 * pdm_score[:, idx] + 2.0 * two_frame_comfort) / 16.0`. Here `pdm_score` is sigmoid(last logit) (`:85`), and `two_frame_comfort` ∈ {0, 1} is `ego_is_two_frame_extended_comfort` between the previous plan's simulated states and each candidate's simulated states, over their overlap (`:132-148`).
- **Weight:** 2/16 = 0.125 on comfort, 14/16 on the learned score. These are fixed constants.
- **When it is used:** only if the previous plan is ≤ 0.55 s old (`AD/constants.py:294` `MOMENTUM_MAX_AGE_S = 0.55`; `momentum.py:128-131`). It is on by default (`constants.py:279, 284`, `DRIVEJEPA_MOMENTUM`).
- **The car state it simulates from:** the newest row's velocity and **zero acceleration** (`planner.py:935-946, 955-956`; `momentum.py:97-120`).

### 4b. Decision-time only and not learned?

**Yes, both.**
- It has no parameters (two constants, 14 and 2).
- It runs only when `past_ego_simulated_states` is not `None`.
- The 0.1.8b training code sets `feats["past_ego_simulated_states"] = None` (`G/s1_build/src/pai_model.py:305`). No training file calls `calibrate_score` (grep over `G/s1_build`, `s8`, `s9`). The `momentum_update` hits in training code are an unrelated "answer key" EMA (`pai_model.py:242`; `pai_model_b.py:161-165`).

**Confidence:** certain.

---

## Risks for approach 4

1. **The force-GT warm-up zeros would fill a 4 s window at handover.** AlpaSim reports velocity and acceleration as exactly 0 on ticks 1–60 (3 s) while the poses move (`DRIVEJEPA_MODE4_ACCEL.md:20`; memory note "force-GT zeroes DynamicState ticks 1..60"). The native driver corrects only the newest row (`AD/planner.py:758-760`: "rewriting history rows would put derived numbers where the interface promises reported ones"). With 8 rows, the first plans after handover would see up to 6–7 rows of "stopped" next to moving positions, a pattern training never contains. Fix: build the velocity of past rows from poses at drive time as training does, or use pose-only history.
2. **Velocity is derived differently in training and driving.** Training differences 10 Hz positions; driving reads the simulator's reported velocity. They agree on the newest row in steady driving, but a history amplifies any offset or lag. The converter reports the differenced acceleration as noisy (|ax| > 4 m/s² at steady cruise on 8 anchors, `s1_converter/REPORT.md:240`).
3. **Acceleration has a mixed record.** It is zeroed in training and driving with no recorded reason beyond "Per brief" (1c). On the released checkpoint a large negative value deepened stops (MODE4_JUNCTION) and amplified slow plans (MODE4_ACCEL, 1 of 6 runaway), while another study "exonerated" it (DJ014). Feeding a history of accelerations reintroduces exactly that channel, ×8. Keep the slots zeroed in history rows, or derive acceleration from history speeds the same way on both sides.
4. **The command in past rows is taken from the future in training.** It is copied into every row from the human's future path (`s1_convert.py:110-113, 125`). Repeating it 8 times gives it more weight, and at drive time it comes from AlpaSim's route. Consider leaving the command out of history rows.
5. **18.8 % of training moments have under 3.5 s of past** (23,828 of 126,981), and the converter silently drops missing rows (`s1_convert.py:123`). A fixed-size input needs explicit padding plus a valid flag. At drive time the first 4 s after session start are the warm-up, so they are also padded or zero (risk 1).
6. **The stored data holds at most 1.5 s.** The token cache holds the newest row only; slim/pad_feature files hold 2–4 rows. 4 s needs a new pass over the egomotion zips. Mid-clip gaps are not flagged in rows: interpolation bridges them silently.
7. **The native driver's rows are one per pose, not 0.5 s.** `_build_ego_status` returns every received pose (`AD/planner.py:832`). History must be resampled to exact 0.5 s offsets by time, with the same interpolation as the converter, or rows will sit at different time offsets from training.
8. **The Paddock route differs from training on top of that.** vy = 0, ax not zeroed, a different relative-pose function, and its 4-row window is tied to camera frames. An approach-4 checkpoint served there would see inputs unlike its training. Approach-3 runs avoided this by using only the native driver.
9. **Earlier evidence on what history adds is offline and linear.** Ego history is a strong predictor of the human's next speeds (+57 % at +1 s, `CEILING_TEST.md:92`), and on NAVSIM k4 beat k1 (MEMORY_STEP2C / H15). No closed-loop run has ever fed history, and nothing shows it helps the pick. Part of the gain may be the model learning to copy the human's speed trend, which in closed loop can lock in a slow-down (the stall pattern noted in MODE4_JUNCTION: "menu slide is STATE-DRIVEN via ego_status feedback").
10. **Upstream's `hist_encoding` is Linear(11 → 256) on one row** (`UP/drive_jepa_model.py:32`). Feeding 8 rows changes the shape of the first generator layer (a new input of 88 values or a sequence encoder), so 0.1.8b's generator weights would not load unchanged into that layer.

---

## What I looked at and could not answer
- **The brief that ordered "acceleration slots zeroed".** Searched: `RD/*/REPORT.md`, `paddock-sid/all md files/*`, `paddock-sid/Alpasim/loop/goals/retraining1.0.md`, `paddock-sid/Alpasim/pr_phase1/gallops_docs/drivejepa-retrain.tmpl.md`, the Gallops commit `4cfdb44` message and `git log -S zero_accel` (Gallops, Paddock). The brief itself was not found on disk.
- **The AlpaSim pose cadence at drive time.** Not measured; it would need a recorded `rollout.asl` or `driver.log` read.
- **Numeric equality of Paddock's `relative_pose` with the adapter's `_relative_pose`.** Not checked.
