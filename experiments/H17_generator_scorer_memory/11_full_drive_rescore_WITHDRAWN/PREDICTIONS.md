# Full-drive re-score: pre-registration

Written 2026-11-16 23:33 IST, shakaal. **No drive has been re-scored when this is written.** The NAS folders that hold the carry and 0.1.8b runs are unreadable at the moment ("Stale file handle"), so nothing from them has been read since the brief arrived. Only the scoring code (local disk) has been read.

## What the old score is (from code, for the record)

`s5_diagnosis/src/stepb.py:31-32` does not compute a score. `score(run)` returns `rollouts[0]` of the run's `aggregate/results-summary.json`, which AlpaSim's evaluation wrote at the end of the run. That score is built as follows (local source `Paddock/paddock-sid/Alpasim/alpasim/src/eval/src/eval/`, commit `affc2ea`):

1. **Window start:** `RemoveTimestepsBeforeEvent(eval_relevant > 0)`, at `aggregation/processing.py:39`.
2. **Window end, first cut:** `RemoveTimestepsAfterEvent(offroad_or_collision > 0)`, at `processing.py:66`.
   - It keeps everything up to and including the first tick with any collision (front, side or rear) or off-road.
3. **Window end, second cut:** `RemoveTimestepsAfterEvent(dist_to_gt_trajectory >= max_dist_to_gt_trajectory)`.
   - It keeps everything up to and including the first tick at 4.0 m or more from the human path.
   - Code: `aggregation/main.py:135-140`. The 4.0 comes from `wizard/configs/base_config.yaml:402`.
   - The modifiers run in that order (`processing.py:750-752`), so the window ends at whichever cut comes first.
4. **Per-drive aggregation** over the remaining ticks (`processing.py:756`):
   - `collision_at_fault` = max of (front or side), `processing.py:46-50`;
   - `offroad` = max;
   - `progress_clipped_rel` = last value of `progress_rel_to_total` (`processing.py:69-90`).
5. **Score:** `eval/aggregation/scene_score.py:75-92`.
   - It is 0 if `collision_at_fault != 0` or `offroad != 0`.
   - Otherwise it is `min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)`, or 1.0 when `gt_dist_traveled_m < 5.0`.
   - Thresholds come from `schema.py:179-185` and `base_config.yaml:406-407`.

**Why the window ends where it does:** the code comment on `max_dist_to_gt_trajectory` (`schema.py:168`) reads only "Maximum distance to the ground truth trajectory to consider for metrics". I found no report that chose 4.0 m for this lane; it is AlpaSim's default. The report will quote anything else found.

Consequence: a crash or off-road that comes after the car is 4 m from the human path is never seen by the score.

## Full-drive score (the only change)

- **Same code, same formula.** The evaluation's own functions are called (`processing.aggregate_and_write_metrics_results_txt` and `scene_score.score_rollout`, imported from the local source unchanged) on each run's own `metrics.parquet`.
- **The only change: both window-end cuts (steps 2 and 3) are removed,** so the window runs to the last metric tick.
- **Unchanged:** the window start (step 1), the per-drive aggregation rules, the thresholds and the formula.
- **Consequences, stated before computing:**
  - an at-fault collision or off-road anywhere after the start gives 0;
  - progress is the last value of the whole drive;
  - a hit from behind no longer freezes progress, and stays non-failing, as before.
- **Sanity check:** with both cuts kept, the code must reproduce every old score (`results-summary.json` `rollouts[0].score`) to 1e-9, for every drive below. If any one fails, no full-drive number is reported for that drive, and the failure is reported instead.

## Drives

| set | drives | runs |
|---|---|---|
| carry | 30 | 12 from the 12-scene check (`cl12/runs/memory_carry_*`) + 18 from CL18 (`cl18/runs/memory_carry_*`) |
| 0.1.8b pass 0.6 | 30 | 12 as on the comparison page (4 image-check re-runs, 8 from 2026-10-31) + 18 from CL18 |
| reference only | 12 reset + 30 for 0.1.7 | reset: `cl12/runs/memory_reset_*`. 0.1.7: `s5_trial_017`, flagged **earlier image** |

**Per drive, the report lists:**
- the old score and the full-drive score;
- the window end;
- the last metric tick;
- every event over the whole drive: each episode of off-road, front collision, side collision and hit from behind, with onset time after handover.

## Decision (all 30 scenes, carry vs 0.1.8b, full-drive score)

Let d = mean over the 30 scenes of (carry − 0.1.8b), using full-drive scores. A **win** is a scene where carry − 0.1.8b > 0.1; a **loss** is a scene where it is < −0.1.

- **KEEP OPEN** if d ≥ +0.033 **and** wins > losses.
- **CLOSE** if d < +0.033 **or** wins ≤ losses.

**Not deciding, reported:**
- the same d, wins and losses on the 18 new scenes alone;
- d over the 30 without the two scenes most in carry's favour (largest full-drive carry − 0.1.8b);
- reset and 0.1.7 means.

For reference, the old-score numbers are d = +0.068, with 6 wins and 0 losses (`out/cl18_table.json`, `reports/CL18_MEMORY.md`).

## Setup

- CPU only: `taskset -c 0-1`, `nice 19`. No GPU, no docker, no simulator runs.
- Writes go only to `reports/`, `out/` and `logs/` of the approach-3 memory folder.
- Script: `src/s19_fulldrive_rescore.py`. Output: `out/fulldrive_rescore.{csv,json}`. Report: `reports/FULLDRIVE_RESCORE.md`.
- **Timing:** step 3 waits until the NAS folders are readable again. This file is not edited after the first number is computed. Any change gets a dated amendment written before the change is used.

## Amendment 1 (2026-11-16 23:35 IST, before any carry or 0.1.8b drive has been read)

Added at Sid's request. This is reported, **not deciding**.

- **Hard vs easy, reported separately.** The hard set is the 12 diagnosis scenes; the easy set is the 18 new scenes.
- **For each set:**
  - each arm's full-drive mean (carry, 0.1.8b; reset and 0.1.7 for reference);
  - the gap d = carry − 0.1.8b;
  - the gap as a share of 0.1.8b's room to improve, d / (1 − 0.1.8b mean). If 0.1.8b's mean is 1.0, this share is reported as undefined.
- **Old scores, same layout.** The same numbers are also given on the old (windowed) scores, so the effect of the window can be read off.

**State of the work at this point:**
- the only drives read are 0.1.7's, by the script test (`logs/fulldrive_017.log`);
- 0.1.7 is reference-only, and none of the decision's inputs has been read.
