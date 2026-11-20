# Closed loop on 20 new scenes: memory carried vs 0.1.8b pass 0.6

Written 2026-11-20, shakaal.
- Pre-registration: `reports/CL20_PREREG.md`.
- Numbers: `out/cl20_table.json` / `.csv`, made by `src/s20_table.py`, log `logs/cl20_table.log`.
- Drives: `{local_store}/runs/new40/`, 7.5 GB.
- Comparison page: port 8786, 20-scene tab.

## Answer by the pre-registered rule: AGAINST

On the 20 new scenes, memory carried scores **0.6353** and 0.1.8b **0.7644**, so d = memory − 0.1.8b = **−0.129**.

All three AGAINST conditions hold, and any one of them is enough:

| condition | needed for AGAINST | result |
|---|---|---|
| mean gap d | ≤ −0.05 | **−0.129** |
| losses vs wins (a scene gap of more than 0.1) | losses > wins | **5 losses, 1 win**, 14 ties |
| at-fault crash inside the score window that 0.1.8b doesn't have | any | **2: 68e53160 (side, +8.70 s) and 8fd8f4f0 (front, +6.45 s)** |

The win is a32549eb. The losses are 29815599, 320c11f3, 34226720, 68e53160 and 8fd8f4f0.

**What this does not show:** this compares two models. The memory model is 0.1.8b plus the memory **and** 930 more training steps, so this test cannot say which of the two caused the difference.

## Per scene (the 20 new scenes)

Score = `stepb.py` `score`. The score window ends at the first crash or off-road, or when the car is 4 m from the human path. "In-window at-fault" means a front or side collision before the window ends. Exit = the first event, at any time after handover. Times are seconds after handover.

| scene | memory | 0.1.8b | gap | memory exit | 0.1.8b exit | window ends, memory / 0.1.8b |
|---|---|---|---|---|---|---|
| 29815599 | 0.000 | 1.000 | **−1.000** | off-road +9.65 | pass | 9.65 / 11.75 |
| 320c11f3 | 0.706 | 1.000 | **−0.294** | front collision +14.10 (after its window) | pass | 7.70 / 15.60 |
| 34226720 | 0.000 | 1.000 | **−1.000** | off-road +8.70 | pass | 8.70 / 14.80 |
| 44409f00 * | 1.000 | 1.000 | 0 | pass | pass | 13.70 / 13.80 |
| 49e0ea24 | 1.000 | 1.000 | 0 | pass | pass | 10.30 / 13.20 |
| 4a495c2d | 1.000 | 1.000 | 0 | pass | pass | 15.55 / 15.20 |
| 5c1b7555 | 1.000 | 1.000 | 0 | pass | pass | 13.95 / 14.90 |
| 68e53160 | 0.000 | 0.296 | **−0.296** | **side collision +8.70 (in window)** | hit from behind +12.70 | 8.70 / 12.70 |
| 6ae687bb | 1.000 | 1.000 | 0 | pass | pass | 13.10 / 16.80 |
| 7747986c | 0.000 | 0.000 | 0 | off-road −0.50 (before handover) | off-road −0.50 (before handover) | 0 / 0 |
| 8578f2c7 | 1.000 | 1.000 | 0 | front collision +15.85 (after its window) | pass | 10.85 / 11.70 |
| 8fd8f4f0 | 0.000 | 1.000 | **−1.000** | **front collision +6.45 (in window)** | pass | 6.45 / 9.00 |
| 9d8fca3b | 0.000 | 0.000 | 0 | front collision +4.35 (in window) | front collision +13.90 (in window) | 4.35 / 13.90 |
| a32549eb | 1.000 | 0.000 | **+1.000** | pass | off-road +6.00 | 15.30 / 6.00 |
| c438a779 | 0.000 | 0.000 | 0 | front collision +8.25 (in window) | front collision +13.50 (in window) | 8.25 / 13.50 |
| dd0f84df | 1.000 | 1.000 | 0 | pass | pass | 16.75 / 16.80 |
| e883822a | 1.000 | 0.991 | +0.009 | pass | pass | 16.35 / 16.85 |
| f005ac5f | 1.000 | 1.000 | 0 | hit from behind +12.50 | hit from behind +12.20 | 12.50 / 12.20 |
| f3a87e7a | 1.000 | 1.000 | 0 | pass | pass | 13.30 / 14.90 |
| f890fe10 | 1.000 | 1.000 | 0 | off-road +7.90 (after its window) | pass | 5.05 / 16.80 |

\* 44409f00 replaced 9e6753c2 (see flag 1).

## Not deciding (pre-registered)

- **Easy scenes:** 10 of the 20 score 1.000 for both drivers.
- **Without the two scenes most in memory's favour** (a32549eb, e883822a): d = **−0.199**.
- **Pick changes:** in memory runs the shadow branch (reset memory on the same input) picks a different final plan on 100 of 780 calls (12.8 %). On the 18 it was about 1 in 8 as well.

## All 50 scenes (secondary)

| block | n | memory | 0.1.8b | d | wins / losses / ties | in-window at-fault, memory / 0.1.8b |
|---|---|---|---|---|---|---|
| 20 new (this test) | 20 | 0.6353 | 0.7644 | −0.129 | 1 / 5 / 14 | 4 / 2 |
| 30 earlier (12 diagnosis + 18) | 30 | 0.8204 | 0.7527 | +0.068 | 6 / 0 / 24 | 2 / 3 |
| **all 50** | 50 | **0.7464** | **0.7574** | **−0.011** | 7 / 5 / 38 | 6 / 5 |

- Without the two scenes most in memory's favour (a32549eb, 04749bb9), d over the 50 is −0.047.
- The 30 earlier drives ran on the previous simulator image (`9c5909da…`), and the 20 new ones on the rebuilt image (`ab16bc78…`). They are pooled because the image check reproduced 6 earlier drives bit for bit.
- The 12 diagnosis scenes are the ones where memory was looked at most. They carry memory's earlier lead: +0.068 over the 30. On scenes it had never been tested on, the lead disappears: +0.027 on the 18 (CL18), and −0.129 on these 20.

## Flags

1. **One scene replaced.**
   - Memory carried on 9e6753c2 hit a simulator error about 12 s in: the route builder reports "route folds back on itself, 180°" (`route_generator.py:258`). The simulator's own resume failed the same way all 3 times, and no metrics were written.
   - It is a map fault, not a driving event. 0.1.8b was never run on that scene.
   - At Sid's instruction it was replaced by **44409f00**, the first scene in a seed-0 shuffle of the same pool the 20 came from. The rule was written before the draw.
   - The failed folder is kept as `runs/memory_carry_9e6753c2.simfail_route_foldback`.
2. **A setup error of mine on the replacement.**
   - The first attempt on 44409f00 failed at startup: the scene file was a symlink to data8t, and the simulator containers mount only the scene cache.
   - The `--preflight-only` check did not catch it, because it never opens the file inside a container.
   - Fixed by putting a real copy in the cache (hash checked), and the same scene was retried. The failed folder is kept (`…44409f00.harnessfail_symlink_not_mounted`).
3. **The at-fault definition differs from CL18.** This brief counts only at-fault crashes inside the score window.
   - Under CL18's definition (the first exit at any time), memory would add four crashes 0.1.8b doesn't have: 320c11f3, 68e53160, 8578f2c7 and 8fd8f4f0.
   - The verdict is AGAINST either way.
4. **Memory's score window tends to close sooner:** it ends earlier than 0.1.8b's on 16 of 20 scenes.
   - On f890fe10, memory drifted 4 m from the human path at +5.05 s and left the road at +7.90 s, yet scores 1.000.
   - On 320c11f3 and 8578f2c7, its front collisions come after the window. As in CL18, the window hides late failures, and here more of them belong to memory.
5. **7747986c starts off-road** (−0.50 s, before handover) in both arms, so it scores 0 for both. It is the scene start, not the driver, and it counts as a tie.
6. **Simulator image rebuilt** (`alpasim-base:0.134.0` = `ab16bc78…`, renderer `97f43e71…`).
   - The image check passed: 6 of 6 drives bit-identical (`out/cl20_imagecheck.json`).
   - The first rebuild attempt was stopped by the root-disk watchdog (the build cache grew 48 GB). The rebuild was resumed after deleting 22 of our own 43 cache entries.
   - Keep-placeholder containers (`siddesh-keep-*`) now hold both images.
7. **Pinning and order:** all 40 drives were pinned to 2 cores at nice 19, carry then 0.1.8b per scene, as in CL18.
8. **Unchanged from CL18:** the comfort re-score is on in both arms, and the shadow is on in carry. The shadow does not touch the driven plan.

## Files

- **Scripts:**
  - `src/s20_queue.sh` (40-drive queue) and `src/s20_imagecheck_queue.sh`;
  - runners `src/s20_run_memory_scene.sh`, `src/s20_run_018bp06_scene.sh` and their `*_imagecheck.sh` copies (output root only);
  - `src/s20_imagecheck_cmp.py`, `src/s20_table.py`, `src/s20_compare_page.py` and `src/s20_compare_page.html`.
- **Scene lists and pins:**
  - `out/cl20_scenes_20_v2.txt` (sha256 `655a3149…`);
  - `out/sim_scenes_new20_v2_26p01.csv` (`350fe886…`);
  - the v1 files are kept.
- **Logs:**
  - `logs/cl20_imagecheck_queue.log`, `logs/cl20_queue.log`, `logs/cl20_table.log` and `logs/cl20_compare_page.log`.
- **Data8t:**
  - drives under `runs/new40/`;
  - scene list, replacement draw and download record under `new40_scenes/`;
  - page under `compare_page/`.
