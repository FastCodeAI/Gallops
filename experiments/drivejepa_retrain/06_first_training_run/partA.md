# S4 Part A — full unseen-clip set (val_full)

**STATUS 2026-10-07 07:53 — DONE.** `04_training_set/val_full/manifest_val_full.parquet` written, all asserts pass, `A2_DONE.json` written last.
Took 6.9 h against the ~5 h target (grading alone 6.2 h), plus a 45 min gap before the manifest step was run.

## Counts

| source | type | clips | anchors converted | D2 drops | samples in manifest | has_label2 | direction_exempt (D3) |
|---|---|---|---|---|---|---|---|
| nurec | A (highway) | 110 | 3,513 | 173 | 3,340 | 3,315 | 0 |
| nurec | B_pass | 57 | 1,824 | 254 | 1,570 | 1,563 | 86 |
| nurec | ordinary | 237 | 7,576 | 595 | 6,981 | 6,964 | 31 |
| raw | A (highway) | 100 | 3,194 | — | 3,194 | 0 | — |
| raw | B_raw (pull-out) | 1 | 32 | — | 32 | 0 | — |
| raw | ordinary | 299 | 9,540 | — | 9,540 | 0 | — |
| **total** | | **804** | **25,679** | **1,022** | **24,657** | **11,842 (48.0 %)** | **117** |

The manifest holds **801 clips**, not 804: three ordinary NuRec scenes had all 32 moments dropped by D2 (778a7121, a62e9c2c, a94b29b1).
By type in the manifest: A 6,534 samples / 210 clips (3,315 graded); B_pass 1,570 / 57 (1,563); B_raw 32 / 1 (0); ordinary 16,521 / 533 (6,964).

- **Pick** (seed 20261008, `val_full/clips.csv`): all 404 eligible nurec_val scenes (404 of 404; typed as `s3_pick.py` types train: A = S0 Run 3
  high-speed flag, B_pass = any around-still-obstacle event and not A, else ordinary) + 400 drawn from 88,013 eligible raw_val clips that are
  not nurec_val, drawn before typing. 499 val chunks, 24 countries. 8 of the NuRec scenes were already in the smoke val set and were reused.
- **Pull-outs in the 400 raw clips: 1** (48dc7e8a: moves off at 6.8 s, 3.4 m lateral, stationary car 2.8 m ahead of the bumper). Scan: 68 clips
  with a move-off, 15 with ≥ 1 m lateral, 1 with a stationary actor ahead, 0 removed by the 45° turn filter. Same detector code as the 285
  train pull-outs (`r3_core.moveoffs` + `attach_actor` via `r3_raw_actors.chunk`). 1 in 400 is in line with train (285 in ~150 k).
- **Raw clips typed A: 100** (hold ≥ 16 m/s for 4.5 s; train's raw ordinary pool excluded such clips, so they are not called ordinary).
  18 of the 400 raw clips never exceed 3 m/s (train's ordinary pool excluded those too; here they stay, typed ordinary).
- **Conversion:** 804 of 804 clips, 0 errors, bonnet mask on 25,679 of 25,679 anchors.
- **Grading:** 404 of 404 scenes ran, 0 scene errors, 0 missing maps; 401 have at least one graded moment. Moments: 11,842 graded,
  1,022 dropped by D2 (7.9 % of NuRec moments; train was 6.2 %), 49 kept ungraded as generator-only rows (47 ego not on any lane,
  2 no 4 s future). Human path ≥ 0.8 on 94.8 % of graded moments. Highway moments (≥ 16 m/s): 2,758, extended anchor index non-empty
  on 94.3 %, original on 16.1 %.
- **Downloads:** 11.6 GB, logged with sha256 in `val_full/download_manifest.csv` (front-wide video 10.65 GB for 796 clips, egomotion
  packages 0.77 GB for 487 chunks, map members 92 MB for 396 scenes, calibration 72 MB, 15 obstacle members 7 MB).

## Wall time

| stage | start → end | wall |
|---|---|---|
| pick + chunk label files + pull-out scan | 00:54 → 00:58 | 4 min |
| labels (804 clips, 25,679 anchors) | 00:58 | 18 s |
| video download (6 threads) | 00:58 → 01:15 | 17 min |
| map fetch (4 threads) | 00:58 → 01:35 | 37 min |
| convert (8 processes) | 01:15 → 01:49 | 34 min |
| grading (10 processes) | 00:59 → 07:08 | 6.2 h (pass 2: 376 scenes in 5.3 h, ≈ 71 per hour) |
| manifest + asserts | 07:47 → 07:53 | 6 min |

## Asserts passed (`s4a_manifest.py`)

Columns and dtypes equal the train manifest; `split == {"val"}`; every clip raw_val and every NuRec clip nurec_val (`assert_val`);
0 exam UUIDs; 0 test UUIDs; 0 train clip UUIDs; 0 sample ids shared with train; sample ids unique; **0 of the 499 val chunks among the
1,398 train chunks**; every graded row has `metric_cache.pkl`, `anchors_scores_index_ext.npy` and `pad_label2.gz` in the shared NAS cache.

## Flags

1. **Late.** Grading ran at about 8.5 process-minutes per scene on a host with load average 24–62; 10 processes could not make 5 h. Pass 1
   also wasted ≈ 10 min waiting on one slow scene, and the manifest step sat for 45 min after grading because my wait job was killed.
2. **Raw highway is new to validation.** 3,194 raw type-A samples have no grades; the smoke val set had no raw highway and no pull-out.
   Per-type readouts on val_full are therefore not comparable with the smoke val numbers for highway (generator readouts now mix graded
   NuRec and ungraded raw).
3. **Pull-out readout rests on one clip (32 samples).**
4. **`s3_common.py` gained a third mode** (`S3_SPLIT=val_full` → `SPLIT == "val"`, `VAL_DIR == "val_full"`, own download manifest);
   train and val modes are unchanged (checked). New scripts: `s4a_pick.py`, `s4a_manifest.py`, `val_full/run_val_full.sh`,
   `val_full/status_loop.py`. The S0 detector's obstacle-member downloads are logged in the val_full manifest, not S0's.
5. Smoke val (`val/manifest_val.parquet`) untouched. Its 8 NuRec scenes are also in val_full; its 12 raw clips are not.
