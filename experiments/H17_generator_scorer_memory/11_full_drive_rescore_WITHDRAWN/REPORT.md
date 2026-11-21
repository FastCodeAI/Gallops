# Full-drive re-score: carry vs 0.1.8b

Computed 2026-11-17 02:15 IST on shakaal, CPU only (`nice 19`, `taskset 0,1`), no simulator runs. Rule and method were fixed beforehand in `reports/FULLDRIVE_RESCORE_PREREG.md` (written 23:33 IST, amendment 1 at 23:35, both before any carry or 0.1.8b drive was read; hashes in `logs/fulldrive_prereg.sha256`, the final one matches).

## Answer by the pre-registered rule: **CLOSE**

On all 30 scenes, full-drive scores:

| | carry | 0.1.8b | gap d | wins / losses / ties |
|---|---|---|---|---|
| **full drive** | **0.6302** | **0.7022** | **−0.072** | **3 / 3 / 24** |
| old (windowed) score | 0.8204 | 0.7527 | +0.068 | 6 / 0 / 24 |

Both CLOSE conditions hold. d = −0.072 is below +0.033, and wins (3) are not more than losses (3).

Not deciding:
- **18 new scenes alone:** carry 0.6087 vs 0.1.8b 0.7604. d = −0.152, with 2 wins, 3 losses and 13 ties.
- **30 scenes without the two scenes most in carry's favour** (`bae463d3` +0.620, `1acf2f5d` +0.165): d = −0.105.

Source for every number: `out/fulldrive_rescore.json`, built by `src/s19_fulldrive_rescore.py`. Tables come from `src/s19_fulldrive_tables.py`, which writes `out/fulldrive_rescore.csv` and `out/fulldrive_tables.md`. The log is `logs/fulldrive_all.log`.

## Sanity check: old scores reproduced

The script runs AlpaSim's own aggregation with the original window kept. For **102 of 102 drives** it reproduces `results-summary.json` `rollouts[0].score` to 1e-9: 0 errors and 0 mismatches (`logs/fulldrive_all.log`, last line). The old means also match the published ones: 0.8204 / 0.7527, d +0.068, 6/0 (`out/cl18_table.json`, `reports/CL18_MEMORY.md`).

## Where the old window ended (step 1)

`s5_diagnosis/src/stepb.py:31-32` does not compute the score. It returns `rollouts[0]` of the run's `aggregate/results-summary.json`. The window is set in AlpaSim's eval code:
- **Start:** at `eval_relevant` (`eval/aggregation/processing.py:39`). This is the handover, +0.00 s, for all 102 drives.
- **End:** at whichever comes first:
  - the first tick with any collision (front, side or rear) or off-road (`processing.py:66`);
  - the first tick ≥ 4.0 m from the human path (`eval/aggregation/main.py:135-140`, value from `wizard/configs/base_config.yaml:402`; every run's own `eval-config.yaml` was checked to hold 4.0).

The last metric tick is +16.75 to +16.85 s for every drive. With both end cuts removed, the full-drive window runs exactly to it on all 102 drives.

What ended the old window:

| arm | 4 m from human path | an event | ran to the last tick |
|---|---|---|---|
| carry (30) | **19** | 5 | 6 |
| 0.1.8b (30) | 14 | 7 | 9 |
| reset (12) | 4 | 5 | 3 |
| 0.1.7 (30) | 17 | 5 | 8 |

Carry leaves the human path by 4 m more often than 0.1.8b. That is where its unseen failures sit.

## What changed (all 15 drives whose score moved)

| drive | old → full | why |
|---|---|---|
| carry `0245ff75` | 1.000 → 0 | 4 m at +13.75; front collision +15.75 |
| carry `032b6f21` | 1.000 → 0 | 4 m at +12.50; front collision +15.50 |
| carry `04394343` | 1.000 → 0 | 4 m at +11.15; off-road from +11.65 |
| carry `026d6a39` | 1.000 → 0 | 4 m at +7.25; off-road +11.20 |
| carry `16b83346` | 1.000 → 0 | 4 m at +5.70; off-road +11.65 |
| carry `04749bb9` | 0.707 → 0 | 4 m at +6.40; off-road +6.90 |
| 0.1.8b `026d6a39` | 1.000 → 0 | 4 m at +7.85; side collision +14.55 |
| 0.1.8b `16b83346` | 1.000 → 0 | 4 m at +6.15; off-road +14.15 |
| 0.1.8b `b1b170d0` | 0.723 → 1.000 | 4 m at +8.10 froze progress; it later rejoined with no event |
| 0.1.8b `98dd56b2` | 0.791 → 1.000 | hit from behind at +12.60 froze progress; a rear hit is not at fault, so it stays non-failing |
| reset `5f064b02` | 0.172 → 0 | hit from behind +9.20 cut the window; front collision +11.45 |
| 0.1.7 `04749bb9` | 0.956 → 0 | 4 m at +10.50; side collision +16.20 |
| 0.1.7 `026d6a39` | 1.000 → 0 | 4 m at +7.80; front collision +10.75 |
| 0.1.7 `b1b170d0` | 0.782 → 1.000 | 4 m at +9.05 froze progress |
| 0.1.7 `bae463d3` | 0.311 → 0.314 | progress after the 4 m cut |

- **Carry:** 6 drives go to 0, and none improve.
- **0.1.8b:** 2 go to 0 (both on scenes where carry also goes to 0), and 2 improve.

The three full-drive losses (`0245ff75`, `032b6f21`, `04394343`) are all on the new 18 scenes. In each, carry fails after the old window while 0.1.8b drives clean to the end. Two of them are the "two late carry crashes" already flagged in `reports/CL18_MEMORY.md`, and `04394343` is a third, an off-road.

## Hard (12 diagnosis scenes) vs easy (18 new), not deciding

The share is gap ÷ 0.1.8b's room to improve, (1 − 0.1.8b mean).

| set | score | carry | 0.1.8b | gap | share of room | W/L/T | reset | 0.1.7 |
|---|---|---|---|---|---|---|---|---|
| hard 12 | full | 0.6623 | 0.6150 | **+0.047** | **+12 %** (of 0.385) | 1/0/11 | 0.5033 | 0.6130 |
| hard 12 | old | 0.7213 | 0.5919 | +0.129 | +32 % (of 0.408) | 3/0/9 | 0.5177 | 0.6742 |
| easy 18 | full | 0.6087 | 0.7604 | **−0.152** | **−63 %** (of 0.240) | 2/3/13 | — | 0.7778 |
| easy 18 | old | 0.8865 | 0.8599 | +0.027 | +19 % (of 0.140) | 3/0/15 | — | 0.8333 |
| all 30 | full | 0.6302 | 0.7022 | −0.072 | −24 % (of 0.298) | 3/3/24 | — | 0.7118 |
| all 30 | old | 0.8204 | 0.7527 | +0.068 | +27 % (of 0.247) | 6/0/24 | — | 0.7697 |

On the hard set, carry's edge shrinks from +0.129 to +0.047 and comes down to one scene, `bae463d3` (+0.620). Its old win on `04749bb9` was an unseen off-road, and its old win on `b1b170d0` was 0.1.8b's frozen progress. On the easy set the sign flips.

## Flags

1. **0.1.7 is an earlier image and is reference only.** Its full-drive mean is 0.7118 (old 0.7697).
2. **Which 0.1.8b runs fill the 12 hard scenes.** The same runs as the comparison page:
   - 4 image-check re-runs (`e3ed554f`, `8023dce5`, `d2715023`, `04749bb9`, from `cl12/imagecheck_018bp06`);
   - 8 runs from 2026-10-31.
3. **`791d68ce` was off-road before the handover** (−3.21 s) in both arms. Its old window ends at +0.00 and it scores 0 under both scores.
4. **Timestamp types.** Timestamps are Int128 in the carry, reset and 0.1.8b files and Float64 in 0.1.7's files. Both were cast to Int64; Float64 is exact below 2^53 µs, and all reproduce.
5. **The 4 m limit is AlpaSim's default.** The only comment on it is `eval/schema.py:168`; no report on this lane chose it.
6. **The full-drive score treats a rear hit as before:** not a failure, and it no longer freezes progress. That is why 0.1.8b `98dd56b2` rises.
7. **Single run per drive.** There is no seed repeat, so each win or loss is one drive.
8. **Backup copy.** `{nas_backup}/FastCode/siddesh` is still empty and `siddesh-physicalai` on the backup has no `s5_trial_017`, so all runs were read from `{nas}`, read-only.
9. **Prereg hash file.** `logs/fulldrive_prereg.sha256` holds two hashes: the original file and the file after amendment 1. Only the second matches now, which is expected since the amendment was appended. The file has not been touched since 23:35:45 IST.

## Per-scene and per-drive tables

Events are times in seconds after handover; `*` marks an event after the old window. Old and full scores are listed for every drive.

| set | scene | carry old → full | 0.1.8b old → full | gap old → full | full result |
|---|---|---|---|---|---|
| hard | `e3ed554f` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| hard | `8023dce5` | 0.000 → **0.000** | 0.000 → **0.000** | +0.000 → **+0.000** | tie |
| hard | `5f064b02` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| hard | `e121e37d` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| hard | `048b974e` | 0.948 → **0.948** | 1.000 → **1.000** | -0.052 → **-0.052** | tie |
| hard | `cdd9448c` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| hard | `d2715023` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| hard | `bd07fb91` | 0.000 → **0.000** | 0.000 → **0.000** | +0.000 → **+0.000** | tie |
| hard | `bae463d3` | 1.000 → **1.000** | 0.380 → **0.380** | +0.620 → **+0.620** | win |
| hard | `d1a25a99` | 0.000 → **0.000** | 0.000 → **0.000** | +0.000 → **+0.000** | tie |
| hard | `b1b170d0` | 1.000 → **1.000** | 0.723 → **1.000** | +0.277 → **+0.000** | tie |
| hard | `04749bb9` | 0.707 → **0.000** | 0.000 → **0.000** | +0.707 → **+0.000** | tie |
| easy | `01d503d4` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `023b7fcc` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `0245ff75` | 1.000 → **0.000** | 1.000 → **1.000** | +0.000 → **-1.000** | loss |
| easy | `026d6a39` | 1.000 → **0.000** | 1.000 → **0.000** | +0.000 → **+0.000** | tie |
| easy | `02e075b9` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `02eadd92` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `032b6f21` | 1.000 → **0.000** | 1.000 → **1.000** | +0.000 → **-1.000** | loss |
| easy | `04394343` | 1.000 → **0.000** | 1.000 → **1.000** | +0.000 → **-1.000** | loss |
| easy | `16b83346` | 1.000 → **0.000** | 1.000 → **0.000** | +0.000 → **+0.000** | tie |
| easy | `1acf2f5d` | 1.000 → **1.000** | 0.835 → **0.835** | +0.165 → **+0.165** | win |
| easy | `2e1ce314` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `6725df2d` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `71d7d03b` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `791d68ce` | 0.000 → **0.000** | 0.000 → **0.000** | +0.000 → **+0.000** | tie |
| easy | `89f2eeb8` | 1.000 → **1.000** | 1.000 → **1.000** | +0.000 → **+0.000** | tie |
| easy | `98dd56b2` | 1.000 → **1.000** | 0.791 → **1.000** | +0.209 → **+0.000** | tie |
| easy | `acd22430` | 0.000 → **0.000** | 0.000 → **0.000** | +0.000 → **+0.000** | tie |
| easy | `b44a4a44` | 0.957 → **0.957** | 0.852 → **0.852** | +0.105 → **+0.105** | win |

| arm | scene | old | full | window end (s) | first ≥ 4 m (s) | last tick (s) | events over the whole drive (s after handover; * = after the old window) |
|---|---|---|---|---|---|---|---|
| carry | `e3ed554f` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| carry | `8023dce5` | 0.000 | 0.000 | +6.30 | +9.05 | +16.85 | front collision +6.30; side collision +10.25*; off-road +13.75* |
| carry | `5f064b02` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| carry | `e121e37d` | 1.000 | 1.000 | +14.65 | +14.65 | +16.80 | none |
| carry | `048b974e` | 0.948 | 0.948 | +16.80 | — | +16.80 | none |
| carry | `cdd9448c` | 1.000 | 1.000 | +14.60 | +14.60 | +16.85 | none |
| carry | `d2715023` | 1.000 | 1.000 | +11.60 | +11.60 | +16.80 | none |
| carry | `bd07fb91` | 0.000 | 0.000 | +5.85 | +6.55 | +16.80 | front collision +5.85; hit from behind +6.10*; off-road +7.35*; front collision +7.45*; side collision +7.55*; hit from behind +7.65*; off-road +13.15*; off-road +13.70* |
| carry | `bae463d3` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| carry | `d1a25a99` | 0.000 | 0.000 | +5.80 | — | +16.75 | off-road +5.80; hit from behind +10.15*; front collision +10.75* |
| carry | `b1b170d0` | 1.000 | 1.000 | +12.95 | +12.95 | +16.80 | none |
| carry | `04749bb9` | 0.707 | 0.000 | +6.40 | +6.40 | +16.80 | off-road +6.90*; off-road +13.20* |
| carry | `01d503d4` | 1.000 | 1.000 | +12.95 | +12.95 | +16.75 | none |
| carry | `023b7fcc` | 1.000 | 1.000 | +16.20 | +16.20 | +16.80 | none |
| carry | `0245ff75` | 1.000 | 0.000 | +13.75 | +13.75 | +16.85 | front collision +15.75*; hit from behind +16.45* |
| carry | `026d6a39` | 1.000 | 0.000 | +7.25 | +7.25 | +16.80 | off-road +11.20*; hit from behind +12.05*; off-road +13.35* |
| carry | `02e075b9` | 1.000 | 1.000 | +14.85 | +14.85 | +16.75 | none |
| carry | `02eadd92` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| carry | `032b6f21` | 1.000 | 0.000 | +12.50 | +12.50 | +16.75 | front collision +15.50*; hit from behind +16.05* |
| carry | `04394343` | 1.000 | 0.000 | +11.15 | +11.15 | +16.80 | off-road +11.65*; off-road +12.15*; off-road +12.45* |
| carry | `16b83346` | 1.000 | 0.000 | +5.70 | +5.70 | +16.85 | off-road +11.65* |
| carry | `1acf2f5d` | 1.000 | 1.000 | +12.35 | +12.35 | +16.80 | none |
| carry | `2e1ce314` | 1.000 | 1.000 | +13.40 | +13.40 | +16.80 | none |
| carry | `6725df2d` | 1.000 | 1.000 | +11.15 | +11.15 | +16.80 | none |
| carry | `71d7d03b` | 1.000 | 1.000 | +12.95 | +12.95 | +16.80 | none |
| carry | `791d68ce` | 0.000 | 0.000 | +0.00 | +6.80 | +16.75 | off-road -3.21 |
| carry | `89f2eeb8` | 1.000 | 1.000 | +13.90 | +13.90 | +16.85 | none |
| carry | `98dd56b2` | 1.000 | 1.000 | +16.30 | +16.30 | +16.80 | none |
| carry | `acd22430` | 0.000 | 0.000 | +4.70 | +12.60 | +16.80 | off-road +4.70; front collision +4.90*; side collision +5.45*; hit from behind +5.55*; off-road +9.60*; front collision +10.15*; side collision +10.55*; hit from behind +10.60* |
| carry | `b44a4a44` | 0.957 | 0.957 | +16.80 | — | +16.80 | none |
| 018b | `e3ed554f` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 018b | `8023dce5` | 0.000 | 0.000 | +6.90 | +8.45 | +16.85 | front collision +6.90; hit from behind +12.60*; front collision +12.85*; side collision +15.60*; hit from behind +16.00* |
| 018b | `5f064b02` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| 018b | `e121e37d` | 1.000 | 1.000 | +15.20 | +15.20 | +16.80 | none |
| 018b | `048b974e` | 1.000 | 1.000 | +16.35 | +16.35 | +16.80 | none |
| 018b | `cdd9448c` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| 018b | `d2715023` | 1.000 | 1.000 | +15.70 | +15.70 | +16.80 | none |
| 018b | `bd07fb91` | 0.000 | 0.000 | +6.95 | +10.10 | +16.80 | front collision +6.95; hit from behind +7.40* |
| 018b | `bae463d3` | 0.380 | 0.380 | +16.85 | — | +16.85 | none |
| 018b | `d1a25a99` | 0.000 | 0.000 | +6.95 | — | +16.75 | off-road +6.95; hit from behind +9.70*; front collision +10.30* |
| 018b | `b1b170d0` | 0.723 | 1.000 | +8.10 | +8.10 | +16.80 | none |
| 018b | `04749bb9` | 0.000 | 0.000 | +12.40 | +14.90 | +16.80 | front collision +12.40; side collision +13.20*; hit from behind +13.25* |
| 018b | `01d503d4` | 1.000 | 1.000 | +14.50 | +14.50 | +16.75 | none |
| 018b | `023b7fcc` | 1.000 | 1.000 | +16.00 | +16.00 | +16.80 | none |
| 018b | `0245ff75` | 1.000 | 1.000 | +15.55 | +15.55 | +16.85 | none |
| 018b | `026d6a39` | 1.000 | 0.000 | +7.85 | +7.85 | +16.80 | side collision +14.55*; front collision +14.95*; off-road +15.10*; side collision +15.70* |
| 018b | `02e075b9` | 1.000 | 1.000 | +15.00 | +15.00 | +16.75 | none |
| 018b | `02eadd92` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 018b | `032b6f21` | 1.000 | 1.000 | +15.50 | +15.50 | +16.75 | none |
| 018b | `04394343` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 018b | `16b83346` | 1.000 | 0.000 | +6.15 | +6.15 | +16.85 | off-road +14.15* |
| 018b | `1acf2f5d` | 0.835 | 0.835 | +16.80 | — | +16.80 | none |
| 018b | `2e1ce314` | 1.000 | 1.000 | +15.80 | +15.80 | +16.80 | none |
| 018b | `6725df2d` | 1.000 | 1.000 | +11.95 | +11.95 | +16.80 | none |
| 018b | `71d7d03b` | 1.000 | 1.000 | +16.20 | +16.20 | +16.80 | none |
| 018b | `791d68ce` | 0.000 | 0.000 | +0.00 | +8.15 | +16.75 | off-road -3.21; off-road +13.30* |
| 018b | `89f2eeb8` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| 018b | `98dd56b2` | 0.791 | 1.000 | +12.60 | — | +16.80 | hit from behind +12.60 |
| 018b | `acd22430` | 0.000 | 0.000 | +7.00 | +14.15 | +16.80 | off-road +7.00; front collision +8.80*; side collision +9.50*; hit from behind +9.65* |
| 018b | `b44a4a44` | 0.852 | 0.852 | +16.80 | — | +16.80 | none |
| reset | `e3ed554f` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| reset | `8023dce5` | 0.000 | 0.000 | +6.35 | +9.30 | +16.85 | front collision +6.35; side collision +13.50*; off-road +14.20*; hit from behind +14.65* |
| reset | `5f064b02` | 0.172 | 0.000 | +9.20 | — | +16.85 | hit from behind +9.20; front collision +11.45* |
| reset | `e121e37d` | 1.000 | 1.000 | +14.90 | +14.90 | +16.80 | none |
| reset | `048b974e` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| reset | `cdd9448c` | 1.000 | 1.000 | +14.85 | +14.85 | +16.85 | none |
| reset | `d2715023` | 1.000 | 1.000 | +12.40 | +12.40 | +16.80 | none |
| reset | `bd07fb91` | 0.000 | 0.000 | +5.90 | +6.95 | +16.80 | front collision +5.90; hit from behind +6.15*; off-road +10.30*; off-road +10.60*; off-road +10.85*; off-road +13.40*; off-road +13.75*; off-road +15.80*; off-road +16.15* |
| reset | `bae463d3` | 0.040 | 0.040 | +16.85 | — | +16.85 | none |
| reset | `d1a25a99` | 0.000 | 0.000 | +5.80 | — | +16.75 | off-road +5.80; hit from behind +10.60*; front collision +11.30*; front collision +16.60* |
| reset | `b1b170d0` | 1.000 | 1.000 | +13.10 | +13.10 | +16.80 | none |
| reset | `04749bb9` | 0.000 | 0.000 | +6.30 | +12.25 | +16.80 | front collision +6.30; side collision +6.70*; hit from behind +6.75*; front collision +9.30*; hit from behind +9.80* |
| 017 | `e3ed554f` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 017 | `8023dce5` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| 017 | `5f064b02` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| 017 | `e121e37d` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 017 | `048b974e` | 1.000 | 1.000 | +13.60 | +13.60 | +16.80 | none |
| 017 | `cdd9448c` | 1.000 | 1.000 | +16.85 | — | +16.85 | none |
| 017 | `d2715023` | 0.041 | 0.041 | +16.80 | — | +16.80 | none |
| 017 | `bd07fb91` | 0.000 | 0.000 | +7.10 | +12.85 | +16.80 | front collision +7.10; hit from behind +7.75* |
| 017 | `bae463d3` | 0.311 | 0.314 | +15.55 | +15.55 | +16.85 | none |
| 017 | `d1a25a99` | 0.000 | 0.000 | +6.25 | — | +16.75 | off-road +6.25; hit from behind +10.50*; front collision +11.15* |
| 017 | `b1b170d0` | 0.782 | 1.000 | +9.05 | +9.05 | +16.80 | none |
| 017 | `04749bb9` | 0.956 | 0.000 | +10.50 | +10.50 | +16.80 | hit from behind +10.60*; off-road +11.80*; off-road +14.55*; hit from behind +15.65*; side collision +16.20*; off-road +16.80* |
| 017 | `01d503d4` | 1.000 | 1.000 | +13.90 | +13.90 | +16.75 | none |
| 017 | `023b7fcc` | 1.000 | 1.000 | +16.35 | +16.35 | +16.80 | none |
| 017 | `0245ff75` | 1.000 | 1.000 | +15.65 | +15.65 | +16.85 | none |
| 017 | `026d6a39` | 1.000 | 0.000 | +7.80 | +7.80 | +16.80 | front collision +10.75*; side collision +11.15*; side collision +14.55*; off-road +14.80*; hit from behind +14.85*; front collision +15.05*; side collision +15.20* |
| 017 | `02e075b9` | 1.000 | 1.000 | +14.80 | +14.80 | +16.75 | none |
| 017 | `02eadd92` | 1.000 | 1.000 | +8.80 | +8.80 | +16.80 | none |
| 017 | `032b6f21` | 1.000 | 1.000 | +14.55 | +14.55 | +16.75 | none |
| 017 | `04394343` | 1.000 | 1.000 | +16.55 | +16.55 | +16.80 | none |
| 017 | `16b83346` | 1.000 | 1.000 | +9.55 | +9.55 | +16.85 | none |
| 017 | `1acf2f5d` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 017 | `2e1ce314` | 1.000 | 1.000 | +16.00 | +16.00 | +16.80 | none |
| 017 | `6725df2d` | 1.000 | 1.000 | +11.60 | +11.60 | +16.80 | none |
| 017 | `71d7d03b` | 1.000 | 1.000 | +15.45 | +15.45 | +16.80 | none |
| 017 | `791d68ce` | 0.000 | 0.000 | +0.00 | — | +16.75 | off-road -3.21 |
| 017 | `89f2eeb8` | 1.000 | 1.000 | +16.80 | +16.80 | +16.85 | none |
| 017 | `98dd56b2` | 1.000 | 1.000 | +16.80 | — | +16.80 | none |
| 017 | `acd22430` | 0.000 | 0.000 | +15.40 | +15.80 | +16.80 | front collision +15.40; side collision +16.75* |
| 017 | `b44a4a44` | 0.000 | 0.000 | +6.60 | — | +16.80 | side collision +6.60; front collision +8.20* |
