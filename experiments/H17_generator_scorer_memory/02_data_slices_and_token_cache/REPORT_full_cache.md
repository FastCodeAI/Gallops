# Drive-JEPA memory, approach 3: build step 2, the full token cache

shakaal, 2026-11-08 (00:05–04:27 IST). One GPU job (the cache of the four slices plus its exactness check, one chain). No training, no closed loop, no docker, no pushes. Sid's change of plan during the step: **part 1 only**; the history check was not pre-registered, fitted or run, and nothing of it was started. Writes only under `{runs}/generator_scorer_memory/cache/` (new) and this memory folder (`src/`, `out/`, `logs/`). Nothing large on shakaal's local disk.

Path abbreviations as in `APPROACH3_RECON.md` (`G`, `RD`, `NAS`, `PAI`, `UP`, `GM` = `gallops-memory/src/gallops/memory/`); `A3` = `{runs}/generator_scorer_memory/`. Plain names: encoder, generator, scorer. Every number below is read from a file named in the same row.

**Read first (verified):** `reports/BUILD1_SLICES_CACHE.md` step 4 (what is cached, the exactness method, the timings); `reports/APPROACH3_RECON.md` D4 (read, not acted on: part 2 was dropped). **Starting model:** `NAS/018b/release/drive_jepa_018b_p06.ckpt`, sha256 recomputed at 00:05 = `3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66` (equals the pin; session `sha256sum`), and again inside every job before loading (`out/s5_cache_<slice>.json: p06_sha_verified_s`); the 495 loaded tensors re-compared with the file after loading, 495 / 495 equal (`agent.loaded_equal`).

## Summary

| | result | files |
|---|---|---|
| cached moments | training **126,981 / 126,981**, picking clips **2,591 / 2,591** (2,000 picking moments flagged), graded held-out **10,516 / 10,516**, raw held-out **9,566 / 9,566**; missing **0** | `A3/cache/<slice>/index_<slice>.parquet`, `out/s5_cache_<slice>.json: counts` |
| disk | **88.6 GB** on the NAS (75.2 training, 1.5 picking, 6.2 graded, 5.7 raw; fp32 tokens + pooled copy + ego row + human future per moment, plus one pooled array per slice) | `out/s5_cache_<slice>.json: disk` |
| compute | encoder GPU time **1.15 GPU-hours** (synchronised, encoder only); wall **3 h 56 min (00:27:02–04:22:52, the four slices back to back; 3 h 50 min inside the four jobs)** for the four slices, **3.5 min inside the check (started 04:23, ended 04:27)** for the exactness check | `out/s5_cache_<slice>.json: timing`, `logs/s5_cache_chain.log` |
| exactness | training **100 % / 0.0 / 0.0**, picking **100 % / 0.0 / 0.0** (pick agreement / max plan difference / max logit difference); graded held-out plans identical **yes (logits yes)**, raw held-out **yes (logits yes)**; cached tokens equal the check forward's tokens on **224/224 + 224/224 + 215/215 + 223/223** of the 4 × ≥ 200 sampled moments | `out/s5_cache_check.json` |
| encoder batch | **batch 1**, not Sid's batch 8: the batch-8 tokens differ from batch-1 tokens by up to 0.021 (flag 1, OPEN 1) | `out/s5_cache_training_STOP.json: batch_check` |

## Before the job

| gate | value at 00:11–00:20 IST | source |
|---|---|---|
| other NAS transfers | none of ours (build 1's conversion finished 11:41 the day before, `A3/raw_heldout/logs/convert.log`); the NIC to the NAS (`enp14s0`, `ip route get 192.168.1.3`) moved 40 KB received / 70 KB sent in a 10 s sample, i.e. idle | session `/proc/net/dev` sample |
| GPU | 47–51 GB of 97.9 GB used by other users' 9 processes (Bench2Drive + policy servers), 6–19 % busy: their usual load; our job peaked at 1.3 GB | `nvidia-smi`; `out/s5_cache_<slice>.json: gpu_before, gpu_peak_alloc_gb` |
| NAS space | 1.7 TB free of 21 TB (93 % used) ≥ 100 GB | `df -h {nas}` |
| memory gate | load 23–26 (< 32), available RAM 74 GB (> 40 GB) | `uptime`, `free -g` |

## What is cached and how

**Tensor (unchanged from build 1).** `feat_flatten[0, :, b, :]` of `ImgEncoder.forward` (`UP/bevformer/simple_image_encoder.py:56-66`): the projector's 256-d tokens of the 16 × 32 grid, the two frames (t − 0.5 s, t) fused as one 2-frame clip (`:33`), **(512, 256) fp32 per moment**; beside it the pooled copy `GM/probe.py:66-72 pool_map` (global mean + 4 × 8 block means = 33 × 256 fp32). The encoder is called exactly as the model calls it (`UP/drive_jepa_model.py:53-57`: ImageNet `Normalize` on both frames, `cat([older, newer], 1)`, one backbone call), eval mode, GridMask asserted off (`s5_cache_full.py`, `assert not bk.grid_mask.training`), fp32 (no autocast).

**Also stored per moment** (the slim file is open anyway; no measure is computed on any of it): the newest ego-status row (11 values, the converter's accelerations kept since the encoder never reads `ego_status`; the model's loader zeroes them, `dataset_s35.py:_finish`) and the (8, 3) human future.

**Files** (`src/s5_cache_full.py`): `A3/cache/<slice>/<clip>.pt` = `{sample_id, anchor_t, tokens (N, 512, 256), pooled (N, 33, 256), ego_status (N, 11), trajectory (N, 8, 3), has_label2}` in time order, one file per clip as in `A3/cache_sample/` (build 1 kept a dict per moment; a stacked tensor gives the index a `row`); `index_<slice>.parquet` with `sample_id, clip, anchor_t, file, row, sha256` (of the file) `, has_label2, type, source, group, in_picking, pooled_row`; `pooled_<slice>.npy` (N, 8,448) fp32, every moment's pooled copy in index order; `index_partial.jsonl` (one line per written file: clip, file, sha256, n, bytes, row0) for resuming. Each file is serialised in memory, hashed, written to a temporary name and renamed, so the index sha is the sha of the bytes on disk. **Resumable:** a clip whose file exists and whose sha256 equals its `index_partial.jsonl` line is skipped and not re-encoded (`resume.clips_already_cached_and_verified`).

**Model loading and hook** = `s4_cache_sample.py:30-38, 48-50` verbatim (strict load through `drive_jepa_agent_s10.py:135-137`, 0.1.7's batch fix on, 495 / 495 tensors equal). **Loader:** the lane's read-only slim dataset (`pai_slim.ReadOnlySlim`, slim PNGs for training / picking / graded held-out, `pad_feature.gz` for the fresh raw held-out which has no slim copies, `A3/raw_heldout/cache/`), `DataLoader` batch 8, **8 spawned workers at nice 19 on all cores** (Sid's one-off exception; `src/run_model_cache.sh` = `run_model.sh` without `taskset`, nice 19 kept), split guard `train` for the training slice and `val` for the three val_full-derived or raw-validation slices (`pai_data.make_guard`, exam / wrong-split clips raise).

**Slices.** training = `RD/s3_assembly/manifest.parquet` (126,981 moments / 4,015 clips / 1,398 chunks); picking = every val_full moment of the 84 picking clips (`NAS/picking/picking_set.parquet` → `RD/s3_assembly/val_full/manifest_val_full.parquet`: 2,591 moments, 1,374 graded, the 2,000 picking moments flagged `in_picking`; 0 clips shared with the graded held-out, session check); graded held-out = `A3/slices/manifest_graded_heldout.parquet` (10,516); raw held-out = `A3/slices/manifest_raw_heldout.parquet` (9,566). Clip and moment counts of the two held-out slices are from build 1; **nothing was measured on them here** (the exactness check reports only identical yes/no).

**Encoder batch (flag 1).** The job encodes its first loader batch twice before writing anything: at the encoder batch and one moment at a time, and once more at the encoder batch (determinism); both must be bit-identical or the job stops (exit 3, nothing cached). First launch, batch 8 as decided: `batch8_equals_batch1_bitwise: false, max_abs_diff 0.0211, batch8_deterministic: true` (`out/s5_cache_training_STOP.json`, `logs/s5_cache_training_batch8_STOP.log`, 00:20–00:26). The check forward the exactness rule compares against is batch 1 (`pai_model.forward_dump`, `G/s1_build/src/pai_model.py:301-310`), so a batch-8 cache cannot be exact against it; the job was relaunched with the encoder at batch 1 (`ENC_BS=1`; the loader still delivers batches of 8), the setting under which build 1 showed the cache reproduces the forward bit for bit. Relaunch check: `batch8_equals_batch1_bitwise: true, max_abs_diff 0.0, batch8_deterministic: true` in all four slices (`out/s5_cache_<slice>.json: batch_check`).

## Part 1 results

### Counts

| slice | manifest moments | cached | missing | clips / files | index sha256 | source |
|---|---|---|---|---|---|---|
| training | 126,981 | 126,981 | 0 | 4015 / 4015 | `ac5561cdb42ecafb…` | `out/s5_cache_training.json` |
| picking clips | 2,591 | 2,591 | 0 | 84 / 84 | `97c5a1dfb6e3b946…` | `out/s5_cache_picking.json` |
| graded held-out | 10,516 | 10,516 | 0 | 355 / 355 | `52422a939a35e1c4…` | `out/s5_cache_graded_heldout.json` |
| raw held-out | 9,566 | 9,566 | 0 | 300 / 300 | `2de890c7facf71eb…` | `out/s5_cache_raw_heldout.json` |

The index asserts: every manifest moment present once, `pooled_row` = 0 … N − 1 in index order (`s5_cache_full.py`, the `assert` before `to_parquet`).

### Disk, GPU, wall

| slice | clip files | pooled array | GPU encoder s (s / moment) | wall | moments / s (wall) | source |
|---|---|---|---|---|---|---|
| training | 70.91 GB | 4.291 GB | 3,700 s (0.0291) | 2 h 30 min | 14.1 | `timing`, `disk` |
| picking clips | 1.45 GB | 0.088 GB | 56 s (0.0216) | 3.6 min | 12.0 | |
| graded held-out | 5.87 GB | 0.355 GB | 209 s (0.0199) | 11.6 min | 15.1 | |
| raw held-out | 5.34 GB | 0.323 GB | 188 s (0.0197) | 1 h 4 min | 2.5 | |
| **total** | **88.6 GB** (83.6 GB in clip files + 5.1 GB pooled arrays) | | **1.15 GPU-h** | **3 h 56 min (00:27:02–04:22:52, the four slices back to back; 3 h 50 min inside the four jobs)** | | `logs/s5_cache_chain.log` |

GPU seconds are the synchronised encoder call including the host-to-device copy of the two frames, so per moment they are above build 1's 0.0243 s zeros-input figure (`BUILD1_SLICES_CACHE.md` step 4) and fell from 0.029 s during the training slice (load 25–33 on the machine, `logs/s5_cache_training.log` progress lines) to 0.020 s on the small slices (load 7–10). The wall is the loader, not the GPU: the GPU was busy ≈ 41 % of the training slice's wall (3,700 of 9,020 s) and the rate rose from 12.7 to 14.2 moments/s as the machine load fell (progress lines every 500 clips). The raw held-out slice ran at 2.5 moments/s, six times slower than the other three: it has no slim copies, so its 8 workers decompress the converter's `pad_feature.gz` files from the NAS (`loader.cache_path`, `loader.slim_root: None`); 64 min for 9,566 moments. Build 1's estimate was ≈ 2 h wall / ≈ 1 GPU-hour at batch 8 for the whole cache (`BUILD1_SLICES_CACHE.md` OPEN 4); the result is 3 h 56 min / 1.15 GPU-hours at batch 1.

### Exactness (`src/s5_cache_check.py`, `out/s5_cache_check.json`, `logs/s5_cache_check.log`)

Sample per slice: the first clips of the index until ≥ 200 moments, whole clips. (a) = the full forward, `pai_model.forward_dump` (batch 1, fp32, the scorer's own pick, as build 1's pass (a)); (b) = the same forward with the encoder module replaced by one returning the cached tokens (`s4_cache_sample.py:51-54, 73-78`), images zeros. Also compared: the tokens the batch-1 forward produces against the cached tokens, bit for bit.

| slice | clips / moments | cached tokens = forward tokens | pick agreement | max plan difference (xy, m) | max logit difference | plans identical | logits identical | pass |
|---|---|---|---|---|---|---|---|---|
| training | 7 / 224 | 224/224 | 100.0 % (0 changed) | 0.0 | 0.0 | yes | yes | pass |
| picking clips | 7 / 224 | 224/224 | 100.0 % (0 changed) | 0.0 | 0.0 | yes | yes | pass |
| graded held-out | 7 / 215 | 215/215 | — (not computed) | — | — | yes | yes | pass |
| raw held-out | 7 / 223 | 223/223 | — (not computed) | — | — | yes | yes | pass |

Rule: training and picking must be 100 % / 0 / 0; held-out slices report only identical yes/no (no grade, no ADE, no pick quality was computed or stored for them: the check script holds the held-out plans only long enough to compare them and writes none of them).

## Flags

1. **Batch-8 encoding is not bit-identical to batch 1** (max token difference 0.021, deterministic). The encoder was run at batch 1, against Sid's "batch 8". Cost: the encoder is 0.029 s / moment at batch 1 against 0.0077 at batch 8 (build 1), i.e. ≈ 1.15 GPU-hours instead of ≈ 0.3; the wall time is loader-bound either way (14 moments/s on the slim-backed slices, 2.5 on the raw held-out; GPU busy ≈ 41 % of the wall). OPEN 1.
2. **Part 2 dropped by Sid mid-step**; the per-moment ego row and human future are still stored beside the tokens (they cost nothing and were read anyway); nothing was computed from them.
3. **The picking "whole clips" are 2,591 moments, not ~2,700**: the 84 picking clips hold 2,591 val_full moments (`manifest_val_full.parquet`), of which 2,000 are picking moments.
4. **The two held-out slices were encoded, not read**: their tokens exist on the NAS; the exactness check compared plans and logits for equality only.
5. **Raw held-out slice six times slower to read** (2.5 moments/s, 64 min for 9,566 moments against 11.6 min for the 10,516 graded moments): no slim copies, `pad_feature.gz` decompressed from the NAS. The cache now holds its tokens, so no later step pays this again.
6. **Machine load went above the 32 start gate during the training slice** (5-min load 33.1 at 00:48, session `/proc/loadavg`; the gate was 25–26 at start, `loadavg_before`). Available RAM never fell below 58 GB (progress lines; the 20 GB stop guard never fired). Our loader is the 8 nice-19 workers Sid allowed; the remaining load was other users' jobs.
7. **The exactness check crashed once before measuring anything**: `s5_cache_check.py` used `idx.clip`, which pandas resolves to the DataFrame's `clip` method, not the index column (`TypeError`, 04:23, `logs/s5_cache_chain.log: check rc=1`). Fixed to `idx["clip"]` and re-run alone at 04:23 (`check relaunch` line, `check rc=0` at 04:27); no cache file was touched between.
8. **Reading the 1.27 GB starting model from the NAS took 232 s once** (picking slice, `p06_sha_verified_s`) against 0.6–0.7 s (page cache) in the other three: a transient NAS stall right after the training slice's last writes. No effect on the result.

## OPEN for Sid

1. **Encoder batch 1 instead of 8 (flag 1).** Recommend: accept the batch-1 cache as built; it is exact against the check forward and the lane's picking-set forward, and the GPU cost difference is 1.15 vs ≈ 0.3 GPU-hours, once. Alternative: re-cache at batch 8 (≈ 0.3 GPU-hours plus the same loader wall) and accept a near-exact cache (token difference ≤ 0.021; its pick agreement is not measured; build 1's bf16 cache, with a token perturbation of similar size, changed 1 pick in 1,022).
2. **Where training reads from.** Recommend: the per-clip `.pt` files (one read per clip, tokens for all its moments, so 8 s windows cost nothing extra); the `pooled_<slice>.npy` arrays are for any pooled-only use. Alternative: a single memory-mapped token array per slice (67 GB for training), which would take another pass over the NAS to build.
3. **Keep or drop the ego row and human future stored beside the tokens.** Recommend: keep (tiny, already written, needed by any training step from the cache). Alternative: strip them (a rewrite of every file).

## Files

- `src/s5_cache_full.py` (the cache job, one slice per run), `src/s5_cache_check.py` (exactness), `src/run_model_cache.sh` (launcher without the 2-core pin) → `logs/s5_cache_{training,picking,graded_heldout,raw_heldout,check,chain}.log`, `logs/s5_cache_training_batch8_STOP.log`, `out/s5_cache_{training,picking,graded_heldout,raw_heldout}.json`, `out/s5_cache_training_STOP.json`, `out/s5_cache_check.json`
- `A3/cache/<slice>/{<clip>.pt, index_<slice>.parquet, pooled_<slice>.npy, index_partial.jsonl}` for the four slices
