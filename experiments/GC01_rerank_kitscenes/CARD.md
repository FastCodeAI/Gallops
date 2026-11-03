<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC01 · A re-ranker over Drive-JEPA's 32 candidates, trained on KITScenes train

**Naren** · KITScenes, nuScenes, neuroNCAP · ◐ mixed · 16 Sep – 12 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

## Question

Released Drive-JEPA is last on KITScenes (ADE@3s 2.364) and first on NAVSIM. Its 32 candidates include one within 0.37 m of the human on average, but its NAVSIM-trained scorer picks one 2.36 m away, about 15th of 32. Does replacing only the choice -- the model frozen, a small selector trained on KITScenes train -- close that gap, and what does it do elsewhere?

## What we learned

Replacing only the choice closes most of the KITScenes gap: ADE@3s 2.521 → 0.973 m against the model's own pick on the same path (paired t −8.0; adapted, open loop), and 0.587 cumulative L2 on nuScenes zero-shot. It learned closeness to the human path and nothing about collisions: on neuroNCAP 0103 with the seeing model it scored 1.85 against 3.01 (9 episodes), and it never picks a stop.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC01.kitscenes-train-h128-t1**<br>the selector: 128 hidden units, label temperature 1.0, epoch 325 of 400 | released | selector (new one-hidden-layer MLP over 18 features per candidate) (0.002561 M); frozen: all of Drive-JEPA: encoder, generator and scorer | kitscenes `train` 4,858 keyframes (train) — 318 of the list's 373 scenes (the first 373 of 533 to download, revision 213352cf); 5,711 keyframes in all. The sha256 is the train.npz export's (checkpoint.json notes.dataset_sha256)<br>kitscenes `train, 55 held-out scenes (seed 0)` 853 keyframes (selection) — chose the epoch by the pick's ADE; the 55 scene ids are in checkpoint.json notes.holdout_scenes<br>kitscenes `val + overlap_train_val` 223 samples (eval) — 140 scenes, none among the 373 (checked by scene id)<br>nuscenes `val` 5,119 samples (eval) — 150 scenes; no nuScenes data went into the selector<br>neuroncap `scene 0103, three scenario pairs, seeds 0-2` 9 episodes (eval) | `gallops.train.rerank`<br>commit `dfa7459` | `{weights}/drivejepa_rerank/kitscenes-train-h128-t1/step_000000325/`<br>sha256 `2050b799e8e88d4fdd46d95233bacf71a97eabf7db975fbf1de17052d66b91fa`<br>shipped as paddock model drivejepa_rerank (its model.yaml pins this directory) |

- **GC01.kitscenes-train-h128-t1:** Listwise cross-entropy against softmax(−ADE) over the 32, full batch, Adam, numpy. Hidden size and temperature won a sweep over {16, 32, 64, 128} × {0.3, 0.5, 1.0} ranked on held-out train scenes only; the sweep also computed the benchmark score of every configuration and the held-out winner was the benchmark's best too, so the paddock page says the benchmark 'was not unseen'. Candidates were dumped through paddock's own KITScenes harness (tools/models/drivejepa/paddock_candidates.py, DRIVEJEPA_IMAGE_GEOMETRY=openscene); that step is paddock's, the data and training steps are gallops'.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC01.kitscenes-train-h128-t1 ★ | KITScenes ADE@3s (m) 2.521 → **0.973** (n=223 samples)<br>all 223 benchmark samples; SE 0.174; t -8.0; paired −1.548 m with scene-clustered SE over 140 scenes; best of the 32 (any selector's ceiling) 0.409; the recorded drivejepa row (devkit loader) 2.364; FDE@3s 2.082; better than the model's pick on 155 samples, worse on 46 | Drive-JEPA's own pick, same path and inputs | ●●●○ strong · adapted, open-loop | paddock docs/models/drivejepa_rerank.md, run 20260921T210621-d7ed08 |
| GC01.kitscenes-train-h128-t1 | KITScenes ADE@3s (m) 1.159 → 0.973 (n=223 samples)<br>all 223 benchmark samples; SE 0.050; t -4.9; paired −0.186 m: it is not just picking 'keep your speed' | the candidate closest to constant velocity | ●●○○ moderate · adapted, open-loop | paddock docs/models/drivejepa_rerank.md, run 20260921T210621-d7ed08 |
| GC01.kitscenes-train-h128-t1 | KITScenes pick ADE@3s on the held-out train scenes (m) 2.515 → 1.050 (n=853 keyframes)<br>55 held-out train scenes, the epoch-selection split; best of the 32 here 0.502; this is the split the epoch was chosen on | Drive-JEPA's own pick | ●●○○ moderate · adapted, open-loop, selected-on-report | paddock docs/models/drivejepa_rerank.md; checkpoint.json notes.holdout_pick_ade |
| GC01.kitscenes-train-h128-t1 | nuScenes cumulative L2 (m) 0.641 → 0.587 (n=5,119 samples)<br>val, all 5,119 samples over 150 scenes, camera-frame fix; t -2.8; zero-shot; paired −0.054 m with SEs over the 150 scenes; ADE@3s 0.949 against 1.069 (t −3.4); point collisions 1.21% against 1.66%, not separated (t −1.5) | constant velocity | ●●○○ moderate · open-loop | paddock docs/models/drivejepa_rerank.md, run 20261012T220930-6194bd |
| GC01.kitscenes-train-h128-t1 | nuScenes cumulative L2 (m) 1.360 → 0.587 (n=5,119 samples)<br>val, all 5,119 samples over 150 scenes, camera-frame fix; t -16; paired −0.772 m, but it mixes two changes: this model runs on the OpenScene crop, drivejepa on native cameras. The selector's own effect at the same pixels is unmeasured on nuScenes | drivejepa with the fix, native cameras (run 20261012T215049-798a0d, not tracked) | ●●○○ moderate · open-loop | paddock docs/models/drivejepa_rerank.md; gallops branch docs/rerank-nuscenes-result |
| GC01.kitscenes-train-h128-t1 | neuroNCAP ncap_score 3.01 → 1.85 (n=9 episodes)<br>scene 0103, three scenario pairs, seeds 0-2, seeing model (camera-frame fix); hit in 6 of 9 against 5; a subset, not a neuroNCAP score. The re-ranker's run 20261012T231620-01570a is not tracked | drivejepa on the same nine episodes (20261012T205915-7153f0) | ●○○○ argued · smoke, no-record | paddock docs/models/drivejepa_rerank.md (Limits); paddock docs/ROADMAP.md |

## Checks that decided something

- **Input-side explanations before the selector** (19 Sep): none moved the pick beyond noise: acceleration zeroed or smoothed ≤ 0.07 m (t ≤ 1.75), route command ≤ 0.07 m (t ≤ 1.75), ego origin at the rear axle −0.09 to +0.02 m (t ≤ 0.7), OpenScene lens distortion −0.14 m (t −1.27); best of 32 stayed 0.35–0.39 m and the pick's mean rank 14.6–15.3 → the deficit is the scorer's choice, so the first target became a re-ranker, not a speed head
- **Rebuild from gallops' data steps**: `gallops.data.rerank keyframes` lists the same 5,711 keyframes the checkpoint trained on; `dataset` on the original dump gives every array of the original train.npz unchanged (its sha256 is checkpoint.json's dataset_sha256); a 5-sample smoke matched the paddock run to the dump's 4-decimal rounding → the data steps moved from paddock into gallops (gallops.data.rerank)
- **The re-ranker as the picker on neuroNCAP candidate dumps (camera-blind)** (23 Sep): picked a clear plan on 0 of the 322 ticks where the driven plan hits although one of the 32 was clear; it never picks a stop → not quoted on the model page, because the dumps were camera-blind; the model stays open-loop only until a full neuroNCAP run pairs it with drivejepa

## Evidence outside this repo

- paddock docs/models/drivejepa_rerank.md: the model page with every number, the rebuild steps and the limits; adapter src/paddock/adapters/models/drivejepa_rerank/ (paddock #54, nuScenes #66).
- paddock results/kitscenes/drivejepa_rerank/20260921T210621-d7ed08 and results/nuscenes/drivejepa_rerank/20261012T220930-6194bd: the tracked runs.
- The neuroNCAP 0103 run neuroncap/drivejepa_rerank/20261012T231620-01570a is not tracked in paddock; only its numbers are, in docs/models/drivejepa_rerank.md.
- The input-side checks (acceleration, route command, ego origin, lens distortion) are on paddock branch drivejepa/selection-diagnostics, not merged.
- The nuScenes headline is added to this repo's README on the unmerged branch docs/rerank-nuscenes-result.
