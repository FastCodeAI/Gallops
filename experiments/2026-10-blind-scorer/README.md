# The scorer experiments of 9–12 Oct 2026 — camera-blind, kept for the record

**Every number these produced was measured on a camera-blind Drive-JEPA.** The
inputs placed the cameras relative to nuScenes' LIDAR_TOP (yawed −90°), so no
image sample landed in the front image; see the correction at the top of
[docs/scorer-cache.md](../../docs/scorer-cache.md) and #8. The scripts and small
results are kept so the record is complete and the pipeline can be re-run on
sighted features. They are not the current state of anything.

What they measured, and what was concluded at the time, is in
[docs/scorer-cache.md](../../docs/scorer-cache.md). What survives the bug: the
grader, the sub-score target definitions, the vocabulary and probe generators,
the two-pass injection with its drift guards, and the real-nuScenes conversion
(all in `src/`). What does not: every fold, learning curve, real-vs-rendered
result and closed-loop score below.

## Paths

Copied as run, with each machine root replaced by a template name. Expand them
with `gallops.toml` before running anything: `{runs}`, `{weights}`,
`{data.nuscenes}`, `{data.navsim.exp}`, `{data.drivejepa_repo}` are the usual
keys; `{dj_cache}` is the scorer-cache root, `{paddock}` / `{paddock_runs}` a
paddock checkout and its runs root, `{gallops}` this repo, `{conda_envs}` /
`{ncap_envs}` the interpreters, `{this_folder}` this directory. Large outputs —
the feature caches (`{dj_cache}/v1`, `v2`, `v3_nusc`, 25 GB), fold logits and the
five scorer `.pt` files — stay where they are and are not in git.

## Layout

| Path | What it is |
|---|---|
| `ops/dj_cache/vocab_v1.py`, `probe_v1.py`, `vocab_v1_probes.py` | the 32-swerve vocabulary (slots 0–31) and 32 straight probe paths (32–63) the v2 and v3 caches were built with |
| `ops/dj_cache/vocab_review/` | the review agent's reports that produced vocabulary v1 |
| `ops/dj_cache/run_*.sh` | the detached chains: cache → labels → folds (`run_cache.sh` v1, `run_cache_v2.sh` v2 at batch 1, `run_nusc.sh` real nuScenes, `run_folds.sh`) |
| `scripts/probe_folds.py` | leave-one-scene-out folds over probes + geometry; variants and the `real` / `realmask` / `realw` flags → `results/v2/probe_*.json` |
| `scripts/context_folds*.py` | the context-feature ablations → `results/v1/context_*.json` |
| `scripts/learning_curve.py` | training on 1–4 scenes → `results/v2/learning_curve_*.json` |
| `scripts/aux_heads.py`, `hazard_targets.py` | the released auxiliary heads against our labels; the hazard-localisation target |
| `scripts/eval_real.py` | override and kept-safe rates on the 66 held-out real scenes |
| `scripts/final_train.py` | the 5-seed scorer the closed-loop adapter loaded; its sidecars are in `results/subscore_weights_sidecar/` |
| `scripts/viz_training.py`, `swerve_vocab.py`, `verify_vocab.py`, `vocab_provisional.py`, `build_plans.py`, `render_v2.py`, `build_page.py`, `page_template.html` | training-sample plots, vocabulary checks, and the shared page of swerves on scenes |
| `scripts/place_test.py`, `place_analyse.py` | the test that exposed the bug: on GT drives, the same plan at the same pose and moment got bit-identical answers whatever the inserted car did |
| `scripts/gt_feature_probe.py`, `lidar_frame_test.py`, `adapter_blind_check.py` | the diagnosis: black and noise frames change nothing; 0.0% of image samples in the frame with LIDAR_TOP, 85% with the ego frame; paddock's adapter old vs fixed |
| `scripts/ego_on_road.py` | the drivable-raster check on real nuScenes (logged ego on the map 99%+ per city) |
| `ops/dj_subscore/` | the paddock adapter checked against the cache (`verify.py`) and the 0103 closed-loop smokes |
| `ops/target_viewer/` | the per-tick target viewer (every plan, its labels and the held-out predictions) |
| `results/v1`, `v2`, `v3_nusc` | the small JSON outputs of each cache's folds and manifests |
