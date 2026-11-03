<!-- Rendered from experiment.toml by `gallops experiments render --write`.
     Edit that file, not this one: make check fails when the two differ. -->

# GC04 · Drive-JEPA retrained on the Bench2Drive base set: Split-v4 and the Final

**Navvrat** · Bench2Drive · ✅ positive · 29 Sep – 13 Oct 2026 · gap-closing

[← all experiments](../../README.md#gap-closing-experiments) · [the account](README.md) · [experiment.toml](experiment.toml)

## Question

Released Drive-JEPA (NAVSIM-v2, ViT-L) drives paddock's 50 Bench2Drive routes to a mean driving score of 57.60 at its native 2 Hz. Does retraining parts of it on the Bench2Drive base set, with the backbone never trained, raise that: first only the scorer head, split into yes/no safety heads (Split-v4), then also the proposal generator with MTD-H teacher paths and an area head (the Final)?

## What we learned

Split-v4 reached a mean DS of 68.1 and the Final 73.6 on 50 routes, against 57.60 for the released model, but each is one run per route on routes that also chose the variant, with no run record committed and same-spot twins of 26 eval routes in the training set. Split-v4's head stages reproduce bitwise from this repo; the Final's three training stages have not been rerun.

## Runs

| Run | Started from | Trained | Data | Code | Checkpoint |
|---|---|---|---|---|---|
| **GC04.split-v4**<br>Split-v4: the scorer head retrained as yes/no heads (collision, lane, TTC, red light); three head stages, split-v4aug kept | released Drive-JEPA (NAVSIM-v2, ViT-L) | scorer head (scorer.pred_score) (1.32 M); frozen: encoder, generator | bench2drive `base less the 150 held-out clips, + route-start frames` 19,950 frames (train) — base 20,449 + route-start 3,000 rows less the 150 held-out clips; labels iPad v2, col/lane/ttc v3, red v4 (Bench2Drive's stop-line rule); split-v4aug adds 8,316 speed-randomised copies of training light frames. manifest: recipes/drivejepa_b2d/data_split_v4.json<br>bench2drive `base, 150 held-out clips` 3,499 frames (selection) — chose every kept epoch; list recovered 29 Sep from the feature cache, confirmed by the bitwise rerun<br>bench2drive `50 eval routes` 50 routes (eval) — the same routes chose which head and which selection rule to keep: selection on the test set | [drivejepa_b2d.toml#split-v4aug](../../recipes/drivejepa_b2d.toml) | `drivejepa_scorer_b2d/split-v4/step 8840`<br>sha256 not recorded |
| **GC04.final**<br>the Final: generator fine-tuned with MTD-H teachers (batch-fixed), split head on its candidates (m05b labels), area head, progress guard | GC04.split-v4 | generator (init_feature, hist_encoding, shared refiner), split head, area head (scorer.pred_area); frozen: encoder | bench2drive `base less the 150 held-out clips, + route-start frames` (train) — as Split-v4, plus the backbone-feature cache, the MTD-H teacher set (Bench2Drive frames only) and the new generator's relabelled candidates. manifest: recipes/drivejepa_b2d/data_final.json (the split head also reads data_split_v4.json)<br>bench2drive `base, 150 held-out clips` 3,499 frames (selection) — chose the kept epoch of all three stages<br>bench2drive `50 eval routes` 50 routes (eval) — the same routes chose which head and which selection rule to keep: selection on the test set | [drivejepa_b2d.toml#final-generator](../../recipes/drivejepa_b2d.toml) | `drivejepa_scorer_b2d/final`<br>sha256 not recorded |

- **GC04.split-v4:** Stages ipad-base-start (kept epoch 17) → split-full (epoch 32, mean AUC 0.940) → split-v4aug (epoch 20, mean AUC 0.940), each initialised from the one before. The 32 proposals are unchanged; only the pick changes. Driven with DRIVEJEPA_PICK=split, DRIVEJEPA_SPLIT_MODE=exponent, DRIVEJEPA_SPLIT_K=4, which are part of the model. The deployed file (release + split-v4aug head, checked tensor by tensor) was written by paddock-ops' writer; this repo's write_checkpoint has never been run on the release.
- **GC04.final:** Generator from the release with its own multimodal loss (expert winner-takes-all + 0.5 × MTD-H pseudo-teachers, up to 4 per frame): Adam 2e-5, batch 32, 10 epochs, seed 0, kept epoch 10 by held-out objective, with the batch fix; holds ~26 GB of feature cache in RAM. Split head from split-v4aug on the new generator's candidates and m05b labels (0.5 m margin on moving actors, expert-masked), kept epoch 25. Area head from the release on v6 step labels, kept epoch 40, shipped beside the weights as area_head.pt. Progress guard (DRIVEJEPA_SPLIT_GUARD=1). Policy env from tags/final-73.6. Deployed file = release + generator + m05b split head, checked tensor by tensor 13 Oct; none of the three stages rerun from this repo.

## Results

★ marks the headline.

| Run | Result | Baseline | Strength | Source |
|---|---|---|---|---|
| GC04.split-v4 | Bench2Drive mean driving score 68.1 (n=50 routes)<br>50 routes, one run each, 30 Sep 2026; inference rate not recorded for these runs; released Drive-JEPA on paddock's 50 routes is 57.60 at 2 Hz and 31.71 at 20 Hz | — | ●○○○ argued · adapted, selected-on-report, no-record | experiments/GC04_bench2drive/README.md |
| GC04.split-v4 | offline split-v4aug mean AUC, 150 held-out clips 0.940<br>kept epoch 20, chosen on the same clips; split-full also 0.940 | — | ●●○○ moderate · selected-on-report | experiments/GC04_bench2drive/README.md |
| GC04.final ★ | Bench2Drive mean driving score 57.60 → **73.6** (n=50 routes)<br>50 routes, one run each, 6 Oct 2026; the Final is driven at PADDOCK_INFER_HZ=2, as the baseline was; no significance test; the base set holds same-scenario, same-location twins of 26 eval routes | released Drive-JEPA at 2 Hz (paddock native50-2hz-40c29d24c37d542a-447e23) | ●○○○ argued · adapted, selected-on-report, no-record | experiments/GC04_bench2drive/README.md |
| GC04.final | Bench2Drive mean driving score 68.1 → 73.6 (n=50 routes)<br>50 routes, one run each; Bench2Drive runs are not repeatable here (rain and foliage render differently), so a single-run difference of a few points is not a result | Split-v4 | ○○○○ unverified · adapted, selected-on-report, no-record | experiments/GC04_bench2drive/README.md |

## Checks that decided something

- **Batch fix: the refiner mixes batch elements** (5 Oct): a dump at batch 64 was a median 1.49 m off the same frames at batch 1; upstream mixes elements in three places (batch_fix.py). Deployment runs batch 1, so closed-loop numbers are unaffected → every generator run behind the Final trains with the fix
- **Split-v4's head stages rerun from this repo** (13 Oct): ipad-base-start, split-full and split-v4aug rerun on the recorded data, each from the previous rerun's output: every head_best.pt bitwise identical (same GPU and torch build); so the held-out list is the original → the recipe stands for Split-v4's training (verified); the dump holds its files (strong: copied ~30 min after the rerun read the same paths)
- **Deployed checkpoints against the release** (13 Oct): Split-v4: its 10 head tensors equal split-v4aug's head_best.pt and its other 495 equal the release. Final: its 10 head tensors equal the m05b head, the generator's 45 tensors its 171, nothing else differs → the recipe's writer replaces the same tensors (argued: never run on the release)

## Evidence outside this repo

- Bench2Drive runs behind 68.1 (30 Sep) and 73.6 (6 Oct): no run record in any repository. paddock's drivejepa_scorer_b2d adapter that drove them existed on 13 Oct only as uncommitted files in one local paddock worktree (rerank-native), on no paddock branch.
- Baseline: paddock results/bench2drive/drivejepa/native50-2hz-40c29d24c37d542a-447e23 (released Drive-JEPA, 2 Hz, DS 57.60; 31.71 at 20 Hz), paddock docs/benchmarks/bench2drive-native-rate.md.
- Training data: the frozen dump drivejepa_b2d-v1 ([datasets.drivejepa_b2d]), made 13 Oct by copying the files the stages read; every file's bytes and sha256 in recipes/drivejepa_b2d/data_split_v4.json and data_final.json. The tooling that built the rows, features, labels and teachers is paddock-side and not in this repo.
- Deployed checkpoints: written by paddock-ops' writers (scorer-retrain/write_scorer_checkpoint.py, generator-ft/step4_write_ckpt.py) with a hand-written sidecar; not under this machine's weights root.
- What still fails on the Final (left turns 0/7, right turns 0/3, ambulance 0/5, road works and two-way accidents 0/5, a stall after a first stop): the team's status page, citing Navvrat's Bench2Drive write-up; in no repository.
