# H01 — 4-camera base: Drive-JEPA's planner on frozen WA-JEPA tokens

**Owner:** Navvrat · **Priority:** 1 of 11 · **Started:** 16 Oct 2026 · **Status:** running — gated by hand, overall
HOLD; no paddock ref carries its adapter yet (hypothesis)

The facts are in [experiment.toml](experiment.toml), and what is off the repository is in
[POINTERS.md](POINTERS.md). This page is the account.

## Why

Released Drive-JEPA reads only its front camera. This puts its planner on WA-JEPA's frozen encoder over four
cameras, and asks whether that base is at least as good where the gate measures. The team agreed it as the base on
19 Oct; every later hypothesis needs that baseline.

## What was done

Drive-JEPA's planner (32 proposals × 8 waypoints, four refinement rounds, a
scorer that picks one) trained on NAVSIM navtrain with its ViT replaced by
AFARI's WA-JEPA encoder, frozen. The encoder runs once, over navtrain's four
cameras (b0, f0, l0, r0), and the planner trains from the cached tokens.

**The paddock adapters are not in paddock yet.** These checkpoints are driven by
paddock's `wajepa_dj` (the 4-camera planner) and `wajepa_dj_run3a` (the sub-score agent-look model) adapters, which on
27 Oct 2026 existed only as uncommitted files in one local paddock worktree
(`rerank-native`); paddock `main` cannot score a checkpoint this recipe writes
until they land there.

| Run | navtest EPDMS, human filter | navtest, CSV score | Bench2Drive, 50 routes, mean DS (2 Hz) | HUGSIM HD-Score, 419 scenarios |
|---|---:|---:|---:|---:|
| Released Drive-JEPA (ViT-L, the authors' checkpoint, same evaluation) | 87.78 | 84.49 | 57.60 | 33.8 † |
| **The 4-camera planner** | **85.27** | **81.98** | 50.07 (18 Oct); 51.70 and 51.95 with the command fix (two runs) | not run |
| **The sub-score agent-look model**, default pick | **87.00** | **83.89** | | |
| **The sub-score agent-look model**, locked pick | **87.30** | **84.09** | 63.29 with the command fix (one run, 26 Oct) | 40.4 † |

† HUGSIM's 22 Oct campaign, which paddock's HUGSIM page marks superseded: six waypoints of each plan sent, the release with momentum off, and this model's run not stored in paddock. Paddock's current figure for the release (momentum on, the whole 4 s plan) is 37.1.

The sub-score agent-look model and its agent decoder: [H04's account](../H04_subscore_scorer_navsim/README.md).

Recipe: [`recipes/drivejepa_wajepa.toml`](../../recipes/drivejepa_wajepa.toml).
`gallops recipe drivejepa_wajepa` prints every command for this machine;
`gallops recipe drivejepa_wajepa <stage> --run` runs one.

## How the code is kept

Both runs trained inside a Drive-JEPA checkout (linhanwang/Drive-JEPA at
`548bb82`, upstream `master`) with navsim's own `run_training.py`. What they
changed is a set of files in navsim's agent packages. Those files are in
[`recipes/drivejepa_wajepa/overlay/<run>/`](../../recipes/drivejepa_wajepa/overlay),
exactly as they ran, and `overlay/<run>.json` lists each file's sha256 as
tracked and as it ran. The `code-<run>` stage rebuilds the tree: `git archive`
of the pinned commit from a site's checkout (the checkout is not touched), the
overlay copied over it, and every overlay file checked against the manifest
first.

A tracked file and the file as it ran differ in one place only: a default that
named a machine path now reads the environment (the recipe sets it). The
manifest's `note` says which file and which lines.

The overlay is not linted (`make lint` covers `src/` and `tests/`): it is
upstream's code with our edits, and reformatting it would break the hashes that
say it is what ran.

## The 4-camera planner

Called Run 2 while it ran: the run ID is `H01.four-camera-planner`. Drive-JEPA's planner,
trained from scratch on frozen WA-JEPA tokens from four cameras. Code and checkpoint paths keep
the name it trained under (`run2`).

Spec: the user's of 15–16 Oct 2026 ("frozen WA-JEPA features → Drive-JEPA
planner"). The changes, all in `drive_jepa_perception_based/`:

| What | Where |
|---|---|
| The frozen encoder: WA-JEPA's V-JEPA 2.1 ViT-L/16 built with WA-JEPA's own code, `encoder.*` loaded strict, `eval()`, no grad, bf16 autocast. Per camera one clip of 2 frames (t − 0.5 s, t), each `img[28:-28]` resized to 512×256 | `bevformer/wajepa.py` |
| Light features (`lidar2img`, `img_shape`, `ego_status`, as v2's builder) and the token builder | `drive_jepa_features.py` |
| Tokens through `Linear(1024, 256)` plus a camera embedding in place of the ViT; `lidar2img` mapped to the 512×256 image once in `forward` | `drive_jepa_model.py` |
| Footprint look on f0, l0, r0 with per-sample routing, `im2col_step = 32` | `bevformer/spatial_cross_attention.py` |
| Global look: each waypoint query attends to rays from all four cameras, `W_o` zero-initialised so it starts as a no-op | `global_look.py`, `traj_refiner.py` |
| Batch fixes: temporal self-attention's `value[0::2]`, the encoder's reference `repeat_interleave`; anchor heights repeated per query (before, a query had 4 distinct anchors, each 4×) | `bevformer/temporal_self_attention.py`, `bevformer/encoder.py` |
| Per-epoch checkpoints + `last.ckpt` (`DRIVEJEPA_CKPT_DIR`); label workers from `DRIVEJEPA_SCORE_WORKERS` | `drive_jepa_agent.py` |

| Stage | What | Output |
|---|---|---|
| `code-run2` | upstream + the 4-camera planner's overlay | `{runs}/drivejepa_wajepa/run2/code` |
| `dump-tokens` | navtrain's 103,288 scenes: light features, targets, tokens (float16, 4 × 512 × 1024). 16 Oct: 0 errors, 5.43 h | `{data.wajepa_tokens}` |
| `verify-tokens` | the cache against a live recompute | exits non-zero on a mismatch |
| `run2-train` | 20 epochs, batch 64, Adam 1e-4, 16-mixed, seed 0; Drive-JEPA's losses and its split (85,109 train / 18,179 val scenes); 7.8 M trainable parameters, 6.1 GB of VRAM, 1.65 s a step | every epoch's checkpoint |
| `run2-navtest` | navsim v2 `run_pdm_score.py` on navtest's 12,146 scenes, tokens live, no momentum-aware selection | navsim's CSV |
| `run2-navtest-score` | the two navtest scores and every sub-score | `navtest.json` |
| `run2-checkpoint` | epoch 19 into `{weights}/wajepa_dj/run2/step_<n>/` with `checkpoint.json` | |

The planner starts from scratch: training loads no released weights
(`checkpoint_path: ''`). The last epoch is kept, not the best: val/score_epoch
was 0.880 at epoch 19, 0.882 at epoch 17.

navtest sub-scores, the 4-camera planner / released: NC .980 / .985, DAC .972 / .986, DDC .990 / .991,
TLC .998 / .998, EP .879 / .891, TTC .973 / .979, LK .967 / .976, HC .977 / .978,
EC .401 / .478.

## The regression gate

Run by hand from local paddock worktrees, because no paddock ref carries the adapter. Against released Drive-JEPA:

| Benchmark | Metric | Released | The 4-camera planner | t | Units | Verdict |
|---|---|---:|---:|---:|---:|---|
| KITScenes | ADE (3 s), m | 2.5375 | 2.0010 | 3.10 | 117 | GAIN |
| nuScenes | L2, 1-2-3 s, m | 2.4314 | 2.0570 | 3.41 | 143 | GAIN |
| NAVSIM, 1,018 scenes | EPDMS | 0.8572 | 0.8440 | −1.28 | 126 | NEUTRAL |
| neuroNCAP, quick check (3 pairs) | NCAP | 3.1794 | 4.0237 | 0.98 | 3 | NEUTRAL |
| Bench2Drive, crash check (route 3717) | DS | 37.67 | 29.20 | | 1 | TOO FEW UNITS |
| AlpaSim, crash check (1 clip) | scene score | 0.0000 | 0.0000 | | 1 | TOO FEW UNITS |
| neuroNCAP, closed loop (15 pairs) | NCAP | 3.6792 | 3.2800 | −0.96 | 15 | NEUTRAL |
| Bench2Drive, closed loop (7 routes) | DS | 59.04 | 80.24 | 1.08 | 7 | NEUTRAL |
| AlpaSim, closed loop (10 scenes) | scene score | 0.4162 | 0.4503 | 0.24 | 10 | NEUTRAL |

Every pair is on the gate's current subsets (paddock #105, 23 Oct; the quick checks #113, 26 Oct), against the queue's
stored released Drive-JEPA runs, and is copied from the gate's output into [experiment.toml](experiment.toml), one block,
in the order every gated record uses. KITScenes and nuScenes ran on 28 Oct, neuroNCAP and AlpaSim on 27 Oct, NAVSIM and
Bench2Drive on 20 Nov. Bench2Drive drives with the command fix (FOLLOW_REFERENCE fed as STRAIGHT), which this model's
closed-loop runs use. On AlpaSim the gate first printed CONFIG DIFFERS on `bridge_args`, `driver_config` and
`driver_profile`: this model drove AlpaSim through AutoE2E's driver profile (six pinhole cameras, front f 990.8 px,
1920 × 1018, no distortion, global shutter, against Drive-JEPA's one front camera at f 1545 px, 1920 × 1080, distortion
left in, rolling shutter; and a wrapper replanning every 0.5 s on the newest frame against Drive-JEPA's own 11-frame
planner), set by an uncommitted edit to paddock's AlpaSim harness. Its two rows are the same runs re-paired with those
three keys allowed; what each difference does to the score is not measured. Overall **HOLD**: nothing regressed, and no
closed-loop benchmark gained.

The 21 Oct gate, on the subsets paddock held before 23 Oct (NAVSIM 1,000 tokens, KITScenes 140 scenes, nuScenes 30 scenes,
Bench2Drive 7 of the 50 routes picked by the baselines' spread, at 2 Hz, without the command fix), is superseded by these
pairs: NAVSIM 0.875 → 0.863 (t −1.29), KITScenes 3.222 → 1.969 (t 6.44, GAIN), nuScenes 2.497 → 1.956 (t 3.03, GAIN), Bench2Drive 58.801 → 62.757 (t 0.36), HOLD.

## Decision

Not decided. The gate says HOLD, and the record stays `running` until the team takes it; no row in the decision log
closes it yet.

## The data

| What | Where it comes from |
|---|---|
| NAVSIM navtrain logs and sensor blobs, navtest | the OpenScene / NAVSIM download, `[datasets.navsim] path` |
| `navtrain_current_1` | the split Drive-JEPA's download script skips; fetched from navsim's S3 (cameras b0, f0, l0, r0 only) and laid over the blobs, `[datasets.navsim] current1` |
| The training metric cache and the 8,192-anchor teacher scores | Drive-JEPA's own preprocessing, at `{data.navsim.exp}/Drive-JEPA-cache/{train_metric_cache_v2_pruned, anchors_scores_index_v2}`; not built here |
| The navtest metric cache | navsim's `run_metric_caching.py`, `[datasets.navsim] metric_cache` |
| WA-JEPA | AFARI-Research/WA-JEPA at `bec2966` (`[code] wajepa`), encoder weights `{weights}/WA-JEPA/model_state_dict.pt` |
| The token cache | `dump-tokens`; `agent-targets` adds the agent decoder's targets beside the tokens |
| The held-out logs | `recipes/drivejepa_wajepa/heldout_logs.txt`: 51 navtrain logs, the agent decoder's held-out set and the sub-score agent-look model's validation split |
| The sub-score agent-look model's teacher list | `teacher-rearok`, beside Drive-JEPA's own list in `[datasets.navsim] drivejepa_cache`, where the metric cache's metadata says the cache is |

## Verified, and how

| Claim | Strength |
|---|---|
| The 4-camera planner's overlay is the code that trained: each file equals the 23 Oct snapshot of the training checkout (upstream + `run2_tracked_23oct.patch` + the untracked files), and no file changed between the launch (16 Oct 17:48) and that snapshot except `drive_jepa_model.py` (19 Oct: a `WAJEPA_GLOBAL_LOOK=0` switch, default on) and `drive_jepa_features.py` (22 Oct: an added class; the overlay holds the file as it was before) | verified, file by file (27 Oct) |
| With the 19 Oct switch at its default, the model is the one that trained | verified: 500 navtest scenes reproduce the 17 Oct epoch-19 CSV on every column (max difference 0) |
| `navtest-score` gives the numbers above from the evaluations' CSVs (85.27 / 81.98, released 87.78 / 84.49) | verified (27 Oct) |
| The recipe's training and evaluation overrides are the runs' own | verified against each run's `code/hydra/overrides.yaml` |
| `drive_jepa_config.py` as it was at launch | **not kept**: the manifest's `as_run_sha256` is null. It differs from the run's in the pick defaults (locked on 25 Oct, after training) and the `agent_decoder_ckpt` default. The gradient never reads the pick, but checkpoint selection (val/score_epoch) did, at gates 1, route 2, no blend; `run3a-train` passes those defaults explicitly |
| `run3a-checks` runs from this repo in the tree `code-run3a` builds | verified (27 Oct): checks 1–3 pass with the 24 Oct numbers (transform error 1.8e-15 m, 768 / 768 waypoints, attention on nothing 1.0000) |
| `dump-tokens` and `verify-tokens` behave as the scripts that ran | **argued**: ported with only the arguments changed (and the dump's exit code); neither was rerun from this repo |
| `run2-train`, `agent-decoder` and `run3a-train` reproduce the recorded checkpoints | **argued**: not rerun. NAVSIM training is seeded but not bitwise deterministic on this GPU |

## Before quoting a number from these checkpoints

- **NAVSIM is not zero-shot**: navtrain trained the planner.
- The 17 Oct evaluation carries the label the spec asked for: "encoder
  possibly selected on navtest".
- **Bench2Drive is zero-shot** (a NAVSIM-trained planner in CARLA), at 2 Hz,
  one run per route, and runs are not repeatable: the two command-fix runs of
  the 4-camera planner differ by a mean of 9.1 DS per route.
- Every closed-loop number above is one run per route or scenario.
- The released Drive-JEPA's 87.78 comes from the same `run_pdm_score.py`, the
  same metric cache and no momentum-aware selection, run on a clean upstream
  tree; the paper's numbers use momentum-aware selection.
