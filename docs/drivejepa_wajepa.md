# Drive-JEPA's planner on frozen WA-JEPA tokens

Drive-JEPA's planner (32 proposals × 8 waypoints, four refinement rounds, a
scorer that picks one) trained on NAVSIM navtrain with its ViT replaced by
AFARI's WA-JEPA encoder, frozen. The encoder runs once, over navtrain's four
cameras (b0, f0, l0, r0), and the planner trains from the cached tokens.

**The paddock adapters are not in paddock yet.** These checkpoints are driven by
paddock's `wajepa_dj` (Run 2) and `wajepa_dj_run3a` (Run 3A) adapters, which on
27 Oct 2026 existed only as uncommitted files in one local paddock worktree
(`rerank-native`); paddock `main` cannot score a checkpoint this recipe writes
until they land there.

| Run | navtest EPDMS, human filter | navtest, CSV score | Bench2Drive, 50 routes, mean DS (2 Hz) | HUGSIM HD-Score, 419 scenarios |
|---|---:|---:|---:|---:|
| Released Drive-JEPA (ViT-L, the authors' checkpoint, same evaluation) | 87.78 | 84.49 | 57.60 | 33.8 |
| **Run 2** | **85.27** | **81.98** | 50.07 (18 Oct); 51.70 and 51.95 with the command fix (two runs) | not run |
| **Run 3A**, default pick | **87.00** | **83.89** | | |
| **Run 3A**, locked pick | **87.30** | **84.09** | 63.29 with the command fix (one run, 26 Oct) | 40.4 |

Recipe: [`recipes/drivejepa_wajepa.toml`](../recipes/drivejepa_wajepa.toml).
`gallops recipe drivejepa_wajepa` prints every command for this machine;
`gallops recipe drivejepa_wajepa <stage> --run` runs one.

## How the code is kept

Both runs trained inside a Drive-JEPA checkout (linhanwang/Drive-JEPA at
`548bb82`, upstream `master`) with navsim's own `run_training.py`. What they
changed is a set of files in navsim's agent packages. Those files are in
[`recipes/drivejepa_wajepa/overlay/<run>/`](../recipes/drivejepa_wajepa/overlay),
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

## Run 2

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
| `code-run2` | upstream + the Run 2 overlay | `{runs}/drivejepa_wajepa/run2/code` |
| `dump-tokens` | navtrain's 103,288 scenes: light features, targets, tokens (float16, 4 × 512 × 1024). 16 Oct: 0 errors, 5.43 h | `{data.wajepa_tokens}` |
| `verify-tokens` | the cache against a live recompute | exits non-zero on a mismatch |
| `run2-train` | 20 epochs, batch 64, Adam 1e-4, 16-mixed, seed 0; Drive-JEPA's losses and its split (85,109 train / 18,179 val scenes); 7.8 M trainable parameters, 6.1 GB of VRAM, 1.65 s a step | every epoch's checkpoint |
| `run2-navtest` | navsim v2 `run_pdm_score.py` on navtest's 12,146 scenes, tokens live, no momentum-aware selection | navsim's CSV |
| `run2-navtest-score` | the two navtest scores and every sub-score | `navtest.json` |
| `run2-checkpoint` | epoch 19 into `{weights}/wajepa_dj/run2/step_<n>/` with `checkpoint.json` | |

The planner starts from scratch: training loads no released weights
(`checkpoint_path: ''`). The last epoch is kept, not the best: val/score_epoch
was 0.880 at epoch 19, 0.882 at epoch 17.

navtest sub-scores, Run 2 / released: NC .980 / .985, DAC .972 / .986, DDC .990 / .991,
TLC .998 / .998, EP .879 / .891, TTC .973 / .979, LK .967 / .976, HC .977 / .978,
EC .401 / .478.

## The agent decoder

Run 3A's agent look reads detected agents, so a decoder is trained first, on its
own, from the same token cache (`agent_decoder.py`, 4.25 M parameters for D):
learned slots, each with a reference point in metres, attend to all 2,048
tokens of the four cameras with the token rays as position codes; each layer
moves its reference point, which is the box centre, and predicts class, box and
velocity. Hungarian-matched loss per layer.
Targets: every vehicle, pedestrian and cyclist within 32 m in x and y at time t,
nearest 64 kept. Train = navtrain minus the 51 held-out logs
(`recipes/drivejepa_wajepa/heldout_logs.txt`, sha1(log) % 20 == 0); navtest is
never read.

| Stage | What | Output |
|---|---|---|
| `code-agent-decoder` | upstream + Run 2's overlay + `agent_decoder.py` and the agent target builder | `{runs}/.../agent_decoder/code` |
| `extract-agents` | the agents of every cached sample, read once from the navtrain logs | `navtrain_agents.pkl` |
| `agent-targets` | `agent_targets.gz` beside each sample's tokens (never overwrites) | in `{data.wajepa_tokens}` |
| `agent-decoder` | AdamW 2e-4, weight decay 0.01, batch 64, fp16, cosine over 12 epochs, gradient clip 35; `QUERIES=300 LAYERS=3` | a checkpoint per epoch |
| `agent-decoder-eval` | held-out metrics of epoch 11 | JSON |

The slot count and depth were chosen by a sweep on the held-out logs (22–23 Oct:
64 × 3, 128 × 3, 64 × 6, 128 × 6, 300 × 3). The pick, **D = 300 slots × 3
layers, epoch 11** (the last of 12): recall 0.497 and precision 0.748 at 2 m,
recall of moving vehicles within 20 m 0.849, on 4,457 held-out samples.

## Run 3A

Spec: the user's of 23 Oct 2026 (revision 2). One new package,
`drive_jepa_run3a/`, a copy of Run 2's with these changes (Run 2's package is
untouched):

| What | Where |
|---|---|
| The agent decoder D loaded whole (queries and layers asserted) and trained on with its detection loss (weight 1) | `drive_jepa_model.py`, `agent_decoder.py` |
| Agent look after each round's refiner, one block shared by the 4 rounds: per waypoint, the k = 16 nearest detections (P(none) < 0.5) moved to the waypoint's time, a relative code in the waypoint's frame, attention with a learned "nothing" slot, `W_o` zero-initialised. The global look is removed | `agent_look.py`, `traj_refiner.py` |
| 12 scorer heads read from each path's 8 waypoint tokens (max over waypoints): gates at-fault collision, drivable area, direction, traffic light, rear safety; goals progress, TTC, lane keeping, comfort, route; edge distance; final score. BCE, smooth L1 for progress and edge distance | `score_module/scorer.py`, `drive_jepa_agent.py` |
| New labels per candidate: rear safety (a lane change that makes a car behind brake harder than 4 m/s²; IDM), edge distance, route share, progress over the 32 paths and PDM-Closed together; map auxiliary targets per waypoint | `score_module/run3a_labels.py`, `score_module/compute_navsim_score.py` |
| The pick: Σ gate weight × log σ(gate) + log of the weighted mean of σ(goals), blended with log σ(final score) | `drive_jepa_model.py` (`pick_score`), `drive_jepa_config.py` |
| Validation on the held-out logs, epoch means; `val/score_epoch` chooses the checkpoint | `planning/training/agent_lightning_module.py` |
| Train = navtrain minus the held-out logs (1,141 logs, 98,831 samples), val = the 51 held-out logs | `config/training/run3a_log_split.yaml`, `run3a_training.yaml` |

| Stage | What | Output |
|---|---|---|
| `code-run3a` | upstream + the Run 3A overlay | `{runs}/.../run3a/code` |
| `run3a-checks` | checks 1–3 of the spec on synthetic inputs: the relative transform, the k-nearest selection, the empty list | exits non-zero on a failure |
| `lane-centrelines` | every lane centreline and the lane graph of the four nuPlan maps, for the labels | one pickle per map |
| `teacher-rearok` | the teacher list: Drive-JEPA's v2 anchors minus those that fail rear safety (with the lane gate). 24 Oct: 17,257,415 → 17,201,172 anchors | beside the metric cache |
| `run3a-train` | 20 epochs, batch 64, Adam 1e-4, 16-mixed, seed 0; planner from scratch; refuses to start without the decoder checkpoint and the centrelines | every epoch's checkpoint |
| `run3a-pick-tune` | on epoch 13: the 12 head outputs for every held-out sample, then a grid over the gate weights, the route weight and the blend | `grid.json` |
| `run3a-navtest-default`, `run3a-navtest` | navtest under the defaults the checkpoint was chosen with, and under the locked pick | navsim's CSVs |
| `run3a-navtest-score` | the two navtest scores and every sub-score | `navtest.json` |
| `run3a-checkpoint` | epoch 13 into `{weights}/wajepa_dj_run3a/run3a/step_<n>/` with `checkpoint.json` | |

Selection and the pick, all on the 51 held-out logs: `val/score_epoch` was best
at epoch 13 (0.9173; epoch 19: 0.9102) under the default pick (gates 1, route 2,
no blend). The grid on epoch 13 then gave gates (0.5, 0.5, 1, 0.5, 0.5), route
2, blend 0.5: 0.9204 against the default's 0.9174 (best of 32: 0.9611). Those
weights are locked in the overlay's `drive_jepa_config.py`.

Training: 1,544 steps an epoch at about 2.0 s a step, 8.3 GiB of VRAM, 13.22 M
trained parameters. Loss shares over the first 50 steps: trajectory 52 %,
detection 26 %, the 12 heads 16 %, map auxiliary 1 %.

navtest sub-scores, Run 3A locked / Run 2: NC .986 / .980, DAC .984 / .972,
DDC .993 / .990, TLC .998 / .998, EP .900 / .879, TTC .981 / .973, LK .977 / .967,
HC .977 / .977, EC .406 / .401. Paired per scene: +2.03 [+1.65, +2.44] over Run 2,
−0.49 [−0.84, −0.11] against the release.

Bench2Drive (26 Oct, 50 routes, 2 Hz, the command fix, one run per route): DS
63.29, RC 79.02, 30 routes completed; Run 2 in the same setup 51.70 / 51.95 (two
runs). HUGSIM (419 scenarios, command only): HD-Score 40.4 against 33.8 for the
release, paired +6.6 [+3.6, +9.7]; KITTI-360 and Waymo have no rear camera, so
Run 3A drove those with a black rear image.

## The data

| What | Where it comes from |
|---|---|
| NAVSIM navtrain logs and sensor blobs, navtest | the OpenScene / NAVSIM download, `[datasets.navsim] path` |
| `navtrain_current_1` | the split Drive-JEPA's download script skips; fetched from navsim's S3 (cameras b0, f0, l0, r0 only) and laid over the blobs, `[datasets.navsim] current1` |
| The training metric cache and the 8,192-anchor teacher scores | Drive-JEPA's own preprocessing, at `{data.navsim.exp}/Drive-JEPA-cache/{train_metric_cache_v2_pruned, anchors_scores_index_v2}`; not built here |
| The navtest metric cache | navsim's `run_metric_caching.py`, `[datasets.navsim] metric_cache` |
| WA-JEPA | AFARI-Research/WA-JEPA at `bec2966` (`[code] wajepa`), encoder weights `{weights}/WA-JEPA/model_state_dict.pt` |
| The token cache | `dump-tokens`; `agent-targets` adds the agent decoder's targets beside the tokens |
| The held-out logs | `recipes/drivejepa_wajepa/heldout_logs.txt`: 51 navtrain logs, the agent decoder's held-out set and Run 3A's validation split |
| Run 3A's teacher list | `teacher-rearok`, beside Drive-JEPA's own list in `[datasets.navsim] drivejepa_cache`, where the metric cache's metadata says the cache is |

## Verified, and how

| Claim | Strength |
|---|---|
| The Run 2 overlay is the code that trained: each file equals the 23 Oct snapshot of the training checkout (upstream + `run2_tracked_23oct.patch` + the untracked files), and no file changed between the launch (16 Oct 17:48) and that snapshot except `drive_jepa_model.py` (19 Oct: a `WAJEPA_GLOBAL_LOOK=0` switch, default on) and `drive_jepa_features.py` (22 Oct: an added class; the overlay holds the file as it was before) | verified, file by file (27 Oct) |
| With the 19 Oct switch at its default, the model is the one that trained | verified: 500 navtest scenes reproduce the 17 Oct epoch-19 CSV on every column (max difference 0) |
| `navtest-score` gives the numbers above from the evaluations' CSVs (85.27 / 81.98, released 87.78 / 84.49) | verified (27 Oct) |
| The recipe's training and evaluation overrides are the runs' own | verified against each run's `code/hydra/overrides.yaml` |
| The Run 3A overlay is the code that trained: no file of the package, its configs or the Lightning module changed between the launch (24 Oct 17:34) and the overlay, except `drive_jepa_config.py` (25 Oct: the locked pick); the tree `code-run3a` builds equals the training checkout but for the machine-path defaults | verified, file by file (27 Oct) |
| `drive_jepa_config.py` as it was at launch | **not kept**: the manifest's `as_run_sha256` is null. Its only change is the pick defaults, which training never reads and checkpoint selection read at gates 1, route 2, no blend |
| The agent decoder's two files are the ones D trained with (last changed 22 Oct 23:11, D trained 23 Oct 15:56–17:11) | verified by timestamps and the code-agent-decoder tree's diff |
| `navtest-score` gives Run 3A's 87.00 / 83.89 and 87.30 / 84.09 | verified (27 Oct) |
| The ported Run 3A and agent decoder scripts compute what the originals did | verified: their syntax trees equal the originals' but for the edits each docstring names (paths to arguments or the environment, the data module's import, an exit code, one renamed variable) |
| `run3a-checks` runs from this repo in the tree `code-run3a` builds | verified (27 Oct): checks 1–3 pass with the 24 Oct numbers (transform error 1.8e-15 m, 768 / 768 waypoints, attention on nothing 1.0000) |
| `dump-tokens` and `verify-tokens` behave as the scripts that ran | **argued**: ported with only the arguments changed (and the dump's exit code); neither was rerun from this repo |
| `run2-train`, `agent-decoder` and `run3a-train` reproduce the recorded checkpoints | **argued**: not rerun. NAVSIM training is seeded but not bitwise deterministic on this GPU |
| Checks 4–7 of the Run 3A spec (W_o = 0 gives identical outputs, overfit 64 scenes, the first 50 steps, the lane-change labels) | ran on 24 Oct (numbers in this doc); their scripts are not in this repo |

## Before quoting a number from these checkpoints

- **NAVSIM is not zero-shot**: navtrain trained the planner.
- The 17 Oct evaluation carries the label the spec asked for: "encoder
  possibly selected on navtest".
- **Bench2Drive is zero-shot** (a NAVSIM-trained planner in CARLA), at 2 Hz,
  one run per route, and runs are not repeatable: the two command-fix runs of
  Run 2 differ by a mean of 9.1 DS per route.
- **Run 3A's pick was tuned on the same 51 logs that chose its epoch**, which
  are navtrain logs, not navtest; its navtest numbers were not used to choose
  anything.
- Every closed-loop number above is one run per route or scenario.
- The released Drive-JEPA's 87.78 comes from the same `run_pdm_score.py`, the
  same metric cache and no momentum-aware selection, run on a clean upstream
  tree; the paper's numbers use momentum-aware selection.
