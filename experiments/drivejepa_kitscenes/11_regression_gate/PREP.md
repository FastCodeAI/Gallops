# Getting the checkpoint ready for the regression gate queue

Checkpoint: "Encoder, generator and scorer trained together" on KITScenes, chosen at 4.5 passes
(`{runs}/drivejepa_kitscenes/generator_joint_kit/s5/ckpt/p04.5.pt`).

Done on shakaal, 2026-10-28 00:55–02:30 IST:
- **No job was submitted.** No GPU was used and nothing was trained.
- **No tracked file was edited in any repo.** Two scripts were added here as untracked files: `prep_export.py` and
  `run_prep.sh`. Their outputs are in `out/prep_{export,load,compare}.json` and `.log`.
- **New files outside this folder:**
  - one weights file on the NAS;
  - one detached Paddock worktree;
  - one untracked `paddock.toml` inside that worktree.

Every step ran on CPU with the Drive-JEPA env Paddock's node uses (`{p4}/naren/envs/drive-jepa`, torch
2.7.1+cu128). Background is in RECON.md and RECON2.md beside this file.

## Key findings

1. **Export.** The new file is
   `{nas}/siddesh/checkpoints/drivejepa_kit_enc_gen_scorer_p045/merged_state_dict.ckpt`: 1,245,120,771 bytes,
   sha256 `91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98`, one top-level key `state_dict`, 495
   tensors.
2. **Checks against the released weights: all PASS.**
   - The key set, shapes and dtypes are identical.
   - 261 tensors differ: encoder 96, generator 155, scorer pick head 10.
   - The projector and the other two scorer heads equal the released weights exactly.
   - The 4 generator head copies are identical to each other and to the trained copy in p04.5.pt.
   - No LoRA or training-only keys are left.
3. **Load the way Paddock's node does: PASS.** It runs `DriveJEPAAgent(config, lr=1e-4, checkpoint_path=…)` then
   `initialize()`, a strict load, in a bare process with none of our code and no `torch.load` shim.
4. **Behaviour, 5 held-out moments: PASS.**
   - The loaded file and the in-memory evaluation build give the same pick on 5/5 moments.
   - The max absolute difference is **0.0** on the 32 menu scores, the raw logits, the 32 candidates and the picked
     8-pose plan.
   - The picks also equal the ones stored by the original held-out evaluation, which ran on GPU.
5. **Clean checkout:** `{paddock_gate_checkout}`, detached at **9585f81**. Its untracked
   `paddock.toml` has the `[queue]` table, and `python -m tools.gate_queue ls` works from it.
6. **`submit` has no dry-run mode.** It was not run. Every check `submit` makes was reproduced by calling the
   queue's own functions read-only (Part 4).
7. **Keep-going is supported** as a submit flag, `--keep-going`. It is stored in the job and passed to
   `tools.gate smoke`.
8. **Blockers: none hard.** origin/main moved to 9671b49 during the session (it adds HUGSIM and WA-JEPA, and leaves
   the gate and Drive-JEPA's settings unchanged). Only 33.8 GB of GPU was free at 02:28, so the closed tier may
   wait for memory.

## Part 1: export

### 1a. Building the merged agent

- `prep_export.py export` calls `final5.build_ours(4.5)` unchanged
  (`10_encoder_generator_scorer/src/final5.py:119-137`). The steps are:
  1. build the released agent with the pass-3 LoRA wrapped (`g0_gpu.build(pass3=True, merged=False)`);
  2. `load_gen` (41 tensors);
  3. `load_lora` (192);
  4. copy `pred_score` (10);
  5. `f_final.merge`, which replaces each LoRA layer with `nn.Linear(W + B@A)`.
- **The one change for CPU:** the devkit's `move_all_modules_to_gpu`
  (`Drive-JEPA/navsim_v2/scripts/inference/run_kitscenes_inference.py:100-105, 115`) was made a no-op. Nothing else
  was touched.
- The released weights' sha256 is asserted inside `g0_gpu.build`
  (`05_generator_survey/src/g0_gpu.py:48`, 740335c8…). RAM was 73.9 GB available at the start, so there was no wait.
- **Time:** about 34 min wall clock. It was disk-bound: the busy hard disk read at about 15 MB/s.

### 1b. Saving

- The file holds `{"state_dict": {"agent." + k: v for k, v in agent.state_dict().items()}}`. It was written to a
  `.tmp` name and then renamed. There is no optimiser state.
- The size is 1.245 GB; the sha256 is in finding 1. The 4 generator-head copies share one storage, because they are
  one module in the model.
- The NAS folder was created. NAS permissions show `rwxrwxrwx`, which is the share's own mapping.

### 1c. Checks against the released weights (`out/prep_export.json`)

| check | result |
|---|---|
| key set equals released (495) | **PASS**: 495 = 495, none missing, none extra |
| shapes and dtypes equal | **PASS**: 0 shape and 0 dtype mismatches (all fp32) |
| tensors that differ from released | **261**, as expected |
| encoder (`_backbone.img_backbone`) | 96 of 292 differ: the 96 LoRA-merged linear layers. Each equals released W + B@A from p04.5.pt with max abs diff 0.0 (α = r = 16, scale 1). Largest change 0.054 |
| generator (`hist_encoding`, `init_feature`, `_trajectory_head.0–3`) | 155 of 171 differ: 38 trained tensors × 4 head copies + 3. The 4 fixed BEV grids per copy are unchanged. Largest change 0.042 |
| scorer pick head (`scorer.pred_score`) | 10 of 10 differ, and they equal p04.5.pt's `pred_score` exactly. Largest change 0.053 |
| projector (`_backbone.projector`) | **PASS**: 2 of 2 equal released |
| other scorer heads (`pred_col_agent`, `pred_area`) | **PASS**: 20 of 20 equal released |
| the 4 generator head copies | **PASS**: 42 tensors per copy, copies 1–3 identical to copy 0, and copy 0 equals p04.5.pt's 38 trained `_trajectory_head.0.*` tensors exactly; `hist_encoding` and `init_feature` equal p04.5.pt too |
| no LoRA or training-only keys | **PASS**: no `lora_`, `key_lora`, `guesser` or `.base.` keys; no non-tensor values |

## Part 2: load and behaviour check (CPU)

### 2a. Strict load the way Paddock's node does it: PASS

The lines followed are `src/paddock/adapters/models/drivejepa/policy.py:156-161` at 9585f81:

```python
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

config = DriveJEPAConfig()
self.agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=self.checkpoint)
self.agent.initialize()
```

`initialize()` is upstream's
(`{drivejepa_repo}/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py:55-66`):

```python
state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
self.load_state_dict({k.replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()})
```

How the process was set up (`prep_export.py load`):
- `PYTHONPATH` was empty, so no Gallops modules were on the path.
- The torch.load shim was never installed. That shim exists only in our evaluation code. Torch 2.7's default
  `weights_only=True` therefore applied, and the file loaded under it.
- The settings followed model.yaml: `cwd` the devkit's `navsim_v2`, `NAVSIM_DEVKIT_ROOT` set, `DRIVE_JEPA_MSDA=auto`.
- `NAVSIM_EXP_ROOT` was our navsim-exp, for the ViT init file. The queue uses Naren's `exp`, which holds the same
  released files: sha256 checked for the checkpoint on 22 Oct; the init file is UNVERIFIED.

Result (`out/prep_load.json`):
- `initialize()` raised nothing, so the strict load succeeded.
- 495 keys were loaded.
- The loaded `state_dict` equals the file tensor for tensor.

### 2b. Behaviour on 5 held-out moments: PASS

- **Which moments:** every 50th of the 223 held-out moments sorted by (scene, keyframe), i.e. indices 0, 50, 100,
  150 and 200:

  | scene | keyframe | split | pick (in-memory build) | pick (loaded file) | pick (stored held-out evaluation, GPU) |
  |---|---|---|---|---|---|
  | 00fc1b9b-005e-554c-7733-c4c0adeca015 | 40 | val | 17 | 17 | 17 |
  | 3a616875-9aef-1054-c506-2c105451c6e9 | 220 | val | 9 | 9 | 9 |
  | 85dbe6d6-47e0-7359-85cc-df98edf61ec3 | 130 | val | 0 | 0 | 0 |
  | ba826f49-37df-54f1-1f9e-b3df77208a5d | 310 | val | 30 | 30 | 30 |
  | e6826f0a-c5c1-c4bb-87cf-8a6dcafad029 | 130 | val | 9 | 9 | 9 |

- **Inputs:** built exactly as the held-out evaluation built them. That is the devkit KITScenes adapter's
  `build_sample(kf, command_mode="gt")` and the agent's own feature builders, as in `09_heldout_comparison/src/final.py:224-248`.
  One feature dict was fed to both agents, in eval mode, on CPU, in fp32.
- **Max absolute difference over the 5:** 32 menu scores **0.0**, raw logits **0.0**, 32 candidates **0.0**, picked
  8-pose plan **0.0**. That is well inside 1e-5.
- **Against the GPU evaluation:** the CPU menu scores differ from the scores stored in the held-out evaluation dump
  by at most 2.8e-6 (CPU vs GPU arithmetic), and every pick is equal.

## Part 3: clean checkout for submitting

### 3a. Worktree

- `{paddock_gate_checkout}` did not exist beforehand.
- It was created with `git worktree add --detach {paddock_gate_checkout} 9585f816c05f…`, run from
  `{paddock}`. HEAD is 9585f81, "gate: swap 2 AlpaSim scenes where every model leaves the
  road at once (#120)", which was origin/main at the time.
- Later, at 02:28, origin/main was **9671b49** (see Part 5). `submit` fetches and uses origin/main itself, so the
  worktree's own HEAD does not decide the job's code (`tools/gate_queue.py:286-318, 333-346`).

### 3b. One-time setup, following the how-to and `docs/queue.md`

Naren's how-to (`{naren_workspace}/renesas/paddock-runs/ops/gate/regression-gate-howto.md:9-21`) has
four steps:

| how-to step | what was done |
|---|---|
| 1. "Log out of shakaal and back in… `id` should list `paddock`." | Not needed: `id siddesh` already lists group `paddock` (gid 1042) |
| 2. "Update your paddock checkout: `git pull origin main`." | Done instead with the fresh worktree at origin/main (3a), so our main checkout, 190 behind with an uncommitted edit, was left alone |
| 3. "Add two lines to your `paddock.toml`: `[queue] root = "{paddock_site}/queue"`, `store = "{paddock_checkpoints}"`" | Done. `Paddock-gate/paddock.toml` is a copy of our site config (`{paddock}/paddock.toml`) with that table appended. It is gitignored (`.gitignore:2`; `git status --ignored` shows `!! paddock.toml`) |
| 4. "Check that it works: `python -m tools.gate_queue ls` prints the queue." | Done. It printed the 7 jobs, exit 0 |

How the pieces fit:
- **Where the site config is read from:** `docs/queue.md:25-26` says "Run these from a paddock checkout, with your
  site `paddock.toml` (it needs the `[queue]` table below)". The submitter needs only `root` and `store`; the rest
  is the worker's (`tools/gate_queue.py:105-108`). The file is found as `$PADDOCK_CONFIG` or `<repo root>/paddock.toml`
  (`src/paddock/config.py:190-192`).
- **Which code runs:** `tools/gate_queue.py:77-79` puts the worktree's own `src/` first on the path, so `paddock`
  resolves to the worktree. This was checked: `paddock.__file__` → `{paddock_gate_checkout}/src/paddock/__init__.py`.
- **Interpreter:** neither document names one for submitters ("paddock's own interpreter", `docs/gate.md:21`). We
  used `{paddock}/.venv/bin/python` (Python 3.11, has `yaml`), called by its full path.
  Nothing was linked into the worktree.
- **Side effect:** running Python created ignored `__pycache__/` folders in the worktree.

## Part 4: the submit, checked without queuing anything

### 4a. Dry-run mode: none

- `submit`'s options are `--model`, `--env`, `--baseline`, `--tier`, `--bench`, `--keep-going`, `--ref`, `--note`
  and a hidden `--repo` (`tools/gate_queue.py:880-888`). None of them plans without acting.
- `submit()` (`tools/gate_queue.py:333-397`) always does four things, so it was **not run**:
  - copies the checkpoint into the store;
  - pushes the commit to `{paddock_site}/queue/repo.git` as `refs/jobs/<id>`;
  - writes `jobs/<id>/job.json`;
  - prints the job id.
- Instead, each thing it would do was reproduced read-only with the queue's own functions:

| what submit would do | found |
|---|---|
| commit | `resolve_commit(None)` = origin/main after a fetch: at 02:28 that is **9671b49** (`tools/gate_queue.py:286-318`) |
| env check | `checked_env` against drivejepa's `model.yaml` at the commit: `DRIVEJEPA_CHECKPOINT` **accepted** (4c) |
| baselines present | `{paddock_site}/queue/baselines/drivejepa.toml` exists, so submit does not refuse (`tools/gate_queue.py:347-361`) |
| checkpoint copy | `store_value` digest computed without copying: the file has no siblings under 1 MB, so it would be copied to `{paddock_checkpoints}/da3038408e58c37c1268f5d018a56609b137a1a14dbd7c991d2ca560a0e77334/merged_state_dict.ckpt` (does not exist yet), made read-only (`tools/gate_queue.py:221-256`). That is 1.25 GB more on the NAS (2.6 TB free) |
| job record | `{"kind": "candidate", "model": "drivejepa", "env": {"DRIVEJEPA_CHECKPOINT": <store copy>}, "tier": "all", "keep_going": false, "ref": "main", "commit": <origin/main>, …}` (`tools/gate_queue.py:375-389`) |
| what the worker runs | `{paddock_site}/venv/bin/python -m tools.gate smoke --model drivejepa --baseline <job>/baseline.toml --tier all --execute --label <job> --out <job>/smoke --json <job>/verdict.json --base-port 9800 --renderer-port 8800 --python {paddock_site}/venv/bin/python --cores 12-15 --cores 16-19 --cores 20-23 --stage-dir {p3}/paddock/stage/<job> --env DRIVEJEPA_CHECKPOINT=<store copy>`, from `{paddock_site}/queue/checkouts/<commit>` (`tools/gate_queue.py:598-621`; `[queue]` in `{paddock_site}/paddock.toml:183-190`) |

Subsets it would run, and the baseline each pairs with (`tools/gate.py:1082-1085, 1594-1602`):

| tier | subset | baseline run | how found | baseline holds the whole current subset? |
|---|---|---|---|---|
| quick | kitscenes | 20261023T195836-e90d98 | own entry | 223 / 223 |
| quick | nuscenes | 20261023T200508-261359 | own entry | 1000 / 1000 |
| quick | navsim | 20261023T202240-de2f00 | own entry | 1018 / 1018 |
| quick | neuroncap_quick | 20261026T195633-2a9a76 | own entry | 3 / 3 |
| quick | bench2drive_quick | 20261026T233957-56d38c | the bench2drive entry | 1 / 1 |
| quick | alpasim_quick | 20261027T005941-d3f828 | the alpasim entry | 1 / 1 |
| closed | neuroncap | 20261026T232704-7bcf0e | own entry | 15 / 15 |
| closed | bench2drive | 20261026T233957-56d38c | own entry | 7 / 7 |
| closed | alpasim | 20261027T005941-d3f828 | own entry | 10 / 10 |

- So no pair should come out as SUBSET DIFFERS.
- Whether any pair comes out as CONFIG DIFFERS can only be known after the runs (UNVERIFIED). The harness and runner
  files that record run config were diffed from each baseline's commit to 9585f81:
  - KITScenes, nuScenes and NAVSIM against d889539 changed only nuScenes' `description` and drivejepa's `model.yaml`
    and `policy.py`. Those two are recorded as `declared_env`, which is a note, not CONFIG DIFFERS.
  - Against 0687080, the neuroNCAP harness change (#114) adds no `config[...]` key.

### 4b. Keep-going: supported, as a submit flag

```python
s.add_argument("--keep-going", action="store_true", help="tools.gate smoke --keep-going")   # tools/gate_queue.py:885
        "keep_going": bool(args.keep_going),                                                 # :385, in job.json
    if job.get("keep_going"):                                                                # :615-616, gate_command
        command.append("--keep-going")
```

In `tools.gate smoke`, `--keep-going` does two things (`tools/gate.py:1960-1983, 2154-2156`):
- it keeps launching after a REGRESSION or INCOMPLETE;
- it starts the closed tier even when a quick subset did not finish or regressed.

### 4c. Env key and baseline coverage

- **`DRIVEJEPA_CHECKPOINT` is accepted.** `checked_env` passed against drivejepa's `model.yaml` at 9585f81
  (`tools/gate_queue.py:275-283`).
  - The keys a job may set there are `DRIVEJEPA_CHECKPOINT`, `DRIVEJEPA_IMAGE_GEOMETRY`, `DRIVEJEPA_MOMENTUM`,
    `DRIVE_JEPA_MSDA`, `NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_MAP_VERSION` and
    `OPENSCENE_DATA_ROOT`.
  - `command_source` was refused, as expected.
  - 9671b49 changes only drivejepa's `benchmarks:` line (it adds `hugsim`), not `runtime.env`.
- **Every subset of `--tier all` has a baseline** (table in 4a).
  - The two crash checks have no entry of their own. They use their benchmark's entry, which is the documented rule
    (`tools/gate.py:1594-1602`; `docs/gate.md:253-254`).
  - The baseline caveats from RECON2.md still apply: the open-loop baselines predate momentum-on, and the AlpaSim
    baseline was put together from runs at two commits.

### 4d. AlpaSim video and where the run folder lands

- **Video: not recorded.** The worker's `[benchmarks.alpasim]` (`{paddock_site}/paddock.toml:170-176`) sets no
  `video`, and the harness reads `self.flag("video")` (`alpasim/harness.py:695`). The baseline run folder has no mp4.
  (`video = "encode"` at `{paddock_site}/paddock.toml:89` is neuroNCAP's.)
- **Where the job's AlpaSim run lands:** `{paddock_runs}/alpasim/drivejepa/<run_id>/` (the worker's runs
  root, `{paddock_site}/paddock.toml:11`). Inside it, as seen in the baseline run `20261027T005941-d3f828`:
  - `<clip>.replans.jsonl` per scene: every replan's sent speed, acceleration, measured sideways values and frame times;
  - `<clip>/rollouts/<clip>/<uuid>/rollout.asl`;
  - `<clip>/controller/` (the controller CSV);
  - `bridge.log`, `bridge_plans.json`, `container_logs.txt`.
- The run.json goes to `{paddock_site}/queue/checkouts/<commit>/results/alpasim/drivejepa/<run_id>/`, and the run id
  appears in `jobs/<job>/verdict.json` and `gate.log`.
- So replays can be rebuilt afterwards from `rollout.asl` and the controller CSV with
  `paddock-sid/Alpasim/tools/make_replays.py` / `make_replays_fullres.py`. These are untested on this layout, and the
  Int128 `metrics.parquet` cast is needed (memory note).

### 4e. Queue and GPU now (02:28 IST)

- **Queue:** no job pending or running. All 7 job folders have a `result.json`, and `ls` shows only finished or
  cancelled jobs. The worker process `tools.gate_queue run` (user paddock) is up.
- **GPU:** 97,887 MiB total, 63,395 used, **33,833 MiB free**.
- **Machine:** load 14.4 / 19.0 / 21.6; 63 GB RAM available; swap 32/63 GB; `/` 17 GB free; `{p3}` (stage) 34 GB
  free; NAS 2.6 TB free.
- The gate waits for 14 GB free before each open-loop subset, 22 GB for neuroNCAP or Bench2Drive, and 31 GB for
  AlpaSim. Lanes add their simulators' shares while side by side (`tools/gate.py:1146-1168`). It gives up after 12 h.

## Part 5: ready to run

**Command** (not run). Submitting is the user's call:

```bash
cd {paddock_gate_checkout} && \
{paddock}/.venv/bin/python -m tools.gate_queue submit --model drivejepa \
    --env DRIVEJEPA_CHECKPOINT={nas}/siddesh/checkpoints/drivejepa_kit_enc_gen_scorer_p045/merged_state_dict.ckpt \
    --tier all \
    --note "Encoder, generator and scorer trained together on KITScenes, 4.5 passes; merged export sha256 91e13bc3"
```

- Add `--keep-going` to run the closed tier even if a quick subset regresses.
- Without it, a REGRESSION or INCOMPLETE in the quick tier stops the job before the closed tier.
- The job will run origin/main as of submit, 9671b49 or later.

**Estimated queue time:** with an empty queue, about 3–3.5 h for `--tier all`. That is roughly 1 h quick plus 2.2 h
closed, from the queue's own Drive-JEPA baseline jobs. Waits for GPU memory come on top, and at 33.8 GB free the
AlpaSim step (31 GB) may wait.

**Things that could block or spoil it** (none blocks submitting):
1. **GPU memory.** Each launch waits for free memory and gives up after 12 h. Other users' load decides this.
2. **Moving origin/main.** It moved twice during this session; 9671b49 adds HUGSIM and WA-JEPA. The gate's files and
   drivejepa's `runtime.env` are unchanged by it, but the job records whatever main is at submit. Use
   `--ref 9585f816c05f` to pin the commit checked here.
3. **Possible CONFIG DIFFERS** on any pair if a harness records a config key its baseline lacks. This can only be
   seen after the runs (UNVERIFIED; see 4a).
4. **Baseline caveats** (RECON2.md, A2b): the open-loop baselines predate momentum-on (a note, not a refusal), and
   the AlpaSim baseline is stitched from two sessions.
5. **Port 9998** is held by another user's dashboard, inside the worker's second lane range 9900–9999. Whether a
   lane binds it is UNVERIFIED; a clash would show as a failed node start in the job's `gate.log`.
6. **The source file is world-writable on the NAS** (share mapping). This does not matter after submit: the queue
   measures its own read-only copy, named by content.
7. **The quick tier can never SHIP**, and the closed tier ships only on a GAIN from neuroNCAP's 15 pairs or
   AlpaSim's 10 scenes (`tools/gate.py:812-834`).

## Submission (2026-10-28 03:25 IST, approved by Sid)

Submitted once, from `{paddock_gate_checkout}` (detached at 9585f81) with
`{paddock}/.venv/bin/python`. No other run was started and no tracked file was edited.

### Before submitting

**What the store folder name is a hash of** (`tools/gate_queue.py:221-256`, `store_value`). It is **not** the file's
own sha256. It is the sha256 of a small JSON text that maps each copied file's name to that file's sha256:

```python
files = [src, *siblings]                      # siblings: other files in the folder under 1 MB
hashes = {name: _sha256(f) for name, f in zip(names, files, strict=True)}
digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
dest = store / digest                         # the copy keeps src.name inside it
```

- Our folder holds only `merged_state_dict.ckpt`, so there are no siblings.
- `sha256(json.dumps({"merged_state_dict.ckpt": "91e13bc3…ebdb98"}, sort_keys=True))` = `da3038408e58c37c…e0a77334`,
  recomputed by hand. That is the folder PREP.md predicted.
- So the queue copies and loads our file, under its own name. The file was re-hashed just before submitting:
  `91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98`.

**Queue and GPU at 03:24 IST:**
- No job pending or running: every job folder had a `result.json`.
- The worker `tools.gate_queue run` (pid 1067408, user paddock) was up.
- GPU: 62,310 MiB free, 34,918 MiB used.

### The submit

```
$ python -m tools.gate_queue submit --model drivejepa \
    --env DRIVEJEPA_CHECKPOINT={nas}/siddesh/checkpoints/drivejepa_kit_enc_gen_scorer_p045/merged_state_dict.ckpt \
    --tier all --keep-going --ref 9585f816c05f \
    --note "Drive-JEPA KITScenes: encoder LoRA + generator + scorer pick head trained together, 4.5 passes (Sid)"
  copying …/merged_state_dict.ckpt -> {paddock_checkpoints}/da3038408e58c37c1268f5d018a56609b137a1a14dbd7c991d2ca560a0e77334/merged_state_dict.ckpt
submitted 20261027T215457-siddesh-94f0: candidate drivejepa at 9585f816c05f, 0 job(s) ahead of it
exit 0
```

No flag was rejected.

### After submitting

**Job:** `20261027T215457-siddesh-94f0`. Folder: `{paddock_site}/queue/jobs/20261027T215457-siddesh-94f0/`.

`job.json`, quoted:

| field | value |
|---|---|
| kind / model | `candidate` / `drivejepa` |
| env | `DRIVEJEPA_CHECKPOINT = {paddock_checkpoints}/da3038408e58c37c1268f5d018a56609b137a1a14dbd7c991d2ca560a0e77334/merged_state_dict.ckpt` |
| sources | source and resolved = our NAS file; digest `da3038…77334`; files `{"merged_state_dict.ckpt": "91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98"}` |
| tier / bench | `all` / `[]` (every subset) |
| keep_going | `true` |
| ref / commit | `9585f816c05f` / `9585f816c05ffc5631c950fff47ee90b383970c8` |
| baseline file | not a job.json field. The worker copied `{paddock_site}/queue/baselines/drivejepa.toml` to `<job>/baseline.toml` when the job started (byte-identical, `cmp`) |

**The stored copy:**
- `{paddock_checkpoints}/da3038…77334/merged_state_dict.ckpt` exists: 1,245,120,771 bytes, mode `r--r--r--`,
  group paddock.
- Its sha256 is `91e13bc38aa76ee386b903a27f582b5459b1ae0d362a0cb090c01db118ebdb98`, **equal** to our export.

**Started at once:** `started.json` 2026-10-27T21:55:32Z (03:25:32 IST, worker 1067408). The gate's command is in
`attempt.json`: `tools.gate smoke --model drivejepa --baseline <job>/baseline.toml --tier all --execute … --cores
12-15 --cores 16-19 --cores 20-23 --keep-going --stage-dir {p3}/paddock/stage/<job> --env
DRIVEJEPA_CHECKPOINT=<store copy>`.

**Baselines for every subset of `--tier all`.** These are from the job's `baseline.toml`, as printed in the plan at
the top of `gate.log`:

| tier | subset | baseline run |
|---|---|---|
| quick | kitscenes | drivejepa:20261023T195836-e90d98 |
| quick | nuscenes | drivejepa:20261023T200508-261359 |
| quick | navsim | drivejepa:20261023T202240-de2f00 |
| quick | neuroncap_quick | drivejepa:20261026T195633-2a9a76 |
| quick | bench2drive_quick | drivejepa:20261026T233957-56d38c (the bench2drive entry) |
| quick | alpasim_quick | drivejepa:20261027T005941-d3f828 (the alpasim entry) |
| closed | neuroncap | drivejepa:20261026T232704-7bcf0e |
| closed | bench2drive | drivejepa:20261026T233957-56d38c |
| closed | alpasim | drivejepa:20261027T005941-d3f828 |

**Checking on it later** (from `{paddock_gate_checkout}`):

```bash
{paddock}/.venv/bin/python -m tools.gate_queue ls          # queue and outcome (done: FAIL/HOLD/SHIP)
tail -f {paddock_site}/queue/jobs/20261027T215457-siddesh-94f0/gate.log              # progress; the verdict table is at the end
ls {paddock_site}/queue/jobs/20261027T215457-siddesh-94f0/smoke/                     # one log per subset, manifest.toml
cat {paddock_site}/queue/jobs/20261027T215457-siddesh-94f0/result.json               # written when the job ends
cat {paddock_site}/queue/jobs/20261027T215457-siddesh-94f0/verdict.json              # every pair, each subset's run id
```

To cancel: `python -m tools.gate_queue cancel 20261027T215457-siddesh-94f0`. Not used.

**Expected duration:** about 3–3.5 h plus any GPU wait. With `--keep-going` the closed tier runs even if a quick
subset regresses. Not waited for, as instructed.
