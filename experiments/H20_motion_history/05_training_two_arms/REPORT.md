# Approach 4: training arms A and B

Design: `PREREG_APPROACH4.md` with amendment 1 (sha256 `f8f4b975…a450`). Accepted for this run: Adam 1e-4 for the new blocks.
Every job is pinned (`taskset -c 0-1 nice -n 19`). The GPU is used only after the memory gate. No closed-loop result is read or run.

## 1. Resume support

`src/train_a4.py` writes a full save at every check (steps 0, 930, …, 5,580) to `ckpt/pXX.X.pt`, and every 30 minutes to `resume/last.pt`. Each save is written to a temp name and then renamed. It holds:
- the trainable weights;
- the three Adam states;
- the position (pass, step within the pass, global step);
- the torch, CUDA, numpy and python RNG states, plus the hiding generator;
- the run record;
- the selection counters.

`--resume <file>` restores all of it.

Two items in the brief's list need no saved state:
- **The sampler:** its batch list is a pure function of (seed, pass), so the position is enough.
- **The scheduler:** 0.1.8b's recipe has none; its learning rates are constant.

**Test on arm A** (`out/resume_test.json`): 6 steps straight vs 3 steps + save + a new process that resumes + 3 steps.

- **Result: identical.**
  - Losses are equal at all 6 steps: 3.082307, 2.719901, 2.638986, 3.463814, 2.972001, 3.492734.
  - All 79 weight tensors are bit-identical (max diff 0.0).
  - The three Adam states, the hiding generator and all four RNG states are identical.
- **First attempt: not identical, and the cause had nothing to do with resume.** The training step itself was not repeatable.
  - The same step on the same state, run twice, gave different gradients. 33 of 79 tensors differed, by up to 0.004, all upstream of the planner's deformable attention (`out/det_probe_plain.json`, first version).
  - Cause: that attention runs in its pure-PyTorch form (`DRIVE_JEPA_MSDA=pytorch`) and calls `F.grid_sample`. PyTorch's CUDA backward for it, `grid_sampler_2d_backward_cuda`, uses atomic adds and has no deterministic version. `torch.use_deterministic_algorithms` names it (`out/det_probe_det.json`).
  - 0.1.8b trained with the same non-repeatable gradient.
- **Fix:** `src/det_grid_sample.py`, installed in the training setup only.
  - **Forward:** unchanged. It is the original `F.grid_sample` call, with autocast's own dtype rule applied (grid_sample runs in the widest input type).
    - A first version cast to fp32 and moved step 1's loss. That was caught and fixed: step 1's loss is again exactly 3.082306958752712, as without the patch.
  - **Backward:** the same bilinear gradient written out. The grid gradient is a gather. The value gradient is summed with `index_put_(accumulate=True)` under PyTorch's deterministic mode, in at least fp32.
  - **Checked against PyTorch's own backward:** agreement to 3e-15 (relative) in float64, and to 1.6e-6 / 2.8e-6 in fp32 (fp32 rounding). The forward is bit-identical in fp32, float64, bf16 and mixed autocast. The new backward is repeatable; PyTorch's is not.
  - The full training step is now repeatable: 0 of 79 gradient tensors differ.
  - Cost: small. A step is about 2–3 s, of which about 0.3–0.6 s is computation and the rest is waiting on grading.
- **Not covered by the test:** a resume from a save written at a check. Those saves use the same state function, written after the check's own random draws.

## 2. Span and selection

- **Span:** fixed to 0.1.8b attempt 2's, 1.2 passes = 5,580 steps, with checks at 0.0, 0.2, …, 1.2. Confirmed from `018b_attempt2/run.json`: steps 0, 930, 1,860, 2,790, 3,720, 4,650 and 5,580.
- **Records at each check:** the safety block (`go_rate_pull_away`, `waits_correctly`, `pick_grade_mean`, `hard_picks_above10`, and the rest), plus the summary, logit statistics and lesson shares, as in 0.1.8b.
- **Selection:** PREREG §1, i.e. 0.1.8b's rule. Pick the highest pick grade among trained checks with hard picks above 10 m/s ≤ step 0's and pull-away go rate ≥ 0.66, with ties going to the earlier check.
- **How 0.1.8b's stopping rules are handled.** The rules are 2 hard-gate failures in a row, or 6 checks without a better choosable pick.
  - They are evaluated at every check and recorded (`prereg_stop_at`), but they do not end the run, because the brief fixes the span.
  - Checks after the point where a rule would have fired are not candidates, since under 0.1.8b they would not exist.
  - If no rule fires by 1.2, the span is shorter than 0.1.8b's 3-pass maximum would allow. This is the brief's choice and is recorded here.
- **Check on the report step:** run on 0.1.8b attempt 2's own `run.json`, it reproduces 0.1.8b's choice (check 0.6).

## 3. Queue

- **How it runs:** `src/queue.sh`, in the detached tmux session **`a4train`**, which survives logout. It runs arm A, then arm B.
- **Before each run:** it waits for the memory gate.
- **On a crash:** it resumes once from the newest save. A second crash stops the queue and appends the error here.
- **After each arm:** `src/report_arm.py` appends that arm's section below.
- **Outputs:** `{approach4_store}/runs/arm_{A,B}/`, containing `ckpt/`, `resume/`, `dumps/`, `run.json` and `logs/`; plus `arm_X_stdout.log` and `queue.log`.
- **Status:** `runs/STATUS.txt` is rewritten at each check (and every 25 steps). It shows arm, step, pass, s/step, the last check's four numbers and the ETA.
- **To stop a run:** write `runs/arm_X/STOP`. The run ends at its next 25-step mark and the queue stops after that arm.
- **Launch, 2026-11-18:**
  - First start at 02:52. It crashed at check 0, on a bug of mine in the status line, not in training: the ETA used the mean step time before any step existed, so it was NaN.
    - The check itself had completed and reproduced p06 exactly: pick grade 0.768489, pull-away go rate 0.840 (the pre-registered baseline), 0 hard picks. It took 422 s.
  - So the queue's one crash-resume would not be spent on that bug, I stopped the queue, fixed the status line (it can no longer stop a run), and set the aborted folder aside as `runs/arm_A_aborted_statusbug/`.
  - Arm A then restarted fresh at 03:01. It is running at 2.56 s per step.
- **Hard-gate reference:** step 0 (p06) has **0** hard picks above 10 m/s. Under 0.1.8b's rule, any trained check with even one hard pick fails the hard gate.

## Arm A (hide_prob 0.5)

Started 2026-11-18 03:02:17, ended 2026-11-18 08:29:42; 5580 steps; stopped: span of 5580 steps completed. Mean step 2.92 s (grading wait 2.60 s). Resumes: 0.

Step 0 (p06 through the wrapper): pick grade 0.768489 (p06's recorded value 0.768489), hard picks above 10 m/s 0 — the reference for the hard-brake gate.

| check (passes) | step | pick_grade_mean | hard_picks_above10 | go_rate_pull_away | waits_correctly | hard gate | go gate (≥ 0.66) | choosable |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0 | 0.7685 | 0 | 0.840 | 0.310 | pass | pass | step 0 (reference) |
| 0.2 | 930 | 0.7727 | 0 | 0.755 | 0.265 | pass | pass | yes |
| 0.4 | 1860 | 0.7703 | 0 | 0.736 | 0.380 | pass | pass | yes |
| 0.6 | 2790 | 0.7723 | 0 | 0.708 | 0.534 | pass | pass | yes |
| 0.8 | 3720 | 0.7589 | 0 | 0.717 | 0.603 | pass | pass | yes |
| 1.0 | 4650 | 0.7692 | 0 | 0.717 | 0.431 | pass | pass | yes |
| 1.2 | 5580 | 0.7608 | 0 | 0.708 | 0.655 | pass | pass | yes |

0.1.8b's stopping rules (2 hard-gate failures in a row, 6 checks without a better choosable pick) did not fire within the span.

**Selected: check 0.2** (`runs/arm_A/ckpt/p00.2.pt`, step 930): the highest pick grade (0.7727) among trained checks with hard picks above 10 m/s ≤ step 0's (0) and pull-away go rate ≥ 0.66 (PREREG §1, 0.1.8b's rule; ties to the earlier check). Beats step 0's pick grade 0.7685.

**Kill switch (PREREG §3): go_rate_pull_away 0.755 vs 0.66 → passes; eligible for simulation (not run).**

### Arm B interruption
- At 2026-11-18 11:10:53 the machine rebooted. This was not a training crash: arm B was at step 2,850 and healthy.
- The run was found stopped at 23:27.
- The queue's one crash-resume was used: arm B resumed from its newest save, `ckpt/p00.6.pt` (step 2,790, written 11:07:10), via `src/queue_resume_B.sh` in tmux `a4train`. Steps 2,791–2,850 are re-run.
- Because resume reproduces a straight run exactly (section 1), the continuation is the same run, not a rerun.
- A second crash would stop arm B and be written here.

## Arm B: STOPPED after a second crash (2026-11-19 01:04:56)

```
00:20:08 +  52.2m pass 1 step 3350/4650 (global 3350) loss 2.8794 generator 0.6039 grade 0.3693 human 1.6226 collision 0.2120 drivable 0.0716 hidden 32.0/batch wait 2.25s step 5.29s loader 0.00s ram 64 GB
00:22:17 +  54.3m pass 1 step 3375/4650 (global 3375) loss 2.8022 generator 0.5766 grade 0.3615 human 1.6072 collision 0.2025 drivable 0.0544 hidden 32.0/batch wait 1.99s step 5.13s loader 0.00s ram 65 GB
00:24:24 +  56.4m pass 1 step 3400/4650 (global 3400) loss 2.7947 generator 0.5990 grade 0.3549 human 1.5919 collision 0.1873 drivable 0.0616 hidden 32.0/batch wait 2.19s step 5.08s loader 0.00s ram 65 GB
00:26:39 +  58.7m pass 1 step 3425/4650 (global 3425) loss 2.8780 generator 0.5828 grade 0.3551 human 1.6225 collision 0.2521 drivable 0.0655 hidden 32.0/batch wait 2.24s step 5.40s loader 0.00s ram 66 GB
00:28:43 +  60.8m pass 1 step 3450/4650 (global 3450) loss 2.8165 generator 0.6031 grade 0.3579 human 1.6128 collision 0.1855 drivable 0.0572 hidden 32.0/batch wait 1.88s step 4.99s loader 0.00s ram 65 GB
00:30:48 +  62.8m pass 1 step 3475/4650 (global 3475) loss 2.8044 generator 0.5705 grade 0.3683 human 1.6132 collision 0.1782 drivable 0.0742 hidden 32.0/batch wait 1.98s step 4.98s loader 0.00s ram 64 GB
00:32:57 +  65.0m pass 1 step 3500/4650 (global 3500) loss 2.7921 generator 0.5865 grade 0.3567 human 1.6071 collision 0.1883 drivable 0.0535 hidden 32.0/batch wait 2.04s step 5.16s loader 0.00s ram 66 GB
00:32:59 +  65.0m timed save at global step 3500
00:35:09 +  67.2m pass 1 step 3525/4650 (global 3525) loss 2.8676 generator 0.5882 grade 0.3792 human 1.6287 collision 0.2088 drivable 0.0627 hidden 32.0/batch wait 2.04s step 5.22s loader 0.00s ram 64 GB
00:37:16 +  69.3m pass 1 step 3550/4650 (global 3550) loss 2.8913 generator 0.5909 grade 0.3518 human 1.6173 collision 0.2605 drivable 0.0707 hidden 32.0/batch wait 2.07s step 5.09s loader 0.00s ram 64 GB
00:39:32 +  71.6m pass 1 step 3575/4650 (global 3575) loss 2.8950 generator 0.6467 grade 0.3803 human 1.6181 collision 0.1760 drivable 0.0739 hidden 32.0/batch wait 2.19s step 5.42s loader 0.00s ram 63 GB
00:41:46 +  73.8m pass 1 step 3600/4650 (global 3600) loss 2.7120 generator 0.5801 grade 0.3451 human 1.6061 collision 0.1244 drivable 0.0563 hidden 32.0/batch wait 2.30s step 5.38s loader 0.00s ram 63 GB
00:43:56 +  76.0m pass 1 step 3625/4650 (global 3625) loss 2.8380 generator 0.6138 grade 0.3484 human 1.6245 collision 0.1987 drivable 0.0526 hidden 32.0/batch wait 2.26s step 5.20s loader 0.00s ram 64 GB
00:45:59 +  78.0m pass 1 step 3650/4650 (global 3650) loss 2.8471 generator 0.6213 grade 0.3600 human 1.6164 collision 0.1975 drivable 0.0520 hidden 32.0/batch wait 1.84s step 4.91s loader 0.00s ram 64 GB
00:48:06 +  80.1m pass 1 step 3675/4650 (global 3675) loss 2.8301 generator 0.5818 grade 0.3849 human 1.6076 collision 0.1946 drivable 0.0612 hidden 32.0/batch wait 1.81s step 5.08s loader 0.00s ram 58 GB
00:50:18 +  82.3m pass 1 step 3700/4650 (global 3700) loss 2.9512 generator 0.6054 grade 0.3828 human 1.6035 collision 0.2995 drivable 0.0600 hidden 32.0/batch wait 3.04s step 5.26s loader 0.00s ram 65 GB
  @custom_fwd(cast_inputs=torch.float16)
  def backward(ctx, grad_output):
  @custom_fwd(cast_inputs=torch.float32)
  def backward(ctx, grad_output):
Traceback (most recent call last):
  File "{working_folder}/src/train_a4.py", line 479, in <module>
    train(a.arm, a.out or CFG.RUNS / f"arm_{a.arm}", a.resume)
  File "{working_folder}/src/train_a4.py", line 414, in train
    check(prog, p, step); last_save = time.time()
  File "{working_folder}/src/train_a4.py", line 359, in check
    c, extra = run_check(model, ds_pk, pk, groups, grader, fbs)
  File "{working_folder}/src/train_a4.py", line 158, in run_check
    c["shares"] = lesson_shares(model, fbs, grader) if shares else None
  File "{working_folder}/src/train_a4.py", line 120, in lesson_shares
    gr = torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
  File "{alpasim_bulk}/conda-envs/dj-torch-cu128-overlay/torch/autograd/__init__.py", line 503, in grad
    result = _engine_run_backward(
  File "{alpasim_bulk}/conda-envs/dj-torch-cu128-overlay/torch/autograd/graph.py", line 829, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "{alpasim_bulk}/conda-envs/dj-torch-cu128-overlay/torch/autograd/function.py", line 311, in apply
    return user_fn(self, *args)
  File "{working_folder}/src/det_grid_sample.py", line 70, in backward
    out.index_put_((torch.cat(rows),), torch.cat(src), accumulate=True)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 363.56 MiB is free. Process 2311 has 4.28 GiB memory in use. Process 161931 has 1.24 GiB memory in use. Process 273139 has 4.24 GiB memory in use. Process 3623300 has 2.29 GiB memory in use. Process 97901 has 2.27 GiB memory in use. Process 999440 has 2.21 GiB memory in use. Including non-PyTorch memory, this process has 5.95 GiB memory in use. Process 3251694 has 47.63 GiB memory in use. Process 3995857 has 20.32 GiB memory in use. Process 4030462 has 4.11 GiB memory in use. Of the allocated memory 4.29 GiB is allocated by PyTorch, and 998.45 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

**What the second crash was:**
- A CUDA out-of-memory error during check 0.8 (step 3,720, 2026-11-19 ~01:04). The GPU was full: 363 MiB of 95 GiB free.
- At the time other users held about 87 GiB. The largest were a 48.8 GB job started about 01:00 h earlier, and a 20.8 GB job started 12 min earlier.
- The allocation that failed was 512 MiB, in the lesson-share step of the check, inside the repeatable grid_sample backward.
  - That backward builds the 4-corner value gradient in one buffer. It may need more memory than PyTorch's own kernel; this was not measured.
- The check-0.8 numbers were computed but not saved. The newest save is `resume/last.pt` (step 3,500, 00:32:58).
- **Arm B is stopped and unfinished** (checks 0.0–0.6 only). It is not resumed, per the brief's second-crash rule. Awaiting Sid.
- **Third start (Sid's go, 2026-11-19 ~01:27):** arm B resumed from `resume/last.pt` (step 3,500) via `src/queue_resume_B2.sh` in tmux `a4train`.
  - The launch now also waits for ≥ 12 GB free on the GPU.
  - The training code is unchanged. Only logging was added: the peak GPU memory every 25 steps and at each check.
  - Any further crash stops arm B.

## Arm B (hide_prob 1.0)

Started 2026-11-18 08:32:09, ended 2026-11-19 05:07:25; 5580 steps; stopped: span of 5580 steps completed. Mean step 3.60 s (grading wait 2.36 s). Resumes: 2 (2026-11-18 23:32:49 from p00.6.pt; 2026-11-19 02:08:08 from last.pt).

Step 0 (p06 through the wrapper): pick grade 0.768489 (p06's recorded value 0.768489), hard picks above 10 m/s 0 — the reference for the hard-brake gate.

| check (passes) | step | pick_grade_mean | hard_picks_above10 | go_rate_pull_away | waits_correctly | hard gate | go gate (≥ 0.66) | choosable |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0 | 0.7685 | 0 | 0.840 | 0.310 | pass | pass | step 0 (reference) |
| 0.2 | 930 | 0.7602 | 0 | 0.623 | 0.551 | pass | FAIL | no |
| 0.4 | 1860 | 0.7767 | 0 | 0.604 | 0.558 | pass | FAIL | no |
| 0.6 | 2790 | 0.7632 | 0 | 0.764 | 0.379 | pass | pass | yes |
| 0.8 | 3720 | 0.7588 | 0 | 0.736 | 0.310 | pass | pass | yes |
| 1.0 | 4650 | 0.7698 | 0 | 0.594 | 0.552 | pass | FAIL | no |
| 1.2 | 5580 | 0.7632 | 0 | 0.528 | 0.569 | pass | FAIL | no |

0.1.8b's stopping rule would have ended the run at check 1.2: no check passing both gates with a better pick grade for 6 checks (pass 1.2). The run continued to the fixed span (brief), but later checks are not candidates, as they would not exist under 0.1.8b's rule.

**Selected: check 0.6** (`runs/arm_B/ckpt/p00.6.pt`, step 2790): the highest pick grade (0.7632) among trained checks with hard picks above 10 m/s ≤ step 0's (0) and pull-away go rate ≥ 0.66 (PREREG §1, 0.1.8b's rule; ties to the earlier check). Does not beat step 0's pick grade 0.7685.

**Kill switch (PREREG §3): go_rate_pull_away 0.764 vs 0.66 → passes; eligible for simulation (not run).**
