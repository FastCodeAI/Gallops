# H05 — State-space memory

**Owner:** Naren · **Priority:** 5 of 11 · **Started:** 21 Oct 2026 (planned) · **Status:** parked, inconclusive (hypothesis) · step 0 run by Siddesh, 3–5 Nov 2026

The facts are in [experiment.toml](experiment.toml). The why, the hypothesis and the plan below are
transcribed from Renesas Progress and Plan, 22 Oct 2026 ("Next steps", item 5, and "Naren · hypothesis to
test"). Only the plan's step 0 has run; "What step 0 did" says what it found.

## Why

- The model's view of time is short: two frames, 0.5 s apart. On neuroNCAP the side crashes are runs
  where it never starts braking although the car was in view for 2 s or more.
- Extended comfort is the base's largest NAVSIM gap (−0.077), and the released model's 0.477 is below
  constant velocity's 0.567: consecutive plans disagree.
- Bench2Drive's stall trap (stuck after a first stop) is a lack of state about why it stopped.
- A state-space layer costs the same per tick however long it has run, so it fits a 2 Hz closed-loop
  driver; a transformer over a long window does not.

It is layer 4 of the team's "Memory: do, on-policy" call (design decision 6), whose layers are on-policy
DAgger, the previous plan as input with 30–50% dropout, carrying slots or scene tokens across ticks, and
a Mamba-style state model that needs sequence training. That call says to judge memory on closed loop
only, because open loop feeds logged history.

## The hypothesis

A state-space layer (Mamba-style: a learned recurrence with a fixed-size state, linear in sequence
length) placed between the frozen encoder and the planner carries the scene forward; the planner's
slots read that state each tick. Expected: earlier, steadier reactions in closed loop and better NAVSIM
extended comfort, with no open-loop loss.

What could make it fail, as the page lists it:

- **Extra capacity, not memory.** A gain could come from the added layer alone. Control: the same layer
  with its state reset every tick.
- **Copying the past.** With its own history it can learn to repeat what it was doing and react late.
  Open loop hides this, so judge on closed loop, and train on closed-loop rounds, not only logs.
- **State leaking between scenes.** The state must reset on `/reset`; the AlpaSim bridge never calls it
  between scenes, so fix that before AlpaSim numbers.

## The plan

| Step | What | Cost | Decides |
|---|---|---|---|
| 0 | Does history beyond 0.5 s carry signal? Cache pooled encoder features per tick for a slice of navtrain and the rendered episodes; fit a small probe on the last k ticks (k = 1, 2, 4, 8) to predict the next seconds of human speed and, on rendered episodes, time to collision. If it stops improving at k = 2, there is little for memory to add. | a feature cache (GPU, hours) plus a CPU probe | whether to build the layer at all |
| 1 | Build the layer in gallops: a diagonal state-space block in plain PyTorch (no custom CUDA kernels), over per-tick pooled scene tokens plus ego status; the plan slots attend to its output. Encoder frozen, features cached; sequences of 8–16 ticks (4–8 s at 2 Hz). | about a day of code | — |
| 2 | Three arms on the same data: (a) state carried across ticks, (b) the same layer reset every tick, (c) released Drive-JEPA. NAVSIM EPDMS and extended comfort on the gate's 1,000 scenarios. | a few GPU-hours per arm | memory versus capacity: (a) must beat (b) |
| 3 | Closed loop: the paddock adapter keeps the state between calls and clears it on reset. The gate's closed tier on (a) and (b): neuroNCAP (when braking starts, impact speed, side 0103 and 0110), Bench2Drive stall routes, AlpaSim. | the closed tier took 2 h 21 min without AlpaSim (20 Oct), per checkpoint | keep it only if (a) beats (b), nothing regresses, and reactions come no later |
| 4 | Port to the 4-camera base if it holds. | — | — |

Training lives in gallops; paddock only runs the checkpoint. Momentum-aware selection (AlpaSim 0.421 →
0.543) is the cheap no-training baseline to beat.

## What step 0 did (3–5 Nov 2026)

These runs are recorded as H05.r1 and H05.r2 in [experiment.toml](experiment.toml); the files they wrote
are off the repository and listed, with sizes and hashes, in [POINTERS.md](POINTERS.md). All numbers are
open loop, on the 176 vendor val logs that have 4 s of history, for the events slice (moments where the
human's speed changes by more than 1 m/s within 2 s), with paired, log-clustered 95% intervals.

- **The cache (r1).** The released model was run over every navtrain moment: 126,694 moments, 0
  failures. A *moment* is the pair of front frames the model reads, now and 0.5 s ago. Everything the
  model computes at a moment is stored, from the encoder map through the 32 plans and their scores.
- **The probe (r2).** A ridge model, a logistic model and a small MLP predict the human's speed and
  braking over the next 3 s. Each reads the last k moments, as a 128-number summary of the encoder map
  plus the 11 ego numbers. k = 1 is what the model sees today.
  - **One extra moment helps speed** (it adds the frame from 1 s ago and the ego state 0.5 s ago):
    −0.021 m/s RMSE [−0.028, −0.014] on 0.512. Of that, −0.013 [−0.019, −0.008] is the camera frame.
    That part may be the extra 128 numbers, not the past.
  - **Further back, only the car's own past helps.** Camera history beyond 1 s adds nothing (−0.0009
    [−0.0045, +0.0025]); ego history adds −0.007 [−0.010, −0.004].
  - **No window predicts braking better than k = 1.** The rise with longer camera windows is likely
    overfitting: the MLP shows the same rise.
- **Against the registered rule.** "Refuted at step 0 if a probe on the last k ticks stops improving
  at k = 2" was not met: ridge speed still improved from k = 2 to k = 4. The control puts that gain on
  the ego state, not the camera.

## Where it stands (5 Nov 2026)

Parked, verdict inconclusive. The probe reads ordinary driving through a lossy 128-number summary and
cannot see a crossing car, so it neither supports nor rules out a trained scene memory. The follow-on
design carries the planner's own outputs forward instead of the camera; that is a different claim, so
it is recorded as H15. Any later "more or longer camera history" work stays here as a new run.

Before step 0 (3 Nov), two things the plan relies on had changed in paddock: the gate's NAVSIM subset
is now 1,018 tokens drawn within groups (#105, merged 23 Oct), and momentum-aware selection is on by
default for Drive-JEPA (#95, merged 26 Oct), so the released arm (c) is the momentum-on model. On the page's third risk: paddock's AlpaSim bridge
on `main` says "Every scene starts with /reset" (`adapters/benchmarks/alpasim/bridge.py`, from #93,
merged 21 Oct); whether that covers the path the page meant was not checked.

## How it will be decided

Step 0 decided not to build the camera layer now. If H05 is resumed, steps 2 and 3 decide whether to
keep it, through paddock's regression gate, with (b) as the control that separates memory from capacity.
