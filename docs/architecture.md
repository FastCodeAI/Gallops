# Architecture

**gallops-v0.** What the model is, what it consumes, what it emits, and which
of those choices are measured rather than argued.

[STRATEGY.md](STRATEGY.md) argues a *shape* — small, camera-only, separate
lateral and longitudinal heads, self-supervised video pretraining, exported to
the target from day one. This document turns that into something buildable, and
it disagrees with STRATEGY in two places. Both disagreements are recorded below
with the measurement that caused them.

Nothing here is implemented. `src/gallops/models/` is still empty.

---

## 1. The contract, which is not negotiable

Everything in this section belongs to `paddock.policy.protocol` and is cited,
not restated — see [conventions.md](conventions.md) for why that rule exists.

A model subclasses `Policy` and implements `infer(obs) -> Trajectory`. Four
properties of that contract drive the architecture more than any preference
does.

### The output is eight waypoints, not six

`conventions.md` says neuroNCAP wants "exactly six", and it does. Six is the
**closed-loop slice**, not the model's output width. NAVSIM's converter builds
`NAVSIM_INTERVAL_S * arange(1, NAVSIM_NUM_POSES + 1)` with `NAVSIM_NUM_POSES =
8` and raises rather than extrapolating:

> A model whose horizon is shorter than four seconds raises instead of being
> [scored] as though it had; refusing names the gap. VisionPilot plans 3 s
> today and will land here until its planner horizon is extended.

**gallops-v0 emits 8 waypoints at `dt = 0.5 s`, a 4.0 s horizon.** Closed loop
takes the first six and drops the rest; NAVSIM needs all eight. A model that
plans 3 s is locked out of the highest-resolution benchmark in the suite.
*Strength: proof — read from the converter.*

### A plan near the origin is mangled, not merely stopped

`conventions.md` records that an all-zero trajectory is a commanded emergency
stop. The neighbourhood of zero is worse than the doc implies. In the
neuroNCAP vehicle model (`neuro-ncap/neuro_ncap/components/vehicle_model.py`):

- `STANDING_STILL_RADIUS = 1` metre, `MAX_ACCELERATION = 6.0` m/s².
- A plan whose **last** waypoint is inside 1 m bypasses the LQR entirely and
  goes to a stopping controller.
- Individual waypoints inside 1 m are `continue`d past — **dropped from the
  tracking reference together with their timestamps.**

So a correct creep-to-stop plan loses its early waypoints and is re-read as a
slow ramp to its surviving tail; move that tail a few centimetres nearer the
origin and the same plan becomes a full emergency stop. There is no continuity
across the boundary. Any output parameterisation whose magnitude passes through
zero sits on that discontinuity — which is why this document does **not** adopt
the factored path × speed head (§3).

The tracker's gains matter for the same reason: `q_lateral = (20.0, 0, 0)`
against `r_lateral = (0.05,)` weights lateral error **400×** the steering-rate
penalty, so it chases a changing reference rather than damping it. A plan that
changes discontinuously between frames is tracked, not filtered.
*Strength: proof — read from the vehicle model.*

### One forward pass per open-loop sequence, if the model is stateless

`Policy.observe()` defaults to running a full `infer` and discarding the plan.
A model that is a pure function of its recorded history overrides it to just
record. On nuScenes that is 4× cheaper; on KITScenes, where a sample is 49
frames, it is 49× cheaper, and bit-identical. `drivejepa`'s adapter already
does this and is the pattern to copy.

This is the strongest argument against recurrence in the model, ahead of any
quantisation concern: **gallops-v0 carries no hidden state.** `self_prime` and
`infer_sequence` exist in the protocol precisely because stateful models
diverge between open and closed loop.

### Calibration is episode state

Rigs are recalibrated between sessions; `conventions.md` records an adapter
that cached the first scene's calibration and mis-warped a quarter of its
samples, visible only as a drifting mean. Whatever warp the model fits must be
cleared in `reset()`, and `diagnostics()` must report the achieved
field-of-view margin. A negative margin means sampling outside what the camera
saw, and it is invisible in the score.

---

## 2. gallops-v0

### Input

| | |
|---|---|
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT` |
| Frames | K frames selected **by timestamp**, never by index |
| Ego | `[speed, yaw_rate]` — two scalars |
| Intent | `command` embedding (4-way) |

**Three front cameras.** Measured from the nuScenes calibration tables: hFOV
64.9° / 64.6° / 64.9°, spaced 54.7° and 57.5° apart, giving **~177° of
coverage with 7–10° of seam overlap** — no gaps. Camera height 1.50 m. A
front-only model cannot see an object 5 m ahead beyond ±3.2 m laterally, and
*side* is 5 of the 20 official neuroNCAP pairs, a quarter of the gate.

Do not read the leaderboard as "more cameras, better side score": OpenPilot
scores 3.08 on side with one camera and VisionPilot scores 0.08 with one
camera. What separates them is the longitudinal policy, not the field of view.
The tri-camera choice is about not being *structurally* blind to a quarter of
the gate. *Strength: geometry is proof; the benefit is argued.*

**Selection by timestamp is not a detail.** Benchmarks run at different rates —
nuScenes and NAVSIM at 2 Hz, KITScenes at 10 Hz, and neuroNCAP calls the model
at **10 Hz**. Index-based history silently means different things in each. With
nearest-timestamp selection at 0.5 s spacing inside a 10 Hz loop, two of three
input slots hold the same frame for five consecutive calls and then both step
at once — a 2 Hz square wave injected into a 10 Hz input. Feature-space
interpolation between bracketing cached frames is the cheap fix; at minimum,
log the achieved history timestamps in `diagnostics()`.

**Why `accel` is excluded, though it is the most informative ego channel.** Not
a preference — a definition gap between train and gate. In closed loop
(`neuro-ncap/neuro_ncap/components/nuscenes_api.py`) the CAN bus is emulated:

```python
speed = np.sqrt((delta_pos.copy() ** 2).sum()) / delta_t   # unsigned norm
can_bus[13] = speed
acc = (speed - prev_speed) / delta_t
can_bus[7] = acc
can_bus[9] = 9.8                                            # gravity, hardcoded
```

`speed` there is an **unsigned position-delta norm including lateral motion**,
not the signed forward velocity nuScenes logs; `accel` is its backward
difference. In training it would be a 50 Hz IMU reading with gravity and pitch
coupling. These are different quantities wearing the same index. **Do not
"improve" the model by adding `accel`.** *Strength: proof — read from the
emulator.*

### Backbone

≤ 12 M parameters, shared across cameras, ImageNet-initialised, distilled
(below). Plain ReLU/ReLU6 convolutions.

**Banned ops, and why the ban is in the architecture rather than the export
script:** squeeze-and-excitation, SiLU/GELU, LayerNorm, attention softmax,
dynamic shapes. These are the operations that push a graph off the integer
path on embedded NPUs. STRATEGY §5 records that VisionPilot's int8 export
contains *no integer convolutions at all* — only `DequantizeLinear` nodes
feeding float convolutions — and ran 30% slower than fp32. VisionPilot is
SiLU throughout. A model is made quantisable when it is designed, not when it
is exported.

**On the ≤ 12 M budget.** STRATEGY says "≲100 M". That is a ceiling from the
deployment target; it is not a budget the data supports. Principal-component
analysis of the 3 s nuScenes target gives **2 components = 99.73%**, and the
intra-scene autocorrelation time is ~4.3 keyframes, so ~28 k anchors are
**~5–6 k effectively independent samples** against a 2-dimensional target.
100 M parameters against that is 20,000 : 1. *Strength: the PCA and
autocorrelation are measured but not yet reproduced inside paddock; the
parameter budget drawn from them is argued.*

### Head — plain trajectory regression

**8 × (x, y), directly, single mode.**

Predicted as a **residual from a fixed, non-learned constant-curvature rollout
of `[v, ω]`**. The ego-extrapolation pathway is then a constant the network
cannot grow into, rather than a learned function competing with the visual
pathway. This is the structural version of a guard that ego-state dropout only
approximates. It also rescales the learning problem usefully: the residual has
a lateral spread of order 0.4 m rather than the raw target's ~20 m
longitudinal spread.

Three things this head deliberately is not:

- **Not factored into path(s) × speed(t).** The factored head is the cleanest
  expression of STRATEGY §3–§4 and it does not survive §1: ~19% of the corpus
  has near-zero arc length over the horizon, where a composed `path(s(t))` has
  no gradient with respect to the path, and the composition puts plan magnitude
  directly on the 1 m discontinuity. Kept as ablation **A1**.
- **Not multi-modal, in v0.** Paddock has already measured the cost of raw
  argmax over proposals: Drive-JEPA's native agent re-ranks its 32 proposals,
  paddock's proxy takes the argmax, and **the entire native−proxy gap is
  `two_frame_extended_comfort`, 0.77114 → 0.42289 (−0.34826)**, with every
  other sub-metric moving < 0.009 and three moving up. *Caveat: those are the
  242-scenario sensor-blob subset figures, not n = 12,146.* A multi-hypothesis
  head is worth having, but it needs hysteresis over fixed named modes and a
  two-frame consistency loss, not a bare argmax. Kept as ablation **A2**.
- **Not silent about lateral versus longitudinal.** The split moves to the loss
  and the report, where `paddock.metrics.trajectory.longitudinal_lateral`
  already computes it. STRATEGY §4 calls the structural split "an argument, not
  a measurement" and "a good first ablation". This document takes it at its
  word.

**Warm-up returns `constant_velocity(obs.speed)`, never zeros**, and the head
clamps away from the standing-still radius rather than drifting across it.

### Distillation

A frozen **V-JEPA 2** teacher, feature-matched into an **L2-normalised** space.

STRATEGY §3 credits self-supervised video pretraining with Drive-JEPA's best-
measured lateral accuracy (0.209) and identifies it as "the part of it that
works". Distillation is how that representation reaches a model small enough to
ship: the teacher is offline-only, the student is what goes to silicon.

Two cautions, both load-bearing:

- **Normalise the target space.** ViT features are heavy-tailed, and a conv
  student inherits the tail. That is exactly what int8 activation quantisation
  cannot absorb — undoing §2's op discipline through the back door.
- **V-JEPA 2's value is video.** Distilling it from 2 Hz keyframes discards
  most of what makes it worth using. This is the reason camera sweeps head the
  precondition list in §5, not a footnote to it.

### Loss

**Whiten before you Huber.** Raw L1 on (x, y) weights the 4 s longitudinal
error (σ ≈ 20 m) roughly 50× the lateral (mean |lateral| at 3 s ≈ 0.36 m).
That reintroduces Drive-JEPA's 11× asymmetry *in the loss* after the head took
care not to build it in — the same failure, one layer down. Since two principal
components carry 99.73%, whitening is nearly free.

| term | why |
|---|---|
| whitened Huber on the 8 waypoints | the only term whose units are the score's units — primary, not auxiliary |
| auxiliary 3-D box head | nuScenes ships boxes at every keyframe: 10⁵–10⁶ supervised targets per epoch against 12 trajectory ones. This is what justifies a 12 M backbone at all, and it is the prerequisite for any collision term. |
| two-frame consistency | the only place A2's instability can be trained away rather than patched at inference |

**A collision penalty against logged nuScenes boxes is not in v0, and that is
deliberate.**
The expert never collides, so the penalty is a constraint already satisfied and
its gradient is near zero everywhere. What works instead is time-to-collision to
the nearest in-path object — densely defined at every keyframe, and what the
score actually depends on. See §5.3.

---

## 3. Ablation arms

Named, so they are commitments rather than intentions. Each is a separate
`train_run`, and a family of adapters is what paddock expects — see
`conventions.md`.

| | arm | what it settles |
|---|---|---|
| **A1** | factored path × speed vs plain head | STRATEGY §4's "good first ablation", which nobody has run. §1 says why it is not the default. |
| **A2** | multi-hypothesis + margin hysteresis over fixed named modes vs single mode | whether mode identity survives at 10 Hz; validate on `two_frame_extended_comfort`, because neuroNCAP scores nothing about smoothness |
| **A3** | mono vs tri-camera | whether the side scenarios move; 25% of the gate |
| **A4** | with vs without V-JEPA 2 distillation | whether §2's central bet pays |

Budget note: each scored variant is a separate ~10.6 h neuroNCAP sweep. Make
the mono variant a debug artefact, not a scored row, until A3 is actually the
question.

---

## 4. Selection

**All 14 neuroNCAP scenes are nuScenes `val` scenes.** Checked 14 of 14 against
paddock's 150-scene val list, which is in turn bit-identical to the devkit's own
`nuscenes.utils.splits.val` — symmetric difference empty, 700/150. Training on
`train` therefore leaves the closed-loop gate honest — but **selecting
checkpoints on `val` silently contaminates the gate**, which is the one number
the whole programme is judged on. *Strength: proof, against the official split.*

| stage | split | metric |
|---|---|---|
| fit | ~600 `train` scenes | training loss |
| select | ~100 held-out `train` scenes | NAVSIM navtest half-split |
| report open-loop | `val` | nuScenes L2 — **tripwire only** |
| gate | neuroNCAP, 2–3 checkpoints | NCAP score |

**The selection metric is NAVSIM, not nuScenes L2.** An ego-only cubic ridge
regression on `[speed, accel, yaw_rate]` — about 30 coefficients, and **no
camera at all** — scores **0.377** cumulative L2 on nuScenes val, against
constant velocity 0.641 and every model in paddock's matrix (Alpamayo 0.787,
Drive-JEPA 0.912, OpenPilot 2.477). The same analysis reproduced paddock's
published constant-velocity figure to three decimals (0.642 against 0.641),
which is what makes it comparable at all.

If that holds, open-loop L2 is not "dominated by ego status" as paddock's
`architecture.md` puts it — it is *closed* by it, and selecting on it selects
for ego-state extrapolation. OpenPilot is the existence proof in the other
direction: worst open-loop in the matrix by 4×, tied-best closed-loop.

**So: report nuScenes L2 against 0.377, not against 0.641.** A model above
0.377 has learned nothing from pixels. *Strength: measured in this analysis,
not yet reproduced inside paddock — see §5.2. Do not cite it as fact until it
is.*

Record in the sidecar: `trained_on = (TrainedOn("nuscenes", "train", ...),)`,
`selection_split` naming the held-out train slice, and a hash of the sorted
sample-id list in `notes` so it can be compared against paddock's
`sequences_sha256`. See [checkpoints.md](checkpoints.md).

Take the canonical split from `nuscenes.utils.splits` rather than restating it,
the same rule as [conventions.md](conventions.md)'s "contracts are imported".
But **record the resolved scene list, not the fact that the devkit produced
it**: the devkit *version* is what defines that list, and a silent bump is
exactly the drift that makes two sites disagree about what they measured.

**Dependency:** paddock's nuScenes harness lives on an unmerged branch. The
open-loop reporting leg depends on it landing.

---

## 5. Preconditions, in order of value

### 5.1 nuScenes camera sweeps

The local copy has **no `sweeps/` directory at all** — verified, and
[data.md](data.md) records it. Keyframes are 2 Hz; sweeps are 12 Hz.

At the measured focal length (fx ≈ 1259 for `CAM_FRONT`), a car crossing at
8 m/s at 20 m range moves **~250 px between frames 0.5 s apart** — 16% of the
image width. That correspondence is outside every early-layer receptive field,
so motion is only recoverable at stride-32 features where it quantises
coarsely. At 0.1 s spacing the same target moves ~50 px. **The 0.5 s spacing is
chosen by our data, not by the task.**

Sweeps are the precondition for three separate things: velocity estimation of
other agents (§2), V-JEPA distillation being *video* distillation (§2), and
history that is continuous at neuroNCAP's 10 Hz (§1). Estimated at ~75 GB for
the three front cameras, against ample free space on the bulk volume — far
cheaper than either long-lead pull [data.md](data.md) currently names.
**Neither repo's data doc mentions it.**

### 5.2 Register the ego-only baseline in paddock

Reproduce the 0.377 figure there, as a scored row, before any gallops
checkpoint is reported against it. It costs 30 floats and it is falsifiable.
Until it exists, §4's selection argument rests on an analysis that lives
nowhere.

### 5.3 Counterfactual hazard supervision

The corpus contains **12.7 s** of braking harder than −3 m/s² across 979 CAN
scenes (19 scenes, 1.9%), and **1.3 s** of lateral acceleration above 3 m/s².
neuroNCAP stages every scenario so that constant velocity collides at full
speed, and avoidance needs sustained hard braking or a firm swerve.

**The gate measures a behaviour the corpus barely contains.** No architecture
recovers that. The construction that can: translate an annotated 3-D box onto
the ego's own recorded future path at a chosen time-to-collision and supervise
the avoidance. The geometry is exact — we hold the boxes and the ego future —
and only the pixels are wrong, which is what `Cosmos-Drive-Dreams`, already on
this machine and unevaluated ([data.md](data.md)), exists to fix.

This is the only item on this list that can move the number the programme is
judged on. *Strength: the corpus statistics are measured but not reproduced in
paddock; the construction is argued.*

---

## 6. Export

STRATEGY §5 is right that export cannot be a later phase, and the check it
implies — count integer convolutions in the ONNX graph — is the right
diagnosis of VisionPilot applied as a gate. It is also aimed at the wrong
toolchain. Counting `QLinearConv` / `ConvInteger` against `DequantizeLinear` →
`Conv` tests **ONNX Runtime's** quantiser. The R-Car X5 does not run ONNX
Runtime, and Renesas' flow normally ingests an **fp32** ONNX and quantises
internally against a calibration set. As written, that gate can pass an
unshippable model and fail a shippable one.

Three checks, cheapest first:

1. **Op-set conformance** — toolchain-independent, and it catches the real
   killers. Fail on any op outside a declared allowlist: Conv, ReLU/ReLU6, Add,
   Concat, MaxPool, AveragePool, nearest-neighbour Resize, Gemm, static
   Reshape/Transpose. This is §2's ban, enforced.
2. **Quantisation simulability** — in-framework per-channel symmetric int8
   post-training quantisation against a declared metric budget; dump per-layer
   activation ranges and flag any layer whose range ratio exceeds ~100, the
   standard predictor of a fp32 fallback.
3. **Then** the ONNX Runtime integer-convolution count, labelled in its own
   docstring as a proxy for a toolchain nobody here has measured.

Fill `params_m`, `macs_g` and `precision` in the sidecar from the first
checkpoint. STRATEGY §5 records that paddock holds no parameter count, no MAC
count and no latency for any model; that column is empty because nobody has
written to it.

---

## 7. Open questions

1. **Does the ego-only baseline reproduce?** §4 and §5.2. Everything about the
   selection protocol turns on it. Closed by landing it in paddock.
2. **The R-Car X5 envelope.** STRATEGY open question #2, unchanged: TOPS,
   memory bandwidth, and whether the toolchain does true integer convolution.
   §6's allowlist is a guess until somebody has the SDK.
3. **Which benchmarks resolve what.** STRATEGY §6, unchanged and still the
   gate. The rank correlations between open-loop and closed-loop across the
   five rows paddock holds are suggestive of an *inverse* relationship, but at
   n = 4–5 they establish nothing. A power analysis still has to happen.
4. **Does distillation from 2 Hz keyframes retain anything?** If sweeps do not
   land, A4 is testing a weakened version of its own hypothesis, and a null
   result would not be informative.
5. **How many frames K?** Left open on purpose; it is a function of §5.1. At
   2 Hz the answer is "as few as possible"; at 12 Hz it is a real choice.
