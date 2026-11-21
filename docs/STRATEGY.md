# Strategy

**Read this before writing code.** It carries what paddock's existing results
already established about what is worth building, and the two constraints that
bound everything we can do today.

Everything below that cites a number is citing a measurement recorded in
paddock, not an opinion. Where something is argued rather than measured, it says
so — that distinction is the standard both repos are held to.

> **Correction (H01, 16 Oct): NAVSIM `navtrain` is no longer absent.** The data table below says it is; it is now
> held at the training site, and the 4-camera planner and the sub-score agent-look model trained on it (`experiments/H01_four_camera_base/README.md`).

> **Correction (2026-10-12): every Drive-JEPA neuroNCAP number on this page is
> camera-blind** (§1's table and §6's 3.57). Paddock placed its cameras relative to
> nuScenes' LIDAR_TOP, which is yawed −90°, and the model reads the image at points
> it treats as lidar-frame (NAVSIM's lidar frame is the ego frame), so none landed
> in the front image and it planned from ego status alone. The fix is paddock #62
> (open; paddock `main` still composes LIDAR_TOP). On scene 0103 the seeing model
> scores frontal 2.36, stationary 5.0 and side 1.67 (run `20261012T205915-7153f0`,
> on #62's branch, 3 runs each, `valid: false`), against 2.14 / 5.0 / 5.0 blind
> (runs 0–2 of scene 0103 in `20260905T220225-f11ac0`), because it now drives at a
> normal pace and the crossing car hits it. A full sweep replaces 3.57. Its
> KITScenes and NAVSIM rows are unaffected (identity lidar2ego there).

> **Correction (2026-11-12), adding to the H01 correction above: what H05 read of `navtrain`.** H05 read its logs, its camera images and
> the vendor's train metric cache: 1,192 graded logs, 126,694 moments with both front frames,
> 91,977 of them graded (`experiments/H05_ssm_memory/`). The images are incomplete there: one copy
> lacks the current frames of 298 logs and a second fills in 134 of them
> (`src/gallops/memory/ticks.py`). Whether `trainval_sensor` is whole and whether
> `synthetic_scenes/` is present were not checked. H01's trainers find it through `[datasets.navsim]`;
> `gallops.memory` reads it through `[datasets.navtrain]` (see `gallops.toml.example`).

---

## Why this repo exists

Paddock answers "which of these models is better, and on what evidence". It now
does that across five benchmarks and four vendor models. Its own roadmap names
the destination, once, in its last line:

> **Training** — the framework's eventual purpose; out of scope until evaluation
> is trustworthy.

Gallops is that purpose. The goal is not to top a public leaderboard; it is to
produce a driving policy that **runs on Renesas silicon**, is **competitive on
closed-loop benchmarks**, and is **ours to understand and ablate**.

The commercial shape matters for the technical decisions, so it is stated
plainly: train on public data first, publish credible comparisons, attract OEMs,
then fine-tune on the data they bring to close gaps they care about.
**Fine-tuning is a product surface, not an internal convenience.**

---

## What paddock's results already tell us

Six findings. They are the most valuable thing this repo inherits.

### 1. Scale is not buying safety

On the full neuroNCAP set — 20 scenario–scene pairs, 10 runs each, n = 200 per
model, every model meeting identical seeded jitter so the comparison is paired
sample-for-sample:

| Model | size | neuroNCAP ↑ | collision rate ↓ |
|---|---|---:|---:|
| Alpamayo 1.5 | 10 B params, 21 GB ckpt, 26 GB VRAM | 3.73 | 32.5% |
| OpenPilot | 60.9 MB ONNX, **CPU only** | 3.60 | 38.0% |
| Drive-JEPA | V-JEPA ViT-L + BEVFormer, 12 GB VRAM | 3.57 | 45.5% |
| VisionPilot | 28.1 M params, 114 MB, CPU | 1.58 | 85.5% |

| paired difference, 200 samples | mean | SE | t |
|---|---:|---:|---:|
| Alpamayo − OpenPilot | +0.12 | 0.15 | 0.80 |
| Alpamayo − Drive-JEPA | +0.16 | 0.14 | 1.12 |
| Alpamayo − VisionPilot | **+2.15** | 0.17 | **12.89** |

> Drive-JEPA's row is camera-blind; see the correction at the top of this page.

**The top three are statistically indistinguishable.** A 10 B vision-language
model needing 26 GB of VRAM is tied, on collision avoidance, with a 60.9 MB
model that runs on a CPU. Only VisionPilot separates, and it separates downward.

This is the strongest reason to believe an automotive-SoC target is realistic.
We are not trying to squeeze a 10 B model into a car. We are trying to reach a
bar that a 60 MB model already reaches.

### 2. ADE is a bad target, and paddock already says so

Drive-JEPA's KITScenes ADE is 2.364 — **worse than a constant-velocity
baseline** at 1.381 — and it wins NAVSIM outright at 0.843 EPDMS against
Alpamayo's 0.608. A model that loses to "keep going straight" on displacement
error is the best planner measured.

Paddock's `architecture.md` gives the reason: open-loop metrics are dominated by
ego status, so a model that continues the current motion scores respectably, and
a good ADE is weak evidence of competence.

**Use ADE as a cheap regression tripwire. Select on closed-loop or NAVSIM.**

### 3. The most useful single number in paddock

From `docs/models/alpamayo15-eval-notes.md`:

> Drive-JEPA's *path* is more accurate (lateral 0.209 vs 0.331), and its entire
> deficit is longitudinal — it does not predict speed well.

| | ADE | longitudinal | lateral |
|---|---:|---:|---:|
| Drive-JEPA | 2.364 | 2.317 | **0.209** |
| Alpamayo 1.5 (medoid) | **0.875** | **0.718** | 0.331 |

Drive-JEPA is not a bad model with a bad ADE. It is an **excellent path
predictor with a broken speed head**, and a joint waypoint regression averaged
an 11× asymmetry into one meaningless number.

> **Superseded for Drive-JEPA, by a later measurement:** its deficit is not a
> speed head. One of the 32 candidates it proposes is within 0.37 m ADE of the
> human; its scorer picks one 2.36 m away. See
> [First target](#first-target--taken-drive-jepas-selection-not-its-head).

### 4. Separate the lateral and longitudinal heads

Three independent signals, all from measurements we hold:

- **Drive-JEPA** — best lateral accuracy of anything measured, worst
  longitudinal. A joint head hid it.
- **VisionPilot** — three physically separate networks (AutoSteer predicts the
  drivable path; AutoSpeed and AutoDrive handle longitudinal). It takes the best
  Emergency-Brake score measured, 96.0 on Bench2Drive, at 28 M parameters on a
  CPU. *(Caveat: its overall Bench2Drive standing is from 25 of 220 routes — see
  §6 — and the same model is last on neuroNCAP.)*
- **Deployment** — a small longitudinal head can run fast while a heavier
  perception and path network runs slower. The split is useful on silicon
  independently of whether it is useful for accuracy.

This is the clearest architectural steer the evidence gives. It is an argument,
not a measurement: nobody has trained the two variants and compared them, and
that is a good first ablation.

### 5. Deployment cannot be a later phase

Paddock already recorded the failure that "we'll quantize at the end" produces.
From `docs/models/VISION_PILOT_GUIDE.md`, on VisionPilot's int8 export:

> The int8 files contain **no integer convolution operations at all** — instead
> they hold 163 to 456 `DequantizeLinear` nodes feeding ordinary floating-point
> convolutions. […] int8 buys you a 3.5× reduction on disk and in memory, and
> costs you about 30% in speed.

A model exported to int8 that got *slower* was never designed for the target.
Export to the target from the first checkpoint, not the last.

The corollary is a measurement gap: **paddock records no parameter count, no MAC
count and no latency for any model.** Closing that is tracked separately in
paddock's `docs/design/model-card-and-latency.md`.

### 6. Almost every ordering has collapsed under n

| measurement | small n | large n |
|---|---|---|
| VisionPilot, NAVSIM | 0.640 (4 scenarios) | **0.431** (12,146) |
| Alpamayo, NAVSIM | 0.226 (4 scenarios) | **0.608** (12,146) |
| Alpamayo, neuroNCAP | 2.46 (1 scene) | **3.73** (20 pairs) |
| Drive-JEPA, neuroNCAP (camera-blind; see the top) | 4.04 (1 scene) | **3.57** (20 pairs) |
| OpenPilot, neuroNCAP | 3.80 | **void** — a one-hot bug; all runs `valid: false` |

Paddock's roadmap draws the lesson — "Sample size is the lesson… Two conclusions
were overturned by running more scenarios, in opposite directions" — but the
implication for training has not been drawn:

**We do not yet know which benchmarks can detect the improvement we intend to
make.** neuroNCAP at n = 200 has a paired SE of ~0.15, so it cannot resolve
anything below roughly a 0.3-point effect — and the entire spread across the top
three models is 0.16. NAVSIM at n = 12,146 resolves much finer. Bench2Drive at
25 of 220 routes, with off-road and minimum-speed penalties disabled at the
pinned revision, resolves nothing we should trust — and it is the one benchmark
where VisionPilot leads, while neuroNCAP puts it last at t = 12.9. **Those two
closed-loop benchmarks violently disagree about the same model, and nobody has
explained why.**

You cannot train toward an improvement you cannot measure. A power analysis of
the suite — what effect size does each column actually resolve? — is the gate,
not a nicety.

---

## What we can actually do here today

Both of these are hard constraints, and both were measured, not assumed.

### Compute: one shared GPU, already saturated

One RTX PRO 6000 Blackwell, 97.9 GB total, with **89.5 GB already resident
across ten other processes**, SM at 100%, drawing 588 of 600 W. Roughly 7.5 GB
of VRAM free. Host RAM: 32 GiB available of 123 GiB, and **swap is 100%
consumed**. Root and one SSD are at 98% and 100% full; only the NAS and one HDD
have room.

Paddock sets `[gpu] serialize = true` for exactly this reason. **Nothing of
training scale runs on this box as it stands.** Securing compute is a
prerequisite, not a detail, and it has a long lead time.

> **Re-measured since, and materially different:** 63 GB of VRAM free, SM at
> 19%, 213 of 600 W, six other processes; 60 GiB of host RAM available. Swap is
> still fully consumed and the box is still shared, so this headroom is not
> something to design against — but "nothing of training scale runs on this box"
> is no longer true as written. The binding constraint today is data, not
> compute. See [architecture.md](architecture.md) §5.

### Data: almost none of it is training data

| dataset | what we hold | usable for training? |
|---|---|---|
| **nuScenes** | full 850 scenes, 34,149 keyframes | **Yes** — the one real corpus. Camera-only: no `sweeps/`, zero radar files, lidar 7,044 keyframes short of the keyframe count. |
| **KITScenes** | 127 of 1,006 sequences from `val` / `overlap_train_val`, and since 2026-09 at least 373 of the 533 `train` sequences | **Partly.** The re-ranker ([First target](#first-target--taken-drive-jepas-selection-not-its-head)) trained on 5,711 keyframes from those 373. Whether the rest of the 2.38 TB `train` split has arrived since has not been re-measured. |
| **NAVSIM / OpenScene** | `navtest` only (136 logs, 12,146 scenes) | **No.** `navtrain` and `trainval_sensor` are absent, as is `synthetic_scenes/`. |

Two long-lead items follow directly: a KITScenes `train` pull is 2.38 TB and
only fits on the NAS, and NAVSIM's training split is a separate, larger
download. If NAVSIM is our highest-resolution benchmark, its training split is
our most valuable corpus.

Also on disk, unevaluated and worth knowing about: `cosmos-transfer2.5` and
`Cosmos-Drive-Dreams` for synthetic generation — a live option given the data
gap, not a hypothetical; cloned-but-unscored repos `vjepa2`, `alpamayo2`,
`sunnypilot`, `auto_e2e`; and four large checkpoints nobody has put through
paddock (UniDriveVLA 8B, ReCogDrive-8B, Qwen3-VL-8B-Thinking, OpenREAD).

---

## What to build

### The shape the evidence argues for

*Turned into a buildable design — with two of the choices below revised
against measurements — in [architecture.md](architecture.md).*

A **small camera-only policy, well under 1 B parameters, with structurally
separate lateral and longitudinal heads, on a self-supervised video-pretrained
encoder, selected on closed-loop, and exported to the target from day one.**

| Choice | Because |
|---|---|
| Small (≲100 M) | §1 — a 60.9 MB CPU model ties a 10 B VLM on collision avoidance, and silicon is the destination |
| Camera-only | Every model in the matrix is camera-only, and our nuScenes copy has no sweeps and empty radar, so fusion is not even evaluable here |
| Separate lateral / longitudinal heads | §3, §4 — though §3's reading of Drive-JEPA is superseded; see [First target](#first-target--taken-drive-jepas-selection-not-its-head) |
| Self-supervised video pretraining | §3 — Drive-JEPA's best-measured lateral accuracy comes from a representation learned without labels. That is the part of it that works. |
| Selected on closed-loop | §2 |
| Exported to target continuously | §5 |

**What not to build:** a large VLM with a reasoning trace. It is the most
interesting model in the matrix and the least suited to this programme — 26 GB
of VRAM, unseeded sampling with a standard deviation of 1.817 on a single
scenario (paddock notes ~200 runs would be needed just to place it), and tied
with a CPU model on safety. It is a research direction, and it is NVIDIA's
already.

### First target — taken: Drive-JEPA's selection, not its head

Measured in paddock (2026-09-16 to 09-19):
- Drive-JEPA's KITScenes deficit is not a speed head. The 32 candidates it
  proposes include one within 0.37 m ADE of the human on average.
- The deficit is its scorer's choice among them. That scorer was trained only
  on NAVSIM's PDM score, and off NAVSIM it ranks close to random: the pick is
  2.36 m, on average the 15th of 32.
- Its inputs were checked first (acceleration, route command, ego origin, lens
  distortion). None moved the pick beyond noise.

So the first target became a **re-ranker**: a small MLP over the frozen model's
candidates, trained on KITScenes train, in `gallops/train/rerank.py`. It proves
the loop, train → checkpoint → sidecar → `drivejepa_rerank` in paddock → a row,
and closes a measured gap on the way. Paddock's model page, now copied in
[experiments/GC01_rerank_kitscenes/paddock/](../experiments/GC01_rerank_kitscenes/paddock/README.md),
has the numbers and the limits. The main limit: it is open-loop only until it
has been through neuroNCAP.

The longitudinal-head idea below is superseded by that finding. It stays as
written, because the reasoning that led to it is worth keeping.

### Candidate first targets — as first written

The point of a first target is to **prove the loop**: train → checkpoint →
sidecar → paddock adapter → a row in the table. Do that with a model whose
quality does not matter yet, exactly as paddock's own guide prescribes for
adapters ("run the constant-velocity template *before* wiring in your model…
nothing else is moving").

- **Fix Drive-JEPA's longitudinal head.** The best-evidenced option (§3): its
  path is the most accurate measured and its entire deficit is speed. Retraining
  a head rather than a backbone is small, cheap, and directly improves a model
  already in the matrix. Produces a `drivejepa_ours` variant — the family-of-
  adapters pattern paddock expects.
- **Retrain VisionPilot's nets.** `auto_speed`, `auto_steer` and `auto_drive`
  are separate Autoware repos that exist so the networks can be retrained. 28 M
  parameters, already deployable, and it has the clearest known weakness in the
  matrix: neuroNCAP side score 0.08, hitting the side obstacle at nearly full
  speed on all 50 runs. That is the one effect in the suite we *know* we can
  measure (t = 12.9).
- **A small model of our own on nuScenes.** The only corpus we actually hold.
  Slowest to a number, but everything about it is ours.
- **AutoE2E — not yet.** It is an open architecture from the Autoware Foundation
  with no published checkpoint, which initially made it look like an ideal
  plumbing test. Treat it as unvalidated: it is a parallel effort from a
  different team at the same organisation that produced VisionPilot, so its
  provenance is not independent evidence of quality — and VisionPilot is the one
  Autoware model we *have* measured, scoring worst on neuroNCAP at 1.58 with an
  85.5% collision rate. Revisit if someone evaluates it.

### Sequencing

1. **Close the measurement gate.** Power-analyse the suite. Explain the
   Bench2Drive/neuroNCAP contradiction. Land paddock's model card and latency
   column. None of this is training work; all of it is what makes training
   legible.
2. **Prove the loop** with a first target whose quality does not matter.
3. **Close one known gap** on a small model — a measured weakness, a fine-tune,
   a paired re-measurement. This is the result shape that attracts an OEM.
4. **Then our own architecture**, with the loop proven, the measurement trusted,
   and a baseline we produced ourselves to beat.

---

## Why two repos

Four reasons, in increasing order of importance:

1. **Engineering.** Paddock's dependency list is six packages and its `make
   check` runs in a clean environment with no GPU. Putting torch in there makes
   its gate heavy and breaks the invariant that paddock never imports a model.
2. **Data isolation.** The strategy is to attract OEMs who bring their own data.
   That data arrives under terms that do not permit it to sit beside a repo we
   publish results from. The separation has to exist before the first
   conversation, not after.
3. **Our own numbers stay trustworthy to us.** Paddock's claim is that an ADE
   and an NCAP score are "statements about the same model, run the same way".
   The moment our model gets a privilege the vendors' do not have — in-process
   access, a special data path, a shortcut past the HTTP node — we lose the
   ability to tell whether we are actually gaining on them.
4. **Paddock's credibility is the commercial asset.** A leaderboard maintained
   by the party that also trains the winning entry is marketing. A leaderboard
   where our model is one adapter among six, scored by machinery that predates
   it and scored four competitors first, is evidence. **Keeping the scorer
   separate from the trainee is what makes the numbers persuasive to somebody
   who does not work here.**

## How the repo is laid out

| Where | What |
|---|---|
| [experiments/](../experiments/) | one folder per experiment: `experiment.toml` (the facts), `CARD.md` (rendered from them), the account, the predictions, and the scripts as they ran |
| [src/gallops/train/](../src/gallops/train/) | the trainers: the KITScenes re-ranker, the swerve finetune, the scorer on cached features, and the preflight and safety checks they must pass |
| [src/gallops/expert/](../src/gallops/expert/) | the expert that draws evasion and braking targets offline, and the grader every plan is judged by |
| [src/gallops/data/](../src/gallops/data/) | training-data builders: the re-ranker's dataset, real nuScenes scenes as episodes |
| [src/gallops/drivejepa_b2d/](../src/gallops/drivejepa_b2d/), [recipes/](../recipes/) | the Bench2Drive training, as a recipe that prints or runs its commands |
| [checkpoint.py](../src/gallops/checkpoint.py), [config.py](../src/gallops/config.py) | the checkpoint contract, and the only code that knows where anything lives on disk |
| [experiments.py](../src/gallops/experiments.py) | the experiment registry: validates every record and renders the README |

The names: a paddock is the enclosure where a horse is paraded and judged
before a race. The gallops is the training ground where it is made fast.

## Open questions

1. **Compute.** The binding constraint on everything past step 1.
2. **The R-Car X5 envelope.** Confirm TOPS, memory bandwidth, and above all
   **whether the toolchain does true integer convolution** — which is exactly
   what VisionPilot's ONNX export did not (§5).
3. **Which benchmarks can resolve what.** See §6. Answer before selecting a
   checkpoint on any of them.
4. **First target.** Taken: the Drive-JEPA re-ranker (see above).
