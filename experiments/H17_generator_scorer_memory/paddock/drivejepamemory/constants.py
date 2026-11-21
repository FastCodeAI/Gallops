# SPDX-License-Identifier: Apache-2.0
"""Every constant the Drive-JEPA interface pins, in one place.

Each entry cites the section of ``INTERFACE_DRIVEJEPA.md`` (and through it the
upstream file:line) that fixes the value. Kept in its own module so both the
planner and the fake read the *same* numbers -- a fake that quietly disagreed
with the adapter would validate nothing.

Nothing here may be "tidied" without changing the doc first: these numbers
describe a trained checkpoint, not a preference.
"""

from __future__ import annotations

# --- Adapter provenance -----------------------------------------------------
# NOT a checkpoint number. This is the one place the adapter's own version is
# written, and everything else derives from it: DriveJEPAPlanner.version_id,
# the launcher's startup provenance line, and the gate that refuses to serve a
# shadowed package. A draft that symlinks this module therefore cannot disagree
# with the code it is serving -- which is the whole point, after 0.1.2 shipped
# a real planner.py whose version_id was the only thing saying which file the
# runtime had actually loaded.

#: Version of the adapter in Alpasim/adapters/drivejepa018bp06/ (0.1.8 = 0.1.7 logic, see below). Bump when planner
#: behaviour changes; drafts that symlink this file follow automatically.
#:
#: 0.1.4 lives in a SIBLING package rather than evolving adapters/drivejepa/
#: in place. 0.1.3 is the provenance for a published row, and its launcher
#: gate reads this very constant (serve_drivejepa_0_1_3.py:83-90): bumping it
#: in the shared module would make 0.1.3 refuse to start, i.e. un-runnable.
#: Everything 0.1.4 does not change is a symlink back to adapters/drivejepa/,
#: so there is still exactly ONE copy of bridge.py, real_model.py, fake_model.py
#: and src/ in existence -- the 0.1.3 invariant, extended across versions.
ADAPTER_VERSION = "0.1.8b-p06-memory5"

# --- 0.1.6: the retrained checkpoint (analysis_3.0/Retrain-data/s4_train) -----
#: 0.1.6 = 0.1.4b + the S4 weights + the two input changes those weights were
#: trained with. The three belong together: the S4 weights never saw a bonnet
#: row or a non-zero acceleration slot, and the shipped weights never saw a
#: masked or zeroed one. Do not mix.
#:
#: Full state dict in upstream's layout (keys "agent._pad_model.*"), encoder
#: tensors copied bit-for-bit from the shipped checkpoint, generator + scorer
#: from the S4 candidate pass. Verified at load by size and sha256.
#:
#: 0.1.7 = 0.1.6 + the S10 generator (s10_retrain final_choice.json, pass "+2")
#: + the batch fix on. The merged file keeps the pass-9 scorer bit for bit
#: (s10_retrain/gen_ckpt.json: 30 of 30 scorer tensors identical), so this
#: version isolates the generator.
#:
#: 0.1.8 = 0.1.7's logic, byte for byte, with three things changed: this
#: version number, the comfort re-score default pinned here (MOMENTUM_DEFAULT
#: below) and printed in the provenance line, and the checkpoint pin, which is
#: set after training. At build time the 0.1.8 weights did not exist, so the
#: three lines below were a PLACEHOLDER. checkpoint_pin() refuses that; the launcher,
#: the worker and the run script all go through it, so nothing can serve or
#: drive 0.1.8 until real values are written here.
#: For the offline build check only, env DRIVEJEPA_CHECKPOINT_OVERRIDE =
#: "<path>|<sha256>|<bytes>" supplies a pin; the file is still verified by size
#: and sha256, the provenance line says OVERRIDE, and run_018_scene.sh refuses
#: to start a closed-loop run while the override is set.
CHECKPOINT_PLACEHOLDER = "PLACEHOLDER-0.1.8b-CHECKPOINT-NOT-PINNED"
#: 0.1.8b-p06 (s10_018b_closed_loop, 2026-10-31): the 0.1.8b snapshot p00.6.pt of training attempt 2
#: (chosen (both gates, below 0.1.7 offline)), exported by s8_018b_build/src/export_018b.py.
CHECKPOINT_PATH = "{retrain_store}/018b/release/drive_jepa_018b_p06.ckpt"
CHECKPOINT_SHA256 = "3d566eb211b5ce0f556680196a46fc0a74da29091ae62ac588910a48d62b8e66"
CHECKPOINT_BYTES = 1273003357
CHECKPOINT_OVERRIDE_ENV = "DRIVEJEPA_CHECKPOINT_OVERRIDE"


def checkpoint_pin() -> tuple[str, str, int, str]:
    """``(path, sha256, n_bytes, source)`` of the weights to load; source is "pinned" or "override".

    Raises RuntimeError while the pin above is still the placeholder and no
    override is given, and on a malformed override.
    """
    import os

    raw = os.environ.get(CHECKPOINT_OVERRIDE_ENV, "").strip()
    if raw:
        parts = raw.split("|")
        if len(parts) != 3 or len(parts[1]) != 64 or not parts[2].isdigit():
            raise RuntimeError(f"{CHECKPOINT_OVERRIDE_ENV}={raw!r} is not '<path>|<sha256>|<bytes>'")
        return parts[0], parts[1], int(parts[2]), "override"
    if CHECKPOINT_PLACEHOLDER in (CHECKPOINT_PATH, CHECKPOINT_SHA256) or CHECKPOINT_BYTES < 0:
        raise RuntimeError(
            "drivejepa 0.1.8b-p06 checkpoint pin is still the PLACEHOLDER (constants.py "
            "CHECKPOINT_PATH/SHA256/BYTES); write the real pin, or set "
            f"{CHECKPOINT_OVERRIDE_ENV} for the offline build check only"
        )
    return CHECKPOINT_PATH, CHECKPOINT_SHA256, CHECKPOINT_BYTES, "pinned"


#: 0.1.7: the three runtime corrections of s10_retrain/train_patch/batch_fix.py
#: (batch_fix.py here is a byte-identical copy, sha256 below) are installed on
#: the model in the worker and switched on for every forward. At batch 1 they
#: leave every tensor as it was (s10_retrain/gate_fix.json: 0.0); they are on
#: so the car runs the code path the S10 generator was trained under.
#: Env DRIVEJEPA_BATCH_FIX=off exists for the build gate only; the launcher
#: refuses to serve with it.
BATCH_FIX = True
BATCH_FIX_ENV = "DRIVEJEPA_BATCH_FIX"
BATCH_FIX_SHA256 = "33b90d9ed5eb08041df480ebb5a2e56f12715579df964ad37b90c9d7d5f9aff7"

#: Rows blacked out at the bottom of each Pipeline-A frame, AFTER crop and
#: resize (the 256-row model input), in [0, 1] space. Training masks the same
#: rows (s1_converter/s1_convert.py BONNET_MASK_ROWS; 37 = highest measured
#: bonnet edge 35 + 2, s2_label2/s21_bonnet.csv). 0 restores 0.1.4b.
BONNET_MASK_ROWS = 37

#: The two acceleration slots of ego_status (indices 5:7) are fed as zeros,
#: as in training (s3_5_smoke/train_patch/dataset_s35.py, zero_accel).
ZERO_ACCELERATION = True
EGO_ACCEL_SLOTS = (5, 7)

#: Decided by Sid for the S5 trial: the §3.5 momentum re-score also runs from
#: zero acceleration (planner._momentum_request -> momentum.py ego_state). Off
#: gives the re-score the reported acceleration, as in 0.1.4b.
ZERO_MOMENTUM_ACCELERATION = True

# --- Pipeline A: the V-JEPA branch (INTERFACE_DRIVEJEPA.md §3.1) ------------
# The branch whose output the network actually consumes.

#: Rows cropped from the top and bottom before resizing, for a 2:1 aspect.
CROP_ROWS = 28
#: cv2.resize target, given as (width, height) -- cv2's argument order.
RESIZE_WH = (512, 256)
#: The resulting tensor shape per frame, (C, H, W).
CAMERA_FEATURE_CHW = (3, 256, 512)
#: Number of frames the V-JEPA encoder consumes (``img_as_video_nframes = 2``).
CONTEXT_FRAMES = 2
#: Nominal spacing between those two frames. Drive-JEPA is a 2 Hz agent.
FRAME_SPACING_US = 500_000

#: ImageNet normalisation applied *inside the model* to Pipeline A, on [0,1]
#: inputs (drive_jepa_model.py:42-47). The planner must NOT pre-apply this --
#: ToTensor's divide-by-255 is where the planner stops.
IMAGENET_MEAN_01 = (0.485, 0.456, 0.406)
IMAGENET_STD_01 = (0.229, 0.224, 0.225)

# --- Pipeline B: the BEV branch (INTERFACE_DRIVEJEPA.md §3.2) ---------------
# Its normalised pixels are never read by forward(); its lidar2img and
# img_shape are. Replicated in full anyway, so the shapes and the ordering are
# exercised and so re-enabling the branch upstream does not silently change us.

#: mmcv imnormalize statistics, on the 0-255 scale.
BEV_MEAN_255 = (123.675, 116.28, 103.53)
BEV_STD_255 = (58.395, 57.12, 57.375)
#: ``to_rgb=True``. See planner._bev_normalize for why this must NOT be "fixed".
BEV_TO_RGB = True
#: RandomScaleImageMultiViewImage(scales=[0.4]) -- one element, so deterministic.
BEV_SCALE = 0.4
#: PadMultiViewImage(size_divisor=32).
BEV_PAD_DIVISOR = 32

#: The fixed BEV camera slot order (bev_feature_build.py:27). Slot 1 must be the
#: front camera, because forward() keeps only ``lidar2img[:, 1:2]``. See §6.1.
BEV_SLOT_ORDER = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")
#: Index forward() slices. Anything else here is a silent misprojection.
FRONT_SLOT_INDEX = 1
#: Minimum slots required for that slice to be in range.
MIN_BEV_SLOTS = 2

#: Per-channel RGB fill of the dummy image that holds slot 0. A real camera
#: frame that is *uniformly* this exact triple is not a thing, which is what
#: makes it usable as a marker the fake can detect (§6.1).
DUMMY_FILL_RGB = (17, 42, 99)

# --- Ego status (INTERFACE_DRIVEJEPA.md §3.3) -------------------------------

#: cat([ego_pose(3), ego_velocity(2), ego_acceleration(2), driving_command(4)]).
EGO_STATUS_DIM = 11
#: Slice offsets into that vector.
EGO_POSE_SLICE = slice(0, 3)
EGO_VELOCITY_SLICE = slice(3, 5)
EGO_ACCEL_SLICE = slice(5, 7)
EGO_COMMAND_SLICE = slice(7, 11)
#: Index of forward velocity within the full vector -- the fake reads this.
EGO_VELOCITY_X_INDEX = 3

#: One-hot layout: [left, straight, right, unknown].
COMMAND_LEFT = (1, 0, 0, 0)
COMMAND_STRAIGHT = (0, 1, 0, 0)
COMMAND_RIGHT = (0, 0, 1, 0)
COMMAND_UNKNOWN = (0, 0, 0, 1)

#: Route -> command thresholds. Taken from AlpaSim's VaVAM preset, which is the
#: convention our own INTERFACE.md §6.3(c) documents: first waypoint at least
#: MIN_LOOKAHEAD_M away; |y| beyond COMMAND_LATERAL_M turns it into a turn.
MIN_LOOKAHEAD_M = 20.0
COMMAND_LATERAL_M = 3.0

# --- Output (INTERFACE_DRIVEJEPA.md §4) -------------------------------------

#: Poses per trajectory, and proposals per forward pass.
NUM_POSES = 8
NUM_PROPOSALS = 32
#: TrajectorySampling(time_horizon=4, interval_length=0.5) -> 0.5 s spacing.
STEP_US = 500_000
#: Sub-scores per proposal; forward() reads the last one as the PDM score.
SCORE_NUM = 6
#: Per-pose layout is (x, y, heading) -- StateSE2Index.
POSE_STATE_SIZE = 3


# ===========================================================================
# 0.1.4 -- the shared 20 Hz referee
# ===========================================================================
# Under the shared referee every model runs on one clock:
#   control_timestep_us == frame_interval_us == 50_000, n_sim_steps == 400.
# Drive-JEPA is still a 2 Hz agent, so the adapter divides: the harness asks
# every 50 ms, real inference happens every 10th ask. Everything below exists
# to make that division safe rather than merely cheap.

#: The replan grid. Drive-JEPA's native rate, and the spacing its context pair
#: and its output waypoints are both built on (FRAME_SPACING_US, STEP_US).
#: The divider keys on elapsed sim time against this grid -- never on a modulo
#: counter and never on ObservationView.drive_count, which is incremented only
#: AFTER the scaffold's "no egomotion yet" early return (scaffold.py:630-638)
#: and so is not the harness's tick index.
REPLAN_PERIOD_US = 500_000

#: Standstill gate, as a SPEED rather than a distance.
#:
#: 0.1.3 gated on a fixed 0.5 m pose delta (planner.py:59,573). That was tuned
#: at 500 ms ticks, where 0.5 m over 0.5 s == 1.0 m/s. The threshold is
#: implicitly `0.5 / dt`, so at the referee's 50 ms ticks it silently became
#: 10.0 m/s: every handover below 36 km/h would have passed the warm-up
#: zero-velocity sentinel straight through to the model, in the one ego_status
#: row forward() reads (INTERFACE_DRIVEJEPA.md 3.3).
#:
#: Expressed as a speed it is tick-rate independent and, at 500 ms, decides
#: bit-for-bit what 0.1.3 decided: dividing by dt=0.5 is exact in binary, so
#: `distance/0.5 <= 1.0` and `distance <= 0.5` compare identically.
HANDOVER_MIN_SPEED_M_S = 1.0

#: Fractional slack on the realized context spacing, as a HARD limit.
#:
#: 0.1.3 carried 0.5 -- anything from 250 ms to 750 ms was accepted silently
#: (planner.py:390,458-459). At 20 Hz with the scaffold's default 8-frame
#: buffer the best available pair is 350 ms, which sits inside that window: the
#: model would have been fed 350 ms of context against a checkpoint trained at
#: 500 ms, with no log line. 0.10 admits only the grid pair at either cadence
#: (at 20 Hz the achievable gaps are multiples of 50 ms, so +-10% == +-50 ms
#: admits exactly 500 000).
CONTEXT_SPACING_TOLERANCE = 0.10

#: Frames the scaffold must retain per camera for the 500 ms pair to exist.
#:
#: At 50 ms pushes, reaching 500 ms back needs the newest plus ten more. The
#: scaffold's default is 8 (scaffold.py:361; serve_planner.py:186-191), which
#: spans only 350 ms -- the bug above. 16 leaves margin without holding many
#: more 1080p JPEGs. Set explicitly by the launcher; never inherited.
DEFAULT_IMAGE_HISTORY = 16

#: How far ahead the controller may sample a returned plan, in microseconds.
#:
#: The MPC builds `now + k*dt_mpc` for k in 0..n_horizon and interpolates the
#: reference there (nonlinear_mpc.py:315-330, linear_mpc.py:180-190). The
#: shipped controller config is n_horizon=20, dt_mpc=0.1 -> 2.0 s. A re-timed
#: cached plan must cover [now, now + this] at every staleness the divider can
#: produce, or the controller is interpolating outside the range it was handed.
MAX_CONTROLLER_LOOKAHEAD_US = 2_000_000


# ===========================================================================
# 0.1.4b -- paper §3.5 momentum-aware selection (DRIVEJEPA_014B)
# ===========================================================================
# 0.1.4b is 0.1.4 plus one thing: the model is handed the trajectory it chose
# at the previous replan, the way upstream's own evaluation loop hands it over
# (run_pdm_score_sequential.py:130-138,140,167-168), so the vendored
# calibrate_score (drive_jepa_model.py:87-88,98-151) runs before the argmax.
# 0.1.4 called agent.forward with image and ego features only, so that branch
# never ran. Frames, features, divider and anchoring are 0.1.4's, unchanged
# (divider.py is a byte-for-byte copy). The worker half lives in momentum.py.

#: Env var. "on" (default) feeds the previous plan; "off" is exact 0.1.4
#: selection. Read once, when the planner is constructed.
MOMENTUM_ENV = "DRIVEJEPA_MOMENTUM"
MOMENTUM_MODES = ("on", "off")
#: 0.1.8: the default is pinned here instead of living as a literal inside
#: planner.momentum_mode. The env var still overrides it; the launcher's
#: provenance line prints both the effective value and this pinned default.
MOMENTUM_DEFAULT = "on"

#: Key under which the planner hands the worker what momentum.py needs. Absent
#: from the feature dict when the switch is off, which is what makes the off
#: path the 0.1.4 call, byte for byte.
MOMENTUM_FEATURE_KEY = "momentum"

#: Upstream drops the previous plan when the two frames are further apart than
#: this, in seconds (run_pdm_score_sequential.py:131). Applied in the worker;
#: repeated here only so the log line and the tests can name it.
MOMENTUM_MAX_AGE_S = 0.55

#: The second provenance line the launcher prints.
MOMENTUM_PROVENANCE = (
    "drivejepa-0.1.8b-p06-memory5 | 0.1.8b-p06 adapter logic and encoder, build-5 generator + scorer + memory blocks from constants.memory_pin, batch fix on (fix1+fix2+fix3), bonnet mask 37 rows, acceleration zeroed in the model input and in the §3.5 re-score"
    " | paper §3.5 momentum-aware selection (upstream calibrate_score)"
)

#: Per-replan outcome codes written to the debug payload.
MOMENTUM_OK = "OK"                  # previous plan fed, score recalibrated
MOMENTUM_NO_PREV = "NO_PREV"        # first replan of the session: plain S
MOMENTUM_DROPPED_AGE = "DROPPED_AGE"  # previous plan older than 0.55 s: plain S


# ===========================================================================
# Memory (approach 3, build 5) -- the only thing this package adds to 0.1.8b-p06
# ===========================================================================
# Everything above is 0.1.8b-p06's constants.py with the version string and the
# provenance text changed. The base checkpoint pin (CHECKPOINT_*) is unchanged:
# the encoder is 0.1.8b pass 0.6. The generator, the scorer and the memory
# blocks come from the build-5 checkpoint pinned here (memory.py loads them over
# the base model; the training-only read-out is skipped).

#: Env var naming the DIRECTORY that holds the memory checkpoint and its sidecar.
#: The directory picks the arm: the mode (carry / reset) is a field of the
#: sidecar, never a command-line flag or an env value.
MEMORY_ENV = "DRIVEJEPA_MEMORY"
MEMORY_CHECKPOINT_NAME = "step0930.pt"
MEMORY_SIDECAR_NAME = "memory_mode.json"
#: runs/build5_short/ckpt/step0930.pt (model version 3), hashed 2026-11-12.
MEMORY_SHA256 = "84359783ef502e1ccc3a2dcff013bf45d215f9fb11e046cc8110c64581434e21"
MEMORY_BYTES = 36748734
MEMORY_VERSION = 3
MEMORY_MODES = ("carry", "reset")
#: Chain rule: the memory is carried only if the previous call of the session is
#: this far back in simulated time; otherwise it restarts from the start state
#: and the call is counted as a chain reset.
MEMORY_CHAIN_MIN_US = 450_000
MEMORY_CHAIN_MAX_US = 550_000
#: Key under which the planner hands the worker (session_uuid, time_us, ego_pose).
MEMORY_FEATURE_KEY = "memory"
#: Key of a control message (session start / close) from the planner to the worker.
MEMORY_CONTROL_KEY = "__memory_control__"
#: Each call also evaluates the other mode's memory on the same input (memory.py).
MEMORY_SHADOW = True
#: Offline proofs only: "off" skips the shadow. The run script refuses to drive with it set.
MEMORY_SHADOW_ENV = "DRIVEJEPA_MEMORY_SHADOW"


def memory_pin() -> dict:
    """The memory checkpoint to load and its mode, read from the sidecar. No torch.

    Refuses: no directory, a missing file, a sidecar whose mode / version / sha /
    base sha is wrong, a file whose size or sha256 differs from the sidecar. A
    file that is not the pinned build-5 checkpoint is accepted only when the
    sidecar says ``"test_copy": true`` (offline proofs); the source is then
    reported as TEST-COPY and the run script refuses to drive on it.
    """
    import hashlib
    import json
    import os
    import pathlib

    raw = os.environ.get(MEMORY_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{MEMORY_ENV} is not set: it must name the directory holding {MEMORY_CHECKPOINT_NAME} and {MEMORY_SIDECAR_NAME}")
    folder = pathlib.Path(raw)
    path, sidecar = folder / MEMORY_CHECKPOINT_NAME, folder / MEMORY_SIDECAR_NAME
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"{folder}: {MEMORY_CHECKPOINT_NAME} or {MEMORY_SIDECAR_NAME} missing")
    side = json.loads(sidecar.read_text())
    if side.get("mode") not in MEMORY_MODES:
        raise RuntimeError(f"{sidecar}: mode {side.get('mode')!r} is not one of {MEMORY_MODES}")
    if side.get("version") != MEMORY_VERSION:
        raise RuntimeError(f"{sidecar}: version {side.get('version')!r} != {MEMORY_VERSION}")
    if side.get("base_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError(f"{sidecar}: base checkpoint sha256 is not the pinned 0.1.8b-p06")
    if side.get("read_out_loaded") is not False:
        raise RuntimeError(f"{sidecar}: read_out_loaded must be false (the read-out is training-only)")
    n_bytes = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    sha = digest.hexdigest()
    if sha != side.get("checkpoint_sha256") or n_bytes != side.get("checkpoint_bytes"):
        raise RuntimeError(f"{path}: sha256 {sha} / {n_bytes} bytes differ from the sidecar")
    test_copy = bool(side.get("test_copy", False))
    if not test_copy and (sha != MEMORY_SHA256 or n_bytes != MEMORY_BYTES):
        raise RuntimeError(f"{path}: not the pinned build-5 checkpoint and the sidecar does not mark a test copy")
    if test_copy and sha == MEMORY_SHA256:
        raise RuntimeError(f"{sidecar}: marks a test copy but the file is the pinned checkpoint")
    return {"dir": str(folder), "path": str(path), "sidecar": str(sidecar), "sha256": sha, "bytes": n_bytes,
            "mode": side["mode"], "version": side["version"], "source": "TEST-COPY" if test_copy else "pinned",
            "base_sha256": CHECKPOINT_SHA256}
