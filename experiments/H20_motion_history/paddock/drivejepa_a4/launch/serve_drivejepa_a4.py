#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""APPROACH 4 (adapters/drivejepa_a4/launch): copy of drafts/drivejepa-0.1.8b-p06/serve_drivejepa_0_1_8b_p06.py; changes: the package is
imported by its own name (adapters/drivejepa_a4), the expected version, the approach-4 config must resolve, a third provenance line.

Serve the Drive-JEPA adapter at 0.1.8 (0.1.7 logic; 0.1.4 + paper §3.5 momentum-aware selection).

    alpasim/.venv/bin/python drafts/drivejepa-0.1.8b-p06/serve_drivejepa_0_1_8b_p06.py \
        --host 0.0.0.0 --port 9149 --device cpu --control-timestep-us 50000

Why this launcher exists rather than `PYTHONPATH=... adapters/serve_planner.py`:
Python puts the *script's own directory* at sys.path[0], ahead of PYTHONPATH.
Running `adapters/serve_planner.py` therefore always imports
`adapters/drivejepa` -- the 0.1.3 package -- no matter what PYTHONPATH says.
This script lives in the draft, so the draft's `drivejepa` package wins, and
`adapters/` is appended after it for the shared scaffold.

**What 0.1.6 adds to 0.1.4.** One thing: at every replan the model is given
the trajectory it chose at the previous replan, the way upstream's evaluation
loop gives it (run_pdm_score_sequential.py:130-138,167-168), so the vendored
``calibrate_score`` runs before the argmax (drive_jepa_model.py:87-91). The
planner keeps the previous plan per rollout; the worker (momentum.py) builds
upstream's three feature keys with upstream's functions. DRIVEJEPA_MOMENTUM=off
restores exact 0.1.4 selection. Everything below is 0.1.4's launcher text.

**What is different about 0.1.4 (inherited).** The referee runs every model on one clock:
control_timestep_us == frame_interval_us == 50_000, n_sim_steps == 400.
Drive-JEPA is still a 2 Hz agent, so the adapter divides -- see
`drivejepa/divider.py`. Three things the 0.1.3 launcher did not do:

  1. It builds a `ReplanDivider` around the planner. The divider owns the
     2 Hz grid; the planner stays a pure function of one observation.
  2. It pins `--image-history` to K.DEFAULT_IMAGE_HISTORY (16) instead of
     inheriting serve_planner's 8. At 50 ms pushes, 8 frames span 350 ms and
     the 0.5 s context pair simply cannot exist -- which 0.1.3 would have
     accepted silently.
  3. It REFUSES `--target-frame-spacing-us` rather than ignoring it. 0.1.3
     accepted the flag through the shared argparse and then never read it
     (serve_planner.py:95-112), so a run could be launched with a frame
     pairing nobody actually asked for. OpenPilot already refuses it
     (serve_planner.py:79-89); this matches that behaviour.

**Why it does not call serve_planner.main().** serve_planner is shared with
0.1.3, openpilot, visionpilot and the blind baseline. Teaching its `_drivejepa`
factory about the divider, the image-history floor and the flag refusal would
change 0.1.3's launch path, and 0.1.3 is the provenance for a published row.
The shared `scaffold` module is still imported from adapters/ -- only the
registry and CLI are local.

**Why 0.1.8 lives in adapters/drivejepa018bp06/ and not adapters/drivejepa/.**
Same reason. 0.1.3's gate reads `constants.ADAPTER_VERSION`
(serve_drivejepa_0_1_3.py:83-90); bumping it in the shared module would make
0.1.3 refuse to start. Everything 0.1.6 does not change -- real_model.py,
fake_model.py, src/ -- is a symlink back to adapters/drivejepa/,
so there is still exactly one copy of each in existence.

Three things enforce provenance, rather than leaving it to convention:

  1. ``_resolve_planner`` refuses to serve a planner whose real path is not
     under ``adapters/drivejepa018bp06/`` -- a re-introduced real file in the draft
     fails at startup instead of quietly winning. That is the 0.1.2 failure
     mode, and the 0.1.3 gate that caught it, pointed at the new package.
  2. The version gate reads ``constants.ADAPTER_VERSION`` from the *imported*
     module, so a stale package cannot claim to be this one.
  3. ``log_provenance`` prints the resolved path into the run's driver.log, so
     every run directory records which file actually drove it.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# 0.1.8 lives in Alpasim/drafts/drivejepa-0.1.8b-p06/ (this file) and Alpasim/adapters/drivejepa018bp06/ (the package),
# next to the shared scaffold in Alpasim/adapters/.
_ADAPTERS = _HERE.parents[1]          # approach 4: this file is adapters/drivejepa_a4/launch/
_LIVE_PACKAGE = _ADAPTERS / "drivejepa_a4"
assert (_ADAPTERS / "scaffold.py").is_file(), _ADAPTERS

# Approach 4: adapters/ holds both the package (drivejepa_a4, imported by its own name) and the shared scaffold.
if str(_ADAPTERS) not in sys.path:
    sys.path.insert(0, str(_ADAPTERS))

#: What this launcher is for. The *expectation*; the reported value always
#: comes out of the imported module, never from this literal.
EXPECTED_VERSION = "0.1.8b-p06-a4"

#: AlpaSim logical_id of the front camera. Same as serve_planner's default.
FRONT_CAMERA_ID = "camera_front_wide_120fov"

logger = logging.getLogger("serve_drivejepa_a4")


def _resolve_planner():
    """Import the planner the way the server will, and prove where it came from.

    Returns ``(planner_module, divider_module, constants_module, real_path)``.
    Raises SystemExit -- at startup, before any GPU or scene time -- if the
    package on the path is not the live one, or is not the version this
    launcher is for.
    """
    # Import the planner FIRST and check where it came from before importing
    # anything else. A wrong `drivejepa` on the path may not even have a
    # divider module, and "ModuleNotFoundError: drivejepa.divider" is a much
    # worse thing to read at 3am than the refusal below.
    import drivejepa_a4.planner as planner

    # realpath, not __file__: the draft's planner.py is a symlink, and the
    # question worth answering in a log is which *source file* ran.
    real_path = Path(os.path.realpath(planner.__file__))

    if real_path.parent != _LIVE_PACKAGE.resolve():
        raise SystemExit(
            f"refusing to serve: planner resolved to {real_path}\n"
            f"expected a file under {_LIVE_PACKAGE.resolve()}\n"
            "A real planner.py has been re-introduced into a draft and is "
            "shadowing the live adapter. That is the 0.1.2 failure mode: the "
            "test suite would still be green against the live file while this "
            "process served a different one. Make it a symlink."
        )

    import drivejepa_a4.constants as constants
    import drivejepa_a4.divider as divider

    divider_path = Path(os.path.realpath(divider.__file__))
    if divider_path.parent != _LIVE_PACKAGE.resolve():
        raise SystemExit(
            f"refusing to serve: divider resolved to {divider_path}\n"
            f"expected a file under {_LIVE_PACKAGE.resolve()}\n"
            "The divider decides which ticks run inference and how a cached "
            "plan is re-anchored. A shadowed copy would be invisible in the "
            "logs and would change every tick's output."
        )

    # 0.1.6: the two files that carry the momentum path. bridge.py decides which
    # model class the worker builds; momentum.py is that class. Either one
    # shadowed would change every replan's pick, or silently drop §3.5.
    import drivejepa_a4.bridge as bridge

    for label, path in (
        ("bridge", Path(os.path.realpath(bridge.__file__))),
        ("momentum", Path(os.path.realpath(Path(planner.__file__).parent / "momentum.py"))),
    ):
        if path.parent != _LIVE_PACKAGE.resolve() or not path.is_file():
            raise SystemExit(
                f"refusing to serve: {label} resolved to {path}\n"
                f"expected a file under {_LIVE_PACKAGE.resolve()}\n"
                "0.1.6's momentum path lives in bridge.py (worker model class) "
                "and momentum.py (upstream's previous-plan keys). A shadowed or "
                "missing copy would drive on plain S while claiming §3.5."
            )

    # 0.1.7: the batch fix must be on. The off value exists for the build gate only.
    if not constants.BATCH_FIX or os.environ.get(constants.BATCH_FIX_ENV, "on").strip().lower() == "off":
        raise SystemExit("refusing to serve: 0.1.8b-p06 drives with the batch fix on; DRIVEJEPA_BATCH_FIX=off is for the build gate only")
    # 0.1.8: the checkpoint pin. constants.CHECKPOINT_* is a PLACEHOLDER until the 0.1.8 weights exist;
    # checkpoint_pin() refuses it unless DRIVEJEPA_CHECKPOINT_OVERRIDE is set (offline build check only).
    try:
        constants.checkpoint_pin()
        constants.a4_pin()            # approach 4: the overlay checkpoint config must resolve before anything is spent
    except RuntimeError as exc:
        raise SystemExit(f"refusing to serve: {exc}") from exc
    version = getattr(constants, "ADAPTER_VERSION", None)
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"refusing to serve: resolved ADAPTER_VERSION {version!r}, "
            f"expected {EXPECTED_VERSION!r}.\n"
            f"drivejepa package came from {real_path}\n"
            "Check sys.path ordering, or bump "
            "adapters/drivejepa018bp06/constants.py."
        )
    return planner, divider, constants, real_path


def log_provenance() -> str:
    """Emit the one startup line that says what is about to drive. Returns it.

    Called by :func:`main` before the port is bound, so it lands in the run's
    ``driver.log``. Kept separate from ``main`` so the entrypoint-resolution
    test can assert on the real thing rather than a re-implementation of it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )
    planner, divider, constants, real_path = _resolve_planner()
    message = (
        f"provenance: adapter_version={constants.ADAPTER_VERSION} "
        f"version_id={planner.DriveJEPAPlanner.version_id} "
        f"planner_module={real_path} "
        f"imported_via={planner.__file__} "
        f"divider_module={Path(os.path.realpath(divider.__file__))} "
        f"replan_period_us={constants.REPLAN_PERIOD_US}"
    )
    logger.info(message)
    # 0.1.6: the second provenance line. The switch is resolved by the same
    # function the planner uses, and the vendored model file is hashed from
    # disk so the line names the calibrate_score that will actually run.
    import hashlib

    mode = planner.momentum_mode("auto")
    ckpt_path, ckpt_sha, ckpt_bytes, ckpt_source = constants.checkpoint_pin()
    env_mode = os.environ.get(constants.MOMENTUM_ENV)
    model_file = (
        _LIVE_PACKAGE.resolve() / "src" / "navsim_v2" / "navsim" / "agents"
        / "drive_jepa_perception_based" / "drive_jepa_model.py"
    )
    model_sha = (
        hashlib.sha256(model_file.read_bytes()).hexdigest()
        if model_file.is_file() else "MISSING"
    )
    line2 = (
        f"{constants.MOMENTUM_PROVENANCE} | "
        f"switch={'on' if mode == 'on' else 'OFF (exact 0.1.4 selection)'} | "
        f"switch_effective={mode} pinned_default={constants.MOMENTUM_DEFAULT} "
        f"env {constants.MOMENTUM_ENV}={'unset' if env_mode is None else repr(env_mode)} | "
        f"checkpoint={ckpt_path} sha256={ckpt_sha} bytes={ckpt_bytes} source={ckpt_source.upper()} | "
        f"previous plan dropped above {constants.MOMENTUM_MAX_AGE_S} s | "
        f"drive_jepa_model.py sha256={model_sha} | "
        f"momentum_module={Path(os.path.realpath(_LIVE_PACKAGE / 'momentum.py'))}"
    )
    logger.info(line2)
    # Approach 4: the third provenance line, the overlay checkpoint (verified again, and logged, by the worker).
    a4 = constants.a4_pin()
    line3 = (f"approach4: arm={a4['arm']} checkpoint={a4['checkpoint']} sha256={a4['sha256']} config={a4['config']} "
             f"ego_history_sha256={constants.A4_EGO_HISTORY_SHA256} history_key={constants.HISTORY_FEATURE_KEY}")
    logger.info(line3)
    return message + "\n" + line2 + "\n" + line3


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Local rather than serve_planner's, so 0.1.3 is untouched."""
    import drivejepa_a4.constants as K

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    parser.add_argument(
        "--port",
        type=int,
        default=9149,
        help="Port to bind. Outside the block the wizard hands out from "
        "baseport. 0.1.3 has 9146, 0.1.4 9147, 0.1.5 9148.",
    )
    parser.add_argument(
        "--front-camera",
        default=FRONT_CAMERA_ID,
        help="AlpaSim logical_id of the front camera.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the Drive-JEPA worker.",
    )
    parser.add_argument(
        "--image-history",
        type=int,
        default=K.DEFAULT_IMAGE_HISTORY,
        help="Frames kept per camera. Must be long enough to reach one full "
        f"{K.FRAME_SPACING_US} us step back at the scenario's push rate; "
        f"default {K.DEFAULT_IMAGE_HISTORY}.",
    )
    parser.add_argument(
        "--control-timestep-us",
        type=int,
        default=50_000,
        help="The scenario's control_timestep_us, which must equal its "
        "frame_interval_us. Checked against the observed frame push interval "
        "on the first observation that carries two frames; a mismatch fails "
        "the rollout. Nothing in the runtime validates this. 0 disables.",
    )
    parser.add_argument(
        "--replan-period-us",
        type=int,
        default=K.REPLAN_PERIOD_US,
        help="Divider grid. Drive-JEPA's native period; changing it means "
        "asking the model for a cadence it was not trained at.",
    )
    parser.add_argument(
        "--target-frame-spacing-us",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # accepted only so it can be REFUSED, below
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build the planner, report readiness, and exit WITHOUT binding a "
        "port. The desk-check that keeps a broken adapter out of the queue.",
    )
    return parser


def build_driver(args: argparse.Namespace):
    """Planner + divider + scaffold. Separated from main() so tests can call it."""
    from scaffold import DriverScaffold

    import drivejepa_a4.constants as K
    from drivejepa_a4.bridge import BridgedDriveJEPA, BridgeError
    from drivejepa_a4.divider import ReplanDivider
    from drivejepa_a4.planner import DriveJEPAPlanner

    # Refused, not ignored: silently dropping a flag someone passed would hand
    # them a run whose frame pairing is not the one they asked for. 0.1.3 took
    # this flag through serve_planner's shared parser and never read it.
    if args.target_frame_spacing_us:
        raise SystemExit(
            "--target-frame-spacing-us is not accepted for drivejepa 0.1.8b-p06 "
            f"(got {args.target_frame_spacing_us}).\n"
            "Drive-JEPA's context pair is fixed at its trained spacing, "
            f"{K.FRAME_SPACING_US} us, and the divider replans on that same "
            "grid. The scenario cadence is set by the driver YAML "
            "(control_timestep_us / frame_interval_us), not by this flag.\n"
            "0.1.3 accepted this flag and silently dropped it; that is the "
            "hole this refusal closes."
        )

    span_us = args.image_history * args.control_timestep_us
    if args.control_timestep_us > 0 and span_us <= K.FRAME_SPACING_US:
        raise SystemExit(
            f"--image-history {args.image_history} at a "
            f"{args.control_timestep_us} us control timestep spans only "
            f"{span_us} us, which cannot hold the {K.FRAME_SPACING_US} us "
            "context pair the model needs.\n"
            f"Use at least {K.FRAME_SPACING_US // args.control_timestep_us + 1} "
            "frames. Refusing at startup rather than failing every tick."
        )

    # Two-process bridge: the model lives in the 3.9 conda env because torch 2.1
    # + mmcv-full are cp39 wheels, while this process needs 3.12 for
    # alpasim_utils/utils_rs. Construction blocks until the worker has the model
    # loaded, so the port still binds only when we are ready to drive.
    try:
        # 0.1.7 (kept in 0.1.8): the worker's stderr (which carries the checkpoint and batch-fix lines) goes to the run's own folder when the runner names one.
        model = BridgedDriveJEPA(device=args.device, stderr_log=os.environ.get("DRIVEJEPA_WORKER_LOG") or None)
    except BridgeError as exc:
        raise SystemExit(
            f"Drive-JEPA bridge could not start: {exc}\n"
            "Build the worker env with Alpasim/build_drivejepa_env.sh"
        ) from exc

    # 0.1.6: resolve the switch here, once, and hand the planner the result,
    # so the provenance line, the planner and the log below cannot disagree.
    from drivejepa_a4.planner import momentum_mode

    mode = momentum_mode("auto")
    if mode == "off":
        logger.warning(
            "momentum OFF by %s=off: this process selects exactly as 0.1.4",
            K.MOMENTUM_ENV,
        )
    else:
        logger.info("momentum on: previous plan fed to the model at every replan")

    planner = DriveJEPAPlanner(
        model_fn=model,
        front_camera_id=args.front_camera,
        expected_frame_interval_us=(args.control_timestep_us or None),
        momentum=mode,
    )
    divider = ReplanDivider(planner, replan_period_us=args.replan_period_us)
    driver = DriverScaffold(
        planner=divider,
        keep_images=True,
        image_history=args.image_history,
    )
    return divider, driver


def main() -> None:
    log_provenance()
    args = build_parser().parse_args()

    divider, driver = build_driver(args)
    logger.info(
        "built ok: version_id=%s image_history=%d control_timestep_us=%d "
        "replan_period_us=%d",
        divider.version_id,
        args.image_history,
        args.control_timestep_us,
        args.replan_period_us,
    )

    if args.check_only:
        logger.info("--check-only: not binding a port. READY.")
        return

    from scaffold import serve

    logger.info("serving on %s:%d", args.host, args.port)
    serve(args.host, args.port, driver)


if __name__ == "__main__":
    main()
