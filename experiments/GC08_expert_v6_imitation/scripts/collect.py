"""Collect inserted-actor episodes on the five non-benchmark train scenes.

This is data generation, not evaluation, so it drives neuroNCAP's engine directly
instead of going through `paddock run`. Two reasons. Our scenarios live outside
paddock's tracked `scenarios_10hz/`, because a training scene must never become a
benchmark scene. And nothing here should land in `results/` -- these episodes are
raw material, not a score.

It still reuses paddock's own launchers for the renderer and the model node, so
the environment a node sees is the one a benchmark run would give it. The only
thing assembled by hand is the engine command, which is the part that has to
differ.

What each episode leaves behind, all of it already written by the engine and the
dump adapter -- no new plumbing:

    CAM_FRONT/<timestamp>.jpg   the raw render the model read (10 Hz)
    ego_poses.json              where the car actually went
    actors.json                 the inserted car's full trajectory
    reference_trajectory.json   the logged human path
    trajectories.json           the plan the model drove
    <dump dir>/ep_*.jsonl       all 32 candidates per tick

`--engine.logger.enabled` is left ON deliberately: it is what writes the frames,
and paddock's own harness turns it off when video is off.

    setsid nohup python3 collect.py > launcher.out 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# The analysis worktree, not main paddock: `load_models()` discovers adapters by
# walking the checkout it is imported from, and `drivejepa_ncapdump` -- the model
# that logs all 32 candidates -- is deliberately kept out of the tracked tree.
PADDOCK = Path(
    os.environ.get("COLLECT_PADDOCK_SRC", "{repos}/paddock-dj-ncap-cands")
)
sys.path.insert(0, str(PADDOCK / "src"))

from paddock.adapters.benchmarks.neuroncap.harness import renderable_cameras  # noqa: E402
from paddock.adapters.benchmarks.neuroncap.launcher import renderer_command  # noqa: E402
from paddock.config import SiteConfig  # noqa: E402
from paddock.registry import load_models  # noqa: E402
from paddock.runtime import proc  # noqa: E402

OPS = Path("{paddock_runs}/ops/dj_collect")
OUT = Path("{runs}/collect")
SCENARIOS = Path("{runs}/scenarios")
SCENES = ["0075", "0064", "0070", "0053", "0058"]
SHAPES = ["stationary", "frontal", "side"]
MODEL = "drivejepa_ncapdump"
WAYPOINT_DT = 0.5


def say(message: str) -> None:
    stamp = time.strftime("%F %H:%M %Z", time.localtime())
    print(f"{stamp} {message}", flush=True)


def free_port(start: int) -> int:
    for port in range(start, start + 200):
        with contextlib.closing(socket.socket()) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


# 3600, not 1800 (2026-10-15 PDT): under the shared box's load, 0075, 0064 and 0070 were still parsing
# their scene at 30 min, while 0053 answered at 16 min.
def wait_for_renderer(port: int, timeout_s: int = 3600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket()) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(10)
    return False


def run_episode(
    cfg,
    spec,
    scene: str,
    shape: str,
    node_url: str,
    renderer_port: int,
    runs: int,
    out_root: Path = OUT,
    state: Path = OPS / "state",
    first_seed: int = 0,
    scenarios: Path = SCENARIOS,
) -> int:
    out_dir = out_root / f"{shape}-{scene}"
    done = state / f"{shape}-{scene}.done"
    if done.exists():
        say(f"  {shape}-{scene}: already collected")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = cfg.benchmarks["neuroncap"]
    # Seeds 0..N-1 are what main.py runs, and the released model is deterministic,
    # so a second collection under it must start past the seeds already used.
    entry = ["main.py"] if first_seed == 0 else [str(OPS / "main_offset.py"), "--first-seed", str(first_seed)]
    command = [
        settings.get("python"),
        "-u",
        *entry,
        "--engine.renderer.port",
        str(renderer_port),
        "--engine.model.port",
        node_url.rsplit(":", 1)[-1],
        "--engine.dataset.data_root",
        str(cfg.datasets["nuscenes"].resolve()),
        "--engine.dataset.version",
        "v1.0-trainval",
        "--engine.dataset.sequence",
        str(int(scene)),
        "--engine.scenario.path",
        str(scenarios / shape / f"{scene}.yaml"),
        "--engine.waypoint-dt",
        str(WAYPOINT_DT),
        "--engine.cameras",
        # Every camera the model DECLARES, not just the one it reads. Drive-JEPA's
        # feature builder fills four slots and raises on a missing one, and the
        # projection it keeps is positional within that list. The logger still
        # writes only CAM_FRONT (`allowed_cams`), which is all the training set
        # needs, so the extra renders cost wall clock and no disk.
        *renderable_cameras(list(spec.cameras or ["CAM_FRONT"])),
        "--engine.logger.log-dir",
        str(out_dir),
        "--runs",
        str(runs),
    ]
    say(f"  {shape}-{scene}: {runs} runs -> {out_dir}")
    result = subprocess.run(command, cwd=settings.get("repo"))
    if result.returncode == 0:
        # The logger's debug renders are ~17 MB an episode of per-step matplotlib
        # frames, four times the training pixels. CAM_FRONT is what the model
        # read and what the dataset needs; the rest is a picture of it.
        for junk in ("FC_TRAJ", "COMBINED_OUTPUTS"):
            for stale in out_dir.rglob(junk):
                if stale.is_dir():
                    for item in stale.rglob("*"):
                        if item.is_file():
                            item.unlink()
                    for item in sorted(stale.rglob("*"), reverse=True):
                        if item.is_dir():
                            item.rmdir()
                    stale.rmdir()
        done.parent.mkdir(parents=True, exist_ok=True)
        done.touch()
    else:
        say(f"  {shape}-{scene}: engine exit {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="seeds per scenario")
    parser.add_argument("--scene", action="append", help="limit to these scenes")
    parser.add_argument("--shape", action="append", help="limit to these shapes")
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="DAgger round. 0 keeps the original paths; later rounds get their own "
        "output, dump and state directories so nothing overwrites a previous round.",
    )
    parser.add_argument("--first-seed", type=int, default=0, help="first jitter seed (round 0 used 0-9)")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS, help="scenario root (shape/scene.yaml)")
    parser.add_argument("--tag", help="names this collection's output and state dirs; default comes from --round")
    args = parser.parse_args()

    scenes = args.scene or SCENES
    shapes = args.shape or SHAPES
    cfg = SiteConfig.load()
    spec = load_models()[MODEL]
    # Round 0 keeps the unsuffixed paths it was collected under; a later round
    # must not land on top of it, because the training set is their UNION.
    suffix = f"_{args.tag}" if args.tag else ("" if args.round == 0 else f"_round{args.round}")
    out_root = OUT.parent / f"{OUT.name}{suffix}"
    state = OPS / f"state{suffix}"
    state.mkdir(parents=True, exist_ok=True)
    say(f"round {args.round}: episodes -> {out_root}")
    say(f"round {args.round}: the checkpoint that drives comes from PADDOCK_CONFIG")

    # One renderer per scene: it loads that scene's hash grids at startup and
    # cannot switch, so a five-scene sweep is five renderers, not one.
    for scene in scenes:
        if all((state / f"{shape}-{scene}.done").exists() for shape in shapes):
            say(f"{scene}: all shapes already collected")
            continue
        renderer_port = free_port(8100)
        command, cwd, env = renderer_command(cfg, scene, renderer_port)
        say(f"{scene}: starting renderer on {renderer_port}")
        log = OPS / f"renderer_{scene}.log"
        with log.open("w") as handle:
            renderer = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            try:
                if not wait_for_renderer(renderer_port):
                    say(f"{scene}: renderer never answered; see {log}")
                    continue
                say(f"{scene}: renderer ready")
                port = free_port(9980)
                with proc.model_node(spec, cfg, port, OPS, echo=say) as node_url:
                    for shape in shapes:
                        run_episode(
                            cfg, spec, scene, shape, node_url, renderer_port, args.runs,
                            out_root=out_root, state=state, first_seed=args.first_seed,
                            scenarios=args.scenarios,
                        )
            finally:
                renderer.terminate()
                try:
                    renderer.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    renderer.kill()
                time.sleep(10)  # let the driver reclaim before the next scene

    total = len(list(state.glob("*.done")))
    say(f"=== collection finished: {total} of {len(scenes) * len(shapes)} scenario-scenes")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
