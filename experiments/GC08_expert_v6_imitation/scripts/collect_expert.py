"""Collect episodes where expert v6 drives, through the real neuroNCAP engine and renderer.

The offline rollout (gallops `expert/rollout.py`) already worked out, tick by tick, the plan the expert
hands the engine on every training episode. Here the engine drives those plans for real: the renderer
draws what the camera sees along the expert's braking and swerving, and the model node (drivejepa_ncapdump
with DRIVEJEPA_DRIVE=expert) hands over the expert's plan while still running and dumping Drive-JEPA's own
32 candidates. The node asserts the engine put the car within 2 cm of where the rollout did, so this also
checks the offline copy against the real engine.

The node learns which episode it is in from a small file this script writes before each engine call
(DRIVEJEPA_EXPERT_CURRENT); one engine call per seed, through main_offset.py so every run lands in run_<k>/.

    PADDOCK_CONFIG=paddock.collect.expert.toml python3 collect_expert.py --tag expert --scenarios <dir> \\
        --current <file> [--scene 0053 ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import collect
from collect import OPS, WAYPOINT_DT, free_port, renderable_cameras, renderer_command, say, wait_for_renderer

from paddock.config import SiteConfig
from paddock.registry import load_models
from paddock.runtime import proc


def run_seed(cfg, spec, scene, shape, seed, node_url, renderer_port, out_root, state, scenarios, current) -> int:
    done = state / f"{shape}-{scene}__run_{seed}.done"
    if done.exists():
        return 0
    out_dir = out_root / f"{shape}-{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    current.write_text(f"{shape}-{scene}/run_{seed}")
    settings = cfg.benchmarks["neuroncap"]
    command = [
        settings.get("python"), "-u", str(OPS / "main_offset.py"), "--first-seed", str(seed),
        "--engine.renderer.port", str(renderer_port),
        "--engine.model.port", node_url.rsplit(":", 1)[-1],
        "--engine.dataset.data_root", str(cfg.datasets["nuscenes"].resolve()),
        "--engine.dataset.version", "v1.0-trainval",
        "--engine.dataset.sequence", str(int(scene)),
        "--engine.scenario.path", str(scenarios / shape / f"{scene}.yaml"),
        "--engine.waypoint-dt", str(WAYPOINT_DT),
        "--engine.cameras", *renderable_cameras(list(spec.cameras or ["CAM_FRONT"])),
        "--engine.logger.log-dir", str(out_dir),
        "--runs", "1",
    ]
    result = subprocess.run(command, cwd=settings.get("repo"))
    run_dir = out_dir / f"run_{seed}"
    for junk in ("FC_TRAJ", "COMBINED_OUTPUTS"):
        stale = run_dir / junk
        if stale.is_dir():
            for item in sorted(stale.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            stale.rmdir()
    if result.returncode == 0:
        done.parent.mkdir(parents=True, exist_ok=True)
        done.touch()
    else:
        say(f"  {shape}-{scene}/run_{seed}: engine exit {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True, help="the file the node reads the episode from")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--shape", action="append")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    scenes = args.scene or collect.SCENES
    shapes = args.shape or collect.SHAPES
    cfg = SiteConfig.load()
    spec = load_models()[collect.MODEL]
    out_root = collect.OUT.parent / f"{collect.OUT.name}_{args.tag}"
    state = OPS / f"state_{args.tag}"
    state.mkdir(parents=True, exist_ok=True)
    say(f"expert-driven collection -> {out_root}")
    for scene in scenes:
        wanted = [(shape, k) for shape in shapes for k in range(args.runs)]
        if all((state / f"{shape}-{scene}__run_{k}.done").exists() for shape, k in wanted):
            say(f"{scene}: already collected")
            continue
        renderer_port = free_port(8100)
        command, cwd, env = renderer_command(cfg, scene, renderer_port)
        say(f"{scene}: starting renderer on {renderer_port}")
        log = OPS / f"renderer_{args.tag}_{scene}.log"
        with log.open("w") as handle:
            renderer = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            try:
                if not wait_for_renderer(renderer_port):
                    say(f"{scene}: renderer never answered; see {log}")
                    continue
                say(f"{scene}: renderer ready")
                port = free_port(9980)
                with proc.model_node(spec, cfg, port, OPS, echo=say) as node_url:
                    for shape, seed in wanted:
                        run_seed(cfg, spec, scene, shape, seed, node_url, renderer_port, out_root, state,
                                 args.scenarios, args.current)
            finally:
                renderer.terminate()
                try:
                    renderer.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    renderer.kill()
                time.sleep(10)
    total = len(list(state.glob("*.done")))
    say(f"=== expert collection finished: {total} of {len(scenes) * len(shapes) * args.runs} episodes")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
