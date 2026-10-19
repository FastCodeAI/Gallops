"""Drive Alpamayo's privileged NeuroNCAP teacher on top of the existing expert.

This is a thin adapter, not a second expert. `draw.teach()` already solves
stop/swerve/brake/least-impact against the grader paddock's own NeuroNCAP
candidate analysis checks out against (see `grader.py`'s module docstring),
so nothing here reimplements candidate generation, collision geometry, or
drivable-area checks -- it only reshapes `teach()`'s output to Alpamayo's
trajectory contract and records it the way `label.py` records Drive-JEPA's.

Paddock's `Trajectory` convention (ego-frame, x forward, y left, metres,
evenly spaced by `waypoint_dt` starting at `t = waypoint_dt`) is the one
`draw.py` already assumes -- `WAYPOINT_DT = 0.5` -- but NeuroNCAP wants exactly
six waypoints where `draw.POSES == 8`. Per paddock's own convention ("a
seventh is dropped"), truncating is correct, not a workaround; `ALPAMAYO_POSES`
exists so a reviewer can see the six is deliberate rather than assume it is a
bug. Confirm this against the live `alpamayo15_teacher` adapter's declared
`waypoint_dt` before wiring -- it is assumed here, not re-derived from paddock
(paddock and gallops do not import each other).

**What this module does NOT do**, and why that is a real scope boundary, not
an oversight:

* It does not drive a live NeuroNCAP episode. That is a paddock-side `Policy`
  (`alpamayo15_teacher` in the earlier, now-split PR) running in an interpreter
  with this package on its `PYTHONPATH` -- the two repos still do not import
  each other, but a model's own launch environment can put both on the path.
  Wiring that adapter is paddock's half of this split, not this module's.
* It does not port the research repo's Frenet-frame candidate generation
  (`rl_training/expert/{frenet,geometry,go_around,...}.py`). That logic solved
  the same problem `draw.teach()` already solves, with no equivalent test
  coverage (`alpamayo-scripts/docs/debugging-the-expert.md`'s own "verification
  status" section admits the fixed version was never re-run end to end against
  a live NeuroNCAP benchmark). Reusing the tested solver instead of porting the
  untested one is the point of this module.
* It does not fully port the paddock adapter's go-around grace-window /
  encounter-commitment state machine. That logic lived in `policy.py` to paper
  over a planner that could flip swerve direction tick to tick; `TeacherState`
  here only *detects and flags* a direction flip in the recorded label
  (`direction_flipped`), rather than forcing commitment, because forcing a
  side `teach()` did not itself choose would mean reaching into its internal
  attempt list, which is not exposed and should not be, since every attempt it
  does choose is independently graded clear. If flips show up often in real
  collected data, the fix belongs in `draw.teach()`'s direction-ordering (so
  every caller benefits), not in a paddock- or Alpamayo-specific override.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .draw import Plan, teach
from .grader import Verdict

#: NeuroNCAP's own waypoint count (CLAUDE.md: "a seventh is dropped, a fifth
#: raises IndexError"); `draw.POSES` (8) is Drive-JEPA's own horizon.
ALPAMAYO_POSES = 6
#: Shared with `draw.WAYPOINT_DT` -- verify against the live adapter's
#: `model.yaml` before relying on it; see the module docstring.
ALPAMAYO_WAYPOINT_DT = 0.5


@dataclass(frozen=True)
class AlpamayoPlan:
    """`draw.Plan`, reshaped to Alpamayo's six-waypoint contract."""

    waypoints: np.ndarray  # (ALPAMAYO_POSES, 3): x, y, heading
    verdict: Verdict
    kind: str
    offset_m: float
    progress_m: float
    drivable: bool

    @property
    def clear(self) -> bool:
        return self.verdict.clear


def _truncate(plan: Plan) -> AlpamayoPlan:
    if len(plan.waypoints) < ALPAMAYO_POSES:
        raise ValueError(
            f"draw.teach() returned {len(plan.waypoints)} waypoints, fewer than "
            f"ALPAMAYO_POSES={ALPAMAYO_POSES} -- draw.POSES must have changed"
        )
    return AlpamayoPlan(
        waypoints=np.asarray(plan.waypoints)[:ALPAMAYO_POSES],
        verdict=plan.verdict,
        kind=plan.kind,
        offset_m=plan.offset_m,
        progress_m=plan.progress_m,
        drivable=plan.drivable,
    )


@dataclass
class TeacherState:
    """Carried across ticks of one episode, to flag (not force) a direction flip.

    See the module docstring for why this flags rather than commits.
    """

    last_kind: str | None = None
    last_offset_sign: float = 0.0
    history: list[tuple[str, float]] = field(default_factory=list)

    def update(self, plan: AlpamayoPlan) -> bool:
        """Record this tick's decision; return whether its swerve direction
        flipped from the last tick that was also swerving."""
        sign = float(np.sign(plan.offset_m))
        flipped = (
            self.last_kind in ("swerve", "swerve+brake")
            and plan.kind in ("swerve", "swerve+brake")
            and sign != 0.0
            and self.last_offset_sign != 0.0
            and sign != self.last_offset_sign
        )
        self.history.append((plan.kind, sign))
        self.last_kind = plan.kind
        if sign != 0.0:
            self.last_offset_sign = sign
        return flipped


def teach_alpamayo(
    ego2world: np.ndarray,
    t0: int,
    speed: float,
    reference_world: np.ndarray,
    targets: list[dict],
    area=None,
    state: TeacherState | None = None,
) -> tuple[AlpamayoPlan, bool]:
    """`draw.teach()`, truncated to six waypoints. Returns (plan, direction_flipped).

    `direction_flipped` is False when `state` is None (nothing to compare against).
    """
    plan = _truncate(teach(ego2world, t0, speed, reference_world, targets, area=area))
    flipped = state.update(plan) if state is not None else False
    return plan, flipped


def label_alpamayo_tick(
    *,
    episode: str,
    timestamp: int,
    ego2world: np.ndarray,
    speed: float,
    reference_world: np.ndarray,
    targets: list[dict],
    frames: dict[str, str] | None,
    area=None,
    state: TeacherState | None = None,
) -> dict:
    """One training example: the privileged plan at this tick, and what produced it.

    Unlike `label.py`'s `label_episode`, there is no separate model whose 32
    candidates get graded here -- the teacher is the thing driving collection,
    so this only records the expert's own trajectory and the frame paths it
    was planned from (`frames` keyed by camera name, e.g. ``{"CAM_FRONT":
    ".../CAM_FRONT/123.jpg", "CAM_FRONT_ZOOM": "..."}``), matching Alpamayo's
    multi-camera rig instead of Drive-JEPA's single `CAM_FRONT`-only check.
    """
    plan, flipped = teach_alpamayo(
        ego2world,
        t0=timestamp,
        speed=speed,
        reference_world=reference_world,
        targets=targets,
        area=area,
        state=state,
    )
    return {
        "episode": episode,
        "timestamp": timestamp,
        "frames": frames,
        "speed": speed,
        "expert": np.asarray(plan.waypoints).round(4).tolist(),
        "expert_kind": plan.kind,
        "expert_clear": plan.clear,
        "expert_drivable": plan.drivable,
        "expert_offset_m": round(plan.offset_m, 3),
        "direction_flipped": flipped,
    }


def write_episode_labels(records: list[dict], out_path: Path) -> None:
    """One JSONL per episode -- `label.py`'s own convention, held to here too."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
