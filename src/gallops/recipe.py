"""Recipes: the ordered stages that produce one model, as tracked data.

A recipe is `recipes/<name>.toml`. Each `[[stage]]` names a module under
`gallops`, the interpreter it runs in, and its arguments. Arguments are
templates, expanded by `SiteConfig.resolve`, so the recipe says *what* a stage
reads and writes and `gallops.toml` says *where* that is -- the same split as
everything else here.

Why a recipe rather than a shell script: a training result is only reproducible
if the exact command is recorded, and a shell script with paths in it is either
untracked or unportable. A recipe is both tracked and portable, and `gallops
recipe <name>` prints the commands this machine would run before running any.

Stages run as `python -m gallops.<module>` with this repo's `src/` on
PYTHONPATH, so a stage can run inside a model's own interpreter (Drive-JEPA's is
Python 3.9) without gallops being installed there.
"""

from __future__ import annotations

import os
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: Drive-JEPA's interpreter is 3.9 (see config.py)
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path

from gallops.config import REPO_ROOT, ConfigError, SiteConfig

RECIPES = REPO_ROOT / "recipes"


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    python: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    #: Stages whose outputs this one reads. Checked, not inferred: a stage that
    #: names an unknown or later stage is an error in the recipe.
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    stages: tuple[Stage, ...]
    #: Recipe-level templates (`{rerank}`, `{out}`), expanded against the site
    #: config first and then available to every stage.
    variables: dict[str, str] = field(default_factory=dict)

    def stage(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        raise ConfigError(f"recipe {self.name!r} has no stage {name!r}; stages: {', '.join(self.names())}")

    def names(self) -> list[str]:
        return [s.name for s in self.stages]


@dataclass(frozen=True)
class Command:
    stage: str
    argv: tuple[str, ...]
    cwd: str | None
    env: dict[str, str]

    def shell(self) -> str:
        """One line a person can paste, for review before anything runs."""
        env = " ".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        where = f"cd {self.cwd} && " if self.cwd else ""
        return f"{where}{env + ' ' if env else ''}{' '.join(self.argv)}"


def load(name: str, root: Path = RECIPES) -> Recipe:
    path = root / f"{name}.toml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in root.glob("*.toml"))) or "none"
        raise ConfigError(f"no recipe {path}; known: {known}")
    raw = tomllib.loads(path.read_text())
    stages = tuple(
        Stage(
            name=s["name"],
            module=s["module"],
            python=s["python"],
            args=tuple(str(a) for a in s.get("args", ())),
            cwd=s.get("cwd"),
            env={k: str(v) for k, v in (s.get("env") or {}).items()},
            after=tuple(s.get("after", ())),
        )
        for s in raw.get("stage", ())
    )
    recipe = Recipe(
        name=name, description=raw.get("description", ""), stages=stages, variables=raw.get("vars", {})
    )
    seen: set[str] = set()
    for s in stages:
        if s.name in seen:
            raise ConfigError(f"recipe {name!r}: stage {s.name!r} appears twice")
        unknown = [d for d in s.after if d not in seen]
        if unknown:
            raise ConfigError(
                f"recipe {name!r}: stage {s.name!r} runs after {unknown}, which are not earlier stages"
            )
        seen.add(s.name)
    return recipe


def commands(recipe: Recipe, cfg: SiteConfig, only: list[str] | None = None) -> list[Command]:
    """The commands this site would run, in recipe order. Nothing is executed."""
    context: dict[str, str | Path] = {"repo": REPO_ROOT}
    for key, value in recipe.variables.items():
        context[key] = cfg.resolve(value, **context)
    wanted = set(only) if only else None
    if wanted:
        for name in wanted:
            recipe.stage(name)  # raises on an unknown stage name
    out = []
    for s in recipe.stages:
        if wanted is not None and s.name not in wanted:
            continue
        python = cfg.resolve(f"{{python.{s.python}}}", **context)
        argv = (python, "-m", f"gallops.{s.module}", *(cfg.resolve(a, **context) for a in s.args))
        env = {k: cfg.resolve(v, **context) for k, v in s.env.items()}
        cwd = cfg.resolve(s.cwd, **context) if s.cwd else None
        out.append(Command(stage=s.name, argv=argv, cwd=cwd, env=env))
    return out


def run(command: Command) -> int:
    """Run one stage with this repo's src/ first on PYTHONPATH. Returns its exit code."""
    env = {**os.environ, **command.env}
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(command.argv, cwd=command.cwd, env=env).returncode
