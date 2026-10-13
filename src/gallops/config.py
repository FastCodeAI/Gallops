"""Site configuration: the only place that knows where anything lives.

This is paddock's invariant, adopted deliberately rather than by imitation:

    tracked in git          what a thing IS       docs, code, dataset names
    site-local (gitignored) where it LIVES        gallops.toml

Tracked files never contain a real path. They refer to locations through
template variables which this module expands:

    {weights}          roots.weights   -- where checkpoints are written
    {runs}             roots.runs      -- logs, samples, anything large
    {site}             the site label
    {data.NAME}        the dataset's path, or its first existing tier
    {data.NAME.tiers}  every tier, colon-joined

`tests/test_portability.py` enforces the split by failing the build if a tracked
file names a real machine root. When it fails, the fix is nearly always a
template variable, not an exemption.

**Point `[roots.weights]` at the same directory as paddock's `[roots.weights]`.**
That is not a coincidence to be tidied away later: it is what lets paddock find
a checkpoint this repo wrote without either side learning the other's layout.
"""

from __future__ import annotations

import os
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9, the model envs
    # gallops itself is 3.11, but a trainer runs inside the model's own
    # interpreter, and Drive-JEPA's is 3.9. tomli is the same parser under the
    # name it had before it entered the standard library.
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_TEMPLATE = re.compile(r"\{([a-zA-Z_][\w.]*)\}")

#: Where the site config lives unless GALLOPS_CONFIG says otherwise. One
#: checkout can serve several layouts -- a laptop and a compute box -- without
#: editing a tracked file.
DEFAULT_CONFIG = REPO_ROOT / "gallops.toml"


class ConfigError(Exception):
    """Site configuration is missing or wrong. Always actionable by the reader."""


@dataclass(frozen=True)
class Dataset:
    """Where one dataset lives here.

    Either a single `path`, or `tiers` -- a search path, fastest first. Tiering
    is not a workaround: an NVMe working set can front an HDD or NAS archive so
    the hot subset is fast without duplicating hundreds of gigabytes.

    **Tiers are not necessarily nested.** Two tiers can hold different scenes
    rather than one being a subset of the other, which is true of the KITScenes
    copies on this workspace today. Training over "the dataset" when the tiers
    disagree trains over whichever tier answered first -- so a training run must
    record the scene list it actually read, not the tier it read from.
    """

    name: str
    path: Path | None = None
    tiers: tuple[Path, ...] = ()
    version: str | None = None
    extra: dict[str, Path] = field(default_factory=dict)

    @property
    def roots(self) -> tuple[Path, ...]:
        if self.tiers:
            return self.tiers
        return (self.path,) if self.path else ()

    def resolve(self) -> Path:
        """The first tier that exists. Raises if none does."""
        for root in self.roots:
            if root.exists():
                return root
        raise ConfigError(
            f"dataset {self.name!r} has no existing root; tried: "
            f"{', '.join(str(r) for r in self.roots) or 'nothing configured'}"
        )

    def find(self, relative: str | Path) -> Path:
        """Locate one entry across tiers, fastest first."""
        for root in self.roots:
            candidate = root / relative
            if candidate.exists():
                return candidate
        raise ConfigError(f"{relative} not found in any tier of dataset {self.name!r}")


@dataclass(frozen=True)
class SiteConfig:
    site: str
    weights: Path
    runs: Path
    datasets: dict[str, Dataset] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> SiteConfig:
        path = Path(path or os.environ.get("GALLOPS_CONFIG") or DEFAULT_CONFIG)
        if not path.exists():
            raise ConfigError(
                f"no site config at {path}\n  cp gallops.toml.example gallops.toml && $EDITOR gallops.toml"
            )
        raw = tomllib.loads(path.read_text())
        roots = raw.get("roots", {})
        missing = [k for k in ("weights", "runs") if k not in roots]
        if missing:
            raise ConfigError(f"{path} is missing [roots] key(s): {', '.join(missing)}")

        reserved = {"path", "tiers", "version"}
        datasets = {}
        for name, body in (raw.get("datasets") or {}).items():
            datasets[name] = Dataset(
                name=name,
                path=_abs(body["path"]) if "path" in body else None,
                tiers=tuple(_abs(t) for t in body.get("tiers", [])),
                version=body.get("version"),
                extra={k: _abs(v) for k, v in body.items() if k not in reserved and isinstance(v, str)},
            )

        return cls(
            site=raw.get("site", "unnamed-site"),
            weights=_abs(roots["weights"]),
            runs=_abs(roots["runs"]),
            datasets=datasets,
            source=path,
        )

    # -- template expansion -----------------------------------------------

    def resolve(self, template: str, **context: str | Path) -> str:
        """Expand template variables in a string from a tracked file.

        Unknown variables raise rather than surviving into a path, where they
        would surface much later as a confusing file-not-found.
        """

        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in context:
                return str(context[key])
            if key in ("weights", "runs", "site"):
                return str(getattr(self, key))
            if key.startswith("data."):
                return self._resolve_data(key)
            raise ConfigError(
                f"unknown template variable {{{key}}} in {template!r}; "
                f"known: weights, runs, site, data.<name>, data.<name>.tiers, "
                f"plus context {sorted(context)}"
            )

        return _TEMPLATE.sub(substitute, template)

    def _resolve_data(self, key: str) -> str:
        parts = key.split(".")
        name = parts[1]
        dataset = self.datasets.get(name)
        if dataset is None:
            known = ", ".join(sorted(self.datasets)) or "none configured"
            raise ConfigError(
                f"dataset {name!r} is not configured in {self.source}\n"
                f"  add a [datasets.{name}] section (see docs/data.md); known: {known}"
            )
        if len(parts) == 3 and parts[2] == "tiers":
            return ":".join(str(r) for r in dataset.roots)
        if len(parts) == 3:
            extra = dataset.extra.get(parts[2])
            if extra is None:
                raise ConfigError(
                    f"dataset {name!r} has no key {parts[2]!r} in {self.source} "
                    f"(has: {', '.join(sorted(dataset.extra)) or 'none'})"
                )
            return str(extra)
        return str(dataset.resolve())

    def run_dir(self, model: str, train_run: str) -> Path:
        """Where one training run's checkpoints go.

        Each name must be one path component: an absolute or nested `train_run`
        would otherwise land outside `{weights}`, or a level deeper than paddock
        looks.
        """
        for label, name in (("model", model), ("train_run", train_run)):
            if name in ("", ".", "..") or Path(name).name != name:
                raise ValueError(f"{label} must be a single directory name, got {name!r}")
        return self.weights / model / train_run

    def checkpoint_dir(self, model: str, train_run: str, step: int) -> Path:
        """Where one checkpoint goes.

        The layout is part of the contract with paddock -- see
        docs/checkpoints.md -- so it lives here rather than being spelled out at
        each call site, where the two would eventually drift.
        """
        return self.run_dir(model, train_run) / f"step_{step:09d}"


def _abs(value: str | Path) -> Path:
    return Path(value).expanduser()
