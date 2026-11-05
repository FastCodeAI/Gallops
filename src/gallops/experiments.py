"""The experiment registry: one `experiment.toml` per experiment, and the README rendered from them.

Gallops exists to produce evidence that a model is better, and evidence that lives only in a
hand-written table drifts: a number is copied, the run behind it is retrained, and nobody can say
which checkpoint the row was about. So each experiment carries its facts in one machine-readable
file -- who ran it, what it started from, what it trained on, which checkpoint it wrote (by weights
sha256), what it measured and how strongly -- and the README's tables, the experiments index and
each card are *rendered* from those files. `tests/test_experiments.py` fails when the README differs from
what this module renders, so the README cannot say something the registry does not.

Layout (docs/experiments.md has the rules and the reasons):

    experiments/
        people.toml          who can own an experiment
        decisions.toml       the decision log, D01, D02, ... in date order
        <ID>_<slug>/
            experiment.toml  this module reads it
            README.md        the narrative
            PREDICTIONS.md   hypothesis experiments: written before the run

Stdlib only, so `make check` runs it on a machine with no GPU. No trainer imports it, but the
training preflight runs the whole test suite under the trainer's interpreter (train/preflight.py,
`code_checks`), and Drive-JEPA's is Python 3.9: so it keeps to 3.9, as config.py does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import posixpath
import re
import textwrap

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9, the model envs; see config.py
    import tomli as tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bump when a field changes meaning; adding an optional field does not need it.
SCHEMA = 1

#: Each experiment's one-page card, rendered whole into its folder; the README's IDs link to it.
CARD = "CARD.md"

#: The README's one picture, rendered whole from the registry.
PICTURE = "docs/assets/experiments.svg"

#: The index page: the tables too long for the README's front page.
INDEX = "experiments/README.md"

#: Where a record on an unmerged branch is linked. A URL, not a machine path.
GITHUB = "https://github.com/FastCodeAI/Gallops"

ID_RE = re.compile(r"^(GC|H)\d{2}$")
FOLDER_RE = re.compile(r"^((GC|H)\d{2})_[a-z0-9]+(_[a-z0-9]+)*$")
RUN_RE = re.compile(r"^((GC|H)\d{2})\.[A-Za-z0-9][A-Za-z0-9_-]*$")
DECISION_RE = re.compile(r"^D\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: What an empty file, or an ensemble of nothing, hashes to. It is valid hex and pins no weights.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
#: One file of a checkpoint, as `sha256sum` prints it: digest, two spaces, name.
MEMBER_RE = re.compile(r"^[0-9a-f]{64}  \S+$")
#: The picture's base box: how wide a line of a baseline's `short` can run, and how many fit.
BASE_WIDTH, BASE_LINES = 22, 2

TRACKS = ("gap-closing", "hypothesis", "enabler")
STATUSES = ("planned", "running", "done", "parked", "dropped")
VERDICTS = ("positive", "negative", "mixed", "inconclusive", "none")
#: Paddock's ladder (docs/conventions.md). One vocabulary, so a reader compares like with like.
STRENGTHS = ("proof", "strong", "moderate", "argued", "unverified")
FLAGS = ("adapted", "blind", "smoke", "selected-on-report", "open-loop", "no-record")
BENCHMARKS = {
    "kitscenes": "KITScenes",
    "nuscenes": "nuScenes",
    "neuroncap": "neuroNCAP",
    "navsim": "NAVSIM",
    "bench2drive": "Bench2Drive",
    "alpasim": "AlpaSim",
    "hugsim": "HUGSIM",
    "comma2k19": "comma2k19",
    "nuplan": "nuPlan",
    "offline": "offline",
}
#: The gate log's columns, in the order paddock's gate tiers list them.
GATE_BENCHES = ("navsim", "kitscenes", "nuscenes", "neuroncap", "bench2drive", "alpasim")
STEP_UNITS = ("epoch", "pass", "optimizer_step")
DATA_ROLES = ("train", "selection", "held-out", "eval")
GATE_TIERS = ("quick", "closed", "all")
#: What paddock's `overall()` returns. An unfinished subset is a HOLD that says so, not a verdict of its own.
GATE_VERDICTS = ("SHIP", "HOLD", "FAIL")
#: Paddock's nine, then `NOT RUN`, which is ours: a benchmark the job's tier left out.
PAIR_VERDICTS = (
    "GAIN",
    "NEUTRAL",
    "REGRESSION",
    "NO RECORDS",
    "TOO FEW UNITS",
    "NO UNIT",
    "CONFIG DIFFERS",
    "INCOMPLETE",
    "SUBSET DIFFERS",
    "NOT RUN",
)

#: Modules that serve every experiment: the contract, the site config, the CLI. Every other module
#: under src/gallops must be claimed by an experiment's `[runs.code]`, so no trainer exists in the
#: repo without the evidence it produced.
SHARED_MODULES = (
    "gallops.checkpoint",
    "gallops.cli",
    "gallops.config",
    "gallops.experiments",
    "gallops.recipe",
)

NUMBER = (int, float, str)  # a str number keeps the precision it was reported at ("87.30")

_TOP: dict[str, tuple[Any, bool]] = {
    "schema": (int, True),
    "id": (str, True),
    "title": (str, True),
    "short": (str, True),
    "track": (str, True),
    "owner": (str, True),
    "contributors": (list, False),
    "benchmarks": (list, True),
    "started": (dt.date, True),
    "ended": (dt.date, False),
    "status": (str, True),
    "verdict": (str, True),
    "priority": (int, False),
    "question": (str, True),
    "hypothesis": (str, False),
    "predictions": (str, False),
    "learned": (str, False),
    "parents": (list, False),
    "narrative": (str, False),
    "branch": (str, False),
    "external": (list, False),
    "runs": (list, False),
    "checks": (list, False),
}
_RUN: dict[str, tuple[Any, bool]] = {
    "id": (str, True),
    "label": (str, True),
    "started": (dt.date, False),
    "ended": (dt.date, False),
    "start_from": (str, False),
    "trained": (list, False),
    "frozen": (list, False),
    "params_trained_m": ((int, float), False),
    "train_git_sha": (str, False),
    "gpu_hours": ((int, float), False),
    "notes": (str, False),
    "code": (dict, False),
    "data": (list, False),
    "checkpoint": (dict, False),
    "results": (list, False),
    "gate": (list, False),
}
_CODE: dict[str, tuple[Any, bool]] = {
    "entry": (str, False),
    "recipe": (str, False),
    "modules": (list, False),
    "as_run": (list, False),
    "upstream": (dict, False),
}
_UPSTREAM: dict[str, tuple[Any, bool]] = {
    "repo": (str, True),
    "commit": (str, True),
    "patches": (list, False),
    "overlay": (str, False),
}
_DATA: dict[str, tuple[Any, bool]] = {
    "dataset": (str, True),
    "split": (str, True),
    "role": (str, True),
    "n": (int, False),
    "unit": (str, False),
    "scene_list": (str, False),
    "built_by": (str, False),
    "manifest_sha256": (str, False),
    "note": (str, False),
}
_CHECKPOINT: dict[str, tuple[Any, bool]] = {
    "model": (str, True),
    "train_run": (str, False),
    "step": (int, False),
    "step_unit": (str, False),
    "weights_sha256": (str, False),
    "file": (str, False),
    "location": (str, False),
    "sidecar": (bool, False),
    "known_wrong": (str, False),
    "shipped_as": (str, False),
    "members": (list, False),
}
_RESULT: dict[str, tuple[Any, bool]] = {
    "bench": (str, True),
    "metric": (str, True),
    "value": (NUMBER, True),
    "better": (str, True),
    "strength": (str, True),
    "baseline": (str, False),
    "baseline_value": (NUMBER, False),
    "n": (int, False),
    "unit": (str, False),
    "subset": (str, False),
    "se": (NUMBER, False),
    "t": (NUMBER, False),
    "flags": (list, False),
    "source": (str, False),
    "headline": (bool, False),
    "note": (str, False),
}
_GATE: dict[str, tuple[Any, bool]] = {
    "date": (dt.date, True),
    "tier": (str, True),
    "verdict": (str, True),
    "baseline": (str, False),
    "paddock_commit": (str, False),
    "manifest": (str, False),
    "pairs": (list, False),
    "note": (str, False),
}
_PAIR: dict[str, tuple[Any, bool]] = {
    "bench": (str, True),
    "verdict": (str, True),
    "metric": (str, False),
    "baseline": (NUMBER, False),
    "candidate": (NUMBER, False),
    "t": (NUMBER, False),
    "mde": (NUMBER, False),
    "units": (int, False),
}
_CHECK: dict[str, tuple[Any, bool]] = {
    "name": (str, True),
    "result": (str, True),
    "date": (dt.date, False),
    "decided": (str, False),
    "source": (str, False),
}
_DECISION: dict[str, tuple[Any, bool]] = {
    "id": (str, True),
    "date": (dt.date, True),
    "short": (str, True),
    "decision": (str, True),
    "by": (list, True),
    "evidence": (list, False),
    "next": (list, False),
    "baseline": (str, False),
    "version": (str, False),
    "source": (str, False),
}
_PERSON: dict[str, tuple[Any, bool]] = {
    "name": (str, True),
    "git": (list, False),
    "focus": (str, False),
}


class RegistryError(Exception):
    """The registry cannot be read at all (a file is not TOML). Content problems are collected instead."""


@dataclass
class Experiment:
    folder: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return self.data.get("id", "")

    @property
    def runs(self) -> list[dict[str, Any]]:
        return self.data.get("runs", [])

    def results(self) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        for run in self.runs:
            for result in run.get("results", []):
                yield run, result

    def headline(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for run, result in self.results():
            if result.get("headline"):
                return run, result
        return None


@dataclass
class Registry:
    root: Path
    experiments: list[Experiment]
    decisions: list[dict[str, Any]]
    people: dict[str, dict[str, Any]]
    errors: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path = REPO_ROOT) -> Registry:
        base = root / "experiments"
        experiments = []
        for path in sorted(base.glob("*/experiment.toml")):
            if path.parent.name.startswith("_"):  # _template
                continue
            experiments.append(Experiment(path.parent, _toml(path)))
        experiments.sort(key=_order)
        decisions = _toml(base / "decisions.toml").get("decision", [])
        people = _toml(base / "people.toml").get("people", {})
        registry = cls(root, experiments, decisions, people)
        registry.errors = validate(registry)
        return registry

    def by_id(self) -> dict[str, Experiment]:
        return {e.id: e for e in self.experiments}

    def run_ids(self) -> dict[str, tuple[Experiment, dict[str, Any]]]:
        return {run.get("id", ""): (e, run) for e in self.experiments for run in e.runs}


def _toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RegistryError(f"{path} is missing") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"{path} is not valid TOML: {exc}") from exc


def _order(e: Experiment) -> tuple[int, int]:
    """GC before H, then by number: the order the work was registered in, which IDs never change."""
    m = ID_RE.match(e.id)
    if not m:
        return (9, 0)
    return (0 if m.group(1) == "GC" else 1, int(e.id[len(m.group(1)) :]))


# ---------------------------------------------------------------- validation


def _fields(where: str, obj: Any, spec: dict[str, tuple[Any, bool]], errors: list[str]) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected a table, got {type(obj).__name__}")
        return False
    for key in obj:
        if key not in spec:
            errors.append(f"{where}: unknown field {key!r}")
    for key, (kind, required) in spec.items():
        if key not in obj:
            if required:
                errors.append(f"{where}: {key!r} is required")
            continue
        value = obj[key]
        # bool is an int in Python; a field that wants a number must not accept true.
        if isinstance(value, bool) and kind is not bool:
            errors.append(f"{where}: {key!r} is a bool")
        elif not isinstance(value, kind):
            errors.append(f"{where}: {key!r} has the wrong type ({type(value).__name__})")
        elif kind == NUMBER and isinstance(value, str) and not _is_number(value):
            errors.append(f"{where}: {key!r} = {value!r} is not a number")
    return True


def _is_number(text: str) -> bool:
    try:
        float(text.removesuffix("%"))
    except ValueError:
        return False
    return True


def _enum(where: str, value: Any, allowed: Iterable[str], errors: list[str]) -> None:
    allowed = tuple(allowed)
    if value is not None and value not in allowed:
        errors.append(f"{where}: {value!r} is not one of {', '.join(allowed)}")


def member(sha: str, name: str) -> str:
    return f"{sha}  {name}"


def ensemble_digest(members: list[str]) -> str:
    """The digest of a checkpoint that is several files (an ensemble of seeds): sha256 over its members'
    `sha256  name` lines, sorted by name. `sha256sum seed_*.pt | sha256sum` reproduces it."""
    lines = sorted(members, key=lambda m: m.split("  ", 1)[1])
    return hashlib.sha256("".join(line + "\n" for line in lines).encode()).hexdigest()


def _members_agree(ck: dict[str, Any]) -> str | None:
    """Why a checkpoint's `weights_sha256` disagrees with its `members`, or None.

    With `file` naming a member, the digest is that file's: the one paddock loads, the others kept as
    second digests. Without it, the checkpoint is the members together, and the digest is theirs.
    """
    members = ck.get("members", [])
    bad = [m for m in members if not isinstance(m, str) or not MEMBER_RE.match(m)]
    if bad:
        return f"members are `sha256  name` lines, as sha256sum prints them: {bad[0]!r}"
    wsha = ck.get("weights_sha256")
    if not members or not wsha:
        return None
    by_name = {m.split("  ", 1)[1]: m.split("  ", 1)[0] for m in members}
    if ck.get("file") in by_name:
        if wsha != by_name[ck["file"]]:
            return f"weights_sha256 is not the digest members give {ck['file']}"
    elif wsha != ensemble_digest(members):
        return "weights_sha256 is not the digest over members, and `file` names none of them"
    return None


def _module_exists(root: Path, dotted: str) -> bool:
    if not dotted.startswith("gallops"):
        return False
    path = root / "src" / Path(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _is_module(ref: str) -> bool:
    return "/" not in ref and not ref.endswith((".py", ".sh", ".md", ".toml", ".json", ".diff"))


def _path_exists(root: Path, ref: str) -> bool:
    return (root / ref.split("#", 1)[0]).exists()


def _recipe(root: Path, ref: str) -> tuple[Path, str]:
    path, _, stage = ref.partition("#")
    return root / path, stage


def recipe_modules(root: Path, ref: str) -> list[str]:
    """The modules a recipe reference runs: one stage's after '#', else every stage's.

    A stage names its module relative to the package (`drivejepa_b2d.train_split_head`) and runs as
    `python -m gallops.<module>` (gallops.recipe), so the claim is the dotted name under `gallops.`.
    """
    path, stage = _recipe(root, ref)
    stages = tomllib.loads(path.read_text()).get("stage", [])
    return [
        f"gallops.{s['module']}" for s in stages if "module" in s and (not stage or s.get("name") == stage)
    ]


def validate(reg: Registry) -> list[str]:
    errors: list[str] = []
    root = reg.root

    for key, person in reg.people.items():
        _fields(f"people.toml [{key}]", person, _PERSON, errors)
    owners = set(reg.people)

    seen_ids: dict[str, Path] = {}
    seen_runs: set[str] = set()
    seen_sha: dict[str, str] = {}
    for e in reg.experiments:
        rel = e.folder.relative_to(root)
        where = f"{rel}/experiment.toml"
        d = e.data
        _fields(where, d, _TOP, errors)
        if d.get("schema") not in (None, SCHEMA):
            errors.append(f"{where}: schema {d.get('schema')} is not {SCHEMA}")
        eid = e.id
        if not ID_RE.match(eid):
            errors.append(f"{where}: id {eid!r} is not GCnn or Hnn")
        m = FOLDER_RE.match(e.folder.name)
        if not m or m.group(1) != eid:
            errors.append(f"{where}: folder must be named {eid}_<slug> (lower-case, underscores)")
        if eid in seen_ids:
            errors.append(f"{where}: id {eid} is also used by {seen_ids[eid]}")
        seen_ids[eid] = rel
        if eid.startswith("GC") and d.get("track") != "gap-closing":
            errors.append(f"{where}: a GC id is a gap-closing experiment")
        if eid.startswith("H") and d.get("track") == "gap-closing":
            errors.append(f"{where}: an H id is a hypothesis or an enabler, not gap closing")
        _enum(where, d.get("track"), TRACKS, errors)
        _enum(where, d.get("status"), STATUSES, errors)
        _enum(where, d.get("verdict"), VERDICTS, errors)
        for bench in d.get("benchmarks", []):
            _enum(f"{where} benchmarks", bench, BENCHMARKS, errors)
        for person in [d.get("owner"), *d.get("contributors", [])]:
            if person is not None and person not in owners:
                errors.append(f"{where}: {person!r} is not in experiments/people.toml")
        if d.get("ended") and d.get("started") and d["ended"] < d["started"]:
            errors.append(f"{where}: ended before it started")
        if d.get("status") in ("done", "parked", "dropped") and not d.get("learned"):
            errors.append(f"{where}: a finished experiment says what it `learned`")
        if d.get("status") == "planned" and d.get("verdict") != "none":
            errors.append(f"{where}: a planned experiment has no verdict yet")
        if d.get("track") == "hypothesis" and not d.get("hypothesis"):
            errors.append(f"{where}: a hypothesis experiment states its `hypothesis`")
        if len(d.get("short", "")) > 40:
            errors.append(f"{where}: `short` is for table cells; keep it to 40 characters")

        branch = d.get("branch", "")
        on_branch = bool(branch)
        refs: list[tuple[str, str]] = []  # (what, path) checked against the working tree
        if d.get("narrative"):
            refs.append(("narrative", d["narrative"]))
        has_results = any(True for _ in e.results())
        if d.get("track") == "hypothesis" and has_results:
            pred = d.get("predictions", "")
            if not pred:
                errors.append(
                    f"{where}: a hypothesis with results names its `predictions` file, or says "
                    f"'none: <why>' -- a prediction written after the result is not one"
                )
            elif not pred.startswith("none:"):
                refs.append(("predictions", pred))

        headlines = 0
        for i, run in enumerate(e.runs):
            rwhere = f"{where} runs[{i}]"
            _fields(rwhere, run, _RUN, errors)
            rid = run.get("id", "")
            rm = RUN_RE.match(rid)
            if not rm or rm.group(1) != eid:
                errors.append(f"{rwhere}: run id {rid!r} must be {eid}.<label>")
            if rid in seen_runs:
                errors.append(f"{rwhere}: run id {rid} is used twice")
            seen_runs.add(rid)
            sha = run.get("train_git_sha")
            if sha is not None and not GIT_SHA_RE.match(sha):
                errors.append(f"{rwhere}: train_git_sha {sha!r} is not a git sha")

            code = run.get("code")
            if code is not None and _fields(f"{rwhere} code", code, _CODE, errors):
                entry = code.get("entry")
                if entry and not _is_module(entry):
                    refs.append(("code.entry", entry))
                for ref in code.get("as_run", []):
                    refs.append(("code.as_run", ref))
                if "recipe" in code:
                    refs.append(("code.recipe", code["recipe"]))
                up = code.get("upstream")
                if up is not None and _fields(f"{rwhere} code.upstream", up, _UPSTREAM, errors):
                    for ref in up.get("patches", []):
                        refs.append(("code.upstream.patches", ref))
                    if up.get("overlay"):
                        refs.append(("code.upstream.overlay", up["overlay"]))
                if not on_branch:
                    mods = list(code.get("modules", []))
                    if entry and _is_module(entry):
                        mods.append(entry)
                    for mod in mods:
                        if not _module_exists(root, mod):
                            errors.append(f"{rwhere} code: module {mod} does not exist under src/")

            for j, data in enumerate(run.get("data", [])):
                dwhere = f"{rwhere} data[{j}]"
                if _fields(dwhere, data, _DATA, errors):
                    _enum(dwhere, data.get("role"), DATA_ROLES, errors)
                    if data.get("scene_list"):
                        refs.append(("data.scene_list", data["scene_list"]))
                    built = data.get("built_by")
                    if built and not _is_module(built):
                        refs.append(("data.built_by", built))
                    elif built and not on_branch and not _module_exists(root, built):
                        errors.append(f"{dwhere}: built_by module {built} does not exist under src/")
                    msha = data.get("manifest_sha256")
                    if msha and not SHA256_RE.match(msha):
                        errors.append(f"{dwhere}: manifest_sha256 is not 64 lower-case hex digits")

            ck = run.get("checkpoint")
            if ck is not None and _fields(f"{rwhere} checkpoint", ck, _CHECKPOINT, errors):
                _enum(f"{rwhere} checkpoint", ck.get("step_unit"), STEP_UNITS, errors)
                why = _members_agree(ck)
                if why:
                    errors.append(f"{rwhere} checkpoint: {why}")
                wsha = ck.get("weights_sha256")
                if wsha is not None:
                    if not SHA256_RE.match(wsha):
                        errors.append(f"{rwhere} checkpoint: weights_sha256 is not 64 lower-case hex digits")
                    elif wsha == EMPTY_SHA256:
                        errors.append(f"{rwhere} checkpoint: weights_sha256 is the SHA-256 of empty input")
                    elif run.get("trained") != []:
                        # `trained = []` pins weights the run did not write: paddock's gate accepts a code
                        # change on the same weights. Two runs that trained cannot have written one file.
                        if wsha in seen_sha:
                            errors.append(
                                f"{rwhere} checkpoint: {wsha[:12]} is also claimed by {seen_sha[wsha]}"
                            )
                        else:
                            seen_sha[wsha] = rid

            for j, res in enumerate(run.get("results", [])):
                xwhere = f"{rwhere} results[{j}]"
                if _fields(xwhere, res, _RESULT, errors):
                    _enum(xwhere, res.get("bench"), BENCHMARKS, errors)
                    _enum(xwhere, res.get("better"), ("higher", "lower"), errors)
                    _enum(xwhere, res.get("strength"), STRENGTHS, errors)
                    for flag in res.get("flags", []):
                        _enum(f"{xwhere} flags", flag, FLAGS, errors)
                    headlines += bool(res.get("headline"))

            if run.get("gate") and not (ck or {}).get("weights_sha256"):
                # A verdict about weights nobody can identify is not evidence.
                errors.append(f"{rwhere}: a gated run pins its checkpoint's weights_sha256")
            for j, gate in enumerate(run.get("gate", [])):
                gwhere = f"{rwhere} gate[{j}]"
                if _fields(gwhere, gate, _GATE, errors):
                    _enum(gwhere, gate.get("tier"), GATE_TIERS, errors)
                    _enum(gwhere, gate.get("verdict"), GATE_VERDICTS, errors)
                    for k, pair in enumerate(gate.get("pairs", [])):
                        pwhere = f"{gwhere} pairs[{k}]"
                        if _fields(pwhere, pair, _PAIR, errors):
                            _enum(pwhere, pair.get("bench"), GATE_BENCHES, errors)
                            _enum(pwhere, pair.get("verdict"), PAIR_VERDICTS, errors)

        if has_results and headlines != 1:
            errors.append(f"{where}: exactly one result is the `headline` (found {headlines})")
        for j, check in enumerate(d.get("checks", [])):
            _fields(f"{where} checks[{j}]", check, _CHECK, errors)
        for j, ext in enumerate(d.get("external", [])):
            if not isinstance(ext, str):
                errors.append(f"{where} external[{j}]: a sentence saying where the evidence lives")

        if not on_branch:
            for what, ref in refs:
                if what == "code.recipe":
                    path, stage = _recipe(root, ref)
                    if not path.is_file():
                        errors.append(f"{where}: {what} {ref} does not exist")
                    elif stage and not recipe_modules(root, ref):
                        errors.append(f"{where}: {what} {ref}: no stage {stage!r} with a module")
                elif not _path_exists(root, ref):
                    errors.append(f"{where}: {what} {ref} does not exist")

    ids = set(seen_ids)
    for e in reg.experiments:
        for parent in e.data.get("parents", []):
            if parent not in ids:
                errors.append(f"{e.id}: parent {parent} is not an experiment")
        for run in e.runs:
            start = run.get("start_from", "")
            if RUN_RE.match(start) and start not in seen_runs:
                errors.append(f"{run.get('id')}: start_from {start} is not a run")

    errors.extend(_validate_decisions(reg, ids | seen_runs, owners))
    errors.extend(_undecided(reg))
    errors.extend(_unclaimed_modules(reg))
    return errors


def _undecided(reg: Registry) -> list[str]:
    """A finished hypothesis was judged by the gate, and the team said what follows from that verdict."""
    errors = []
    for e in reg.experiments:
        if e.data.get("track") != "hypothesis" or e.data.get("status") != "done":
            continue
        gates = [gate for run in e.runs for gate in run.get("gate", [])]
        if not gates:
            errors.append(f"{e.id}: a done hypothesis has a gate verdict on one of its runs")
        verdict = e.data.get("verdict")
        if verdict == "none":
            errors.append(f"{e.id}: a done hypothesis says how it came out (`verdict` is not none)")
        if verdict == "positive" and not any(gate.get("verdict") == "SHIP" for gate in gates):
            errors.append(f"{e.id}: a positive hypothesis has a gate SHIP")
        # Any older decision will not do: D11 names H01 as the base, and cannot have said what follows
        # from a gate run after it. The same day counts, since a team decides on the day's verdict.
        last = max((g["date"] for g in gates if isinstance(g.get("date"), dt.date)), default=dt.date.min)
        if not any(_cites(d, e.id) and d["date"] >= last for d in reg.decisions):
            errors.append(
                f"{e.id}: a done hypothesis is cited as evidence by a decision "
                "dated on or after its last gate"
            )
    return errors


def _cites(decision: dict[str, Any], experiment: str) -> bool:
    # A non-string is `_validate_decisions`' to report, with a message; it must not raise here first.
    refs = [ref for ref in decision.get("evidence", []) if isinstance(ref, str)]
    return isinstance(decision.get("date"), dt.date) and experiment in {r.split(".", 1)[0] for r in refs}


def _validate_decisions(reg: Registry, known: set[str], owners: set[str]) -> list[str]:
    errors: list[str] = []
    previous: dt.date | None = None
    for i, d in enumerate(reg.decisions):
        where = f"decisions.toml decision[{i}]"
        if not _fields(where, d, _DECISION, errors):
            continue
        want = f"D{i + 1:02d}"
        if d.get("id") != want:
            # The log is linear: numbered in order, never renumbered, so D07 always means one thing.
            errors.append(f"{where}: id {d.get('id')!r} should be {want} (decisions are numbered in order)")
        date = d.get("date")
        if previous and date and date < previous:
            errors.append(f"{where}: {d.get('id')} is dated before the decision above it")
        previous = date or previous
        for who in d.get("by", []):
            if who != "team" and who not in owners:
                errors.append(f"{where}: by {who!r} is not in people.toml (or 'team')")
        for ref in [*d.get("evidence", []), *d.get("next", [])]:
            if ref not in known:
                errors.append(f"{where}: {ref} is not an experiment or run")
        if d.get("baseline") and len(_base_lines(str(d.get("short", "")))) > BASE_LINES:
            # Refused, not cut: a box that drops words names a different model and nothing fails.
            errors.append(
                f"{where}: a baseline's `short` names it in the picture, which has room for "
                f"{BASE_LINES} lines of {BASE_WIDTH} characters"
            )
    return errors


def all_modules(root: Path) -> list[str]:
    """Every non-package module under src/gallops, dotted."""
    src = root / "src"
    out = []
    for path in sorted((src / "gallops").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        out.append(".".join(path.relative_to(src).with_suffix("").parts))
    return out


def claimed_modules(reg: Registry) -> dict[str, list[str]]:
    """module -> the runs whose `[runs.code]` names it (directly, by package, or through a recipe)."""
    claims: dict[str, list[str]] = {}
    for e in reg.experiments:
        if e.data.get("branch"):
            continue
        for run in e.runs:
            code = run.get("code", {})
            names = list(code.get("modules", []))
            entry = code.get("entry")
            if entry and _is_module(entry):
                names.append(entry)
            if code.get("recipe") and _recipe(reg.root, code["recipe"])[0].is_file():
                names.extend(recipe_modules(reg.root, code["recipe"]))
            for data in run.get("data", []):
                if data.get("built_by") and _is_module(data["built_by"]):
                    names.append(data["built_by"])
            for name in names:
                claims.setdefault(name, []).append(run.get("id", ""))
    return claims


def module_claims(reg: Registry) -> dict[str, list[str]]:
    """Every module under src/gallops -> the runs that used it; package claims cover their modules."""
    claims = claimed_modules(reg)
    out: dict[str, list[str]] = {}
    for module in all_modules(reg.root):
        runs: list[str] = []
        for name, rids in claims.items():
            if module == name or module.startswith(name + "."):
                runs.extend(r for r in rids if r not in runs)
        out[module] = runs
    return out


def _unclaimed_modules(reg: Registry) -> list[str]:
    return [
        f"src: {module} is used by no experiment's [runs.code] and is not in SHARED_MODULES"
        for module, runs in module_claims(reg).items()
        if not runs and module not in SHARED_MODULES
    ]


# ---------------------------------------------------------------- rendering

STRENGTH_DOTS = {
    "proof": "●●●●",
    "strong": "●●●○",
    "moderate": "●●○○",
    "argued": "●○○○",
    "unverified": "○○○○",
}
VERDICT_MARK = {
    "positive": "✅ positive",
    "negative": "❌ negative",
    "mixed": "◐ mixed",
    "inconclusive": "❔ inconclusive",
    "none": "☑️ done",
}
STATUS_MARK = {"planned": "📝 planned", "running": "⏳ running", "parked": "⏸ parked", "dropped": "🗑 dropped"}
PAIR_MARK = {
    "GAIN": "🟢",
    "NEUTRAL": "⚪",
    "REGRESSION": "🔴",
    "NO RECORDS": "▫️",
    "TOO FEW UNITS": "▫️",
    "NO UNIT": "▫️",
    "CONFIG DIFFERS": "▫️",
    "INCOMPLETE": "▫️",
    "SUBSET DIFFERS": "▫️",
    "NOT RUN": "—",
}
GATE_MARK = {"SHIP": "🚀 **SHIP**", "HOLD": "⏸ **HOLD**", "FAIL": "⛔ **FAIL**"}


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _num(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:g}"


def _date(value: dt.date | None) -> str:
    return f"{value.day} {value:%b}" if value else ""


def _span(d: dict[str, Any]) -> str:
    start, end = d.get("started"), d.get("ended")
    if not start:
        return ""
    if not end or end == start:
        return f"{_date(start)} {start.year}"
    if end.year != start.year:
        return f"{_date(start)} {start.year} – {_date(end)} {end.year}"
    return f"{_date(start)} – {_date(end)} {end.year}"


def mark(d: dict[str, Any]) -> str:
    """The one emoji-and-word a table shows for where an experiment stands."""
    if d.get("status") == "done":
        return VERDICT_MARK[d.get("verdict", "none")]
    if d.get("status") == "parked" and d.get("verdict") not in (None, "none"):
        return f"⏸ parked ({d['verdict']})"
    return STATUS_MARK.get(d.get("status", ""), d.get("status", ""))


def person(reg: Registry, key: str) -> str:
    return key if key in reg.people else f"{key}?"


def owners(reg: Registry, d: dict[str, Any]) -> str:
    return ", ".join(person(reg, k) for k in [d.get("owner", ""), *d.get("contributors", [])])


def rel(path: str, base: str) -> str:
    """`path` (repo-root relative, maybe with #fragment) as a link from a file in folder `base`."""
    target, sep, frag = path.partition("#")
    out = posixpath.relpath(target, base or ".") if target else ""
    return out + ("/" if target.endswith("/") and not out.endswith("/") else "") + sep + frag


def link(d: dict[str, Any], path: str | None = None, base: str = "") -> str:
    """A link to a record: relative in the working tree, or to the branch on GitHub while it is unmerged."""
    path = path or d.get("narrative", "")
    if not path:
        return ""
    if d.get("branch"):
        return f"{GITHUB}/blob/{d['branch']}/{path}"
    return rel(path, base)


def _ref_kind(d: dict[str, Any]) -> str:
    """What `branch` names: a branch, or the commit a closed PR keeps once its branch is deleted."""
    return "commit" if GIT_SHA_RE.match(d["branch"]) else "branch"


def record(d: dict[str, Any], base: str = "") -> str:
    if not d.get("narrative"):
        return "—"
    if d.get("branch"):
        return f"[on {_ref_kind(d)} ⎇]({link(d, base=base)})"
    return f"[account]({link(d, base=base)})"


def card_path(reg: Registry, e: Experiment) -> str:
    """Where an experiment's card is rendered, relative to the repo root."""
    return f"{e.folder.relative_to(reg.root).as_posix()}/{CARD}"


def card(reg: Registry, eid: str, base: str = "") -> str:
    """A link to an experiment's card from a file in folder `base`."""
    by_id = reg.by_id()
    return rel(card_path(reg, by_id[eid]), base) if eid in by_id else ""


def benches(d: dict[str, Any]) -> str:
    return ", ".join(BENCHMARKS.get(b, b) for b in d.get("benchmarks", []))


def strength(res: dict[str, Any]) -> str:
    s = res.get("strength", "unverified")
    flags = res.get("flags", [])
    text = f"{STRENGTH_DOTS[s]} {s}"
    return text + (f" · {', '.join(flags)}" if flags else "")


def result_text(res: dict[str, Any], bold: bool = True) -> str:
    value = _num(res["value"])
    value = f"**{value}**" if bold else value
    bench = BENCHMARKS.get(res["bench"], res["bench"])
    head = f"{bench} {res['metric']} "
    if "baseline_value" in res:
        head += f"{_num(res['baseline_value'])} → {value}"
    else:
        head += value
    n = f"n={_num(res['n'])}" + (f" {res['unit']}s" if res.get("unit") else "") if "n" in res else ""
    return head + (f" ({n})" if n else "")


def tag(run: dict[str, Any] | None) -> str:
    sha = ((run or {}).get("checkpoint") or {}).get("weights_sha256")
    return f"`{sha[:12]}`" if sha else "—"


def latest_gate(e: Experiment) -> tuple[dict[str, Any], dict[str, Any]] | None:
    gates = [(run, g) for run in e.runs for g in run.get("gate", [])]
    return max(gates, key=lambda rg: rg[1]["date"]) if gates else None


def render_badges(reg: Registry) -> str:
    counts = {t: sum(e.data.get("track") == t for e in reg.experiments) for t in TRACKS}
    gated = sum(1 for e in reg.experiments for run in e.runs if run.get("gate"))
    versions = [d["version"] for d in reg.decisions if d.get("version")]
    version = versions[-1] if versions else "none_yet"
    badges = [
        ("gap_closing", f"{counts['gap-closing']}_experiments", "2E8B4E", "#gap-closing-experiments"),
        ("hypotheses", f"{counts['hypothesis']}_open_or_tested", "1F6FEB", "#hypothesis-experiments"),
        ("gated_checkpoints", str(gated), "B4560A", f"{INDEX}#regression-gate-log"),
        ("gallops_version", version.replace("-", "--"), "555555", "#where-we-are"),
    ]
    return "\n".join(
        f'  <a href="{href}"><img src="https://img.shields.io/badge/{label}-{msg}-{color}.svg" '
        f'alt="{label.replace("_", " ")}: {msg.replace("_", " ").replace("--", "-")}"></a>'
        for label, msg, color, href in badges
    )


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def render_status(reg: Registry) -> str:
    lines = []
    baselines = [d for d in reg.decisions if d.get("baseline")]
    if baselines:
        lines.append(f"- **Base model:** {baselines[-1]['baseline']}.")
    versions = [d for d in reg.decisions if d.get("version")]
    if versions:
        lines.append(f"- **Shipped version:** `{versions[-1]['version']}` ({versions[-1]['id']}).")
    else:
        lines.append(
            "- **Shipped version:** none yet. A version is earned by a gate SHIP "
            "([how](docs/experiments.md#versions))."
        )
    gates = [(e, run, g) for e in reg.experiments for run in e.runs for g in run.get("gate", [])]
    if gates:
        e, run, g = max(gates, key=lambda x: x[2]["date"])
        bad = [BENCHMARKS[p["bench"]] for p in g.get("pairs", []) if p.get("verdict") == "REGRESSION"]
        why = f" — regressed on {', '.join(bad)}" if bad else ""
        lines.append(
            f"- **Last gate run:** {_date(g['date'])} · [{run['id']}]({card(reg, e.id)}) · "
            f"{GATE_MARK[g['verdict']]}{why}."
        )
    counts = {
        t: sum(e.data.get("status") == "running" and e.data.get("track") == t for e in reg.experiments)
        for t in ("hypothesis", "enabler")
    }
    now = (
        f"- **In progress:** {_count(counts['hypothesis'], 'hypothesis', 'hypotheses')}, "
        f"{_count(counts['enabler'], 'enabler', 'enablers')}."
    )
    planned = sorted(
        (e for e in reg.experiments if e.data.get("status") == "planned"),
        key=lambda e: e.data.get("priority", 99),
    )
    if planned:
        items = ", ".join(f"[{e.id}]({card(reg, e.id)}) {e.data['short']}" for e in planned[:3])
        now += f" Next by priority: {items}."
    lines.append(now)
    return "\n".join(lines)


def _base_lines(short: str) -> list[str]:
    # The box is titled "Base", so a decision's own "Base: " would only repeat it.
    return textwrap.wrap(short.removeprefix("Base: "), BASE_WIDTH)


def _box(x: int, w: int, fill: str, title: str, lines: list[str], dashed: bool = False) -> str:
    """One stage of the picture. White text on a solid fill reads on GitHub's light and dark pages alike."""
    ink = "#768390" if dashed else "#FFFFFF"
    stroke = ' stroke="#768390" stroke-width="2" stroke-dasharray="6 5"' if dashed else ""
    out = [f'  <rect x="{x}" y="24" width="{w}" height="108" rx="10" fill="{fill}"{stroke}/>']
    cx = x + w // 2
    out.append(
        f'  <text x="{cx}" y="58" text-anchor="middle" font-size="15" font-weight="700" fill="{ink}">'
        f"{html.escape(title)}</text>"
    )
    for k, line in enumerate(lines):
        out.append(
            f'  <text x="{cx}" y="{82 + 20 * k}" text-anchor="middle" font-size="13" fill="{ink}">'
            f"{html.escape(line)}</text>"
        )
    return "\n".join(out)


def render_picture(reg: Registry) -> str:
    """The README's picture: how the work flows, with the counts as they stand.

    Five stages and one loop. It stays this small on purpose: one node per experiment drew 40 boxes
    nobody could read, and the decision log already says what fed what.
    """
    tracks = {t: [e for e in reg.experiments if e.data.get("track") == t] for t in TRACKS}
    gc = tracks["gap-closing"]
    first = min((e.data["started"] for e in gc), default=None)
    last = max((e.data.get("ended") or e.data["started"] for e in gc), default=None)
    span = f"{_date(first)} – {_date(last)}" if first and last else ""
    gated = sum(1 for e in reg.experiments for run in e.runs if run.get("gate"))
    bases = [d for d in reg.decisions if d.get("baseline")]
    base = bases[-1] if bases else None
    versions = [d["version"] for d in reg.decisions if d.get("version")]
    hypotheses = _count(len(tracks["hypothesis"]), "hypothesis", "hypotheses")
    enablers = _count(len(tracks["enabler"]), "enabler", "enablers")
    grey = "#768390"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-2 0 1004 206" role="img" '
        'aria-label="Gap closing, then hypotheses of one change each, judged against the base by the '
        'regression gate, until the gate ships a version" '
        "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\">",
        "  <title>How the work flows</title>",
        f'  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{grey}"/></marker></defs>',
        _box(0, 226, "#1E6B3A", "Phase 1: gap closing", [_count(len(gc), "experiment", "experiments"), span]),
        _box(
            266,
            164,
            "#B4560A",
            "Base",
            _base_lines(base["short"]) if base else [],
        ),
        _box(
            470,
            200,
            "#1F5FBF",
            "Phase 2: hypotheses",
            [f"{hypotheses}, one change each", enablers],
        ),
        _box(710, 136, "#57606A", "Regression gate", [f"{gated} gated so far"]),
        _box(
            898,
            102,
            "#1E6B3A" if versions else "none",
            versions[-1] if versions else "gallops-v0",
            ["shipped"] if versions else ["none yet"],
            dashed=not versions,
        ),
    ]
    for x1, x2 in ((226, 262), (430, 466), (670, 706)):
        parts.append(
            f'  <line x1="{x1}" y1="78" x2="{x2}" y2="78" stroke="{grey}" stroke-width="2" '
            'marker-end="url(#a)"/>'
        )
    parts += [
        f'  <line x1="846" y1="78" x2="894" y2="78" stroke="{grey}" stroke-width="2" marker-end="url(#a)"/>',
        f'  <text x="870" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="{grey}">'
        "SHIP</text>",
        f'  <path d="M778 132 V176 H570 V136" fill="none" stroke="{grey}" stroke-width="2" '
        'marker-end="url(#a)"/>',
        f'  <text x="674" y="196" text-anchor="middle" font-size="13" fill="{grey}">'
        "FAIL or HOLD: recorded, back to the hypotheses</text>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def render_gap_closing(reg: Registry) -> str:
    rows = ["| ID | Experiment | Owner | Headline result | Strength | Outcome |", "|---|---|---|---|---|---|"]
    for e in reg.experiments:
        d = e.data
        if d.get("track") != "gap-closing":
            continue
        head = e.headline()
        res = result_text(head[1]) if head else "—"
        strg = strength(head[1]) if head else "—"
        rows.append(
            f"| [**{e.id}**]({card(reg, e.id)}) | {_cell(d['short'])} | {owners(reg, d)} | {_cell(res)} | "
            f"{strg} | {mark(d)} |"
        )
    return "\n".join(rows)


def first_sentence(text: str) -> str:
    """A table cell's worth: up to the first full stop that ends a sentence; the card keeps the rest."""
    m = re.search(r"[.!?](?=\s+[A-Z(])", text)
    return text[: m.end()] if m else text


def render_hypotheses(reg: Registry) -> str:
    rows = [
        "| ID | Hypothesis | Owner | Status | Headline result | Strength | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in reg.experiments:
        d = e.data
        if d.get("track") != "hypothesis":
            continue
        head = e.headline()
        gate = latest_gate(e)
        gate_text = f"{GATE_MARK[gate[1]['verdict']]} {_date(gate[1]['date'])}" if gate else "—"
        rows.append(
            f"| [**{e.id}**]({card(reg, e.id)}) | {_cell(d['short'])} | {owners(reg, d)} | {mark(d)} | "
            f"{_cell(result_text(head[1])) if head else '—'} | {strength(head[1]) if head else '—'} | "
            f"{gate_text} |"
        )
    return "\n".join(rows)


def render_enablers(reg: Registry) -> str:
    rows = [
        "| ID | Enabler | Owner | Priority | Status | What it gives the team |",
        "|---|---|---|:-:|---|---|",
    ]
    for e in reg.experiments:
        d = e.data
        if d.get("track") != "enabler":
            continue
        gives = first_sentence(d.get("learned") or d["question"])
        rows.append(
            f"| [**{e.id}**]({card(reg, e.id, 'experiments')}) | {_cell(d['short'])} | {owners(reg, d)} | "
            f"{d.get('priority', '—')} | {mark(d)} | {_cell(gives)} |"
        )
    return "\n".join(rows)


def _pair_cell(pair: dict[str, Any] | None) -> str:
    if not pair:
        return "—"
    text = PAIR_MARK[pair["verdict"]]
    if "baseline" in pair and "candidate" in pair:
        text += f" {_num(pair['baseline'])} → {_num(pair['candidate'])}"
    return text


def render_gate(reg: Registry) -> str:
    header = (
        "| Date | Run | Checkpoint | Tier | "
        + " | ".join(BENCHMARKS[b] for b in GATE_BENCHES)
        + " | Verdict |"
    )
    rows = [header, "|---|---|---|---|" + "---|" * len(GATE_BENCHES) + "---|"]
    entries = sorted(
        ((e, run, g) for e in reg.experiments for run in e.runs for g in run.get("gate", [])),
        key=lambda x: (x[2]["date"], x[1]["id"]),
    )
    for e, run, g in entries:
        pairs = {p["bench"]: p for p in g.get("pairs", [])}
        cells = " | ".join(_pair_cell(pairs.get(b)) for b in GATE_BENCHES)
        rows.append(
            f"| {_date(g['date'])} {g['date'].year} | [{run['id']}]({card(reg, e.id, 'experiments')}) | "
            f"{tag(run)} | "
            f"{g['tier']} | "
            f"{cells} | {GATE_MARK[g['verdict']]} |"
        )
    if not entries:
        rows.append("| — | no gate run recorded yet |" + " |" * (len(GATE_BENCHES) + 2))
    return "\n".join(rows)


def _refs(reg: Registry, refs: list[str], base: str = "") -> str:
    """Experiments or runs, each linked to its experiment's card from a file in folder `base`."""
    runs = reg.run_ids()
    out = []
    for ref in refs:
        eid = runs[ref][0].id if ref in runs else ref
        out.append(f"[{ref}]({card(reg, eid, base)})")
    return ", ".join(out) if out else "—"


def render_decisions(reg: Registry) -> str:
    rows = ["| # | Date | Decision | By | Evidence | Led to |", "|---|---|---|---|---|---|"]
    for d in reg.decisions:
        by = ", ".join(d.get("by", []))
        evidence = _refs(reg, d.get("evidence", []), "experiments")
        rows.append(
            f"| **{d['id']}** | {_date(d['date'])} {d['date'].year} | **{_cell(d['short'])}.** "
            f"{_cell(d['decision'])} | {by} | "
            f"{evidence} | {_refs(reg, d.get('next', []), 'experiments')} |"
        )
    return "\n".join(rows)


def _data_text(run: dict[str, Any]) -> str:
    parts = []
    for data in run.get("data", []):
        n = f"{_num(data['n'])} {data.get('unit', '')}".strip() if "n" in data else ""
        text = f"{data['dataset']} `{data['split']}`" + (f" {n}" if n else "") + f" ({data['role']})"
        if data.get("note"):
            text += f" — {data['note']}"
        parts.append(text)
    return "<br>".join(_cell(p) for p in parts) or "—"


def _trained_text(run: dict[str, Any]) -> str:
    text = ", ".join(run.get("trained", [])) or "—"
    if run.get("params_trained_m"):
        text += f" ({_num(run['params_trained_m'])} M)"
    if run.get("frozen"):
        text += f"; frozen: {', '.join(run['frozen'])}"
    return _cell(text)


def _checkpoint_text(run: dict[str, Any]) -> str:
    ck = run.get("checkpoint")
    if not ck:
        return "—"
    where = ck.get("location") or "/".join(
        x
        for x in (ck.get("model", ""), ck.get("train_run", ""), f"step {ck['step']}" if "step" in ck else "")
        if x
    )
    text = f"`{where}`"
    if ck.get("weights_sha256"):
        whose = f" ({ck['file']})" if ck.get("file") and ck.get("members") else ""
        if ck.get("members") and not ck.get("file"):
            whose = f" (over {len(ck['members'])} files)"
        text += f"<br>sha256{whose} `{ck['weights_sha256']}`"
    else:
        text += "<br>sha256 not recorded"
    if ck.get("members"):
        files = ", ".join(f"{m.split('  ', 1)[1]} `{m[:12]}`" for m in ck["members"])
        text += f"<br>files: {files}"
    if ck.get("shipped_as"):
        text += f"<br>shipped as {ck['shipped_as']}"
    if ck.get("known_wrong"):
        text += f"<br>⚠️ sidecar: {ck['known_wrong']}"
    return _cell(text)


def _code_text(d: dict[str, Any], run: dict[str, Any], base: str = "") -> str:
    code = run.get("code") or {}
    bits = []
    if code.get("entry"):
        entry = code["entry"]
        # A script links to its folder, not its file: as-run scripts keep the working names the work was
        # done under, which the rendered pages do not repeat (tests/test_drivejepa_retrain.py).
        folder = Path(entry).parent.as_posix()
        bits.append(
            f"`{entry}`"
            if _is_module(entry)
            else f"[scripts in {Path(folder).name}/]({link(d, folder + '/', base)})"
        )
    if code.get("recipe"):
        bits.append(f"[{code['recipe'].split('/')[-1]}]({link(d, code['recipe'].split('#')[0], base)})")
    if run.get("train_git_sha"):
        bits.append(f"commit `{run['train_git_sha']}`")
    up = code.get("upstream")
    if up:
        bits.append(f"{up['repo']} @ `{up['commit']}`" + (" + patches" if up.get("patches") else ""))
    return "<br>".join(bits) or "—"


#: Where a card's "back" link lands, by track: the README's tables, or the index for enablers.
_BACK = {
    "gap-closing": "README.md#gap-closing-experiments",
    "hypothesis": "README.md#hypothesis-experiments",
    "enabler": f"{INDEX}#enablers",
}


def render_card(reg: Registry, e: Experiment) -> str:
    """One experiment's card: everything its record holds, on one page in its own folder.

    The README keeps one row per experiment and links its ID here, so the README stays the same size as
    experiments are added. Rendered whole; the file is never edited by hand.
    """
    d = e.data
    base = e.folder.relative_to(reg.root).as_posix()
    out = [
        "<!-- Rendered from experiment.toml by `gallops experiments render --write`.",
        "     Edit that file, not this one: make check fails when the two differ. -->",
        "",
        f"# {e.id} · {d['title']}",
        "",
        f"**{owners(reg, d)}** · {benches(d)} · {mark(d)} · {_span(d)} · {d['track']}",
        "",
    ]
    nav = [f"[← all experiments]({rel(_BACK[d['track']], base)})"]
    if d.get("narrative"):
        where = f" (on {_ref_kind(d)} ⎇)" if d.get("branch") else ""
        nav.append(f"[the account]({link(d, base=base)})" + where)
    nav.append(f"[experiment.toml]({rel(base + '/experiment.toml', base)})")
    out += [" · ".join(nav), ""]
    if d.get("branch"):
        kind = _ref_kind(d)
        ref = d["branch"][:7] if kind == "commit" else d["branch"]
        out += [f"> The record is on {kind} `{ref}`, not merged yet; its links go to that {kind}.", ""]
    meta = []
    if d.get("priority"):
        meta.append(f"**Priority:** {d['priority']}")
    if d.get("parents"):
        meta.append(f"**Builds on:** {_refs(reg, d['parents'], base)}")
    led = [
        x["id"]
        for x in reg.decisions
        if e.id in x.get("evidence", []) or any(r.startswith(e.id + ".") for r in x.get("evidence", []))
    ]
    if led:
        meta.append(
            f"**Evidence for:** {', '.join(led)} ([decision log]({rel(INDEX + '#decision-log', base)}))"
        )
    if meta:
        out += [" · ".join(meta), ""]
    out += ["## Question", "", d["question"], ""]
    if d.get("hypothesis"):
        out += ["## Hypothesis", "", d["hypothesis"], ""]
    if d.get("predictions"):
        pred = d["predictions"]
        if pred.startswith("none:"):
            text = f"⚠️ Not pre-registered: {pred[5:].strip()}"
        else:
            text = f"[{Path(pred).name}]({link(d, pred, base)}), written before the run."
        out += ["## Predictions", "", text, ""]
    if d.get("learned"):
        out += ["## What we learned", "", d["learned"], ""]
    if e.runs:
        out += [
            "## Runs",
            "",
            "| Run | Started from | Trained | Data | Code | Checkpoint |",
            "|---|---|---|---|---|---|",
        ]
        for run in e.runs:
            out.append(
                f"| **{run['id']}**<br>{_cell(run['label'])} | {_cell(run.get('start_from', '—'))} | "
                f"{_trained_text(run)} | {_data_text(run)} | {_code_text(d, run, base)} | "
                f"{_checkpoint_text(run)} |"
            )
        out.append("")
        notes = [f"- **{run['id']}:** {run['notes']}" for run in e.runs if run.get("notes")]
        if notes:
            out.extend([*notes, ""])
    results = list(e.results())
    if results:
        out += ["## Results", "", "★ marks the headline.", ""]
        out += ["| Run | Result | Baseline | Strength | Source |", "|---|---|---|---|---|"]
        for run, res in results:
            star = " ★" if res.get("headline") else ""
            extra = []
            if res.get("subset"):
                extra.append(res["subset"])
            if "se" in res:
                extra.append(f"SE {_num(res['se'])}")
            if "t" in res:
                extra.append(f"t {_num(res['t'])}")
            if res.get("note"):
                extra.append(res["note"])
            text = result_text(res, bold=bool(star)) + (f"<br>{'; '.join(extra)}" if extra else "")
            out.append(
                f"| {run['id']}{star} | {_cell(text)} | {_cell(res.get('baseline', '—'))} | "
                f"{strength(res)} | {_cell(res.get('source', '—'))} |"
            )
        out.append("")
    gates = [(run, g) for run in e.runs for g in run.get("gate", [])]
    if gates:
        out += ["## Regression gate", ""]
    for run, g in gates:
        head = f"**{run['id']}** — {_date(g['date'])} {g['date'].year}, {g['tier']} tier"
        head += f", against {g['baseline']}" if g.get("baseline") else ""
        out += [f"{head}: {GATE_MARK[g['verdict']]}", ""]
        if g.get("pairs"):
            out.append("| Benchmark | Metric | Baseline | Candidate | t | MDE | Units | Verdict |")
            out.append("|---|---|---|---|---|---|---|---|")
            for pr in g["pairs"]:
                out.append(
                    f"| {BENCHMARKS[pr['bench']]} | {pr.get('metric', '')} | "
                    f"{_num(pr.get('baseline', ''))} | {_num(pr.get('candidate', ''))} | "
                    f"{_num(pr.get('t', ''))} | {_num(pr.get('mde', ''))} | {_num(pr.get('units', ''))} | "
                    f"{PAIR_MARK[pr['verdict']]} {pr['verdict']} |"
                )
            out.append("")
        if g.get("note"):
            out += [g["note"], ""]
    if d.get("checks"):
        out += ["## Checks that decided something", ""]
        for c in d["checks"]:
            when = f" ({_date(c['date'])})" if c.get("date") else ""
            decided = f" → {c['decided']}" if c.get("decided") else ""
            out.append(f"- **{c['name']}**{when}: {c['result']}{decided}")
        out.append("")
    if d.get("external"):
        out += ["## Evidence outside this repo", "", *[f"- {x}" for x in d["external"]], ""]
    return "\n".join(out).rstrip() + "\n"


def render_index(reg: Registry) -> str:
    rows = ["| ID | Experiment | Track | Owner | When | Status | Folder |", "|---|---|---|---|---|---|---|"]
    for e in reg.experiments:
        d = e.data
        folder = e.folder.relative_to(reg.root / "experiments").as_posix()
        rows.append(
            f"| [{e.id}]({card(reg, e.id, 'experiments')}) | {_cell(d['title'])} | {d['track']} | "
            f"{owners(reg, d)} | {_span(d)} | {mark(d)} | "
            f"[{folder}/]({folder}/) |"
        )
    return "\n".join(rows)


def render_code_map(reg: Registry) -> str:
    rows = ["| Module | Used by | Kind |", "|---|---|---|"]
    for module, runs in module_claims(reg).items():
        kind = "shared" if module in SHARED_MODULES else ("experiment" if runs else "**unclaimed**")
        rows.append(f"| `{module}` | {', '.join(runs) or '—'} | {kind} |")
    return "\n".join(rows)


README_BLOCKS = {
    "badges": render_badges,
    "status": render_status,
    "gap-closing": render_gap_closing,
    "hypotheses": render_hypotheses,
}
INDEX_BLOCKS = {
    "index": render_index,
    "enablers": render_enablers,
    "gate": render_gate,
    "decisions": render_decisions,
    "code-map": render_code_map,
}

#: Which file each set of blocks is rendered into, relative to the repo root.
TARGETS = {"README.md": README_BLOCKS, INDEX: INDEX_BLOCKS}


def render(reg: Registry, blocks: dict[str, Any]) -> dict[str, str]:
    return {name: fn(reg) for name, fn in blocks.items()}


def markers(name: str) -> tuple[str, str]:
    return f"<!-- experiments:{name} -->", f"<!-- /experiments:{name} -->"


def apply(text: str, rendered: dict[str, str]) -> str:
    """Replace each marked block's body; refuse a file that is missing a marker."""
    for name, body in rendered.items():
        start, end = markers(name)
        i, j = text.find(start), text.find(end)
        if i < 0 or j < i:
            raise RegistryError(f"no {start} ... {end} block to render into")
        text = text[: i + len(start)] + "\n" + body + "\n" + text[j:]
    return text


def rendered(reg: Registry) -> dict[str, str]:
    """Every file the registry renders, repo-root path -> the text it should hold.

    The README and the index have marked blocks inside hand-written prose; the picture and each card
    are rendered whole.
    """
    out = {}
    for target, blocks in TARGETS.items():
        out[target] = apply((reg.root / target).read_text(), render(reg, blocks))
    out[PICTURE] = render_picture(reg)
    for e in reg.experiments:
        out[card_path(reg, e)] = render_card(reg, e)
    return out


def stale(reg: Registry) -> list[str]:
    """The rendered files that are missing or differ from what the registry renders now."""
    return [
        target
        for target, text in rendered(reg).items()
        if not (reg.root / target).is_file() or (reg.root / target).read_text() != text
    ]


def write(reg: Registry) -> list[str]:
    changed = stale(reg)
    for target, text in rendered(reg).items():
        if target in changed:
            (reg.root / target).parent.mkdir(parents=True, exist_ok=True)
            (reg.root / target).write_text(text)
    return changed
