"""Recipes resolve on any site that configures what they name, and their stages exist.

A recipe is the record of how a checkpoint was made, so a broken one fails late and expensively: a stage
that names a module that is not there, or a template no site can expand, surfaces hours into a run. These
checks need no GPU, no torch and no data.
"""

from __future__ import annotations

import ast
import fnmatch
import json
from pathlib import Path

import pytest

from gallops import recipe as recipes
from gallops.config import REPO_ROOT, ConfigError, SiteConfig

SRC = REPO_ROOT / "src" / "gallops"

#: Modules that run inside Drive-JEPA's own interpreter (Python 3.9), and what they import from gallops.
PY39 = (
    "__init__.py",
    "drivejepa_b2d/__init__.py",
    "drivejepa_b2d/provenance.py",
    "drivejepa_b2d/batch_fix.py",
    "drivejepa_b2d/host.py",
    "drivejepa_b2d/train_generator.py",
    # gallops.drivejepa_aux, and every gallops module importing it loads under 3.9 (listed from
    # sys.modules in Drive-JEPA's interpreter, not by reading imports)
    "drivejepa_aux/__init__.py",
    "drivejepa_aux/bev.py",
    "drivejepa_aux/targets.py",
    "drivejepa_aux/nusc.py",
    "drivejepa_aux/inputs.py",
    "drivejepa_aux/heads.py",
    "drivejepa_aux/step.py",
    "drivejepa_aux/split.py",
    "drivejepa_aux/cache.py",
    "drivejepa_aux/metrics.py",
    "drivejepa_aux/preflight.py",
    "drivejepa_aux/publish.py",
    "drivejepa_aux/train.py",
    "drivejepa_aux/report.py",
    "drivejepa_b2d/sidecar.py",
    "checkpoint.py",
    "config.py",
    "data/__init__.py",
    "data/nuscenes_real.py",
    "expert/__init__.py",
    "expert/grader.py",
    "expert/lanes.py",
    "train/__init__.py",
    "train/losses.py",
    "train/preflight.py",
    "train/swerve.py",
    "train/swerve_data.py",
)


def write_site(tmp: Path) -> SiteConfig:
    for d in ("dump", "navsim"):
        (tmp / d).mkdir(exist_ok=True)
    path = tmp / "gallops.toml"
    path.write_text(
        f"""
site = "test"
[roots]
weights = "{tmp}/weights"
runs = "{tmp}/runs"
[python]
train = "{tmp}/py/train"
drivejepa = "{tmp}/py/drivejepa"
[code]
drivejepa = "{tmp}/Drive-JEPA"
[datasets.navsim]
path = "{tmp}/navsim"
exp = "{tmp}/navsim-exp"
[datasets.drivejepa_b2d]
path = "{tmp}/dump"
"""
    )
    return SiteConfig.load(path)


@pytest.fixture
def site(tmp_path: Path) -> SiteConfig:
    return write_site(tmp_path)


def test_every_stage_resolves(site: SiteConfig) -> None:
    rec = recipes.load("drivejepa_b2d")
    cmds = recipes.commands(rec, site)
    assert [c.stage for c in cmds] == rec.names()
    for c in cmds:
        text = " ".join([*c.argv, c.cwd or "", *c.env.values()])
        assert "{" not in text, f"{c.stage}: unexpanded template in {text}"
        assert c.argv[1] == "-m" and c.argv[2].startswith("gallops.")


def test_every_stage_module_exists() -> None:
    for s in recipes.load("drivejepa_b2d").stages:
        path = SRC / (s.module.replace(".", "/") + ".py")
        assert path.exists(), f"stage {s.name} runs {s.module}, which is not at {path}"


def test_stages_run_in_the_interpreter_they_name(site: SiteConfig) -> None:
    for c in recipes.commands(recipes.load("drivejepa_b2d"), site):
        assert c.argv[0] in {str(p) for p in site.python.values()}
    gen = recipes.commands(recipes.load("drivejepa_b2d"), site, ["final-generator"])[0]
    assert gen.argv[0] == str(site.python["drivejepa"])
    assert gen.cwd == f"{site.code['drivejepa']}/navsim_v2", (
        "the agent loads ./data/8192.npy relative to the cwd"
    )


@pytest.mark.parametrize("rel", PY39)
def test_drivejepa_side_is_python39(rel: str) -> None:
    """These run in Drive-JEPA's Python 3.9 env; 3.10+ syntax would fail only there, hours into a run."""
    ast.parse((SRC / rel).read_text(), filename=rel, feature_version=(3, 9))


def test_only_named_stages_and_unknown_names_fail(site: SiteConfig) -> None:
    rec = recipes.load("drivejepa_b2d")
    assert [c.stage for c in recipes.commands(rec, site, ["split-full"])] == ["split-full"]
    with pytest.raises(ConfigError, match="no stage"):
        recipes.commands(rec, site, ["split-v5"])


def test_a_stage_must_follow_what_it_reads(tmp_path: Path) -> None:
    (tmp_path / "bad.toml").write_text(
        '[[stage]]\nname = "a"\nmodule = "m"\npython = "train"\nafter = ["b"]\n'
        '[[stage]]\nname = "b"\nmodule = "m"\npython = "train"\n'
    )
    with pytest.raises(ConfigError, match="not earlier stages"):
        recipes.load("bad", root=tmp_path)


def test_python_and_code_tables(site: SiteConfig) -> None:
    assert (
        site.resolve("{python.train} {code.drivejepa}") == f"{site.python['train']} {site.code['drivejepa']}"
    )
    with pytest.raises(ConfigError, match=r"no \[python\] entry 'missing'"):
        site.resolve("{python.missing}")


def test_a_site_without_the_interpreter_cannot_resolve(tmp_path: Path) -> None:
    site = write_site(tmp_path)
    bare = SiteConfig(
        site=site.site, weights=site.weights, runs=site.runs, datasets=site.datasets, source=site.source
    )
    with pytest.raises(ConfigError, match="python"):
        recipes.commands(recipes.load("drivejepa_b2d"), bare)


def test_the_held_out_split_is_tracked() -> None:
    """Every head stage selects its epoch on these 150 clips; the recipe reads them from the repo, not from a
    site path."""
    clips = json.loads((REPO_ROOT / "recipes/drivejepa_b2d/heldout_150.json").read_text())["val"]
    assert len(clips) == len(set(clips)) == 150


def test_the_dump_holds_everything_the_recipe_reads(site: SiteConfig) -> None:
    """Every path a stage reads from the data dump matches a file in the tracked manifests: a stage reading
    something the dump does not hold would only fail on a site that has nothing but the dump."""
    dump = str(site.datasets["drivejepa_b2d"].path) + "/"
    files: dict = {}
    for m in sorted((REPO_ROOT / "recipes/drivejepa_b2d").glob("data_*.json")):
        files.update(json.loads(m.read_text())["files"])
    for c in recipes.commands(recipes.load("drivejepa_b2d"), site):
        for arg in c.argv[3:]:
            if not arg.startswith(dump):
                continue
            rel = arg[len(dump) :]
            hit = any(fnmatch.fnmatch(f, rel) or f.startswith(rel.rstrip("/") + "/") for f in files)
            assert hit, f"{c.stage} reads {rel}, which no manifest lists"


def test_provenance_finds_the_repo_root_config_does() -> None:
    """provenance restates config.REPO_ROOT so it can run in a model's own interpreter, where config does not
    import; a copy is only acceptable with a check that it still agrees."""
    from gallops.drivejepa_b2d import provenance

    assert provenance.REPO_ROOT == REPO_ROOT
