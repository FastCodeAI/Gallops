"""The experiment registry: every record valid, the README rendered from it, no trainer without its evidence.

The README's tables are the evidence for the model this repo builds, so they are rendered, not typed:
`gallops experiments render --write` writes them from experiments/*/experiment.toml, and these tests fail
when the README says anything the registry does not. The negative tests build a small repository in a
temporary directory and break it one way at a time, so each rule is shown to bite.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from gallops import experiments as exps

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- the real registry


@pytest.fixture(scope="module")
def registry() -> exps.Registry:
    return exps.Registry.load(REPO_ROOT)


def test_every_record_is_valid(registry) -> None:
    assert registry.errors == [], "\n".join(registry.errors)


def test_the_readme_is_what_the_registry_renders(registry) -> None:
    stale = exps.stale(registry)
    assert stale == [], f"{stale} out of date: run `gallops experiments render --write`"


def test_paddocks_link_into_this_readme_still_lands() -> None:
    """Paddock's README links Gallops#training-so-far for its trained Drive-JEPA rows (paddock 9a51c15)."""
    assert '<a id="training-so-far"></a>' in (REPO_ROOT / "README.md").read_text()


def test_the_template_fills_into_a_valid_record(tmp_path) -> None:
    """`gallops experiments new` copies experiments/_template; what it writes must pass `make check`."""
    root = _repo(tmp_path, record=None, decisions="")
    folder = root / "experiments" / "H01_thing"
    folder.mkdir(parents=True)
    fill = {
        "{{ID}}": "H01",
        "{{ID_LOWER}}": "h01",
        "{{FOLDER}}": "H01_thing",
        "{{TRACK}}": "hypothesis",
        "{{OWNER}}": "Naren",
        "{{DATE}}": "2026-11-03",
    }
    for src in (REPO_ROOT / "experiments" / "_template").iterdir():
        text = src.read_text()
        for key, value in fill.items():
            text = text.replace(key, value)
        (folder / src.name).write_text(text)
    reg = exps.Registry.load(root)
    # The template claims no module yet; the fake repo's one trainer is unclaimed by design here.
    assert [e for e in reg.errors if "is used by no experiment" not in e] == []


# ---------------------------------------------------------------- a small repository to break

SHA = "a" * 64

RECORD = f"""\
schema = 1
id = "H01"
title = "A thing that helps"
short = "a thing"
track = "hypothesis"
owner = "Naren"
benchmarks = ["navsim"]
started = 2026-10-21
status = "done"
verdict = "positive"
question = "Does the thing help?"
hypothesis = "It raises EPDMS by 0.01; a NEUTRAL with MDE under 0.01 refutes it."
learned = "It did."
predictions = "experiments/H01_thing/PREDICTIONS.md"
narrative = "experiments/H01_thing/README.md"

[[runs]]
id = "H01.r1"
label = "first run"
start_from = "released"

  [runs.code]
  modules = ["gallops.train.thing"]

  [runs.checkpoint]
  model = "thing"
  weights_sha256 = "{SHA}"

  [[runs.results]]
  bench = "navsim"
  metric = "EPDMS"
  baseline_value = 0.875
  value = "0.885"
  better = "higher"
  strength = "moderate"
  headline = true

  [[runs.gate]]
  date = 2026-10-28
  tier = "quick"
  verdict = "HOLD"

    [[runs.gate.pairs]]
    bench = "navsim"
    verdict = "NEUTRAL"
"""

DECISIONS = """\
[[decision]]
id = "D01"
date = 2026-10-20
short = "try the thing"
decision = "Try the thing."
by = ["team"]
next = ["H01"]

[[decision]]
id = "D02"
date = 2026-10-29
short = "keep the thing"
decision = "Keep the thing."
by = ["Naren"]
evidence = ["H01.r1"]
"""


def _blocks(names) -> str:
    return "\n".join(f"{start}\n{end}" for start, end in (exps.markers(n) for n in names)) + "\n"


def _repo(tmp_path: Path, record: str | None = RECORD, decisions: str = DECISIONS) -> Path:
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "people.toml").write_text('[people.Naren]\nname = "Narendiran"\n')
    (root / "experiments" / "decisions.toml").write_text(decisions)
    (root / "README.md").write_text(_blocks(exps.README_BLOCKS))
    (root / "experiments" / "README.md").write_text(_blocks(exps.INDEX_BLOCKS))
    pkg = root / "src" / "gallops"
    (pkg / "train").mkdir(parents=True)
    for name in ("__init__.py", "checkpoint.py", "train/__init__.py", "train/thing.py"):
        (pkg / name).write_text("")
    if record is not None:
        folder = root / "experiments" / "H01_thing"
        folder.mkdir()
        (folder / "experiment.toml").write_text(record)
        (folder / "README.md").write_text("# H01\n")
        (folder / "PREDICTIONS.md").write_text("# H01 predictions\n")
    return root


def test_the_small_repository_is_valid_and_renders(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    assert reg.errors == []
    assert exps.write(reg) == ["README.md", "experiments/README.md"]
    assert exps.stale(exps.Registry.load(reg.root)) == []
    readme = (reg.root / "README.md").read_text()
    assert "[**H01**](#h01)" in readme and "**0.885**" in readme and "`aaaaaaaaaaaa`" in readme


@pytest.mark.parametrize(
    ("change", "old", "new", "error"),
    [
        ("a sha that is not one", f'"{SHA}"', '"abc123"', "weights_sha256 is not 64"),
        (
            "a gated run with no sha",
            f'  weights_sha256 = "{SHA}"\n',
            "",
            "pins its checkpoint's weights_sha256",
        ),
        ("a parent that does not exist", "learned =", 'parents = ["GC09"]\nlearned =', "parent GC09 is not"),
        (
            "a module that does not exist",
            '"gallops.train.thing"',
            '"gallops.train.nothing"',
            "does not exist",
        ),
        (
            "a trainer nobody claims",
            'modules = ["gallops.train.thing"]',
            "modules = []",
            "is used by no experiment",
        ),
        (
            "results with no predictions",
            'predictions = "experiments/H01_thing/PREDICTIONS.md"\n',
            "",
            "predictions",
        ),
        ("a narrative that is not there", "H01_thing/README.md", "H01_thing/NOTES.md", "does not exist"),
        ("an unknown field", "learned =", 'colour = "red"\nlearned =', "unknown field 'colour'"),
        ("a strength off the ladder", '"moderate"', '"very strong"', "is not one of"),
        ("an id that is not its folder's", 'id = "H01"', 'id = "H02"', "folder must be named H02_"),
        ("a harmless note (a control)", "headline = true", "headline = true\n  note = 'x'", None),
        ("no headline", "  headline = true\n", "", "exactly one result is the `headline`"),
        ("a run id under another experiment", 'id = "H01.r1"', 'id = "H03.r1"', "must be H01.<label>"),
        ("an owner nobody knows", 'owner = "Naren"', 'owner = "Nobody"', "not in experiments/people.toml"),
        ("a finished experiment that learned nothing", 'learned = "It did."\n', "", "`learned`"),
    ],
)
def test_each_rule_bites(tmp_path, change, old, new, error) -> None:
    assert old in RECORD, change
    reg = exps.Registry.load(_repo(tmp_path, RECORD.replace(old, new, 1)))
    if error is None:  # a control: a harmless change stays valid
        assert reg.errors == [], change
    else:
        assert any(error in e for e in reg.errors), f"{change}: {reg.errors}"


def test_a_recipe_stage_claims_the_module_it_runs(tmp_path) -> None:
    """Stages name modules relative to the package and run as `python -m gallops.<module>`."""
    root = _repo(
        tmp_path, RECORD.replace('modules = ["gallops.train.thing"]', 'recipe = "recipes/thing.toml#fit"')
    )
    (root / "recipes").mkdir()
    (root / "recipes" / "thing.toml").write_text('[[stage]]\nname = "fit"\nmodule = "train.thing"\n')
    reg = exps.Registry.load(root)
    assert reg.errors == []
    assert exps.module_claims(reg)["gallops.train.thing"] == ["H01.r1"]


def test_decisions_are_numbered_in_date_order(tmp_path) -> None:
    swapped = DECISIONS.replace("date = 2026-10-29", "date = 2026-10-19")
    reg = exps.Registry.load(_repo(tmp_path, decisions=swapped))
    assert any("dated before the decision above it" in e for e in reg.errors)


def test_a_decision_cites_only_what_exists(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path, decisions=DECISIONS.replace('"H01.r1"', '"H01.r9"')))
    assert any("H01.r9 is not an experiment or run" in e for e in reg.errors)


def test_a_hand_edited_readme_is_caught(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    exps.write(reg)
    readme = reg.root / "README.md"
    readme.write_text(readme.read_text().replace("**0.885**", "**0.950**"))
    assert exps.stale(exps.Registry.load(reg.root)) == ["README.md"]


def test_a_record_on_an_unmerged_branch_is_linked_there_not_checked_here(tmp_path) -> None:
    on_branch = RECORD.replace("learned =", 'branch = "train/thing"\nlearned =').replace(
        '"experiments/H01_thing/README.md"', '"docs/thing.md"'
    )
    reg = exps.Registry.load(_repo(tmp_path, on_branch))
    # its narrative is a path on the branch, so it is not looked for here; its module claim does not count
    assert [e for e in reg.errors if "docs/thing.md" in e] == []
    assert any("gallops.train.thing is used by no experiment" in e for e in reg.errors)
    exps.write(exps.Registry.load(reg.root))
    assert f"{exps.GITHUB}/blob/train/thing/docs/thing.md" in (reg.root / "README.md").read_text()


def test_numbers_keep_the_precision_they_were_reported_at() -> None:
    assert exps._num("87.30") == "87.30"
    assert exps._num(126981) == "126,981"
    assert exps._num(0.5433) == "0.5433"


def test_dates_render_as_a_reader_says_them() -> None:
    assert (
        exps._span({"started": dt.date(2026, 10, 16), "ended": dt.date(2026, 10, 17)})
        == "16 Oct – 17 Oct 2026"
    )
    assert exps._span({"started": dt.date(2026, 9, 26), "ended": dt.date(2026, 9, 26)}) == "26 Sep 2026"
