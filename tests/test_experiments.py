"""The experiment registry: every record valid, the README rendered from it, no trainer without its evidence.

The README's tables are the evidence for the model this repo builds, so they are rendered, not typed:
`gallops experiments render --write` writes them from experiments/*/experiment.toml, and these tests fail
when the README says anything the registry does not. The negative tests build a small repository in a
temporary directory and break it one way at a time, so each rule is shown to bite.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import subprocess
import sys
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


def test_the_readme_stays_a_front_page() -> None:
    """One row per experiment and links out. Detail goes on cards and in experiments/README.md."""
    text = (REPO_ROOT / "README.md").read_text()
    assert len(text.splitlines()) <= 120 and len(text.encode()) <= 16 * 1024


def test_paddocks_link_into_this_readme_still_lands() -> None:
    """Paddock's README links Gallops#training-so-far for its trained Drive-JEPA rows (paddock 9a51c15)."""
    assert '<a id="training-so-far"></a>' in (REPO_ROOT / "README.md").read_text()


def test_the_registry_imports_where_tomllib_does_not_exist() -> None:
    """The preflight runs this suite under Drive-JEPA's Python 3.9, which has tomli and no tomllib."""
    # Under 3.11+ hide tomllib and offer it as tomli, the 3.9 layout; under a real 3.9 there is no
    # tomllib to hide, and the module's own fallback is what runs.
    code = "\n".join(
        [
            "import sys",
            "try:",
            "    import tomllib",
            "    sys.modules['tomli'], sys.modules['tomllib'] = tomllib, None",
            "except ModuleNotFoundError:",
            "    pass",
            "import gallops.experiments",
            "print('ok')",
        ]
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


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
    assert exps.write(reg) == [
        "README.md",
        "experiments/README.md",
        "docs/assets/experiments.svg",
        "experiments/H01_thing/CARD.md",
    ]
    assert exps.stale(exps.Registry.load(reg.root)) == []
    readme = (reg.root / "README.md").read_text()
    # the README links each ID to its card, and keeps one row per experiment
    assert "[**H01**](experiments/H01_thing/CARD.md)" in readme and "**0.885**" in readme
    card = (reg.root / "experiments" / "H01_thing" / "CARD.md").read_text()
    # the card holds the detail, with links relative to its own folder
    assert f"`{SHA}`" in card and "[the account](README.md)" in card
    assert "[← all experiments](../../README.md#hypothesis-experiments)" in card


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


def test_the_picture_counts_what_the_registry_holds(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    exps.write(reg)
    picture = reg.root / "docs" / "assets" / "experiments.svg"
    svg = picture.read_text()
    assert "1 hypothesis, one change each" in svg and "1 gated so far" in svg and "none yet" in svg
    picture.write_text(svg.replace("1 gated so far", "9 gated so far"))
    assert exps.stale(exps.Registry.load(reg.root)) == ["docs/assets/experiments.svg"]


def test_a_baseline_the_picture_cannot_hold_is_refused(tmp_path) -> None:
    """The box took two lines and dropped the rest, so it named a different model and nothing failed."""
    long = "Base: the 4-camera stack with state-space memory and the sub-score scorer"
    decisions = DECISIONS.replace(
        'short = "try the thing"\n', f'short = "{long}"\nbaseline = "the 4-camera stack"\n'
    )
    errors = exps.Registry.load(_repo(tmp_path, decisions=decisions)).errors
    assert any("baseline's `short`" in e for e in errors)


def test_a_count_of_one_is_singular(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    reg.experiments[0].data["status"] = "running"
    assert "**In progress:** 1 hypothesis, 0 enablers." in exps.render_status(reg)


def test_each_decision_leads_with_its_short_name(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    assert "| **try the thing.** Try the thing. |" in exps.render_decisions(reg)


def test_a_hand_edited_or_missing_card_is_caught(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    exps.write(reg)
    card = reg.root / "experiments" / "H01_thing" / "CARD.md"
    card.write_text(card.read_text().replace("It did.", "It did, and more."))
    assert exps.stale(exps.Registry.load(reg.root)) == ["experiments/H01_thing/CARD.md"]
    card.unlink()
    assert exps.stale(exps.Registry.load(reg.root)) == ["experiments/H01_thing/CARD.md"]


def test_a_record_on_an_unmerged_branch_is_linked_there_not_checked_here(tmp_path) -> None:
    on_branch = RECORD.replace("learned =", 'branch = "train/thing"\nlearned =').replace(
        '"experiments/H01_thing/README.md"', '"docs/thing.md"'
    )
    reg = exps.Registry.load(_repo(tmp_path, on_branch))
    # its narrative is a path on the branch, so it is not looked for here; its module claim does not count
    assert [e for e in reg.errors if "docs/thing.md" in e] == []
    assert any("gallops.train.thing is used by no experiment" in e for e in reg.errors)
    exps.write(exps.Registry.load(reg.root))
    card = (reg.root / "experiments" / "H01_thing" / "CARD.md").read_text()
    assert f"{exps.GITHUB}/blob/train/thing/docs/thing.md" in card


def test_a_record_whose_branch_was_deleted_is_linked_at_its_commit(tmp_path) -> None:
    """A closed PR keeps its head commit after the branch goes, so `branch` can name that commit."""
    sha = "1c60e59" + "0" * 33
    gone = RECORD.replace("learned =", f'branch = "{sha}"\nlearned =').replace(
        '"experiments/H01_thing/README.md"', '"docs/thing.md"'
    )
    reg = exps.Registry.load(_repo(tmp_path, gone))
    exps.write(exps.Registry.load(reg.root))
    card = (reg.root / "experiments" / "H01_thing" / "CARD.md").read_text()
    assert f"{exps.GITHUB}/blob/{sha}/docs/thing.md" in card
    assert "on commit `1c60e59`" in card and "on branch" not in card


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


# ---------------------------------------------------------------- pinning a checkpoint


def _step(tmp_path: Path, files: dict[str, bytes], digest: str | None = None) -> Path:
    from gallops import checkpoint as ck

    d = tmp_path / "weights" / "m" / "r" / "step_000000003"
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    ck.write(
        d, ck.Checkpoint(model="m", step=3, train_git_sha="abc1234", train_run="r", weights_sha256=digest)
    )
    return d


def _pin(d: Path, *args: object):
    """`gallops experiments checkpoint d ...`, with a site whose weights root is beside the step dir's."""
    try:
        from typer.testing import CliRunner

        from gallops.cli import app
    except (ImportError, TypeError) as exc:
        # The preflight runs this file under Drive-JEPA's Python 3.9, where the CLI cannot load (recipe.py's
        # tomllib, then typer on `X | None`) and is never run. Anywhere with tomllib, the interpreter
        # make check uses, a failed import is a failure and must not turn into a skip.
        if importlib.util.find_spec("tomllib") is not None:
            raise
        pytest.skip(f"gallops.cli does not load on this interpreter: {exc}")

    site = d.parents[3]
    config = site / "gallops.toml"
    config.write_text(f'[roots]\nweights = "{site / "weights"}"\nruns = "{site / "runs"}"\n')
    return CliRunner().invoke(
        app, ["experiments", "checkpoint", str(d), *map(str, args)], env={"GALLOPS_CONFIG": str(config)}
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_pinning_refuses_to_guess_which_weights_file_is_the_checkpoint(tmp_path) -> None:
    """The swerve trainer writes model.pt and agent.ckpt; paddock loads agent.ckpt, the smaller one."""
    d = _step(tmp_path, {"agent.ckpt": b"A" * 10, "model.pt": b"M" * 20})
    out = _pin(d)
    assert out.exit_code == 2 and "agent.ckpt, model.pt" in out.output
    out = _pin(d, "--file", "agent.ckpt")
    assert out.exit_code == 0, out.output
    assert f'weights_sha256 = "{_sha(b"A" * 10)}"' in out.output and 'file = "agent.ckpt"' in out.output
    assert f'"{_sha(b"M" * 20)}  model.pt"' in out.output  # the other file is kept, with its digest


def test_file_must_name_a_weights_file(tmp_path) -> None:
    """A scorer.json beside the weights exists, and is not what gets pinned."""
    d = _step(tmp_path, {"reranker.npz": b"w"})
    (d / "scorer.json").write_text("{}")
    for name in ("scorer.json", "nothing.pt"):
        out = _pin(d, "--file", name)
        assert out.exit_code == 2 and "is not a weights file" in out.output, name
        assert out.exception is None or isinstance(out.exception, SystemExit), out.exception


def test_an_ensemble_is_pinned_by_its_members_together(tmp_path) -> None:
    seeds = {f"seed_{i}.pt": bytes([i]) * 8 for i in range(3)}
    out = _pin(_step(tmp_path, seeds), "--ensemble")
    assert out.exit_code == 0, out.output
    listing = "".join(
        f"{_sha(data)}  {name}\n" for name, data in sorted(seeds.items())
    )  # sha256sum seed_*.pt
    assert f'weights_sha256 = "{_sha(listing.encode())}"' in out.output
    assert "file =" not in out.output


def test_an_ensemble_of_one_file_is_refused(tmp_path) -> None:
    """One file gives no members, so the digest would be over nothing: the same for every such directory."""
    out = _pin(_step(tmp_path, {"seed_0.pt": b"one seed"}), "--ensemble")
    assert out.exit_code == 2 and "two or more" in out.output
    assert _sha(b"") not in out.output


def test_the_digest_of_nothing_pins_nothing(tmp_path) -> None:
    block = f'  weights_sha256 = "{SHA}"\n'
    empty = RECORD.replace(block, f'  weights_sha256 = "{_sha(b"")}"\n', 1)
    reg = exps.Registry.load(_repo(tmp_path, empty))
    assert any("SHA-256 of empty input" in e for e in reg.errors), reg.errors


def test_a_sidecar_digest_is_checked_against_the_bytes_not_copied(tmp_path) -> None:
    d = _step(tmp_path, {"reranker.npz": b"rewritten after the sidecar"}, digest="0" * 64)
    out = _pin(d)
    assert out.exit_code == 1 and f"hashes to {_sha(b'rewritten after the sidecar')}" in out.output
    out = _pin(d, "--trust-sidecar")
    assert out.exit_code == 0 and "NOT re-hashed" in out.output


def test_location_is_the_layout_configs_and_a_stray_directory_is_flagged(tmp_path) -> None:
    home = _step(tmp_path, {"reranker.npz": b"x"})
    out = _pin(home)
    assert out.exit_code == 0 and "warning" not in out.output
    assert 'location = "{weights}/m/r/step_000000003/"' in out.output
    stray = tmp_path / "copies" / "m" / "r" / "step_000000003"
    stray.parent.mkdir(parents=True)
    home.rename(stray)
    (tmp_path / "copies" / "weights").mkdir()
    out = _pin(stray)
    assert out.exit_code == 0 and "is not where the layout puts this checkpoint" in out.output
    assert 'location = "{weights}/m/r/step_000000003/"' in out.output


@pytest.mark.parametrize(
    ("members", "file", "error"),
    [
        ([f"{SHA}  agent.ckpt", f"{'b' * 64}  model.pt"], "agent.ckpt", None),
        ([f"{'c' * 64}  agent.ckpt", f"{'b' * 64}  model.pt"], "agent.ckpt", "not the digest members give"),
        ([f"{'c' * 64}  seed_0.pt", f"{'b' * 64}  seed_1.pt"], None, "not the digest over members"),
        ([f"{SHA} agent.ckpt"], "agent.ckpt", "as sha256sum prints them"),
    ],
)
def test_members_agree_with_the_pinned_digest(tmp_path, members, file, error) -> None:
    block = f'  weights_sha256 = "{SHA}"\n'
    extra = (f'  file = "{file}"\n' if file else "") + f"  members = {members!r}\n".replace("'", '"')
    reg = exps.Registry.load(_repo(tmp_path, RECORD.replace(block, block + extra, 1)))
    if error is None:
        assert reg.errors == []
    else:
        assert any(error in e for e in reg.errors), reg.errors
