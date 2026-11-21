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
import shutil
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


def _front_page_problems(text: str) -> list[str]:
    """What keeps a README from being a front page, read from the prose around its rendered blocks.

    The tables grow a row per experiment and IDs are never retired, so a cap on the whole page would
    one day fail a PR that only registers a hypothesis. The prose is what can bloat, and can move.
    """
    for name in exps.README_BLOCKS:
        start, end = exps.markers(name)
        text = text[: text.index(start)] + text[text.index(end) + len(end) :]
    lines, size = len(text.splitlines()), len(text.encode())
    if lines <= PROSE_LINES and size <= PROSE_BYTES:
        return []
    return [
        f"README prose is {lines} lines and {size} bytes (limit {PROSE_LINES} and {PROSE_BYTES}): "
        "move the detail to a card, experiments/README.md or docs/"
    ]


def test_the_readme_stays_a_front_page() -> None:
    """One row per experiment and links out. Detail goes on cards and in experiments/README.md."""
    assert _front_page_problems((REPO_ROOT / "README.md").read_text()) == []


def test_the_front_page_cap_counts_prose_and_not_experiments() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    _, end = exps.markers("hypotheses")
    assert _front_page_problems(readme.replace(end, "| a row |\n" * 200 + end)) == []
    assert "move the detail" in _front_page_problems(readme + "prose\n" * 200)[0]


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

#: The prose around the README's rendered blocks, which stood at 72 lines and 3.1 KB when this cap was set.
PROSE_LINES, PROSE_BYTES = 90, 6 * 1024
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
verdict = "inconclusive"
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
            "a done hypothesis with no verdict",
            'verdict = "inconclusive"',
            'verdict = "none"',
            "how it came out",
        ),
        ("positive with no SHIP", 'verdict = "inconclusive"', 'verdict = "positive"', "has a gate SHIP"),
        ("a gate verdict paddock never prints", 'verdict = "HOLD"', 'verdict = "INCOMPLETE"', "INCOMPLETE"),
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


def test_a_finished_hypothesis_was_gated_and_decided(tmp_path) -> None:
    i = RECORD.index("  [[runs.gate]]")
    reg = exps.Registry.load(_repo(tmp_path / "a", RECORD[:i]))
    assert any("has a gate verdict" in e for e in reg.errors), reg.errors
    reg = exps.Registry.load(_repo(tmp_path / "b", decisions=""))
    assert any("cited as evidence by a decision" in e for e in reg.errors), reg.errors


def test_the_decision_that_closes_a_hypothesis_came_after_its_gate(tmp_path) -> None:
    """An older decision cannot have said what follows from a gate run after it, as D11 could not for H01."""
    before = DECISIONS.replace("date = 2026-10-29", "date = 2026-10-27")
    reg = exps.Registry.load(_repo(tmp_path / "a", decisions=before))
    assert any("cited as evidence by a decision" in e for e in reg.errors), reg.errors
    same_day = DECISIONS.replace("date = 2026-10-29", "date = 2026-10-28")
    assert exps.Registry.load(_repo(tmp_path / "b", decisions=same_day)).errors == []


def test_one_gated_run_is_enough_among_several(tmp_path) -> None:
    earlier = '[[runs]]\nid = "H01.r0"\nlabel = "an earlier run, never gated"\n\n'
    record = RECORD.replace("[[runs]]\n", earlier + "[[runs]]\n", 1)
    assert exps.Registry.load(_repo(tmp_path, record)).errors == []


def test_a_bad_evidence_entry_is_an_error_and_not_a_traceback(tmp_path) -> None:
    decisions = DECISIONS.replace('evidence = ["H01.r1"]', "evidence = [1]")
    reg = exps.Registry.load(_repo(tmp_path, decisions=decisions))
    assert any("1 is not an experiment or run" in e for e in reg.errors), reg.errors


@pytest.mark.parametrize(
    ("trained", "shared"),
    [("trained = []", True), ('trained = ["scorer"]', False), ("", False)],
)
def test_only_a_run_that_trained_nothing_may_share_a_digest(tmp_path, trained, shared) -> None:
    """Paddock's gate accepts a code change on the same weights (H08); two trainings cannot write one file."""
    second = f"""
[[runs]]
id = "H01.r2"
label = "a code change on the same weights"
{trained}

  [runs.checkpoint]
  model = "drivejepa"
  train_run = "r2"
  weights_sha256 = "{SHA}"
"""
    errors = exps.Registry.load(_repo(tmp_path, RECORD + second)).errors
    assert (errors == []) if shared else any("is also claimed by H01.r1" in e for e in errors), errors


@pytest.mark.parametrize("verdict", ["NO UNIT", "CONFIG DIFFERS", "INCOMPLETE", "SUBSET DIFFERS"])
def test_every_undecided_pair_paddock_prints_can_be_copied_in(tmp_path, verdict) -> None:
    """A HOLD caused by one of these has to be recordable, or "copy it verbatim" fails `make check`."""
    reg = exps.Registry.load(_repo(tmp_path, RECORD.replace('verdict = "NEUTRAL"', f'verdict = "{verdict}"')))
    assert reg.errors == []
    assert verdict in exps.render_card(reg, reg.experiments[0])


def test_an_unfinished_hypothesis_needs_neither(tmp_path) -> None:
    running = RECORD[: RECORD.index("  [[runs.gate]]")].replace('status = "done"', 'status = "running"')
    assert exps.Registry.load(_repo(tmp_path, running, decisions="")).errors == []


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


def test_the_picture_names_the_base_without_repeating_the_word(tmp_path) -> None:
    """The base box stands between the phases, so a date there would put 6 Nov before a phase of 16 Oct."""
    decisions = DECISIONS.replace(
        'short = "try the thing"\n', 'short = "Base: the released model"\nbaseline = "the released model"\n'
    )
    svg = exps.render_picture(exps.Registry.load(_repo(tmp_path, decisions=decisions)))
    assert ">Base</text>" in svg and ">the released model</text>" in svg and "Base: the" not in svg


def test_the_picture_does_not_draw_the_base_as_what_phase_1_produced(tmp_path) -> None:
    """The base is the released model, which predates gap closing: an arrow into its box says otherwise."""
    svg = exps.render_picture(exps.Registry.load(_repo(tmp_path)))
    assert 'x1="226"' not in svg, "no arrow leaves the Phase 1 box"
    assert 'x1="430"' in svg and 'x1="670"' in svg, "the base still feeds Phase 2, and Phase 2 the gate"


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


def _cli_module():
    """gallops.cli, or skip where it cannot load."""
    try:
        from gallops import cli
    except (ImportError, TypeError) as exc:
        # The preflight runs this file under Drive-JEPA's Python 3.9, where the CLI cannot load (recipe.py's
        # tomllib, then typer on `X | None`) and is never run. Anywhere with tomllib, the interpreter
        # make check uses, a failed import is a failure and must not turn into a skip.
        if importlib.util.find_spec("tomllib") is not None:
            raise
        pytest.skip(f"gallops.cli does not load on this interpreter: {exc}")
    return cli


def _cli(args: list[str], env: dict[str, str] | None = None):
    """Invoke the CLI, or skip where it cannot load."""
    app = _cli_module().app
    from typer.testing import CliRunner

    return CliRunner().invoke(app, args, env=env)


def _pin(d: Path, *args: object):
    """`gallops experiments checkpoint d ...`, with a site whose weights root is beside the step dir's."""
    site = d.parents[3]
    config = site / "gallops.toml"
    config.write_text(f'[roots]\nweights = "{site / "weights"}"\nruns = "{site / "runs"}"\n')
    return _cli(["experiments", "checkpoint", str(d), *map(str, args)], {"GALLOPS_CONFIG": str(config)})


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


# ---------------------------------------------------------------- results PRs

_TITLE = "H01.r1: NAVSIM EPDMS 0.875 → 0.885, gate HOLD"
_NO_GATE = RECORD.split("  [[runs.gate]]")[0]
_LATER_DECISION = """
[[decision]]
id = "D03"
date = 2026-10-30
short = "drop the thing"
decision = "Drop the thing."
by = ["Naren"]
evidence = ["H01"]
"""
_SECOND_RUN = """
[[runs]]
id = "H01.r2"
label = "second run"
start_from = "released"

  [runs.code]
  modules = ["gallops.train.thing"]

  [[runs.results]]
  bench = "navsim"
  metric = "EPDMS"
  value = "0.890"
  better = "higher"
  strength = "argued"
"""


def _rendered(reg: exps.Registry, ref: str = "H01.r1") -> tuple[str, str]:
    """A results PR as an author leaves it: the record's title and body, with the TODOs written."""
    title, _, body = exps.render_pr(reg, ref).partition("\n\n")
    return title, "\n".join(
        "Written." if line.startswith(exps.PR_TODO) else line for line in body.splitlines()
    )


def test_a_results_pr_is_rendered_from_its_record(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    title, body = _rendered(reg)
    assert title == _TITLE
    assert "<!-- gallops:pr H01.r1 -->" in body
    assert "- **Hypothesis:** It raises EPDMS by 0.01; a NEUTRAL with MDE under 0.01 refutes it." in body
    assert "- **Result:** NAVSIM EPDMS 0.875 → **0.885** · ●●○○ moderate" in body
    assert "- **Gate:** ⏸ **HOLD**, 28 Oct 2026, quick tier" in body
    assert "- **Decision:** D02: keep the thing" in body
    assert "- **Record:** `experiments/H01_thing/CARD.md`" in body
    assert "Why the gate has not run" not in body
    assert exps.render_pr(reg, "H01") == exps.render_pr(reg, "H01.r1"), "an experiment means its headline"
    assert exps.check_pr(reg, title, body, ["H01.r1"]) == []


def test_a_run_with_no_gate_says_so_and_asks_why(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path, _NO_GATE))
    text = exps.render_pr(reg, "H01.r1")
    assert text.splitlines()[0] == "H01.r1: NAVSIM EPDMS 0.875 → 0.885, gate not run"
    assert "- **Gate:** not run" in text
    assert f"## Why the gate has not run\n\n{exps.PR_TODO}" in text


def test_a_run_with_a_gate_and_no_result_can_be_reported(tmp_path) -> None:
    """H02.timing holds a gate verdict and no result."""
    results = RECORD[RECORD.index("  [[runs.results]]") : RECORD.index("  [[runs.gate]]")]
    reg = exps.Registry.load(_repo(tmp_path, RECORD.replace(results, "")))
    text = exps.render_pr(reg, "H01.r1")
    assert text.splitlines()[0] == "H01.r1: gate HOLD"
    assert "- **Result:** none in this run" in text


def test_a_run_reports_its_own_result_when_the_headline_is_elsewhere(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path, RECORD + _SECOND_RUN))
    assert exps.render_pr(reg, "H01.r2").splitlines()[0] == "H01.r2: NAVSIM EPDMS 0.890, gate not run"
    assert exps.render_pr(reg, "H01").splitlines()[0] == _TITLE


@pytest.mark.parametrize(
    ("ref", "error"),
    [("H99", "neither an experiment nor a run"), ("H01.r9", "neither an experiment nor a run")],
)
def test_a_pr_names_something_the_registry_holds(tmp_path, ref, error) -> None:
    with pytest.raises(ValueError, match=error):
        exps.render_pr(exps.Registry.load(_repo(tmp_path)), ref)


def test_a_record_with_nothing_to_report_has_no_results_pr(tmp_path) -> None:
    start = RECORD.index("  [[runs.results]]")
    reg = exps.Registry.load(_repo(tmp_path, RECORD[:start]))
    with pytest.raises(ValueError, match="no headline result"):
        exps.render_pr(reg, "H01")
    with pytest.raises(ValueError, match="not a results PR"):
        exps.render_pr(reg, "H01.r1")


@pytest.mark.parametrize(
    ("break_it", "error"),
    [
        (lambda t, b: ("H01.r1: EPDMS is up", b), "the title must be"),
        (lambda t, b: (t, b.replace("**0.885**", "**0.895**")), "differs from the record"),
        (lambda t, b: (t, b.replace("HOLD", "SHIP")), "differs from the record"),
        (lambda t, b: (t, b + f"\n{exps.PR_TODO} later\n"), "is still in the body"),
        (lambda t, b: (t, b + "\nline" * (exps.PR_PROSE_LINES + 1)), "hand-written lines"),
        (lambda t, b: (t, b + b), "2 rendered blocks"),
        (lambda t, b: (t, b.replace("gallops:pr H01.r1", "gallops:pr H77.r1")), "neither an experiment"),
    ],
)
def test_a_results_pr_that_differs_from_its_record_is_refused(tmp_path, break_it, error) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    title, body = break_it(*_rendered(reg))
    problems = exps.check_pr(reg, title, body, ["H01.r1"])
    assert any(error in p for p in problems), problems


def test_a_run_reports_its_latest_gate_however_the_record_lists_them() -> None:
    older = {"date": dt.date(2026, 10, 20), "verdict": "FAIL"}
    newer = {"date": dt.date(2026, 10, 28), "verdict": "HOLD"}
    for gates in ([older, newer], [newer, older]):
        assert exps.pr_title({"id": "H01.r1", "gate": gates}, None) == "H01.r1: gate HOLD"


def test_a_results_pr_gives_the_latest_decision_that_cites_it(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path, decisions=DECISIONS + _LATER_DECISION))
    assert "- **Decision:** D03: drop the thing" in exps.render_pr(reg, "H01.r1")


def test_a_run_holding_the_headline_reports_it_wherever_it_is_listed(tmp_path) -> None:
    first = """  [[runs.results]]
  bench = "navsim"
  metric = "EPDMS"
  value = "0.100"
  better = "higher"
  strength = "argued"

"""
    record = RECORD.replace("  [[runs.results]]\n", first + "  [[runs.results]]\n", 1)
    reg = exps.Registry.load(_repo(tmp_path, record))
    assert exps.render_pr(reg, "H01.r1").splitlines()[0] == _TITLE


def test_a_block_left_behind_by_the_base_says_to_merge_it(tmp_path) -> None:
    """A decision on main citing the run changes the merge; re-running on the branch prints the old block."""
    reg = exps.Registry.load(_repo(tmp_path))
    title, body = _rendered(reg)
    (reg.root / "experiments" / "decisions.toml").write_text(DECISIONS + _LATER_DECISION)
    assert exps.check_pr(exps.Registry.load(reg.root), title, body, ["H01.r1"]) == [
        "the rendered block differs from the record: merge the base, then re-run "
        "`gallops experiments pr H01.r1`"
    ]


def test_one_results_pr_reports_one_experiment(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    problems = exps.check_pr(reg, *_rendered(reg), ["H01.r1", "H02.r1"])
    assert problems == ["it also changes results in H02: one PR per experiment"]


def test_a_result_with_no_rendered_block_is_refused_until_it_says_why(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    assert "gallops experiments pr <ID>" in exps.check_pr(reg, "any title", "any body", ["H01.r1"])[0]
    assert (
        exps.check_pr(reg, "any title", "Fixes a flag.\nNot a results PR: a flag was wrong.", ["H01.r1"])
        == []
    )
    assert exps.check_pr(reg, "any title", "Not a results PR: ", ["H01.r1"]) != [], "a reason is required"


def test_a_pr_that_changes_no_result_is_left_alone(tmp_path) -> None:
    reg = exps.Registry.load(_repo(tmp_path))
    assert exps.check_pr(reg, "docs: a typo", "", []) == []


def test_a_body_saved_by_the_web_editor_still_matches(tmp_path) -> None:
    """GitHub's editor saves CRLF, which would otherwise fail every PR someone touched up in the browser."""
    reg = exps.Registry.load(_repo(tmp_path))
    title, body = _rendered(reg)
    assert exps.check_pr(reg, title, body.replace("\n", "\r\n"), ["H01.r1"]) == []
    assert exps.check_pr(reg, title + " ", body, ["H01.r1"]) == [], "so may a trailing space in the title"


def test_the_pr_template_does_not_pass_for_a_results_pr(tmp_path) -> None:
    """Leaving the template in place says nothing, and neither does a reason hidden in a comment."""
    reg = exps.Registry.load(_repo(tmp_path))
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text()
    assert exps.check_pr(reg, "any title", template, ["H01.r1"]) != []
    hidden = "<!--\nNot a results PR: hidden from the reviewer\n-->"
    assert exps.check_pr(reg, "any title", hidden, ["H01.r1"]) != [], "the reason has to be visible"


def _commit(root: Path, message: str) -> None:
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root)]
    if not (root / ".git").exists():
        subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", message, "--allow-empty"], check=True)


def _with_base(tmp_path: Path, record: str | None = RECORD) -> Path:
    root = _repo(tmp_path, record)
    template = root / "experiments" / "_template"
    template.mkdir()
    (template / "experiment.toml").write_text("started = {{DATE}}\n")  # not TOML, like the real one
    _commit(root, "base")
    return root


def test_changing_a_result_or_a_gate_is_what_makes_a_results_pr(tmp_path) -> None:
    root = _with_base(tmp_path)
    record = root / "experiments" / "H01_thing" / "experiment.toml"
    for old, new in [('value = "0.885"', 'value = "0.886"'), ('verdict = "HOLD"', 'verdict = "FAIL"')]:
        record.write_text(RECORD.replace(old, new))
        assert exps.changed_results(exps.Registry.load(root), "HEAD") == ["H01.r1"], new


def test_registering_an_experiment_is_not_a_results_pr_until_it_reports_something(tmp_path) -> None:
    """H19 (then H15) was registered, with a run and its predictions, before anything was graded."""
    root = _with_base(tmp_path, None)
    folder = root / "experiments" / "H01_thing"
    folder.mkdir()
    (folder / "experiment.toml").write_text(RECORD[: RECORD.index("  [[runs.results]]")])
    assert exps.changed_results(exps.Registry.load(root), "HEAD") == []
    (folder / "experiment.toml").write_text(RECORD)
    assert exps.changed_results(exps.Registry.load(root), "HEAD") == ["H01.r1"]


def test_a_result_that_is_removed_is_a_change_too(tmp_path) -> None:
    """IDs are marked `dropped`, not freed, so a record that vanishes needs a reason someone can read."""
    root = _with_base(tmp_path)
    shutil.rmtree(root / "experiments" / "H01_thing")
    assert exps.changed_results(exps.Registry.load(root), "HEAD") == ["H01.r1"]


def test_a_results_pr_reports_a_run_it_changed(tmp_path) -> None:
    """A title that names H01.r1 must not stand for a PR that only moved H01.r2's number."""
    root = _with_base(tmp_path, RECORD + _SECOND_RUN)
    record = root / "experiments" / "H01_thing" / "experiment.toml"
    record.write_text((RECORD + _SECOND_RUN).replace('value = "0.890"', 'value = "0.891"'))
    reg = exps.Registry.load(root)
    changed = exps.changed_results(reg, "HEAD")
    assert changed == ["H01.r2"]
    assert exps.check_pr(reg, *_rendered(reg, "H01.r2"), changed) == []
    assert exps.check_pr(reg, *_rendered(reg, "H01.r1"), changed) == [
        "it reports H01.r1 but changes H01.r2: report a run it changed"
    ]
    assert exps.check_pr(reg, *_rendered(reg, "H01.r1"), ["H01.r1", "H01.r2"]) == [], "one of several is fine"


def test_what_a_record_says_around_its_results_is_no_change(tmp_path) -> None:
    root = _with_base(tmp_path)
    folder = root / "experiments" / "H01_thing"
    (folder / "experiment.toml").write_text(
        "# a new comment\n" + RECORD.replace('learned = "It did."', 'learned = "It did, a little."')
    )
    assert exps.changed_results(exps.Registry.load(root), "HEAD") == []


def test_a_folder_that_moves_reports_nothing_new(tmp_path) -> None:
    root = _with_base(tmp_path)
    (root / "experiments" / "H01_thing").rename(root / "experiments" / "H01_renamed")
    assert exps.changed_results(exps.Registry.load(root), "HEAD") == []


def test_pr_prints_the_title_then_the_body_and_pr_check_reads_the_environment(tmp_path, monkeypatch) -> None:
    """On a repository of its own: the real one holds uncommitted results while someone records them."""
    root = _with_base(tmp_path)
    monkeypatch.setattr(_cli_module(), "_registry", lambda: exps.Registry.load(root))
    printed = _cli(["experiments", "pr", "H01.r1"])
    assert printed.exit_code == 0
    assert printed.output.splitlines()[:3] == [_TITLE, "", "<!-- gallops:pr H01.r1 -->"]
    assert _cli(["experiments", "pr", "H99"]).exit_code == 2

    def check(title: str, body: str):
        return _cli(["experiments", "pr-check", "--base", "HEAD"], {"PR_TITLE": title, "PR_BODY": body})

    written = printed.output.partition("\n\n")[2].replace(exps.PR_TODO, "Written:")
    assert check(_TITLE, written).exit_code == 0
    assert check("H01.r1: better", written).exit_code == 1
    assert _cli(["experiments", "pr-check", "--base", "no-such-commit"]).exit_code == 2


# ---------------------------------------------------------------- CI


def _workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text())


def test_ci_runs_make_check_and_holds_no_gate_of_its_own() -> None:
    """Anything that can reject work belongs in the Makefile, so a green run here is a green run there."""
    steps = _workflow("check.yml")["jobs"]["check"]["steps"]
    assert [s["run"] for s in steps if "run" in s] == ["make setup", "make check"]


def test_a_prs_text_never_reaches_a_shell() -> None:
    """`${{ }}` inside `run` is pasted into the script before it runs, so a PR title could run commands."""
    for name in ("check.yml", "results-pr.yml"):
        for job in _workflow(name)["jobs"].values():
            assert not [s["run"] for s in job["steps"] if "${{" in s.get("run", "")], name
