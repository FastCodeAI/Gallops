"""`compat.zip_strict` is `zip(..., strict=True)` on every interpreter a trainer runs in."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from gallops import compat
from gallops.config import REPO_ROOT

CASES = [
    ([1, 2], ["a", "b"]),
    ([1, 2], ["a"]),
    ([1], ["a", "b"]),
    ([1, 2], [3, 4], [5]),
    ([1], [3, 4], [5, 6]),
    ([1, 2], [3], [5, 6]),
    ([], []),
    ([1, 2, 3],),
]


def _outcome(fn, *args):
    try:
        return ("ok", list(fn(*args)))
    except ValueError as exc:
        return ("error", str(exc))


@pytest.mark.parametrize("args", CASES)
def test_fallback_matches_expected(args):
    expected = {
        0: ("ok", [(1, "a"), (2, "b")]),
        1: ("error", "zip() argument 2 is shorter than argument 1"),
        2: ("error", "zip() argument 2 is longer than argument 1"),
        3: ("error", "zip() argument 3 is shorter than arguments 1-2"),
        4: ("error", "zip() argument 2 is longer than argument 1"),
        5: ("error", "zip() argument 2 is shorter than argument 1"),
        6: ("ok", []),
        7: ("ok", [(1,), (2,), (3,)]),
    }[CASES.index(args)]
    assert _outcome(compat._zip_strict, *args) == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="the built-in to compare with is 3.10+")
@pytest.mark.parametrize("args", CASES)
def test_fallback_matches_builtin(args):
    builtin = _outcome(lambda *a: zip(*a, strict=True), *args)
    assert _outcome(compat._zip_strict, *args) == builtin
    assert _outcome(compat.zip_strict, *args) == builtin


def test_no_arguments():
    assert list(compat.zip_strict()) == []


# The two places that call the built-in on purpose: the shim itself (behind its version check) and
# the test above that compares the shim with the built-in (skipped below 3.10).
_BUILTIN_ON_PURPOSE = {"src/gallops/compat.py", "tests/test_compat.py"}


def strict_zips(path: Path) -> list[int]:
    """Lines of `path` that call `zip(..., strict=...)`, which is a TypeError on Python 3.9."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and any(k.arg == "strict" for k in node.keywords)
    ]


def test_strict_zips_are_found(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("a = list(zip(x, y))\nb = list(zip(\n    x, y, strict=True\n))\n")
    assert strict_zips(sample) == [2]


def test_no_strict_zip_outside_the_shim():
    """The preflight runs every test under the training interpreter (Python 3.9), so one
    `zip(..., strict=True)` anywhere in `src/` or `tests/` refuses every training run there.
    Use `compat.zip_strict`. Merges from `main` brought such calls in once already."""
    found = {
        f"{rel}:{line}"
        for path in sorted([*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "tests").rglob("*.py")])
        if (rel := path.relative_to(REPO_ROOT).as_posix()) not in _BUILTIN_ON_PURPOSE
        for line in strict_zips(path)
    }
    assert not found, f"zip(..., strict=) fails on Python 3.9; use gallops.compat.zip_strict: {sorted(found)}"
