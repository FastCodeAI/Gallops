"""`compat.zip_strict` is `zip(..., strict=True)` on every interpreter a trainer runs in."""

from __future__ import annotations

import sys

import pytest

from gallops import compat

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
