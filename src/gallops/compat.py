"""Small stand-ins for Python 3.10+ built-ins, for code that also runs in a model's own interpreter.

gallops itself targets 3.11, but a trainer runs inside the environment its model was built for, and
Drive-JEPA's is Python 3.9. A trainer's preflight runs the whole test suite in that interpreter
(`train/preflight.py`), so a module that only imports on 3.10+ fails it even when the trainer never
calls that module. These helpers keep the 3.10+ behaviour, error included, on both.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import Any


def zip_strict(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """`zip(*iterables, strict=True)`, on Python 3.9 as well.

    Raises the same ValueError, with the same message, as the built-in when the
    lengths differ. Lazy like `zip`: the error surfaces when the shorter input runs out.
    """
    if sys.version_info >= (3, 10):  # noqa: UP036 - the block exists for 3.9, below the project's floor
        return zip(*iterables, strict=True)
    return _zip_strict(*iterables)


def _span(n: int) -> str:
    """CPython's wording for the first n arguments: "argument 1" or "arguments 1-n"."""
    return "argument 1" if n == 1 else f"arguments 1-{n}"


def _zip_strict(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    # Pulls from each input in order and stops at the first one that runs out, as the built-in does,
    # so later inputs are not advanced past the point the error is raised.
    iterators = [iter(it) for it in iterables]
    if not iterators:
        return
    sentinel = object()
    while True:
        items = [next(iterators[0], sentinel)]
        if items[0] is sentinel:
            # argument 1 ran out: every other input must be empty too
            for j, rest in enumerate(iterators[1:], start=1):
                if next(rest, sentinel) is not sentinel:
                    raise ValueError(f"zip() argument {j + 1} is longer than {_span(j)}")
            return
        for k, it in enumerate(iterators[1:], start=1):
            item = next(it, sentinel)
            if item is sentinel:
                raise ValueError(f"zip() argument {k + 1} is shorter than {_span(k)}")
            items.append(item)
        yield tuple(items)
