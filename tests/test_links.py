"""Every relative link in a Markdown file points at a file that exists.

The experiment accounts move between folders as the registry settles (docs/experiments.md), and a link
that silently stops resolving is how a reader loses the evidence behind a number. External links and
in-page anchors are not followed; GitHub links to an unmerged branch are external by design.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _markdown_files() -> list[Path]:
    # Tracked plus not-yet-added files, the same set tests/test_portability.py scans.
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if (REPO_ROOT / line).is_file()]


def broken_links(root: Path, paths: list[Path]) -> list[str]:
    """Relative links in `paths` that do not land on a file inside `root`.

    Inside, not merely existing: a link to a sibling checkout (`../paddock`) resolves only where the
    two repos happen to sit side by side, so `make check` would pass or fail by directory layout.
    """
    root = root.resolve()
    broken = []
    for path in paths:
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            file = (path.parent / target.split("#", 1)[0]).resolve()
            where = path.relative_to(root)
            if not file.is_relative_to(root):
                broken.append(f"{where}: {target} (outside the repository; link it by URL)")
            elif not file.exists():
                broken.append(f"{where}: {target}")
    return broken


def test_every_relative_markdown_link_resolves() -> None:
    broken = broken_links(REPO_ROOT, _markdown_files())
    assert broken == [], "\n".join(broken)


def test_a_link_out_of_the_repository_is_refused_even_where_it_resolves(tmp_path) -> None:
    """The case that passed beside a paddock checkout and failed everywhere else."""
    (tmp_path / "paddock").mkdir()
    repo = tmp_path / "gallops"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "x.md").write_text("ok")
    doc = repo / "CLAUDE.md"
    doc.write_text("[Paddock](../paddock) and [x](docs/x.md) and [web](https://example.org)")
    assert broken_links(repo, [doc]) == ["CLAUDE.md: ../paddock (outside the repository; link it by URL)"]
