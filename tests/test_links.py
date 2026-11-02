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


def test_every_relative_markdown_link_resolves() -> None:
    broken = []
    for path in _markdown_files():
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            file = target.split("#", 1)[0]
            if not (path.parent / file).exists():
                broken.append(f"{path.relative_to(REPO_ROOT)}: {target}")
    assert broken == [], "\n".join(broken)
