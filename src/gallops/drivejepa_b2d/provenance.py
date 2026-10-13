"""Which commit of this repo produced an output.

Stdlib only and valid Python 3.9: the generator stage records it from inside
Drive-JEPA's interpreter, where gallops is not installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_sha() -> str:
    """HEAD of this checkout, suffixed `-dirty` when a tracked file differs from it.

    A dirty sha is still recorded rather than refused: a stage run from an
    uncommitted edit is worth knowing about, not worth losing.
    """

    def git(*args: str) -> str:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
        return out.stdout.strip()

    sha = git("rev-parse", "HEAD") or "unknown"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
