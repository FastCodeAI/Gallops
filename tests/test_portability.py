"""The portability gate.

Gallops will be run on more than one machine — a workstation, a cluster node,
whatever an OEM engagement puts in front of us — and a single machine-specific
path in a tracked file is enough to make the repo work here and fail there. The
failure lands on whoever tries second, with no clue why.

So: tracked files must not name a location that exists on only one machine.
Where something lives belongs in `gallops.toml`, which is gitignored. Tracked
files refer to locations through template variables ({weights}, {runs},
{data.NAME}) that `gallops.config` expands.

Placeholders are fine — `/path/to/nuscenes` in an example hurts nobody. What is
banned is a real root from a real machine.

This is paddock's gate, adopted on purpose. It is the single mechanism that has
kept that repo working at a second site, and a training repo has strictly more
opportunity to hardcode a path than an evaluation one does.

If this fails on a legitimate case, the fix is nearly always a template
variable, not an entry in EXEMPT.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roots that belong to one machine rather than to any Linux box.
MACHINE_ROOTS = (
    "/shakaal",
    "/spiral",
    "/datadrive",
    "/mnt/",
    "/home/",
    "/proj/",
    "/scratch/",
    "/Users/",
)

# A path under one person's home directory, written without the root above:
# "$HOME/<their checkout>" names one machine's layout exactly as "/home/<them>/"
# does. Dotfiles are left alone -- "~/.cache" is the same place for everybody.
HOME_RELATIVE = re.compile(r"(?:\$HOME|\$\{HOME\}|(?<![\w/.])~)/(?![.{$<])[\w-]")

# Universal on any Linux box, so not a portability risk.
ALLOWED = (
    "/usr/",
    "/bin/",
    "/tmp/",
    "/etc/",
    "/dev/null",
    "/proc/",
)

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".pkl",
    ".ckpt",
    ".onnx",
    ".pt",
    ".safetensors",
    ".gz",
    ".tar",
    ".pyc",
}

# This file necessarily contains the very patterns it bans.
EXEMPT = {"tests/test_portability.py"}


def tracked_files() -> list[Path]:
    """Files git would ship: tracked, plus untracked ones that are not ignored.

    Untracked-but-not-ignored matters: a file staged for the next commit should
    fail this gate now, not after it is pushed.
    """
    if not (REPO_ROOT / ".git").exists():
        # A fresh checkout before `git init`. Fall back to walking the tree so
        # the gate still means something in the first minutes of the repo.
        return [
            p
            for p in REPO_ROOT.rglob("*")
            if p.is_file() and not any(part.startswith(".") or part == "runs" for part in p.parts)
        ]
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def offending_lines(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel in EXEMPT or path.suffix in BINARY_SUFFIXES:
        return []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        for root in MACHINE_ROOTS:
            index = line.find(root)
            if index == -1:
                continue
            if any(allowed in line[index:] for allowed in ALLOWED):
                continue
            hits.append(f"{rel}:{number}: {line.strip()}")
            break
        else:
            if HOME_RELATIVE.search(line):
                hits.append(f"{rel}:{number}: {line.strip()}")
    return hits


def test_no_machine_paths_in_tracked_files() -> None:
    hits = [hit for path in tracked_files() for hit in offending_lines(path)]
    assert not hits, (
        "Tracked files must not name a machine-specific path.\n"
        "Use a template variable ({weights}, {runs}, {data.NAME}) and put the real\n"
        "location in gallops.toml, which is gitignored.\n\n" + "\n".join(hits)
    )


def test_the_gate_can_actually_see_a_violation() -> None:
    """A gate that cannot fail is not a gate.

    Paddock learned this the expensive way in a different form: a check that
    silently passes is worse than no check, because people stop looking.
    """
    probe = REPO_ROOT / "tests" / "_probe_not_written.py"
    assert offending_lines(probe) == [], "probe file should not exist"
    # Confirm the matcher itself works, without touching the filesystem.
    line = "root = '/home/someone/data'"
    assert any(r in line for r in MACHINE_ROOTS)
    assert HOME_RELATIVE.search("A=$HOME/some-checkout/tools")
    assert HOME_RELATIVE.search("cd ~/work/repo")
    assert not HOME_RELATIVE.search("open('~/.cache/huggingface/token')")
