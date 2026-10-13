"""Stage check-data: is this site's copy of the recipe's data the one the models were trained on?

The training inputs are a frozen dump (one directory, [datasets.drivejepa_b2d] in gallops.toml); its file
list, sizes and sha256 are tracked in recipes/drivejepa_b2d/data_*.json. Missing or resized files fail in
seconds; --hash also compares contents, which reads every byte (5 GB for Split-v4 with the released
checkpoint, 33 GB with the Final).

A stage that silently trained on a regenerated or half-copied file would produce a plausible checkpoint that
is not the recorded model, so the recipe runs this first. Running named stages only (`gallops recipe
drivejepa_b2d <stage> --run`) skips it unless it is named too. Stdlib only.

    python -m gallops.drivejepa_b2d.check_data --root <dump> --manifest <data_*.json> [--hash]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the dump directory")
    ap.add_argument(
        "--manifest", type=Path, required=True, action="append", help="tracked data_*.json; repeatable"
    )
    ap.add_argument("--hash", action="store_true", help="also compare sha256 (reads every file)")
    a = ap.parse_args()
    files: dict = {}
    for m in a.manifest:
        files.update(json.loads(m.read_text())["files"])
    bad = []
    for rel, want in sorted(files.items()):
        path = a.root / rel
        if not path.is_file():
            bad.append(f"missing  {rel}")
        elif path.stat().st_size != want["bytes"]:
            bad.append(f"size     {rel}: {path.stat().st_size} != {want['bytes']}")
        elif a.hash and sha256(path) != want["sha256"]:
            bad.append(f"content  {rel}: sha256 differs")
    total = sum(f["bytes"] for f in files.values())
    print(
        f"{len(files)} files, {total / 1e9:.2f} GB under {a.root}: {len(bad)} problem(s)"
        f"{'' if a.hash else ' (sizes only; --hash compares contents)'}"
    )
    for line in bad[:50]:
        print("  " + line)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
