"""Stages code-run2 and code-run3a: rebuild the Drive-JEPA tree a run trained in.

Upstream Drive-JEPA at the commit the manifest pins, exported from a site's checkout with `git archive` (the
checkout itself is not touched: no worktree, no branch), then the run's overlay laid over it --
recipes/drivejepa_wajepa/overlay/<run>/, the files that differ from upstream, as they ran.

Every overlay file is checked against the manifest's sha256 before it is copied, so a tree built from an
edited overlay is refused rather than trained in. The manifest also records each file's sha256 as it ran; the
two differ only where a machine path was replaced by an environment variable (the manifest's note says
which), because a tracked file may not name a machine root (tests/test_portability.py).

The tree is written to a staging directory and renamed into place, so --out is either complete or absent, and
an existing --out is refused. OVERLAY.json beside navsim_v2/ records the manifest and this repo's commit.

    python -m gallops.drivejepa_wajepa.code --drivejepa <checkout> --overlay <recipes/.../overlay/run2>
        --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from gallops.drivejepa_b2d.provenance import git_sha


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(overlay: Path) -> dict:
    manifest = overlay.with_suffix(".json")
    if not manifest.is_file():
        raise SystemExit(f"no manifest {manifest} beside the overlay {overlay}")
    return json.loads(manifest.read_text())


def check_overlay(overlay: Path, manifest: dict) -> list[str]:
    """The overlay's files, each matching the manifest. Refuses a missing, extra or edited file."""
    listed = sorted(manifest["files"])
    present = sorted(str(p.relative_to(overlay)) for p in overlay.rglob("*") if p.is_file())
    if listed != present:
        raise SystemExit(
            f"overlay {overlay} and its manifest disagree: only in the manifest "
            f"{sorted(set(listed) - set(present))}, only on disk {sorted(set(present) - set(listed))}"
        )
    edited = [rel for rel in listed if sha256(overlay / rel) != manifest["files"][rel]["sha256"]]
    if edited:
        raise SystemExit(f"overlay files differ from the manifest: {edited}")
    return listed


def export(checkout: Path, sha: str, dest: Path) -> None:
    """`git archive <sha>` of the checkout, unpacked into dest."""
    proc = subprocess.Popen(
        ["git", "-C", str(checkout), "archive", "--format=tar", sha], stdout=subprocess.PIPE
    )
    assert proc.stdout is not None
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(dest, filter="data")
    if proc.wait() != 0:
        raise SystemExit(f"git archive {sha} failed in {checkout}: is the commit in that checkout?")


def build(checkout: Path, overlay: Path, out: Path, sha: str | None = None) -> dict:
    manifest = load_manifest(overlay)
    files = check_overlay(overlay, manifest)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists; a run's tree is built once, into a new directory")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pinned = sha or manifest["upstream"]["sha"]
    export(checkout, pinned, tmp)
    replaced = []
    for rel in files:
        dest = tmp / rel
        if dest.exists():
            replaced.append(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay / rel, dest)
    record = {
        "run": manifest["run"],
        "upstream": {**manifest["upstream"], "sha": pinned},
        "overlay": {rel: manifest["files"][rel]["sha256"] for rel in files},
        "replaced_upstream_files": replaced,
        "gallops_git_sha": git_sha(),
    }
    (tmp / "OVERLAY.json").write_text(json.dumps(record, indent=2) + "\n")
    tmp.rename(out)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--drivejepa", type=Path, required=True, help="a Drive-JEPA checkout holding the pinned commit"
    )
    ap.add_argument("--overlay", type=Path, required=True, help="recipes/drivejepa_wajepa/overlay/<run>")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    record = build(a.drivejepa, a.overlay, a.out)
    print(
        f"{a.out}: Drive-JEPA {record['upstream']['sha'][:12]} + {len(record['overlay'])} overlay files "
        f"({len(record['replaced_upstream_files'])} replace upstream files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
