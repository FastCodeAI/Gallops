"""Stages code-run2, code-agent-decoder and code-run3a: rebuild the Drive-JEPA tree a run trained in.

Upstream Drive-JEPA at the commit the manifests pin, exported from a site's checkout with `git archive` (the
checkout itself is not touched: no worktree, no branch), then the run's overlays laid over it in order --
recipes/drivejepa_wajepa/overlay/<name>/, the files that differ from upstream, as they ran. The agent
decoder's tree is Run 2's overlay plus its own two files; Run 3A's package is self-contained.

Every overlay file is checked against the manifest's sha256 before it is copied, so a tree built from an
edited overlay is refused rather than trained in. The manifest also records each file's sha256 as it ran; the
two differ only where a machine path was replaced by an environment variable (the manifest's note says
which), because a tracked file may not name a machine root (tests/test_portability.py).

The tree is written to a staging directory and renamed into place, so --out is either complete or absent, and
an existing --out is refused. OVERLAY.json beside navsim_v2/ records every overlay's files and hashes, and
this repo's commit.

    python -m gallops.drivejepa_wajepa.code --drivejepa <checkout> --overlay <recipes/.../overlay/run2>
        [--overlay <...>] --out <dir>
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


def build(checkout: Path, overlays: list[Path], out: Path, sha: str | None = None) -> dict:
    """Upstream at the pinned commit, then each overlay in order (a later one wins on a shared file)."""
    manifests = [load_manifest(o) for o in overlays]
    files = [check_overlay(o, m) for o, m in zip(overlays, manifests, strict=True)]
    pins = {m["upstream"]["sha"] for m in manifests}
    if len(pins) != 1:
        raise SystemExit(f"the overlays pin different upstream commits: {sorted(pins)}")
    if out.exists():
        raise SystemExit(f"refusing: {out} exists; a run's tree is built once, into a new directory")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pinned = sha or pins.pop()
    export(checkout, pinned, tmp)
    replaced, layers = [], []
    for overlay, manifest, rels in zip(overlays, manifests, files, strict=True):
        for rel in rels:
            dest = tmp / rel
            if dest.exists():
                replaced.append(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(overlay / rel, dest)
        layers.append(
            {"run": manifest["run"], "files": {rel: manifest["files"][rel]["sha256"] for rel in rels}}
        )
    record = {
        "upstream": {**manifests[0]["upstream"], "sha": pinned},
        "overlays": layers,
        "replaced_files": sorted(set(replaced)),
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
    ap.add_argument(
        "--overlay",
        type=Path,
        action="append",
        required=True,
        help="recipes/drivejepa_wajepa/overlay/<run>; repeat to layer several, in order",
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    record = build(a.drivejepa, a.overlay, a.out)
    layers = ", ".join(f"{layer['run']} ({len(layer['files'])} files)" for layer in record["overlays"])
    print(
        f"{a.out}: Drive-JEPA {record['upstream']['sha'][:12]} + {layers}; "
        f"{len(record['replaced_files'])} upstream files replaced"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
