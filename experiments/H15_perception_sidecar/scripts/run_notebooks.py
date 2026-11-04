"""Execute H15's review notebooks in place, keeping their outputs, as they were run on 4 Nov 2026.

    <Drive-JEPA's interpreter> experiments/H15_perception_sidecar/scripts/run_notebooks.py \
        experiments/H15_perception_sidecar/notebooks/01_dataset.ipynb [...]

Run it with Drive-JEPA's interpreter (`[python] drivejepa` in gallops.toml): it has nbclient and
ipykernel, and jupyter_client starts the `python3` kernel with the interpreter that launches it, so the
notebooks run where torch 2.1 and the vendor's packages are. Each notebook finds the repository and
gallops.toml from its own folder (so it runs only from inside the checkout) and prints no machine path.
After a run, grep the notebook for the site's data and checkout roots before committing: the
portability gate does not know this site's roots.

Prints any error or stderr output per cell, and exits 1 if a cell failed.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run(path: Path, timeout: int = 1800) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    )
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    ok = True
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                print(f"{path.name} cell {i}: {out['ename']}: {out['evalue']}")
                ok = False
            if out.get("name") == "stderr":
                print(f"{path.name} cell {i}, stderr: {out['text'][:300]}")
    print(f"{path.name}: {'executed' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    results = [run(Path(arg).resolve()) for arg in sys.argv[1:]]
    sys.exit(0 if results and all(results) else 1)
