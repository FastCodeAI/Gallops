"""One-off: copy the as-run pipeline into src/gallops/drivejepa_b2d_retrain with only import and path edits.
Every replacement must hit, so a stale pattern fails here rather than leaving a machine path behind."""
import pathlib, sys

OPS = pathlib.Path('{ops}')
B = OPS / 'b2d_drivejepa'
DST = pathlib.Path('{ops}/worktrees/gallops-b2d-p2/src/gallops/drivejepa_b2d_retrain')
PKG = 'gallops.drivejepa_b2d_retrain'
W = f'from {PKG} import wiring as W'

P1 = B / 'phase1/src'
P2 = B / 'phase2/src'
FILES = {
    'frames.py': (P1 / 'data.py', [
        ('import wiring as W\n', W + '\n'),
    ]),
    'ade.py': (P1 / 'ade.py', [
        ('import wiring as W\nfrom data import FrameCache, _to_tensor\n', f'from {PKG}.frames import FrameCache, _to_tensor\n'),
    ]),
    'build_cache.py': (P1 / 'build_cache.py', [
        ('sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport wiring as W  # noqa: E402\n', W + '\n'),
        ('json.load(open(W.ROOT / "splits/clips.json"))', 'json.load(open(W.SPLITS))'),
    ]),
    'train_lora.py': (P1 / 'train_b2d.py', [
        ('sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport wiring as W  # noqa: E402\n', W + '\n'),
        ('import ade as A  # noqa: E402\nimport data as D  # noqa: E402\n', f'from {PKG} import ade as A  # noqa: E402\nfrom {PKG} import frames as D  # noqa: E402\n'),
        ('json.load(open(W.ROOT / "splits/clips.json"))', 'json.load(open(W.SPLITS))'),
    ]),
    'merge_lora.py': (P1 / 'merge_pass.py', [
        ('import wiring as W\n', W + '\n'),
        ('import data as D\n', f'from {PKG} import frames as D\n'),
        ('open(W.ROOT / "sid_e75aaac/f_final.py")', 'open(W.SID_TRAIN / "f_final.py")'),
        ('json.load(open(W.ROOT / "splits/clips.json"))', 'json.load(open(W.SPLITS))'),
    ]),
    'build_vocab.py': (B / 'vocab/build_vocab.py', [
        ('HERE = os.path.dirname(os.path.abspath(__file__))\n', ''),
        ('default=f"{HERE}/paths_cache.npz"', 'required=True'),
        ('ap.add_argument("--out", default=HERE)', 'ap.add_argument("--out", required=True)'),
    ]),
    'scorer.py': (OPS / 'mtd/scorer.py', []),
    'grade.py': (P2 / 'grade2.py', [
        ("sys.path.insert(0, '{ops}/mtd')\nimport scorer as SC\n", f'from {PKG} import scorer as SC\n'),
        ('import math, os, sys\n', 'import math\n'),
    ]),
    'grade_all.py': (P2 / 'grade_all.py', [
        ('import grade2 as G2\n', f'from {PKG} import grade as G2\n{W}\n'),
        ("D = '{data.b2d_retrain}'; CTX = f'{D}/context'\nV = '{ops}/b2d_drivejepa/vocab/b2d_2048.npy'\n", "D = str(W.DATA); CTX = f'{D}/context'\nV = str(W.VOCAB / 'b2d_2048.npy')\n"),
        ("json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json'))", 'json.load(open(W.SPLITS))'),
    ]),
    'teachers.py': (P2 / 'teachers.py', [
        ("import json, os, re, numpy as np\n", f"import json, os, re, numpy as np\n{W}\n"),
        ("D = '{data.b2d_retrain}'; G = f'{D}/grade_v2'; OUT = f'{D}/teachers_2048_v2.npz'; RES = '{ops}/b2d_drivejepa/phase2/step_a_v2_numbers.json'\n", "D = str(W.DATA); G = f'{D}/grade_v2'; OUT = f'{D}/teachers_2048_v2.npz'; RES = f'{D}/step_a_v2_numbers.json'\n"),
        ("json.load(open('{ops}/b2d_drivejepa/phase1/splits/clips.json'))", 'json.load(open(W.SPLITS))'),
        ("    p = f'{ops}/b2d_drivejepa/phase2/logs/{f}'; m = re.findall(r'GRADE_DONE (\\d+) (\\d+) (\\d+)', open(p).read())\n", "    p = f'{D}/logs/{f}'; m = re.findall(r'GRADE_DONE (\\d+) (\\d+) (\\d+)', open(p).read()) if os.path.exists(p) else []\n"),
    ]),
    'samples.py': (P2 / 'data2.py', [
        ("import json, os, sys\n", "import json, os\n"),
        ("import grade2 as G2\nsys.path.insert(0, '{ops}/b2d_drivejepa/phase1/src')\nimport wiring as W\nD = '{data.b2d_retrain}'\n", f"from {PKG} import grade as G2\n{W}\nD = str(W.DATA)\n"),
        ("json.load(open(W.ROOT / 'splits/clips.json'))", 'json.load(open(W.SPLITS))'),
        ("np.load('{ops}/b2d_drivejepa/vocab/paths_cache.npz', allow_pickle=True)", "np.load(W.VOCAB / 'paths_cache.npz', allow_pickle=True)"),
        ("from data import FrameCache, _to_tensor\n", f"from {PKG}.frames import FrameCache, _to_tensor\n"),
    ]),
    'train_stage2.py': (P2 / 'train_b2d.py', [
        ("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nsys.path.insert(0, '{ops}/b2d_drivejepa/phase1/src')\nimport wiring as W\n", W + '\n'),
        ("import grade2 as G2\nimport data2 as D2\nsys.path.insert(0, '{ops}/mtd/stage4')\nimport sca_fix\n", f"from {PKG} import grade as G2\nfrom {PKG} import samples as D2\nfrom gallops.drivejepa_b2d import batch_fix\n"),
        ("sca_fix.apply_all()\n", "batch_fix.apply_all()\n"),
        ("VOCAB = '{ops}/b2d_drivejepa/vocab/b2d_2048.npy'; ", "VOCAB = W.VOCAB / 'b2d_2048.npy'; "),
        ("import argparse, hashlib, json, math, os, random, sys, time\n", "import argparse, hashlib, json, math, os, random, time\n"),
        ("mtd/stage4/sca_fix.apply_all(): upstream's refiner", "gallops.drivejepa_b2d.batch_fix.apply_all(): upstream's refiner"),
    ]),
    'export_agent.py': (P2 / 'export_ckpt.py', [
        ("import hashlib, json, sys, torch\n", f"import hashlib, sys, torch\n{W}\n"),
        ("d = f'{runs}/b2d_drivejepa/phase2/runs/p2_05nov/ckpt_epoch{e:02d}'", "d = str(W.RUNS.parent / f'phase2/runs/p2_05nov/ckpt_epoch{e:02d}')"),
    ]),
}

DST.mkdir(parents=True, exist_ok=True)
for name, (src, edits) in FILES.items():
    text = src.read_text()
    for old, new in edits:
        n = text.count(old)
        assert n >= 1, (name, old[:70])
        text = text.replace(old, new)
    (DST / name).write_text(text)
    print(name, 'from', src.relative_to(OPS), len(edits), 'edits')
