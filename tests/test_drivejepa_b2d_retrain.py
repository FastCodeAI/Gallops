"""recipes/drivejepa_b2d_retrain.toml and H16's record: the recipe resolves, its modules stay Python 3.9, the
grade keeps Bench2Drive's coefficients, and the record's numbers are the ones in the tables they cite.

No GPU, no torch, no data. Nothing here runs a trainer: H16's README says what was run instead."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

import pytest

from gallops import experiments as exps
from gallops import recipe as recipes
from gallops.config import REPO_ROOT, SiteConfig

PKG = REPO_ROOT / "src" / "gallops" / "drivejepa_b2d_retrain"
FOLDER = REPO_ROOT / "experiments" / "H16_drivejepa_b2d_retrain"
TABLES = FOLDER / "tables"
CLIPS = REPO_ROOT / "recipes" / "drivejepa_b2d_retrain" / "clips.json"
RECIPE = "drivejepa_b2d_retrain"
MODULES = sorted(p.name for p in PKG.glob("*.py"))


def write_site(tmp: Path) -> SiteConfig:
    for d in ("navsim", "clips"):
        (tmp / d).mkdir(exist_ok=True)
    path = tmp / "gallops.toml"
    path.write_text(
        f"""
site = "test"
[roots]
weights = "{tmp}/weights"
runs = "{tmp}/runs"
[python]
drivejepa = "{tmp}/py/drivejepa"
[code]
drivejepa = "{tmp}/Drive-JEPA"
gallops_kitscenes = "{tmp}/Gallops-e75aaac"
[datasets.navsim]
path = "{tmp}/navsim"
exp = "{tmp}/navsim-exp"
[datasets.b2d_clips]
path = "{tmp}/clips"
"""
    )
    return SiteConfig.load(path)


@pytest.fixture
def site(tmp_path: Path) -> SiteConfig:
    return write_site(tmp_path)


def constant(module: str, name: str):
    """A module-level literal, read from the source: the modules import torch, which `make check` has not."""
    tree = ast.parse((PKG / module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if isinstance(node.value, ast.Call):  # COEF = dict(ped=0.5, ...)
                return {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {module}")


def methods(module: str, cls: str) -> set[str]:
    tree = ast.parse((PKG / module).read_text())
    found = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {n.name for n in found.body if isinstance(n, ast.FunctionDef)}


def run(rid: str) -> dict:
    return exps.Registry.load().run_ids()[rid][1]


def test_every_stage_resolves(site: SiteConfig) -> None:
    rec = recipes.load(RECIPE)
    cmds = recipes.commands(rec, site)
    assert [c.stage for c in cmds] == rec.names()
    for c in cmds:
        text = " ".join([*c.argv, c.cwd or "", *c.env.values()])
        assert "{" not in text, f"{c.stage}: unexpanded template in {text}"
        assert c.argv[0] == str(site.python["drivejepa"]), "everything runs in Drive-JEPA's interpreter"
        assert c.argv[1] == "-m" and c.argv[2].startswith("gallops.drivejepa_b2d_retrain.")


def test_every_stage_module_exists() -> None:
    for s in recipes.load(RECIPE).stages:
        path = REPO_ROOT / "src" / "gallops" / (s.module.replace(".", "/") + ".py")
        assert path.exists(), f"stage {s.name} runs {s.module}, which is not at {path}"


def test_a_stage_sets_every_location_its_modules_read(site: SiteConfig) -> None:
    """wiring.py reads locations from the environment and falls back to an impossible path, so a stage the
    recipe forgot one for would fail hours in, on the first file it opened there."""
    names = set(re.findall(r'_site\("(\w+)"\)', (PKG / "wiring.py").read_text()))
    assert names >= {"B2D_SPLITS", "B2D_CLIPS", "B2D_OUT", "B2D_VOCAB", "B2D_DATA", "SID_KITSCENES"}
    for c in recipes.commands(recipes.load(RECIPE), site):
        if c.stage in ("frame-cache", "lora-train", "lora-merge", "stage2-train", "stage2-finetune"):
            assert names <= set(c.env), f"{c.stage} leaves {names - set(c.env)} unset"


@pytest.mark.parametrize("rel", MODULES)
def test_drivejepa_side_is_python39(rel: str) -> None:
    """These run in Drive-JEPA's Python 3.9 env; 3.10+ syntax would fail only there, hours into a run."""
    ast.parse((PKG / rel).read_text(), filename=rel, feature_version=(3, 9))


def test_the_package_imports_nothing_at_package_level() -> None:
    assert not [n for n in ast.parse((PKG / "__init__.py").read_text()).body if not isinstance(n, ast.Expr)]


def test_the_trainer_keeps_what_drive_jepa_calls() -> None:
    """make check has no torch, so this is the guard that the trainer's agent still overrides what it must."""
    assert {"__init__", "compute_score"} <= methods("train_stage2.py", "B2DAgent")
    assert {"__init__", "__len__"} <= methods("samples.py", "Tables")
    assert {"__init__", "__len__", "__getitem__"} <= methods("samples.py", "Samples")
    assert {"__init__", "clip", "frame"} <= methods("frames.py", "FrameCache")


def test_the_split_is_tracked_and_the_two_sides_share_no_route() -> None:
    split = json.loads(CLIPS.read_text())
    assert (len(split["clips"]), len(split["train"]), len(split["heldout"])) == (944, 897, 47)
    assert sorted(split["train"] + split["heldout"]) == sorted(split["clips"])
    assert len(split["twins_removed"]) == 24 and not set(split["twins_removed"]) & set(split["clips"])

    def route(clip: str) -> str:
        return re.search(r"(Town\d+\w*?)_Route(\d+)", clip).group(0)

    assert not {route(c) for c in split["train"]} & {route(c) for c in split["heldout"]}


def test_the_grade_keeps_bench2drive_s_coefficients() -> None:
    """statistics_manager.py's penalties, and the adaptation the record states: 1.0 m/s on 0.5 s segments."""
    assert constant("grade.py", "COEF") == dict(ped=0.5, veh=0.6, static=0.65, red=0.7, stop=0.8)
    assert constant("grade.py", "SPEED_T") == 1.0
    assert constant("grade.py", "OFF_REAR") == 1.3886
    assert constant("scorer.py", "MARGIN") == 0.3


def test_teachers_are_the_spec_s() -> None:
    text = (PKG / "teachers.py").read_text()
    assert "S[o[i] : o[i + 1]] >= 0.95" in text, "the teacher threshold moved"
    assert "if len(t) > 256:" in text, "the cap moved"


def test_the_record_pins_what_the_code_checks() -> None:
    enc = run("H16.r1")["checkpoint"]["weights_sha256"]
    source = (PKG / "train_stage2.py").read_text()
    assert f'ENC_SHA = "{enc}"' in source, (
        "Stage 2 refuses any other encoder, so the record must name this one"
    )
    assert run("H16.r1")["start_from"] == f"released ({constant('wiring.py', 'AGENT_SHA')[:12]})"
    data = {d["manifest_sha256"] for d in run("H16.r2")["data"] if d.get("manifest_sha256")}
    sums = (TABLES / "vocab_SHA256SUMS").read_text()
    assert any(f"{sha}  b2d_2048.npy" in sums for sha in data)


def table(name: str) -> list[dict]:
    with (TABLES / name).open(newline="") as f:
        return list(csv.DictReader(f))


def result(rid: str, metric: str) -> dict:
    return next(r for r in run(rid)["results"] if r["metric"].startswith(metric))


def test_the_offline_numbers_are_the_epoch_tables() -> None:
    rows = table("p2_05nov_epochs.csv")
    assert [int(r["epoch"]) for r in rows] == list(range(1, 21))
    e18 = rows[17]
    assert f"{float(e18['pick_grade']):.4f}" == result("H16.r2", "pick grade")["value"]
    assert f"{float(e18['pick_ade_3s']):.4f}" == result("H16.r2", "scorer-pick ADE")["value"]
    assert run("H16.r2")["checkpoint"]["step"] == 18
    ft = table("p2ft_06nov_epochs.csv")
    assert [int(r["epoch"]) for r in ft] == list(range(1, 9)), "the record says 8 of 20 epochs"
    res = result("H16.r3", "pick grade")
    assert f"{float(ft[7]['pick_grade']):.4f}" == res["value"]
    assert f"{float(rows[7]['pick_grade']):.4f}" == res["baseline_value"]


def test_the_closed_loop_means_are_the_per_route_table() -> None:
    rows = table("closed_loop_13_routes.csv")
    res = result("H16.r2", "mean driving score")
    assert len(rows) == res["n"] == 13
    assert [int(r["queue_position"]) for r in rows] == list(range(1, 14)), (
        "the note says the front of the queue"
    )
    assert "PARTIAL" in res["metric"] and "smoke" in res["flags"], (
        "13 of 50 routes must never read as a score"
    )
    assert not res.get("headline")

    def mean(key: str) -> str:
        return f"{sum(float(r[key]) for r in rows) / len(rows):.2f}"

    assert mean("h16_r2_ds") == res["value"]
    assert mean("released_ds") == res["baseline_value"]
    assert "45.36" in res["note"] and mean("run3a_ds") == "45.36"


def test_the_gate_block_is_the_verdict_files() -> None:
    files = [json.loads((TABLES / n).read_text()) for n in ("verdict_ncap.json", "verdict_openloop.json")]
    (block,) = run("H16.r2")["gate"]
    assert {f["overall"] for f in files} == {block["verdict"]}
    verdicts = [p for f in files for p in f["pairs"]]
    for pair in block["pairs"]:
        if pair["verdict"] == "NOT RUN":
            continue
        got = [p for p in verdicts if p["bench"] == pair["bench"] and p["units"] == pair["units"]]
        assert len(got) == 1, pair
        assert pair["verdict"] == got[0]["verdict"]
        assert pair["baseline"] == round(got[0]["baseline_mean"], 3)
        assert pair["candidate"] == round(got[0]["candidate_mean"], 3)
        assert pair["t"] == round(got[0]["t"], 2)
        assert pair["mde"] == round(got[0]["mde"], 3)


def test_the_benchmarks_with_no_result_say_so() -> None:
    assert not json.loads((TABLES / "verdict_alpasim.json").read_text())["pairs"]
    missing = {p["bench"] for g in run("H16.r2")["gate"] for p in g["pairs"] if p["verdict"] == "NOT RUN"}
    assert missing == {"navsim", "alpasim", "bench2drive"}
