"""recipes/drivejepa_wajepa.toml: the recipe resolves, its overlay is what the manifest says, and the stages
that can run without a GPU or navsim do what they claim.

The overlay is the record of the agent code a run trained with; these checks are what stop it drifting from
that record unnoticed. No GPU, no torch, no data."""

from __future__ import annotations

import ast
import csv
import json
import subprocess
from pathlib import Path

import pytest

from gallops import recipe as recipes
from gallops.checkpoint import Checkpoint, SidecarError
from gallops.config import REPO_ROOT, SiteConfig
from gallops.drivejepa_wajepa import code, navtest_score, train, write_checkpoint

SRC = REPO_ROOT / "src" / "gallops"
OVERLAYS = REPO_ROOT / "recipes" / "drivejepa_wajepa" / "overlay"
RUNS = sorted(p.stem for p in OVERLAYS.glob("*.json"))

#: Run in Drive-JEPA's own interpreter (Python 3.9), with what they import from gallops.
PY39 = (
    "__init__.py",
    "drivejepa_b2d/__init__.py",
    "drivejepa_b2d/provenance.py",
    "drivejepa_wajepa/__init__.py",
    "drivejepa_wajepa/dump_tokens.py",
    "drivejepa_wajepa/verify_tokens.py",
    "drivejepa_wajepa/train.py",
    "drivejepa_wajepa/eval_navtest.py",
    "drivejepa_wajepa/extract_agents.py",
    "drivejepa_wajepa/agent_targets.py",
    "drivejepa_wajepa/agent_decoder_data.py",
    "drivejepa_wajepa/train_agent_decoder.py",
    "drivejepa_wajepa/lane_centrelines.py",
    "drivejepa_wajepa/teacher_rearok.py",
    "drivejepa_wajepa/run3a_checks.py",
    "drivejepa_wajepa/pick_tune.py",
)


def write_site(tmp: Path) -> SiteConfig:
    for d in ("navsim", "tokens"):
        (tmp / d).mkdir(exist_ok=True)
    path = tmp / "gallops.toml"
    path.write_text(
        f"""
site = "test"
[roots]
weights = "{tmp}/weights"
runs = "{tmp}/runs"
[python]
train = "{tmp}/py/train"
drivejepa = "{tmp}/py/drivejepa"
[code]
drivejepa = "{tmp}/Drive-JEPA"
wajepa = "{tmp}/WA-JEPA"
[datasets.navsim]
path = "{tmp}/navsim"
exp = "{tmp}/navsim-exp"
maps = "{tmp}/maps"
current1 = "{tmp}/current1"
metric_cache = "{tmp}/metric_cache"
drivejepa_cache = "{tmp}/drivejepa-cache"
[datasets.wajepa_tokens]
path = "{tmp}/tokens"
"""
    )
    return SiteConfig.load(path)


@pytest.fixture
def site(tmp_path: Path) -> SiteConfig:
    return write_site(tmp_path)


def test_every_stage_resolves(site: SiteConfig) -> None:
    rec = recipes.load("drivejepa_wajepa")
    cmds = recipes.commands(rec, site)
    assert [c.stage for c in cmds] == rec.names()
    for c in cmds:
        text = " ".join([*c.argv, c.cwd or "", *c.env.values()])
        assert "{" not in text, f"{c.stage}: unexpanded template in {text}"
        assert c.argv[1] == "-m" and c.argv[2].startswith("gallops.")


def test_every_stage_module_exists() -> None:
    for s in recipes.load("drivejepa_wajepa").stages:
        path = SRC / (s.module.replace(".", "/") + ".py")
        assert path.exists(), f"stage {s.name} runs {s.module}, which is not at {path}"


def test_navsim_stages_run_in_drivejepas_interpreter(site: SiteConfig) -> None:
    for c in recipes.commands(recipes.load("drivejepa_wajepa"), site):
        module = c.argv[2].removeprefix("gallops.").replace(".", "/") + ".py"
        if module in PY39:
            assert c.argv[0] == str(site.python["drivejepa"]), c.stage


def test_the_training_overrides_are_the_runs_own(site: SiteConfig) -> None:
    """run2-train's overrides are Run 2's code/hydra/overrides.yaml (16 Oct 17:48), the cache path aside."""
    (cmd,) = recipes.commands(recipes.load("drivejepa_wajepa"), site, ["run2-train"])
    overrides = list(cmd.argv[cmd.argv.index("--") + 1 :])
    assert overrides == [
        "agent=drive_jepa_perception_based_agent",
        "experiment_name=wajepa_dj_v2",
        "train_test_split=navtrain",
        "dataloader.params.batch_size=64",
        "dataloader.params.num_workers=8",
        "trainer.params.max_epochs=20",
        "trainer.params.strategy=auto",
        f"cache_path={site.datasets['wajepa_tokens'].path}",
        "use_cache_without_dataset=True",
        "force_cache_computation=False",
    ]


def test_run3a_trains_with_its_recorded_overrides_and_refuses_without_its_inputs(site: SiteConfig) -> None:
    """Run 3A's code/hydra/overrides.yaml (24 Oct 17:34), plus the pick defaults it ran with: val/score_epoch,
    which chose the epoch, reads the config's pick, and the overlay's config now holds the pick locked after
    training (#23 review 🟠 2). Its model starts the agent decoder from random weights, silently, when
    RUN3A_AGENT_DECODER_CKPT names no file, so the stage must require it."""
    (cmd,) = recipes.commands(recipes.load("drivejepa_wajepa"), site, ["run3a-train"])
    argv = list(cmd.argv)
    overrides = argv[argv.index("--") + 1 :]
    assert overrides == [
        "agent=drive_jepa_run3a_agent",
        "experiment_name=run3a",
        "train_test_split=navtrain",
        "dataloader.params.batch_size=64",
        "dataloader.params.num_workers=8",
        "trainer.params.max_epochs=20",
        "trainer.params.strategy=auto",
        "trainer.params.precision=16-mixed",
        f"cache_path={site.datasets['wajepa_tokens'].path}",
        "use_cache_without_dataset=True",
        "force_cache_computation=False",
        "+agent.config.pick_gate_weights=[1.0,1.0,1.0,1.0,1.0]",
        "+agent.config.pick_alpha=0.0",
    ]
    assert argv[argv.index("--config-name") + 1] == "run3a_training"
    required = [argv[i + 1] for i, a in enumerate(argv) if a == "--require"]
    assert cmd.env["RUN3A_AGENT_DECODER_CKPT"] in required
    assert cmd.env["RUN3A_CENTRELINES"] in required


def test_the_teacher_list_goes_beside_the_metric_cache(site: SiteConfig) -> None:
    """compute_navsim_score finds a sample's teachers by swapping the metric cache's directory name in its
    path."""
    (cmd,) = recipes.commands(recipes.load("drivejepa_wajepa"), site, ["teacher-rearok"])
    argv = list(cmd.argv)
    mc, out = Path(argv[argv.index("--metric-cache") + 1]), Path(argv[argv.index("--out") + 1])
    assert mc.name == "train_metric_cache_v2_pruned" and out.parent == mc.parent
    teacher_dir = "anchors_scores_index_v2_rearok_lane"
    config = OVERLAYS / "run3a/navsim_v2/navsim/agents/drive_jepa_run3a/drive_jepa_config.py"
    assert out.name == teacher_dir and f'teacher_dir: str = "{teacher_dir}"' in config.read_text()


def test_the_held_out_logs_are_run3as_validation_split() -> None:
    """The 51 held-out logs chose Run 3A's epoch and pick, and are what the agent decoder holds out."""
    held = [
        ln.strip() for ln in (REPO_ROOT / "recipes/drivejepa_wajepa/heldout_logs.txt").read_text().split("\n")
    ]
    held = [h for h in held if h]
    split = (
        OVERLAYS / "run3a/navsim_v2/navsim/planning/script/config/training/run3a_log_split.yaml"
    ).read_text()
    sections: dict[str, list[str]] = {}
    key = None
    for line in split.splitlines():
        if line and not line.startswith((" ", "#")):
            key = line.rstrip(":")
            sections[key] = []
        elif line.strip().startswith("- ") and key:
            sections[key].append(line.strip()[2:])
    assert len(held) == len(set(held)) == 51
    assert sorted(sections["val_logs"]) == sorted(held)
    assert not set(sections["train_logs"]) & set(held)


@pytest.mark.parametrize("rel", PY39)
def test_drivejepa_side_is_python39(rel: str) -> None:
    ast.parse((SRC / rel).read_text(), filename=rel, feature_version=(3, 9))


@pytest.mark.parametrize("run", RUNS)
def test_overlay_matches_its_manifest(run: str) -> None:
    overlay = OVERLAYS / run
    manifest = code.load_manifest(overlay)
    files = code.check_overlay(overlay, manifest)  # raises on a missing, extra or edited file
    assert files and manifest["run"] == run
    assert len(manifest["upstream"]["sha"]) == 40
    for rel in files:
        entry = manifest["files"][rel]
        if (
            entry["sha256"] != entry["as_run_sha256"]
        ):  # an edit, or a file whose as-run copy was not kept (null)
            assert "note" in entry, f"{rel} differs from the file as run and the manifest does not say why"


@pytest.mark.parametrize("run", RUNS)
def test_overlay_is_python39(run: str) -> None:
    for path in (OVERLAYS / run).rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 9))


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "upstream"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/a.py").write_text("a = 'upstream'\n")
    (repo / "pkg/b.py").write_text("b = 'upstream'\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "upstream")
    return repo, git(repo, "rev-parse", "HEAD").strip()


def make_overlay(tmp_path: Path, files: dict[str, str], pin: str = "0" * 40) -> Path:
    overlay = tmp_path / "overlay" / "run"
    entries = {}
    for rel, text in files.items():
        (overlay / rel).parent.mkdir(parents=True, exist_ok=True)
        (overlay / rel).write_text(text)
        entries[rel] = {"sha256": code.sha256(overlay / rel), "as_run_sha256": code.sha256(overlay / rel)}
    manifest = {"run": "run", "upstream": {"repo": "x", "sha": pin}, "files": entries}
    overlay.with_suffix(".json").write_text(json.dumps(manifest))
    return overlay


def test_code_builds_upstream_plus_overlay(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, sha = upstream
    overlay = make_overlay(tmp_path, {"pkg/a.py": "a = 'ours'\n", "pkg/new.py": "n = 1\n"})
    out = tmp_path / "tree"
    record = code.build(repo, [overlay], out, sha=sha)
    assert (out / "pkg/a.py").read_text() == "a = 'ours'\n"
    assert (out / "pkg/b.py").read_text() == "b = 'upstream'\n"
    assert (out / "pkg/new.py").read_text() == "n = 1\n"
    assert record["replaced_files"] == ["pkg/a.py"]
    assert json.loads((out / "OVERLAY.json").read_text())["upstream"]["sha"] == sha
    assert not git(repo, "status", "--porcelain"), "the checkout must not be touched"
    with pytest.raises(SystemExit, match="exists"):
        code.build(repo, [overlay], out, sha=sha)


def test_code_layers_overlays_in_order(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, sha = upstream
    first = make_overlay(tmp_path / "1", {"pkg/a.py": "a = 'first'\n", "pkg/c.py": "c = 1\n"})
    second = make_overlay(tmp_path / "2", {"pkg/a.py": "a = 'second'\n"})
    record = code.build(repo, [first, second], tmp_path / "tree", sha=sha)
    assert (tmp_path / "tree/pkg/a.py").read_text() == "a = 'second'\n"
    assert (tmp_path / "tree/pkg/c.py").read_text() == "c = 1\n"
    assert [layer["files"] and len(layer["files"]) for layer in record["overlays"]] == [2, 1]


def test_code_refuses_overlays_pinning_different_commits(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, sha = upstream
    first = make_overlay(tmp_path / "1", {"pkg/a.py": "a = 1\n"})
    second = make_overlay(tmp_path / "2", {"pkg/b.py": "b = 1\n"}, pin="1" * 40)
    with pytest.raises(SystemExit, match="different upstream commits"):
        code.build(repo, [first, second], tmp_path / "tree", sha=sha)


def test_code_refuses_an_edited_overlay(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, sha = upstream
    overlay = make_overlay(tmp_path, {"pkg/a.py": "a = 'ours'\n"})
    (overlay / "pkg/a.py").write_text("a = 'edited'\n")
    with pytest.raises(SystemExit, match="differ from the manifest"):
        code.build(repo, [overlay], tmp_path / "tree", sha=sha)
    assert not (tmp_path / "tree").exists()


def test_code_refuses_an_unlisted_file(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, sha = upstream
    overlay = make_overlay(tmp_path, {"pkg/a.py": "a = 'ours'\n"})
    (overlay / "pkg/stray.py").write_text("x = 1\n")
    with pytest.raises(SystemExit, match="disagree"):
        code.build(repo, [overlay], tmp_path / "tree", sha=sha)


CSV_COLS = ["token", "valid", "score", *navtest_score.GATES, *(c for c, _ in navtest_score.SOFT)]


def write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def row(token: str, gates: tuple, soft: tuple, score: float, valid: str = "True") -> dict:
    r = {"token": token, "valid": valid, "score": score}
    r.update(dict(zip(navtest_score.GATES, gates, strict=True)))
    r.update(dict(zip((c for c, _ in navtest_score.SOFT), soft, strict=True)))
    return r


def test_navtest_score_recombines_the_human_filtered_columns(tmp_path: Path) -> None:
    rows = [
        # all soft terms: (5*0.8 + 5*1 + 2*1 + 2*0.5 + 2*1) / 16 = 0.875
        row("a", (1, 1, 1, 1), (0.8, 1, 1, 0.5, 1), 0.9),
        # EC blank: left out of this scene's mean -> (5*1 + 5*1 + 2*1 + 2*1) / 14 = 1; one gate halves it
        row("b", (1, 0.5, 1, 1), (1, 1, 1, 1, ""), 0.7),
        row("c", (0, 0, 0, 0), (0, 0, 0, 0, 0), 0.0, valid="False"),  # invalid: dropped
        row("average_all_frames", (1, 1, 1, 1), (1, 1, 1, 1, 1), 1.0),  # summary row: dropped
    ]
    s = navtest_score.score(str(write_csv(tmp_path / "x.csv", rows)))
    assert s["scenes"] == 2
    assert s["epdms_human_filter"] == pytest.approx(100 * (0.875 + 0.5) / 2)
    assert s["csv_score"] == pytest.approx(80.0)
    assert s["sub_scores"]["EC"] == pytest.approx(1.0)  # the blank is not a zero


def test_navtest_score_reads_the_latest_run_of_a_glob(tmp_path: Path) -> None:
    for stamp in ("2026.10.17.13.45.48", "2026.10.18.09.00.00"):
        (tmp_path / stamp).mkdir()
        write_csv(tmp_path / stamp / "r.csv", [row("a", (1, 1, 1, 1), (1, 1, 1, 1, 1), 1.0)])
    assert navtest_score.resolve(str(tmp_path / "*" / "*.csv")).endswith("2026.10.18.09.00.00/r.csv")
    with pytest.raises(SystemExit, match="no CSV"):
        navtest_score.resolve(str(tmp_path / "none" / "*.csv"))


def test_train_refuses_an_existing_checkpoint_dir(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "OVERLAY.json").write_text("{}")
    (tmp_path / "ckpts").mkdir()
    with pytest.raises(SystemExit, match="never overwritten"):
        train.main(["--tree", str(tree), "--ckpt-dir", str(tmp_path / "ckpts"), "--", "agent=x"])


def test_train_refuses_a_missing_required_input(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "OVERLAY.json").write_text("{}")
    with pytest.raises(SystemExit, match="required inputs missing"):
        train.main(
            ["--tree", str(tree), "--ckpt-dir", str(tmp_path / "c"), "--require", str(tmp_path / "no.pt")]
        )


def test_train_refuses_a_tree_the_code_stage_did_not_build(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="OVERLAY.json"):
        train.main(["--tree", str(tmp_path), "--ckpt-dir", str(tmp_path / "ckpts")])


def test_train_sets_what_both_runs_had(tmp_path: Path) -> None:
    env = train.environment(tmp_path, tmp_path / "ckpts", 12)
    assert env["WANDB_MODE"] == "offline"
    assert env["DRIVEJEPA_SCORE_WORKERS"] == "12"
    assert env["DRIVEJEPA_CKPT_DIR"] == str(tmp_path / "ckpts")
    assert env["NAVSIM_DEVKIT_ROOT"] == env["PYTHONPATH"] == str(tmp_path / "navsim_v2")
    cmd = train.command(tmp_path, "run3a_training", ["agent=x"])
    assert cmd[1].endswith("navsim/planning/script/run_training.py")
    assert cmd[2:] == ["--config-name", "run3a_training", "agent=x"]


def _record(**over: object) -> Checkpoint:
    kw: dict = dict(
        global_step=26580,
        epoch=19,
        n_params=7_768_355,
        dtypes={"float32"},
        model="wajepa_dj",
        train_run="run2",
        trained_on_pairs=["navsim:navtrain"],
        selection_split=None,
        src_name="epoch_19.ckpt",
        src_sha256="a" * 64,
        manifest_name="run2.json",
        manifest_sha256="b" * 64,
        upstream={"sha": "548bb82"},
        policy_env={},
        note="",
        publisher_sha="c" * 40,
    )
    kw.update(over)
    return write_checkpoint.record_for(**kw)


def test_checkpoint_stage_builds_a_valid_sidecar() -> None:
    """#22 review 🔴 1: the stage set precision "16-mixed", which the sidecar refuses, so it never ran."""
    with pytest.raises(SidecarError, match="precision"):
        Checkpoint(
            model="wajepa_dj",
            step=1,
            train_git_sha="c" * 40,
            train_run="run2",
            precision=write_checkpoint.TRAINING_PRECISION,
        )
    rec = _record()
    assert rec.precision == "fp32"
    assert rec.notes["training_precision"] == "16-mixed"
    assert _record(dtypes={"float32", "float16"}).precision == "mixed"


def test_checkpoint_stage_steps_by_global_step_and_counts_params_once() -> None:
    """#22 review 🟠 2: the step directory is the optimizer step (the epoch goes in notes), and params_m is
    the distinct tensors, so it agrees with the record's 7.8 M."""
    rec = _record()
    assert rec.step == 26580 and rec.notes["epoch"] == 19
    assert rec.params_m == 7.77


def test_summarize_counts_tied_tensors_once() -> None:
    torch = pytest.importorskip("torch")
    shared = torch.zeros(3, 4)
    state = {"round0.w": shared, "round1.w": shared, "b": torch.zeros(5), "step": 3}
    n, dtypes = write_checkpoint.summarize(state)
    assert n == 17
    assert dtypes == {"float32"}


def test_agent_decoder_and_centreline_stages_make_their_output_dirs() -> None:
    """#23 review 🔴 1: both stages wrote into a directory no stage created, so ran only where it was
    made by hand. Both modules need torch or nuplan, so this guards the structure: train() makes its out dir
    first (refusing an existing one), and the centreline script makes its out dir before the map loop."""
    tad = ast.parse((SRC / "drivejepa_wajepa" / "train_agent_decoder.py").read_text())
    fn = next(n for n in tad.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    assert ast.unparse(fn.body[0]) == "os.makedirs(out_dir)"
    lc = ast.parse((SRC / "drivejepa_wajepa" / "lane_centrelines.py").read_text())
    made = [i for i, n in enumerate(lc.body) if ast.unparse(n) == "os.makedirs(OUT, exist_ok=True)"]
    loop = next(i for i, n in enumerate(lc.body) if isinstance(n, ast.For))
    assert made and made[0] < loop
