"""Command line for gallops.

Deliberately small. This repo's job is to produce checkpoints; the interesting
verbs will arrive with the training code. What exists today is what you need to
confirm the plumbing before writing any of it -- which is the same discipline
paddock's `adding-a-model` guide prescribes for adapters: get the boring path
green first, so that when something moves you know it was the model.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from gallops import checkpoint as ckpt
from gallops import experiments as exps
from gallops import recipe as recipes
from gallops.config import REPO_ROOT, ConfigError, SiteConfig

app = typer.Typer(add_completion=False, help="Training for driving policies.")


@app.command()
def config() -> None:
    """Show the resolved site configuration.

    The first thing to run in a new checkout. If this is wrong, everything
    downstream is wrong in a way that looks like a data bug.
    """
    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    typer.echo(f"site     {cfg.site}")
    typer.echo(f"source   {cfg.source}")
    typer.echo(f"weights  {cfg.weights}{'' if cfg.weights.exists() else '   (MISSING)'}")
    typer.echo(f"runs     {cfg.runs}{'' if cfg.runs.exists() else '   (MISSING)'}")
    for table in ("python", "code"):
        for name, path in sorted(getattr(cfg, table).items()):
            typer.echo(f"{table:<8} {name:<12} {path}{'' if path.exists() else '   (MISSING)'}")
    if not cfg.datasets:
        typer.echo("datasets none configured")
        return
    typer.echo("datasets")
    for name, dataset in sorted(cfg.datasets.items()):
        roots = ", ".join(str(r) for r in dataset.roots)
        present = any(r.exists() for r in dataset.roots)
        typer.echo(f"  {name:<12} {roots}{'' if present else '   (MISSING)'}")


@app.command()
def show(directory: str) -> None:
    """Print the sidecar of a checkpoint directory.

    Use it to check what a checkpoint claims about itself before spending GPU
    hours scoring it -- particularly `trained_on`, which decides whether a
    benchmark result from it is contaminated.
    """
    try:
        record = ckpt.read(directory)
    except ckpt.SidecarError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(record.to_dict(), indent=2, sort_keys=True))


@app.command()
def recipe(
    name: str,
    stages: Annotated[
        list[str] | None, typer.Argument(help="stages to show or run; all when omitted")
    ] = None,
    run: Annotated[bool, typer.Option("--run", help="run the stages in order; otherwise only print")] = False,
) -> None:
    """Print, or run, the commands of a recipe (recipes/<name>.toml) on this machine.

    Printing is the default: read what would run, with this site's paths filled
    in, before spending GPU hours on it.
    """
    try:
        cfg = SiteConfig.load()
        rec = recipes.load(name)
        cmds = recipes.commands(rec, cfg, stages or None)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    for cmd in cmds:
        typer.echo(f"# {cmd.stage}\n{cmd.shell()}\n")
        if run:
            code = recipes.run(cmd)
            if code:
                typer.secho(f"stage {cmd.stage} exited {code}; stopping", fg=typer.colors.RED, err=True)
                raise typer.Exit(code)


experiments_app = typer.Typer(
    add_completion=False,
    help="The experiment registry: experiments/<ID>_<slug>/experiment.toml, and the README rendered from it.",
)
app.add_typer(experiments_app, name="experiments")


def _registry() -> exps.Registry:
    try:
        return exps.Registry.load()
    except exps.RegistryError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


@experiments_app.command("check")
def experiments_check() -> None:
    """Validate every experiment, the decision log, the code map, and that the README is rendered.

    The same checks `make check` runs; this prints all of them at once.
    """
    reg = _registry()
    problems = list(reg.errors)
    if not problems:
        problems = [f"{t} is out of date: run `gallops experiments render --write`" for t in exps.stale(reg)]
    for problem in problems:
        typer.secho(problem, fg=typer.colors.RED, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"{len(reg.experiments)} experiments, {len(reg.decisions)} decisions: ok")


@experiments_app.command("render")
def experiments_render(
    write: Annotated[bool, typer.Option("--write", help="rewrite the rendered files in place")] = False,
) -> None:
    """Render the README's blocks, the index and every experiment's CARD.md from the registry.

    Without --write, only list the files that would change.
    """
    reg = _registry()
    if reg.errors:
        for problem in reg.errors:
            typer.secho(problem, fg=typer.colors.RED, err=True)
        typer.secho("fix the registry before rendering from it", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if write:
        changed = exps.write(reg)
        typer.echo("rewrote " + ", ".join(changed) if changed else "already up to date")
        return
    changed = exps.stale(reg)
    typer.echo("would rewrite " + ", ".join(changed) if changed else "already up to date")


@experiments_app.command("new")
def experiments_new(
    eid: Annotated[str, typer.Argument(help="the next free id: GCnn or Hnn")],
    slug: Annotated[str, typer.Argument(help="lower_case_words naming it")],
    track: Annotated[str, typer.Option(help="hypothesis | enabler | gap-closing")] = "hypothesis",
    owner: Annotated[str, typer.Option(help="a name from experiments/people.toml")] = "",
) -> None:
    """Start an experiment folder from experiments/_template, before the first run.

    PREDICTIONS.md comes with it: fill it in and commit it before anything is trained, because a
    prediction written after the result is not one.
    """
    import datetime as dt
    import shutil

    folder = exps.REPO_ROOT / "experiments" / f"{eid}_{slug}"
    if not exps.FOLDER_RE.match(folder.name) or not exps.ID_RE.match(eid):
        typer.secho(f"{folder.name} is not <GCnn|Hnn>_<lower_case_slug>", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if track not in exps.TRACKS:
        typer.secho(f"--track is one of {', '.join(exps.TRACKS)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    taken = {e.id for e in _registry().experiments}
    if eid in taken or folder.exists():
        typer.secho(f"{eid} is taken: ids are never reused", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    shutil.copytree(exps.REPO_ROOT / "experiments" / "_template", folder)
    fill = {
        "{{ID}}": eid,
        "{{ID_LOWER}}": eid.lower(),
        "{{FOLDER}}": folder.name,
        "{{TRACK}}": track,
        "{{OWNER}}": owner or "?",
        "{{DATE}}": dt.date.today().isoformat(),
    }
    for path in folder.rglob("*"):
        if path.is_file():
            text = path.read_text()
            for key, value in fill.items():
                text = text.replace(key, value)
            path.write_text(text)
    typer.echo(f"created {folder.relative_to(exps.REPO_ROOT)}/ -- write PREDICTIONS.md before the first run")


#: What a weights file in a step directory is called, by the trainers this repo has.
_WEIGHTS = (".safetensors", ".ckpt", ".pt", ".pth", ".npz", ".bin")


@experiments_app.command("checkpoint")
def experiments_checkpoint(
    directory: Annotated[str, typer.Argument(help="a step directory holding checkpoint.json")],
    file: Annotated[
        str | None, typer.Option("--file", help="the weights file paddock loads, when there are several")
    ] = None,
    ensemble: Annotated[
        bool, typer.Option("--ensemble", help="the weights files are one checkpoint together (seeds)")
    ] = False,
    trust_sidecar: Annotated[
        bool, typer.Option("--trust-sidecar", help="copy the sidecar's digest without hashing the weights")
    ] = False,
) -> None:
    """Print the [runs.checkpoint] block for a checkpoint directory, pinned by its weights' sha256.

    The weights are hashed here, and a sidecar whose digest disagrees is refused: the record pins the
    bytes on disk, not what the sidecar remembers. A directory with several weights files is refused
    until you say what the checkpoint is: --file names the one paddock loads (the others are recorded
    as members, each with its digest), --ensemble pins them all together (docs/experiments.md).
    """
    from pathlib import Path

    def fail(message: str, code: int = 2) -> typer.Exit:
        typer.secho(message, fg=typer.colors.RED, err=True)
        return typer.Exit(code)

    path = Path(directory)
    try:
        record = ckpt.read(path)
    except ckpt.SidecarError as exc:
        raise fail(str(exc)) from exc
    if file and ensemble:
        raise fail("--file or --ensemble, not both")
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in _WEIGHTS)
    if not files:
        raise fail(f"no weights file ({', '.join(_WEIGHTS)}) in {path}")
    if file:
        if not (path / file).is_file():
            raise fail(f"no {file} in {path}")
    elif not ensemble and len(files) > 1:
        names = ", ".join(p.name for p in files)
        raise fail(
            f"{len(files)} weights files in {path}: {names}\n"
            "  name the one paddock loads with --file, or pin them all together with --ensemble"
        )
    chosen = file or (None if ensemble else files[0].name)

    members: list[str] = []
    if trust_sidecar:
        if not record.weights_sha256:
            raise fail("the sidecar has no weights_sha256 to trust; drop --trust-sidecar")
        digest = record.weights_sha256
    else:
        shas = {p.name: ckpt.sha256_file(p) for p in files}
        members = [exps.member(shas[name], name) for name in sorted(shas)] if len(shas) > 1 else []
        digest = shas[chosen] if chosen else exps.ensemble_digest(members)
        claimed = record.weights_sha256
        if claimed and claimed != digest:
            what = chosen or f"the {len(files)} files together"
            raise fail(f"the sidecar says {claimed}, but {what} hashes to {digest}", code=1)

    lines = [
        "[runs.checkpoint]",
        f'model = "{record.model}"',
        f'train_run = "{record.train_run}"',
        f"step = {record.step}",
        'step_unit = "?"  # epoch | pass | optimizer_step: the sidecar does not say which',
        f'weights_sha256 = "{digest}"'
        + ("  # from the sidecar, NOT re-hashed (--trust-sidecar)" if trust_sidecar else ""),
    ]
    if chosen:
        lines.append(f'file = "{chosen}"')
    if members:
        lines.append("members = [")
        lines += [f'  "{m}",' for m in members]
        lines.append("]")
    lines += [
        f'location = "{{weights}}/{record.model}/{record.train_run}/step_{record.step:09d}/"',
        "sidecar = true",
        f"# train_git_sha = {record.train_git_sha}  (put it on the run)",
    ]
    for t in record.trained_on:
        lines.append(f"# trained_on: {t.dataset} / {t.split}" + (f" ({t.fraction})" if t.fraction else ""))
    if record.selection_split:
        lines.append(f"# selection_split: {record.selection_split}")
    typer.echo("\n".join(lines))


@app.command("train-rerank")
def train_rerank(
    dataset: str = typer.Argument(..., help="dataset from `python -m gallops.data.rerank dataset`"),
    train_run: str = typer.Option(..., help="label for this run; names the checkpoint directory"),
    model: str = typer.Option("drivejepa_rerank", help="paddock adapter name"),
    hidden: int = typer.Option(128, help="MLP hidden units (0 = linear)"),
    temperature: float = typer.Option(1.0, help="soft-label sharpness on ADE"),
    epochs: int = typer.Option(400),
    holdout_frac: float = typer.Option(0.15, help="fraction of training scenes used to pick the epoch"),
) -> None:
    """Train a re-ranker over a frozen model's candidates and write a checkpoint.

    See `gallops.train.rerank` for the method and what is written.
    """
    from pathlib import Path

    from gallops.train import rerank

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    settings = rerank.Settings(
        hidden=hidden, temperature=temperature, epochs=epochs, holdout_frac=holdout_frac
    )
    result = rerank.train(
        Path(dataset),
        cfg,
        model=model,
        train_run=train_run,
        settings=settings,
        train_git_sha=rerank.git_sha(REPO_ROOT),
    )
    typer.echo(f"wrote      {result.directory}")
    typer.echo(f"epoch      {result.epoch}")
    # "selection split", not "held-out": the epoch was chosen on these scenes, so
    # the pick ADE is optimistic. Report from a split nothing was selected on.
    typer.echo(
        f"selection  {result.holdout_samples} samples / {result.holdout_scenes} scenes: "
        f"pick ADE {result.selection_pick_ade:.3f}"
        + (
            f", model's own pick {result.holdout_model_pick_ade:.3f}"
            if result.holdout_model_pick_ade is not None
            else ""
        )
        + f", best of the candidates {result.holdout_best_ade:.3f}"
    )


if __name__ == "__main__":
    app()
