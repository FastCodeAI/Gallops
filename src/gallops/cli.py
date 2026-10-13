"""Command line for gallops.

Deliberately small. This repo's job is to produce checkpoints; the interesting
verbs will arrive with the training code. What exists today is what you need to
confirm the plumbing before writing any of it -- which is the same discipline
paddock's `adding-a-model` guide prescribes for adapters: get the boring path
green first, so that when something moves you know it was the model.
"""

from __future__ import annotations

import json

import typer

from gallops import checkpoint as ckpt
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
