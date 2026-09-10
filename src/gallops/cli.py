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
from gallops.config import ConfigError, SiteConfig

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


if __name__ == "__main__":
    app()
