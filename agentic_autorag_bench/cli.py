"""Typer CLI entry point for the bench suite."""

from __future__ import annotations

import typer

app = typer.Typer(name="agentic-autorag-bench", help="Agentic AutoRAG benchmark suite")


@app.command()
def version() -> None:
    """Print the bench suite version."""
    from agentic_autorag_bench import __version__

    print(__version__)


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to paper-mode YAML"),
) -> None:
    """Run the (method × seed) matrix described by the YAML config."""
    from agentic_autorag_bench.run import run_cli

    run_cli(config)


@app.command()
def analyze(
    results_dir: str = typer.Option("results/", "--results-dir", help="Where the matrix run wrote outputs"),
    output: str = typer.Option("paper_artifacts/", "--output", "-o", help="Where to write Table_1.tex + figures"),
) -> None:
    """Aggregate committed results into paper artifacts (LaTeX table + comparison figures)."""
    import logging

    from agentic_autorag_bench.analyze import analyze as run_analyze

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    run_analyze(results_dir, output)


if __name__ == "__main__":
    app()
