"""Typer CLI entry point for the bench suite."""

from __future__ import annotations

import typer

app = typer.Typer(name="autorag-bench", help="Agentic AutoRAG benchmark suite")


@app.command()
def version() -> None:
    """Print the bench suite version."""
    from autorag_bench import __version__

    print(__version__)


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to paper-mode YAML"),
    methods: str | None = typer.Option(
        None, "--methods", "-m", help="Comma-separated subset (random,bayesian,agentic,autorag_ragas,autorag_mcq)"
    ),
    seeds: str | None = typer.Option(
        None, "--seeds", help="Comma-separated seed list; overrides config.seeds"
    ),
) -> None:
    """Run the (method × seed) matrix described by the YAML config."""
    raise NotImplementedError("Wired up in the orchestrator commit")


@app.command()
def analyze(
    results_dir: str = typer.Option("results/", "--results-dir", help="Where the matrix run wrote outputs"),
    output: str = typer.Option("paper_artifacts/", "--output", "-o", help="Where to write Table_1.tex + figures"),
) -> None:
    """Aggregate committed results into paper artifacts (LaTeX table + trajectory figure)."""
    raise NotImplementedError("Wired up in the analyzer commit")


if __name__ == "__main__":
    app()
