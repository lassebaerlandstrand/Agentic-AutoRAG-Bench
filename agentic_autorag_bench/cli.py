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
    methods: list[str] = typer.Option(
        None,
        "--methods",
        "-m",
        help="Subset of methods to run (must be present in the config; repeat flag for multiple).",
    ),
    debug_prompts: bool = typer.Option(
        True,
        "--debug-prompts/--no-debug-prompts",
        help="For the agentic method, log every proposer prompt + response to run.log. "
             "Defaults to ON because proposer-side bugs aren't reproducible from per-trial "
             "JSON alone; pass --no-debug-prompts to silence.",
    ),
    clean: bool = typer.Option(
        True,
        "--clean/--no-clean",
        help="Reset the per-method dirs about to be run (and any matching "
             "@k checkpoint dirs) so their contents reflect the current run "
             "only. ``figures/`` is NOT wiped — matrix figures are staged "
             "and atomically swapped at end-of-run so the previous figures "
             "stay readable throughout. Method dirs not in this run, "
             "``.shared_cache/``, and user files at output_root are also "
             "preserved. Pass --no-clean to keep prior files in place "
             "without resuming trial state (use --resume for that).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume each selected method from its last successfully completed "
             "trial. Per-(method, seed) directories with prior trial state on "
             "disk continue from trial K+1; empty dirs start fresh. A trial "
             "interrupted mid-evaluation is discarded and re-attempted. "
             "Implies --no-clean (mutually exclusive with --clean). Typical "
             "use after a Ctrl+C: `--methods bayesian --resume`.",
    ),
) -> None:
    """Run the (method × seed) matrix described by the YAML config."""
    if resume and clean:
        raise typer.BadParameter(
            "--resume and --clean are mutually exclusive: --resume needs the "
            "prior method dirs intact to continue from. Pass `--no-clean` "
            "explicitly together with --resume, or drop --resume to start fresh."
        )
    from agentic_autorag_bench.run import run_cli

    run_cli(
        config,
        methods=methods or None,
        debug_prompts=debug_prompts,
        clean=clean,
        resume=resume,
    )


@app.command()
def analyze(
    results_dir: str = typer.Option("results_paper/", "--results-dir", help="Where the matrix run wrote outputs"),
    output: str = typer.Option(
        "",
        "--output", "-o",
        help="Where to write the figures/ subtree. Empty (default) means write "
             "alongside the run data, i.e. <results-dir>/figures/. Pass an "
             "explicit path to re-render figures into a separate directory.",
    ),
) -> None:
    """Re-render matrix-level figures + Table_1.md from a committed results tree.

    Per-method and per-seed figures are auto-emitted by ``run`` and live under
    the run's output tree; this command re-renders the matrix-level views
    (and ``Table_1.md``) without re-running the matrix.
    """
    import logging

    from agentic_autorag_bench.analyze import analyze as run_analyze

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    run_analyze(results_dir, output or results_dir)


if __name__ == "__main__":
    app()
