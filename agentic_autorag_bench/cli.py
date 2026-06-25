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
    clean: bool | None = typer.Option(
        None,
        "--clean/--no-clean",
        help="Reset the per-method dirs about to be run (and any matching "
        "@k checkpoint dirs) so their contents reflect the current run "
        "only. Defaults ON for a fresh run, but ``--resume`` implies "
        "``--no-clean`` automatically. ``figures/`` is NOT wiped — matrix "
        "figures are staged and atomically swapped at end-of-run so the "
        "previous figures stay readable throughout. Method dirs not in this "
        "run, ``.shared_cache/``, and user files at output_root are also "
        "preserved.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume each selected method from its last successfully completed "
        "trial. Per-(method, seed) directories with prior trial state on "
        "disk continue from trial K+1; empty dirs start fresh. A trial "
        "interrupted mid-evaluation is discarded and re-attempted. "
        "Implies --no-clean. Typical use after a Ctrl+C or a crash: "
        "`--resume` (optionally with `-m motpe`).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow a --clean start to wipe method dirs that already contain "
        "COMPLETED hold-out results. Without it, a clean start refuses to "
        "delete finished work (so an accidental re-launch of the documented "
        "`run` command after a crash can't destroy days of results) — pass "
        "--resume to continue instead, or --force to deliberately restart.",
    ),
) -> None:
    """Run the (method × seed) matrix described by the YAML config."""
    # ``--resume`` implies ``--no-clean``: only error if the user EXPLICITLY
    # asked for --clean alongside --resume (clean is True, not the None default).
    if resume and clean is True:
        raise typer.BadParameter(
            "--resume and --clean are mutually exclusive: --resume needs the "
            "prior method dirs intact to continue from. Drop --clean (--resume "
            "already implies --no-clean), or drop --resume to start a fresh clean run."
        )
    effective_clean = False if resume else (True if clean is None else clean)
    from agentic_autorag_bench.run import run_cli

    run_cli(
        config,
        methods=methods or None,
        clean=effective_clean,
        resume=resume,
        force=force,
    )


@app.command("replay-holdout")
def replay_holdout(
    config: str = typer.Option(..., "--config", "-c", help="Path to paper-mode YAML (same one passed to `run`)"),
    n_runs: int = typer.Option(
        3,
        "--n-runs",
        help="Target number of hold-out evals per (method, seed). The "
        "end-of-search benchmark_results.json counts as run 1; this "
        "command tops up to n_runs by writing run_002.json, "
        "run_003.json, ... under holdout_replays/. Default 3.",
    ),
    methods: list[str] = typer.Option(
        None,
        "--methods",
        "-m",
        help="Subset of base methods to top up. Matches both bare method "
        "dirs and @k checkpoint variants (e.g. --methods agentic_score "
        "covers agentic_score, agentic_score@10, agentic_score@20).",
    ),
    include_checkpoints: bool = typer.Option(
        True,
        "--include-checkpoints/--no-include-checkpoints",
        help="Replay the @k checkpoint dirs alongside the main method dirs. "
        "Default ON so the headline figure has consistent N across all "
        "bars. Pass --no-include-checkpoints to skip them (e.g. when "
        "iterating on the main bars to save eval cost).",
    ),
) -> None:
    """Top up each (method, seed) dir to N hold-out evals so the matrix
    figure shows mean ± SD across replays.

    Idempotent: re-running picks up only the missing run indices. After
    every eval, re-applies the cross-method content-filter union and
    re-renders matrix figures + Table_1.md.
    """
    from agentic_autorag_bench.replay import replay_holdout_cli

    replay_holdout_cli(
        config,
        n_runs=n_runs,
        methods=methods or None,
        include_checkpoints=include_checkpoints,
    )


@app.command()
def pareto(
    config: str = typer.Option(..., "--config", "-c", help="Path to a pareto YAML (e.g. configs/unidoc_pareto.yaml)"),
    figure_only: bool = typer.Option(
        False,
        "--figure-only",
        help="Skip the search; just re-render figures/pareto.png from an existing "
        "results_*/agentic_cost/seed_N/ (details/history.jsonl). Useful "
        "for tweaking the figure without re-running the optimizer.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an interrupted agentic_cost search from its last completed trial.",
    ),
) -> None:
    """Run agentic_cost's search on the UniDoc corpus and render a Syftr-style
    cost-vs-accuracy Pareto figure of its trials.

    Scores trials on the optimizer's own self-generated exam (no held-out QA).
    Downloads the UniDoc corpus on first run if it isn't present yet.
    """
    from agentic_autorag_bench.pareto import pareto_cli

    pareto_cli(config, figure_only=figure_only, resume=resume)


@app.command("kb-greedy")
def kb_greedy(
    config: str = typer.Option(..., "--config", "-c", help="Path to paper-mode YAML (same one passed to `run`)"),
    seed: int = typer.Option(42, "--seed", help="Seed label for the kb_greedy/seed_<n>/ output dir."),
) -> None:
    """Evaluate the KB's strongest pipeline once on the held-out gold (no search).

    A reference bar: the most-capable config (strongest LLM/embedder/reranker,
    max chunk/top_k/reranker_top_n) scored once on the same hold-out as the
    matrix methods. Writes ``kb_greedy/seed_<n>/benchmark_results.json``.
    """
    import asyncio
    import logging

    from agentic_autorag_bench.methods.kb_greedy import run_kb_greedy

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run_kb_greedy(config, seed=seed))


@app.command()
def analyze(
    results_dir: str = typer.Option("results_paper/", "--results-dir", help="Where the matrix run wrote outputs"),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
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
