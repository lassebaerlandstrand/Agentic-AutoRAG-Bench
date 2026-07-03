"""Typer CLI entry point for the bench suite."""

from __future__ import annotations

from pathlib import Path

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


@app.command("prepare-splits")
def prepare_splits(
    qa: str = typer.Option(..., "--qa", help="Path to a prepared benchmark qa.json"),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output dir for holdout_qa.json + optimization_qa.json + provenance"
    ),
    stratify_key: str | None = typer.Option(
        None,
        "--stratify-key",
        help="metadata field to stratify by (n_hops, type, question_type); auto-detected if omitted",
    ),
    holdout_size: int = typer.Option(300, "--holdout-size", help="Held-out gold slice size"),
    seed: int = typer.Option(42, "--seed", help="Deterministic split seed"),
) -> None:
    """Draw a stratified held-out slice from a benchmark qa.json.

    Replaces the biased contiguous held-out slice: the held-out mirrors the
    pool's difficulty mix, doc-less rows are excluded, and every remaining usable
    row becomes the optimization reservoir the real-QA exam is built from.
    """
    import logging

    from agentic_autorag.benchmarks import load_qa

    from agentic_autorag_bench.splits import stratified_split, write_splits

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    pairs = load_qa(Path(qa))
    result = stratified_split(pairs, stratify_key=stratify_key, holdout_size=holdout_size, seed=seed)
    paths = write_splits(result, Path(output))
    prov = result.provenance
    print(f"Stratified split by {prov.stratify_key!r} (seed={prov.seed})")
    print(f"  pool: {prov.n_pool_usable} usable of {prov.n_pool_total} ({prov.n_excluded_no_docs} doc-less excluded)")
    print(f"  holdout ({prov.holdout_size}):   {prov.holdout_distribution}  -> {paths['holdout']}")
    print(f"  reservoir ({prov.opt_size}): {prov.opt_distribution}  -> {paths['optimization']}")
    print(f"  disjoint: {prov.disjoint}  provenance -> {paths['provenance']}")


@app.command("real-exam")
def real_exam(
    project_config: str = typer.Option(
        ..., "--project-config", help="Project YAML (corpus_path + model aliases + examiner model)"
    ),
    splits_dir: str = typer.Option(
        ..., "--splits-dir", help="Directory from prepare-splits (optimization_qa.json + split_provenance.json)"
    ),
    output: str = typer.Option(..., "--output", "-o", help="Destination for the real-QA exam JSON"),
    exam_size: int = typer.Option(100, "--exam-size", help="Number of tier-C questions in the exam"),
    extractor_model: str | None = typer.Option(
        None, "--extractor-model", help="LLM for span extraction; defaults to the project's examiner model"
    ),
    concurrency: int = typer.Option(10, help="Concurrent extraction calls"),
    seed: int = typer.Option(42, "--seed", help="Deterministic draw seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build a 100 %-grounded real-QA exam from a benchmark's own questions.

    Every question is tier C (verbatim evidence spans) and the difficulty mix
    matches the held-out slice; ungroundable draws are replaced until each
    stratum is full. Point ``examiner.custom_exam_path`` at the output.
    """
    import asyncio
    import json
    import logging

    from agentic_autorag.benchmarks import load_qa
    from agentic_autorag.config.loader import load_config
    from agentic_autorag.engine._io import load_direct_read_corpus
    from agentic_autorag.litellm_runtime import configure_litellm_runtime, install_model_aliases

    from agentic_autorag_bench.real_exam import build_real_exam, write_real_exam

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    if not verbose:
        for noisy in ("LiteLLM", "litellm", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    splits = Path(splits_dir)
    optimization_pool = load_qa(splits / "optimization_qa.json")
    holdout_distribution = json.loads((splits / "split_provenance.json").read_text(encoding="utf-8"))[
        "holdout_distribution"
    ]

    project = load_config(project_config)
    configure_litellm_runtime(project.model_aliases)
    install_model_aliases(project.model_aliases)
    doc_ids, texts = load_direct_read_corpus(Path(project.meta.corpus_path))
    corpus = dict(zip(doc_ids, texts, strict=True))

    exam, provenance = asyncio.run(
        build_real_exam(
            optimization_pool,
            corpus,
            holdout_distribution,
            extractor_model=extractor_model or project.agent.examiner_model,
            reasoning_effort=project.agent.examiner_reasoning_effort,
            exam_size=exam_size,
            fuzzy_threshold=project.examiner.source_fact_verify_fuzzy_threshold,
            concurrency=concurrency,
            seed=seed,
        )
    )
    prov_path = write_real_exam(exam, provenance, Path(output))
    print(f"Real-QA exam: {len(exam)} tier-C questions -> {output}")
    print(f"  target mix (held-out): {provenance.target_distribution}")
    print(f"  exam mix:              {provenance.exam_distribution}")
    print(f"  extracted per stratum: {provenance.per_stratum_extracted}")
    print(f"  provenance -> {prov_path}")


if __name__ == "__main__":
    app()
