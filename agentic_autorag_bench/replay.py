"""Top-up hold-out replays for already-discovered best configs.

Each (method, seed) dir under ``output_root`` carries a single hold-out
score in ``benchmark_results.json`` from end-of-search. That's run 1.
``replay-holdout`` re-evaluates the same ``best_config.yaml`` against the
same hold-out QA two more times (or however many runs ``--n-runs``
declares), writing ``holdout_replays/run_002.json``,
``holdout_replays/run_003.json``, ... so the matrix chart's "mean ± SD"
bars are computed from N independent hold-out evals rather than a
bootstrap over per-question rows of a single eval.

Idempotent: re-running this command finds existing replay files, skips
them, and only runs the missing ones. After all evals, the union
content-filter exclusion runs again so any CONTENT_FILTER row in a
replay file participates in the cross-method denominator, and matrix
figures are re-rendered with the new ``mean ± SD``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import yaml
from agentic_autorag.config.models import TrialConfig
from agentic_autorag.litellm_runtime import configure_litellm_runtime

from agentic_autorag_bench._holdout_registry import apply_union_exclusion
from agentic_autorag_bench.benchmarks.runner import BenchmarkRunner
from agentic_autorag_bench.plots import make_matrix_figures
from agentic_autorag_bench.run import BenchConfig

logger = logging.getLogger("agentic_autorag_bench.run")

_REPLAYS_DIR_NAME = "holdout_replays"
_REPLAY_FILE_RE = re.compile(r"run_(\d+)\.json$")
_NON_METHOD_DIRS = {"figures", ".shared_cache", "_figures_staging", "_figures_previous"}


def _existing_replay_indices(seed_dir: Path) -> set[int]:
    """Return the set of run indices that already have a replay file.

    ``benchmark_results.json`` is run 1 by convention. Replay files live
    in ``holdout_replays/run_NNN.json`` with NNN >= 2 (so they don't clash
    with the original).
    """
    indices: set[int] = set()
    if (seed_dir / "benchmark_results.json").exists():
        indices.add(1)
    replays_dir = seed_dir / _REPLAYS_DIR_NAME
    if replays_dir.is_dir():
        for p in replays_dir.glob("run_*.json"):
            m = _REPLAY_FILE_RE.search(p.name)
            if m:
                indices.add(int(m.group(1)))
    return indices


def _discover_targets(
    output_root: Path,
    *,
    methods_filter: set[str] | None,
    include_checkpoints: bool,
) -> list[Path]:
    """Find every ``{method}/seed_*/`` dir eligible for hold-out replays.

    A dir is eligible iff it carries both ``best_config.yaml`` (so we have
    a config to re-evaluate) and ``benchmark_results.json`` (so run 1
    already happened and the dir is past the end-of-search stage).
    ``@k`` checkpoint method dirs are included by default; pass
    ``include_checkpoints=False`` to skip them.

    ``methods_filter``, when set, restricts to method names that are
    either an exact match or whose base method (before ``@``) matches.
    This means ``--methods agentic_score`` includes both
    ``agentic_score/`` and ``agentic_score@10/``, which is what a user
    asking "top up the agentic-score family" expects.
    """
    targets: list[Path] = []
    if not output_root.exists():
        return targets
    for method_dir in sorted(output_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name in _NON_METHOD_DIRS:
            continue
        is_checkpoint = "@" in method_dir.name
        if is_checkpoint and not include_checkpoints:
            continue
        if methods_filter is not None:
            base = method_dir.name.split("@", 1)[0]
            if method_dir.name not in methods_filter and base not in methods_filter:
                continue
        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            if not (seed_dir / "best_config.yaml").is_file():
                continue
            if not (seed_dir / "benchmark_results.json").is_file():
                logger.warning(
                    "Skipping %s: best_config.yaml exists but benchmark_results.json "
                    "(run 1) is missing — search may have crashed before hold-out",
                    seed_dir,
                )
                continue
            targets.append(seed_dir)
    return targets


async def replay_holdout(
    config_path: str | Path,
    *,
    n_runs: int = 3,
    methods: list[str] | None = None,
    include_checkpoints: bool = True,
) -> None:
    """Bring every eligible (method, seed) dir up to ``n_runs`` hold-out evals.

    Loads the bench config to get ``hold_out`` settings (judge model, limit,
    concurrency) and the benchmark adapter identity. Each replay reuses the
    same ``BenchmarkRunner.evaluate(...)`` entry point as the in-run hold-out
    so scoring semantics match byte-for-byte.

    The original ``benchmark_results.json`` is never overwritten by the
    replay loop — only the post-pass ``apply_union_exclusion`` may rewrite
    it (idempotently) to update ``excluded_question_ids`` if a new replay
    surfaced a content-filter row.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    configure_litellm_runtime()
    bench = BenchConfig.load(config_path)
    benchmark = BenchmarkRunner(
        name=bench.benchmark.name,
        output_dir=bench.benchmark.output_dir,
        split=bench.benchmark.split,
        sample_size=bench.benchmark.sample_size,
        seed=bench.benchmark.prep_seed,
    )
    benchmark.prepare()

    methods_filter = set(methods) if methods else None
    targets = _discover_targets(
        bench.output_root,
        methods_filter=methods_filter,
        include_checkpoints=include_checkpoints,
    )
    if not targets:
        logger.info(
            "No replay targets under %s (methods_filter=%s, include_checkpoints=%s)",
            bench.output_root,
            methods_filter,
            include_checkpoints,
        )
        return

    plan: list[tuple[Path, list[int]]] = []
    for seed_dir in targets:
        existing = _existing_replay_indices(seed_dir)
        missing = sorted(set(range(1, n_runs + 1)) - existing)
        if missing:
            plan.append((seed_dir, missing))
        else:
            logger.info(
                "%s already has %d runs; nothing to do",
                seed_dir.relative_to(bench.output_root),
                len(existing),
            )

    total_evals = sum(len(missing) for _, missing in plan)
    logger.info(
        "Replay plan: %d eval(s) across %d (method, seed) dir(s); target n_runs=%d",
        total_evals,
        len(plan),
        n_runs,
    )

    for seed_dir, missing in plan:
        replays_dir = seed_dir / _REPLAYS_DIR_NAME
        replays_dir.mkdir(parents=True, exist_ok=True)
        best_config_path = seed_dir / "best_config.yaml"
        config_dict = yaml.safe_load(best_config_path.read_text(encoding="utf-8"))
        trial_config = TrialConfig(**config_dict)
        rel = seed_dir.relative_to(bench.output_root)
        for run_idx in missing:
            if run_idx == 1:
                # Run 1 is reserved for the original benchmark_results.json,
                # which the search loop wrote at end-of-search. If it's
                # missing here, the dir was rejected by _discover_targets;
                # we should never reach this branch.
                logger.warning(
                    "Refusing to write run_001 from replay loop; %s should already have benchmark_results.json",
                    seed_dir,
                )
                continue
            out_path = replays_dir / f"run_{run_idx:03d}.json"
            logger.info("=" * 60)
            logger.info("REPLAY %s | run %d → %s", rel, run_idx, out_path.name)
            logger.info("=" * 60)
            await benchmark.evaluate(
                project_config_path=str(bench.project_config_path),
                trial_config=trial_config,
                output_path=out_path,
                judge_model=bench.hold_out_judge_model,
                limit=bench.hold_out_limit,
                concurrency=bench.hold_out_concurrency,
                exclude_question_types=bench.hold_out_exclude_question_types,
            )

    # Cross-method union exclusion now sees the new replay rows; rewrites
    # every benchmark_results.json + holdout_replays/*.json with the
    # widened excluded set. Idempotent — running with no new files is a
    # cheap no-op rewrite.
    apply_union_exclusion(bench.output_root)
    make_matrix_figures(bench.output_root)


def replay_holdout_cli(
    config_path: str,
    *,
    n_runs: int = 3,
    methods: list[str] | None = None,
    include_checkpoints: bool = True,
) -> None:
    """Sync wrapper for the Typer CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run_logger = logging.getLogger("agentic_autorag_bench.run")
    run_logger.setLevel(logging.INFO)
    asyncio.run(
        replay_holdout(
            config_path,
            n_runs=n_runs,
            methods=methods,
            include_checkpoints=include_checkpoints,
        )
    )
