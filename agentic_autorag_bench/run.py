"""Matrix orchestrator: iterate over (method × seed), run each, score held-out.

The bench config (``configs/hotpot_paper.yaml``) declares which methods, which
seeds, what budget, and what held-out scoring settings to use. This module
loads it, prepares HotpotQA once, sets up a shared framework Orchestrator
whose ``evaluate_trial`` is the evaluator every sequential method calls, then
runs each method-seed pair into ``output_root/<method>/seed_<n>/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from agentic_autorag.config.models import TrialConfig
from agentic_autorag.litellm_runtime import configure_litellm_runtime
from agentic_autorag.orchestrator import Orchestrator

from agentic_autorag_bench._holdout_registry import apply_union_exclusion
from agentic_autorag_bench.benchmarks.hotpot_qa import HotpotQABenchmark
from agentic_autorag_bench.methods.agentic import AgenticOptimizer
from agentic_autorag_bench.methods.autorag.driver import AutoRAGOptimizer
from agentic_autorag_bench.methods.bayesian import BayesianSearch
from agentic_autorag_bench.methods.random import RandomSearch
from agentic_autorag_bench.plots import (
    make_matrix_figures,
    make_method_figures,
    make_seed_figures,
)
from agentic_autorag_bench.types import Budget, SearchResult, TrialResult

logger = logging.getLogger("agentic_autorag_bench.run")

STOCHASTIC_METHODS = {"random", "bayesian", "agentic"}
DETERMINISTIC_METHODS = {"autorag_ragas", "autorag_mcq"}
ALL_METHODS = STOCHASTIC_METHODS | DETERMINISTIC_METHODS


def _clear_output_root_for(output_root: Path, methods: list[str]) -> list[str]:
    """Wipe per-run artifacts for the methods about to be run, plus the
    cross-method ``figures/`` dir.

    Scoped on purpose: partial runs (e.g. ``-m agentic``) should compose
    with previous results for the other methods, so only the method dirs
    we're about to overwrite get reset. ``figures/`` is always wiped
    because ``make_matrix_figures`` regenerates it at the end of the run
    from whatever combination of methods now lives in the tree (new
    agentic + old random + old bayesian, say).

    ``.shared_cache/`` and any user files (notes, scratch dirs) are
    untouched: they're not in the wipe set. Backups remain the user's
    responsibility — the bench is last-run-wins for the targeted methods,
    not for the whole tree.

    Returns the names that were removed, for logging.
    """
    if not output_root.exists():
        return []
    targets = [*methods, "figures"]
    removed: list[str] = []
    for name in targets:
        child = output_root / name
        if not child.exists():
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(name)
    return removed


@dataclass
class BenchConfig:
    project_config_path: Path
    methods: list[str]
    seeds: list[int]
    max_trials: int
    hotpot_split: str
    hotpot_sample_size: int | None
    hotpot_prep_seed: int
    hotpot_output_dir: Path
    hold_out_limit: int | None
    hold_out_judge_model: str | None
    hold_out_concurrency: int
    output_root: Path

    @classmethod
    def load(cls, config_path: str | Path) -> BenchConfig:
        """Load and resolve paths.

        Path conventions:
        - ``project_config`` is a sibling-yaml reference, so it resolves
          relative to *this* config's directory.
        - ``hotpot.output_dir`` and ``output_root`` resolve relative to the
          *current working directory*, matching the framework's convention
          for ``meta.corpus_path`` in the project YAML. Mixing the two would
          cause the bench to prepare data at one path and the framework to
          look for it at another.
        """
        config_path = Path(config_path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_path = (config_path.parent / raw["project_config"]).resolve()
        unknown = set(raw["methods"]) - ALL_METHODS
        if unknown:
            raise ValueError(f"Unknown methods in {config_path}: {sorted(unknown)}")
        return cls(
            project_config_path=project_path,
            methods=list(raw["methods"]),
            seeds=list(raw.get("seeds", [42])),
            max_trials=int(raw["budget"]["max_trials"]),
            hotpot_split=raw["hotpot"]["split"],
            hotpot_sample_size=raw["hotpot"].get("sample_size"),
            hotpot_prep_seed=int(raw["hotpot"].get("prep_seed", 42)),
            hotpot_output_dir=Path(raw["hotpot"]["output_dir"]).resolve(),
            hold_out_limit=raw["hold_out"].get("limit"),
            hold_out_judge_model=raw["hold_out"].get("judge_model"),
            hold_out_concurrency=int(raw["hold_out"].get("concurrency", 10)),
            output_root=Path(raw["output_root"]).resolve(),
        )


def _persist_search_result(sr: SearchResult, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "search_result.json").write_text(json.dumps(sr.to_dict(), indent=2), encoding="utf-8")
    (dest / "best_config.yaml").write_text(yaml.safe_dump(sr.best_config, sort_keys=False), encoding="utf-8")
    # Agentic's framework Orchestrator writes a richer ``history.jsonl`` during
    # ``orch.run()`` — per-question results, judge breakdown, proposer meta,
    # diagnosis, frontier flags. The bench's reduced HistoryEntry is a strict
    # subset (trial_number / config / score / metrics / eval_usd), all of which
    # also exist on the framework's TrialRecord, so analyze.py reads either
    # equivalently. Don't overwrite the rich version.
    history_path = dest / "history.jsonl"
    if sr.method != "agentic" or not history_path.exists():
        history_path.write_text(
            "\n".join(json.dumps(h.to_dict()) for h in sr.history) + ("\n" if sr.history else ""),
            encoding="utf-8",
        )
    (dest / "optimizer_meta.json").write_text(
        json.dumps(
            {
                "method": sr.method,
                "seed": sr.seed,
                "deterministic": sr.deterministic,
                "optimizer_usd": sr.optimizer_usd,
                "trial_usd_total": sr.trial_usd_total,
                "wall_clock_s": sr.wall_clock_s,
                "n_trials_completed": len(sr.history),
                "extras": sr.extras,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _build_optimizer(
    name: str,
    *,
    project,
    bench: BenchConfig,
    output_dir: Path,
    debug_prompts: bool = False,
):
    if name == "random":
        return RandomSearch(project=project)
    if name == "bayesian":
        return BayesianSearch(project=project, storage_dir=output_dir)
    if name == "agentic":
        return AgenticOptimizer(
            config_path=str(bench.project_config_path),
            output_dir=str(output_dir),
            debug_prompts=debug_prompts,
        )
    if name in {"autorag_ragas", "autorag_mcq"}:
        variant = "ragas" if name == "autorag_ragas" else "mcq"
        return AutoRAGOptimizer(
            config_path=str(bench.project_config_path),
            output_dir=str(output_dir),
            qa_variant=variant,
        )
    raise ValueError(f"Unknown method {name!r}")


async def run_matrix(
    config_path: str | Path,
    *,
    methods_override: list[str] | None = None,
    debug_prompts: bool = False,
    clean: bool = True,
) -> None:
    # litellm.drop_params=True so provider-specific params (seed, temperature
    # on gpt-5) are silently dropped instead of erroring. The framework's
    # ``agentic-autorag optimize`` CLI calls this; the bench has its own entry
    # point and must call it explicitly to inherit identical LLM semantics.
    configure_litellm_runtime()
    bench = BenchConfig.load(config_path)
    if methods_override is not None:
        unknown = set(methods_override) - ALL_METHODS
        if unknown:
            raise ValueError(f"Unknown methods in --methods override: {sorted(unknown)}")
        missing = set(methods_override) - set(bench.methods)
        if missing:
            raise ValueError(
                f"--methods includes {sorted(missing)} which are not in {config_path}; "
                "edit the config or pick a subset of its methods"
            )
        bench.methods = [m for m in bench.methods if m in set(methods_override)]

    if clean:
        removed = _clear_output_root_for(bench.output_root, bench.methods)
        if removed:
            logger.info(
                "Cleared %s under %s before run; other entries (including "
                "method dirs not in this run) preserved. Pass --no-clean "
                "to keep them.",
                sorted(removed), bench.output_root,
            )

    benchmark = HotpotQABenchmark(
        output_dir=bench.hotpot_output_dir,
        split=bench.hotpot_split,
        sample_size=bench.hotpot_sample_size,
        seed=bench.hotpot_prep_seed,
    )
    benchmark.prepare()

    # Shared orchestrator: provides the evaluator that every sequential method
    # calls per trial. setup() is idempotent — the parsed corpus, exam.json,
    # and ingredient cache live under the project YAML's meta.output_dir
    # (./results/.shared_cache by default) so they're reused across methods.
    logger.info("Setting up shared orchestrator (will generate exam.json on first run)")
    shared = Orchestrator(str(bench.project_config_path))
    shared.evaluator.quiet_per_question = True
    try:
        await shared.setup()

        async def evaluator(config: TrialConfig) -> TrialResult:
            return TrialResult.from_exam_result(await shared.evaluate_trial(config))

        budget = Budget(max_trials=bench.max_trials)

        for method_name in bench.methods:
            seeds_for_method = bench.seeds if method_name in STOCHASTIC_METHODS else [None]
            for seed in seeds_for_method:
                seed_label = f"seed_{seed}" if seed is not None else "default"
                method_dir = bench.output_root / method_name / seed_label
                logger.info("=" * 60)
                logger.info("RUNNING %s | %s", method_name, seed_label)
                logger.info("=" * 60)

                optimizer = _build_optimizer(
                    method_name,
                    project=shared.config,
                    bench=bench,
                    output_dir=method_dir,
                    debug_prompts=debug_prompts,
                )
                try:
                    sr = await optimizer.search(evaluator, budget, seed=seed)
                except Exception:
                    logger.exception("%s seed=%s failed", method_name, seed)
                    continue

                _persist_search_result(sr, method_dir)
                logger.info(
                    "%s seed=%s done | best_score=%.3f | trials=%d | wall=%.1fs | trial_usd=$%.4f | optim_usd=$%.4f",
                    method_name, seed,
                    max((h.score for h in sr.history), default=0.0),
                    len(sr.history),
                    sr.wall_clock_s,
                    sr.trial_usd_total,
                    sr.optimizer_usd,
                )

                # Held-out scoring on the same QA, same evaluator semantics.
                trial_config = TrialConfig(**sr.best_config)
                await benchmark.evaluate(
                    project_config_path=str(bench.project_config_path),
                    trial_config=trial_config,
                    output_path=method_dir / "benchmark_results.json",
                    judge_model=bench.hold_out_judge_model,
                    limit=bench.hold_out_limit,
                    concurrency=bench.hold_out_concurrency,
                )

                # Per-seed figures: render as soon as one (method, seed) finishes
                # its hold-out so the user can inspect a run mid-matrix.
                make_seed_figures(method_dir)

            # Per-method figures: aggregate every seed for this method now that
            # the inner loop has closed. Cross-method matrix figures wait for
            # the outer loop.
            make_method_figures(bench.output_root / method_name)
    finally:
        await shared.cleanup()

    # Cross-method content-filter exclusion: drop any question that any
    # method's best config got rejected on, so all rows score the same
    # denominator. Runs after every hold-out so the union is complete.
    apply_union_exclusion(bench.output_root)

    # Matrix figures see the union-exclusion-adjusted hold-out scores; calling
    # before apply_union_exclusion would bake stale per-question denominators
    # into the table.
    make_matrix_figures(bench.output_root)


def run_cli(
    config_path: str,
    *,
    methods: list[str] | None = None,
    debug_prompts: bool = False,
    clean: bool = True,
) -> None:
    """Sync wrapper for the Typer CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    # Chatty libraries — we want their warnings, not their per-call INFO lines.
    # Mirrors Agentic-AutoRAG/agentic_autorag/cli.py so per-query "Batches:"
    # bars and per-load model-init INFO chatter don't pollute the terminal.
    for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run_logger = logging.getLogger("agentic_autorag_bench.run")
    run_logger.setLevel(logging.INFO)
    asyncio.run(run_matrix(
        config_path,
        methods_override=methods,
        debug_prompts=debug_prompts,
        clean=clean,
    ))
