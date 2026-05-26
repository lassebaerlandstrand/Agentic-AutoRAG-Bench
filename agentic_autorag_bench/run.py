"""Matrix orchestrator: iterate over (method × seed), run each, score held-out.

A bench config (e.g. ``configs/hotpot_paper.yaml``) declares which methods,
which seeds, what budget, what benchmark to materialise, and what held-out
scoring settings to use. This module loads it, prepares the benchmark once,
sets up a shared framework Orchestrator whose ``evaluate_trial`` is the
evaluator every sequential method calls, then runs each method-seed pair into
``output_root/<method>/seed_<n>/``.

One bench config = one benchmark. To evaluate multiple benchmarks, run the
matrix once per benchmark config (each into its own ``output_root``); the
shared ``Orchestrator`` is built around a single corpus and exam.json, so
multi-benchmark belongs at the config layer, not inside one run.
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
from agentic_autorag.cost_ledger import CostLedger, get_active_ledger, reset_active_ledger, set_active_ledger
from agentic_autorag.litellm_runtime import configure_litellm_runtime
from agentic_autorag.orchestrator import Orchestrator

from agentic_autorag_bench._holdout_registry import apply_union_exclusion
from agentic_autorag_bench.benchmarks.runner import BenchmarkRunner
from agentic_autorag_bench.methods.agentic import AgenticOptimizer
from agentic_autorag_bench.methods.autorag.driver import AutoRAGOptimizer, resolve_autorag_python
from agentic_autorag_bench.methods.bayesian import BayesianSearch
from agentic_autorag_bench.methods.random import RandomSearch
from agentic_autorag_bench.plots import (
    make_matrix_figures,
    make_method_figures,
    make_seed_figures,
)
from agentic_autorag_bench.types import Budget, SearchResult, TrialResult

logger = logging.getLogger("agentic_autorag_bench.run")

STOCHASTIC_METHODS = {"random", "bayesian", "agentic_score", "agentic_cost"}
DETERMINISTIC_METHODS = {"autorag_ragas", "autorag_our_exam"}
ALL_METHODS = STOCHASTIC_METHODS | DETERMINISTIC_METHODS
# Methods that share the bench's ``shared`` Orchestrator via ``evaluate_trial``.
# These need the bench to install a per-(method, seed) cost ledger and reset
# ``shared._seen_emb_fps`` between runs; agentic/autorag instantiate their own
# Orchestrators so they manage their own ledger lifecycle.
_SHARED_EVALUATOR_METHODS = {"random", "bayesian"}


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


def _write_bench_metadata(output_root: Path, bench: BenchConfig) -> None:
    """Persist the benchmark + run identity at ``output_root/bench_metadata.json``.

    Downstream readers (``analyze.py``, ``plots.py``) consult this file to
    surface the right benchmark name in tables and figure titles, so they
    don't have to re-parse the source YAML config.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": {
            "name": bench.benchmark.name,
            "split": bench.benchmark.split,
            "sample_size": bench.benchmark.sample_size,
            "prep_seed": bench.benchmark.prep_seed,
        },
        "project_config_path": str(bench.project_config_path),
        "methods": bench.methods,
        "seeds": bench.seeds,
        "max_trials": bench.max_trials,
    }
    (output_root / "bench_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


@dataclass(frozen=True)
class BenchmarkSpec:
    """Which benchmark to materialise, and how.

    ``name`` is the adapter key from the framework's ``ADAPTERS`` registry
    (``agentic_autorag.benchmarks.__init__.py``) — e.g. ``hotpot_qa``,
    ``musique``, ``multihop_rag``. Dispatch happens inside
    ``BenchmarkRunner.prepare()``.
    """

    name: str
    split: str
    sample_size: int | None
    prep_seed: int
    output_dir: Path


@dataclass
class BenchConfig:
    project_config_path: Path
    methods: list[str]
    seeds: list[int]
    max_trials: int
    benchmark: BenchmarkSpec
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
        - ``benchmark.output_dir`` and ``output_root`` resolve relative to the
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
        b = raw["benchmark"]
        benchmark = BenchmarkSpec(
            name=b["name"],
            split=b["split"],
            sample_size=b.get("sample_size"),
            prep_seed=int(b.get("prep_seed", 42)),
            output_dir=Path(b["output_dir"]).resolve(),
        )
        return cls(
            project_config_path=project_path,
            methods=list(raw["methods"]),
            seeds=list(raw.get("seeds", [42])),
            max_trials=int(raw["budget"]["max_trials"]),
            benchmark=benchmark,
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
    if not sr.method.startswith("agentic_") or not history_path.exists():
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
                "prompt_tokens": sr.prompt_tokens,
                "completion_tokens": sr.completion_tokens,
                "embedding_tokens": sr.embedding_tokens,
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
    if name in {"agentic_score", "agentic_cost"}:
        return AgenticOptimizer(
            config_path=str(bench.project_config_path),
            output_dir=str(output_dir),
            cost_aware=(name == "agentic_cost"),
            debug_prompts=debug_prompts,
        )
    if name in {"autorag_ragas", "autorag_our_exam"}:
        variant = "ragas" if name == "autorag_ragas" else "our_exam"
        return AutoRAGOptimizer(
            config_path=str(bench.project_config_path),
            output_dir=str(output_dir),
            qa_variant=variant,
        )
    raise ValueError(f"Unknown method {name!r}")


def _make_metered_evaluator(shared: Orchestrator, method_dir: Path):
    """Wrap ``shared.evaluate_trial`` with per-trial ledger snapshot/delta.

    Used by methods that drive the shared Orchestrator (random / bayesian).
    Captures the cost-ledger delta over each trial, writes one line to
    ``method_dir/trial_cost_ledger.jsonl``, flushes any pending cache-event
    credits the orchestrator queued via ``_credit_embedding_build``, and
    returns a ``TrialResult`` with token totals filled in from the delta.

    Per-trial accounting excludes pre-trial setup spend (exam generation,
    endpoint verification, probe-phase embedding builds) because those land
    in the ledger before this evaluator's first ``snapshot`` call. This is
    the bench's fairness convention — see ``TrialResult`` for the full rule.
    """
    trial_counter = [0]

    async def evaluator(config: TrialConfig) -> TrialResult:
        trial_counter[0] += 1
        trial_num = trial_counter[0]
        ledger = get_active_ledger()
        before = ledger.snapshot() if ledger is not None else None

        exam_result = await shared.evaluate_trial(config)

        if ledger is not None and before is not None:
            delta = ledger.delta_since(before)
            try:
                with (method_dir / "trial_cost_ledger.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"trial_number": trial_num, "buckets": delta}) + "\n")
            except OSError:
                logger.warning("Failed to append trial_cost_ledger.jsonl", exc_info=True)
            shared._flush_pending_cache_events(trial_num)
            prompt_tokens = sum(int(b["prompt_tokens"]) for b in delta.values())
            completion_tokens = sum(int(b["completion_tokens"]) for b in delta.values())
            embedding_tokens = sum(int(b["embedding_input_tokens"]) for b in delta.values())
        else:
            prompt_tokens = int(getattr(exam_result, "total_prompt_tokens", 0))
            completion_tokens = int(getattr(exam_result, "total_completion_tokens", 0))
            embedding_tokens = 0

        return TrialResult(
            score=float(exam_result.score),
            metrics={
                "answer_accuracy": float(exam_result.answer_accuracy),
                "mean_retrieval_quality": float(exam_result.mean_retrieval_quality),
                "mean_em": float(exam_result.mean_em),
                "mean_f1": float(exam_result.mean_f1),
            },
            eval_usd=float(exam_result.total_llm_cost_usd),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding_tokens=embedding_tokens,
        )

    return evaluator


async def _run_optimizer_with_ledger(
    optimizer,
    *,
    method_name: str,
    shared: Orchestrator,
    method_dir: Path,
    budget: Budget,
    seed: int | None,
) -> SearchResult:
    """Run ``optimizer.search`` with a per-(method, seed) cost-ledger context.

    For methods that drive the shared Orchestrator (``random``, ``bayesian``,
    plus ``autorag_*`` which uses the shared evaluator for the rescore step):
    install a fresh ``CostLedger`` so per-trial token deltas can be captured,
    reset the shared orchestrator's cache-credit bookkeeping
    (``_seen_emb_fps``, ``_pending_cache_events``) so first-use credits apply
    per (method, seed), and redirect ``shared.output_dir`` so the per-trial
    accounting files land in this method dir.

    ``agentic_*`` instantiates its own ``Orchestrator`` and manages the
    ledger lifecycle internally — bypass the wrapper there.
    """
    if method_name.startswith("agentic_"):
        async def _stub_evaluator(_config: TrialConfig) -> TrialResult:  # pragma: no cover
            raise RuntimeError(f"{method_name} should not call the bench evaluator")
        return await optimizer.search(_stub_evaluator, budget, seed=seed)

    original_output_dir = shared.output_dir
    shared._seen_emb_fps.clear()
    shared._pending_cache_events.clear()
    shared.output_dir = method_dir
    ledger = CostLedger()
    token = set_active_ledger(ledger)
    try:
        evaluator = _make_metered_evaluator(shared, method_dir)
        sr = await optimizer.search(evaluator, budget, seed=seed)
    finally:
        try:
            # ``bench_ledger.json`` (not ``cost_breakdown.json``) so a future
            # ``glob('**/cost_breakdown.json')`` does not conflate this
            # bench-side ledger (rescore-only) with autorag's own per-run
            # ``autorag_project/cost_breakdown.json`` (enumeration + qa-gen).
            (method_dir / "bench_ledger.json").write_text(
                json.dumps(ledger.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("Failed to write bench_ledger.json", exc_info=True)
        reset_active_ledger(token)
        shared.output_dir = original_output_dir
    return sr


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

    # Fail-fast if an AutoRAG method survived the -m filter but the venv set
    # up by scripts/setup_autorag_venv.sh isn't resolvable. Without this, a
    # 5-method run would spend hours on agentic/random/bayesian before the
    # AutoRAG rows discover the missing interpreter at search() time.
    if any(m.startswith("autorag") for m in bench.methods):
        resolve_autorag_python()

    if clean:
        removed = _clear_output_root_for(bench.output_root, bench.methods)
        if removed:
            logger.info(
                "Cleared %s under %s before run; other entries (including "
                "method dirs not in this run) preserved. Pass --no-clean "
                "to keep them.",
                sorted(removed), bench.output_root,
            )

    benchmark = BenchmarkRunner(
        name=bench.benchmark.name,
        output_dir=bench.benchmark.output_dir,
        split=bench.benchmark.split,
        sample_size=bench.benchmark.sample_size,
        seed=bench.benchmark.prep_seed,
    )
    benchmark.prepare()

    # Persist benchmark identity at output_root so analyze/plots can surface
    # the right name in tables and titles without needing to re-load the
    # YAML config. Survives ``_clear_output_root_for`` (file, not in the
    # targets list); rewritten on every run so it tracks the current spec.
    _write_bench_metadata(bench.output_root, bench)

    # Shared orchestrator: provides the evaluator that every sequential method
    # calls per trial. setup() is idempotent — the parsed corpus, exam.json,
    # and ingredient cache live under the project YAML's meta.output_dir
    # (./results/.shared_cache by default) so they're reused across methods.
    logger.info("Setting up shared orchestrator (will generate exam.json on first run)")
    shared = Orchestrator(str(bench.project_config_path))
    shared.evaluator.quiet_per_question = True
    try:
        await shared.setup()

        budget = Budget(max_trials=bench.max_trials)

        for method_name in bench.methods:
            seeds_for_method = bench.seeds if method_name in STOCHASTIC_METHODS else [None]
            for seed in seeds_for_method:
                seed_label = f"seed_{seed}" if seed is not None else "default"
                method_dir = bench.output_root / method_name / seed_label
                method_dir.mkdir(parents=True, exist_ok=True)
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
                    sr = await _run_optimizer_with_ledger(
                        optimizer,
                        method_name=method_name,
                        shared=shared,
                        method_dir=method_dir,
                        budget=budget,
                        seed=seed,
                    )
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
