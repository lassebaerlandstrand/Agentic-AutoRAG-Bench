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
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from agentic_autorag.config.models import TrialConfig
from agentic_autorag.cost_ledger import CostLedger, get_active_ledger, reset_active_ledger, set_active_ledger
from agentic_autorag.litellm_runtime import configure_litellm_runtime
from agentic_autorag.orchestrator import Orchestrator

from agentic_autorag_bench._holdout_registry import apply_union_exclusion
from agentic_autorag_bench.benchmarks.runner import BenchmarkRunner
from agentic_autorag_bench.methods.agentic import AgenticOptimizer
from agentic_autorag_bench.methods.bayesian import BayesianSearch
from agentic_autorag_bench.methods.random import RandomSearch
from agentic_autorag_bench.plots import (
    make_matrix_figures,
    make_method_figures,
    make_seed_figures,
)
from agentic_autorag_bench.types import Budget, HistoryEntry, SearchResult, TrialResult

logger = logging.getLogger("agentic_autorag_bench.run")

STOCHASTIC_METHODS = {"random", "bayesian", "agentic_score", "agentic_cost"}
# Kept (empty) so deterministic methods can be reintroduced without rewiring
# the dispatch loop in ``run_matrix``. AutoRAG variants used to live here
# before being moved to agentic_autorag_bench/_deprecated/autorag/.
DETERMINISTIC_METHODS: set[str] = set()
ALL_METHODS = STOCHASTIC_METHODS | DETERMINISTIC_METHODS
# Methods that share the bench's ``shared`` Orchestrator via ``evaluate_trial``.
# These need the bench to install a per-(method, seed) cost ledger and reset
# ``shared._seen_emb_fps`` between runs; agentic instantiates its own
# Orchestrator so it manages its own ledger lifecycle.
_SHARED_EVALUATOR_METHODS = {"random", "bayesian"}


def _clear_output_root_for(output_root: Path, methods: list[str]) -> list[str]:
    """Wipe per-run artifacts for the methods about to be re-run.

    Scoped on purpose: partial runs (e.g. ``-m agentic_score``) should
    compose with previous results for the other methods, so only the
    method dirs we're about to overwrite get reset. Checkpoint variants
    (``<method>@<k>/``) are wiped alongside their parent method, since
    they're produced from the parent's history and must stay in sync.

    ``figures/`` is deliberately NOT wiped here: matrix figures are
    rendered to a staging dir and atomically swapped at end-of-run by
    ``_swap_in_staged_figures``. That keeps the previous run's
    cross-method figures readable for the entire duration of a new run.

    ``.shared_cache/``, ``bench_metadata.json``, and any user files
    (notes, scratch dirs) are untouched: they're not in the wipe set.

    Returns the names that were removed, for logging.
    """
    if not output_root.exists():
        return []
    method_prefixes = tuple(f"{m}@" for m in methods)
    checkpoint_dirs = [
        child.name
        for child in output_root.iterdir()
        if child.is_dir() and child.name.startswith(method_prefixes)
    ]
    targets = [*methods, *checkpoint_dirs]
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


_FIGURES_STAGING_NAME = "_figures_staging"


def _write_bench_metadata(output_root: Path, bench: BenchConfig) -> None:
    """Persist the benchmark + run identity at ``output_root/bench_metadata.json``.

    Downstream readers (``analyze.py``, ``plots.py``) consult this file to
    surface the right benchmark name in tables and figure titles, so they
    don't have to re-parse the source YAML config. Rewritten on every run
    so it tracks the current spec.
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
        "checkpoints": bench.checkpoints,
    }
    (output_root / "bench_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _swap_in_staged_figures(output_root: Path) -> None:
    """Atomically replace ``output_root/figures/`` with the freshly rendered
    contents of ``output_root/_figures_staging/``.

    The staging dance keeps the previous run's matrix figures visible at
    their normal path for the entire duration of the new run — only the
    final swap touches ``figures/``. If matrix rendering fails mid-run,
    the staging dir is left in place for inspection and the old
    ``figures/`` is undisturbed.

    POSIX guarantees atomicity on each rename; the brief window between
    the two renames is the smallest achievable without backup-copy
    overhead. The intermediate ``_figures_previous/`` directory is removed
    once the new figures are in place.
    """
    staging = output_root / _FIGURES_STAGING_NAME
    if not staging.is_dir():
        return
    final = output_root / "figures"
    backup = output_root / "_figures_previous"
    if final.exists():
        if backup.exists():
            shutil.rmtree(backup)
        final.rename(backup)
    staging.rename(final)
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


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
    # Per-method early-stopping checkpoints. After a method's full
    # max_trials-budget search finishes, the orchestrator additionally
    # evaluates the best config seen in ``history[:k]`` on the held-out QA
    # for each declared k < max_trials, writing ``<method>@<k>/seed_<n>/``
    # alongside the bare ``<method>/seed_<n>/`` directory. Lets the paper
    # show "ours at trial k vs. baseline at trial max_trials" without
    # running extra search trials. Unset methods get [] (no extra
    # checkpoints — only the natural full-budget directory).
    checkpoints: dict[str, list[int]] = field(default_factory=dict)

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
        max_trials = int(raw["budget"]["max_trials"])
        raw_checkpoints = raw.get("checkpoints") or {}
        unknown_methods = set(raw_checkpoints) - ALL_METHODS
        if unknown_methods:
            raise ValueError(
                f"Unknown methods in checkpoints block of {config_path}: "
                f"{sorted(unknown_methods)}"
            )
        checkpoints: dict[str, list[int]] = {}
        for method, ks in raw_checkpoints.items():
            cleaned = sorted({int(k) for k in ks if 0 < int(k) < max_trials})
            if cleaned:
                checkpoints[method] = cleaned
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
            max_trials=max_trials,
            benchmark=benchmark,
            hold_out_limit=raw["hold_out"].get("limit"),
            hold_out_judge_model=raw["hold_out"].get("judge_model"),
            hold_out_concurrency=int(raw["hold_out"].get("concurrency", 10)),
            output_root=Path(raw["output_root"]).resolve(),
            checkpoints=checkpoints,
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


def _read_trial_cost_ledger(method_dir: Path) -> list[dict]:
    """Parse ``method_dir/trial_cost_ledger.jsonl`` into per-trial bucket dicts.

    The ledger is written for every method (framework's
    ``_finalize_trial_accounting`` for agentic_*, bench's
    ``_make_metered_evaluator`` for random/bayesian). Each line has
    ``{"trial_number": int, "buckets": {bucket_name: {usd, prompt_tokens,
    completion_tokens, embedding_input_tokens, n_calls, ...}}}`` and may
    include a ``"status"`` field (agentic writes ``"failed"`` for trials
    whose evaluation errored — the spend is real and stays in the sum).
    """
    path = method_dir / "trial_cost_ledger.jsonl"
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _checkpoint_costs(ledger: list[dict], k: int) -> tuple[float, float]:
    """Return ``(optimizer_usd, trial_usd_total)`` for an @k early-stopping run.

    Reads per-trial cost deltas captured during the parent's full-budget run
    and reconstructs what an actual early-stop-at-k run would have paid:

    - ``optimizer_usd``: ``agent_proposal`` summed over trials ``1..(k-1)``.
      Trial 1's bucket bundles ``propose_initial`` plus the
      ``analyze_and_propose`` call that produced trial 2's config; each
      subsequent trial K's bucket holds only the ``analyze_and_propose``
      call that produced trial K+1's config (see
      ``Orchestrator._run_with_ledger``). Summing over ``1..(k-1)`` yields
      exactly ``propose_initial + analyze_and_propose × (k-1)``, which is
      what an actual @k run pays — it stops the loop before the
      ``analyze_and_propose`` at the end of trial k fires.
    - ``trial_usd_total``: ``rag_eval + judge`` summed over trials ``1..k``.
      Includes the per-trial judge spend that
      ``ExamResult.total_llm_cost_usd`` (RAG-only) systematically omits.

    Returns ``(0.0, 0.0)`` for an empty ledger — caller falls back to the
    prorated/history-sum path so checkpointing still works on legacy result
    trees written before the ledger was added.
    """
    if not ledger:
        return 0.0, 0.0
    proposer_usd = sum(
        float(e.get("buckets", {}).get("agent_proposal", {}).get("usd", 0.0))
        for e in ledger
        if int(e.get("trial_number", 0)) < k
    )
    trial_usd = sum(
        float(e.get("buckets", {}).get("rag_eval", {}).get("usd", 0.0))
        + float(e.get("buckets", {}).get("judge", {}).get("usd", 0.0))
        for e in ledger
        if int(e.get("trial_number", 0)) <= k
    )
    return proposer_usd, trial_usd


async def _evaluate_checkpoints(
    sr: SearchResult,
    *,
    method_name: str,
    seed: int | None,
    bench: BenchConfig,
    benchmark: BenchmarkRunner,
) -> None:
    """For each declared checkpoint k < max_trials, write a sibling @k result
    directory using the best config from ``history[:k]``.

    Lets the paper compare ``method@10`` vs. ``method@20`` vs. ``method@40``
    on the same held-out QA without paying for additional search trials —
    the head of the existing trajectory is exactly what an early-stopped
    run would have produced. The reduced ``SearchResult`` written into each
    ``<method>@<k>/seed_<n>/`` reports cumulative-cost-at-k using the
    parent's ``trial_cost_ledger.jsonl`` (see ``_checkpoint_costs`` for the
    bucket-attribution rule). ``wall_clock_s`` stays prorated by trial count
    because per-trial wall time isn't recorded in the bench's per-trial JSONL.
    A fresh held-out eval runs per checkpoint — the framework's judge caches
    per ``(config_hash, question_id)`` so identical configs across
    checkpoints pay nothing extra.
    """
    ckpts = bench.checkpoints.get(method_name, [])
    if not ckpts:
        return
    seed_label = f"seed_{seed}" if seed is not None else "default"
    parent_dir = bench.output_root / method_name / seed_label
    ledger = _read_trial_cost_ledger(parent_dir)
    total = len(sr.history)
    for k in ckpts:
        if k >= total:
            continue
        sliced: list[HistoryEntry] = sr.history[:k]
        if not sliced:
            continue
        best = max(sliced, key=lambda h: h.score)

        ck_method = f"{method_name}@{k}"
        ck_dir = bench.output_root / ck_method / seed_label

        proposer_usd, trial_usd = _checkpoint_costs(ledger, k)
        # Fallback when the ledger is missing (legacy result trees): prorate
        # optimizer_usd, sum history's RAG-only eval_usd. Same behaviour the
        # old prorated path had — losing only the judge contribution.
        if not ledger:
            proposer_usd = sr.optimizer_usd * (k / total)
            trial_usd = sum(h.eval_usd for h in sliced)
        fraction = k / total
        sr_at_k = SearchResult(
            method=ck_method,
            seed=sr.seed,
            deterministic=sr.deterministic,
            best_config=best.config,
            history=sliced,
            optimizer_usd=proposer_usd,
            trial_usd_total=trial_usd,
            wall_clock_s=sr.wall_clock_s * fraction,
            prompt_tokens=sum(h.prompt_tokens for h in sliced),
            completion_tokens=sum(h.completion_tokens for h in sliced),
            embedding_tokens=sum(h.embedding_tokens for h in sliced),
            extras={
                **dict(sr.extras),
                "checkpoint_at": k,
                "parent_method": method_name,
                "parent_max_trials": total,
            },
        )
        _persist_search_result(sr_at_k, ck_dir)

        trial_config = TrialConfig(**best.config)
        await benchmark.evaluate(
            project_config_path=str(bench.project_config_path),
            trial_config=trial_config,
            output_path=ck_dir / "benchmark_results.json",
            judge_model=bench.hold_out_judge_model,
            limit=bench.hold_out_limit,
            concurrency=bench.hold_out_concurrency,
        )
        make_seed_figures(ck_dir)
        logger.info(
            "  checkpoint %s seed=%s @%d done | best_score=%.3f | trial_usd=$%.4f",
            method_name, seed, k, best.score, sr_at_k.trial_usd_total,
        )


def _build_optimizer(
    name: str,
    *,
    project,
    bench: BenchConfig,
    output_dir: Path,
    resume: bool = False,
):
    if name == "random":
        return RandomSearch(project=project, storage_dir=output_dir, resume=resume)
    if name == "bayesian":
        return BayesianSearch(project=project, storage_dir=output_dir, resume=resume)
    if name in {"agentic_score", "agentic_cost"}:
        return AgenticOptimizer(
            config_path=str(bench.project_config_path),
            output_dir=str(output_dir),
            cost_aware=(name == "agentic_cost"),
            resume=resume,
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

    ``eval_usd`` sums every USD bucket in the ledger delta (RAG generation,
    judge, query expansion, any other trial-phase LLM call) rather than
    using ``ExamResult.total_llm_cost_usd``, which is documented RAG-only
    and would silently omit the judge from random/bayesian's reported
    ``trial_usd_total``. The agentic adapter folds judge in by reading
    ``cost_breakdown.json``; this keeps the bench-side methods symmetric.

    ``_current_phase`` is promoted to ``"trial"`` so any
    ``_credit_embedding_build`` event fired by ``evaluate_trial`` is tagged
    with the trial number in ``cache_events.jsonl`` — the shared
    Orchestrator stays at ``"setup"`` otherwise because random/bayesian
    never call the framework's per-trial phase-setting path.
    """
    trial_counter = [0]

    async def evaluator(config: TrialConfig) -> TrialResult:
        trial_counter[0] += 1
        trial_num = trial_counter[0]
        ledger = get_active_ledger()
        before = ledger.snapshot() if ledger is not None else None

        shared._current_phase = "trial"
        exam_result = await shared.evaluate_trial(config)

        if ledger is not None and before is not None:
            delta = ledger.delta_since(before)
            try:
                with (method_dir / "trial_cost_ledger.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"trial_number": trial_num, "buckets": delta}) + "\n")
            except OSError:
                logger.warning("Failed to append trial_cost_ledger.jsonl", exc_info=True)
            shared._flush_pending_cache_events(trial_num)
            eval_usd = sum(float(b["usd"]) for b in delta.values())
            prompt_tokens = sum(int(b["prompt_tokens"]) for b in delta.values())
            completion_tokens = sum(int(b["completion_tokens"]) for b in delta.values())
            embedding_tokens = sum(int(b["embedding_input_tokens"]) for b in delta.values())
        else:
            eval_usd = float(exam_result.total_llm_cost_usd)
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
            eval_usd=eval_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding_tokens=embedding_tokens,
        )

    return evaluator


def _seed_seen_emb_fps_from_history(shared: Orchestrator, method_dir: Path) -> None:
    """Mark every embedding fingerprint already paid for by this (method,
    seed)'s prior trials as seen on the shared orchestrator.

    Used on ``--resume`` for shared-evaluator methods (random / bayesian).
    Without this, the first post-resume encounter of an embedder that the
    prior run already credited to ``embedding_build`` would charge it a
    second time, breaking the bench's "first use per (method, seed) only"
    accounting rule. We walk this method's own ``history.jsonl`` (the
    canonical record of completed trials' configs) and re-derive each
    config's ``emb_fp`` via the exact code path the live evaluator uses,
    so the seeded set matches whatever the un-interrupted run would have
    accumulated up to the same trial.
    """
    history_path = method_dir / "history.jsonl"
    if not history_path.exists():
        return
    corpus_hash = shared._corpus_cache_key()
    n_seeded = 0
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            trial_config = TrialConfig(**data["config"])
            emb_fp = trial_config.to_structural().embeddings_fingerprint(corpus_hash)
        except Exception:
            logger.warning(
                "Could not re-derive emb_fp for a history line in %s; first "
                "post-resume encounter of that embedder may be charged again.",
                history_path, exc_info=True,
            )
            continue
        shared._seen_emb_fps.add(emb_fp)
        n_seeded += 1
    if n_seeded:
        logger.info(
            "Seeded %d embedding fingerprint(s) into shared._seen_emb_fps from %s",
            n_seeded, history_path,
        )


async def _run_optimizer_with_ledger(
    optimizer,
    *,
    method_name: str,
    shared: Orchestrator,
    method_dir: Path,
    budget: Budget,
    seed: int | None,
    resume: bool = False,
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

    On ``resume``: ``_seen_emb_fps`` is repopulated from the prior trials'
    configs (read straight from this method's ``history.jsonl``) so the
    first post-resume encounter of an embedder we already paid for does
    not double-charge the ``embedding_build`` bucket.
    """
    if method_name.startswith("agentic_"):
        async def _stub_evaluator(_config: TrialConfig) -> TrialResult:  # pragma: no cover
            raise RuntimeError(f"{method_name} should not call the bench evaluator")
        return await optimizer.search(_stub_evaluator, budget, seed=seed)

    original_output_dir = shared.output_dir
    shared._seen_emb_fps.clear()
    shared._pending_cache_events.clear()
    if resume:
        _seed_seen_emb_fps_from_history(shared, method_dir)
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


_RESUME_STATE_FILES = ("history.jsonl", "optuna.db", "rng_state.pkl")


def _has_prior_trial_state(method_dir: Path) -> bool:
    """A (method, seed) dir is resume-able if any of its per-trial state
    artifacts already exist. Empty dirs (e.g. created on this run by the
    main loop's ``method_dir.mkdir``) are treated as fresh starts so
    ``--resume`` is a no-op for methods that haven't started yet.
    """
    return any((method_dir / name).exists() for name in _RESUME_STATE_FILES)


async def run_matrix(
    config_path: str | Path,
    *,
    methods_override: list[str] | None = None,
    clean: bool = True,
    resume: bool = False,
) -> None:
    # litellm.drop_params=True so provider-specific params (seed, temperature
    # on gpt-5) are silently dropped instead of erroring. The framework's
    # ``agentic-autorag optimize`` CLI calls this; the bench has its own entry
    # point and must call it explicitly to inherit identical LLM semantics.
    configure_litellm_runtime()
    if clean and resume:
        # CLI parses this case too, but keep the guard here so library
        # callers can't accidentally request both.
        raise ValueError("clean=True and resume=True are mutually exclusive")
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
                "Cleared %s under %s before run; figures/, .shared_cache/, "
                "and method dirs not in this run are preserved. Pass "
                "--no-clean to resume a partial run within a method.",
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
    # the right name in tables and figure titles without re-loading the YAML.
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

                # ``method_dir.mkdir(exist_ok=True)`` above just created the
                # dir if it didn't exist; the partial-state check has to
                # look for actual artifact files, not just dir existence.
                resume_this_method = resume and _has_prior_trial_state(method_dir)
                if resume and not resume_this_method:
                    logger.info(
                        "--resume passed but %s has no prior trial state; "
                        "starting from trial 1", method_dir,
                    )
                optimizer = _build_optimizer(
                    method_name,
                    project=shared.config,
                    bench=bench,
                    output_dir=method_dir,
                    resume=resume_this_method,
                )
                try:
                    sr = await _run_optimizer_with_ledger(
                        optimizer,
                        method_name=method_name,
                        shared=shared,
                        method_dir=method_dir,
                        budget=budget,
                        seed=seed,
                        resume=resume_this_method,
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

                # Synthesize @k early-stopping checkpoints from the same
                # trajectory. Each writes its own sibling directory + held-out
                # results + figures, so the cross-method matrix figures will
                # surface them automatically.
                await _evaluate_checkpoints(
                    sr,
                    method_name=method_name,
                    seed=seed,
                    bench=bench,
                    benchmark=benchmark,
                )

            # Per-method figures: aggregate every seed for this method now that
            # the inner loop has closed. Cross-method matrix figures wait for
            # the outer loop. Checkpoint variants get the same treatment.
            make_method_figures(bench.output_root / method_name)
            for k in bench.checkpoints.get(method_name, []):
                if k >= bench.max_trials:
                    continue
                ck_dir = bench.output_root / f"{method_name}@{k}"
                if ck_dir.is_dir():
                    make_method_figures(ck_dir)
    finally:
        await shared.cleanup()

    # Cross-method content-filter exclusion: drop any question that any
    # method's best config got rejected on, so all rows score the same
    # denominator. Runs after every hold-out so the union is complete.
    apply_union_exclusion(bench.output_root)

    # Matrix figures see the union-exclusion-adjusted hold-out scores; calling
    # before apply_union_exclusion would bake stale per-question denominators
    # into the table. Rendered into a staging dir first; the previous run's
    # figures/ stays readable until the swap at the very end.
    staging_dir = bench.output_root / _FIGURES_STAGING_NAME
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    make_matrix_figures(bench.output_root, figures_dir=staging_dir)
    _swap_in_staged_figures(bench.output_root)


def run_cli(
    config_path: str,
    *,
    methods: list[str] | None = None,
    clean: bool = True,
    resume: bool = False,
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
        clean=clean,
        resume=resume,
    ))
