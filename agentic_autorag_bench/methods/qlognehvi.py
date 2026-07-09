"""GP-BO degradation-reference baseline: Ax/BoTorch qLogNEHVI (Barker et al., 2025).

Barker et al. ("Faster, Cheaper, Better: MO HPO for LLM and RAG Systems", arXiv:2502.18635)
tune RAG with Ax's multi-objective GP-BO, whose default acquisition for two objectives is
**qLogNEHVI** (q-Log Noisy Expected Hypervolume Improvement). qLogNEHVI is *inherently
multi-objective* (it maximizes expected hypervolume improvement), so this baseline lives in the
cost-aware Pareto experiment — maximize ``answer_accuracy`` ↑, minimize ``mean_llm_cost_per_query_usd`` ↓
— NOT the single-objective accuracy experiment (a single-objective GP-BO would use qLogNEI, a
different acquisition; out of scope here).

Why it is a *degradation reference*, not a method-we-beat (STRONG_BASELINE_PLAN.md §6): Ax one-hot
encodes every unordered categorical, so our 13 embedders + ~39 generators + 5 rerankers +
4 query-expansion strategies become 60+ mostly-binary dimensions, and Ax has no native define-by-run
conditional structure (``compressor_llm`` only matters under an active compressor, etc.). GP-BO is
documented to degrade past ~20 one-hot dims, so on this space it is expected to perform at/below random
— a legitimate finding that engages Barker et al. (who validated only a tiny 4-LLM space) and explains
why TPE-family / reasoning methods are needed on realistic RAG spaces.

Design (so the ax-free core is unit-testable and only the thin service-loop glue needs the dep):
- ``ax_parameters(search_space)`` builds the *flattened* Ax parameter list (all dims unconditional —
  Ax can't gate them). Pure; no ax import.
- ``decode_params(params, search_space)`` maps one flat Ax parametrization back to a valid
  ``TrialConfig``, applying the SAME conditional gating + grid-snapping the random/Optuna samplers use
  (overlap < chunk_size, reranker_top_n <= top_k, stage-LLMs only when their stage is active, hybrid_alpha
  only under hybrid+alpha). Pure; no ax import.
- ``QLogNEHVISearch.search`` runs Ax's ``AxClient`` ask/tell loop, decodes each suggestion, resamples
  infeasible ones via ``log_trial_failure``, and reports accuracy + per-query cost. ``import ax`` is
  deferred into ``search`` so the bench imports cleanly without the optional dependency.

STATUS: the encode/decode core is unit-tested (``tests/test_qlognehvi.py``); the Ax service loop is
written against Ax's Service API (``AxClient`` + ``ObjectiveProperties``; MO ⇒ qLogNEHVI by default) but
is UNVERIFIED at runtime pending ``uv add ax-platform`` — Ax pulls botorch/gpytorch and pins torch, so
adding it deliberately (and re-checking the sentence-transformers torch version) is a maintainer call.
Not in any default config's ``methods`` list; wire it into the Pareto experiment when the dep is added.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_autorag.config.models import (
    GRAPH_INDEX_TYPES,
    DiscreteValues,
    IndexType,
    ProjectConfig,
    SearchSpace,
    TrialConfig,
    _dim_midpoint,
    _dim_min_value,
)

from agentic_autorag_bench.methods._logging import log_trial_banner
from agentic_autorag_bench.methods._sampler import _snap_to_nearest
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")

MAX_RESAMPLE_ATTEMPTS = 1000

_AX_IMPORT_HINT = (
    "qlognehvi requires the optional Ax dependency. Install it with "
    "`uv add ax-platform` (pulls botorch + gpytorch; verify it does not downgrade "
    "the torch your sentence-transformers embedders use), then re-run."
)


def _is_discrete(dim) -> bool:
    return isinstance(dim, DiscreteValues)


def ax_parameters(search_space: SearchSpace) -> list[dict]:
    """Flattened Ax parameter list for the search space.

    Every dimension is emitted unconditionally (Ax has no define-by-run gating);
    conditional validity is enforced later in :func:`decode_params`. Unordered
    categoricals become ``choice`` parameters (Ax one-hot encodes them — the
    source of the high-dimensional degradation); numeric ``DiscreteValues`` become
    ordered ``choice``; numeric ranges become ``range``. ``chunk_token_overlap``
    and ``reranker_top_n`` are emitted over their FULL declared range and gated to
    ``< chunk_size`` / ``<= top_k`` at decode time. Stage-LLM params are emitted
    only when their pool is non-empty; ``reasoning`` only when some generator
    supports it.
    """
    ss = search_space
    params: list[dict] = []

    def _choice(name: str, values: list, *, ordered: bool) -> dict:
        return {"name": name, "type": "choice", "values": list(values), "is_ordered": ordered, "sort_values": False}

    def _numeric(name: str, dim, *, is_float: bool) -> dict:
        if _is_discrete(dim):
            vals = [float(v) for v in dim.values] if is_float else [int(v) for v in dim.values]
            return _choice(name, vals, ordered=True)
        lo, hi = (float(dim.min), float(dim.max)) if is_float else (int(dim.min), int(dim.max))
        return {"name": name, "type": "range", "bounds": [lo, hi], "value_type": "float" if is_float else "int"}

    params.append(_choice("chunking_strategy", ss.chunking.strategies, ordered=False))
    params.append(_choice("embedding_model", list(ss.embedding.models), ordered=False))
    params.append(_numeric("chunk_token_size", ss.chunking.chunk_token_size, is_float=False))
    params.append(_numeric("chunk_token_overlap", ss.chunking.chunk_token_overlap, is_float=False))
    params.append(_choice("index_type", [it.value for it in ss.retrieval.index_types], ordered=False))
    params.append(_numeric("top_k", ss.retrieval.top_k, is_float=False))
    params.append(_choice("bm25_vector_fusion", ss.retrieval.bm25_vector_fusion, ordered=False))
    params.append(_numeric("hybrid_alpha", ss.retrieval.hybrid_alpha, is_float=True))
    params.append(_choice("long_context_reorder", list(ss.retrieval.long_context_reorder), ordered=False))
    params.append(_choice("reranker", ss.reranker.models, ordered=False))
    params.append(_numeric("reranker_top_n", ss.reranker.top_n, is_float=False))
    params.append(_choice("query_expansion", ss.query_expansion.strategies, ordered=False))
    params.append(_choice("passage_compressor", ss.passage_compressor.strategies, ordered=False))
    params.append(_choice("generator_llm", ss.generator.models, ordered=False))
    params.append(_numeric("temperature", ss.temperature, is_float=True))
    if ss.passage_compressor.models:
        params.append(_choice("compressor_llm", ss.passage_compressor.models, ordered=False))
    if ss.query_expansion.models:
        params.append(_choice("expander_llm", ss.query_expansion.models, ordered=False))
    if any(ss.is_reasoning_allowed(m) for m in ss.generator.models):
        params.append(_choice("reasoning", [False, True], ordered=False))
    if any(it in GRAPH_INDEX_TYPES for it in ss.retrieval.index_types) and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        params.append(_choice("graph_query_mode", gr.graph_query_modes, ordered=False))
        params.append(_numeric("graph_top_k", gr.graph_top_k, is_float=False))
    return params


def _snap_numeric(value, dim, *, is_float: bool):
    """Snap an Ax-suggested value onto a discrete grid (no-op for ranges)."""
    if _is_discrete(dim):
        grid = [float(v) for v in dim.values] if is_float else [int(v) for v in dim.values]
        snapped = _snap_to_nearest(float(value), grid)
        return float(snapped) if is_float else int(snapped)
    return float(value) if is_float else int(round(float(value)))


def decode_params(params: dict, search_space: SearchSpace) -> TrialConfig:
    """Map one flat Ax parametrization to a valid ``TrialConfig``.

    Applies the same conditional gating + grid-snapping as the random/Optuna
    samplers, so an Ax suggestion over the flattened (unconditional) space lands
    on a feasible config (or one ``validate_trial`` can still reject for a genuine
    cross-field violation, which the search loop resamples).
    """
    ss = search_space

    chunking_strategy = params["chunking_strategy"]
    embedding_model = params["embedding_model"]
    chunk_token_size = _snap_numeric(params["chunk_token_size"], ss.chunking.chunk_token_size, is_float=False)

    # overlap gated < chunk_size, then snapped to the legal grid.
    overlap_dim = ss.chunking.chunk_token_overlap
    raw_overlap = _snap_numeric(params["chunk_token_overlap"], overlap_dim, is_float=False)
    if _is_discrete(overlap_dim):
        legal = [int(v) for v in overlap_dim.values if v < chunk_token_size]
        chunk_token_overlap = int(_snap_to_nearest(raw_overlap, legal)) if legal else int(_dim_min_value(overlap_dim))
    else:
        chunk_token_overlap = max(int(overlap_dim.min), min(raw_overlap, chunk_token_size - 1))

    index_type = IndexType(params["index_type"])
    top_k = _snap_numeric(params["top_k"], ss.retrieval.top_k, is_float=False)

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        bm25_vector_fusion = params["bm25_vector_fusion"]
        if bm25_vector_fusion == "alpha":
            hybrid_alpha = round(float(params["hybrid_alpha"]), 4)
        else:
            hybrid_alpha = round(_dim_midpoint(ss.retrieval.hybrid_alpha), 4)
    else:
        bm25_vector_fusion = ss.retrieval.bm25_vector_fusion[0]
        hybrid_alpha = round(_dim_midpoint(ss.retrieval.hybrid_alpha), 4)

    reranker = params["reranker"]
    if reranker != "none":
        rn_dim = ss.reranker.top_n
        raw_top_n = _snap_numeric(params["reranker_top_n"], rn_dim, is_float=False)
        if _is_discrete(rn_dim):
            legal = [int(v) for v in rn_dim.values if v <= top_k]
            reranker_top_n = int(_snap_to_nearest(raw_top_n, legal)) if legal else int(_dim_min_value(rn_dim))
        else:
            reranker_top_n = max(int(rn_dim.min), min(raw_top_n, top_k))
    else:
        reranker_top_n = int(_dim_min_value(ss.reranker.top_n))

    query_expansion = params["query_expansion"]
    long_context_reorder = bool(params["long_context_reorder"])
    passage_compressor = params["passage_compressor"]
    generator_llm = params["generator_llm"]
    compressor_llm = params.get("compressor_llm") if passage_compressor != "none" else None
    expander_llm = params.get("expander_llm") if query_expansion != "none" else None
    temperature = round(float(params["temperature"]), 4)
    reasoning = bool(params.get("reasoning", False)) if ss.is_reasoning_allowed(generator_llm) else False

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        graph_query_mode = params.get("graph_query_mode", "hybrid")
        graph_top_k = _snap_numeric(params.get("graph_top_k", 60), ss.graph_retrieval.graph_top_k, is_float=False)
    else:
        graph_query_mode = "hybrid"
        graph_top_k = 60

    return TrialConfig(
        chunking_strategy=chunking_strategy,
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
        embedding_model=embedding_model,
        index_type=index_type,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        bm25_vector_fusion=bm25_vector_fusion,
        long_context_reorder=long_context_reorder,
        passage_compressor=passage_compressor,
        reranker=reranker,
        reranker_top_n=reranker_top_n,
        query_expansion=query_expansion,
        generator_llm=generator_llm,
        compressor_llm=compressor_llm,
        expander_llm=expander_llm,
        temperature=temperature,
        reasoning=reasoning,
        graph_query_mode=graph_query_mode,
        graph_top_k=graph_top_k,
    )


@dataclass
class QLogNEHVISearch:
    """Ax/BoTorch multi-objective GP-BO (qLogNEHVI) over the flattened search space.

    Multi-objective only (accuracy ↑, per-query cost ↓) — requires ``meta.cost_aware``.
    The bench computes the comparable hypervolume separately (framework helper, shared
    reference point), so this method only needs to produce per-trial accuracy + cost.
    """

    project: ProjectConfig
    storage_dir: Path
    resume: bool = False
    name: str = "qlognehvi"
    deterministic: bool = False

    async def search(self, evaluator: Evaluator, budget: Budget, *, seed: int | None = None) -> SearchResult:
        if budget.max_trials is None:
            raise ValueError("qlognehvi search requires budget.max_trials")
        if not bool(self.project.meta.cost_aware):
            raise NotImplementedError(
                "qlognehvi is multi-objective (accuracy + cost) and only runs in the cost-aware "
                "Pareto experiment. For a single-objective accuracy GP-BO use a qLogNEI variant (out of scope)."
            )
        try:
            from ax.service.ax_client import AxClient, ObjectiveProperties
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(_AX_IMPORT_HINT) from exc

        ax_client = AxClient(random_seed=seed)
        ax_client.create_experiment(
            name="qlognehvi_rag",
            parameters=ax_parameters(self.project.search_space),
            # Two objectives ⇒ Ax's ModularBoTorchGenerator defaults to qLogNEHVI.
            objectives={
                "answer_accuracy": ObjectiveProperties(minimize=False),
                "mean_llm_cost_per_query_usd": ObjectiveProperties(minimize=True),
            },
        )

        history: list[HistoryEntry] = []
        n_validation_rejects = 0
        t_start = time.monotonic()
        for trial_num in range(1, budget.max_trials + 1):
            config = None
            trial_index = None
            for _ in range(MAX_RESAMPLE_ATTEMPTS):
                raw_params, trial_index = ax_client.get_next_trial()
                candidate = decode_params(raw_params, self.project.search_space)
                violations = self.project.validate_trial(candidate)
                if not violations:
                    config = candidate
                    break
                logger.debug("trial %d ax suggestion rejected: %s", trial_num, "; ".join(violations))
                ax_client.log_trial_failure(trial_index)
                n_validation_rejects += 1

            if config is None or trial_index is None:
                raise RuntimeError(
                    f"qlognehvi could not find a valid config after {MAX_RESAMPLE_ATTEMPTS} Ax suggestions "
                    f"on trial {trial_num}; the feasible volume is near zero."
                )

            log_trial_banner(logger, trial_num, budget.max_trials, config)
            try:
                result = await evaluator(config)
            except Exception:
                logger.exception("trial %d evaluation failed; marking failed", trial_num)
                ax_client.log_trial_failure(trial_index)
                continue

            ax_client.complete_trial(
                trial_index,
                raw_data={
                    "answer_accuracy": float(result.answer_accuracy),
                    "mean_llm_cost_per_query_usd": float(result.mean_llm_cost_per_query_usd),
                },
            )
            history.append(
                HistoryEntry(
                    trial_number=trial_num,
                    config=config.to_prompt_dump(include_graph=self.project.uses_graph()),
                    answer_accuracy=result.answer_accuracy,
                    metrics=result.metrics,
                    eval_usd=result.eval_usd,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    embedding_tokens=result.embedding_tokens,
                    mean_llm_cost_per_query_usd=result.mean_llm_cost_per_query_usd,
                )
            )

        if not history:
            raise RuntimeError("qlognehvi search produced no successful trials")

        best_entry = max(history, key=lambda h: h.answer_accuracy)
        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=0.0,
            trial_usd_total=sum(h.eval_usd for h in history),
            wall_clock_s=time.monotonic() - t_start,
            prompt_tokens=sum(h.prompt_tokens for h in history),
            completion_tokens=sum(h.completion_tokens for h in history),
            embedding_tokens=sum(h.embedding_tokens for h in history),
            extras={"n_validation_rejects": n_validation_rejects, "acquisition": "qLogNEHVI"},
        )
