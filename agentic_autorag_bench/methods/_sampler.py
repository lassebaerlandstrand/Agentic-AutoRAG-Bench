"""Conditional samplers: random and Optuna define-by-run.

Both honour the same active-dimension gates the agent's proposer must respect
(graph fields only render for graph index types, ``hybrid_alpha`` only when
hybrid, ``reranker_top_n`` only when a reranker is selected). Optuna's TPE
surrogate therefore observes only relevant dimensions.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agentic_autorag.config.models import (
    GRAPH_INDEX_TYPES,
    IndexType,
    NumericRange,
    SearchSpace,
    TrialConfig,
)

if TYPE_CHECKING:
    import optuna


def _midpoint(r: NumericRange) -> float:
    return (r.min + r.max) / 2.0


def _filter_compatible_embeddings(
    embedding_models: list[str],
    chunk_token_size: int,
    embedding_token_limits: dict[str, int],
) -> list[str]:
    return [
        m for m in embedding_models
        if embedding_token_limits.get(m) is None or chunk_token_size <= embedding_token_limits[m]
    ]


def sample_random(
    rng: random.Random,
    search_space: SearchSpace,
    embedding_token_limits: dict[str, int] | None = None,
) -> TrialConfig:
    """Uniform random sample from the search space, gated by validators."""
    embedding_token_limits = embedding_token_limits or {}
    ss = search_space

    chunking_strategy = rng.choice(ss.chunking.strategies)
    cs_lo = int(ss.chunking.chunk_token_size.min)
    cs_hi = int(ss.chunking.chunk_token_size.max)
    chunk_token_size = rng.randint(cs_lo, cs_hi)

    co_lo = int(ss.chunking.chunk_token_overlap.min)
    co_hi = max(co_lo, min(int(ss.chunking.chunk_token_overlap.max), chunk_token_size - 1))
    chunk_token_overlap = rng.randint(co_lo, co_hi)

    compatible = _filter_compatible_embeddings(ss.embedding_models, chunk_token_size, embedding_token_limits)
    if not compatible:
        max_supported = max(embedding_token_limits.values()) if embedding_token_limits else cs_hi
        chunk_token_size = max(cs_lo, min(chunk_token_size, max_supported))
        chunk_token_overlap = min(chunk_token_overlap, max(co_lo, chunk_token_size - 1))
        compatible = _filter_compatible_embeddings(
            ss.embedding_models, chunk_token_size, embedding_token_limits
        ) or list(ss.embedding_models)
    embedding_model = rng.choice(compatible)

    index_type = rng.choice(list(ss.index_types))
    top_k = rng.randint(int(ss.top_k.min), int(ss.top_k.max))

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        hybrid_alpha = round(rng.uniform(ss.hybrid_alpha.min, ss.hybrid_alpha.max), 4)
    else:
        hybrid_alpha = round(_midpoint(ss.hybrid_alpha), 4)

    reranker = rng.choice(ss.reranker.models)
    if reranker != "none":
        rn_lo = int(ss.reranker.top_n.min)
        rn_hi = max(rn_lo, min(int(ss.reranker.top_n.max), top_k))
        reranker_top_n = rng.randint(rn_lo, rn_hi)
    else:
        reranker_top_n = int(ss.reranker.top_n.min)

    query_expansion = rng.choice(ss.query_expansion)
    llm_model = rng.choice(ss.llm_models)
    temperature = round(rng.uniform(ss.temperature.min, ss.temperature.max), 4)
    reasoning = rng.choice([False, True]) if ss.is_reasoning_allowed(llm_model) else False

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        graph_query_mode = rng.choice(gr.graph_query_modes)
        graph_top_k = rng.randint(int(gr.graph_top_k.min), int(gr.graph_top_k.max))
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
        reranker=reranker,
        reranker_top_n=reranker_top_n,
        query_expansion=query_expansion,
        llm_model=llm_model,
        temperature=temperature,
        reasoning=reasoning,
        graph_query_mode=graph_query_mode,
        graph_top_k=graph_top_k,
    )


def sample_optuna(
    trial: optuna.Trial,
    search_space: SearchSpace,
    embedding_token_limits: dict[str, int] | None = None,
) -> TrialConfig:
    """Define-by-run sample. Raises ``optuna.TrialPruned`` if no embedding fits."""
    import optuna

    embedding_token_limits = embedding_token_limits or {}
    ss = search_space

    chunking_strategy = trial.suggest_categorical("chunking_strategy", ss.chunking.strategies)
    cs_lo = int(ss.chunking.chunk_token_size.min)
    cs_hi = int(ss.chunking.chunk_token_size.max)
    chunk_token_size = trial.suggest_int("chunk_token_size", cs_lo, cs_hi)

    co_lo = int(ss.chunking.chunk_token_overlap.min)
    co_hi = max(co_lo, min(int(ss.chunking.chunk_token_overlap.max), chunk_token_size - 1))
    chunk_token_overlap = trial.suggest_int("chunk_token_overlap", co_lo, co_hi)

    compatible = _filter_compatible_embeddings(ss.embedding_models, chunk_token_size, embedding_token_limits)
    if not compatible:
        raise optuna.TrialPruned(f"No embedding model supports chunk_token_size={chunk_token_size}")
    embedding_model = trial.suggest_categorical("embedding_model", compatible)

    index_type = IndexType(trial.suggest_categorical("index_type", [it.value for it in ss.index_types]))
    top_k = trial.suggest_int("top_k", int(ss.top_k.min), int(ss.top_k.max))

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        hybrid_alpha = trial.suggest_float("hybrid_alpha", ss.hybrid_alpha.min, ss.hybrid_alpha.max)
    else:
        hybrid_alpha = _midpoint(ss.hybrid_alpha)

    reranker = trial.suggest_categorical("reranker", ss.reranker.models)
    if reranker != "none":
        rn_lo = int(ss.reranker.top_n.min)
        rn_hi = max(rn_lo, min(int(ss.reranker.top_n.max), top_k))
        reranker_top_n = trial.suggest_int("reranker_top_n", rn_lo, rn_hi)
    else:
        reranker_top_n = int(ss.reranker.top_n.min)

    query_expansion = trial.suggest_categorical("query_expansion", ss.query_expansion)
    llm_model = trial.suggest_categorical("llm_model", ss.llm_models)
    temperature = trial.suggest_float("temperature", ss.temperature.min, ss.temperature.max)
    reasoning = (
        trial.suggest_categorical("reasoning", [False, True])
        if ss.is_reasoning_allowed(llm_model)
        else False
    )

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        graph_query_mode = trial.suggest_categorical("graph_query_mode", gr.graph_query_modes)
        graph_top_k = trial.suggest_int("graph_top_k", int(gr.graph_top_k.min), int(gr.graph_top_k.max))
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
        reranker=reranker,
        reranker_top_n=reranker_top_n,
        query_expansion=query_expansion,
        llm_model=llm_model,
        temperature=temperature,
        reasoning=reasoning,
        graph_query_mode=graph_query_mode,
        graph_top_k=graph_top_k,
    )
