"""Conditional samplers: random and Optuna define-by-run.

Both honour the same active-dimension gates the agent's proposer must respect
(graph fields only render for graph index types, ``hybrid_alpha`` only when
hybrid, ``reranker_top_n`` only when a reranker is selected). Optuna's TPE
surrogate therefore observes only relevant dimensions.

Discrete vs continuous numeric dims: ``top_k``, ``hybrid_alpha``,
``reranker.top_n``, ``chunk_token_size`` and ``chunk_token_overlap`` may be
either a ``NumericRange`` (continuous; AutoRAG can't fairly sample inside
one) or a ``DiscreteValues`` set. The ``_sample_int`` / ``_sample_float`` /
``_suggest_int`` / ``_suggest_float`` helpers dispatch on type so the rest
of the sampler stays declarative.

Optuna and dependent DiscreteValues dims: when a dim's legal set depends on
another parameter just sampled this trial (``reranker.top_n <= top_k``,
``chunk_token_overlap < chunk_token_size``), we use ``suggest_int`` with
per-trial dynamic bounds and post-snap the result to the nearest legal grid
value, rather than ``suggest_categorical`` over the full set with a snap-back
on illegal picks. The dynamic-int pattern is Optuna's documented recipe (FAQ
"search spaces specified in each call") — int distributions can vary their
bounds per trial without corrupting TPE's parameter-space hash, unlike
categorical.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agentic_autorag.config.models import (
    GRAPH_INDEX_TYPES,
    DiscreteValues,
    IndexType,
    NumericDim,
    SearchSpace,
    TrialConfig,
    _dim_midpoint,
    _dim_min_value,
)

if TYPE_CHECKING:
    import optuna


def _sample_int(rng: random.Random, dim: NumericDim) -> int:
    if isinstance(dim, DiscreteValues):
        return int(rng.choice(dim.values))
    return rng.randint(int(dim.min), int(dim.max))


def _sample_float(rng: random.Random, dim: NumericDim) -> float:
    if isinstance(dim, DiscreteValues):
        return float(rng.choice(dim.values))
    return rng.uniform(dim.min, dim.max)


def _suggest_int(trial: optuna.Trial, name: str, dim: NumericDim) -> int:
    if isinstance(dim, DiscreteValues):
        return int(trial.suggest_categorical(name, [int(v) for v in dim.values]))
    return trial.suggest_int(name, int(dim.min), int(dim.max))


def _suggest_float(trial: optuna.Trial, name: str, dim: NumericDim) -> float:
    if isinstance(dim, DiscreteValues):
        return float(trial.suggest_categorical(name, [float(v) for v in dim.values]))
    return trial.suggest_float(name, float(dim.min), float(dim.max))


def _filter_discrete_le(dim: DiscreteValues, upper_inclusive: int | float) -> list[float | int]:
    """Discrete values ``<= upper_inclusive``. Empty when no value qualifies."""
    return [v for v in dim.values if v <= upper_inclusive]


def _filter_discrete_lt(dim: DiscreteValues, upper_exclusive: int | float) -> list[float | int]:
    """Discrete values ``< upper_exclusive``. Empty when no value qualifies."""
    return [v for v in dim.values if v < upper_exclusive]


def _snap_to_nearest(value: int | float, sorted_values: list[int] | list[float]) -> int | float:
    """Snap ``value`` to the closest entry in a sorted-ascending list.

    Ties go to the lower entry (Python ``min`` returns the first match when
    the key is equal). Caller guarantees ``sorted_values`` is non-empty.
    """
    return min(sorted_values, key=lambda v: abs(v - value))


def _sample_chunk_token_overlap(
    rng: random.Random,
    ss: SearchSpace,
    chunk_token_size: int,
) -> int:
    """Sample chunk_token_overlap with the ``overlap < chunk_token_size`` filter."""
    dim = ss.chunking.chunk_token_overlap
    if isinstance(dim, DiscreteValues):
        legal = _filter_discrete_lt(dim, chunk_token_size)
        if not legal:
            raise RuntimeError(
                f"No DiscreteValue in chunk_token_overlap={dim.values} is "
                f"< chunk_token_size={chunk_token_size}. SearchSpace.chunk_overlap_feasible "
                "validator should have caught this at config load."
            )
        return int(rng.choice(legal))
    co_lo = int(dim.min)
    co_hi = max(co_lo, min(int(dim.max), chunk_token_size - 1))
    return rng.randint(co_lo, co_hi)


def _suggest_chunk_token_overlap(
    trial: optuna.Trial,
    ss: SearchSpace,
    chunk_token_size: int,
) -> int:
    """Optuna analog of :func:`_sample_chunk_token_overlap`.

    DiscreteValues: ``suggest_int`` over the legal interval (values strictly
    less than ``chunk_token_size``), then snap to the nearest grid value.
    Optuna's int distributions support per-trial bounds, so this avoids the
    categorical snap-back surrogate corruption.
    """
    dim = ss.chunking.chunk_token_overlap
    if isinstance(dim, DiscreteValues):
        legal = [int(v) for v in dim.values if v < chunk_token_size]
        if not legal:
            raise RuntimeError(
                f"No DiscreteValue in chunk_token_overlap={dim.values} is "
                f"< chunk_token_size={chunk_token_size}. "
                "SearchSpace.chunk_overlap_feasible validator should have "
                "caught this at config load."
            )
        lo, hi = legal[0], legal[-1]
        chosen = lo if lo == hi else trial.suggest_int("chunk_token_overlap", lo, hi)
        return int(_snap_to_nearest(chosen, legal))
    co_lo = int(dim.min)
    co_hi = max(co_lo, min(int(dim.max), chunk_token_size - 1))
    return trial.suggest_int("chunk_token_overlap", co_lo, co_hi)


def _sample_reranker_top_n(rng: random.Random, dim: NumericDim, top_k: int) -> int:
    """Sample reranker_top_n bounded by ``top_k`` (the upstream retrieval size)."""
    if isinstance(dim, DiscreteValues):
        legal = _filter_discrete_le(dim, top_k)
        if not legal:
            raise RuntimeError(
                f"No DiscreteValue in reranker.top_n={dim.values} is <= "
                f"top_k={top_k}. SearchSpace.reranker_top_n_feasible validator "
                "should have caught this at config load."
            )
        return int(rng.choice(legal))
    rn_lo = int(dim.min)
    rn_hi = max(rn_lo, min(int(dim.max), top_k))
    return rng.randint(rn_lo, rn_hi)


def _suggest_reranker_top_n(trial: optuna.Trial, dim: NumericDim, top_k: int) -> int:
    """Optuna analog of :func:`_sample_reranker_top_n`.

    DiscreteValues: ``suggest_int`` over the legal interval (values
    ``<= top_k``), then snap to the nearest grid value. This is Optuna's
    canonical pattern for dependent parameters (FAQ: per-trial int bounds
    are allowed; only categorical value sets must stay fixed). TPE learns
    from the suggested int and the actual score together — no mislabelled
    trials.
    """
    if isinstance(dim, DiscreteValues):
        legal = [int(v) for v in dim.values if v <= top_k]
        if not legal:
            raise RuntimeError(
                f"No DiscreteValue in reranker.top_n={dim.values} is <= "
                f"top_k={top_k}. SearchSpace.reranker_top_n_feasible validator "
                "should have caught this at config load."
            )
        lo, hi = legal[0], legal[-1]
        chosen = lo if lo == hi else trial.suggest_int("reranker_top_n", lo, hi)
        return int(_snap_to_nearest(chosen, legal))
    rn_lo = int(dim.min)
    rn_hi = max(rn_lo, min(int(dim.max), top_k))
    return trial.suggest_int("reranker_top_n", rn_lo, rn_hi)


def sample_random(
    rng: random.Random,
    search_space: SearchSpace,
) -> TrialConfig:
    """Uniform random sample from the search space, gated by validators."""
    ss = search_space

    chunking_strategy = rng.choice(ss.chunking.strategies)
    embedding_model = rng.choice(list(ss.embedding.models))
    chunk_token_size = _sample_int(rng, ss.chunking.chunk_token_size)
    chunk_token_overlap = _sample_chunk_token_overlap(rng, ss, chunk_token_size)

    index_type = rng.choice(list(ss.retrieval.index_types))
    top_k = _sample_int(rng, ss.retrieval.top_k)

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        bm25_vector_fusion = rng.choice(ss.retrieval.bm25_vector_fusion)
        # hybrid_alpha only feeds alpha-blend; skip sampling it under rrf.
        if bm25_vector_fusion == "alpha":
            hybrid_alpha = round(_sample_float(rng, ss.retrieval.hybrid_alpha), 4)
        else:
            hybrid_alpha = round(_dim_midpoint(ss.retrieval.hybrid_alpha), 4)
    else:
        bm25_vector_fusion = ss.retrieval.bm25_vector_fusion[0]
        hybrid_alpha = round(_dim_midpoint(ss.retrieval.hybrid_alpha), 4)

    reranker = rng.choice(ss.reranker.models)
    if reranker != "none":
        reranker_top_n = _sample_reranker_top_n(rng, ss.reranker.top_n, top_k)
    else:
        reranker_top_n = int(_dim_min_value(ss.reranker.top_n))

    query_expansion = rng.choice(ss.query_expansion.strategies)
    long_context_reorder = rng.choice(ss.retrieval.long_context_reorder)
    passage_compressor = rng.choice(ss.passage_compressor.strategies)
    # Per-stage LLMs: generator always sampled; compressor/expander only
    # when the stage actually runs (matches TrialConfig's cross-field
    # validator). Each stage draws from its own pool.
    generator_llm = rng.choice(ss.generator.models)
    compressor_llm = rng.choice(ss.passage_compressor.models) if passage_compressor != "none" else None
    expander_llm = rng.choice(ss.query_expansion.models) if query_expansion != "none" else None
    temperature = round(rng.uniform(ss.temperature.min, ss.temperature.max), 4)
    reasoning = rng.choice([False, True]) if ss.is_reasoning_allowed(generator_llm) else False

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        graph_query_mode = rng.choice(gr.graph_query_modes)
        graph_top_k = _sample_int(rng, gr.graph_top_k)
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


def config_to_optuna_params(config: TrialConfig, search_space: SearchSpace) -> dict[str, object]:
    """Inverse of :func:`sample_optuna`: the fixed-parameter dict that, enqueued
    via ``study.enqueue_trial`` and replayed through ``sample_optuna``, reproduces
    ``config`` exactly.

    Only the parameters ``sample_optuna`` would actually ``suggest_*`` for this
    config's active conditional branch are emitted, with the types Optuna expects
    (``index_type`` as its string ``.value``, discrete/range ints as ``int``,
    ``hybrid_alpha``/``temperature`` as ``float``). The two snap-to-grid dims
    (``chunk_token_overlap``, ``reranker_top_n``) are emitted only when their
    legal set spans more than one value — exactly the condition under which
    ``sample_optuna`` issues a ``suggest_int`` for them. Fidelity matters for
    warm-start: a missing key would let TPE resample that dimension, so the
    evaluated seed would drift from the agent's proposal. The round-trip is
    covered by a property test.
    """
    ss = search_space
    params: dict[str, object] = {}

    params["chunking_strategy"] = config.chunking_strategy
    params["embedding_model"] = config.embedding_model
    params["chunk_token_size"] = int(config.chunk_token_size)

    co_dim = ss.chunking.chunk_token_overlap
    if isinstance(co_dim, DiscreteValues):
        legal = [int(v) for v in co_dim.values if v < config.chunk_token_size]
        if legal and legal[0] != legal[-1]:
            params["chunk_token_overlap"] = int(config.chunk_token_overlap)
    else:
        params["chunk_token_overlap"] = int(config.chunk_token_overlap)

    params["index_type"] = getattr(config.index_type, "value", config.index_type)
    params["top_k"] = int(config.top_k)

    if config.index_type == IndexType.HYBRID_BM25_VECTOR:
        params["bm25_vector_fusion"] = config.bm25_vector_fusion
        if config.bm25_vector_fusion == "alpha":
            params["hybrid_alpha"] = float(config.hybrid_alpha)

    params["reranker"] = config.reranker
    if config.reranker != "none":
        rn_dim = ss.reranker.top_n
        if isinstance(rn_dim, DiscreteValues):
            legal = [int(v) for v in rn_dim.values if v <= config.top_k]
            if legal and legal[0] != legal[-1]:
                params["reranker_top_n"] = int(config.reranker_top_n)
        else:
            params["reranker_top_n"] = int(config.reranker_top_n)

    params["query_expansion"] = config.query_expansion
    params["long_context_reorder"] = config.long_context_reorder
    params["passage_compressor"] = config.passage_compressor
    params["generator_llm"] = config.generator_llm
    if config.passage_compressor != "none":
        params["compressor_llm"] = config.compressor_llm
    if config.query_expansion != "none":
        params["expander_llm"] = config.expander_llm
    params["temperature"] = float(config.temperature)
    if ss.is_reasoning_allowed(config.generator_llm):
        params["reasoning"] = config.reasoning

    if config.index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        params["graph_query_mode"] = config.graph_query_mode
        params["graph_top_k"] = int(config.graph_top_k)

    return params


def sample_optuna(
    trial: optuna.Trial,
    search_space: SearchSpace,
) -> TrialConfig:
    """Define-by-run sample over the full search space."""
    ss = search_space

    chunking_strategy = trial.suggest_categorical("chunking_strategy", ss.chunking.strategies)
    embedding_model = trial.suggest_categorical("embedding_model", list(ss.embedding.models))
    chunk_token_size = _suggest_int(trial, "chunk_token_size", ss.chunking.chunk_token_size)
    chunk_token_overlap = _suggest_chunk_token_overlap(trial, ss, chunk_token_size)

    index_type = IndexType(trial.suggest_categorical("index_type", [it.value for it in ss.retrieval.index_types]))
    top_k = _suggest_int(trial, "top_k", ss.retrieval.top_k)

    if index_type == IndexType.HYBRID_BM25_VECTOR:
        bm25_vector_fusion = trial.suggest_categorical("bm25_vector_fusion", ss.retrieval.bm25_vector_fusion)
        if bm25_vector_fusion == "alpha":
            hybrid_alpha = _suggest_float(trial, "hybrid_alpha", ss.retrieval.hybrid_alpha)
        else:
            hybrid_alpha = _dim_midpoint(ss.retrieval.hybrid_alpha)
    else:
        bm25_vector_fusion = ss.retrieval.bm25_vector_fusion[0]
        hybrid_alpha = _dim_midpoint(ss.retrieval.hybrid_alpha)

    reranker = trial.suggest_categorical("reranker", ss.reranker.models)
    if reranker != "none":
        reranker_top_n = _suggest_reranker_top_n(trial, ss.reranker.top_n, top_k)
    else:
        reranker_top_n = int(_dim_min_value(ss.reranker.top_n))

    query_expansion = trial.suggest_categorical("query_expansion", ss.query_expansion.strategies)
    long_context_reorder = trial.suggest_categorical("long_context_reorder", ss.retrieval.long_context_reorder)
    passage_compressor = trial.suggest_categorical("passage_compressor", ss.passage_compressor.strategies)
    # Per-stage LLMs: generator always sampled; compressor/expander only
    # when the stage runs (TPE skips dead dimensions on the inactive
    # branches of conditional suggests).
    generator_llm = trial.suggest_categorical("generator_llm", ss.generator.models)
    if passage_compressor != "none":
        compressor_llm = trial.suggest_categorical("compressor_llm", ss.passage_compressor.models)
    else:
        compressor_llm = None
    if query_expansion != "none":
        expander_llm = trial.suggest_categorical("expander_llm", ss.query_expansion.models)
    else:
        expander_llm = None
    temperature = trial.suggest_float("temperature", ss.temperature.min, ss.temperature.max)
    reasoning = (
        trial.suggest_categorical("reasoning", [False, True]) if ss.is_reasoning_allowed(generator_llm) else False
    )

    if index_type in GRAPH_INDEX_TYPES and ss.graph_retrieval is not None:
        gr = ss.graph_retrieval
        graph_query_mode = trial.suggest_categorical("graph_query_mode", gr.graph_query_modes)
        graph_top_k = _suggest_int(trial, "graph_top_k", gr.graph_top_k)
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
