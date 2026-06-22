"""qLogNEHVI (Ax GP-BO) baseline — ax-free core (encode + flatten-decode) + guards.

The Ax dependency is optional and not installed by default, so these tests cover the
parts that don't need it: the search-space encode, the flat-parametrization decode
(conditional gating + grid-snapping), and the dependency / cost-aware guards. The Ax
service loop itself is exercised only once ``ax-platform`` is added.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from agentic_autorag.config.models import (
    AgentConfig,
    ChunkingSearchSpace,
    DiscreteValues,
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    MetaConfig,
    NumericRange,
    PassageCompressorSearchSpace,
    ProjectConfig,
    QueryExpansionSearchSpace,
    RerankerSearchSpace,
    RetrievalSearchSpace,
    SearchSpace,
)

from agentic_autorag_bench.methods._sampler import sample_random
from agentic_autorag_bench.methods.qlognehvi import (
    QLogNEHVISearch,
    ax_parameters,
    decode_params,
)
from agentic_autorag_bench.types import Budget, TrialResult


def _rich_space() -> SearchSpace:
    return SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive", "fixed"],
            chunk_token_size=DiscreteValues(values=[256, 512]),
            chunk_token_overlap=DiscreteValues(values=[0, 32, 64]),
        ),
        embedding=EmbeddingSearchSpace(models=["m1", "m2"]),
        retrieval=RetrievalSearchSpace(
            index_types=[IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
            top_k=NumericRange(min=3, max=20),
            hybrid_alpha=NumericRange(min=0.0, max=1.0),
            bm25_vector_fusion=["alpha", "rrf"],
            long_context_reorder=[False, True],
        ),
        reranker=RerankerSearchSpace(
            models=["none", "BAAI/bge-reranker-v2-m3"], top_n=DiscreteValues(values=[3, 5, 10])
        ),
        query_expansion=QueryExpansionSearchSpace(strategies=["none", "hyde"], models=["ollama/llama3.2"]),
        passage_compressor=PassageCompressorSearchSpace(
            strategies=["none", "tree_summarize"], models=["ollama/llama3.2"]
        ),
        generator=GeneratorSearchSpace(models=["ollama/llama3.2", "ollama/mistral"]),
        temperature=NumericRange(min=0.0, max=1.0),
    )


def _project(cost_aware: bool) -> ProjectConfig:
    return ProjectConfig(
        meta=MetaConfig(cost_aware=cost_aware),
        search_space=_rich_space(),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2", examiner_model="ollama/llama3.2", judge_model="ollama/llama3.2"
        ),
    )


def _flat_params(c) -> dict:
    """The flat Ax parametrization corresponding to a TrialConfig (all params present)."""
    return {
        "chunking_strategy": c.chunking_strategy,
        "embedding_model": c.embedding_model,
        "chunk_token_size": c.chunk_token_size,
        "chunk_token_overlap": c.chunk_token_overlap,
        "index_type": c.index_type.value,
        "top_k": c.top_k,
        "bm25_vector_fusion": c.bm25_vector_fusion,
        "hybrid_alpha": c.hybrid_alpha,
        "long_context_reorder": c.long_context_reorder,
        "reranker": c.reranker,
        "reranker_top_n": c.reranker_top_n,
        "query_expansion": c.query_expansion,
        "passage_compressor": c.passage_compressor,
        "generator_llm": c.generator_llm,
        "temperature": c.temperature,
        "compressor_llm": c.compressor_llm,
        "expander_llm": c.expander_llm,
        "reasoning": c.reasoning,
    }


def test_ax_parameters_encodes_categoricals_and_ranges() -> None:
    ps = {p["name"]: p for p in ax_parameters(_rich_space())}
    # Unordered categoricals -> one-hot choice params (the high-dim degradation source).
    assert ps["embedding_model"]["type"] == "choice" and ps["embedding_model"]["is_ordered"] is False
    assert ps["generator_llm"]["values"] == ["ollama/llama3.2", "ollama/mistral"]
    # Continuous dims -> range.
    assert ps["top_k"]["type"] == "range" and ps["top_k"]["bounds"] == [3, 20]
    assert ps["hybrid_alpha"]["type"] == "range"
    # DiscreteValues numeric -> ordered choice.
    assert ps["chunk_token_size"]["type"] == "choice" and ps["chunk_token_size"]["is_ordered"] is True
    # Conditional stage-LLMs emitted unconditionally (Ax can't gate them; decode does).
    assert "compressor_llm" in ps and "expander_llm" in ps


def test_decode_round_trip_reproduces_configs() -> None:
    """A flat parametrization for any sampled config decodes back to that config —
    the gating + snapping in decode_params lands on the same feasible point."""
    ss = _rich_space()
    rng = random.Random(7)
    for i in range(60):
        original = sample_random(rng, ss)
        decoded = decode_params(_flat_params(original), ss)
        assert decoded.to_prompt_dump(include_graph=False) == original.to_prompt_dump(include_graph=False), (
            f"decode mismatch on draw {i}:\n original={original.to_prompt_dump(include_graph=False)}\n"
            f" decoded={decoded.to_prompt_dump(include_graph=False)}"
        )


def test_decode_gates_inactive_stage_llms() -> None:
    """When the compressor/expander stage is off, decode forces its LLM to None even
    if Ax suggested a model for the (unconditional) param."""
    ss = _rich_space()
    rng = random.Random(0)
    base = sample_random(rng, ss)
    params = _flat_params(base)
    params["passage_compressor"] = "none"
    params["query_expansion"] = "none"
    params["compressor_llm"] = "ollama/llama3.2"  # Ax suggests a value; must be dropped
    params["expander_llm"] = "ollama/llama3.2"
    decoded = decode_params(params, ss)
    assert decoded.compressor_llm is None
    assert decoded.expander_llm is None


def test_decode_snaps_reranker_top_n_within_top_k() -> None:
    ss = _rich_space()
    rng = random.Random(3)
    base = sample_random(rng, ss)
    params = _flat_params(base)
    params["reranker"] = "BAAI/bge-reranker-v2-m3"
    params["top_k"] = 4
    params["reranker_top_n"] = 10  # > top_k and off-grid; must snap to a legal value <= top_k
    decoded = decode_params(params, ss)
    assert decoded.reranker_top_n in {3, 5, 10}
    assert decoded.reranker_top_n <= 4


@pytest.mark.asyncio
async def test_search_requires_cost_aware(tmp_path: Path) -> None:
    """qLogNEHVI is multi-objective: it refuses a single-objective (cost_aware=False) project."""
    opt = QLogNEHVISearch(project=_project(False), storage_dir=tmp_path)

    async def _evaluator(config):  # pragma: no cover - never reached
        return TrialResult(answer_accuracy=0.5, metrics={}, eval_usd=0.0)

    with pytest.raises(NotImplementedError, match="multi-objective"):
        await opt.search(_evaluator, Budget(max_trials=2), seed=1)


@pytest.mark.asyncio
async def test_search_without_ax_raises_helpful_error(tmp_path: Path) -> None:
    """With ax-platform absent, a cost-aware run raises a clear install hint rather
    than an opaque ImportError. (Skips if ax happens to be installed.)"""
    try:
        import ax  # noqa: F401

        pytest.skip("ax-platform is installed; the guard path is not exercised")
    except ImportError:
        pass

    opt = QLogNEHVISearch(project=_project(True), storage_dir=tmp_path)

    async def _evaluator(config):  # pragma: no cover - never reached
        return TrialResult(answer_accuracy=0.5, metrics={}, eval_usd=0.0)

    with pytest.raises(ImportError, match="uv add ax-platform"):
        await opt.search(_evaluator, Budget(max_trials=2), seed=1)
