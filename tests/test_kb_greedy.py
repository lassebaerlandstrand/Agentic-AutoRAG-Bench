"""Tests for the kb_greedy single-config reference.

``build_strongest_config`` must reproduce the examiner's Tier-4-strong probe
(strongest LLM/embedder/reranker, max chunk/top_k/reranker_top_n) and override a
graph index to a non-graph one. ``run_kb_greedy`` must evaluate that single
config exactly once on the hold-out — no search loop. Both paths are mocked: the
model ranking is replaced by an identity (weakest→strongest = input order) so no
network/KB ranking is needed, and the evaluator is an AsyncMock spy.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from agentic_autorag.config.models import (
    AgentConfig,
    ChunkingSearchSpace,
    DiscreteValues,
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    MetaConfig,
    NumericRange,
    ProjectConfig,
    RerankerSearchSpace,
    RetrievalSearchSpace,
    SearchSpace,
    TrialConfig,
)

from agentic_autorag_bench.methods.kb_greedy import build_strongest_config, run_kb_greedy

# Identity ranking: rank_models_for_probes returns its input list unchanged, so
# "strongest" = the last model in each search-space list.
_RANK_IDENTITY = "agentic_autorag_bench.methods.kb_greedy.rank_models_for_probes"


def _search_space(index_types: list[IndexType] | None = None) -> SearchSpace:
    return SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive", "fixed"],
            chunk_token_size=DiscreteValues(values=[128, 512]),
            chunk_token_overlap=DiscreteValues(values=[0, 32, 64]),
        ),
        embedding=EmbeddingSearchSpace(models=["weak_embed", "strong_embed"]),
        retrieval=RetrievalSearchSpace(
            index_types=index_types or [IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
            top_k=NumericRange(min=3, max=20),
            hybrid_alpha=NumericRange(min=0.0, max=1.0),
            bm25_vector_fusion=["alpha", "rrf"],
            long_context_reorder=[False, True],
        ),
        reranker=RerankerSearchSpace(
            models=["none", "BAAI/bge-reranker-v2-m3"],
            top_n=DiscreteValues(values=[3, 10]),
        ),
        generator=GeneratorSearchSpace(models=["ollama/llama3.2", "ollama/mistral"]),
        temperature=NumericRange(min=0.0, max=1.0),
    )


def _project(index_types: list[IndexType] | None = None) -> ProjectConfig:
    return ProjectConfig(
        meta=MetaConfig(corpus_description="A tiny test corpus."),
        search_space=_search_space(index_types),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2",
            examiner_model="ollama/llama3.2",
            judge_model="ollama/llama3.2",
        ),
    )


async def _identity_rank(model_names, *args, **kwargs):
    return list(model_names)


@pytest.mark.asyncio
async def test_build_strongest_config_picks_kb_strongest() -> None:
    with patch(_RANK_IDENTITY, new=AsyncMock(side_effect=_identity_rank)):
        trial = await build_strongest_config(_project())

    assert trial.generator_llm == "ollama/mistral"
    assert trial.embedding_model == "strong_embed"
    assert trial.reranker == "BAAI/bge-reranker-v2-m3"
    assert trial.chunk_token_size == 512  # max chunk
    assert trial.top_k == 20  # max top_k
    assert trial.reranker_top_n == 10  # max reranker_top_n
    assert trial.index_type == IndexType.HYBRID_BM25_VECTOR  # strongest non-graph index


@pytest.mark.asyncio
async def test_build_strongest_config_overrides_graph_index() -> None:
    """A graph index (held-out runner can't build graphs) is replaced by the
    strongest available non-graph index — HYBRID_BM25_VECTOR preferred."""
    project = _project()  # space = [vector_only, hybrid_bm25_vector]
    graph_trial = TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="strong_embed",
        index_type=IndexType.HYBRID_GRAPH_VECTOR,
        top_k=20,
        reranker="BAAI/bge-reranker-v2-m3",
        reranker_top_n=10,
        generator_llm="ollama/mistral",
        temperature=0.0,
    )
    with (
        patch(_RANK_IDENTITY, new=AsyncMock(side_effect=_identity_rank)),
        patch(
            "agentic_autorag_bench.methods.kb_greedy.select_probe_configs",
            return_value=[("Tier4-strong", graph_trial)],
        ),
    ):
        trial = await build_strongest_config(project)

    assert trial.index_type == IndexType.HYBRID_BM25_VECTOR


def _write_bench_config(tmp_path: Path) -> Path:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(yaml.safe_dump({"meta": {"output_dir": str(tmp_path / "_cache")}}))
    config = {
        "project_config": str(project_yaml),
        "methods": ["random"],
        "seeds": [1],
        "budget": {"max_trials": 40},
        "benchmark": {
            "name": "hotpot_qa",
            "split": "validation",
            "sample_size": 100,
            "prep_seed": 42,
            "output_dir": str(tmp_path / "_data"),
        },
        "hold_out": {"limit": 10, "judge_model": "test", "concurrency": 1},
        "output_root": str(tmp_path / "results"),
    }
    config_yaml = tmp_path / "bench_config.yaml"
    config_yaml.write_text(yaml.safe_dump(config))
    return config_yaml


@pytest.mark.asyncio
async def test_run_kb_greedy_evaluates_once(tmp_path: Path) -> None:
    """kb_greedy scores exactly one config on the hold-out and writes the seed
    dir the matrix figures read."""
    config_path = _write_bench_config(tmp_path)
    fixed = TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="strong_embed",
        index_type=IndexType.VECTOR_ONLY,
        top_k=20,
        reranker="strong_reranker",
        reranker_top_n=10,
        generator_llm="strong_llm",
        temperature=0.0,
    )
    evaluate = AsyncMock(return_value={"answer_accuracy": 0.7})

    with (
        patch("agentic_autorag_bench.methods.kb_greedy.build_strongest_config", new=AsyncMock(return_value=fixed)),
        patch("agentic_autorag_bench.methods.kb_greedy.load_config", return_value=_project()),
        patch("agentic_autorag_bench.benchmarks.runner.BenchmarkRunner.evaluate", new=evaluate),
        patch("agentic_autorag_bench.benchmarks.runner.BenchmarkRunner.prepare"),
        patch("agentic_autorag_bench.plots.make_seed_figures"),
        patch("agentic_autorag_bench.plots.make_matrix_figures"),
        patch("agentic_autorag_bench._holdout_registry.apply_union_exclusion"),
        patch("agentic_autorag.litellm_runtime.configure_litellm_runtime"),
    ):
        await run_kb_greedy(config_path, seed=7)

    evaluate.assert_awaited_once()
    # The single config scored is the strongest one.
    assert evaluate.await_args.kwargs["trial_config"] is fixed

    seed_dir = tmp_path / "results" / "kb_greedy" / "seed_7"
    assert (seed_dir / "best_config.yaml").exists()
    assert (seed_dir / "optimizer_meta.json").exists()
    assert evaluate.await_args.kwargs["output_path"] == seed_dir / "benchmark_results.json"
