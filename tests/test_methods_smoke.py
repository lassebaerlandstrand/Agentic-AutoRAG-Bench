"""Fast smoke: every sequential method runs to completion under a mocked evaluator.

Catches Optimizer-protocol drift (signature, return shape) without paying real
LLM cost. The agentic and AutoRAG methods are exercised separately — they
require a real Orchestrator and a subprocess respectively, neither of which is
fast enough for the default test path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_autorag.config.models import (
    ChunkingSearchSpace,
    IndexType,
    NumericRange,
    ProjectConfig,
    RerankerSearchSpace,
    SearchSpace,
    TrialConfig,
)

from agentic_autorag_bench.methods.bayesian import BayesianSearch
from agentic_autorag_bench.methods.random import RandomSearch
from agentic_autorag_bench.types import Budget, TrialResult


def _tiny_project() -> ProjectConfig:
    return ProjectConfig(
        search_space=SearchSpace(
            chunking=ChunkingSearchSpace(
                strategies=["recursive"],
                chunk_token_size=NumericRange(min=256, max=512),
                chunk_token_overlap=NumericRange(min=0, max=64),
            ),
            embedding_models=["sentence-transformers/all-MiniLM-L6-v2"],
            index_types=[IndexType.VECTOR_ONLY],
            top_k=NumericRange(min=3, max=10),
            hybrid_alpha=NumericRange(min=0.0, max=1.0),
            reranker=RerankerSearchSpace(
                models=["none"],
                top_n=NumericRange(min=3, max=10),
            ),
            query_expansion=["none"],
            llm_models=["ollama/llama3.2"],
            temperature=NumericRange(min=0.0, max=1.0),
        )
    )


def _make_evaluator(scores: list[float]):
    """Return a callable that scores trials in order from ``scores``, looping."""
    counter = {"i": 0}

    async def evaluator(config: TrialConfig) -> TrialResult:
        score = scores[counter["i"] % len(scores)]
        counter["i"] += 1
        return TrialResult(
            score=score,
            metrics={"answer_accuracy": score, "mean_em": score, "mean_f1": score, "mean_retrieval_quality": score},
            eval_usd=0.001,
        )

    return evaluator


@pytest.mark.asyncio
async def test_random_search_runs_to_completion() -> None:
    project = _tiny_project()
    optimizer = RandomSearch(project=project)
    evaluator = _make_evaluator([0.3, 0.7, 0.5])

    sr = await optimizer.search(evaluator, Budget(max_trials=3), seed=42)

    assert sr.method == "random"
    assert sr.seed == 42
    assert sr.deterministic is False
    assert len(sr.history) == 3
    assert max(h.score for h in sr.history) == 0.7
    assert sr.best_config["chunking_strategy"] == "recursive"
    assert sr.trial_usd_total == pytest.approx(3 * 0.001)


@pytest.mark.asyncio
async def test_random_search_is_seed_reproducible() -> None:
    project = _tiny_project()
    optimizer = RandomSearch(project=project)
    evaluator_a = _make_evaluator([0.1, 0.2, 0.3])
    evaluator_b = _make_evaluator([0.1, 0.2, 0.3])

    a = await optimizer.search(evaluator_a, Budget(max_trials=3), seed=7)
    b = await optimizer.search(evaluator_b, Budget(max_trials=3), seed=7)

    # Same seed → same proposed configs → same history
    assert [h.config for h in a.history] == [h.config for h in b.history]


@pytest.mark.asyncio
async def test_random_search_different_seeds_diverge() -> None:
    project = _tiny_project()
    optimizer = RandomSearch(project=project)
    evaluator_a = _make_evaluator([0.5])
    evaluator_b = _make_evaluator([0.5])

    a = await optimizer.search(evaluator_a, Budget(max_trials=5), seed=1)
    b = await optimizer.search(evaluator_b, Budget(max_trials=5), seed=2)

    # Different seeds → at least one different config in the history
    assert any(h_a.config != h_b.config for h_a, h_b in zip(a.history, b.history, strict=True))


@pytest.mark.asyncio
async def test_bayesian_search_runs_to_completion(tmp_path: Path) -> None:
    project = _tiny_project()
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.3, 0.7, 0.5])

    sr = await optimizer.search(evaluator, Budget(max_trials=3), seed=42)

    assert sr.method == "bayesian"
    assert len(sr.history) == 3
    assert max(h.score for h in sr.history) == 0.7
    assert (tmp_path / "optuna.db").exists()
    assert (tmp_path / "optuna_sampler.pkl").exists()


@pytest.mark.asyncio
async def test_random_rejects_when_budget_missing() -> None:
    project = _tiny_project()
    optimizer = RandomSearch(project=project)
    evaluator = _make_evaluator([0.5])

    with pytest.raises(ValueError, match="max_trials"):
        await optimizer.search(evaluator, Budget(), seed=0)


@pytest.mark.asyncio
async def test_bayesian_rejects_when_budget_missing(tmp_path: Path) -> None:
    project = _tiny_project()
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.5])

    with pytest.raises(ValueError, match="max_trials"):
        await optimizer.search(evaluator, Budget(), seed=0)


def _multi_embedding_project() -> ProjectConfig:
    """Search space with heterogeneous embedding token limits.

    Regression coverage for the Optuna ``CategoricalDistribution`` static-domain
    constraint: a previous implementation filtered the embedding list against
    the sampled ``chunk_token_size`` and crashed on trial 2 with
    ``CategoricalDistribution does not support dynamic value space``. Three
    models with different limits and a chunk range that straddles them is the
    minimum case that exercises this.
    """
    return ProjectConfig(
        search_space=SearchSpace(
            chunking=ChunkingSearchSpace(
                strategies=["recursive"],
                chunk_token_size=NumericRange(min=128, max=512),
                chunk_token_overlap=NumericRange(min=0, max=64),
            ),
            embedding_models=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-m3",
            ],
            index_types=[IndexType.VECTOR_ONLY],
            top_k=NumericRange(min=3, max=10),
            hybrid_alpha=NumericRange(min=0.0, max=1.0),
            reranker=RerankerSearchSpace(
                models=["none"],
                top_n=NumericRange(min=3, max=10),
            ),
            query_expansion=["none"],
            llm_models=["ollama/llama3.2"],
            temperature=NumericRange(min=0.0, max=1.0),
        ),
        embedding_token_limits={
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "BAAI/bge-large-en-v1.5": 512,
            "BAAI/bge-m3": 8192,
        },
    )


@pytest.mark.asyncio
async def test_bayesian_with_mixed_embedding_limits_runs_all_trials(tmp_path: Path) -> None:
    """All 8 trials must complete — none pruned by the previous dynamic-domain crash."""
    project = _multi_embedding_project()
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.3, 0.4, 0.5, 0.6, 0.7, 0.55, 0.45, 0.35])

    sr = await optimizer.search(evaluator, Budget(max_trials=8), seed=42)

    assert len(sr.history) == 8
    for entry in sr.history:
        embedding = entry.config["embedding_model"]
        chunk_size = entry.config["chunk_token_size"]
        limit = project.embedding_token_limits[embedding]
        assert chunk_size <= limit, (
            f"trial {entry.trial_number}: {embedding} (limit {limit}) got chunk_size={chunk_size}"
        )


@pytest.mark.asyncio
async def test_random_with_mixed_embedding_limits_respects_per_embedding_chunk_bound() -> None:
    project = _multi_embedding_project()
    optimizer = RandomSearch(project=project)
    evaluator = _make_evaluator([0.5] * 20)

    sr = await optimizer.search(evaluator, Budget(max_trials=20), seed=42)

    assert len(sr.history) == 20
    for entry in sr.history:
        embedding = entry.config["embedding_model"]
        chunk_size = entry.config["chunk_token_size"]
        limit = project.embedding_token_limits[embedding]
        assert chunk_size <= limit


@pytest.mark.asyncio
async def test_bayesian_with_mixed_embedding_limits_explores_all_embeddings(tmp_path: Path) -> None:
    """Static embedding categorical → Bayesian's first few trials see every embedding,
    not just the ones compatible with whatever chunk_token_size happened to land first.
    """
    project = _multi_embedding_project()
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.5] * 15)

    sr = await optimizer.search(evaluator, Budget(max_trials=15), seed=42)

    seen_embeddings = {h.config["embedding_model"] for h in sr.history}
    assert seen_embeddings == set(project.search_space.embedding_models), (
        f"Expected all three embeddings, saw {seen_embeddings}"
    )
