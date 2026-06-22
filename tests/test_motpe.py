"""MO-TPE baseline: behavioral equivalence to syftr, config-driven objective
mode, warm-start enqueue, and the warm-start inverse-mapping round-trip.

These are the correctness-critical tests for the strong baseline. The LLM is
never called: the warm-start proposer is mocked, and the round-trip is pure
sampler math.
"""

from __future__ import annotations

import random
from pathlib import Path

import optuna
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
    TrialConfig,
)

from agentic_autorag_bench.methods._sampler import (
    config_to_optuna_params,
    sample_optuna,
    sample_random,
)
from agentic_autorag_bench.methods.motpe import (
    SAMPLER_KWARGS,
    MOTPESearch,
    _make_sampler,
)
from agentic_autorag_bench.types import Budget, TrialResult

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _rich_search_space() -> SearchSpace:
    """A space that exercises every conditional branch sample_optuna gates on:
    vector + hybrid index (→ bm25_vector_fusion + hybrid_alpha), a real reranker
    (→ reranker_top_n), query expansion (→ expander_llm), passage compression
    (→ compressor_llm), and both DiscreteValues and NumericRange numeric dims.
    Graph branches are out of scope (no graph index in the paper spaces)."""
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
            models=["none", "BAAI/bge-reranker-v2-m3"],
            top_n=DiscreteValues(values=[3, 5, 10]),
        ),
        query_expansion=QueryExpansionSearchSpace(strategies=["none", "hyde"], models=["ollama/llama3.2"]),
        passage_compressor=PassageCompressorSearchSpace(
            strategies=["none", "tree_summarize"], models=["ollama/llama3.2"]
        ),
        generator=GeneratorSearchSpace(models=["ollama/llama3.2", "ollama/mistral"]),
        temperature=NumericRange(min=0.0, max=1.0),
    )


def _project(cost_aware: bool, search_space: SearchSpace | None = None) -> ProjectConfig:
    return ProjectConfig(
        meta=MetaConfig(cost_aware=cost_aware, corpus_description="A tiny test corpus."),
        search_space=search_space or _rich_search_space(),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2",
            examiner_model="ollama/llama3.2",
            judge_model="ollama/llama3.2",
        ),
    )


def _make_evaluator(scores: list[float], costs: list[float] | None = None):
    counter = {"i": 0}

    async def evaluator(config: TrialConfig) -> TrialResult:
        i = counter["i"]
        score = scores[i % len(scores)]
        cost = (costs[i % len(costs)] if costs else 0.0)
        counter["i"] += 1
        return TrialResult(
            answer_accuracy=score,
            metrics={"answer_accuracy": score, "mean_em": score, "mean_f1": score, "mean_retrieval_quality": score},
            eval_usd=0.001,
            mean_llm_cost_per_query_usd=cost,
        )

    return evaluator


# --------------------------------------------------------------- equivalence


def test_sampler_kwargs_match_syftr_published_config() -> None:
    """Behavioral-equivalence: our sampler construction == syftr's published
    MO-TPE config (syftr/optuna_helper.py). Tests the data path, not prompts."""
    assert SAMPLER_KWARGS == {
        "multivariate": True,
        "group": True,
        "constant_liar": True,
        "n_startup_trials": 10,
    }


def test_make_sampler_is_tpe() -> None:
    assert isinstance(_make_sampler(seed=1), optuna.samplers.TPESampler)


# ----------------------------------------------------- objective-mode toggle


@pytest.mark.asyncio
async def test_cost_aware_drives_objective_directions(tmp_path: Path, monkeypatch) -> None:
    """``meta.cost_aware`` selects single- vs multi-objective, mirroring how the
    agentic optimizer flips score↔cost on the same flag."""
    captured: dict = {}
    real_create = optuna.create_study

    def spy(*args, **kwargs):
        captured["direction"] = kwargs.get("direction")
        captured["directions"] = kwargs.get("directions")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", spy)
    vector_only = SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive"],
            chunk_token_size=NumericRange(min=256, max=512),
            chunk_token_overlap=NumericRange(min=0, max=64),
        ),
        embedding=EmbeddingSearchSpace(models=["m1"]),
        retrieval=RetrievalSearchSpace(index_types=[IndexType.VECTOR_ONLY], top_k=NumericRange(min=3, max=10)),
        reranker=RerankerSearchSpace(models=["none"], top_n=NumericRange(min=3, max=10)),
        query_expansion=QueryExpansionSearchSpace(strategies=["none"], models=[]),
        passage_compressor=PassageCompressorSearchSpace(strategies=["none"], models=[]),
        generator=GeneratorSearchSpace(models=["ollama/llama3.2"]),
        temperature=NumericRange(min=0.0, max=1.0),
    )

    # Multi-objective when cost_aware.
    await MOTPESearch(project=_project(True, vector_only), storage_dir=tmp_path / "mo").search(
        _make_evaluator([0.5, 0.6, 0.7], costs=[0.003, 0.001, 0.002]), Budget(max_trials=3), seed=1
    )
    assert captured["directions"] == ["maximize", "minimize"]
    assert captured["direction"] is None

    # Single-objective otherwise.
    await MOTPESearch(project=_project(False, vector_only), storage_dir=tmp_path / "so").search(
        _make_evaluator([0.5, 0.6, 0.7]), Budget(max_trials=3), seed=1
    )
    assert captured["direction"] == "maximize"
    assert captured["directions"] is None


@pytest.mark.asyncio
async def test_multiobjective_records_per_query_cost(tmp_path: Path) -> None:
    """The MO run threads ``mean_llm_cost_per_query_usd`` onto every history row
    (the second objective + what the Pareto figure reads)."""
    sr = await MOTPESearch(project=_project(True), storage_dir=tmp_path).search(
        _make_evaluator([0.4, 0.6, 0.8], costs=[0.004, 0.002, 0.006]), Budget(max_trials=3), seed=1
    )
    assert sr.extras["cost_aware"] is True
    costs = {round(h.mean_llm_cost_per_query_usd, 6) for h in sr.history}
    assert costs == {0.004, 0.002, 0.006}
    # best_config is the max-accuracy point in both modes.
    assert sr.best_config == max(sr.history, key=lambda h: h.answer_accuracy).config


# -------------------------------------------------- warm-start inverse round-trip


def test_config_to_optuna_params_round_trip() -> None:
    """Enqueueing the inverse params and replaying through ``sample_optuna``
    reproduces the original config EXACTLY across every conditional branch.

    This is the fidelity guarantee warm-start relies on: a missing/incorrect
    param would let TPE resample that dim, so the evaluated seed would drift
    from the agent's proposal.
    """
    ss = _rich_search_space()
    rng = random.Random(12345)
    for i in range(80):
        original = sample_random(rng, ss)
        params = config_to_optuna_params(original, ss)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=i))
        study.enqueue_trial(params, skip_if_exists=False)
        trial = study.ask()
        replayed = sample_optuna(trial, ss)
        assert replayed.to_prompt_dump(include_graph=False) == original.to_prompt_dump(include_graph=False), (
            f"round-trip mismatch on draw {i}:\n"
            f" original={original.to_prompt_dump(include_graph=False)}\n"
            f" replayed={replayed.to_prompt_dump(include_graph=False)}\n"
            f" params={params}"
        )


# ------------------------------------------------------- warm-start end-to-end


def _grid_config(top_k: int) -> TrialConfig:
    """A valid config in the discrete vector-only space below, distinct by top_k."""
    return TrialConfig(
        chunking_strategy="recursive",
        chunk_token_size=256,
        chunk_token_overlap=0,
        embedding_model="m1",
        index_type=IndexType.VECTOR_ONLY,
        top_k=top_k,
        reranker="none",
        reranker_top_n=3,
        query_expansion="none",
        passage_compressor="none",
        generator_llm="ollama/llama3.2",
        temperature=0.0,
        reasoning=False,
    )


@pytest.mark.asyncio
async def test_warmstart_evaluates_cold_proposer_configs_first(tmp_path: Path, monkeypatch) -> None:
    """``motpe_warmstart`` enqueues the cold proposer's configs so the first
    asks return the agent's frozen prior (not random startup draws)."""
    from agentic_autorag.optimizer.reasoning_agent import ReasoningAgent

    space = SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive"],
            chunk_token_size=DiscreteValues(values=[256, 512]),
            chunk_token_overlap=DiscreteValues(values=[0, 64]),
        ),
        embedding=EmbeddingSearchSpace(models=["m1"]),
        retrieval=RetrievalSearchSpace(index_types=[IndexType.VECTOR_ONLY], top_k=DiscreteValues(values=[3, 5, 10])),
        reranker=RerankerSearchSpace(models=["none"], top_n=DiscreteValues(values=[3, 5, 10])),
        query_expansion=QueryExpansionSearchSpace(strategies=["none"], models=[]),
        passage_compressor=PassageCompressorSearchSpace(strategies=["none"], models=[]),
        generator=GeneratorSearchSpace(models=["ollama/llama3.2"]),
        temperature=NumericRange(min=0.0, max=0.0),
    )
    project = _project(False, space)

    canned = [_grid_config(3), _grid_config(5), _grid_config(10)]
    calls = {"i": 0}

    async def fake_propose_initial(self, corpus_description):  # noqa: ANN001
        cfg = canned[calls["i"] % len(canned)]
        calls["i"] += 1
        return cfg

    monkeypatch.setattr(ReasoningAgent, "propose_initial", fake_propose_initial)

    optimizer = MOTPESearch(project=project, storage_dir=tmp_path, warm_start=True, name="motpe_warmstart")
    sr = await optimizer.search(_make_evaluator([0.5] * 3), Budget(max_trials=3), seed=1)

    # The 3 distinct cold configs were enqueued and evaluated first, in order.
    assert [h.config["top_k"] for h in sr.history[:3]] == [3, 5, 10]
    assert sr.extras["n_warmstart"] == 3
    assert sr.method == "motpe_warmstart"
