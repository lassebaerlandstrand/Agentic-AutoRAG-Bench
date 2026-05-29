"""Fast smoke: every sequential method runs to completion under a mocked evaluator.

Catches Optimizer-protocol drift (signature, return shape) without paying real
LLM cost. The agentic method is exercised separately — it requires a real
Orchestrator, which is not fast enough for the default test path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agentic_autorag.config.models import (
    AgentConfig,
    ChunkingSearchSpace,
    DiscreteValues,
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    NumericRange,
    PassageCompressorSearchSpace,
    ProjectConfig,
    QueryExpansionSearchSpace,
    RerankerSearchSpace,
    RetrievalSearchSpace,
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
            embedding=EmbeddingSearchSpace(models=["sentence-transformers/all-MiniLM-L6-v2"]),
            retrieval=RetrievalSearchSpace(
                index_types=[IndexType.VECTOR_ONLY],
                top_k=NumericRange(min=3, max=10),
                hybrid_alpha=NumericRange(min=0.0, max=1.0),
            ),
            reranker=RerankerSearchSpace(
                models=["none"],
                top_n=NumericRange(min=3, max=10),
            ),
            query_expansion=QueryExpansionSearchSpace(strategies=["none"], models=[]),
            passage_compressor=PassageCompressorSearchSpace(strategies=["none"], models=[]),
            generator=GeneratorSearchSpace(models=["ollama/llama3.2"]),
            temperature=NumericRange(min=0.0, max=1.0),
        ),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2",
            examiner_model="ollama/llama3.2",
            judge_model="ollama/llama3.2",
        ),
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
    """Search space with three embedders so the sampler must explore them all."""
    return ProjectConfig(
        search_space=SearchSpace(
            chunking=ChunkingSearchSpace(
                strategies=["recursive"],
                chunk_token_size=NumericRange(min=128, max=512),
                chunk_token_overlap=NumericRange(min=0, max=64),
            ),
            embedding=EmbeddingSearchSpace(
                models=[
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "BAAI/bge-large-en-v1.5",
                    "BAAI/bge-m3",
                ],
            ),
            retrieval=RetrievalSearchSpace(
                index_types=[IndexType.VECTOR_ONLY],
                top_k=NumericRange(min=3, max=10),
                hybrid_alpha=NumericRange(min=0.0, max=1.0),
            ),
            reranker=RerankerSearchSpace(
                models=["none"],
                top_n=NumericRange(min=3, max=10),
            ),
            query_expansion=QueryExpansionSearchSpace(strategies=["none"], models=[]),
            passage_compressor=PassageCompressorSearchSpace(strategies=["none"], models=[]),
            generator=GeneratorSearchSpace(models=["ollama/llama3.2"]),
            temperature=NumericRange(min=0.0, max=1.0),
        ),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2",
            examiner_model="ollama/llama3.2",
            judge_model="ollama/llama3.2",
        ),
    )


@pytest.mark.asyncio
async def test_bayesian_with_multiple_embeddings_runs_all_trials(tmp_path: Path) -> None:
    """All 8 trials complete with a multi-embedder search space."""
    project = _multi_embedding_project()
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.3, 0.4, 0.5, 0.6, 0.7, 0.55, 0.45, 0.35])

    sr = await optimizer.search(evaluator, Budget(max_trials=8), seed=42)

    assert len(sr.history) == 8


@pytest.mark.asyncio
async def test_random_with_multiple_embeddings_runs_all_trials() -> None:
    project = _multi_embedding_project()
    optimizer = RandomSearch(project=project)
    evaluator = _make_evaluator([0.5] * 20)

    sr = await optimizer.search(evaluator, Budget(max_trials=20), seed=42)

    assert len(sr.history) == 20


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
    assert seen_embeddings == set(project.search_space.embedding.models), (
        f"Expected all three embeddings, saw {seen_embeddings}"
    )


def _discrete_project() -> ProjectConfig:
    """SearchSpace with DiscreteValues for all 5 fairness-critical numeric dims.

    Used to exercise the discrete-grid code path in both sample_random and
    sample_optuna (the helpers ``_sample_int`` / ``_suggest_int`` etc. and
    the per-trial filters for chunk_overlap < chunk_size and reranker_top_n
    <= top_k).
    """
    return ProjectConfig(
        search_space=SearchSpace(
            chunking=ChunkingSearchSpace(
                strategies=["recursive"],
                chunk_token_size=DiscreteValues(values=[256, 512]),
                chunk_token_overlap=DiscreteValues(values=[0, 64]),
            ),
            embedding=EmbeddingSearchSpace(models=["sentence-transformers/all-MiniLM-L6-v2"]),
            retrieval=RetrievalSearchSpace(
                index_types=[IndexType.VECTOR_ONLY],
                top_k=DiscreteValues(values=[3, 5, 10]),
                hybrid_alpha=DiscreteValues(values=[0.0, 0.5, 1.0]),
            ),
            reranker=RerankerSearchSpace(
                models=["none", "BAAI/bge-reranker-v2-m3"],
                top_n=DiscreteValues(values=[3, 5, 10]),
            ),
            query_expansion=QueryExpansionSearchSpace(strategies=["none"], models=["ollama/llama3.2"]),
            passage_compressor=PassageCompressorSearchSpace(strategies=["none"], models=["ollama/mistral"]),
            generator=GeneratorSearchSpace(models=["ollama/llama3.2", "ollama/mistral"]),
            temperature=NumericRange(min=1.0, max=1.0),
        ),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2",
            examiner_model="ollama/llama3.2",
            judge_model="ollama/llama3.2",
        ),
    )


def _is_int_in(value: int, allowed: list[float | int]) -> bool:
    return value in [int(v) for v in allowed]


@pytest.mark.asyncio
async def test_random_search_with_discrete_values_lands_in_grid() -> None:
    """Every sampled value for the 5 fairness-critical dims must come from
    its DiscreteValues option set (no continuous draws when the dim is
    discrete)."""
    project = _discrete_project()
    optimizer = RandomSearch(project=project)
    evaluator = _make_evaluator([0.5] * 20)

    sr = await optimizer.search(evaluator, Budget(max_trials=20), seed=42)

    for h in sr.history:
        assert _is_int_in(h.config["top_k"], [3, 5, 10])
        assert _is_int_in(h.config["chunk_token_size"], [256, 512])
        assert _is_int_in(h.config["chunk_token_overlap"], [0, 64])
        # reranker_top_n only meaningful when a real reranker is picked.
        if h.config["reranker"] != "none":
            assert _is_int_in(h.config["reranker_top_n"], [3, 5, 10])
            assert h.config["reranker_top_n"] <= h.config["top_k"]
        # chunk_token_overlap < chunk_token_size invariant.
        assert h.config["chunk_token_overlap"] < h.config["chunk_token_size"]


@pytest.mark.asyncio
async def test_random_search_with_discrete_values_picks_per_stage_llms() -> None:
    """generator_llm / expander_llm / compressor_llm draw from their own pools."""
    project = _discrete_project()
    # Force query_expansion + passage_compressor to enable expander_llm/compressor_llm.
    project.search_space.query_expansion.strategies = ["hyde"]
    project.search_space.passage_compressor.strategies = ["tree_summarize"]
    optimizer = RandomSearch(project=project)
    evaluator = _make_evaluator([0.5] * 10)

    sr = await optimizer.search(evaluator, Budget(max_trials=10), seed=42)

    for h in sr.history:
        assert h.config["generator_llm"] in {"ollama/llama3.2", "ollama/mistral"}
        assert h.config["expander_llm"] == "ollama/llama3.2"
        assert h.config["compressor_llm"] == "ollama/mistral"


@pytest.mark.asyncio
async def test_bayesian_with_discrete_values_lands_in_grid(tmp_path: Path) -> None:
    """Optuna's categorical suggest must produce values in the discrete sets,
    with snap-back for top_k-incompatible reranker_top_n picks."""
    project = _discrete_project()
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.5] * 12)

    sr = await optimizer.search(evaluator, Budget(max_trials=12), seed=42)

    for h in sr.history:
        assert _is_int_in(h.config["top_k"], [3, 5, 10])
        assert _is_int_in(h.config["chunk_token_size"], [256, 512])
        if h.config["reranker"] != "none":
            assert h.config["reranker_top_n"] <= h.config["top_k"]


@pytest.mark.asyncio
async def test_bayesian_reranker_top_n_lands_on_grid_and_respects_top_k(tmp_path: Path) -> None:
    """Optuna now uses dynamic int bounds + snap-to-grid for reranker_top_n
    (not categorical with snap-back). Every sampled value must (a) be in the
    DiscreteValues grid and (b) be <= top_k. This is the regression test for
    the migration off categorical snap-back.
    """
    project = _discrete_project()
    # Force the reranker to be active so reranker_top_n is meaningful.
    project.search_space.reranker.models = ["BAAI/bge-reranker-v2-m3"]
    optimizer = BayesianSearch(project=project, storage_dir=tmp_path)
    evaluator = _make_evaluator([0.5] * 15)

    sr = await optimizer.search(evaluator, Budget(max_trials=15), seed=42)

    for h in sr.history:
        assert _is_int_in(h.config["reranker_top_n"], [3, 5, 10])
        assert h.config["reranker_top_n"] <= h.config["top_k"], (
            f"reranker_top_n={h.config['reranker_top_n']} > top_k={h.config['top_k']} — "
            "dynamic-int-bounds branch should keep reranker_top_n within top_k"
        )


@pytest.mark.asyncio
async def test_random_resume_continues_from_last_trial(tmp_path: Path) -> None:
    """Half a run with storage on disk; restart with ``resume=True`` and the
    loop picks up at trial K+1 with full history merged."""
    project = _tiny_project()

    # First leg: 2 trials of a 4-trial budget; interrupt after 2 by capping
    # the evaluator's available scores.
    interrupting_evaluator = _make_evaluator([0.1, 0.2])
    optimizer_a = RandomSearch(project=project, storage_dir=tmp_path)

    # Use a smaller budget so the search returns cleanly after 2 trials.
    sr_a = await optimizer_a.search(interrupting_evaluator, Budget(max_trials=2), seed=42)
    assert len(sr_a.history) == 2
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "rng_state.pkl").exists()
    assert (tmp_path / "wall_clock.json").exists()

    # Second leg: resume up to 4 trials. The fresh evaluator is "trial 3+ only".
    resumed_evaluator = _make_evaluator([0.3, 0.4])
    optimizer_b = RandomSearch(project=project, storage_dir=tmp_path, resume=True)
    sr_b = await optimizer_b.search(resumed_evaluator, Budget(max_trials=4), seed=42)

    # History is the merge: trials 1-2 from leg A, trials 3-4 from leg B.
    assert len(sr_b.history) == 4
    scores = [h.score for h in sr_b.history]
    assert scores == [0.1, 0.2, 0.3, 0.4], scores
    # trial_usd_total accumulates across legs.
    assert sr_b.trial_usd_total == pytest.approx(4 * 0.001)
    # Wall clock accumulates across legs.
    assert sr_b.wall_clock_s > 0


@pytest.mark.asyncio
async def test_random_resume_same_rng_point_on_in_flight_interrupt(tmp_path: Path) -> None:
    """The RNG state is saved AFTER a trial completes, so an interrupted trial
    is fully discarded — restart re-draws the same config from the same RNG
    point as the original would have."""
    project = _tiny_project()

    # Run all 5 trials in one shot to capture the "ground truth" history.
    truth_optimizer = RandomSearch(project=project, storage_dir=tmp_path / "truth")
    truth_history = (await truth_optimizer.search(_make_evaluator([0.5] * 5), Budget(max_trials=5), seed=99)).history

    # Now simulate an interrupted+resumed run: 3 trials, then 2 more.
    partial_dir = tmp_path / "partial"
    part_a = await RandomSearch(project=project, storage_dir=partial_dir).search(
        _make_evaluator([0.5] * 3), Budget(max_trials=3), seed=99,
    )
    assert len(part_a.history) == 3

    part_b = await RandomSearch(project=project, storage_dir=partial_dir, resume=True).search(
        _make_evaluator([0.5] * 2), Budget(max_trials=5), seed=99,
    )
    assert len(part_b.history) == 5

    # Trial-by-trial configs from the interrupted+resumed run must equal those
    # from the un-interrupted ground-truth — that's the resume contract.
    assert [h.config for h in part_b.history] == [h.config for h in truth_history]


@pytest.mark.asyncio
async def test_bayesian_resume_continues_from_last_trial(tmp_path: Path) -> None:
    project = _tiny_project()

    sr_a = await BayesianSearch(project=project, storage_dir=tmp_path).search(
        _make_evaluator([0.1, 0.2]), Budget(max_trials=2), seed=42,
    )
    assert len(sr_a.history) == 2
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "optuna.db").exists()

    sr_b = await BayesianSearch(project=project, storage_dir=tmp_path, resume=True).search(
        _make_evaluator([0.3, 0.4]), Budget(max_trials=4), seed=42,
    )
    assert len(sr_b.history) == 4
    scores = [h.score for h in sr_b.history]
    assert scores == [0.1, 0.2, 0.3, 0.4], scores
    assert sr_b.trial_usd_total == pytest.approx(4 * 0.001)


@pytest.mark.asyncio
async def test_bayesian_does_not_wipe_prior_state_on_fresh_start(tmp_path: Path) -> None:
    """Wiping a non-empty storage dir is the bench-level ``--clean`` flag's
    job (``_clear_output_root_for``). The optimizer must NOT silently
    delete prior optuna.db / history.jsonl when constructed with
    ``resume=False`` — that would destroy user data on a partial-state
    restart where the user explicitly chose ``--no-clean``.
    """
    project = _tiny_project()

    await BayesianSearch(project=project, storage_dir=tmp_path).search(
        _make_evaluator([0.1, 0.2]), Budget(max_trials=2), seed=42,
    )
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "optuna.db").exists()

    # Construct a fresh BayesianSearch (resume=False) and confirm prior
    # files are still on disk. We don't run search again — that would
    # exercise the buggy ``--no-clean`` semantic (loop runs from 1, sqlite
    # still has prior trials), which is out of scope for this test.
    BayesianSearch(project=project, storage_dir=tmp_path)
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "optuna.db").exists()


@pytest.mark.asyncio
async def test_bayesian_resume_self_heals_missing_history_jsonl(tmp_path: Path) -> None:
    """A run started with a pre-resume version of the bench writes
    ``optuna.db`` + ``trial_cost_ledger.jsonl`` per trial but NOT
    ``history.jsonl`` (that was end-of-method-only). On ``--resume``, we
    reconstruct the missing history.jsonl from those two files so the
    user's prior work isn't lost."""
    project = _tiny_project()

    # Run 2 trials normally to populate optuna.db + history.jsonl.
    await BayesianSearch(project=project, storage_dir=tmp_path).search(
        _make_evaluator([0.4, 0.5]), Budget(max_trials=2), seed=42,
    )

    # Simulate the pre-resume layout: history.jsonl was never written.
    # ``trial_cost_ledger.jsonl`` would also not have been written by the
    # bare optimizer (it's written by the bench's _make_metered_evaluator
    # wrapper); synthesize a minimal one so the reconstruction can sum
    # eval_usd. Wall-clock file likewise didn't exist pre-resume.
    history_path = tmp_path / "history.jsonl"
    history_path.unlink()
    (tmp_path / "wall_clock.json").unlink()
    bucket = {
        "rag_eval": {
            "usd": 0.001, "prompt_tokens": 100, "completion_tokens": 10,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "embedding_input_tokens": 0, "n_calls": 1,
        }
    }
    import json as _json
    (tmp_path / "trial_cost_ledger.jsonl").write_text(
        _json.dumps({"trial_number": 1, "buckets": bucket}) + "\n"
        + _json.dumps({"trial_number": 2, "buckets": bucket}) + "\n",
        encoding="utf-8",
    )
    assert (tmp_path / "optuna.db").exists()
    assert not history_path.exists()

    # Resume: self-heal kicks in, reconstructs history.jsonl, continues to 4.
    sr = await BayesianSearch(project=project, storage_dir=tmp_path, resume=True).search(
        _make_evaluator([0.6, 0.7]), Budget(max_trials=4), seed=42,
    )
    assert len(sr.history) == 4
    # First two scores came from optuna.db's stored values.
    assert sr.history[0].score == 0.4
    assert sr.history[1].score == 0.5
    # New trials run as normal.
    assert sr.history[2].score == 0.6
    assert sr.history[3].score == 0.7
    # Reconstructed configs are non-empty (sample_optuna replay worked).
    assert sr.history[0].config["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    # Cost from the synthesized ledger flows through.
    assert sr.history[0].eval_usd == pytest.approx(0.001)
    # The self-heal also persisted history.jsonl on disk.
    assert history_path.exists()
