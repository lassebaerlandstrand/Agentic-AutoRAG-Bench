"""End-to-end checks for the expanded HotpotQA paper search space.

The paper config (``configs/hotpot_paper_project.yaml``) was widened on
2026-05-27 after the original Marker-Inc AutoRAG baseline was dropped — see
``agentic_autorag_bench/_deprecated/README.md`` for context. These tests
guard the wider config against three classes of regression:

  1. The YAML still parses against the framework's ``ProjectConfig`` schema
     (catches schema drift in either repo).
  2. Random and Optuna TPE samplers both fan out over the new search space
     without violating any of the framework's per-trial validators —
     including conditional dims (``hybrid_alpha`` gated on
     ``bm25_vector_fusion``, ``reranker_top_n <= top_k``, ``overlap <
     chunk_token_size``, expander/compressor LLMs gated by their strategy).
  3. Reasoning is toggled per-trial only for generators that actually
     support it (else stays False), and ``reasoning_effort`` is treated as
     a per-run constant, not a per-trial axis.
"""

from __future__ import annotations

import random as pyrand
from pathlib import Path

import optuna
import pytest
import yaml
from agentic_autorag.config.models import IndexType, ProjectConfig

from agentic_autorag_bench.methods._sampler import sample_optuna, sample_random

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hotpot_paper_project.yaml"


def _load_project() -> ProjectConfig:
    raw = yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8"))
    return ProjectConfig(**raw)


def test_project_yaml_parses() -> None:
    project = _load_project()
    ss = project.search_space
    assert len(ss.embedding.models) >= 10
    assert len(ss.generator.models) >= 30
    assert ss.generator.reasoning is True
    assert ss.generator.reasoning_effort == "medium"
    assert "hyde" in ss.query_expansion.strategies
    assert "mixedbread-ai/mxbai-rerank-xsmall-v1" in ss.reranker.models
    assert ss.chunking.strategies == ["recursive", "fixed"]
    assert ss.retrieval.long_context_reorder == [False, True]


def test_chunk_dims_remain_discrete() -> None:
    """Chunk size + overlap enter the on-disk chunking cache key; a
    continuous NumericRange would create one cache entry per unique integer
    value and explode storage. Discrete grids are mandatory here."""
    project = _load_project()
    ss = project.search_space
    assert hasattr(ss.chunking.chunk_token_size, "values"), "chunk_token_size must be DiscreteValues (cache key)"
    assert hasattr(ss.chunking.chunk_token_overlap, "values"), "chunk_token_overlap must be DiscreteValues (cache key)"


def test_random_sampler_500x_zero_violations() -> None:
    """Every random sample must pass ``project.validate_trial``. Failures
    indicate the sampler missed a conditional gate."""
    project = _load_project()
    rng = pyrand.Random(0)
    n = 500
    for _ in range(n):
        cfg = sample_random(rng, project.search_space)
        assert not project.validate_trial(cfg)


def test_optuna_sampler_100x_zero_violations() -> None:
    """Same guarantee for the MO-TPE path."""
    project = _load_project()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    n = 100
    for i in range(n):
        trial = study.ask()
        cfg = sample_optuna(trial, project.search_space)
        viols = project.validate_trial(cfg)
        assert not viols, f"trial {i} produced {viols}"
        study.tell(trial, 0.5)


def test_random_sampler_covers_every_discrete_choice() -> None:
    """500 random samples should hit every entry in the small categorical
    dims. If this regresses, the sampler probably grew a hidden filter."""
    project = _load_project()
    ss = project.search_space
    rng = pyrand.Random(42)
    seen_strategies: set[str] = set()
    seen_qe: set[str] = set()
    seen_rerankers: set[str] = set()
    seen_index_types: set[IndexType] = set()
    for _ in range(500):
        cfg = sample_random(rng, ss)
        seen_strategies.add(cfg.chunking_strategy)
        seen_qe.add(cfg.query_expansion)
        seen_rerankers.add(cfg.reranker)
        seen_index_types.add(cfg.index_type)
    assert seen_strategies == set(ss.chunking.strategies)
    assert seen_qe == set(ss.query_expansion.strategies)
    assert seen_rerankers == set(ss.reranker.models)
    assert seen_index_types == set(ss.retrieval.index_types)


def test_reasoning_only_toggled_for_capable_generators() -> None:
    """When the search space declares ``reasoning: true``, the sampler may
    set reasoning=True only when LiteLLM reports the generator supports it.
    All other trials must be reasoning=False, independent of the per-trial
    coin flip — otherwise we'd send unsupported params downstream."""
    project = _load_project()
    ss = project.search_space
    rng = pyrand.Random(123)
    for _ in range(200):
        cfg = sample_random(rng, ss)
        if cfg.reasoning:
            assert ss.is_reasoning_allowed(cfg.generator_llm), (
                f"reasoning=True but {cfg.generator_llm!r} is not reasoning-capable"
            )


def test_continuous_dims_actually_continuous() -> None:
    """``top_k`` and ``hybrid_alpha`` should produce non-grid values when
    sampled, exercising MO-TPE's continuous path. (Discrete-only would
    silently revert the sampler to a categorical surrogate.)"""
    project = _load_project()
    ss = project.search_space
    rng = pyrand.Random(7)
    hybrid_alphas: list[float] = []
    top_ks: list[int] = []
    for _ in range(50):
        cfg = sample_random(rng, ss)
        if cfg.index_type == IndexType.HYBRID_BM25_VECTOR and cfg.bm25_vector_fusion == "alpha":
            hybrid_alphas.append(cfg.hybrid_alpha)
        top_ks.append(cfg.top_k)
    # NumericRange-sampled alphas should not all snap to {0, 0.5, 1.0}
    if hybrid_alphas:
        coarse = {round(a, 1) for a in hybrid_alphas}
        assert len(coarse) > 3, f"hybrid_alpha samples look discrete: {sorted(hybrid_alphas)[:10]}"
    # top_k spans 3..20 — should observe enough variety
    assert len(set(top_ks)) > 5


@pytest.mark.parametrize("n", [1, 5, 50])
def test_optuna_persists_through_validator_rejections(n: int) -> None:
    """Even if every n-th candidate is rejected, TPE should keep sampling
    without raising — the candidate pool is large enough that infeasible
    points are rare and the sampler should recover quickly."""
    project = _load_project()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=n),
    )
    for _ in range(n * 5):
        trial = study.ask()
        cfg = sample_optuna(trial, project.search_space)
        viols = project.validate_trial(cfg)
        if viols:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, 0.5)
