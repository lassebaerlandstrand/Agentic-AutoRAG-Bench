"""Tests for the checkpoint-evaluator @k mechanism.

After a method's full-budget search finishes, ``_evaluate_checkpoints``
slices ``history[:k]`` for each declared checkpoint, finds the best
trial in that prefix, writes a sibling ``<method>@<k>/seed_<n>/`` result
directory, and re-runs held-out scoring on that prefix's best config.

We exercise the slice + write path with a stubbed
``BenchmarkRunner.evaluate`` so the test stays offline. Held-out cost
is the most expensive part of the real flow; stubbing it keeps the test
fast and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentic_autorag_bench.benchmarks.runner import BenchmarkRunner
from agentic_autorag_bench.run import BenchConfig, BenchmarkSpec, _evaluate_checkpoints
from agentic_autorag_bench.types import HistoryEntry, SearchResult


def _minimal_trial_config_dict() -> dict:
    """A dict that round-trips through ``TrialConfig(**d)`` without erroring.

    Mirrors what ``config.to_prompt_dump(include_graph=False)`` produces:
    every required field present, no graph fields.
    """
    return {
        "chunking_strategy": "recursive",
        "chunk_token_size": 512,
        "chunk_token_overlap": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "index_type": "vector_only",
        "top_k": 5,
        "hybrid_alpha": 0.5,
        "bm25_vector_fusion": "alpha",
        "long_context_reorder": False,
        "passage_compressor": "none",
        "reranker": "none",
        "reranker_top_n": 3,
        "query_expansion": "none",
        "generator_llm": "azure/gpt-4o-mini",
        "compressor_llm": None,
        "expander_llm": None,
        "temperature": 1.0,
        "reasoning": False,
    }


def _bench(tmp_path: Path, checkpoints: dict[str, list[int]]) -> BenchConfig:
    return BenchConfig(
        project_config_path=tmp_path / "project.yaml",
        methods=["agentic_score"],
        seeds=[0],
        max_trials=40,
        benchmark=BenchmarkSpec(
            name="hotpot_qa",
            split="validation",
            sample_size=10,
            prep_seed=42,
            output_dir=tmp_path / "bench_data",
        ),
        hold_out_limit=10,
        hold_out_judge_model=None,
        hold_out_concurrency=1,
        output_root=tmp_path / "results",
        checkpoints=checkpoints,
    )


def _search_result(n_trials: int, peak_at: int) -> SearchResult:
    """Synthesize a SearchResult with ``n_trials`` history entries; the
    best score sits at 1-indexed ``peak_at``."""
    base = _minimal_trial_config_dict()
    history = [
        HistoryEntry(
            trial_number=i + 1,
            config=dict(base),
            score=0.5 + (0.10 if i + 1 == peak_at else 0.0) + 0.001 * i,
            metrics={"answer_accuracy": 0.5},
            eval_usd=0.10,
            prompt_tokens=100,
            completion_tokens=20,
            embedding_tokens=0,
        )
        for i in range(n_trials)
    ]
    return SearchResult(
        method="agentic_score",
        seed=0,
        deterministic=False,
        best_config=history[-1].config,
        history=history,
        optimizer_usd=2.0,
        trial_usd_total=sum(h.eval_usd for h in history),
        wall_clock_s=400.0,
    )


async def test_writes_checkpoint_dirs_with_correct_best(tmp_path: Path) -> None:
    """For each k in checkpoints, the @k directory is written with the
    best config from ``history[:k]``."""
    bench = _bench(tmp_path, {"agentic_score": [10, 20]})
    sr = _search_result(n_trials=40, peak_at=15)  # global max within first 20 trials at #15

    benchmark = BenchmarkRunner(
        name="hotpot_qa",
        output_dir=tmp_path / "bench_data",
        split="validation",
        sample_size=10,
        seed=42,
    )
    benchmark.evaluate = AsyncMock(return_value={"answer_accuracy": 0.7})

    await _evaluate_checkpoints(
        sr,
        method_name="agentic_score",
        seed=0,
        bench=bench,
        benchmark=benchmark,
    )
    run_root = bench.output_root

    # @10: best is somewhere in trials 1-10 (deterministic by score formula)
    ck10 = run_root / "agentic_score@10" / "seed_0"
    assert ck10.is_dir()
    meta10 = json.loads((ck10 / "optimizer_meta.json").read_text())
    assert meta10["method"] == "agentic_score@10"
    assert meta10["n_trials_completed"] == 10
    assert meta10["extras"]["checkpoint_at"] == 10
    assert meta10["extras"]["parent_method"] == "agentic_score"

    # @20: should peak at trial 15 (the synthesized global max within first 20)
    ck20 = run_root / "agentic_score@20" / "seed_0"
    assert ck20.is_dir()
    sr20 = json.loads((ck20 / "search_result.json").read_text())
    assert sr20["method"] == "agentic_score@20"
    assert len(sr20["history"]) == 20
    best20 = max(sr20["history"], key=lambda h: h["score"])
    assert best20["trial_number"] == 15

    # Held-out evaluator was invoked once per checkpoint, each with
    # output_path under its own @k dir.
    assert benchmark.evaluate.await_count == 2
    output_paths = {
        call.kwargs["output_path"]
        for call in benchmark.evaluate.await_args_list
    }
    assert ck10 / "benchmark_results.json" in output_paths
    assert ck20 / "benchmark_results.json" in output_paths


async def test_skips_checkpoints_at_or_above_max_trials(tmp_path: Path) -> None:
    """A checkpoint equal to (or exceeding) the actual trial count is the
    natural full-budget run and is already written elsewhere — no @k dir
    should be produced."""
    bench = _bench(tmp_path, {"agentic_score": [10, 40, 60]})
    sr = _search_result(n_trials=40, peak_at=5)

    benchmark = BenchmarkRunner(
        name="hotpot_qa",
        output_dir=tmp_path / "bench_data",
        split="validation",
        sample_size=10,
        seed=42,
    )
    benchmark.evaluate = AsyncMock(return_value={})

    await _evaluate_checkpoints(
        sr,
        method_name="agentic_score",
        seed=0,
        bench=bench,
        benchmark=benchmark,
    )
    run_root = bench.output_root

    assert (run_root / "agentic_score@10").is_dir()
    assert not (run_root / "agentic_score@40").exists()
    assert not (run_root / "agentic_score@60").exists()
    assert benchmark.evaluate.await_count == 1


async def test_no_checkpoints_is_noop(tmp_path: Path) -> None:
    """A method without a checkpoints entry produces no @k dirs and makes
    no held-out calls."""
    bench = _bench(tmp_path, {})
    sr = _search_result(n_trials=10, peak_at=3)

    benchmark = BenchmarkRunner(
        name="hotpot_qa",
        output_dir=tmp_path / "bench_data",
        split="validation",
        sample_size=10,
        seed=42,
    )
    benchmark.evaluate = AsyncMock(return_value={})

    await _evaluate_checkpoints(
        sr,
        method_name="agentic_score",
        seed=0,
        bench=bench,
        benchmark=benchmark,
    )

    if bench.output_root.exists():
        assert [c.name for c in bench.output_root.iterdir()] == []
    benchmark.evaluate.assert_not_awaited()


async def test_cumulative_cost_reflects_only_sliced_trials(tmp_path: Path) -> None:
    """``trial_usd_total`` for @k is the sum of trial costs over the first
    k trials, NOT the parent's full total."""
    bench = _bench(tmp_path, {"agentic_score": [5]})
    sr = _search_result(n_trials=40, peak_at=1)

    benchmark = BenchmarkRunner(
        name="hotpot_qa",
        output_dir=tmp_path / "bench_data",
        split="validation",
        sample_size=10,
        seed=42,
    )
    benchmark.evaluate = AsyncMock(return_value={})

    await _evaluate_checkpoints(
        sr,
        method_name="agentic_score",
        seed=0,
        bench=bench,
        benchmark=benchmark,
    )
    run_root = bench.output_root

    meta5 = json.loads(
        (run_root / "agentic_score@5" / "seed_0" / "optimizer_meta.json").read_text()
    )
    # 5 trials × $0.10/trial = $0.50 — exact match
    assert meta5["trial_usd_total"] == pytest.approx(0.5)
    # Prorated optimizer cost: 5/40 * 2.0 = 0.25
    assert meta5["optimizer_usd"] == pytest.approx(0.25)
