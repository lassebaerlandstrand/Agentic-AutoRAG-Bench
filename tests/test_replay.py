"""Tests for the hold-out replay command and mean±SD aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import yaml

from agentic_autorag_bench._holdout_registry import apply_union_exclusion
from agentic_autorag_bench.analyze import (
    MethodResult,
    aggregate_by_method,
    load_results,
    mean_sd,
)
from agentic_autorag_bench.replay import (
    _discover_targets,
    _existing_replay_indices,
)

# Minimal valid TrialConfig — replay loads best_config.yaml via TrialConfig(**),
# so the test fixture must be a parseable config. Mirrors the smallest viable
# config from results_hotpot/random/seed_1/best_config.yaml.
_MINIMAL_TRIAL_CONFIG = {
    "chunking_strategy": "fixed",
    "chunk_token_size": 256,
    "chunk_token_overlap": 0,
    "embedding_model": "Snowflake/snowflake-arctic-embed-xs",
    "index_type": "vector_only",
    "top_k": 5,
    "long_context_reorder": False,
    "passage_compressor": "none",
    "reranker": "none",
    "query_expansion": "none",
    "generator_llm": "bedrock/test-model",
    "temperature": 1.0,
    "reasoning": False,
}

# -------------------------------------------------------------- mean_sd helper


def test_mean_sd_returns_zero_width_for_n_less_than_2() -> None:
    """N=1 has no defined SD; chart writers must see a zero-height error bar."""
    mean, lo, hi = mean_sd(np.array([0.7]))
    assert mean == lo == hi == 0.7


def test_mean_sd_empty_array_returns_zeros() -> None:
    assert mean_sd(np.array([])) == (0.0, 0.0, 0.0)


def test_mean_sd_uses_sample_sd_ddof_1() -> None:
    """Sample SD (ddof=1) is the small-N convention; assert math directly."""
    values = np.array([0.5, 0.6, 0.7])
    mean, lo, hi = mean_sd(values)
    expected_sd = float(np.std(values, ddof=1))
    assert abs(mean - 0.6) < 1e-9
    assert abs((hi - mean) - expected_sd) < 1e-9
    assert abs((mean - lo) - expected_sd) < 1e-9


def test_mean_sd_skips_nan() -> None:
    """A run where every judge call failed produces NaN; mean_sd must skip it."""
    values = np.array([0.5, 0.7, np.nan])
    mean, lo, hi = mean_sd(values)
    assert abs(mean - 0.6) < 1e-9
    assert hi > mean > lo


# ----------------------------------------------- _existing_replay_indices


def test_existing_replay_indices_counts_primary_as_run_1(tmp_path: Path) -> None:
    seed_dir = tmp_path / "agentic_score" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "benchmark_results.json").write_text("{}")
    assert _existing_replay_indices(seed_dir) == {1}


def test_existing_replay_indices_picks_up_replays(tmp_path: Path) -> None:
    seed_dir = tmp_path / "agentic_score" / "seed_1"
    (seed_dir / "holdout_replays").mkdir(parents=True)
    (seed_dir / "benchmark_results.json").write_text("{}")
    (seed_dir / "holdout_replays" / "run_002.json").write_text("{}")
    (seed_dir / "holdout_replays" / "run_003.json").write_text("{}")
    assert _existing_replay_indices(seed_dir) == {1, 2, 3}


def test_existing_replay_indices_handles_missing_primary(tmp_path: Path) -> None:
    """A pre-search dir (no benchmark_results.json yet) reports an empty set."""
    seed_dir = tmp_path / "agentic_score" / "seed_1"
    seed_dir.mkdir(parents=True)
    assert _existing_replay_indices(seed_dir) == set()


# ----------------------------------------------- _discover_targets


def _seed_dir_with_primary(root: Path, method: str, seed: int) -> Path:
    """Materialise a (method, seed) dir with best_config.yaml + run-1 results."""
    seed_dir = root / method / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "best_config.yaml").write_text(yaml.safe_dump(_MINIMAL_TRIAL_CONFIG))
    (seed_dir / "benchmark_results.json").write_text(json.dumps({"per_question": []}))
    return seed_dir


def test_discover_targets_skips_dirs_without_best_config(tmp_path: Path) -> None:
    _seed_dir_with_primary(tmp_path, "agentic_score", 1)
    # method without best_config (e.g., search crashed pre-completion)
    incomplete = tmp_path / "agentic_cost" / "seed_1"
    incomplete.mkdir(parents=True)
    (incomplete / "benchmark_results.json").write_text("{}")
    targets = _discover_targets(tmp_path, methods_filter=None, include_checkpoints=True)
    assert len(targets) == 1
    assert targets[0].parent.name == "agentic_score"


def test_discover_targets_includes_checkpoints_by_default(tmp_path: Path) -> None:
    _seed_dir_with_primary(tmp_path, "agentic_score", 1)
    _seed_dir_with_primary(tmp_path, "agentic_score@10", 1)
    _seed_dir_with_primary(tmp_path, "agentic_score@20", 1)
    targets = _discover_targets(tmp_path, methods_filter=None, include_checkpoints=True)
    assert {t.parent.name for t in targets} == {
        "agentic_score", "agentic_score@10", "agentic_score@20",
    }


def test_discover_targets_excludes_checkpoints_when_requested(tmp_path: Path) -> None:
    _seed_dir_with_primary(tmp_path, "agentic_score", 1)
    _seed_dir_with_primary(tmp_path, "agentic_score@10", 1)
    targets = _discover_targets(tmp_path, methods_filter=None, include_checkpoints=False)
    assert {t.parent.name for t in targets} == {"agentic_score"}


def test_discover_targets_methods_filter_matches_base(tmp_path: Path) -> None:
    """``--methods agentic_score`` includes both ``agentic_score/`` and ``agentic_score@10/``
    so a user asking "top up the agentic-score family" gets the whole family.
    """
    _seed_dir_with_primary(tmp_path, "agentic_score", 1)
    _seed_dir_with_primary(tmp_path, "agentic_score@10", 1)
    _seed_dir_with_primary(tmp_path, "bayesian", 1)
    targets = _discover_targets(
        tmp_path, methods_filter={"agentic_score"}, include_checkpoints=True,
    )
    assert {t.parent.name for t in targets} == {"agentic_score", "agentic_score@10"}


def test_discover_targets_skips_non_method_dirs(tmp_path: Path) -> None:
    """``figures/``, ``.shared_cache/`` etc. are infrastructure, not method dirs."""
    _seed_dir_with_primary(tmp_path, "agentic_score", 1)
    (tmp_path / "figures").mkdir()
    (tmp_path / ".shared_cache").mkdir()
    targets = _discover_targets(tmp_path, methods_filter=None, include_checkpoints=True)
    assert len(targets) == 1


# -------------------------------------------------- union exclusion covers replays


def _write_benchmark(path: Path, ids: list[str], filtered_ids: list[str]) -> None:
    """Write a minimal benchmark_results.json shape, marking ``filtered_ids`` rows
    with the framework's CONTENT_FILTER sentinel so ``apply_union_exclusion``
    picks them up.
    """
    from agentic_autorag.examiner._errors import CONTENT_FILTER_SENTINEL
    per_q: list[dict[str, Any]] = []
    for qid in ids:
        row = {"id": qid, "em": 1.0, "f1": 1.0, "judge": 1, "retrieved_doc_ids": []}
        if qid in filtered_ids:
            row["error"] = CONTENT_FILTER_SENTINEL
            row["em"] = 0.0
            row["f1"] = 0.0
            row["judge"] = None
        per_q.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_total": len(ids), "n_valid": len(ids),
        "em": 0.5, "f1": 0.5, "llm_judge_accuracy": 0.5,
        "per_question": per_q,
        "judge_model": "test-judge",
    }))


def test_union_exclusion_includes_content_filter_from_replays(tmp_path: Path) -> None:
    """CONTENT_FILTER hit in a replay must propagate to every benchmark file's
    excluded_question_ids, so all methods score the same denominator."""
    seed_dir = tmp_path / "agentic_score" / "seed_1"
    primary = seed_dir / "benchmark_results.json"
    replay = seed_dir / "holdout_replays" / "run_002.json"
    # Run 1: clean. Run 2: surfaces a filter on q_filtered.
    _write_benchmark(primary, ["q_a", "q_b", "q_filtered"], filtered_ids=[])
    _write_benchmark(replay, ["q_a", "q_b", "q_filtered"], filtered_ids=["q_filtered"])
    # A second method's benchmark with no filter — verifies union propagates.
    other_primary = tmp_path / "random" / "seed_1" / "benchmark_results.json"
    _write_benchmark(other_primary, ["q_a", "q_b", "q_filtered"], filtered_ids=[])

    registry = apply_union_exclusion(tmp_path)
    assert "q_filtered" in registry["excluded_question_ids"]

    # Every benchmark file got rewritten with the union id in excluded_question_ids.
    for path in (primary, replay, other_primary):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "q_filtered" in data["excluded_question_ids"], (
            f"{path.relative_to(tmp_path)} missed the union id"
        )

    # by_run reports the replay file with a richer key that disambiguates run.
    assert any("/run_002" in k for k in registry["by_run"]), (
        f"replay key missing from by_run: {list(registry['by_run'].keys())}"
    )


# ------------------------------------------------------- aggregation across replays


def test_aggregate_uses_mean_sd_across_replays() -> None:
    """3 replays with known per-run means → triple is (mean, mean-sd, mean+sd)."""
    def _bm(em: float, judge_correct: int) -> dict:
        return {
            "per_question": [{"id": "q1", "em": em, "f1": em, "judge": judge_correct}],
            "excluded_question_ids": [],
        }
    # Three runs with EM = 0.5, 0.6, 0.7 → mean=0.6, sd(ddof=1)=0.1.
    result = MethodResult(
        method="agentic_score",
        seed=1,
        benchmarks=[_bm(0.5, 1), _bm(0.6, 1), _bm(0.7, 0)],
        optimizer_meta={},
        history=[],
    )
    stats = aggregate_by_method([result])
    em_mean, em_lo, em_hi = stats["agentic_score"]["em"]
    expected_sd = float(np.std([0.5, 0.6, 0.7], ddof=1))
    assert abs(em_mean - 0.6) < 1e-9
    assert abs((em_hi - em_mean) - expected_sd) < 1e-9
    assert stats["agentic_score"]["n_runs_total"] == 3
    assert stats["agentic_score"]["n_runs_per_seed"] == [3]


def test_aggregate_falls_back_to_zero_width_for_n_1() -> None:
    """N=1 (no replays yet) yields lo=hi=mean so the chart draws a flat bar."""
    bm = {
        "per_question": [{"id": "q1", "em": 0.5, "f1": 0.5, "judge": 1}],
        "excluded_question_ids": [],
    }
    result = MethodResult(
        method="random", seed=1, benchmarks=[bm], optimizer_meta={}, history=[],
    )
    stats = aggregate_by_method([result])
    em_mean, em_lo, em_hi = stats["random"]["em"]
    assert em_mean == em_lo == em_hi == 0.5
    assert stats["random"]["n_runs_total"] == 1


def test_load_results_picks_up_holdout_replays(tmp_path: Path) -> None:
    """``load_results`` must include every ``holdout_replays/run_NNN.json``
    in the returned ``MethodResult.benchmarks`` list, in numbered order."""
    seed_dir = tmp_path / "agentic_score" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "benchmark_results.json").write_text(json.dumps({
        "per_question": [{"id": "q1", "em": 0.5, "f1": 0.5, "judge": 1}],
    }))
    (seed_dir / "holdout_replays").mkdir()
    (seed_dir / "holdout_replays" / "run_002.json").write_text(json.dumps({
        "per_question": [{"id": "q1", "em": 0.6, "f1": 0.6, "judge": 1}],
    }))
    (seed_dir / "holdout_replays" / "run_003.json").write_text(json.dumps({
        "per_question": [{"id": "q1", "em": 0.7, "f1": 0.7, "judge": 0}],
    }))
    results = load_results(tmp_path)
    assert len(results) == 1
    assert len(results[0].benchmarks) == 3
    # Order: primary first, then replays by numeric ascending.
    em_per_run, _, _ = results[0].per_run_means()
    assert list(em_per_run) == [0.5, 0.6, 0.7]


# -------------------------------------------------- replay command idempotency


@pytest.mark.asyncio
async def test_replay_holdout_skips_existing_runs(tmp_path: Path) -> None:
    """Pre-populate run_002.json; assert replay only fills the missing run_003."""
    from agentic_autorag_bench.replay import replay_holdout

    # Materialise a results tree.
    output_root = tmp_path / "results"
    seed_dir = _seed_dir_with_primary(output_root, "agentic_score", 1)
    (seed_dir / "holdout_replays").mkdir()
    # Already have run 2 from a prior partial replay; only run 3 is missing.
    _write_benchmark(seed_dir / "holdout_replays" / "run_002.json", ["q1"], filtered_ids=[])

    # Materialise a config that BenchConfig.load can read.
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(yaml.safe_dump({"meta": {"output_dir": str(tmp_path / "_cache")}}))
    config_yaml = tmp_path / "bench_config.yaml"
    config_yaml.write_text(yaml.safe_dump({
        "project_config": str(project_yaml),
        "methods": ["agentic_score"],
        "seeds": [1],
        "budget": {"max_trials": 1},
        "benchmark": {
            "name": "hotpot_qa", "split": "validation",
            "sample_size": 100, "prep_seed": 42,
            "output_dir": str(tmp_path / "_data"),
        },
        "hold_out": {"limit": 10, "judge_model": "test", "concurrency": 1},
        "output_root": str(output_root),
    }))

    # Mock the evaluator + the post-pass functions so the test does no LLM work.
    eval_calls: list[Path] = []
    async def _fake_evaluate(*, output_path: Path, **kwargs) -> dict:
        eval_calls.append(Path(output_path))
        _write_benchmark(Path(output_path), ["q1"], filtered_ids=[])
        return {}

    with (
        patch(
            "agentic_autorag_bench.replay.BenchmarkRunner.evaluate",
            new=AsyncMock(side_effect=_fake_evaluate),
        ),
        patch("agentic_autorag_bench.replay.BenchmarkRunner.prepare"),
        patch("agentic_autorag_bench.replay.make_matrix_figures"),
        patch("agentic_autorag_bench.replay.configure_litellm_runtime"),
    ):
        await replay_holdout(config_yaml, n_runs=3)

    # Only run_003 should have been written; run_002 was already there.
    assert eval_calls == [seed_dir / "holdout_replays" / "run_003.json"]
    assert (seed_dir / "holdout_replays" / "run_003.json").exists()


@pytest.mark.asyncio
async def test_replay_holdout_is_idempotent_when_full(tmp_path: Path) -> None:
    """Running with no missing runs triggers zero evals (only union + figures)."""
    from agentic_autorag_bench.replay import replay_holdout

    output_root = tmp_path / "results"
    seed_dir = _seed_dir_with_primary(output_root, "agentic_score", 1)
    (seed_dir / "holdout_replays").mkdir()
    _write_benchmark(seed_dir / "holdout_replays" / "run_002.json", ["q1"], filtered_ids=[])
    _write_benchmark(seed_dir / "holdout_replays" / "run_003.json", ["q1"], filtered_ids=[])

    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(yaml.safe_dump({"meta": {"output_dir": str(tmp_path / "_cache")}}))
    config_yaml = tmp_path / "bench_config.yaml"
    config_yaml.write_text(yaml.safe_dump({
        "project_config": str(project_yaml),
        "methods": ["agentic_score"],
        "seeds": [1],
        "budget": {"max_trials": 1},
        "benchmark": {
            "name": "hotpot_qa", "split": "validation",
            "sample_size": 100, "prep_seed": 42,
            "output_dir": str(tmp_path / "_data"),
        },
        "hold_out": {"limit": 10, "judge_model": "test", "concurrency": 1},
        "output_root": str(output_root),
    }))

    eval_calls: list[Path] = []
    async def _count_call(*, output_path: Path, **kwargs) -> dict:
        eval_calls.append(Path(output_path))
        return {}

    with (
        patch(
            "agentic_autorag_bench.replay.BenchmarkRunner.evaluate",
            new=AsyncMock(side_effect=_count_call),
        ),
        patch("agentic_autorag_bench.replay.BenchmarkRunner.prepare"),
        patch("agentic_autorag_bench.replay.make_matrix_figures"),
        patch("agentic_autorag_bench.replay.configure_litellm_runtime"),
    ):
        await replay_holdout(config_yaml, n_runs=3)

    assert eval_calls == [], "replay_holdout should not re-run already-complete dirs"
