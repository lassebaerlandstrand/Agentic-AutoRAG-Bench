"""Unit tests for the shared bench types."""

from __future__ import annotations

from types import SimpleNamespace

from agentic_autorag_bench.types import Budget, HistoryEntry, SearchResult, TrialResult


def test_budget_defaults_unbounded() -> None:
    b = Budget()
    assert b.max_trials is None
    assert b.max_wall_clock_s is None


def test_trial_result_from_exam_result() -> None:
    exam = SimpleNamespace(
        answer_accuracy=0.80,
        mean_retrieval_quality=0.62,
        mean_em=0.70,
        mean_f1=0.78,
        total_llm_cost_usd=0.0123,
    )
    tr = TrialResult.from_exam_result(exam)
    assert tr.answer_accuracy == 0.80
    assert tr.metrics["answer_accuracy"] == 0.80
    assert tr.metrics["mean_em"] == 0.70
    assert tr.eval_usd == 0.0123


def test_trial_result_handles_missing_cost() -> None:
    exam = SimpleNamespace(
        answer_accuracy=0.5,
        mean_retrieval_quality=0.5,
        mean_em=0.5,
        mean_f1=0.5,
    )
    tr = TrialResult.from_exam_result(exam)
    assert tr.eval_usd == 0.0


def test_search_result_serializes_round_trip() -> None:
    sr = SearchResult(
        method="random",
        seed=1,
        deterministic=False,
        best_config={"top_k": 5},
        history=[
            HistoryEntry(trial_number=1, config={"top_k": 3}, answer_accuracy=0.4, metrics={"em": 0.3}, eval_usd=0.01),
            HistoryEntry(trial_number=2, config={"top_k": 5}, answer_accuracy=0.6, metrics={"em": 0.5}, eval_usd=0.01),
        ],
        optimizer_usd=0.0,
        trial_usd_total=0.02,
        wall_clock_s=12.5,
    )
    d = sr.to_dict()
    assert d["method"] == "random"
    assert d["seed"] == 1
    assert len(d["history"]) == 2
    assert d["history"][0]["answer_accuracy"] == 0.4
    assert d["trial_usd_total"] == 0.02
