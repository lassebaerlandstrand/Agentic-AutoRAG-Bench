"""Cross-method content-filter exclusion."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_autorag_bench._holdout_registry import apply_union_exclusion


def _write_run(root: Path, method: str, seed: str, rows: list[dict], judge_model: str = "judge") -> Path:
    seed_dir = root / method / seed
    seed_dir.mkdir(parents=True, exist_ok=True)
    f = seed_dir / "benchmark_results.json"
    f.write_text(
        json.dumps(
            {
                "benchmark": "hotpot",
                "n_total": len(rows),
                "n_valid": sum(1 for r in rows if r.get("error") is None),
                "em": 0.0,
                "f1": 0.0,
                "llm_judge_accuracy": None,
                "per_question": rows,
                "judge_model": judge_model,
            }
        )
    )
    return f


def _row(qid: str, em: float = 1.0, f1: float = 1.0, judge: int | None = 1, error: str | None = None) -> dict:
    return {
        "id": qid,
        "em": em,
        "f1": f1,
        "judge": judge,
        "retrieved_doc_ids": [],
        "supporting_doc_ids": [],
        "error": error,
    }


def test_no_filter_rejections_writes_empty_registry(tmp_path: Path) -> None:
    _write_run(tmp_path, "agentic", "seed_42", [_row("q1"), _row("q2")])
    _write_run(tmp_path, "random", "seed_42", [_row("q1"), _row("q2")])

    registry = apply_union_exclusion(tmp_path)

    assert registry["excluded_question_ids"] == []
    assert (tmp_path / "filtered_questions.json").exists()


def test_union_drops_question_from_every_method_denominator(tmp_path: Path) -> None:
    """If method A flagged q5 as content-filtered, method B (which scored q5
    fine) must also drop q5 — otherwise A and B have different denominators."""
    a_rows = [_row("q1"), _row("q2"), _row("q5", em=0.0, f1=0.0, judge=None, error="CONTENT_FILTER")]
    b_rows = [_row("q1"), _row("q2"), _row("q5", em=1.0, f1=1.0, judge=1)]
    _write_run(tmp_path, "agentic", "seed_42", a_rows)
    _write_run(tmp_path, "random", "seed_42", b_rows)

    registry = apply_union_exclusion(tmp_path)

    assert registry["excluded_question_ids"] == ["q5"]
    assert "agentic/seed_42" in registry["by_run"]
    assert "random/seed_42" not in registry["by_run"]

    a = json.loads((tmp_path / "agentic/seed_42/benchmark_results.json").read_text())
    b = json.loads((tmp_path / "random/seed_42/benchmark_results.json").read_text())
    assert a["excluded_question_ids"] == ["q5"]
    assert b["excluded_question_ids"] == ["q5"]
    # Both methods denominate over the same 2 questions (q1, q2).
    assert a["n_valid"] == 2
    assert b["n_valid"] == 2
    # Method B scored q5=1 originally; after exclusion its mean over q1,q2 is 1.0
    # (not 3/3=1.0 by coincidence — the point is it's averaged over 2).
    assert b["em"] == 1.0


def test_method_specific_filter_not_excluded(tmp_path: Path) -> None:
    """Auth errors (PERMANENT) are NOT shared across methods — those stay as
    per-method skipped rows but don't propagate to the union."""
    rows = [_row("q1"), _row("q2", em=0.0, f1=0.0, judge=None, error="PERMANENT_LLM_ERROR")]
    _write_run(tmp_path, "agentic", "seed_42", rows)
    _write_run(tmp_path, "random", "seed_42", [_row("q1"), _row("q2")])

    registry = apply_union_exclusion(tmp_path)

    assert registry["excluded_question_ids"] == []
    # Random's q2 is still in its score (its em=1.0); agentic's q2 is still
    # tagged as PERMANENT (it stays in per_question with that error).
    a = json.loads((tmp_path / "agentic/seed_42/benchmark_results.json").read_text())
    # Aggregate uses only valid rows (excludes permanent) so em=1.0/1=1.0
    assert a["n_valid"] == 1
    assert a["em"] == 1.0


def test_recall_and_judge_recomputed_after_exclusion(tmp_path: Path) -> None:
    """Aggregate recompute must walk recalls + judge with the excluded set
    dropped from both numerator and denominator."""
    rows = [
        {**_row("q1", judge=1), "retrieved_doc_ids": ["d1"], "supporting_doc_ids": ["d1"]},
        {**_row("q2", judge=0), "retrieved_doc_ids": ["d2"], "supporting_doc_ids": ["d2"]},
        {
            **_row("qBAD", em=0.0, f1=0.0, judge=None, error="CONTENT_FILTER"),
            "retrieved_doc_ids": [],
            "supporting_doc_ids": ["dBAD"],
        },
    ]
    _write_run(tmp_path, "agentic", "seed_42", rows)

    apply_union_exclusion(tmp_path)

    a = json.loads((tmp_path / "agentic/seed_42/benchmark_results.json").read_text())
    assert a["n_valid"] == 2
    assert a["llm_judge_accuracy"] == 0.5  # 1+0 over 2
    assert a["recall_at_1"] == 1.0  # both valid rows retrieve their gold doc at rank 1
    # Single-gold degradation: joint_recall == recall, mrr_complete == mrr_first.
    assert a["joint_recall_at_1"] == 1.0
    assert a["mrr_first"] == 1.0
    assert a["mrr_complete"] == 1.0


def test_existing_retrieval_aggregates_survive_rows_without_doc_ids(tmp_path: Path) -> None:
    """No surviving row carries ``supporting_doc_ids`` -> there is nothing to
    recompute retrieval from, so the aggregates already on file must be left
    alone. This function rewrites benchmark_results.json in place, so nulling
    them would destroy the only copy (the failure mode hit when per-question
    rows are slimmed for distribution)."""
    rows = [_row("q1", judge=1), _row("q2", judge=0)]  # _row() has empty doc-id lists
    f = _write_run(tmp_path, "agentic", "seed_42", rows)
    data = json.loads(f.read_text())
    data.update({"recall_at_1": 0.42, "joint_recall_at_5": 0.25, "mrr_first": 0.7, "mrr_complete": 0.5})
    f.write_text(json.dumps(data))

    apply_union_exclusion(tmp_path)

    a = json.loads(f.read_text())
    assert a["recall_at_1"] == 0.42
    assert a["joint_recall_at_5"] == 0.25
    assert a["mrr_first"] == 0.7
    assert a["mrr_complete"] == 0.5
    assert a["llm_judge_accuracy"] == 0.5  # answer aggregates still recomputed


def test_multi_hop_aggregates_separate_first_and_complete(tmp_path: Path) -> None:
    """Multi-hop questions must produce joint_recall_at_k < recall_at_k when
    only one of the two gold docs is retrieved, and mrr_complete must reflect
    the rank of the last-needed gold doc."""
    rows = [
        {  # both gold at ranks 1,2 — complete at rank 2.
            **_row("q1", judge=1),
            "retrieved_doc_ids": ["d1", "d2", "x"],
            "supporting_doc_ids": ["d1", "d2"],
        },
        {  # only one gold ever retrieved — first=1, complete=None.
            **_row("q2", judge=0),
            "retrieved_doc_ids": ["d1", "x", "y"],
            "supporting_doc_ids": ["d1", "d2"],
        },
    ]
    _write_run(tmp_path, "agentic", "seed_42", rows)

    apply_union_exclusion(tmp_path)

    a = json.loads((tmp_path / "agentic/seed_42/benchmark_results.json").read_text())
    # Partial recall@2: q1=1.0 + q2=0.5 over 2 → 0.75.
    assert a["recall_at_2"] == 0.75
    # Joint recall@2: q1=1 + q2=0 over 2 → 0.5.
    assert a["joint_recall_at_2"] == 0.5
    # Both questions hit first gold at rank 1 → mrr_first = 1.0.
    assert a["mrr_first"] == 1.0
    # q1 complete at rank 2 (=0.5), q2 never → mrr_complete = 0.25.
    assert a["mrr_complete"] == 0.25
