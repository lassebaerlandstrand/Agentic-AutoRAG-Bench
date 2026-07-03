"""Tests for deterministic stratified benchmark splits (bench tooling)."""

from __future__ import annotations

from collections import Counter

import pytest
from agentic_autorag.benchmarks.schema import BenchmarkQAPair

from agentic_autorag_bench.splits import detect_stratify_key, stratified_split


def _pool(counts: dict[str, int], *, key: str = "n_hops", with_docs: bool = True) -> list[BenchmarkQAPair]:
    """Build a synthetic pool with ``counts`` rows per stratum label."""
    pairs: list[BenchmarkQAPair] = []
    i = 0
    for label, n in counts.items():
        for _ in range(n):
            pairs.append(
                BenchmarkQAPair(
                    id=f"{label}_{i}",
                    question=f"q{i}",
                    gold_answers=[f"a{i}"],
                    supporting_doc_ids=["d1", "d2"] if with_docs else [],
                    metadata={key: label},
                )
            )
            i += 1
    return pairs


def test_split_is_disjoint_and_reservoir_is_remainder() -> None:
    pool = _pool({"2": 500, "3": 300, "4": 200})  # 1000 usable
    result = stratified_split(pool, stratify_key="n_hops", holdout_size=300, seed=1)
    assert len(result.holdout) == 300
    assert len(result.optimization) == 700  # every remaining usable row
    holdout_ids = {p.id for p in result.holdout}
    opt_ids = {p.id for p in result.optimization}
    assert holdout_ids.isdisjoint(opt_ids)
    assert result.provenance.disjoint is True
    # Together the two slices cover every usable row exactly once.
    assert holdout_ids | opt_ids == {p.id for p in pool}


def test_holdout_matches_pool_distribution() -> None:
    pool = _pool({"2": 500, "3": 300, "4": 200})  # 50/30/20 %
    result = stratified_split(pool, stratify_key="n_hops", holdout_size=100, seed=1)
    hd = Counter(p.metadata["n_hops"] for p in result.holdout)
    assert hd == {"2": 50, "3": 30, "4": 20}


def test_split_is_deterministic_under_seed() -> None:
    pool = _pool({"2": 500, "3": 300, "4": 200})
    a = stratified_split(pool, stratify_key="n_hops", holdout_size=100, seed=7)
    b = stratified_split(pool, stratify_key="n_hops", holdout_size=100, seed=7)
    assert [p.id for p in a.holdout] == [p.id for p in b.holdout]
    assert [p.id for p in a.optimization] == [p.id for p in b.optimization]


def test_different_seed_changes_membership() -> None:
    pool = _pool({"2": 500, "3": 300, "4": 200})
    a = stratified_split(pool, stratify_key="n_hops", holdout_size=100, seed=1)
    b = stratified_split(pool, stratify_key="n_hops", holdout_size=100, seed=2)
    assert {p.id for p in a.holdout} != {p.id for p in b.holdout}


def test_docless_rows_excluded() -> None:
    pool = _pool({"2": 100}) + _pool({"3": 50}, with_docs=False)
    result = stratified_split(pool, stratify_key="n_hops", holdout_size=40, seed=1)
    assert result.provenance.n_excluded_no_docs == 50
    assert result.provenance.n_pool_usable == 100
    # No stratum-3 (doc-less) row can appear in either slice.
    assert all(p.metadata["n_hops"] == "2" for p in result.holdout + result.optimization)


def test_oversized_holdout_raises() -> None:
    pool = _pool({"2": 100})
    with pytest.raises(ValueError):
        stratified_split(pool, stratify_key="n_hops", holdout_size=200, seed=1)


def test_null_query_rows_retained_as_abstention_stratum() -> None:
    # null_query rows carry no supporting docs but are benchmark-verified
    # unanswerable, so the split keeps them as their own stratum.
    pool = _pool({"comparison_query": 120, "inference_query": 120}, key="question_type") + _pool(
        {"null_query": 60}, key="question_type", with_docs=False
    )
    result = stratified_split(pool, stratify_key="question_type", holdout_size=100, seed=1)
    prov = result.provenance
    assert prov.n_excluded_no_docs == 0  # abstention rows are not "excluded no docs"
    assert prov.n_abstention_retained == 60
    assert prov.n_pool_usable == 300
    # Proportional share in both slices (60 of 300 usable -> 20 of a 100 holdout).
    hd = Counter((p.metadata or {}).get("question_type") for p in result.holdout)
    od = Counter((p.metadata or {}).get("question_type") for p in result.optimization)
    assert hd["null_query"] == 20
    assert od["null_query"] == 40


def test_docless_answerable_still_excluded_when_abstention_retained() -> None:
    # Only null_query doc-less rows are kept; a doc-less *answerable* row is
    # still dropped (it can't be grounded or scored for retrieval).
    pool = (
        _pool({"comparison_query": 100}, key="question_type")
        + _pool({"null_query": 40}, key="question_type", with_docs=False)
        + _pool({"inference_query": 30}, key="question_type", with_docs=False)
    )
    result = stratified_split(pool, stratify_key="question_type", holdout_size=50, seed=1)
    prov = result.provenance
    assert prov.n_excluded_no_docs == 30  # the doc-less inference rows
    assert prov.n_abstention_retained == 40
    assert prov.n_pool_usable == 140  # 100 answerable + 40 abstention
    surviving = result.holdout + result.optimization
    assert all((p.metadata or {}).get("question_type") != "inference_query" for p in surviving)


def test_detect_stratify_key() -> None:
    assert detect_stratify_key(_pool({"a": 3}, key="type")) == "type"
    assert detect_stratify_key(_pool({"a": 3}, key="question_type")) == "question_type"
    assert detect_stratify_key(_pool({"2": 3}, key="n_hops")) == "n_hops"


def test_detect_stratify_key_missing_raises() -> None:
    pairs = [BenchmarkQAPair(id="x", question="q", gold_answers=["a"], supporting_doc_ids=["d"], metadata={"foo": 1})]
    with pytest.raises(ValueError):
        detect_stratify_key(pairs)
