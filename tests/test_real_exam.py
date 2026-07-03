"""Tests for the benchmark real-QA exam builder (framework ``ground_exam`` mocked).

The builder's job is the benchmark policy on top of ``ground_exam``: fill each
stratum to a held-out-matched target with fully grounded (tier-C) questions,
replacing ungroundable draws, so the finished exam is exactly ``exam_size`` and
100 % tier C. The grounder itself is mocked — a question grounds iff its id
contains ``ok`` — so these tests exercise the top-up/replace policy, not the LLM.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from agentic_autorag.config.models import OpenEndedQuestion

from agentic_autorag_bench.real_exam import build_real_exam, map_reasoning_type


def _to_tier_c(q: OpenEndedQuestion) -> OpenEndedQuestion:
    docs = list(q.supporting_doc_ids)
    return OpenEndedQuestion(
        id=q.id,
        question=q.question,
        canonical_answer=q.canonical_answer,
        answer_variants=list(q.answer_variants),
        reasoning_type=q.reasoning_type,
        source_doc_ids=docs,
        source_spans=[f"span {d}" for d in docs],
        supporting_doc_ids=docs,
    )


async def _fake_ground_exam(questions, corpus, **kwargs):
    """Upgrade groundable questions (id contains 'ok'); keep the rest tier B."""
    out = [_to_tier_c(q) if "ok" in q.id else q for q in questions]
    return out, None


def _pool(spec: dict[str, tuple[int, int]], *, key: str = "type") -> list[BenchmarkQAPair]:
    """spec maps stratum -> (n_groundable, n_ungroundable)."""
    pairs: list[BenchmarkQAPair] = []
    i = 0
    for stratum, (n_ok, n_bad) in spec.items():
        for _ in range(n_ok):
            pairs.append(
                BenchmarkQAPair(
                    id=f"{stratum}_ok_{i}",
                    question=f"q{i}",
                    gold_answers=[f"a{i}", f"alt{i}"],
                    supporting_doc_ids=[f"d{i}"],
                    metadata={key: stratum},
                )
            )
            i += 1
        for _ in range(n_bad):
            pairs.append(
                BenchmarkQAPair(
                    id=f"{stratum}_bad_{i}",
                    question=f"q{i}",
                    gold_answers=[f"a{i}"],
                    supporting_doc_ids=[f"d{i}"],
                    metadata={key: stratum},
                )
            )
            i += 1
    return pairs


def test_reasoning_type_never_numeric() -> None:
    for key, val in [("type", "bridge"), ("type", "comparison"), ("question_type", "temporal_query"), ("n_hops", "4")]:
        assert map_reasoning_type(key, val) not in {"numeric", "numeric_single"}


@pytest.mark.asyncio
async def test_exam_is_all_tier_c_and_matches_holdout_mix() -> None:
    pool = _pool({"bridge": (60, 0), "comparison": (40, 0)})
    holdout_distribution = {"bridge": 60, "comparison": 40}
    with patch("agentic_autorag_bench.real_exam.ground_exam", new=_fake_ground_exam):
        exam, prov = await build_real_exam(
            pool, {}, holdout_distribution, extractor_model="test/model", exam_size=50, concurrency=4
        )
    assert len(exam) == 50
    assert all(q.grounding_tier == "C" for q in exam)
    assert prov.all_tier_c is True
    # 50 apportioned 60/40 -> 30 bridge / 20 comparison, matched to the held-out.
    assert prov.exam_distribution == {"bridge": 30, "comparison": 20}
    assert prov.target_distribution == {"bridge": 30, "comparison": 20}


@pytest.mark.asyncio
async def test_replaces_ungroundable_until_target_met() -> None:
    # Half of the stratum's candidates never ground; the builder must draw past
    # them. A full, all-'ok' exam of the target size proves replacement worked —
    # taking the first N candidates would have pulled in 'bad' (tier-B) rows.
    pool = _pool({"bridge": (20, 20)})
    holdout_distribution = {"bridge": 10}
    with patch("agentic_autorag_bench.real_exam.ground_exam", new=_fake_ground_exam):
        exam, prov = await build_real_exam(
            pool, {}, holdout_distribution, extractor_model="test/model", exam_size=10, concurrency=4
        )
    assert len(exam) == 10
    assert all(q.grounding_tier == "C" for q in exam)
    assert all("bad" not in q.id for q in exam)
    # It had to extract more candidates than it kept, because some failed to ground.
    assert prov.per_stratum_extracted["bridge"] > 10


@pytest.mark.asyncio
async def test_fails_loud_when_stratum_cannot_fill() -> None:
    # The comparison stratum is entirely ungroundable, but the held-out mix still
    # demands comparison questions — the builder must raise, not silently skew.
    pool = _pool({"bridge": (90, 0), "comparison": (0, 10)})
    holdout_distribution = {"bridge": 90, "comparison": 10}
    with (
        patch("agentic_autorag_bench.real_exam.ground_exam", new=_fake_ground_exam),
        pytest.raises(ValueError, match="exhausted"),
    ):
        await build_real_exam(pool, {}, holdout_distribution, extractor_model="test/model", exam_size=50, concurrency=4)
