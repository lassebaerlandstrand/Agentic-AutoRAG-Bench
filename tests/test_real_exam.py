"""Tests for the benchmark real-QA exam builder (framework ``ground_exam`` mocked).

The builder's job is the benchmark policy on top of ``ground_exam``: fill each
answerable stratum to a held-out-matched target with fully grounded (tier-C)
questions, replacing ungroundable draws, while the ``null_query`` stratum passes
benchmark-verified unanswerable rows through without grounding. The grounder is
mocked — a question grounds iff its id contains ``ok`` — so these tests exercise
the top-up/replace policy and the abstention pass-through, not the LLM.
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


def _mixed_pool() -> list[BenchmarkQAPair]:
    """60 groundable answerable rows + 40 benchmark-verified unanswerable rows."""
    pairs: list[BenchmarkQAPair] = []
    for i in range(60):
        pairs.append(
            BenchmarkQAPair(
                id=f"inference_ok_{i}",
                question=f"q{i}",
                gold_answers=[f"a{i}"],
                supporting_doc_ids=[f"d{i}"],
                metadata={"question_type": "inference_query"},
            )
        )
    for i in range(60, 100):
        pairs.append(
            BenchmarkQAPair(
                id=f"null_{i}",
                question=f"unanswerable {i}?",
                gold_answers=["Insufficient information."],
                supporting_doc_ids=[],
                metadata={"question_type": "null_query"},
            )
        )
    return pairs


@pytest.mark.asyncio
async def test_abstention_stratum_passes_through_without_grounding() -> None:
    pool = _mixed_pool()
    holdout_distribution = {"inference_query": 60, "null_query": 40}
    with patch("agentic_autorag_bench.real_exam.ground_exam", new=_fake_ground_exam):
        exam, prov = await build_real_exam(
            pool, {}, holdout_distribution, extractor_model="test/model", exam_size=50, concurrency=4
        )
    assert len(exam) == 50
    abstention = [q for q in exam if q.reasoning_type is None]
    answerable = [q for q in exam if q.reasoning_type is not None]
    # 50 apportioned 60/40 -> 30 answerable / 20 abstention.
    assert len(answerable) == 30
    assert len(abstention) == 20
    # Answerable are span-grounded; abstention are bare tier-A with the gold
    # insufficiency statement and no docs/spans.
    assert all(q.grounding_tier == "C" for q in answerable)
    assert all(q.grounding_tier == "A" for q in abstention)
    assert all(q.canonical_answer == "Insufficient information." for q in abstention)
    assert all(not q.supporting_doc_ids and not q.source_spans for q in abstention)
    # Provenance: all_tier_c is asserted over the answerable subset only.
    assert prov.all_tier_c is True
    assert prov.n_abstention == 20
    assert prov.exam_distribution == {"inference_query": 30, "null_query": 20}


@pytest.mark.asyncio
async def test_abstention_stratum_fails_loud_when_too_few_rows() -> None:
    pool = [
        BenchmarkQAPair(
            id=f"inference_ok_{i}",
            question=f"q{i}",
            gold_answers=[f"a{i}"],
            supporting_doc_ids=[f"d{i}"],
            metadata={"question_type": "inference_query"},
        )
        for i in range(50)
    ] + [
        BenchmarkQAPair(
            id=f"null_{i}",
            question=f"q{i}",
            gold_answers=["Insufficient information."],
            supporting_doc_ids=[],
            metadata={"question_type": "null_query"},
        )
        for i in range(3)
    ]
    holdout_distribution = {"inference_query": 50, "null_query": 50}  # wants ~25 abstention, only 3 exist
    with (
        patch("agentic_autorag_bench.real_exam.ground_exam", new=_fake_ground_exam),
        pytest.raises(ValueError, match="abstention stratum"),
    ):
        await build_real_exam(pool, {}, holdout_distribution, extractor_model="test/model", exam_size=50, concurrency=4)


class _FakePipelineConfig:
    llm_timeout_s = 10.0


class _FakeTiming:
    model_s = 0.0


class _FakeRetrieval:
    documents: list = []
    timing = _FakeTiming()
    expansion_cost = {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


class _FakePipeline:
    """Minimal pipeline returning a fixed answer, for scoring a built question."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.config = _FakePipelineConfig()

    async def retrieve(self, _q):
        return _FakeRetrieval()

    async def prepare_context(self, _q, _r):
        return "", {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def generate(self, _prompt):
        return self._answer, {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


@pytest.mark.asyncio
async def test_null_query_row_scores_through_framework_abstention_path() -> None:
    """Behavioral equivalence (bench->framework): a MultiHop null_query row,
    mapped by the builder, scores correct when the system abstains and wrong
    when it hallucinates — through the real OpenEndedEvaluator, judge mocked."""
    from unittest.mock import AsyncMock

    from agentic_autorag.examiner.evaluator import OpenEndedEvaluator

    pool = [
        BenchmarkQAPair(
            id="null_row",
            question="What is the fictional planet Qorb's population?",
            gold_answers=["Insufficient information."],
            supporting_doc_ids=[],
            metadata={"question_type": "null_query"},
        )
    ]
    with patch("agentic_autorag_bench.real_exam.ground_exam", new=_fake_ground_exam):
        exam, _ = await build_real_exam(
            pool, {}, {"null_query": 1}, extractor_model="test/model", exam_size=1, concurrency=1
        )
    assert exam[0].grounding_tier == "A"

    evaluator = OpenEndedEvaluator(concurrency=1, judge_model="judge/test")
    # Correct abstention -> judge YES(1) -> correct.
    with patch(
        "agentic_autorag.examiner.evaluator.llm_judge",
        new=AsyncMock(return_value=1),
    ):
        correct = await evaluator.evaluate(_FakePipeline("The context does not say."), exam)
    assert correct.question_results[0].correct is True

    # Hallucinated answer -> judge NO(0) -> wrong (diagnosis mocked away).
    with (
        patch("agentic_autorag.examiner.evaluator.llm_judge", new=AsyncMock(return_value=0)),
        patch(
            "agentic_autorag.examiner.evaluator.llm_diagnose_failure",
            new=AsyncMock(return_value="context_present_but_wrong"),
        ),
    ):
        wrong = await evaluator.evaluate(_FakePipeline("Two billion."), exam)
    assert wrong.question_results[0].correct is False
    assert wrong.question_results[0].failure_class == "generation_wrong"
