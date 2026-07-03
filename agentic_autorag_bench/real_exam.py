"""Build a 100 %-grounded real-QA exam from a benchmark's own held-in questions.

Each optimization-reservoir question carries a gold answer and the ids of its
supporting corpus documents but no span-level evidence. This module recovers
tier-C grounding by reusing the framework's ``ground_exam`` (LLM span extraction
+ ``verify_source_facts``), and layers the benchmark policy on top: the exam is
exactly ``exam_size`` questions, every one tier-C, with a difficulty mix matched
to the held-out slice. Because some questions can't be grounded, candidates are
drawn per stratum lazily and the ones that fail are **replaced** until each
stratum reaches its target — so the finished exam is 100 % span-grounded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

from agentic_autorag.benchmarks.schema import BenchmarkQAPair
from agentic_autorag.config.models import OpenEndedQuestion
from agentic_autorag.examiner.ground_exam import ground_exam
from pydantic import BaseModel

from agentic_autorag_bench.splits import _hamilton_allocation, _stratum_of, detect_stratify_key

logger = logging.getLogger(__name__)

DEFAULT_EXAM_SIZE = 100

# Difficulty label -> exam reasoning_type. Never numeric: these answers are
# names/entities, and a numeric answer_format_hint would wrongly demand a
# number. Uniform across datasets so it can't bias a cross-method comparison.
# MuSiQue (keyed on n_hops) is entirely bridge-style multi-hop composition.
_REASONING_TYPE_MAP: dict[tuple[str, str], str] = {
    ("type", "bridge"): "bridge",
    ("type", "comparison"): "comparison",
    ("question_type", "comparison_query"): "comparison",
    ("question_type", "inference_query"): "inference",
    ("question_type", "temporal_query"): "inference",
}
_DEFAULT_REASONING_TYPE = "inference"


def map_reasoning_type(stratify_key: str, stratum_value: str) -> str:
    """Map a difficulty label to a non-numeric exam ``reasoning_type``."""
    if stratify_key == "n_hops":
        return "bridge"
    return _REASONING_TYPE_MAP.get((stratify_key, str(stratum_value)), _DEFAULT_REASONING_TYPE)


class RealExamProvenance(BaseModel):
    """Reproducibility record written next to the exam file."""

    n_pool: int
    exam_size: int
    all_tier_c: bool
    extractor_model: str
    fuzzy_threshold: float
    stratify_key: str
    seed: int
    qa_sha256: str
    target_distribution: dict[str, int]
    exam_distribution: dict[str, int]
    per_stratum_extracted: dict[str, int]


def _to_tier_b(qa: BenchmarkQAPair, reasoning_type: str) -> OpenEndedQuestion | None:
    """Convert a benchmark QA pair into a tier-B question for grounding."""
    gold = [g for g in qa.gold_answers if g and g.strip()]
    if not gold:
        return None
    return OpenEndedQuestion(
        id=qa.id,
        question=qa.question,
        canonical_answer=gold[0],
        answer_variants=gold[1:],
        reasoning_type=reasoning_type,
        supporting_doc_ids=list(qa.supporting_doc_ids),
    )


async def _fill_stratum(
    candidates: list[OpenEndedQuestion],
    corpus: dict[str, str],
    target: int,
    *,
    stratum: str,
    extractor_model: str,
    reasoning_effort: str | None,
    fuzzy_threshold: float,
    concurrency: int,
) -> tuple[list[OpenEndedQuestion], int]:
    """Ground candidates in order until ``target`` reach tier C; replace failures.

    Returns the selected tier-C questions and how many candidates were extracted.
    Raises if the candidate list is exhausted before reaching the target.
    """
    selected: list[OpenEndedQuestion] = []
    pos = 0
    while len(selected) < target:
        need = target - len(selected)
        batch = candidates[pos : pos + need]
        if not batch:
            raise ValueError(
                f"stratum {stratum!r} exhausted: grounded {len(selected)} of {len(candidates)} "
                f"candidates but need {target}. Widen the split (larger optimization reservoir)."
            )
        pos += len(batch)
        grounded, _ = await ground_exam(
            batch,
            corpus,
            extractor_model=extractor_model,
            reasoning_effort=reasoning_effort,
            fuzzy_threshold=fuzzy_threshold,
            concurrency=concurrency,
        )
        for q in grounded:
            if q.grounding_tier == "C" and len(selected) < target:
                selected.append(q)
    return selected, pos


async def build_real_exam(
    optimization_pool: list[BenchmarkQAPair],
    corpus: dict[str, str],
    holdout_distribution: dict[str, int],
    *,
    extractor_model: str,
    reasoning_effort: str | None = None,
    exam_size: int = DEFAULT_EXAM_SIZE,
    fuzzy_threshold: float = 0.9,
    stratify_key: str | None = None,
    concurrency: int = 10,
    seed: int = 42,
) -> tuple[list[OpenEndedQuestion], RealExamProvenance]:
    """Build a 100 %-tier-C real-QA exam matched to the held-out difficulty mix.

    ``holdout_distribution`` (per-stratum held-out counts) sets the mix: the exam
    is apportioned across strata in the same proportions. Each stratum is filled
    to its target with grounded questions, replacing ungroundable ones — so every
    returned question is tier C and the strata sum to ``exam_size``.
    """
    key = stratify_key or detect_stratify_key(optimization_pool)
    targets = _hamilton_allocation(exam_size, {k: v for k, v in holdout_distribution.items() if v > 0})

    by_stratum: dict[str, list[BenchmarkQAPair]] = {}
    for qa in optimization_pool:
        by_stratum.setdefault(_stratum_of(qa, key), []).append(qa)

    rng = random.Random(seed)
    exam: list[OpenEndedQuestion] = []
    exam_dist: dict[str, int] = {}
    extracted: dict[str, int] = {}
    for stratum in sorted(targets):
        target = targets[stratum]
        if target == 0:
            continue
        members = sorted(by_stratum.get(stratum, []), key=lambda qa: qa.id)
        rng.shuffle(members)
        reasoning_type = map_reasoning_type(key, stratum)
        candidates = [q for q in (_to_tier_b(m, reasoning_type) for m in members) if q is not None]
        selected, n_extracted = await _fill_stratum(
            candidates,
            corpus,
            target,
            stratum=stratum,
            extractor_model=extractor_model,
            reasoning_effort=reasoning_effort,
            fuzzy_threshold=fuzzy_threshold,
            concurrency=concurrency,
        )
        exam.extend(selected)
        exam_dist[stratum] = len(selected)
        extracted[stratum] = n_extracted

    exam.sort(key=lambda q: q.id)
    if len(exam) != exam_size:
        raise AssertionError(f"exam has {len(exam)} questions, expected {exam_size}")
    all_tier_c = all(q.grounding_tier == "C" for q in exam)
    if not all_tier_c:
        raise AssertionError("benchmark exam must be 100 % tier C but some questions are not grounded")

    qa_sha = hashlib.sha256(
        json.dumps([p.model_dump(mode="json") for p in optimization_pool], sort_keys=True).encode("utf-8")
    ).hexdigest()
    provenance = RealExamProvenance(
        n_pool=len(optimization_pool),
        exam_size=len(exam),
        all_tier_c=all_tier_c,
        extractor_model=extractor_model,
        fuzzy_threshold=fuzzy_threshold,
        stratify_key=key,
        seed=seed,
        qa_sha256=qa_sha,
        target_distribution=dict(sorted(targets.items())),
        exam_distribution=dict(sorted(exam_dist.items())),
        per_stratum_extracted=dict(sorted(extracted.items())),
    )
    return exam, provenance


def write_real_exam(
    exam: list[OpenEndedQuestion],
    provenance: RealExamProvenance,
    output_path: Path,
) -> Path:
    """Write the exam JSON and a sibling ``<stem>_provenance.json``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([q.model_dump(mode="json") for q in exam], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prov_path = output_path.with_name(output_path.stem + "_provenance.json")
    prov_path.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return prov_path
