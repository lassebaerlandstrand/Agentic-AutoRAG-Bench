"""Convert the framework's cached open-ended ``exam.json`` into AutoRAG's ``qa.parquet``.

Used by the ``autorag_our_exam`` bench method: AutoRAG runs against the same
open-ended exam our framework generates, enabling an apples-to-apples
comparison on identical questions. Each exam entry carries:
- ``question`` (string)
- ``canonical_answer`` (string)
- ``answer_variants`` (list of acceptable paraphrases incl. canonical)
- ``source_doc_ids`` (list of supporting doc filenames, e.g. "alopecurus.md")

We translate this 1:1 to AutoRAG's required schema:
- ``qid`` (string)
- ``query`` (string)
- ``retrieval_gt`` (2D list of doc_id strings: outer list is conjunctive over
  supporting facts that must each be matched, inner list is disjunctive over
  acceptable doc ids for one fact — see ``autorag.evaluation.metric.retrieval``)
- ``generation_gt`` (list of acceptable answer strings)

Doc-id alignment: AutoRAG matches retrieval_gt against ``corpus.parquet.doc_id``.
Our corpus_export writes ``doc_id = path.stem`` (no extension), so we strip
the trailing ".md" / ".txt" from source_doc_ids here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DOC_SUFFIXES = (".md", ".txt", ".markdown", ".mdx")


def _strip_doc_suffix(doc_id: str) -> str:
    for sfx in _DOC_SUFFIXES:
        if doc_id.endswith(sfx):
            return doc_id[: -len(sfx)]
    return doc_id


def export_open_exam_to_parquet(exam_json: Path, out_path: Path) -> int:
    """Write each exam question to ``out_path`` as one (qid, query, retrieval_gt, generation_gt) row."""
    data = json.loads(exam_json.read_text(encoding="utf-8"))
    # Legacy format wrapped the list under ``{"questions": [...]}``; current
    # framework writes the list directly. Accept both.
    if isinstance(data, dict):
        questions = data.get("questions")
    else:
        questions = data
    if not isinstance(questions, list):
        raise ValueError(f"Unexpected exam.json shape at {exam_json}")

    rows: list[dict] = []
    for i, q in enumerate(questions):
        question_text = q.get("question") or ""
        canonical = q.get("canonical_answer") or ""
        variants = q.get("answer_variants") or []
        generation_gt: list[str] = list(variants) if variants else ([canonical] if canonical else [])
        if not (question_text and generation_gt):
            continue

        source_doc_ids_raw = q.get("source_doc_ids") or []
        # AutoRAG's retrieval_gt is a 2D list. The actual ``retrieval_recall``
        # implementation treats each *outer* element as a required supporting
        # fact (counted in ``hits`` only if at least one inner doc is in the
        # prediction) and the *inner* list as acceptable alternatives for that
        # fact. Multi-hop QA needs all supporting docs found, so we emit one
        # outer entry per source doc (singleton inner lists). On single-hop
        # this collapses to ``[[doc]]`` — identical to AutoRAG's own
        # single-hop QA generator output.
        if source_doc_ids_raw:
            retrieval_gt = [[_strip_doc_suffix(d)] for d in source_doc_ids_raw]
        else:
            retrieval_gt = [[]]

        rows.append(
            {
                "qid": q.get("id") or f"q_{i}",
                "query": question_text,
                "retrieval_gt": retrieval_gt,
                "generation_gt": generation_gt,
            }
        )

    if not rows:
        raise RuntimeError(f"No QA rows produced from {exam_json}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)
