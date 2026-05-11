"""Convert the framework's cached open-ended ``exam.json`` into AutoRAG's ``qa.parquet``.

The framework's exam is open-ended QA, not MCQ. Each entry carries:
- ``question`` (string)
- ``canonical_answer`` (string)
- ``answer_variants`` (list of acceptable paraphrases incl. canonical)
- ``source_doc_ids`` (list of supporting doc filenames, e.g. "alopecurus.md")

We translate this 1:1 to AutoRAG's required schema:
- ``qid`` (string)
- ``query`` (string)
- ``retrieval_gt`` (2D list of doc_id strings — the inner list is a single
  AND-conjunction; we use one inner list = all supporting docs)
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


def export_mcq_exam_to_parquet(exam_json: Path, out_path: Path) -> int:
    """Write each exam question to ``out_path`` as one (qid, query, retrieval_gt, generation_gt) row.

    The function name retains the legacy "mcq" suffix because the bench's
    method registry uses ``autorag_mcq`` for "AutoRAG running against our
    framework's exam." The exam itself is now open-ended QA — see module
    docstring.
    """
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
        # AutoRAG's retrieval_gt is a 2D list: outer = OR (any of these
        # disjunctions satisfies retrieval), inner = AND (all of these
        # doc_ids must be retrieved). For multi-hop QA where ALL supporting
        # docs are needed, one inner list with all the docs is the correct
        # shape.
        if source_doc_ids_raw:
            retrieval_gt = [[_strip_doc_suffix(d) for d in source_doc_ids_raw]]
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
