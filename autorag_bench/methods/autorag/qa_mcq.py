"""Convert the framework's cached ``exam.json`` into AutoRAG's ``qa.parquet``.

For the MCQ-ablation variant, AutoRAG receives our exam questions (re-shaped as
free-form QA where the gold ``generation_gt`` is the correct option text). The
custom ``mcq_accuracy`` metric (see ``mcq_metric.py``) scores by normalized
substring match, mirroring the framework's MCQ scoring logic.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import pandas as pd


def export_mcq_exam_to_parquet(exam_json: Path, out_path: Path) -> int:
    """Write each exam question to ``out_path`` with question + correct option."""
    data = json.loads(exam_json.read_text(encoding="utf-8"))
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError(f"Unexpected exam.json shape at {exam_json}")

    rows: list[dict] = []
    for i, q in enumerate(questions):
        question_text = q.get("question") or ""
        options = q.get("options") or []
        correct_idx = int(q.get("correct_index", 0))
        if not options or correct_idx >= len(options):
            continue
        correct_text = options[correct_idx]
        # Format the prompt to make AutoRAG's fstring template trivial: bake
        # the options into ``query`` itself so the generator sees the full MCQ.
        labeled = "\n".join(f"({string.ascii_uppercase[j]}) {opt}" for j, opt in enumerate(options))
        full_query = f"{question_text}\n{labeled}"

        retrieval_gt = [[]]
        if "source_doc_ids" in q and q["source_doc_ids"]:
            retrieval_gt = [list(q["source_doc_ids"])]

        rows.append(
            {
                "qid": q.get("id") or f"mcq_{i}",
                "query": full_query,
                "retrieval_gt": retrieval_gt,
                "generation_gt": [correct_text],
            }
        )

    if not rows:
        raise RuntimeError(f"No MCQ rows produced from {exam_json}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)
