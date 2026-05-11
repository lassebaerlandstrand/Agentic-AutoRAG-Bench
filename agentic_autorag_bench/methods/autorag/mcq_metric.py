"""``mcq_accuracy`` metric registered with AutoRAG via PYTHONPATH plugin.

Normalized-substring match: the prediction is correct if the gold option's
canonical form is a substring of the prediction's canonical form (lowercase,
ASCII-only, punctuation stripped). Mirrors the framework's MCQ scoring so the
AutoRAG-MCQ ablation row scores on the same notion of "answered correctly"
as the agentic run.
"""

from __future__ import annotations

import re
import string
import unicodedata


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mcq_accuracy(generation_gt: list[str], generations: list[str]) -> list[float]:
    """Return per-row 1.0 / 0.0 scores. AutoRAG metrics are list-in, list-out."""
    scores: list[float] = []
    for gold_options, pred in zip(generation_gt, generations, strict=True):
        if isinstance(gold_options, str):
            gold_list = [gold_options]
        else:
            gold_list = list(gold_options)
        norm_pred = _normalize(pred or "")
        hit = any(_normalize(opt) in norm_pred for opt in gold_list if opt)
        scores.append(1.0 if hit else 0.0)
    return scores
