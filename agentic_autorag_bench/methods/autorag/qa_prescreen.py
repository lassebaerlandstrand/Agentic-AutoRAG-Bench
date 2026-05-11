"""Pre-screen ``qa.parquet`` rows that the provider's content filter rejects.

AutoRAG's ``evaluate`` subprocess aborts the whole enumeration as soon as a
single QA row triggers the provider's content policy (Azure's
``ResponsibleAIPolicyViolation``). The bench has no hook into the subprocess
to skip-and-continue, so we filter upstream: send each question through a
cheap question-only completion using the same provider, drop the rows that
get rejected, and let AutoRAG enumerate the survivors.

This is a lower bound on filtered rows — AutoRAG's metric prompts (RAGAS, MCQ)
may include the gold answer and retrieved context, either of which can trigger
the filter when the question alone doesn't. For paper coverage that's good
enough: it transforms "AutoRAG row missing from the table" into "AutoRAG row
present with N pre-screened drops noted."
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
from agentic_autorag.examiner._errors import is_content_filter_error
from agentic_autorag.litellm_runtime import acompletion_with_cost

logger = logging.getLogger("agentic_autorag_bench.run")

_PRESCREEN_PROMPT_TEMPLATE = (
    "Briefly answer the following question. If the question is unanswerable, say 'unknown'.\n\n"
    "Question: {question}"
)


async def _probe_one(model: str, qid: str, query: str, timeout_s: float) -> tuple[str, bool, str]:
    """Return ``(qid, kept, reason)``. ``kept=False`` only on content-filter
    rejection — other errors (auth, timeout) are passed through as ``kept=True``
    so we don't silently shrink the QA set on transient infra problems."""
    try:
        await acompletion_with_cost(
            cost_category="autorag_prescreen",
            model=model,
            messages=[{"role": "user", "content": _PRESCREEN_PROMPT_TEMPLATE.format(question=query)}],
            num_retries=1,
            timeout=timeout_s,
        )
        return qid, True, ""
    except Exception as exc:
        if is_content_filter_error(exc):
            return qid, False, type(exc).__name__
        # Non-filter error: keep the row, log loudly so the user notices.
        logger.warning("Pre-screen errored on %s (kept): %s: %s", qid, type(exc).__name__, exc)
        return qid, True, ""


async def prescreen_qa_for_content_filter(
    qa_parquet: Path,
    *,
    model: str,
    concurrency: int = 5,
    timeout_s: float = 20.0,
) -> list[str]:
    """Drop content-filter-rejecting rows from ``qa_parquet`` in place.

    Returns the list of dropped ``qid`` values. Idempotent: if the parquet
    has already been pre-screened, every remaining row passes and the function
    returns ``[]``.
    """
    df = pd.read_parquet(qa_parquet)
    if df.empty:
        return []

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(qid: str, query: str) -> tuple[str, bool, str]:
        async with sem:
            return await _probe_one(model, qid, query, timeout_s)

    coros = [_bounded(str(row.qid), str(row.query)) for row in df.itertuples(index=False)]
    results = await asyncio.gather(*coros)
    dropped = [qid for qid, kept, _ in results if not kept]

    if not dropped:
        logger.info("AutoRAG pre-screen: %d/%d rows kept (no content-filter rejections)", len(df), len(df))
        return []

    kept_df = df[~df["qid"].astype(str).isin(set(dropped))].reset_index(drop=True)
    kept_df.to_parquet(qa_parquet)
    logger.warning(
        "AutoRAG pre-screen: dropped %d/%d rows due to content-filter (qids: %s)",
        len(dropped), len(df), dropped,
    )
    return dropped
