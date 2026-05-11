"""AutoRAG QA pre-screen drops content-filter rejections before the subprocess runs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agentic_autorag_bench.methods.autorag.qa_prescreen import prescreen_qa_for_content_filter


class _ContentPolicy(Exception):
    pass


_ContentPolicy.__name__ = "ContentPolicyViolationError"


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "qa.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


async def _fake_acompletion_factory(banned: set[str]):
    """Return an awaitable that raises content-filter for any prompt containing
    a banned substring, so the test can pick exactly which rows get dropped."""
    async def fake(*, messages, **_kwargs):
        prompt = messages[0]["content"]
        for term in banned:
            if term in prompt:
                raise _ContentPolicy(f"blocked: {term}")
        # Return a tuple shape that matches acompletion_with_cost.
        return (object(), {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0})
    return fake


@pytest.mark.asyncio
async def test_prescreen_drops_filtered_row(tmp_path: Path) -> None:
    p = _write_parquet(tmp_path, [
        {"qid": "q1", "query": "Who wrote Hamlet?", "retrieval_gt": [["d1"]], "generation_gt": ["Shakespeare"]},
        {"qid": "q2", "query": "POISONED-TOPIC explain it", "retrieval_gt": [["d2"]], "generation_gt": ["x"]},
    ])

    fake = await _fake_acompletion_factory({"POISONED-TOPIC"})
    with patch("agentic_autorag_bench.methods.autorag.qa_prescreen.acompletion_with_cost", fake):
        dropped = await prescreen_qa_for_content_filter(p, model="azure/gpt-4o-mini", concurrency=2)

    assert dropped == ["q2"]
    remaining = pd.read_parquet(p)
    assert list(remaining["qid"]) == ["q1"]


@pytest.mark.asyncio
async def test_prescreen_keeps_rows_on_non_filter_error(tmp_path: Path) -> None:
    """Auth/timeout errors must NOT drop rows — those are infra problems, not
    content rejections. Silently shrinking the QA set on infra errors would
    hide failures."""
    p = _write_parquet(tmp_path, [
        {"qid": "q1", "query": "Who wrote Hamlet?", "retrieval_gt": [["d1"]], "generation_gt": ["Shakespeare"]},
    ])

    async def fake_auth_error(**_kwargs):
        class _Auth(Exception):
            pass
        _Auth.__name__ = "AuthenticationError"
        raise _Auth("invalid key")

    with patch(
        "agentic_autorag_bench.methods.autorag.qa_prescreen.acompletion_with_cost",
        fake_auth_error,
    ):
        dropped = await prescreen_qa_for_content_filter(p, model="azure/gpt-4o-mini", concurrency=1)

    assert dropped == []
    remaining = pd.read_parquet(p)
    assert list(remaining["qid"]) == ["q1"]


@pytest.mark.asyncio
async def test_prescreen_handles_empty_parquet(tmp_path: Path) -> None:
    p = _write_parquet(tmp_path, [])
    dropped = await prescreen_qa_for_content_filter(p, model="azure/gpt-4o-mini")
    assert dropped == []
