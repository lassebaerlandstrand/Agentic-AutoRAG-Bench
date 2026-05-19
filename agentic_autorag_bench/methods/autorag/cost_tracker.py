"""LLM-call accountant for the AutoRAG subprocess.

Runs as the entry point inside the AutoRAG venv: monkey-patches the OpenAI
SDK and the boto3 bedrock-runtime client so every chat completion writes one
JSONL line with ``{model, prompt_tokens, completion_tokens}`` to the file at
``$AUTORAG_COST_LOG``. Then it invokes ``autorag.cli.cli()`` in-process with
the remaining argv so the patches stay live for the whole run.

The driver post-processes the log into USD via ``litellm.cost_per_token``.
Anything that does not flow through these two SDK surfaces (e.g. local
rerankers, embedding models) is correctly excluded from the meter.

RAGAS QA generation runs in a *separate* subprocess (``qa_ragas.py``) that
this script never wraps, so exam-creation cost is naturally outside the
meter — fair across baselines that reuse the agentic exam.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path


def _open_log() -> tuple[Path, threading.Lock] | tuple[None, None]:
    path_str = os.environ.get("AUTORAG_COST_LOG")
    if not path_str:
        return None, None
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, threading.Lock()


_LOG_PATH, _LOG_LOCK = _open_log()


def _write(record: dict) -> None:
    if _LOG_PATH is None:
        return
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with _LOG_LOCK, _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def _record_openai(response, source: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    model = getattr(response, "model", None) or "unknown"
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if prompt_tokens == 0 and completion_tokens == 0:
        return
    _write({
        "source": source,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })


def _patch_openai() -> None:
    import openai
    from openai.resources.chat.completions import AsyncCompletions, Completions

    _orig_sync = Completions.create
    _orig_async = AsyncCompletions.create

    def _sync_create(self, *args, **kwargs):
        response = _orig_sync(self, *args, **kwargs)
        if not kwargs.get("stream", False):
            _record_openai(response, source="openai")
        return response

    async def _async_create(self, *args, **kwargs):
        response = await _orig_async(self, *args, **kwargs)
        if not kwargs.get("stream", False):
            _record_openai(response, source="openai")
        return response

    Completions.create = _sync_create
    AsyncCompletions.create = _async_create
    # Touch openai to discourage unused-import lint; the side effect is the
    # class attribute reassignment above.
    _ = openai


def _record_bedrock(response: dict, model_id: str) -> None:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not usage:
        return
    prompt_tokens = int(usage.get("inputTokens", 0) or 0)
    completion_tokens = int(usage.get("outputTokens", 0) or 0)
    if prompt_tokens == 0 and completion_tokens == 0:
        return
    _write({
        "source": "bedrock",
        "model": model_id or "unknown",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })


def _patch_bedrock() -> None:
    """Wrap ``boto3.client('bedrock-runtime')`` so every ``converse`` logs usage.

    Bedrock returns ``response['usage']['inputTokens'/'outputTokens']``, so the
    bookkeeping is identical to OpenAI's ``response.usage``. ``converse_stream``
    is patched but ignored — AutoRAG's eval path doesn't stream.
    """
    try:
        import boto3
    except ImportError:
        return

    _orig_client = boto3.client

    def _patched_client(service_name, *args, **kwargs):
        client = _orig_client(service_name, *args, **kwargs)
        if service_name != "bedrock-runtime":
            return client

        _orig_converse = client.converse

        def _patched_converse(*c_args, **c_kwargs):
            response = _orig_converse(*c_args, **c_kwargs)
            _record_bedrock(response, c_kwargs.get("modelId", ""))
            return response

        client.converse = _patched_converse
        return client

    boto3.client = _patched_client

    # Also patch the default session's client factory, since llama-index's
    # BedrockConverse builds the client via ``boto3.Session(...).client(...)``
    # rather than the top-level ``boto3.client`` helper.
    from boto3.session import Session

    _orig_session_client = Session.client

    def _patched_session_client(self, service_name, *args, **kwargs):
        client = _orig_session_client(self, service_name, *args, **kwargs)
        if service_name != "bedrock-runtime":
            return client

        _orig_converse = client.converse

        def _patched_converse(*c_args, **c_kwargs):
            response = _orig_converse(*c_args, **c_kwargs)
            _record_bedrock(response, c_kwargs.get("modelId", ""))
            return response

        client.converse = _patched_converse
        return client

    Session.client = _patched_session_client


def main() -> None:
    _patch_openai()
    _patch_bedrock()
    # Hand off to the autorag CLI. ``sys.argv[0]`` becomes 'autorag' so click's
    # error messages and ``--help`` output read naturally. ``cli()`` raises
    # SystemExit on completion; we let it propagate.
    sys.argv = ["autorag", *sys.argv[1:]]
    from autorag.cli import cli

    cli()


if __name__ == "__main__":
    main()
