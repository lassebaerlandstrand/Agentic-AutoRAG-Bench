"""Tests for the AutoRAG cost log + summarizer.

We don't import ``cost_tracker.py`` here — that module monkey-patches
``openai`` / ``boto3`` and assumes it runs inside the AutoRAG venv. The
*summarizer* (``_summarize_cost_log``) is bench-side code, so we exercise
it directly with a hand-built JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_autorag_bench.methods.autorag.driver import _summarize_cost_log


def _write_log(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def test_summarize_missing_log_returns_zero(tmp_path: Path) -> None:
    summary = _summarize_cost_log(tmp_path / "nonexistent.jsonl")
    assert summary == {"total_usd": 0.0, "buckets": {}, "by_source": {}, "n_calls": 0}


def test_summarize_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(
        "not-json\n"
        + json.dumps({"model": "gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 10}) + "\n"
        + "\n"  # blank
        + "{also bad\n",
        encoding="utf-8",
    )
    summary = _summarize_cost_log(log)
    assert summary["n_calls"] == 1
    assert "gpt-4o-mini" in summary["buckets"]


def test_summarize_buckets_per_model_and_sums(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    _write_log(
        log,
        [
            {"source": "openai", "model": "gpt-4o-mini", "prompt_tokens": 1000, "completion_tokens": 500},
            {"source": "openai", "model": "gpt-4o-mini", "prompt_tokens": 2000, "completion_tokens": 100},
            {"source": "bedrock", "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
             "prompt_tokens": 500, "completion_tokens": 200},
        ],
    )
    summary = _summarize_cost_log(log)
    assert summary["n_calls"] == 3
    assert set(summary["buckets"]) == {"gpt-4o-mini", "anthropic.claude-haiku-4-5-20251001-v1:0"}
    gpt = summary["buckets"]["gpt-4o-mini"]
    assert gpt["n_calls"] == 2
    assert gpt["prompt_tokens"] == 3000
    assert gpt["completion_tokens"] == 600
    # gpt-4o-mini is in litellm.model_cost, so usd must be > 0.
    assert gpt["usd"] > 0
    # And the model-level usd must sum to the top-level total.
    assert summary["total_usd"] == pytest.approx(
        sum(b["usd"] for b in summary["buckets"].values()),
        rel=1e-9,
    )


def test_summarize_unknown_model_contributes_zero(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    _write_log(
        log,
        [
            {"model": "some-private-vllm-deployment-not-in-litellm",
             "prompt_tokens": 12345, "completion_tokens": 6789},
        ],
    )
    summary = _summarize_cost_log(log)
    assert summary["n_calls"] == 1
    # Token counts are still recorded; just no USD attribution.
    bucket = summary["buckets"]["some-private-vllm-deployment-not-in-litellm"]
    assert bucket["prompt_tokens"] == 12345
    assert bucket["completion_tokens"] == 6789
    assert bucket["usd"] == 0.0
    assert summary["total_usd"] == 0.0


def test_summarize_accumulates_cache_token_fields(tmp_path: Path) -> None:
    """Cache token fields flow from log records into the per-model bucket."""
    log = tmp_path / "calls.jsonl"
    _write_log(
        log,
        [
            {
                "source": "openai",
                "model": "gpt-4o-mini",
                "prompt_tokens": 5000,
                "completion_tokens": 100,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 0,
            },
            {
                "source": "openai",
                "model": "gpt-4o-mini",
                "prompt_tokens": 2000,
                "completion_tokens": 50,
                "cache_read_input_tokens": 1500,
                "cache_creation_input_tokens": 0,
            },
        ],
    )
    summary = _summarize_cost_log(log)
    bucket = summary["buckets"]["gpt-4o-mini"]
    assert bucket["cache_read_input_tokens"] == 4500
    assert bucket["cache_creation_input_tokens"] == 0


def test_summarize_legacy_records_without_cache_fields(tmp_path: Path) -> None:
    """Records written before cache tracking existed must still parse, with cache totals = 0."""
    log = tmp_path / "calls.jsonl"
    _write_log(
        log,
        [
            {"model": "gpt-4o-mini", "prompt_tokens": 1000, "completion_tokens": 500},
        ],
    )
    summary = _summarize_cost_log(log)
    bucket = summary["buckets"]["gpt-4o-mini"]
    assert bucket["cache_read_input_tokens"] == 0
    assert bucket["cache_creation_input_tokens"] == 0
    # And usd remains identical to the pre-cache-tracking behavior since
    # cost_per_token(..., cache_*=0) is a no-op vs omitting the args.
    assert bucket["usd"] > 0


def test_summarize_cached_call_charges_less_than_uncached(tmp_path: Path) -> None:
    """Same total prompt tokens, one with cache reads: cached version must cost less.

    This is the regression for the original bug — previously cache fields were
    dropped and identical token totals produced identical USD regardless of
    cache usage.
    """
    uncached_log = tmp_path / "uncached.jsonl"
    _write_log(
        uncached_log,
        [
            {"model": "gpt-4o-mini", "prompt_tokens": 10000, "completion_tokens": 100,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        ],
    )
    cached_log = tmp_path / "cached.jsonl"
    _write_log(
        cached_log,
        [
            {"model": "gpt-4o-mini", "prompt_tokens": 10000, "completion_tokens": 100,
             "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 0},
        ],
    )
    uncached = _summarize_cost_log(uncached_log)
    cached = _summarize_cost_log(cached_log)
    assert cached["total_usd"] < uncached["total_usd"], (
        f"cached={cached['total_usd']} should be < uncached={uncached['total_usd']}; "
        "if equal, litellm.cost_per_token isn't honoring cache_read_input_tokens"
    )


def test_scrub_trial_artifacts_walks_yaml_and_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_scrub_trial_artifacts`` must rewrite every yaml/csv/json in the tree,
    not just the top-level extracted_sample.yaml the original scrubber covered.
    """
    from agentic_autorag_bench.methods.autorag.driver import _scrub_trial_artifacts

    secret = "SECRET_KEY_VALUE_12345"
    monkeypatch.setenv("AZURE_API_KEY", secret)

    trial = tmp_path / "trial_0"
    (trial / "node_line" / "node").mkdir(parents=True)
    leaks = {
        trial / "config.yaml": f"api_key: {secret}\n",
        trial / "node_line" / "summary.csv": f"col,api_key\nrow,{secret}\n",
        trial / "node_line" / "node" / "summary.csv": f"params,{secret}\n",
        trial / "translation_notes.json": f'{{"api_key": "{secret}"}}\n',
    }
    untouched = trial / "node_line" / "node" / "0.parquet"
    untouched.write_bytes(b"binary-blob-" + secret.encode())  # parquet not in glob
    for path, contents in leaks.items():
        path.write_text(contents, encoding="utf-8")

    _scrub_trial_artifacts(trial)

    placeholder = "${AZURE_API_KEY}"
    for path in leaks:
        text = path.read_text(encoding="utf-8")
        assert secret not in text, f"{path} still contains the literal key"
        assert placeholder in text, f"{path} missing placeholder"
    # Parquet (and any other extension) must NOT be touched — the scrubber's
    # contract is text-only; touching binary would corrupt parquet/sqlite.
    assert secret.encode() in untouched.read_bytes()


def test_cost_tracker_script_is_present() -> None:
    """The driver references the wrapper by path — make sure it exists."""
    script = Path(__file__).resolve().parents[1] / "agentic_autorag_bench" / "methods" / "autorag" / "cost_tracker.py"
    assert script.exists(), f"cost_tracker.py missing at {script}"
    text = script.read_text(encoding="utf-8")
    # Sanity: the monkey-patch targets we depend on are mentioned by name so
    # an accidental rename surfaces here instead of at runtime.
    assert "Completions.create" in text
    assert "AsyncCompletions.create" in text
    assert "bedrock-runtime" in text
    assert "AUTORAG_COST_LOG" in text
