"""Verify every model in a paper config is loadable / callable.

Walks ``search_space`` in the YAML, plus ``agent.*_model`` and
``query_expansion.models``, and probes each model end-to-end:

- LLMs: one ``litellm.completion`` call with ``Say 'hello'`` (max_tokens=8).
- Embedders: load via ``sentence_transformers.SentenceTransformer`` and encode
  one short string. Truncates a chunk-sized 600-token probe to surface the
  ``max_seq_length`` mismatch that would silently bias retrieval scoring.
- Rerankers: load via ``sentence_transformers.CrossEncoder`` (works for
  bge-reranker-v2-m3 even though native_config routes it through
  ``flag_embedding_reranker`` — CrossEncoder is FlagEmbedding's fallback).

Run from the framework venv (litellm + sentence_transformers required):

    .venv/bin/python scripts/preflight_search_space.py configs/hotpot_paper_project.yaml

Exits non-zero if any model fails; per-model pass/fail prints inline so a
failure mid-batch doesn't hide successes that already ran.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    seconds: float


@dataclass
class Bucket:
    label: str
    results: list[Result] = field(default_factory=list)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _collect_llms(cfg: dict) -> list[str]:
    """Unique LLMs across agent.* and search_space stages."""
    seen: dict[str, None] = {}
    for key in ("optimizer_model", "examiner_model", "judge_model"):
        v = cfg.get("agent", {}).get(key)
        if v:
            seen.setdefault(v, None)
    ss = cfg.get("search_space", {})
    for stage in ("generator", "query_expansion", "passage_compressor"):
        for m in ss.get(stage, {}).get("models", []) or []:
            seen.setdefault(m, None)
    return list(seen.keys())


def _free_torch_memory() -> None:
    """Drop GPU/CPU caches between model loads so 9 embedders don't OOM."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _probe_llm(model: str, timeout: float = 30.0) -> Result:
    """Send the cheapest possible chat completion that still works on
    reasoning-capable models. ``max_tokens=128`` because gpt-oss / nemotron /
    o-series spend tokens on hidden reasoning before emitting visible output;
    at max_tokens=8 they truncate to empty content. 128 is enough headroom
    for a "say hello" without bloating cost.

    Reasoning-capable models (o-series, gpt-5-*) reject any ``temperature``
    override on Azure, but ``drop_params=True`` (set below) strips the
    unsupported arg silently.
    """
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True

    t0 = time.perf_counter()
    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'hello'."}],
            "max_tokens": 128,
            "temperature": 1.0,
            "timeout": timeout,
        }
        resp = litellm.completion(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        dt = time.perf_counter() - t0
        if not text:
            return Result(model, False, "empty response", dt)
        return Result(model, True, f"reply={text[:40]!r}", dt)
    except Exception as e:  # noqa: BLE001 — preflight wants the message, not the type
        dt = time.perf_counter() - t0
        return Result(model, False, f"{type(e).__name__}: {str(e)[:200]}", dt)


def _probe_embedder(model: str, chunk_size_tokens: int = 512) -> Result:
    """Load + encode a short sentence; warn if max_seq_length < chunk_size.

    ``chunk_size_tokens`` mirrors search_space.chunking.chunk_token_size. An
    embedder whose ``max_seq_length`` is below that value will silently
    truncate every chunk at retrieval time, biasing its score downward — the
    classic MiniLM (max=256) gotcha.
    """
    t0 = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model, trust_remote_code=True)
        max_seq = int(getattr(st, "max_seq_length", 0) or 0)
        vec = st.encode("This is a preflight probe.", show_progress_bar=False)
        dim = int(vec.shape[-1])
        del st
        _free_torch_memory()
        dt = time.perf_counter() - t0
        note = f"dim={dim}, max_seq={max_seq}"
        if max_seq and max_seq < chunk_size_tokens:
            note += f" ⚠ TRUNCATES at chunk_size={chunk_size_tokens}"
        return Result(model, True, note, dt)
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return Result(model, False, f"{type(e).__name__}: {str(e)[:200]}", dt)


def _probe_reranker(model: str) -> Result:
    """Score one (query, passage) pair via CrossEncoder.

    CrossEncoder handles every reranker in our search space — bge-reranker-v2-m3
    works through it (it IS a cross-encoder architecture; FlagEmbedding is
    just a wrapper). Native_config routes bge to ``flag_embedding_reranker``
    at run time, but the underlying model load works the same.
    """
    if model == "none":
        return Result(model, True, "pass-through (no model to load)", 0.0)
    t0 = time.perf_counter()
    try:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(model, trust_remote_code=True)
        scores = ce.predict([("query", "candidate passage about the query.")])
        score = float(scores[0]) if hasattr(scores, "__iter__") else float(scores)
        del ce
        _free_torch_memory()
        dt = time.perf_counter() - t0
        return Result(model, True, f"score={score:.3f}", dt)
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return Result(model, False, f"{type(e).__name__}: {str(e)[:200]}", dt)


def _print_bucket(bucket: Bucket) -> tuple[int, int]:
    print(f"\n=== {bucket.label} ({len(bucket.results)} entries) ===")
    n_ok = sum(1 for r in bucket.results if r.ok)
    n_fail = len(bucket.results) - n_ok
    for r in bucket.results:
        mark = "✓" if r.ok else "✗"
        print(f"  {mark} {r.name:60s} {r.seconds:6.1f}s  {r.detail}")
    print(f"  → {n_ok}/{len(bucket.results)} passed")
    return n_ok, n_fail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="paper config YAML")
    parser.add_argument(
        "--skip",
        choices=("llms", "embedders", "rerankers"),
        action="append",
        default=[],
        help="skip a category (repeatable)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Agentic-AutoRAG" / ".env",
        help="path to .env (default: sibling Agentic-AutoRAG/.env)",
    )
    args = parser.parse_args()

    if args.env.exists():
        load_dotenv(args.env, override=False)
        print(f"Loaded env from {args.env}")
    else:
        print(f"WARN: env file {args.env} not found; relying on shell environment")

    cfg = _load_config(args.config)
    chunk_size = int(
        cfg.get("search_space", {})
        .get("chunking", {})
        .get("chunk_token_size", {})
        .get("values", [512])[0]
    )

    buckets: list[Bucket] = []

    if "llms" not in args.skip:
        llms = _collect_llms(cfg)
        b = Bucket(label="LLMs (litellm.completion: \"Say 'hello'\", max_tokens=128)")
        for m in llms:
            print(f"  probing LLM: {m} ...", flush=True)
            b.results.append(_probe_llm(m))
        buckets.append(b)

    if "embedders" not in args.skip:
        embedders = list(cfg.get("search_space", {}).get("embedding", {}).get("models", []) or [])
        b = Bucket(
            label=f"Embedders (SentenceTransformer.encode; chunk_size={chunk_size} for truncation check)"
        )
        for m in embedders:
            print(f"  probing embedder: {m} ...", flush=True)
            b.results.append(_probe_embedder(m, chunk_size_tokens=chunk_size))
        buckets.append(b)

    if "rerankers" not in args.skip:
        rerankers = list(cfg.get("search_space", {}).get("reranker", {}).get("models", []) or [])
        b = Bucket(label="Rerankers (CrossEncoder.predict on a single pair)")
        for m in rerankers:
            print(f"  probing reranker: {m} ...", flush=True)
            b.results.append(_probe_reranker(m))
        buckets.append(b)

    print("\n" + "=" * 64)
    print("PREFLIGHT SUMMARY")
    print("=" * 64)
    total_ok = total_fail = 0
    for b in buckets:
        n_ok, n_fail = _print_bucket(b)
        total_ok += n_ok
        total_fail += n_fail

    print("\n" + "=" * 64)
    print(f"TOTAL: {total_ok} passed, {total_fail} failed")
    print("=" * 64)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
