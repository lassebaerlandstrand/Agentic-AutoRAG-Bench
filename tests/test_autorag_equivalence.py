"""Behavioral equivalence between AutoRAG's HF loaders and our framework's loaders.

The benchmark's fairness rests on the assumption that a config translated from
AutoRAG's ``extracted_sample.yaml`` to our ``TrialConfig`` produces the same
retrieval and reranking behavior on both sides. Name-level checks
(``test_autorag_translator.py``) catch schema drift but cannot detect
implementation-level drift: AutoRAG's ``flag_embedding_reranker`` wraps
``FlagEmbedding.FlagReranker`` while our framework wraps
``sentence_transformers.CrossEncoder`` — both load the same HF weights but may
post-process scores differently (sigmoid vs. raw logits, batch padding,
normalization defaults). The same risk applies to embedding loaders: both
sides nominally use the same HF model, but llama-index's
``HuggingFaceEmbedding`` and direct ``sentence_transformers`` use can diverge
on normalization or pooling defaults.

These tests run the same inputs through both stacks and check the outputs
agree (cosine ≥ 0.999 for embeddings, Spearman ≥ 0.9 for reranker orderings).
Two outcomes lock down trust:

  - PASS → the AutoRAG row's translated winning config produces the same
    retrieval/reranking as the framework methods on the same corpus, so any
    score gap reflects optimizer quality, not implementation drift.
  - FAIL → translation surfaces a behavioral mismatch we must either correct
    (e.g. switch the framework's reranker class to match AutoRAG's) or
    document as a known limitation in the paper appendix.

The tests shell out to the AutoRAG venv via the ``AUTORAG_PYTHON`` env var,
mirroring ``qa_ragas.py``. They auto-skip when the venv isn't configured so
they don't block fast iteration. Model weights are downloaded on first run
(MiniLM is ~90 MB, the reranker ~1.1 GB), then cached by HuggingFace Hub.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap

import pytest

# Equivalence threshold for L2-normalized embeddings: 1.0 would require bit
# identity, which floating-point GEMM differences across torch versions can
# break. 0.999 is tight enough to flag normalization / pooling / truncation
# divergences without flagging numerical noise.
EMBEDDING_COSINE_MIN = 0.999

# Top-k Spearman threshold for rerankers: scores need not be identical (one
# side may sigmoid the logits) but the *ranking* must agree to within a
# coefficient of 0.9 across documents, otherwise different documents end up
# at the top of the reranker output and the downstream generator sees
# different evidence.
RERANKER_SPEARMAN_MIN = 0.9

# Single representative model per loader path. Adding more models linearly
# inflates first-run weight-download cost; the loader path is the same
# regardless of model name, so one is enough for the smoke.
SMOKE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SMOKE_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Fixed probe data — five short factual sentences exercise tokenization,
# pooling, and similarity without needing a real corpus.
EMBEDDING_PROBE_TEXTS = [
    "The Eiffel Tower is in Paris, France.",
    "Photosynthesis converts light energy into chemical energy.",
    "Mount Everest is the tallest mountain above sea level.",
    "Shakespeare wrote Hamlet around 1600.",
    "The mitochondrion is the powerhouse of the cell.",
]

# Probe pairs for the reranker: same query with five candidate passages
# spanning relevant/irrelevant. A reranker must agree on the *relative*
# ranking even if its raw scores differ.
RERANKER_PROBE_QUERY = "Where is the Eiffel Tower located?"
RERANKER_PROBE_DOCS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Paris is the capital and most populous city of France.",
    "The Statue of Liberty is on Liberty Island in New York Harbor.",
    "Photosynthesis converts light energy into chemical energy.",
    "Mount Everest is the tallest mountain above sea level.",
]


def _autorag_python() -> str | None:
    return os.environ.get("AUTORAG_PYTHON")


# AutoRAG-side embedding probe. llama-index's logging is unconditional and
# lands on stdout, so the probe writes JSON to a path passed as argv[1]
# instead of trying to share stdout with log lines.
_AUTORAG_EMBEDDING_PROBE = textwrap.dedent(
    """
    import json, sys
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    payload = json.loads(sys.stdin.read())
    embed = HuggingFaceEmbedding(model_name=payload["model"])
    vectors = embed.get_text_embedding_batch(payload["texts"])
    with open(sys.argv[1], "w") as f:
        json.dump({"vectors": vectors}, f)
    """
).strip()


# AutoRAG-side reranker probe: load ``BAAI/bge-reranker-v2-m3`` via
# FlagEmbedding (matches the ``flag_embedding_reranker`` module) OR via
# sentence-transformers CrossEncoder (matches ``sentence_transformer_reranker``).
# Same out-file convention as the embedding probe.
_AUTORAG_RERANKER_PROBE = textwrap.dedent(
    """
    import json, sys

    payload = json.loads(sys.stdin.read())
    loader = payload["loader"]
    model = payload["model"]
    pairs = [[payload["query"], doc] for doc in payload["docs"]]

    if loader == "flag_embedding":
        from FlagEmbedding import FlagReranker
        scorer = FlagReranker(model, use_fp16=False)
        scores = scorer.compute_score(pairs, normalize=False)
    elif loader == "sentence_transformer":
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model)
        scores = ce.predict(pairs).tolist()
    else:
        raise SystemExit("unknown loader: " + loader)
    with open(sys.argv[1], "w") as f:
        json.dump({"scores": [float(s) for s in scores]}, f)
    """
).strip()


def _run_autorag_probe(script: str, payload: dict, tmp_path) -> dict:
    python = _autorag_python()
    if not python:
        raise RuntimeError("AUTORAG_PYTHON not set")
    out_path = tmp_path / "probe_out.json"
    result = subprocess.run(
        [python, "-c", script, str(out_path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"AutoRAG probe exited rc={result.returncode}; stderr=\n{result.stderr.rstrip()}"
        )
    return json.loads(out_path.read_text())


def _cosine(u, v) -> float:
    import numpy as np

    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _spearman(xs, ys) -> float:
    """Spearman rank correlation without scipy. Both inputs are score lists."""
    import numpy as np

    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    n = len(a)
    if n < 2:
        return 1.0
    sa, sb = ra.std(), rb.std()
    if sa == 0.0 or sb == 0.0:
        return 1.0 if (sa == 0.0 and sb == 0.0) else 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
def test_huggingface_embedding_equivalence_minilm(tmp_path) -> None:
    """AutoRAG's HuggingFaceEmbedding agrees with our SentenceTransformer path
    on every probe string to cosine ≥ 0.999.

    Catches normalization, pooling, and truncation divergences that name-only
    checks miss.
    """
    from sentence_transformers import SentenceTransformer

    autorag = _run_autorag_probe(
        _AUTORAG_EMBEDDING_PROBE,
        {"model": SMOKE_EMBEDDING_MODEL, "texts": EMBEDDING_PROBE_TEXTS},
        tmp_path,
    )
    autorag_vecs = autorag["vectors"]

    bench_model = SentenceTransformer(SMOKE_EMBEDDING_MODEL)
    bench_vecs = bench_model.encode(EMBEDDING_PROBE_TEXTS).tolist()

    assert len(autorag_vecs) == len(bench_vecs) == len(EMBEDDING_PROBE_TEXTS)
    for i, (av, bv) in enumerate(zip(autorag_vecs, bench_vecs, strict=True)):
        cos = _cosine(av, bv)
        assert cos >= EMBEDDING_COSINE_MIN, (
            f"text {i!r} cosine {cos:.6f} < {EMBEDDING_COSINE_MIN} — AutoRAG and "
            f"framework loaders diverge for {SMOKE_EMBEDDING_MODEL!r}"
        )


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
def test_sentence_transformer_reranker_equivalence(tmp_path) -> None:
    """AutoRAG's ``sentence_transformer_reranker`` agrees with our framework's
    CrossEncoder on probe ordering to Spearman ≥ 0.9.

    This is the lower-risk reranker mapping (both sides nominally use
    ``sentence_transformers.CrossEncoder``); a failure here means the test
    setup itself is broken.
    """
    from sentence_transformers import CrossEncoder

    autorag = _run_autorag_probe(
        _AUTORAG_RERANKER_PROBE,
        {
            "loader": "sentence_transformer",
            "model": SMOKE_RERANKER_MODEL,
            "query": RERANKER_PROBE_QUERY,
            "docs": RERANKER_PROBE_DOCS,
        },
        tmp_path,
    )
    autorag_scores = autorag["scores"]

    ce = CrossEncoder(SMOKE_RERANKER_MODEL)
    pairs = [[RERANKER_PROBE_QUERY, doc] for doc in RERANKER_PROBE_DOCS]
    bench_scores = ce.predict(pairs).tolist()

    rho = _spearman(autorag_scores, bench_scores)
    assert rho >= RERANKER_SPEARMAN_MIN, (
        f"sentence_transformer reranker Spearman {rho:.3f} < {RERANKER_SPEARMAN_MIN}. "
        f"autorag={autorag_scores} bench={bench_scores}"
    )


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
def test_flag_embedding_reranker_vs_framework_crossencoder(tmp_path) -> None:
    """AutoRAG maps ``BAAI/bge-reranker-v2-m3`` to ``flag_embedding_reranker``
    (FlagEmbedding's FlagReranker), but our framework loads every reranker
    through ``sentence_transformers.CrossEncoder``. Verify the orderings still
    agree (Spearman ≥ 0.9): the underlying weights are the same HF checkpoint,
    so cross-class score divergence at the *ordering* level would be a real
    fairness bug.

    Expected outcomes:
      - Pass: scores may differ in absolute value (raw logits vs. potentially
        sigmoid'd), but rankings agree → translation is honest.
      - Fail: switch the AutoRAG mapping for bge-reranker-v2-m3 to
        ``sentence_transformer_reranker`` so both sides use the same class,
        and document in the appendix.
    """
    from sentence_transformers import CrossEncoder

    bge = "BAAI/bge-reranker-v2-m3"
    autorag = _run_autorag_probe(
        _AUTORAG_RERANKER_PROBE,
        {
            "loader": "flag_embedding",
            "model": bge,
            "query": RERANKER_PROBE_QUERY,
            "docs": RERANKER_PROBE_DOCS,
        },
        tmp_path,
    )
    autorag_scores = autorag["scores"]

    ce = CrossEncoder(bge)
    pairs = [[RERANKER_PROBE_QUERY, doc] for doc in RERANKER_PROBE_DOCS]
    bench_scores = ce.predict(pairs).tolist()

    rho = _spearman(autorag_scores, bench_scores)
    assert rho >= RERANKER_SPEARMAN_MIN, (
        f"flag_embedding vs CrossEncoder Spearman {rho:.3f} < {RERANKER_SPEARMAN_MIN}. "
        f"autorag(flag_embedding)={autorag_scores} bench(CrossEncoder)={bench_scores}. "
        "Either switch native_config.RERANKER_MODULE_MAP['BAAI/bge-reranker-v2-m3'] to "
        "'sentence_transformer_reranker' or document the divergence."
    )


# AutoRAG-side bedrock_converse probe. Issues a single non-streaming chat
# completion via ``BedrockConverse.complete`` (the same path AutoRAG's
# llama_index_llm node walks). Writes a JSON payload with the response text.
_AUTORAG_BEDROCK_CONVERSE_PROBE = textwrap.dedent(
    """
    import json, sys
    from llama_index.llms.bedrock_converse import BedrockConverse

    payload = json.loads(sys.stdin.read())
    llm = BedrockConverse(
        model=payload["model"],
        region_name=payload["region"],
        temperature=0.0,
        max_tokens=64,
    )
    resp = llm.complete(payload["prompt"])
    with open(sys.argv[1], "w") as f:
        json.dump({"text": str(resp).strip()}, f)
    """
).strip()


# Cheapest bedrock entry in the paper search space — minimises live-API spend
# for the smoke. The model selection mirrors the production llm_models list
# so we exercise the same boto3 client_creation path the bench actually uses.
SMOKE_BEDROCK_MODEL = "us.meta.llama3-1-8b-instruct-v1:0"
BEDROCK_PROBE_PROMPT = (
    "Reply with exactly one word and nothing else: the capital of France."
)


def _aws_creds_present() -> bool:
    return (
        bool(os.environ.get("AWS_ACCESS_KEY_ID"))
        and bool(os.environ.get("AWS_SECRET_ACCESS_KEY"))
        and bool(os.environ.get("AWS_REGION_NAME"))
    )


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
@pytest.mark.skipif(not _aws_creds_present(), reason="AWS bedrock creds not set in env")
def test_bedrock_converse_vs_litellm_bedrock_smoke(tmp_path) -> None:
    """End-to-end smoke: same prompt + model on AutoRAG's ``BedrockConverse``
    (the modern path patched into ``autorag.generator_models`` — see
    scripts/autorag_patches.py) and the framework's litellm ``bedrock/*``
    route must both return non-empty text and broadly agree.

    The two stacks ultimately call the same AWS Bedrock Converse endpoint
    with the same model id, so identical outputs at T=0 are the natural
    expectation. We assert weakly (non-empty + 1-token overlap) so harmless
    formatting jitter (trailing punctuation, "Paris." vs "Paris") doesn't
    flap the test — but a hard *divergence* (one side returns gibberish or
    refuses) still fails. Burns one short Bedrock call per side; gated on
    AWS env creds so dev machines without bedrock access skip cleanly.
    """
    region = os.environ["AWS_REGION_NAME"]

    autorag = _run_autorag_probe(
        _AUTORAG_BEDROCK_CONVERSE_PROBE,
        {"model": SMOKE_BEDROCK_MODEL, "region": region, "prompt": BEDROCK_PROBE_PROMPT},
        tmp_path,
    )
    autorag_text = autorag["text"]
    assert autorag_text, "AutoRAG BedrockConverse returned empty text"

    # Framework side: same call via litellm.completion. Sync API to keep the
    # test simple — the bench's async wrapper is the same code path under the
    # hood, just awaited.
    import litellm

    resp = litellm.completion(
        model=f"bedrock/{SMOKE_BEDROCK_MODEL}",
        messages=[{"role": "user", "content": BEDROCK_PROBE_PROMPT}],
        temperature=0.0,
        max_tokens=64,
    )
    bench_text = resp.choices[0].message.content.strip()
    assert bench_text, "litellm bedrock returned empty text"

    # Weak agreement: at least one tokenised word in common after normalisation.
    # The prompt asks for a one-word answer, so the intersection should contain
    # at least "paris" (or the model's chosen city). A divergence like one side
    # refusing ("I cannot answer") would have zero intersection with a city
    # name — that's the failure mode this guards against.
    autorag_words = {w.strip(".,!?\"'").lower() for w in autorag_text.split()}
    bench_words = {w.strip(".,!?\"'").lower() for w in bench_text.split()}
    overlap = autorag_words & bench_words
    assert overlap, (
        f"BedrockConverse vs litellm produced disjoint answers: "
        f"autorag={autorag_text!r} bench={bench_text!r}. "
        f"Either the model_id is being mis-translated on one side, or one "
        f"stack is silently refusing — investigate before trusting AutoRAG "
        f"bedrock rows in the paper."
    )
