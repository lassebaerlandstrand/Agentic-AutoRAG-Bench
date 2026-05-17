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


# AutoRAG-side hybrid_rrf probe. Calls ``autorag.nodes.hybridretrieval.hybrid_rrf``
# with a single batched query (the function expects per-query lists of (id_list,
# score_list) tuples wrapped in an outer tuple over retriever modes). Probe
# inputs use distinct scores to avoid tie-handling differences (AutoRAG uses
# pandas ``method="min"`` for rank assignment; our ``_rrf_merge`` ranks by
# enumerate position, which is unstable for ties).
_AUTORAG_HYBRID_RRF_PROBE = textwrap.dedent(
    """
    import json, sys
    from autorag.nodes.hybridretrieval.hybrid_rrf import hybrid_rrf

    payload = json.loads(sys.stdin.read())
    ids = (
        [payload["semantic_ids"]],  # one query
        [payload["lexical_ids"]],
    )
    scores = (
        [payload["semantic_scores"]],
        [payload["lexical_scores"]],
    )
    fused_ids, fused_scores = hybrid_rrf(
        ids,
        scores,
        payload["top_k"],
        weight=payload["weight"],
    )
    with open(sys.argv[1], "w") as f:
        json.dump({"ids": fused_ids[0], "scores": [float(s) for s in fused_scores[0]]}, f)
    """
).strip()


# AutoRAG-side query_decompose parser probe. We don't need a real LLM call
# for equivalence on the parser — we feed synthetic LLM outputs through
# AutoRAG's ``get_query_decompose`` and compare against our framework's
# ``_parse_decompose``. Both should produce identical lists.
_AUTORAG_QUERY_DECOMPOSE_PARSER_PROBE = textwrap.dedent(
    """
    import json, sys
    from autorag.nodes.queryexpansion.query_decompose import get_query_decompose

    payload = json.loads(sys.stdin.read())
    results = []
    for case in payload["cases"]:
        out = get_query_decompose(case["query"], case["answer"])
        results.append(out)
    with open(sys.argv[1], "w") as f:
        json.dump({"results": results}, f)
    """
).strip()


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
def test_query_decompose_parser_matches_autorag(tmp_path) -> None:
    """Framework's ``_parse_decompose`` agrees with AutoRAG's
    ``get_query_decompose`` on every probe input.

    The parser is the part of query_decompose that affects the downstream
    retrieval — wrong parsing = wrong sub-queries = different retrieval
    fingerprint. Equivalence here means the cross-framework comparison is
    grounded in identical sub-query lists for matching LLM outputs.
    """
    from agentic_autorag.engine.pipeline import _parse_decompose

    probes = [
        {"query": "Q1", "answer": "1: Where is Paris?\n2: When built?"},
        {"query": "Q2", "answer": "The question needs no decomposition"},
        {"query": "Q3", "answer": "THE QUESTION NEEDS NO DECOMPOSITION"},
        {"query": "Q4", "answer": "Decompositions:\n1: First sub-q\n2: Second sub-q\n3: Third"},
        {"query": "Q5", "answer": ""},
        {"query": "Q6", "answer": "no colons no structure here"},
        {"query": "Q7", "answer": "1: lone sub-question"},
        {
            "query": "Q8",
            "answer": "Decompositions:\n1: A?\n2: B?\nrandom text\n3: C?",
        },
        # Leading/trailing whitespace — both sides strip().
        {"query": "Q9", "answer": "   The question needs no decomposition   "},
        # Multiple colons in one line — split(":", 1) only on first.
        {
            "query": "Q10",
            "answer": "1: A: B: C\n2: D: E",
        },
        # Tabs and varied spacing.
        {"query": "Q11", "answer": "1:\tspaced sub-q\n2:    indented sub-q"},
        # Decompositions header at the START followed immediately by content.
        {"query": "Q12", "answer": "Decompositions:\n1: only one\n"},
        # No numbered prefix — colons in keywords like "Note:".
        {"query": "Q13", "answer": "Note: The question needs no decomposition"},
        # Single subquestion with non-numeric prefix.
        {"query": "Q14", "answer": "Question: What is X?"},
        # Mixed valid + truly broken lines.
        {"query": "Q15", "answer": "1: ok\nrandom\n2: ok2\nrandom too\n3: ok3"},
    ]

    autorag = _run_autorag_probe(
        _AUTORAG_QUERY_DECOMPOSE_PARSER_PROBE,
        {"cases": probes},
        tmp_path,
    )
    autorag_results = autorag["results"]

    for probe, autorag_out in zip(probes, autorag_results, strict=True):
        bench_out = _parse_decompose(probe["answer"], probe["query"])
        assert bench_out == autorag_out, (
            f"probe {probe['answer']!r}: framework {bench_out} != AutoRAG {autorag_out}. "
            "Parser divergence — investigate before trusting query_decompose rows."
        )


# AutoRAG-side long_context_reorder probe. Instantiate ``LongContextReorder``
# (BasePromptMaker requires a project_dir, otherwise unused for ``_pure``)
# and call ``_pure`` with a minimal prompt template containing the two
# placeholders the function expects. We then parse out the ordered passage
# sequence by splitting on the join delimiter (AutoRAG uses ``"\n\n"``).
_AUTORAG_LONG_CONTEXT_REORDER_PROBE = textwrap.dedent(
    """
    import json, sys
    from autorag.nodes.promptmaker.long_context_reorder import LongContextReorder

    payload = json.loads(sys.stdin.read())
    reorderer = LongContextReorder(project_dir=payload["project_dir"])
    # AutoRAG's _pure substitutes {query} and {retrieved_contents}; we use a
    # marker before retrieved_contents so we can pull the ordered passages
    # back out by splitting the result.
    prompt_template = "MARKER:{retrieved_contents}"
    prompts = reorderer._pure(
        prompt_template,
        [payload["query"]],
        [payload["passages"]],
        [payload["scores"]],
    )
    body = prompts[0].split("MARKER:", 1)[1]
    # AutoRAG joins with double newline.
    ordered = body.split("\\n\\n")
    with open(sys.argv[1], "w") as f:
        json.dump({"ordered_passages": ordered}, f)
    """
).strip()


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
def test_long_context_reorder_passage_order_matches_autorag(tmp_path) -> None:
    """Framework's ``RAGPipeline.prepare_context(long_context_reorder=True)``
    produces the same passage *order* as AutoRAG's ``LongContextReorder._pure``.

    Both append the top-by-score passage to the END of the original
    (unsorted) retrieved list. The join delimiter differs (``"\\n"`` in our
    framework vs ``"\\n\\n"`` in AutoRAG); we compare the passage sequence
    only, not the joined string. That join divergence is documented in the
    paper appendix — changing the framework join would shift grader behaviour
    on every existing config.
    """
    from agentic_autorag.config.models import RuntimeConfig
    from agentic_autorag.engine.pipeline import (
        RAGPipeline,
        RetrievalResult,
        RetrievalTiming,
        RetrievedDocument,
    )

    probes = [
        {
            "name": "monotonic_decreasing",
            "passages": ["P0", "P1", "P2", "P3", "P4"],
            "scores": [0.9, 0.7, 0.5, 0.3, 0.1],
            "expected_top_value": "P0",
        },
        {
            "name": "top_in_middle",
            "passages": ["P0", "P1", "P2", "P3", "P4"],
            "scores": [0.1, 0.3, 0.9, 0.5, 0.2],
            "expected_top_value": "P2",
        },
        {
            "name": "two_passages",
            "passages": ["onlyA", "onlyB"],
            "scores": [0.2, 0.8],
            "expected_top_value": "onlyB",
        },
        {
            # Top is at the end already — duplication does not change order,
            # only doubles the last entry.
            "name": "top_at_end",
            "passages": ["P0", "P1", "P2"],
            "scores": [0.1, 0.2, 0.9],
            "expected_top_value": "P2",
        },
        {
            # Top is at the start — original order ends with non-top item;
            # duplication appends the top.
            "name": "top_at_start",
            "passages": ["P0", "P1", "P2"],
            "scores": [0.9, 0.2, 0.1],
            "expected_top_value": "P0",
        },
        {
            # Large list (10 passages) — exercise sorted vs. original mismatch
            # over a non-trivial span.
            "name": "ten_passages_zigzag",
            "passages": [f"X{i}" for i in range(10)],
            "scores": [0.5, 0.95, 0.3, 0.7, 0.1, 0.8, 0.4, 0.6, 0.2, 0.55],
            "expected_top_value": "X1",
        },
    ]

    for probe in probes:
        autorag = _run_autorag_probe(
            _AUTORAG_LONG_CONTEXT_REORDER_PROBE,
            {
                "project_dir": str(tmp_path),
                "query": "what is the answer",
                "passages": probe["passages"],
                "scores": probe["scores"],
            },
            tmp_path,
        )
        autorag_order = autorag["ordered_passages"]

        # Framework side: instantiate the pipeline with long_context_reorder=True
        # and synthesize a RetrievalResult.
        import asyncio

        from unittest.mock import MagicMock

        cfg = RuntimeConfig(generator_llm="test/model", long_context_reorder=True)
        pipe = RAGPipeline(
            vector_store=MagicMock(),
            graph_store=None,
            config=cfg,
            embedder=MagicMock(),
            index_type=__import__("agentic_autorag.config.models", fromlist=["IndexType"]).IndexType.VECTOR_ONLY,
        )
        result = RetrievalResult(
            documents=[
                RetrievedDocument(id=f"d{i}", text=t, score=s)
                for i, (t, s) in enumerate(zip(probe["passages"], probe["scores"], strict=True))
            ],
            timing=RetrievalTiming(),
            expansion_cost={"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        context, _ = asyncio.run(pipe.prepare_context("q", result))
        bench_order = context.split("\n")

        assert bench_order == autorag_order, (
            f"probe {probe['name']!r}: framework order {bench_order} != "
            f"AutoRAG order {autorag_order}. AutoRAG semantics: append top-by-score "
            f"to the original list; our prepare_context must mirror that."
        )
        # Sanity: the duplicate at the end is the top-scored passage.
        assert bench_order[-1] == probe["expected_top_value"]


@pytest.mark.skipif(_autorag_python() is None, reason="AUTORAG_PYTHON not set")
def test_hybrid_rrf_top_k_matches_autorag(tmp_path) -> None:
    """Framework's ``RAGPipeline._rrf_merge`` agrees with AutoRAG's ``hybrid_rrf``
    on top-K ranking for distinct-score probes.

    The formulas are mathematically equivalent:
      - AutoRAG: ``1 / (r + rrf_k)`` where ``r`` is 1-indexed rank (from
        ``rank_df.rank(ascending=False, method="min")``).
      - Framework: ``1.0 / (k + rank + 1)`` where ``rank`` is 0-indexed from
        ``enumerate(list)``.
    Substituting ``r = rank + 1`` shows them identical when ties are absent.
    This test exercises three overlap regimes (full, partial, disjoint) and
    asserts identical top-3 ranking.
    """
    from agentic_autorag.engine.pipeline import RAGPipeline

    probes = [
        {
            "name": "partial_overlap",
            "semantic_ids": ["a", "b", "c", "d"],
            "semantic_scores": [0.9, 0.7, 0.5, 0.3],
            "lexical_ids": ["b", "c", "e", "f"],
            "lexical_scores": [0.95, 0.6, 0.4, 0.2],
        },
        {
            "name": "full_overlap",
            "semantic_ids": ["x", "y", "z"],
            "semantic_scores": [0.9, 0.5, 0.2],
            "lexical_ids": ["z", "y", "x"],
            "lexical_scores": [0.8, 0.4, 0.1],
        },
        {
            "name": "disjoint",
            "semantic_ids": ["a", "b", "c"],
            "semantic_scores": [0.9, 0.6, 0.3],
            "lexical_ids": ["x", "y", "z"],
            "lexical_scores": [0.9, 0.6, 0.3],
        },
        {
            # Reverse ordering on one side — common when BM25 and vector
            # disagree sharply (e.g. lexical match vs semantic paraphrase).
            "name": "reverse_lexical",
            "semantic_ids": ["p", "q", "r", "s"],
            "semantic_scores": [0.95, 0.75, 0.55, 0.35],
            "lexical_ids": ["s", "r", "q", "p"],
            "lexical_scores": [0.96, 0.76, 0.56, 0.36],
        },
        {
            # Single overlap at the top — the one shared doc should win.
            "name": "single_overlap_top",
            "semantic_ids": ["a", "b", "c", "d", "e"],
            "semantic_scores": [0.99, 0.7, 0.5, 0.3, 0.1],
            "lexical_ids": ["a", "x", "y", "z", "w"],
            "lexical_scores": [0.97, 0.65, 0.45, 0.25, 0.05],
        },
        {
            # Single overlap at the bottom — much weaker fusion signal.
            "name": "single_overlap_bottom",
            "semantic_ids": ["a", "b", "c", "d", "e"],
            "semantic_scores": [0.99, 0.7, 0.5, 0.3, 0.1],
            "lexical_ids": ["x", "y", "z", "w", "e"],
            "lexical_scores": [0.97, 0.65, 0.45, 0.25, 0.05],
        },
        {
            # Tiny score gap — sensitive to the exact rrf_k value.
            "name": "tight_score_gap",
            "semantic_ids": ["a", "b", "c"],
            "semantic_scores": [0.501, 0.500, 0.499],
            "lexical_ids": ["c", "b", "a"],
            "lexical_scores": [0.601, 0.600, 0.599],
        },
        {
            # Large list (10 items each) with partial overlap so RRF ranks
            # diverge. Pure-disjoint long lists tie pairwise on RRF score
            # (s_i and l_i both end up at rank i+1 on their own retriever),
            # and tie-breaking depends on pandas DataFrame index order vs.
            # our dict-insertion order — that's an undefined-behaviour gap
            # for disjoint top-3, not a formula bug. Partial overlap forces
            # at least one fused-rank dominance.
            "name": "long_list_partial_overlap",
            "semantic_ids": [f"s{i}" for i in range(10)],
            "semantic_scores": [round(1.0 - 0.07 * i, 4) for i in range(10)],
            "lexical_ids": ["s2", "s5", "s8"] + [f"l{i}" for i in range(7)],
            "lexical_scores": [
                0.98, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05
            ],
        },
    ]

    for probe in probes:
        autorag = _run_autorag_probe(
            _AUTORAG_HYBRID_RRF_PROBE,
            {
                "semantic_ids": probe["semantic_ids"],
                "semantic_scores": probe["semantic_scores"],
                "lexical_ids": probe["lexical_ids"],
                "lexical_scores": probe["lexical_scores"],
                "top_k": 3,
                "weight": 60,
            },
            tmp_path,
        )
        autorag_top = autorag["ids"][:3]

        list_a = [{"id": i, "score": s, "text": ""} for i, s in zip(probe["semantic_ids"], probe["semantic_scores"], strict=True)]
        list_b = [{"id": i, "score": s, "text": ""} for i, s in zip(probe["lexical_ids"], probe["lexical_scores"], strict=True)]
        bench_merged = RAGPipeline._rrf_merge(list_a, list_b, k=60)
        bench_top = [d["id"] for d in bench_merged[:3]]

        assert autorag_top == bench_top, (
            f"probe {probe['name']!r}: AutoRAG top-3 {autorag_top} != bench top-3 {bench_top}. "
            "RRF formula divergence — investigate before trusting bm25_vector_fusion='rrf' "
            "rows in the paper."
        )
