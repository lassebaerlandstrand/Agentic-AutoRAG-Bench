"""Auto-generate AutoRAG's ``config.yaml`` strictly mirroring our ``SearchSpace``.

AutoRAG searches a richer native space (``passage_compressor``, ``prompt_maker``
template tuning, LongLLMLingua, additional ``query_expansion`` modules). Letting
AutoRAG enumerate those would give it search dimensions Random / Bayesian /
Agentic don't have. This module restricts AutoRAG to **only** the dimensions in
our ``SearchSpace`` and freezes everything else at sensible defaults.

Every excluded dimension is recorded in ``translation_notes.json`` next to the
generated config; the paper appendix lists them verbatim so reviewers can verify
the mirror is honest.
"""

from __future__ import annotations

from agentic_autorag.config.models import IndexType, NumericRange, SearchSpace

MCQ_PROMPT_TEMPLATE = (
    "Answer the following multiple-choice question by giving the text of the correct option.\n"
    "\n"
    "Context:\n"
    "{retrieved_contents}\n"
    "\n"
    "{query}\n"
    "\n"
    "Answer with only the text of the correct option, nothing else."
)

FREE_FORM_PROMPT_TEMPLATE = (
    "Use the following context to answer the question.\n"
    "\n"
    "Context:\n"
    "{retrieved_contents}\n"
    "\n"
    "Question: {query}\n"
    "Answer with only the answer itself: no explanation, no quotes."
)

# Explicit reranker mapping. The previous heuristic (substring match) was
# brittle — silent fallback to ``sentence_transformer_reranker`` mis-loaded
# Qwen3-Reranker (architecture mismatch). We now require an explicit entry
# and raise on unknown rerankers so drift is caught at config-generation time.
RERANKER_MODULE_MAP: dict[str, str] = {
    "BAAI/bge-reranker-v2-m3": "flag_embedding_reranker",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "sentence_transformer_reranker",
    "Alibaba-NLP/gte-reranker-modernbert-base": "sentence_transformer_reranker",
    "mixedbread-ai/mxbai-rerank-xsmall-v1": "sentence_transformer_reranker",
    "mixedbread-ai/mxbai-rerank-base-v2": "sentence_transformer_reranker",
}

# Discretization grid sizes for AutoRAG's enumeration. Higher → more faithful
# to our continuous space, but multiplicatively more pipeline runs per node.
CHUNK_SIZE_GRID_N = 5
CHUNK_OVERLAP_GRID_N = 3
TOP_K_GRID_N = 5
RERANKER_TOP_K_GRID_N = 3
TEMPERATURE_GRID_N = 1  # our search space pins temperature; one value is enough


def _discretize_int(r: NumericRange, n: int) -> list[int]:
    lo, hi = int(r.min), int(r.max)
    if lo == hi or n <= 1:
        return [lo] if lo == hi else [lo, hi]
    step = (hi - lo) / (n - 1)
    return sorted({int(round(lo + i * step)) for i in range(n)})


def _discretize_float(r: NumericRange, n: int, *, precision: int = 2) -> list[float]:
    lo, hi = float(r.min), float(r.max)
    if lo == hi or n <= 1:
        return [round(lo, precision)] if lo == hi else [round(lo, precision), round(hi, precision)]
    step = (hi - lo) / (n - 1)
    return sorted({round(lo + i * step, precision) for i in range(n)})


def _reranker_module_for(model: str) -> str:
    if model not in RERANKER_MODULE_MAP:
        raise KeyError(
            f"No AutoRAG reranker module mapping for {model!r}. "
            f"Add it to RERANKER_MODULE_MAP in native_config.py before running the AutoRAG baseline."
        )
    return RERANKER_MODULE_MAP[model]


def generate_autorag_config(
    search_space: SearchSpace,
    *,
    qa_variant: str,
) -> tuple[dict, dict]:
    """Generate AutoRAG yaml dict + translation notes.

    Parameters
    ----------
    search_space:
        Source-of-truth ``SearchSpace``. Every present dimension is mirrored;
        no other dimensions appear in the AutoRAG config.
    qa_variant:
        ``"mcq"`` registers the ``mcq_accuracy`` metric and the MCQ prompt
        template. ``"ragas"`` uses ``g_eval`` and the free-form template.
    """
    if qa_variant not in {"mcq", "ragas"}:
        raise ValueError(f"qa_variant must be 'mcq' or 'ragas', got {qa_variant!r}")

    ss = search_space
    chunk_sizes = _discretize_int(ss.chunking.chunk_token_size, CHUNK_SIZE_GRID_N)
    chunk_overlaps = _discretize_int(ss.chunking.chunk_token_overlap, CHUNK_OVERLAP_GRID_N)
    top_ks = _discretize_int(ss.top_k, TOP_K_GRID_N)
    reranker_top_ks = _discretize_int(ss.reranker.top_n, RERANKER_TOP_K_GRID_N)
    temperatures = _discretize_float(ss.temperature, TEMPERATURE_GRID_N)

    chunker_modules = [
        {
            "module_type": "llama_index_chunk",
            "chunk_method": ["token" if strategy == "recursive" else strategy],
            "chunk_size": chunk_sizes,
            "chunk_overlap": chunk_overlaps,
        }
        for strategy in ss.chunking.strategies
    ]

    retrieval_modules: list[dict] = []
    if IndexType.VECTOR_ONLY in ss.index_types:
        retrieval_modules.append({"module_type": "vectordb", "embedding_model": list(ss.embedding_models)})
    if IndexType.HYBRID_BM25_VECTOR in ss.index_types:
        # AutoRAG's ``hybrid_cc.weight`` is BM25's contribution; ours is the
        # vector's. Translation: ``weight = 1.0 - hybrid_alpha``.
        bm25_lo = round(1.0 - ss.hybrid_alpha.max, 4)
        bm25_hi = round(1.0 - ss.hybrid_alpha.min, 4)
        retrieval_modules.append(
            {"module_type": "hybrid_cc", "weight_range": [bm25_lo, bm25_hi], "normalize_method": ["mm", "tmm"]}
        )

    reranker_modules: list[dict] = []
    for model in ss.reranker.models:
        if model == "none":
            reranker_modules.append({"module_type": "pass_passage_reranker"})
        else:
            reranker_modules.append({"module_type": _reranker_module_for(model), "model_name": model})

    query_expansion_modules: list[dict] = []
    for qe in ss.query_expansion:
        if qe == "none":
            query_expansion_modules.append({"module_type": "pass_query_expansion"})
        elif qe == "hyde":
            query_expansion_modules.append({"module_type": "hyde"})
        elif qe == "multi_query":
            query_expansion_modules.append({"module_type": "multi_query_expansion"})
        else:
            raise ValueError(f"Unknown query_expansion {qe!r} — add an explicit AutoRAG mapping")

    generator_modules = [
        {"module_type": "llama_index_llm", "llm": llm, "temperature": temperatures}
        for llm in ss.llm_models
    ]

    if qa_variant == "mcq":
        gen_metrics = ["mcq_accuracy"]
        prompt_template = MCQ_PROMPT_TEMPLATE
    else:
        gen_metrics = ["g_eval"]
        prompt_template = FREE_FORM_PROMPT_TEMPLATE

    config = {
        "node_lines": [
            {
                "node_line_name": "pre_retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "query_expansion",
                        "modules": query_expansion_modules,
                        "strategy": {"metrics": ["retrieval_f1"], "top_k": top_ks},
                    }
                ],
            },
            {
                "node_line_name": "retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "modules": retrieval_modules,
                        "strategy": {"metrics": ["retrieval_f1", "retrieval_recall"], "top_k": top_ks},
                    }
                ],
            },
            {
                "node_line_name": "post_retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "passage_reranker",
                        "modules": reranker_modules,
                        "strategy": {"metrics": ["retrieval_f1"], "top_k": reranker_top_ks},
                    },
                    {
                        "node_type": "prompt_maker",
                        "modules": [{"module_type": "fstring", "prompt": [prompt_template]}],
                        "strategy": {"metrics": gen_metrics},
                    },
                    {
                        "node_type": "generator",
                        "modules": generator_modules,
                        "strategy": {"metrics": gen_metrics},
                    },
                ],
            },
        ]
    }

    excluded_dimensions = [
        "passage_compressor (tree_summarize / refine / longllmlingua)",
        "passage_filter (similarity_threshold_cutoff / percentile_cutoff / recency_filter)",
        "prompt_maker template tuning beyond the single fstring",
        "query_expansion modules outside {none, hyde, multi_query} (e.g. query_decompose)",
    ]
    if qa_variant == "mcq":
        excluded_dimensions.append("g_eval / sem_score / bleu / rouge generation metrics — replaced by mcq_accuracy")

    notes = {
        "qa_variant": qa_variant,
        "excluded_dimensions": excluded_dimensions,
        "discretization": {
            "chunk_size": chunk_sizes,
            "chunk_overlap": chunk_overlaps,
            "top_k": top_ks,
            "reranker_top_k": reranker_top_ks,
            "temperature": temperatures,
        },
        "reranker_module_map": {m: RERANKER_MODULE_MAP.get(m, "pass_passage_reranker") for m in ss.reranker.models},
        "hybrid_alpha_convention": "AutoRAG's hybrid_cc.weight is BM25's; we pass weight = 1 - hybrid_alpha",
    }
    return config, notes
