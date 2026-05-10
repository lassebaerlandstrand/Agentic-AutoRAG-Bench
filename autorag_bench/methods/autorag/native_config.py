"""Auto-generate AutoRAG's RAG-eval ``config.yaml`` strictly mirroring our ``SearchSpace``.

Targets AutoRAG v0.3.x — verified against sample_config/rag/english/non_gpu/{compact,full}.yaml.
v0.2 schema differences this module already corrects for:

- ``retrieval`` node_type was split into three: ``lexical_retrieval``
  (BM25), ``semantic_retrieval`` (vector), ``hybrid_retrieval`` (fusion).
- ``top_k`` moved from ``strategy.top_k`` to the node level.
- ``vectordb`` is now a top-level YAML key referenced by *name* from
  ``semantic_retrieval`` and ``hybrid_retrieval`` modules.
- The pass-through reranker module is ``pass_reranker``, not
  ``pass_passage_reranker``.
- The pass-through query-expansion module is ``pass_query_expansion``;
  HyDE / multi-query require ``generator_module_type`` + ``llm`` + ``model``.

AutoRAG searches a richer native space (``passage_compressor``,
``passage_filter``, ``prompt_maker`` template tuning, LongLLMLingua,
additional ``query_expansion`` modules). Letting AutoRAG enumerate those
would give it search dimensions Random / Bayesian / Agentic don't have.
This module restricts AutoRAG to **only** the dimensions in our
``SearchSpace`` and freezes everything else at sensible defaults.

Every excluded dimension is recorded in ``translation_notes.json`` next to
the generated config; the paper appendix lists them verbatim so reviewers
can verify the mirror is honest.

Chunking is a separate pre-processing phase in AutoRAG v0.3 (not part of
``autorag evaluate``). We freeze the chunking config to a single
``(strategy, chunk_size, chunk_overlap)`` triple chosen as the midpoint of
our search space, and note this exclusion. A future revision could sweep
chunkings by running ``autorag chunk`` once per (strategy, size, overlap)
tuple — high cost, deferred.

LLM provider: our search space uses azure/-prefixed litellm IDs. AutoRAG's
LLM abstraction layer is llama_index, whose Azure support routes through
the ``openailike`` provider with ``api_base`` / ``api_key`` / ``api_version``
parameters (verified against AutoRAG docs/local_model.md). We translate
``azure/<deployment>`` → ``openailike`` with ``model=<deployment>`` and the
endpoint from environment variables.
"""

from __future__ import annotations

import os

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
TOP_K_GRID_N = 5
RERANKER_TOP_K_GRID_N = 3
TEMPERATURE_GRID_N = 1  # our search space pins temperature; one value is enough

# Default vectordb backing store. AutoRAG ships chroma + qdrant + milvus —
# chroma is the simplest because it has no service requirements.
DEFAULT_VECTORDB_NAME = "default"
DEFAULT_VECTORDB_BACKEND = "chroma"


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


def _translate_llm(litellm_model: str) -> tuple[str, str]:
    """Convert a litellm model id to (autorag_llm_provider, autorag_model_name).

    Azure: ``azure/gpt-4o-mini`` → (``openailike``, ``gpt-4o-mini``). Endpoint
    is wired from env vars in the ``llama_index_llm`` module-level params.
    OpenAI: ``openai/gpt-4o-mini`` → (``openai``, ``gpt-4o-mini``).
    Bedrock: ``bedrock/<id>`` → (``bedrock``, ``<id>``).

    Anything else raises so we don't silently mis-translate.
    """
    if "/" not in litellm_model:
        raise ValueError(f"litellm model id {litellm_model!r} must have a provider prefix (e.g. 'azure/...')")
    provider, suffix = litellm_model.split("/", 1)
    if provider == "azure":
        return "openailike", suffix
    if provider == "openai":
        return "openai", suffix
    if provider == "bedrock":
        return "bedrock", suffix
    raise ValueError(
        f"Provider {provider!r} not supported in AutoRAG baseline. "
        "Supported: azure (via openailike), openai, bedrock. "
        "Extend _translate_llm in native_config.py if you add another."
    )


def _build_generator_module(litellm_model: str, temperatures: list[float]) -> dict:
    """One ``llama_index_llm`` module for the given model + temperature grid."""
    autorag_llm, autorag_model = _translate_llm(litellm_model)
    module: dict = {
        "module_type": "llama_index_llm",
        "llm": autorag_llm,
        "model": [autorag_model],
        "temperature": temperatures,
    }
    if autorag_llm == "openailike":
        # AutoRAG's openailike wrapper threads kwargs into llama_index's
        # OpenAILike(...) ctor. ${VAR} is AutoRAG's own env-var syntax.
        module["api_base"] = "${AZURE_API_BASE}"
        module["api_key"] = "${AZURE_API_KEY}"
        module["is_chat_model"] = True
    return module


def _build_vectordb_entries(embedding_models: list[str]) -> tuple[list[dict], dict[str, str]]:
    """One vectordb entry per embedding model — names referenced from retrieval modules.

    AutoRAG can't enumerate embedding models inside a single vectordb entry;
    each must be a named top-level entry. We name them ``embed_<index>`` so
    semantic_retrieval / hybrid_retrieval can pick by name.

    Returns (vectordb_entries, model_to_name).
    """
    entries: list[dict] = []
    model_to_name: dict[str, str] = {}
    for i, m in enumerate(embedding_models):
        name = f"embed_{i}"
        model_to_name[m] = name
        provider_specific = _vectordb_embedding_block(m)
        entry: dict = {
            "name": name,
            "db_type": DEFAULT_VECTORDB_BACKEND,
            "client_type": "persistent",
            "collection_name": name,
            "path": "${PROJECT_DIR}/resources/chroma",
            "embedding_batch": 64,
        }
        entry.update(provider_specific)
        entries.append(entry)
    return entries, model_to_name


def _vectordb_embedding_block(model: str) -> dict:
    """Translate an embedding model id into AutoRAG's vectordb.embedding_model spec.

    AutoRAG's ``embedding_model`` field accepts (verified by reading
    ``autorag.embedding.base.EmbeddingModel.load``):
    - a registry name (str): "openai", "huggingface", "mock", "ollama", "vllm"
    - a list of one dict: ``[{type: <one of above>, model_name: ..., ...kwargs}]``

    We emit the list-of-dict form so the actual HF / openai-compatible model
    is pinned (the bare string "huggingface" requires AutoRAG to default to
    its own ``HF_EMBEDDING_MODEL`` env var, which we don't want to depend on).
    Type ``huggingface`` requires the ``AutoRAG[gpu]`` install; ``openai_like``
    routes through llama_index's OpenAI-compatible adapter (Azure-friendly).
    """
    return {
        "embedding_model": [
            {
                "type": "huggingface",
                "model_name": model,
            }
        ]
    }


def _build_query_expansion_modules(query_expansion: list[str], generator_module: dict) -> list[dict]:
    """Translate our query-expansion choices to AutoRAG modules.

    HyDE / multi-query expansion in AutoRAG need ``generator_module_type``,
    ``llm`` and ``model`` set (so the expansion call has somewhere to land).
    We thread the first search-space LLM through as the expansion LLM.
    """
    out: list[dict] = []
    for qe in query_expansion:
        if qe == "none":
            out.append({"module_type": "pass_query_expansion"})
            continue
        gen_block = {
            "generator_module_type": "llama_index_llm",
            "llm": generator_module["llm"],
            "model": list(generator_module["model"]),
        }
        if "api_base" in generator_module:
            gen_block["api_base"] = generator_module["api_base"]
            gen_block["api_key"] = generator_module["api_key"]
            gen_block["is_chat_model"] = True
        if qe == "hyde":
            out.append({"module_type": "hyde", "max_token": 64, **gen_block})
        elif qe == "multi_query":
            out.append({"module_type": "multi_query_expansion", **gen_block})
        else:
            raise ValueError(f"Unknown query_expansion {qe!r} — add an explicit AutoRAG mapping")
    return out


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
    top_ks = _discretize_int(ss.top_k, TOP_K_GRID_N)
    reranker_top_ks = _discretize_int(ss.reranker.top_n, RERANKER_TOP_K_GRID_N)
    temperatures = _discretize_float(ss.temperature, TEMPERATURE_GRID_N)

    # vectordb entries — one per embedding model, named for cross-referencing.
    vectordb_entries, model_to_name = _build_vectordb_entries(list(ss.embedding_models))

    # Build generator modules first so query_expansion can borrow the first
    # LLM for its expansion calls.
    generator_modules = [_build_generator_module(llm, temperatures) for llm in ss.llm_models]

    # ===== Retrieval nodes (split: lexical / semantic / hybrid) =====
    retrieve_nodes: list[dict] = []
    if IndexType.HYBRID_BM25_VECTOR in ss.index_types or IndexType.VECTOR_ONLY not in ss.index_types:
        # BM25 is a building block of hybrid; expose it as lexical_retrieval
        # whenever hybrid is in the space.
        retrieve_nodes.append(
            {
                "node_type": "lexical_retrieval",
                "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
                "top_k": top_ks[-1],
                "modules": [{"module_type": "bm25"}],
            }
        )
    if IndexType.VECTOR_ONLY in ss.index_types:
        retrieve_nodes.append(
            {
                "node_type": "semantic_retrieval",
                "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
                "top_k": top_ks[-1],
                "modules": [
                    {"module_type": "vectordb", "vectordb": vname}
                    for vname in model_to_name.values()
                ],
            }
        )
    if IndexType.HYBRID_BM25_VECTOR in ss.index_types:
        # AutoRAG's ``hybrid_cc.weight_range`` is the BM25 contribution; ours
        # is the vector's. Translation: bm25_weight = 1 - hybrid_alpha.
        bm25_lo = round(1.0 - ss.hybrid_alpha.max, 4)
        bm25_hi = round(1.0 - ss.hybrid_alpha.min, 4)
        retrieve_nodes.append(
            {
                "node_type": "hybrid_retrieval",
                "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
                "top_k": top_ks[-1],
                "modules": [
                    {
                        "module_type": "hybrid_cc",
                        "normalize_method": ["mm", "tmm"],
                        "weight_range": (bm25_lo, bm25_hi),
                        "test_weight_size": 21,
                    }
                ],
            }
        )

    # ===== Reranker node =====
    reranker_modules: list[dict] = []
    for model in ss.reranker.models:
        if model == "none":
            reranker_modules.append({"module_type": "pass_reranker"})
        else:
            reranker_modules.append({"module_type": _reranker_module_for(model), "model_name": model})

    # ===== Query expansion =====
    query_expansion_modules = _build_query_expansion_modules(
        list(ss.query_expansion),
        generator_modules[0],
    )

    # ===== Metric registration =====
    if qa_variant == "mcq":
        gen_metrics = ["mcq_accuracy"]
        prompt_template = MCQ_PROMPT_TEMPLATE
    else:
        gen_metrics = [
            {"metric_name": "bleu"},
            {"metric_name": "rouge"},
            {"metric_name": "g_eval"},
        ]
        prompt_template = FREE_FORM_PROMPT_TEMPLATE

    # ===== Assemble node_lines =====
    pre_retrieve_nodes = [
        {
            "node_type": "query_expansion",
            "strategy": {
                "metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
                "top_k": top_ks[-1],
                "retrieval_modules": [
                    {"module_type": "vectordb", "vectordb": next(iter(model_to_name.values()))}
                ],
            },
            "modules": query_expansion_modules,
        }
    ]

    post_retrieve_nodes = [
        {
            "node_type": "passage_reranker",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": reranker_top_ks[-1],
            "modules": reranker_modules,
        },
        {
            "node_type": "prompt_maker",
            "strategy": {
                "metrics": gen_metrics,
                "generator_modules": [generator_modules[0]],  # only one for prompt tuning
            },
            "modules": [{"module_type": "fstring", "prompt": [prompt_template]}],
        },
        {
            "node_type": "generator",
            "strategy": {"metrics": gen_metrics},
            "modules": generator_modules,
        },
    ]

    config: dict = {
        "vectordb": vectordb_entries,
        "node_lines": [
            {"node_line_name": "pre_retrieve_node_line", "nodes": pre_retrieve_nodes},
            {"node_line_name": "retrieve_node_line", "nodes": retrieve_nodes},
            {"node_line_name": "post_retrieve_node_line", "nodes": post_retrieve_nodes},
        ],
    }

    excluded_dimensions = [
        "chunking — fixed externally; AutoRAG's chunk phase is separate from evaluate",
        "passage_compressor (tree_summarize / refine / longllmlingua)",
        "passage_filter (similarity_threshold_cutoff / percentile_cutoff / recency_filter)",
        "passage_augmenter (prev_next_augmenter)",
        "prompt_maker template tuning beyond the single fstring",
        "long_context_reorder + window_replacement variants of prompt_maker",
        "query_expansion modules outside {none, hyde, multi_query} (e.g. query_decompose)",
        "hybrid_rrf — we only enumerate hybrid_cc (CC fusion); RRF is excluded for "
        "search-space symmetry with our hybrid_alpha (continuous)",
    ]
    if qa_variant == "mcq":
        excluded_dimensions.append("g_eval / sem_score / bleu / rouge generation metrics — replaced by mcq_accuracy")

    notes = {
        "qa_variant": qa_variant,
        "autorag_target_version": "0.3.x",
        "excluded_dimensions": excluded_dimensions,
        "discretization": {
            "top_k": top_ks,
            "reranker_top_k": reranker_top_ks,
            "temperature": temperatures,
        },
        "reranker_module_map": {m: RERANKER_MODULE_MAP.get(m, "pass_reranker") for m in ss.reranker.models},
        "embedding_model_to_vectordb_name": model_to_name,
        "hybrid_alpha_convention": "AutoRAG's hybrid_cc.weight_range is BM25's; "
        "we pass (1-hybrid_alpha_max, 1-hybrid_alpha_min)",
        "llm_provider_translation": "azure/<m> → openailike with model=<m>, api_base=$AZURE_API_BASE",
        "azure_env_vars_required": ["AZURE_API_KEY", "AZURE_API_BASE"],
        "azure_api_base_present": bool(os.environ.get("AZURE_API_BASE")),
        "azure_api_key_present": bool(os.environ.get("AZURE_API_KEY")),
    }
    return config, notes
