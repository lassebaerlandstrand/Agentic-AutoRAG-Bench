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

# Embedding batch tuned for GPU throughput. AutoRAG's default is 100; the
# upstream HuggingFaceEmbedding default is 10. With six HF embedders and 19k
# docs the corpus ingest dominates setup time, and HF's own perf guide
# recommends batch sizes ≥128 on GPU ("quantized models reach peak throughput
# at batch size 128"). 256 is comfortably in-VRAM once
# _patch_free_embedder_after_ingest (see scripts/autorag_patches.py) keeps
# only one embedder resident at a time.
EMBEDDING_BATCH = 256

# Use fp16 weights for HF embedders. Halves VRAM and ~1.5× throughput on the
# 4080. Numeric drift from fp32 → fp16 is well below the cosine ≥ 0.999
# threshold the existing equivalence test enforces (test_autorag_equivalence
# .test_huggingface_embedding_equivalence_minilm); these sentence-transformer
# models are calibrated for half-precision inference.
EMBEDDING_TORCH_DTYPE = "float16"


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

    Azure: ``azure/gpt-4o-mini`` → (``openai``, ``gpt-4o-mini``) with the
    base URL pointed at Azure's OpenAI-compatible endpoint
    ``<AZURE_API_BASE>/openai/v1``. Verified manually: OpenAILike drops
    ``is_chat_model`` through AutoRAG's ``pop_params`` because the field
    is a Pydantic class attribute (not declared on ``__init__``), so the
    chat/completion routing breaks. The plain ``openai`` provider's
    metadata-driven chat detection recognises ``gpt-4o-mini`` etc. as chat
    models without needing the ``is_chat_model`` knob.
    OpenAI: ``openai/gpt-4o-mini`` → (``openai``, ``gpt-4o-mini``).
    Bedrock: ``bedrock/<id>`` → (``bedrock_converse``, ``<id>``). We route
    through the modern ``BedrockConverse`` LLM (registered as a new AutoRAG
    provider by scripts/autorag_patches.py) because AutoRAG 0.3's bundled
    deprecated ``llama_index.llms.bedrock.Bedrock`` hard-restricts ``model``
    to a fixed pre-2024 registry — our search-space entries (Llama 3.1, Nova
    2 Lite, Claude Haiku 4.5) all fail there with a ``context_size`` error.

    Anything else raises so we don't silently mis-translate.
    """
    if "/" not in litellm_model:
        raise ValueError(f"litellm model id {litellm_model!r} must have a provider prefix (e.g. 'azure/...')")
    provider, suffix = litellm_model.split("/", 1)
    if provider in ("azure", "openai"):
        return "openai", suffix
    if provider == "bedrock":
        return "bedrock_converse", suffix
    raise ValueError(
        f"Provider {provider!r} not supported in AutoRAG baseline. "
        "Supported: azure (via openai+v1-compat base), openai, bedrock (via bedrock_converse). "
        "Extend _translate_llm in native_config.py if you add another."
    )


def _build_generator_module(litellm_model: str, temperatures: list[float]) -> dict:
    """One ``llama_index_llm`` module for the given model + temperature grid."""
    autorag_llm, autorag_model = _translate_llm(litellm_model)
    is_azure = litellm_model.startswith("azure/")
    is_bedrock = litellm_model.startswith("bedrock/")
    module: dict = {
        "module_type": "llama_index_llm",
        "llm": autorag_llm,
        "model": [autorag_model],
        "temperature": temperatures,
    }
    if is_azure:
        # Azure's cognitive-services hosts respond to standard OpenAI chat
        # completions under ``<base>/openai/v1``. ``${VAR}`` is AutoRAG's
        # env-var substitution syntax (autorag.utils.util.convert_env_in_dict).
        module["api_base"] = "${AZURE_API_BASE}/openai/v1"
        module["api_key"] = "${AZURE_API_KEY}"
    if is_bedrock:
        # ``BedrockConverse.__init__`` accepts ``region_name`` and is happy to
        # pick up AWS credentials from boto3's standard env-var chain
        # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY). The region is the only
        # thing boto3 can't infer: it reads AWS_REGION / AWS_DEFAULT_REGION but
        # *not* the litellm-convention AWS_REGION_NAME the rest of the bench
        # uses, so we plumb it explicitly. AutoRAG substitutes ``${VAR}`` at
        # YAML load.
        module["region_name"] = "${AWS_REGION_NAME}"
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
            # ``${PROJECT_DIR}/resources`` is symlinked by the driver to a
            # corpus-keyed shared cache (see methods/autorag/driver.py:
            # ``_setup_resources_cache``); AutoRAG's ``filter_exist_ids``
            # then short-circuits re-embedding on subsequent runs.
            "path": "${PROJECT_DIR}/resources/chroma",
            "embedding_batch": EMBEDDING_BATCH,
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
                # fp16 weights — halve VRAM and ~1.5× throughput on the 4080.
                # llama-index's HuggingFaceEmbedding forwards ``model_kwargs``
                # to ``SentenceTransformer(... model_kwargs=...)``, which
                # passes ``torch_dtype`` straight to ``transformers``'
                # ``from_pretrained``.
                "model_kwargs": {"torch_dtype": EMBEDDING_TORCH_DTYPE},
            }
        ]
    }


def _build_query_expansion_modules(query_expansion: list[str], generator_module: dict) -> list[dict]:
    """Translate our query-expansion choices to AutoRAG modules.

    HyDE / multi-query expansion in AutoRAG need ``generator_module_type``,
    ``llm`` and ``model`` set (so the expansion call has somewhere to land).
    We thread the first search-space LLM through as the expansion LLM, copying
    whichever provider-specific auth keys it carries (Azure: api_base/api_key,
    Bedrock: region_name).
    """
    out: list[dict] = []
    for qe in query_expansion:
        if qe == "none":
            out.append({"module_type": "pass_query_expansion"})
            continue
        gen_block: dict = {
            "generator_module_type": "llama_index_llm",
            "llm": generator_module["llm"],
            "model": list(generator_module["model"]),
        }
        for key in ("api_base", "api_key", "region_name"):
            if key in generator_module:
                gen_block[key] = generator_module[key]
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

    # AutoRAG v0.3 ALWAYS requires all three retrieval node_types when a
    # passage_reranker follows: lexical and semantic emit suffixed columns
    # (``retrieved_contents_lexical`` / ``_semantic``); hybrid is the only
    # node that produces the un-suffixed ``retrieved_contents`` the reranker
    # consumes (autorag/nodes/passagereranker/run.py drops the suffixed
    # columns unconditionally). So we always emit all three.
    #
    # AutoRAG hybrid_cc's ``weight`` is the *semantic* weight: weight=1.0 →
    # semantic-only, weight=0.0 → lexical/BM25-only (verified against
    # autorag/nodes/hybridretrieval/hybrid_cc.py docstring). Our SearchSpace's
    # ``hybrid_alpha`` uses the same convention via ``HybridAlphaReranker``
    # (relevance = alpha*vector + (1-alpha)*fts), so the two map 1:1 with no
    # inversion.
    semantic_lo = round(float(ss.hybrid_alpha.min), 4)
    semantic_hi = round(float(ss.hybrid_alpha.max), 4)
    if IndexType.HYBRID_BM25_VECTOR in ss.index_types:
        weight_lo, weight_hi = semantic_lo, semantic_hi
    else:
        # vector_only only — pin hybrid_cc weight=1.0 so the fusion is fully
        # semantic and acts as a pass-through of the semantic retriever.
        weight_lo, weight_hi = 1.0, 1.0

    retrieve_nodes: list[dict] = [
        {
            "node_type": "lexical_retrieval",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": top_ks[-1],
            "modules": [{"module_type": "bm25", "bm25_tokenizer": ["porter_stemmer", "space"]}],
        },
        {
            "node_type": "semantic_retrieval",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": top_ks[-1],
            "modules": [
                {"module_type": "vectordb", "vectordb": vname}
                for vname in model_to_name.values()
            ],
        },
        {
            "node_type": "hybrid_retrieval",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": top_ks[-1],
            "modules": [
                {
                    "module_type": "hybrid_cc",
                    "normalize_method": ["mm", "tmm"],
                    # AutoRAG's custom YAML loader recognises the string form
                    # ``"(a, b)"`` as a tuple (autorag.utils.util:convert_string_to_tuple_in_dict).
                    # PyYAML can't dump Python tuples → we emit the string form
                    # directly so AutoRAG re-tuplifies on load.
                    "weight_range": f"({weight_lo}, {weight_hi})",
                    "test_weight_size": 21 if weight_lo != weight_hi else 1,
                }
            ],
        },
    ]

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
    # Both variants score the winning AutoRAG pipeline through our framework's
    # held-out evaluator afterwards, so AutoRAG's *internal* metric just needs
    # to be reasonable for ranking. We use ``rouge`` (token overlap with the
    # gold answer) — cheap, deterministic, no LLM judge cost.
    # ``mcq_accuracy`` (our custom substring-match metric) would require
    # patching AutoRAG's ``GENERATION_METRIC_FUNC_DICT`` at runtime; we keep
    # it simple and use built-in metrics only.
    if qa_variant == "mcq":
        gen_metrics = ["rouge"]
        prompt_template = MCQ_PROMPT_TEMPLATE
    else:
        gen_metrics = ["rouge", "bleu"]
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
    excluded_dimensions.append(
        "custom ``mcq_accuracy`` metric — replaced by AutoRAG's built-in rouge "
        "(winning config is re-scored through our framework's evaluator anyway, "
        "so AutoRAG's internal ranking signal need only correlate with our final metric)"
    )

    has_bedrock = any(m.startswith("bedrock/") for m in ss.llm_models)

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
        "hybrid_alpha_convention": "AutoRAG hybrid_cc.weight is the *semantic* weight "
        "(weight=1.0 → semantic-only), identical to our hybrid_alpha; passed straight through",
        "llm_provider_translation": (
            "azure/<m> → openai with model=<m>, api_base=$AZURE_API_BASE/openai/v1; "
            "bedrock/<m> → bedrock_converse with model=<m>, region_name=$AWS_REGION_NAME "
            "(bedrock_converse is registered by scripts/autorag_patches.py; the deprecated "
            "bedrock provider can't load 2024+ model IDs)"
        ),
        "azure_env_vars_required": ["AZURE_API_KEY", "AZURE_API_BASE"],
        "azure_api_base_present": bool(os.environ.get("AZURE_API_BASE")),
        "azure_api_key_present": bool(os.environ.get("AZURE_API_KEY")),
        "bedrock_in_search_space": has_bedrock,
        "bedrock_env_vars_required": (
            ["AWS_REGION_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] if has_bedrock else []
        ),
        "aws_region_name_present": bool(os.environ.get("AWS_REGION_NAME")),
        "aws_access_key_id_present": bool(os.environ.get("AWS_ACCESS_KEY_ID")),
    }
    return config, notes
