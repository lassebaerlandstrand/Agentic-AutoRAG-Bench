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

from agentic_autorag.config.models import (
    DiscreteValues,
    IndexType,
    NumericDim,
    SearchSpace,
)
from agentic_autorag.engine.pipeline import (
    _DEFAULT_REFINE_PROMPT_TMPL,
    _DEFAULT_TREE_SUMMARIZE_PROMPT_TMPL,
)
from agentic_autorag.engine.pipeline import _MULTI_QUERY_PROMPT as FRAMEWORK_MULTI_QUERY_PROMPT
from agentic_autorag.examiner.prompts import NAIVE_RAG_PROMPT, answer_format_hint

# Mirror the framework's NAIVE_RAG_PROMPT so AutoRAG's internal rouge metric
# optimises under the same prompt the bench's rescore uses for every method.
# A diverging prompt would mean AutoRAG enumerates one objective (rouge over
# its static prompt) then gets scored on another (judge-acc over the
# framework's prompt with per-question hints), biasing its winner selection.
#
# Two substitutions:
#   - ``{context}`` -> ``{retrieved_contents}``: AutoRAG's fstring module
#     uses this variable name.
#   - ``{question}`` -> ``{query}``: same.
# The per-question ``{answer_format_hint}`` placeholder is replaced at
# generation time with the framework's neutral fallback. AutoRAG's static
# config can't inject per-question hints (no per-row prompt substitution),
# so the fallback is the most-honest reproduction of the framework's prompt
# inside AutoRAG's constraints. Disclosed in the paper's accounting note.
FREE_FORM_PROMPT_TEMPLATE = (
    NAIVE_RAG_PROMPT
    .replace("{context}", "{retrieved_contents}")
    .replace("{question}", "{query}")
    .replace("{answer_format_hint}", answer_format_hint(None, None))
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
}

# Internal hybrid_cc alpha sweep size — historical knob; superseded by the
# per-value module enumeration in ``_build_hybrid_modules`` which mirrors the
# framework's discrete ``hybrid_alpha`` grid exactly. Kept here only because
# tests still reference the constant. 21 (the AutoRAG default) would give
# AutoRAG ~21
# alpha-tuning evaluations per call while adaptive methods sample one alpha
# per trial. We drop it to 5 to roughly match adaptive sampling density.
HYBRID_CC_TEST_WEIGHT_SIZE = 5

# Pinned BM25 tokenizer and CC normalize method. AutoRAG was enumerating these
# (porter_stemmer/space, mm/tmm) under prior translator versions, giving
# AutoRAG a silent 2x x 2x = 4x advantage at the hybrid+lexical path the
# framework has no equivalent knob to match. Pinning to single values restores
# parity. porter_stemmer + mm are AutoRAG's documented defaults.
BM25_TOKENIZER_PINNED = "porter_stemmer"
HYBRID_CC_NORMALIZE_METHOD_PINNED = "mm"

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


def _require_discrete_int(dim: NumericDim, name: str) -> list[int]:
    """Extract the int option set from a DiscreteValues dim, or raise.

    Section-1 AutoRAG baselines must use DiscreteValues for ``top_k``,
    ``reranker.top_n``, ``chunk_token_size``, and ``chunk_token_overlap``:
    AutoRAG enumerates lists via ``itertools.product``, so a continuous
    NumericRange has no fair on-line mapping. Section-2 (continuous Pareto)
    is framework-only; if it ever tries to translate to AutoRAG, fail loud.
    """
    if not isinstance(dim, DiscreteValues):
        raise ValueError(
            f"AutoRAG translation requires DiscreteValues for {name!r} "
            f"(got NumericRange [{dim.min}, {dim.max}]). AutoRAG enumerates "
            "node-level params via itertools.product over list-valued keys; "
            "a continuous range cannot be sampled fairly inside one evaluator "
            "run. Use DiscreteValues in the Section-1 config."
        )
    return [int(v) for v in dim.values]


def _require_discrete_float(dim: NumericDim, name: str) -> list[float]:
    """As :func:`_require_discrete_int` but for float dims (e.g. hybrid_alpha)."""
    if not isinstance(dim, DiscreteValues):
        raise ValueError(
            f"AutoRAG translation requires DiscreteValues for {name!r} "
            f"(got NumericRange [{dim.min}, {dim.max}]). "
            "Use DiscreteValues in the Section-1 config."
        )
    return [float(v) for v in dim.values]


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


_HYBRID_FUSION_MODULE_MAP: dict[str, str] = {
    "alpha": "hybrid_cc",
    "rrf": "hybrid_rrf",
}


_PASSAGE_COMPRESSOR_MODULE_MAP: dict[str, str] = {
    "none": "pass_compressor",
    "tree_summarize": "tree_summarize",
    "refine": "refine",
}

# Pin the prompt for every llama_index compressor variant. AutoRAG's
# ``LlamaIndexCompressor`` picks ``prompt`` for non-chat LLMs and ``chat_prompt``
# for chat-capable LLMs (``is_chat_model`` returns True for ``Bedrock``). Pinning
# both to the same string forces AutoRAG into a deterministic branch matching
# the framework's prepare_context output regardless of model type.
_COMPRESSOR_PROMPT_TMPL: dict[str, str] = {
    "tree_summarize": _DEFAULT_TREE_SUMMARIZE_PROMPT_TMPL,
    "refine": _DEFAULT_REFINE_PROMPT_TMPL,
}


def _build_passage_compressor_modules(
    search_space: SearchSpace,
    compressor_llm_modules: list[dict],
    temperatures: list[float],
) -> list[dict]:
    """Emit AutoRAG passage_compressor modules.

    For every non-"none" compressor type, emits one module per LLM in
    ``compressor_llm_modules`` (built from ``ss.passage_compressor.models``)
    so AutoRAG's strategy can pick a compressor LLM independently of the
    generator LLM. The ``prompt`` and ``chat_prompt`` kwargs are pinned to
    the framework's templates so AutoRAG runs against the same wording
    regardless of the underlying LLM's chat-mode. ``temperature`` is pinned
    explicitly to prevent reasoning models from falling back to llama_index
    defaults.
    """
    modules: list[dict] = []
    for compressor in search_space.passage_compressor.strategies:
        if compressor == "none":
            modules.append({"module_type": "pass_compressor"})
            continue
        if compressor not in _PASSAGE_COMPRESSOR_MODULE_MAP:
            raise ValueError(
                f"Unknown passage_compressor {compressor!r}. "
                "Add an entry to _PASSAGE_COMPRESSOR_MODULE_MAP."
            )
        for gen_mod in compressor_llm_modules:
            compressor_module: dict = {
                "module_type": _PASSAGE_COMPRESSOR_MODULE_MAP[compressor],
                "llm": gen_mod["llm"],
                "model": list(gen_mod["model"]),
                "temperature": list(temperatures),
                "prompt": _COMPRESSOR_PROMPT_TMPL[compressor],
                "chat_prompt": _COMPRESSOR_PROMPT_TMPL[compressor],
            }
            for key in ("api_base", "api_key", "region_name"):
                if key in gen_mod:
                    compressor_module[key] = gen_mod[key]
            modules.append(compressor_module)
    return modules


def _build_prompt_maker_modules(search_space: SearchSpace, prompt_template: str) -> list[dict]:
    """Emit prompt_maker modules from ``search_space.retrieval.long_context_reorder``.

    ``False`` → ``fstring`` (substitute only); ``True`` →
    ``long_context_reorder`` (append top-by-score passage to the end before
    substitution).
    """
    modules: list[dict] = []
    for enabled in search_space.retrieval.long_context_reorder:
        if enabled is False:
            modules.append({"module_type": "fstring", "prompt": [prompt_template]})
        elif enabled is True:
            modules.append({"module_type": "long_context_reorder", "prompt": [prompt_template]})
        else:
            raise ValueError(
                f"long_context_reorder values must be bool; got {enabled!r}"
            )
    return modules


def _build_hybrid_modules(search_space: SearchSpace, alpha_values: list[float]) -> list[dict]:
    """Emit one AutoRAG hybrid_retrieval module per enumerated fusion strategy.

    ``"alpha"`` → one ``hybrid_cc`` module per value in ``alpha_values``,
    pinned to that exact weight (``weight_range=(v, v)``, ``test_weight_size=1``).
    This forces AutoRAG to enumerate exactly the framework's discrete
    ``hybrid_alpha`` grid. The previous implementation passed a range plus
    ``test_weight_size`` (5), which gave AutoRAG access to intermediate
    values like ``0.25`` that the framework's search space disallows — a
    silent fairness break.

    ``"rrf"`` → ``hybrid_rrf`` with ``weight=60`` (rrf_k) pinned.

    ``normalize_method`` is pinned to a single value to match parity with
    the framework's lexical+fts fusion path (which has no equivalent knob).
    """
    modules: list[dict] = []
    for fusion in search_space.retrieval.bm25_vector_fusion:
        if fusion == "alpha":
            for v in alpha_values:
                vr = round(float(v), 4)
                modules.append(
                    {
                        "module_type": "hybrid_cc",
                        "normalize_method": HYBRID_CC_NORMALIZE_METHOD_PINNED,
                        # AutoRAG's YAML loader re-tuplifies the string form
                        # ``"(a, b)"`` (autorag.utils.util.convert_string_to_tuple_in_dict).
                        # PyYAML can't dump Python tuples, so we emit the string.
                        "weight_range": f"({vr}, {vr})",
                        "test_weight_size": 1,
                    }
                )
        elif fusion == "rrf":
            # AutoRAG's HybridRRF.run_evaluator enumerates ``weight`` (its name
            # for rrf_k) across ``np.linspace(weight_range[0], weight_range[1],
            # weight_range[1] - weight_range[0] + 1)``. Pin the range to a
            # single point so the enumeration collapses to ``[60]`` — matches
            # the framework's ``_rrf_merge`` k=60 default.
            modules.append({"module_type": "hybrid_rrf", "weight_range": "(60, 60)"})
        else:
            raise ValueError(
                f"Unknown bm25_vector_fusion {fusion!r}. "
                f"Add an entry to _HYBRID_FUSION_MODULE_MAP."
            )
    return modules


def _build_query_expansion_modules(
    query_expansion: list[str],
    expander_llm_modules: list[dict],
    temperatures: list[float],
) -> list[dict]:
    """Translate our query-expansion choices to AutoRAG modules.

    For each non-"none" strategy, emits one module per LLM in
    ``expander_llm_modules`` (built from ``ss.query_expansion.models``) so
    AutoRAG's strategy can pick the expander LLM independently of the
    generator. HyDE / multi-query / query_decompose all need
    ``generator_module_type`` / ``llm`` / ``model`` plus the
    provider-specific auth keys. ``temperature`` is pinned to prevent
    llama_index defaults on reasoning models.
    """
    out: list[dict] = []
    for qe in query_expansion:
        if qe == "none":
            out.append({"module_type": "pass_query_expansion"})
            continue
        for gen_mod in expander_llm_modules:
            gen_block: dict = {
                "generator_module_type": "llama_index_llm",
                "llm": gen_mod["llm"],
                "model": list(gen_mod["model"]),
                "temperature": list(temperatures),
            }
            for key in ("api_base", "api_key", "region_name"):
                if key in gen_mod:
                    gen_block[key] = gen_mod[key]
            if qe == "hyde":
                # NOTE: AutoRAG's HyDE has an upstream source bug at
                # nodes/queryexpansion/hyde.py:34 — ``(prompt if not bool(prompt)
                # else hyde_prompt)`` inverts the falsy check, so passing a
                # custom ``prompt`` is silently ignored and AutoRAG falls back
                # to its default. We don't attempt to override it from here;
                # the framework's HyDE prompt has been aligned to AutoRAG's
                # exact default (see Agentic-AutoRAG/agentic_autorag/engine/
                # pipeline.py::_HYDE_PROMPT) so both methods produce
                # hypothetical documents under the same instruction.
                out.append({"module_type": "hyde", "max_token": 64, **gen_block})
            elif qe == "multi_query":
                # Pass the framework's MultiQuery prompt so AutoRAG uses the
                # same expansion instruction the framework methods do. AutoRAG
                # honors the custom ``prompt`` here (no source bug like HyDE)
                # via ``prompt.format(query=x)`` in
                # nodes/queryexpansion/multi_query_expansion.py.
                out.append({
                    "module_type": "multi_query_expansion",
                    "prompt": FRAMEWORK_MULTI_QUERY_PROMPT,
                    **gen_block,
                })
            elif qe == "query_decompose":
                # Passing ``prompt: ""`` forces AutoRAG into the
                # ``bool(prompt) is False`` branch in
                # ``QueryDecompose._pure``, which substitutes the question
                # cleanly via ``decompose_prompt.format(question=query)`` —
                # matching the framework's behaviour. With the default
                # ``prompt=decompose_prompt`` AutoRAG instead wraps the prompt
                # as ``f"prompt: {decompose_prompt}\n\n question: {query}"``,
                # leaving a literal ``{question}`` placeholder in the example
                # slot.
                out.append({"module_type": "query_decompose", "prompt": "", **gen_block})
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
        ``"our_exam"`` uses the framework's open-ended exam via the MCQ-style
        prompt template (still cheap rouge as AutoRAG's internal metric).
        ``"ragas"`` uses ``g_eval`` and the free-form template against a
        RAGAS-bootstrapped QA set.
    """
    if qa_variant not in {"our_exam", "ragas"}:
        raise ValueError(f"qa_variant must be 'our_exam' or 'ragas', got {qa_variant!r}")

    ss = search_space
    top_ks = _require_discrete_int(ss.retrieval.top_k, "retrieval.top_k")
    reranker_top_ks = _require_discrete_int(ss.reranker.top_n, "reranker.top_n")
    # Temperature stays a NumericRange (pinned to a single value in every
    # paper config); we emit AutoRAG with a single point at the lower bound.
    if ss.temperature.min != ss.temperature.max:
        raise ValueError(
            "AutoRAG translation expects temperature to be pinned to a single "
            f"value (got [{ss.temperature.min}, {ss.temperature.max}]). "
            "Set min=max in the search space."
        )
    temperatures = [round(float(ss.temperature.min), 2)]

    # vectordb entries — one per embedding model, named for cross-referencing.
    vectordb_entries, model_to_name = _build_vectordb_entries(list(ss.embedding.models))

    # Per-stage LLM modules. Generator gets the (typically larger) generator
    # pool; expander/compressor get their own cheaper pools — matches the
    # framework's TrialConfig generator_llm / expander_llm / compressor_llm
    # split. AutoRAG strategy then enumerates the right pool at each node.
    generator_modules = [_build_generator_module(llm, temperatures) for llm in ss.generator.models]
    expander_modules = [_build_generator_module(llm, temperatures) for llm in ss.query_expansion.models]
    compressor_modules = [_build_generator_module(llm, temperatures) for llm in ss.passage_compressor.models]

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
    hybrid_alpha_values = _require_discrete_float(ss.retrieval.hybrid_alpha, "retrieval.hybrid_alpha")
    if IndexType.HYBRID_BM25_VECTOR in ss.retrieval.index_types:
        hybrid_weights = sorted(round(float(v), 4) for v in hybrid_alpha_values)
    else:
        # vector_only only — pin hybrid_cc weight=1.0 so the fusion is fully
        # semantic and acts as a pass-through of the semantic retriever.
        hybrid_weights = [1.0]

    # ``top_k`` is a node-level parameter in AutoRAG. ``Node.from_dict``
    # (autorag/schema/node.py:50) routes every non-strategy/non-modules key
    # into ``node_params``; ``get_param_combinations`` then merges those into
    # each module's ``module_param`` and runs ``make_combinations``
    # (autorag/utils/util.py:137) which ``itertools.product``s every
    # list-valued key. So passing a *list* of top_k values causes AutoRAG to
    # enumerate (top_k, module-params) pairs natively — no outer loop needed.
    # Pinning to ``top_ks[-1]`` (a prior version of this translator) silently
    # locked AutoRAG to a single top_k value at every retrieval node, which
    # is a strict fairness regression vs. random/Bayesian/agentic methods
    # that sample top_k freely. All retrieval module signatures
    # (bm25.py:200, vectordb.py:85, hybrid_rrf.py:13, hybrid_cc.py:57)
    # accept ``top_k: int`` per call, so the list is consumed correctly.
    retrieve_nodes: list[dict] = [
        {
            "node_type": "lexical_retrieval",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": top_ks,
            # bm25_tokenizer is pinned to a single value (porter_stemmer); the
            # framework's lexical-side scoring has no equivalent tokenizer
            # knob, so enumerating it here would give AutoRAG a silent
            # advantage. See translator audit (project_autorag_translator_audit).
            "modules": [{"module_type": "bm25", "bm25_tokenizer": BM25_TOKENIZER_PINNED}],
        },
        {
            "node_type": "semantic_retrieval",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": top_ks,
            "modules": [
                {"module_type": "vectordb", "vectordb": vname}
                for vname in model_to_name.values()
            ],
        },
        {
            "node_type": "hybrid_retrieval",
            "strategy": {"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]},
            "top_k": top_ks,
            "modules": _build_hybrid_modules(ss, hybrid_weights),
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
    # All LLMs in the *expander* stage's pool are emitted at the expander
    # node so AutoRAG's strategy picks the expander LLM independently of
    # the generator. Matches the framework's per-stage ``expander_llm`` field.
    query_expansion_modules = _build_query_expansion_modules(
        list(ss.query_expansion.strategies),
        expander_modules,
        temperatures,
    )

    # ===== Metric registration =====
    # Both variants score the winning AutoRAG pipeline through our framework's
    # held-out evaluator afterwards, so AutoRAG's *internal* metric just needs
    # to be reasonable for ranking. We use ``rouge`` (token overlap with the
    # gold answer) — cheap, deterministic, no LLM judge cost.
    # Both variants use the open-ended prompt template. ``our_exam`` is the
    # framework's open-ended exam (despite the legacy variable name once
    # implying multi-choice), so the prompt must not reference non-existent
    # "options" — a MCQ-framed prompt against an open-ended exam makes the
    # generator answer as if it were selecting from choices, severely biasing
    # AutoRAG's score downward.
    #
    # Generator metric is rouge (token overlap with the gold answer). For
    # factoid-extraction tasks like HotpotQA the gold answers are short
    # canonical strings, and rouge correlates with judge accuracy as well or
    # better than sem_score — semantic-similarity metrics can rank
    # wrong-but-similar answers (e.g. "1996" vs gold "1995", or near-miss
    # surnames) high because they embed near the gold. The bench's
    # downstream judge-based rescore is the final scoring authority for
    # cross-method comparison.
    if qa_variant == "our_exam":
        gen_metrics = ["rouge"]
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

    post_retrieve_nodes: list[dict] = [
        {
            "node_type": "passage_reranker",
            # Order-sensitive metrics for the rerank node: NDCG, MAP, MRR.
            # Set metrics (retrieval_f1 / recall / precision) would tie all
            # rerankers because reranking only re-orders the input set —
            # set membership is identical across rerankers, so the metric
            # produces identical scores and AutoRAG's ``is_best`` selection
            # collapses to first-row-wins (pass_reranker). NDCG/MAP/MRR
            # reward putting gold docs earlier, distinguishing rerankers as
            # they're designed to be. This matches AutoRAG's published
            # rerank-node convention (retrieve nodes use set metrics, rerank
            # nodes use order-sensitive ones — autorag/evaluation/metric/
            # retrieval.py ships all three).
            "strategy": {"metrics": ["retrieval_ndcg", "retrieval_map", "retrieval_mrr"]},
            # See top_k comment on retrieve_nodes above — same enumeration
            # rule applies: list-valued node-level top_k is swept by AutoRAG
            # via make_combinations. Pinning to ``reranker_top_ks[-1]`` (a
            # prior translator bug) locked AutoRAG out of smaller reranker
            # top_n values that adaptive methods could sample.
            "top_k": reranker_top_ks,
            "modules": reranker_modules,
        },
    ]
    # Omit the passage_compressor node entirely when only "none" is
    # enumerated — AutoRAG would otherwise evaluate a no-op module per trial.
    if any(c != "none" for c in ss.passage_compressor.strategies):
        # AutoRAG's passage_compressor node restricts strategy.metrics to
        # the retrieval-token metric family (validated at
        # autorag/nodes/passagecompressor/run.py:82-89). Using generator
        # metrics like rouge / bleu here raises ``ValueError: metrics must be
        # one of ...``.
        post_retrieve_nodes.append(
            {
                "node_type": "passage_compressor",
                "strategy": {
                    "metrics": [
                        "retrieval_token_f1",
                        "retrieval_token_recall",
                        "retrieval_token_precision",
                    ]
                },
                "modules": _build_passage_compressor_modules(ss, compressor_modules, temperatures),
            }
        )
    post_retrieve_nodes.extend(
        [
            {
                "node_type": "prompt_maker",
                "strategy": {
                    "metrics": gen_metrics,
                    # All generator-stage LLMs are exposed at prompt_maker too
                    # so AutoRAG's strategy enumerates the full per-stage pool
                    # at every node that touches an LLM.
                    "generator_modules": generator_modules,
                },
                "modules": _build_prompt_maker_modules(ss, prompt_template),
            },
            {
                "node_type": "generator",
                "strategy": {"metrics": gen_metrics},
                "modules": generator_modules,
            },
        ]
    )

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
        "passage_compressor longllmlingua module (LLMLingua dependency excluded)",
        "passage_filter (similarity_threshold_cutoff / percentile_cutoff / recency_filter)",
        "passage_augmenter (prev_next_augmenter)",
        "prompt_maker template tuning beyond the single fstring (and "
        "long_context_reorder when enumerated)",
        "window_replacement variants of prompt_maker",
    ]
    excluded_dimensions.append(
        "AutoRAG's internal generation metric is ``rouge`` "
        "(winning config is re-scored through our framework's evaluator anyway, "
        "so AutoRAG's internal ranking signal need only correlate with our final metric)"
    )

    has_bedrock = any(m.startswith("bedrock/") for m in ss.all_llm_models())

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
