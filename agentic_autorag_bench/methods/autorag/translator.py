"""Translate AutoRAG's resolved ``extracted_sample.yaml`` → our ``TrialConfig``.

Targets AutoRAG v0.3.x. The key shape differences from v0.2:
- Retrieval is one of ``lexical_retrieval`` / ``semantic_retrieval`` /
  ``hybrid_retrieval`` instead of a single ``retrieval`` node_type.
- ``vectordb`` is a top-level YAML key (a list of named entries); the
  ``semantic_retrieval`` module's ``vectordb: <name>`` references it.
- ``top_k`` lives at the module level (or node level — both occur in
  practice). We read whichever is set.

Anything missing falls back to the search-space minimum so the result is
always a valid ``TrialConfig``. Chunking is not enumerated by AutoRAG v0.3
inside ``autorag evaluate`` (it's a separate pre-step) so we fall back to
search-space defaults and surface a hint in translation_notes.json.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from agentic_autorag.config.models import (
    IndexType,
    SearchSpace,
    TrialConfig,
    _dim_midpoint,
    _dim_min_value,
)


def _walk_nodes(extracted: dict) -> dict[str, dict]:
    """Map node_type → node dict, with the first occurrence winning.

    AutoRAG's extracted_sample.yaml has one node per node_type since it's
    the resolved (best-of) pipeline.
    """
    nodes: dict[str, dict] = {}
    for line in extracted.get("node_lines", []) or []:
        for node in line.get("nodes", []) or []:
            ntype = node.get("node_type")
            if ntype and ntype not in nodes:
                nodes[ntype] = node
    return nodes


def _winning_module(node: dict) -> dict:
    modules = node.get("modules", []) or []
    return modules[0] if modules else {}


def _scalar_or_first(value: object) -> object:
    return value[0] if isinstance(value, list) and value else value


_CAMEL_TO_SNAKE_RE = re.compile(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])")

# AutoRAG class names that don't follow ordinary CamelCase splitting. The
# generic regex turns ``HyDE`` into ``hy_de`` and ``VectorDB`` into
# ``vector_db`` because of the trailing all-caps segment; the modules
# themselves are referenced in snake_case YAML as ``hyde`` and ``vectordb``.
_NORMALIZE_OVERRIDES: dict[str, str] = {
    "hy_de": "hyde",
    "vector_db": "vectordb",
}


def _normalize_module_type(s: str | None) -> str:
    """AutoRAG v0.3 extract_best_config emits CamelCase ``module_type`` strings
    (e.g. ``PassReranker``, ``VectorDB``, ``HybridCC``, ``LlamaIndexLLM``)
    even though the input config uses snake_case. Normalise to lowercase
    snake_case so both forms match the translator's branches.
    """
    if not s:
        return ""
    normalised = _CAMEL_TO_SNAKE_RE.sub("_", s.strip()).lower()
    return _NORMALIZE_OVERRIDES.get(normalised, normalised)


def _vectordb_to_embedding(extracted: dict) -> dict[str, str | None]:
    """Reverse-map: vectordb name → its embedding model id.

    AutoRAG v0.3 accepts ``embedding_model`` as either a string registry
    key, or a list of one dict: ``[{type: huggingface, model_name: ...}]``.
    We unwrap both shapes to a single model id. Returns None if the entry
    can't be resolved.
    """
    out: dict[str, str | None] = {}
    for entry in extracted.get("vectordb", []) or []:
        name = entry.get("name")
        if not name:
            continue
        em = entry.get("embedding_model")
        if isinstance(em, list) and em:
            spec = em[0] if isinstance(em[0], dict) else {}
            out[name] = spec.get("model_name") or spec.get("type")
        elif isinstance(em, str):
            out[name] = em
        else:
            out[name] = None
    return out


def _read_top_k(node: dict, module: dict) -> int | None:
    """top_k may be on the module (v0.3 common form), or at node level (also v0.3)."""
    if "top_k" in module:
        return int(_scalar_or_first(module["top_k"]))
    if "top_k" in node:
        return int(_scalar_or_first(node["top_k"]))
    strategy = node.get("strategy") or {}
    if "top_k" in strategy:
        return int(_scalar_or_first(strategy["top_k"]))
    return None


def _autorag_llm_to_litellm(
    llm_provider: str | None, model: str | None, api_base: str | None = None
) -> str | None:
    """Reverse-map AutoRAG generator output to a litellm model id.

    Mirror of native_config._translate_llm. Disambiguates ``openai`` between
    real OpenAI vs Azure by checking whether ``api_base`` contains an Azure
    host. The translator returns the raw assembled string; the caller
    validates against the search space.

    ``bedrock_converse`` is the modern provider native_config emits for
    bedrock/* entries (see scripts/autorag_patches.py); ``bedrock`` is kept
    for backward compatibility with older extracted_sample.yaml files.
    """
    if not llm_provider or not model:
        return None
    api_base_str = (api_base or "").lower()
    is_azure = "azure" in api_base_str or "cognitiveservices" in api_base_str
    if llm_provider == "openai":
        return f"azure/{model}" if is_azure else f"openai/{model}"
    if llm_provider in {"openailike", "azure"}:
        return f"azure/{model}"
    if llm_provider in {"bedrock", "bedrock_converse"}:
        return f"bedrock/{model}"
    return None


def _extract_stage_llm(module: dict, llm_models: list[str]) -> str | None:
    """Read the LLM picked by AutoRAG's strategy at an LLM-bearing node.

    For passage_compressor, the module dict carries ``llm`` + ``model``
    directly. For query_expansion, the same fields are exposed (the
    expansion module's ``generator_module_type`` is ``llama_index_llm``
    with ``llm``/``model`` siblings). Returns a litellm id when the
    AutoRAG provider+model resolves to a known search-space LLM, else
    None — the caller decides whether absence is fatal.
    """
    provider = module.get("llm")
    model_value = module.get("model")
    model = _scalar_or_first(model_value) if model_value is not None else None
    api_base = module.get("api_base")
    candidate = _autorag_llm_to_litellm(provider, model, api_base)
    if candidate and candidate in llm_models:
        return candidate
    if provider and provider in llm_models:
        return provider
    return None


def translate_extracted_to_trial_config(
    extracted_yaml_path: Path | str,
    search_space: SearchSpace,
) -> TrialConfig:
    raw = yaml.safe_load(Path(extracted_yaml_path).read_text(encoding="utf-8"))
    nodes = _walk_nodes(raw)
    vectordb_index = _vectordb_to_embedding(raw)

    pc_strategies = search_space.passage_compressor.strategies
    qe_strategies = search_space.query_expansion.strategies
    default_passage_compressor = "none" if "none" in pc_strategies else pc_strategies[0]
    default_query_expansion = "none" if "none" in qe_strategies else qe_strategies[0]
    fields: dict = {
        "chunking_strategy": search_space.chunking.strategies[0],
        "chunk_token_size": int(_dim_min_value(search_space.chunking.chunk_token_size)),
        "chunk_token_overlap": int(_dim_min_value(search_space.chunking.chunk_token_overlap)),
        "embedding_model": search_space.embedding.models[0],
        "index_type": search_space.retrieval.index_types[0],
        "top_k": int(_dim_min_value(search_space.retrieval.top_k)),
        "hybrid_alpha": round(_dim_midpoint(search_space.retrieval.hybrid_alpha), 4),
        "bm25_vector_fusion": search_space.retrieval.bm25_vector_fusion[0],
        "long_context_reorder": search_space.retrieval.long_context_reorder[0],
        "passage_compressor": default_passage_compressor,
        "reranker": "none" if "none" in search_space.reranker.models else search_space.reranker.models[0],
        "reranker_top_n": int(_dim_min_value(search_space.reranker.top_n)),
        "query_expansion": default_query_expansion,
        # Per-stage LLMs. Defaults: generator gets the first generator-pool
        # LLM (always set); compressor/expander are None when their stage
        # default is "none", else the first LLM in that stage's pool.
        # Overridden below from AutoRAG's resolved picks at each node.
        "generator_llm": search_space.generator.models[0],
        "compressor_llm": (
            None if default_passage_compressor == "none" else search_space.passage_compressor.models[0]
        ),
        "expander_llm": (
            None if default_query_expansion == "none" else search_space.query_expansion.models[0]
        ),
        "temperature": float(search_space.temperature.min),
        "reasoning": False,
    }

    # Chunker (v0.2 left this in the eval YAML; v0.3 separates chunking).
    # We still read it if present — the v0.3 ``autorag evaluate`` doesn't
    # emit chunker but a separate ``chunker.yaml`` consumer could write one.
    chunker_node = nodes.get("chunker") or nodes.get("chunking")
    if chunker_node:
        m = _winning_module(chunker_node)
        cm = _scalar_or_first(m.get("chunk_method", ""))
        if isinstance(cm, str):
            if cm.lower() in {"token", "recursive", "recursivecharacter"}:
                fields["chunking_strategy"] = "recursive"
            elif cm in search_space.chunking.strategies:
                fields["chunking_strategy"] = cm
        if "chunk_size" in m:
            fields["chunk_token_size"] = int(_scalar_or_first(m["chunk_size"]))
        if "chunk_overlap" in m:
            fields["chunk_token_overlap"] = int(_scalar_or_first(m["chunk_overlap"]))
        if fields["chunk_token_overlap"] >= fields["chunk_token_size"]:
            fields["chunk_token_overlap"] = max(0, fields["chunk_token_size"] - 1)

    # ===== Retrieval: v0.3 has three separate node types =====
    sem = nodes.get("semantic_retrieval")
    hybrid = nodes.get("hybrid_retrieval")
    lex = nodes.get("lexical_retrieval")

    if hybrid:
        # In v0.3 all three retrieval nodes always run; ``hybrid_retrieval``
        # produces the un-suffixed retrieval columns the reranker consumes.
        # AutoRAG's ``weight`` is the SEMANTIC weight (weight=1.0 →
        # semantic-only, weight=0.0 → BM25-only), identical convention to our
        # ``hybrid_alpha``. Use directly with no inversion.
        m = _winning_module(hybrid)
        mtype = _normalize_module_type(m.get("module_type"))
        if mtype == "hybrid_rrf":
            # RRF: ``weight`` here is rrf_k (default 60), not the semantic
            # mix — ignore for hybrid_alpha and instead flag the fusion mode.
            if "rrf" not in search_space.retrieval.bm25_vector_fusion:
                raise ValueError(
                    "AutoRAG resolved a hybrid_rrf module but the search space's "
                    "bm25_vector_fusion does not include 'rrf'. The native_config "
                    "must be regenerated against the current search space."
                )
            fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
            fields["bm25_vector_fusion"] = "rrf"
        elif mtype in {"hybrid_cc", ""}:
            # ``hybrid_cc`` (or pre-normalised module name) → alpha-blend.
            fields["bm25_vector_fusion"] = "alpha"
            weight = m.get("weight") if "weight" in m else m.get("weight_range")
            chosen_weight = float(_scalar_or_first(weight)) if weight is not None else None
            if chosen_weight is None or chosen_weight >= 1.0:
                # Fully semantic — collapse to vector_only when the space supports it.
                if IndexType.VECTOR_ONLY in search_space.retrieval.index_types:
                    fields["index_type"] = IndexType.VECTOR_ONLY
                else:
                    fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
                    fields["hybrid_alpha"] = 1.0
            else:
                fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
                fields["hybrid_alpha"] = round(max(0.0, min(1.0, chosen_weight)), 4)
        else:
            raise ValueError(
                f"Unknown hybrid_retrieval module_type {mtype!r} from AutoRAG. "
                "Add a translator branch before trusting this row."
            )
        tk = _read_top_k(hybrid, m)
        if tk is not None:
            fields["top_k"] = tk
    elif sem:
        m = _winning_module(sem)
        fields["index_type"] = IndexType.VECTOR_ONLY
        tk = _read_top_k(sem, m)
        if tk is not None:
            fields["top_k"] = tk
    elif lex:
        # Lexical-only — best-effort: treat as hybrid with vector=0.
        m = _winning_module(lex)
        if IndexType.HYBRID_BM25_VECTOR in search_space.retrieval.index_types:
            fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
            fields["hybrid_alpha"] = 0.0
        tk = _read_top_k(lex, m)
        if tk is not None:
            fields["top_k"] = tk

    # Embedding model — read from semantic_retrieval's vectordb reference,
    # regardless of which retrieval node "won". Falls back to defaults if
    # missing.
    if sem:
        sem_m = _winning_module(sem)
        if _normalize_module_type(sem_m.get("module_type")) == "vectordb":
            vname = sem_m.get("vectordb")
            if vname and vname in vectordb_index and vectordb_index[vname] in search_space.embedding.models:
                fields["embedding_model"] = vectordb_index[vname]
            elif "embedding_model" in sem_m and _scalar_or_first(sem_m["embedding_model"]) in search_space.embedding.models:
                fields["embedding_model"] = _scalar_or_first(sem_m["embedding_model"])

    # ===== Reranker =====
    reranker_node = nodes.get("passage_reranker")
    if reranker_node:
        m = _winning_module(reranker_node)
        mtype = _normalize_module_type(m.get("module_type"))
        if mtype in {"pass_reranker", "pass_passage_reranker"}:
            fields["reranker"] = "none"
        else:
            model_name = m.get("model_name") or m.get("model")
            if model_name and model_name in search_space.reranker.models:
                fields["reranker"] = model_name
            elif "none" in search_space.reranker.models:
                fields["reranker"] = "none"
        if fields["reranker"] != "none":
            tk = _read_top_k(reranker_node, m)
            if tk is not None:
                fields["reranker_top_n"] = tk

    # ===== Query expansion =====
    qe_node = nodes.get("query_expansion")
    if qe_node:
        m = _winning_module(qe_node)
        mtype = _normalize_module_type(m.get("module_type"))
        if mtype == "pass_query_expansion":
            fields["query_expansion"] = "none"
            fields["expander_llm"] = None
        elif mtype == "hyde" and "hyde" in qe_strategies:
            fields["query_expansion"] = "hyde"
        elif mtype == "multi_query_expansion" and "multi_query" in qe_strategies:
            fields["query_expansion"] = "multi_query"
        elif mtype == "query_decompose":
            if "query_decompose" not in qe_strategies:
                raise ValueError(
                    "AutoRAG resolved a query_decompose module, but the search space's "
                    "query_expansion.strategies does not include 'query_decompose'. The "
                    "native_config must be regenerated against the current search space."
                )
            fields["query_expansion"] = "query_decompose"
        elif mtype in qe_strategies:
            fields["query_expansion"] = mtype
        # When the resolved strategy actually runs an LLM, read which one
        # AutoRAG's strategy picked. None means the strategy was
        # ``pass_query_expansion``. Falls back to the search-space's first
        # LLM when the AutoRAG module doesn't surface llm/model fields —
        # legacy ``extracted_sample.yaml`` files (and some pre-v0.3.x
        # fixtures) omit them.
        if fields["query_expansion"] != "none":
            chosen_expander = _extract_stage_llm(m, list(search_space.query_expansion.models))
            fields["expander_llm"] = chosen_expander or search_space.query_expansion.models[0]

    # ===== Passage compressor =====
    pc_node = nodes.get("passage_compressor")
    if pc_node:
        m = _winning_module(pc_node)
        mtype = _normalize_module_type(m.get("module_type"))
        if mtype in {"pass_compressor", ""}:
            fields["passage_compressor"] = "none"
            fields["compressor_llm"] = None
        elif mtype in {"tree_summarize", "refine"}:
            if mtype not in pc_strategies:
                raise ValueError(
                    f"AutoRAG resolved a {mtype!r} passage_compressor module, but the "
                    f"search space's passage_compressor.strategies does not include "
                    f"{mtype!r}. The native_config must be regenerated against the "
                    "current search space."
                )
            fields["passage_compressor"] = mtype
        else:
            raise ValueError(
                f"Unknown passage_compressor module_type {mtype!r} from AutoRAG. "
                "Add a translator branch before trusting this row."
            )
        # Same fallback as expander: when the compressor actually runs,
        # read AutoRAG's pick, else fall back to first LLM in pool.
        if fields["passage_compressor"] != "none":
            chosen_compressor = _extract_stage_llm(m, list(search_space.passage_compressor.models))
            fields["compressor_llm"] = chosen_compressor or search_space.passage_compressor.models[0]

    # ===== Prompt maker =====
    pm_node = nodes.get("prompt_maker")
    if pm_node:
        m = _winning_module(pm_node)
        mtype = _normalize_module_type(m.get("module_type"))
        if mtype == "long_context_reorder":
            if True not in search_space.retrieval.long_context_reorder:
                raise ValueError(
                    "AutoRAG resolved a long_context_reorder prompt_maker module, but "
                    "the search space's retrieval.long_context_reorder does not include "
                    "True. The native_config must be regenerated against the current "
                    "search space."
                )
            fields["long_context_reorder"] = True
        elif mtype in {"fstring", ""}:
            fields["long_context_reorder"] = False
        else:
            raise ValueError(
                f"Unknown prompt_maker module_type {mtype!r} from AutoRAG. "
                "Add a translator branch before trusting this row."
            )

    # ===== Generator =====
    gen = nodes.get("generator")
    if gen:
        m = _winning_module(gen)
        chosen_generator = _extract_stage_llm(m, list(search_space.generator.models))
        if chosen_generator is not None:
            fields["generator_llm"] = chosen_generator
        if "temperature" in m:
            fields["temperature"] = float(_scalar_or_first(m["temperature"]))

    if fields["reranker"] != "none" and fields["reranker_top_n"] > fields["top_k"]:
        fields["reranker_top_n"] = fields["top_k"]

    return TrialConfig.model_validate(fields)
