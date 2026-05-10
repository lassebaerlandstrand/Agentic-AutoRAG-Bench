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

from pathlib import Path

import yaml

from agentic_autorag.config.models import IndexType, NumericRange, SearchSpace, TrialConfig


def _midpoint(r: NumericRange) -> float:
    return (r.min + r.max) / 2.0


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


def _normalize_module_type(s: str | None) -> str:
    """AutoRAG v0.3 extract_best_config emits CamelCase ``module_type`` strings
    (e.g. ``PassReranker``, ``VectorDB``, ``HybridCC``, ``LlamaIndexLLM``)
    even though the input config uses snake_case. Normalise to lowercase
    snake-equivalent for matching.
    """
    if not s:
        return ""
    return s.strip().lower()


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
    """
    if not llm_provider or not model:
        return None
    api_base_str = (api_base or "").lower()
    is_azure = "azure" in api_base_str or "cognitiveservices" in api_base_str
    if llm_provider == "openai":
        return f"azure/{model}" if is_azure else f"openai/{model}"
    if llm_provider in {"openailike", "azure"}:
        return f"azure/{model}"
    if llm_provider == "bedrock":
        return f"bedrock/{model}"
    return None


def translate_extracted_to_trial_config(
    extracted_yaml_path: Path | str,
    search_space: SearchSpace,
) -> TrialConfig:
    raw = yaml.safe_load(Path(extracted_yaml_path).read_text(encoding="utf-8"))
    nodes = _walk_nodes(raw)
    vectordb_index = _vectordb_to_embedding(raw)

    fields: dict = {
        "chunking_strategy": search_space.chunking.strategies[0],
        "chunk_token_size": int(search_space.chunking.chunk_token_size.min),
        "chunk_token_overlap": int(search_space.chunking.chunk_token_overlap.min),
        "embedding_model": search_space.embedding_models[0],
        "index_type": search_space.index_types[0],
        "top_k": int(search_space.top_k.min),
        "hybrid_alpha": round(_midpoint(search_space.hybrid_alpha), 4),
        "reranker": "none" if "none" in search_space.reranker.models else search_space.reranker.models[0],
        "reranker_top_n": int(search_space.reranker.top_n.min),
        "query_expansion": "none" if "none" in search_space.query_expansion else search_space.query_expansion[0],
        "llm_model": search_space.llm_models[0],
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
        # In v0.3, all three retrieval nodes always run; ``hybrid_retrieval``
        # produces the un-suffixed retrieval columns the reranker consumes.
        # The chosen BM25 weight tells us where on the BM25↔vector spectrum
        # the optimum landed: weight=0 → fully vector → hybrid_alpha=1.0;
        # weight=1 → fully BM25 → hybrid_alpha=0.0.
        m = _winning_module(hybrid)
        weight = m.get("weight") if "weight" in m else m.get("weight_range")
        chosen_weight = float(_scalar_or_first(weight)) if weight is not None else None
        # Map the BM25 weight to (index_type, hybrid_alpha):
        if chosen_weight is None or chosen_weight in (0.0, 0):
            # Fully vector, treated as vector_only when our space allows it.
            if IndexType.VECTOR_ONLY in search_space.index_types:
                fields["index_type"] = IndexType.VECTOR_ONLY
            else:
                fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
                fields["hybrid_alpha"] = 1.0
        else:
            fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
            fields["hybrid_alpha"] = round(max(0.0, min(1.0, 1.0 - chosen_weight)), 4)
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
        if IndexType.HYBRID_BM25_VECTOR in search_space.index_types:
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
            if vname and vname in vectordb_index and vectordb_index[vname] in search_space.embedding_models:
                fields["embedding_model"] = vectordb_index[vname]
            elif "embedding_model" in sem_m and _scalar_or_first(sem_m["embedding_model"]) in search_space.embedding_models:
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
        elif mtype == "hyde" and "hyde" in search_space.query_expansion:
            fields["query_expansion"] = "hyde"
        elif mtype == "multi_query_expansion" and "multi_query" in search_space.query_expansion:
            fields["query_expansion"] = "multi_query"
        elif mtype in search_space.query_expansion:
            fields["query_expansion"] = mtype

    # ===== Generator =====
    gen = nodes.get("generator")
    if gen:
        m = _winning_module(gen)
        provider = m.get("llm")
        model = _scalar_or_first(m.get("model")) if m.get("model") is not None else None
        api_base = m.get("api_base")
        # First try the v0.3 reverse-map (provider + model + base → litellm).
        candidate = _autorag_llm_to_litellm(provider, model, api_base)
        if candidate and candidate in search_space.llm_models:
            fields["llm_model"] = candidate
        # Fall back to v0.2 form: ``llm`` already contained the full litellm id.
        elif provider and provider in search_space.llm_models:
            fields["llm_model"] = provider
        if "temperature" in m:
            fields["temperature"] = float(_scalar_or_first(m["temperature"]))

    if fields["reranker"] != "none" and fields["reranker_top_n"] > fields["top_k"]:
        fields["reranker_top_n"] = fields["top_k"]

    return TrialConfig.model_validate(fields)
