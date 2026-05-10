"""Translate AutoRAG's resolved ``extracted_sample.yaml`` → our ``TrialConfig``.

After ``autorag evaluate`` finishes, the winning pipeline is materialised as
``extracted_sample.yaml`` (one resolved module per node). Because ``native_config``
constrains AutoRAG to exactly our dimensions, every winning module corresponds
1:1 to a ``TrialConfig`` field — this is structured field extraction, not a
lossy mapping. Anything missing falls back to the search-space minimum so the
result is always a valid ``TrialConfig``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_autorag.config.models import IndexType, NumericRange, SearchSpace, TrialConfig


def _midpoint(r: NumericRange) -> float:
    return (r.min + r.max) / 2.0


def _walk_nodes(extracted: dict) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    for line in extracted.get("node_lines", []) or []:
        for node in line.get("nodes", []) or []:
            ntype = node.get("node_type")
            if ntype:
                nodes[ntype] = node
    return nodes


def _winning_module(node: dict) -> dict:
    modules = node.get("modules", []) or []
    return modules[0] if modules else {}


def _scalar_or_first(value: object) -> object:
    return value[0] if isinstance(value, list) and value else value


def translate_extracted_to_trial_config(
    extracted_yaml_path: Path | str,
    search_space: SearchSpace,
) -> TrialConfig:
    raw = yaml.safe_load(Path(extracted_yaml_path).read_text(encoding="utf-8"))
    nodes = _walk_nodes(raw)

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

    chunker = nodes.get("chunker") or nodes.get("chunking")
    if chunker:
        m = _winning_module(chunker)
        if "chunk_method" in m:
            cm = _scalar_or_first(m["chunk_method"])
            if cm == "token":
                fields["chunking_strategy"] = "recursive"
            elif cm in search_space.chunking.strategies:
                fields["chunking_strategy"] = cm
        if "chunk_size" in m:
            fields["chunk_token_size"] = int(_scalar_or_first(m["chunk_size"]))
        if "chunk_overlap" in m:
            fields["chunk_token_overlap"] = int(_scalar_or_first(m["chunk_overlap"]))
        if fields["chunk_token_overlap"] >= fields["chunk_token_size"]:
            fields["chunk_token_overlap"] = max(0, fields["chunk_token_size"] - 1)

    retrieval = nodes.get("retrieval") or nodes.get("retrieve")
    if retrieval:
        m = _winning_module(retrieval)
        mtype = m.get("module_type", "")
        if mtype == "vectordb":
            fields["index_type"] = IndexType.VECTOR_ONLY
            embed = _scalar_or_first(m.get("embedding_model"))
            if embed in search_space.embedding_models:
                fields["embedding_model"] = embed
        elif mtype in {"hybrid_cc", "hybrid_rrf", "bm25"}:
            fields["index_type"] = IndexType.HYBRID_BM25_VECTOR
            weight = m.get("weight")
            if weight is not None:
                # AutoRAG weight is BM25's; we store vector's complement.
                w = float(_scalar_or_first(weight))
                fields["hybrid_alpha"] = round(max(0.0, min(1.0, 1.0 - w)), 4)
        if "top_k" in m:
            fields["top_k"] = int(_scalar_or_first(m["top_k"]))
        else:
            strat = retrieval.get("strategy", {}) or {}
            tk = strat.get("top_k")
            if tk:
                fields["top_k"] = int(_scalar_or_first(tk))

    reranker = nodes.get("passage_reranker")
    if reranker:
        m = _winning_module(reranker)
        if m.get("module_type") == "pass_passage_reranker":
            fields["reranker"] = "none"
        else:
            model_name = m.get("model_name") or m.get("model")
            if model_name and model_name in search_space.reranker.models:
                fields["reranker"] = model_name
            elif "none" in search_space.reranker.models:
                fields["reranker"] = "none"
        if "top_k" in m and fields["reranker"] != "none":
            fields["reranker_top_n"] = int(_scalar_or_first(m["top_k"]))

    qe = nodes.get("query_expansion")
    if qe:
        m = _winning_module(qe)
        mtype = m.get("module_type", "")
        if mtype == "pass_query_expansion":
            fields["query_expansion"] = "none"
        elif mtype == "hyde" and "hyde" in search_space.query_expansion:
            fields["query_expansion"] = "hyde"
        elif mtype == "multi_query_expansion" and "multi_query" in search_space.query_expansion:
            fields["query_expansion"] = "multi_query"
        elif mtype in search_space.query_expansion:
            fields["query_expansion"] = mtype

    gen = nodes.get("generator")
    if gen:
        m = _winning_module(gen)
        llm = m.get("llm") or m.get("model")
        if llm and llm in search_space.llm_models:
            fields["llm_model"] = llm
        if "temperature" in m:
            fields["temperature"] = float(_scalar_or_first(m["temperature"]))

    if fields["reranker"] != "none" and fields["reranker_top_n"] > fields["top_k"]:
        fields["reranker_top_n"] = fields["top_k"]

    return TrialConfig.model_validate(fields)
