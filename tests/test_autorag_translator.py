"""Tests for the AutoRAG ``extracted_sample.yaml`` → ``TrialConfig`` translator."""

from __future__ import annotations

import yaml

from agentic_autorag.config.models import (
    ChunkingSearchSpace,
    IndexType,
    NumericRange,
    RerankerSearchSpace,
    SearchSpace,
)

from autorag_bench.methods.autorag.translator import translate_extracted_to_trial_config


def _curated_space() -> SearchSpace:
    return SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive", "fixed"],
            chunk_token_size=NumericRange(min=256, max=512),
            chunk_token_overlap=NumericRange(min=0, max=64),
        ),
        embedding_models=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"],
        index_types=[IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
        top_k=NumericRange(min=3, max=20),
        hybrid_alpha=NumericRange(min=0.0, max=1.0),
        reranker=RerankerSearchSpace(
            models=["none", "BAAI/bge-reranker-v2-m3"],
            top_n=NumericRange(min=3, max=10),
        ),
        query_expansion=["none", "hyde", "multi_query"],
        llm_models=["bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"],
        temperature=NumericRange(min=1.0, max=1.0),
    )


def _write_extracted_yaml(tmp_path, content: dict) -> str:
    path = tmp_path / "extracted_sample.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return str(path)


def test_translates_vector_only_winner(tmp_path) -> None:
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "chunker",
                        "modules": [{"chunk_method": "token", "chunk_size": 384, "chunk_overlap": 32}],
                    },
                    {
                        "node_type": "retrieval",
                        "modules": [{"module_type": "vectordb", "embedding_model": "BAAI/bge-m3", "top_k": 10}],
                    },
                    {"node_type": "passage_reranker", "modules": [{"module_type": "pass_passage_reranker"}]},
                    {"node_type": "query_expansion", "modules": [{"module_type": "pass_query_expansion"}]},
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
                                "temperature": 1.0,
                            }
                        ],
                    },
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())

    assert config.chunking_strategy == "recursive"  # token → recursive
    assert config.chunk_token_size == 384
    assert config.chunk_token_overlap == 32
    assert config.index_type == IndexType.VECTOR_ONLY
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.top_k == 10
    assert config.reranker == "none"
    assert config.query_expansion == "none"
    assert config.llm_model == "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_translates_hybrid_inverts_weight(tmp_path) -> None:
    """If AutoRAG picks BM25 weight=0.3, we store hybrid_alpha=0.7 (vector weight)."""
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "modules": [{"module_type": "hybrid_cc", "weight": 0.3, "top_k": 8}],
                    },
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.index_type == IndexType.HYBRID_BM25_VECTOR
    assert config.hybrid_alpha == 0.7
    assert config.top_k == 8


def test_translates_reranker_with_explicit_model(tmp_path) -> None:
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "modules": [{"module_type": "vectordb", "embedding_model": "BAAI/bge-m3", "top_k": 10}],
                    },
                    {
                        "node_type": "passage_reranker",
                        "modules": [
                            {
                                "module_type": "flag_embedding_reranker",
                                "model_name": "BAAI/bge-reranker-v2-m3",
                                "top_k": 5,
                            }
                        ],
                    },
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.reranker == "BAAI/bge-reranker-v2-m3"
    assert config.reranker_top_n == 5
    assert config.top_k == 10  # the clamp invariant: reranker_top_n <= top_k


def test_clamps_reranker_top_n_to_top_k(tmp_path) -> None:
    """reranker_top_n must be <= top_k; the translator clamps when AutoRAG's
    discretization picks an inconsistent pair."""
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "modules": [{"module_type": "vectordb", "embedding_model": "BAAI/bge-m3", "top_k": 5}],
                    },
                    {
                        "node_type": "passage_reranker",
                        "modules": [
                            {"module_type": "flag_embedding_reranker", "model_name": "BAAI/bge-reranker-v2-m3", "top_k": 10}
                        ],
                    },
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.reranker_top_n == config.top_k == 5


def test_translates_query_expansion_module_types(tmp_path) -> None:
    for autorag_type, expected in [("hyde", "hyde"), ("multi_query_expansion", "multi_query")]:
        extracted = {
            "node_lines": [
                {"nodes": [{"node_type": "query_expansion", "modules": [{"module_type": autorag_type}]}]}
            ]
        }
        path = _write_extracted_yaml(tmp_path, extracted)
        config = translate_extracted_to_trial_config(path, _curated_space())
        assert config.query_expansion == expected, f"AutoRAG {autorag_type} should map to {expected}"


def test_falls_back_to_search_space_minimum_when_node_missing(tmp_path) -> None:
    extracted = {"node_lines": [{"nodes": []}]}
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.top_k == 3  # search space min
    assert config.reranker == "none"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_clamps_overlap_below_chunk_size(tmp_path) -> None:
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "chunker",
                        "modules": [{"chunk_method": "token", "chunk_size": 100, "chunk_overlap": 100}],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.chunk_token_overlap < config.chunk_token_size


def test_handles_list_valued_resolved_fields(tmp_path) -> None:
    """Older AutoRAG versions left singleton lists in extracted_sample.yaml."""
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "retrieval",
                        "modules": [{"module_type": "vectordb", "embedding_model": ["BAAI/bge-m3"], "top_k": [12]}],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.top_k == 12
