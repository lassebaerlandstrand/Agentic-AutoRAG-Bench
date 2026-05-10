"""Tests for the AutoRAG v0.3.x ``extracted_sample.yaml`` → ``TrialConfig`` translator."""

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
        llm_models=["azure/gpt-4o-mini"],
        temperature=NumericRange(min=1.0, max=1.0),
    )


def _write_extracted_yaml(tmp_path, content: dict) -> str:
    path = tmp_path / "extracted_sample.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return str(path)


def test_translates_v03_semantic_retrieval_with_vectordb_reference(tmp_path) -> None:
    """v0.3 names retrieval modules by vectordb registry name; we resolve back to the HF model."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_1",
                "db_type": "chroma",
                "embedding_model": "huggingface",
                "embedding_model_kwargs": {"model_name": "BAAI/bge-m3"},
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "semantic_retrieval",
                        "strategy": {"metrics": ["retrieval_f1"]},
                        "modules": [{"module_type": "vectordb", "vectordb": "embed_1", "top_k": 10}],
                    },
                    {
                        "node_type": "passage_reranker",
                        "modules": [{"module_type": "pass_reranker"}],
                    },
                    {"node_type": "query_expansion", "modules": [{"module_type": "pass_query_expansion"}]},
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "openailike",
                                "model": "gpt-4o-mini",
                                "temperature": 1.0,
                            }
                        ],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.index_type == IndexType.VECTOR_ONLY
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.top_k == 10
    assert config.reranker == "none"
    assert config.query_expansion == "none"
    assert config.llm_model == "azure/gpt-4o-mini"


def test_translates_v03_hybrid_retrieval_inverts_weight(tmp_path) -> None:
    """v0.3 hybrid_cc.weight on extracted_sample = BM25 weight; we invert to hybrid_alpha."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": "huggingface",
                "embedding_model_kwargs": {"model_name": "BAAI/bge-m3"},
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_cc", "weight": 0.3, "top_k": 8}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.index_type == IndexType.HYBRID_BM25_VECTOR
    assert config.hybrid_alpha == 0.7
    assert config.top_k == 8


def test_translates_v03_reranker_with_explicit_model(tmp_path) -> None:
    extracted = {
        "vectordb": [
            {
                "name": "embed_1",
                "embedding_model": "huggingface",
                "embedding_model_kwargs": {"model_name": "BAAI/bge-m3"},
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "semantic_retrieval",
                        "modules": [{"module_type": "vectordb", "vectordb": "embed_1", "top_k": 10}],
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
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.reranker == "BAAI/bge-reranker-v2-m3"
    assert config.reranker_top_n == 5
    assert config.top_k == 10  # the clamp invariant: reranker_top_n <= top_k


def test_accepts_pass_reranker_v03_and_pass_passage_reranker_v02(tmp_path) -> None:
    """Both spellings of the pass-through reranker module are accepted."""
    for spelling in ("pass_reranker", "pass_passage_reranker"):
        extracted = {
            "node_lines": [
                {"nodes": [{"node_type": "passage_reranker", "modules": [{"module_type": spelling}]}]}
            ]
        }
        path = _write_extracted_yaml(tmp_path, extracted)
        config = translate_extracted_to_trial_config(path, _curated_space())
        assert config.reranker == "none", f"{spelling!r} should map to reranker=none"


def test_clamps_reranker_top_n_to_top_k(tmp_path) -> None:
    extracted = {
        "vectordb": [
            {
                "name": "embed_1",
                "embedding_model": "huggingface",
                "embedding_model_kwargs": {"model_name": "BAAI/bge-m3"},
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "semantic_retrieval",
                        "modules": [{"module_type": "vectordb", "vectordb": "embed_1", "top_k": 5}],
                    },
                    {
                        "node_type": "passage_reranker",
                        "modules": [
                            {"module_type": "flag_embedding_reranker", "model_name": "BAAI/bge-reranker-v2-m3", "top_k": 10}
                        ],
                    },
                ]
            }
        ],
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
                        "modules": [{"chunk_method": "Token", "chunk_size": 100, "chunk_overlap": 100}],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.chunk_token_overlap < config.chunk_token_size


def test_falls_back_when_vectordb_name_not_in_registry(tmp_path) -> None:
    """If extracted_sample.yaml references a vectordb name that isn't declared, use the search-space default."""
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "semantic_retrieval",
                        "modules": [{"module_type": "vectordb", "vectordb": "missing_name", "top_k": 7}],
                    }
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.top_k == 7
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_openailike_model_translates_to_azure_litellm_id(tmp_path) -> None:
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "openailike",
                                "model": "gpt-4o-mini",
                            }
                        ],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.llm_model == "azure/gpt-4o-mini"
