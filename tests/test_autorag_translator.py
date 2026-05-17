"""Tests for the AutoRAG v0.3.x ``extracted_sample.yaml`` → ``TrialConfig`` translator."""

from __future__ import annotations

import yaml

from agentic_autorag.config.models import (
    ChunkingSearchSpace,
    DiscreteValues,
    IndexType,
    NumericRange,
    RerankerSearchSpace,
    SearchSpace,
    StageLLMs,
)

from agentic_autorag_bench.methods.autorag.translator import translate_extracted_to_trial_config


def _curated_space() -> SearchSpace:
    return SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive", "fixed"],
            chunk_token_size=DiscreteValues(values=[256, 512]),
            chunk_token_overlap=DiscreteValues(values=[0, 64]),
        ),
        embedding_models=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"],
        index_types=[IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
        top_k=DiscreteValues(values=[3, 5, 10, 15, 20]),
        hybrid_alpha=DiscreteValues(values=[0.0, 0.5, 1.0]),
        reranker=RerankerSearchSpace(
            models=["none", "BAAI/bge-reranker-v2-m3"],
            top_n=DiscreteValues(values=[3, 5, 10]),
        ),
        query_expansion=["none", "hyde", "multi_query"],
        llm_models=StageLLMs.uniform(["azure/gpt-4o-mini"]),
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
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
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
    assert config.generator_llm == "azure/gpt-4o-mini"


def test_translates_v03_hybrid_retrieval_passes_weight_through_as_alpha(tmp_path) -> None:
    """AutoRAG's hybrid_cc.weight is the SEMANTIC weight (weight=1.0 → semantic-only),
    identical convention to our hybrid_alpha: passed through directly with no inversion.
    """
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
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
    assert config.hybrid_alpha == 0.3
    assert config.top_k == 8


def test_translates_v03_hybrid_weight_one_collapses_to_vector_only_when_allowed(tmp_path) -> None:
    """weight=1.0 → fully semantic; collapse to vector_only when the search space allows it."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_cc", "weight": 1.0, "top_k": 5}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.index_type == IndexType.VECTOR_ONLY


def test_translates_v03_hybrid_rrf_sets_bm25_vector_fusion_to_rrf(tmp_path) -> None:
    """When AutoRAG resolves ``hybrid_rrf`` as the winning hybrid module, the
    translator sets ``bm25_vector_fusion='rrf'`` (and leaves ``hybrid_alpha``
    at its inert default since RRF ignores it)."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_rrf", "weight": 60, "top_k": 7}],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.bm25_vector_fusion = ["alpha", "rrf"]
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, space)
    assert config.index_type == IndexType.HYBRID_BM25_VECTOR
    assert config.bm25_vector_fusion == "rrf"
    assert config.top_k == 7


def test_translator_raises_when_hybrid_rrf_resolved_but_not_in_search_space(tmp_path) -> None:
    """If AutoRAG emits ``hybrid_rrf`` but the bench's search space does not
    enumerate RRF, the translator raises rather than silently re-labelling the
    row as alpha-blend — paper integrity over silent fallback."""
    import pytest

    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_rrf", "weight": 60, "top_k": 5}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    # _curated_space() has the default bm25_vector_fusion=['alpha'] only.
    with pytest.raises(ValueError, match="hybrid_rrf"):
        translate_extracted_to_trial_config(path, _curated_space())


def test_translator_recognizes_query_decompose(tmp_path) -> None:
    """``query_decompose`` winning expansion → ``query_expansion='query_decompose'``."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "query_expansion",
                        "modules": [{"module_type": "query_decompose"}],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.query_expansion = ["none", "query_decompose"]
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, space)
    assert config.query_expansion == "query_decompose"


def test_translator_raises_when_query_decompose_resolved_but_excluded(tmp_path) -> None:
    import pytest

    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "query_expansion",
                        "modules": [{"module_type": "query_decompose"}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    # _curated_space() has query_expansion={'none', 'hyde', 'multi_query'} — no 'query_decompose'.
    with pytest.raises(ValueError, match="query_decompose"):
        translate_extracted_to_trial_config(path, _curated_space())


def test_translator_recognizes_tree_summarize_passage_compressor(tmp_path) -> None:
    """``tree_summarize`` winning compressor → ``passage_compressor='tree_summarize'``."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "passage_compressor",
                        "modules": [{"module_type": "tree_summarize", "llm": "openai", "model": "gpt-4o-mini"}],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.passage_compressor = ["none", "tree_summarize", "refine"]
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, space)
    assert config.passage_compressor == "tree_summarize"


def test_translator_pass_compressor_maps_to_none(tmp_path) -> None:
    """``pass_compressor`` → ``passage_compressor='none'``."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "passage_compressor",
                        "modules": [{"module_type": "pass_compressor"}],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.passage_compressor = ["none", "tree_summarize"]
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, space)
    assert config.passage_compressor == "none"


def test_translator_raises_when_compressor_resolved_but_excluded(tmp_path) -> None:
    import pytest

    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "passage_compressor",
                        "modules": [{"module_type": "refine"}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    # _curated_space() has passage_compressor=['none'] by default.
    with pytest.raises(ValueError, match="refine"):
        translate_extracted_to_trial_config(path, _curated_space())


def test_translator_recognizes_long_context_reorder(tmp_path) -> None:
    """When AutoRAG resolves ``long_context_reorder`` as the winning
    prompt_maker module, the translator sets ``long_context_reorder=True``."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "prompt_maker",
                        "modules": [
                            {"module_type": "long_context_reorder", "prompt": ["...{retrieved_contents}..."]}
                        ],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.long_context_reorder = [False, True]
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, space)
    assert config.long_context_reorder is True


def test_translator_raises_on_long_context_reorder_when_excluded_from_search_space(tmp_path) -> None:
    """If AutoRAG emits long_context_reorder but the search space excludes True,
    raise rather than silently downgrading the row."""
    import pytest

    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "prompt_maker",
                        "modules": [{"module_type": "long_context_reorder", "prompt": ["x"]}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    # _curated_space() has the default long_context_reorder=[False].
    with pytest.raises(ValueError, match="long_context_reorder"):
        translate_extracted_to_trial_config(path, _curated_space())


def test_translator_raises_on_unknown_prompt_maker_module(tmp_path) -> None:
    """Unknown prompt_maker module_type raises rather than silently mapping."""
    import pytest

    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "prompt_maker",
                        "modules": [{"module_type": "window_replacement", "prompt": ["x"]}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    with pytest.raises(ValueError, match="window_replacement"):
        translate_extracted_to_trial_config(path, _curated_space())


def test_translator_raises_on_unknown_hybrid_module(tmp_path) -> None:
    """Defensive: an unexpected hybrid_retrieval module_type from AutoRAG (e.g.
    a new fusion strategy not yet wired up) should raise so the bench fails
    loudly instead of silently mis-labelling the row."""
    import pytest

    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_future_unknown", "top_k": 5}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    with pytest.raises(ValueError, match="hybrid_future_unknown"):
        translate_extracted_to_trial_config(path, _curated_space())


def test_native_config_then_translator_roundtrip_for_pipeline_dimensions(tmp_path) -> None:
    """An AutoRAG config emitted by ``generate_autorag_config`` for the
    pipeline dimensions (``bm25_vector_fusion``, ``long_context_reorder``,
    ``passage_compressor``, ``query_expansion``) must round-trip through
    the translator into matching ``TrialConfig`` values, so that picking
    ``hybrid_rrf + long_context_reorder + tree_summarize + query_decompose``
    on the AutoRAG side resolves to the same configuration when re-evaluated
    by the framework.
    """
    from agentic_autorag_bench.methods.autorag.native_config import generate_autorag_config

    space = _curated_space()
    space.bm25_vector_fusion = ["alpha", "rrf"]
    space.long_context_reorder = [False, True]
    space.passage_compressor = ["none", "tree_summarize", "refine"]
    space.query_expansion = ["none", "hyde", "multi_query", "query_decompose"]

    ar_config, _ = generate_autorag_config(space, qa_variant="ragas")

    # Simulate AutoRAG resolving the "max-feature" winners: pick the LAST
    # module from each new-dimension node. This is the worst-case for the
    # translator — all the new branches fire.
    def _pick_last(node_type: str) -> dict:
        for line in ar_config["node_lines"]:
            for node in line["nodes"]:
                if node["node_type"] == node_type:
                    return {**node, "modules": [node["modules"][-1]]}
        raise AssertionError(f"{node_type} not in config")

    resolved = {
        "vectordb": ar_config["vectordb"],
        "node_lines": [
            {
                "nodes": [
                    _pick_last("query_expansion"),
                    _pick_last("hybrid_retrieval"),
                    _pick_last("passage_reranker"),
                    _pick_last("passage_compressor"),
                    _pick_last("prompt_maker"),
                    _pick_last("generator"),
                ]
            }
        ],
    }
    # Set the hybrid_retrieval module's top_k explicitly so the translator
    # has something to read (AutoRAG's resolved YAML stores top_k at module
    # level after the strategy selects a value).
    for node in resolved["node_lines"][0]["nodes"]:
        if node["node_type"] == "hybrid_retrieval":
            node["modules"][0]["top_k"] = 10

    path = _write_extracted_yaml(tmp_path, resolved)
    trial = translate_extracted_to_trial_config(path, space)

    # The "last module" picks per node correspond to:
    #   query_expansion: query_decompose (4th option)
    #   hybrid_retrieval: hybrid_rrf (2nd option)
    #   passage_compressor: refine (3rd option)
    #   prompt_maker: long_context_reorder (2nd option, since False/True ordering)
    assert trial.query_expansion == "query_decompose"
    assert trial.bm25_vector_fusion == "rrf"
    assert trial.index_type == IndexType.HYBRID_BM25_VECTOR
    assert trial.passage_compressor == "refine"
    assert trial.long_context_reorder is True
    assert trial.top_k == 10


def test_native_config_then_translator_roundtrip_with_per_stage_llms(tmp_path) -> None:
    """When AutoRAG picks different LLMs at compressor / expander / generator
    nodes (the per-stage independence the new design exposes), the translator
    must produce a TrialConfig where compressor_llm, expander_llm, and
    generator_llm reflect AutoRAG's actual picks — NOT the search-space
    first-LLM default. This is the fairness fix: the bench evaluates the
    pipeline AutoRAG actually ran, not the one we wish it had run."""
    extracted = {
        "vectordb": [{"name": "embed_0", "embedding_model": "BAAI/bge-m3"}],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "query_expansion",
                        "modules": [
                            {
                                "module_type": "query_decompose",
                                "llm": "openai",
                                "model": ["o4-mini"],
                                "api_base": "https://example.azure.com/openai/v1",
                                "prompt": "",
                            }
                        ],
                    },
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_cc", "weight": 1.0, "top_k": 5}],
                    },
                    {
                        "node_type": "passage_compressor",
                        "modules": [
                            {
                                "module_type": "tree_summarize",
                                "llm": "openai",
                                "model": ["gpt-4o-mini"],
                                "api_base": "https://example.azure.com/openai/v1",
                            }
                        ],
                    },
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "openai",
                                "model": ["o4-mini"],
                                "api_base": "https://example.azure.com/openai/v1",
                            }
                        ],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.llm_models = StageLLMs.uniform(["azure/gpt-4o-mini", "azure/o4-mini"])
    space.passage_compressor = ["none", "tree_summarize"]
    space.query_expansion = ["none", "query_decompose"]
    path = _write_extracted_yaml(tmp_path, extracted)
    trial = translate_extracted_to_trial_config(path, space)

    assert trial.passage_compressor == "tree_summarize"
    assert trial.compressor_llm == "azure/gpt-4o-mini"
    assert trial.query_expansion == "query_decompose"
    assert trial.expander_llm == "azure/o4-mini"
    assert trial.generator_llm == "azure/o4-mini"


def test_translator_sets_stage_llm_to_none_when_stage_inactive(tmp_path) -> None:
    """When AutoRAG resolves pass_compressor / pass_query_expansion, the
    corresponding compressor_llm / expander_llm must be None — the TrialConfig
    validator rejects a non-None LLM on a dead stage."""
    extracted = {
        "vectordb": [{"name": "embed_0", "embedding_model": "BAAI/bge-m3"}],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "query_expansion",
                        "modules": [{"module_type": "pass_query_expansion"}],
                    },
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_cc", "weight": 1.0, "top_k": 5}],
                    },
                    {
                        "node_type": "passage_compressor",
                        "modules": [{"module_type": "pass_compressor"}],
                    },
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "openai",
                                "model": ["gpt-4o-mini"],
                                "api_base": "https://example.azure.com/openai/v1",
                            }
                        ],
                    },
                ]
            }
        ],
    }
    space = _curated_space()
    space.passage_compressor = ["none", "tree_summarize"]
    space.query_expansion = ["none", "hyde"]
    path = _write_extracted_yaml(tmp_path, extracted)
    trial = translate_extracted_to_trial_config(path, space)

    assert trial.passage_compressor == "none"
    assert trial.compressor_llm is None
    assert trial.query_expansion == "none"
    assert trial.expander_llm is None
    assert trial.generator_llm == "azure/gpt-4o-mini"


def test_translates_v03_hybrid_weight_zero_pins_alpha_to_bm25_only(tmp_path) -> None:
    """weight=0.0 → BM25-only → hybrid_alpha=0.0 with HYBRID_BM25_VECTOR index_type."""
    extracted = {
        "vectordb": [
            {
                "name": "embed_0",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
            },
        ],
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "modules": [{"module_type": "hybrid_cc", "weight": 0.0, "top_k": 5}],
                    },
                ]
            }
        ],
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.index_type == IndexType.HYBRID_BM25_VECTOR
    assert config.hybrid_alpha == 0.0


def test_translates_v03_reranker_with_explicit_model(tmp_path) -> None:
    extracted = {
        "vectordb": [
            {
                "name": "embed_1",
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
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
                "embedding_model": [{"type": "huggingface", "model_name": "BAAI/bge-m3"}],
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


def test_openai_with_azure_base_translates_to_azure_litellm_id(tmp_path) -> None:
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "openai",
                                "model": "gpt-4o-mini",
                                "api_base": "https://david-test.cognitiveservices.azure.com/openai/v1",
                            }
                        ],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.generator_llm == "azure/gpt-4o-mini"


def test_openai_without_azure_base_remains_openai(tmp_path) -> None:
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "openai",
                                "model": "gpt-4o-mini",
                                # No api_base → not Azure
                            }
                        ],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    # The curated space's llm_models only contains azure/gpt-4o-mini, so
    # openai/gpt-4o-mini falls back to the search-space default.
    config = translate_extracted_to_trial_config(path, _curated_space())
    assert config.generator_llm == "azure/gpt-4o-mini"  # search-space default


def test_openailike_legacy_translates_to_azure(tmp_path) -> None:
    """Older v0.3 configs may still use ``llm: openailike`` — accept for back-compat."""
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
    assert config.generator_llm == "azure/gpt-4o-mini"


def _bedrock_space() -> SearchSpace:
    space = _curated_space()
    space.llm_models = StageLLMs.uniform(["bedrock/us.meta.llama3-1-8b-instruct-v1:0", "azure/gpt-4o-mini"])
    return space


def test_bedrock_converse_provider_reverse_maps_to_bedrock_litellm_id(tmp_path) -> None:
    """``bedrock_converse`` is the modern provider native_config emits for
    bedrock/* entries (the deprecated ``bedrock`` provider rejects 2024+ model
    IDs). The translator must reverse it back to the litellm ``bedrock/<m>``
    form so re-scoring round-trips."""
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "bedrock_converse",
                                "model": "us.meta.llama3-1-8b-instruct-v1:0",
                                "region_name": "us-east-1",
                            }
                        ],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _bedrock_space())
    assert config.generator_llm == "bedrock/us.meta.llama3-1-8b-instruct-v1:0"


def test_legacy_bedrock_provider_still_reverse_maps(tmp_path) -> None:
    """Old extracted_sample.yaml files (predating the bedrock_converse switch)
    still carry ``llm: bedrock``. Keep accepting that for back-compat so we
    can re-score historical artifacts."""
    extracted = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "node_type": "generator",
                        "modules": [
                            {
                                "module_type": "llama_index_llm",
                                "llm": "bedrock",
                                "model": "us.meta.llama3-1-8b-instruct-v1:0",
                            }
                        ],
                    }
                ]
            }
        ]
    }
    path = _write_extracted_yaml(tmp_path, extracted)
    config = translate_extracted_to_trial_config(path, _bedrock_space())
    assert config.generator_llm == "bedrock/us.meta.llama3-1-8b-instruct-v1:0"


def test_normalize_module_type_handles_autorag_camelcase() -> None:
    """AutoRAG's extract_best_config emits CamelCase module class names; the
    translator must recognise them as the snake_case forms used in the input
    YAML and in the translator's match arms."""
    from agentic_autorag_bench.methods.autorag.translator import _normalize_module_type

    pairs = {
        "HybridCC": "hybrid_cc",
        "HybridRRF": "hybrid_rrf",
        "PassReranker": "pass_reranker",
        "PassQueryExpansion": "pass_query_expansion",
        "MultiQueryExpansion": "multi_query_expansion",
        "QueryDecompose": "query_decompose",
        "TreeSummarize": "tree_summarize",
        "Refine": "refine",
        "LongContextReorder": "long_context_reorder",
        "Fstring": "fstring",
        "VectorDB": "vectordb",
        "BM25": "bm25",
        "PassCompressor": "pass_compressor",
        "HyDE": "hyde",
        "LlamaIndexLLM": "llama_index_llm",
    }
    for camel, snake in pairs.items():
        assert _normalize_module_type(camel) == snake, f"{camel} -> {snake}"
    # snake_case inputs pass through unchanged.
    for snake in pairs.values():
        assert _normalize_module_type(snake) == snake
