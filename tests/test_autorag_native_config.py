"""Tests for the AutoRAG v0.3.x search-space mirror."""

from __future__ import annotations

import pytest

from agentic_autorag.config.models import (
    ChunkingSearchSpace,
    IndexType,
    NumericRange,
    RerankerSearchSpace,
    SearchSpace,
)

from autorag_bench.methods.autorag.native_config import (
    FREE_FORM_PROMPT_TEMPLATE,
    MCQ_PROMPT_TEMPLATE,
    RERANKER_MODULE_MAP,
    generate_autorag_config,
)


def _curated_space() -> SearchSpace:
    return SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive", "fixed"],
            chunk_token_size=NumericRange(min=256, max=512),
            chunk_token_overlap=NumericRange(min=0, max=64),
        ),
        embedding_models=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-m3",
        ],
        index_types=[IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
        top_k=NumericRange(min=3, max=20),
        hybrid_alpha=NumericRange(min=0.0, max=1.0),
        reranker=RerankerSearchSpace(
            models=["none", "BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
            top_n=NumericRange(min=3, max=10),
        ),
        query_expansion=["none", "hyde", "multi_query"],
        llm_models=["azure/gpt-4o-mini"],
        temperature=NumericRange(min=1.0, max=1.0),
    )


def _all_nodes(config: dict) -> list[dict]:
    return [n for line in config["node_lines"] for n in line["nodes"]]


def _find_node(config: dict, node_type: str) -> dict:
    for n in _all_nodes(config):
        if n["node_type"] == node_type:
            return n
    raise AssertionError(f"node_type {node_type!r} not in config")


class TestGenerateAutoragConfig:
    def test_mcq_variant_uses_mcq_prompt_and_metric(self) -> None:
        config, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        assert notes["qa_variant"] == "mcq"
        prompt_node = _find_node(config, "prompt_maker")
        assert prompt_node["modules"][0]["prompt"][0] == MCQ_PROMPT_TEMPLATE
        gen_node = _find_node(config, "generator")
        assert gen_node["strategy"]["metrics"] == ["mcq_accuracy"]

    def test_ragas_variant_uses_free_form_and_g_eval(self) -> None:
        config, _ = generate_autorag_config(_curated_space(), qa_variant="ragas")
        prompt_node = _find_node(config, "prompt_maker")
        assert prompt_node["modules"][0]["prompt"][0] == FREE_FORM_PROMPT_TEMPLATE
        gen_metrics = _find_node(config, "generator")["strategy"]["metrics"]
        # Free-form metrics: bleu / rouge / g_eval as dicts (v0.3 metric form).
        names = {m["metric_name"] if isinstance(m, dict) else m for m in gen_metrics}
        assert "g_eval" in names

    def test_rejects_unknown_qa_variant(self) -> None:
        with pytest.raises(ValueError, match="qa_variant"):
            generate_autorag_config(_curated_space(), qa_variant="bogus")

    def test_v03_node_types_present(self) -> None:
        """v0.3 split retrieval into three node_types — verify none use the v0.2 'retrieval' name."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        ntypes = {n["node_type"] for n in _all_nodes(config)}
        # v0.3 uses these node names
        assert "lexical_retrieval" in ntypes
        assert "semantic_retrieval" in ntypes
        assert "hybrid_retrieval" in ntypes
        assert "passage_reranker" in ntypes
        assert "query_expansion" in ntypes
        assert "prompt_maker" in ntypes
        assert "generator" in ntypes
        # v0.2 'retrieval' should be absent
        assert "retrieval" not in ntypes

    def test_top_k_is_at_node_level_not_in_strategy(self) -> None:
        """v0.3 moved top_k from strategy → node level for retrieval/reranker."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        for node_type in {"lexical_retrieval", "semantic_retrieval", "hybrid_retrieval", "passage_reranker"}:
            node = _find_node(config, node_type)
            assert "top_k" in node, f"{node_type} missing top_k at node level"
            assert "top_k" not in node.get("strategy", {}), f"{node_type} should not have top_k under strategy"

    def test_semantic_retrieval_references_vectordb_by_name(self) -> None:
        """v0.3: vectordb is declared top-level, referenced by name from semantic_retrieval."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        vectordb_names = {entry["name"] for entry in config["vectordb"]}
        assert vectordb_names  # at least one vectordb entry
        sem = _find_node(config, "semantic_retrieval")
        for m in sem["modules"]:
            if m["module_type"] == "vectordb":
                assert m["vectordb"] in vectordb_names

    def test_one_vectordb_entry_per_embedding_model(self) -> None:
        config, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        # 2 embedding models in the curated space → 2 named vectordb entries.
        assert len(config["vectordb"]) == 2
        assert notes["embedding_model_to_vectordb_name"] == {
            "sentence-transformers/all-MiniLM-L6-v2": "embed_0",
            "BAAI/bge-m3": "embed_1",
        }

    def test_huggingface_embedding_models_use_explicit_huggingface_block(self) -> None:
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        for entry in config["vectordb"]:
            assert entry["embedding_model"] == "huggingface"
            assert "embedding_model_kwargs" in entry
            assert entry["embedding_model_kwargs"]["model_name"] in {
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-m3",
            }

    def test_hybrid_alpha_inverts_to_bm25_weight(self) -> None:
        """AutoRAG's hybrid_cc.weight_range is BM25's; ours is the vector's."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        hybrid_node = _find_node(config, "hybrid_retrieval")
        hybrid_mod = next(m for m in hybrid_node["modules"] if m["module_type"] == "hybrid_cc")
        bm25_lo, bm25_hi = hybrid_mod["weight_range"]
        # Our hybrid_alpha range is [0.0, 1.0] → BM25 weight range is [0.0, 1.0]
        assert bm25_lo == 0.0
        assert bm25_hi == 1.0

    def test_pass_through_reranker_uses_pass_reranker_v03_name(self) -> None:
        """v0.3 renamed pass_passage_reranker → pass_reranker."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        reranker_node = _find_node(config, "passage_reranker")
        modules = {m.get("model_name", "<pass>"): m["module_type"] for m in reranker_node["modules"]}
        assert modules["<pass>"] == "pass_reranker"

    def test_known_rerankers_use_explicit_module_mapping(self) -> None:
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        reranker_node = _find_node(config, "passage_reranker")
        modules = {m.get("model_name", "<pass>"): m["module_type"] for m in reranker_node["modules"]}
        assert modules["BAAI/bge-reranker-v2-m3"] == "flag_embedding_reranker"
        assert modules["cross-encoder/ms-marco-MiniLM-L-6-v2"] == "sentence_transformer_reranker"

    def test_unknown_reranker_raises_explicit_error(self) -> None:
        space = _curated_space()
        space.reranker = RerankerSearchSpace(
            models=["totally-fake-reranker/v9000"],
            top_n=NumericRange(min=3, max=10),
        )
        with pytest.raises(KeyError, match="No AutoRAG reranker module mapping"):
            generate_autorag_config(space, qa_variant="mcq")

    def test_azure_llm_translates_to_openailike(self) -> None:
        config, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        gen = _find_node(config, "generator")
        assert len(gen["modules"]) == 1
        mod = gen["modules"][0]
        assert mod["llm"] == "openailike"
        assert mod["model"] == ["gpt-4o-mini"]
        assert mod["api_base"] == "${AZURE_API_BASE}"
        assert mod["api_key"] == "${AZURE_API_KEY}"
        assert "azure/<m> → openailike" in notes["llm_provider_translation"]

    def test_translation_notes_record_excluded_dimensions(self) -> None:
        _, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        excluded = " ".join(notes["excluded_dimensions"])
        assert "chunking" in excluded  # explicitly noted as a v0.3 exclusion
        assert "passage_compressor" in excluded
        assert "passage_filter" in excluded
        assert "prompt_maker template tuning" in excluded
        assert "hybrid_rrf" in excluded

    def test_discretization_grid_recorded(self) -> None:
        _, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        grid = notes["discretization"]
        assert grid["top_k"][0] == 3 and grid["top_k"][-1] == 20
        assert len(grid["top_k"]) == 5
        assert grid["reranker_top_k"][0] == 3 and grid["reranker_top_k"][-1] == 10

    def test_query_expansion_modules_include_pass_hyde_and_multi_query(self) -> None:
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        qe_node = _find_node(config, "query_expansion")
        mtypes = [m["module_type"] for m in qe_node["modules"]]
        assert "pass_query_expansion" in mtypes
        assert "hyde" in mtypes
        assert "multi_query_expansion" in mtypes

    def test_query_expansion_strategy_carries_retrieval_modules(self) -> None:
        """v0.3 query_expansion node embeds retrieval_modules in strategy so it can score query rewrites."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        qe_node = _find_node(config, "query_expansion")
        assert "retrieval_modules" in qe_node["strategy"]
        assert qe_node["strategy"]["retrieval_modules"]


class TestRerankerModuleMap:
    def test_curated_three_rerankers_are_all_mapped(self) -> None:
        for model in [
            "BAAI/bge-reranker-v2-m3",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ]:
            assert model in RERANKER_MODULE_MAP, f"{model} should be in RERANKER_MODULE_MAP"
