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

from agentic_autorag_bench.methods.autorag.native_config import (
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
    def test_mcq_variant_uses_mcq_prompt_and_rouge_metric(self) -> None:
        """The mcq variant uses the MCQ prompt template + rouge as the internal AutoRAG metric.

        Custom mcq_accuracy was retired because it required runtime-patching of
        AutoRAG's frozen metric registry. Rouge (token overlap with the gold
        answer text) is a reasonable monotonic proxy for substring match on
        short MCQ-style answers, and we re-score winners through our framework
        evaluator anyway.
        """
        config, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        assert notes["qa_variant"] == "mcq"
        prompt_node = _find_node(config, "prompt_maker")
        assert prompt_node["modules"][0]["prompt"][0] == MCQ_PROMPT_TEMPLATE
        gen_node = _find_node(config, "generator")
        assert gen_node["strategy"]["metrics"] == ["rouge"]

    def test_ragas_variant_uses_free_form_and_rouge_plus_bleu(self) -> None:
        config, _ = generate_autorag_config(_curated_space(), qa_variant="ragas")
        prompt_node = _find_node(config, "prompt_maker")
        assert prompt_node["modules"][0]["prompt"][0] == FREE_FORM_PROMPT_TEMPLATE
        gen_metrics = _find_node(config, "generator")["strategy"]["metrics"]
        assert set(gen_metrics) == {"rouge", "bleu"}

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

    def test_huggingface_embedding_models_use_list_of_dict_form(self) -> None:
        """AutoRAG's vectordb expects embedding_model as a list-of-one-dict for HF models."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        for entry in config["vectordb"]:
            em = entry["embedding_model"]
            assert isinstance(em, list) and len(em) == 1
            spec = em[0]
            assert spec["type"] == "huggingface"
            assert spec["model_name"] in {
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-m3",
            }

    def test_hybrid_weight_range_passes_alpha_through_as_semantic_weight(self) -> None:
        """AutoRAG hybrid_cc.weight is the SEMANTIC weight (weight=1.0 → semantic-only),
        matching our hybrid_alpha convention 1:1 — passed through with no inversion.

        AutoRAG's YAML loader interprets the literal string ``"(a, b)"`` as a
        2-tuple (utils.util.convert_string_to_tuple_in_dict). PyYAML can't
        dump tuples, so we emit the string form directly.
        """
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        hybrid_node = _find_node(config, "hybrid_retrieval")
        hybrid_mod = next(m for m in hybrid_node["modules"] if m["module_type"] == "hybrid_cc")
        assert hybrid_mod["weight_range"] == "(0.0, 1.0)"

    def test_hybrid_weight_pinned_to_one_when_only_vector_only_in_search_space(self) -> None:
        """vector_only-only space → hybrid_cc weight pinned at 1.0 (fully semantic, BM25
        contribution zeroed) so hybrid_retrieval is effectively a pass-through of the
        semantic retriever."""
        space = _curated_space()
        space.index_types = [IndexType.VECTOR_ONLY]
        config, _ = generate_autorag_config(space, qa_variant="mcq")
        hybrid_mod = next(
            m
            for n in _all_nodes(config)
            if n["node_type"] == "hybrid_retrieval"
            for m in n["modules"]
            if m["module_type"] == "hybrid_cc"
        )
        assert hybrid_mod["weight_range"] == "(1.0, 1.0)"

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

    def test_azure_llm_translates_to_openai_with_v1_base(self) -> None:
        """Azure routes through AutoRAG's ``openai`` provider (not ``openailike``).

        Reason: AutoRAG's ``pop_params`` filters kwargs against
        ``OpenAILike.__init__``'s declared parameter names. ``is_chat_model`` is a
        Pydantic class attribute (not a declared init param), so it gets dropped
        and the LLM defaults to ``is_chat_model=False`` — which routes chat
        models like gpt-4o-mini to ``/completions`` (400 Bad Request from Azure).
        The plain ``openai`` provider uses model-name-based chat detection that
        recognises gpt-4o-mini correctly without needing the flag.
        """
        config, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        gen = _find_node(config, "generator")
        assert len(gen["modules"]) == 1
        mod = gen["modules"][0]
        assert mod["llm"] == "openai"
        assert mod["model"] == ["gpt-4o-mini"]
        # Azure cognitive-services hosts respond to standard OpenAI chat
        # completions under ``/openai/v1``.
        assert mod["api_base"] == "${AZURE_API_BASE}/openai/v1"
        assert mod["api_key"] == "${AZURE_API_KEY}"
        assert "azure/<m> → openai" in notes["llm_provider_translation"]

    def test_bedrock_llm_translates_to_bedrock_converse_with_region(self) -> None:
        """Bedrock models route through ``bedrock_converse`` (a patched-in provider).

        Reason: AutoRAG 0.3 bundles ``llama-index-llms-bedrock`` (deprecated),
        whose ``Bedrock`` class hard-restricts ``model`` to a pre-2024 registry
        (no Llama 3.1, Nova 2, Claude Haiku 4.5). ``scripts/autorag_patches.py``
        registers ``BedrockConverse`` under the new provider name. The region
        must be passed explicitly because boto3 doesn't read ``AWS_REGION_NAME``
        (it reads AWS_REGION / AWS_DEFAULT_REGION).
        """
        space = _curated_space()
        space.llm_models = ["bedrock/us.meta.llama3-1-8b-instruct-v1:0"]
        config, notes = generate_autorag_config(space, qa_variant="mcq")
        mod = _find_node(config, "generator")["modules"][0]
        assert mod["llm"] == "bedrock_converse"
        assert mod["model"] == ["us.meta.llama3-1-8b-instruct-v1:0"]
        assert mod["region_name"] == "${AWS_REGION_NAME}"
        # No Azure-style keys leak onto bedrock modules.
        assert "api_base" not in mod and "api_key" not in mod
        assert notes["bedrock_in_search_space"] is True
        assert "AWS_REGION_NAME" in notes["bedrock_env_vars_required"]
        assert "bedrock_converse" in notes["llm_provider_translation"]

    def test_bedrock_query_expansion_threads_region_when_first_llm_is_bedrock(self) -> None:
        """HyDE / multi_query borrow auth from generator_modules[0]. If that's
        bedrock_converse, the QE block must carry ``region_name`` so the
        expansion LLM has a working endpoint."""
        space = _curated_space()
        space.llm_models = ["bedrock/us.meta.llama3-1-8b-instruct-v1:0", "azure/gpt-4o-mini"]
        config, _ = generate_autorag_config(space, qa_variant="mcq")
        qe_node = _find_node(config, "query_expansion")
        hyde = next(m for m in qe_node["modules"] if m["module_type"] == "hyde")
        assert hyde["llm"] == "bedrock_converse"
        assert hyde["region_name"] == "${AWS_REGION_NAME}"
        assert "api_base" not in hyde and "api_key" not in hyde

    def test_bedrock_not_in_search_space_omits_bedrock_notes(self) -> None:
        """When no bedrock entries exist, the notes shouldn't claim AWS env vars are required."""
        _, notes = generate_autorag_config(_curated_space(), qa_variant="mcq")
        assert notes["bedrock_in_search_space"] is False
        assert notes["bedrock_env_vars_required"] == []

    def test_embedding_batch_tuned_for_gpu(self) -> None:
        """Embedding batch should be GPU-tuned (>= 128 per HF perf docs). The
        previous value (64) underutilised the 4080; bumped to 256 once the
        free-after-ingest patch keeps only one embedder resident at a time."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        for entry in config["vectordb"]:
            assert entry["embedding_batch"] >= 128, (
                f"vectordb entry {entry['name']} has embedding_batch="
                f"{entry['embedding_batch']}; raise to ≥128 to use GPU bandwidth"
            )

    def test_huggingface_embeddings_use_fp16(self) -> None:
        """HF embedders should load in fp16. fp32→fp16 cosine drift is ~1e-6
        per a smoke check (well below the 0.999 threshold in
        test_huggingface_embedding_equivalence_minilm), and we halve VRAM."""
        config, _ = generate_autorag_config(_curated_space(), qa_variant="mcq")
        for entry in config["vectordb"]:
            spec = entry["embedding_model"][0]
            assert spec["model_kwargs"]["torch_dtype"] == "float16", (
                f"{entry['name']} missing fp16 model_kwargs"
            )

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
