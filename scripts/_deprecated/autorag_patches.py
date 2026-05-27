"""Runtime patches applied inside ``.autorag-venv`` at interpreter startup.

The setup script (``scripts/setup_autorag_venv.sh``) drops this file as
``bench_autorag_patches.py`` in the autorag venv's site-packages, paired with a
``bench_autorag_patches.pth`` line. Python loads ``.pth`` files at startup and
runs lines that begin with ``import``, so the patch fires before AutoRAG's
``evaluate`` (or any other) entry point ever touches Chroma.

Patches applied:

- ``autorag.vectordb.chroma.Chroma.add_embedding`` — chunk inserts so they fit
  under Chroma's SQLite-backed 5461-row batch cap. Without this, ingest fails
  on any corpus >~5k documents with ``Batch size of N is greater than max
  batch size of 5461``. The bench's hotpot corpora are ~19k docs.

- ``autorag.generator_models["bedrock_converse"]`` — register the modern
  ``BedrockConverse`` LLM as a new provider. AutoRAG 0.3 ships
  ``llama-index-llms-bedrock`` (deprecated), whose ``Bedrock`` class
  hard-restricts ``model`` to a fixed pre-2024 registry. The paper's
  search-space bedrock entries (us.meta.llama3-1-8b, global.amazon.nova-2-lite,
  global.anthropic.claude-haiku-4-5) are all 2024+ and aren't in that registry,
  so they fail at construction with a ``context_size`` error. The Converse API
  has no such restriction. We surface it as a separate provider name
  (``bedrock_converse``) rather than replacing ``bedrock`` so the deprecated
  path keeps working for anyone with a pinned older config.

- ``autorag.generator_models["vertex"]`` — register Google Vertex AI's
  llama_index integration so generator entries prefixed ``vertex_ai/`` in the
  bench config can be exercised by the AutoRAG baseline. AutoRAG 0.3 has no
  built-in Vertex provider; native_config.py translates ``vertex_ai/<model>``
  to ``(vertex, <model>)`` and passes ``project`` / ``location`` via env-var
  substitution.

- ``autorag.nodes.semanticretrieval.vectordb.vectordb_ingest_huggingface`` —
  release each SentenceTransformer's VRAM after its corpus ingest. Upstream
  AutoRAG loops through every vectordb in ``Evaluator.start_trial`` and
  ingests each in turn, but the embedder instance stays referenced inside
  the ``BaseVectorStore`` for the rest of the evaluator's lifetime. With six
  HuggingFace embedders (MiniLM + 5×1024-dim models ≈ 7.5 GB FP32) this
  hard-pins ~7.5 GB of VRAM during ingest before retrieval even starts —
  catastrophic when the 4080 (16 GB) shares with another workload. AutoRAG's
  retrieval node *already* re-loads the model per module evaluation
  (``Initialize retrieval node`` / ``Deleting retrieval node`` log lines in
  base.py), so dropping the ingest-time reference is safe.

- ``TreeSummarize._pure`` / ``Refine._pure`` / ``LlamaIndexLLM.__pure_generate``
  / ``LlamaIndexLLM.__pure_chat`` — per-row content-filter tolerance.
  Upstream AutoRAG hands every per-row LLM task to ``process_batch`` which
  uses ``asyncio.gather`` without ``return_exceptions``, so a single
  Azure ``ResponsibleAIPolicyViolation`` aborts the entire enumeration.
  Patched: wrap each task so content-filter rejections substitute an
  empty result, which scores 0 in retrieval_token_f1 / rouge — dragging
  the failing module's average down so AutoRAG's strategy correctly
  de-prefers filter-prone modules (typically pass_compressor /
  pass_query_expansion win on filter-prone rows). Other exceptions still
  propagate.
"""

from __future__ import annotations

from typing import Any


def _patch_chroma_add_embedding() -> None:
    # Below Chroma's documented 5461 hard limit, leaves headroom for the few
    # extra columns ``add()`` writes alongside (ids + embeddings + metadata).
    BATCH = 5000
    try:
        from autorag.vectordb.chroma import Chroma
    except ImportError:
        return  # AutoRAG not installed in this interpreter; nothing to patch.

    def add_embedding(self, ids, embeddings):
        for start in range(0, len(ids), BATCH):
            end = start + BATCH
            self.collection.add(ids=ids[start:end], embeddings=embeddings[start:end])

    Chroma.add_embedding = add_embedding


def _patch_register_bedrock_converse() -> None:
    try:
        import autorag
        from llama_index.llms.bedrock_converse import BedrockConverse
    except ImportError:
        return  # AutoRAG or bedrock-converse not installed in this interpreter.

    # AutoRAG's generator interface calls ``acomplete`` on the llm instance,
    # but the upstream ``BedrockConverse.acomplete`` is missing on some
    # llama-index releases. Mirror AutoRAG's own ``AutoRAGBedrock`` shim so
    # the sync ``complete`` is re-exposed as async.
    class AutoRAGBedrockConverse(BedrockConverse):
        async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            return self.complete(prompt, formatted=formatted, **kwargs)

    autorag.generator_models["bedrock_converse"] = AutoRAGBedrockConverse


def _patch_register_vertex() -> None:
    try:
        import autorag
        from llama_index.llms.vertex import Vertex
    except ImportError:
        return  # AutoRAG or llama-index-llms-vertex not installed.

    autorag.generator_models["vertex"] = Vertex


def _patch_free_embedder_after_ingest() -> None:
    try:
        from autorag.nodes.semanticretrieval import vectordb as _vdb_mod
        from autorag import evaluator as _eval_mod
    except ImportError:
        return  # AutoRAG not installed in this interpreter; nothing to patch.

    original = _vdb_mod.vectordb_ingest_huggingface

    def _wrapped(vectordb, corpus_data):
        original(vectordb, corpus_data)
        # The SentenceTransformer (and its tokenizer) live under
        # ``vectordb.embedding._model`` / ``._tokenizer`` (HuggingFaceEmbedding
        # internals). Drop the references so Python GC + torch's CUDA caching
        # allocator can release the VRAM before the next vectordb ingests.
        # Retrieval re-loads each model lazily per module, so this doesn't
        # break query-time embedding (verified against base.py:18/28 logs:
        # ``Initialize retrieval node - VectorDB`` / ``Loading
        # SentenceTransformer model``).
        emb = getattr(vectordb, "embedding", None)
        if emb is not None:
            for attr in ("_model", "_tokenizer"):
                try:
                    delattr(emb, attr)
                except AttributeError:
                    pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    _vdb_mod.vectordb_ingest_huggingface = _wrapped
    # ``evaluator.py`` imports ``vectordb_ingest_huggingface`` by name into
    # its own module namespace (``from autorag.nodes.semanticretrieval.vectordb
    # import vectordb_ingest_huggingface``), so patching only the source
    # module wouldn't rebind that local reference. Patch both.
    _eval_mod.vectordb_ingest_huggingface = _wrapped


_CONTENT_FILTER_MARKERS: tuple[str, ...] = (
    "content_filter",
    "responsibleaipolicyviolation",
    "content management policy",
)


def _is_content_filter_exc(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CONTENT_FILTER_MARKERS)


def _patch_content_filter_row_tolerance() -> None:
    import asyncio

    try:
        from autorag.nodes.passagecompressor.tree_summarize import TreeSummarize
        from autorag.nodes.passagecompressor.refine import Refine
        from autorag.nodes.generator.llama_index_llm import LlamaIndexLLM
        from autorag.utils.util import get_event_loop, process_batch
        from llama_index.core import PromptTemplate
        from llama_index.core.base.llms.types import CompletionResponse, ChatResponse, ChatMessage
        from llama_index.core.prompts import PromptType
        from llama_index.core.prompts.utils import is_chat_model
        from llama_index.core.response_synthesizers import (
            TreeSummarize as _LITreeSummarize,
            Refine as _LIRefine,
        )
    except ImportError:
        return

    async def _safe(coro, fallback):
        try:
            return await coro
        except BaseException as exc:
            if _is_content_filter_exc(exc):
                return fallback
            raise

    def _tree_summarize_pure(self, queries, contents, prompt=None, chat_prompt=None, batch=16):
        if prompt is not None and not is_chat_model(self.llm):
            summary_template = PromptTemplate(prompt, prompt_type=PromptType.SUMMARY)
        elif chat_prompt is not None and is_chat_model(self.llm):
            summary_template = PromptTemplate(chat_prompt, prompt_type=PromptType.SUMMARY)
        else:
            summary_template = None
        summarizer = _LITreeSummarize(llm=self.llm, summary_template=summary_template, use_async=True)
        tasks = [
            _safe(summarizer.aget_response(query, content), fallback="")
            for query, content in zip(queries, contents)
        ]
        loop = get_event_loop()
        return loop.run_until_complete(process_batch(tasks, batch_size=batch))

    def _refine_pure(self, queries, contents, prompt=None, chat_prompt=None, batch=16):
        if prompt is not None and not is_chat_model(self.llm):
            refine_template = PromptTemplate(prompt, prompt_type=PromptType.REFINE)
        elif chat_prompt is not None and is_chat_model(self.llm):
            refine_template = PromptTemplate(chat_prompt, prompt_type=PromptType.REFINE)
        else:
            refine_template = None
        summarizer = _LIRefine(llm=self.llm, refine_template=refine_template, verbose=True)
        tasks = [
            _safe(summarizer.aget_response(query, content), fallback="")
            for query, content in zip(queries, contents)
        ]
        loop = get_event_loop()
        return loop.run_until_complete(process_batch(tasks, batch_size=batch))

    TreeSummarize._pure = _tree_summarize_pure
    Refine._pure = _refine_pure

    # LlamaIndexLLM has name-mangled ``__pure_generate`` / ``__pure_chat``.
    # Build CompletionResponse / ChatResponse fallbacks with empty content
    # so downstream metric calculation gives this row 0 for rouge / bleu /
    # retrieval_token_f1 — dragging the failing LLM's average down so
    # AutoRAG's strategy de-prefers filter-prone LLMs at the generator and
    # query_expansion (LlamaIndexLLM-backed) nodes.
    _empty_completion = CompletionResponse(text="")
    _empty_chat = ChatResponse(message=ChatMessage(role="assistant", content=""))

    _orig_pure_generate = LlamaIndexLLM._LlamaIndexLLM__pure_generate
    _orig_pure_chat = LlamaIndexLLM._LlamaIndexLLM__pure_chat

    def _safe_pure_generate(self, prompts, **kwargs):
        tasks = [
            _safe(self.llm_instance.acomplete(prompt), fallback=_empty_completion)
            for prompt in prompts
        ]
        loop = get_event_loop()
        results = loop.run_until_complete(process_batch(tasks, batch_size=self.batch))
        generated_texts = [r.text for r in results]
        tokenized_ids = self.get_default_tokenized_ids(generated_texts)
        pseudo_log_probs = self.get_default_log_probs(tokenized_ids)
        return generated_texts, tokenized_ids, pseudo_log_probs

    def _safe_pure_chat(self, prompts, **kwargs):
        llama_index_messages = [
            [ChatMessage(role=msg["role"], content=msg["content"]) for msg in message]
            for message in prompts
        ]
        tasks = [
            _safe(self.llm_instance.achat(msg), fallback=_empty_chat)
            for msg in llama_index_messages
        ]
        loop = get_event_loop()
        results = loop.run_until_complete(process_batch(tasks, batch_size=self.batch))
        generated_texts = [r.message.content for r in results]
        if all(r.logprobs is not None for r in results if r is not _empty_chat):
            # Logprobs absent on the filtered rows; force the no-logprob branch
            # so downstream uses pseudo logprobs uniformly.
            tokenized_ids = self.get_default_tokenized_ids(generated_texts)
            pseudo_log_probs = self.get_default_log_probs(tokenized_ids)
        else:
            tokenized_ids = self.get_default_tokenized_ids(generated_texts)
            pseudo_log_probs = self.get_default_log_probs(tokenized_ids)
        return generated_texts, tokenized_ids, pseudo_log_probs

    LlamaIndexLLM._LlamaIndexLLM__pure_generate = _safe_pure_generate
    LlamaIndexLLM._LlamaIndexLLM__pure_chat = _safe_pure_chat


_patch_chroma_add_embedding()
_patch_register_bedrock_converse()
_patch_register_vertex()
_patch_free_embedder_after_ingest()
_patch_content_filter_row_tolerance()
