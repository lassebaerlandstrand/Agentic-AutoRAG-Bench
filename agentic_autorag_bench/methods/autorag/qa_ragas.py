"""Bootstrap RAGAS-style QA inside the AutoRAG venv.

AutoRAG's QA generator imports llama-index, openai, and various other libraries
that conflict with our base deps. Instead of importing it here we shell out to
a tiny Python program in the AutoRAG venv. The program is held in this file as
a string so the bench repo carries no AutoRAG-specific imports at parse time.

API shape is AutoRAG v0.3.x (verified against 0.3.22): the QA chain lives at
``autorag.data.qa.*`` (it was ``autorag.data.beta.*`` in pre-0.3 builds), and
the data-creation step uses the documented ``Corpus(...).sample(...)``-style
chain — see https://marker-inc-korea.github.io/AutoRAG/data_creation/tutorial.html.
"""

from __future__ import annotations

import logging
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger("agentic_autorag_bench.run")

# The bootstrap runs inside ``.autorag-venv`` (numpy<2, AutoRAG-pinned deps),
# so we can't share it with our base venv — hence the subprocess + string-held
# program. ``AUTORAG_BOOTSTRAP_MODEL`` carries the bench's litellm-style id
# (``azure/<deployment>`` or ``openai/<model>``) so the script can do the same
# provider translation that ``native_config._translate_llm`` does for the
# evaluate step's generator LLM.
_BOOTSTRAP_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    from pathlib import Path

    import pandas as pd
    from autorag.data.qa.query.llama_gen_query import factoid_query_gen
    from autorag.data.qa.sample import random_single_hop
    from autorag.data.qa.schema import Corpus
    from autorag.data.qa.generation_gt.llama_index_gen_gt import (
        make_basic_gen_gt,
        make_concise_gen_gt,
    )
    from llama_index.llms.openai import OpenAI

    corpus_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    sample_n = int(sys.argv[3])
    litellm_model = os.environ.get("AUTORAG_BOOTSTRAP_MODEL", "")
    if not litellm_model or "/" not in litellm_model:
        raise SystemExit(
            "AUTORAG_BOOTSTRAP_MODEL must be set to '<provider>/<model>' "
            "(e.g. 'azure/gpt-4o-mini'); got " + repr(litellm_model)
        )

    # Translate the bench's litellm id to a llama-index OpenAI instance. This
    # mirrors native_config._translate_llm: ``azure/<deployment>`` routes
    # through Azure's OpenAI-compat v1 shim at ``<AZURE_API_BASE>/openai/v1``.
    provider, _, model_name = litellm_model.partition("/")

    # Azure/OpenAI reasoning models (o1/o3/o4-series) reject any temperature
    # override — the API only accepts the model's default (1.0) and returns
    # ``BadRequestError 400 unsupported_value`` for anything else. llama_index's
    # OpenAI wrapper defaults to temperature=0.1 unconditionally, so without
    # this branch the RAGAS bootstrap dies on its first ``achat`` (typically
    # inside ``make_basic_gen_gt``). The main bench path doesn't hit this
    # because ``configure_litellm_runtime`` sets ``litellm.drop_params=True``,
    # but the bootstrap runs in the AutoRAG subprocess and uses llama_index
    # directly, so we have to handle it here.
    is_reasoning = any(model_name.startswith(p) for p in ("o1", "o3", "o4", "o5"))
    extra_kwargs = {"temperature": 1.0} if is_reasoning else {}

    if provider == "azure":
        base = os.environ.get("AZURE_API_BASE")
        key = os.environ.get("AZURE_API_KEY")
        if not (base and key):
            raise SystemExit(
                "AZURE_API_BASE and AZURE_API_KEY must be set for the QA bootstrap "
                "when AUTORAG_BOOTSTRAP_MODEL uses the azure/ prefix."
            )
        llm = OpenAI(
            model=model_name,
            api_base=base.rstrip("/") + "/openai/v1",
            api_key=key,
            **extra_kwargs,
        )
    elif provider == "openai":
        llm = OpenAI(model=model_name, **extra_kwargs)
    else:
        raise SystemExit(
            "Unsupported provider for the RAGAS bootstrap: " + repr(provider)
            + ". Add a branch in qa_ragas.py if you add another."
        )

    corpus_df = pd.read_parquet(corpus_path)
    corpus = Corpus(corpus_df=corpus_df)
    qa = (
        corpus.sample(random_single_hop, n=sample_n, random_state=42)
        .map(lambda df: df.reset_index(drop=True))
        .make_retrieval_gt_contents()
        .batch_apply(factoid_query_gen, llm=llm)
        .batch_apply(make_basic_gen_gt, llm=llm)
        .batch_apply(make_concise_gen_gt, llm=llm)
    )

    # ``qa.to_parquet`` would overwrite the corpus parquet; the docstring
    # documents ``qa.data.to_parquet(...)`` as the per-frame-only escape hatch.
    # AutoRAG's evaluate step reads exactly these four columns.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qa.data[["qid", "query", "retrieval_gt", "generation_gt"]].reset_index(drop=True).to_parquet(out_path)
    print("wrote " + str(len(qa.data)) + " rows to " + str(out_path))
    """
).strip()


def export_ragas_qa_via_subprocess(
    corpus_parquet: Path,
    out_path: Path,
    *,
    sample_n: int,
    llm_model: str,
    autorag_python: str,
) -> None:
    """Invoke AutoRAG's RAGAS bootstrap inside the AutoRAG venv.

    ``llm_model`` is the bench's litellm-style id (``azure/<deployment>`` or
    ``openai/<model>``); the embedded script handles the provider translation.
    """
    if not autorag_python:
        raise RuntimeError("autorag_python is required to run the RAGAS bootstrap")
    cmd = [autorag_python, "-c", _BOOTSTRAP_SCRIPT, str(corpus_parquet), str(out_path), str(sample_n)]
    import os as _os
    # ``_os.environ`` first, ``AUTORAG_BOOTSTRAP_MODEL`` last so the bench's
    # value wins even if the parent shell happens to export it.
    env = {**_os.environ, "AUTORAG_BOOTSTRAP_MODEL": llm_model}
    logger.info("Running AutoRAG RAGAS bootstrap (n=%d, model=%s)", sample_n, llm_model)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.stdout:
        logger.info(result.stdout.rstrip())
    if result.stderr:
        logger.warning(result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"AutoRAG RAGAS bootstrap exited with rc={result.returncode}")
