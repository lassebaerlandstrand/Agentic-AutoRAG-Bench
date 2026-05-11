"""Bootstrap RAGAS-style QA inside the AutoRAG venv.

AutoRAG's QA generator imports llama-index, openai, and various other libraries
that conflict with our base deps. Instead of importing it here we shell out to
a tiny Python program in the AutoRAG venv. The program is held in this file as
a string so the bench repo carries no AutoRAG-specific imports at parse time.
"""

from __future__ import annotations

import logging
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger("agentic_autorag_bench.run")

_BOOTSTRAP_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import pandas as pd
    from autorag.data.beta.query.llama_gen_query import factoid_query_gen
    from autorag.data.beta.sample import random_single_hop
    from autorag.data.beta.schema import Raw
    from autorag.data.beta.generation_gt.llama_index_gen_gt import (
        make_basic_gen_gt,
        make_concise_gen_gt,
    )
    from llama_index.llms.openai import OpenAI
    # The exact LLM is parameterised by the caller via env var.
    import os

    corpus_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    sample_n = int(sys.argv[3])
    model = os.environ.get("AUTORAG_BOOTSTRAP_MODEL", "gpt-4o-mini")

    corpus_df = pd.read_parquet(corpus_path)
    raw = Raw(corpus_df)

    sampled = random_single_hop(raw.data, n=sample_n, random_state=42)
    qa = sampled.copy()
    llm = OpenAI(model=model)
    qa = factoid_query_gen(qa, llm)
    qa = make_basic_gen_gt(qa, llm)
    qa = make_concise_gen_gt(qa, llm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qa.to_parquet(out_path, index=False)
    print(f"wrote {len(qa)} rows to {out_path}")
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
    """Invoke AutoRAG's RAGAS bootstrap inside the AutoRAG venv."""
    if not autorag_python:
        raise RuntimeError("autorag_python is required to run the RAGAS bootstrap")
    cmd = [autorag_python, "-c", _BOOTSTRAP_SCRIPT, str(corpus_parquet), str(out_path), str(sample_n)]
    env = {"AUTORAG_BOOTSTRAP_MODEL": llm_model}
    logger.info("Running AutoRAG RAGAS bootstrap (n=%d, model=%s)", sample_n, llm_model)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env={**env, **__import__("os").environ})
    if result.stdout:
        logger.info(result.stdout.rstrip())
    if result.stderr:
        logger.warning(result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"AutoRAG RAGAS bootstrap exited with rc={result.returncode}")
