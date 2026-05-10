"""Export a corpus directory to AutoRAG's ``corpus.parquet`` schema.

AutoRAG expects three columns: ``doc_id``, ``contents``, ``metadata``. We
treat one file in the corpus directory as one AutoRAG document. The
``metadata`` column is a dict; ``last_modified_datetime`` is required by
AutoRAG's chunker.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def export_corpus_to_parquet(corpus_dir: Path, out_path: Path) -> int:
    """Write every ``*.md`` / ``*.txt`` file under ``corpus_dir`` to ``out_path``.

    Returns the number of documents exported. Files are sorted by name so the
    parquet ordering is deterministic across runs (AutoRAG hashes ``doc_id``;
    a stable sort means the same run produces the same fingerprint).
    """
    rows: list[dict] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        rows.append(
            {
                "doc_id": path.stem,
                "contents": path.read_text(encoding="utf-8"),
                "metadata": {"last_modified_datetime": datetime.now().isoformat()},
            }
        )
    if not rows:
        raise RuntimeError(f"No documents found under {corpus_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)
