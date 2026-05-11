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
"""

from __future__ import annotations


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


_patch_chroma_add_embedding()
