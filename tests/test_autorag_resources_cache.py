"""Tests for the AutoRAG resources-cache symlink + key hash."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_autorag_bench.methods.autorag.driver import (
    _compute_resources_cache_key,
    _setup_resources_cache,
)


def _write_parquet_with_content(path: Path, payload: bytes) -> Path:
    """Helper: write a file whose bytes seed the cache hash. The cache key
    function hashes raw file bytes, so any deterministic content works."""
    path.write_bytes(payload)
    return path


def test_cache_key_changes_when_corpus_content_changes(tmp_path: Path) -> None:
    """Different corpus bytes → different cache key (so the cache invalidates
    on a fresh corpus). Otherwise we'd serve stale Chroma collections for a
    new corpus."""
    a = _write_parquet_with_content(tmp_path / "a.parquet", b"corpus-A")
    b = _write_parquet_with_content(tmp_path / "b.parquet", b"corpus-B")
    embedders = ["sentence-transformers/all-MiniLM-L6-v2"]
    key_a = _compute_resources_cache_key(a, embedders)
    key_b = _compute_resources_cache_key(b, embedders)
    assert key_a != key_b


def test_cache_key_changes_when_embedder_set_changes(tmp_path: Path) -> None:
    """Same corpus + different embedder list → different key. AutoRAG names
    Chroma collections ``embed_0`` … ``embed_N`` positionally, so reordering
    embedders shifts which collection holds which model's vectors. Reusing
    the cache across embedder-list changes would mismatch collections to
    models — incorrect retrieval, not just slower."""
    corpus = _write_parquet_with_content(tmp_path / "c.parquet", b"corpus")
    key_one = _compute_resources_cache_key(corpus, ["BAAI/bge-large-en-v1.5"])
    key_two = _compute_resources_cache_key(corpus, ["BAAI/bge-m3"])
    key_swap = _compute_resources_cache_key(corpus, ["A", "B"])
    key_swap_rev = _compute_resources_cache_key(corpus, ["B", "A"])
    assert key_one != key_two, "different single embedder must yield different key"
    assert key_swap != key_swap_rev, "embedder ordering matters (collection indices are positional)"


def test_cache_key_stable_for_same_inputs(tmp_path: Path) -> None:
    """Determinism — same corpus + same embedders → same key. Otherwise the
    cache would never hit across runs."""
    corpus = _write_parquet_with_content(tmp_path / "c.parquet", b"same")
    embedders = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
    k1 = _compute_resources_cache_key(corpus, embedders)
    k2 = _compute_resources_cache_key(corpus, embedders)
    assert k1 == k2


def test_setup_resources_cache_creates_symlink_to_cache_dir(tmp_path: Path) -> None:
    autorag_dir = tmp_path / "autorag_project"
    autorag_dir.mkdir()
    cache_root = tmp_path / ".shared_cache"
    cache_root.mkdir()
    cache_dir = _setup_resources_cache(autorag_dir, cache_root, "abc123")
    resources = autorag_dir / "resources"
    assert resources.is_symlink(), "resources/ must be a symlink so AutoRAG writes through to the cache"
    assert resources.resolve() == cache_dir.resolve()
    assert cache_dir.name == "autorag_resources_abc123"


def test_setup_resources_cache_replaces_stale_symlink_with_new_key(tmp_path: Path) -> None:
    """When the cache key changes (e.g. corpus or embedder set changes), the
    symlink must rebind to the new cache dir — otherwise we'd silently keep
    using the old cache after an invalidating change."""
    autorag_dir = tmp_path / "autorag_project"
    autorag_dir.mkdir()
    cache_root = tmp_path / ".shared_cache"
    cache_root.mkdir()
    first = _setup_resources_cache(autorag_dir, cache_root, "key-A")
    second = _setup_resources_cache(autorag_dir, cache_root, "key-B")
    assert first != second
    assert (autorag_dir / "resources").resolve() == second.resolve()


def test_setup_resources_cache_migrates_existing_dir_contents(tmp_path: Path) -> None:
    """If a previous run wrote ``resources/`` as a real directory (pre-caching),
    we must migrate its contents to the cache instead of throwing them away —
    otherwise the first cache-enabled run after an upgrade silently re-embeds."""
    autorag_dir = tmp_path / "autorag_project"
    autorag_dir.mkdir()
    cache_root = tmp_path / ".shared_cache"
    cache_root.mkdir()
    legacy_resources = autorag_dir / "resources"
    legacy_resources.mkdir()
    (legacy_resources / "bm25_porter_stemmer.pkl").write_bytes(b"old-bm25-pickle")
    (legacy_resources / "vectordb.yaml").write_text("legacy: true")

    cache_dir = _setup_resources_cache(autorag_dir, cache_root, "k")

    assert (autorag_dir / "resources").is_symlink()
    assert (cache_dir / "bm25_porter_stemmer.pkl").read_bytes() == b"old-bm25-pickle"
    assert (cache_dir / "vectordb.yaml").read_text() == "legacy: true"


@pytest.mark.parametrize("cache_root_exists", [True, False])
def test_setup_resources_cache_idempotent(tmp_path: Path, cache_root_exists: bool) -> None:
    """Calling twice with the same key is a no-op — no surprise re-link, no
    deletion of the cache contents."""
    autorag_dir = tmp_path / "autorag_project"
    autorag_dir.mkdir()
    cache_root = tmp_path / ".shared_cache"
    if cache_root_exists:
        cache_root.mkdir()
    cache_dir = _setup_resources_cache(autorag_dir, cache_root, "k1")
    (cache_dir / "marker").write_bytes(b"present")
    again = _setup_resources_cache(autorag_dir, cache_root, "k1")
    assert again == cache_dir
    assert (cache_dir / "marker").read_bytes() == b"present"
