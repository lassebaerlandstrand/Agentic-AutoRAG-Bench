"""Tests for the cost-aware Pareto experiment (command + figure + labels)."""

from __future__ import annotations

import json
from unittest.mock import patch

import yaml

from agentic_autorag_bench.pareto import (
    ParetoConfig,
    _describe_config,
    _ensure_corpus,
    _load_trial_points,
    _read_frontier_meta,
    _short_model,
    make_pareto_figure,
)


def _write_history(seed_dir, rows: list[dict]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _trial(n: int, cost: float, score: float, pareto: bool, **config) -> dict:
    """A rich agentic history row (only the fields the figure reads)."""
    base_config = {
        "generator_llm": "azure/gpt-4o-mini",
        "chunking_strategy": "recursive",
        "index_type": "vector_only",
        "query_expansion": "none",
        "reranker": "none",
    }
    base_config.update(config)
    return {
        "trial_number": n,
        "score": score,
        "mean_llm_cost_per_query_usd": cost,
        "is_pareto_optimal": pareto,
        "config": base_config,
    }


# ------------------------------------------------------------- _short_model


def test_short_model_strips_provider_and_vendor() -> None:
    assert _short_model("bedrock/moonshotai.kimi-k2.5") == "kimi-k2.5"
    assert _short_model("azure/gpt-4o-mini") == "gpt-4o-mini"
    assert _short_model("bedrock/us.meta.llama3-3-70b-instruct-v1:0") == "llama3-3-70b-instruct-v1"
    assert _short_model("vertex_ai/gemini-2.5-flash") == "gemini-2.5-flash"
    assert _short_model("") == "?"


def test_short_model_keeps_version_dots_when_not_a_vendor() -> None:
    # "gpt-5.4-mini" has a dot in the version, but "gpt-5" is not a vendor token.
    assert _short_model("azure/gpt-5.4-mini") == "gpt-5.4-mini"


# ----------------------------------------------------------- _describe_config


def test_describe_config_full_pipeline() -> None:
    desc = _describe_config(
        {
            "generator_llm": "bedrock/moonshotai.kimi-k2.5",
            "chunking_strategy": "recursive",
            "index_type": "hybrid_bm25_vector",
            "query_expansion": "hyde",
            "reranker": "BAAI/bge-reranker-v2-m3",
        }
    )
    assert desc == (
        "kimi-k2.5 RAG with Recursive Splitting, Hybrid Retrieval, HyDE, "
        "and bge-reranker-v2-m3 reranking"
    )


def test_describe_config_omits_none_expansion_and_reranker() -> None:
    desc = _describe_config(
        {
            "generator_llm": "azure/gpt-4o-mini",
            "chunking_strategy": "fixed",
            "index_type": "vector_only",
            "query_expansion": "none",
            "reranker": "none",
        }
    )
    # "none" query_expansion and reranker drop out; two clauses join with "and".
    assert desc == "gpt-4o-mini RAG with Token Splitting and Dense Retrieval"


def test_describe_config_query_expansion_labels() -> None:
    for raw, label in [("multi_query", "Multi-Query"), ("query_decompose", "Query Decompose")]:
        desc = _describe_config(
            {"generator_llm": "azure/gpt-4o", "chunking_strategy": "recursive",
             "index_type": "vector_only", "query_expansion": raw, "reranker": "none"}
        )
        assert label in desc


# ----------------------------------------------------------- trial loading


def test_load_trial_points_filters_and_parses(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    _write_history(
        seed_dir,
        [
            _trial(1, 0.0010, 0.50, pareto=False),
            _trial(2, 0.0020, 0.80, pareto=True),
            _trial(3, 0.0, 0.40, pareto=False),  # cost 0 -> dropped (log axis)
            {"trial_number": 4, "score": None, "mean_llm_cost_per_query_usd": 0.001},  # no score -> dropped
        ],
    )
    points = _load_trial_points(seed_dir)
    assert [p.trial_number for p in points] == [1, 2]
    assert points[1].is_pareto is True
    assert points[1].cost_per_query == 0.0020
    assert points[0].score == 0.50


def test_frontier_subset_sorted_by_cost(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    _write_history(
        seed_dir,
        [
            _trial(1, 0.0030, 0.85, pareto=True),
            _trial(2, 0.0010, 0.60, pareto=True),
            _trial(3, 0.0050, 0.86, pareto=False),
            _trial(4, 0.0020, 0.78, pareto=True),
        ],
    )
    points = _load_trial_points(seed_dir)
    frontier = sorted((p for p in points if p.is_pareto), key=lambda p: p.cost_per_query)
    assert [p.trial_number for p in frontier] == [2, 4, 1]  # by ascending cost


def test_read_frontier_meta(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "frontier.json").write_text(
        json.dumps({"knee_trial": 4, "recommended_trial": 1, "max_score_trial": 1, "frontier": []})
    )
    meta = _read_frontier_meta(seed_dir)
    assert meta == {"knee": 4, "recommended": 1, "max_score": 1}


def test_read_frontier_meta_missing_file(tmp_path) -> None:
    assert _read_frontier_meta(tmp_path) == {}


# ------------------------------------------------------------------- figure


def test_make_pareto_figure_emits_png(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    _write_history(
        seed_dir,
        [
            _trial(1, 0.0012, 0.50, pareto=False, generator_llm="bedrock/moonshotai.kimi-k2.5"),
            _trial(2, 0.0010, 0.60, pareto=True, index_type="hybrid_bm25_vector"),
            _trial(3, 0.0020, 0.82, pareto=True, query_expansion="hyde"),
            _trial(4, 0.0035, 0.84, pareto=True, reranker="BAAI/bge-reranker-v2-m3"),
            _trial(5, 0.0040, 0.55, pareto=False),
        ],
    )
    (seed_dir / "frontier.json").write_text(
        json.dumps({"knee_trial": 3, "recommended_trial": 4, "max_score_trial": 4, "frontier": []})
    )
    out = tmp_path / "figures" / "pareto.png"
    make_pareto_figure(seed_dir, out, domain="healthcare")
    assert out.exists()
    assert out.stat().st_size > 0


def test_make_pareto_figure_single_trial_no_frontier(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    _write_history(seed_dir, [_trial(1, 0.0015, 0.50, pareto=False)])
    out = tmp_path / "figures" / "pareto.png"
    make_pareto_figure(seed_dir, out)  # no frontier, no frontier.json -> must not crash
    assert out.exists()
    assert out.stat().st_size > 0


def test_make_pareto_figure_empty_history_writes_nothing(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "history.jsonl").write_text("")
    out = tmp_path / "figures" / "pareto.png"
    make_pareto_figure(seed_dir, out)
    assert not out.exists()


# --------------------------------------------------------------- config load


def test_pareto_config_load_resolves_paths(tmp_path) -> None:
    (tmp_path / "proj.yaml").write_text("meta:\n  corpus_path: ./corpus\n")
    (tmp_path / "entry.yaml").write_text(
        yaml.safe_dump(
            {
                "project_config": "./proj.yaml",
                "seed": 1,
                "budget": {"max_trials": 40},
                "corpus": {"domain": "healthcare", "max_pdfs": 230, "max_images": 20},
                "output_root": "./results_unidoc",
            }
        )
    )
    cfg = ParetoConfig.load(tmp_path / "entry.yaml")
    assert cfg.project_config_path == (tmp_path / "proj.yaml").resolve()
    assert cfg.seed == 1
    assert cfg.max_trials == 40
    assert cfg.corpus_domain == "healthcare"
    assert cfg.corpus_max_pdfs == 230
    assert cfg.corpus_max_images == 20


# ------------------------------------------------------------- corpus prep


def _cfg(output_root) -> ParetoConfig:
    from pathlib import Path

    return ParetoConfig(
        project_config_path=Path("proj.yaml"),
        seed=1,
        max_trials=40,
        corpus_domain="healthcare",
        corpus_max_pdfs=230,
        corpus_max_images=20,
        output_root=Path(output_root),
    )


def test_ensure_corpus_skips_when_populated(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "healthcare_0000001.pdf").write_bytes(b"%PDF-1.4 stub")
    with patch("agentic_autorag_bench.unidoc_corpus.download_unidoc_corpus") as mock_dl:
        _ensure_corpus(corpus, _cfg(tmp_path))
    mock_dl.assert_not_called()


def test_ensure_corpus_downloads_when_empty(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    with patch("agentic_autorag_bench.unidoc_corpus.download_unidoc_corpus") as mock_dl:
        _ensure_corpus(corpus, _cfg(tmp_path))
    mock_dl.assert_called_once()
    _, kwargs = mock_dl.call_args
    assert kwargs["domain"] == "healthcare"
    assert kwargs["max_pdfs"] == 230
    assert kwargs["max_images"] == 20
