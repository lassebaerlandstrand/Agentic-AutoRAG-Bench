"""Tests for the cost-aware Pareto experiment (command + figure + labels)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.pareto import (
    ParetoConfig,
    _ensure_corpus,
    method_seed_complete,
    run_pareto,
)
from agentic_autorag_bench.plots import (
    _attainment_curve,
    _attainment_stats,
    _describe_config,
    _load_method_seed_points,
    _load_trial_points,
    _pooled_frontier,
    _select_labeled_frontier,
    _short_model,
    _TrialPoint,
    compute_pareto_hypervolumes,
    make_pareto_attainment_figure,
    make_pareto_attainment_median_figure,
    make_pareto_comparison_figure,
    make_pareto_cost_and_embeddings_figure,
    make_pareto_figure,
    make_pareto_frontier_annotated_figure,
    make_pareto_hv_convergence_figure,
    make_pareto_median_hv_combined_figure,
)


def _point(n: int, cost: float, acc: float) -> _TrialPoint:
    return _TrialPoint(trial_number=n, cost_per_query=cost, answer_accuracy=acc, is_pareto=False, config={})


def _write_history(seed_dir, rows: list[dict]) -> None:
    history_path = RunLayout(base=seed_dir).history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


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
        "answer_accuracy": score,
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
            "top_k": 20,
            "query_expansion": "hyde",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "reranker_top_n": 5,
        }
    )
    # No "RAG" filler; top_k attaches to retrieval, top_n to the reranker.
    assert desc == (
        "kimi-k2.5 with Recursive Splitting, Hybrid Retrieval (top_k=20), HyDE, "
        "and bge-reranker-v2-m3 reranking (top_n=5)"
    )
    assert "RAG" not in desc


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
    # No top_k/reranker_top_n keys here -> no parenthetical suffixes.
    assert desc == "gpt-4o-mini with Token Splitting and Dense Retrieval"


def test_describe_config_top_n_only_when_reranker_present() -> None:
    desc = _describe_config(
        {
            "generator_llm": "azure/gpt-4o",
            "chunking_strategy": "recursive",
            "index_type": "vector_only",
            "top_k": 12,
            "query_expansion": "none",
            "reranker": "none",
            "reranker_top_n": 7,  # ignored because reranker is "none"
        }
    )
    assert "Dense Retrieval (top_k=12)" in desc
    assert "top_n" not in desc


def test_describe_config_query_expansion_labels() -> None:
    for raw, label in [("multi_query", "Multi-Query"), ("query_decompose", "Query Decompose")]:
        desc = _describe_config(
            {
                "generator_llm": "azure/gpt-4o",
                "chunking_strategy": "recursive",
                "index_type": "vector_only",
                "query_expansion": raw,
                "reranker": "none",
            }
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
            # no accuracy -> dropped
            {"trial_number": 4, "answer_accuracy": None, "mean_llm_cost_per_query_usd": 0.001},
        ],
    )
    points = _load_trial_points(seed_dir)
    assert [p.trial_number for p in points] == [1, 2]
    assert points[1].is_pareto is True
    assert points[1].cost_per_query == 0.0020
    assert points[0].answer_accuracy == 0.50


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
    out = tmp_path / "figures" / "pareto.png"
    make_pareto_figure(seed_dir, out, domain="healthcare")
    assert out.exists()
    assert out.stat().st_size > 0


def test_make_pareto_figure_single_trial_no_frontier(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    _write_history(seed_dir, [_trial(1, 0.0015, 0.50, pareto=False)])
    out = tmp_path / "figures" / "pareto.png"
    make_pareto_figure(seed_dir, out)  # single trial, no frontier -> must not crash
    assert out.exists()
    assert out.stat().st_size > 0


def test_make_pareto_figure_empty_history_writes_nothing(tmp_path) -> None:
    seed_dir = tmp_path / "agentic_cost" / "seed_1"
    history_path = RunLayout(base=seed_dir).history
    history_path.parent.mkdir(parents=True)
    history_path.write_text("")
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


# ----------------------------------------------- two-method hypervolume


def test_compute_pareto_hypervolumes_uses_shared_reference() -> None:
    """Each method's per-seed HV is scored against ONE passed-in reference point and
    reported as the fraction of the ``[0,1] x [0, cost_ref]`` box dominated, so the
    numbers are comparable. One seed here, so mean == min == max."""
    method_seed_points = {
        "agentic_cost": [[_point(1, 0.001, 0.6), _point(2, 0.002, 0.8)]],
        "motpe": [[_point(1, 0.001, 0.5), _point(2, 0.003, 0.7)]],
    }
    hv = compute_pareto_hypervolumes(method_seed_points, 0.006)

    assert hv["cost_reference"] == pytest.approx(0.006)
    assert hv["score_reference"] == 0.0
    assert hv["methods"]["agentic_cost"]["n_seeds"] == 1
    # Hand-computed staircase areas (0.0038 / 0.0031) normalized by the ref (0.006).
    assert hv["methods"]["agentic_cost"]["hypervolume_mean"] == pytest.approx(0.0038 / 0.006)
    assert hv["methods"]["motpe"]["hypervolume_mean"] == pytest.approx(0.0031 / 0.006)
    # The dominating frontier has the larger hypervolume.
    assert hv["methods"]["agentic_cost"]["hypervolume_mean"] > hv["methods"]["motpe"]["hypervolume_mean"]


def test_compute_pareto_hypervolumes_aggregates_across_seeds() -> None:
    """Two seeds with different frontiers -> mean / min / max span both."""
    method_seed_points = {
        "m": [
            [_point(1, 0.001, 0.6), _point(2, 0.002, 0.8)],  # seed A: raw HV 0.0038
            [_point(1, 0.001, 0.5), _point(2, 0.003, 0.7)],  # seed B: raw HV 0.0031
        ],
    }
    hv = compute_pareto_hypervolumes(method_seed_points, 0.006)["methods"]["m"]
    assert hv["n_seeds"] == 2
    assert hv["hypervolume_min"] == pytest.approx(0.0031 / 0.006)
    assert hv["hypervolume_max"] == pytest.approx(0.0038 / 0.006)
    assert hv["hypervolume_mean"] == pytest.approx((0.0038 + 0.0031) / 2 / 0.006)


def test_compute_pareto_hypervolumes_dominated_point_does_not_change_hv() -> None:
    """A dominated trial is dropped from the frontier, so it leaves HV unchanged."""
    base = [_point(1, 0.001, 0.6), _point(3, 0.003, 0.9)]
    with_dominated = base + [_point(2, 0.002, 0.5)]  # dominated by trial 1
    hv_base = compute_pareto_hypervolumes({"m": [base]}, 0.006)["methods"]["m"]["hypervolume_mean"]
    hv_dom = compute_pareto_hypervolumes({"m": [with_dominated]}, 0.006)["methods"]["m"]["hypervolume_mean"]
    assert hv_dom == pytest.approx(hv_base)


def test_make_pareto_comparison_figure_emits_png(tmp_path) -> None:
    method_points = {
        "agentic_cost": [_point(1, 0.001, 0.6), _point(2, 0.002, 0.8), _point(3, 0.004, 0.55)],
        "motpe": [_point(1, 0.0012, 0.5), _point(2, 0.003, 0.7)],
    }
    out = tmp_path / "figures" / "pareto_comparison.png"
    make_pareto_comparison_figure(method_points, out, domain="healthcare")
    assert out.exists() and out.stat().st_size > 0


def test_make_pareto_comparison_figure_no_points_is_noop(tmp_path) -> None:
    out = tmp_path / "figures" / "pareto_comparison.png"
    make_pareto_comparison_figure({"agentic_cost": [], "motpe": []}, out)
    assert not out.exists()


# ------------------------------------------- multi-seed attainment / convergence


def _write_run_tree(root: Path) -> None:
    """Two methods, three seeds each, a few trials per seed."""
    for method, base in (("agentic_cost", 0.6), ("motpe", 0.5)):
        for s in (1, 2, 3):
            rows = [
                _trial(1, 0.0010, base, pareto=True),
                _trial(2, 0.0020, base + 0.1, pareto=True),
                _trial(3, 0.0040 + 0.0005 * s, base + 0.15, pareto=True),
            ]
            _write_history(root / method / f"seed_{s}", rows)


def test_load_method_seed_points_groups_seeds_per_method(tmp_path) -> None:
    _write_run_tree(tmp_path)
    msp = _load_method_seed_points(tmp_path)
    assert set(msp) == {"agentic_cost", "motpe"}
    # every method keeps its three seeds separate (not pooled into one list)
    assert [len(seeds) for seeds in msp.values()] == [3, 3]
    assert all(len(pts) == 3 for seeds in msp.values() for pts in seeds)


def test_attainment_curve_pads_zero_below_cheapest() -> None:
    # An empirical attainment function is 0 (not NaN) below the cheapest trial:
    # a run with nothing at that budget attains nothing there.
    import numpy as np

    pts = [_point(1, 0.010, 0.6), _point(2, 0.020, 0.8)]
    curve = _attainment_curve(pts, np.array([0.005, 0.010, 0.015, 0.020, 0.050]))
    assert list(curve) == [0.0, 0.6, 0.6, 0.8, 0.8]  # 0 below cheapest, then best-so-far


def test_attainment_stats_frontier_shows_single_seed_cheap_point() -> None:
    # One seed explores a cheap config the other never reaches. The frontier (max)
    # must show it; the worst-seed floor (min) is 0 there, so the band would be
    # empty at that cost.
    import numpy as np

    seed_cheap = [_point(1, 0.001, 0.5), _point(2, 0.004, 0.7)]
    seed_dear = [_point(1, 0.004, 0.65)]  # nothing under 0.004
    lo, _med, hi = _attainment_stats([seed_cheap, seed_dear], np.array([0.001, 0.004]))
    assert hi[0] == 0.5 and lo[0] == 0.0  # frontier reaches the cheap point; floor does not


def test_make_pareto_attainment_figure_emits_png(tmp_path) -> None:
    _write_run_tree(tmp_path)
    out = tmp_path / "figures" / "pareto_attainment.png"
    make_pareto_attainment_figure(tmp_path, out, domain="healthcare")
    assert out.exists() and out.stat().st_size > 0


def test_make_pareto_attainment_median_figure_emits_png(tmp_path) -> None:
    _write_run_tree(tmp_path)
    out = tmp_path / "figures" / "pareto_attainment_median.png"
    make_pareto_attainment_median_figure(tmp_path, out, domain="healthcare")
    assert out.exists() and out.stat().st_size > 0


def test_make_pareto_hv_convergence_figure_emits_png(tmp_path) -> None:
    _write_run_tree(tmp_path)
    out = tmp_path / "figures" / "pareto_hv_convergence.png"
    make_pareto_hv_convergence_figure(tmp_path, out, domain="healthcare", cost_ref=0.006)
    assert out.exists() and out.stat().st_size > 0


def test_make_pareto_median_hv_combined_figure_emits_png(tmp_path) -> None:
    _write_run_tree(tmp_path)
    out = tmp_path / "figures" / "pareto_median_and_hypervolume.png"
    make_pareto_median_hv_combined_figure(tmp_path, out, domain="healthcare", cost_ref=0.006)
    assert out.exists() and out.stat().st_size > 0


def test_make_pareto_median_hv_combined_figure_no_points_is_noop(tmp_path) -> None:
    (tmp_path / "agentic_cost" / "seed_1").mkdir(parents=True)  # empty: no history
    out = tmp_path / "figures" / "pareto_median_and_hypervolume.png"
    make_pareto_median_hv_combined_figure(tmp_path, out, cost_ref=0.006)
    assert not out.exists()


def test_multiseed_figures_no_points_is_noop(tmp_path) -> None:
    (tmp_path / "agentic_cost" / "seed_1").mkdir(parents=True)  # empty: no history
    make_pareto_attainment_figure(tmp_path, tmp_path / "figures" / "pareto_attainment.png")
    make_pareto_attainment_median_figure(tmp_path, tmp_path / "figures" / "pareto_attainment_median.png")
    make_pareto_hv_convergence_figure(tmp_path, tmp_path / "figures" / "pareto_hv_convergence.png", cost_ref=0.006)
    assert not (tmp_path / "figures" / "pareto_attainment.png").exists()
    assert not (tmp_path / "figures" / "pareto_attainment_median.png").exists()
    assert not (tmp_path / "figures" / "pareto_hv_convergence.png").exists()


# ------------------------------------------- annotated frontier + cost/embeddings


def test_pooled_frontier_pools_seeds_and_drops_dominated() -> None:
    # seed 2's cheap+strong point dominates seed 1's first; both a dear-weak point
    # (seed 2) and the dominated point fall off the pooled frontier.
    seed1 = [_point(1, 0.0010, 0.50), _point(2, 0.0020, 0.70)]
    seed2 = [_point(1, 0.0008, 0.55), _point(2, 0.0015, 0.40)]
    frontier = _pooled_frontier([seed1, seed2])
    costs = [round(p.cost_per_query, 4) for p in frontier]
    assert costs == [0.0008, 0.0020]  # sorted by cost, dominated points removed


def test_select_labeled_frontier_caps_and_keeps_endpoints() -> None:
    frontier = [_point(i, 0.0001 * (i + 1), 0.30 + 0.05 * i) for i in range(9)]
    picked = _select_labeled_frontier(frontier, max_labels=6)
    assert len(picked) == 6
    assert picked[0] is frontier[0] and picked[-1] is frontier[-1]  # both endpoints kept
    # under the cap: everything is returned untouched
    assert _select_labeled_frontier(frontier[:4], max_labels=6) == frontier[:4]


def test_make_pareto_frontier_annotated_figure_emits_png(tmp_path) -> None:
    _write_run_tree(tmp_path)
    out = tmp_path / "figures" / "pareto_frontier_configs.png"
    make_pareto_frontier_annotated_figure(tmp_path, out, domain="healthcare")
    assert out.exists() and out.stat().st_size > 0


def _write_meta(seed_dir: Path, *, optimizer_usd: float, trial_usd: float, embedding_tokens: float) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "optimizer_meta.json").write_text(
        json.dumps(
            {"optimizer_usd": optimizer_usd, "trial_usd_total": trial_usd, "embedding_tokens": embedding_tokens}
        ),
        encoding="utf-8",
    )


def test_make_pareto_cost_and_embeddings_figure_emits_png(tmp_path) -> None:
    for method, opt in (("agentic_cost", 0.7), ("motpe", 0.0)):
        for s in (1, 2, 3):
            _write_meta(
                tmp_path / method / f"seed_{s}",
                optimizer_usd=opt,
                trial_usd=3.0 + 0.1 * s,
                embedding_tokens=1e7 * (1 + s) if method == "motpe" else 1.2e7,
            )
    out = tmp_path / "figures" / "cost_and_embeddings.png"
    make_pareto_cost_and_embeddings_figure(tmp_path, out, domain="healthcare")
    assert out.exists() and out.stat().st_size > 0


def test_make_pareto_cost_and_embeddings_figure_no_meta_is_noop(tmp_path) -> None:
    (tmp_path / "agentic_cost" / "seed_1").mkdir(parents=True)  # no optimizer_meta.json
    out = tmp_path / "figures" / "cost_and_embeddings.png"
    make_pareto_cost_and_embeddings_figure(tmp_path, out)
    assert not out.exists()


# ------------------------------------------ methods/seeds config + selection


def test_pareto_config_parses_methods_and_seeds(tmp_path) -> None:
    (tmp_path / "proj.yaml").write_text("meta:\n  corpus_path: ./corpus\n  output_dir: ./out/.shared_cache\n")
    (tmp_path / "entry.yaml").write_text(
        yaml.safe_dump(
            {
                "project_config": "./proj.yaml",
                "seed": 2,
                "budget": {"max_trials": 30},
                "methods": ["agentic_cost", "motpe"],
                "seeds": [1, 2, 3],
                "output_root": "./out",
            }
        )
    )
    cfg = ParetoConfig.load(tmp_path / "entry.yaml")
    assert cfg.methods == ["agentic_cost", "motpe"]
    assert cfg.seeds == [1, 2, 3]
    assert cfg.max_trials == 30


def test_pareto_config_defaults_methods_and_seeds(tmp_path) -> None:
    (tmp_path / "proj.yaml").write_text("meta:\n  corpus_path: ./corpus\n  output_dir: ./out/.shared_cache\n")
    (tmp_path / "entry.yaml").write_text(
        yaml.safe_dump(
            {
                "project_config": "./proj.yaml",
                "seed": 5,
                "budget": {"max_trials": 30},
                "output_root": "./out",
            }
        )
    )
    cfg = ParetoConfig.load(tmp_path / "entry.yaml")
    # default method slate (canonical run order) and seeds default to [seed]
    assert cfg.methods == ["agentic_cost", "random", "motpe_warm", "motpe"]
    assert cfg.seeds == [5]


# -------------------------------------------------- completion oracle (disk)


def test_method_seed_complete_reads_optimizer_meta(tmp_path) -> None:
    root = tmp_path / "out"
    d = root / "motpe" / "seed_1"
    d.mkdir(parents=True)
    assert method_seed_complete(root, "motpe", 1, 30) is False  # no meta yet
    (d / "optimizer_meta.json").write_text(json.dumps({"n_trials_completed": 12}))
    assert method_seed_complete(root, "motpe", 1, 30) is False  # short of target
    (d / "optimizer_meta.json").write_text(json.dumps({"n_trials_completed": 30}))
    assert method_seed_complete(root, "motpe", 1, 30) is True
    (d / "optimizer_meta.json").write_text(json.dumps({"n_trials_completed": 31}))
    assert method_seed_complete(root, "motpe", 1, 30) is True  # >= target
    (d / "optimizer_meta.json").write_text("{ not json")
    assert method_seed_complete(root, "motpe", 1, 30) is False  # corrupt -> not done


# -------------------------------------- run_pareto: setup-only + method gate


def _write_pareto_configs(tmp_path, *, methods=None, seeds=None, max_trials=3):
    """A minimal real (entry, project) config pair; returns (entry_path, cache_dir)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "healthcare_stub.pdf").write_bytes(b"%PDF-1.4 stub")
    out = tmp_path / "out"
    cache = out / ".shared_cache"
    (tmp_path / "proj.yaml").write_text(f"meta:\n  corpus_path: {corpus}\n  output_dir: {cache}\n")
    data = {
        "project_config": "./proj.yaml",
        "seed": 1,
        "budget": {"max_trials": max_trials},
        "corpus": {"domain": "healthcare", "max_pdfs": 5, "max_images": 0},
        "output_root": str(out),
    }
    if methods is not None:
        data["methods"] = methods
    if seeds is not None:
        data["seeds"] = seeds
    entry = tmp_path / "entry.yaml"
    entry.write_text(yaml.safe_dump(data))
    return entry, cache


def _mock_orchestrator():
    inst = MagicMock()
    inst.setup = AsyncMock()
    inst.cleanup = AsyncMock()
    return inst


def test_run_pareto_setup_only_writes_marker_and_runs_no_trials(tmp_path) -> None:
    from agentic_autorag_bench.pareto import SETUP_MARKER

    entry, cache = _write_pareto_configs(tmp_path)
    orch = _mock_orchestrator()
    with (
        patch("agentic_autorag.litellm_runtime.configure_litellm_runtime") as m_lite,
        patch("agentic_autorag_bench.pareto._ensure_corpus"),
        patch("agentic_autorag.orchestrator.Orchestrator", return_value=orch),
        patch("agentic_autorag_bench.methods.agentic.AgenticOptimizer") as m_ag,
        patch("agentic_autorag_bench.methods.random.RandomSearch") as m_rand,
        patch("agentic_autorag_bench.methods.motpe.MOTPESearch") as m_motpe,
    ):
        asyncio.run(run_pareto(entry, setup_only=True))

    assert (cache / SETUP_MARKER).exists()  # single-writer warmup marker
    orch.setup.assert_awaited_once()
    m_lite.assert_called_once()
    m_ag.assert_not_called()  # no trials run under --setup-only
    m_rand.assert_not_called()
    m_motpe.assert_not_called()


def test_run_pareto_agentic_only_skips_shared_orchestrator_and_figures(tmp_path) -> None:
    entry, cache = _write_pareto_configs(tmp_path)
    sr = MagicMock()
    sr.history = []
    ag_inst = MagicMock()
    ag_inst.search = AsyncMock(return_value=sr)
    with (
        patch("agentic_autorag.litellm_runtime.configure_litellm_runtime"),
        patch("agentic_autorag_bench.pareto._ensure_corpus"),
        patch("agentic_autorag.orchestrator.Orchestrator") as m_orch,
        patch("agentic_autorag_bench.methods.agentic.AgenticOptimizer", return_value=ag_inst) as m_ag,
        patch("agentic_autorag_bench.methods.random.RandomSearch") as m_rand,
        patch("agentic_autorag_bench.methods.motpe.MOTPESearch") as m_motpe,
        patch("agentic_autorag_bench.run._persist_search_result") as m_persist,
    ):
        asyncio.run(run_pareto(entry, methods=["agentic_cost"], seed=1))

    m_ag.assert_called_once()
    m_orch.assert_not_called()  # agentic_cost carries its own orchestrator
    m_rand.assert_not_called()
    m_motpe.assert_not_called()
    m_persist.assert_called_once()
    # a --methods subset defers figures to the --figure-only finalize pass
    assert not (cache.parent / "hypervolume.json").exists()


def test_run_pareto_motpe_warm_requires_completed_random_transfer_source(tmp_path) -> None:
    entry, _cache = _write_pareto_configs(tmp_path)
    orch = _mock_orchestrator()
    with (
        patch("agentic_autorag.litellm_runtime.configure_litellm_runtime"),
        patch("agentic_autorag_bench.pareto._ensure_corpus"),
        patch("agentic_autorag.orchestrator.Orchestrator", return_value=orch),
        patch("agentic_autorag_bench.methods.motpe.MOTPESearch") as m_motpe,
        pytest.raises(RuntimeError, match="transfer prior"),
    ):
        # motpe_warm without random in the invocation, and no completed random on disk
        asyncio.run(run_pareto(entry, methods=["motpe_warm"], seed=1))
    m_motpe.assert_not_called()
    orch.cleanup.assert_awaited_once()  # shared orchestrator still torn down
