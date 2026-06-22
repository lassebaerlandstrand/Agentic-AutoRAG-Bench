"""Tests for the bench analyzer."""

from __future__ import annotations

import json

import numpy as np
import yaml
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.analyze import (
    aggregate_by_method,
    analyze,
    bootstrap_ci,
    load_results,
    write_efficiency_figure,
    write_holdout_scores_figure,
    write_markdown_table,
)


def test_bootstrap_ci_returns_mean_within_bounds() -> None:
    np.random.seed(0)
    values = np.random.normal(0.5, 0.1, size=200)
    mean, lo, hi = bootstrap_ci(values)
    assert lo <= mean <= hi
    assert abs(mean - values.mean()) < 1e-9


def test_bootstrap_ci_handles_empty_array() -> None:
    mean, lo, hi = bootstrap_ci(np.array([]))
    assert mean == lo == hi == 0.0


def test_bootstrap_ci_constant_values_have_zero_width() -> None:
    values = np.full(100, 0.7)
    mean, lo, hi = bootstrap_ci(values)
    assert abs(mean - 0.7) < 1e-9
    assert abs(hi - lo) < 1e-9


def _write_method_dir(root, method: str, seed: int | None, em_scores: list[float], judge_scores: list[bool]) -> None:
    """Mirror BenchmarkResult: per-question rows carry ``judge: int | None``."""
    seed_dir = (root / method) / (f"seed_{seed}" if seed is not None else "default")
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "benchmark_results.json").write_text(
        json.dumps(
            {
                "n_total": len(em_scores),
                "em": float(np.mean(em_scores)),
                "f1": float(np.mean(em_scores)),
                "llm_judge_accuracy": float(np.mean([1.0 if v else 0.0 for v in judge_scores])),
                "mrr_first": 0.5,
                "mrr_complete": 0.4,
                "joint_recall_at_2": 0.6,
                "joint_recall_at_5": 0.7,
                "joint_recall_at_10": 0.8,
                "per_question": [
                    {"em": em, "f1": em, "judge": 1 if jg else 0}
                    for em, jg in zip(em_scores, judge_scores, strict=True)
                ],
            }
        )
    )
    (seed_dir / "optimizer_meta.json").write_text(
        json.dumps(
            {
                "method": method,
                "seed": seed,
                "wall_clock_s": 100.0,
                "optimizer_usd": 0.05,
                "trial_usd_total": 0.5,
                "n_trials_completed": 30,
            }
        )
    )
    history_path = RunLayout(base=seed_dir).history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "\n".join(
            json.dumps({"trial_number": i + 1, "config": {}, "score": float(em), "metrics": {}, "eval_usd": 0.01})
            for i, em in enumerate(em_scores)
        )
    )


def test_per_question_judge_returns_nan_when_judge_missing(tmp_path) -> None:
    """Hold-out eval calls the judge for every row; ``judge=None`` means the call
    failed (timeout / parse / content filter). Drop those rows from the
    denominator by emitting NaN rather than falling back to EM."""
    seed_dir = (tmp_path / "agentic_score") / "seed_1"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "benchmark_results.json").write_text(
        json.dumps(
            {
                "n_total": 4,
                "em": 0.5,
                "f1": 0.5,
                "mrr": 0.5,
                "per_question": [
                    {"em": 1.0, "f1": 1.0, "judge": 1},
                    {"em": 0.0, "f1": 0.0, "judge": 1},
                    {"em": 0.0, "f1": 0.0, "judge": 0},
                    {"em": 0.0, "f1": 0.0, "judge": None},  # judge call failed
                ],
            }
        )
    )
    results = load_results(tmp_path)
    judge = results[0].per_question_judge
    assert judge[0] == 1.0
    assert judge[1] == 1.0
    assert judge[2] == 0.0
    assert np.isnan(judge[3])


def test_bootstrap_ci_drops_nan_rows() -> None:
    """``bootstrap_ci`` should ignore NaN entries (judge-call failures)."""
    values = np.array([1.0, 1.0, 0.0, np.nan, 1.0])
    mean, lo, hi = bootstrap_ci(values)
    # 3/4 of the non-NaN rows are 1.0, so mean ≈ 0.75
    assert abs(mean - 0.75) < 1e-9
    assert lo <= mean <= hi


def test_load_results_round_trip(tmp_path) -> None:
    _write_method_dir(tmp_path, "random", 1, [1.0, 0.0, 1.0, 0.0], [True, False, True, False])
    _write_method_dir(tmp_path, "random", 2, [0.0, 1.0, 0.0, 1.0], [False, True, False, True])
    results = load_results(tmp_path)
    assert len(results) == 2
    assert {r.seed for r in results} == {1, 2}
    assert results[0].method == "random"


def test_aggregate_pools_seeds(tmp_path) -> None:
    _write_method_dir(tmp_path, "agentic_score", 1, [1.0] * 10, [True] * 10)
    _write_method_dir(tmp_path, "agentic_score", 2, [0.0] * 10, [False] * 10)
    results = load_results(tmp_path)
    stats = aggregate_by_method(results)

    assert stats["agentic_score"]["n_seeds"] == 2
    em_mean, _em_lo, _em_hi = stats["agentic_score"]["em"]
    assert abs(em_mean - 0.5) < 0.05  # pooled mean across the two seeds


def test_write_holdout_scores_figure_skips_when_no_methods(tmp_path) -> None:
    out_path = tmp_path / "figure_holdout_scores.png"
    write_holdout_scores_figure({}, out_path)
    # No methods → no file emitted (matches the table's "no results yet" early-return shape).
    assert not out_path.exists()


def test_write_efficiency_figure_skips_when_no_methods(tmp_path) -> None:
    out_path = tmp_path / "figure_efficiency.png"
    write_efficiency_figure({}, out_path)
    assert not out_path.exists()


def test_write_markdown_table_emits_pipe_table(tmp_path) -> None:
    stats = {
        "agentic_score": {
            "n_seeds": 3,
            "em": (0.5, 0.45, 0.55),
            "f1": (0.7, 0.65, 0.75),
            "judge": (0.85, 0.80, 0.90),
            "mrr_first": 0.92,
            "mrr_complete": 0.40,
            "joint_recall_at_2": 0.70,
            "joint_recall_at_5": 0.85,
            "joint_recall_at_10": 0.92,
            "wall_clock_s_mean": 1200.0,
            "optimizer_usd_mean": 0.10,
            "trial_usd_mean": 1.50,
        }
    }
    out_path = tmp_path / "Table_1.md"
    write_markdown_table(stats, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "| Method |" in text
    assert "| Agentic (Ours) |" in text  # paper-facing display label
    # New format: ``mean ± SD`` across replays. The (0.5, 0.45, 0.55) triple
    # in the stats dict reads as mean=0.5, SD=0.05.
    assert "0.500 ± 0.050" in text
    # Headline multi-hop retrieval columns must be present.
    assert "Joint-R@2" in text
    assert "MRR-complete" in text
    # Search $ = optimizer + trial ($0.10 + $1.50).
    assert "$1.6000" in text


def test_write_markdown_table_handles_empty_stats(tmp_path) -> None:
    out_path = tmp_path / "Table_1.md"
    write_markdown_table({}, out_path)
    assert "no results yet" in out_path.read_text(encoding="utf-8")


def test_analyze_emits_all_artifacts(tmp_path) -> None:
    """End-to-end: results tree → matrix figures + Table_1.md under figures/."""
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "artifacts"
    _write_method_dir(results_dir, "agentic_score", 1, [1.0, 0.0, 1.0, 1.0], [True, False, True, True])
    _write_method_dir(results_dir, "agentic_score", 2, [1.0, 1.0, 0.0, 1.0], [True, True, False, True])
    _write_method_dir(results_dir, "random", 1, [0.0, 0.0, 1.0, 0.0], [False, False, True, False])
    _write_method_dir(results_dir, "motpe", 1, [1.0, 0.0, 1.0, 0.0], [True, False, True, False])

    analyze(results_dir, output_dir)

    figures_dir = output_dir / "figures"
    # Canonical names emitted by plots.make_matrix_figures.
    for name in (
        "Table_1.md",
        "holdout_metrics.png",
        "score_per_trial.png",
        "best_so_far.png",
        "cost_breakdown.png",
        "token_breakdown.png",
        "cost_and_embeddings.png",
    ):
        path = figures_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"
    # efficiency moved to appendix
    assert (figures_dir / "appendix" / "efficiency.png").exists()
    # The legacy figure_trajectory.png was removed (superseded by best_so_far.png).
    assert not (figures_dir / "figure_trajectory.png").exists()


def test_yaml_round_trip_unrelated_to_pyyaml_warnings(tmp_path) -> None:
    """Sanity check that yaml import works in test context (catch path issues)."""
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"a": 1}))
    assert yaml.safe_load(p.read_text()) == {"a": 1}
