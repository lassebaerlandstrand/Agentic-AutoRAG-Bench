"""Tests for the bench analyzer."""

from __future__ import annotations

import json

import numpy as np
import yaml

from agentic_autorag_bench.analyze import (
    aggregate_by_method,
    analyze,
    bootstrap_ci,
    load_results,
    write_efficiency_figure,
    write_holdout_scores_figure,
    write_latex_table,
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
                "mrr": 0.5,
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
    (seed_dir / "history.jsonl").write_text(
        "\n".join(
            json.dumps({"trial_number": i + 1, "config": {}, "score": float(em), "metrics": {}, "eval_usd": 0.01})
            for i, em in enumerate(em_scores)
        )
    )


def test_per_question_judge_falls_back_to_em_when_judge_none(tmp_path) -> None:
    """When ``judge`` is None, accuracy = EM>0.5 (framework only runs judge on EM=0)."""
    seed_dir = (tmp_path / "agentic") / "seed_1"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "benchmark_results.json").write_text(
        json.dumps(
            {
                "n_total": 4,
                "em": 0.5,
                "f1": 0.5,
                "mrr": 0.5,
                "per_question": [
                    {"em": 1.0, "f1": 1.0, "judge": None},  # EM=1 → correct
                    {"em": 0.0, "f1": 0.0, "judge": 1},     # EM=0, judge YES → correct
                    {"em": 0.0, "f1": 0.0, "judge": 0},     # EM=0, judge NO → wrong
                    {"em": 0.0, "f1": 0.0, "judge": None},  # EM=0, no judge run → wrong
                ],
            }
        )
    )
    results = load_results(tmp_path)
    judge = results[0].per_question_judge
    assert list(judge) == [1.0, 1.0, 0.0, 0.0]


def test_load_results_round_trip(tmp_path) -> None:
    _write_method_dir(tmp_path, "random", 1, [1.0, 0.0, 1.0, 0.0], [True, False, True, False])
    _write_method_dir(tmp_path, "random", 2, [0.0, 1.0, 0.0, 1.0], [False, True, False, True])
    results = load_results(tmp_path)
    assert len(results) == 2
    assert {r.seed for r in results} == {1, 2}
    assert results[0].method == "random"


def test_aggregate_pools_seeds(tmp_path) -> None:
    _write_method_dir(tmp_path, "agentic", 1, [1.0] * 10, [True] * 10)
    _write_method_dir(tmp_path, "agentic", 2, [0.0] * 10, [False] * 10)
    results = load_results(tmp_path)
    stats = aggregate_by_method(results)

    assert stats["agentic"]["n_seeds"] == 2
    em_mean, _em_lo, _em_hi = stats["agentic"]["em"]
    assert abs(em_mean - 0.5) < 0.05  # pooled mean across the two seeds


def test_write_latex_table_emits_booktabs(tmp_path) -> None:
    stats = {
        "agentic": {
            "n_seeds": 3,
            "em": (0.5, 0.45, 0.55),
            "f1": (0.7, 0.65, 0.75),
            "judge": (0.85, 0.80, 0.90),
            "mrr": 0.92,
            "wall_clock_s_mean": 1200.0,
            "optimizer_usd_mean": 0.10,
            "trial_usd_mean": 1.50,
        }
    }
    out_path = tmp_path / "Table_1.tex"
    write_latex_table(stats, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "\\toprule" in text
    assert "\\bottomrule" in text
    assert "agentic" in text
    assert "[0.450, 0.550]" in text  # EM CI


def test_write_latex_table_handles_empty_stats(tmp_path) -> None:
    out_path = tmp_path / "Table_1.tex"
    write_latex_table({}, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "no results yet" in text


def test_write_holdout_scores_figure_skips_when_no_methods(tmp_path) -> None:
    out_path = tmp_path / "figure_holdout_scores.pdf"
    write_holdout_scores_figure({}, out_path)
    # No methods → no file emitted (matches the table's "no results yet" early-return shape).
    assert not out_path.exists()


def test_write_efficiency_figure_skips_when_no_methods(tmp_path) -> None:
    out_path = tmp_path / "figure_efficiency.pdf"
    write_efficiency_figure({}, out_path)
    assert not out_path.exists()


def test_analyze_emits_all_artifacts(tmp_path) -> None:
    """End-to-end: results tree → table + three figures, all non-empty."""
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "artifacts"
    _write_method_dir(results_dir, "agentic", 1, [1.0, 0.0, 1.0, 1.0], [True, False, True, True])
    _write_method_dir(results_dir, "agentic", 2, [1.0, 1.0, 0.0, 1.0], [True, True, False, True])
    _write_method_dir(results_dir, "random", 1, [0.0, 0.0, 1.0, 0.0], [False, False, True, False])
    _write_method_dir(results_dir, "autorag_mcq", None, [1.0, 0.0, 1.0, 0.0], [True, False, True, False])

    analyze(results_dir, output_dir)

    for name in (
        "Table_1.tex",
        "figure_holdout_scores.pdf",
        "figure_efficiency.pdf",
        "figure_trajectory.pdf",
    ):
        path = output_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"


def test_yaml_round_trip_unrelated_to_pyyaml_warnings(tmp_path) -> None:
    """Sanity check that yaml import works in test context (catch path issues)."""
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"a": 1}))
    assert yaml.safe_load(p.read_text()) == {"a": 1}
