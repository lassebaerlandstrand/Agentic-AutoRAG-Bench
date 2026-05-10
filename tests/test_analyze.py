"""Tests for the bench analyzer."""

from __future__ import annotations

import json

import numpy as np
import yaml

from autorag_bench.analyze import (
    aggregate_by_method,
    bootstrap_ci,
    load_results,
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
                    {"em": em, "f1": em, "llm_judge_correct": jg}
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


def test_yaml_round_trip_unrelated_to_pyyaml_warnings(tmp_path) -> None:
    """Sanity check that yaml import works in test context (catch path issues)."""
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"a": 1}))
    assert yaml.safe_load(p.read_text()) == {"a": 1}
