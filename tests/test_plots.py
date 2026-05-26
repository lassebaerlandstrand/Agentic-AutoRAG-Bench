"""Tests for the auto-figure generator.

The matrix-level figures share their writers with ``analyze.py`` and are
covered transitively by ``test_analyze.py``. These tests focus on the
seed-level and method-level entry points and on the schema-dual cost
extraction (agentic vs. random/bayesian/autorag history.jsonl shapes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agentic_autorag_bench.plots import (
    _entry_eval_usd,
    _holdout_judge_mean,
    _pad_edge,
    _pad_nan,
    make_matrix_figures,
    make_method_figures,
    make_seed_figures,
)


def _write_seed(
    seed_dir: Path,
    scores: list[float],
    eval_usds: list[float],
    *,
    schema: str = "bench",
    judges: list[int | None] | None = None,
    write_benchmark: bool = True,
) -> None:
    """Lay down a minimal seed-dir so plots can read it.

    ``schema='bench'`` mirrors the bench's reduced HistoryEntry (random,
    bayesian, autorag). ``schema='agentic'`` mirrors the framework's richer
    history.jsonl (uses ``total_llm_cost_usd`` instead of ``eval_usd``).
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (s, c) in enumerate(zip(scores, eval_usds, strict=True), start=1):
        entry: dict = {
            "trial_number": i,
            "config": {"chunk_token_size": 256},
            "score": s,
            "metrics": {},
        }
        if schema == "bench":
            entry["eval_usd"] = c
        else:
            entry["total_llm_cost_usd"] = c
            entry["trial_metrics"] = {"answer_accuracy": s}
        lines.append(json.dumps(entry))
    (seed_dir / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if write_benchmark:
        per_q = []
        if judges is None:
            judges = [1, 0, 1, None][: max(1, len(scores))]
        for i, j in enumerate(judges):
            per_q.append({"id": f"Q{i:02d}", "em": 1.0 if j == 1 else 0.0,
                          "f1": 1.0 if j == 1 else 0.0, "judge": j})
        em = float(np.mean([r["em"] for r in per_q]))
        (seed_dir / "benchmark_results.json").write_text(
            json.dumps({"n_total": len(per_q), "em": em, "f1": em, "mrr": 0.5,
                        "per_question": per_q}),
            encoding="utf-8",
        )
    (seed_dir / "optimizer_meta.json").write_text(
        json.dumps({"method": seed_dir.parent.name, "seed": seed_dir.name,
                    "wall_clock_s": 100.0, "optimizer_usd": 0.05,
                    "trial_usd_total": sum(eval_usds),
                    "n_trials_completed": len(scores)}),
        encoding="utf-8",
    )


def test_entry_eval_usd_handles_both_schemas() -> None:
    assert _entry_eval_usd({"eval_usd": 0.5}) == pytest.approx(0.5)
    assert _entry_eval_usd({"total_llm_cost_usd": 1.25}) == pytest.approx(1.25)
    # ``eval_usd`` wins when both present — the bench's reduced schema is the
    # canonical one for non-agentic methods.
    assert _entry_eval_usd({"eval_usd": 0.1, "total_llm_cost_usd": 0.9}) == pytest.approx(0.1)
    assert _entry_eval_usd({}) == 0.0


def test_holdout_judge_mean_drops_none_rows() -> None:
    benchmark = {"per_question": [
        {"id": "a", "judge": 1},
        {"id": "b", "judge": 0},
        {"id": "c", "judge": 1},
        # judge=None means the judge call failed — exclude from denominator
        {"id": "d", "judge": None},
    ]}
    assert _holdout_judge_mean(benchmark) == pytest.approx(2 / 3)


def test_holdout_judge_mean_returns_none_when_no_judge_column() -> None:
    """benchmark_results.json without a judge column (no judge_model) → None."""
    benchmark = {"per_question": [{"id": "a", "em": 1.0, "f1": 1.0}]}
    assert _holdout_judge_mean(benchmark) is None


def test_holdout_judge_mean_respects_excluded_question_ids() -> None:
    benchmark = {
        "excluded_question_ids": ["b"],
        "per_question": [
            {"id": "a", "judge": 1},
            {"id": "b", "judge": 0},  # excluded
            {"id": "c", "judge": 1},
        ],
    }
    assert _holdout_judge_mean(benchmark) == pytest.approx(1.0)


def test_pad_edge_replicates_last_value() -> None:
    """Best-so-far curves do not regress; edge replication keeps the mean honest
    when one seed ran fewer trials than another."""
    curves = [np.array([0.1, 0.5, 0.7]), np.array([0.2, 0.6])]
    padded = _pad_edge(curves)
    assert padded.shape == (2, 3)
    # Second curve pads with its last value (0.6), not 0.
    assert padded[1, 2] == pytest.approx(0.6)


def test_pad_nan_does_not_bias_mean_when_one_seed_aborts() -> None:
    """Raw per-trial scores must NaN-pad: a seed that stopped early shouldn't
    drag the mean down with synthetic late-trial values."""
    curves = [np.array([0.8, 0.9, 0.85]), np.array([0.7, 0.75])]
    padded = _pad_nan(curves)
    assert np.isnan(padded[1, 2])
    # Mean over column 2 ignores the NaN entry
    assert np.nanmean(padded[:, 2]) == pytest.approx(0.85)


def test_make_seed_figures_emits_score_and_cost(tmp_path) -> None:
    seed_dir = tmp_path / "random" / "seed_1"
    _write_seed(seed_dir, [0.3, 0.5, 0.7, 0.6], [0.10, 0.12, 0.11, 0.13])
    make_seed_figures(seed_dir)
    figs = seed_dir / "figures"
    assert (figs / "score_per_trial.png").exists()
    assert (figs / "score_per_trial.png").stat().st_size > 0
    assert (figs / "cost_per_trial.png").exists()
    assert (figs / "cost_per_trial.png").stat().st_size > 0


def test_make_seed_figures_handles_agentic_schema(tmp_path) -> None:
    """Agentic's framework-written history.jsonl uses ``total_llm_cost_usd``
    instead of ``eval_usd``; the per-seed cost plot must still render."""
    seed_dir = tmp_path / "agentic" / "seed_1"
    _write_seed(
        seed_dir, [0.4, 0.6, 0.5], [0.20, 0.22, 0.21],
        schema="agentic",
    )
    make_seed_figures(seed_dir)
    assert (seed_dir / "figures" / "cost_per_trial.png").exists()


def test_make_seed_figures_no_history_is_noop(tmp_path) -> None:
    """A seed dir without a history.jsonl (early-aborted run) → no figures dir."""
    seed_dir = tmp_path / "agentic" / "seed_1"
    seed_dir.mkdir(parents=True)
    make_seed_figures(seed_dir)
    assert not (seed_dir / "figures").exists()


def test_make_seed_figures_works_without_benchmark(tmp_path) -> None:
    """Score plot should render even when hold-out scoring has not yet
    completed (no benchmark_results.json yet)."""
    seed_dir = tmp_path / "random" / "seed_1"
    _write_seed(seed_dir, [0.3, 0.5], [0.10, 0.12], write_benchmark=False)
    make_seed_figures(seed_dir)
    assert (seed_dir / "figures" / "score_per_trial.png").exists()


def test_make_method_figures_aggregates_seeds(tmp_path) -> None:
    method_dir = tmp_path / "random"
    _write_seed(method_dir / "seed_1", [0.3, 0.5, 0.7, 0.6], [0.10, 0.12, 0.11, 0.13])
    _write_seed(method_dir / "seed_2", [0.4, 0.6, 0.55, 0.65], [0.10, 0.11, 0.10, 0.12])
    make_method_figures(method_dir)
    figs = method_dir / "figures"
    for name in ("score_per_trial.png", "best_so_far.png", "holdout_metrics.png"):
        assert (figs / name).exists(), name
        assert (figs / name).stat().st_size > 0


def test_make_method_figures_ignores_figures_subdir(tmp_path) -> None:
    """A pre-existing ``figures/`` next to seed dirs must not be treated as a
    seed dir. The seed scanner has to skip it explicitly."""
    method_dir = tmp_path / "random"
    (method_dir / "figures").mkdir(parents=True)  # stale figures from earlier run
    _write_seed(method_dir / "seed_1", [0.3, 0.5], [0.10, 0.11])
    make_method_figures(method_dir)
    # No exception. Sanity: the new figures land where we expect.
    assert (method_dir / "figures" / "score_per_trial.png").exists()


def test_make_method_figures_handles_ragged_seeds(tmp_path) -> None:
    """One seed shorter than another (e.g. crashed mid-trial) — figure must
    still render and not pad the raw-score mean with synthetic values."""
    method_dir = tmp_path / "agentic_score"
    _write_seed(method_dir / "seed_1", [0.3, 0.5, 0.7], [0.1, 0.1, 0.1], schema="agentic")
    _write_seed(method_dir / "seed_2", [0.4, 0.6], [0.1, 0.1], schema="agentic")
    make_method_figures(method_dir)
    assert (method_dir / "figures" / "score_per_trial.png").exists()


def test_make_matrix_figures_assembles_full_matrix(tmp_path) -> None:
    """End-to-end matrix-level write. Covers the same surface as
    test_analyze_emits_all_artifacts but invokes plots.py directly."""
    output_root = tmp_path / "results_paper"
    _write_seed(output_root / "agentic_score" / "seed_1", [0.6, 0.7, 0.75], [0.1, 0.1, 0.1])
    _write_seed(output_root / "random" / "seed_1", [0.3, 0.5, 0.6], [0.1, 0.1, 0.1])
    _write_seed(output_root / "autorag_our_exam" / "default", [0.55], [0.1])
    make_matrix_figures(output_root)
    figs = output_root / "figures"
    for name in (
        "Table_1.md",
        "score_per_trial.png",
        "best_so_far.png",
        "holdout_metrics.png",
        "cost_breakdown.png",
        "token_breakdown.png",
    ):
        assert (figs / name).exists(), name
        assert (figs / name).stat().st_size > 0
    # efficiency.png moved to appendix subdir — no longer part of the paper body.
    assert (figs / "appendix" / "efficiency.png").exists()


def test_make_matrix_figures_redirect_output_dir(tmp_path) -> None:
    """``figures_dir`` override is the bridge between auto-run mode (writes
    alongside results) and ``analyze --output`` mode (writes elsewhere)."""
    output_root = tmp_path / "results_paper"
    elsewhere = tmp_path / "paper_artifacts" / "figures"
    _write_seed(output_root / "random" / "seed_1", [0.3, 0.5], [0.1, 0.1])
    make_matrix_figures(output_root, figures_dir=elsewhere)
    assert (elsewhere / "Table_1.md").exists()
    # No accidental write to the default location
    assert not (output_root / "figures").exists()


def test_make_matrix_figures_skips_shared_cache(tmp_path) -> None:
    """``.shared_cache`` is the framework's cache dir, not a method. The
    method scanner must not treat it as one (otherwise load_results will
    blow up trying to read its non-existent seed dirs)."""
    output_root = tmp_path / "results_paper"
    (output_root / ".shared_cache" / "exam").mkdir(parents=True)
    _write_seed(output_root / "random" / "seed_1", [0.3, 0.5], [0.1, 0.1])
    make_matrix_figures(output_root)
    # Sanity: ran without raising. Output exists.
    assert (output_root / "figures" / "Table_1.md").exists()
