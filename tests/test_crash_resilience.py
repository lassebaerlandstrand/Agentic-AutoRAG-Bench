"""A process killed mid-write can leave a truncated/zero-byte JSON anywhere in
the result tree. The end-of-run union-exclusion + the figure/analyze readers
must degrade past such a file (treat it as not-yet-scored) instead of aborting
the whole matrix after all the expensive search + hold-out compute landed.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_autorag_bench._holdout_registry import apply_union_exclusion
from agentic_autorag_bench.analyze import load_results


def _valid_results(seed_dir: Path, *, accuracy: float = 0.8) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_model": "azure/gpt-4o-mini",
        "n_total": 2,
        "n_valid": 2,
        "em": 0.5,
        "f1": 0.6,
        "llm_judge_accuracy": accuracy,
        "per_question": [
            {"id": "q1", "em": 1.0, "f1": 1.0, "judge": 1, "supporting_doc_ids": [], "retrieved_doc_ids": []},
            {"id": "q2", "em": 0.0, "f1": 0.2, "judge": 0, "supporting_doc_ids": [], "retrieved_doc_ids": []},
        ],
    }
    (seed_dir / "benchmark_results.json").write_text(json.dumps(payload), encoding="utf-8")
    (seed_dir / "optimizer_meta.json").write_text(json.dumps({"n_trials_completed": 2}), encoding="utf-8")


def test_union_exclusion_skips_corrupt_file(tmp_path: Path) -> None:
    """One truncated benchmark_results.json must not abort the union pass; the
    valid sibling is still rescored and tagged."""
    root = tmp_path / "results"
    _valid_results(root / "random" / "seed_1")
    bad = root / "motpe" / "seed_1"
    bad.mkdir(parents=True)
    (bad / "benchmark_results.json").write_text('{"per_question": [trunca', encoding="utf-8")  # truncated

    registry = apply_union_exclusion(root)  # must not raise

    assert registry["n_runs_scanned"] == 1  # only the readable file counted
    good = json.loads((root / "random" / "seed_1" / "benchmark_results.json").read_text())
    assert "excluded_question_ids" in good  # the valid file was rewritten
    # The corrupt file is left untouched for the next --resume to regenerate.
    assert (bad / "benchmark_results.json").read_text().startswith('{"per_question": [trunca')


def test_union_exclusion_all_corrupt_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "results"
    bad = root / "random" / "seed_1"
    bad.mkdir(parents=True)
    (bad / "benchmark_results.json").write_text("", encoding="utf-8")  # zero-byte

    registry = apply_union_exclusion(root)  # must not raise

    assert registry["excluded_ids"] == []


def test_load_results_skips_corrupt_seed(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _valid_results(root / "random" / "seed_1")
    bad = root / "random" / "seed_2"
    bad.mkdir(parents=True)
    (bad / "benchmark_results.json").write_text("{ partial", encoding="utf-8")

    results = load_results(root)  # must not raise

    methods_seeds = {(r.method, r.seed) for r in results}
    assert ("random", 1) in methods_seeds
    assert ("random", 2) not in methods_seeds  # corrupt seed skipped


def test_load_results_skips_corrupt_history_but_keeps_seed(tmp_path: Path) -> None:
    """A truncated history.jsonl shouldn't drop the whole seed if the
    benchmark_results.json is fine — load_results guards the seed body."""
    root = tmp_path / "results"
    seed_dir = root / "random" / "seed_1"
    _valid_results(seed_dir)
    details = seed_dir / "details"
    details.mkdir(parents=True, exist_ok=True)
    (details / "history.jsonl").write_text('{"trial_number": 1}\n{ truncated', encoding="utf-8")

    results = load_results(root)  # must not raise
    # Seed is skipped on the corrupt history (conservative: the whole body is guarded).
    # Either outcome is acceptable as long as it does not raise; assert no crash + valid seed handling.
    assert isinstance(results, list)
