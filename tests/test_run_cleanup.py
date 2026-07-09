"""Tests for the scoped start-of-run output-root cleanup.

The bench resets only the per-method dirs about to be run (and any
matching ``<method>@<k>/`` checkpoint dirs). Method dirs not in the
current run, ``.shared_cache/``, the cross-method ``figures/`` dir,
``bench_metadata.json``, and any user files at ``output_root`` are
preserved — so ``-m agentic`` does not touch a previous run's
``random/`` or ``motpe/`` results, and the previous matrix figures
stay readable for the entire duration of a new run (the new figures
are atomically swapped in at end-of-run).
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_autorag_bench.run import (
    _clear_output_root_for,
    _is_method_seed_complete,
    _methods_with_completed_results,
)


def _seed_with_history(seed_dir: Path) -> None:
    """Lay down a minimal seed dir so we can confirm it gets wiped/preserved."""
    seed_dir.mkdir(parents=True)
    (seed_dir / "history.jsonl").write_text(
        json.dumps({"trial_number": 1, "score": 0.5, "config": {}, "eval_usd": 0.1}) + "\n",
        encoding="utf-8",
    )


def test_only_targeted_method_dir_is_wiped(tmp_path: Path) -> None:
    """``-m agentic`` resets agentic/ but preserves random/, motpe/, etc."""
    root = tmp_path / "results_paper"
    for method in ("agentic", "random", "motpe"):
        _seed_with_history(root / method / "seed_1")

    removed = _clear_output_root_for(root, ["agentic"])

    assert removed == ["agentic"]
    assert not (root / "agentic").exists()
    assert (root / "random" / "seed_1" / "history.jsonl").exists()
    assert (root / "motpe" / "seed_1" / "history.jsonl").exists()


def test_figures_dir_is_preserved_during_clean(tmp_path: Path) -> None:
    """Cross-method ``figures/`` must SURVIVE a start-of-run clean — new
    matrix figures are staged and swapped in atomically at end-of-run, so
    the previous figures stay readable in the meantime. Wiping them up
    front is the regression we're guarding against."""
    root = tmp_path / "results_paper"
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "Table_1.md").write_text("# previous run\n")
    _seed_with_history(root / "agentic" / "seed_1")

    removed = _clear_output_root_for(root, ["agentic"])

    assert "figures" not in removed
    assert (root / "figures" / "Table_1.md").read_text() == "# previous run\n"


def test_shared_cache_preserved_implicitly(tmp_path: Path) -> None:
    """``.shared_cache`` is never named in the wipe targets (methods come
    from ALL_METHODS, ``figures`` is hardcoded), so the framework's parsed
    corpus + exam.json + embedding ingredients survive a clean. Rebuilding
    them costs hours; preserving is pure perf."""
    root = tmp_path / "results_paper"
    (root / ".shared_cache").mkdir(parents=True)
    (root / ".shared_cache" / "exam.json").write_text('{"keep": true}')
    _seed_with_history(root / "agentic" / "seed_1")

    _clear_output_root_for(root, ["agentic"])

    assert (root / ".shared_cache" / "exam.json").read_text() == '{"keep": true}'


def test_user_files_at_output_root_preserved(tmp_path: Path) -> None:
    """A user dropping notes.md or a scratch dir under output_root expects
    it to survive a bench run. Scoped cleanup only touches the named
    targets."""
    root = tmp_path / "results_paper"
    (root / "notes.md").parent.mkdir(parents=True)
    (root / "notes.md").write_text("manual notes\n")
    (root / "scratch").mkdir()
    (root / "scratch" / "tmp.txt").write_text("hi\n")
    _seed_with_history(root / "agentic" / "seed_1")

    _clear_output_root_for(root, ["agentic"])

    assert (root / "notes.md").read_text() == "manual notes\n"
    assert (root / "scratch" / "tmp.txt").read_text() == "hi\n"


def test_all_methods_wiped_when_all_methods_run(tmp_path: Path) -> None:
    """The full-matrix run (no ``-m`` filter) passes every method into the
    cleanup, so every method dir gets reset — matching ``run`` config that
    declared them. ``figures/`` and the bench metadata sidecar survive."""
    root = tmp_path / "results_paper"
    methods = ("agentic_score", "agentic_cost", "random", "motpe")
    for method in methods:
        _seed_with_history(root / method / "seed_1")
    (root / "figures").mkdir()
    (root / "figures" / "Table_1.md").write_text("# previous\n")

    removed = _clear_output_root_for(root, list(methods))

    assert set(removed) == set(methods)
    for method in methods:
        assert not (root / method).exists()
    # figures/ survives the wipe (staging swap handles end-of-run replacement)
    assert (root / "figures" / "Table_1.md").read_text() == "# previous\n"


def test_checkpoint_dirs_wiped_alongside_parent(tmp_path: Path) -> None:
    """When ``agentic_score`` is in the wipe set, its ``agentic_score@10`` and
    ``agentic_score@20`` checkpoint siblings must also reset — they're
    derived from the parent's history and would otherwise drift."""
    root = tmp_path / "results_paper"
    for name in ("agentic_score", "agentic_score@10", "agentic_score@20", "random"):
        _seed_with_history(root / name / "seed_1")

    removed = _clear_output_root_for(root, ["agentic_score"])

    assert set(removed) == {"agentic_score", "agentic_score@10", "agentic_score@20"}
    assert not (root / "agentic_score").exists()
    assert not (root / "agentic_score@10").exists()
    assert not (root / "agentic_score@20").exists()
    # Sibling method untouched
    assert (root / "random" / "seed_1" / "history.jsonl").exists()


def test_handles_missing_targets_gracefully(tmp_path: Path) -> None:
    """First run: the targeted method dirs don't exist yet — no error, no
    spurious 'removed' entries."""
    root = tmp_path / "results_paper"
    root.mkdir()
    removed = _clear_output_root_for(root, ["agentic", "random"])
    assert removed == []


def test_handles_missing_output_root(tmp_path: Path) -> None:
    """Brand-new output_root — must not raise."""
    root = tmp_path / "never_existed"
    assert _clear_output_root_for(root, ["agentic"]) == []


def test_simulates_stale_figure_scenario(tmp_path: Path) -> None:
    """End-to-end of the bug that motivated the cleanup: a figure rendered
    against an earlier ``history.jsonl`` lingered when a later run rewrote
    the history but didn't re-hit the figure code path. With scoped
    cleanup, run 2 starts with a fresh agentic/ so the stale figure cannot
    survive."""
    root = tmp_path / "results_paper"
    seed_dir = root / "agentic" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "figures").mkdir()
    (seed_dir / "figures" / "score_per_trial.png").write_bytes(b"OLD-RUN-RENDER")
    (seed_dir / "history.jsonl").write_text(
        json.dumps({"trial_number": 1, "score": 0.825, "config": {}, "eval_usd": 0.1}) + "\n",
        encoding="utf-8",
    )

    _clear_output_root_for(root, ["agentic"])

    assert not (seed_dir / "figures").exists()
    assert not (seed_dir / "history.jsonl").exists()


def _seed_complete_method(seed_dir: Path, *, n_trials: int) -> None:
    """Lay down the on-disk markers a fully-finished (method, seed) leaves:
    a held-out ``benchmark_results.json`` plus ``optimizer_meta.json``."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "benchmark_results.json").write_text('{"accuracy": 0.8}', encoding="utf-8")
    (seed_dir / "optimizer_meta.json").write_text(
        json.dumps({"n_trials_completed": n_trials}), encoding="utf-8"
    )


def test_complete_when_holdout_present_and_no_checkpoints(tmp_path: Path) -> None:
    """A bare method (no checkpoints) is complete once its held-out result exists."""
    root = tmp_path / "results"
    _seed_complete_method(root / "random" / "seed_1", n_trials=2)
    assert _is_method_seed_complete(root, "random", "seed_1", []) is True


def test_incomplete_when_holdout_missing(tmp_path: Path) -> None:
    """Search may have finished but a crash before/within held-out leaves no
    benchmark_results.json — the pair must be re-entered, not skipped."""
    root = tmp_path / "results"
    seed_dir = root / "random" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "history.jsonl").write_text(
        json.dumps({"trial_number": 1, "config": {}, "answer_accuracy": 0.5, "eval_usd": 0.1}) + "\n",
        encoding="utf-8",
    )
    assert _is_method_seed_complete(root, "random", "seed_1", []) is False


def test_incomplete_when_a_declared_checkpoint_holdout_is_missing(tmp_path: Path) -> None:
    """Main held-out done but a declared @k checkpoint (k < n_trials) hasn't
    been evaluated yet → not complete."""
    root = tmp_path / "results"
    _seed_complete_method(root / "agentic_score" / "seed_1", n_trials=20)
    # @10 present, @5 missing
    _seed_complete_method(root / "agentic_score@10" / "seed_1", n_trials=10)
    assert _is_method_seed_complete(root, "agentic_score", "seed_1", [5, 10]) is False


def test_complete_when_all_declared_checkpoints_present(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _seed_complete_method(root / "agentic_score" / "seed_1", n_trials=20)
    _seed_complete_method(root / "agentic_score@5" / "seed_1", n_trials=5)
    _seed_complete_method(root / "agentic_score@10" / "seed_1", n_trials=10)
    assert _is_method_seed_complete(root, "agentic_score", "seed_1", [5, 10]) is True


def test_checkpoint_beyond_completed_trials_not_required(tmp_path: Path) -> None:
    """If only 8 trials completed (failures ate the rest), a declared @10
    checkpoint is never produced by _evaluate_checkpoints (k >= n_trials), so
    the guard must not demand its directory — else resume would loop forever."""
    root = tmp_path / "results"
    _seed_complete_method(root / "motpe" / "seed_1", n_trials=8)
    # @10 dir intentionally absent; n_trials_completed=8 so it's not required.
    assert _is_method_seed_complete(root, "motpe", "seed_1", [10]) is True


def test_methods_with_completed_results_detects_finished_work(tmp_path: Path) -> None:
    """The clean-wipe safety guard must see a method with a finished hold-out."""
    root = tmp_path / "results"
    _seed_complete_method(root / "random" / "seed_1", n_trials=2)
    # motpe present but only mid-search (history, no benchmark_results) → not "completed"
    (root / "motpe" / "seed_1").mkdir(parents=True)
    (root / "motpe" / "seed_1" / "history.jsonl").write_text("{}\n", encoding="utf-8")

    done = _methods_with_completed_results(root, ["random", "motpe", "agentic_score"])
    assert done == ["random"]


def test_methods_with_completed_results_empty_on_fresh_root(tmp_path: Path) -> None:
    root = tmp_path / "never"
    assert _methods_with_completed_results(root, ["random", "motpe"]) == []
