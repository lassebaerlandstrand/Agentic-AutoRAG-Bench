"""Tests for bench-config parsing of the new ``checkpoints`` block plus
the matrix-level figure staging swap.

The bench writes matrix figures to ``output_root/_figures_staging/`` and
atomically swaps that into ``output_root/figures/`` at the very end of a
run. Until the swap, the previous run's figures stay readable at the
normal path. ``_clear_output_root_for`` no longer wipes ``figures/`` at
start-of-run for this reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_autorag_bench.run import (
    BenchConfig,
    _swap_in_staged_figures,
)

# ---- BenchConfig.checkpoints parsing -----------------------------------


def test_bench_checkpoints_parse_drops_out_of_range(tmp_path: Path) -> None:
    """Checkpoints in the YAML that are ``>= max_trials`` are silently
    pruned: the bare ``<method>/`` directory IS the full-budget result, so
    a separate ``@max_trials`` entry would be redundant."""
    cfg = tmp_path / "hotpot_paper.yaml"
    project = tmp_path / "project.yaml"
    project.write_text("dummy: true")
    cfg.write_text(
        "project_config: ./project.yaml\n"
        "methods: [random, agentic_score]\n"
        "seeds: [1]\n"
        "budget: {max_trials: 40}\n"
        "checkpoints:\n"
        "  agentic_score: [10, 20, 40, 60]\n"
        "benchmark: {name: hotpot_qa, split: validation, sample_size: 10, output_dir: ./bench}\n"
        "hold_out: {limit: 10, judge_model: null, concurrency: 1}\n"
        "output_root: ./results\n",
        encoding="utf-8",
    )
    bench = BenchConfig.load(cfg)
    assert bench.checkpoints == {"agentic_score": [10, 20]}


def test_bench_checkpoints_parse_rejects_unknown_method(tmp_path: Path) -> None:
    """A checkpoints block referencing a non-existent method should fail
    loudly at config-load time, not silently noop at runtime."""
    cfg = tmp_path / "hotpot_paper.yaml"
    project = tmp_path / "project.yaml"
    project.write_text("dummy: true")
    cfg.write_text(
        "project_config: ./project.yaml\n"
        "methods: [random]\n"
        "seeds: [1]\n"
        "budget: {max_trials: 40}\n"
        "checkpoints:\n"
        "  agentic_not_a_method: [10]\n"
        "benchmark: {name: hotpot_qa, split: validation, sample_size: 10, output_dir: ./bench}\n"
        "hold_out: {limit: 10, judge_model: null, concurrency: 1}\n"
        "output_root: ./results\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agentic_not_a_method"):
        BenchConfig.load(cfg)


def test_bench_checkpoints_default_is_empty(tmp_path: Path) -> None:
    """A bench config without a checkpoints block should leave the dict
    empty — no @k dirs get written and the matrix is just the natural
    full-budget run."""
    cfg = tmp_path / "hotpot_paper.yaml"
    project = tmp_path / "project.yaml"
    project.write_text("dummy: true")
    cfg.write_text(
        "project_config: ./project.yaml\n"
        "methods: [random]\n"
        "seeds: [1]\n"
        "budget: {max_trials: 40}\n"
        "benchmark: {name: hotpot_qa, split: validation, sample_size: 10, output_dir: ./bench}\n"
        "hold_out: {limit: 10, judge_model: null, concurrency: 1}\n"
        "output_root: ./results\n",
        encoding="utf-8",
    )
    bench = BenchConfig.load(cfg)
    assert bench.checkpoints == {}


# ---- _swap_in_staged_figures atomic replacement ------------------------


def test_swap_replaces_old_figures_with_staging(tmp_path: Path) -> None:
    """Happy path: ``figures/`` exists, ``_figures_staging/`` exists with new
    content; after swap, ``figures/`` reflects the staging contents and
    staging is gone."""
    root = tmp_path / "results_hotpot"
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "old_figure.png").write_bytes(b"OLD")
    (root / "_figures_staging").mkdir()
    (root / "_figures_staging" / "new_figure.png").write_bytes(b"NEW")

    _swap_in_staged_figures(root)

    assert (root / "figures" / "new_figure.png").read_bytes() == b"NEW"
    assert not (root / "figures" / "old_figure.png").exists()
    assert not (root / "_figures_staging").exists()


def test_swap_creates_figures_when_none_existed(tmp_path: Path) -> None:
    """First-ever run: ``figures/`` doesn't exist yet. The staging swap
    must still install the new figures."""
    root = tmp_path / "results_hotpot"
    (root / "_figures_staging").mkdir(parents=True)
    (root / "_figures_staging" / "fresh.png").write_bytes(b"FRESH")

    _swap_in_staged_figures(root)

    assert (root / "figures" / "fresh.png").read_bytes() == b"FRESH"
    assert not (root / "_figures_staging").exists()


def test_swap_is_noop_when_no_staging(tmp_path: Path) -> None:
    """If matrix rendering produced no staging dir (e.g. partial run with
    no completed methods), the swap silently leaves ``figures/`` alone."""
    root = tmp_path / "results_hotpot"
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "preserved.png").write_bytes(b"KEEP ME")

    _swap_in_staged_figures(root)

    assert (root / "figures" / "preserved.png").read_bytes() == b"KEEP ME"


def test_swap_recovers_from_leftover_previous_backup(tmp_path: Path) -> None:
    """A prior crash mid-swap could leave a stale ``_figures_previous/``
    directory in place. The next swap should still succeed (overwrites
    the stale backup)."""
    root = tmp_path / "results_hotpot"
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "current.png").write_bytes(b"CURRENT")
    (root / "_figures_previous").mkdir()
    (root / "_figures_previous" / "stale.png").write_bytes(b"STALE")
    (root / "_figures_staging").mkdir()
    (root / "_figures_staging" / "new.png").write_bytes(b"NEW")

    _swap_in_staged_figures(root)

    assert (root / "figures" / "new.png").read_bytes() == b"NEW"
    assert not (root / "_figures_previous").exists()
    assert not (root / "_figures_staging").exists()
