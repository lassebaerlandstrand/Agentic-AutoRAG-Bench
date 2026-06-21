"""Tests for plots.py method-name discovery + ordering helpers.

After per-method @k checkpoints became first-class result directories,
matrix-level rendering had to stop iterating the hardcoded
``METHOD_ORDER`` and start discovering whatever's on disk. These tests
lock down the ordering rule: base methods in declared order, then each
base's ``@k`` variants by ascending k, then anything else.
"""

from __future__ import annotations

from pathlib import Path

from agentic_autorag_bench._figstyle import METHOD_COLOR
from agentic_autorag_bench._figstyle import color_for as _color_for
from agentic_autorag_bench.plots import (
    _discover_method_names,
    _is_sequential,
    _order_methods,
)


def test_order_methods_groups_checkpoints_after_parent() -> None:
    """`@10` and `@20` of a base method appear immediately after the base,
    sorted by k."""
    names = [
        "random",
        "agentic_score@20",
        "agentic_score",
        "bayesian",
        "agentic_score@10",
        "agentic_cost",
        "agentic_cost@10",
    ]
    assert _order_methods(names) == [
        "agentic_score",
        "agentic_score@10",
        "agentic_score@20",
        "agentic_cost",
        "agentic_cost@10",
        "random",
        "bayesian",
    ]


def test_order_methods_handles_only_checkpoints() -> None:
    """Edge case: the base method dir doesn't exist (yet) but @k variants do."""
    names = ["agentic_score@10", "agentic_score@20"]
    assert _order_methods(names) == ["agentic_score@10", "agentic_score@20"]


def test_order_methods_puts_unknown_names_last_alphabetical() -> None:
    """Defensive: stray dirs (e.g. user-created notes) don't crash, they
    sort alphabetically after the known methods."""
    names = ["random", "zsh_temp", "agentic_score", "alpha_test"]
    assert _order_methods(names) == [
        "agentic_score",
        "random",
        "alpha_test",
        "zsh_temp",
    ]


def test_color_for_inherits_parent_color() -> None:
    """``@k`` variants must paint with the base method's color so the legend
    stays consistent."""
    assert _color_for("agentic_score") == METHOD_COLOR["agentic_score"]
    assert _color_for("agentic_score@10") == METHOD_COLOR["agentic_score"]
    assert _color_for("agentic_cost@20") == METHOD_COLOR["agentic_cost"]


def test_color_for_unknown_method_gets_default() -> None:
    assert _color_for("totally_made_up") == "#888888"


def test_is_sequential_extends_to_checkpoints() -> None:
    """Checkpoint variants are sequential iff their parent base is."""
    assert _is_sequential("agentic_score")
    assert _is_sequential("agentic_score@10")
    assert _is_sequential("agentic_cost@20")
    assert _is_sequential("random")
    assert not _is_sequential("autorag_our_exam")  # not in SEQUENTIAL
    assert not _is_sequential("autorag_our_exam@10")


def test_discover_method_names_skips_non_method_dirs(tmp_path: Path) -> None:
    """``figures/``, ``.shared_cache/``, and the staging dirs sit next to
    method dirs but must NOT be treated as methods."""
    root = tmp_path / "results"
    for name in (
        "agentic_score",
        "agentic_score@10",
        "random",
        "figures",
        ".shared_cache",
        "_figures_staging",
        "_figures_previous",
    ):
        (root / name).mkdir(parents=True)

    discovered = _discover_method_names(root)
    assert discovered == ["agentic_score", "agentic_score@10", "random"]


def test_discover_method_names_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert _discover_method_names(tmp_path / "does_not_exist") == []
