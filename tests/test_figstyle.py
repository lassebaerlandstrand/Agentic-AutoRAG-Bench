"""Tests for the shared figure-style helpers."""

from __future__ import annotations

from agentic_autorag_bench._figstyle import (
    color_for,
    display_label,
    fig_width_for,
)


def test_display_label_base_methods() -> None:
    assert display_label("agentic_score") == "Agentic (Ours)"
    assert display_label("random") == "Random"
    assert display_label("motpe") == "MO-TPE"
    assert display_label("motpe_warmstart") == "MO-TPE (KB warm-start)"
    assert display_label("agentic_nokb") == "Agentic (no KB)"


def test_display_label_checkpoints() -> None:
    """@k checkpoints render compactly; the '(Ours)' is implied by color."""
    assert display_label("agentic_score@10") == "Agentic@10"
    assert display_label("agentic_score@20") == "Agentic@20"


def test_display_label_unknown_falls_back_to_hyphenated() -> None:
    assert display_label("some_new_method") == "some-new-method"


def test_color_for_checkpoint_inherits_base() -> None:
    assert color_for("agentic_score@10") == color_for("agentic_score")


def test_color_for_unknown_is_fallback_gray() -> None:
    assert color_for("mystery") == "#888888"


def test_no_autorag_in_display_labels() -> None:
    """AutoRAG was cut — no method label should mention it."""
    for m in ("agentic_score", "agentic_cost", "random", "motpe", "motpe_warmstart", "agentic_score@10"):
        assert "autorag" not in display_label(m).lower()


def test_fig_width_grows_then_caps() -> None:
    narrow = fig_width_for(2)
    wide = fig_width_for(8)
    assert wide > narrow
    assert fig_width_for(100) <= 16.0
