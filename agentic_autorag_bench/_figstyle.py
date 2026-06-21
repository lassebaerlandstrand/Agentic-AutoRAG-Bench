"""Shared matplotlib styling for paper-ready benchmark figures.

One source of truth for method colors, paper-facing display names, and the
small layout helpers (value labels, outside legend, width scaling) so
``plots.py`` and ``analyze.py`` render consistently.

Display names are paper-facing ONLY — the internal method keys, the config
YAML, and the on-disk ``results_*/<method>/`` directories are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer

# One stable color per BASE method (matplotlib tab10); ``@k`` checkpoint
# variants inherit their base method's color via ``color_for``.
METHOD_COLOR: dict[str, str] = {
    "agentic_score": "#1f77b4",  # blue
    "agentic_cost": "#17becf",  # cyan
    "random": "#ff7f0e",  # orange
    "bayesian": "#2ca02c",  # green
}
_FALLBACK_COLOR = "#888888"

# Paper-facing display names (display-only). ``agentic_cost`` is mapped for
# when the cost-aware run lands; it does not appear in the current figure set.
_DISPLAY_LABEL: dict[str, str] = {
    "agentic_score": "Agentic (Ours)",
    "agentic_cost": "Agentic-Pareto (Ours)",
    "random": "Random",
    "bayesian": "Bayesian (TPE)",
}


def base_method(method: str) -> str:
    """Strip an ``@k`` checkpoint suffix to the base method name."""
    return method.split("@", 1)[0]


def color_for(method: str) -> str:
    """Stable color for a method; ``@k`` variants inherit the base color."""
    if method in METHOD_COLOR:
        return METHOD_COLOR[method]
    return METHOD_COLOR.get(base_method(method), _FALLBACK_COLOR)


def display_label(method: str) -> str:
    """Paper-facing label for a method name.

    ``agentic_score`` → "Agentic (Ours)"; ``agentic_score@10`` → "Agentic@10";
    ``bayesian`` → "Bayesian (TPE)"; unknown names fall back to a hyphenated
    form so nothing renders as raw snake_case.
    """
    if "@" in method:
        base, k = method.split("@", 1)
        if base_method(base).startswith("agentic"):
            return f"Agentic@{k}"
        return f"{display_label(base)}@{k}"
    if method in _DISPLAY_LABEL:
        return _DISPLAY_LABEL[method]
    return method.replace("_", "-")


def apply_paper_style() -> None:
    """Nudge matplotlib rcParams toward paper-legible defaults. Idempotent."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            # A touch more gap than matplotlib's default 6.0 so plain in-axes
            # titles breathe. Figures with an outside legend set their own (larger)
            # pad via ``legend_outside`` and are unaffected.
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            # Slightly larger than matplotlib's default 0.1 so the tight crop
            # leaves a little margin around the figure (notably above the title).
            "savefig.pad_inches": 0.15,
        }
    )


def fig_width_for(n_groups: int, *, base: float = 6.0, per_group: float = 1.1, cap: float = 16.0) -> float:
    """Figure width that grows with the number of x-axis groups so labels and
    bars don't crowd. Capped so the figure stays printable."""
    return min(cap, base + per_group * max(0, n_groups - 3))


def style_method_xticks(ax: Axes, methods: list[str]) -> None:
    """Set x-tick labels to display names, rotated so long ``@k`` names don't
    collide. Assumes ticks are already at ``range(len(methods))``."""
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [display_label(m) for m in methods],
        rotation=25,
        ha="right",
    )


def add_bar_value_labels(
    ax: Axes,
    bars: BarContainer,
    *,
    fmt: str = "{:.2f}",
    rotation: int = 90,
    fontsize: float = 6.5,
    y_offsets: list[float] | None = None,
) -> None:
    """Annotate each bar with its height just above the bar (or its error cap).

    ``y_offsets`` optionally lifts each label clear of an asymmetric error
    bar's upper whisker (pass the per-bar upper-error magnitudes). Zero-height
    bars (e.g. a not-yet-run method) are skipped to avoid clutter at the floor.
    Labels use ``clip_on=False`` so a tall bar's label is not chopped at the
    axis ceiling.
    """
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height <= 0:
            continue
        extra = y_offsets[i] if y_offsets is not None else 0.0
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height + extra),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
        )


# Title pad (points) that lifts an axes title clear of an outside legend so
# the two don't crowd. Tuned against the ~4-inch-tall paper figures.
_TITLE_PAD_ABOVE_LEGEND = 34


def legend_outside(ax: Axes, *, ncol: int, title: str | None = None, **kwargs) -> None:
    """Place the legend in a horizontal strip above the axes, clear of data.

    Pass ``title`` to also set an axes title lifted above the legend (the
    title pad and legend offset are tuned together here so callers don't have
    to hand-balance the two).
    """
    if title is not None:
        ax.set_title(title, pad=_TITLE_PAD_ABOVE_LEGEND)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncol,
        frameon=False,
        **kwargs,
    )
