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

# Standard matplotlib tab10 colours (dropping brown, which looks muddy, and
# olive, which reads too close to green). Assigned so the set that co-occurs in
# nearly every chart -- {agentic, MO-TPE cold, MO-TPE warm, random} -- lands on
# four well-separated, high-contrast hues: blue, green, purple, orange. Agentic
# (Ours) is the tab blue (the "darker blue"), shared by score/cost so it reads
# the same in every experiment. ``@k`` checkpoint variants inherit their base
# method's colour via ``color_for``.
METHOD_COLOR: dict[str, str] = {
    "agentic_score": "#1f77b4",  # blue — "Agentic (Ours)"
    "agentic_cost": "#1f77b4",  # same blue — consistent "Ours" across experiments
    "agentic_nokb": "#17becf",  # cyan (agentic ablation: KB off)
    "agentic_nodiag": "#d62728",  # red (agentic ablation: diagnosis off)
    "agentic_nokb_nodiag": "#e377c2",  # pink (maximally ablated agentic baseline)
    "random": "#ff7f0e",  # orange
    "motpe": "#2ca02c",  # green (MO-TPE cold)
    "motpe_warm": "#9467bd",  # purple (MO-TPE warm)
    "qlognehvi": "#7f7f7f",  # gray (GP-BO reference; cite-only)
}
_FALLBACK_COLOR = "#888888"  # kb_greedy and any unlisted method (neutral gray)

# Paper-facing display names (display-only). The internal method keys, config
# YAML, and on-disk ``results_*/<method>/`` directories are unchanged.
_DISPLAY_LABEL: dict[str, str] = {
    "agentic_score": "Agentic (Ours)",
    "agentic_cost": "Agentic (Ours)",
    "agentic_nokb": "Agentic (no KB)",
    "agentic_nodiag": "Agentic (no diagnosis)",
    "agentic_nokb_nodiag": "Agentic (no KB, no diag)",
    "random": "Random",
    "motpe": "MO-TPE",
    "motpe_warm": "MO-TPE (warm)",
    "qlognehvi": "GP-BO (qLogNEHVI)",
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
    ``motpe`` → "MO-TPE"; unknown names fall back to a hyphenated
    form so nothing renders as raw snake_case.
    """
    if "@" in method:
        base, k = method.split("@", 1)
        b = base_method(base)
        if b.startswith("agentic"):
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
            # Embed TrueType (Type-42) fonts in vector output instead of
            # matplotlib's default Type-3 bitmap fonts, which AAAI and most
            # venues reject. Applies to every PDF/PS/EPS figure this style drives.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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


def use_paper_serif() -> None:
    """Switch matplotlib to the paper's body serif (TeX Gyre Termes = newtxtext).

    Registers the Termes OTF files if present (they ship with TeX Live) and
    selects them for text and math; falls back to matplotlib's built-in STIX Two
    (also a Times clone) when they are not found. Paper figures are outlined
    before use, so this only affects how glyphs are drawn, not what is embedded.
    """
    import glob

    import matplotlib as mpl
    from matplotlib import font_manager as fm

    termes = glob.glob("/usr/share/texmf/**/texgyretermes-*.otf", recursive=True)
    if termes:
        for f in termes:
            fm.fontManager.addfont(f)
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["TeX Gyre Termes"],
                "mathtext.fontset": "custom",
                "mathtext.rm": "TeX Gyre Termes",
                "mathtext.it": "TeX Gyre Termes:italic",
                "mathtext.bf": "TeX Gyre Termes:bold",
            }
        )
    else:
        mpl.rcParams.update(
            {"font.family": "serif", "font.serif": ["STIX Two Text", "DejaVu Serif"], "mathtext.fontset": "stix"}
        )


def outline_pdf_fonts(path) -> None:
    """Convert all text in a PDF figure to vector outlines (no embedded fonts).

    Ghostscript's ``-dNoOutputFonts`` re-renders every glyph as a path, so the
    result carries neither Type-3 nor CID/Identity-H fonts, both of which AAAI's
    font rules flag (Type-3 is an explicit desk-reject). No-op for non-PDF paths
    or when Ghostscript is unavailable.
    """
    import shutil
    import subprocess
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() != ".pdf" or shutil.which("gs") is None:
        return
    tmp = p.parent / (p.stem + "__outlined.pdf")
    r = subprocess.run(
        ["gs", "-o", str(tmp), "-sDEVICE=pdfwrite", "-dNoOutputFonts", str(p)],
        capture_output=True,
    )
    if r.returncode == 0 and tmp.exists():
        tmp.replace(p)
    elif tmp.exists():
        tmp.unlink()


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
