#!/usr/bin/env python
"""Render the cross-dataset *best-so-far* figure (Exp-1), a companion to
``score_per_trial_3panel`` from ``paper_figures.py``.

``score_per_trial`` plots the raw per-trial validation-exam accuracy (noisy,
one zig-zag line per method). This script plots the *best-so-far* incumbent --
``np.maximum.accumulate`` of the same per-trial accuracy -- which is the curve
that tells the sample-efficiency story (``agentic@10`` >= baselines@30):
monotone, so far easier to read across three small panels.

Layout matches ``score_per_trial`` exactly (1x3 panels, one per dataset, shared
0-1 y-axis, one line per searching method, shared legend below) so the two
figures read as a pair, with a ``mean +/- 1 SD across seeds`` band per method
(same +/- SD reported in Table 1). ``kb_greedy`` runs no search, so it is drawn
as a dashed full-width reference line at its validation accuracy -- the
no-search level the searching methods have to climb above. The single-panel
per-dataset ``best_so_far.png`` from ``plots.py`` is unchanged; this is the
paper's one-figure cross-dataset version.

Data/aggregation reuse ``analyze.load_results`` and the per-trial
``answer_accuracy`` field so the curves cannot drift from ``Table_1`` or from
``score_per_trial``; styling/colors/labels reuse ``paper_figures`` so the two
figures are visually consistent.

Run:  ``uv run python scripts/best_so_far_figure.py``  (writes the PDF + PNG to
``experiment-1/figures_paper/``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agentic_autorag_bench._figstyle import apply_paper_style, color_for  # noqa: E402
from agentic_autorag_bench.analyze import load_results  # noqa: E402

# Reuse the paper figure's dataset order, line set, and labels verbatim so this
# figure and ``score_per_trial`` stay a matched pair. ``paper_figures`` is a
# sibling module in scripts/ (on sys.path[0] when this file is run directly).
from paper_figures import DATASETS, FIGURE_METHODS, REFERENCE_METHODS, label_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP1_ROOT = REPO_ROOT / "experiment-1"
OUT_DIR = EXP1_ROOT / "figures_paper"


# ---------------------------------------------------------------- data helpers


def _seed_best(results: list, method: str) -> list[np.ndarray]:
    """Per-seed best-so-far curves (cumulative max of per-trial accuracy)."""
    out: list[np.ndarray] = []
    for r in results:
        if r.method != method or not r.history:
            continue
        hist = sorted(r.history, key=lambda e: int(e.get("trial_number", 0)))
        scores = np.array([float(e.get("answer_accuracy", 0.0)) for e in hist])
        out.append(np.maximum.accumulate(scores))
    return out


def _pad_edge(curves: list[np.ndarray]) -> np.ndarray:
    """Edge-pad ragged best-so-far curves to the max length.

    A seed that stopped at trial T<max keeps its final incumbent past T (the
    best-so-far cannot regress), so edge replication -- not NaN -- keeps the
    mean honest. Mirrors ``plots._pad_edge``.
    """
    width = max(c.size for c in curves)
    return np.array([np.pad(c, (0, width - c.size), mode="edge") for c in curves])


def _reference_values(results: list, method: str) -> np.ndarray:
    """Per-seed validation accuracy for a no-search reference method.

    ``kb_greedy`` evaluates a single config once (scored on the validation exam
    by ``scripts/score_kb_greedy_validation.py``), so its history holds one
    entry per seed: a flat level to beat, not a trajectory.
    """
    return np.array(
        [float(r.history[-1].get("answer_accuracy", 0.0)) for r in results if r.method == method and r.history],
        dtype=float,
    )


# ---------------------------------------------------------------- figure


def build_best_so_far_figure(
    per_dataset_results: list[tuple[str, str, list]], out_path: Path, *, show_ref_band: bool = True
) -> None:
    """3-panel best-so-far figure (mean +/- 1 SD), mirroring ``build_figure``."""
    apply_paper_style()
    n = len(per_dataset_results)
    font_overrides = {
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11.5,
        # The figure is saved with ``savefig.bbox="tight"``, so the widest artist
        # sets the file's width. At 12pt the one-row legend is wider than the
        # panels and the crop leaves dead space either side of them; at 10pt the
        # panels become the widest artist, so they span the file edge to edge.
        # The narrower file is magnified more at \textwidth, which cancels out:
        # the legend still renders at ~12pt on the page, the axis labels larger.
        "legend.fontsize": 10,
    }
    with plt.rc_context(font_overrides):
        fig, axes = plt.subplots(1, n, figsize=(9.6, 3.1), sharey=True)
        if n == 1:
            axes = [axes]

        handles: dict[str, object] = {}
        for ax, (_dir, header, results) in zip(axes, per_dataset_results, strict=True):
            for method in FIGURE_METHODS:
                per_seed = _seed_best(results, method)
                if not per_seed:
                    continue
                color = color_for(method)
                if len(per_seed) >= 2:
                    padded = _pad_edge(per_seed)
                    mean = padded.mean(axis=0)
                    sd = padded.std(axis=0, ddof=1)  # sample SD, matches Table 1
                    x = np.arange(1, padded.shape[1] + 1)
                    ax.fill_between(x, mean - sd, mean + sd, alpha=0.13, color=color, lw=0)
                    (line,) = ax.plot(x, mean, "-", color=color, linewidth=1.6)
                else:
                    best = per_seed[0]
                    (line,) = ax.plot(np.arange(1, best.size + 1), best, "-", color=color, linewidth=1.6)
                handles.setdefault(method, line)
            # No-search references have no trajectory: draw the level to beat as
            # a dashed full-width line, band first so the line stays on top.
            for method in REFERENCE_METHODS:
                ref = _reference_values(results, method)
                if not ref.size:
                    continue
                color = color_for(method)
                mean = float(ref.mean())
                if show_ref_band and ref.size >= 2:
                    sd = float(ref.std(ddof=1))  # sample SD, matches the curves
                    ax.axhspan(mean - sd, mean + sd, color=color, alpha=0.13, lw=0)
                handles.setdefault(method, ax.axhline(mean, linestyle="--", color=color, linewidth=1.6))
            ax.set_title(header)
            ax.set_xlabel("Trial")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[0].set_ylabel("Best-so-far accuracy")

        ordered = [m for m in (*FIGURE_METHODS, *REFERENCE_METHODS) if m in handles]
        fig.legend(
            [handles[m] for m in ordered],
            [label_for(m) for m in ordered],
            loc="lower center",
            ncol=len(ordered),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------- entry point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-ref-band", action="store_true", help="Draw the KB Greedy reference line without its +/- SD band"
    )
    args = parser.parse_args()

    dataset_dirs = [(EXP1_ROOT / d, hdr) for d, hdr in DATASETS]
    missing = [str(d) for d, _ in dataset_dirs if not d.is_dir()]
    if missing:
        raise SystemExit("missing dataset dirs:\n  " + "\n  ".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_dataset_results = [(d.name, hdr, load_results(d)) for d, hdr in dataset_dirs]

    for out in (OUT_DIR / "best_so_far_3panel.pdf", OUT_DIR / "best_so_far_3panel.png"):
        build_best_so_far_figure(per_dataset_results, out, show_ref_band=not args.no_ref_band)
        print("wrote", out)


if __name__ == "__main__":
    main()
