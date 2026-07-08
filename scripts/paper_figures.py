#!/usr/bin/env python
"""Render the paper-ready Experiment-1 table and per-trial figure from data.

Experiment 1 spans three QA datasets (HotpotQA, MuSiQue, MultiHop-RAG), each
run for 3 full-rerun seeds. The in-run pipeline emits one single-dataset
``Table_1.md`` and ``score_per_trial.png`` per dataset (see ``analyze.py`` /
``plots.py``); the paper instead needs ONE cross-dataset table and ONE
cross-dataset figure. This script produces exactly those two artifacts, read
straight from the committed ``experiment-1/`` tree so the numbers cannot drift
from the source of truth:

* ``table1_answer_quality.tex`` -- held-out EM / F1 / Judge (mean +/- SD across
  seeds) for every method, three dataset column-groups, ``kb-greedy`` as a
  non-search reference row. This is a drop-in LaTeX ``table*`` for the paper's
  ``tab:holdout``. Regenerate and re-paste rather than hand-editing numbers.
* ``score_per_trial_3panel.pdf`` -- per-trial validation-exam accuracy vs. trial
  index, one panel per dataset, one line per searching method (mean across the
  seeds). Vector PDF for the paper's ``fig:score-per-trial``.

Aggregation reuses ``analyze.load_results`` + ``aggregate_by_method`` so the
mean / sample-SD is byte-identical to ``Table_1.md``; styling reuses
``_figstyle`` so colors match the other paper figures.

Run:  ``uv run python scripts/paper_figures.py``  (``--dry-run`` prints paths).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agentic_autorag_bench._figstyle import apply_paper_style, color_for, display_label  # noqa: E402
from agentic_autorag_bench.analyze import (  # noqa: E402
    aggregate_by_method,
    load_results,
    read_benchmark_pretty_name,
)

# ---------------------------------------------------------------- configuration

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP1_ROOT = REPO_ROOT / "experiment-1"
OUT_DIR = EXP1_ROOT / "figures_paper"

# On-disk dataset dir -> compact header name used in the table and figure.
DATASETS: list[tuple[str, str]] = [
    ("hotpot", "HotpotQA"),
    ("musique", "MuSiQue"),
    ("multihop", "MultiHop-RAG"),
]

# Searching methods, in the paper's row/line order (baselines first, ascending
# to the full agentic method). ``@k`` checkpoints are table rows only.
TABLE_METHODS = [
    "random",
    "motpe",
    "motpe_warm",
    "agentic_nokb_nodiag",
    "agentic_score@10",
    "agentic_score@20",
    "agentic_score",
]
# Non-search reference row(s), rendered below a rule and never bold-eligible.
REFERENCE_METHODS = ["kb_greedy"]
# Methods that get a per-trial line (exclude @k prefixes and the no-search ref).
FIGURE_METHODS = ["random", "motpe", "motpe_warm", "agentic_nokb_nodiag", "agentic_score"]
# Methods (left-to-right) shown as bars in the cost/embeddings figure: our full
# method first, then its @10/@20 checkpoints, then the maximally-ablated agentic
# baseline, then the statistical baselines. Keeping the @10/@20 checkpoints pairs
# cost with the sample-efficiency story. ``kb_greedy`` is excluded: it runs no
# search loop, so it has zero search cost and zero embedding tokens.
COST_METHODS = [
    "agentic_score",
    "agentic_score@10",
    "agentic_score@20",
    "agentic_nokb_nodiag",
    "random",
    "motpe",
    "motpe_warm",
]

# One hue per dataset for the grouped cost/embedding bars, in ``DATASETS`` order.
# These identify a DATASET (not a method), so they intentionally do not use
# ``color_for``; the legend and bar position disambiguate. Distinct tab10 hues.
_DATASET_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Short paper-facing labels (override _figstyle's longer defaults where needed).
PAPER_LABEL = {
    "random": "Random",
    "motpe": "MO-TPE",
    "motpe_warm": "MO-TPE (warm)",
    "agentic_nokb_nodiag": "Agentic (no KB/diag)",
    "agentic_score@10": "Agentic@10",
    "agentic_score@20": "Agentic@20",
    "agentic_score": "Agentic (Ours)",
    "kb_greedy": "KB Greedy",
}

METRICS = [("em", "EM"), ("f1", "F1"), ("judge", "Judge")]


def label_for(method: str) -> str:
    return PAPER_LABEL.get(method, display_label(method))


# ---------------------------------------------------------------- data loading


def load_dataset(dataset_dir: Path):
    """Return (results, stats, pretty_name) for one dataset dir."""
    results = load_results(dataset_dir)
    stats = aggregate_by_method(results)
    pretty = read_benchmark_pretty_name(dataset_dir)
    return results, stats, pretty


def mean_sd_of(stats: dict, method: str, metric: str) -> tuple[float, float] | None:
    """(mean, sd) for a method/metric, or None if the method is absent.

    ``aggregate_by_method`` stores each metric as the ``mean_sd`` triple
    ``(mean, mean - sd, mean + sd)``; recover the SD as ``hi - mean``.
    """
    if method not in stats:
        return None
    mean, _lo, hi = stats[method][metric]
    return mean, hi - mean


# ---------------------------------------------------------------- LaTeX table


def _nolead(x: float) -> str:
    """Format to 3 decimals and drop the leading zero (0.486 -> .486)."""
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def _cell(mean: float, sd: float, *, bold: bool) -> str:
    m = _nolead(mean)
    body = f"\\mathbf{{{m}}}" if bold else m
    return f"${body}_{{\\pm {_nolead(sd)}}}$"


def build_latex_table(per_dataset: list[tuple[str, str, dict]]) -> str:
    """Assemble the cross-dataset answer-quality ``table*``.

    ``per_dataset`` is a list of ``(dir_name, header_name, stats)`` in column
    order. Bold marks the per-column max over the searching rows (rounded to the
    printed 3 decimals so display-ties both bold); the reference row is excluded
    from the max and never bold.
    """
    # Bold set: (dataset_index, metric_key) -> set of methods holding the max.
    bold: dict[tuple[int, str], set[str]] = {}
    for di, (_dir, _hdr, stats) in enumerate(per_dataset):
        for metric, _lbl in METRICS:
            rounded = {m: round(ms[0], 3) for m in TABLE_METHODS if (ms := mean_sd_of(stats, m, metric)) is not None}
            if not rounded:
                continue
            top = max(rounded.values())
            bold[(di, metric)] = {m for m, v in rounded.items() if v == top}

    def row(method: str) -> str:
        cells = [label_for(method)]
        for di, (_dir, _hdr, stats) in enumerate(per_dataset):
            for metric, _lbl in METRICS:
                ms = mean_sd_of(stats, method, metric)
                if ms is None:
                    cells.append("--")
                    continue
                is_bold = method in bold.get((di, metric), set())
                cells.append(_cell(ms[0], ms[1], bold=is_bold))
        return " & ".join(cells) + r" \\"

    headers = [hdr for _dir, hdr, _stats in per_dataset]
    top_groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{h}}}" for h in headers)
    cmids = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(per_dataset)))
    sub = "Method & " + " & ".join(["EM & F1 & Judge"] * len(per_dataset))

    lines = [
        "% GENERATED by scripts/paper_figures.py -- do not hand-edit numbers; regenerate and re-paste.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}l ccc ccc ccc@{}}",
        r"\toprule",
        "& " + top_groups + r" \\",
        cmids,
        sub + r" \\",
        r"\midrule",
    ]
    lines += [row(m) for m in TABLE_METHODS]
    lines.append(r"\midrule")
    lines += [row(m) for m in REFERENCE_METHODS]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{CAPTION -- see main.tex.}",
        r"\label{tab:holdout}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figure


def _pad_nan(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack ragged 1-D arrays into a 2-D array, NaN-padding short rows so an
    aborted seed does not flatten the tail (mirrors ``plots._pad_nan``)."""
    width = max(a.size for a in arrays)
    out = np.full((len(arrays), width), np.nan)
    for i, a in enumerate(arrays):
        out[i, : a.size] = a
    return out


def _seed_scores(results: list, method: str) -> list[np.ndarray]:
    """Per-seed arrays of per-trial validation-exam accuracy for one method."""
    out: list[np.ndarray] = []
    for r in results:
        if r.method != method or not r.history:
            continue
        hist = sorted(r.history, key=lambda e: int(e.get("trial_number", 0)))
        out.append(np.array([float(e.get("answer_accuracy", 0.0)) for e in hist]))
    return out


def build_figure(per_dataset_results: list[tuple[str, str, list]], out_path: Path) -> None:
    apply_paper_style()
    n = len(per_dataset_results)
    # Full-width figure* scaled down to ~\textwidth: bump fonts above the paper
    # default so the per-panel titles and the legend stay legible after the
    # downscale (mirrors build_cost_figure). Scoped so other figures are unaffected.
    font_overrides = {
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
    }
    with plt.rc_context(font_overrides):
        fig, axes = plt.subplots(1, n, figsize=(9.6, 3.1), sharey=True)
        if n == 1:
            axes = [axes]

        handles: dict[str, object] = {}
        for ax, (_dir, header, results) in zip(axes, per_dataset_results, strict=True):
            for method in FIGURE_METHODS:
                per_seed = _seed_scores(results, method)
                if not per_seed:
                    continue
                color = color_for(method)
                if len(per_seed) >= 2:
                    padded = _pad_nan(per_seed)
                    with np.errstate(invalid="ignore"):
                        mean = np.nanmean(padded, axis=0)
                    x = np.arange(1, padded.shape[1] + 1)
                    (line,) = ax.plot(x, mean, "o-", color=color, markersize=1.4, linewidth=1.4)
                else:
                    scores = per_seed[0]
                    (line,) = ax.plot(
                        np.arange(1, scores.size + 1), scores, "o-", color=color, markersize=1.4, linewidth=1.4
                    )
                handles.setdefault(method, line)
            ax.set_title(header)
            ax.set_xlabel("Trial")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[0].set_ylabel("Validation-exam accuracy")

        ordered = [m for m in FIGURE_METHODS if m in handles]
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


# ---------------------------------------------------------------- cost figure


def _minmax(vals: list[float]) -> tuple[float, float, float]:
    """``(mean, lower, upper)`` where lower/upper are the min-max whisker
    magnitudes about the mean (``mean - min`` and ``max - mean``)."""
    mean = float(np.mean(vals))
    return mean, mean - min(vals), max(vals) - mean


def _cost_totals(stats: dict, method: str) -> tuple[float, float, tuple[float, float, float]]:
    """``(opt_mean, trial_mean, (total_mean, lo, hi))`` for one method.

    The whisker is the min-max of the per-seed *totals* (optimizer + trial),
    which is the right quantity for a stacked bar's error mark. Absent method →
    all zeros so the bar renders empty rather than erroring.
    """
    if method not in stats:
        return 0.0, 0.0, (0.0, 0.0, 0.0)
    s = stats[method]
    seed_totals = [o + t for o, t in zip(s["optimizer_usd_list"], s["trial_usd_list"], strict=True)]
    return s["optimizer_usd_mean"], s["trial_usd_mean"], _minmax(seed_totals)


def _embed_millions(stats: dict, method: str) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` embedding tokens in millions, min-max across seeds."""
    if method not in stats:
        return 0.0, 0.0, 0.0
    return _minmax([e / 1e6 for e in stats[method]["embedding_tokens_list"]])


_OPT_GRAY = "#9e9e9e"  # neutral hue for the optimizer-reasoning segment (gray split style)


def build_cost_figure(per_dataset: list[tuple[str, str, dict]], out_path: Path, *, split_style: str = "gray") -> None:
    """Cross-dataset search-cost + embedding-footprint figure (Exp-1).

    Two horizontal-bar panels sharing one y-axis of methods (top to bottom: our
    full method, its @10/@20 checkpoints, the maximally-ablated agentic baseline,
    then the statistical baselines). Left panel: per-seed search cost (USD), each
    bar split into optimizer reasoning (proposer + diagnoser calls) and trial
    evaluation (RAG generation + judge); the statistical baselines have no
    optimizer-reasoning cost. Right panel: cache-aware embedding tokens (millions),
    a single solid bar. Within each method the three datasets are color-coded bars,
    each with a min-max-across-seeds whisker on the total and its value printed at
    the bar's end. Horizontal bars keep the long method names unrotated.

    ``split_style`` picks how the optimizer segment is drawn: ``"gray"`` paints it
    a single neutral gray (dataset hue stays on the trial segment); ``"hatch"``
    keeps the dataset hue and marks the optimizer segment with a hatch texture.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_paper_style()
    methods = COST_METHODS
    labels = [label_for(m) for m in methods]
    nm = len(methods)
    n = len(per_dataset)
    # Method 0 sits at the top of the axis; the datasets are stacked as sub-bars
    # within each method's slot (dataset 0 highest, matching the legend order).
    centers = np.arange(nm)[::-1].astype(float)
    thick = 0.24
    step = 0.27
    offsets = [((n - 1) / 2 - j) * step for j in range(n)]

    # Full-width ``figure*`` scaled down to ~\textwidth: set fonts larger than the
    # paper default so they stay legible after the downscale. Scoped to this figure
    # so ``score_per_trial`` is unaffected.
    font_overrides = {
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13.5,
    }
    value_fontsize = 12  # number at each bar's end

    def _annotate_ends(ax, y, ends, hi, fmt):
        for yi, ev, hv in zip(y, ends, hi, strict=True):
            if ev > 0:
                ax.annotate(fmt(ev), (ev + hv + pad, yi), va="center", ha="left", fontsize=value_fontsize)

    with plt.rc_context(font_overrides):
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 6.4), sharey=True)

        # ---- Left panel: search cost, split optimizer reasoning vs trial eval ----
        ax = axs[0]
        cost = []
        for _dir, _header, stats in per_dataset:
            rows = [_cost_totals(stats, m) for m in methods]
            cost.append(
                (
                    np.array([r[0] for r in rows]),  # opt_mean
                    np.array([r[1] for r in rows]),  # trial_mean
                    np.array([r[2][0] for r in rows]),  # total_mean (= opt + trial)
                    np.array([r[2][1] for r in rows]),  # lo
                    np.array([r[2][2] for r in rows]),  # hi
                )
            )
        xmax = max(float((total + hi).max()) for _o, _t, total, _lo, hi in cost)
        pad = 0.012 * xmax
        for j, (opt, trial, total, lo, hi) in enumerate(cost):
            color = _DATASET_COLORS[j % len(_DATASET_COLORS)]
            y = centers + offsets[j]
            # Trial-evaluation segment in the dataset hue, stacked on the optimizer foot.
            ax.barh(y, trial, height=thick, left=opt, color=color, zorder=3)
            # Optimizer-reasoning foot: neutral gray, or the dataset hue with a hatch.
            if split_style == "hatch":
                ax.barh(y, opt, height=thick, color=color, hatch="////", edgecolor="white", linewidth=0.0, zorder=3)
            else:
                ax.barh(y, opt, height=thick, color=_OPT_GRAY, zorder=3)
            ax.errorbar(total, y, xerr=[lo, hi], fmt="none", ecolor="#333", elinewidth=0.9, capsize=2.5, zorder=4)
            _annotate_ends(ax, y, total, hi, lambda v: f"${v:.2f}")
        ax.set_title("Search cost (USD)")
        ax.set_xlim(0, xmax * 1.20)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        # ---- Right panel: embedding tokens, single solid bar per dataset ----
        ax = axs[1]
        emb = []
        for _dir, _header, stats in per_dataset:
            rows = [_embed_millions(stats, m) for m in methods]
            emb.append(
                (
                    np.array([r[0] for r in rows]),
                    np.array([r[1] for r in rows]),
                    np.array([r[2] for r in rows]),
                )
            )
        xmax = max(float((mean + hi).max()) for mean, _lo, hi in emb)
        pad = 0.012 * xmax
        for j, (mean, lo, hi) in enumerate(emb):
            color = _DATASET_COLORS[j % len(_DATASET_COLORS)]
            y = centers + offsets[j]
            ax.barh(
                y,
                mean,
                height=thick,
                color=color,
                xerr=[lo, hi],
                capsize=2.5,
                error_kw={"ecolor": "#333", "lw": 0.9},
                zorder=3,
            )
            _annotate_ends(ax, y, mean, hi, lambda v: f"{v:.0f}")
        ax.set_title("Embedding tokens (M)")
        ax.set_xlim(0, xmax * 1.20)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        axs[0].set_yticks(centers)
        axs[0].set_yticklabels(labels)
        axs[0].set_ylim(centers.min() - 0.6, centers.max() + 0.6)

        handles = [
            Patch(color=_DATASET_COLORS[j % len(_DATASET_COLORS)], label=hdr)
            for j, (_dir, hdr, _stats) in enumerate(per_dataset)
        ]
        if split_style == "hatch":
            opt_handle = Patch(facecolor=_OPT_GRAY, edgecolor="white", hatch="////", label="Optimizer reasoning")
        else:
            opt_handle = Patch(color=_OPT_GRAY, label="Optimizer reasoning")
        handles.append(opt_handle)
        handles.append(Line2D([], [], color="#333", lw=0.9, marker="|", markersize=9, label="Min–max across seeds"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.01))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------- entry point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print resolved paths and exit")
    args = ap.parse_args()

    dataset_dirs = [(EXP1_ROOT / d, hdr) for d, hdr in DATASETS]
    table_path = OUT_DIR / "table1_answer_quality.tex"
    fig_path = OUT_DIR / "score_per_trial_3panel.pdf"
    # Canonical cost figure uses the hatched optimizer/trial split; the gray
    # split is kept as an alternate for comparison.
    cost_pdf = OUT_DIR / "cost_and_embeddings.pdf"
    cost_png = OUT_DIR / "cost_and_embeddings.png"
    cost_gray_pdf = OUT_DIR / "cost_and_embeddings_gray.pdf"
    cost_gray_png = OUT_DIR / "cost_and_embeddings_gray.png"

    if args.dry_run:
        print("experiment-1 root:", EXP1_ROOT)
        for d, hdr in dataset_dirs:
            print(f"  dataset {hdr:14s} -> {d}  (exists={d.is_dir()})")
        print("table  ->", table_path)
        print("figure ->", fig_path)
        print("cost   ->", cost_pdf, "(+ .png)  [hatched, canonical]")
        print("cost   ->", cost_gray_pdf, "(+ .png)  [gray alt]")
        return

    missing = [str(d) for d, _ in dataset_dirs if not d.is_dir()]
    if missing:
        raise SystemExit("missing dataset dirs:\n  " + "\n  ".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_dataset_table: list[tuple[str, str, dict]] = []
    per_dataset_results: list[tuple[str, str, list]] = []
    for dataset_dir, header in dataset_dirs:
        results, stats, _pretty = load_dataset(dataset_dir)
        per_dataset_table.append((dataset_dir.name, header, stats))
        per_dataset_results.append((dataset_dir.name, header, results))

    table_path.write_text(build_latex_table(per_dataset_table), encoding="utf-8")
    print("wrote", table_path)

    build_figure(per_dataset_results, fig_path)
    print("wrote", fig_path)

    build_cost_figure(per_dataset_table, cost_pdf, split_style="hatch")
    build_cost_figure(per_dataset_table, cost_png, split_style="hatch")
    build_cost_figure(per_dataset_table, cost_gray_pdf, split_style="gray")
    build_cost_figure(per_dataset_table, cost_gray_png, split_style="gray")
    print("wrote", cost_png, "and", cost_gray_png, "(+ pdfs)")


if __name__ == "__main__":
    main()
