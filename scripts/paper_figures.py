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
  index, one panel per dataset, one line per searching method, +/-SD band across
  the seeds. Vector PDF for the paper's ``fig:score-per-trial``.

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


def _cell(mean: float, sd: float, *, bold: bool) -> str:
    body = f"\\mathbf{{{mean:.3f}}}" if bold else f"{mean:.3f}"
    return f"${body}_{{\\pm {sd:.3f}}}$"


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
            rounded = {
                m: round(ms[0], 3)
                for m in TABLE_METHODS
                if (ms := mean_sd_of(stats, m, metric)) is not None
            }
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
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l ccc ccc ccc}",
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
        r"\end{tabular}%",
        r"}",
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
    fig, axes = plt.subplots(1, n, figsize=(9.6, 3.1), sharey=True)
    if n == 1:
        axes = [axes]

    handles: dict[str, object] = {}
    for ax, (_dir, header, results) in zip(axes, per_dataset_results):
        for method in FIGURE_METHODS:
            per_seed = _seed_scores(results, method)
            if not per_seed:
                continue
            color = color_for(method)
            if len(per_seed) >= 2:
                padded = _pad_nan(per_seed)
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(padded, axis=0)
                    std = np.nanstd(padded, axis=0)
                x = np.arange(1, padded.shape[1] + 1)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
                (line,) = ax.plot(x, mean, "o-", color=color, markersize=3, linewidth=1.4)
            else:
                scores = per_seed[0]
                (line,) = ax.plot(
                    np.arange(1, scores.size + 1), scores, "o-", color=color, markersize=3, linewidth=1.4
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


# ---------------------------------------------------------------- entry point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print resolved paths and exit")
    args = ap.parse_args()

    dataset_dirs = [(EXP1_ROOT / d, hdr) for d, hdr in DATASETS]
    table_path = OUT_DIR / "table1_answer_quality.tex"
    fig_path = OUT_DIR / "score_per_trial_3panel.pdf"

    if args.dry_run:
        print("experiment-1 root:", EXP1_ROOT)
        for d, hdr in dataset_dirs:
            print(f"  dataset {hdr:14s} -> {d}  (exists={d.is_dir()})")
        print("table  ->", table_path)
        print("figure ->", fig_path)
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


if __name__ == "__main__":
    main()
