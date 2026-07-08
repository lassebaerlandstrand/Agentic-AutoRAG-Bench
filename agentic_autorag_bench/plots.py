"""Auto-generated benchmark figures.

Called from ``run.py`` at three nesting levels so the user sees plots as soon
as enough data exists to draw them:

- ``make_seed_figures(seed_dir)`` — after one ``(method, seed)`` run finishes
  its hold-out scoring. Writes ``<seed_dir>/figures/``.
- ``make_method_figures(method_dir)`` — after the seed loop for one method
  closes. Pools all seeds of that method. Writes ``<method_dir>/figures/``.
- ``make_matrix_figures(output_root)`` — after the full matrix completes
  (post union-exclusion). Writes ``<output_root>/figures/``.

Every function is idempotent and best-effort: a corrupt history.jsonl in one
seed will not abort the matrix. The standalone ``analyze`` CLI calls
``make_matrix_figures(...)`` on a committed results tree to regenerate the
matrix-level summary without re-running the matrix.

Figure file names are stable across all three levels (``score_per_trial.png``,
``best_so_far.png``, ``holdout_metrics.png``, ``efficiency.png``,
``cost_breakdown.png``, ``cost_per_trial.png``) so a reader can navigate
``<output_root>/`` → ``<output_root>/<method>/`` → ``<output_root>/<method>/<seed>/``
and recognise the same view zoomed at each level.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench._figstyle import (
    apply_paper_style,
    display_label,
    fig_width_for,
    legend_outside,
    style_method_xticks,
)
from agentic_autorag_bench._figstyle import (
    color_for as _color_for,
)

logger = logging.getLogger("agentic_autorag_bench.run")

# Stable display order for BASE methods. Mirrors ``analyze.METHOD_ORDER``;
# the two must agree. ``@k`` checkpoint variants (e.g. ``agentic_score@10``)
# are interleaved at render time by ``_discover_method_names`` so they
# appear immediately after their parent base method, in ascending k order.
METHOD_ORDER = [
    "agentic_score",
    "agentic_cost",
    "agentic_nokb",
    "agentic_nodiag",
    "agentic_nokb_nodiag",
    "motpe",
    "motpe_warm",
    "qlognehvi",
    "random",
]

# Methods whose per-trial trajectory is meaningful. ``@k`` checkpoint
# variants inherit sequential-ness from their parent (they're a strict
# prefix of the parent's history) via ``_is_sequential``.
SEQUENTIAL = {
    "agentic_score",
    "agentic_cost",
    "agentic_nokb",
    "agentic_nodiag",
    "agentic_nokb_nodiag",
    "motpe",
    "motpe_warm",
    "qlognehvi",
    "random",
}

# Directory names that live next to method dirs under ``output_root`` but are
# not method results. ``_seed_dirs`` and ``make_matrix_figures`` skip these.
_NON_METHOD_DIRS = {"figures", ".shared_cache", "_figures_staging", "_figures_previous"}


def _is_sequential(method: str) -> bool:
    """Whether ``method`` has a per-trial trajectory worth plotting.

    Treats ``<base>@<k>`` as sequential iff ``<base>`` is sequential.
    """
    return method in SEQUENTIAL or method.split("@", 1)[0] in SEQUENTIAL


def _order_methods(method_names) -> list[str]:
    """Sort an iterable of method names by ``METHOD_ORDER`` (base first, then
    each base's ``@k`` variants by ascending k). Names not in ``METHOD_ORDER``
    and not ``<base>@<k>`` of one come last, alphabetically."""
    names = set(method_names)
    ordered: list[str] = []
    seen: set[str] = set()
    for base in METHOD_ORDER:
        if base in names:
            ordered.append(base)
            seen.add(base)
        prefix = f"{base}@"
        ks: list[tuple[int, str]] = []
        for n in names:
            if n.startswith(prefix):
                suffix = n[len(prefix) :]
                if suffix.isdigit():
                    ks.append((int(suffix), n))
        for _, n in sorted(ks):
            ordered.append(n)
            seen.add(n)
    for n in sorted(names - seen):
        ordered.append(n)
    return ordered


def _discover_method_names(output_root: Path) -> list[str]:
    """All method dir names under ``output_root``, ordered per
    ``_order_methods``. Empty if ``output_root`` doesn't exist."""
    if not output_root.exists():
        return []
    on_disk = {d.name for d in output_root.iterdir() if d.is_dir() and d.name not in _NON_METHOD_DIRS}
    return _order_methods(on_disk)


# ---------------------------------------------------------------- I/O helpers


def _entry_score(e: dict) -> float:
    return float(e.get("answer_accuracy", 0.0))


def _entry_eval_usd(e: dict) -> float:
    """Per-trial evaluation cost across both history schemas.

    The framework writes ``total_llm_cost_usd`` into agentic's
    ``history.jsonl``; the bench's reduced ``HistoryEntry`` writes
    ``eval_usd`` for random/motpe. Both name the same quantity.
    """
    if "eval_usd" in e:
        return float(e["eval_usd"])
    return float(e.get("total_llm_cost_usd", 0.0))


def _read_history(seed_dir: Path) -> list[dict]:
    path = RunLayout(base=seed_dir).history
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        # A kill mid-write can truncate the last line; keep the good prefix so
        # trajectory figures still render rather than aborting the whole plot.
        logger.warning("Truncated/corrupt history at %s; using %d good line(s)", path, len(out), exc_info=True)
    out.sort(key=lambda e: int(e.get("trial_number", 0)))
    return out


def _read_benchmark(seed_dir: Path) -> dict | None:
    path = seed_dir / "benchmark_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Unreadable/corrupt benchmark_results at %s; skipping", path, exc_info=True)
        return None


def _seed_dirs(method_dir: Path) -> list[Path]:
    return sorted(p for p in method_dir.iterdir() if p.is_dir() and p.name not in _NON_METHOD_DIRS)


def _method_dirs(output_root: Path) -> list[Path]:
    return sorted(p for p in output_root.iterdir() if p.is_dir() and p.name not in _NON_METHOD_DIRS)


def _benchmark_cost_per_question(benchmark: dict) -> float | None:
    """Mean LLM cost per question on hold-out, after union-exclusion.

    Sums ``input_tokens × in_price + completion_tokens × out_price`` per row
    (already computed by LiteLLM and persisted as ``llm_cost_usd``), means
    across the non-excluded rows. This is the per-query deploy cost of the
    winning pipeline — the right denominator for a cost–quality Pareto,
    since search-time cost (covered by ``efficiency.png``) is what the
    researcher pays once, while cost-per-question is what production pays
    forever.

    Returns None if no row carries a cost — caller skips that method.
    """
    excluded = set(benchmark.get("excluded_question_ids") or [])
    costs: list[float] = []
    for r in benchmark.get("per_question", []):
        if r.get("id") in excluded:
            continue
        v = r.get("llm_cost_usd")
        if v is None:
            continue
        costs.append(float(v))
    if not costs:
        return None
    return float(np.mean(costs))


def _holdout_judge_mean(benchmark: dict) -> float | None:
    """Mean LLM-judge accuracy, dropping rows where the judge call failed.

    ``judge: None`` in a per-question row means the judge timed out / failed
    to parse / hit a content filter — not "judge said incorrect". Drop those
    rows from the denominator. Returns None if no judge column at all.
    """
    excluded = set(benchmark.get("excluded_question_ids") or [])
    vals: list[float] = []
    saw_judge = False
    for r in benchmark.get("per_question", []):
        if r.get("id") in excluded:
            continue
        v = r.get("judge")
        if v is None:
            saw_judge = saw_judge or "judge" in r
            continue
        saw_judge = True
        vals.append(1.0 if v == 1 else 0.0)
    if not saw_judge:
        return None
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _pad_edge(curves: list[np.ndarray]) -> np.ndarray:
    """Pad ragged curves to the max length using edge replication.

    Correct for best-so-far curves: if seed K stopped at trial T<max, its
    best-so-far does not regress past trial T, so edge replication keeps the
    mean honest.
    """
    max_len = max(len(c) for c in curves)
    return np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in curves])


def _pad_nan(curves: list[np.ndarray]) -> np.ndarray:
    """Pad ragged curves to the max length using NaN.

    Use for raw per-trial scores. A seed that stopped at trial T should not
    contribute to mean/std past T — replication would flatten the tail with
    a synthetic value. ``np.nanmean`` / ``np.nanstd`` handle the pads.
    """
    max_len = max(len(c) for c in curves)
    return np.array(
        [np.pad(c.astype(float), (0, max_len - len(c)), mode="constant", constant_values=np.nan) for c in curves]
    )


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _integer_xticks(ax) -> None:
    """Force integer-only ticks on the x-axis.

    Trial numbers are integers; with few data points matplotlib's default
    auto-ticker interpolates to decimals (e.g. 1.0, 1.2, 1.4 for two trials)
    which is meaningless for a trial-index axis.
    """
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _safely(name: str, fn: Callable[[], None]) -> None:
    """Run a figure-writer; log+swallow failures.

    Figures are best-effort: a corrupt seed should not abort the matrix or
    block downstream analysis. Each call site keeps its own try/except so
    one bad figure doesn't take the rest down.
    """
    try:
        fn()
    except Exception:
        logger.warning("Figure %s failed", name, exc_info=True)


# -------------------------------------------------------------------- per-seed


def make_seed_figures(seed_dir: Path) -> None:
    """Render figures for a single ``(method, seed)`` run.

    Writes into ``seed_dir/figures/``. Reads ``history.jsonl`` for the
    trajectory and ``benchmark_results.json`` for the hold-out reference
    line (if scoring has completed; absence is non-fatal).
    """
    history = _read_history(seed_dir)
    if not history:
        return

    figures_dir = seed_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    method = seed_dir.parent.name
    seed_label = seed_dir.name
    color = _color_for(method)

    trial_nums = np.array([int(e["trial_number"]) for e in history])
    scores = np.array([_entry_score(e) for e in history])
    eval_usds = np.array([_entry_eval_usd(e) for e in history])
    benchmark = _read_benchmark(seed_dir)

    _safely(
        f"seed score_per_trial: {seed_dir}",
        lambda: _seed_score_per_trial(
            figures_dir / "score_per_trial.png",
            trial_nums,
            scores,
            benchmark,
            method,
            seed_label,
            color,
        ),
    )
    _safely(
        f"seed cost_per_trial: {seed_dir}",
        lambda: _seed_cost_per_trial(
            figures_dir / "cost_per_trial.png",
            trial_nums,
            eval_usds,
            method,
            seed_label,
            color,
        ),
    )


def _seed_score_per_trial(
    out_path: Path,
    trial_nums: np.ndarray,
    scores: np.ndarray,
    benchmark: dict | None,
    method: str,
    seed_label: str,
    color: str,
) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(trial_nums, scores, "o-", color=color, label="Trial score", alpha=0.9)
    if len(trial_nums) > 1:
        best = np.maximum.accumulate(scores)
        ax.plot(trial_nums, best, "--", color="black", label="Best so far", alpha=0.55)
    if benchmark is not None:
        judge = _holdout_judge_mean(benchmark)
        if judge is not None:
            ax.axhline(
                judge,
                color="firebrick",
                linestyle=":",
                alpha=0.7,
                label=f"Hold-out judge = {judge:.2f}",
            )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Exam score")
    ax.set_ylim(0, 1)
    # With a single trial, matplotlib auto-axis zooms into [0.95, 1.05] which
    # is meaningless. Force a sensible frame so the dot reads as "trial 1 of
    # 1", and pin the single integer tick (MaxNLocator decays into decimal
    # labels when xlim is this narrow).
    if len(trial_nums) == 1:
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([int(trial_nums[0])])
    else:
        _integer_xticks(ax)
    ax.set_title(f"{method} / {seed_label}: score per trial")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _seed_cost_per_trial(
    out_path: Path,
    trial_nums: np.ndarray,
    eval_usds: np.ndarray,
    method: str,
    seed_label: str,
    color: str,
) -> None:
    plt = _import_matplotlib()
    fig, (ax_per, ax_cum) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax_per.bar(trial_nums, eval_usds, color=color, alpha=0.8)
    ax_per.set_xlabel("Trial number")
    ax_per.set_ylabel("Per-trial cost (USD)")
    ax_per.set_title("Per-trial cost")
    ax_per.grid(alpha=0.3, axis="y")

    ax_cum.plot(trial_nums, np.cumsum(eval_usds), "o-", color=color)
    ax_cum.set_xlabel("Trial number")
    ax_cum.set_ylabel("Cumulative cost (USD)")
    ax_cum.set_title("Cumulative cost")
    ax_cum.grid(alpha=0.3)

    if len(trial_nums) == 1:
        for sub in (ax_per, ax_cum):
            sub.set_xlim(0.5, 1.5)
            sub.set_xticks([int(trial_nums[0])])
    else:
        _integer_xticks(ax_per)
        _integer_xticks(ax_cum)

    fig.suptitle(f"{method} / {seed_label}: trial cost", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ per-method


def make_method_figures(method_dir: Path) -> None:
    """Render figures aggregated across seeds for one method.

    Writes into ``method_dir/figures/``. For deterministic methods with one
    seed, the per-seed and per-method views collapse to the same plot — we
    still write the method-level files so the layout is consistent.
    """
    seed_dirs = _seed_dirs(method_dir)
    if not seed_dirs:
        return

    figures_dir = method_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    method = method_dir.name
    color = _color_for(method)

    seed_runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for sd in seed_dirs:
        hist = _read_history(sd)
        if not hist:
            continue
        trial_nums = np.array([int(e["trial_number"]) for e in hist])
        scores = np.array([_entry_score(e) for e in hist])
        best = np.maximum.accumulate(scores)
        seed_runs.append((sd.name, trial_nums, scores, best))

    # Trajectory plots only make sense for sequential methods (those with a
    # real per-trial sweep in ``history.jsonl``).
    if seed_runs and _is_sequential(method):
        _safely(
            f"method score_per_trial: {method_dir}",
            lambda: _method_score_per_trial(
                figures_dir / "score_per_trial.png",
                method,
                color,
                seed_runs,
            ),
        )
        _safely(
            f"method best_so_far: {method_dir}",
            lambda: _method_best_so_far(
                figures_dir / "best_so_far.png",
                method,
                color,
                seed_runs,
            ),
        )

    # Hold-out per seed — independent of history availability.
    seed_metrics: list[tuple[str, dict]] = []
    for sd in seed_dirs:
        bm = _read_benchmark(sd)
        if bm is not None:
            seed_metrics.append((sd.name, bm))
    if seed_metrics:
        _safely(
            f"method holdout_metrics: {method_dir}",
            lambda: _method_holdout_metrics(
                figures_dir / "holdout_metrics.png",
                method,
                seed_metrics,
            ),
        )


def _method_score_per_trial(
    out_path: Path,
    method: str,
    color: str,
    seed_runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    raw_curves = [scores for _, _, scores, _ in seed_runs]
    if len(raw_curves) >= 2:
        padded = _pad_nan(raw_curves)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
        x = np.arange(1, padded.shape[1] + 1)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15,
            color=color,
            label="mean ± std",
        )
        ax.plot(x, mean, "-", color=color, alpha=0.9)
    for name, trial_nums, scores, _ in seed_runs:
        ax.plot(trial_nums, scores, "o-", alpha=0.6, label=name)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Exam score (per trial)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{method}: score per trial")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _integer_xticks(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _method_best_so_far(
    out_path: Path,
    method: str,
    color: str,
    seed_runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    best_curves = [best for _, _, _, best in seed_runs]
    if len(best_curves) >= 2:
        padded = _pad_edge(best_curves)
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        x = np.arange(1, padded.shape[1] + 1)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15,
            color=color,
            label="mean ± std",
        )
        ax.plot(x, mean, "-", color=color, alpha=0.9)
    for name, trial_nums, _, best in seed_runs:
        ax.plot(trial_nums, best, "o-", alpha=0.6, label=name)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best-so-far exam score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{method}: best-so-far trajectory")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _integer_xticks(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _method_holdout_metrics(
    out_path: Path,
    method: str,
    seed_metrics: list[tuple[str, dict]],
) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    labels = [name for name, _ in seed_metrics]
    x = np.arange(len(labels))
    bw = 0.27
    metric_specs: list[tuple[str, str, Callable[[dict], float]]] = [
        ("EM", "#1f77b4", lambda b: float(b.get("em", 0.0))),
        ("F1", "#ff7f0e", lambda b: float(b.get("f1", 0.0))),
        ("Judge", "#2ca02c", lambda b: _holdout_judge_mean(b) or 0.0),
    ]
    for i, (label, mcolor, getter) in enumerate(metric_specs):
        vals = [getter(bm) for _, bm in seed_metrics]
        ax.bar(x + (i - 1) * bw, vals, bw, color=mcolor, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out score")
    ax.set_title(f"{method}: hold-out metrics per seed")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- matrix


def make_matrix_figures(
    output_root: Path,
    *,
    figures_dir: Path | None = None,
    benchmark_pretty_name: str | None = None,
) -> None:
    """Render cross-method figures into ``figures_dir`` (default
    ``output_root/figures/``).

    Idempotent: reads the full tree under ``output_root`` and rewrites every
    matrix-level figure plus ``Table_1.md`` from scratch. Safe to call at any
    point — partial matrices render partial figures.

    ``figures_dir`` is the output override. The in-run hook leaves it None so
    everything stays under ``output_root``; ``analyze.analyze`` passes an
    explicit override when the user wants a separate paper-artifact directory.

    ``benchmark_pretty_name`` titles the Markdown table. When omitted, the
    name is recovered from ``run_metadata.json`` at ``output_root`` so the
    in-run hook doesn't have to plumb it explicitly.

    Delegates the bootstrap-CI stats (``aggregate_by_method``) and the
    summary writers (``write_markdown_table``, ``write_holdout_scores_figure``,
    ``write_efficiency_figure``) to ``analyze.py`` to keep one source of
    truth for the table the paper consumes.
    """
    if not output_root.exists():
        return
    method_dirs = _method_dirs(output_root)
    if not method_dirs:
        return

    # Late import to avoid a plots ↔ analyze import cycle (analyze does not
    # import plots, but a future caller might add the inverse).
    from agentic_autorag_bench.analyze import (
        aggregate_by_method,
        load_results,
        read_benchmark_pretty_name,
        write_efficiency_figure,
        write_holdout_scores_figure,
        write_markdown_table,
    )

    if figures_dir is None:
        figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if benchmark_pretty_name is None:
        benchmark_pretty_name = read_benchmark_pretty_name(output_root)

    # Trajectory plots only need history.jsonl; render them first so a
    # mid-matrix run with no hold-out scoring yet still gets a useful view.
    _safely(
        "matrix score_per_trial.png",
        lambda: _matrix_score_per_trial(figures_dir / "score_per_trial.png", output_root),
    )
    _safely(
        "matrix best_so_far.png",
        lambda: _matrix_best_so_far(figures_dir / "best_so_far.png", output_root),
    )

    # The remaining figures need hold-out scoring. ``load_results`` skips a
    # seed dir that has no benchmark_results.json (with a warning).
    results = load_results(output_root)
    if not results:
        logger.info(
            "make_matrix_figures: no hold-out results yet under %s; wrote trajectory plots only",
            output_root,
        )
        return
    stats = aggregate_by_method(results)

    _safely(
        "matrix Table_1.md",
        lambda: write_markdown_table(
            stats,
            figures_dir / "Table_1.md",
            benchmark_pretty_name=benchmark_pretty_name,
        ),
    )
    _safely(
        "matrix holdout_metrics.png",
        lambda: write_holdout_scores_figure(stats, figures_dir / "holdout_metrics.png"),
    )
    _safely(
        "matrix cost_breakdown.png",
        lambda: _matrix_cost_breakdown(figures_dir / "cost_breakdown.png", stats),
    )
    _safely(
        "matrix token_breakdown.png",
        lambda: _matrix_token_breakdown(figures_dir / "token_breakdown.png", stats),
    )
    _safely(
        "matrix cost_and_embeddings.png",
        lambda: _matrix_cost_and_embeddings(figures_dir / "cost_and_embeddings.png", stats),
    )
    # ``efficiency`` and ``score_vs_cost`` are appendix-only — F1+F2+F3+F3b+F4
    # cover the paper's body. Keep rendering them so the appendix has
    # something to point at without re-running the matrix.
    appendix_dir = figures_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    _safely(
        "appendix efficiency.png",
        lambda: write_efficiency_figure(stats, appendix_dir / "efficiency.png"),
    )
    _safely(
        "appendix score_vs_cost.png",
        lambda: _matrix_score_vs_cost(appendix_dir / "score_vs_cost.png", results, stats),
    )


def _base_sequential_methods(output_root: Path) -> list[str]:
    """Base sequential methods (no ``@k`` checkpoints).

    Trajectory plots exclude ``@k`` variants: a checkpoint's history is a
    strict prefix of its parent's curve, so plotting it would just redraw a
    truncated copy of the same line.
    """
    return [m for m in _discover_method_names(output_root) if _is_sequential(m) and "@" not in m]


def _matrix_score_per_trial(out_path: Path, output_root: Path) -> None:
    """Per-trial exam score across base methods (mean ± std across seeds).

    NaN-padded so an aborted seed does not flatten the tail. ``@k`` checkpoint
    variants are excluded (they're prefixes of their parent's curve).
    """
    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    has_data = False
    for method in _base_sequential_methods(output_root):
        method_dir = output_root / method
        if not method_dir.is_dir():
            continue
        per_seed_scores: list[np.ndarray] = []
        for sd in _seed_dirs(method_dir):
            hist = _read_history(sd)
            if hist:
                per_seed_scores.append(np.array([_entry_score(e) for e in hist]))
        if not per_seed_scores:
            continue
        color = _color_for(method)
        if len(per_seed_scores) >= 2:
            padded = _pad_nan(per_seed_scores)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(padded, axis=0)
                std = np.nanstd(padded, axis=0)
            x = np.arange(1, padded.shape[1] + 1)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
            ax.plot(x, mean, "o-", color=color, markersize=4, label=display_label(method))
        else:
            scores = per_seed_scores[0]
            ax.plot(
                np.arange(1, scores.size + 1),
                scores,
                "o-",
                color=color,
                markersize=4,
                label=display_label(method),
            )
        has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Self-generated exam score (per trial)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-trial self-generated exam score across methods")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    _integer_xticks(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _matrix_best_so_far(out_path: Path, output_root: Path) -> None:
    """Best-so-far trajectory across base methods (mean ± std across seeds).

    ``@k`` checkpoint variants are excluded (prefixes of the parent curve).
    """
    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    has_data = False
    for method in _base_sequential_methods(output_root):
        method_dir = output_root / method
        if not method_dir.is_dir():
            continue
        per_seed_best: list[np.ndarray] = []
        for sd in _seed_dirs(method_dir):
            hist = _read_history(sd)
            if hist:
                scores = np.array([_entry_score(e) for e in hist])
                per_seed_best.append(np.maximum.accumulate(scores))
        if not per_seed_best:
            continue
        color = _color_for(method)
        if len(per_seed_best) >= 2:
            padded = _pad_edge(per_seed_best)
            mean = padded.mean(axis=0)
            std = padded.std(axis=0)
            x = np.arange(1, padded.shape[1] + 1)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
            ax.plot(x, mean, "-", color=color, label=display_label(method))
        else:
            best = per_seed_best[0]
            ax.plot(np.arange(1, best.size + 1), best, "-", color=color, label=display_label(method))
        has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best-so-far exam score")
    ax.set_ylim(0, 1)
    ax.set_title("Best-so-far trajectory across methods (mean ± std across seeds)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    _integer_xticks(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _matrix_score_vs_cost(out_path: Path, results: list, stats: dict[str, dict]) -> None:
    """Pipeline cost-quality Pareto: held-out judge vs. per-question LLM cost.

    Each method is one point. X-axis is the *deploy-time* cost of running
    the winning pipeline on a single question, computed as
    ``mean(input_tokens × in_price + completion_tokens × out_price)`` over
    the union-excluded hold-out rows — i.e. the cost a production user
    pays per query. Y-axis is held-out LLM-judge accuracy with ± SD
    across hold-out replays from ``aggregate_by_method``. This is
    distinct from ``efficiency.png``, which uses *total search cost*
    (the one-time bill to find the winner). Both matter; this one is
    the better Pareto for "should I deploy this?".

    Horizontal error bars are per-seed std on cost (zero with a single seed).
    Vertical error bars are ± SD across hold-out replays (zero when only the
    end-of-search eval exists).
    """
    apply_paper_style()
    plt = _import_matplotlib()
    methods = _order_methods(stats.keys())
    if not methods:
        return

    by_method: dict[str, list] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    has_data = False
    for m in methods:
        runs = by_method.get(m, [])
        if not runs:
            continue
        cpq_seeds: list[float] = []
        for r in runs:
            cpq = _benchmark_cost_per_question(r.benchmark)
            if cpq is not None:
                cpq_seeds.append(cpq)
        if not cpq_seeds:
            continue
        x_mean = float(np.mean(cpq_seeds))
        x_err = float(np.std(cpq_seeds)) if len(cpq_seeds) > 1 else 0.0
        y_mean, y_lo, y_hi = stats[m]["judge"]
        y_err = [[y_mean - y_lo], [y_hi - y_mean]]
        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_err if x_err > 0 else None,
            yerr=y_err,
            fmt="o",
            color=_color_for(m),
            markersize=9,
            capsize=3,
            elinewidth=1.0,
            label=display_label(m),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel("Mean LLM cost per question on hold-out (USD)")
    ax.set_ylabel("Held-out LLM-Judge accuracy")
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.set_title("Pipeline cost-quality Pareto")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", frameon=False, markerscale=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _matrix_cost_breakdown(out_path: Path, stats: dict[str, dict]) -> None:
    """Stacked bar: optimizer reasoning vs trial evaluation USD per method.

    The plan's third layer (``exam_generation``) is deliberately omitted: under
    the bench's fairness rule, exam-gen cost is excluded from the bench tally
    because only ``agentic_*`` creates an exam. Including it would penalize
    that method for work the others don't do. Embedding tokens have no USD
    here (local execution); they appear on the companion ``token_breakdown.png``.

    Bars annotated with the totals so a $0 stack (e.g. agentic with an
    unpriced examiner/optimizer model) is still readable.
    """
    apply_paper_style()
    plt = _import_matplotlib()
    methods = _order_methods(stats.keys())
    if not methods:
        return
    reasoning = np.array([stats[m]["optimizer_usd_mean"] for m in methods])
    trial = np.array([stats[m]["trial_usd_mean"] for m in methods])
    totals = reasoning + trial
    fig, ax = plt.subplots(figsize=(fig_width_for(len(methods)), 4.0))
    x = np.arange(len(methods))
    ax.bar(x, reasoning, color="#1f77b4", label="Optimizer reasoning (agent)")
    ax.bar(x, trial, bottom=reasoning, color="#ff7f0e", label="Trial evaluation (RAG + judge)")
    for xi, total in zip(x, totals, strict=True):
        if total > 0:
            ax.text(xi, total, f"${total:.3f}", ha="center", va="bottom", fontsize=8)
    style_method_xticks(ax, methods)
    ax.set_ylabel("Mean search cost per seed (USD)")
    if totals.max() > 0:
        ax.set_ylim(0, totals.max() * 1.18)
    legend_outside(ax, ncol=2, title="Search cost by source")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _matrix_token_breakdown(out_path: Path, stats: dict[str, dict]) -> None:
    """Two-panel stacked bar of LLM and embedding tokens per method.

    Left panel: LLM input vs output tokens (mean per seed). Right panel:
    embedding input tokens (mean per seed). This is the recommended
    primary cost-comparability view for the paper — wall-clock varies with
    rate limits and cache state, USD varies with which provider you bill,
    but tokens are deterministic and cache-aware (first-use-per-(method,
    seed) rule from the framework's cost ledger).
    """
    apply_paper_style()
    plt = _import_matplotlib()
    methods = _order_methods(stats.keys())
    if not methods:
        return
    prompt = np.array([stats[m]["prompt_tokens_mean"] for m in methods])
    completion = np.array([stats[m]["completion_tokens_mean"] for m in methods])
    embed = np.array([stats[m]["embedding_tokens_mean"] for m in methods])

    panel_w = fig_width_for(len(methods), base=5.0, per_group=0.8, cap=9.0)
    fig, (ax_llm, ax_emb) = plt.subplots(1, 2, figsize=(2 * panel_w, 4.0))

    ax_llm.bar(np.arange(len(methods)), prompt, color="#1f77b4", label="LLM input")
    ax_llm.bar(np.arange(len(methods)), completion, bottom=prompt, color="#ff7f0e", label="LLM output")
    style_method_xticks(ax_llm, methods)
    ax_llm.set_ylabel("Mean tokens per seed")
    ax_llm.set_title("LLM tokens (input + output)")
    ax_llm.legend(loc="upper right", frameon=False)
    ax_llm.grid(axis="y", alpha=0.3)

    ax_emb.bar(np.arange(len(methods)), embed, color="#2ca02c", label="Embedding input")
    style_method_xticks(ax_emb, methods)
    ax_emb.set_ylabel("Mean tokens per seed")
    ax_emb.set_title("Embedding tokens (cache-aware)")
    ax_emb.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _matrix_cost_and_embeddings(out_path: Path, stats: dict[str, dict]) -> None:
    """Two-panel: search cost by source (left) + embedding tokens (right).

    The companion to ``token_breakdown`` with the LLM-token panel swapped for
    the dollar cost-by-source stack, so one figure pairs the USD bill of search
    with the (local, cache-aware) embedding-token footprint. Exam-generation
    cost is excluded from the left panel under the bench's fairness rule (only
    ``agentic_*`` creates an exam).
    """
    apply_paper_style()
    plt = _import_matplotlib()
    methods = _order_methods(stats.keys())
    if not methods:
        return
    reasoning = np.array([stats[m]["optimizer_usd_mean"] for m in methods])
    trial = np.array([stats[m]["trial_usd_mean"] for m in methods])
    totals = reasoning + trial
    embed = np.array([stats[m]["embedding_tokens_mean"] for m in methods])

    panel_w = fig_width_for(len(methods), base=5.0, per_group=0.8, cap=9.0)
    fig, (ax_cost, ax_emb) = plt.subplots(1, 2, figsize=(2 * panel_w, 4.0))
    x = np.arange(len(methods))

    ax_cost.bar(x, reasoning, color="#1f77b4", label="Optimizer reasoning (agent)")
    ax_cost.bar(x, trial, bottom=reasoning, color="#ff7f0e", label="Trial evaluation (RAG + judge)")
    for xi, total in zip(x, totals, strict=True):
        if total > 0:
            ax_cost.text(xi, total, f"${total:.3f}", ha="center", va="bottom", fontsize=8)
    style_method_xticks(ax_cost, methods)
    ax_cost.set_ylabel("Mean search cost per seed (USD)")
    ax_cost.set_title("Search cost by source")
    if totals.max() > 0:
        ax_cost.set_ylim(0, totals.max() * 1.18)
    ax_cost.legend(loc="upper right", frameon=False)
    ax_cost.grid(axis="y", alpha=0.3)

    ax_emb.bar(x, embed, color="#2ca02c", label="Embedding input")
    style_method_xticks(ax_emb, methods)
    ax_emb.set_ylabel("Mean tokens per seed")
    ax_emb.set_title("Embedding tokens (cache-aware)")
    ax_emb.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Cost-aware Pareto figures (Exp-2 / UniDoc).
#
# Relocated here (from pareto.py) so every benchmark figure lives in one
# module. ``make_pareto_figure`` renders a single-method Syftr-style scatter
# (gray trial cloud + the optimizer's self-marked frontier, numbered and
# described in a side legend). ``make_pareto_comparison_figure`` overlays every
# method's frontier, and ``compute_pareto_hypervolumes`` scores each frontier
# against a SHARED reference point (pooled across all methods) so the
# hypervolumes are comparable. X-axis is deploy-time cost **per query**.
# =====================================================================

# Distinct index colours for the numbered frontier-config markers (these encode a
# config's rank on the frontier, NOT a method identity). Standard matplotlib
# tab10, dropping brown; blue is last since the markers sit on the blue Agentic
# line. Cycled if a frontier has more points than colours.
_FRONTIER_COLORS = [
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
    "#1f77b4",  # blue
]


def _frontier_color(i: int):
    """Colour for the ``i``-th (1-based) numbered frontier marker."""
    return _FRONTIER_COLORS[(i - 1) % len(_FRONTIER_COLORS)]

_CHUNKING_LABELS = {"recursive": "Recursive Splitting", "fixed": "Token Splitting"}
_INDEX_LABELS = {
    "vector_only": "Dense Retrieval",
    "hybrid_bm25_vector": "Hybrid Retrieval",
    "graph_only": "Graph Retrieval",
    "hybrid_graph_vector": "Hybrid Graph Retrieval",
}
_QUERY_EXPANSION_LABELS = {
    "hyde": "HyDE",
    "multi_query": "Multi-Query",
    "query_decompose": "Query Decompose",
}
# Vendor/region tokens to strip from a model id so labels read "kimi-k2.5"
# rather than "moonshotai.kimi-k2.5" or "us.meta.llama...".
_MODEL_VENDOR_TOKENS = {
    "moonshotai",
    "us",
    "global",
    "amazon",
    "meta",
    "mistral",
    "google",
    "qwen",
    "nvidia",
    "zai",
    "minimax",
    "openai",
    "ai21",
    "cohere",
    "anthropic",
}


def _short_model(name: str) -> str:
    """Compact display name for a LiteLLM model id (drop provider/vendor cruft)."""
    if not name:
        return "?"
    tail = name.split("/")[-1].split(":")[0]
    parts = tail.split(".")
    while len(parts) > 1 and parts[0].lower() in _MODEL_VENDOR_TOKENS:
        parts = parts[1:]
    return ".".join(parts)


def _join_clause(head: str, rest: list[str]) -> str:
    if not rest:
        return head
    if len(rest) == 1:
        return f"{head} with {rest[0]}"
    if len(rest) == 2:
        return f"{head} with {rest[0]} and {rest[1]}"
    return f"{head} with {', '.join(rest[:-1])}, and {rest[-1]}"


def _describe_config(config: dict) -> str:
    """One-line description of a trial config for the figure legend.

    Includes ``top_k`` (retrieval depth) and ``top_n`` (reranker depth) since
    they're the levers the cost-aware optimizer most often trades against cost.
    """
    head = _short_model(config.get("generator_llm", ""))
    rest: list[str] = []
    if (ck := _CHUNKING_LABELS.get(config.get("chunking_strategy"))) is not None:
        rest.append(ck)
    if (idx := _INDEX_LABELS.get(config.get("index_type"))) is not None:
        top_k = config.get("top_k")
        rest.append(f"{idx} (top_k={top_k})" if top_k is not None else idx)
    if (qe := _QUERY_EXPANSION_LABELS.get(config.get("query_expansion"))) is not None:
        rest.append(qe)
    reranker = config.get("reranker")
    if reranker and reranker != "none":
        top_n = config.get("reranker_top_n")
        label = f"{_short_model(reranker)} reranking"
        rest.append(f"{label} (top_n={top_n})" if top_n is not None else label)
    return _join_clause(head, rest)


@dataclass
class _TrialPoint:
    trial_number: int
    cost_per_query: float
    answer_accuracy: float
    is_pareto: bool
    config: dict


def _load_trial_points(seed_dir: Path) -> list[_TrialPoint]:
    """Trials with a usable (cost>0, accuracy) pair, from the rich agentic history."""
    points: list[_TrialPoint] = []
    for row in _read_history(seed_dir):
        cost = row.get("mean_llm_cost_per_query_usd")
        score = row.get("answer_accuracy")
        if cost is None or score is None or float(cost) <= 0.0:
            continue
        points.append(
            _TrialPoint(
                trial_number=int(row.get("trial_number", 0)),
                cost_per_query=float(cost),
                answer_accuracy=float(score),
                is_pareto=bool(row.get("is_pareto_optimal", False)),
                config=row.get("config") or {},
            )
        )
    return points


def make_pareto_figure(seed_dir: Path, out_path: Path, *, domain: str = "") -> None:
    """Render the cost-vs-exam-accuracy Pareto figure.

    Gray cloud of every trial; the optimizer's Pareto-optimal trials drawn as
    colored, numbered markers connected along the frontier and described in a
    side legend. No-op (no file) when there is nothing plottable.
    """
    points = _load_trial_points(seed_dir)
    if not points:
        logger.warning("No plottable trials under %s; skipping pareto figure", seed_dir)
        return

    frontier = sorted((p for p in points if p.is_pareto), key=lambda p: p.cost_per_query)

    apply_paper_style()
    plt = _import_matplotlib()
    # Narrow plotting axes (in line with the other scatter figures) so the
    # narrow cost range isn't stretched across the page; the config legend
    # sits to the right and the tight bbox grows the canvas to fit it.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Cloud of all trials.
    ax.scatter(
        [p.cost_per_query for p in points],
        [p.answer_accuracy * 100 for p in points],
        s=42,
        c="#d9d9d9",
        edgecolors="none",
        alpha=0.7,
        zorder=1,
        label="All trials",
    )

    # Frontier line (sorted by cost ascending).
    if len(frontier) >= 2:
        ax.plot(
            [p.cost_per_query for p in frontier],
            [p.answer_accuracy * 100 for p in frontier],
            color="#7f7f7f",
            lw=1.1,
            zorder=2,
            label="Pareto frontier",
        )

    # Colored, numbered frontier points + side-legend descriptions.
    for i, p in enumerate(frontier, start=1):
        ax.scatter(
            p.cost_per_query,
            p.answer_accuracy * 100,
            s=110,
            color=_frontier_color(i),
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            label=f"{i}. {_describe_config(p.config)}",
        )
        ax.annotate(
            str(i),
            (p.cost_per_query, p.answer_accuracy * 100),
            textcoords="offset points",
            xytext=(6, 5),
            fontweight="bold",
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Exam accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"{display_label('agentic_cost')} cost vs exam accuracy on UniDoc{title_domain}")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
        title="Frontier configurations",
        title_fontsize=9,
        borderaxespad=0.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class _FrontierRecord:
    """Minimal record for the framework's ``compute_frontier`` / ``compute_hypervolume``
    (which read ``trial_number`` / ``answer_accuracy`` / ``mean_llm_cost_per_query_usd``)."""

    trial_number: int
    answer_accuracy: float
    mean_llm_cost_per_query_usd: float


def compute_pareto_hypervolumes(method_seed_points: dict[str, list[list[_TrialPoint]]], cost_ref: float) -> dict:
    """Per-method NORMALIZED hypervolume, aggregated across seeds, against a shared,
    space-derived reference point.

    Each seed's frontier is recomputed (framework ``compute_frontier``), scored
    against ``ref_point=(0, cost_ref)``, and divided by ``cost_ref`` (the area of the
    ideal ``[0,1] x [0, cost_ref]`` box), so it reads as the fraction of the
    achievable cost-quality box that seed's frontier dominates, in ``[0, 1]``. The
    per-seed fractions are summarized as mean / min / max so no single seed is
    privileged and the table matches the seed-aggregated figures. ``cost_ref`` is a
    property of the search space (see ``space_derived_cost_reference``), shared
    across every method so the numbers are comparable.
    """
    from agentic_autorag.optimizer import pareto as fpareto

    ref_point = (0.0, cost_ref)
    out: dict = {"score_reference": 0.0, "cost_reference": cost_ref, "methods": {}}
    for method, seeds in method_seed_points.items():
        per_seed = []
        for pts in seeds:
            records = [_FrontierRecord(p.trial_number, p.answer_accuracy, p.cost_per_query) for p in pts]
            frontier = fpareto.compute_frontier(records)
            per_seed.append(fpareto.compute_hypervolume(frontier, ref_point=ref_point) / cost_ref)
        if not per_seed:
            continue
        out["methods"][method] = {
            "hypervolume_mean": sum(per_seed) / len(per_seed),
            "hypervolume_min": min(per_seed),
            "hypervolume_max": max(per_seed),
            "hypervolume_per_seed": per_seed,
            "n_seeds": len(per_seed),
            "n_trials_per_seed": [len(pts) for pts in seeds],
        }
    return out


def make_pareto_comparison_figure(
    method_points: dict[str, list[_TrialPoint]],
    out_path: Path,
    *,
    domain: str = "",
) -> None:
    """Overlay every method's trial cloud + Pareto frontier (one seed) on one
    figure. The hypervolume numbers live in ``hypervolume.json`` and the
    hypervolume-over-trials figure, both aggregated across seeds; this scatter is
    the shape of a single run. No-op when nothing is plottable."""
    from agentic_autorag.optimizer import pareto as fpareto

    if not any(method_points.values()):
        logger.warning("No plottable trials; skipping pareto comparison figure")
        return

    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for method, pts in method_points.items():
        if not pts:
            continue
        color = _color_for(method)
        ax.scatter(
            [p.cost_per_query for p in pts],
            [p.answer_accuracy * 100 for p in pts],
            s=28,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
        records = [_FrontierRecord(p.trial_number, p.answer_accuracy, p.cost_per_query) for p in pts]
        frontier_tns = {r.trial_number for r in fpareto.compute_frontier(records)}
        frontier = sorted((p for p in pts if p.trial_number in frontier_tns), key=lambda p: p.cost_per_query)
        label = display_label(method)
        if len(frontier) >= 2:
            ax.plot(
                [p.cost_per_query for p in frontier],
                [p.answer_accuracy * 100 for p in frontier],
                color=color,
                lw=1.6,
                marker="o",
                ms=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=3,
                label=label,
            )
        elif frontier:
            ax.scatter(
                [frontier[0].cost_per_query],
                [frontier[0].answer_accuracy * 100],
                color=color,
                s=80,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
                label=label,
            )
        else:
            ax.plot([], [], color=color, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Exam accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"Cost vs exam accuracy on UniDoc{title_domain} (Pareto frontiers)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------- multi-seed Pareto figures
#
# ``make_pareto_comparison_figure`` + ``hypervolume.json`` read a SINGLE seed per
# method (the driver ``seed``). The two figures below read EVERY seed of every
# method and show the seed spread as a shaded band, so the paper can report seed
# robustness without drawing one line per (method, seed).

_ATTAINMENT_GRID_POINTS = 240


def _is_our_method(method: str) -> bool:
    """Whether to visually emphasise ``method`` as the paper's own optimizer."""
    return method.split("@", 1)[0].startswith("agentic")


def _load_method_seed_points(output_root: Path) -> dict[str, list[list[_TrialPoint]]]:
    """Per method (``_order_methods``-ordered), the trial points of each seed that
    has >=1 plottable trial. Keeps seeds separate (unlike the single-seed
    ``method_points`` the comparison figure uses) so the band figures below can
    show inter-seed spread."""
    out: dict[str, list[list[_TrialPoint]]] = {}
    for method in _discover_method_names(output_root):
        seed_curves = [_load_trial_points(sd) for sd in _seed_dirs(output_root / method)]
        seed_curves = [pts for pts in seed_curves if pts]
        if seed_curves:
            out[method] = seed_curves
    return out


def _attainment_curve(points: list[_TrialPoint], grid: np.ndarray) -> np.ndarray:
    """Empirical attainment function: best accuracy among trials with ``cost <= x``
    for each ``x`` in ``grid``, and 0 where ``x`` is below the seed's cheapest trial.

    Zero (not NaN) is the correct value below the cheapest trial: a run that found
    no configuration at or under budget ``x`` has nothing to deploy there, so its
    attained accuracy is 0. Monotone non-decreasing in ``x``."""
    ordered = sorted(points, key=lambda p: p.cost_per_query)
    costs = np.array([p.cost_per_query for p in ordered])
    best = np.maximum.accumulate(np.array([p.answer_accuracy for p in ordered]))
    idx = np.searchsorted(costs, grid, side="right") - 1
    return np.where(idx >= 0, best[idx.clip(0)], 0.0)


def _attainment_grid(method_seed_points: dict[str, list[list[_TrialPoint]]]) -> np.ndarray | None:
    """Shared log-spaced cost grid spanning every method's cheapest..dearest trial.
    None when there is nothing plottable."""
    all_costs = [p.cost_per_query for seeds in method_seed_points.values() for pts in seeds for p in pts]
    if not all_costs:
        return None
    return np.logspace(np.log10(min(all_costs)), np.log10(max(all_costs)), _ATTAINMENT_GRID_POINTS)


def _attainment_stats(seed_points: list[list[_TrialPoint]], grid: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(min, median, max)`` of the seeds' empirical attainment curves on ``grid``.

    Each is a per-grid-point aggregate across seeds. ``max`` is the best any seed
    attained (the pooled frontier); ``median`` is the typical seed; ``min`` is the
    worst. Because curves are 0-padded below a seed's cheapest trial, all three are
    defined over the whole grid."""
    curves = np.vstack([_attainment_curve(pts, grid) for pts in seed_points])
    return curves.min(axis=0), np.median(curves, axis=0), curves.max(axis=0)


def _plot_attainment(
    ax, method_seed_points: dict[str, list[list[_TrialPoint]]], grid: np.ndarray, *, central: str
) -> None:
    """Draw each method's attainment on ``ax``: a central line + a min-max band.

    ``central`` picks the line: ``"max"`` is the best-across-seeds frontier (every
    frontier point is on the line), ``"median"`` is the typical seed. The line is
    drawn over its full positive extent, so it reaches the cheapest costs a seed
    explored. The min-max band is drawn ONLY where every seed has coverage (its
    worst seed has a trial that cheap), i.e. where a seed-spread estimate is
    honest; in the partial-coverage cheap region the line stands alone rather than
    a band collapsing to 0. Our own method is emphasised."""
    for method in _order_methods(method_seed_points):
        lo, med, hi = _attainment_stats(method_seed_points[method], grid)
        line = hi if central == "max" else med
        color, emph = _color_for(method), _is_our_method(method)
        band = lo > 0  # every seed has a trial this cheap -> spread is meaningful
        ax.fill_between(grid, lo * 100, hi * 100, where=band, color=color,
                        alpha=0.18 if emph else 0.10, lw=0, zorder=4 if emph else 2)
        ax.plot(grid, np.where(line > 0, line * 100, np.nan), color=color, lw=2.4 if emph else 1.7,
                solid_capstyle="round", zorder=6 if emph else 3, label=display_label(method))


def make_pareto_attainment_figure(output_root: Path, out_path: Path, *, domain: str = "") -> None:
    """Cost->accuracy frontier across seeds: best-across-seeds line + min-max band.

    The line is the empirical attainment frontier (best accuracy any seed reached at
    or below each cost), so every frontier point is shown, including cheap configs
    only some seeds explored. The band is the seed min-max, drawn where all seeds
    have coverage. Companion ``make_pareto_attainment_median_figure`` swaps the line
    for the median (typical run). No-op when nothing is plottable."""
    method_seed_points = _load_method_seed_points(output_root)
    grid = _attainment_grid(method_seed_points)
    if grid is None:
        logger.warning("No plottable trials under %s; skipping pareto attainment figure", output_root)
        return

    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _plot_attainment(ax, method_seed_points, grid, central="max")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Exam accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"Cost vs exam accuracy on UniDoc{title_domain} (frontier across seeds)")
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              title="line = best across seeds, band = min-max across seeds")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_median_attainment_panel(ax, method_seed_points, grid, *, domain: str) -> None:
    """Median-seed cost->accuracy attainment (line + min-max band) onto ``ax``.
    Shared by the standalone median figure and the combined landscape figure so
    the two never drift."""
    _plot_attainment(ax, method_seed_points, grid, central="median")
    ax.set_xscale("log")
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Exam accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"Cost vs exam accuracy on UniDoc{title_domain}")
    ax.legend(loc="lower right", frameon=False,
              title="line = median seed, band = min-max across seeds")


def make_pareto_attainment_median_figure(output_root: Path, out_path: Path, *, domain: str = "") -> None:
    """The ``make_pareto_attainment_figure`` companion whose line is the MEDIAN seed.

    Same 0-padded band; the line is the typical seed's attainment rather than the
    best-across-seeds frontier. The median line reaches only costs a majority of
    seeds explored, so it sits below the frontier line in the cheap region and
    reads as "what a single run typically attains". No-op when nothing is
    plottable."""
    method_seed_points = _load_method_seed_points(output_root)
    grid = _attainment_grid(method_seed_points)
    if grid is None:
        logger.warning("No plottable trials under %s; skipping pareto median attainment figure", output_root)
        return

    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _draw_median_attainment_panel(ax, method_seed_points, grid, domain=domain)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pooled_frontier(seed_points: list[list[_TrialPoint]]) -> list[_TrialPoint]:
    """The Pareto frontier of a method's trials pooled across all seeds, sorted by
    cost ascending. This is the best cost-accuracy trade-off the method actually
    reached anywhere, so it traces the top edge of that method's min-max band."""
    from agentic_autorag.optimizer import pareto as fpareto

    pooled = [p for pts in seed_points for p in pts]
    if not pooled:
        return []
    # Trial numbers collide across seeds; index into the pooled list instead.
    records = [_FrontierRecord(i, p.answer_accuracy, p.cost_per_query) for i, p in enumerate(pooled)]
    keep = {r.trial_number for r in fpareto.compute_frontier(records)}
    return sorted((p for i, p in enumerate(pooled) if i in keep), key=lambda p: p.cost_per_query)


def _select_labeled_frontier(frontier: list[_TrialPoint], max_labels: int) -> list[_TrialPoint]:
    """Thin a cost-sorted frontier to at most ``max_labels`` points for legibility.

    Always keeps the cheapest and most-accurate endpoints; fills the rest with the
    points that sit just above the largest accuracy jumps, so the labelled subset
    spans the frontier's whole accuracy range instead of clustering."""
    if len(frontier) <= max_labels:
        return list(frontier)
    keep = {0, len(frontier) - 1}
    interior = sorted(
        range(1, len(frontier) - 1),
        key=lambda i: frontier[i].answer_accuracy - frontier[i - 1].answer_accuracy,
        reverse=True,
    )
    for i in interior:
        if len(keep) >= max_labels:
            break
        keep.add(i)
    return [frontier[i] for i in sorted(keep)]


def make_pareto_frontier_annotated_figure(
    output_root: Path,
    out_path: Path,
    *,
    domain: str = "",
    hero_method: str = "agentic_cost",
    max_labels: int = 6,
) -> None:
    """The seed-aggregated frontier view (``make_pareto_attainment_figure``) with our
    own frontier's concrete configurations called out.

    Every method keeps its best-across-seeds frontier line + min-max band, so the
    head-to-head is the same fair comparison. The pooled Pareto frontier of
    ``hero_method`` lies exactly on that method's line, so a handful of its points
    are marked with numbered dots, each described in a side legend the way
    ``make_pareto_figure`` does for a single seed. This pairs "we dominate the
    frontier" with "here are the configs that get you there". No-op when nothing is
    plottable."""
    from matplotlib.lines import Line2D

    method_seed_points = _load_method_seed_points(output_root)
    grid = _attainment_grid(method_seed_points)
    if grid is None:
        logger.warning("No plottable trials under %s; skipping annotated frontier figure", output_root)
        return

    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    _plot_attainment(ax, method_seed_points, grid, central="max")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Exam accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"Cost vs exam accuracy on UniDoc{title_domain} (frontier across seeds)")

    # Keep the method legend inside; the config legend goes outside on the right.
    method_legend = ax.legend(
        loc="lower right", frameon=False, fontsize=9,
        title="line = best across seeds, band = min-max across seeds",
    )
    ax.add_artist(method_legend)

    frontier = _pooled_frontier(method_seed_points.get(hero_method, []))
    if frontier:
        labeled = _select_labeled_frontier(frontier, max_labels)
        handles: list[Line2D] = []
        for i, p in enumerate(labeled, start=1):
            c = _frontier_color(i)
            ax.scatter(p.cost_per_query, p.answer_accuracy * 100, s=150, color=c,
                       edgecolors="black", linewidths=0.7, zorder=9)
            ax.annotate(str(i), (p.cost_per_query, p.answer_accuracy * 100), ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color="white", zorder=10)
            handles.append(Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor=c,
                                  markeredgecolor="black", markeredgewidth=0.6,
                                  label=f"{i}. {_describe_config(p.config)}"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
                  title=f"{display_label(hero_method)} frontier configurations",
                  title_fontsize=9, borderaxespad=0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _running_hypervolume(points: list[_TrialPoint], ref_point: tuple[float, float]) -> np.ndarray:
    """Best-so-far hypervolume after each trial in trial order (monotone)."""
    from agentic_autorag.optimizer import pareto as fpareto

    ordered = sorted(points, key=lambda p: p.trial_number)
    records = [_FrontierRecord(p.trial_number, p.answer_accuracy, p.cost_per_query) for p in ordered]
    hvs = np.empty(len(records))
    for k in range(1, len(records) + 1):
        hvs[k - 1] = fpareto.compute_hypervolume(fpareto.compute_frontier(records[:k]), ref_point=ref_point)
    return hvs


def _draw_hv_convergence_panel(ax, method_seed_points, *, domain: str, cost_ref: float) -> None:
    """Anytime normalized hypervolume vs. trials (mean + min-max band) onto ``ax``.
    Shared by the standalone HV figure and the combined landscape figure."""
    ref_point = (0.0, cost_ref)
    for method in _order_methods(method_seed_points):
        curves = _pad_edge([_running_hypervolume(pts, ref_point) / cost_ref for pts in method_seed_points[method]])
        x = np.arange(1, curves.shape[1] + 1)
        color, emph = _color_for(method), _is_our_method(method)
        lo, hi = curves.min(axis=0), curves.max(axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18 if emph else 0.10, lw=0, zorder=4 if emph else 2)
        # Very thin same-colour edges so each band's extent stays readable where
        # the fills overlap (this figure has the densest band overlap of the set).
        for edge in (lo, hi):
            ax.plot(x, edge, color=color, lw=0.6, alpha=0.5, zorder=4 if emph else 2)
        ax.plot(x, curves.mean(axis=0), color=color, lw=2.4 if emph else 1.7, solid_capstyle="round",
                zorder=6 if emph else 3, label=display_label(method))

    _integer_xticks(ax)
    ax.set_xlabel("Trials evaluated")
    ax.set_ylabel("Normalized hypervolume")
    ax.grid(alpha=0.3)
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"Hypervolume over trials on UniDoc{title_domain}")
    ax.legend(loc="lower right", frameon=False, title="line = mean, band = min-max across seeds")


def make_pareto_hv_convergence_figure(output_root: Path, out_path: Path, *, domain: str = "", cost_ref: float) -> None:
    """Anytime normalized hypervolume vs. trials: mean over seeds + min-max band.

    Running best-so-far hypervolume against the shared, space-derived ``cost_ref``
    (the same reference used for ``hypervolume.json``, so this figure and the table
    agree), normalized by ``cost_ref`` so each curve reads as the fraction of the
    cost-quality box dominated so far. Shows convergence speed AND inter-seed
    spread. No-op when nothing is plottable."""
    method_seed_points = _load_method_seed_points(output_root)
    all_costs = [p.cost_per_query for seeds in method_seed_points.values() for pts in seeds for p in pts]
    if not all_costs:
        logger.warning("No plottable trials under %s; skipping HV convergence figure", output_root)
        return

    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _draw_hv_convergence_panel(ax, method_seed_points, domain=domain, cost_ref=cost_ref)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_pareto_median_hv_combined_figure(
    output_root: Path, out_path: Path, *, domain: str = "", cost_ref: float
) -> None:
    """Wide landscape figure pairing the two seed-aggregated views side by side:
    the median cost-accuracy frontier (left) and the hypervolume-over-trials curve
    (right). Reuses the exact panels of the two standalone figures, so it stays in
    sync with them. No-op when nothing is plottable."""
    method_seed_points = _load_method_seed_points(output_root)
    grid = _attainment_grid(method_seed_points)
    if grid is None:
        logger.warning("No plottable trials under %s; skipping combined median+HV figure", output_root)
        return

    apply_paper_style()
    plt = _import_matplotlib()
    # Full-width figure* downscaled to ~\textwidth: bump fonts modestly above the
    # paper default so both panels stay legible after the downscale. This figure
    # saves with bbox_inches="tight" (unlike build_cost_figure), which crops the
    # margins and magnifies fonts once stretched to \textwidth, so the values here
    # are smaller than a naive width-scaling would suggest. Scoped to this figure so
    # the standalone panels that share these helpers are unaffected; the smaller
    # legend.title_fontsize keeps the long band-legend titles from dominating.
    font_overrides = {
        "axes.titlesize": 14,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "legend.title_fontsize": 9.5,
    }
    with plt.rc_context(font_overrides):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.4, 4.8))
        _draw_median_attainment_panel(ax_l, method_seed_points, grid, domain=domain)
        _draw_hv_convergence_panel(ax_r, method_seed_points, domain=domain, cost_ref=cost_ref)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _read_seed_cost_embed(seed_dir: Path) -> tuple[float, float, float] | None:
    """``(optimizer_usd, trial_usd, embedding_tokens)`` from a Pareto seed's
    ``optimizer_meta.json``; None if it is missing or unreadable.

    Same three quantities the Exp-1 ``cost_and_embeddings`` panel reads out of
    ``aggregate_by_method``, but the Pareto path has no hold-out ``stats`` object,
    so we read each seed's meta directly."""
    path = seed_dir / "optimizer_meta.json"
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Unreadable optimizer_meta at %s; skipping", path, exc_info=True)
        return None
    return (
        float(m.get("optimizer_usd", 0.0)),
        float(m.get("trial_usd_total", 0.0)),
        float(m.get("embedding_tokens", 0.0)),
    )


def _minmax_err(per_seed: list[float]) -> tuple[float, list[float]]:
    """``(mean, [[down], [up]])`` min-max whisker magnitudes for an errorbar."""
    mean = float(np.mean(per_seed))
    return mean, [[mean - min(per_seed)], [max(per_seed) - mean]]


def make_pareto_cost_and_embeddings_figure(output_root: Path, out_path: Path, *, domain: str = "") -> None:
    """Two-panel search cost + embedding footprint for the Pareto (Exp-2) run, with
    min-max seed whiskers.

    Left: stacked search cost per method (optimizer reasoning + trial evaluation),
    a min-max whisker on the total. Right: embedding input tokens per method (in
    millions), min-max whisker. The Exp-2 analogue of Exp-1's ``cost_and_embeddings``
    but sourced per seed from ``optimizer_meta.json`` and showing the seed spread.
    No-op when nothing is plottable."""
    methods = _discover_method_names(output_root)
    data: dict[str, list[tuple[float, float, float]]] = {}
    for m in methods:
        rows = [r for sd in _seed_dirs(output_root / m) if (r := _read_seed_cost_embed(sd)) is not None]
        if rows:
            data[m] = rows
    methods = [m for m in methods if m in data]
    if not methods:
        logger.warning("No optimizer_meta under %s; skipping cost_and_embeddings figure", output_root)
        return

    opt_mean = np.array([float(np.mean([r[0] for r in data[m]])) for m in methods])
    trial_mean = np.array([float(np.mean([r[1] for r in data[m]])) for m in methods])
    total_stats = [_minmax_err([r[0] + r[1] for r in data[m]]) for m in methods]
    total_mean = np.array([t[0] for t in total_stats])
    total_err = np.hstack([np.array(t[1]) for t in total_stats])
    emb_stats = [_minmax_err([r[2] / 1e6 for r in data[m]]) for m in methods]
    emb_mean = np.array([e[0] for e in emb_stats])
    emb_err = np.hstack([np.array(e[1]) for e in emb_stats])

    apply_paper_style()
    plt = _import_matplotlib()
    panel_w = fig_width_for(len(methods), base=5.0, per_group=0.8, cap=9.0)
    fig, (ax_cost, ax_emb) = plt.subplots(1, 2, figsize=(2 * panel_w, 4.2))
    x = np.arange(len(methods))

    ax_cost.bar(x, opt_mean, color="#1f77b4", label="Optimizer reasoning (agent)")
    ax_cost.bar(x, trial_mean, bottom=opt_mean, color="#ff7f0e", label="Trial evaluation (RAG + judge)")
    ax_cost.errorbar(x, total_mean, yerr=total_err, fmt="none", ecolor="black", capsize=4, lw=1.0,
                     zorder=5, label="min-max across seeds")
    style_method_xticks(ax_cost, methods)
    ax_cost.set_ylabel("Mean search cost per seed (USD)")
    ax_cost.set_title("Search cost by source")
    ax_cost.set_ylim(0, float((total_mean + total_err[1]).max()) * 1.15)
    ax_cost.legend(loc="upper right", frameon=False)
    ax_cost.grid(axis="y", alpha=0.3)

    ax_emb.bar(x, emb_mean, color="#2ca02c", yerr=emb_err, capsize=4,
               error_kw={"ecolor": "black", "lw": 1.0})
    style_method_xticks(ax_emb, methods)
    ax_emb.set_ylabel("Mean embedding tokens per seed (millions)")
    ax_emb.set_title("Embedding tokens (cache-aware)")
    ax_emb.set_ylim(0, float((emb_mean + emb_err[1]).max()) * 1.15)
    ax_emb.grid(axis="y", alpha=0.3)

    title_domain = f" ({domain})" if domain else ""
    fig.suptitle(f"Search cost and embedding footprint on UniDoc{title_domain}", fontsize=12, y=1.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
