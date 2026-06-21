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
METHOD_ORDER = ["agentic_score", "agentic_cost", "random", "bayesian"]

# Methods whose per-trial trajectory is meaningful. ``@k`` checkpoint
# variants inherit sequential-ness from their parent (they're a strict
# prefix of the parent's history) via ``_is_sequential``.
SEQUENTIAL = {"agentic_score", "agentic_cost", "random", "bayesian"}

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
    ``eval_usd`` for random/bayesian. Both name the same quantity.
    """
    if "eval_usd" in e:
        return float(e["eval_usd"])
    return float(e.get("total_llm_cost_usd", 0.0))


def _read_history(seed_dir: Path) -> list[dict]:
    path = RunLayout(base=seed_dir).history
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    out.sort(key=lambda e: int(e.get("trial_number", 0)))
    return out


def _read_benchmark(seed_dir: Path) -> dict | None:
    path = seed_dir / "benchmark_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
