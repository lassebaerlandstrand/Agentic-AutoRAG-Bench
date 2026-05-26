"""Aggregate committed results into matrix-level figures and Table_1.md.

Reads ``<results_dir>/<method>/seed_<n>/{benchmark_results.json,
optimizer_meta.json, history.jsonl}`` for every (method, seed) the matrix
produced.

The ``run`` command auto-emits every matrix figure as the matrix runs; this
module's ``analyze`` entry point is the re-render path — point it at a
committed ``results_dir`` and it rewrites every figure under
``<output_dir>/figures/`` without re-running the matrix.

Statistical method (``bootstrap_ci``, ``aggregate_by_method``): nonparametric
bootstrap (1000 boots) on the held-out per-question EM/F1/Judge scores,
paired across methods that share the same test questions. Reported as
``mean [lo, hi]``.

The figure writers (``write_markdown_table``, ``write_holdout_scores_figure``,
``write_efficiency_figure``, ``write_trajectory_figure``) are also called by
``plots.make_matrix_figures`` so the in-run hook and the standalone
``analyze`` command share one implementation. They take an explicit
``out_path`` so plots.py can route each one into ``figures/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("agentic_autorag_bench.run")

N_BOOTSTRAP = 1000
CI_ALPHA = 0.05

# Stable display order shared by the LaTeX, Markdown, and figure writers so the
# paper's narrative ("agentic vs. random/bayesian vs. AutoRAG") reads the same
# everywhere.
METHOD_ORDER = ["agentic_score", "agentic_cost", "random", "bayesian", "autorag_our_exam", "autorag_ragas"]

# Display name per benchmark adapter key. Used in the Markdown table title so
# the paper-ready file reads with the canonical dataset+variant string, not the
# snake_case adapter id. Missing entries fall back to the adapter name.
BENCHMARK_PRETTY_NAMES = {
    "hotpot_qa": "HotpotQA-distractor",
    "musique": "MuSiQue-Ans",
    "multihop_rag": "MultiHop-RAG",
}


def read_benchmark_pretty_name(results_dir: Path) -> str:
    """Resolve the benchmark's display name from ``bench_metadata.json``.

    Returns the canonical name when ``run.py`` wrote a sidecar metadata file
    (every run since multi-benchmark support landed); falls back to ``"Benchmark"``
    for older trees that predate it.
    """
    meta_path = Path(results_dir) / "bench_metadata.json"
    if not meta_path.exists():
        return "Benchmark"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Benchmark"
    name = (meta.get("benchmark") or {}).get("name") or ""
    return BENCHMARK_PRETTY_NAMES.get(name, name or "Benchmark")


def _ordered_methods(stats: dict[str, dict]) -> list[str]:
    return [m for m in METHOD_ORDER if m in stats]


@dataclass
class MethodResult:
    method: str
    seed: int | None
    benchmark: dict
    optimizer_meta: dict
    history: list[dict]

    def _scoring_rows(self) -> list[dict]:
        """Per-question rows used for bootstrap, with the union-exclusion
        applied. ``excluded_question_ids`` is populated by the post-matrix
        union pass — any id in that list is dropped from every method so all
        rows bootstrap over the same denominator. Falls back to "all rows"
        when the registry is absent (older results dir)."""
        excluded = set(self.benchmark.get("excluded_question_ids") or [])
        return [
            r for r in self.benchmark.get("per_question", [])
            if r.get("id") not in excluded
        ]

    @property
    def per_question_em(self) -> np.ndarray:
        return np.array([float(r.get("em", 0.0)) for r in self._scoring_rows()])

    @property
    def per_question_f1(self) -> np.ndarray:
        return np.array([float(r.get("f1", 0.0)) for r in self._scoring_rows()])

    @property
    def per_question_judge(self) -> np.ndarray:
        # Schema (BenchmarkResult.per_question -> QAResult): ``judge: int | None``
        # where 1=correct, 0=incorrect, None=parse/timeout failure. The hold-out
        # evaluator (FreeFormEvaluator) calls the judge for *every* row when
        # judge_model is set — unlike the framework's trial-time evaluator which
        # only calls the judge on EM=0 — so judge=None here means the judge call
        # failed (timeout, parse error, content filter), not "EM already
        # decided". Drop those rows from the denominator: return NaN so callers
        # using np.nanmean treat them as missing rather than biased.
        out = []
        for r in self._scoring_rows():
            v = r.get("judge")
            if v == 1:
                out.append(1.0)
            elif v == 0:
                out.append(0.0)
            else:
                out.append(np.nan)
        return np.array(out)


_NON_METHOD_DIRS = {"figures", ".shared_cache"}


def load_results(results_dir: Path) -> list[MethodResult]:
    out: list[MethodResult] = []
    method_dirs = sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name not in _NON_METHOD_DIRS
    )
    for method_dir in method_dirs:
        # The per-method ``figures/`` subdir is co-located with seed dirs in
        # the auto-run layout; exclude it from the seed scan.
        seed_dirs = sorted(
            p for p in method_dir.iterdir()
            if p.is_dir() and p.name not in _NON_METHOD_DIRS
        )
        for seed_dir in seed_dirs:
            bench_path = seed_dir / "benchmark_results.json"
            meta_path = seed_dir / "optimizer_meta.json"
            history_path = seed_dir / "history.jsonl"
            if not bench_path.exists():
                logger.warning("Skipping %s/%s: no benchmark_results.json", method_dir.name, seed_dir.name)
                continue
            benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
            optimizer_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            history = []
            if history_path.exists():
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
            seed: int | None
            if seed_dir.name.startswith("seed_"):
                seed = int(seed_dir.name.removeprefix("seed_"))
            else:
                seed = None
            out.append(
                MethodResult(
                    method=method_dir.name,
                    seed=seed,
                    benchmark=benchmark,
                    optimizer_meta=optimizer_meta,
                    history=history,
                )
            )
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = CI_ALPHA) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for ``values`` under the empirical bootstrap.

    NaN entries are dropped before resampling — ``per_question_judge`` uses NaN
    to flag rows where the judge call itself failed (timeout / parse error),
    which would otherwise propagate through ``np.choice``-mean.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0
    clean = values[~np.isnan(values)] if values.dtype.kind == "f" else values
    if clean.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed=42)
    boots = rng.choice(clean, size=(n_boot, clean.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(clean.mean()), lo, hi


def aggregate_by_method(results: list[MethodResult]) -> dict[str, dict]:
    """Pool per-question scores across seeds, then bootstrap.

    For a method with K seeds × N questions, treats all K*N scores as the
    sample. Reports mean = average across seeds and questions, CI via bootstrap.

    Caveat: pooled bootstrap treats per-question rows as i.i.d., which
    slightly underestimates variance when K>1 seeds share the same RAG
    pipeline (within-seed correlation). A cluster bootstrap over seed-means
    would be tighter; we pool because for K=3 seeds the cluster bootstrap is
    very high-variance. Cross-method comparisons should use a paired test
    (paired bootstrap of per-question differences) rather than reading
    overlapping CIs as "no significant difference".
    """
    by_method: dict[str, list[MethodResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    out: dict[str, dict] = {}
    for method, runs in by_method.items():
        em_pool = np.concatenate([r.per_question_em for r in runs]) if runs else np.array([])
        f1_pool = np.concatenate([r.per_question_f1 for r in runs]) if runs else np.array([])
        judge_pool = np.concatenate([r.per_question_judge for r in runs]) if runs else np.array([])

        wall_clocks = [float(r.optimizer_meta.get("wall_clock_s", 0.0)) for r in runs]
        optim_usds = [float(r.optimizer_meta.get("optimizer_usd", 0.0)) for r in runs]
        trial_usds = [float(r.optimizer_meta.get("trial_usd_total", 0.0)) for r in runs]
        prompt_toks = [int(r.optimizer_meta.get("prompt_tokens", 0)) for r in runs]
        completion_toks = [int(r.optimizer_meta.get("completion_tokens", 0)) for r in runs]
        embed_toks = [int(r.optimizer_meta.get("embedding_tokens", 0)) for r in runs]

        retrieval_fields = (
            "mrr_first", "mrr_complete",
            "joint_recall_at_2", "joint_recall_at_5", "joint_recall_at_10",
        )
        retrieval_means: dict[str, float] = {}
        for fname in retrieval_fields:
            vals = [float(v) for v in (r.benchmark.get(fname) for r in runs) if v is not None]
            retrieval_means[fname] = float(np.mean(vals)) if vals else 0.0

        out[method] = {
            "n_seeds": len(runs),
            "em": bootstrap_ci(em_pool),
            "f1": bootstrap_ci(f1_pool),
            "judge": bootstrap_ci(judge_pool),
            **retrieval_means,
            "wall_clock_s_mean": float(np.mean(wall_clocks)) if wall_clocks else 0.0,
            "optimizer_usd_mean": float(np.mean(optim_usds)) if optim_usds else 0.0,
            "trial_usd_mean": float(np.mean(trial_usds)) if trial_usds else 0.0,
            "prompt_tokens_mean": float(np.mean(prompt_toks)) if prompt_toks else 0.0,
            "completion_tokens_mean": float(np.mean(completion_toks)) if completion_toks else 0.0,
            "embedding_tokens_mean": float(np.mean(embed_toks)) if embed_toks else 0.0,
            "wall_clock_s_list": wall_clocks,
            "optimizer_usd_list": optim_usds,
            "trial_usd_list": trial_usds,
            "prompt_tokens_list": prompt_toks,
            "completion_tokens_list": completion_toks,
            "embedding_tokens_list": embed_toks,
        }
    return out


def write_markdown_table(
    stats: dict[str, dict],
    out_path: Path,
    *,
    benchmark_pretty_name: str = "Benchmark",
) -> None:
    """Per-method results as a Markdown pipe-table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for m in METHOD_ORDER:
        if m not in stats:
            continue
        s = stats[m]
        em_m, em_lo, em_hi = s["em"]
        f1_m, f1_lo, f1_hi = s["f1"]
        j_m, j_lo, j_hi = s["judge"]
        label = m.replace("_", "-")
        rows.append(
            f"| {label} "
            f"| {em_m:.3f} [{em_lo:.3f}, {em_hi:.3f}] "
            f"| {f1_m:.3f} [{f1_lo:.3f}, {f1_hi:.3f}] "
            f"| {j_m:.3f} [{j_lo:.3f}, {j_hi:.3f}] "
            f"| {s['joint_recall_at_2']:.3f} "
            f"| {s['joint_recall_at_5']:.3f} "
            f"| {s['mrr_complete']:.3f} "
            f"| {s['mrr_first']:.3f} "
            f"| {_fmt_tok(s.get('prompt_tokens_mean', 0.0))} "
            f"| {_fmt_tok(s.get('completion_tokens_mean', 0.0))} "
            f"| {_fmt_tok(s.get('embedding_tokens_mean', 0.0))} "
            f"| ${s['optimizer_usd_mean']:.4f} "
            f"| ${s['trial_usd_mean']:.4f} "
            f"| {s['wall_clock_s_mean']:.0f}s¹ |"
        )
    header = (
        "| Method | EM | Token-F1 | LLM Judge | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | "
        "LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Wall |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(rows) if rows else "| _(no results yet)_ | | | | | | | | | | | | | |"
    text = (
        f"# {benchmark_pretty_name} held-out scores\n\n"
        "Mean and 95% bootstrap CIs over per-question metrics, pooled across seeds. "
        "Token / cost / wall columns are mean across seeds.\n\n"
        f"{header}\n{body}\n\n"
        "¹ Wall-clock is reported for context only — rate limits and shared caches "
        "make it an unfair primary metric. Token counts are the recommended cost proxy.\n"
    )
    out_path.write_text(text, encoding="utf-8")
    logger.info("wrote %s", out_path)


def _fmt_tok(n: float) -> str:
    """Format a token count compactly (e.g. 12.3M / 456k / 789)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


def write_holdout_scores_figure(stats: dict[str, dict], out_path: Path) -> None:
    """Grouped bars of held-out EM, Token-F1, and LLM-Judge accuracy per method.

    Error bars use the bootstrap 95% CI from ``aggregate_by_method``. This is the
    primary "which method scores best?" view — the LaTeX table carries the same
    numbers but the grouping makes cross-method comparison readable at a glance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = _ordered_methods(stats)
    if not methods:
        return

    metrics = [("em", "Exact Match"), ("f1", "Token F1"), ("judge", "LLM Judge")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(methods))
    bar_width = 0.27

    for i, (metric, label) in enumerate(metrics):
        means = [stats[m][metric][0] for m in methods]
        # Asymmetric error bars from bootstrap CI.
        lo_err = [stats[m][metric][0] - stats[m][metric][1] for m in methods]
        hi_err = [stats[m][metric][2] - stats[m][metric][0] for m in methods]
        ax.bar(
            x + (i - 1) * bar_width,
            means,
            bar_width,
            yerr=[lo_err, hi_err],
            capsize=3,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "-") for m in methods])
    ax.set_ylabel("Score (held-out)")
    ax.set_title("Held-out evaluation scores (mean, 95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def write_efficiency_figure(stats: dict[str, dict], out_path: Path) -> None:
    """Score-vs-cost and score-vs-wallclock scatter (1x2 panel).

    Each method is one point; vertical bars are the held-out Judge CI, horizontal
    bars are the per-seed std of the corresponding axis (or zero for the
    deterministic AutoRAG variants, which have a single run).

    This is the headline "value" plot: a method in the upper-left corner of either
    panel is the best score per dollar / per second.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = _ordered_methods(stats)
    if not methods:
        return

    fig, (ax_cost, ax_time) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    for m in methods:
        s = stats[m]
        judge_m, judge_lo, judge_hi = s["judge"]
        score_yerr = [[judge_m - judge_lo], [judge_hi - judge_m]]
        # Total search cost = optimizer-side + trial-side per seed.
        cost_seeds = [
            o + t for o, t in zip(s["optimizer_usd_list"], s["trial_usd_list"], strict=True)
        ]
        wall_seeds = s["wall_clock_s_list"]
        cost_m = float(np.mean(cost_seeds)) if cost_seeds else 0.0
        wall_m = float(np.mean(wall_seeds)) if wall_seeds else 0.0
        # Use std as a simple per-seed spread indicator; 0 for n_seeds=1 (autorag).
        cost_err = float(np.std(cost_seeds)) if len(cost_seeds) > 1 else 0.0
        wall_err = float(np.std(wall_seeds)) if len(wall_seeds) > 1 else 0.0
        label = m.replace("_", "-")
        ax_cost.errorbar(cost_m, judge_m, xerr=cost_err, yerr=score_yerr, fmt="o", capsize=3, label=label)
        ax_time.errorbar(wall_m, judge_m, xerr=wall_err, yerr=score_yerr, fmt="o", capsize=3, label=label)

    for ax, xlabel, title in (
        (ax_cost, "Total search cost (USD)", "Score vs. cost"),
        (ax_time, "Wall-clock (s)", "Score vs. wall-clock"),
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Held-out LLM-Judge accuracy")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax_cost.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def write_trajectory_figure(results: list[MethodResult], out_path: Path) -> None:
    """Best-so-far vs trial-number for the sequential methods (mean ± std across seeds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[MethodResult]] = {}
    for r in results:
        if r.method in {"random", "bayesian", "agentic"}:
            by_method.setdefault(r.method, []).append(r)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for method, runs in by_method.items():
        if not runs:
            continue
        # Best-so-far per seed; pad to max length so np.stack works
        per_seed_curves: list[np.ndarray] = []
        for r in runs:
            scores = [float(h["score"]) for h in r.history]
            if not scores:
                continue
            running = np.maximum.accumulate(scores)
            per_seed_curves.append(running)
        if not per_seed_curves:
            continue
        max_len = max(len(c) for c in per_seed_curves)
        padded = np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in per_seed_curves])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best-so-far exam score")
    ax.set_title("Optimization trajectory (mean ± std across seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def analyze(results_dir: str | Path, output_dir: str | Path) -> None:
    """Regenerate matrix-level figures + Table_1.md from a committed results tree.

    Writes everything under ``<output_dir>/figures/`` to match the in-run
    layout (``run.py`` emits the same files under
    ``<results_dir>/figures/``). When the user passes
    ``--output <results_dir>`` (or omits the flag), this is exactly what the
    run hook produced — but the call still rewrites, so it is the canonical
    re-render path after edits to the figure code.

    Also writes the legacy single best-so-far trajectory figure
    ``figure_trajectory.png`` for backward compat with paper drafts that
    embed it under that name; the new ``best_so_far.png`` (written by
    ``make_matrix_figures``) is the same plot under the canonical name.
    """
    from agentic_autorag_bench.plots import make_matrix_figures

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    benchmark_pretty_name = read_benchmark_pretty_name(results_dir)
    make_matrix_figures(
        results_dir,
        figures_dir=figures_dir,
        benchmark_pretty_name=benchmark_pretty_name,
    )
    # Best-so-far trajectory is also emitted by make_matrix_figures as
    # ``best_so_far.png``; keep the legacy name alongside it for any
    # downstream consumer still referencing it. Skipped when hold-out scoring
    # is missing for every seed (load_results returns []), since the plot
    # would be empty.
    results = load_results(results_dir)
    if results:
        write_trajectory_figure(results, figures_dir / "figure_trajectory.png")
    else:
        logger.info(
            "analyze: no hold-out results under %s — trajectory + bars only; "
            "table requires per-question scores",
            results_dir,
        )
