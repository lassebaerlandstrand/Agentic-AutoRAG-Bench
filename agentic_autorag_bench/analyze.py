"""Aggregate committed results into paper artifacts.

Reads ``results/<method>/seed_<n>/{benchmark_results.json, optimizer_meta.json,
history.jsonl}`` for every (method, seed) the matrix produced. Emits:

- ``paper_artifacts/Table_1.tex``: per-method held-out scores with bootstrap
  95% CIs, plus optimizer/trial $ split and wall-clock.
- ``paper_artifacts/figure_trajectory.pdf``: best-so-far vs trial number for
  the three sequential methods (mean ± std across seeds). AutoRAG variants
  are excluded — their trajectory shape is per-node greedy, not per-trial.

Statistical method: nonparametric bootstrap (1000 boots) on the held-out
per-question EM/F1/Judge scores, paired across methods that share the same
test questions. Reported as ``mean [lo, hi]``.
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


@dataclass
class MethodResult:
    method: str
    seed: int | None
    benchmark: dict
    optimizer_meta: dict
    history: list[dict]

    @property
    def per_question_em(self) -> np.ndarray:
        return np.array([float(r.get("em", 0.0)) for r in self.benchmark.get("per_question", [])])

    @property
    def per_question_f1(self) -> np.ndarray:
        return np.array([float(r.get("f1", 0.0)) for r in self.benchmark.get("per_question", [])])

    @property
    def per_question_judge(self) -> np.ndarray:
        # Schema (BenchmarkResult.per_question -> QAResult): ``judge: int | None``
        # where 1=correct, 0=incorrect, None=disabled or parse-fail. For aggregate
        # accuracy, fall back to EM when judge is None (matches framework's
        # OpenEndedEvaluator: judge only runs when EM=0, so judge=None implies EM
        # already decided the verdict — which we infer from em>0.5).
        out = []
        for r in self.benchmark.get("per_question", []):
            v = r.get("judge")
            if v == 1:
                out.append(1.0)
            elif v == 0:
                out.append(0.0)
            else:  # judge is None — verdict already set by EM
                out.append(1.0 if float(r.get("em", 0.0)) > 0.5 else 0.0)
        return np.array(out)


def load_results(results_dir: Path) -> list[MethodResult]:
    out: list[MethodResult] = []
    for method_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
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
    """Return (mean, lo, hi) for ``values`` under the empirical bootstrap."""
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed=42)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(values.mean()), lo, hi


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

        out[method] = {
            "n_seeds": len(runs),
            "em": bootstrap_ci(em_pool),
            "f1": bootstrap_ci(f1_pool),
            "judge": bootstrap_ci(judge_pool),
            "mrr": float(np.mean([float(r.benchmark.get("mrr", 0.0)) for r in runs])),
            "wall_clock_s_mean": float(np.mean(wall_clocks)) if wall_clocks else 0.0,
            "optimizer_usd_mean": float(np.mean(optim_usds)) if optim_usds else 0.0,
            "trial_usd_mean": float(np.mean(trial_usds)) if trial_usds else 0.0,
            "wall_clock_s_list": wall_clocks,
            "optimizer_usd_list": optim_usds,
            "trial_usd_list": trial_usds,
        }
    return out


def write_latex_table(stats: dict[str, dict], out_path: Path) -> None:
    """Method × {EM, F1, Judge, MRR, $, wall-clock} as a booktabs table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    method_order = ["agentic", "random", "bayesian", "autorag_mcq", "autorag_ragas"]
    rows = []
    for m in method_order:
        if m not in stats:
            continue
        s = stats[m]
        em_m, em_lo, em_hi = s["em"]
        f1_m, f1_lo, f1_hi = s["f1"]
        j_m, j_lo, j_hi = s["judge"]
        rows.append(
            f"  {m.replace('_', '-')} & "
            f"{em_m:.3f} [{em_lo:.3f}, {em_hi:.3f}] & "
            f"{f1_m:.3f} [{f1_lo:.3f}, {f1_hi:.3f}] & "
            f"{j_m:.3f} [{j_lo:.3f}, {j_hi:.3f}] & "
            f"{s['mrr']:.3f} & "
            f"\\${s['optimizer_usd_mean']:.2f} / \\${s['trial_usd_mean']:.2f} & "
            f"{s['wall_clock_s_mean']:.0f}s \\\\"
        )
    body = "\n".join(rows) if rows else "  \\multicolumn{6}{c}{no results yet} \\\\"
    table = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\caption{HotpotQA-distractor held-out scores. Mean and bootstrap 95\\% CIs over per-question metrics, "
        "pooled across seeds. Cost column: optimizer-side / trial-side USD. Wall-clock is mean across seeds.}\n"
        "  \\label{tab:hotpot_main}\n"
        "  \\begin{tabular}{lccccrr}\n"
        "  \\toprule\n"
        "  Method & EM & Token-F1 & Judge & MRR & Cost (\\$) & Wall \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    out_path.write_text(table, encoding="utf-8")
    logger.info("wrote %s", out_path)


METHOD_ORDER = ["agentic", "random", "bayesian", "autorag_mcq", "autorag_ragas"]


def _ordered_methods(stats: dict[str, dict]) -> list[str]:
    return [m for m in METHOD_ORDER if m in stats]


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
    fig.savefig(out_path)
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

    fig, (ax_cost, ax_time) = plt.subplots(1, 2, figsize=(9.5, 3.8))

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
    fig.savefig(out_path)
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
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def analyze(results_dir: str | Path, output_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    results = load_results(results_dir)
    if not results:
        logger.warning("No results found under %s — nothing to analyze", results_dir)
        return
    stats = aggregate_by_method(results)
    write_latex_table(stats, output_dir / "Table_1.tex")
    write_holdout_scores_figure(stats, output_dir / "figure_holdout_scores.pdf")
    write_efficiency_figure(stats, output_dir / "figure_efficiency.pdf")
    write_trajectory_figure(results, output_dir / "figure_trajectory.pdf")
