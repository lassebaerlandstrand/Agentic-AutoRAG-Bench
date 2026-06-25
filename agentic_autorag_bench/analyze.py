"""Aggregate committed results into matrix-level figures and Table_1.md.

Reads ``<results_dir>/<method>/seed_<n>/{benchmark_results.json,
optimizer_meta.json, history.jsonl}`` for every (method, seed) the matrix
produced, plus any ``holdout_replays/run_NNN.json`` files written by
``replay-holdout`` (each is a full re-evaluation of the same best config
against the same hold-out QA).

The ``run`` command auto-emits every matrix figure as the matrix runs; this
module's ``analyze`` entry point is the re-render path — point it at a
committed ``results_dir`` and it rewrites every figure under
``<output_dir>/figures/`` without re-running the matrix.

Statistical method (``mean_sd``, ``aggregate_by_method``): for each
(method, seed), the per-run EM / F1 / Judge mean is computed from each
hold-out eval; the matrix figures then plot ``mean ± SD`` across those
N runs (typically N=3 after ``replay-holdout``). When N=1 the SD is
zero so error bars vanish — graceful degradation for dirs that haven't
been replayed yet. ``bootstrap_ci`` is kept available for plots that
still need per-question-row resampling (none currently).

The figure writers (``write_markdown_table``, ``write_holdout_scores_figure``,
``write_efficiency_figure``) are also called by ``plots.make_matrix_figures``
so the in-run hook and the standalone ``analyze`` command share one
implementation. They take an explicit ``out_path`` so plots.py can route each
one into ``figures/``. Shared method colors / display names / layout helpers
live in ``_figstyle``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench._figstyle import (
    add_bar_value_labels,
    apply_paper_style,
    color_for,
    display_label,
    fig_width_for,
    legend_outside,
    style_method_xticks,
)

logger = logging.getLogger("agentic_autorag_bench.run")

N_BOOTSTRAP = 1000
CI_ALPHA = 0.05

# Stable display order shared by the Markdown table and figure writers so the
# paper's narrative ("agentic vs. MO-TPE/random") reads the same everywhere.
METHOD_ORDER = [
    "agentic_score",
    "agentic_cost",
    "agentic_nokb",
    "agentic_nodiag",
    "agentic_opro",
    "motpe",
    "motpe_warm",
    "qlognehvi",
    "random",
]


def _order_methods_for_analyze(method_names) -> list[str]:
    """Mirrors ``plots._order_methods``: keeps base methods in declared order
    and groups each base's ``@k`` checkpoint variants immediately after it,
    sorted by ascending k. Names outside this scheme come last alphabetically."""
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

    Returns the canonical name when ``run.py`` wrote the sidecar metadata
    file; falls back to ``"Benchmark"`` for trees that lack it.
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
    return _order_methods_for_analyze(stats.keys())


@dataclass
class MethodResult:
    method: str
    seed: int | None
    benchmarks: list[dict]  # run 1 first, then replays in numbered order
    optimizer_meta: dict
    history: list[dict]

    @property
    def benchmark(self) -> dict:
        """The primary (run 1) hold-out result. Kept for back-compat with
        callers that need a single benchmark dict — cost-per-question on
        the score-vs-cost Pareto, retrieval-metric reads, etc."""
        return self.benchmarks[0]

    @staticmethod
    def _scoring_rows_for(benchmark: dict) -> list[dict]:
        """Per-question rows used for aggregation, with the union-exclusion
        applied. ``excluded_question_ids`` is populated by the post-matrix
        union pass — any id in that list is dropped from every method so all
        rows aggregate over the same denominator. Falls back to "all rows"
        when the registry is absent (older results dir)."""
        excluded = set(benchmark.get("excluded_question_ids") or [])
        return [r for r in benchmark.get("per_question", []) if r.get("id") not in excluded]

    @staticmethod
    def _row_em(rows: list[dict]) -> np.ndarray:
        return np.array([float(r.get("em", 0.0)) for r in rows])

    @staticmethod
    def _row_f1(rows: list[dict]) -> np.ndarray:
        return np.array([float(r.get("f1", 0.0)) for r in rows])

    @staticmethod
    def _row_judge(rows: list[dict]) -> np.ndarray:
        # Schema (BenchmarkResult.per_question -> QAResult): ``judge: int | None``
        # where 1=correct, 0=wrong, -1=NO_ANSWER (the model abstained /
        # "insufficient context"), None=judge call failed (timeout / parse
        # error). The hold-out evaluator calls the judge for *every* row when
        # judge_model is set.
        #
        # Abstention (-1) counts as INCORRECT (0.0), matching the optimizer's
        # trial-time accuracy (examiner: correct / valid, abstention in the
        # denominator but not the numerator) — so hold-out scores the same
        # objective the search optimized. Dropping abstentions instead would
        # inflate any config that games the judge by refusing to answer hard
        # questions. Only None (a measurement failure, surfaced separately as
        # n_judge_invalid) is dropped via NaN so np.nanmean skips it.
        out: list[float] = []
        for r in rows:
            v = r.get("judge")
            if v is None:
                out.append(np.nan)
            elif v == 1:
                out.append(1.0)
            else:  # 0 (wrong) or -1 (abstention)
                out.append(0.0)
        return np.array(out)

    def per_run_means(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (em_per_run, f1_per_run, judge_per_run) with one mean per
        hold-out eval. Length equals ``len(self.benchmarks)``.

        Each entry is the mean across union-excluded per-question rows for
        that run. Judge means use ``np.nanmean`` to skip judge-failed rows
        (see ``_row_judge``); a run with zero valid judge rows reports NaN.
        """
        em_means: list[float] = []
        f1_means: list[float] = []
        judge_means: list[float] = []
        for bm in self.benchmarks:
            rows = self._scoring_rows_for(bm)
            em_arr = self._row_em(rows)
            f1_arr = self._row_f1(rows)
            jg_arr = self._row_judge(rows)
            em_means.append(float(em_arr.mean()) if em_arr.size else 0.0)
            f1_means.append(float(f1_arr.mean()) if f1_arr.size else 0.0)
            if jg_arr.size and not np.all(np.isnan(jg_arr)):
                judge_means.append(float(np.nanmean(jg_arr)))
            else:
                judge_means.append(float("nan"))
        return np.array(em_means), np.array(f1_means), np.array(judge_means)

    # Pooled per-question accessors — kept so plots.py and tests that
    # used them on a single-run MethodResult still work. Concatenates
    # across replays, which is what bootstrap-on-pooled-rows callers
    # would have asked for anyway.

    @property
    def per_question_em(self) -> np.ndarray:
        return (
            np.concatenate([self._row_em(self._scoring_rows_for(bm)) for bm in self.benchmarks])
            if self.benchmarks
            else np.array([])
        )

    @property
    def per_question_f1(self) -> np.ndarray:
        return (
            np.concatenate([self._row_f1(self._scoring_rows_for(bm)) for bm in self.benchmarks])
            if self.benchmarks
            else np.array([])
        )

    @property
    def per_question_judge(self) -> np.ndarray:
        return (
            np.concatenate([self._row_judge(self._scoring_rows_for(bm)) for bm in self.benchmarks])
            if self.benchmarks
            else np.array([])
        )


_NON_METHOD_DIRS = {"figures", ".shared_cache"}


def load_results(results_dir: Path) -> list[MethodResult]:
    out: list[MethodResult] = []
    method_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name not in _NON_METHOD_DIRS)
    for method_dir in method_dirs:
        # The per-method ``figures/`` subdir is co-located with seed dirs in
        # the auto-run layout; exclude it from the seed scan.
        seed_dirs = sorted(p for p in method_dir.iterdir() if p.is_dir() and p.name not in _NON_METHOD_DIRS)
        for seed_dir in seed_dirs:
            bench_path = seed_dir / "benchmark_results.json"
            meta_path = seed_dir / "optimizer_meta.json"
            history_path = RunLayout(base=seed_dir).history
            if not bench_path.exists():
                logger.warning("Skipping %s/%s: no benchmark_results.json", method_dir.name, seed_dir.name)
                continue
            # A process killed mid-write can leave a truncated json anywhere in
            # this seed tree. Skip the whole (method, seed) on any read error
            # rather than abort the matrix render — the next --resume rewrites it.
            try:
                benchmarks: list[dict] = [json.loads(bench_path.read_text(encoding="utf-8"))]
                replays_dir = seed_dir / "holdout_replays"
                if replays_dir.is_dir():
                    for rp in sorted(replays_dir.glob("run_*.json")):
                        benchmarks.append(json.loads(rp.read_text(encoding="utf-8")))
                optimizer_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                history = []
                if history_path.exists():
                    for line in history_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            history.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "Skipping %s/%s: unreadable/corrupt result file", method_dir.name, seed_dir.name, exc_info=True
                )
                continue
            seed: int | None = int(seed_dir.name.removeprefix("seed_")) if seed_dir.name.startswith("seed_") else None
            out.append(
                MethodResult(
                    method=method_dir.name,
                    seed=seed,
                    benchmarks=benchmarks,
                    optimizer_meta=optimizer_meta,
                    history=history,
                )
            )
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = CI_ALPHA) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for ``values`` under the empirical bootstrap.

    Retained for callers that still want per-question-row resampling on a
    single eval. The matrix figures use ``mean_sd`` over per-run means
    instead, since hold-out replays now provide that signal directly.

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


def mean_sd(values: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mean, mean-sd, mean+sd)`` over ``values``.

    Reports sample standard deviation (``ddof=1``) so a 3-replay matrix
    answers "how spread are these 3 reps?" rather than the asymptotic
    population SD. With N<2 the SD is undefined; this returns
    ``(mean, mean, mean)`` so the chart writers' asymmetric ``yerr``
    plumbing produces a zero-height error bar instead of erroring out.

    The triple shape matches ``bootstrap_ci`` so call sites that
    destructure ``(mean, lo, hi)`` keep working unchanged.

    NaN entries are dropped before stats — judge-failed rows propagate as
    NaN through ``per_run_means`` for a run where every judge call timed
    out, and shouldn't poison the cross-run aggregate.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0
    clean = values[~np.isnan(values)] if values.dtype.kind == "f" else values
    if clean.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(clean.mean())
    if clean.size < 2:
        return mean, mean, mean
    sd = float(np.std(clean, ddof=1))
    return mean, mean - sd, mean + sd


def aggregate_by_method(results: list[MethodResult]) -> dict[str, dict]:
    """Aggregate held-out scores per method as ``mean ± SD across runs``.

    For each (method, seed) we compute one EM / F1 / Judge mean per
    hold-out eval (``MethodResult.per_run_means``). Across seeds we
    concatenate those per-run means and report ``mean_sd`` over the
    union — so a 1-seed × 3-replay matrix yields N=3 per metric, a
    2-seed × 3-replay matrix yields N=6, etc.

    Sample SD (ddof=1) is used; the matrix figures' captions advertise
    this so reviewers can sanity-check the math.

    Search-side stats (wall_clock_s, optimizer_usd, trial_usd_total,
    token totals) are unchanged — they're per-seed scalars taken from
    ``optimizer_meta.json`` and unaffected by hold-out replays.

    Retrieval metrics are read from the primary benchmark dict; they're
    deterministic given the same config (the retrieval index doesn't
    re-rank between hold-out evals).
    """
    by_method: dict[str, list[MethodResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    out: dict[str, dict] = {}
    for method, runs in by_method.items():
        em_chunks: list[np.ndarray] = []
        f1_chunks: list[np.ndarray] = []
        judge_chunks: list[np.ndarray] = []
        for r in runs:
            em_run, f1_run, judge_run = r.per_run_means()
            em_chunks.append(em_run)
            f1_chunks.append(f1_run)
            judge_chunks.append(judge_run)
        em_runs = np.concatenate(em_chunks) if em_chunks else np.array([])
        f1_runs = np.concatenate(f1_chunks) if f1_chunks else np.array([])
        judge_runs = np.concatenate(judge_chunks) if judge_chunks else np.array([])
        n_runs_per_seed = [len(r.benchmarks) for r in runs]

        wall_clocks = [float(r.optimizer_meta.get("wall_clock_s", 0.0)) for r in runs]
        optim_usds = [float(r.optimizer_meta.get("optimizer_usd", 0.0)) for r in runs]
        trial_usds = [float(r.optimizer_meta.get("trial_usd_total", 0.0)) for r in runs]
        prompt_toks = [int(r.optimizer_meta.get("prompt_tokens", 0)) for r in runs]
        completion_toks = [int(r.optimizer_meta.get("completion_tokens", 0)) for r in runs]
        embed_toks = [int(r.optimizer_meta.get("embedding_tokens", 0)) for r in runs]

        retrieval_fields = (
            "mrr_first",
            "mrr_complete",
            "joint_recall_at_2",
            "joint_recall_at_5",
            "joint_recall_at_10",
        )
        retrieval_means: dict[str, float] = {}
        for fname in retrieval_fields:
            vals = [float(v) for v in (r.benchmark.get(fname) for r in runs) if v is not None]
            retrieval_means[fname] = float(np.mean(vals)) if vals else 0.0

        out[method] = {
            "n_seeds": len(runs),
            "n_runs_per_seed": n_runs_per_seed,
            "n_runs_total": int(sum(n_runs_per_seed)),
            "em": mean_sd(em_runs),
            "f1": mean_sd(f1_runs),
            "judge": mean_sd(judge_runs),
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
    """Per-method results as a Markdown pipe-table.

    EM / F1 / Judge columns are formatted as ``mean ± SD`` across the
    hold-out replays for that method (typically N=3 after
    ``replay-holdout``). ``N`` is reported alongside so a method that
    hasn't been replayed yet (N=1, SD=0) is visually obvious.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for m in _order_methods_for_analyze(stats.keys()):
        s = stats[m]
        em_m, em_lo, em_hi = s["em"]
        f1_m, f1_lo, f1_hi = s["f1"]
        j_m, j_lo, j_hi = s["judge"]
        em_sd = em_hi - em_m
        f1_sd = f1_hi - f1_m
        j_sd = j_hi - j_m
        n_runs = s.get("n_runs_total", s.get("n_seeds", 1))
        search_usd = s["optimizer_usd_mean"] + s["trial_usd_mean"]
        rows.append(
            f"| {display_label(m)} "
            f"| {em_m:.3f} ± {em_sd:.3f} "
            f"| {f1_m:.3f} ± {f1_sd:.3f} "
            f"| {j_m:.3f} ± {j_sd:.3f} "
            f"| {n_runs} "
            f"| {s['joint_recall_at_2']:.3f} "
            f"| {s['joint_recall_at_5']:.3f} "
            f"| {s['mrr_complete']:.3f} "
            f"| {s['mrr_first']:.3f} "
            f"| {_fmt_tok(s.get('prompt_tokens_mean', 0.0))} "
            f"| {_fmt_tok(s.get('completion_tokens_mean', 0.0))} "
            f"| {_fmt_tok(s.get('embedding_tokens_mean', 0.0))} "
            f"| ${s['optimizer_usd_mean']:.4f} "
            f"| ${s['trial_usd_mean']:.4f} "
            f"| ${search_usd:.4f} "
            f"| {s['wall_clock_s_mean']:.0f}s¹ |"
        )
    header = (
        "| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | "
        "LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(rows) if rows else "| _(no results yet)_ | | | | | | | | | | | | | | | |"
    text = (
        f"# {benchmark_pretty_name} held-out scores\n\n"
        "EM / F1 / Judge columns: mean ± sample SD (ddof=1) across N hold-out "
        "replays per method. N=1 means the method has only its end-of-search "
        "eval; run `replay-holdout` to bring it to N=3. Token / cost / wall "
        "columns are mean across seeds (search-side, unaffected by hold-out "
        "replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to "
        "find the winning config).\n\n"
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

    Error bars are ``± sample SD`` across N hold-out replays per method
    (run ``replay-holdout`` to populate N>=2 replays; until then N=1 and
    the bars have zero-height whiskers). This is the primary "which method
    scores best?" view — the LaTeX table carries the same numbers but
    the grouping makes cross-method comparison readable at a glance.
    """
    apply_paper_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = _ordered_methods(stats)
    if not methods:
        return

    metrics = [("em", "Exact Match"), ("f1", "Token F1"), ("judge", "LLM Judge")]
    fig, ax = plt.subplots(figsize=(fig_width_for(len(methods)), 4.4))
    x = np.arange(len(methods))
    bar_width = 0.27

    for i, (metric, label) in enumerate(metrics):
        means = [stats[m][metric][0] for m in methods]
        lo_err = [stats[m][metric][0] - stats[m][metric][1] for m in methods]
        hi_err = [stats[m][metric][2] - stats[m][metric][0] for m in methods]
        # Bars colored per metric; methods are distinguished on the x-axis.
        bars = ax.bar(
            x + (i - 1) * bar_width,
            means,
            bar_width,
            yerr=[lo_err, hi_err],
            capsize=2,
            label=label,
        )
        # Value labels sit just above each bar's upper error cap.
        add_bar_value_labels(ax, bars, y_offsets=hi_err)

    style_method_xticks(ax, methods)
    ax.set_ylabel("Score (held-out)")
    ax.set_ylim(0, 1.0)
    # N can differ per method (e.g. a method not yet replayed). The table
    # reports exact per-row N; here just note the range when it varies.
    n_values = sorted({stats[m].get("n_runs_total", 1) for m in methods})
    n_label = f"N={n_values[0]}" if len(n_values) == 1 else f"N={n_values[0]}–{n_values[-1]}"
    legend_outside(ax, ncol=3, title=f"Held-out evaluation scores (mean ± SD, {n_label})")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def write_efficiency_figure(stats: dict[str, dict], out_path: Path) -> None:
    """Score-vs-cost and score-vs-wallclock scatter (1x2 panel).

    Each method is one point; vertical bars are ± SD of held-out Judge
    across hold-out replays (zero if only the end-of-search eval exists),
    horizontal bars are the per-seed std of the corresponding axis (zero
    with a single seed).

    This is the headline "value" plot: a method in the upper-left corner of either
    panel is the best score per dollar / per second.
    """
    apply_paper_style()
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
        cost_seeds = [o + t for o, t in zip(s["optimizer_usd_list"], s["trial_usd_list"], strict=True)]
        wall_seeds = s["wall_clock_s_list"]
        cost_m = float(np.mean(cost_seeds)) if cost_seeds else 0.0
        wall_m = float(np.mean(wall_seeds)) if wall_seeds else 0.0
        # Per-seed spread indicator; 0 with a single seed.
        cost_err = float(np.std(cost_seeds)) if len(cost_seeds) > 1 else 0.0
        wall_err = float(np.std(wall_seeds)) if len(wall_seeds) > 1 else 0.0
        color = color_for(m)
        label = display_label(m)
        ax_cost.errorbar(cost_m, judge_m, xerr=cost_err, yerr=score_yerr, fmt="o", color=color, capsize=3, label=label)
        ax_time.errorbar(wall_m, judge_m, xerr=wall_err, yerr=score_yerr, fmt="o", color=color, capsize=3, label=label)

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


def analyze(results_dir: str | Path, output_dir: str | Path) -> None:
    """Regenerate matrix-level figures + Table_1.md from a committed results tree.

    Writes everything under ``<output_dir>/figures/`` to match the in-run
    layout (``run.py`` emits the same files under
    ``<results_dir>/figures/``). When the user passes
    ``--output <results_dir>`` (or omits the flag), this is exactly what the
    run hook produced — but the call still rewrites, so it is the canonical
    re-render path after edits to the figure code.
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
