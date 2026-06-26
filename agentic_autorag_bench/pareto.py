"""Cost-aware Pareto experiment: ``agentic_cost`` vs ``random`` vs ``motpe_warm`` on UniDoc.

Three cost-aware searches on the UniDoc (healthcare) PDF corpus, scored on the
optimizer's own self-generated exam — no held-out QA. ``agentic_cost`` is the full
agentic optimizer (Pareto-aware reasoning); ``random`` is the exploration floor and
the transfer source; ``motpe_warm`` is the two-objective MO-TPE warm-started from
``random`` (all of random's completed trials injected as a free, uncounted transfer
prior). All minimize the SAME ``mean_llm_cost_per_query_usd`` and maximize the SAME
exam accuracy on the SAME exam, so the comparison is fair. ``random`` runs before
``motpe_warm`` (a hard data dependency).

``make_pareto_figure`` renders a single-method Syftr-style scatter (a gray cloud of
every trial with the optimizer's self-marked Pareto frontier highlighted, numbered,
and described in a side legend). ``make_pareto_comparison_figure`` overlays every
method's frontier, and ``compute_pareto_hypervolumes`` scores each frontier against
a SHARED reference point (pooled across all methods) so the hypervolumes are
comparable. X-axis is deploy-time cost **per query**.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentic_autorag_bench._figstyle import apply_paper_style, color_for, display_label
from agentic_autorag_bench.plots import _import_matplotlib, _read_history

logger = logging.getLogger("agentic_autorag_bench.run")

_FRONTIER_COLORMAP = "tab10"
_FRONTIER_COLORMAP_LARGE = "tab20"


# --------------------------------------------------------------------- config


@dataclass
class ParetoConfig:
    """Thin entry config for the ``pareto`` command.

    Path conventions mirror ``BenchConfig``: ``project_config`` resolves
    relative to this config's directory; ``output_root`` and the project
    YAML's ``meta.corpus_path`` resolve relative to the current working
    directory (the framework's convention).
    """

    project_config_path: Path
    seed: int
    max_trials: int
    corpus_domain: str
    corpus_max_pdfs: int
    corpus_max_images: int
    output_root: Path

    @classmethod
    def load(cls, config_path: str | Path) -> ParetoConfig:
        config_path = Path(config_path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_path = (config_path.parent / raw["project_config"]).resolve()
        corpus = raw.get("corpus") or {}
        return cls(
            project_config_path=project_path,
            seed=int(raw.get("seed", 1)),
            max_trials=int(raw["budget"]["max_trials"]),
            corpus_domain=str(corpus.get("domain", "healthcare")),
            corpus_max_pdfs=int(corpus.get("max_pdfs", 230)),
            corpus_max_images=int(corpus.get("max_images", 20)),
            output_root=Path(raw["output_root"]).resolve(),
        )


def _read_corpus_path(project_config_path: Path) -> Path:
    """``meta.corpus_path`` from the project YAML, resolved like the orchestrator."""
    raw = yaml.safe_load(Path(project_config_path).read_text(encoding="utf-8"))
    return Path(raw["meta"]["corpus_path"]).resolve()


def _ensure_corpus(corpus_path: Path, cfg: ParetoConfig) -> None:
    """Download the UniDoc corpus if ``corpus_path`` has no PDFs yet."""
    if corpus_path.is_dir() and any(corpus_path.glob("*.pdf")):
        logger.info(
            "UniDoc corpus already present at %s (%d PDFs); skipping download",
            corpus_path,
            len(list(corpus_path.glob("*.pdf"))),
        )
        return
    from agentic_autorag_bench.unidoc_corpus import download_unidoc_corpus

    logger.info(
        "UniDoc corpus missing at %s; downloading %d %s PDF(s) + %d image(s)",
        corpus_path,
        cfg.corpus_max_pdfs,
        cfg.corpus_domain,
        cfg.corpus_max_images,
    )
    download_unidoc_corpus(
        corpus_path,
        domain=cfg.corpus_domain,
        max_pdfs=cfg.corpus_max_pdfs,
        max_images=cfg.corpus_max_images,
    )


# --------------------------------------------------------------------- command


async def _stub_evaluator(_config):  # pragma: no cover - never invoked
    """AgenticOptimizer drives its own internal evaluator; this must not run."""
    raise RuntimeError("agentic_cost manages its own evaluator; the stub must not be called")


async def run_pareto(config_path: str | Path, *, figure_only: bool = False, resume: bool = False) -> None:
    """Run the cost-aware Pareto comparison — ``agentic_cost`` (full agentic) vs
    ``random`` (floor + transfer source) vs ``motpe_warm`` (two-objective MO-TPE
    warm-started from ``random``) — then render the multi-frontier figure with a
    shared-reference-point hypervolume.

    All three methods evaluate the SAME self-generated exam on the SAME corpus:
    the shared orchestrator (used for the random / motpe_warm evaluator) and
    agentic_cost's own orchestrator load the same project config, so the corpus
    index + exam.json cache is shared. Only the proposer differs. ``random`` runs
    before ``motpe_warm`` — a hard data dependency, since ``motpe_warm`` injects
    all of the paired ``random`` cell's completed trials as a free, uncounted
    transfer prior.
    """
    from agentic_autorag.litellm_runtime import configure_litellm_runtime
    from agentic_autorag.orchestrator import Orchestrator

    from agentic_autorag_bench.methods.agentic import AgenticOptimizer
    from agentic_autorag_bench.methods.motpe import MOTPESearch
    from agentic_autorag_bench.methods.random import RandomSearch
    from agentic_autorag_bench.run import _persist_search_result, _run_optimizer_with_ledger
    from agentic_autorag_bench.types import Budget

    cfg = ParetoConfig.load(config_path)
    seed_label = f"seed_{cfg.seed}"
    agentic_dir = cfg.output_root / "agentic_cost" / seed_label
    random_dir = cfg.output_root / "random" / seed_label
    motpe_warm_dir = cfg.output_root / "motpe_warm" / seed_label

    if not figure_only:
        configure_litellm_runtime()
        corpus_path = _read_corpus_path(cfg.project_config_path)
        _ensure_corpus(corpus_path, cfg)
        budget = Budget(max_trials=cfg.max_trials)

        logger.info("Setting up shared orchestrator for the Pareto comparison (exam generated on first run)")
        shared = Orchestrator(str(cfg.project_config_path))
        shared.evaluator.quiet_per_question = True
        try:
            await shared.setup()

            # agentic_cost — full agentic, cost-aware (self-contained orchestrator).
            agentic_dir.mkdir(parents=True, exist_ok=True)
            logger.info("=" * 60)
            logger.info("PARETO | agentic_cost | seed=%d | max_trials=%d", cfg.seed, cfg.max_trials)
            logger.info("=" * 60)
            agentic_opt = AgenticOptimizer(
                config_path=str(cfg.project_config_path),
                output_dir=str(agentic_dir),
                cost_aware=True,
                resume=resume,
            )
            sr_a = await agentic_opt.search(_stub_evaluator, budget, seed=cfg.seed)
            _persist_search_result(sr_a, agentic_dir)
            logger.info(
                "agentic_cost done | trials=%d | best_accuracy=%.3f",
                len(sr_a.history),
                max((h.answer_accuracy for h in sr_a.history), default=0.0),
            )

            # random — exploration floor AND motpe_warm's transfer source; drives
            # the shared bench evaluator (cost_aware=True in the project config
            # makes every trial record the same mean_llm_cost_per_query_usd).
            # MUST run before motpe_warm.
            random_dir.mkdir(parents=True, exist_ok=True)
            logger.info("=" * 60)
            logger.info("PARETO | random | seed=%d | max_trials=%d", cfg.seed, cfg.max_trials)
            logger.info("=" * 60)
            resume_random = resume and (random_dir / "rng_state.pkl").exists()
            random_opt = RandomSearch(
                project=shared.config,
                storage_dir=random_dir,
                resume=resume_random,
            )
            sr_r = await _run_optimizer_with_ledger(
                random_opt,
                method_name="random",
                shared=shared,
                method_dir=random_dir,
                budget=budget,
                seed=cfg.seed,
                resume=resume_random,
            )
            _persist_search_result(sr_r, random_dir)
            logger.info(
                "random done | trials=%d | best_accuracy=%.3f",
                len(sr_r.history),
                max((h.answer_accuracy for h in sr_r.history), default=0.0),
            )

            # motpe_warm — two-objective MO-TPE warm-started from the paired random
            # cell (all of random's completed trials injected as a free, uncounted
            # transfer prior), driving the shared bench evaluator. cost_aware=True
            # in the project config makes it minimize the same
            # mean_llm_cost_per_query_usd as agentic_cost.
            motpe_warm_dir.mkdir(parents=True, exist_ok=True)
            logger.info("=" * 60)
            logger.info("PARETO | motpe_warm | seed=%d | max_trials=%d", cfg.seed, cfg.max_trials)
            logger.info("=" * 60)
            resume_warm = resume and (motpe_warm_dir / "optuna.db").exists()
            motpe_warm_opt = MOTPESearch(
                project=shared.config,
                storage_dir=motpe_warm_dir,
                name="motpe_warm",
                warm_transfer=True,
                transfer_source_dir=random_dir,
                resume=resume_warm,
            )
            sr_w = await _run_optimizer_with_ledger(
                motpe_warm_opt,
                method_name="motpe_warm",
                shared=shared,
                method_dir=motpe_warm_dir,
                budget=budget,
                seed=cfg.seed,
                resume=resume_warm,
            )
            _persist_search_result(sr_w, motpe_warm_dir)
            logger.info(
                "motpe_warm done | trials=%d | best_accuracy=%.3f",
                len(sr_w.history),
                max((h.answer_accuracy for h in sr_w.history), default=0.0),
            )
        finally:
            await shared.cleanup()

    figures_dir = cfg.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Per-method detailed scatter (agentic_cost carries the self-marked frontier
    # + config legend); the head-to-head lives in the comparison figure.
    make_pareto_figure(agentic_dir, figures_dir / "pareto_agentic_cost.png", domain=cfg.corpus_domain)

    method_points = {
        "agentic_cost": _load_trial_points(agentic_dir),
        "random": _load_trial_points(random_dir),
        "motpe_warm": _load_trial_points(motpe_warm_dir),
    }
    method_points = {m: pts for m, pts in method_points.items() if pts}
    if not method_points:
        logger.warning("No plottable trials for any method; skipping comparison figure + hypervolume.json")
        return

    hv_info = compute_pareto_hypervolumes(method_points)
    (cfg.output_root / "hypervolume.json").write_text(json.dumps(hv_info, indent=2), encoding="utf-8")
    make_pareto_comparison_figure(
        method_points, hv_info, figures_dir / "pareto_comparison.png", domain=cfg.corpus_domain
    )
    logger.info(
        "Wrote %s + hypervolume.json (shared cost_ref=%.5f; HV %s)",
        figures_dir / "pareto_comparison.png",
        hv_info["cost_reference"],
        {m: round(d["hypervolume"], 5) for m, d in hv_info["methods"].items()},
    )


def pareto_cli(config_path: str, *, figure_only: bool = False, resume: bool = False) -> None:
    """Sync wrapper for the Typer CLI."""
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("agentic_autorag_bench.run").setLevel(logging.INFO)
    asyncio.run(run_pareto(config_path, figure_only=figure_only, resume=resume))


# ------------------------------------------------------------- config labels

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


# ------------------------------------------------------------------- figure


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
    cmap_name = _FRONTIER_COLORMAP if len(frontier) <= 10 else _FRONTIER_COLORMAP_LARGE
    cmap = plt.get_cmap(cmap_name)
    for i, p in enumerate(frontier, start=1):
        ax.scatter(
            p.cost_per_query,
            p.answer_accuracy * 100,
            s=110,
            color=cmap((i - 1) % cmap.N),
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
    ax.set_title(f"UniDoc{title_domain} — {display_label('agentic_cost')} cost vs. accuracy (self-generated exam)")

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


# ----------------------------------------------- two-method Pareto comparison


@dataclass
class _FrontierRecord:
    """Minimal record for the framework's ``compute_frontier`` / ``compute_hypervolume``
    (which read ``trial_number`` / ``answer_accuracy`` / ``mean_llm_cost_per_query_usd``)."""

    trial_number: int
    answer_accuracy: float
    mean_llm_cost_per_query_usd: float


def compute_pareto_hypervolumes(method_points: dict[str, list[_TrialPoint]]) -> dict:
    """Per-method Pareto frontier + hypervolume against a SHARED reference point.

    The reference point is computed once over the POOLED trials of every method
    (``cost_ref = 2 × max positive cost`` across both methods, ``score_ref = 0``),
    so the two hypervolumes are directly comparable. Computing HV per-method with
    each method's own ``max(cost)`` would make the two numbers incommensurable —
    the Exp-2 fairness landmine. Frontiers are recomputed here via the framework's
    ``compute_frontier`` (not any optimizer's self-marked flag) so both methods are
    treated identically.
    """
    from agentic_autorag.optimizer import pareto as fpareto

    all_costs = [p.cost_per_query for pts in method_points.values() for p in pts]
    cost_ref = fpareto.cost_reference(all_costs)
    ref_point = (0.0, cost_ref)
    out: dict = {"score_reference": 0.0, "cost_reference": cost_ref, "methods": {}}
    for method, pts in method_points.items():
        records = [_FrontierRecord(p.trial_number, p.answer_accuracy, p.cost_per_query) for p in pts]
        frontier = fpareto.compute_frontier(records)
        hv = fpareto.compute_hypervolume(frontier, ref_point=ref_point)
        out["methods"][method] = {
            "hypervolume": hv,
            "n_trials": len(records),
            "n_frontier": len(frontier),
            "frontier_trials": [r.trial_number for r in frontier],
        }
    return out


def make_pareto_comparison_figure(
    method_points: dict[str, list[_TrialPoint]],
    hv_info: dict,
    out_path: Path,
    *,
    domain: str = "",
) -> None:
    """Overlay both methods' trial clouds + Pareto frontiers on one figure, with
    each method's shared-reference-point hypervolume in the legend. No-op when
    nothing is plottable."""
    if not any(method_points.values()):
        logger.warning("No plottable trials; skipping pareto comparison figure")
        return

    apply_paper_style()
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for method, pts in method_points.items():
        if not pts:
            continue
        color = color_for(method)
        ax.scatter(
            [p.cost_per_query for p in pts],
            [p.answer_accuracy * 100 for p in pts],
            s=28,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
        info = hv_info["methods"].get(method, {})
        frontier_tns = set(info.get("frontier_trials", []))
        frontier = sorted((p for p in pts if p.trial_number in frontier_tns), key=lambda p: p.cost_per_query)
        hv = info.get("hypervolume", 0.0)
        label = f"{display_label(method)} (HV={hv:.4f})"
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
    ax.set_title(f"UniDoc{title_domain} — cost vs. accuracy Pareto frontiers")
    ax.legend(loc="lower right", frameon=False, fontsize=9, title="Frontier (shared HV ref point)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
