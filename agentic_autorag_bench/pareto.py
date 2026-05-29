"""Cost-aware Pareto experiment: run ``agentic_cost`` on UniDoc, plot its trials.

A single ``agentic_cost`` search on the UniDoc (healthcare) PDF corpus, scored
on the optimizer's own self-generated exam — no held-out QA. The optimizer
already writes everything the figure needs: each ``history.jsonl`` row carries
``score``, ``mean_llm_cost_per_query_usd``, and ``is_pareto_optimal`` (the flag
is recomputed over all trials after every trial and the whole file rewritten,
so the final file reflects the final frontier), and ``frontier.json`` carries
the knee / recommended / max-score trial numbers.

``make_pareto_figure`` renders a Syftr-style scatter: a gray cloud of every
trial with the optimizer's Pareto frontier highlighted, numbered, and described
in a side legend. X-axis is deploy-time cost **per query** (not Syftr's
per-100-calls).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentic_autorag_bench._figstyle import apply_paper_style, display_label
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
            corpus_path, len(list(corpus_path.glob("*.pdf"))),
        )
        return
    from agentic_autorag_bench.unidoc_corpus import download_unidoc_corpus

    logger.info(
        "UniDoc corpus missing at %s; downloading %d %s PDF(s) + %d image(s)",
        corpus_path, cfg.corpus_max_pdfs, cfg.corpus_domain, cfg.corpus_max_images,
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


async def run_pareto(
    config_path: str | Path, *, figure_only: bool = False, resume: bool = False
) -> None:
    """Run (or skip) the agentic_cost search, then render the Pareto figure."""
    from agentic_autorag.litellm_runtime import configure_litellm_runtime

    from agentic_autorag_bench.methods.agentic import AgenticOptimizer
    from agentic_autorag_bench.run import _persist_search_result
    from agentic_autorag_bench.types import Budget

    cfg = ParetoConfig.load(config_path)
    seed_dir = cfg.output_root / "agentic_cost" / f"seed_{cfg.seed}"

    if not figure_only:
        configure_litellm_runtime()
        corpus_path = _read_corpus_path(cfg.project_config_path)
        _ensure_corpus(corpus_path, cfg)

        seed_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60)
        logger.info("PARETO RUN | agentic_cost | seed=%d | max_trials=%d", cfg.seed, cfg.max_trials)
        logger.info("=" * 60)
        optimizer = AgenticOptimizer(
            config_path=str(cfg.project_config_path),
            output_dir=str(seed_dir),
            cost_aware=True,
            resume=resume,
        )
        sr = await optimizer.search(_stub_evaluator, Budget(max_trials=cfg.max_trials), seed=cfg.seed)
        _persist_search_result(sr, seed_dir)
        logger.info(
            "agentic_cost pareto run done | trials=%d | best_score=%.3f",
            len(sr.history), max((h.score for h in sr.history), default=0.0),
        )

    figures_dir = cfg.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    make_pareto_figure(seed_dir, figures_dir / "pareto.png", domain=cfg.corpus_domain)
    logger.info("Wrote %s", figures_dir / "pareto.png")


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
    "moonshotai", "us", "global", "amazon", "meta", "mistral", "google", "qwen",
    "nvidia", "zai", "minimax", "openai", "ai21", "cohere", "anthropic",
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
    score: float
    is_pareto: bool
    config: dict


def _load_trial_points(seed_dir: Path) -> list[_TrialPoint]:
    """Trials with a usable (cost>0, score) pair, from the rich agentic history."""
    points: list[_TrialPoint] = []
    for row in _read_history(seed_dir):
        cost = row.get("mean_llm_cost_per_query_usd")
        score = row.get("score")
        if cost is None or score is None or float(cost) <= 0.0:
            continue
        points.append(
            _TrialPoint(
                trial_number=int(row.get("trial_number", 0)),
                cost_per_query=float(cost),
                score=float(score),
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
        [p.score * 100 for p in points],
        s=42, c="#d9d9d9", edgecolors="none", alpha=0.7, zorder=1, label="All trials",
    )

    # Frontier line (sorted by cost ascending).
    if len(frontier) >= 2:
        ax.plot(
            [p.cost_per_query for p in frontier],
            [p.score * 100 for p in frontier],
            color="#7f7f7f", lw=1.1, zorder=2, label="Pareto frontier",
        )

    # Colored, numbered frontier points + side-legend descriptions.
    cmap_name = _FRONTIER_COLORMAP if len(frontier) <= 10 else _FRONTIER_COLORMAP_LARGE
    cmap = plt.get_cmap(cmap_name)
    for i, p in enumerate(frontier, start=1):
        ax.scatter(
            p.cost_per_query, p.score * 100,
            s=110, color=cmap((i - 1) % cmap.N), edgecolors="black", linewidths=0.6, zorder=4,
            label=f"{i}. {_describe_config(p.config)}",
        )
        ax.annotate(
            str(i), (p.cost_per_query, p.score * 100),
            textcoords="offset points", xytext=(6, 5), fontweight="bold", zorder=6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Exam accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    title_domain = f" ({domain})" if domain else ""
    ax.set_title(f"UniDoc{title_domain} — {display_label('agentic_cost')} cost vs. accuracy (self-generated exam)")

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
        title="Frontier configurations", title_fontsize=9, borderaxespad=0.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
