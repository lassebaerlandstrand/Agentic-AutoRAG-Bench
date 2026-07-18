"""Cost-aware Pareto experiment: ``agentic_cost`` vs ``random`` vs ``motpe`` vs ``motpe_warm`` on UniDoc.

Four cost-aware searches on the UniDoc (healthcare) PDF corpus, scored on the
optimizer's own self-generated exam — no held-out QA. ``agentic_cost`` is the full
agentic optimizer (Pareto-aware reasoning); ``random`` is the exploration floor and
the transfer source; ``motpe`` is the cold two-objective MO-TPE (no transfer prior);
``motpe_warm`` is the SAME MO-TPE warm-started from ``random`` (all of random's
completed trials injected as a free, uncounted transfer prior). The cold→warm
frontier gap isolates the value of that transfer prior (warm-start ablation). All
minimize the SAME ``mean_llm_cost_per_query_usd`` and maximize the SAME exam accuracy
on the SAME exam, so the comparison is fair. ``random`` runs before ``motpe_warm``
(a hard data dependency); cold ``motpe`` has none.

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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Pareto figures live in the shared ``plots`` module (all benchmark figures in
# one place); the command below just drives them.
from agentic_autorag_bench.plots import (
    _load_method_seed_points,
    _load_trial_points,
    compute_pareto_hypervolumes,
    make_pareto_attainment_figure,
    make_pareto_attainment_median_figure,
    make_pareto_comparison_figure,
    make_pareto_cost_and_embeddings_figure,
    make_pareto_figure,
    make_pareto_frontier_annotated_figure,
    make_pareto_hv_convergence_figure,
    make_pareto_hypervolume_box_figure,
    make_pareto_median_hv_combined_figure,
)

logger = logging.getLogger("agentic_autorag_bench.run")

# Canonical method slate + run order for the Pareto comparison. ``random`` runs
# before ``motpe_warm`` (its on-disk transfer source); cold ``motpe`` has no
# ordering constraint. ``agentic_cost`` uses its own self-contained orchestrator.
_ALL_METHODS = ["agentic_cost", "random", "motpe_warm", "motpe"]
# Methods that share the single bench evaluator orchestrator (built once per run).
_SHARED_ORCH_METHODS = frozenset({"random", "motpe_warm", "motpe"})


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
    # Defaulted so direct construction (tests / ad-hoc) is valid; ``load`` always
    # sets both from the entry YAML (or the canonical slate / ``[seed]``).
    methods: list[str] = field(default_factory=lambda: list(_ALL_METHODS))
    seeds: list[int] = field(default_factory=lambda: [1])

    @classmethod
    def load(cls, config_path: str | Path) -> ParetoConfig:
        config_path = Path(config_path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_path = (config_path.parent / raw["project_config"]).resolve()
        corpus = raw.get("corpus") or {}
        seed = int(raw.get("seed", 1))
        return cls(
            project_config_path=project_path,
            seed=seed,
            max_trials=int(raw["budget"]["max_trials"]),
            corpus_domain=str(corpus.get("domain", "healthcare")),
            corpus_max_pdfs=int(corpus.get("max_pdfs", 230)),
            corpus_max_images=int(corpus.get("max_images", 20)),
            output_root=Path(raw["output_root"]).resolve(),
            methods=[str(m) for m in (raw.get("methods") or _ALL_METHODS)],
            seeds=[int(s) for s in (raw.get("seeds") or [seed])],
        )


def _read_corpus_path(project_config_path: Path) -> Path:
    """``meta.corpus_path`` from the project YAML, resolved like the orchestrator."""
    raw = yaml.safe_load(Path(project_config_path).read_text(encoding="utf-8"))
    return Path(raw["meta"]["corpus_path"]).resolve()


def _read_shared_cache_dir(project_config_path: Path) -> Path:
    """``meta.output_dir`` from the project YAML — the shared cache the orchestrator
    writes (corpus parse, ``exam.json``, probe indexes). Resolved like the orchestrator."""
    raw = yaml.safe_load(Path(project_config_path).read_text(encoding="utf-8"))
    return Path(raw["meta"]["output_dir"]).resolve()


# The search space caps input context (top_k x chunk size) but not output length,
# so the answer is bounded here, set above the longest answer any run produced.
MAX_ANSWER_TOKENS = 2048


def space_derived_cost_reference(project_config_path: Path) -> float:
    """Hypervolume cost-axis reference: the maximum per-query LLM cost the search
    space can express.

    The priciest generator in the pool answers the largest retrieval context the
    space allows (``top_k`` max x ``chunk_token_size`` max tokens) plus a
    full-length answer. This is a property of the space and the price catalogue,
    not of which configs a run happened to sample, so it is reproducible and shared
    across every method. Prices come from litellm — the same catalogue that priced
    the trials — so the reference and the measured costs sit on one scale.
    """
    import litellm

    space = yaml.safe_load(Path(project_config_path).read_text(encoding="utf-8"))["search_space"]
    max_context = space["retrieval"]["top_k"]["max"] * max(space["chunking"]["chunk_token_size"]["values"])
    max_in = 0.0
    max_out = 0.0
    for model in space["generator"]["models"]:
        try:
            info = litellm.get_model_info(model)
        except Exception:
            continue
        max_in = max(max_in, info.get("input_cost_per_token") or 0.0)
        max_out = max(max_out, info.get("output_cost_per_token") or 0.0)
    if max_in <= 0.0 and max_out <= 0.0:
        raise ValueError("no generator model in the pool has a litellm price; cannot derive a cost reference")
    return max_context * max_in + MAX_ANSWER_TOKENS * max_out


SETUP_MARKER = ".setup_complete"


def method_seed_complete(output_root: Path, method: str, seed: int, max_trials: int) -> bool:
    """Disk-truth completion oracle for one Pareto ``(method, seed)`` cell.

    A cell is DONE iff its ``optimizer_meta.json`` reports
    ``n_trials_completed >= max_trials``. The pareto path never writes
    ``benchmark_results.json`` (no held-out eval), so the Exp-1 sentinel does not
    apply here. This is both the Exp-2 scheduler's oracle and run_pareto's
    ``motpe_warm`` transfer-source guard, so the two always agree.
    """
    meta = Path(output_root) / method / f"seed_{seed}" / "optimizer_meta.json"
    if not meta.exists():
        return False
    try:
        n = int(json.loads(meta.read_text(encoding="utf-8")).get("n_trials_completed", 0))
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    return n >= int(max_trials)


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


async def run_pareto(
    config_path: str | Path,
    *,
    figure_only: bool = False,
    resume: bool = False,
    methods: list[str] | None = None,
    seed: int | None = None,
    setup_only: bool = False,
) -> None:
    """Run the cost-aware Pareto comparison — ``agentic_cost`` (full agentic) vs
    ``random`` (floor + transfer source) vs ``motpe`` (cold two-objective MO-TPE)
    vs ``motpe_warm`` (same MO-TPE warm-started from ``random``) — then render the
    multi-frontier figure with a shared-reference-point hypervolume.

    All methods evaluate the SAME self-generated exam on the SAME corpus: the
    shared orchestrator (used for the random / motpe / motpe_warm evaluator) and
    agentic_cost's own orchestrator load the same project config, so the corpus
    index + ``exam.json`` cache is shared. Only the proposer differs.

    Selective / concurrent execution (the Exp-2 scheduler drives this):

    * ``setup_only=True`` — warm the shared caches (corpus parse + ``exam.json`` +
      probe indexes) with a SINGLE writer and write a ``.setup_complete`` marker,
      then return without running any trial. This must precede any concurrent
      per-method fan-out, since the corpus/exam caches are non-atomic.
    * ``methods`` — run only this subset (canonical order preserved), applying the
      ``seed`` override. A subset invocation renders NO figures (the finalize pass,
      ``figure_only=True``, does). ``methods=None`` keeps the original one-process
      behaviour: run all four in order and render figures at the end.
    * ``motpe_warm`` reads the paired ``random`` cell's history as its transfer
      prior; if ``random`` is not in this invocation, the same-seed ``random`` cell
      must already be complete on disk (else a clear error).
    """
    from agentic_autorag.litellm_runtime import configure_litellm_runtime
    from agentic_autorag.orchestrator import Orchestrator

    from agentic_autorag_bench.methods.agentic import AgenticOptimizer
    from agentic_autorag_bench.methods.motpe import MOTPESearch
    from agentic_autorag_bench.methods.random import RandomSearch
    from agentic_autorag_bench.run import _persist_search_result, _run_optimizer_with_ledger
    from agentic_autorag_bench.types import Budget

    cfg = ParetoConfig.load(config_path)
    eff_seed = cfg.seed if seed is None else int(seed)
    # ``methods=None`` = full default run (renders figures); a subset = a
    # scheduler per-cell run (figures deferred to the --figure-only finalize).
    run_all = methods is None
    selected = list(cfg.methods) if run_all else list(methods)
    unknown = [m for m in selected if m not in _ALL_METHODS]
    if unknown:
        raise ValueError(f"unknown pareto method(s) {unknown}; choose from {_ALL_METHODS}")

    seed_label = f"seed_{eff_seed}"
    agentic_dir = cfg.output_root / "agentic_cost" / seed_label
    random_dir = cfg.output_root / "random" / seed_label
    motpe_dir = cfg.output_root / "motpe" / seed_label
    motpe_warm_dir = cfg.output_root / "motpe_warm" / seed_label

    # ---- setup-only: single-writer warmup, then a marker (no trials) ----------
    if setup_only:
        configure_litellm_runtime()
        corpus_path = _read_corpus_path(cfg.project_config_path)
        _ensure_corpus(corpus_path, cfg)
        logger.info("PARETO SETUP | warming shared corpus + exam cache (single writer)")
        orch = Orchestrator(str(cfg.project_config_path))
        try:
            await orch.setup()
        finally:
            await orch.cleanup()
        cache_dir = _read_shared_cache_dir(cfg.project_config_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        marker = cache_dir / SETUP_MARKER
        marker.write_text("setup complete\n", encoding="utf-8")
        logger.info("PARETO SETUP | done; wrote %s", marker)
        return

    if not figure_only:
        configure_litellm_runtime()
        corpus_path = _read_corpus_path(cfg.project_config_path)
        _ensure_corpus(corpus_path, cfg)
        budget = Budget(max_trials=cfg.max_trials)

        # The shared bench evaluator orchestrator is only needed by
        # random/motpe/motpe_warm; agentic_cost carries its own. Skip building it
        # (and its setup()) when only agentic_cost is selected.
        need_shared = any(m in _SHARED_ORCH_METHODS for m in selected)
        shared = None
        if need_shared:
            logger.info("Setting up shared orchestrator for the Pareto comparison (exam reused from cache)")
            shared = Orchestrator(str(cfg.project_config_path))
            shared.evaluator.quiet_per_question = True
            await shared.setup()
        try:
            if "agentic_cost" in selected:
                # agentic_cost — full agentic, cost-aware (self-contained orchestrator).
                agentic_dir.mkdir(parents=True, exist_ok=True)
                logger.info("=" * 60)
                logger.info("PARETO | agentic_cost | seed=%d | max_trials=%d", eff_seed, cfg.max_trials)
                logger.info("=" * 60)
                agentic_opt = AgenticOptimizer(
                    config_path=str(cfg.project_config_path),
                    output_dir=str(agentic_dir),
                    cost_aware=True,
                    resume=resume,
                )
                sr_a = await agentic_opt.search(_stub_evaluator, budget, seed=eff_seed)
                _persist_search_result(sr_a, agentic_dir)
                logger.info(
                    "agentic_cost done | trials=%d | best_accuracy=%.3f",
                    len(sr_a.history),
                    max((h.answer_accuracy for h in sr_a.history), default=0.0),
                )

            if "random" in selected:
                # random — exploration floor AND motpe_warm's transfer source; drives
                # the shared bench evaluator (cost_aware=True in the project config
                # makes every trial record the same mean_llm_cost_per_query_usd).
                # MUST run (or already be complete on disk) before motpe_warm.
                random_dir.mkdir(parents=True, exist_ok=True)
                logger.info("=" * 60)
                logger.info("PARETO | random | seed=%d | max_trials=%d", eff_seed, cfg.max_trials)
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
                    seed=eff_seed,
                    resume=resume_random,
                )
                _persist_search_result(sr_r, random_dir)
                logger.info(
                    "random done | trials=%d | best_accuracy=%.3f",
                    len(sr_r.history),
                    max((h.answer_accuracy for h in sr_r.history), default=0.0),
                )

            if "motpe_warm" in selected:
                # motpe_warm — two-objective MO-TPE warm-started from the paired random
                # cell (all of random's completed trials injected as a free, uncounted
                # transfer prior), driving the shared bench evaluator. When random is
                # not part of THIS invocation it must already be complete on disk.
                if "random" not in selected and not method_seed_complete(
                    cfg.output_root, "random", eff_seed, cfg.max_trials
                ):
                    raise RuntimeError(
                        f"motpe_warm needs a completed random cell as its transfer prior, but "
                        f"{random_dir} is missing or incomplete "
                        f"(need optimizer_meta.json n_trials_completed >= {cfg.max_trials}). "
                        f"Run the same-seed random cell first."
                    )
                motpe_warm_dir.mkdir(parents=True, exist_ok=True)
                logger.info("=" * 60)
                logger.info("PARETO | motpe_warm | seed=%d | max_trials=%d", eff_seed, cfg.max_trials)
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
                    seed=eff_seed,
                    resume=resume_warm,
                )
                _persist_search_result(sr_w, motpe_warm_dir)
                logger.info(
                    "motpe_warm done | trials=%d | best_accuracy=%.3f",
                    len(sr_w.history),
                    max((h.answer_accuracy for h in sr_w.history), default=0.0),
                )

            if "motpe" in selected:
                # motpe (cold) — the SAME two-objective MO-TPE as motpe_warm but with
                # NO transfer prior (warm_transfer defaults False, no transfer_source_dir),
                # driving the shared bench evaluator. The cold→warm frontier gap isolates
                # the value of random's free transfer prior (warm-start ablation). No data
                # dependency, so ordering is free.
                motpe_dir.mkdir(parents=True, exist_ok=True)
                logger.info("=" * 60)
                logger.info("PARETO | motpe | seed=%d | max_trials=%d", eff_seed, cfg.max_trials)
                logger.info("=" * 60)
                resume_cold = resume and (motpe_dir / "optuna.db").exists()
                motpe_opt = MOTPESearch(
                    project=shared.config,
                    storage_dir=motpe_dir,
                    name="motpe",
                    resume=resume_cold,
                )
                sr_c = await _run_optimizer_with_ledger(
                    motpe_opt,
                    method_name="motpe",
                    shared=shared,
                    method_dir=motpe_dir,
                    budget=budget,
                    seed=eff_seed,
                    resume=resume_cold,
                )
                _persist_search_result(sr_c, motpe_dir)
                logger.info(
                    "motpe done | trials=%d | best_accuracy=%.3f",
                    len(sr_c.history),
                    max((h.answer_accuracy for h in sr_c.history), default=0.0),
                )
        finally:
            if shared is not None:
                await shared.cleanup()

    # Figures render only on the full default run or an explicit --figure-only
    # pass (the finalize step). A scheduler per-cell subset leaves them alone so
    # no partial/misleading hypervolume.json is written mid-experiment.
    if not (run_all or figure_only):
        return

    figures_dir = cfg.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Per-method detailed scatter (agentic_cost carries the self-marked frontier
    # + config legend); the head-to-head lives in the comparison figure.
    make_pareto_figure(agentic_dir, figures_dir / "pareto_agentic_cost.png", domain=cfg.corpus_domain)

    method_points = {
        "agentic_cost": _load_trial_points(agentic_dir),
        "random": _load_trial_points(random_dir),
        "motpe": _load_trial_points(motpe_dir),
        "motpe_warm": _load_trial_points(motpe_warm_dir),
    }
    method_points = {m: pts for m, pts in method_points.items() if pts}
    if not method_points:
        logger.warning("No plottable trials for any method; skipping comparison figure + hypervolume.json")
        return

    cost_ref = space_derived_cost_reference(cfg.project_config_path)
    hv_info = compute_pareto_hypervolumes(_load_method_seed_points(cfg.output_root), cost_ref)
    (cfg.output_root / "hypervolume.json").write_text(json.dumps(hv_info, indent=2), encoding="utf-8")
    make_pareto_comparison_figure(
        method_points, figures_dir / "pareto_comparison.png", domain=cfg.corpus_domain
    )
    logger.info(
        "Wrote %s + hypervolume.json (space-derived cost_ref=%.5f; normalized HV mean %s)",
        figures_dir / "pareto_comparison.png",
        hv_info["cost_reference"],
        {m: round(d["hypervolume_mean"], 4) for m, d in hv_info["methods"].items()},
    )

    # Multi-seed figures (read every seed, show the seed spread as a band).
    # Best-effort: a failure here must not discard the hypervolume.json /
    # comparison figure already written above.
    try:
        make_pareto_attainment_figure(
            cfg.output_root, figures_dir / "pareto_cost_accuracy.png", domain=cfg.corpus_domain
        )
    except Exception:
        logger.warning("cost-accuracy figure failed", exc_info=True)
    try:
        make_pareto_attainment_median_figure(
            cfg.output_root, figures_dir / "pareto_cost_accuracy_median.png", domain=cfg.corpus_domain
        )
    except Exception:
        logger.warning("cost-accuracy median figure failed", exc_info=True)
    try:
        make_pareto_frontier_annotated_figure(
            cfg.output_root, figures_dir / "pareto_frontier_configs.png", domain=cfg.corpus_domain
        )
    except Exception:
        logger.warning("annotated frontier figure failed", exc_info=True)
    try:
        make_pareto_cost_and_embeddings_figure(
            cfg.output_root, figures_dir / "cost_and_embeddings.png", domain=cfg.corpus_domain
        )
    except Exception:
        logger.warning("cost_and_embeddings figure failed", exc_info=True)
    try:
        make_pareto_hv_convergence_figure(
            cfg.output_root, figures_dir / "pareto_hypervolume.png", domain=cfg.corpus_domain, cost_ref=cost_ref
        )
    except Exception:
        logger.warning("hypervolume figure failed", exc_info=True)
    try:
        make_pareto_hypervolume_box_figure(
            cfg.output_root, figures_dir / "pareto_hypervolume_box.png", domain=cfg.corpus_domain
        )
    except Exception:
        logger.warning("hypervolume box figure failed", exc_info=True)
    try:
        make_pareto_median_hv_combined_figure(
            cfg.output_root, figures_dir / "pareto_median_and_hypervolume.png",
            domain=cfg.corpus_domain, cost_ref=cost_ref,
        )
    except Exception:
        logger.warning("combined median+HV figure failed", exc_info=True)


def pareto_cli(
    config_path: str,
    *,
    figure_only: bool = False,
    resume: bool = False,
    methods: list[str] | None = None,
    seed: int | None = None,
    setup_only: bool = False,
) -> None:
    """Sync wrapper for the Typer CLI."""
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("agentic_autorag_bench.run").setLevel(logging.INFO)
    asyncio.run(
        run_pareto(
            config_path,
            figure_only=figure_only,
            resume=resume,
            methods=methods,
            seed=seed,
            setup_only=setup_only,
        )
    )
