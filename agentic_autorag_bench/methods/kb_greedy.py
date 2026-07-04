"""kb_greedy — single-config reference: score the KB's strongest pipeline once.

No search, no exam generation. Builds the "most capable" config — the
Tier-4-strong probe configuration (strongest LLM + strongest embedder + best
reranker + max chunk size + max top_k + max reranker_top_n) — by reusing the
examiner's probe builder, then evaluates it once on the held-out gold test. It
answers two reviewer questions cheaply: "does iterative search beat starting
from the KB's best guess?" and "does the agent blindly follow the KB?".

Not a registered search method; driven by its own CLI subcommand. The on-disk
contract is ``output_root/kb_greedy/seed_<n>/benchmark_results.json`` (+
``best_config.yaml``), which the matrix figures auto-discover.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from agentic_autorag.config.knowledge_base import KnowledgeBase
from agentic_autorag.config.loader import load_config
from agentic_autorag.config.models import (
    GRAPH_INDEX_TYPES,
    DiscreteValues,
    IndexType,
    ProjectConfig,
    TrialConfig,
    _dim_max_value,
    _dim_min_value,
)
from agentic_autorag.examiner.probe_selector import rank_models_for_probes, select_probe_configs

logger = logging.getLogger("agentic_autorag_bench.kb_greedy")

# When the strongest probe lands on a graph index (the held-out runner can't
# build graphs), fall back to the strongest available non-graph index.
_NON_GRAPH_PREFERENCE = (IndexType.HYBRID_BM25_VECTOR, IndexType.VECTOR_ONLY)


async def build_strongest_config(project: ProjectConfig) -> TrialConfig:
    """Assemble the KB's most-capable pipeline = the Tier-4-strong probe config.

    Mirrors the orchestrator's KB load + probe ranking, then reuses
    ``select_probe_configs`` (weakest→strongest) and takes the strongest probe
    verbatim so kb_greedy matches the examiner's own notion of "most capable".
    """
    ss = project.search_space

    # KB + embedding_token_limits must be populated before select_probe_configs,
    # which caps the Tier-4 chunk size to the strongest embedder's max_tokens.
    try:
        kb: KnowledgeBase | None = KnowledgeBase()
    except Exception as e:
        logger.warning("Could not load knowledge base: %s. Ranking falls back to LLM/order.", e)
        kb = None
    if kb:
        embed_models = kb._embeddings.get("models", {})
        for name in ss.embedding.models:
            entry = embed_models.get(name)
            if entry and entry.get("max_tokens"):
                project.embedding_token_limits[name] = int(entry["max_tokens"])

    optimizer_model = project.agent.optimizer_model
    all_llms = ss.all_llm_models()
    reasoning_allowed = {m: ss.is_reasoning_allowed(m) for m in all_llms}
    ranked_llms = await rank_models_for_probes(
        all_llms,
        "llm",
        kb,
        optimizer_model,
        reasoning_allowed=reasoning_allowed,
        reasoning_effort=ss.generator.reasoning_effort,
    )
    ranked_embeds = await rank_models_for_probes(ss.embedding.models, "embedding", kb, optimizer_model)
    ranked_rerankers = await rank_models_for_probes(ss.reranker.models, "reranker", kb, optimizer_model)

    probes = select_probe_configs(
        project,
        ranked_llms=ranked_llms,
        ranked_embeds=ranked_embeds,
        ranked_rerankers=ranked_rerankers,
    )
    if not probes:
        raise RuntimeError("kb_greedy: probe builder produced no valid config")
    _label, strongest = probes[-1]

    # Snap the capability levers to their search-grid extremes. The probe builder
    # can emit search-space-invalid values (overlap = chunk_size // 10,
    # temperature = 0.0 even when the space pins it) and, if a narrow space dedups
    # Tier-4 into Tier-3, can drop top_k / reranker_top_n below their max. Forcing
    # the grid extremes keeps kb_greedy the "most capable" config AND a point the
    # search methods could also have reached, regardless of dedup collapse.
    updates: dict = {
        "top_k": int(_dim_max_value(ss.retrieval.top_k)),
        "reranker_top_n": int(_dim_max_value(ss.reranker.top_n)),
    }
    overlap_dim = ss.chunking.chunk_token_overlap
    if isinstance(overlap_dim, DiscreteValues):
        valid = [v for v in overlap_dim.values if v < strongest.chunk_token_size]
        if valid:
            updates["chunk_token_overlap"] = max(valid)
    t_min, t_max = _dim_min_value(ss.temperature), _dim_max_value(ss.temperature)
    updates["temperature"] = min(max(strongest.temperature, t_min), t_max)
    strongest = strongest.model_copy(update=updates)

    if strongest.index_type in GRAPH_INDEX_TYPES:
        replacement = next((it for it in _NON_GRAPH_PREFERENCE if it in ss.retrieval.index_types), None)
        if replacement is None:
            raise RuntimeError("kb_greedy: no non-graph index_type available in the search space")
        strongest = strongest.model_copy(update={"index_type": replacement})

    violations = project.validate_trial(strongest)
    if violations:
        raise RuntimeError("kb_greedy strongest config invalid:\n" + "\n".join(f"- {v}" for v in violations))
    return strongest


async def run_kb_greedy(config_path: str | Path, *, seed: int = 42) -> None:
    """Build the strongest config and evaluate it once on the held-out gold."""
    from agentic_autorag.litellm_runtime import configure_litellm_runtime

    from agentic_autorag_bench._holdout_registry import apply_union_exclusion
    from agentic_autorag_bench.benchmarks.runner import BenchmarkRunner
    from agentic_autorag_bench.plots import make_matrix_figures, make_seed_figures
    from agentic_autorag_bench.run import BenchConfig

    configure_litellm_runtime()
    bench = BenchConfig.load(config_path)
    benchmark = BenchmarkRunner(
        name=bench.benchmark.name,
        output_dir=bench.benchmark.output_dir,
        split=bench.benchmark.split,
        sample_size=bench.benchmark.sample_size,
        seed=bench.benchmark.prep_seed,
    )
    benchmark.prepare()

    project = load_config(str(bench.project_config_path))
    trial = await build_strongest_config(project)
    logger.info(
        "kb_greedy strongest config | llm=%s embed=%s reranker=%s index=%s chunk=%d top_k=%d rerank_top_n=%d",
        trial.generator_llm,
        trial.embedding_model,
        trial.reranker,
        trial.index_type.value,
        trial.chunk_token_size,
        trial.top_k,
        trial.reranker_top_n,
    )

    seed_dir = bench.output_root / "kb_greedy" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "best_config.yaml").write_text(
        yaml.safe_dump(trial.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    # Minimal optimizer_meta so analyze / _is_method_seed_complete are satisfied
    # (kb_greedy runs zero search trials, so all optimizer costs are zero).
    (seed_dir / "optimizer_meta.json").write_text(
        json.dumps(
            {
                "method": "kb_greedy",
                "seed": seed,
                "deterministic": True,
                "optimizer_usd": 0.0,
                "trial_usd_total": 0.0,
                "wall_clock_s": 0.0,
                "n_trials_completed": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    await benchmark.evaluate(
        project_config_path=str(bench.project_config_path),
        trial_config=trial,
        output_path=seed_dir / "benchmark_results.json",
        qa_path_override=bench.hold_out_qa_path,
        judge_model=bench.hold_out_judge_model,
        limit=bench.hold_out_limit,
        concurrency=bench.hold_out_concurrency,
        exclude_question_types=bench.hold_out_exclude_question_types,
    )

    # Figures are best-effort: the eval result is already persisted, and the
    # matrix figure may not render until the other methods exist.
    try:
        make_seed_figures(seed_dir)
        apply_union_exclusion(bench.output_root)
        make_matrix_figures(bench.output_root)
    except Exception as e:
        logger.warning("kb_greedy: figure generation skipped (%s)", e)
