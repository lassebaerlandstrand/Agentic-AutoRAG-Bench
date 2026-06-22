"""Agentic AutoRAG (ours) — thin adapter over the framework's optimization loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agentic_autorag.orchestrator import Orchestrator
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")


@dataclass
class AgenticOptimizer:
    """Wraps ``Orchestrator.run()`` so it conforms to the bench Optimizer protocol.

    The framework's orchestrator owns the agent, the proposer LLM, the cost
    ledger, and the per-trial evaluator. It also writes its own ``history.jsonl``
    to ``output_dir``. This adapter just re-shapes the run's outputs into a
    ``SearchResult``; the ``evaluator`` callback is intentionally unused (the
    framework's internal evaluator is the same code path the other methods reach
    through their ``evaluator`` callback, so fairness is preserved).

    ``cost_aware`` chooses between the score- and Pareto-aware variants:
    ``agentic_score`` (``False``, highest-score) and ``agentic_cost`` (``True``,
    Pareto-aware cheapest-best). The flag is propagated by overriding
    ``meta.cost_aware`` on the loaded project config so the YAML's default is
    irrelevant.

    ``use_knowledge_base`` / ``use_diagnosis`` are the ablation toggles. They
    register as separate bench methods (``agentic_nokb`` runs with the KB-off
    hook; ``agentic_nodiag`` skips the per-question diagnosis stage) via an
    explicit ``method_name``. Both default on, so the headline ``agentic_score``
    / ``agentic_cost`` runs are unaffected.
    """

    config_path: str
    output_dir: str
    cost_aware: bool
    use_knowledge_base: bool = True
    use_diagnosis: bool = True
    deterministic: bool = False
    resume: bool = False
    method_name: str | None = None

    @property
    def name(self) -> str:
        if self.method_name:
            return self.method_name
        return "agentic_cost" if self.cost_aware else "agentic_score"

    async def search(
        self,
        evaluator: Evaluator,  # noqa: ARG002 — see docstring
        budget: Budget,
        *,
        seed: int | None = None,
    ) -> SearchResult:
        if budget.max_trials is None:
            raise ValueError("Agentic search requires budget.max_trials")

        orch = Orchestrator(
            self.config_path,
            output_dir_override=self.output_dir,
            seed=seed,
            resume=self.resume,
            skip_final_report=True,
            use_knowledge_base=self.use_knowledge_base,
            use_diagnosis=self.use_diagnosis,
        )
        orch.evaluator.quiet_per_question = True
        # NB: the framework enables ``litellm.drop_params=True`` so reasoning
        # models that don't accept ``seed`` (e.g. azure/o4-mini) silently
        # ignore it. With ``optimizer_model: azure/o4-mini`` the per-seed
        # proposer trajectories are driven only by intrinsic LLM nondeterminism,
        # not by the bench's ``seeds: [1,2,3]`` knob. The paper appendix calls
        # this out — agentic's cross-seed variance is *not* a controlled-seed
        # signal for non-seed-accepting models, while random/motpe variance
        # is genuinely re-randomised.
        # Honour the bench-side budget + the per-method cost-aware flag.
        # Both override the YAML so the same project_config drives both
        # agentic_score and agentic_cost runs without duplication.
        orch.config.meta.max_trials = budget.max_trials
        orch.config.meta.cost_aware = self.cost_aware

        t_start = time.monotonic()
        await orch.run()

        include_graph = orch.config.uses_graph()
        history: list[HistoryEntry] = [
            HistoryEntry(
                trial_number=record.trial_number,
                config=record.config.to_prompt_dump(include_graph=include_graph),
                answer_accuracy=float(record.answer_accuracy),
                metrics={
                    "answer_accuracy": float(record.answer_accuracy),
                    "mean_retrieval_quality": float(record.mean_retrieval_quality),
                    "mean_em": float(record.mean_em),
                    "mean_f1": float(record.mean_f1),
                },
                eval_usd=float(record.total_llm_cost_usd),
                prompt_tokens=int(record.total_prompt_tokens),
                completion_tokens=int(record.total_completion_tokens),
                embedding_tokens=int(record.total_embedding_tokens),
                mean_llm_cost_per_query_usd=float(record.mean_llm_cost_per_query_usd),
            )
            for record in orch.history.records
        ]

        if not history:
            raise RuntimeError("Agentic run produced no successful trials")

        best_entry = max(history, key=lambda h: h.answer_accuracy)

        # ``trial_usd_total`` covers the full per-trial eval spend: each
        # question's RAG generation (``eval_usd`` summed from history) plus
        # the judge calls that score it. Judge is a benchmark-only cost (not
        # paid in production) but still part of what a method "spent" while
        # being benchmarked. Read from cost_breakdown.json alongside the
        # optimizer-side agent_proposal cost.
        optimizer_usd, judge_usd = _read_run_costs_from_ledger_dump(self.output_dir)
        trial_usd_total = sum(h.eval_usd for h in history) + judge_usd

        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=optimizer_usd,
            trial_usd_total=trial_usd_total,
            wall_clock_s=time.monotonic() - t_start,
            prompt_tokens=sum(h.prompt_tokens for h in history),
            completion_tokens=sum(h.completion_tokens for h in history),
            embedding_tokens=sum(h.embedding_tokens for h in history),
            extras={"output_dir": self.output_dir, "cost_aware": self.cost_aware},
        )


def _read_run_costs_from_ledger_dump(output_dir: str) -> tuple[float, float]:
    """Read ``(agent_proposal_usd, judge_usd)`` from ``cost_breakdown.json``.

    Returns ``(0.0, 0.0)`` if the file is missing or unparseable. Caller folds
    ``agent_proposal`` into ``optimizer_usd`` and ``judge`` into
    ``trial_usd_total``.
    """
    import json
    from pathlib import Path

    path = RunLayout(base=Path(output_dir)).cost_breakdown
    if not path.exists():
        return 0.0, 0.0
    try:
        buckets = json.loads(path.read_text(encoding="utf-8")).get("buckets", {})
        return (
            float(buckets.get("agent_proposal", {}).get("usd", 0.0)),
            float(buckets.get("judge", {}).get("usd", 0.0)),
        )
    except Exception:
        logger.warning("Could not parse cost_breakdown.json at %s", path, exc_info=True)
        return 0.0, 0.0
