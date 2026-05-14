"""Agentic AutoRAG (ours) — thin adapter over the framework's optimization loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agentic_autorag.orchestrator import Orchestrator

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
    """

    config_path: str
    output_dir: str
    name: str = "agentic"
    deterministic: bool = False
    debug_prompts: bool = False

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
            debug_prompts=self.debug_prompts,
        )
        orch.evaluator.quiet_per_question = True
        # NB: the framework enables ``litellm.drop_params=True`` so reasoning
        # models that don't accept ``seed`` (e.g. azure/o4-mini) silently
        # ignore it. With ``optimizer_model: azure/o4-mini`` the per-seed
        # proposer trajectories are driven only by intrinsic LLM nondeterminism,
        # not by the bench's ``seeds: [1,2,3]`` knob. The paper appendix calls
        # this out — agentic's cross-seed variance is *not* a controlled-seed
        # signal for non-seed-accepting models, while random/bayesian variance
        # is genuinely re-randomised.
        # Honour the bench-side budget. The YAML may carry a different value
        # for developer use; the bench overrides it for the paper run.
        orch.config.meta.max_trials = budget.max_trials

        t_start = time.monotonic()
        await orch.run()

        include_graph = orch.config.uses_graph()
        history: list[HistoryEntry] = [
            HistoryEntry(
                trial_number=record.trial_number,
                config=record.config.to_prompt_dump(include_graph=include_graph),
                score=float(record.score),
                metrics={
                    "answer_accuracy": float(record.answer_accuracy),
                    "mean_retrieval_quality": float(record.mean_retrieval_quality),
                    "mean_em": float(record.mean_em),
                    "mean_f1": float(record.mean_f1),
                },
                eval_usd=float(record.total_llm_cost_usd),
            )
            for record in orch.history.records
        ]

        if not history:
            raise RuntimeError("Agentic run produced no successful trials")

        best_entry = max(history, key=lambda h: h.score)

        # Cost ledger is reset between runs but the orchestrator already logged
        # its own breakdown. We approximate from history (rag_eval = sum of
        # eval_usd) and read the most recent agent_proposal cost from
        # cost_breakdown.json if present.
        optimizer_usd = _read_optimizer_cost_from_ledger_dump(self.output_dir)
        trial_usd_total = sum(h.eval_usd for h in history)

        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=optimizer_usd,
            trial_usd_total=trial_usd_total,
            wall_clock_s=time.monotonic() - t_start,
            extras={"output_dir": self.output_dir},
        )


def _read_optimizer_cost_from_ledger_dump(output_dir: str) -> float:
    """Pull ``agent_proposal`` USD from ``cost_breakdown.json`` if the framework wrote one."""
    import json
    from pathlib import Path

    path = Path(output_dir) / "cost_breakdown.json"
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("buckets", {}).get("agent_proposal", {}).get("usd", 0.0))
    except Exception:
        logger.warning("Could not parse cost_breakdown.json at %s", path, exc_info=True)
        return 0.0
